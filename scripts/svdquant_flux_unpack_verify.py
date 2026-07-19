#!/usr/bin/env python3
# svdquant_flux_unpack_verify.py — reconstruct a nunchaku class-A linear using
# the EXACT kernel-layout un-permutation (ported verbatim from the reference
# nunchaku loader /home/alex/Wan2GP/shared/qtypes/nunchaku_int4.py) and prove it
# matches the bf16 ground-truth weight in flux1-dev.
#
# The Phase-2 sweep proved the naive [out,in/2] row-major read gives cos~0: the
# public checkpoint stores qweight AND wscales AND smooth AND the low-rank
# matrices in a warp/mma-packed layout. This script applies the real unpack:
#   W = (unpack_w4a4(qweight) * expand(unpack_wscales(wscales))) / unpack(smooth)
#         + unpack_lowrank(proj_up, down=False) @ unpack_lowrank(proj_down, down=True)
# and reports cos vs ground truth. Expect >= 0.99 (int4 + rank32 closely
# approximates the original bf16 weight).
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python scripts/svdquant_flux_unpack_verify.py

import torch
from safetensors import safe_open

INT4 = "/home/alex/.serenity/models/checkpoints/nunchaku/svdq-int4_r32-flux.1-dev.safetensors"
BF16 = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"
GROUP = 64

LAYERS = {
    "transformer_blocks.0.out_proj": "double_blocks.0.img_attn.proj.weight",   # [3072,3072]
    "transformer_blocks.0.qkv_proj": "double_blocks.0.img_attn.qkv.weight",    # [9216,3072]
    "transformer_blocks.0.mlp_fc1":  "double_blocks.0.img_mlp.0.weight",       # [12288,3072]
    "transformer_blocks.0.mlp_fc2":  "double_blocks.0.img_mlp.2.weight",       # [3072,12288]
}


# ── verbatim ports from nunchaku_int4.py ──────────────────────────────────────
def _unpack_int4_from_int8(qweight):
    q = qweight.to(torch.uint8)
    low = (q & 0x0F).to(torch.int16)
    high = ((q >> 4) & 0x0F).to(torch.int16)
    low -= (low >= 8).to(torch.int16) * 16
    high -= (high >= 8).to(torch.int16) * 16
    stacked = torch.stack((low, high), dim=-1)
    return stacked.reshape(qweight.shape[0], qweight.shape[1] * 2)


def _unpack_nunchaku_w4a4_weight(qweight, out_features, in_features):
    if qweight.dtype != torch.int8:
        return _unpack_int4_from_int8(qweight)
    if qweight.numel() != out_features * in_features // 2:
        return _unpack_int4_from_int8(qweight)
    mem_n, mem_k, num_k_unrolls = 128, 64, 2
    if out_features % mem_n != 0 or in_features % (mem_k * num_k_unrolls) != 0:
        return _unpack_int4_from_int8(qweight)
    n_tiles = out_features // mem_n
    k_tiles = in_features // mem_k
    packed_i32 = qweight.view(torch.int32)
    packed_i32 = packed_i32.view(n_tiles, k_tiles, 1, 8, 8, 4, 2, 2, 1)
    vals = torch.stack(
        [(packed_i32 >> shift) & 0xF for shift in (0, 4, 8, 12, 16, 20, 24, 28)],
        dim=-1,
    )
    vals = vals.permute(0, 3, 6, 4, 8, 1, 2, 7, 5, 9).contiguous()
    vals = vals.view(out_features, in_features).to(torch.int16)
    vals -= (vals >= 8).to(torch.int16) * 16
    return vals


def _unpack_nunchaku_wscales(wscales, out_features, in_features, group_size):
    if wscales is None or wscales.ndim != 2:
        return wscales
    if in_features % group_size != 0:
        return wscales
    groups = in_features // group_size
    if wscales.shape != (groups, out_features):
        return wscales
    warp_n, num_lanes = 128, 32
    s_pack_size = min(max(warp_n // num_lanes, 2), 8)
    num_s_lanes = min(num_lanes, warp_n // s_pack_size)
    num_s_packs = warp_n // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    if out_features % warp_s != 0:
        return wscales
    packed = wscales.view(
        out_features // warp_s, groups, num_s_packs, num_s_lanes // 4, 4, s_pack_size // 2, 2
    )
    unpacked = packed.permute(0, 2, 3, 5, 4, 6, 1).contiguous()
    return unpacked.view(out_features, groups).transpose(0, 1).contiguous()


def _unpack_nunchaku_scale_vector(scale, size):
    if scale is None or scale.ndim != 1 or scale.numel() != size:
        return scale
    warp_n, num_lanes = 128, 32
    s_pack_size = min(max(warp_n // num_lanes, 2), 8)
    num_s_lanes = min(num_lanes, warp_n // s_pack_size)
    num_s_packs = warp_n // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    if size % warp_s != 0:
        return scale
    packed = scale.reshape(size // warp_s, 1, num_s_packs, num_s_lanes // 4, 4, s_pack_size // 2, 2)
    unpacked = packed.permute(0, 2, 3, 5, 4, 6, 1).contiguous()
    return unpacked.view(size)


def _expand_group_scales(scales, group_size):
    return scales.transpose(0, 1).repeat_interleave(group_size, dim=1)


def _unpack_lowrank_weight(weight, down):
    if weight is None or weight.ndim != 2:
        return weight
    c, r = weight.shape
    reg_n, reg_k = 1, 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    pack_n = n_pack_size * num_n_lanes * reg_n
    pack_k = k_pack_size * num_k_lanes * reg_k
    if down:
        if r % pack_n != 0 or c % pack_k != 0:
            return weight
        r_packs, c_packs = r // pack_n, c // pack_k
    else:
        if c % pack_n != 0 or r % pack_k != 0:
            return weight
        c_packs, r_packs = c // pack_n, r // pack_k
    weight = weight.view(
        c_packs, r_packs, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, reg_n, reg_k
    )
    weight = weight.permute(0, 1, 4, 2, 6, 5, 3, 7).contiguous()
    weight = weight.view(c_packs, r_packs, pack_n, pack_k)
    if down:
        weight = weight.permute(1, 2, 0, 3).contiguous().view(r, c)
    else:
        weight = weight.permute(0, 2, 1, 3).contiguous().view(c, r)
    return weight


def cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm())).item()


def main():
    torch.set_grad_enabled(False)
    fi = safe_open(INT4, framework="pt", device="cpu")
    fb = safe_open(BF16, framework="pt", device="cpu")
    print(f"{'layer':38s} {'shape':14s} {'cos_int4only':>13s} {'cos_full':>10s}")
    allcos = []
    for pfx, gt in LAYERS.items():
        qweight = fi.get_tensor(pfx + ".qweight")
        wscales = fi.get_tensor(pfx + ".wscales")
        lora_down = fi.get_tensor(pfx + ".lora_down")
        lora_up = fi.get_tensor(pfx + ".lora_up")
        smooth_f = fi.get_tensor(pfx + ".smooth")
        W_orig = fb.get_tensor(gt).float()
        out_f, in_f = W_orig.shape

        qvals = _unpack_nunchaku_w4a4_weight(qweight, out_f, in_f).float()
        ws = _unpack_nunchaku_wscales(wscales, out_f, in_f, GROUP)
        scales = _expand_group_scales(ws, GROUP).float()
        base = qvals * scales
        smooth = _unpack_nunchaku_scale_vector(smooth_f, in_f).float()
        base = base / smooth
        cos_int4 = cos(base, W_orig)   # int4 branch only (no low-rank)

        pd = _unpack_lowrank_weight(lora_down, down=True).float()   # [rank,in]
        pu = _unpack_lowrank_weight(lora_up, down=False).float()    # [out,rank]
        W = base + pu @ pd
        cos_full = cos(W, W_orig)
        allcos.append(cos_full)
        print(f"{pfx:38s} {str(list(W_orig.shape)):14s} {cos_int4:+13.6f} {cos_full:+10.6f}")

    print(f"\nmin cos_full across layers = {min(allcos):.6f}")
    if min(allcos) >= 0.99:
        print(">>> CONVENTION CONFIRMED (public nunchaku layout): "
              "qweight+wscales+smooth+lowrank are warp/mma-packed; "
              "smooth folds as DIVIDE; sign=twos_complement. cos>=0.99 vs ground truth.")
    else:
        print("*** still below 0.99 — investigate further.")


if __name__ == "__main__":
    main()
