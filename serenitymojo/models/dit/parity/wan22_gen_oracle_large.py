#!/usr/bin/env python3
# Wan2.2-TI2V-5B DiT oracle generator — LARGE GRID variant (DEV-ONLY).
#
# Same canonical WanModel as wan22_gen_oracle.py, but a larger video grid so the
# token count S is multi-block (>512) and the math-mode SDPA path would be heavy
# / OOM at head_dim=128. Dumps the full-forward reference for the tiled-SDPA gate.
#
#   - wan22_full_x_in_large.bin        raw latent input [in_dim, F, H, W] f32
#   - wan22_full_context_raw_large.bin raw text tokens [ctx_len, text_dim] f32
#   - wan22_full_out_large.bin         final unpatchified output [out_dim,F,H,W] f32
#   - wan22_grid_large.txt             grid metadata
#
# Run: /home/alex/SimpleTuner/.venv/bin/python wan22_gen_oracle_large.py
import os, sys, math
import numpy as np
import torch

import importlib.util, types
WAN_ROOT = "/home/alex/Wan2.2"
pkg_wan = types.ModuleType("wan"); pkg_wan.__path__ = [os.path.join(WAN_ROOT, "wan")]
pkg_mods = types.ModuleType("wan.modules"); pkg_mods.__path__ = [os.path.join(WAN_ROOT, "wan", "modules")]
sys.modules["wan"] = pkg_wan
sys.modules["wan.modules"] = pkg_mods
spec_a = importlib.util.spec_from_file_location(
    "wan.modules.attention", os.path.join(WAN_ROOT, "wan", "modules", "attention.py"))
mod_a = importlib.util.module_from_spec(spec_a)
sys.modules["wan.modules.attention"] = mod_a
spec_a.loader.exec_module(mod_a)
spec_m = importlib.util.spec_from_file_location(
    "wan.modules.model", os.path.join(WAN_ROOT, "wan", "modules", "model.py"))
mod_m = importlib.util.module_from_spec(spec_m)
sys.modules["wan.modules.model"] = mod_m
spec_m.loader.exec_module(mod_m)
WanModel = mod_m.WanModel


def _sdpa_flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                          softmax_scale=None, q_scale=None, causal=False,
                          window_size=(-1, -1), deterministic=False,
                          dtype=torch.bfloat16, version=None):
    out_dtype = q.dtype
    q = q.to(dtype); k = k.to(dtype); v = v.to(dtype)
    if q_scale is not None:
        q = q * q_scale
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    attn_mask = None
    if k_lens is not None:
        b, _, lk, _ = kt.shape
        mask = torch.zeros(b, 1, 1, lk, dtype=torch.bool, device=kt.device)
        for i, kl in enumerate(k_lens.tolist()):
            mask[i, :, :, int(kl):] = True
        if mask.any():
            attn_mask = torch.zeros(b, 1, 1, lk, dtype=qt.dtype, device=kt.device)
            attn_mask.masked_fill_(mask, float("-inf"))
    o = torch.nn.functional.scaled_dot_product_attention(
        qt, kt, vt, attn_mask=attn_mask, dropout_p=0.0, scale=softmax_scale)
    o = o.transpose(1, 2).contiguous()
    return o.to(out_dtype)

mod_m.flash_attention = _sdpa_flash_attention

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-bf16"
DEV = "cuda"

# LARGE grid: F=4 frames (patch_f=1), H=64, W=64 latent -> patch (1,2,2) ->
# grid (4,32,32) -> S = 4*32*32 = 4096 tokens (multi-block; math-mode [1,24,
# 4096,4096] f32 scores ~1.6 GB/block would OOM at head_dim=128).
IN_DIM = 48
F_LAT, H_LAT, W_LAT = 4, 64, 64
CTX_LEN = 12
TEXT_DIM = 4096
TIMESTEP = 500.0

def dump(name, t):
    a = t.detach().to(torch.float32).cpu().contiguous().numpy().astype(np.float32)
    a.tofile(os.path.join(HERE, name + ".bin"))
    with open(os.path.join(HERE, name + ".shape"), "w") as f:
        f.write(" ".join(str(d) for d in a.shape))
    return a

def main():
    cfg = dict(model_type="ti2v", patch_size=(1, 2, 2), text_len=512,
               in_dim=IN_DIM, dim=3072, ffn_dim=14336, freq_dim=256,
               text_dim=TEXT_DIM, out_dim=IN_DIM, num_heads=24, num_layers=30,
               qk_norm=True, cross_attn_norm=True, eps=1e-6)
    model = WanModel(**cfg)

    from safetensors.torch import load_file
    sd = {}
    import glob
    for shard in sorted(glob.glob(os.path.join(CKPT, "*.safetensors"))):
        sd.update(load_file(shard))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    miss_real = [k for k in missing if k != "freqs"]
    print("missing (non-freqs):", miss_real)
    assert not miss_real, f"missing weights: {miss_real}"
    del sd

    model = model.to(torch.bfloat16).eval()
    model = model.to(DEV)

    g = torch.Generator(device="cpu").manual_seed(1234)
    x_lat = torch.randn(IN_DIM, F_LAT, H_LAT, W_LAT, generator=g, dtype=torch.float32)
    ctx = torch.randn(CTX_LEN, TEXT_DIM, generator=g, dtype=torch.float32) * 0.5

    x_in = [x_lat.to(DEV).to(torch.bfloat16)]
    ctx_in = [ctx.to(DEV).to(torch.bfloat16)]
    t = torch.tensor([TIMESTEP], device=DEV, dtype=torch.float32)

    F_p = F_LAT // 1; H_p = H_LAT // 2; W_p = W_LAT // 2
    n_patches = F_p * H_p * W_p
    seq_len = n_patches

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x_in, t, ctx_in, seq_len=seq_len)
    full_out = out[0]

    dump("wan22_full_x_in_large", x_lat)
    dump("wan22_full_context_raw_large", ctx)
    dump("wan22_full_out_large", full_out)

    with open(os.path.join(HERE, "wan22_grid_large.txt"), "w") as f:
        f.write(f"F={F_p} H={H_p} W={W_p} seq_len={seq_len} n_patches={n_patches} "
                f"text_len=512 ctx_len={CTX_LEN} dim=3072 nheads=24 headdim=128 "
                f"in_dim={IN_DIM} ffn=14336 timestep={TIMESTEP}\n")
        f.write(f"axes_dims=[44,42,42] theta=10000\n")
        f.write(f"latent F={F_LAT} H={H_LAT} W={W_LAT}\n")

    print("seq_len", seq_len, "full_out", tuple(full_out.shape))
    print("OK")

if __name__ == "__main__":
    main()
