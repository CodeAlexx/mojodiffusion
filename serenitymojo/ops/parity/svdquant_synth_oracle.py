#!/usr/bin/env python3
# svdquant_synth_oracle.py — synthetic, self-consistent fixture + reference for
# the SVDQuant Class-A int4 linear (ops/svdquant.mojo).
#
# NO real checkpoint needed. We build a known BF16 weight W [out, in], quantize
# it with the SAME convention the Mojo kernel decodes (group-64 symmetric int4,
# per-group-per-output scale = max|W_group| / 7, round, clamp to [-7,7], pack
# two nibbles/byte along the input dim), and also emit random x / lora_down /
# lora_up / smooth / bias. We compute the reference y (and the dequantized ref
# W) in torch and dump everything to a .safetensors the Mojo gate reads back.
#
# Convention MUST match ops/svdquant.mojo defaults:
#   SIGN_MODE    = twos_complement  (nibble n<8 → n, n>=8 → n-16)
#   NIBBLE_ORDER = lo_even          (low nibble → even input index)
# Flip SIGN_MODE / NIBBLE_ORDER here in lock-step with the Mojo comptimes.
#
# Faithfulness: the dequant (both sides) uses the STORED bf16 scale, so W
# reconstruction is essentially exact. The reference y rounds the same
# intermediates Mojo stores in BF16 (W, xs, down) so the cos bar (>=0.999) has
# comfortable headroom. Non-degenerate randn data only (never modular fills).
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python \
#         serenitymojo/ops/parity/svdquant_synth_oracle.py

import torch
from safetensors.torch import save_file

# ── Convention flags (keep in lock-step with ops/svdquant.mojo) ───────────────
SIGN_MODE = "twos_complement"   # or "offset8"
NIBBLE_ORDER = "lo_even"        # or "hi_even"

OUT = 256      # output features (even)
IN = 192       # input features (even, multiple of 64)
RANK = 32
M = 4          # activation rows
GROUP = 64

FIXTURE = "/home/alex/mojodiffusion/serenitymojo/ops/parity/svdq_synth_fixture.safetensors"


def main():
    torch.manual_seed(0)
    assert IN % 2 == 0 and IN % GROUP == 0 and OUT % 2 == 0
    groups = IN // GROUP

    # ── Known dense weight, then group-64 symmetric int4 quantize ────────────
    W = (torch.randn(OUT, IN) * 0.5).float()

    # Per-(output, group) scale = max|W_group| / 7  (symmetric, avoids -8).
    Wg = W.view(OUT, groups, GROUP)
    scale_true = Wg.abs().amax(dim=2) / 7.0            # [OUT, groups]
    scale_true = torch.clamp(scale_true, min=1e-8)

    # Store scale as BF16 (this is what both sides dequant with).
    wscales_bf16 = scale_true.to(torch.bfloat16)       # [OUT, groups]
    scale_stored = wscales_bf16.float()                # [OUT, groups]

    # Quantize with the true scale, dequant with the stored bf16 scale.
    q = torch.round(W / scale_true.repeat_interleave(GROUP, dim=1))
    q = torch.clamp(q, -7, 7).to(torch.int64)          # [OUT, IN], ints in [-7,7]
    scale_full = scale_stored.repeat_interleave(GROUP, dim=1)   # [OUT, IN]
    W_deq = (q.float() * scale_full)                   # dequantized, f32
    W_deq_bf16 = W_deq.to(torch.bfloat16)              # matches Mojo BF16 W

    # ── Pack q into nibbles: qweight I8 [OUT, IN/2] ──────────────────────────
    # twos_complement nibble = q & 0xF  (−1→15, −7→9, ...). offset8 nibble = q+8.
    if SIGN_MODE == "twos_complement":
        nib = (q & 0xF).to(torch.int64)                # [OUT, IN]
    elif SIGN_MODE == "offset8":
        nib = (q + 8).to(torch.int64)
    else:
        raise ValueError(SIGN_MODE)

    half = IN // 2
    qbytes = torch.zeros(OUT, half, dtype=torch.int64)
    even = nib[:, 0::2]   # input indices 0,2,4,...  -> [OUT, half]
    odd = nib[:, 1::2]    # input indices 1,3,5,...  -> [OUT, half]
    if NIBBLE_ORDER == "lo_even":
        packed = (even & 0xF) | ((odd & 0xF) << 4)
    elif NIBBLE_ORDER == "hi_even":
        packed = (odd & 0xF) | ((even & 0xF) << 4)
    else:
        raise ValueError(NIBBLE_ORDER)
    qbytes = packed
    # Store as signed int8 (I8) — reinterpret the 0..255 byte as two's-comp int8.
    qbyte_u8 = (qbytes & 0xFF).to(torch.int64)
    qbyte_i8 = torch.where(qbyte_u8 >= 128, qbyte_u8 - 256, qbyte_u8).to(torch.int8)

    # wscales on disk is [in/64, out] (group-major). We computed [OUT, groups];
    # transpose to [groups, OUT].
    wscales_disk = wscales_bf16.t().contiguous()       # [groups, OUT], bf16

    # ── Low-rank + smoothing + bias + activation ─────────────────────────────
    lora_down = (torch.randn(IN, RANK) * 0.1).to(torch.bfloat16)     # [in, rank]
    lora_up = (torch.randn(OUT, RANK) * 0.1).to(torch.bfloat16)      # [out, rank]
    smooth = (torch.randn(IN) * 0.2 + 1.0).to(torch.bfloat16)        # [in]
    bias = (torch.randn(OUT) * 0.1).to(torch.bfloat16)               # [out]
    x = (torch.randn(M, IN) * 1.0).to(torch.bfloat16)                # [M, in]

    # ── Reference y (mirrors Mojo storage: xs/W/down rounded to BF16) ────────
    xf = x.float()
    sf = smooth.float()
    xs = (xf * sf).to(torch.bfloat16).float()                        # smooth kernel
    Wb = W_deq_bf16.float()                                          # [out, in]
    main = xs @ Wb.t()                                               # [M, out]
    down = (xf @ lora_down.float()).to(torch.bfloat16).float()       # [M, rank]
    up = down @ lora_up.float().t()                                  # [M, out]
    ref_y = (main + up + bias.float())                              # [M, out], f32
    ref_W = Wb                                                       # [out, in], f32

    tensors = {
        "qweight": qbyte_i8,                 # I8   [out, in/2]
        "wscales": wscales_disk,             # BF16 [in/64, out]
        "lora_down": lora_down,              # BF16 [in, rank]
        "lora_up": lora_up,                  # BF16 [out, rank]
        "smooth": smooth,                    # BF16 [in]
        "bias": bias,                        # BF16 [out]
        "x": x,                              # BF16 [M, in]
        "ref_y": ref_y.contiguous(),         # F32  [M, out]
        "ref_W": ref_W.contiguous(),         # F32  [out, in]
    }
    save_file(tensors, FIXTURE)
    print(f"wrote {FIXTURE}")
    print(f"  SIGN_MODE={SIGN_MODE} NIBBLE_ORDER={NIBBLE_ORDER}")
    print(f"  OUT={OUT} IN={IN} RANK={RANK} M={M} groups={groups}")
    print(f"  qweight {tuple(qbyte_i8.shape)} i8   |q| range [{int(q.min())},{int(q.max())}]")
    print(f"  wscales {tuple(wscales_disk.shape)} bf16")
    print(f"  ref_W   {tuple(ref_W.shape)} f32  ref_y {tuple(ref_y.shape)} f32")
    # Sanity: reconstruct from the packed bytes here (mirror the Mojo decode).
    _self_check(qbyte_i8, scale_stored, W_deq_bf16, groups)


def _self_check(qbyte_i8, scale_stored, W_deq_bf16, groups):
    """Decode the packed bytes back the way Mojo will, and confirm it matches
    W_deq — catches packing/order bugs before the Mojo gate even runs."""
    u8 = (qbyte_i8.to(torch.int64) & 0xFF)              # [OUT, half]
    if NIBBLE_ORDER == "lo_even":
        lo = u8 & 0xF
        hi = (u8 >> 4) & 0xF
    else:
        hi = u8 & 0xF
        lo = (u8 >> 4) & 0xF
    even = lo
    odd = hi
    OUT = qbyte_i8.shape[0]
    IN = qbyte_i8.shape[1] * 2
    nib = torch.zeros(OUT, IN, dtype=torch.int64)
    nib[:, 0::2] = even
    nib[:, 1::2] = odd
    if SIGN_MODE == "twos_complement":
        qv = torch.where(nib >= 8, nib - 16, nib)
    else:
        qv = nib - 8
    scale_full = scale_stored.repeat_interleave(IN // groups, dim=1)
    rec = (qv.float() * scale_full).to(torch.bfloat16).float()
    err = (rec - W_deq_bf16.float()).abs().max().item()
    print(f"  self-check max|rec - W_deq| = {err:.3e} (expect 0.0)")
    assert err == 0.0, "packer self-check FAILED — packing/convention bug"


if __name__ == "__main__":
    main()
