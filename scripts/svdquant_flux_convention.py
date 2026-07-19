#!/usr/bin/env python3
# svdquant_flux_convention.py — determine the REAL SVDQuant int4 convention by
# reconstructing a nunchaku class-A linear and matching it against the exact
# bf16 ground-truth weight it was quantized from (flux1-dev).
#
# SVDQuant stores a SMOOTHED, int4+low-rank decomposition of each linear:
#     W_smooth ≈ dequant4(qweight, wscales) + lora_up @ lora_down^T
# and smoothing migrates a per-input-channel scale between activation and
# weight. So the ORIGINAL weight is recovered by folding `smooth` back in along
# the input dim (direction unknown). We sweep:
#     SIGN_MODE    ∈ {twos_complement, offset8}
#     NIBBLE_ORDER ∈ {lo_even, hi_even}
#     smooth fold  ∈ {none, *smooth, /smooth, *smooth_orig, /smooth_orig}
# = 20 cheap tries per layer, and report cos vs the bf16 ground truth. The
# correct convention is the one hitting cos ~0.98–0.999 (int4+rank32 closely
# approximates the original). If nothing clears 0.9, the class-A qweight is
# likely Marlin-permuted (not plain [out,in/2] row-major) — we STOP and print
# the full table rather than hardcode past it.
#
# Ground-truth key map (nunchaku diffusers-ish -> BFL flux1-dev), all 1:1 [out,in]:
#   transformer_blocks.0.out_proj  -> double_blocks.0.img_attn.proj.weight [3072,3072]
#   transformer_blocks.0.qkv_proj  -> double_blocks.0.img_attn.qkv.weight  [9216,3072]
#   transformer_blocks.0.mlp_fc1   -> double_blocks.0.img_mlp.0.weight     [12288,3072]
#
# Run: /home/alex/serenityflow-v2/.venv/bin/python scripts/svdquant_flux_convention.py

import torch
from safetensors import safe_open

INT4 = "/home/alex/.serenity/models/checkpoints/nunchaku/svdq-int4_r32-flux.1-dev.safetensors"
BF16 = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"

# nunchaku prefix -> BFL ground-truth weight key
LAYERS = {
    "transformer_blocks.0.out_proj": "double_blocks.0.img_attn.proj.weight",
    "transformer_blocks.0.qkv_proj": "double_blocks.0.img_attn.qkv.weight",
    "transformer_blocks.0.mlp_fc1":  "double_blocks.0.img_mlp.0.weight",
}

SIGN_MODES = ["twos_complement", "offset8"]
NIBBLE_ORDERS = ["lo_even", "hi_even"]


def dequant4(qweight_i8, wscales_bf16, sign_mode, nibble_order):
    """qweight I8 [out, in/2] -> W_q float [out, in]. wscales bf16 [in/64, out]."""
    out, half = qweight_i8.shape
    in_f = half * 2
    u8 = (qweight_i8.to(torch.int64) & 0xFF)                 # [out, half]
    lo = u8 & 0xF
    hi = (u8 >> 4) & 0xF
    if nibble_order == "lo_even":
        even_nib, odd_nib = lo, hi
    else:  # hi_even
        even_nib, odd_nib = hi, lo
    nib = torch.zeros(out, in_f, dtype=torch.int64)
    nib[:, 0::2] = even_nib
    nib[:, 1::2] = odd_nib
    if sign_mode == "twos_complement":
        q = torch.where(nib >= 8, nib - 16, nib)
    else:  # offset8
        q = nib - 8
    groups = wscales_bf16.shape[0]                           # in/64
    assert groups == in_f // 64, f"wscales groups {groups} != in/64 {in_f//64}"
    # wscales [in/64, out] -> scale_full[o, k] = wscales[k//64, o]
    scale = wscales_bf16.float().t().contiguous()            # [out, in/64]
    scale_full = scale.repeat_interleave(64, dim=1)          # [out, in]
    return q.float() * scale_full


def cos(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm())).item()


def main():
    torch.set_grad_enabled(False)
    fi = safe_open(INT4, framework="pt", device="cpu")
    fb = safe_open(BF16, framework="pt", device="cpu")

    best_global = None
    for pfx, gt_key in LAYERS.items():
        qweight = fi.get_tensor(pfx + ".qweight")           # I8 [out, in/2]
        wscales = fi.get_tensor(pfx + ".wscales")           # BF16 [in/64, out]
        lora_down = fi.get_tensor(pfx + ".lora_down").float()  # [in, rank]
        lora_up = fi.get_tensor(pfx + ".lora_up").float()      # [out, rank]
        smooth = fi.get_tensor(pfx + ".smooth").float()        # [in]
        smooth_orig = fi.get_tensor(pfx + ".smooth_orig").float()
        W_orig = fb.get_tensor(gt_key).float()                 # [out, in]

        out, in_f = W_orig.shape
        L = lora_up @ lora_down.t()                            # [out, in] low-rank
        assert L.shape == W_orig.shape, (L.shape, W_orig.shape)

        folds = {
            "none": lambda W: W,
            "*smooth": lambda W: W * smooth[None, :],
            "/smooth": lambda W: W / smooth[None, :],
            "*smooth_orig": lambda W: W * smooth_orig[None, :],
            "/smooth_orig": lambda W: W / smooth_orig[None, :],
        }

        print(f"\n================ {pfx}  ->  {gt_key}   W{list(W_orig.shape)} ================")
        print(f"{'sign':16s} {'nibble':8s} {'fold':14s}  cos_vs_ground_truth")
        rows = []
        for sm in SIGN_MODES:
            for no in NIBBLE_ORDERS:
                Wq = dequant4(qweight, wscales, sm, no)        # [out, in]
                W_recon = Wq + L                               # smoothed-space recon
                for fname, ffn in folds.items():
                    c = cos(ffn(W_recon), W_orig)
                    rows.append((c, sm, no, fname))
        rows.sort(key=lambda r: -r[0])
        for c, sm, no, fname in rows:
            mark = "   <-- BEST" if (c, sm, no, fname) == rows[0] else ""
            print(f"{sm:16s} {no:8s} {fname:14s}  {c:+.6f}{mark}")
        top = rows[0]
        print(f"  WINNER[{pfx}]: sign={top[1]} nibble={top[2]} fold={top[3]} cos={top[0]:.6f}")
        if best_global is None:
            best_global = top

    # Cross-layer agreement on (sign, nibble). fold direction is the same too.
    print("\n================ SUMMARY ================")
    # recompute winners per layer for the summary
    summary = []
    for pfx, gt_key in LAYERS.items():
        qweight = fi.get_tensor(pfx + ".qweight")
        wscales = fi.get_tensor(pfx + ".wscales")
        lora_down = fi.get_tensor(pfx + ".lora_down").float()
        lora_up = fi.get_tensor(pfx + ".lora_up").float()
        smooth = fi.get_tensor(pfx + ".smooth").float()
        smooth_orig = fi.get_tensor(pfx + ".smooth_orig").float()
        W_orig = fb.get_tensor(gt_key).float()
        L = lora_up @ lora_down.t()
        folds = {
            "none": lambda W: W,
            "*smooth": lambda W: W * smooth[None, :],
            "/smooth": lambda W: W / smooth[None, :],
            "*smooth_orig": lambda W: W * smooth_orig[None, :],
            "/smooth_orig": lambda W: W / smooth_orig[None, :],
        }
        best = (-2, None, None, None)
        for sm in SIGN_MODES:
            for no in NIBBLE_ORDERS:
                Wq = dequant4(qweight, wscales, sm, no)
                for fname, ffn in folds.items():
                    c = cos(ffn(Wq + L), W_orig)
                    if c > best[0]:
                        best = (c, sm, no, fname)
        summary.append((pfx, best))
        print(f"  {pfx:38s} sign={best[1]:16s} nibble={best[2]:8s} fold={best[3]:14s} cos={best[0]:.6f}")

    signs = set(b[1] for _, b in summary)
    nibs = set(b[2] for _, b in summary)
    folds_ = set(b[3] for _, b in summary)
    allcos = [b[0] for _, b in summary]
    print(f"\n  agreement: SIGN={signs} NIBBLE={nibs} FOLD={folds_}")
    print(f"  min cos across layers = {min(allcos):.6f}")
    if min(allcos) < 0.9:
        print("\n  *** WALL: no combo clears 0.9 on all layers — qweight is likely")
        print("  *** Marlin-permuted / not plain [out,in/2] row-major, OR a wscales")
        print("  *** transpose. STOP — do not hardcode. Report the table above.")
    elif len(signs) == 1 and len(nibs) == 1:
        print(f"\n  >>> CONVENTION CONFIRMED: SIGN_MODE={signs.pop()} "
              f"NIBBLE_ORDER={nibs.pop()} smooth_fold={folds_}")
    else:
        print("\n  *** layers disagree on sign/nibble — investigate before setting defaults.")


if __name__ == "__main__":
    main()
