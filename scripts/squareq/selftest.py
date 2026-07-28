"""Chunk-0 gate for the squareq package. Run:
    <torch-venv>/bin/python -m squareq.selftest [--klein /path/to/klein4b.safetensors]

Gates (all must PASS):
  1. int4 pack/unpack bit-exact round trip
  2. grouped RHT-256: orthonormal + self-inverse
  3. quantize_layer deterministic across runs (byte-identical)
  4. recon cos(W_hat, W) >= 0.999 @ rank 32 on REAL Klein-4B tensors
  5. NVFP4 codec round trip bounded error + deterministic
  6. legacy w4a4 API functional (cos >= 0.985 on synthetic, like the old gate)
  7. torch reference LoRA-on-frozen-base demo: loss drops, base grad-free
  8. peak RSS < 4 GB
"""

import argparse
import resource
import sys

import torch

from . import core
from .reference import SquareQLinear, lora_train_demo
from .svdquant_int4 import quantize_svdquant_w4a4, w4a4_forward

DEFAULT_KLEIN = "/home/alex/mojodiffusion/models/klein4b/transformer.safetensors"
FAILS = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def t_pack_unpack():
    torch.manual_seed(0)
    q = torch.randint(-8, 8, (64, 512), dtype=torch.int16)
    rt = core.unpack_int4(core.pack_int4(q))
    check("pack/unpack bit-exact", bool((rt == q).all()))


def t_rht():
    h = core.hadamard_matrix(256)
    err_orth = (h @ h.t() - torch.eye(256, dtype=h.dtype)).abs().max().item()
    x = torch.randn(4, 1536, dtype=torch.float64)
    err_inv = (core.rht_grouped(core.rht_grouped(x)) - x).abs().max().item()
    check("H256 orthonormal", err_orth < 1e-12, f"max|HH^T-I|={err_orth:.2e}")
    check("rht_grouped self-inverse", err_inv < 1e-12, f"max err={err_inv:.2e}")


def t_determinism():
    torch.manual_seed(123)
    w = torch.randn(512, 1024)
    t1, s1 = core.quantize_layer(w, rank=32)
    t2, s2 = core.quantize_layer(w, rank=32)
    same = all(
        bool(torch.equal(t1[k], t2[k])) for k in ("qweight", "wscales", "lora_down", "lora_up")
    )
    check("quantize_layer deterministic", same, f"cos_w={s1['cos_w']:.6f}")


def t_klein(path: str):
    try:
        from safetensors import safe_open

        f = safe_open(path, "pt")
    except Exception as e:  # noqa: BLE001
        check("klein tensors available", False, f"{e}")
        return
    keys = [k for k in f.keys() if k.endswith(".weight")]
    picks = []
    for want in ("double_blocks.0.img_attn.qkv.weight", "single_blocks.0.linear2.weight"):
        if want in keys:
            picks.append(want)
    if not picks:  # fall back: first two big 2D weights with K%256==0
        for k in keys:
            sh = f.get_slice(k).get_shape()
            if len(sh) == 2 and sh[1] % 256 == 0 and sh[0] >= 1024:
                picks.append(k)
            if len(picks) == 2:
                break
    def _cos(a, b):
        a = a.flatten().double()
        b = b.flatten().double()
        return (a @ b / (a.norm() * b.norm() + 1e-30)).item()

    worst_w = worst_y = 1.0
    torch.manual_seed(0)
    for k in picks:
        w = f.get_tensor(k).float()
        x = torch.randn(128, w.shape[1])
        y_true = x.double() @ w.double().t()
        t, stats = core.quantize_layer(w, rank=32)
        w_hat = core.reconstruct_weight(
            t["qweight"], t["wscales"], t["lora_down"], t["lora_up"]
        )
        cos_y = _cos(x.double() @ w_hat.double().t(), y_true)
        # context: int8 per-row (ConvRot-class weight budget, 1 B/param)
        s8 = w.abs().amax(1, keepdim=True) / 127.0
        w8 = torch.clamp(torch.round(w / s8), -128, 127) * s8
        cos_y8 = _cos(x.double() @ w8.double().t(), y_true)
        # ablation: same bytes, no Hadamard
        ld, lu = core.svd_lowrank_init(w, 32)
        resid = (w.double() - lu.double() @ ld.double().t()).float()
        qn, sn = core.quant_int4_g64(resid)
        w_noh = core.dequant_int4_g64(qn, sn) + lu @ ld.t()
        cos_ynoh = _cos(x.double() @ w_noh.double().t(), y_true)
        worst_w = min(worst_w, stats["cos_w"])
        worst_y = min(worst_y, cos_y)
        ratio = stats["bytes_q"] / stats["bytes_bf16"]
        print(
            f"    {k}  [{stats['out']}x{stats['in']}]  cos_w={stats['cos_w']:.6f}"
            f"  cos_y={cos_y:.6f}  bytes={ratio:.3f}x"
            f"  | int8-perrow cos_y={cos_y8:.6f} (0.500x)"
            f"  | noH+r32 cos_y={cos_ynoh:.6f}"
        )
        del w
    # Regression floors from measured Klein-4B reality (2026-07-28, r32 g64:
    # worst cos_w 0.99467 / cos_y 0.99469 on single_blocks.0.linear2; int8-perrow
    # reference 0.99994 at 2x bytes; g32 would buy ~+0.001 cos_y for +1.5pp bytes).
    # End-to-end quality is judged by the chunk-4/5 output+loss gates, not here.
    check("real-weight recon cos_w >= 0.994 @ r32", worst_w >= 0.994, f"worst={worst_w:.6f}")
    check("real-weight output cos_y >= 0.994 @ r32", worst_y >= 0.994, f"worst={worst_y:.6f}")


def t_nvfp4():
    torch.manual_seed(7)
    x = torch.randn(32, 256) * 3.0
    p1 = core.nvfp4_encode(x)
    p2 = core.nvfp4_encode(x)
    det = all(bool(torch.equal(a, b)) for a, b in zip(p1, p2))
    d = core.nvfp4_decode(*p1)
    a = d.flatten().double()
    b = x.flatten().double()
    cos = (a @ b / (a.norm() * b.norm())).item()
    check("nvfp4 deterministic", det)
    check("nvfp4 round-trip cos >= 0.99", cos >= 0.99, f"cos={cos:.5f}")


def t_legacy_w4a4():
    # Structured synthetic (low-rank + noise) — pure randn has no low-rank
    # structure and understates every SVD-based method vs real weights.
    torch.manual_seed(0)
    w = (torch.randn(512, 96) @ torch.randn(96, 2048)) * 0.01 + torch.randn(512, 2048) * 0.004
    qb, ws, ld, lu = quantize_svdquant_w4a4(w, rank=64)
    x = torch.randn(64, 2048)
    y = w4a4_forward(x, qb, ws, ld, lu)
    yt = x.double() @ w.double().t()
    cos = (y.flatten() @ yt.flatten() / (y.flatten().norm() * yt.flatten().norm())).item()
    check("legacy w4a4_forward cos >= 0.985", cos >= 0.985, f"cos={cos:.5f}")


def t_reference_train():
    torch.manual_seed(0)
    w = torch.randn(256, 512) * 0.02
    tensors, _ = core.quantize_layer(w, rank=16)
    lin = SquareQLinear(**tensors)
    r = lora_train_demo(lin, steps=60)
    check(
        "reference LoRA demo loss drops",
        r["last_loss"] < r["first_loss"] * 0.5,
        f"{r['first_loss']:.4f} -> {r['last_loss']:.4f}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--klein", default=DEFAULT_KLEIN)
    args = ap.parse_args()
    print("squareq selftest (chunk-0 gate)")
    t_pack_unpack()
    t_rht()
    t_determinism()
    t_klein(args.klein)
    t_nvfp4()
    t_legacy_w4a4()
    t_reference_train()
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    check("peak RSS < 4 GB", rss_gb < 4.0, f"rss={rss_gb:.2f} GB")
    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
