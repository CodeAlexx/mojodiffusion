"""Gate for the `convrot_w8_v1` offline packer. CPU only. Run:
    python -m squareq.selftest_convrot [--h3 /path/to/H3/transformer]

Gates (all must PASS):
  1. H_256 @ H_256 == I in float64; rht_grouped == right-multiply by the
     explicit block-diagonal H_bd; rht_grouped is self-inverse
  2. x @ W^T == (x @ H_bd) @ Rrot^T — the identity the Mojo runtime uses to
     rotate activations instead of un-rotating weights
  3. quant_int8_perchannel round trip: symmetric range, exact dequant identity,
     cos(W_hat, W) >= 0.999 on structured-random and on REAL H3 tensors
  4. GO/NO-GO — rotated int8 beats plain per-CHANNEL int8 by >= GO_SNR_DB on
     EVERY layer family (qkv_proj, out_proj, fc1, fc2) across several blocks,
     with per-tensor int8 as a third reference point and the input-channel
     outlier ratio reported alongside (it predicts whether H3 has the disease
     the rotation cures)
  5. chunked quantize_layer_convrot == whole-tensor composition, bit-identical
  6. peak RSS < 8 GB
"""

import argparse
import resource
import sys

import torch

from . import core
from .source import open_source

DEFAULT_H3 = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/transformer"
FAMILIES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
DEFAULT_BLOCKS = (0, 25, 49)

# GO/NO-GO on the rotation itself (set by the team lead from the OneTrainer
# prior art: dxqb's PR #1632 W8A8 ConvRot on Qwen was "not proven useful").
# NOTE ON THE COS BAR: at int8 weight precision cos is already ~0.9999, so a
# +0.0005 cos improvement is unreachable by ANY int8 method — the bar would
# fail a working rotation. cos ~= 1 - err^2/(2||W||^2), so it compresses the
# whole int8 error range into the 5th decimal. SNR is the discriminating
# metric and the verdict is taken on it; the cos delta is reported alongside.
GO_SNR_DB = 1.5
GO_COS_DELTA = 0.0005
FAILS = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def _snr_db(ref: torch.Tensor, hat: torch.Tensor) -> float:
    r = ref.flatten().double()
    e = (hat.flatten().double() - r)
    return float(20.0 * torch.log10(r.norm() / (e.norm() + 1e-300)))


def _int8_perrow_plain(w: torch.Tensor) -> torch.Tensor:
    """The baseline convrot is measured against: per-output-row absmax int8 on
    the UNROTATED weight, same 1 byte/param budget, same scale count."""
    x = w.float()
    s = (x.abs().amax(1, keepdim=True) / 127.0).clamp(min=1e-30)
    return torch.clamp(torch.round(x / s), -127, 127) * s


def analyze(w: torch.Tensor, x: torch.Tensor, hblock: int = core.HBLOCK) -> dict:
    """Everything the go/no-go needs for one weight, accumulated over row
    chunks in float64 so mlp.fc1 [28672, 5376] costs the same RSS as a small
    one. Rows are independent under the rotation, the per-row scale, and
    y = x @ W^T (a W row-chunk is a y column-chunk), so chunking changes
    nothing but the summation order.

    Outlier statistics, both predictors of whether rotation can help:
      col_ratio  max_c(colmax) / median_c(colmax), colmax[c] = max_r |W[r,c]|.
                 The LRQ-DiT-style INPUT-channel outlier ratio. Big means a few
                 input channels dominate, which is the disease rotation cures.
      row_spike  mean over rows of max_c|W[r,c]| / median_c|W[r,c]|. This is
                 what per-row absmax int8 actually wastes its grid on, measured
                 before and after the rotation."""
    out_f, in_f = w.shape
    rows = max(1, min(out_f, (32 << 20) // in_f))

    # pass 1: global absmax (per-tensor baseline) + per-input-channel maxima
    amax = 0.0
    colmax = torch.zeros(in_f)
    for i in range(0, out_f, rows):
        c = w[i : i + rows].float().abs()
        amax = max(amax, float(c.max()))
        colmax = torch.maximum(colmax, c.amax(0))
        del c
    s_tns = max(amax / 127.0, 1e-30)

    acc = dict.fromkeys(
        ("w2", "e_rot", "e_row", "e_tns", "y2", "ey_rot", "ey_row",
         "d_rot", "h2_rot", "d_row", "h2_row", "spike_pre", "spike_post"), 0.0
    )
    xd = x.double()
    for i in range(0, out_f, rows):
        wc = w[i : i + rows].float()
        rrot = core.rht_grouped(wc.double(), block=hblock).float()
        q, sc = core.quant_int8_perchannel(rrot)
        hat_rot = core.rht_grouped(
            core.dequant_int8_perchannel(q, sc).double(), block=hblock
        ).float()
        hat_row = _int8_perrow_plain(wc)
        hat_tns = torch.clamp(torch.round(wc / s_tns), -127, 127) * s_tns

        wd = wc.double()
        hr, hw = hat_rot.double(), hat_row.double()
        acc["w2"] += float((wd * wd).sum())
        acc["e_rot"] += float(((hr - wd) ** 2).sum())
        acc["e_row"] += float(((hw - wd) ** 2).sum())
        acc["e_tns"] += float(((hat_tns.double() - wd) ** 2).sum())
        acc["d_rot"] += float((hr * wd).sum())
        acc["h2_rot"] += float((hr * hr).sum())
        acc["d_row"] += float((hw * wd).sum())
        acc["h2_row"] += float((hw * hw).sum())

        y_true = xd @ wd.t()
        acc["y2"] += float((y_true * y_true).sum())
        acc["ey_rot"] += float(((xd @ hat_rot.double().t() - y_true) ** 2).sum())
        acc["ey_row"] += float(((xd @ hat_row.double().t() - y_true) ** 2).sum())

        aw = wc.abs()
        ar = rrot.abs()
        acc["spike_pre"] += float(
            (aw.amax(1) / aw.median(1).values.clamp(min=1e-30)).sum()
        )
        acc["spike_post"] += float(
            (ar.amax(1) / ar.median(1).values.clamp(min=1e-30)).sum()
        )
        del wc, rrot, q, sc, hat_rot, hat_row, hat_tns, wd, hr, hw, y_true, aw, ar

    def db(e):  # SNR against the exact weight
        return 20.0 * torch.log10(
            torch.tensor(acc["w2"] ** 0.5 / (e**0.5 + 1e-300))
        ).item()

    def db_y(e):
        return 20.0 * torch.log10(
            torch.tensor(acc["y2"] ** 0.5 / (e**0.5 + 1e-300))
        ).item()

    def cos(d, h2):
        return d / ((h2**0.5) * (acc["w2"] ** 0.5) + 1e-300)

    return {
        "snr_rot": db(acc["e_rot"]), "snr_row": db(acc["e_row"]),
        "snr_tns": db(acc["e_tns"]),
        "snr_y_rot": db_y(acc["ey_rot"]), "snr_y_row": db_y(acc["ey_row"]),
        "cos_rot": cos(acc["d_rot"], acc["h2_rot"]),
        "cos_row": cos(acc["d_row"], acc["h2_row"]),
        "err_drop": 1.0 - acc["e_rot"] / acc["e_row"],
        "col_ratio": float(colmax.max() / colmax.median().clamp(min=1e-30)),
        "spike_pre": acc["spike_pre"] / out_f,
        "spike_post": acc["spike_post"] / out_f,
    }


def t_rotation():
    h = core.hadamard_matrix(256)
    err_i = (h @ h - torch.eye(256, dtype=torch.float64)).abs().max().item()
    check("H256 @ H256 == I (float64)", err_i < 1e-12, f"max|HH-I|={err_i:.3e}")

    # rht_grouped must BE right-multiplication by the block-diagonal H_bd; that
    # equivalence is the contract the Mojo reader's `dequant @ H_bd` relies on.
    torch.manual_seed(0)
    k, blk = 1024, 256
    x = torch.randn(37, k, dtype=torch.float64)
    hbd = torch.zeros(k, k, dtype=torch.float64)
    for i in range(k // blk):
        hbd[i * blk : (i + 1) * blk, i * blk : (i + 1) * blk] = h
    err_bd = (core.rht_grouped(x, block=blk) - x @ hbd).abs().max().item()
    check("rht_grouped == x @ H_bd", err_bd < 1e-13, f"max err={err_bd:.3e}")

    err_inv = (core.rht_grouped(core.rht_grouped(x)) - x).abs().max().item()
    check("rht_grouped self-inverse", err_inv < 1e-12, f"max err={err_inv:.3e}")


def t_gemm_identity():
    """The identity the Mojo runtime is built on. Storing Rrot = W @ H_bd means

        x @ W^T = x @ (Rrot @ H_bd)^T = (x @ H_bd) @ Rrot^T     (H_bd symmetric)

    so the runtime never un-rotates the weight: it rotates the ACTIVATION by the
    same block-diagonal H_bd and feeds the stored int8 straight to the GEMM.
    That is also where the rotation earns most of its accuracy — the activation
    side — which is the runtime half's number to measure, not this packer's."""
    torch.manual_seed(3)
    w = torch.randn(96, 512, dtype=torch.float64) * 0.02
    x = torch.randn(17, 512, dtype=torch.float64)
    rrot = core.rht_grouped(w)
    err = (core.rht_grouped(x) @ rrot.t() - x @ w.t()).abs().max().item()
    check("x @ W^T == (x @ H_bd) @ Rrot^T", err < 1e-12, f"max err={err:.3e}")


def t_quant_synthetic():
    torch.manual_seed(0)
    # Structured (low-rank + noise + a few row outliers) — pure randn has no
    # outliers, which is exactly what a rotation is for, so it would flatter
    # the unrotated baseline.
    w = (torch.randn(1024, 64) @ torch.randn(64, 2048)) * 0.01 + torch.randn(1024, 2048) * 0.004
    w[torch.arange(0, 1024, 7), torch.randint(0, 2048, (147,))] *= 40.0

    rrot = core.rht_grouped(w.double()).float()
    q, s = core.quant_int8_perchannel(rrot)
    check("int8 range symmetric [-127,127]",
          int(q.min()) >= -127 and int(q.max()) <= 127,
          f"min={int(q.min())} max={int(q.max())}")
    check("wscale dtype f32 / shape [out]",
          s.dtype == torch.float32 and tuple(s.shape) == (w.shape[0],),
          f"{s.dtype} {tuple(s.shape)}")
    d = core.dequant_int8_perchannel(q, s)
    check("dequant == q * scale exactly",
          bool(torch.equal(d, q.float() * s.unsqueeze(1))))

    w_hat = core.reconstruct_weight_convrot(q, s)
    cos = _cos(w_hat, w)
    check("synthetic cos(W_hat, W) >= 0.999", cos >= 0.999, f"cos={cos:.6f}")


def t_chunk_identity():
    torch.manual_seed(1)
    w = torch.randn(3000, 512) * 0.02
    tensors, stats = core.quantize_layer_convrot(w)
    rrot = core.rht_grouped(w.double()).float()
    q_ref, s_ref = core.quant_int8_perchannel(rrot)
    same = bool(torch.equal(tensors["qweight"], q_ref)) and bool(
        torch.equal(tensors["wscale"], s_ref)
    )
    check("chunked encode bit-identical to whole-tensor", same,
          f"rows={w.shape[0]} cos_w={stats['cos_w']:.6f}")
    check("stats.rank == 0 (no low-rank branch)", stats["rank"] == 0)
    check("bytes_q == out*in + 4*out",
          stats["bytes_q"] == w.numel() + 4 * w.shape[0],
          f"{stats['bytes_q']} ({stats['bytes_q'] / stats['bytes_bf16']:.3f}x bf16)")


def t_real(path: str, blocks):
    """GO/NO-GO: does rotating actually buy anything on H3's OWN weights?
    One tensor of each family per block, so a marginal family cannot hide
    behind a good one."""
    try:
        src = open_source(path)
    except Exception as e:  # noqa: BLE001
        check("H3 tensors available", False, f"{e}")
        return
    per_family: dict = {}
    worst_cos = 1.0
    torch.manual_seed(0)
    with src as f:
        avail = set(f.keys())
        picks = [f"blocks.{b}.{fam}.weight" for b in blocks for fam in FAMILIES]
        picks = [k for k in picks if k in avail]
        if not picks:
            check("H3 tensors available", False, f"no block tensors in {path}")
            return
        print(f"    {'tensor':34s} {'shape':14s} {'SNRw rot/row/tns':>22s}"
              f" {'gain':>7s} {'cos rot/row':>21s} {'dcos':>9s}"
              f" {'colratio':>9s} {'spike pre>post':>15s}")
        for k in picks:
            w = f.get_tensor(k).float()
            out_f, in_f = w.shape
            x = torch.randn(64, in_f)
            r = analyze(w, x)
            fam = k.split(".", 2)[2].rsplit(".", 1)[0]
            per_family.setdefault(fam, []).append(r)
            worst_cos = min(worst_cos, r["cos_rot"])
            print(
                f"    {k:34s} {f'{out_f}x{in_f}':14s}"
                f" {r['snr_rot']:6.2f}/{r['snr_row']:6.2f}/{r['snr_tns']:6.2f}"
                f" {r['snr_rot'] - r['snr_row']:+6.2f}"
                f"  {r['cos_rot']:.6f}/{r['cos_row']:.6f}"
                f" {r['cos_rot'] - r['cos_row']:+.6f}"
                f" {r['col_ratio']:8.2f}x"
                f" {r['spike_pre']:6.1f}>{r['spike_post']:5.1f}"
            )
            del w, x
    print()
    gains = []
    for fam in FAMILIES:
        rs = per_family.get(fam)
        if not rs:
            continue
        g = sum(r["snr_rot"] - r["snr_row"] for r in rs) / len(rs)
        gy = sum(r["snr_y_rot"] - r["snr_y_row"] for r in rs) / len(rs)
        dc = sum(r["cos_rot"] - r["cos_row"] for r in rs) / len(rs)
        ed = sum(r["err_drop"] for r in rs) / len(rs)
        cr = sum(r["col_ratio"] for r in rs) / len(rs)
        sp = sum(r["spike_pre"] for r in rs) / len(rs)
        sq = sum(r["spike_post"] for r in rs) / len(rs)
        gains.append(g)
        print(f"    FAMILY {fam:16s} n={len(rs)}  gain {g:+.2f} dB weight /"
              f" {gy:+.2f} dB output  dcos {dc:+.6f}"
              f"  err-energy -{100 * ed:.0f}%"
              f"  col_ratio {cr:.1f}x  row_spike {sp:.1f} -> {sq:.1f}")
    print()
    check("real-weight cos(W_hat, W) >= 0.999", worst_cos >= 0.999,
          f"worst={worst_cos:.6f}")
    check(f"GO/NO-GO: every family gains >= {GO_SNR_DB} dB vs plain per-channel int8",
          bool(gains) and min(gains) >= GO_SNR_DB,
          f"worst family {min(gains):+.2f} dB, mean {sum(gains) / len(gains):+.2f} dB"
          if gains else "no families measured")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h3", default=DEFAULT_H3)
    ap.add_argument("--blocks", default=",".join(str(b) for b in DEFAULT_BLOCKS),
                    help="block indices to sweep; every family is measured in each")
    args = ap.parse_args()
    print("convrot_w8_v1 selftest (CPU)")
    t_rotation()
    t_gemm_identity()
    t_quant_synthetic()
    t_chunk_identity()
    t_real(args.h3, [int(b) for b in args.blocks.split(",") if b.strip()])
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    check("peak RSS < 8 GB", rss_gb < 8.0, f"rss={rss_gb:.2f} GB")
    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
