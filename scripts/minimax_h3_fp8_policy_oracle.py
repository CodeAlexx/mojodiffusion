#!/usr/bin/env python3
"""Oracle for MiniMax-H3 unit 14 — the fp8 residency policy and E4M3 encoder.

Two independent things end up in one reference file because they answer one
question ("does H3 run on the 3090 Ti in fp8?") from two directions:

  A. FOOTPRINT.  Re-parses the converter's own key plan and classifies all 638
     tensors in Python, so the Mojo accounting in fp8_policy.mojo is checked
     against a second implementation rather than against my arithmetic.

     The parse is anchored to ground truth we did not write: the converter
     printed its own totals at the bottom of that file (638 keys, 12 fp32,
     66280430080 bytes) and this script ASSERTS against them before emitting
     anything. A silently drifting regex would fail here rather than quietly
     move the verdict.

  B. E4M3 BYTES.  torch.float8_e4m3fn is the authority for the rounding, and
     the values fed through it are chosen to break a naive encoder:

       * exact ties at every mantissa step (round-half-to-EVEN, not away)
       * the subnormal/normal boundary at 2^-6 and the m==8 promotion
       * the 16 -> 8 mantissa carry at the top of a binade
       * saturation above 448, which E4M3-fn clamps rather than making inf
       * negative zero, which must stay 0x80 and not become 0x00
       * values below half the smallest subnormal, which must flush to zero

     Plus real per-row matrices, because the row scale is where a per-tensor
     scheme and a per-row scheme diverge and H3's projections are wide enough
     for that to matter.

Writes: output/minimax_h3_fp8/fp8_policy_ref.safetensors
        output/minimax_h3_fp8/key_plan.csv    (normalized: key,rows,cols)
"""

import math
import os
import re
import sys

import torch
from safetensors.torch import save_file

PLAN = "/home/alex/minimax_h3_ref/transformer_key_plan.txt"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_fp8"

F32_PREFIXES = (
    "proj_in.",
    "audio_proj_in.",
    "time_embedder.",
    "proj_out.",
    "audio_proj_out.",
)
FP8_MIN_ELEMENTS = 1_000_000
E4M3_MAX = 448.0

F32_KEEP, BF16_KEEP, FP8_ROW, ADALN = 0, 1, 2, 3


# Copied VERBATIM from `modules_to_not_convert` in the reference's own int8
# consumer recipe, PR 14355 commit 80453959c (2026-08-02 15:55Z):
#   proj_in, audio_proj_in, context_embedder, time_embedder, time_proj,
#   token_refiner, norm_out, proj_out, audio_proj_out
# Six are already covered by the fp32 and adaLN rules; these three are not.
VENDOR_PROTECTED = ("context_embedder", "token_refiner", "time_proj")


def is_adaln(key):
    # `norm_out` must be named: it is `final_layer.adaln_proj` in the original
    # checkpoint but diffusers renames it, so a substring test on "adaln" misses
    # 28.9 M evictable parameters.
    # `norm_out.linear`, not `norm_out.`: the sibling norm_out.norm.weight is
    # the RMSNorm gain, applied per token and therefore not evictable.
    return "adaln" in key or key.startswith("norm_out.linear")


def classify(key, rows, cols):
    if key.startswith(F32_PREFIXES):
        return F32_KEEP
    if is_adaln(key):
        return ADALN
    if key.startswith(VENDOR_PROTECTED):
        return BF16_KEEP
    if cols > 0 and rows * cols >= FP8_MIN_ELEMENTS:
        return FP8_ROW
    return BF16_KEEP


def policy_bytes(cls, rows, cols):
    n = rows * cols if cols > 0 else rows
    if cls == F32_KEEP:
        return n * 4
    if cls == FP8_ROW:
        return n + rows * 4  # E4M3 byte per element + f32 scale per output row
    return n * 2


def parse_plan():
    rows = []
    for line in open(PLAN):
        m = re.search(r"->\?\s+(\S+)\s+\[([\d, ]+)\]\s+(F32|BF16)", line)
        if m:
            shape = [int(x) for x in m.group(2).split(",")]
            rows.append((m.group(1), shape, m.group(3)))
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plan = parse_plan()

    # ── anchor the parse to the converter's own printed footer ───────────────
    text = open(PLAN).read()
    want_keys = int(re.search(r"planned diffusers keys:\s*(\d+)", text).group(1))
    want_f32 = int(re.search(r"fp32 diffusers keys\s*:\s*(\d+)", text).group(1))
    want_bytes = int(re.search(r"total output bytes\s*:\s*(\d+)", text).group(1))
    got_bytes = sum(
        math.prod(s) * (4 if d == "F32" else 2) for _, s, d in plan
    )
    assert len(plan) == want_keys, f"parsed {len(plan)} keys, plan says {want_keys}"
    assert sum(1 for _, _, d in plan if d == "F32") == want_f32
    assert got_bytes == want_bytes, f"parsed {got_bytes} bytes, plan says {want_bytes}"
    print(f"parse anchored: {want_keys} keys, {want_f32} fp32, {want_bytes} bytes")

    # ── A. footprint ─────────────────────────────────────────────────────────
    keys_n = [0] * 4
    params = [0] * 4
    pbytes = [0] * 4
    adaln_out_rows = 0
    csv = []
    for key, shape, dtype in plan:
        r = shape[0]
        c = shape[1] if len(shape) == 2 else 0
        cls = classify(key, r, c)
        # The dtype column must agree with the name-based fp32 rule — if it
        # ever does not, the five-prefix claim in the intake is wrong.
        assert (cls == F32_KEEP) == (dtype == "F32"), f"{key}: {dtype} vs class {cls}"
        keys_n[cls] += 1
        params[cls] += math.prod(shape)
        pbytes[cls] += policy_bytes(cls, r, c)
        if cls == ADALN and c > 0:
            adaln_out_rows += r
        csv.append(f"{key},{r},{c}")
    open(f"{OUT_DIR}/key_plan.csv", "w").write("\n".join(csv) + "\n")

    non_adaln = sum(pbytes[i] for i in range(4) if i != ADALN)
    gib = 1024**3
    print()
    names = ["F32_KEEP", "BF16_KEEP", "FP8_ROW", "ADALN"]
    for i in range(4):
        print(
            f"  {names[i]:10s} {keys_n[i]:4d} keys  {params[i]/1e9:8.4f} B params"
            f"  policy {pbytes[i]/gib:7.3f} GiB"
        )
    print()
    print(f"  bf16 everything resident        {got_bytes/gib:7.3f} GiB")
    print(f"  fp8, adaLN kept as weights      {(non_adaln+pbytes[ADALN])/gib:7.3f} GiB")
    # DISTINCT TIMESTEPS, not steps: build_row_timesteps returns torch.unique
    # over {video_t, audio_t, max(video_t, NOISE_AUG), 1.0}, and H3's video and
    # audio run SEPARATE schedules (shift 12.0 vs 3.0) so they never coincide.
    # The two conditioning values are pinned for the whole run.
    for steps in (25, 50):
        for mode, extra in (("t2va", 0), ("fl2va", 1), ("ref2va", 2)):
            distinct = 2 * steps + extra
            cache = adaln_out_rows * distinct * 2
            print(
                f"  fp8, adaLN evicted {mode:6s} @{steps:3d} steps  "
                f"{distinct:4d} distinct  {(non_adaln+cache)/gib:7.3f} GiB "
                f"  (cache {cache/2**20:6.1f} MiB)"
            )

    tensors = {
        "want.keys": torch.tensor(keys_n, dtype=torch.int64),
        "want.params": torch.tensor(params, dtype=torch.int64),
        "want.bytes": torch.tensor(pbytes, dtype=torch.int64),
        "want.adaln_out_rows": torch.tensor([adaln_out_rows], dtype=torch.int64),
        "want.bf16_all_bytes": torch.tensor([got_bytes], dtype=torch.int64),
        # indexed by DISTINCT TIMESTEPS: t2va@25, ref2va@50 — the extremes
        "want.resident_50t": torch.tensor(
            [non_adaln + adaln_out_rows * 50 * 2], dtype=torch.int64
        ),
        "want.resident_102t": torch.tensor(
            [non_adaln + adaln_out_rows * 102 * 2], dtype=torch.int64
        ),
        "want.adaln_resident": torch.tensor(
            [non_adaln + pbytes[ADALN]], dtype=torch.int64
        ),
    }

    # ── B. E4M3 bytes ────────────────────────────────────────────────────────
    values = []

    # every exact tie between adjacent E4M3 codes, both signs: this is the
    # round-half-to-even surface and it is dense enough to catch an off-by-one
    # in the mantissa carry
    for exp in range(-9, 9):
        for m in range(8):
            lo = math.ldexp(1.0 + m / 8.0, exp)
            hi = math.ldexp(1.0 + (m + 1) / 8.0, exp)
            values += [lo, (lo + hi) / 2.0, -(lo + hi) / 2.0]

    # subnormal grid and its boundary, including the m==8 promotion to 2^-6
    for m in range(10):
        values += [math.ldexp(m / 8.0, -6), math.ldexp((m + 0.5) / 8.0, -6)]

    # The overflow boundary, densely. torch does NOT saturate: values that
    # round past 448 land on the m==7 code, which E4M3-fn reserves for NaN.
    # 464 is the exact tie between m==6 (448) and m==7, so ties-to-even keeps
    # it finite; anything above it does not. This span pins the boundary down
    # by measurement instead of by my reasoning about it.
    values += [456.0, 463.9, 464.0, 464.1, 470.0, 479.0, 479.9, 480.0]
    values += [-456.0, -464.0, -464.1, -470.0, -480.0]

    # saturation, flush-to-zero, signed zero, and the exact max
    values += [
        448.0, 448.1, 500.0, 1e4, 1e30,
        -448.0, -449.0, -1e30,
        0.0, -0.0,
        math.ldexp(1.0, -10),          # half the smallest subnormal: ties to 0
        math.ldexp(1.0, -10) * 1.001,  # just above: rounds up to 2^-9
        math.ldexp(1.0, -30),
    ]

    v = torch.tensor(values, dtype=torch.float32)
    sb = v.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int64)
    tensors["in.scalars"] = v
    tensors["want.scalar_bytes"] = sb

    overflow = ((sb & 0x7F) == 0x7F).sum().item()
    print()
    print(
        f"  E4M3 scalars: {len(values)} values, {overflow} of them overflow to the"
        " NaN code"
    )
    print(
        "    torch does not saturate; our encoder does, deliberately. The gate"
        " requires"
    )
    print(
        "    bit-exactness everywhere torch stays finite and +-448 everywhere it"
        " does not."
    )

    # real per-row matrices: normal, heavy-tailed, one all-zero row, one row
    # whose absmax sits exactly on 448 after scaling
    torch.manual_seed(0x48_33)
    rows, cols = 6, 512
    w = torch.randn(rows, cols, dtype=torch.float32)
    w[1] *= 40.0            # a row that saturates hard before scaling
    w[2] *= 1e-4            # a row that lands mostly in the subnormal grid
    w[3] = 0.0              # all-zero row -> scale 1.0
    w[4, 7] = 448.0 * w[4].abs().max()  # force one element onto the top code
    w = w.to(torch.bfloat16).to(torch.float32)  # the checkpoint is bf16

    scale = w.abs().amax(dim=1) / E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (w / scale[:, None]).to(torch.float8_e4m3fn)
    deq = q.to(torch.float32) * scale[:, None]

    tensors["in.matrix"] = w
    tensors["want.row_scale"] = scale
    tensors["want.matrix_bytes"] = q.view(torch.uint8).to(torch.int64)
    tensors["want.matrix_dequant"] = deq

    # Per row, because a single headline number is unreadable here: rows 1 and
    # 4 are deliberately pathological and would otherwise dominate and hide the
    # only figure that describes real weights.
    #
    # Per-row absmax scaling gives each row 448 : 2^-9 of usable range, about
    # 2^17.8. Anything below absmax/229376 flushes to zero — a 100% relative
    # error on that element. That is not a defect in the encoder, it is the
    # format, and it only bites rows with a single huge outlier (row 4 has one
    # element 448x the rest, so most of the row falls off the bottom).
    err = (deq - w).abs()
    rel = err / w.abs().clamp_min(1e-30)
    labels = [
        "gaussian (what real weights look like)",
        "gaussian x40 (saturates before scaling)",
        "gaussian x1e-4 (lives in the subnormals)",
        "all zero",
        "one element 448x the rest",
        "gaussian",
    ]
    print()
    print("  E4M3 per-row round trip, 512 columns each:")
    for r in range(rows):
        nz = w[r].abs() > 0
        flushed = int(((deq[r] == 0) & nz).sum())
        if nz.any():
            print(
                f"    row {r}  max_abs {err[r].max().item():10.4g}"
                f"  max_rel {rel[r][nz].max().item():8.4g}"
                f"  mean_rel {rel[r][nz].mean().item():8.4g}"
                f"  flushed {flushed:3d}/512   {labels[r]}"
            )
        else:
            print(
                f"    row {r}  exact zeros, scale {scale[r].item():.1f}"
                f"                                            {labels[r]}"
            )
    ok = w[0].abs() > 0
    print(
        f"    headline for real weights: mean_rel {rel[0][ok].mean().item():.4g},"
        f" max_rel {rel[0][ok].max().item():.4g}  (3 mantissa bits => 2^-4 half-ULP bound)"
    )

    save_file(tensors, f"{OUT_DIR}/fp8_policy_ref.safetensors")
    print()
    print(f"wrote {OUT_DIR}/fp8_policy_ref.safetensors")
    print(f"wrote {OUT_DIR}/key_plan.csv  ({len(csv)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
