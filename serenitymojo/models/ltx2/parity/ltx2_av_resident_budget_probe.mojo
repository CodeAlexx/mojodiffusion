# serenitymojo/models/ltx2/parity/ltx2_av_resident_budget_probe.mojo
#
# How much VRAM does the AV fp8-resident block store ACTUALLY cost, and how much
# time does it save per block visit?
#
# The `--resident_blocks N` knob is sized against measured free headroom, not a
# guess: the AV arm peaks at 7197 MiB of 24564, so ~17.3 GiB is the budget. The
# per-block cost is NOT uniform — blocks 0-3 and 47 are BF16 boundary blocks
# (2 B/param, stored verbatim) while blocks 4-46 are fp8 (1 B/param, stored raw
# and dequanted per visit) — so a single "MiB/block" number cannot size the knob.
# This probe reports both classes separately and the per-visit materialise time
# streamed vs resident, so the production N is chosen from data.
#
# LTX2_RESIDENT_N (default 6) = how many blocks to park, from block 0.
#
# Run: rm -f serenitymojo.mojopkg; pixi run mojo build -O2 -I . -Xlinker -lm \
#   -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_resident_budget_probe.mojo \
#   -o /tmp/ltx2_av_resident_budget && /tmp/ltx2_av_resident_budget

from std.gpu.host import DeviceContext
from std.collections import List
from std.time import perf_counter_ns

from serenitymojo.io.env import env_int, env_or, serenity_checkpoint
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.models.ltx2.ltx2_av_stack import LTX2AVBlockSource
from serenitymojo.offload.ltx2_block_stream import LTX2BlockStream

comptime CKPT_FP8_NAME = "ltx-2.3-22b-distilled-fp8.safetensors"


def _time_block(
    src: LTX2AVBlockSource, bi: Int, reps: Int, ctx: DeviceContext
) raises -> Tuple[Float64, Float64]:
    """(min, mean) ms to materialise block `bi`, after warming THAT block."""
    var w_warm = src.get_block(bi, ctx)
    _ = w_warm^
    ctx.synchronize()
    var best = 1.0e18
    var total = 0.0
    for _ in range(reps):
        var t0 = perf_counter_ns()
        var w = src.get_block(bi, ctx)
        ctx.synchronize()
        var ms = Float64(perf_counter_ns() - t0) / 1.0e6
        _ = w^
        if ms < best:
            best = ms
        total += ms
    return (best, total / Float64(reps))


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var ckpt_fp8 = env_or(
        "LTX2_AV_CKPT_FP8", serenity_checkpoint(String(CKPT_FP8_NAME)),
    )
    var n_res = env_int("LTX2_RESIDENT_N", 6)

    print("=== LTX-2 AV fp8-resident block store: VRAM + time budget ===")
    print("  checkpoint:", ckpt_fp8)

    # ── per-block fp8 census (header-only, no VRAM) ──────────────────────────
    var probe = LTX2BlockStream.open(ckpt_fp8)
    var nb = probe.block_count()
    var n_bf16_blocks = 0
    var n_fp8_blocks = 0
    for b in range(nb):
        if probe.fp8_tensor_count(b) == 0:
            n_bf16_blocks += 1
        else:
            n_fp8_blocks += 1
    print("  blocks:", nb, " BF16-boundary:", n_bf16_blocks, " fp8:", n_fp8_blocks)

    # ── streamed materialise cost (residency OFF) ────────────────────────────
    # Block 0 is BF16-boundary, block 4 is the first fp8 block: time both, since
    # only the fp8 class actually pays a dequant. REPS iterations after a warm-up
    # of the SAME block — a warm-up on a different block class does not warm the
    # right path (a BF16-boundary resident hit is a pure Arc share and runs NO
    # dequant kernel, so it cannot stand in for an fp8 block's first dequant).
    # min is reported alongside mean: page-cache pressure on this box (58/62 GiB
    # in buff/cache) makes single samples off a 29.5 GB mmap wildly noisy, and
    # min is the least-contaminated estimator of the real cost.
    var REPS = env_int("LTX2_RESIDENT_REPS", 5)
    var src_off = LTX2AVBlockSource.open(ckpt_fp8, cfg)

    var t_stream_bf16 = _time_block(src_off, 0, REPS, ctx)
    var t_stream_fp8 = _time_block(src_off, 4, REPS, ctx)
    print("  [streamed] block 0 (BF16-boundary): min", t_stream_bf16[0],
          "ms  mean", t_stream_bf16[1], "ms")
    print("  [streamed] block 4 (fp8)          : min", t_stream_fp8[0],
          "ms  mean", t_stream_fp8[1], "ms")

    # ── resident store over blocks 0..n_res-1 ────────────────────────────────
    var src_on = LTX2AVBlockSource.open(ckpt_fp8, cfg)
    t0 = perf_counter_ns()
    src_on.enable_resident(0, n_res - 1, ctx)
    ctx.synchronize()
    var t_enable = Float64(perf_counter_ns() - t0) / 1.0e6
    var rb = src_on.resident_bytes()
    print("  [resident] blocks 0 ..", n_res - 1, " enable took", t_enable, "ms")
    print("  [resident] resident_bytes:", rb, "=", Float64(rb) / 1048576.0, "MiB")

    # Split the total across the two block classes present in [0, n_res).
    var nb_b = 0
    var nf_b = 0
    for b in range(n_res):
        if probe.fp8_tensor_count(b) == 0:
            nb_b += 1
        else:
            nf_b += 1
    print("  [resident] range contains", nb_b, "BF16-boundary +", nf_b, "fp8 blocks")

    var t_res_bf16 = _time_block(src_on, 0, REPS, ctx)
    print("  [resident] block 0 (BF16-boundary): min", t_res_bf16[0],
          "ms  mean", t_res_bf16[1], "ms")
    print("  [saving]  BF16 block per visit    :",
          t_stream_bf16[0] - t_res_bf16[0], "ms (min-vs-min)")

    if n_res > 4:
        var t_res_fp8 = _time_block(src_on, 4, REPS, ctx)
        print("  [resident] block 4 (fp8)          : min", t_res_fp8[0],
              "ms  mean", t_res_fp8[1], "ms")
        var save_fp8 = t_stream_fp8[0] - t_res_fp8[0]
        print("  [saving]  fp8 block per visit     :", save_fp8, "ms (min-vs-min)")
        # 96 visits/step = 48 forward + 48 backward. This is an UPPER BOUND on
        # the win: it credits every visit the fp8-block saving, but 5 of the 48
        # blocks are BF16-boundary and save a different amount.
        print("  [saving]  projected per STEP (96 visits):",
              save_fp8 * 96.0 / 1000.0, "s  [UPPER BOUND]")
    print("PROBE DONE")
