# ops/tests/svdquant_recon_vs_gemm_bench.mojo — DECISIVE reconstruct-vs-GEMM
# split for the int4-resident LTX2 path (nsys's QdstrmImporter is broken on this
# box — "wrong event order" — so we measure directly with CUDA-event sync).
#
# The int4-resident block-load reconstructs each quantized linear to a dense BF16
# weight (svdquant_reconstruct_weight_raw = dequant4 + rank-32 low-rank GEMM +
# add), then the block forward runs the real GEMM x @ Wᵀ (vendor.blas). This
# bench times BOTH on the DOMINANT LTX2 layer shape (ff.net.0.proj: in=4096,
# out=16384, rank=32, group=64) at the two stage sequence lengths, to decide:
#
#   fused-W4A16 ceiling = t_recon / (t_recon + t_fwd)   (what Phase A can remove)
#   W4A4 win           ≈ halving t_fwd (int4 tensor cores 2x) + removing t_recon
#
# Values are irrelevant (timing is value-independent) → synthesize raw buffers.
#
# Build:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -O2 -I . \
#     serenitymojo/ops/tests/svdquant_recon_vs_gemm_bench.mojo -o /tmp/svdq_bench
# Run:
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib /tmp/svdq_bench

from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.svdquant import (
    svdquant_reconstruct_weight_raw,
    svdquant_dequant_class_a,
    svdquant_linear,
    SvdquantLinearA,
)
from serenitymojo.ops.svdquant_w4a4 import SvdquantW4A4, svdquant_linear_w4a4


def _raw(nbytes: Int, shape: List[Int], dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    """Uninitialized device Tensor of the given shape/dtype (timing-only)."""
    var buf = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    return Tensor(buf^, shape.copy(), dtype)


def _bench(m: Int, in_f: Int, out_f: Int, rank: Int, ctx: DeviceContext) raises:
    var group = 64
    var half = in_f // 2
    var groups = in_f // group
    # int4 weight bundle (raw, correct shapes)
    var qweight = _raw(out_f * half, [out_f, half], STDtype.I8, ctx)          # [out, in/2]
    var wscales = _raw(groups * out_f * 2, [groups, out_f], STDtype.BF16, ctx)  # [in/64, out]
    var lora_down = _raw(in_f * rank * 2, [in_f, rank], STDtype.BF16, ctx)     # [in, R]
    var lora_up = _raw(out_f * rank * 2, [out_f, rank], STDtype.BF16, ctx)     # [out, R]
    var x = _raw(m * in_f * 2, [m, in_f], STDtype.BF16, ctx)                   # [M, in]

    # Reconstruct once to get a dense W operand for the forward-GEMM timing.
    var W = svdquant_reconstruct_weight_raw(qweight, wscales, lora_down, lora_up, in_f, out_f, ctx)

    comptime WARMUP = 5
    comptime ITERS = 40

    # --- reconstruct (dequant4 + low-rank GEMM[out,in] + add), per eval ---
    for _ in range(WARMUP):
        var _w = svdquant_reconstruct_weight_raw(qweight, wscales, lora_down, lora_up, in_f, out_f, ctx)
        _ = _w^
    ctx.synchronize()
    var r0 = perf_counter_ns()
    for _ in range(ITERS):
        var _w = svdquant_reconstruct_weight_raw(qweight, wscales, lora_down, lora_up, in_f, out_f, ctx)
        _ = _w^
    ctx.synchronize()
    var r1 = perf_counter_ns()
    var recon_us = Float64(r1 - r0) / Float64(ITERS) / 1000.0

    # --- forward GEMM x @ Wᵀ (vendor.blas, the real block-forward linear) ---
    for _ in range(WARMUP):
        var _y = linear(x, W, None, ctx)
        _ = _y^
    ctx.synchronize()
    var f0 = perf_counter_ns()
    for _ in range(ITERS):
        var _y = linear(x, W, None, ctx)
        _ = _y^
    ctx.synchronize()
    var f1 = perf_counter_ns()
    var fwd_us = Float64(f1 - f0) / Float64(ITERS) / 1000.0

    # --- BREAKDOWN: dequant-only (the int4→bf16 kernel) ---
    for _ in range(WARMUP):
        var _d = svdquant_dequant_class_a(qweight, wscales, in_f, out_f, ctx)
        _ = _d^
    ctx.synchronize()
    var d0 = perf_counter_ns()
    for _ in range(ITERS):
        var _d = svdquant_dequant_class_a(qweight, wscales, in_f, out_f, ctx)
        _ = _d^
    ctx.synchronize()
    var d1 = perf_counter_ns()
    var dq_us = Float64(d1 - d0) / Float64(ITERS) / 1000.0

    # --- BREAKDOWN: low-rank WEIGHT GEMM lora_up @ lora_downᵀ → [out,in] (K=32) ---
    for _ in range(WARMUP):
        var _l = linear(lora_up, lora_down, None, ctx)
        _ = _l^
    ctx.synchronize()
    var l0 = perf_counter_ns()
    for _ in range(ITERS):
        var _l = linear(lora_up, lora_down, None, ctx)
        _ = _l^
    ctx.synchronize()
    var l1 = perf_counter_ns()
    var lr_us = Float64(l1 - l0) / Float64(ITERS) / 1000.0

    # --- svdquant_linear: dequant + activation-side low-rank (NO weight materialize) ---
    var bias = _raw(out_f * 2, [out_f], STDtype.BF16, ctx)
    var smooth = _raw(in_f * 2, [in_f], STDtype.BF16, ctx)
    var wq = SvdquantLinearA(qweight^, wscales^, lora_down^, lora_up^, smooth^, bias^, in_f, out_f, rank)
    for _ in range(WARMUP):
        var _y = svdquant_linear(x, wq, ctx)
        _ = _y^
    ctx.synchronize()
    var s0 = perf_counter_ns()
    for _ in range(ITERS):
        var _y = svdquant_linear(x, wq, ctx)
        _ = _y^
    ctx.synchronize()
    var s1 = perf_counter_ns()
    var sq_us = Float64(s1 - s0) / Float64(ITERS) / 1000.0

    # --- NET W4A4 path: fwht_quant + int4 GEMM + rescale + bf16 low-rank + bias ---
    # (K = in_f must be 2048/4096/8192; rank 128 per-out). Synthetic weight (timing).
    var w4_us: Float64 = -1.0
    if in_f == 2048 or in_f == 4096 or in_f == 8192:
        var rank = 128
        var qw4 = _raw(out_f * (in_f // 2), [out_f, in_f // 2], STDtype.U8, ctx)
        var ws4 = _raw(out_f * 2, [out_f], STDtype.BF16, ctx)
        var ld4 = _raw(in_f * rank * 2, [in_f, rank], STDtype.BF16, ctx)
        var lu4 = _raw(out_f * rank * 2, [out_f, rank], STDtype.BF16, ctx)
        var bs4 = _raw(out_f * 2, [out_f], STDtype.BF16, ctx)
        var w4 = SvdquantW4A4(qw4^, ws4^, ld4^, lu4^, bs4^, in_f, out_f, rank)
        var xb = _raw(m * in_f * 2, [m, in_f], STDtype.BF16, ctx)
        for _ in range(WARMUP):
            var _y = svdquant_linear_w4a4(xb, w4, ctx)
            _ = _y^
        ctx.synchronize()
        var w0 = perf_counter_ns()
        for _ in range(ITERS):
            var _y = svdquant_linear_w4a4(xb, w4, ctx)
            _ = _y^
        ctx.synchronize()
        var w1 = perf_counter_ns()
        w4_us = Float64(w1 - w0) / Float64(ITERS) / 1000.0

    var total = recon_us + fwd_us
    var recon_pct = 100.0 * recon_us / total
    var a_ceiling = recon_pct  # fused-W4A16 removes reconstruct → best-case % saved
    # W4A4: remove reconstruct AND ~halve the forward GEMM (int4 2x tensor cores)
    var w4a4_total = fwd_us * 0.5
    var w4a4_speedup = total / w4a4_total

    print("")
    print("=== ff.net.0.proj  M=", m, " in=", in_f, " out=", out_f, " r=", rank, " ===")
    print("  reconstruct (dequant+lowrankGEMM+add) : ", recon_us, " us/eval")
    print("    - dequant4 kernel only              : ", dq_us, " us")
    print("    - low-rank WEIGHT GEMM (K=32→[o,i])  : ", lr_us, " us  <-- materializes dense W")
    print("    - (remainder = add + alloc)         : ", recon_us - dq_us - lr_us, " us")
    print("  forward GEMM  x @ Wᵀ                  : ", fwd_us, " us/eval")
    print("  reconstruct fraction of (recon+fwd)   : ", recon_pct, " %")
    print("  svdquant_linear (dequant + ACT low-rank, no weight-materialize):")
    print("    end-to-end                          : ", sq_us, " us/eval")
    print("    vs (reconstruct + fwd GEMM)         : ", total, " us  --> ", 100.0*(1.0 - sq_us/total), " % faster, ZERO quality change")
    print("  --> fused-W4A16 (Phase A) ceiling     : ", a_ceiling, " % faster (removes reconstruct only)")
    print("  --> W4A4 (Phase B) est speedup        : ", w4a4_speedup, "x (remove recon + halve GEMM)")
    if w4_us > 0.0:
        print("  W4A4 NET (fwht+quant+int4gemm+rescale+lowrank+bias): ", w4_us, " us/eval")
        print("  --> W4A4 NET speedup vs (recon+fwd)   : ", total / w4_us, "x  (MEASURED, full runtime)")


def main() raises:
    var ctx = DeviceContext()
    print("SVDQuant int4 reconstruct-vs-GEMM split — LTX-2.3 ff MLP (RTX 3090 Ti)")
    print("Dominant layer: in=4096 out=16384 (48 of these, the biggest weight)")
    _bench(1536, 4096, 16384, 32, ctx)   # Stage1 seq S_V=1536
    _bench(6144, 4096, 16384, 32, ctx)   # Stage2 seq S_V2=6144
    print("")
    print("done.")
