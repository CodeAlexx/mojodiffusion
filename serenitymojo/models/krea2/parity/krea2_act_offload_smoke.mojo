# models/krea2/parity/krea2_act_offload_smoke.mojo
# ──────────────────────────────────────────────────────────────────────────────
# ACTIVATION-OFFLOAD GO/NO-GO smoke (the requirement smoke for KREA2_ACT_OFFLOAD).
#
# The act-offload lever DELETES the streaming backward's per-block recompute (re-
# running krea2_single_stream_block_lora to regenerate Krea2BlockSaved) by SAVING
# the full Krea2BlockSaved to HOST in the forward and RESTORING it in the backward.
# That trades ~1 forward GEMM pass/block for a ~1.72GB D2H (save) + ~1.72GB H2D
# (restore) per block. The win exists ONLY if the transfer HIDES behind compute on
# the copy stream. This smoke measures, on the REAL shapes (L=4864, bf16):
#
#   A. RECOMPUTE cost  — time krea2_single_stream_block_lora (the thing deleted).
#   B. PCIe bandwidth  — pinned vs pageable D2H + H2D GB/s at the 1.72GB block size.
#   C. OVERLAPPED xfer — async D2H (cuMemcpyDtoHAsync on a copy stream) WHILE the
#                        block forward runs on the default stream; wall(compute +
#                        overlapped xfer) − wall(compute-alone) = un-hidden xfer.
#
#   VERDICT: if (un-hidden D2H + un-hidden H2D) per block < recompute per block →
#            offload WINS → build the full pageable-store + 2-pinned-slot path.
#            Else transfer-bound on the 3090's PCIe → DEAD END (the 12B-on-24GB
#            recompute is cheaper than offloading 48GB/step over PCIe).
#
# Timing-only: weights/inputs are random (we measure wall time, not correctness).
# Build needs the flash shim (the block forward pulls cuDNN flash on real_len<L).
# ──────────────────────────────────────────────────────────────────────────────

from std.ffi import external_call
from std.time import perf_counter_ns
from std.collections import List, Optional
from std.gpu.host import DeviceContext, HostBuffer, DeviceBuffer, DeviceStream, DeviceEvent
from std.gpu.host._nvidia_cuda import CUDA, CUstream

from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, Krea2BlockSaved,
    krea2_single_stream_block_lora,
)

comptime TArc = ArcPointer[Tensor]

# ── production krea2 dims (the giger-cache LFULL bucket) ──────────────────────
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM       # 6144
comptime MLPDIM = 16384
comptime L = 4864                         # LFULL = LTMAX(768) + IMGLEN(4096)
comptime HALF = HEADDIM // 2
comptime EPS = Float32(1e-6)
comptime RANK = 16
comptime ALPHA = Float32(16.0)

# the real_len the giger cache hits (lt∈{458..647} → real_len∈{4554..4743}); pick a
# representative so the forward runs the cuDNN flash-padmask path it uses in training.
comptime REAL_LEN = 4647


# ── D2H async DMA (cuMemcpyDtoHAsync_v2) — turbo_loader only has the H2D side ──
def _cu_memcpy_dtoh_async(
    dst_host_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    src_device_ptr: UInt64,
    nbytes: Int,
    stream: CUstream,
) -> Int32:
    return external_call["cuMemcpyDtoHAsync_v2", Int32](
        dst_host_ptr, src_device_ptr, nbytes, stream
    )


def _cu_memcpy_htod_async(
    dst_device_ptr: UInt64,
    src_host_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    nbytes: Int,
    stream: CUstream,
) -> Int32:
    return external_call["cuMemcpyHtoDAsync_v2", Int32](
        dst_device_ptr, src_host_ptr, nbytes, stream
    )


def _r(var shape: List[Int], seed: UInt64, ctx: DeviceContext) raises -> TArc:
    # random bf16 device tensor (krea2 acts/weights are bf16 in production).
    var f32 = randn(shape^, seed, STDtype.F32, ctx)
    return TArc(_to_bf16(f32, ctx))


def _to_bf16(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(t, STDtype.BF16, ctx)


def _r2(a: Int, b: Int, seed: UInt64, ctx: DeviceContext) raises -> TArc:
    var s = List[Int](); s.append(a); s.append(b)
    return _r(s^, seed, ctx)


def _r1(a: Int, seed: UInt64, ctx: DeviceContext) raises -> TArc:
    var s = List[Int](); s.append(a)
    return _r(s^, seed, ctx)


def _rf1(a: Int, seed: UInt64, ctx: DeviceContext) raises -> TArc:
    # F32 1-D random (the norm/mod SCALE params are F32 in production; the block
    # casts them down per the mixed-precision path).
    var s = List[Int](); s.append(a)
    return TArc(randn(s^, seed, STDtype.F32, ctx))


def _rope_tile(heads: Int, seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    # [L*heads, HALF] random unit-ish table (cos/sin values; timing-only).
    var s = List[Int](); s.append(L * heads); s.append(HALF)
    return randn(s^, seed, STDtype.F32, ctx)


def _block_saved_bytes() -> Int:
    # mirrors Krea2BlockSaved at L=4864, bf16 (2B) + flash stats f32 (4B).
    var Ffeat = 1 * L * FEATURES * 2
    var Hh = 1 * L * HEADS * HEADDIM * 2
    var KVh = 1 * L * KVHEADS * HEADDIM * 2
    var Mlp = 1 * L * MLPDIM * 2
    var saved = (
        Ffeat + Ffeat                  # x, xm
        + Hh + KVh + KVh               # q_pre, k_pre, v
        + Hh + KVh + Hh + Hh           # q_rope, k_rope, k_full, v_full
        + Ffeat + Ffeat + Ffeat + Ffeat + Ffeat + Ffeat + Ffeat  # attn_flat gate_pre sg gated a x1 xm2
        + Mlp + Mlp + Mlp + Ffeat      # mlp_gate mlp_up sw m
        + Ffeat + Ffeat                # xn, xn2
    )
    var flash = 4 * Hh + (1 * HEADS * L * 1 * 4)  # q/k/v/o bf16 + stats f32
    return saved + flash


def main() raises:
    var ctx = DeviceContext()
    var NBYTES = _block_saved_bytes()
    print("==== KREA2 ACTIVATION-OFFLOAD GO/NO-GO SMOKE ====")
    print("L=", L, " FEATURES=", FEATURES, " MLPDIM=", MLPDIM, " real_len=", REAL_LEN)
    print("Krea2BlockSaved bytes (the per-block offload payload) =",
          Float64(NBYTES) / 1e9, "GB")
    print("")

    # ── build one real block (random; timing-only) ───────────────────────────
    var seed = UInt64(101)
    var x = _r2(L, FEATURES, seed, ctx); seed += 1   # [1,L,F] flattened as [L,F]
    var xr = TArc(_reshape3(x, ctx))
    var vec = _to_bf16(randn(_s2(1, 6 * FEATURES), seed, STDtype.F32, ctx), ctx); seed += 1

    var w = Krea2BlockWeights(
        _r2(HEADS * HEADDIM, FEATURES, seed, ctx),       # wq
        _r2(KVHEADS * HEADDIM, FEATURES, seed + 1, ctx), # wk
        _r2(KVHEADS * HEADDIM, FEATURES, seed + 2, ctx), # wv
        _r2(FEATURES, FEATURES, seed + 3, ctx),          # gate_w
        _r2(FEATURES, FEATURES, seed + 4, ctx),          # wo
        _r2(MLPDIM, FEATURES, seed + 5, ctx),            # mlp_gate_w
        _r2(MLPDIM, FEATURES, seed + 6, ctx),            # mlp_up_w
        _r2(FEATURES, MLPDIM, seed + 7, ctx),            # mlp_down_w
        _rf1(HEADDIM, seed + 8, ctx),                    # qnorm_scale (F32 scale)
        _rf1(HEADDIM, seed + 9, ctx),                    # knorm_scale (F32)
        _rf1(FEATURES, seed + 10, ctx),                  # prenorm_scale (F32)
        _rf1(FEATURES, seed + 11, ctx),                  # postnorm_scale (F32)
        _r1(6 * FEATURES, seed + 12, ctx),               # mod_lin (bf16, matches the bf16 vec)
    )
    seed += 20
    # no LoRA adapters (None) — the recompute forward path is identical w/wo;
    # timing the FROZEN forward recompute (what the backward re-runs).
    var lora = Krea2BlockLora(
        None, None, None, None, None, None, None, None
    )

    var cos_q = _rope_tile(HEADS, seed, ctx); seed += 1
    var sin_q = _rope_tile(HEADS, seed, ctx); seed += 1
    var cos_k = _rope_tile(KVHEADS, seed, ctx); seed += 1
    var sin_k = _rope_tile(KVHEADS, seed, ctx); seed += 1
    var cos0 = randn(_s2(L, HALF), seed, STDtype.F32, ctx); seed += 1
    var sin0 = randn(_s2(L, HALF), seed, STDtype.F32, ctx); seed += 1
    var rl = Optional[Int](REAL_LEN)

    # ── A. RECOMPUTE cost ────────────────────────────────────────────────────
    # warmup
    for _ in range(2):
        var f = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            xr.copy(), vec, w, lora, cos0, sin0, cos_q, sin_q, cos_k, sin_k, EPS, ctx, rl,
        )
        _ = f^
    ctx.synchronize()
    comptime ITERS = 10
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        var f = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            xr.copy(), vec, w, lora, cos0, sin0, cos_q, sin_q, cos_k, sin_k, EPS, ctx, rl,
        )
        _ = f^
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var recompute_ms = Float64(t1 - t0) / 1e6 / Float64(ITERS)
    print("A. RECOMPUTE (block forward) =", recompute_ms, "ms/block")
    print("   × 28 blocks =", recompute_ms * 28.0 / 1000.0, "s/step (the deleted recompute)")
    print("")

    # ── B. PCIe bandwidth at the block size (pinned vs pageable) ─────────────
    var dev = ctx.enqueue_create_buffer[DType.uint8](NBYTES)
    dev.enqueue_fill(UInt8(7))
    var pinned = ctx.enqueue_create_host_buffer[DType.uint8](NBYTES)  # PINNED
    ctx.synchronize()
    # pinned D2H (default stream, sync) — median of 5
    var d2h_ms = _bench_d2h_sync(dev, pinned, NBYTES, ctx)
    var h2d_ms = _bench_h2d_sync(dev, pinned, NBYTES, ctx)
    print("B. PCIe (pinned, 1.72GB block):")
    print("   D2H =", d2h_ms, "ms  =", Float64(NBYTES)/1e9/(d2h_ms/1e3), "GB/s")
    print("   H2D =", h2d_ms, "ms  =", Float64(NBYTES)/1e9/(h2d_ms/1e3), "GB/s")
    print("   per step (28 blocks): D2H", d2h_ms*28.0/1e3, "s + H2D", h2d_ms*28.0/1e3, "s (if NOT overlapped)")
    print("")

    # ── C. OVERLAPPED transfer (async D2H on copy stream || compute) ─────────
    var copy_stream = ctx.create_stream()
    var ev = ctx.create_event[disable_timing=True]()
    # baseline: compute-alone wall (one block forward).
    ctx.synchronize()
    var c0 = perf_counter_ns()
    var fc = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
        xr.copy(), vec, w, lora, cos0, sin0, cos_q, sin_q, cos_k, sin_k, EPS, ctx, rl,
    )
    _ = fc^
    ctx.synchronize()
    var c1 = perf_counter_ns()
    var compute_alone_ms = Float64(c1 - c0) / 1e6

    # overlapped: dispatch async D2H on the copy stream, then run compute on the
    # default stream, then fence both. wall − compute_alone = un-hidden D2H.
    ctx.synchronize()
    var o0 = perf_counter_ns()
    var rc = Int(_cu_memcpy_dtoh_async(
        pinned.unsafe_ptr(), UInt64(Int(dev.unsafe_ptr())), NBYTES, CUDA(copy_stream),
    ))
    if rc != 0:
        raise Error(String("cuMemcpyDtoHAsync rc=") + String(rc))
    copy_stream.record_event(ev)
    var fo = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
        xr.copy(), vec, w, lora, cos0, sin0, cos_q, sin_q, cos_k, sin_k, EPS, ctx, rl,
    )
    _ = fo^
    ctx.stream().enqueue_wait_for(ev)   # default waits for the copy
    ctx.synchronize()
    var o1 = perf_counter_ns()
    var overlapped_ms = Float64(o1 - o0) / 1e6
    var unhidden_d2h_ms = overlapped_ms - compute_alone_ms
    if unhidden_d2h_ms < 0.0:
        unhidden_d2h_ms = 0.0

    print("C. OVERLAPPED:")
    print("   compute-alone (1 block fwd) =", compute_alone_ms, "ms")
    print("   compute + overlapped D2H    =", overlapped_ms, "ms")
    print("   UN-HIDDEN D2H per block     =", unhidden_d2h_ms, "ms")
    print("")

    # ── VERDICT ──────────────────────────────────────────────────────────────
    # Conservative: assume H2D restore un-hides symmetrically to D2H (same PCIe,
    # opposite direction). Per-block offload cost ≈ 2 × un-hidden-one-direction.
    var unhidden_per_block = unhidden_d2h_ms * 2.0
    print("==== VERDICT ====")
    print("recompute/block      =", recompute_ms, "ms")
    print("un-hidden xfer/block =", unhidden_per_block, "ms (2× un-hidden D2H, sym H2D)")
    if unhidden_per_block < recompute_ms:
        print("GO: offload WINS by", recompute_ms - unhidden_per_block,
              "ms/block (×28 =", (recompute_ms - unhidden_per_block)*28.0/1e3, "s/step) — build A.")
    else:
        print("NO-GO: transfer-bound. un-hidden xfer >= recompute → the 12B-on-24GB",
              "recompute is cheaper than offloading over PCIe. DEAD END, do not build.")


# ── small helpers ────────────────────────────────────────────────────────────
def _s2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _reshape3(t: TArc, ctx: DeviceContext) raises -> Tensor:
    return reshape(t[], _s3(1, L, FEATURES), ctx)


def _s3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def _bench_d2h_sync(
    dev: DeviceBuffer[DType.uint8], host: HostBuffer[DType.uint8], nbytes: Int, ctx: DeviceContext
) raises -> Float64:
    for _ in range(2):
        ctx.enqueue_copy(dst_buf=host, src_buf=dev)
    ctx.synchronize()
    var best = Float64(1e18)
    for _ in range(5):
        var a = perf_counter_ns()
        ctx.enqueue_copy(dst_buf=host, src_buf=dev)
        ctx.synchronize()
        var b = perf_counter_ns()
        var ms = Float64(b - a) / 1e6
        if ms < best: best = ms
    return best


def _bench_h2d_sync(
    dev: DeviceBuffer[DType.uint8], host: HostBuffer[DType.uint8], nbytes: Int, ctx: DeviceContext
) raises -> Float64:
    for _ in range(2):
        ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    var best = Float64(1e18)
    for _ in range(5):
        var a = perf_counter_ns()
        ctx.enqueue_copy(dst_buf=dev, src_buf=host)
        ctx.synchronize()
        var b = perf_counter_ns()
        var ms = Float64(b - a) / 1e6
        if ms < best: best = ms
    return best
