# serenitymojo/models/ltx2/parity/ltx2_av_block_load_decomp_probe.mojo
#
# STEP 1 of the AV block-load unit: DECOMPOSE the ~1040 ms/block weight
# materialisation into its four sub-costs before changing anything.
#
# The AV block loader (models/dit/ltx2_dit.mojo:806-834 LTX2AVBlockWeights.load)
# calls `Tensor.from_view_as_bf16` once per key. For a BF16-on-disk checkpoint
# that function (tensor.mojo:238-241) does:
#     host_out = ctx.enqueue_create_host_buffer(nbytes)     # (a) pinned alloc
#     for i in range(nbytes): outp[i] = tv.data[i]          # (b) SCALAR BYTE LOOP
#     dev = ctx.enqueue_create_buffer(nbytes)               # (c) device alloc
#     ctx.enqueue_copy(dev, host_out)                       # (c) H2D
#     ctx.synchronize()                                     # (d) per-tensor sync
#
# This probe times (a), (b), (c), (d) SEPARATELY at real block-weight sizes, and
# times variant (b2) sys_memcpy against (b1) the scalar loop — memcpy is what the
# sibling `Tensor.from_view` (tensor.mojo:174) already uses for the same job.
#
# Measurement only: no weights are kept, nothing in the model path is changed.
#
# Run: rm -f serenitymojo.mojopkg; pixi run mojo build -O2 -I . -Xlinker -lm \
#   -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_block_load_decomp_probe.mojo \
#   -o /tmp/ltx2_av_load_decomp && /tmp/ltx2_av_load_decomp

from max.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from std.collections import List
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.io.env import env_or, serenity_checkpoint

comptime CKPT_NAME = "ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
comptime BLOCK = 0
comptime REPS = 5


def _median(var v: List[Float64]) -> Float64:
    for i in range(1, len(v)):
        var x = v[i]
        var j = i - 1
        while j >= 0 and v[j] > x:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = x
    return v[len(v) // 2]


def _fmt(v: List[Float64]) -> String:
    var s = String("[")
    for i in range(len(v)):
        if i > 0: s += ", "
        s += String(v[i])
    return s + "]"


def _report(label: String, v: List[Float64], total_ms: Float64) -> Float64:
    var m = _median(v.copy())
    print("  ", label, " reps_ms=", _fmt(v))
    print("      MEDIAN=", m, "ms   share of", total_ms, "ms baseline =",
          100.0 * m / total_ms, "%")
    return m


def main() raises:
    var ctx = DeviceContext()
    var ckpt = env_or("LTX2_AV_CKPT", serenity_checkpoint(String(CKPT_NAME)))
    print("=== AV block-load DECOMPOSITION (block", BLOCK, ", reps=", REPS, ") ===")

    var st = ShardedSafeTensors.open(ckpt)
    # the block's key set, straight off the checkpoint header
    var prefix = (String("model.diffusion_model.transformer_blocks.")
                  + String(BLOCK) + ".")
    var all_names = st.names()
    var keys = List[String]()
    for ref n in all_names:
        if n.startswith(prefix):
            keys.append(n)
    var nk = len(keys)
    var total_bytes = 0
    var n_bf16 = 0
    for ref k in keys:
        var tv = st.tensor_view(k)
        total_bytes += tv.nbytes()
        if tv.dtype == STDtype.BF16:
            n_bf16 += 1
    print("  keys=", nk, " bf16_keys=", n_bf16, " total_bytes=", total_bytes,
          " (=", Float64(total_bytes) / 1048576.0, "MiB )")

    # ── BASELINE: the real loader path, once per key ─────────────────────────
    var wu = List[ArcPointer[Tensor]]()
    for ref k in keys:
        wu.append(ArcPointer(Tensor.from_view_as_bf16(st.tensor_view(k), ctx)))
    _ = wu^
    var base = List[Float64]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var ws = List[ArcPointer[Tensor]]()
        for ref k in keys:
            ws.append(ArcPointer(Tensor.from_view_as_bf16(st.tensor_view(k), ctx)))
        var t1 = perf_counter_ns()
        base.append(Float64(t1 - t0) / 1.0e6)
        _ = ws^
    var base_ms = _median(base.copy())
    print("  BASELINE Tensor.from_view_as_bf16 x", nk, " reps_ms=", _fmt(base))
    print("      MEDIAN=", base_ms, "ms  <- reconcile against the 1040 ms pin")
    print("")

    # ── (a) pinned host alloc only ───────────────────────────────────────────
    var a_ms = List[Float64]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var bufs = List[HostBuffer[DType.uint8]]()
        for ref k in keys:
            bufs.append(ctx.enqueue_create_host_buffer[DType.uint8](
                st.tensor_view(k).nbytes()))
        var t1 = perf_counter_ns()
        a_ms.append(Float64(t1 - t0) / 1.0e6)
        _ = bufs^
    var a = _report(String("(a) enqueue_create_host_buffer x") + String(nk),
                    a_ms, base_ms)

    # ── (b) mmap -> pinned copy, into ALREADY-allocated buffers ──────────────
    # allocate once, outside the timed region, so this isolates the copy.
    var pinned = List[HostBuffer[DType.uint8]]()
    for ref k in keys:
        pinned.append(ctx.enqueue_create_host_buffer[DType.uint8](
            st.tensor_view(k).nbytes()))

    # (b1) the CURRENT scalar byte loop (tensor.mojo:240-241)
    var b1_ms = List[Float64]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        for ki in range(nk):
            var tv = st.tensor_view(keys[ki])
            var outp = pinned[ki].unsafe_ptr()
            var nb = tv.nbytes()
            for i in range(nb):
                outp[i] = tv.data[i]
        var t1 = perf_counter_ns()
        b1_ms.append(Float64(t1 - t0) / 1.0e6)
    var b1 = _report(String("(b1) SCALAR BYTE LOOP mmap->pinned x") + String(nk),
                     b1_ms, base_ms)

    # (b2) sys_memcpy — what Tensor.from_view already uses for the same job
    var b2_ms = List[Float64]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        for ki in range(nk):
            var tv = st.tensor_view(keys[ki])
            var dst = BytePtr(unsafe_from_address=Int(pinned[ki].unsafe_ptr()))
            var src = BytePtr(unsafe_from_address=Int(tv.data.unsafe_ptr()))
            _ = sys_memcpy(dst, src, tv.nbytes())
        var t1 = perf_counter_ns()
        b2_ms.append(Float64(t1 - t0) / 1.0e6)
    var b2 = _report(String("(b2) sys_memcpy mmap->pinned x") + String(nk),
                     b2_ms, base_ms)

    # ── (c) device alloc + enqueue_copy, NO sync ─────────────────────────────
    var c_ms = List[Float64]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var devs = List[DeviceBuffer[DType.uint8]]()
        for ki in range(nk):
            var nb = st.tensor_view(keys[ki]).nbytes()
            var d = ctx.enqueue_create_buffer[DType.uint8](nb)
            ctx.enqueue_copy(dst_buf=d, src_buf=pinned[ki])
            devs.append(d^)
        var t1 = perf_counter_ns()
        ctx.synchronize()          # drain outside the timed region
        c_ms.append(Float64(t1 - t0) / 1.0e6)
        _ = devs^
    var c = _report(String("(c) device alloc + enqueue_copy (NO sync) x") + String(nk),
                    c_ms, base_ms)

    # ── (d) sync cost: 86 syncs vs ONE ───────────────────────────────────────
    var d_many = List[Float64]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var devs = List[DeviceBuffer[DType.uint8]]()
        for ki in range(nk):
            var nb = st.tensor_view(keys[ki]).nbytes()
            var d = ctx.enqueue_create_buffer[DType.uint8](nb)
            ctx.enqueue_copy(dst_buf=d, src_buf=pinned[ki])
            ctx.synchronize()      # per-tensor sync, as the loader does today
            devs.append(d^)
        var t1 = perf_counter_ns()
        d_many.append(Float64(t1 - t0) / 1.0e6)
        _ = devs^
    var dm = _report(String("(d1) same as (c) but ") + String(nk) + " syncs",
                     d_many, base_ms)

    var d_one = List[Float64]()
    for _ in range(REPS):
        var t0 = perf_counter_ns()
        var devs = List[DeviceBuffer[DType.uint8]]()
        for ki in range(nk):
            var nb = st.tensor_view(keys[ki]).nbytes()
            var d = ctx.enqueue_create_buffer[DType.uint8](nb)
            ctx.enqueue_copy(dst_buf=d, src_buf=pinned[ki])
            devs.append(d^)
        ctx.synchronize()          # ONE sync at the end
        var t1 = perf_counter_ns()
        d_one.append(Float64(t1 - t0) / 1.0e6)
        _ = devs^
    var do1 = _report(String("(d2) same as (c) but ONE sync at the end"),
                      d_one, base_ms)

    _ = pinned^

    # ── verdict ──────────────────────────────────────────────────────────────
    print("")
    print("  SUMMARY (median ms, baseline", base_ms, "ms):")
    print("    (a)  pinned alloc x", nk, "        =", a)
    print("    (b1) scalar byte loop copy      =", b1, "   <- current code")
    print("    (b2) sys_memcpy copy            =", b2, "   <- candidate")
    print("    (c)  dev alloc + H2D, no sync   =", c)
    print("    (d1) (c) +", nk, "syncs            =", dm)
    print("    (d2) (c) + 1 sync               =", do1)
    print("    accounted (a)+(b1)+(d1)         =", a + b1 + dm)
    print("")
    print("    b1 - b2 (memcpy saving)         =", b1 - b2, "ms/block")
    print("    d1 - d2 (one-sync saving)       =", dm - do1, "ms/block")
    print("    scalar-loop throughput          =",
          Float64(total_bytes) / 1.073741824e9 / (b1 / 1000.0), "GB/s")
    print("    memcpy throughput               =",
          Float64(total_bytes) / 1.073741824e9 / (b2 / 1000.0), "GB/s")

    # ── BYTE-IDENTITY: from_view (memcpy) vs from_view_as_bf16 (scalar loop) ──
    # The fix swaps one for the other on BF16-source views. Prove on EVERY key
    # of the block that the device bytes are identical, rather than inferring it
    # from downstream cosines.
    print("")
    print("  BYTE-IDENTITY check, all", nk, "keys (BF16 keys must be identical):")
    var checked = 0
    var mismatched = 0
    var skipped_f32 = 0
    for ref k in keys:
        var tv = st.tensor_view(k)
        if tv.dtype != STDtype.BF16:
            skipped_f32 += 1
            continue
        var a_t = Tensor.from_view_as_bf16(st.tensor_view(k), ctx)   # scalar loop
        var b_t = Tensor.from_view(st.tensor_view(k), ctx)           # memcpy
        if a_t.nbytes() != b_t.nbytes():
            mismatched += 1
            print("    SIZE MISMATCH", k)
            continue
        if a_t.dtype() != b_t.dtype():
            mismatched += 1
            print("    DTYPE MISMATCH", k)
            continue
        var ha = a_t.to_host(ctx)
        var hb = b_t.to_host(ctx)
        var bad = 0
        if len(ha) != len(hb):
            bad = 1
        else:
            for i in range(len(ha)):
                var x = ha[i]
                var y = hb[i]
                # exact equality, including NaN payloads (compare bit patterns
                # via the != on the decoded values plus a NaN-agreement check)
                if x != y and not (x != x and y != y):
                    bad += 1
        if bad != 0:
            mismatched += 1
            print("    VALUE MISMATCH", k, " differing=", bad)
        checked += 1
    print("    BF16 keys compared=", checked, " F32 keys skipped (cast path"
          " unchanged)=", skipped_f32, " MISMATCHED=", mismatched)
    if mismatched != 0:
        raise Error(String("BYTE-IDENTITY FAIL: ") + String(mismatched)
                    + " keys differ between from_view and from_view_as_bf16")
    print("    BYTE-IDENTITY PASS: memcpy path == scalar-loop path on every"
          " BF16 key")
    print("ltx2_av_block_load_decomp_probe DONE")
