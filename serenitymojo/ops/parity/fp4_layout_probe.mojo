# ops/parity/fp4_layout_probe.mojo — verify cuBLASLt's TILED scale-factor
# layout for VEC16_UE4M3 NVFP4 GEMM with NON-uniform scales (chunk 7 step 2;
# the uniform-scale probe was layout-invariant by design).
#
# Layout under test (the tcgen05/CUTLASS Sm1xx block-scaled layout): for an
# operand with R logical rows and K/16 scale columns, the scale tensor is
# stored in 512-byte tiles of 128 rows x 4 scale-cols:
#   phys(r, c) = (r//128 * ceil(K/64) + c//4) * 512
#              + (r % 32) * 16 + ((r // 32) % 4) * 4 + (c % 4)
# with rows padded to 128 and scale-cols padded to 4.
#
# Verification: random e2m1 codes + RANDOM ue4m3 scales packed via phys();
# host reference computes with the LOGICAL scales. cos ~= 1.0 <=> layout right.
#
# Build:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     -Xlinker -lcuda -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/ops/parity/fp4_layout_probe.mojo -o output/checks/fp4_layout_probe
# Run: LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib \
#     output/checks/fp4_layout_probe

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.ops.fp4_gemm import fp4_gemm_nt_rc

comptime M = 256
comptime N = 256
comptime K = 128
comptime KB = K // 16          # 8 scale columns


def _e2m1_value(code: Int) -> Float64:
    var mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    var v = mags[code & 0x7]
    if (code & 0x8) != 0:
        return -v
    return v


def _ue4m3_value(b: Int) -> Float64:
    if b == 0:
        return 0.0
    var e = ((b >> 3) & 0xF) - 7
    var m = Float64(b & 0x7) / 8.0
    var v = (1.0 + m)
    var p = e
    while p > 0:
        v *= 2.0
        p -= 1
    while p < 0:
        v *= 0.5
        p += 1
    return v


def _phys(r: Int, c: Int, kb_pad4: Int) -> Int:
    """Tiled scale-factor physical byte offset (see header comment)."""
    var tile = (r // 128) * (kb_pad4 // 4) + (c // 4)
    return tile * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + (c % 4)


def main() raises:
    var ctx = DeviceContext()
    var kb_pad = ((KB + 3) // 4) * 4
    var rows_pad_m = ((M + 127) // 128) * 128
    var rows_pad_n = ((N + 127) // 128) * 128
    var a_sc_bytes = (rows_pad_m // 128) * (kb_pad // 4) * 512
    var b_sc_bytes = (rows_pad_n // 128) * (kb_pad // 4) * 512

    # packed codes
    var a_host = ctx.enqueue_create_host_buffer[DType.uint8](M * K // 2)
    var b_host = ctx.enqueue_create_host_buffer[DType.uint8](N * K // 2)
    var seed: Int = 424242
    var ap = a_host.unsafe_ptr()
    for i in range(M * K // 2):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        ap[i] = UInt8(seed & 0xFF)
    var bp = b_host.unsafe_ptr()
    for i in range(N * K // 2):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        bp[i] = UInt8(seed & 0xFF)

    # LOGICAL random scales (ue4m3 bytes in a benign range 0x30..0x40)
    var a_log = List[Int]()
    for _ in range(M * KB):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        a_log.append(0x30 + (seed % 16))
    var b_log = List[Int]()
    for _ in range(N * KB):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        b_log.append(0x30 + (seed % 16))

    # pack into the tiled layout (padding bytes = 0, harmless)
    var as_host = ctx.enqueue_create_host_buffer[DType.uint8](a_sc_bytes)
    for i in range(a_sc_bytes):
        as_host.unsafe_ptr()[i] = 0
    for r in range(M):
        for c in range(KB):
            as_host.unsafe_ptr()[_phys(r, c, kb_pad)] = UInt8(a_log[r * KB + c])
    var bs_host = ctx.enqueue_create_host_buffer[DType.uint8](b_sc_bytes)
    for i in range(b_sc_bytes):
        bs_host.unsafe_ptr()[i] = 0
    for r in range(N):
        for c in range(KB):
            bs_host.unsafe_ptr()[_phys(r, c, kb_pad)] = UInt8(b_log[r * KB + c])

    var a_dev = ctx.enqueue_create_buffer[DType.uint8](M * K // 2)
    var b_dev = ctx.enqueue_create_buffer[DType.uint8](N * K // 2)
    var as_dev = ctx.enqueue_create_buffer[DType.uint8](a_sc_bytes)
    var bs_dev = ctx.enqueue_create_buffer[DType.uint8](b_sc_bytes)
    var d_dev = ctx.enqueue_create_buffer[DType.uint8](M * N * 4)
    ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
    ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)
    ctx.enqueue_copy(dst_buf=as_dev, src_buf=as_host)
    ctx.enqueue_copy(dst_buf=bs_dev, src_buf=bs_host)
    ctx.synchronize()

    var rc = fp4_gemm_nt_rc(a_dev, as_dev, b_dev, bs_dev, d_dev, M, N, K, ctx)
    if rc != 0:
        print("fp4_layout_probe: gemm rc=", rc, " — UNSUPPORTED at these dims")
        return
    ctx.synchronize()
    var d_host = ctx.enqueue_create_host_buffer[DType.uint8](M * N * 4)
    ctx.enqueue_copy(dst_buf=d_host, src_buf=d_dev)
    ctx.synchronize()
    var dp = d_host.unsafe_ptr().bitcast[Float32]()

    var dot: Float64 = 0.0
    var ng: Float64 = 0.0
    var nr: Float64 = 0.0
    var max_rel: Float64 = 0.0
    for mi in range(M):
        for ni in range(N):
            var acc: Float64 = 0.0
            for ki in range(K):
                var abyte = Int(ap[mi * (K // 2) + (ki >> 1)])
                var bbyte = Int(bp[ni * (K // 2) + (ki >> 1)])
                var acode = (abyte >> 4) & 0xF if (ki & 1) == 1 else abyte & 0xF
                var bcode = (bbyte >> 4) & 0xF if (ki & 1) == 1 else bbyte & 0xF
                var asc = _ue4m3_value(a_log[mi * KB + ki // 16])
                var bsc = _ue4m3_value(b_log[ni * KB + ki // 16])
                acc += _e2m1_value(acode) * asc * _e2m1_value(bcode) * bsc
            var got = Float64(dp[mi * N + ni])
            dot += got * acc
            ng += got * got
            nr += acc * acc
            if acc != 0.0:
                var rel = (got - acc) / acc
                if rel < 0:
                    rel = -rel
                if rel > max_rel:
                    max_rel = rel
    var cos = dot / (sqrt(ng) * sqrt(nr) + 1e-30)
    print("fp4_layout_probe: cos=", cos, " max_rel=", max_rel)
    if cos >= 0.99999:
        print("fp4_layout_probe: LAYOUT CONFIRMED (tiled 128x4, 32-row interleave)")
    else:
        print("fp4_layout_probe: LAYOUT WRONG — mapping differs, iterate")
