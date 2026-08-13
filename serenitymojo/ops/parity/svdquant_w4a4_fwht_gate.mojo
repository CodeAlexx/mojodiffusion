# svdquant_w4a4_fwht_gate.mojo — kernel gate for the fused FWHT + int4 quant
# (ops/svdquant_w4a4.mojo). Verifies the GPU butterfly + per-token quant matches
# a host FWHT+quant reference (same math), on a non-degenerate row. MJ-1099 B.3b.
#
# Build: rm -f serenitymojo.mojopkg && pixi run mojo build -O2 -I . \
#   serenitymojo/ops/parity/svdquant_w4a4_fwht_gate.mojo -o /tmp/w4a4_fwht
# Run:  LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/w4a4_fwht

from std.math import sqrt, sin
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.svdquant_w4a4 import fwht_quant


def _host_fwht_quant(x: List[Float32], K: Int) -> List[Float32]:
    """Host reference: FWHT (in-place butterfly) → normalize → per-token int4
    quant → dequant. Returns the dequantized rotated row (what the kernel makes)."""
    var r = x.copy()
    var length = 1
    while length < K:
        var pp = 0
        while pp < K // 2:
            var blk = pp // length
            var within = pp % length
            var j = blk * (2 * length) + within
            var a = r[j]; var b = r[j + length]
            r[j] = a + b
            r[j + length] = a - b
            pp += 1
        length *= 2
    var inv = 1.0 / sqrt(Float32(K))
    var amax: Float32 = 0.0
    for i in range(K):
        r[i] = r[i] * inv
        var av = r[i] if r[i] >= 0.0 else -r[i]
        if av > amax:
            amax = av
    var scale = amax / 7.0
    if scale == 0.0:
        scale = 1.0
    var out = List[Float32]()
    for i in range(K):
        var q = r[i] / scale
        var rq = Float32(Int(q + 0.5)) if q >= 0.0 else Float32(Int(q - 0.5))
        if rq > 7.0: rq = 7.0
        if rq < -8.0: rq = -8.0
        out.append(rq * scale)
    return out^


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var dot: Float64 = 0.0; var na: Float64 = 0.0; var nb: Float64 = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def main() raises:
    var ctx = DeviceContext()
    var M = 8
    var K = 4096
    # non-degenerate rows (sinusoidal, per-row phase)
    var host = List[Float32]()
    for m in range(M):
        for k in range(K):
            host.append(sin(Float32(k) * 0.017 + Float32(m) * 0.3) + 0.2 * sin(Float32(k) * 0.101))
    var x = Tensor.from_host(host, [M, K], STDtype.BF16, ctx)

    var packed = fwht_quant(x, ctx)
    var half0 = K // 2
    var xq_host = ctx.enqueue_create_host_buffer[DType.uint8](M * half0)
    ctx.enqueue_copy(dst_buf=xq_host, src_buf=packed.xq.buf)
    ctx.synchronize()
    var xq_p = xq_host.unsafe_ptr()             # U8 bytes
    var xs_hf = packed.xscale.to_host(ctx)      # [M] bf16→f32

    var worst: Float64 = 1.0
    var half = K // 2
    for m in range(M):
        # reference row (bf16-rounded input to match the kernel's load)
        var rowin = List[Float32]()
        for k in range(K):
            rowin.append(host[m * K + k].cast[DType.bfloat16]().cast[DType.float32]())
        var refv = _host_fwht_quant(rowin, K)
        # dequant the kernel output: scale[m] * int4(xq[m, :])
        var scale = xs_hf[m]
        var moj = List[Float32]()
        for b in range(half):
            var byte = Int(xq_p[m * half + b])   # U8 byte → Int
            var lo = byte & 0xF; var hi = (byte >> 4) & 0xF
            var qe = lo - 16 if lo >= 8 else lo
            var qo = hi - 16 if hi >= 8 else hi
            moj.append(Float32(qe) * scale)
            moj.append(Float32(qo) * scale)
        var c = _cos(moj, refv)
        if c < worst:
            worst = c
    _ = xq_host^
    print("[w4a4-fwht] worst row cos(kernel, host-FWHT+quant) =", worst)
    if worst >= 0.999:
        print("[w4a4-fwht] PASS (FWHT butterfly + int4 quant correct)")
    else:
        print("[w4a4-fwht] FAIL — butterfly/quant mismatch")
        raise Error("w4a4 FWHT kernel gate FAILED")
