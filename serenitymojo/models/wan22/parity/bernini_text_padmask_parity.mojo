# Bernini UMT5 tail-mask gate.
#
# The creator passes a 512-row text buffer plus its true token length to
# varlen cuDNN attention. Verify that the bounded full-buffer mask path matches
# attention over the physically trimmed first VALID rows.

from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention_flash import (
    sdpa_flash_infer_fwd_rect,
    sdpa_flash_infer_fwd_rect_padmask,
)
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import slice


comptime B = 1
comptime SQ = 8
comptime SKV = 512
comptime VALID = 8
comptime H = 2
comptime DH = 128


def main() raises:
    var ctx = DeviceContext()
    var q = randn([B, SQ, H, DH], 20260601, STDtype.BF16, ctx)
    var k = randn([B, SKV, H, DH], 20260602, STDtype.BF16, ctx)
    var v = randn([B, SKV, H, DH], 20260603, STDtype.BF16, ctx)
    var k_short = slice(k, 1, 0, VALID, ctx)
    var v_short = slice(v, 1, 0, VALID, ctx)
    var scale = Float32(1.0 / Float32(DH) ** Float32(0.5))
    var expected = sdpa_flash_infer_fwd_rect[B, SQ, VALID, H, DH](
        q, k_short, v_short, scale, ctx
    ).to_host(ctx)
    var actual = sdpa_flash_infer_fwd_rect_padmask[B, SQ, SKV, H, DH](
        q, k, v, VALID, scale, ctx
    ).to_host(ctx)
    if len(actual) != len(expected):
        raise Error("Bernini text padmask length mismatch")
    var dot = 0.0
    var aa = 0.0
    var rr = 0.0
    var max_abs = 0.0
    for i in range(len(actual)):
        var a = Float64(actual[i])
        var r = Float64(expected[i])
        dot += a * r
        aa += a * a
        rr += r * r
        var d = a - r
        if d < 0.0:
            d = -d
        if d > max_abs:
            max_abs = d
    var cosine = dot / (aa * rr) ** 0.5
    print("Bernini text padmask: cos=", cosine, " max_abs=", max_abs)
    if cosine < 0.999 or max_abs > 0.02:
        raise Error("Bernini text padmask parity FAIL")
    print("GATE PASS Bernini text padmask matches trimmed creator semantics")
