# klein_rect_rope_parity.mojo — product landscape RoPE vs official diffusers.

from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.sampling.klein_sampler import _rope_host

comptime FIXTURE = "/tmp/klein_rect_rope.safetensors"
comptime HEIGHT = 56
comptime WIDTH = 72
comptime N_IMG = HEIGHT * WIDTH
comptime TEXT = 512
comptime S = TEXT + N_IMG
comptime HEADS = 2
comptime HALF = 64
comptime ROWS = S * HEADS


def main() raises:
    var ctx = DeviceContext()
    var rope = _rope_host[N_IMG, TEXT, S, HEADS, HEIGHT, WIDTH]()
    if len(rope[0]) != ROWS * HALF or len(rope[1]) != ROWS * HALF:
        raise Error("Klein rectangular compact RoPE shape mismatch")
    var cos = Tensor.from_host(rope[0].copy(), [ROWS, HALF], STDtype.F32, ctx)
    var sin = Tensor.from_host(rope[1].copy(), [ROWS, HALF], STDtype.F32, ctx)

    var fx = ShardedSafeTensors.open(FIXTURE)
    var cos_ref = Tensor.from_view(fx.tensor_view("cos"), ctx).to_host(ctx)
    var sin_ref = Tensor.from_view(fx.tensor_view("sin"), ctx).to_host(ctx)
    var h = ParityHarness(0.99999)
    var cos_result = h.compare(cos, cos_ref, ctx)
    var sin_result = h.compare(sin, sin_ref, ctx)
    print("klein landscape rope cos:", cos_result)
    print("klein landscape rope sin:", sin_result)
    if not cos_result.passed or not sin_result.passed:
        raise Error("Klein rectangular RoPE parity failed")
    print("PASS: Klein 56x72 packed-grid RoPE matches official diffusers")
