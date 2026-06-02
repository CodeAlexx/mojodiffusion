# klein9b_dit_smoke.mojo - real-weight FLUX.2 Klein 9B DiT block smoke.
#
# This is not image generation yet. It proves the Mojo Klein DiT path can load
# real 9B BF16 checkpoint tensors and execute the core architecture on a tiny
# token grid entirely on GPU: projections, timestep MLP, one double-stream
# block, one single-stream block, and the final projection.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.klein_dit import Klein9BDiT, build_klein_rope_tables


comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
comptime N_IMG = 4
comptime N_TXT = 8
comptime S = N_IMG + N_TXT


def _zeros(var shape: List[Int], dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    var vals = List[Float32](capacity=n)
    for _ in range(n):
        vals.append(0.0)
    return Tensor.from_host(vals, shape^, dtype, ctx)


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "  [stat]", name, "mean=", Float32(mean), "std=", Float32(sqrt(var_)),
        "absmax=", Float32(amax), "n=", n,
    )


def _print_shape(label: String, t: Tensor):
    var s = t.shape()
    print(label, s[0], s[1], s[2])


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein 9B DiT smoke - real BF16 weights, truncated 1+1 blocks ===")
    print("[load]", KLEIN9B_PATH)
    var model = Klein9BDiT.load(KLEIN9B_PATH, ctx)

    var img_shape = List[Int]()
    img_shape.append(1)
    img_shape.append(N_IMG)
    img_shape.append(128)
    var txt_shape = List[Int]()
    txt_shape.append(1)
    txt_shape.append(N_TXT)
    txt_shape.append(12288)
    var t_shape = List[Int]()
    t_shape.append(1)

    var img = _zeros(img_shape^, STDtype.BF16, ctx)
    var txt = _zeros(txt_shape^, STDtype.BF16, ctx)
    var tvals = List[Float32]()
    tvals.append(0.5)
    var timestep = Tensor.from_host(tvals, t_shape^, STDtype.F32, ctx)
    var rope = build_klein_rope_tables[N_IMG, N_TXT, 32, 128](ctx, STDtype.BF16)

    print("[forward] N_IMG", N_IMG, "N_TXT", N_TXT, "S", S)
    var out = model.forward_truncated[N_IMG, N_TXT, S](
        img, txt, timestep, rope[0], rope[1], ctx
    )
    _print_shape("  out shape:", out)
    _stats("out", out, ctx)
    print("Klein 9B DiT smoke PASS")
