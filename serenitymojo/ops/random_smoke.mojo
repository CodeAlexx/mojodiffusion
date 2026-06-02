# random_smoke.mojo - GPU randn smoke test.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn


comptime N = 262144
comptime SEED = UInt64(42)


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _stats(name: String, h: List[Float32]) raises:
    var n = len(h)
    var s = Float64(0.0)
    var s2 = Float64(0.0)
    var amax = Float64(0.0)
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
        var av = _abs(v)
        if av > amax:
            amax = av
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(name, "mean=", Float32(mean), "std=", Float32(sqrt(var_)), "absmax=", Float32(amax))
    if _abs(mean) > 0.015:
        raise Error("randn smoke: mean out of tolerance")
    if _abs(sqrt(var_) - 1.0) > 0.025:
        raise Error("randn smoke: std out of tolerance")


def main() raises:
    var ctx = DeviceContext()
    var shape = List[Int]()
    shape.append(N)
    var a = randn(shape.copy(), SEED, STDtype.F32, ctx)
    var b = randn(shape.copy(), SEED, STDtype.F32, ctx)
    var c = randn(shape.copy(), SEED + 1, STDtype.F32, ctx)

    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    var ch = c.to_host(ctx)
    _stats("randn_f32", ah)

    var rust_ref = List[Float32]()
    rust_ref.append(-1.979273677)
    rust_ref.append(-0.333371311)
    rust_ref.append(-1.608397007)
    rust_ref.append(-0.442455173)
    rust_ref.append(-0.347588181)
    rust_ref.append(-0.401655287)
    rust_ref.append(-0.117208749)
    rust_ref.append(0.078687891)
    rust_ref.append(0.245065019)
    rust_ref.append(0.053716935)
    rust_ref.append(-0.845145702)
    rust_ref.append(0.500127196)
    rust_ref.append(-0.114719711)
    rust_ref.append(-1.448853970)
    rust_ref.append(1.071474195)
    rust_ref.append(-1.489436030)
    var max_ref = Float64(0.0)
    for i in range(len(rust_ref)):
        var d = _abs(Float64(ah[i] - rust_ref[i]))
        if d > max_ref:
            max_ref = d
    print("rust_std_rng_first16_max_abs=", Float32(max_ref))
    if max_ref > 2.0e-6:
        raise Error("randn smoke: first values differ from Rust rand 0.8 StdRng reference")

    var max_same = Float64(0.0)
    var diff_count = 0
    for i in range(N):
        var d = _abs(Float64(ah[i] - bh[i]))
        if d > max_same:
            max_same = d
        if ah[i] != ch[i]:
            diff_count += 1
    print("same_seed_max_abs=", Float32(max_same), "different_seed_diff_count=", diff_count)
    if max_same != 0.0:
        raise Error("randn smoke: same seed was not deterministic")
    if diff_count < N // 2:
        raise Error("randn smoke: different seed did not change enough values")
    print("GPU randn smoke PASS")
