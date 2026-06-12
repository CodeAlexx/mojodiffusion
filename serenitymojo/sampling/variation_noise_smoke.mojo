from std.math import sqrt

from serenitymojo.sampling.variation_noise import swarm_variation_noise_chw


def _abs64(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _check_close(name: String, got: Float64, want: Float64) raises:
    if _abs64(got - want) > 1.0e-5:
        raise Error(name + String(" mismatch: got ") + String(got) + String(" want ") + String(want))


def main() raises:
    var base: List[Float32] = [1.0, 0.0]
    var variation: List[Float32] = [0.0, 1.0]
    var zero = swarm_variation_noise_chw(base, variation, 1, 2, 1, 0.0)
    _check_close("strength0[0]", Float64(zero[0]), 1.0)
    _check_close("strength0[1]", Float64(zero[1]), 0.0)

    var one = swarm_variation_noise_chw(base, variation, 1, 2, 1, 1.0)
    _check_close("strength1[0]", Float64(one[0]), 0.0)
    _check_close("strength1[1]", Float64(one[1]), 1.0)

    var half = swarm_variation_noise_chw(base, variation, 1, 2, 1, 0.5)
    var want = sqrt(0.5)
    _check_close("strength_half[0]", Float64(half[0]), want)
    _check_close("strength_half[1]", Float64(half[1]), want)
    print("variation_noise_smoke: pass")
