from std.math import acos, sin, sqrt


def _abs64(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _clamp_dot(x: Float64) -> Float64:
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return x


def _chw_index(c: Int, h: Int, w: Int, height: Int, width: Int) -> Int:
    return c * height * width + h * width + w


def _column_dot(
    base: List[Float32], variation: List[Float32], c: Int, w: Int,
    height: Int, width: Int,
) -> Float64:
    var lo2 = Float64(0.0)
    var hi2 = Float64(0.0)
    var dot = Float64(0.0)
    for h in range(height):
        var idx = _chw_index(c, h, w, height, width)
        var lo = Float64(base[idx])
        var hi = Float64(variation[idx])
        lo2 += lo * lo
        hi2 += hi * hi
        dot += lo * hi
    if lo2 <= 1.0e-30 or hi2 <= 1.0e-30:
        return 1.0
    return _clamp_dot(dot / (sqrt(lo2) * sqrt(hi2)))


def swarm_variation_noise_chw(
    base: List[Float32], variation: List[Float32], channels: Int, height: Int,
    width: Int, strength: Float64,
) raises -> List[Float32]:
    """SwarmKSampler-compatible slerp over a CHW latent slice.

    Swarm calls `slerp(var_seed_strength, base_noise, variation_noise)` after
    removing the batch dimension. For image latents this normalizes along CHW
    dim=1, i.e. each channel/column vector over height.
    """
    var total = channels * height * width
    if channels <= 0 or height <= 0 or width <= 0:
        raise Error("swarm_variation_noise_chw: shape dimensions must be positive")
    if len(base) != total or len(variation) != total:
        raise Error("swarm_variation_noise_chw: noise length does not match CHW shape")
    if strength <= 0.0:
        return base.copy()
    if strength >= 1.0:
        return variation.copy()

    var out = base.copy()
    var dot_sum = Float64(0.0)
    var dot_count = 0
    for c in range(channels):
        for w in range(width):
            dot_sum += _column_dot(base, variation, c, w, height, width)
            dot_count += 1

    if dot_count > 0 and dot_sum / Float64(dot_count) > 0.9995:
        # Mirrors Swarm's near-identical-vector fallback exactly.
        for i in range(total):
            out[i] = Float32(Float64(base[i]) * strength + Float64(variation[i]) * (1.0 - strength))
        return out^

    for c in range(channels):
        for w in range(width):
            var dot = _column_dot(base, variation, c, w, height, width)
            var omega = acos(dot)
            var so = sin(omega)
            var low_coef = Float64(1.0) - strength
            var high_coef = strength
            if _abs64(so) > 1.0e-12:
                low_coef = sin((1.0 - strength) * omega) / so
                high_coef = sin(strength * omega) / so
            for h in range(height):
                var idx = _chw_index(c, h, w, height, width)
                out[idx] = Float32(
                    low_coef * Float64(base[idx])
                    + high_coef * Float64(variation[idx])
                )
    return out^
