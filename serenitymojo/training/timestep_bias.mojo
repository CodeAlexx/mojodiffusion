# training/timestep_bias.mojo — timestep biasing (Wave 2A item 2f).
#
# Pure host F32 scalar math. Ports EDv2
# EriDiffusion-v2/crates/eridiffusion-core/src/training/features/timestep_bias.rs
# (Strategy enum :15, apply_bias :90) VERBATIM.
#
# Reshapes a sampled training timestep `t in [0, total)` to up-weight high-noise
# / low-noise regimes or restrict to a sub-range, WITHOUT changing the base
# sampler. `total` is the trainer's NUM_TRAIN_TIMESTEPS (typically 1000); for the
# flow-matching sigma path the same blend applies with total=1.0.
#
# ── Strategy enum (comptime ints) ─────────────────────────────────────────────
#   TSB_NONE     0  -> identity (DEFAULT-OFF: returns t unchanged)
#   TSB_LATER    1  -> pull toward total: t' = t + m*(total - t)
#   TSB_EARLIER  2  -> pull toward 0:     t' = t * (1 - m)
#   TSB_RANGE    3  -> linear remap [0,total) -> [lo*total, hi*total)
#
# Default-off invariance: TSB_NONE returns `t` byte-unchanged.
#
# Mojo 1.0.0b1.


comptime TSB_NONE = 0
comptime TSB_LATER = 1
comptime TSB_EARLIER = 2
comptime TSB_RANGE = 3


@always_inline
def _clamp01(x: Float32) -> Float32:
    if x < Float32(0.0):
        return Float32(0.0)
    if x > Float32(1.0):
        return Float32(1.0)
    return x


def apply_bias(
    t: Float32, total: Float32, strategy: Int,
    multiplier: Float32, range_min: Float32, range_max: Float32,
) -> Float32:
    """Apply timestep bias to one sampled `t in [0, total)`.

    Mirrors timestep_bias.rs:90-110.
    - TSB_NONE: identity (default-off).
    - TSB_LATER:  m clamped [0,1]; t' = t + m*(total - t).
    - TSB_EARLIER: m clamped [0,1]; t' = t*(1 - m).
    - TSB_RANGE: lo=clamp01(range_min); hi=max(clamp01(range_max), lo);
                 frac=clamp01(t/total); t' = (lo + frac*(hi - lo))*total.
    """
    if strategy == TSB_NONE:
        return t
    elif strategy == TSB_LATER:
        var m = _clamp01(multiplier)
        return t + m * (total - t)
    elif strategy == TSB_EARLIER:
        var m = _clamp01(multiplier)
        return t * (Float32(1.0) - m)
    elif strategy == TSB_RANGE:
        var lo = _clamp01(range_min)
        var hi = _clamp01(range_max)
        if hi < lo:
            hi = lo
        var frac = _clamp01(t / total)
        return (lo + frac * (hi - lo)) * total
    else:
        # Unknown strategy -> identity (safe default-off fallback).
        return t
