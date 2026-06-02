# training/noise_modifiers.mojo — noise perturbations applied to the sampled
# training noise BEFORE the forward pass (Wave 2B item 2e).
#
# Ports EDv2
# EriDiffusion-v2/crates/eridiffusion-core/src/training/features/noise_modifiers.rs
# (maybe_apply_offset_noise :55, maybe_apply_input_perturbation :83,
#  maybe_apply_multires_noise :120) VERBATIM math.
#
# The Klein loop builds its noise HOST-side as a flat List[Float32] in TOKEN
# layout [N_IMG, C] (tokens-major, channel last) — see train_klein_real.mojo
# `_host_noise`. So these modifiers run on the host list before it is uploaded.
# All extra randn draws use the SAME Box-Muller-on-PCG stream as `_host_noise`
# so the perturbation is deterministic and reproducible per seed.
#
#   offset noise:  with prob `prob`, noise[t,c] += weight * off[c],
#                  off[c] = randn (one per channel, broadcast over tokens).
#                  Mirrors maybe_apply_offset_noise: per-channel [.,C,1,1] randn
#                  scaled by weight, broadcast over spatial (here: tokens).
#                  weight<=0 OR prob<=0 OR draw>=prob => UNCHANGED (no draw alloc
#                  beyond the Bernoulli probe; matches the Rust short-circuits).
#   input perturb: noise[i] += gamma * randn  (one randn per element).
#                  Mirrors maybe_apply_input_perturbation. gamma<=0 => UNCHANGED.
#   multires:      4D-NCHW pyramid noise. The Rust guards `dims.len() != 4` and
#                  returns clone() otherwise; TOKEN-space noise is 2D, so this is
#                  a documented NO-OP here (kept as a gated stub so enabling it
#                  in token-space is a safe no-op, not a crash). Wired off by
#                  default (multires_iterations==0).
#
# Order matches the Rust trainer: offset FIRST, then input-perturbation
# (noise_modifiers.rs header: "Applied AFTER offset noise").
#
# Default-off invariance: with weight=0, prob=0, gamma=0, iterations=0 the
# returned list is BYTE-IDENTICAL to the input (no draws, no arithmetic).
#
# Mojo 1.0.0b1. Pure host F32 (no device).

from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from serenitymojo.training.schedule import sample_timestep_uniform


# Box-Muller normal stream on a PCG state — EXACTLY the generator `_host_noise`
# uses in train_klein_real.mojo (same constants, same draw order). Returns a
# List[Float32] of `n` N(0,1) draws seeded by `seed`.
def _host_randn(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 9007199254740992.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 11) & 0xFFFFFFFFFFFFF)) * (1.0 / 9007199254740992.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


# Standard<f32> uniform draw (top-24-bits/2^24) from the shared ChaCha12 stream —
# the Bernoulli probe for offset noise (mirrors rng.gen::<f32>() in the Rust).
def _bernoulli_uniform(seed: UInt64) -> Float32:
    return sample_timestep_uniform(seed)


# ── offset noise (in place on a host list) ───────────────────────────────────
# layout is [n_tokens, channels] row-major; off[c] broadcast over tokens.
def apply_offset_noise_host(
    mut noise: List[Float32], n_tokens: Int, channels: Int,
    weight: Float32, prob: Float32, off_seed: UInt64, bern_seed: UInt64,
):
    """noise[t,c] += weight * off[c] with probability `prob`.

    Mirrors maybe_apply_offset_noise: weight<=0 OR prob<=0 short-circuits with
    NO change and NO draw; otherwise draws one uniform — if draw>=prob, no
    change (matches `rng.gen() >= prob => clone()`). When it fires, one randn
    per channel, broadcast over tokens, scaled by weight."""
    if weight <= Float32(0.0) or prob <= Float32(0.0):
        return
    if _bernoulli_uniform(bern_seed) >= prob:
        return
    var off = _host_randn(channels, off_seed)
    for t in range(n_tokens):
        var base = t * channels
        for c in range(channels):
            noise[base + c] = noise[base + c] + weight * off[c]


# ── input perturbation (in place on a host list) ─────────────────────────────
def apply_input_perturbation_host(
    mut noise: List[Float32], gamma: Float32, seed: UInt64,
):
    """noise[i] += gamma * randn. Mirrors maybe_apply_input_perturbation:
    gamma<=0 short-circuits with NO change and NO draw."""
    if gamma <= Float32(0.0):
        return
    var pert = _host_randn(len(noise), seed)
    for i in range(len(noise)):
        noise[i] = noise[i] + gamma * pert[i]


# ── combined dispatcher used by the trainer (offset THEN input-perturb) ──────
# multires is a documented no-op in token-space (Rust 4D-only guard). The
# `iterations`/`discount` args are accepted so the wire site can pass cfg fields
# unconditionally; when iterations>0 in 2D token space we mirror the Rust
# `dims.len()!=4 => clone()` no-op (and warn once via the return flag).
def apply_noise_modifiers_host(
    mut noise: List[Float32], n_tokens: Int, channels: Int,
    offset_weight: Float32, offset_prob: Float32,
    input_perturb_gamma: Float32,
    multires_iterations: Int, multires_discount: Float32,
    step_seed: UInt64,
) raises -> Bool:
    """Apply offset noise then input perturbation IN PLACE.

    Token-space training noise is 2D [tokens, channels], so 4D multires pyramid
    noise is not implemented here. Enabling it raises instead of silently doing
    nothing. All-off (weights/gamma/iterations all 0) leaves `noise`
    byte-identical."""
    apply_offset_noise_host(
        noise, n_tokens, channels, offset_weight, offset_prob,
        step_seed * UInt64(101) + UInt64(1), step_seed * UInt64(103) + UInt64(2),
    )
    apply_input_perturbation_host(
        noise, input_perturb_gamma, step_seed * UInt64(107) + UInt64(3),
    )
    if multires_iterations > 0 and multires_discount > Float32(0.0):
        raise Error("apply_noise_modifiers_host: multires noise requires a 4D NCHW path; token-space trainer cannot apply it")
    return False
