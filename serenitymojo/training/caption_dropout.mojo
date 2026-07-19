# training/caption_dropout.mojo — caption dropout for classifier-free guidance
# (Wave 2B item 2d).
#
# Ports EDv2
# EriDiffusion-v2/crates/eridiffusion-core/src/training/features/caption_dropout.rs
# (maybe_drop_caption :30, drop_caption :48) VERBATIM:
#
#   if prob <= 0.0:            return cond  (NO rng draw — default-off byte-exact)
#   if rng.gen::<f32>() < prob: return uncond
#   else:                       return cond
#
# ── The draw ──────────────────────────────────────────────────────────────────
# Rust `rng.gen::<f32>()` is rand 0.8.5 `Standard<f32>` = top-24-bits(word)/2^24.
# That is exactly the value `sample_timestep_uniform` already produces in
# schedule.mojo (`_standard_f64(w0)`, one ChaCha12 word at word_pos 0 of the
# seed's stream). We reuse that primitive so the per-step uniform draw matches
# the device RNG stream byte-for-byte. The Bernoulli decision is `draw < prob`.
#
# ── Caller contract ───────────────────────────────────────────────────────────
# This module decides WHETHER to drop (a pure host bool). The trainer owns the
# two embeddings (conditional text tokens vs the cached uncond/empty-prompt
# embedding) and swaps the device tensor itself at the text-token select site.
# Keeping the decision host-side avoids threading the move-only Tensor through
# here. Default-off (prob<=0) NEVER draws and NEVER drops.
#
# AGENT-DEFAULT for review: the uncond embedding the trainer swaps to is a
# ZERO text-token tensor (empty-prompt analog), documented at the wire site in
# train_klein_real.mojo. Klein's precache does not currently cache a separate
# negative/empty caption per training sample, so a zero embedding is the
# cheapest reproducible "unconditional" that requires no extra cache plumbing.
#
# Mojo 1.0.0b1. Pure host scalar math (no device).

from serenitymojo.training.schedule import sample_timestep_uniform


def caption_dropout_draw(seed: UInt64) -> Float32:
    """The Standard<f32> uniform draw used for the Bernoulli trial. Identical to
    schedule.mojo `sample_timestep_uniform` (rand 0.8.5 gen::<f32>())."""
    return sample_timestep_uniform(seed)


def should_drop_caption(seed: UInt64, prob: Float32) -> Bool:
    """Return True when this step's caption should be swapped for the uncond
    embedding. Mirrors caption_dropout.rs maybe_drop_caption:

      prob <= 0.0  -> False  (default-off, NO draw)
      draw < prob  -> True   (drop to uncond)
      else         -> False  (keep conditional)
    """
    if prob <= Float32(0.0):
        return False
    return caption_dropout_draw(seed) < prob
