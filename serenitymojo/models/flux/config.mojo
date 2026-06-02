# models/flux/config.mojo — Flux (flux1-dev) per-variant config accessor.
#
# Binding user rule (2026-05-31): NO hardcoded arch/recipe. The shared block of
# the config (inner_dim / head_dim / n_heads / mlp_hidden / rope_theta / block
# counts / recipe) READS the JSON (serenitymojo/configs/flux.json) via the same
# read_model_config() reader Klein uses. Pointing at a config file is not "coding
# params in" — the params live in the JSON, verified against the checkpoint
# header (the BFL flux1-dev key shapes documented in inference-flame
# flux1_dit.rs:13-34).
#
# FLUX-SPECIFIC STACK CONSTANTS NOT IN TrainConfig
#   The shared TrainConfig (training/train_config.mojo) carries the block-level
#   dims every model needs. Flux additionally has three STACK-level pieces that
#   are NOT block-level and therefore not in TrainConfig:
#     axes_dims_rope = [16, 56, 56]   3-axis RoPE split (sum = head_dim = 128;
#                                     halves 8+28+28 = 64 = head_dim/2)
#     has_guidance   = True           Dev variant has guidance_in embed
#     vector_dim     = 768            CLIP-pooled -> vector_in embed
#   These feed the RoPE-table builder and the timestep/guidance/vector embeds at
#   the STACK level (Phase 2+), NOT the double/single blocks. The blocks consume
#   precomputed (cos, sin) tables + precomputed modulation vectors, exactly like
#   Klein's blocks. So Phase-1 block parity does not need them; they are exposed
#   here as comptime accessors for the stack phase.
#
# Source of truth for these three: inference-flame/src/models/flux1_dit.rs
#   Flux1Config::default() (lines 94-112): axes_dims_rope [16,56,56],
#   has_guidance true (Dev), vector_dim 768, rope_theta 10000.0, mlp_ratio 4.0.

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime FLUX_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/flux.json"

# Flux-specific stack constants (NOT in TrainConfig; from flux1_dit.rs:94-112).
comptime FLUX_AXES_DIMS_ROPE_0 = 16
comptime FLUX_AXES_DIMS_ROPE_1 = 56
comptime FLUX_AXES_DIMS_ROPE_2 = 56
comptime FLUX_HAS_GUIDANCE = True       # Dev variant
comptime FLUX_VECTOR_DIM = 768          # CLIP pooled -> vector_in
comptime FLUX_MLP_RATIO = 4             # GELU MLP: hidden = inner_dim * 4


def flux_dev() raises -> TrainConfig:
    return read_model_config(String(FLUX_CONFIG))
