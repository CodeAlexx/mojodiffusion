# models/ernie/config.mojo — ERNIE-Image per-variant config accessor.
#
# Binding user rule (2026-05-31): NO hardcoded arch/recipe in a model file. This
# helper READS the variant's config file (serenitymojo/configs/ernie_image.json),
# the single source of truth (verified against the checkpoint header by
# models/dit/ernie_contract.mojo). Mirrors models/klein/config.mojo EXACTLY.
#
# ERNIE field mapping into the shared TrainConfig:
#   inner_dim            -> d_model              (DiT hidden = 4096)
#   in_channels          -> in_channels          (latent channels = 128)
#   joint_attention_dim  -> joint_attention_dim  (text_in_dim = 3072, Mistral hidden)
#   out_channels         -> out_channels          (128)
#   num_double           -> num_double            (0 — ERNIE has no double-stream)
#   num_single           -> num_single            (36 single-stream blocks)
#   num_heads            -> n_heads                (32)
#   head_dim             -> head_dim               (128)
#   mlp_hidden           -> mlp_hidden             (FFN = 12288; GELU-gated, NOT 2x)
#   timestep_dim         -> timestep_dim           (sinusoidal embed dim = hidden = 4096)
#   rope_theta           -> rope_theta             (256)
#
# The 3-axis RoPE split (32/48/48) is an architectural CONTRACT constant; it
# lives in models/dit/ernie_contract.mojo as comptime values (exactly like Klein
# keeps H/Dh comptime), not as a runtime recipe scalar. The shared rope_theta is
# carried by the config; the per-axis split is comptime.

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime ERNIE_IMAGE_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/ernie_image.json"


def ernie_image() raises -> TrainConfig:
    return read_model_config(String(ERNIE_IMAGE_CONFIG))
