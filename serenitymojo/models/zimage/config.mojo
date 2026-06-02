# models/zimage/config.mojo — Z-Image (NextDiT) per-model config accessor.
#
# Binding user rule (2026-05-31): NO hardcoded arch/recipe in a model file. This
# helper READS the variant's config file (serenitymojo/configs/zimage.json), the
# single source of truth. Mirrors models/klein/config.mojo and
# models/ernie/config.mojo EXACTLY.
#
# Z-Image field mapping into the shared TrainConfig (from
# inference-flame/src/models/zimage_nextdit.rs NextDiTConfig::default() and the
# diffusers ZImageTransformer2DModel):
#   inner_dim            -> d_model              (DiT hidden = 3840 = 30*128)
#   in_channels          -> in_channels          (latent channels = 16)
#   joint_attention_dim  -> joint_attention_dim  (cap_feat_dim = 2560, Qwen3-4B)
#   out_channels         -> out_channels          (16)
#   num_double           -> num_double            (0 — Z-Image has no double-stream)
#   num_single           -> num_single            (30 main layers)
#   num_heads            -> n_heads                (30)
#   head_dim             -> head_dim               (128)
#   mlp_hidden           -> mlp_hidden             (SwiGLU per-gate hidden = 10240)
#   timestep_dim         -> timestep_dim           (t_embedder_hidden = 1024)
#   rope_theta           -> rope_theta             (256)
#
# Z-Image-specific architectural CONTRACT constants are kept comptime in the
# model code (exactly like Klein/Ernie keep H/Dh and the rope axis split), NOT as
# runtime recipe scalars. These live in the JSON as documentation and are skipped
# by read_model_config (unknown keys are ignored):
#   num_noise_refiner=2, num_context_refiner=2, patch_size=2, min_mod=256,
#   rope_axes_dim=[32,48,48] (sum=64=head_dim/2), time_scale=1000,
#   pad_tokens_multiple=32, norm_eps=1e-5 (block RMSNorm + qk RMSNorm),
#   final_norm_eps=1e-6 (final-layer LayerNorm).
#
# RoPE CONVENTION (parity-critical, see models/zimage/block.mojo header): Z-Image
# uses INTERLEAVED RoPE (pair (x[2i],x[2i+1])), NOT half-split. Confirmed against
# diffusers transformer_z_image.py apply_rotary_emb (torch.view_as_complex on
# reshape(..., -1, 2) = adjacent pairs) and the Mojo forward oracle
# models/dit/zimage_dit.mojo (rope_interleaved). This DIFFERS from Ernie, which is
# half-split. Block eps is 1e-5 (Z-Image), NOT Ernie's 1e-6.

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime ZIMAGE_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/zimage.json"


def zimage() raises -> TrainConfig:
    return read_model_config(String(ZIMAGE_CONFIG))
