# models/chroma/config.mojo — Chroma (lodestones Chroma1-HD) per-model config.
#
# Binding user rule (2026-05-31): NO hardcoded arch/recipe — the dims+recipe live
# in serenitymojo/configs/chroma.json and are read by the same read_model_config
# reader Klein/Flux use. The JSON is verified against the REAL safetensors header
# of the Chroma1-HD transformer shards (read with python struct, this session):
#   transformer_blocks.0.attn.to_q.weight        BF16 [3072, 3072]
#   transformer_blocks.0.attn.add_q_proj.weight  BF16 [3072, 3072]
#   transformer_blocks.0.ff.net.0.proj.weight    BF16 [12288, 3072]
#   single_transformer_blocks.0.proj_mlp.weight  BF16 [12288, 3072]
#   single_transformer_blocks.0.proj_out.weight  BF16 [3072, 15360]  (= [D, D+Fmlp])
#   x_embedder.weight                            BF16 [3072, 64]
#   context_embedder.weight                      BF16 [3072, 4096]
#   proj_out.weight                              BF16 [64, 3072]
#   => 19 double blocks + 38 single blocks
# => D=3072, H=24, Dh=128, Fmlp=12288, in_ch=64, txt_ch=4096, out_ch=64.
#
# Source-fidelity recipe cite (EriDiffusion-v2 chroma.rs):
#   crates/eridiffusion-core/src/models/chroma.rs:84-114
#     NUM_DOUBLE_BLOCKS = 19, NUM_SINGLE_BLOCKS = 38, DIM = 3072,
#     NUM_HEADS = 24, HEAD_DIM = 128, IN_CHANNELS = 64,
#     JOINT_ATTN_DIM = 4096 (T5-XXL hidden), MLP_HIDDEN = 12288, NORM_EPS = 1e-6.
#   LoRA targets chroma.rs:44-64 (the per-block trained projections; see
#   models/chroma/chroma_block.mojo header for the full slot map).
#   schedule shift 1.15 from models/dit/chroma_contract.mojo:52
#     (CHROMA_SCHEDULE_SHIFT_X100 = 115).
#
# CHROMA vs FLUX (the deltas that matter to this surface):
#   (1) Chroma stores SEPARATE attn projections (to_q / to_k / to_v with bias,
#       add_q_proj / add_k_proj / add_v_proj for txt) and a SEPARATE single-block
#       proj_mlp. Flux fuses them (qkv [3D,D]; single linear1 [3D+Fmlp,D]). The
#       BLOCK MATH is identical once the separate matrices are row-stacked, so the
#       Chroma block fwd/bwd REUSES the proven Flux block; the fuse happens in
#       models/chroma/weights.mojo (the loader). See chroma_block.mojo.
#   (2) Chroma has NO guidance / vector(CLIP-pooled) embed. Its modulation comes
#       from the distilled_guidance_layer APPROXIMATOR producing per-row mod vecs
#       (chroma_dit.mojo). Both are STACK-level (the block consumes precomputed
#       ModVecs/SingleModVecs exactly like Flux/Klein) → out of per-block scope.

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime CHROMA_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/chroma.json"

# Chroma-specific STACK-level constants (NOT in TrainConfig; not block-level).
# 3-axis RoPE split (sum = head_dim = 128), from chroma_contract.mojo:83-85.
comptime CHROMA_AXES_DIMS_ROPE_0 = 16
comptime CHROMA_AXES_DIMS_ROPE_1 = 56
comptime CHROMA_AXES_DIMS_ROPE_2 = 56
comptime CHROMA_HAS_GUIDANCE = False    # Chroma has NO guidance vec (delta vs Flux Dev)
comptime CHROMA_HAS_VECTOR = False      # Chroma has NO CLIP-pooled vector embed
comptime CHROMA_MLP_RATIO = 4           # GELU MLP hidden = inner_dim * 4 = 12288


def chroma1_hd() raises -> TrainConfig:
    return read_model_config(String(CHROMA_CONFIG))
