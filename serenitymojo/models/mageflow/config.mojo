# serenitymojo/models/mageflow/config.mojo — Mage-Flow-Base training spec.
#
# Dims + LoRA recipe accessor for the Mage-Flow-Base LoRA trainer
# (training/train_mageflow_real.mojo). Values CONFIRMED against
#   /home/alex/.serenity/models/checkpoints/Mage-Flow-Base/transformer/config.json
#     in_channels 128 / out_channels 128 / context_in_dim 2560 / hidden_size 3072
#     num_heads 24 / depth 12 / mlp_ratio 4.0 (FFN 12288) / axes [16,56,56]
#     theta 10000 / static_shift 6.0 / schedule_mode "z-image" / bf16
# and the Mage-Flow-Turbo transformer safetensors header (SAME arch; Base
# transformer shard still downloading at build time — the weights loader
# re-checks shapes against the real Base header on load):
#     img_in.weight [3072,128], txt_in.weight [3072,2560], txt_norm.weight [2560]
#     time_text_embed.timestep_embedder.linear_1.weight [3072,256]
#     norm_out.linear.weight [6144,3072], proj_out.weight [128,3072]
#     transformer_blocks.{0..11}.* — 32 tensors / 679,662,592 bytes per block.
#
# The block math itself is BANKED: qwenimage_block.mojo double_block_lora_
# forward/backward, GATED vs the real mage_flow block (incl text-identity rope)
# by models/mageflow/parity/mageflow_block_lora_parity.mojo (PASS, cos>=0.999).
#
# Mojo 1.0.0b1.


@fieldwise_init
struct MageFlowTrainSpec(Copyable, Movable, ImplicitlyCopyable):
    # ── architecture (frozen; from the Base transformer config.json) ──────────
    var depth: Int               # 12 double-stream blocks, 0 single
    var inner_dim: Int           # 3072 (= num_heads * head_dim)
    var num_heads: Int           # 24
    var head_dim: Int            # 128
    var mlp_hidden: Int          # 12288 (mlp_ratio 4.0)
    var in_channels: Int         # 128 packed latent channels
    var out_channels: Int        # 128 velocity channels
    var context_in_dim: Int      # 2560 (Qwen3-VL context width)
    var timestep_dim: Int        # 256 sinusoid width
    var rope_theta: Float64      # 10000.0
    var eps: Float32             # 1e-6 (LayerNorm/RMSNorm)
    # ── recipe defaults (config JSON overrides) ───────────────────────────────
    var flow_shift: Float32      # 6.0 (config.json static_shift; schedule z-image)
    var default_rank: Int        # 16
    var default_alpha: Float32   # 16.0
    var default_lr: Float32      # 1e-4

    @staticmethod
    def mageflow_base() -> MageFlowTrainSpec:
        return MageFlowTrainSpec(
            12, 3072, 24, 128, 12288, 128, 128, 2560, 256,
            Float64(10000.0), Float32(1e-6),
            Float32(6.0), 16, Float32(16.0), Float32(1e-4),
        )
