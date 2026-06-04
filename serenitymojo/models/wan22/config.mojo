# models/wan22/config.mojo — Wan2.2-TI2V-5B DiT per-variant TRAINING config.
#
# Dims CONFIRMED from the real safetensors header of
#   /home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-bf16/
#       diffusion_pytorch_model-00001-of-00003.safetensors
#   blocks.0.self_attn.q.weight  BF16 [3072,3072]  -> dim=3072
#   blocks.0.self_attn.norm_q.weight BF16 [3072]    (qk-rms over head_dim)
#   blocks.0.cross_attn.{q,k,v,o}.weight BF16 [3072,3072]
#   blocks.0.ffn.0.weight BF16 [14336,3072] ; ffn.2.weight BF16 [3072,14336] -> ffn=14336
#   blocks.0.modulation BF16 [1,6,3072]  (per-token AdaLN, 6 chunks)
#   blocks.0.norm3.{weight,bias} BF16 [3072]  (affine LN before cross-attn)
#   text_embedding.0.weight BF16 [3072,4096] -> text_dim=4096
#   => dim=3072, num_heads=24, head_dim=128 (3072/24), ffn_dim=14336,
#      num_layers=30, text_dim=4096, text_len=512.
#
# Recipe cited from EriDiffusion-v2 crates/eridiffusion-core/src/models/wan22.rs:
#   - TI2V-5B consts (wan22.rs:129-141): num_layers=30, dim=3072, ffn_dim=14336,
#     num_heads=24, head_dim=128, eps=1e-6, rope_theta=10000.0.
#   - RoPE 3-axis interleaved, FULL-dim axes [44,42,42] (head_dim=128, d6=21;
#     wan22_dit.mojo:82-91 wan22_rope_axes; matches model.py rope_apply).
#   - qk_norm=True (RMSNorm norm_q/norm_k over head_dim), cross_attn_norm=True
#     (affine LN norm3 before cross-attn).
#   - LoRA targets: 8 attention projections per block (wan22.rs:199-206
#     LoraTarget): self_attn.{q,k,v,o} + cross_attn.{q,k,v,o}; in=out=dim each.
#   - rank/alpha default 32/32 (scale 1.0) — flame wan22 trainer default.
#
# Wan2.2 is ALL single-image-stream blocks (no two-stream join). It maps onto the
# shared TrainConfig num_double slot = 30, num_single = 0 (the block kind is the
# WanAttentionBlock, not a Klein/Qwen double block).

# Self-contained per-model dims+recipe for the Wan2.2 TRAINING surface. The
# shared TrainConfig (training/train_config.mojo) carries the full optimizer/
# schedule recipe consumed by the real trainer; this accessor carries only the
# block-shape dims + LoRA recipe the per-model block/stack surface needs.
@fieldwise_init
struct Wan22TrainConfig(Copyable, Movable, ImplicitlyCopyable):
    var num_layers: Int       # 30 WanAttentionBlocks
    var dim: Int              # 3072 == num_heads*head_dim
    var ffn_dim: Int          # 14336
    var num_heads: Int        # 24
    var head_dim: Int         # 128
    var in_dim: Int           # 48 (patch latent channels)
    var out_dim: Int          # 48
    var freq_dim: Int         # 256 (timestep embedding)
    var text_dim: Int         # 4096 (raw context channels)
    var text_len: Int         # 512 (cross-attn kv length)
    var eps: Float32          # 1e-6 (LN / RMS)
    var rope_theta: Float32   # 10000
    var lora_rank: Int        # 32
    var lora_alpha: Float32   # 32 (scale = alpha/rank = 1.0)

    @staticmethod
    def ti2v_5b() -> Wan22TrainConfig:
        return Wan22TrainConfig(
            num_layers=30, dim=3072, ffn_dim=14336, num_heads=24, head_dim=128,
            in_dim=48, out_dim=48, freq_dim=256, text_dim=4096, text_len=512,
            eps=1.0e-6, rope_theta=10000.0, lora_rank=32, lora_alpha=32.0,
        )


# The 8 LoRA target projections per block (EDv2 wan22.rs:199-206 LoraTarget):
#   self_attn.q, self_attn.k, self_attn.v, self_attn.o,
#   cross_attn.q, cross_attn.k, cross_attn.v, cross_attn.o   (in=out=dim each).
def wan22_lora_targets() -> List[String]:
    var t = List[String]()
    t.append(String("self_attn.q"))
    t.append(String("self_attn.k"))
    t.append(String("self_attn.v"))
    t.append(String("self_attn.o"))
    t.append(String("cross_attn.q"))
    t.append(String("cross_attn.k"))
    t.append(String("cross_attn.v"))
    t.append(String("cross_attn.o"))
    return t^
