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
    var boundary: Float32     # dual high/low-noise expert timestep split
                              # (T2V 0.875 ; I2V 0.900 — Wan2.2 official recipe)

    @staticmethod
    def ti2v_5b() -> Wan22TrainConfig:
        return Wan22TrainConfig(
            num_layers=30, dim=3072, ffn_dim=14336, num_heads=24, head_dim=128,
            in_dim=48, out_dim=48, freq_dim=256, text_dim=4096, text_len=512,
            eps=1.0e-6, rope_theta=10000.0, lora_rank=32, lora_alpha=32.0,
            boundary=0.875,
        )

    # ── Wan2.2-A14B (the WIRED training engine — dim=5120, 40 blocks) ────────────
    # dim=5120, ffn=13824, heads=40, head_dim=128, freq_dim=256, patch (1,2,2),
    # qk_norm, cross_attn_norm, eps 1e-6, Wan2.1 VAE (16ch), umt5. in_dim/out_dim
    # here are the Conv3d CHANNEL counts (pre-patchify); packed IN_CH = in_dim*1*2*2.
    #   T2V-A14B: in_dim=16 (latent only) -> packed 64 ; boundary 0.875.
    #   I2V-A14B: in_dim=36 = noisy_latent(16) + y(20) [y = mask(4)+image_latent(16)]
    #             -> packed 144 ; out_dim STILL 16 (velocity of the 16-ch latent);
    #             boundary 0.900. Block/stack/LoRA/dual-expert compute is IDENTICAL
    #             to T2V (WanCrossAttention: no CLIP, no k_img/v_img — musubi
    #             model.py:379-380). Only the input channels + boundary differ.
    @staticmethod
    def t2v_a14b() -> Wan22TrainConfig:
        return Wan22TrainConfig(
            num_layers=40, dim=5120, ffn_dim=13824, num_heads=40, head_dim=128,
            in_dim=16, out_dim=16, freq_dim=256, text_dim=4096, text_len=512,
            eps=1.0e-6, rope_theta=10000.0, lora_rank=32, lora_alpha=32.0,
            boundary=0.875,
        )

    @staticmethod
    def i2v_a14b() -> Wan22TrainConfig:
        return Wan22TrainConfig(
            num_layers=40, dim=5120, ffn_dim=13824, num_heads=40, head_dim=128,
            in_dim=36, out_dim=16, freq_dim=256, text_dim=4096, text_len=512,
            eps=1.0e-6, rope_theta=10000.0, lora_rank=32, lora_alpha=32.0,
            boundary=0.900,
        )

    # ── Wan 2.1 T2V — SINGLE EXPERT (no dual high/low-noise) ─────────────────────
    # Wan2.1 uses the SAME WanAttentionBlock as Wan2.2 (WanCrossAttention, no CLIP,
    # no k_img/v_img) with a SINGLE DiT (boundary=None → no timestep split). The
    # per-batch modulation of 2.1 is numerically identical to the 2.2 per-token
    # block when the single timestep's e0 is broadcast across all tokens (musubi
    # force_v2_1_time_embedding), so the certified 2.2 block/stack/LoRA compute
    # carries over UNCHANGED. Same flow-match (target=noise-x0), same 10-target
    # LoRA set (attn q/k/v/o ×2 + ffn.0 + ffn.2), same Wan2.1 VAE + umt5, in_dim=16.
    # `boundary` is a SENTINEL 0.0 here to mark single-expert (the trainer's Wan2.1
    # path never consults it — no dual switch).
    #
    #   T2V-14B: identical dims to T2V-A14B (dim=5120, ffn=13824, heads=40,
    #            layers=40) — only the dual-expert is dropped.
    @staticmethod
    def t2v_14b_wan21() -> Wan22TrainConfig:
        return Wan22TrainConfig(
            num_layers=40, dim=5120, ffn_dim=13824, num_heads=40, head_dim=128,
            in_dim=16, out_dim=16, freq_dim=256, text_dim=4096, text_len=512,
            eps=1.0e-6, rope_theta=10000.0, lora_rank=32, lora_alpha=32.0,
            boundary=0.0,
        )

    #   T2V-1.3B: dim=1536, ffn=8960, heads=12, head_dim=128 (1536/12), layers=30,
    #             in_dim=16, freq_dim=256, patch(1,2,2), eps 1e-6. Same S=256 /
    #             IN_CH=64 patch geometry as A14B (latent [16,1,32,32]); only
    #             dim/heads/blocks differ → a distinct comptime monomorphization
    #             (H=12) in the trainer.
    @staticmethod
    def t2v_1p3b_wan21() -> Wan22TrainConfig:
        return Wan22TrainConfig(
            num_layers=30, dim=1536, ffn_dim=8960, num_heads=12, head_dim=128,
            in_dim=16, out_dim=16, freq_dim=256, text_dim=4096, text_len=512,
            eps=1.0e-6, rope_theta=10000.0, lora_rank=32, lora_alpha=32.0,
            boundary=0.0,
        )


# The 10 LoRA target projections per block trained by the wired engine
# (build_wan22_lora_set): the 8 attention projections (EDv2 wan22.rs:199-206
# LoraTarget) self_attn.{q,k,v,o} + cross_attn.{q,k,v,o} (in=out=dim each), PLUS
# the two FFN linears ffn.0 (dim->ffn) and ffn.2 (ffn->dim). IDENTICAL for T2V
# and I2V — the I2V block is the same WanAttentionBlock (only the patch-embedding
# input channels differ, which is NOT a LoRA target).
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
    t.append(String("ffn.0"))
    t.append(String("ffn.2"))
    return t^
