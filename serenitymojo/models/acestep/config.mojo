# models/acestep/config.mojo — ACE-Step-1.5 DiT per-variant TRAINING config.
#
# Dims CONFIRMED from the real safetensors header of
#   /home/alex/ACE-Step-1.5/checkpoints/acestep-v15-turbo/model.safetensors
#   decoder.layers.0.self_attn.q_proj.weight  BF16 [2048,2048] -> q_dim=2048, hidden=2048
#   decoder.layers.0.self_attn.k_proj.weight  BF16 [1024,2048] -> kv_dim=1024 (8 kv heads*128)
#   decoder.layers.0.self_attn.v_proj.weight  BF16 [1024,2048]
#   decoder.layers.0.self_attn.o_proj.weight  BF16 [2048,2048]
#   decoder.layers.0.self_attn.{q,k}_norm.weight BF16 [128] (per-head qk-rms over head_dim)
#   decoder.layers.0.cross_attn.{q,k,v,o}_proj.weight same GQA shapes; q/k_norm [128]
#   decoder.layers.0.{self_attn_norm,cross_attn_norm,mlp_norm}.weight BF16 [2048] (affine RMS)
#   decoder.layers.0.mlp.gate_proj.weight BF16 [6144,2048] ; up_proj [6144,2048]
#   decoder.layers.0.mlp.down_proj.weight BF16 [2048,6144]  -> intermediate=6144 (SwiGLU)
#   decoder.layers.0.scale_shift_table  BF16 [1,6,2048]  (per-SAMPLE AdaLN, 6 chunks [H])
#   => hidden=2048, num_heads=16, num_kv_heads=8 (GQA n_rep=2), head_dim=128,
#      intermediate=6144, num_layers=24, rope_theta=1e6, eps=1e-6.
#
# Recipe cited from EriDiffusion-v2 crates/eridiffusion-core/src/models/acestep.rs:
#   - AceStepConfig (acestep.rs:68-79): hidden_size=2048, num_heads=16,
#     num_kv_heads=8, head_dim=128, intermediate_size=6144, num_layers=24,
#     rope_theta=1_000_000.0, rms_norm_eps=1e-6 (turbo defaults, acestep.rs:192-199).
#   - Block (dit_layer_forward, acestep.rs:843-913): scale_shift_table[1,6,H]+
#     timestep_proj -> chunk6 (shift_msa,scale_msa,gate_msa,c_shift,c_scale,c_gate),
#     PER-SAMPLE [H] AdaLN: norm_hs=(1+scale_msa)*rms(x)+shift_msa (modulate);
#     x=hidden+attn*gate_msa (residual_gate); cross-attn plain residual;
#     mlp_in=(1+c_scale)*rms(x)+c_shift; SwiGLU(gate/up/down, no bias);
#     x=x+mlp_out*c_gate (residual_gate).
#   - Self/cross attn (acestep.rs:916-1060): q/k/v/o linears NO bias; per-head
#     qk-rms (q_norm/k_norm [head_dim]); self-attn HALFSPLIT RoPE (Qwen3 rotate_half),
#     cross-attn NO RoPE; GQA repeat_kv n_rep = num_heads/num_kv_heads = 2.
#   - LoRA targets (acestep.rs:38-66 AceStepLoraTarget, 8/block): self_attn.{q,k,v,o}
#     + cross_attn.{q,k,v,o}. q/o: in=hidden out=q_dim; k/v: in=hidden out=kv_dim.

from std.collections import List


@fieldwise_init
struct AceStepTrainConfig(Copyable, Movable, ImplicitlyCopyable):
    var num_layers: Int       # 24 DiT layers
    var hidden_size: Int      # 2048
    var num_heads: Int        # 16
    var num_kv_heads: Int     # 8  (GQA n_rep = 16/8 = 2)
    var head_dim: Int         # 128
    var intermediate: Int     # 6144 (SwiGLU)
    var in_channels: Int      # 192 (context 128 + acoustic 64)
    var acoustic_dim: Int     # 64
    var patch_size: Int       # 2
    var rope_theta: Float64   # 1e6
    var rms_norm_eps: Float32 # 1e-6
    var sliding_window: Int   # 128
    var lora_rank: Int        # 16
    var lora_alpha: Float32   # 16 (scale = alpha/rank = 1.0)

    @staticmethod
    def turbo() -> AceStepTrainConfig:
        return AceStepTrainConfig(
            num_layers=24, hidden_size=2048, num_heads=16, num_kv_heads=8,
            head_dim=128, intermediate=6144, in_channels=192, acoustic_dim=64,
            patch_size=2, rope_theta=1_000_000.0, rms_norm_eps=1.0e-6,
            sliding_window=128, lora_rank=16, lora_alpha=16.0,
        )

    def n_rep(self) -> Int:
        return self.num_heads // self.num_kv_heads


# LoRA targets: 8 attention projections per block (acestep.rs:38-66, 353-361).
# q/o: in=hidden out=q_dim(=hidden) ; k/v: in=hidden out=kv_dim.
def acestep_lora_targets() -> List[String]:
    var t = List[String]()
    t.append(String("self_attn.q_proj"))
    t.append(String("self_attn.k_proj"))
    t.append(String("self_attn.v_proj"))
    t.append(String("self_attn.o_proj"))
    t.append(String("cross_attn.q_proj"))
    t.append(String("cross_attn.k_proj"))
    t.append(String("cross_attn.v_proj"))
    t.append(String("cross_attn.o_proj"))
    return t^
