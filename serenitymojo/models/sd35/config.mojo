# serenitymojo/models/sd35/config.mojo — SD3.5 MMDiT per-variant dims + recipe.
#
# SD3.5 is a JOINT MMDiT: each `joint_blocks.{i}` is a JointTransformerBlock with
# a dual stream (image `x_block` + text `context_block`) coupled by ONE joint
# attention (concat ctx+x -> SDPA -> split). Closest existing analog is the Klein
# FLUX.2 DOUBLE-stream block (models/klein/double_block.mojo), but SD3.5 differs:
#   - token norm  : LayerNorm (no affine, eps 1e-6)   — Klein uses RMSNorm
#   - MLP         : fc1 -> GELU(tanh) -> fc2           — Klein uses SwiGLU
#   - NO RoPE     : pos_embed added once pre-block     — Klein uses interleaved RoPE
#   - gating      : out = x + gate[:,None,:] * proj(att) (broadcast mul + add)
#   - modulation  : shift/scale/gate are PER-SAMPLE [hidden] vectors (from
#                   adaLN_modulation.1(silu(c))), broadcast over tokens
#   - mods count  : context_block 6 (2 if pre_only/last); x_block 6 (9 if dual-attn)
#   - QK norm     : RMSNorm over head_dim (ln_q / ln_k, no bias)
#   - joint order : ctx FIRST, then x (concat axis = sequence)
#
# DIMS CONFIRMED from the real checkpoint header
#   /home/alex/.serenity/models/checkpoints/sd3.5_medium.safetensors
#   (read via python struct, keys under "model.diffusion_model."):
#     x_embedder.proj.weight            [1536, 16, 2, 2]  -> hidden 1536, in_ch 16, patch 2
#     context_embedder.weight           [1536, 4096]      -> context_dim 4096
#     t_embedder.mlp.0.weight           [1536, 256]       -> timestep_dim 256
#     joint_blocks.0.x_block.attn.qkv.weight   [4608, 1536] -> 3*1536; heads 24, head_dim 64
#     joint_blocks.0.x_block.attn.ln_q.weight  [64]         -> head_dim 64 (RMSNorm)
#     joint_blocks.0.x_block.mlp.fc1.weight    [6144, 1536] -> mlp_hidden 6144 (4*hidden), GELU
#     joint_blocks.0.x_block.adaLN_modulation.1.weight   [13824, 1536] -> 9*1536 (dual)
#     joint_blocks.0.context_block.adaLN_modulation.1.weight [9216, 1536] -> 6*1536
#     final_layer.linear.weight         [64, 1536]        -> 2*2*16 patch-vector out
#     n joint blocks = 24; dual-attention blocks = 0..12; pre_only block = 23
#   sd3.5_large: depth 38, hidden 2432, heads 38, head_dim 64, NO dual attention.
#
# RECIPE cited from EDv2 trainer
#   crates/eridiffusion-cli/src/bin/train_sd35.rs:
#     :5-11  logit-normal timestep (shift default 1.0); sigma=(floor(t)+1)/1000;
#            noisy = sigma*noise + (1-sigma)*latent; target = noise - latent (rectified flow)
#     :11    loss = mean MSE in F32 (no v-pred preconditioning)
#     :49    TIMESTEP_SHIFT_DEFAULT = 1.0   (inference-time schedule uses 3.0, :146)
#     :46    NUM_TRAIN_TIMESTEPS = 1000
#   crates/eridiffusion-core/src/models/sd35.rs:
#     :32-39 LoRA targets (kohya SD3 SingleDiTBlock set): per joint block
#            x_block.attn.{qkv,proj}, x_block.mlp.{fc1,fc2},
#            context_block.attn.{qkv,proj}, context_block.mlp.{fc1,fc2},
#            dual blocks add x_block.attn2.{qkv,proj}.
#     :314   rank = config.lora_rank ; alpha = config.lora_alpha.


from serenitymojo.io.dtype import STDtype


struct SD35Config(Copyable, Movable):
    var name: String
    var checkpoint: String
    var hidden: Int            # model dim (== n_heads * head_dim)
    var context_dim: Int       # context_embedder input (T5/CLIP joint dim)
    var pooled_dim: Int        # y_embedder input (pooled CLIP)
    var timestep_dim: Int      # t_embedder sinusoidal dim
    var in_channels: Int       # latent channels
    var patch_size: Int
    var n_heads: Int
    var head_dim: Int
    var mlp_hidden: Int        # fc1 output (GELU); 4*hidden
    var depth: Int             # number of joint_blocks
    var num_dual_blocks: Int   # first N x_blocks carry attn2 (Medium); 0 for Large
    # ── recipe ──
    var lr: Float32
    var lora_rank: Int
    var lora_alpha: Float32
    var timestep_shift: Float32
    var num_train_timesteps: Int
    var eps: Float32           # optimizer epsilon
    var qk_norm_eps: Float32   # RMSNorm eps for ln_q/ln_k
    var ln_eps: Float32        # LayerNorm (no affine) eps

    def __init__(
        out self,
        var name: String, var checkpoint: String,
        hidden: Int, context_dim: Int, pooled_dim: Int, timestep_dim: Int,
        in_channels: Int, patch_size: Int, n_heads: Int, head_dim: Int,
        mlp_hidden: Int, depth: Int, num_dual_blocks: Int,
        lr: Float32, lora_rank: Int, lora_alpha: Float32,
        timestep_shift: Float32, num_train_timesteps: Int,
        eps: Float32, qk_norm_eps: Float32, ln_eps: Float32,
    ):
        self.name = name^
        self.checkpoint = checkpoint^
        self.hidden = hidden
        self.context_dim = context_dim
        self.pooled_dim = pooled_dim
        self.timestep_dim = timestep_dim
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.mlp_hidden = mlp_hidden
        self.depth = depth
        self.num_dual_blocks = num_dual_blocks
        self.lr = lr
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.timestep_shift = timestep_shift
        self.num_train_timesteps = num_train_timesteps
        self.eps = eps
        self.qk_norm_eps = qk_norm_eps
        self.ln_eps = ln_eps


def sd35_medium() -> SD35Config:
    # Confirmed from sd3.5_medium.safetensors header (see file docstring).
    return SD35Config(
        String("sd3_5_medium"),
        String("/home/alex/.serenity/models/checkpoints/sd3.5_medium.safetensors"),
        1536,    # hidden
        4096,    # context_dim
        2048,    # pooled_dim (CLIP-L 768 + CLIP-G 1280)
        256,     # timestep_dim
        16,      # in_channels
        2,       # patch_size
        24,      # n_heads
        64,      # head_dim
        6144,    # mlp_hidden (4*hidden)
        24,      # depth
        13,      # num_dual_blocks (blocks 0..12)
        Float32(1e-4),   # lr
        16,              # lora_rank
        Float32(16.0),   # lora_alpha
        Float32(1.0),    # timestep_shift (train default; sample uses 3.0)
        1000,            # num_train_timesteps
        Float32(1e-8),   # optimizer eps
        Float32(1e-6),   # qk_norm_eps (RMSNorm)
        Float32(1e-6),   # ln_eps (LayerNorm no-affine)
    )


def sd35_large() -> SD35Config:
    return SD35Config(
        String("sd3_5_large"),
        String("/home/alex/.serenity/models/checkpoints/sd3.5_large.safetensors"),
        2432,    # hidden
        4096,    # context_dim
        2048,    # pooled_dim
        256,     # timestep_dim
        16,      # in_channels
        2,       # patch_size
        38,      # n_heads
        64,      # head_dim
        9728,    # mlp_hidden (4*hidden)
        38,      # depth
        0,       # num_dual_blocks (Large has no dual attention)
        Float32(1e-4),
        16,
        Float32(16.0),
        Float32(1.0),
        1000,
        Float32(1e-8),
        Float32(1e-6),
        Float32(1e-6),
    )
