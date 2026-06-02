# training/train_config.mojo — the ONE training+model config descriptor.
#
# Carries the COMPLETE per-run description read from a JSON config file
# (serenitymojo/configs/<model>.json): model arch + LoRA/optimizer recipe +
# cadence + checkpoint/vae paths. This is the SINGLE SOURCE OF TRUTH — no
# trainer or sampler may hardcode arch/dims/recipe (binding user rule,
# 2026-05-31: "all trainers must read params from config files").
#
# Design note (OneTrainer parity): OneTrainer's BaseConfig declares every field
# with a (name, default, type) triple and a generic from_dict mutates them by
# name. Mojo 1.0.0b1 has NO runtime reflection, so we adapt the SAME shape:
# `TrainConfig.default()` provides every default; train_config_reader mutates
# fields by JSON key as it parses. Adding a model = adding a JSON file, never code.
#
# Mojo constraint: the attention SHAPE generics (H, Dh, N_IMG, N_TXT, S) are
# COMPTIME params of the stack-forward functions and cannot be purely file-
# driven. The trainer keeps them comptime and ASSERTS they match this config
# (H*Dh == d_model, H == n_heads, Dh == head_dim). Everything else is runtime.

from std.collections import List


struct TrainConfig(Copyable, Movable):
    # ── identity + paths ──
    var name: String          # "model_type" (e.g. "klein")
    var checkpoint: String    # base model safetensors path
    var vae: String           # vae safetensors path
    var validation_prompts_file: String  # shared sample prompt/caps JSON

    # ── architecture (from the checkpoint header; carried in the config file) ──
    var d_model: Int               # inner_dim (== n_heads * head_dim)
    var in_channels: Int           # img_in input channels
    var joint_attention_dim: Int   # txt_in input channels
    var out_channels: Int          # final_layer.linear output channels
    var num_double: Int            # double-stream block count
    var num_single: Int            # single-stream block count
    var n_heads: Int
    var head_dim: Int
    var mlp_hidden: Int            # SwiGLU per-gate hidden (fc1 stores 2*this)
    var timestep_dim: Int          # time_in input dim
    var rope_theta: Float64

    # ── recipe ──
    var lr: Float32
    var lora_rank: Int
    var lora_alpha: Float32
    var timestep_shift: Float32
    var eps: Float32               # optimizer epsilon
    var weight_decay: Float32
    var beta1: Float32
    var beta2: Float32
    var max_grad_norm: Float32

    # ── lr scheduler (Wave 2A item 2a; default-off == flat constant lr) ──
    # lr_scheduler: 0=Constant 1=Linear 2=Cosine 3=CosineRestarts 4=Polynomial 5=Rex
    # (matches training/lr_schedule.mojo LR_* comptime ints). Default 0 + warmup 0
    # makes lr_for_step return cfg.lr for every step => baseline byte-unchanged.
    var lr_scheduler: Int
    var lr_warmup_steps: Int
    var lr_min_factor: Float32
    var lr_cycles: Float32

    # ── loss weighting (Wave 2A item 2b; default-off == unweighted MSE) ──
    # min_snr_gamma < 0 is the "off" sentinel (no MIN-SNR weighting). debiased
    # off by default. is_v_prediction true for flow-matching trainers (Klein).
    var min_snr_gamma: Float32
    var debiased: Bool

    # ── combined loss (Wave 2A item 2c; default-off == MSE-only) ──
    # mse=1, mae=0, huber=0 reproduces the bare MSE loss exactly.
    var loss_mse_strength: Float32
    var loss_mae_strength: Float32
    var loss_huber_strength: Float32

    # ── timestep bias (Wave 2A item 2f; default-off == identity) ──
    # 0=None 1=Later 2=Earlier 3=Range (matches timestep_bias.mojo TSB_* ints).
    var timestep_bias_strategy: Int
    var timestep_bias_multiplier: Float32
    var timestep_bias_range_min: Float32
    var timestep_bias_range_max: Float32

    # ── timestep distribution (Wave 2A item 2g; default keeps production path) ──
    # -1 = "production default" (logit-normal+qwen-shift via sample_timestep_logit_normal);
    # 0=Uniform 1=Sigmoid 2=LogitNormal (matches timestep_dist.mojo TSD_* ints).
    # Leaving this at -1 means the existing schedule.mojo path is used unchanged.
    var timestep_distribution: Int
    var timestep_noising_weight: Float32
    var timestep_noising_bias: Float32

    # ── caption dropout (Wave 2B item 2d; default-off == 0 == never drop) ──
    # With prob>0, each step draws a uniform and (if draw<prob) swaps the
    # conditional text embedding for the cached uncond (zero) embedding. p=0
    # never draws and never drops => baseline byte-unchanged.
    var caption_dropout_prob: Float32

    # ── noise modifiers (Wave 2B item 2e; ALL default-off) ──
    # offset_noise_weight<=0 OR offset_noise_prob<=0 => no offset noise.
    # input_perturbation<=0 => no input perturbation. multires_iterations==0 OR
    # multires_discount<=0 => no pyramid noise. All-off == pure-Gaussian noise.
    var offset_noise_weight: Float32
    var offset_noise_prob: Float32
    var input_perturbation: Float32
    var multires_iterations: Int
    var multires_discount: Float32

    # ── gradient accumulation (Wave 2B item 2h; default-off == 1) ──
    # Accumulate grads across N micro-steps, then clip+AdamW once every N.
    # grad_accum_steps=1 == current per-step behavior.
    var grad_accum_steps: Int

    # ── EMA (Wave 2B item 2i; default-off == disabled, no shadow alloc) ──
    # ema_enabled False => no shadow copies, no EMA update. The schedule fields
    # mirror EDv2 ema_advanced.rs EmaConfig (diffusers power-decay).
    var ema_enabled: Bool
    var ema_inv_gamma: Float32
    var ema_power: Float32
    var ema_update_after_step: Int
    var ema_min_decay: Float32
    var ema_max_decay: Float32

    # ── adapter algo selector (Wave 2B item 2j; default-off == plain LoRA) ──
    # 0=plain LoRA (low-rank A/B), 1=LyCORIS Full (full-shape weight delta),
    # 2=LyCORIS LoHa (Hadamard of two rank-r products; loha_adapter.mojo),
    # 3=DoRA (weight-decomposed LoRA: magnitude × normalized direction;
    #   dora_adapter.mojo), 4=LyCORIS LoKr (Kronecker product delta;
    #   lokr_adapter.mojo). 1..4 are primitives gated by their *_smoke.mojo and
    #   fail loud in the Klein stack until the integration wave wires them.
    var adapter_algo: Int

    # ── cadence ──
    var max_steps: Int
    var save_every: Int
    var sample_every: Int

    def n_layers(self) -> Int:
        """Total block count (back-compat convenience)."""
        return self.num_double + self.num_single

    @staticmethod
    def default() -> TrainConfig:
        """All-defaults descriptor; the reader overwrites present keys by name.
        Mirrors OneTrainer BaseConfig.default_values."""
        return TrainConfig(
            name=String("unknown"),
            checkpoint=String(""),
            vae=String(""),
            validation_prompts_file=String(""),
            d_model=0,
            in_channels=0,
            joint_attention_dim=0,
            out_channels=0,
            num_double=0,
            num_single=0,
            n_heads=0,
            head_dim=0,
            mlp_hidden=0,
            timestep_dim=256,
            rope_theta=Float64(2000.0),
            lr=Float32(1.0e-4),
            lora_rank=16,
            lora_alpha=Float32(16.0),
            timestep_shift=Float32(1.0),
            eps=Float32(1.0e-8),
            weight_decay=Float32(0.01),
            beta1=Float32(0.9),
            beta2=Float32(0.999),
            max_grad_norm=Float32(1.0),
            lr_scheduler=0,                  # Constant (default-off)
            lr_warmup_steps=0,               # no warmup (default-off)
            lr_min_factor=Float32(0.0),
            lr_cycles=Float32(1.0),
            min_snr_gamma=Float32(-1.0),     # <0 sentinel = off
            debiased=False,
            loss_mse_strength=Float32(1.0),  # MSE-only (default-off)
            loss_mae_strength=Float32(0.0),
            loss_huber_strength=Float32(0.0),
            timestep_bias_strategy=0,        # None (identity, default-off)
            timestep_bias_multiplier=Float32(0.0),
            timestep_bias_range_min=Float32(0.0),
            timestep_bias_range_max=Float32(1.0),
            timestep_distribution=-1,        # production logit-normal path (default)
            timestep_noising_weight=Float32(0.0),
            timestep_noising_bias=Float32(0.0),
            caption_dropout_prob=Float32(0.0),   # never drop (default-off)
            offset_noise_weight=Float32(0.0),    # no offset noise (default-off)
            offset_noise_prob=Float32(0.0),
            input_perturbation=Float32(0.0),     # no input perturbation (default-off)
            multires_iterations=0,               # no pyramid noise (default-off)
            multires_discount=Float32(0.0),
            grad_accum_steps=1,                  # per-step AdamW (default-off)
            ema_enabled=False,                   # no shadow params (default-off)
            ema_inv_gamma=Float32(1.0),          # EDv2 EmaConfig defaults
            ema_power=Float32(0.6667),
            ema_update_after_step=0,
            ema_min_decay=Float32(0.0),
            ema_max_decay=Float32(0.9999),
            adapter_algo=0,                      # plain LoRA (default-off)
            max_steps=3000,
            save_every=500,
            sample_every=250,
        )

    def __init__(
        out self, var name: String, var checkpoint: String, var vae: String,
        var validation_prompts_file: String,
        d_model: Int, in_channels: Int, joint_attention_dim: Int, out_channels: Int,
        num_double: Int, num_single: Int, n_heads: Int, head_dim: Int,
        mlp_hidden: Int, timestep_dim: Int, rope_theta: Float64,
        lr: Float32, lora_rank: Int, lora_alpha: Float32, timestep_shift: Float32,
        eps: Float32, weight_decay: Float32, beta1: Float32, beta2: Float32,
        max_grad_norm: Float32,
        lr_scheduler: Int, lr_warmup_steps: Int, lr_min_factor: Float32, lr_cycles: Float32,
        min_snr_gamma: Float32, debiased: Bool,
        loss_mse_strength: Float32, loss_mae_strength: Float32, loss_huber_strength: Float32,
        timestep_bias_strategy: Int, timestep_bias_multiplier: Float32,
        timestep_bias_range_min: Float32, timestep_bias_range_max: Float32,
        timestep_distribution: Int, timestep_noising_weight: Float32,
        timestep_noising_bias: Float32,
        caption_dropout_prob: Float32,
        offset_noise_weight: Float32, offset_noise_prob: Float32,
        input_perturbation: Float32,
        multires_iterations: Int, multires_discount: Float32,
        grad_accum_steps: Int,
        ema_enabled: Bool, ema_inv_gamma: Float32, ema_power: Float32,
        ema_update_after_step: Int, ema_min_decay: Float32, ema_max_decay: Float32,
        adapter_algo: Int,
        max_steps: Int, save_every: Int, sample_every: Int,
    ):
        self.name = name^
        self.checkpoint = checkpoint^
        self.vae = vae^
        self.validation_prompts_file = validation_prompts_file^
        self.d_model = d_model
        self.in_channels = in_channels
        self.joint_attention_dim = joint_attention_dim
        self.out_channels = out_channels
        self.num_double = num_double
        self.num_single = num_single
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.mlp_hidden = mlp_hidden
        self.timestep_dim = timestep_dim
        self.rope_theta = rope_theta
        self.lr = lr
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.timestep_shift = timestep_shift
        self.eps = eps
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_grad_norm = max_grad_norm
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_min_factor = lr_min_factor
        self.lr_cycles = lr_cycles
        self.min_snr_gamma = min_snr_gamma
        self.debiased = debiased
        self.loss_mse_strength = loss_mse_strength
        self.loss_mae_strength = loss_mae_strength
        self.loss_huber_strength = loss_huber_strength
        self.timestep_bias_strategy = timestep_bias_strategy
        self.timestep_bias_multiplier = timestep_bias_multiplier
        self.timestep_bias_range_min = timestep_bias_range_min
        self.timestep_bias_range_max = timestep_bias_range_max
        self.timestep_distribution = timestep_distribution
        self.timestep_noising_weight = timestep_noising_weight
        self.timestep_noising_bias = timestep_noising_bias
        self.caption_dropout_prob = caption_dropout_prob
        self.offset_noise_weight = offset_noise_weight
        self.offset_noise_prob = offset_noise_prob
        self.input_perturbation = input_perturbation
        self.multires_iterations = multires_iterations
        self.multires_discount = multires_discount
        self.grad_accum_steps = grad_accum_steps
        self.ema_enabled = ema_enabled
        self.ema_inv_gamma = ema_inv_gamma
        self.ema_power = ema_power
        self.ema_update_after_step = ema_update_after_step
        self.ema_min_decay = ema_min_decay
        self.ema_max_decay = ema_max_decay
        self.adapter_algo = adapter_algo
        self.max_steps = max_steps
        self.save_every = save_every
        self.sample_every = sample_every
