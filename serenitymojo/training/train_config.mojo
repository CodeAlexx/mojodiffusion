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


# Shared encodings for trainer mode/target selectors. These are runtime config
# values; model kernels still choose shape-critical paths with comptime params.
comptime TRAIN_MODALITY_VIDEO = 0
comptime TRAIN_MODALITY_AV = 1
comptime TRAIN_MODALITY_AUDIO = 2

comptime LORA_TARGET_LEGACY_VIDEO_ATTN1 = 0
comptime LORA_TARGET_LTX2_T2V = 1
comptime LORA_TARGET_LTX2_V2V = 2
comptime LORA_TARGET_LTX2_AUDIO = 3
comptime LORA_TARGET_LTX2_AUDIO_REF_ONLY_IC = 4
comptime LORA_TARGET_LTX2_FULL = 5

# OneTrainer TrainingMethod string enum tags. Preserve baseline LoRA behavior
# unless a config explicitly selects FINE_TUNE/full finetune.
comptime TRAINING_METHOD_LORA = 0
comptime TRAINING_METHOD_FINE_TUNE = 1

# OneTrainer GradientCheckpointingMethod string enum:
# OFF | ON | CPU_OFFLOADED. Keep these as typed ints so downstream Mojo code
# can branch without string compares in hot setup paths.
comptime GRADIENT_CHECKPOINTING_OFF = 0
comptime GRADIENT_CHECKPOINTING_ON = 1
comptime GRADIENT_CHECKPOINTING_CPU_OFFLOADED = 2

# OneTrainer DataType string enum. These are storage/request tags only; model
# code must choose kernels without silently upcasting stored BF16/F16/FP8 values.
comptime TRAIN_DTYPE_NONE = 0
comptime TRAIN_DTYPE_FLOAT_8 = 1
comptime TRAIN_DTYPE_FLOAT_16 = 2
comptime TRAIN_DTYPE_FLOAT_32 = 3
comptime TRAIN_DTYPE_BFLOAT_16 = 4
comptime TRAIN_DTYPE_TFLOAT_32 = 5
comptime TRAIN_DTYPE_INT_8 = 6
comptime TRAIN_DTYPE_NFLOAT_4 = 7
comptime TRAIN_DTYPE_FLOAT_W8A8 = 8
comptime TRAIN_DTYPE_INT_W8A8 = 9
comptime TRAIN_DTYPE_GGUF = 10
comptime TRAIN_DTYPE_GGUF_A8_FLOAT = 11
comptime TRAIN_DTYPE_GGUF_A8_INT = 12

# OneTrainer Optimizer tags. Current real LoRA loops still execute AdamW; these
# fields let loop validation fail loud instead of ignoring an unsupported preset.
comptime TRAIN_OPTIMIZER_ADAMW = 0
comptime TRAIN_OPTIMIZER_ADAM = 1
comptime TRAIN_OPTIMIZER_ADAFACTOR = 2
comptime TRAIN_OPTIMIZER_CAME = 3
comptime TRAIN_OPTIMIZER_LION = 4
comptime TRAIN_OPTIMIZER_PRODIGY = 5
comptime TRAIN_OPTIMIZER_SGD = 6
comptime TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW = 7

comptime TRAIN_TIME_UNIT_EPOCH = 0
comptime TRAIN_TIME_UNIT_STEP = 1
comptime TRAIN_TIME_UNIT_SECOND = 2
comptime TRAIN_TIME_UNIT_MINUTE = 3
comptime TRAIN_TIME_UNIT_HOUR = 4
comptime TRAIN_TIME_UNIT_NEVER = 5
comptime TRAIN_TIME_UNIT_ALWAYS = 6

comptime EMA_MODE_OFF = 0
comptime EMA_MODE_GPU = 1
comptime EMA_MODE_CPU = 2


struct TrainConfig(Copyable, Movable):
    # ── identity + paths ──
    var name: String          # "model_type" (e.g. "klein")
    var checkpoint: String    # base model safetensors path
    var vae: String           # vae safetensors path
    var validation_prompts_file: String  # shared sample prompt/caps JSON
    var base_model_name: String
    var workspace_dir: String
    var cache_dir: String
    var output_model_destination: String
    var output_model_format: String
    var concept_file_name: String
    var sample_definition_file_name: String

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

    # -- OneTrainer training method selector --
    # 0=LORA, 1=FINE_TUNE. This is mechanical config/mode wiring only; model
    # train loops must opt into full-finetune behavior explicitly.
    var training_method: Int

    # ── OneTrainer runtime/control fields ──
    var batch_size: Int
    var epochs: Int
    var stop_training_after: Int
    var stop_training_after_unit: Int
    var resolution: String
    var frames: String
    var seed: UInt64
    var compile_model: Bool
    var dataloader_threads: Int
    var latent_caching: Bool
    var clear_cache_before_training: Bool
    var only_cache: Bool
    var tensorboard: Bool
    var tensorboard_always_on: Bool
    var samples_to_tensorboard: Bool

    # ── OneTrainer dtype policy tags ──
    var train_dtype: Int
    var fallback_train_dtype: Int
    var weight_dtype: Int
    var output_dtype: Int
    var lora_weight_dtype: Int
    var embedding_weight_dtype: Int
    var unet_weight_dtype: Int
    var prior_weight_dtype: Int
    var transformer_weight_dtype: Int
    var text_encoder_weight_dtype: Int
    var text_encoder_2_weight_dtype: Int
    var text_encoder_3_weight_dtype: Int
    var text_encoder_4_weight_dtype: Int
    var vae_weight_dtype: Int

    # ── OneTrainer model-part flags/names used by presets ──
    var unet_model_name: String
    var prior_model_name: String
    var transformer_model_name: String
    var text_encoder_model_name: String
    var text_encoder_2_model_name: String
    var text_encoder_3_model_name: String
    var text_encoder_4_model_name: String
    var vae_model_name: String
    var unet_train: Bool
    var prior_train: Bool
    var transformer_train: Bool
    var text_encoder_train: Bool
    var text_encoder_2_train: Bool
    var text_encoder_3_train: Bool
    var text_encoder_4_train: Bool
    var vae_train: Bool

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
    var optimizer: Int
    var optimizer_eps2: Float32
    var optimizer_clip_threshold: Float32
    var optimizer_decay_rate: Float32
    var optimizer_relative_step: Bool
    var optimizer_scale_parameter: Bool
    var optimizer_warmup_init: Bool
    var optimizer_fused: Bool
    var optimizer_fused_back_pass: Bool
    var optimizer_stochastic_rounding: Bool

    # ── OneTrainer layer/quantization selectors ──
    var layer_filter: String
    var layer_filter_preset: String
    var layer_filter_regex: Bool
    var quantization_layer_filter: String
    var quantization_layer_filter_preset: String
    var quantization_layer_filter_regex: Bool

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
    var ema_mode: Int
    var ema_decay: Float32
    var ema_update_step_interval: Int

    # ── OneTrainer offload/checkpoint policy ──
    # Defaults mirror OneTrainer TrainConfig:
    # gradient_checkpointing=ON, enable_async_offloading=True,
    # enable_activation_offloading=True, layer_offload_fraction=0.0.
    # CPU-side activation/layer offload activates only for CPU_OFFLOADED. These
    # are policy scalars only; they must not convert activations to host F32.
    var gradient_checkpointing: Int
    var enable_async_offloading: Bool
    var enable_activation_offloading: Bool
    var layer_offload_fraction: Float64

    # ── adapter algo selector (Wave 2B item 2j; default-off == plain LoRA) ──
    # 0=plain LoRA (low-rank A/B), 1=LyCORIS Full (full-shape weight delta),
    # 2=LyCORIS LoHa (Hadamard of two rank-r products; loha_adapter.mojo),
    # 3=DoRA (weight-decomposed LoRA: magnitude × normalized direction;
    #   dora_adapter.mojo), 4=LyCORIS LoKr (Kronecker product delta;
    #   lokr_adapter.mojo). 1..4 are primitives gated by their *_smoke.mojo and
    #   fail loud in the Klein stack until the integration wave wires them.
    var adapter_algo: Int

    # ── cached-input / AV trainer contract (default-off) ──
    # train_modality: 0=video, 1=audio-video, 2=audio.
    # lora_target_preset mirrors musubi LTX2 presets:
    #   0=legacy_video_attn1, 1=t2v, 2=v2v, 3=audio,
    #   4=audio_ref_only_ic, 5=full.
    # Production AV trainers should set modality=AV, require cached video/text/
    # audio inputs, and keep the hot forward/loss/backward/update loop on device.
    var train_modality: Int
    var lora_target_preset: Int
    var dataset_cache_dir: String
    var require_cached_video_latents: Bool
    var require_cached_text_embeddings: Bool
    var require_cached_audio_latents: Bool
    var hot_loop_device_only: Bool
    var video_loss_weight: Float32
    var audio_loss_weight: Float32

    # ── OneTrainer validation / sample / save / backup cadence ──
    var validation: Bool
    var validate_after: Int
    var validate_after_unit: Int
    var continue_last_backup: Bool
    var sample_after: Int
    var sample_after_unit: Int
    var sample_skip_first: Int
    var non_ema_sampling: Bool
    var backup_after: Int
    var backup_after_unit: Int
    var rolling_backup: Bool
    var rolling_backup_count: Int
    var backup_before_save: Bool
    var save_every_unit: Int
    var save_skip_first: Int
    var save_filename_prefix: String

    # ── OneTrainer masked/prior-preservation flags ──
    var masked_training: Bool
    var unmasked_probability: Float32
    var unmasked_weight: Float32
    var normalize_masked_area_loss: Bool
    var masked_prior_preservation_weight: Float32
    var custom_conditioning_image: Bool

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
            base_model_name=String(""),
            workspace_dir=String("workspace/run"),
            cache_dir=String("workspace-cache/run"),
            output_model_destination=String(""),
            output_model_format=String("SAFETENSORS"),
            concept_file_name=String("training_concepts/concepts.json"),
            sample_definition_file_name=String("training_samples/samples.json"),
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
            training_method=TRAINING_METHOD_LORA,
            batch_size=1,
            epochs=100,
            stop_training_after=0,
            stop_training_after_unit=TRAIN_TIME_UNIT_NEVER,
            resolution=String("512"),
            frames=String("25"),
            seed=UInt64(42),
            compile_model=False,
            dataloader_threads=2,
            latent_caching=True,
            clear_cache_before_training=True,
            only_cache=False,
            tensorboard=True,
            tensorboard_always_on=False,
            samples_to_tensorboard=True,
            train_dtype=TRAIN_DTYPE_FLOAT_16,
            fallback_train_dtype=TRAIN_DTYPE_BFLOAT_16,
            weight_dtype=TRAIN_DTYPE_NONE,
            output_dtype=TRAIN_DTYPE_FLOAT_32,
            lora_weight_dtype=TRAIN_DTYPE_FLOAT_32,
            embedding_weight_dtype=TRAIN_DTYPE_FLOAT_32,
            unet_weight_dtype=TRAIN_DTYPE_FLOAT_32,
            prior_weight_dtype=TRAIN_DTYPE_FLOAT_32,
            transformer_weight_dtype=TRAIN_DTYPE_FLOAT_32,
            text_encoder_weight_dtype=TRAIN_DTYPE_FLOAT_32,
            text_encoder_2_weight_dtype=TRAIN_DTYPE_FLOAT_32,
            text_encoder_3_weight_dtype=TRAIN_DTYPE_FLOAT_32,
            text_encoder_4_weight_dtype=TRAIN_DTYPE_FLOAT_32,
            vae_weight_dtype=TRAIN_DTYPE_FLOAT_32,
            unet_model_name=String(""),
            prior_model_name=String(""),
            transformer_model_name=String(""),
            text_encoder_model_name=String(""),
            text_encoder_2_model_name=String(""),
            text_encoder_3_model_name=String(""),
            text_encoder_4_model_name=String(""),
            vae_model_name=String(""),
            unet_train=True,
            prior_train=True,
            transformer_train=True,
            text_encoder_train=True,
            text_encoder_2_train=True,
            text_encoder_3_train=True,
            text_encoder_4_train=True,
            vae_train=True,
            lr=Float32(1.0e-4),
            lora_rank=16,
            lora_alpha=Float32(16.0),
            timestep_shift=Float32(1.0),
            eps=Float32(1.0e-8),
            weight_decay=Float32(0.01),
            beta1=Float32(0.9),
            beta2=Float32(0.999),
            max_grad_norm=Float32(1.0),
            optimizer=TRAIN_OPTIMIZER_ADAMW,
            optimizer_eps2=Float32(0.0),
            optimizer_clip_threshold=Float32(0.0),
            optimizer_decay_rate=Float32(0.0),
            optimizer_relative_step=False,
            optimizer_scale_parameter=False,
            optimizer_warmup_init=False,
            optimizer_fused=False,
            optimizer_fused_back_pass=False,
            optimizer_stochastic_rounding=True,
            layer_filter=String(""),
            layer_filter_preset=String("full"),
            layer_filter_regex=False,
            quantization_layer_filter=String(""),
            quantization_layer_filter_preset=String("full"),
            quantization_layer_filter_regex=False,
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
            ema_mode=EMA_MODE_OFF,
            ema_decay=Float32(0.999),
            ema_update_step_interval=5,
            gradient_checkpointing=GRADIENT_CHECKPOINTING_ON,
            enable_async_offloading=True,
            enable_activation_offloading=True,
            layer_offload_fraction=Float64(0.0),
            adapter_algo=0,                      # plain LoRA (default-off)
            train_modality=TRAIN_MODALITY_VIDEO,
            lora_target_preset=LORA_TARGET_LEGACY_VIDEO_ATTN1,
            dataset_cache_dir=String(""),
            require_cached_video_latents=False,
            require_cached_text_embeddings=False,
            require_cached_audio_latents=False,
            hot_loop_device_only=False,
            video_loss_weight=Float32(1.0),
            audio_loss_weight=Float32(0.0),
            validation=False,
            validate_after=1,
            validate_after_unit=TRAIN_TIME_UNIT_EPOCH,
            continue_last_backup=False,
            sample_after=10,
            sample_after_unit=TRAIN_TIME_UNIT_MINUTE,
            sample_skip_first=0,
            non_ema_sampling=True,
            backup_after=30,
            backup_after_unit=TRAIN_TIME_UNIT_MINUTE,
            rolling_backup=False,
            rolling_backup_count=3,
            backup_before_save=True,
            save_every_unit=TRAIN_TIME_UNIT_NEVER,
            save_skip_first=0,
            save_filename_prefix=String(""),
            masked_training=False,
            unmasked_probability=Float32(0.1),
            unmasked_weight=Float32(0.1),
            normalize_masked_area_loss=False,
            masked_prior_preservation_weight=Float32(0.0),
            custom_conditioning_image=False,
            max_steps=3000,
            save_every=500,
            sample_every=250,
        )

    def __init__(
        out self, var name: String, var checkpoint: String, var vae: String,
        var validation_prompts_file: String, var base_model_name: String,
        var workspace_dir: String, var cache_dir: String,
        var output_model_destination: String, var output_model_format: String,
        var concept_file_name: String, var sample_definition_file_name: String,
        d_model: Int, in_channels: Int, joint_attention_dim: Int, out_channels: Int,
        num_double: Int, num_single: Int, n_heads: Int, head_dim: Int,
        mlp_hidden: Int, timestep_dim: Int, rope_theta: Float64,
        training_method: Int,
        batch_size: Int, epochs: Int, stop_training_after: Int,
        stop_training_after_unit: Int, var resolution: String, var frames: String,
        seed: UInt64, compile_model: Bool, dataloader_threads: Int,
        latent_caching: Bool, clear_cache_before_training: Bool, only_cache: Bool,
        tensorboard: Bool, tensorboard_always_on: Bool, samples_to_tensorboard: Bool,
        train_dtype: Int, fallback_train_dtype: Int, weight_dtype: Int,
        output_dtype: Int, lora_weight_dtype: Int, embedding_weight_dtype: Int,
        unet_weight_dtype: Int, prior_weight_dtype: Int, transformer_weight_dtype: Int,
        text_encoder_weight_dtype: Int, text_encoder_2_weight_dtype: Int,
        text_encoder_3_weight_dtype: Int, text_encoder_4_weight_dtype: Int,
        vae_weight_dtype: Int,
        var unet_model_name: String, var prior_model_name: String,
        var transformer_model_name: String, var text_encoder_model_name: String,
        var text_encoder_2_model_name: String, var text_encoder_3_model_name: String,
        var text_encoder_4_model_name: String, var vae_model_name: String,
        unet_train: Bool, prior_train: Bool, transformer_train: Bool,
        text_encoder_train: Bool, text_encoder_2_train: Bool,
        text_encoder_3_train: Bool, text_encoder_4_train: Bool, vae_train: Bool,
        lr: Float32, lora_rank: Int, lora_alpha: Float32, timestep_shift: Float32,
        eps: Float32, weight_decay: Float32, beta1: Float32, beta2: Float32,
        max_grad_norm: Float32,
        optimizer: Int, optimizer_eps2: Float32, optimizer_clip_threshold: Float32,
        optimizer_decay_rate: Float32, optimizer_relative_step: Bool,
        optimizer_scale_parameter: Bool, optimizer_warmup_init: Bool,
        optimizer_fused: Bool, optimizer_fused_back_pass: Bool,
        optimizer_stochastic_rounding: Bool,
        var layer_filter: String, var layer_filter_preset: String,
        layer_filter_regex: Bool, var quantization_layer_filter: String,
        var quantization_layer_filter_preset: String,
        quantization_layer_filter_regex: Bool,
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
        ema_mode: Int, ema_decay: Float32, ema_update_step_interval: Int,
        gradient_checkpointing: Int, enable_async_offloading: Bool,
        enable_activation_offloading: Bool, layer_offload_fraction: Float64,
        adapter_algo: Int,
        train_modality: Int, lora_target_preset: Int, var dataset_cache_dir: String,
        require_cached_video_latents: Bool, require_cached_text_embeddings: Bool,
        require_cached_audio_latents: Bool, hot_loop_device_only: Bool,
        video_loss_weight: Float32, audio_loss_weight: Float32,
        validation: Bool, validate_after: Int, validate_after_unit: Int,
        continue_last_backup: Bool,
        sample_after: Int, sample_after_unit: Int, sample_skip_first: Int,
        non_ema_sampling: Bool, backup_after: Int, backup_after_unit: Int,
        rolling_backup: Bool, rolling_backup_count: Int, backup_before_save: Bool,
        save_every_unit: Int, save_skip_first: Int, var save_filename_prefix: String,
        masked_training: Bool, unmasked_probability: Float32, unmasked_weight: Float32,
        normalize_masked_area_loss: Bool, masked_prior_preservation_weight: Float32,
        custom_conditioning_image: Bool,
        max_steps: Int, save_every: Int, sample_every: Int,
    ):
        self.name = name^
        self.checkpoint = checkpoint^
        self.vae = vae^
        self.validation_prompts_file = validation_prompts_file^
        self.base_model_name = base_model_name^
        self.workspace_dir = workspace_dir^
        self.cache_dir = cache_dir^
        self.output_model_destination = output_model_destination^
        self.output_model_format = output_model_format^
        self.concept_file_name = concept_file_name^
        self.sample_definition_file_name = sample_definition_file_name^
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
        self.training_method = training_method
        self.batch_size = batch_size
        self.epochs = epochs
        self.stop_training_after = stop_training_after
        self.stop_training_after_unit = stop_training_after_unit
        self.resolution = resolution^
        self.frames = frames^
        self.seed = seed
        self.compile_model = compile_model
        self.dataloader_threads = dataloader_threads
        self.latent_caching = latent_caching
        self.clear_cache_before_training = clear_cache_before_training
        self.only_cache = only_cache
        self.tensorboard = tensorboard
        self.tensorboard_always_on = tensorboard_always_on
        self.samples_to_tensorboard = samples_to_tensorboard
        self.train_dtype = train_dtype
        self.fallback_train_dtype = fallback_train_dtype
        self.weight_dtype = weight_dtype
        self.output_dtype = output_dtype
        self.lora_weight_dtype = lora_weight_dtype
        self.embedding_weight_dtype = embedding_weight_dtype
        self.unet_weight_dtype = unet_weight_dtype
        self.prior_weight_dtype = prior_weight_dtype
        self.transformer_weight_dtype = transformer_weight_dtype
        self.text_encoder_weight_dtype = text_encoder_weight_dtype
        self.text_encoder_2_weight_dtype = text_encoder_2_weight_dtype
        self.text_encoder_3_weight_dtype = text_encoder_3_weight_dtype
        self.text_encoder_4_weight_dtype = text_encoder_4_weight_dtype
        self.vae_weight_dtype = vae_weight_dtype
        self.unet_model_name = unet_model_name^
        self.prior_model_name = prior_model_name^
        self.transformer_model_name = transformer_model_name^
        self.text_encoder_model_name = text_encoder_model_name^
        self.text_encoder_2_model_name = text_encoder_2_model_name^
        self.text_encoder_3_model_name = text_encoder_3_model_name^
        self.text_encoder_4_model_name = text_encoder_4_model_name^
        self.vae_model_name = vae_model_name^
        self.unet_train = unet_train
        self.prior_train = prior_train
        self.transformer_train = transformer_train
        self.text_encoder_train = text_encoder_train
        self.text_encoder_2_train = text_encoder_2_train
        self.text_encoder_3_train = text_encoder_3_train
        self.text_encoder_4_train = text_encoder_4_train
        self.vae_train = vae_train
        self.lr = lr
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.timestep_shift = timestep_shift
        self.eps = eps
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer
        self.optimizer_eps2 = optimizer_eps2
        self.optimizer_clip_threshold = optimizer_clip_threshold
        self.optimizer_decay_rate = optimizer_decay_rate
        self.optimizer_relative_step = optimizer_relative_step
        self.optimizer_scale_parameter = optimizer_scale_parameter
        self.optimizer_warmup_init = optimizer_warmup_init
        self.optimizer_fused = optimizer_fused
        self.optimizer_fused_back_pass = optimizer_fused_back_pass
        self.optimizer_stochastic_rounding = optimizer_stochastic_rounding
        self.layer_filter = layer_filter^
        self.layer_filter_preset = layer_filter_preset^
        self.layer_filter_regex = layer_filter_regex
        self.quantization_layer_filter = quantization_layer_filter^
        self.quantization_layer_filter_preset = quantization_layer_filter_preset^
        self.quantization_layer_filter_regex = quantization_layer_filter_regex
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
        self.ema_mode = ema_mode
        self.ema_decay = ema_decay
        self.ema_update_step_interval = ema_update_step_interval
        self.gradient_checkpointing = gradient_checkpointing
        self.enable_async_offloading = enable_async_offloading
        self.enable_activation_offloading = enable_activation_offloading
        self.layer_offload_fraction = layer_offload_fraction
        self.adapter_algo = adapter_algo
        self.train_modality = train_modality
        self.lora_target_preset = lora_target_preset
        self.dataset_cache_dir = dataset_cache_dir^
        self.require_cached_video_latents = require_cached_video_latents
        self.require_cached_text_embeddings = require_cached_text_embeddings
        self.require_cached_audio_latents = require_cached_audio_latents
        self.hot_loop_device_only = hot_loop_device_only
        self.video_loss_weight = video_loss_weight
        self.audio_loss_weight = audio_loss_weight
        self.validation = validation
        self.validate_after = validate_after
        self.validate_after_unit = validate_after_unit
        self.continue_last_backup = continue_last_backup
        self.sample_after = sample_after
        self.sample_after_unit = sample_after_unit
        self.sample_skip_first = sample_skip_first
        self.non_ema_sampling = non_ema_sampling
        self.backup_after = backup_after
        self.backup_after_unit = backup_after_unit
        self.rolling_backup = rolling_backup
        self.rolling_backup_count = rolling_backup_count
        self.backup_before_save = backup_before_save
        self.save_every_unit = save_every_unit
        self.save_skip_first = save_skip_first
        self.save_filename_prefix = save_filename_prefix^
        self.masked_training = masked_training
        self.unmasked_probability = unmasked_probability
        self.unmasked_weight = unmasked_weight
        self.normalize_masked_area_loss = normalize_masked_area_loss
        self.masked_prior_preservation_weight = masked_prior_preservation_weight
        self.custom_conditioning_image = custom_conditioning_image
        self.max_steps = max_steps
        self.save_every = save_every
        self.sample_every = sample_every

    def is_lora_training(self) -> Bool:
        return self.training_method == TRAINING_METHOD_LORA

    def is_full_finetune_training(self) -> Bool:
        return self.training_method == TRAINING_METHOD_FINE_TUNE

    def validate_training_method_config(self) raises:
        if (
            self.training_method != TRAINING_METHOD_LORA
            and self.training_method != TRAINING_METHOD_FINE_TUNE
        ):
            raise Error(
                String("TrainConfig: invalid training_method tag ")
                + String(self.training_method)
            )

    def _valid_dtype_tag(self, v: Int) -> Bool:
        return (
            v == TRAIN_DTYPE_NONE
            or v == TRAIN_DTYPE_FLOAT_8
            or v == TRAIN_DTYPE_FLOAT_16
            or v == TRAIN_DTYPE_FLOAT_32
            or v == TRAIN_DTYPE_BFLOAT_16
            or v == TRAIN_DTYPE_TFLOAT_32
            or v == TRAIN_DTYPE_INT_8
            or v == TRAIN_DTYPE_NFLOAT_4
            or v == TRAIN_DTYPE_FLOAT_W8A8
            or v == TRAIN_DTYPE_INT_W8A8
            or v == TRAIN_DTYPE_GGUF
            or v == TRAIN_DTYPE_GGUF_A8_FLOAT
            or v == TRAIN_DTYPE_GGUF_A8_INT
        )

    def _valid_time_unit_tag(self, v: Int) -> Bool:
        return (
            v == TRAIN_TIME_UNIT_EPOCH
            or v == TRAIN_TIME_UNIT_STEP
            or v == TRAIN_TIME_UNIT_SECOND
            or v == TRAIN_TIME_UNIT_MINUTE
            or v == TRAIN_TIME_UNIT_HOUR
            or v == TRAIN_TIME_UNIT_NEVER
            or v == TRAIN_TIME_UNIT_ALWAYS
        )

    def _valid_optimizer_tag(self, v: Int) -> Bool:
        return (
            v == TRAIN_OPTIMIZER_ADAMW
            or v == TRAIN_OPTIMIZER_ADAM
            or v == TRAIN_OPTIMIZER_ADAFACTOR
            or v == TRAIN_OPTIMIZER_CAME
            or v == TRAIN_OPTIMIZER_LION
            or v == TRAIN_OPTIMIZER_PRODIGY
            or v == TRAIN_OPTIMIZER_SGD
            or v == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW
        )

    def optimizer_is_adamw(self) -> Bool:
        return self.optimizer == TRAIN_OPTIMIZER_ADAMW

    def validate_lora_adamw_loop_policy(self, trainer_name: String) raises:
        if not self.is_lora_training():
            raise Error(
                trainer_name
                + String(" currently implements training_method=LORA only; parsed training_method tag ")
                + String(self.training_method)
            )
        if not self.optimizer_is_adamw():
            raise Error(
                trainer_name
                + String(" currently implements optimizer=ADAMW only; parsed optimizer tag ")
                + String(self.optimizer)
            )

    def validate_onetrainer_policy_config(self) raises:
        if not self._valid_optimizer_tag(self.optimizer):
            raise Error(String("TrainConfig: invalid optimizer tag ") + String(self.optimizer))
        if not self._valid_time_unit_tag(self.stop_training_after_unit):
            raise Error("TrainConfig: invalid stop_training_after_unit tag")
        if not self._valid_time_unit_tag(self.validate_after_unit):
            raise Error("TrainConfig: invalid validate_after_unit tag")
        if not self._valid_time_unit_tag(self.sample_after_unit):
            raise Error("TrainConfig: invalid sample_after_unit tag")
        if not self._valid_time_unit_tag(self.backup_after_unit):
            raise Error("TrainConfig: invalid backup_after_unit tag")
        if not self._valid_time_unit_tag(self.save_every_unit):
            raise Error("TrainConfig: invalid save_every_unit tag")
        if self.batch_size <= 0:
            raise Error("TrainConfig: batch_size must be > 0")
        if self.dataloader_threads < 0:
            raise Error("TrainConfig: dataloader_threads must be >= 0")
        if self.backup_after < 0 or self.save_every < 0 or self.sample_after < 0:
            raise Error("TrainConfig: sample/save/backup cadence values must be >= 0")
        if self.rolling_backup_count < 0:
            raise Error("TrainConfig: rolling_backup_count must be >= 0")
        if self.ema_mode != EMA_MODE_OFF and self.ema_mode != EMA_MODE_GPU and self.ema_mode != EMA_MODE_CPU:
            raise Error(String("TrainConfig: invalid ema mode tag ") + String(self.ema_mode))
        if self.ema_update_step_interval < 0:
            raise Error("TrainConfig: ema_update_step_interval must be >= 0")
        if self.unmasked_probability < Float32(0.0) or self.unmasked_probability > Float32(1.0):
            raise Error("TrainConfig: unmasked_probability must be within 0..1")
        if (
            not self._valid_dtype_tag(self.train_dtype)
            or not self._valid_dtype_tag(self.fallback_train_dtype)
            or not self._valid_dtype_tag(self.weight_dtype)
            or not self._valid_dtype_tag(self.output_dtype)
            or not self._valid_dtype_tag(self.lora_weight_dtype)
            or not self._valid_dtype_tag(self.embedding_weight_dtype)
            or not self._valid_dtype_tag(self.unet_weight_dtype)
            or not self._valid_dtype_tag(self.prior_weight_dtype)
            or not self._valid_dtype_tag(self.transformer_weight_dtype)
            or not self._valid_dtype_tag(self.text_encoder_weight_dtype)
            or not self._valid_dtype_tag(self.text_encoder_2_weight_dtype)
            or not self._valid_dtype_tag(self.text_encoder_3_weight_dtype)
            or not self._valid_dtype_tag(self.text_encoder_4_weight_dtype)
            or not self._valid_dtype_tag(self.vae_weight_dtype)
        ):
            raise Error("TrainConfig: invalid dtype tag")

    def gradient_checkpointing_enabled(self) -> Bool:
        return (
            self.gradient_checkpointing == GRADIENT_CHECKPOINTING_ON
            or self.gradient_checkpointing == GRADIENT_CHECKPOINTING_CPU_OFFLOADED
        )

    def gradient_checkpointing_offload(self) -> Bool:
        return self.gradient_checkpointing == GRADIENT_CHECKPOINTING_CPU_OFFLOADED

    def activation_offload_enabled(self) -> Bool:
        return self.gradient_checkpointing_offload() and self.enable_activation_offloading

    def layer_offload_enabled(self) -> Bool:
        return (
            self.gradient_checkpointing_offload()
            and self.layer_offload_fraction > Float64(0.0)
        )

    def async_offload_enabled_for_cuda(self, is_cuda: Bool) -> Bool:
        return is_cuda and self.enable_async_offloading

    def validate_offload_checkpoint_config(self) raises:
        if (
            self.gradient_checkpointing != GRADIENT_CHECKPOINTING_OFF
            and self.gradient_checkpointing != GRADIENT_CHECKPOINTING_ON
            and self.gradient_checkpointing != GRADIENT_CHECKPOINTING_CPU_OFFLOADED
        ):
            raise Error(
                String("TrainConfig: invalid gradient_checkpointing tag ")
                + String(self.gradient_checkpointing)
            )
        if self.layer_offload_fraction < Float64(0.0) or self.layer_offload_fraction > Float64(1.0):
            raise Error("TrainConfig: layer_offload_fraction must be within 0..1")
