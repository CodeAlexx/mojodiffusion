# train_config.mojo — runtime training scalars (the model-AGNOSTIC recipe).
# Port subset of Serenity modules/util/config/TrainConfig.py — only the scalars
# the shared pipeline (optimizer / loss / grad / driver) consumes. Per-model dims
# and shapes are comptime params on the model spec, NOT here (Mojo attention dims
# must be comptime — see PORT_MAP §4).
#
# Pure host scalars. No Tensor, no GPU. The optimizer-state / activation dtype is
# always BF16 (port policy); this struct carries no dtype field by design.

from std.builtin.dtype import DType


# TimestepDistribution enum kinds (Serenity modules/util/enum/TimestepDistribution.py
# is a string enum; we assign stable integer codes in declaration order so the
# Mojo port can branch on them in ModelSetupNoiseMixin._get_timestep_discrete).
comptime TSDIST_UNIFORM           = 0
comptime TSDIST_SIGMOID           = 1
comptime TSDIST_LOGIT_NORMAL      = 2
comptime TSDIST_HEAVY_TAIL        = 3
comptime TSDIST_COS_MAP           = 4
comptime TSDIST_INVERTED_PARABOLA = 5


@fieldwise_init
struct TrainConfig(Copyable, Movable):
    # optimizer (AdamW defaults mirror Serenity/torch)
    var learning_rate: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32
    var stochastic_rounding: Bool      # bf16 SR on param update (Serenity default for bf16)

    # loop / accumulation
    var epochs: Int
    var batch_size: Int
    var gradient_accumulation_steps: Int

    # grad clipping (clip_grad_norm is Optional in reference trainer; <=0 here means "off")
    var clip_grad_norm: Float32

    # LR schedule (resolved separately into LrSchedule; these are the inputs)
    var lr_scheduler_kind: Int
    var warmup_steps: Float64
    var lr_num_cycles: Float64
    var lr_min_factor: Float64

    # loss weighting (LossWeight enum kind) + params
    var loss_weight_kind: Int
    var min_snr_gamma: Float32

    # LoRA / LyCORIS. Values align with serenitymojo.training.train_config:
    # 0=lora, 2=loha, 3=dora, 4=lokr, 5=oft, 6=boft, 7=locon.
    # BOFT is parsed as an explicit error by TrainConfigReader; the value is
    # documented here only so stale numeric configs remain unambiguous.
    var lora_rank: Int
    var lora_alpha: Float32
    var adapter_algo: Int

    # rng
    var seed: UInt32

    # timestep / noising schedule (ModelSetupNoiseMixin._get_timestep_discrete).
    # Serenity TrainConfig.py defaults (lines 1020-1025):
    #   min_noising_strength=0.0, max_noising_strength=1.0,
    #   timestep_distribution=UNIFORM(0), noising_weight=0.0,
    #   noising_bias=0.0, timestep_shift=1.0.
    var timestep_distribution: Int    # TimestepDistribution enum kind (0=UNIFORM,1=SIGMOID,2=LOGIT_NORMAL,3=HEAVY_TAIL,4=COS_MAP,5=INVERTED_PARABOLA)
    var min_noising_strength: Float32
    var max_noising_strength: Float32
    var noising_weight: Float32
    var noising_bias: Float32
    var timestep_shift: Float32
    # dynamic_timestep_shifting (Serenity TrainConfig.py:1026, default False).
    # When True, BaseZImageSetup.predict (:109,116) uses the per-image μ from
    # calculate_timestep_shift(H, W) as the schedule shift instead of the static
    # config.timestep_shift.
    var dynamic_timestep_shifting: Bool

    # guidance scale (config.transformer.guidance_scale; BaseFlux2Setup.py:133 /
    # Flux2Sampler.py:113). Only consumed by guidance-distilled variants
    # (Klein 9B guidance_embeds=True). Serenity training default guidance_scale = 1.0
    # (TrainConfig.py:289; the four #flux2 presets do NOT override it). Ignored when
    # guidance_embeds=False. NB: 4.0 is the diffusers *inference* pipeline default, not
    # Serenity's training default.
    var guidance_scale: Float32

    @staticmethod
    def adamw_lora_defaults() -> TrainConfig:
        """A sane LoRA/AdamW default set (overridden per run). Mirrors common
        Serenity LoRA presets: lr 1e-4, betas (0.9, 0.999), eps 1e-8, wd 0.01."""
        return TrainConfig(
            learning_rate=Float32(1e-4),
            beta1=Float32(0.9),
            beta2=Float32(0.999),
            eps=Float32(1e-8),
            weight_decay=Float32(0.01),
            stochastic_rounding=True,
            epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            clip_grad_norm=Float32(1.0),
            lr_scheduler_kind=0,        # CONSTANT
            warmup_steps=Float64(0.0),
            lr_num_cycles=Float64(1.0),
            lr_min_factor=Float64(0.0),
            loss_weight_kind=0,         # CONSTANT
            min_snr_gamma=Float32(5.0),
            lora_rank=16,
            lora_alpha=Float32(16.0),
            adapter_algo=0,
            seed=UInt32(0),
            # noising-schedule defaults mirror Serenity TrainConfig.py:1020-1025
            timestep_distribution=TSDIST_UNIFORM,
            min_noising_strength=Float32(0.0),
            max_noising_strength=Float32(1.0),
            noising_weight=Float32(0.0),
            noising_bias=Float32(0.0),
            timestep_shift=Float32(1.0),
            dynamic_timestep_shifting=False,   # Serenity TrainConfig.py:1026
            guidance_scale=Float32(1.0),       # Serenity training default (TrainConfig.py:289; #flux2 presets do not override)
        )
