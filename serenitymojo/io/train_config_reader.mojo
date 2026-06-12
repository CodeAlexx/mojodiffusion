# train_config_reader.mojo — read a model config JSON into a TrainConfig.
#
# Reads serenitymojo/configs/<model>.json (the SINGLE SOURCE OF TRUTH for arch +
# recipe + paths — binding user rule 2026-05-31). Unlike the earlier version,
# the MODEL DIMS are read FROM THE FILE, not caller-supplied: a trainer/sampler
# passes only the path. Adding a new model = adding a JSON file, never code.
#
# ── Why a hand-rolled parser ─────────────────────────────────────────────────
# io/json_header.mojo is a flat-safetensors-header parser; model configs are
# general JSON (floats, sci-notation, bools, strings, a nested "optimizer"). We
# REUSE json_header's cursor primitives (_Cursor peek/advance/skip_ws/expect,
# escape-aware _parse_string, balanced-brace _skip_value) and add a general
# scalar reader (number/bool/null/string). Mojo 1.0.0b1 has no runtime
# reflection, so — mirroring OneTrainer BaseConfig.from_dict — we start from
# TrainConfig.default() and overwrite each field as its JSON key is parsed,
# skipping unknown keys. Missing keys keep their default.
#
# ── Keys read (top level) ────────────────────────────────────────────────────
#   model_type(str)→name | checkpoint(str) | vae(str) | validation_prompts_file(str)
#   inner_dim→d_model | in_channels | joint_attention_dim | out_channels
#   num_double | num_single | num_heads→n_heads | head_dim | mlp_hidden
#   timestep_dim | rope_theta | learning_rate→lr | lora_rank | lora_alpha
#   timestep_shift | max_grad_norm | max_steps | save_every | sample_every
#   optimizer.{eps,weight_decay,beta1,beta2}
#   training_method/train_method/method
#   train_modality/ltx2_mode | lora_target_preset | dataset_cache_dir/cache_dir
#   require_cached_video_latents | require_cached_text_embeddings
#   require_cached_audio_latents | hot_loop_device_only
#   video_loss_weight | audio_loss_weight
#   continue_last_backup | gradient_checkpointing | enable_async_offloading
#   enable_activation_offloading | layer_offload_fraction
#
# Mojo 1.0.0b1: `def` not `fn`; no Python.

from std.collections import List
from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value
from serenitymojo.training.train_config import (
    TrainConfig,
    TRAIN_MODALITY_VIDEO, TRAIN_MODALITY_AV, TRAIN_MODALITY_AUDIO,
    LORA_TARGET_LEGACY_VIDEO_ATTN1, LORA_TARGET_LTX2_T2V,
    LORA_TARGET_LTX2_V2V, LORA_TARGET_LTX2_AUDIO,
    LORA_TARGET_LTX2_AUDIO_REF_ONLY_IC, LORA_TARGET_LTX2_FULL,
    TRAINING_METHOD_LORA, TRAINING_METHOD_FINE_TUNE,
    GRADIENT_CHECKPOINTING_OFF, GRADIENT_CHECKPOINTING_ON,
    GRADIENT_CHECKPOINTING_CPU_OFFLOADED,
    TRAIN_DTYPE_NONE, TRAIN_DTYPE_FLOAT_8, TRAIN_DTYPE_FLOAT_16,
    TRAIN_DTYPE_FLOAT_32, TRAIN_DTYPE_BFLOAT_16, TRAIN_DTYPE_TFLOAT_32,
    TRAIN_DTYPE_INT_8, TRAIN_DTYPE_NFLOAT_4, TRAIN_DTYPE_FLOAT_W8A8,
    TRAIN_DTYPE_INT_W8A8, TRAIN_DTYPE_GGUF, TRAIN_DTYPE_GGUF_A8_FLOAT,
    TRAIN_DTYPE_GGUF_A8_INT,
    TRAIN_OPTIMIZER_ADAMW, TRAIN_OPTIMIZER_ADAFACTOR,
    TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW, TRAIN_OPTIMIZER_ADAMW_8BIT,
    TRAIN_TIME_UNIT_EPOCH, TRAIN_TIME_UNIT_STEP, TRAIN_TIME_UNIT_SECOND,
    TRAIN_TIME_UNIT_MINUTE, TRAIN_TIME_UNIT_HOUR, TRAIN_TIME_UNIT_NEVER,
    TRAIN_TIME_UNIT_ALWAYS,
    EMA_MODE_OFF, EMA_MODE_GPU, EMA_MODE_CPU,
)


# ── General JSON number parser (signed, fractional, scientific) ──────────────
# json_header._parse_int only does non-negative integers. We parse a full JSON
# number to Float64 by hand (no atof dependency — not verified present). Config
# numbers are small-magnitude so Float64 accumulation is exact enough.
def _parse_number(mut cur: _Cursor) raises -> Float64:
    cur.skip_ws()
    var start = cur.pos
    var neg = False
    var c = cur.peek()
    if c == 0x2D:  # '-'
        neg = True
        cur.advance()
    elif c == 0x2B:  # '+'
        cur.advance()

    var mantissa = Float64(0.0)
    var any_digit = False
    while not cur.at_end():
        var ch = cur.peek()
        if ch >= 0x30 and ch <= 0x39:
            mantissa = mantissa * 10.0 + Float64(ch - 0x30)
            any_digit = True
            cur.advance()
        else:
            break
    var frac_digits = 0
    if cur.peek() == 0x2E:  # '.'
        cur.advance()
        while not cur.at_end():
            var ch = cur.peek()
            if ch >= 0x30 and ch <= 0x39:
                mantissa = mantissa * 10.0 + Float64(ch - 0x30)
                frac_digits += 1
                any_digit = True
                cur.advance()
            else:
                break
    if not any_digit:
        raise Error(
            String("JSON config: expected number at byte ") + String(start)
        )
    var exp = 0
    var exp_neg = False
    var ech = cur.peek()
    if ech == 0x65 or ech == 0x45:  # 'e' / 'E'
        cur.advance()
        var es = cur.peek()
        if es == 0x2D:
            exp_neg = True
            cur.advance()
        elif es == 0x2B:
            cur.advance()
        var have_exp = False
        while not cur.at_end():
            var ch2 = cur.peek()
            if ch2 >= 0x30 and ch2 <= 0x39:
                exp = exp * 10 + (ch2 - 0x30)
                have_exp = True
                cur.advance()
            else:
                break
        if not have_exp:
            raise Error(
                String("JSON config: malformed exponent at byte ") + String(cur.pos)
            )
        if exp_neg:
            exp = -exp
    var net = exp - frac_digits
    var value = mantissa
    if net > 0:
        for _ in range(net):
            value = value * 10.0
    elif net < 0:
        for _ in range(-net):
            value = value / 10.0
    if neg:
        value = -value
    return value


# Bool / null skip-or-read. Returns 1.0 for true, 0.0 for false/null.
def _read_literal(mut cur: _Cursor) raises -> Float64:
    cur.skip_ws()
    var c = cur.peek()
    if c == 0x74:  # 't' -> true
        for _ in range(4):
            cur.advance()
        return 1.0
    if c == 0x66:  # 'f' -> false
        for _ in range(5):
            cur.advance()
        return 0.0
    if c == 0x6E:  # 'n' -> null
        for _ in range(4):
            cur.advance()
        return 0.0
    raise Error(
        String("JSON config: expected true/false/null at byte ") + String(cur.pos)
    )


@fieldwise_init
struct _Scalar(Copyable, Movable):
    var is_string: Bool
    var s: String
    var num: Float64


def _read_scalar(mut cur: _Cursor) raises -> _Scalar:
    cur.skip_ws()
    var c = cur.peek()
    if c == 0x22:  # string
        return _Scalar(True, _parse_string(cur), 0.0)
    if c == 0x74 or c == 0x66 or c == 0x6E:  # true/false/null
        return _Scalar(False, String(""), _read_literal(cur))
    return _Scalar(False, String(""), _parse_number(cur))


def _read_string_required(mut cur: _Cursor, field: String) raises -> String:
    var sc = _read_scalar(cur)
    if not sc.is_string:
        raise Error(String("JSON config: ") + field + String(" must be a string"))
    return sc.s.copy()


def _read_bool(mut cur: _Cursor) raises -> Bool:
    return _read_scalar(cur).num != 0.0


def _dtype_int(s: String) raises -> Int:
    if s == "NONE":
        return TRAIN_DTYPE_NONE
    elif s == "FLOAT_8":
        return TRAIN_DTYPE_FLOAT_8
    elif s == "FLOAT_16":
        return TRAIN_DTYPE_FLOAT_16
    elif s == "FLOAT_32":
        return TRAIN_DTYPE_FLOAT_32
    elif s == "BFLOAT_16":
        return TRAIN_DTYPE_BFLOAT_16
    elif s == "TFLOAT_32":
        return TRAIN_DTYPE_TFLOAT_32
    elif s == "INT_8":
        return TRAIN_DTYPE_INT_8
    elif s == "NFLOAT_4":
        return TRAIN_DTYPE_NFLOAT_4
    elif s == "FLOAT_W8A8":
        return TRAIN_DTYPE_FLOAT_W8A8
    elif s == "INT_W8A8":
        return TRAIN_DTYPE_INT_W8A8
    elif s == "GGUF":
        return TRAIN_DTYPE_GGUF
    elif s == "GGUF_A8_FLOAT":
        return TRAIN_DTYPE_GGUF_A8_FLOAT
    elif s == "GGUF_A8_INT":
        return TRAIN_DTYPE_GGUF_A8_INT
    raise Error(String("JSON config: unknown DataType '") + s + String("'"))


def _optimizer_int(s: String) raises -> Int:
    # T1.C fail-loud contract: only the optimizers a Mojo trainer actually
    # implements parse (ADAMW default fused path; ADAFACTOR /
    # SCHEDULE_FREE_ADAMW / ADAMW_8BIT via training/levers.mojo
    # levers_optimizer_step). Every other recognized tag
    # (ADAM/CAME/LION/PRODIGY/SGD/...) is rejected AT CONFIG LOAD instead of
    # silently training AdamW.
    if s == "ADAMW" or s == "ADAMW_ADV":
        return TRAIN_OPTIMIZER_ADAMW
    elif s == "ADAMW_8BIT":
        # T2.A: bnb block-wise 8-bit AdamW (training/adamw8bit.mojo, gated by
        # training/tests/adamw8bit_parity.mojo vs bnb 0.49.2 dumps). Pre-T2.A
        # this tag silently aliased to plain ADAMW.
        return TRAIN_OPTIMIZER_ADAMW_8BIT
    elif s == "ADAFACTOR":
        return TRAIN_OPTIMIZER_ADAFACTOR
    elif s == "SCHEDULE_FREE_ADAMW":
        return TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW
    elif (
        s == "ADAM" or s == "ADAM_8BIT"
        or s == "CAME" or s == "CAME_8BIT"
        or s == "LION" or s == "LION_8BIT" or s == "LION_ADV"
        or s == "PRODIGY" or s == "PRODIGY_PLUS_SCHEDULE_FREE"
        or s == "PRODIGY_ADV"
        or s == "SGD" or s == "SGD_8BIT"
    ):
        raise Error(
            String("JSON config: optimizer '") + s
            + String("' is not implemented in the Mojo trainers; supported:")
            + String(" ADAMW, ADAMW_8BIT, ADAFACTOR, SCHEDULE_FREE_ADAMW")
        )
    raise Error(
        String("JSON config: unknown Optimizer '") + s
        + String("'; supported: ADAMW, ADAMW_8BIT, ADAFACTOR,")
        + String(" SCHEDULE_FREE_ADAMW")
    )


def _time_unit_int(s: String) raises -> Int:
    if s == "EPOCH" or s == "epoch":
        return TRAIN_TIME_UNIT_EPOCH
    elif s == "STEP" or s == "step":
        return TRAIN_TIME_UNIT_STEP
    elif s == "SECOND" or s == "second":
        return TRAIN_TIME_UNIT_SECOND
    elif s == "MINUTE" or s == "minute":
        return TRAIN_TIME_UNIT_MINUTE
    elif s == "HOUR" or s == "hour":
        return TRAIN_TIME_UNIT_HOUR
    elif s == "NEVER" or s == "never":
        return TRAIN_TIME_UNIT_NEVER
    elif s == "ALWAYS" or s == "always":
        return TRAIN_TIME_UNIT_ALWAYS
    raise Error(String("JSON config: unknown TimeUnit '") + s + String("'"))


def _ema_mode_int(s: String) raises -> Int:
    if s == "OFF" or s == "off":
        return EMA_MODE_OFF
    elif s == "GPU" or s == "gpu":
        return EMA_MODE_GPU
    elif s == "CPU" or s == "cpu":
        return EMA_MODE_CPU
    elif s == "EMA" or s == "ema":
        # serenity-trainer UI emits "EMA" (TrainerConfigModel ema_options is
        # OFF/EMA); the T1.B lora_ema.mojo shadows live on HOST mirrors.
        return EMA_MODE_CPU
    raise Error(String("JSON config: unknown EMAMode '") + s + String("'"))


# Parse a nested optimizer object, mutating the active optimizer policy on cfg.
def _parse_optimizer(mut cur: _Cursor, mut cfg: TrainConfig) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        var field = _parse_string(cur)
        cur.expect(0x3A)
        if field == "optimizer":
            cfg.optimizer = _optimizer_int(_read_string_required(cur, String("optimizer.optimizer")))
        elif field == "eps":
            cfg.eps = Float32(_read_scalar(cur).num)
        elif field == "eps2":
            cfg.optimizer_eps2 = Float32(_read_scalar(cur).num)
        elif field == "weight_decay":
            cfg.weight_decay = Float32(_read_scalar(cur).num)
        elif field == "beta1":
            cfg.beta1 = Float32(_read_scalar(cur).num)
        elif field == "beta2":
            cfg.beta2 = Float32(_read_scalar(cur).num)
        elif field == "clip_threshold":
            cfg.optimizer_clip_threshold = Float32(_read_scalar(cur).num)
        elif field == "decay_rate":
            cfg.optimizer_decay_rate = Float32(_read_scalar(cur).num)
        elif field == "relative_step":
            cfg.optimizer_relative_step = _read_bool(cur)
        elif field == "scale_parameter":
            cfg.optimizer_scale_parameter = _read_bool(cur)
        elif field == "warmup_init":
            cfg.optimizer_warmup_init = _read_bool(cur)
        elif field == "warmup_steps":
            cfg.optimizer_warmup_steps = Int(_read_scalar(cur).num)
        elif field == "fused":
            cfg.optimizer_fused = _read_bool(cur)
        elif field == "fused_back_pass":
            cfg.optimizer_fused_back_pass = _read_bool(cur)
        elif field == "stochastic_rounding":
            cfg.optimizer_stochastic_rounding = _read_bool(cur)
        else:
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x7D:
            cur.advance()
            break
        raise Error(
            String("JSON config: expected ',' or '}' in optimizer at byte ")
            + String(cur.pos)
        )


def _parse_optimizer_defaults(mut cur: _Cursor, mut cfg: TrainConfig) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        _ = _parse_string(cur)
        cur.expect(0x3A)
        cur.skip_ws()
        if cur.peek() == 0x7B:
            _parse_optimizer(cur, cfg)
        else:
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x7D:
            cur.advance()
            break
        raise Error(
            String("JSON config: expected ',' or '}' in optimizer_defaults at byte ")
            + String(cur.pos)
        )


def _set_part_model_name(mut cfg: TrainConfig, part: String, var value: String):
    if part == "unet":
        cfg.unet_model_name = value^
    elif part == "prior":
        cfg.prior_model_name = value^
    elif part == "transformer":
        cfg.transformer_model_name = value^
    elif part == "text_encoder":
        cfg.text_encoder_model_name = value^
    elif part == "text_encoder_2":
        cfg.text_encoder_2_model_name = value^
    elif part == "text_encoder_3":
        cfg.text_encoder_3_model_name = value^
    elif part == "text_encoder_4":
        cfg.text_encoder_4_model_name = value^
    elif part == "vae":
        cfg.vae_model_name = value^


def _set_part_train(mut cfg: TrainConfig, part: String, value: Bool):
    if part == "unet":
        cfg.unet_train = value
    elif part == "prior":
        cfg.prior_train = value
    elif part == "transformer":
        cfg.transformer_train = value
    elif part == "text_encoder":
        cfg.text_encoder_train = value
    elif part == "text_encoder_2":
        cfg.text_encoder_2_train = value
    elif part == "text_encoder_3":
        cfg.text_encoder_3_train = value
    elif part == "text_encoder_4":
        cfg.text_encoder_4_train = value
    elif part == "vae":
        cfg.vae_train = value


def _set_part_dtype(mut cfg: TrainConfig, part: String, value: Int):
    if part == "unet":
        cfg.unet_weight_dtype = value
    elif part == "prior":
        cfg.prior_weight_dtype = value
    elif part == "transformer":
        cfg.transformer_weight_dtype = value
    elif part == "text_encoder":
        cfg.text_encoder_weight_dtype = value
    elif part == "text_encoder_2":
        cfg.text_encoder_2_weight_dtype = value
    elif part == "text_encoder_3":
        cfg.text_encoder_3_weight_dtype = value
    elif part == "text_encoder_4":
        cfg.text_encoder_4_weight_dtype = value
    elif part == "vae":
        cfg.vae_weight_dtype = value


def _parse_model_part(mut cur: _Cursor, mut cfg: TrainConfig, part: String) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        var field = _parse_string(cur)
        cur.expect(0x3A)
        if field == "model_name":
            _set_part_model_name(cfg, part, _read_string_required(cur, part + String(".model_name")))
        elif field == "train":
            _set_part_train(cfg, part, _read_bool(cur))
        elif field == "weight_dtype":
            _set_part_dtype(cfg, part, _dtype_int(_read_string_required(cur, part + String(".weight_dtype"))))
        else:
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x7D:
            cur.advance()
            break
        raise Error(
            String("JSON config: expected ',' or '}' in model part at byte ")
            + String(cur.pos)
        )


def _parse_quantization(mut cur: _Cursor, mut cfg: TrainConfig) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        var field = _parse_string(cur)
        cur.expect(0x3A)
        if field == "layer_filter":
            cfg.quantization_layer_filter = _read_string_required(cur, String("quantization.layer_filter"))
        elif field == "layer_filter_preset":
            cfg.quantization_layer_filter_preset = _read_string_required(cur, String("quantization.layer_filter_preset"))
        elif field == "layer_filter_regex":
            cfg.quantization_layer_filter_regex = _read_bool(cur)
        else:
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x7D:
            cur.advance()
            break
        raise Error(
            String("JSON config: expected ',' or '}' in quantization at byte ")
            + String(cur.pos)
        )


# ── Wave 2 enum string -> comptime-int mappers (fail loud on unknown). ───────
# Each maps the EDv2 canonical config STRING to the comptime-Int encoding the
# Wave 2 builders chose (LR_* in lr_schedule.mojo, TSB_* in timestep_bias.mojo,
# TSD_* in schedule.mojo, adapter_algo full=1). An UNKNOWN string raises (the
# task's "fail loud, not silently default" rule) so a typo in a config never
# silently degrades to the default-off path.
#
# AGENT-DEFAULT (flagged for review): the accepted string spellings below.
def _lr_scheduler_int(s: String) raises -> Int:
    # LR_CONSTANT 0 / LR_LINEAR 1 / LR_COSINE 2 / LR_COSINE_RESTARTS 3
    # / LR_POLYNOMIAL 4 / LR_REX 5  (lr_schedule.mojo).
    if s == "constant" or s == "CONSTANT":
        return 0
    elif s == "linear" or s == "LINEAR":
        return 1
    elif s == "cosine" or s == "COSINE":
        return 2
    elif s == "cosine_with_restarts" or s == "COSINE_WITH_RESTARTS":
        return 3
    elif s == "polynomial" or s == "POLYNOMIAL":
        return 4
    elif s == "rex" or s == "REX":
        return 5
    raise Error(
        String("JSON config: unknown lr_scheduler '") + s
        + "' (expected constant|linear|cosine|cosine_with_restarts|polynomial|rex)"
    )


def _loss_fn_int(s: String) raises -> Int:
    # LOSS_FN_MSE 0 / LOSS_FN_HUBER 1 / LOSS_FN_SMOOTH_L1 2
    # (train_config.mojo; T1.A torch-semantics loss selector).
    if s == "mse" or s == "MSE":
        return 0
    elif s == "huber" or s == "HUBER":
        return 1
    elif s == "smooth_l1" or s == "SMOOTH_L1":
        return 2
    raise Error(
        String("JSON config: unknown loss_fn '") + s
        + "' (expected mse|huber|smooth_l1)"
    )


def _timestep_bias_int(s: String) raises -> Int:
    # TSB_NONE 0 / TSB_LATER 1 / TSB_EARLIER 2 / TSB_RANGE 3 (timestep_bias.mojo).
    if s == "none":
        return 0
    elif s == "later":
        return 1
    elif s == "earlier":
        return 2
    elif s == "range":
        return 3
    raise Error(
        String("JSON config: unknown timestep_bias_strategy '") + s
        + "' (expected none|later|earlier|range)"
    )


def _timestep_distribution_int(s: String) raises -> Int:
    # TSD_UNIFORM 0 / TSD_SIGMOID 1 / TSD_LOGIT_NORMAL 2 (schedule.mojo).
    # OneTrainer carries additional distribution tags; preserve them as policy
    # tags even where current loops still need a separate implementation gate.
    if s == "uniform" or s == "UNIFORM":
        return 0
    elif s == "sigmoid" or s == "SIGMOID":
        return 1
    elif s == "logit_normal" or s == "LOGIT_NORMAL":
        return 2
    elif s == "HEAVY_TAIL" or s == "heavy_tail":
        return 3
    elif s == "COS_MAP" or s == "cos_map":
        return 4
    elif s == "INVERTED_PARABOLA" or s == "inverted_parabola":
        return 5
    raise Error(
        String("JSON config: unknown timestep_distribution '") + s
        + "' (expected uniform|sigmoid|logit_normal|HEAVY_TAIL|COS_MAP|INVERTED_PARABOLA)"
    )


def _adapter_algo_int(s: String) raises -> Int:
    # 0 = plain LoRA, 1 = LyCORIS Full (full_adapter.mojo), 2 = LyCORIS LoHa
    # (loha_adapter.mojo), 3 = DoRA (dora_adapter.mojo), 4 = LyCORIS LoKr
    # (lokr_adapter.mojo), 5 = Diag-OFT (oft_adapter.mojo), 6 = BOFT
    # (boft_adapter.mojo) — all gated by their *_smoke.mojo; stack-dispatch is a
    # later integration wave (the trainer selector fails loud for 1..6).
    if s == "lora":
        return 0
    elif s == "full":
        return 1
    elif s == "loha":
        return 2
    elif s == "dora":
        return 3
    elif s == "lokr":
        return 4
    elif s == "oft":
        return 5
    elif s == "boft":
        return 6
    raise Error(
        String("JSON config: unknown algo '") + s + "' (expected lora|full|loha|dora|lokr|oft|boft)"
    )


def _training_method_int(s: String) raises -> Int:
    # OneTrainer TrainingMethod values plus common full-finetune shorthands.
    if s == "LORA" or s == "lora" or s == "LoRA":
        return TRAINING_METHOD_LORA
    elif (
        s == "FINE_TUNE"
        or s == "fine_tune"
        or s == "FineTune"
        or s == "finetune"
        or s == "FINETUNE"
        or s == "full"
        or s == "FULL"
        or s == "Full"
    ):
        return TRAINING_METHOD_FINE_TUNE
    raise Error(
        String("JSON config: unknown training_method '") + s
        + String("' (expected LORA|FINE_TUNE|fine_tune|full|finetune)")
    )


def _gradient_checkpointing_int(s: String) raises -> Int:
    # OneTrainer GradientCheckpointingMethod values.
    if s == "OFF" or s == "off":
        return GRADIENT_CHECKPOINTING_OFF
    elif s == "ON" or s == "on":
        return GRADIENT_CHECKPOINTING_ON
    elif s == "CPU_OFFLOADED" or s == "cpu_offloaded":
        return GRADIENT_CHECKPOINTING_CPU_OFFLOADED
    raise Error(
        String("JSON config: unknown gradient_checkpointing '") + s
        + String("' (expected OFF|ON|CPU_OFFLOADED)")
    )


def _checked_train_modality(v: Int) raises -> Int:
    if v == TRAIN_MODALITY_VIDEO or v == TRAIN_MODALITY_AV or v == TRAIN_MODALITY_AUDIO:
        return v
    raise Error(
        String("JSON config: unknown train_modality ")
        + String(v) + String(" (expected 0=video, 1=av, 2=audio)")
    )


def _train_modality_int(s: String) raises -> Int:
    if s == "video" or s == "v":
        return TRAIN_MODALITY_VIDEO
    elif s == "av" or s == "audio_video" or s == "va":
        return TRAIN_MODALITY_AV
    elif s == "audio" or s == "a":
        return TRAIN_MODALITY_AUDIO
    raise Error(
        String("JSON config: unknown train_modality '") + s
        + String("' (expected video|av|audio)")
    )


def _checked_lora_target_preset(v: Int) raises -> Int:
    if (
        v == LORA_TARGET_LEGACY_VIDEO_ATTN1
        or v == LORA_TARGET_LTX2_T2V
        or v == LORA_TARGET_LTX2_V2V
        or v == LORA_TARGET_LTX2_AUDIO
        or v == LORA_TARGET_LTX2_AUDIO_REF_ONLY_IC
        or v == LORA_TARGET_LTX2_FULL
    ):
        return v
    raise Error(
        String("JSON config: unknown lora_target_preset ")
        + String(v)
        + String(" (expected 0=legacy_video_attn1, 1=t2v, 2=v2v, 3=audio, 4=audio_ref_only_ic, 5=full)")
    )


def _lora_target_preset_int(s: String) raises -> Int:
    if s == "legacy_video_attn1" or s == "legacy":
        return LORA_TARGET_LEGACY_VIDEO_ATTN1
    elif s == "t2v":
        return LORA_TARGET_LTX2_T2V
    elif s == "v2v":
        return LORA_TARGET_LTX2_V2V
    elif s == "audio":
        return LORA_TARGET_LTX2_AUDIO
    elif s == "audio_ref_only_ic":
        return LORA_TARGET_LTX2_AUDIO_REF_ONLY_IC
    elif s == "full":
        return LORA_TARGET_LTX2_FULL
    raise Error(
        String("JSON config: unknown lora_target_preset '") + s
        + String("' (expected legacy_video_attn1|t2v|v2v|audio|audio_ref_only_ic|full)")
    )


# ── Read a whole file's bytes via raw syscalls (pure-Mojo, no Python). ───────
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, BytePtr, O_RDONLY
from std.memory import alloc


def _read_file_bytes(path: String) raises -> List[UInt8]:
    """Read the entire file at `path` into a host byte list."""
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("train_config_reader: cannot open ") + path)
    var out = List[UInt8]()
    comptime CHUNK = 65536
    var buf = alloc[UInt8](CHUNK)
    var offset = 0
    while True:
        var n = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, offset)
        if n < 0:
            buf.free()
            _ = sys_close(fd)
            raise Error("train_config_reader: read error")
        if n == 0:
            break
        for i in range(n):
            out.append(buf[i])
        offset += n
        if n < CHUNK:
            break
    buf.free()
    _ = sys_close(fd)
    return out^


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: read a model config JSON into a fully-populated TrainConfig.
# Arch + recipe + paths ALL come from the file; the caller passes only the path.
# Missing keys keep TrainConfig.default() values.
# ─────────────────────────────────────────────────────────────────────────────
def read_model_config(json_path: String) raises -> TrainConfig:
    """Parse `json_path` and return a complete TrainConfig (arch+recipe+paths)."""
    var bytes = _read_file_bytes(json_path)
    var cur = _Cursor(bytes^)
    var cfg = TrainConfig.default()

    cur.expect(0x7B)  # top-level '{'
    cur.skip_ws()
    if cur.peek() == 0x7D:  # empty object
        cur.advance()
        return cfg^

    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)  # ':'

        # strings
        if key == "model_type":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.name = sc.s
        elif key == "base_model_name":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.base_model_name = sc.s
        elif key == "checkpoint":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.checkpoint = sc.s
        elif key == "vae":
            cur.skip_ws()
            if cur.peek() == 0x7B:
                _parse_model_part(cur, cfg, String("vae"))
            else:
                var sc = _read_scalar(cur)
                if sc.is_string:
                    cfg.vae = sc.s
        elif key == "validation_prompts_file" or key == "sample_prompts_file":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.validation_prompts_file = sc.s
        elif key == "sample_definition_file_name":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.sample_definition_file_name = sc.s
                if cfg.validation_prompts_file == String(""):
                    cfg.validation_prompts_file = sc.s
        elif key == "workspace_dir":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.workspace_dir = sc.s
        elif key == "cache_dir":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.cache_dir = sc.s
                cfg.dataset_cache_dir = sc.s
        elif key == "output_model_destination":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.output_model_destination = sc.s
        elif key == "output_model_format":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.output_model_format = sc.s
        elif key == "concept_file_name":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.concept_file_name = sc.s
        elif key == "resolution":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.resolution = sc.s
        elif key == "frames":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.frames = sc.s
        elif key == "quantized_resident":
            # T2.B quantized-resident base weights. Fail loud on unknown tags
            # at config-load time (OneTrainer fail-fast discipline).
            var sc = _read_scalar(cur)
            if sc.is_string:
                if (
                    sc.s != String("") and sc.s != String("OFF")
                    and sc.s != String("fp8_e4m3")
                ):
                    raise Error(
                        String("train config: unsupported quantized_resident '")
                        + sc.s + "' (supported: OFF, fp8_e4m3)"
                    )
                cfg.quantized_resident = sc.s
        elif key == "controlnet_layers":
            # T2.E ControlNet training (default-off 0). Fail loud on negatives.
            var n = Int(_read_scalar(cur).num)
            if n < 0:
                raise Error(
                    String("train config: controlnet_layers must be >= 0, got ")
                    + String(n)
                )
            cfg.controlnet_layers = n
        elif key == "controlnet_scale":
            cfg.controlnet_scale = _read_scalar(cur).num
        elif key == "controlnet_checkpoint":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.controlnet_checkpoint = sc.s
        # arch (ints)
        elif key == "inner_dim":
            cfg.d_model = Int(_read_scalar(cur).num)
        elif key == "in_channels":
            cfg.in_channels = Int(_read_scalar(cur).num)
        elif key == "joint_attention_dim":
            cfg.joint_attention_dim = Int(_read_scalar(cur).num)
        elif key == "out_channels":
            cfg.out_channels = Int(_read_scalar(cur).num)
        elif key == "num_double":
            cfg.num_double = Int(_read_scalar(cur).num)
        elif key == "num_single":
            cfg.num_single = Int(_read_scalar(cur).num)
        elif key == "num_heads":
            cfg.n_heads = Int(_read_scalar(cur).num)
        elif key == "head_dim":
            cfg.head_dim = Int(_read_scalar(cur).num)
        elif key == "mlp_hidden":
            cfg.mlp_hidden = Int(_read_scalar(cur).num)
        elif key == "timestep_dim":
            cfg.timestep_dim = Int(_read_scalar(cur).num)
        elif key == "rope_theta":
            cfg.rope_theta = _read_scalar(cur).num
        # recipe
        elif key == "learning_rate":
            cfg.lr = Float32(_read_scalar(cur).num)
        elif key == "batch_size":
            cfg.batch_size = Int(_read_scalar(cur).num)
        elif key == "epochs":
            cfg.epochs = Int(_read_scalar(cur).num)
        elif key == "stop_training_after":
            cfg.stop_training_after = Int(_read_scalar(cur).num)
            if cfg.stop_training_after_unit == TRAIN_TIME_UNIT_STEP:
                cfg.max_steps = cfg.stop_training_after
        elif key == "stop_training_after_unit":
            cfg.stop_training_after_unit = _time_unit_int(
                _read_string_required(cur, String("stop_training_after_unit"))
            )
            if cfg.stop_training_after > 0 and cfg.stop_training_after_unit == TRAIN_TIME_UNIT_STEP:
                cfg.max_steps = cfg.stop_training_after
        elif key == "seed":
            cfg.seed = UInt64(Int(_read_scalar(cur).num))
        elif key == "compile":
            cfg.compile_model = _read_bool(cur)
        elif key == "dataloader_threads":
            cfg.dataloader_threads = Int(_read_scalar(cur).num)
        elif key == "latent_caching":
            cfg.latent_caching = _read_bool(cur)
        elif key == "clear_cache_before_training":
            cfg.clear_cache_before_training = _read_bool(cur)
        elif key == "only_cache":
            cfg.only_cache = _read_bool(cur)
        elif key == "tensorboard":
            cfg.tensorboard = _read_bool(cur)
        elif key == "tensorboard_always_on":
            cfg.tensorboard_always_on = _read_bool(cur)
        elif key == "samples_to_tensorboard":
            cfg.samples_to_tensorboard = _read_bool(cur)
        elif key == "lora_rank":
            cfg.lora_rank = Int(_read_scalar(cur).num)
        elif key == "lora_alpha":
            cfg.lora_alpha = Float32(_read_scalar(cur).num)
        elif key == "timestep_shift":
            cfg.timestep_shift = Float32(_read_scalar(cur).num)
        elif key == "max_grad_norm" or key == "clip_grad_norm":
            cfg.max_grad_norm = Float32(_read_scalar(cur).num)
        elif key == "max_steps" or key == "max_train_steps":
            cfg.max_steps = Int(_read_scalar(cur).num)
        elif key == "save_every" or key == "save_every_n_steps" or key == "save_after":
            cfg.save_every = Int(_read_scalar(cur).num)
        elif key == "sample_every" or key == "sample_every_n_steps":
            cfg.sample_every = Int(_read_scalar(cur).num)
            cfg.sample_after = cfg.sample_every
            cfg.sample_after_unit = TRAIN_TIME_UNIT_STEP
        elif key == "optimizer":
            _parse_optimizer(cur, cfg)
        elif key == "optimizer_defaults":
            _parse_optimizer_defaults(cur, cfg)
        elif key == "training_method" or key == "train_method" or key == "method":
            var sc = _read_scalar(cur)
            if not sc.is_string:
                raise Error("JSON config: training_method/train_method/method must be a string")
            cfg.training_method = _training_method_int(sc.s)
        elif key == "train_dtype":
            cfg.train_dtype = _dtype_int(_read_string_required(cur, String("train_dtype")))
        elif key == "fallback_train_dtype":
            cfg.fallback_train_dtype = _dtype_int(_read_string_required(cur, String("fallback_train_dtype")))
        elif key == "weight_dtype":
            cfg.weight_dtype = _dtype_int(_read_string_required(cur, String("weight_dtype")))
        elif key == "output_dtype":
            cfg.output_dtype = _dtype_int(_read_string_required(cur, String("output_dtype")))
        elif key == "lora_weight_dtype":
            cfg.lora_weight_dtype = _dtype_int(_read_string_required(cur, String("lora_weight_dtype")))
        elif key == "embedding_weight_dtype":
            cfg.embedding_weight_dtype = _dtype_int(_read_string_required(cur, String("embedding_weight_dtype")))
        elif key == "unet" or key == "prior" or key == "transformer" or key == "text_encoder" or key == "text_encoder_2" or key == "text_encoder_3" or key == "text_encoder_4":
            _parse_model_part(cur, cfg, key)
        elif key == "unet_weight_dtype":
            cfg.unet_weight_dtype = _dtype_int(_read_string_required(cur, String("unet_weight_dtype")))
        elif key == "prior_weight_dtype":
            cfg.prior_weight_dtype = _dtype_int(_read_string_required(cur, String("prior_weight_dtype")))
        elif key == "transformer_weight_dtype":
            cfg.transformer_weight_dtype = _dtype_int(_read_string_required(cur, String("transformer_weight_dtype")))
        elif key == "text_encoder_weight_dtype":
            cfg.text_encoder_weight_dtype = _dtype_int(_read_string_required(cur, String("text_encoder_weight_dtype")))
        elif key == "text_encoder_2_weight_dtype":
            cfg.text_encoder_2_weight_dtype = _dtype_int(_read_string_required(cur, String("text_encoder_2_weight_dtype")))
        elif key == "text_encoder_3_weight_dtype":
            cfg.text_encoder_3_weight_dtype = _dtype_int(_read_string_required(cur, String("text_encoder_3_weight_dtype")))
        elif key == "text_encoder_4_weight_dtype":
            cfg.text_encoder_4_weight_dtype = _dtype_int(_read_string_required(cur, String("text_encoder_4_weight_dtype")))
        elif key == "vae_weight_dtype":
            cfg.vae_weight_dtype = _dtype_int(_read_string_required(cur, String("vae_weight_dtype")))
        elif key == "train_unet":
            cfg.unet_train = _read_bool(cur)
        elif key == "train_prior":
            cfg.prior_train = _read_bool(cur)
        elif key == "train_transformer":
            cfg.transformer_train = _read_bool(cur)
        elif key == "train_text_encoder":
            cfg.text_encoder_train = _read_bool(cur)
        elif key == "train_text_encoder_2":
            cfg.text_encoder_2_train = _read_bool(cur)
        elif key == "quantization":
            _parse_quantization(cur, cfg)
        elif key == "layer_filter":
            cfg.layer_filter = _read_string_required(cur, String("layer_filter"))
        elif key == "layer_filter_preset":
            cfg.layer_filter_preset = _read_string_required(cur, String("layer_filter_preset"))
        elif key == "layer_filter_regex":
            cfg.layer_filter_regex = _read_bool(cur)
        # ── Wave 2A: lr scheduler ──
        elif key == "lr_scheduler" or key == "learning_rate_scheduler":
            var sc = _read_scalar(cur)
            if not sc.is_string:
                raise Error("JSON config: lr_scheduler/learning_rate_scheduler must be a string")
            cfg.lr_scheduler = _lr_scheduler_int(sc.s)
        elif key == "lr_warmup_steps" or key == "learning_rate_warmup_steps":
            cfg.lr_warmup_steps = Int(_read_scalar(cur).num)
        elif key == "optimizer_warmup_steps":
            # T1.C: schedule-free AdamW internal warmup (also accepted as
            # optimizer.warmup_steps in the nested optimizer object).
            cfg.optimizer_warmup_steps = Int(_read_scalar(cur).num)
        elif key == "lr_min_factor" or key == "learning_rate_min_factor":
            cfg.lr_min_factor = Float32(_read_scalar(cur).num)
        elif key == "lr_cycles" or key == "learning_rate_cycles":
            cfg.lr_cycles = Float32(_read_scalar(cur).num)
        # ── Wave 2A: loss weighting ──
        elif key == "min_snr_gamma":
            cfg.min_snr_gamma = Float32(_read_scalar(cur).num)
        elif key == "debiased":
            cfg.debiased = _read_scalar(cur).num != 0.0
        # ── Wave 2A: combined loss strengths ──
        elif key == "loss_mse_strength" or key == "mse_strength":
            cfg.loss_mse_strength = Float32(_read_scalar(cur).num)
        elif key == "loss_mae_strength" or key == "mae_strength":
            cfg.loss_mae_strength = Float32(_read_scalar(cur).num)
        elif key == "loss_huber_strength" or key == "huber_strength":
            cfg.loss_huber_strength = Float32(_read_scalar(cur).num)
        # ── T1.A: torch-semantics loss fn + flow min-SNR-γ (levers.mojo) ──
        elif key == "loss_fn":
            var sc = _read_scalar(cur)
            if not sc.is_string:
                raise Error("JSON config: loss_fn must be a string")
            cfg.loss_fn = _loss_fn_int(sc.s)
        elif key == "huber_delta":
            cfg.huber_delta = Float32(_read_scalar(cur).num)
        elif key == "smooth_l1_beta":
            cfg.smooth_l1_beta = Float32(_read_scalar(cur).num)
        elif key == "min_snr_gamma_flow":
            cfg.min_snr_gamma_flow = Float32(_read_scalar(cur).num)
        # ── Wave 2A: timestep bias ──
        elif key == "timestep_bias_strategy":
            var sc = _read_scalar(cur)
            if not sc.is_string:
                raise Error("JSON config: timestep_bias_strategy must be a string")
            cfg.timestep_bias_strategy = _timestep_bias_int(sc.s)
        elif key == "timestep_bias_multiplier":
            cfg.timestep_bias_multiplier = Float32(_read_scalar(cur).num)
        elif key == "timestep_bias_range_min":
            cfg.timestep_bias_range_min = Float32(_read_scalar(cur).num)
        elif key == "timestep_bias_range_max":
            cfg.timestep_bias_range_max = Float32(_read_scalar(cur).num)
        # ── Wave 2A: timestep distribution ──
        elif key == "timestep_distribution":
            var sc = _read_scalar(cur)
            if not sc.is_string:
                raise Error("JSON config: timestep_distribution must be a string")
            cfg.timestep_distribution = _timestep_distribution_int(sc.s)
        elif key == "timestep_noising_weight" or key == "noising_weight":
            cfg.timestep_noising_weight = Float32(_read_scalar(cur).num)
        elif key == "timestep_noising_bias" or key == "noising_bias":
            cfg.timestep_noising_bias = Float32(_read_scalar(cur).num)
        # ── Wave 2B: caption dropout ──
        elif key == "caption_dropout_prob":
            cfg.caption_dropout_prob = Float32(_read_scalar(cur).num)
        # ── Wave 2B: noise modifiers ──
        elif key == "offset_noise_weight":
            cfg.offset_noise_weight = Float32(_read_scalar(cur).num)
        elif key == "offset_noise_prob":
            cfg.offset_noise_prob = Float32(_read_scalar(cur).num)
        elif key == "input_perturbation" or key == "perturbation_noise_weight":
            cfg.input_perturbation = Float32(_read_scalar(cur).num)
        elif key == "multires_iterations":
            cfg.multires_iterations = Int(_read_scalar(cur).num)
        elif key == "multires_discount":
            cfg.multires_discount = Float32(_read_scalar(cur).num)
        # ── Wave 2B: gradient accumulation ──
        elif key == "grad_accum_steps" or key == "gradient_accumulation_steps":
            cfg.grad_accum_steps = Int(_read_scalar(cur).num)
        # ── Wave 2B: EMA ──
        elif key == "ema":
            cfg.ema_mode = _ema_mode_int(_read_string_required(cur, String("ema")))
            cfg.ema_enabled = cfg.ema_mode != EMA_MODE_OFF
        elif key == "ema_enabled":
            cfg.ema_enabled = _read_scalar(cur).num != 0.0
        elif key == "ema_inv_gamma":
            cfg.ema_inv_gamma = Float32(_read_scalar(cur).num)
        elif key == "ema_power":
            cfg.ema_power = Float32(_read_scalar(cur).num)
        elif key == "ema_update_after_step":
            cfg.ema_update_after_step = Int(_read_scalar(cur).num)
        elif key == "ema_min_decay":
            cfg.ema_min_decay = Float32(_read_scalar(cur).num)
        elif key == "ema_max_decay":
            cfg.ema_max_decay = Float32(_read_scalar(cur).num)
        elif key == "ema_decay":
            cfg.ema_decay = Float32(_read_scalar(cur).num)
            cfg.ema_max_decay = cfg.ema_decay
        elif key == "ema_update_step_interval":
            cfg.ema_update_step_interval = Int(_read_scalar(cur).num)
        # ── OneTrainer offload/checkpoint policy ──
        elif key == "gradient_checkpointing":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.gradient_checkpointing = _gradient_checkpointing_int(sc.s)
            elif sc.num != 0.0:
                # Mirrors OneTrainer migration of legacy bool True -> ON.
                cfg.gradient_checkpointing = GRADIENT_CHECKPOINTING_ON
            else:
                cfg.gradient_checkpointing = GRADIENT_CHECKPOINTING_OFF
        elif key == "enable_async_offloading":
            cfg.enable_async_offloading = _read_scalar(cur).num != 0.0
        elif key == "enable_activation_offloading":
            cfg.enable_activation_offloading = _read_scalar(cur).num != 0.0
        elif key == "layer_offload_fraction":
            cfg.layer_offload_fraction = _read_scalar(cur).num
        # ── Wave 2B: adapter algo selector ──
        elif key == "algo" or key == "adapter_algo":
            var sc = _read_scalar(cur)
            if not sc.is_string:
                raise Error("JSON config: algo must be a string")
            cfg.adapter_algo = _adapter_algo_int(sc.s)
        # ── T2.G LoKr knobs (SimpleTuner lycoris_config parity; algo=4 only) ──
        elif key == "lokr_factor":
            cfg.lokr_factor = Int(_read_scalar(cur).num)
        elif key == "lokr_factor_attn":
            cfg.lokr_factor_attn = Int(_read_scalar(cur).num)
        elif key == "lokr_factor_ff":
            cfg.lokr_factor_ff = Int(_read_scalar(cur).num)
        elif key == "lokr_factor_single":
            cfg.lokr_factor_single = Int(_read_scalar(cur).num)
        elif key == "lokr_decompose_both" or key == "decompose_both":
            cfg.lokr_decompose_both = _read_scalar(cur).num != 0.0
        elif key == "lokr_full_matrix" or key == "full_matrix":
            cfg.lokr_full_matrix = _read_scalar(cur).num != 0.0
        elif key == "lokr_targets":
            var sc = _read_scalar(cur)
            if sc.is_string:
                if sc.s == "attn":
                    cfg.lokr_targets = 1
                elif sc.s == "attn+ff":
                    cfg.lokr_targets = 2
                elif sc.s == "all":
                    cfg.lokr_targets = 3
                else:
                    raise Error(
                        String("JSON config: unknown lokr_targets '") + sc.s
                        + "' (expected attn|attn+ff|all)"
                    )
            else:
                var n = Int(sc.num)
                if n < 1 or n > 3:
                    raise Error("JSON config: lokr_targets must be 1..3")
                cfg.lokr_targets = n
        elif key == "init_lokr_norm":
            var v = _read_scalar(cur).num
            if v < 0.0:
                raise Error("JSON config: init_lokr_norm must be >= 0")
            cfg.init_lokr_norm = v
        # ── cached-input / AV trainer contract ──
        elif key == "train_modality" or key == "ltx2_mode" or key == "modality":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.train_modality = _train_modality_int(sc.s)
            else:
                cfg.train_modality = _checked_train_modality(Int(sc.num))
        elif key == "lora_target_preset" or key == "ltx2_lora_target_preset":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.lora_target_preset = _lora_target_preset_int(sc.s)
            else:
                cfg.lora_target_preset = _checked_lora_target_preset(Int(sc.num))
        elif key == "dataset_cache_dir" or key == "train_cache_dir":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.dataset_cache_dir = sc.s
        elif key == "require_cached_video_latents" or key == "cache_video_latents":
            cfg.require_cached_video_latents = _read_scalar(cur).num != 0.0
        elif key == "require_cached_text_embeddings" or key == "cache_text_embeddings":
            cfg.require_cached_text_embeddings = _read_scalar(cur).num != 0.0
        elif key == "require_cached_audio_latents" or key == "cache_audio_latents":
            cfg.require_cached_audio_latents = _read_scalar(cur).num != 0.0
        elif key == "hot_loop_device_only" or key == "device_hot_loop":
            cfg.hot_loop_device_only = _read_scalar(cur).num != 0.0
        elif key == "video_loss_weight":
            cfg.video_loss_weight = Float32(_read_scalar(cur).num)
        elif key == "audio_loss_weight":
            cfg.audio_loss_weight = Float32(_read_scalar(cur).num)
        elif key == "validation":
            cfg.validation = _read_bool(cur)
        elif key == "validate_after":
            cfg.validate_after = Int(_read_scalar(cur).num)
        elif key == "validate_after_unit":
            cfg.validate_after_unit = _time_unit_int(_read_string_required(cur, String("validate_after_unit")))
        elif key == "continue_last_backup":
            cfg.continue_last_backup = _read_bool(cur)
        elif key == "sample_after":
            cfg.sample_after = Int(_read_scalar(cur).num)
            if cfg.sample_after_unit == TRAIN_TIME_UNIT_STEP:
                cfg.sample_every = cfg.sample_after
        elif key == "sample_after_unit":
            cfg.sample_after_unit = _time_unit_int(_read_string_required(cur, String("sample_after_unit")))
            if cfg.sample_after > 0 and cfg.sample_after_unit == TRAIN_TIME_UNIT_STEP:
                cfg.sample_every = cfg.sample_after
        elif key == "sample_skip_first":
            cfg.sample_skip_first = Int(_read_scalar(cur).num)
        elif key == "non_ema_sampling":
            cfg.non_ema_sampling = _read_bool(cur)
        elif key == "backup_after":
            cfg.backup_after = Int(_read_scalar(cur).num)
        elif key == "backup_after_unit":
            cfg.backup_after_unit = _time_unit_int(_read_string_required(cur, String("backup_after_unit")))
        elif key == "rolling_backup":
            cfg.rolling_backup = _read_bool(cur)
        elif key == "rolling_backup_count":
            cfg.rolling_backup_count = Int(_read_scalar(cur).num)
        elif key == "backup_before_save":
            cfg.backup_before_save = _read_bool(cur)
        elif key == "save_every_unit" or key == "save_after_unit":
            cfg.save_every_unit = _time_unit_int(_read_string_required(cur, String("save_every_unit")))
        elif key == "save_skip_first":
            cfg.save_skip_first = Int(_read_scalar(cur).num)
        elif key == "save_filename_prefix":
            cfg.save_filename_prefix = _read_string_required(cur, String("save_filename_prefix"))
        elif key == "masked_training":
            cfg.masked_training = _read_bool(cur)
        elif key == "unmasked_probability":
            cfg.unmasked_probability = Float32(_read_scalar(cur).num)
        elif key == "unmasked_weight":
            cfg.unmasked_weight = Float32(_read_scalar(cur).num)
        elif key == "normalize_masked_area_loss":
            cfg.normalize_masked_area_loss = _read_bool(cur)
        elif key == "masked_prior_preservation_weight":
            cfg.masked_prior_preservation_weight = Float32(_read_scalar(cur).num)
        elif key == "custom_conditioning_image":
            cfg.custom_conditioning_image = _read_bool(cur)
        else:
            _skip_value(cur)  # skip unknown top-level keys

        cur.skip_ws()
        var c = cur.peek()
        if c == 0x2C:  # ','
            cur.advance()
            continue
        if c == 0x7D:  # '}'
            cur.advance()
            break
        raise Error(
            String("JSON config: expected ',' or '}' at top level at byte ")
            + String(cur.pos)
        )

    cfg.validate_training_method_config()
    cfg.validate_offload_checkpoint_config()
    cfg.validate_onetrainer_policy_config()
    return cfg^
