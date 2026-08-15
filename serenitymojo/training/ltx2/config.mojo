# config.mojo -- LTX-2.3 AV trainer config and CLI contract.
#
# Runtime contract: Mojo parses the launch shape and carries concrete training
# policy. Python/TOML references are development oracles only.
#
# LEVERS (feature #2): `--config <json>` loads a TrainConfig-shaped lever set via
# read_model_config (loss_fn/huber_delta/smooth_l1_beta/min_snr_gamma_flow,
# lr_scheduler/lr_warmup_steps, optimizer). Absent -> TrainConfig.default() =
# every lever off (byte-identical to the pre-levers behavior). ltx2-specific argv
# still WINS for its own fields: after the arg loop we copy the ltx2-effective
# learning_rate/max_steps INTO cfg.levers (.lr/.max_steps) so the LR scheduler
# base+horizon track the ltx2 knobs, and constant+warmup0 returns base exactly.

from std.collections import List
from serenitymojo.training.train_config import (
    TrainConfig, LOSS_FN_MSE, LOSS_FN_HUBER, LOSS_FN_SMOOTH_L1,
    TRAIN_OPTIMIZER_ADAMW,
)
from serenitymojo.io.train_config_reader import read_model_config


comptime MODE_VIDEO = 0
comptime MODE_AV = 1
comptime MODE_AUDIO = 2

comptime VERSION_20 = 20
comptime VERSION_23 = 23

comptime PRESET_T2V = 0
comptime PRESET_V2V = 1
comptime PRESET_AUDIO = 2
comptime PRESET_AUDIO_REF_ONLY_IC = 3
comptime PRESET_FULL = 4

comptime SIGMA_SHIFTED_LOGIT_NORMAL = 0
comptime SIGMA_UNIFORM = 1

comptime SHIFT_LEGACY = 0
comptime SHIFT_STRETCHED = 1


def _substr_bytes(s: String, start: Int, end: Int) -> String:
    var out = String("")
    var b = s.as_bytes()
    for i in range(start, end):
        out += chr(Int(b[i]))
    return out


def _strip_arg_value(arg: String, key: String) -> String:
    var prefix = key + String("=")
    if arg.startswith(prefix):
        return _substr_bytes(arg, prefix.byte_length(), arg.byte_length())
    return String("")


def _parse_nonnegative_int(s: String) raises -> Int:
    if s.byte_length() == 0:
        raise Error("expected integer, got empty string")
    var value = 0
    var b = s.as_bytes()
    for i in range(s.byte_length()):
        var c = b[i]
        if c < UInt8(0x30) or c > UInt8(0x39):
            raise Error(String("expected integer, got ") + s)
        value = value * 10 + Int(c - UInt8(0x30))
    return value


def _parse_bool(s: String) raises -> Bool:
    var v = s
    if v == "1" or v == "true" or v == "yes" or v == "on":
        return True
    if v == "0" or v == "false" or v == "no" or v == "off":
        return False
    raise Error(String("expected bool, got ") + s)


def _parse_float32(s: String) raises -> Float32:
    if s.byte_length() == 0:
        raise Error("expected float, got empty string")
    var b = s.as_bytes()
    var i = 0
    var sign = Float64(1.0)
    if b[0] == UInt8(0x2D):
        sign = Float64(-1.0)
        i = 1
    elif b[0] == UInt8(0x2B):
        i = 1

    var value = Float64(0.0)
    var saw_digit = False
    while i < s.byte_length() and b[i] >= UInt8(0x30) and b[i] <= UInt8(0x39):
        value = value * Float64(10.0) + Float64(Int(b[i] - UInt8(0x30)))
        saw_digit = True
        i += 1
    if i < s.byte_length() and b[i] == UInt8(0x2E):
        i += 1
        var place = Float64(0.1)
        while i < s.byte_length() and b[i] >= UInt8(0x30) and b[i] <= UInt8(0x39):
            value += Float64(Int(b[i] - UInt8(0x30))) * place
            place *= Float64(0.1)
            saw_digit = True
            i += 1
    if not saw_digit:
        raise Error(String("expected float, got ") + s)

    var exp10 = 0
    if i < s.byte_length() and (b[i] == UInt8(0x65) or b[i] == UInt8(0x45)):
        i += 1
        var exp_sign = 1
        if i < s.byte_length() and b[i] == UInt8(0x2D):
            exp_sign = -1
            i += 1
        elif i < s.byte_length() and b[i] == UInt8(0x2B):
            i += 1
        if i >= s.byte_length():
            raise Error(String("expected exponent digits, got ") + s)
        while i < s.byte_length():
            if b[i] < UInt8(0x30) or b[i] > UInt8(0x39):
                raise Error(String("expected float, got ") + s)
            exp10 = exp10 * 10 + Int(b[i] - UInt8(0x30))
            i += 1
        exp10 *= exp_sign
    if i != s.byte_length():
        raise Error(String("expected float, got ") + s)

    var factor = Float64(1.0)
    if exp10 > 0:
        for _ in range(exp10):
            factor *= Float64(10.0)
    elif exp10 < 0:
        for _ in range(-exp10):
            factor *= Float64(0.1)
    return Float32(sign * value * factor)


def mode_from_string(s: String) raises -> Int:
    if s == "video" or s == "v":
        return MODE_VIDEO
    if s == "av" or s == "va":
        return MODE_AV
    if s == "audio" or s == "a":
        return MODE_AUDIO
    raise Error(String("invalid LTX2 mode: ") + s)


def mode_name(mode: Int) -> String:
    if mode == MODE_VIDEO:
        return String("video")
    if mode == MODE_AUDIO:
        return String("audio")
    return String("av")


def version_from_string(s: String) raises -> Int:
    if s == "2.0" or s == "20":
        return VERSION_20
    if s == "2.3" or s == "23":
        return VERSION_23
    raise Error(String("invalid LTX version: ") + s)


def version_name(version: Int) -> String:
    if version == VERSION_20:
        return String("2.0")
    return String("2.3")


def preset_from_string(s: String) raises -> Int:
    if s == "t2v":
        return PRESET_T2V
    if s == "v2v":
        return PRESET_V2V
    if s == "audio":
        return PRESET_AUDIO
    if s == "audio_ref_only_ic":
        return PRESET_AUDIO_REF_ONLY_IC
    if s == "full":
        return PRESET_FULL
    raise Error(String("invalid LoRA target preset: ") + s)


def preset_name(preset: Int) -> String:
    if preset == PRESET_T2V:
        return String("t2v")
    if preset == PRESET_V2V:
        return String("v2v")
    if preset == PRESET_AUDIO:
        return String("audio")
    if preset == PRESET_AUDIO_REF_ONLY_IC:
        return String("audio_ref_only_ic")
    return String("full")


def sigma_sampling_from_string(s: String) raises -> Int:
    if s == "shifted_logit_normal" or s == "sigma":
        return SIGMA_SHIFTED_LOGIT_NORMAL
    if s == "uniform":
        return SIGMA_UNIFORM
    raise Error(String("invalid timestep_sampling: ") + s)


def shift_mode_from_string(s: String) raises -> Int:
    if s == "legacy" or s == "classic" or s == "old":
        return SHIFT_LEGACY
    if s == "stretched" or s == "v2" or s == "upstream":
        return SHIFT_STRETCHED
    raise Error(String("invalid shifted_logit_mode: ") + s)


struct LTX2TrainerConfig(Copyable, Movable):
    var ltx2_checkpoint: String
    var dataset_cache_dir: String
    var output_dir: String
    var resume_from: String
    var validation_prompts_cache: String
    var sample_latents_cache: String
    var ltx_mode: Int
    var ltx_version: Int
    var lora_target_preset: Int
    var lora_rank: Int
    var lora_alpha: Float32
    var learning_rate: Float32
    var weight_decay: Float32
    var batch_size: Int
    var gradient_accumulation_steps: Int
    var max_steps: Int
    var save_every: Int
    var sample_every: Int
    var seed: UInt64
    var video_loss_weight: Float32
    var audio_loss_weight: Float32
    var independent_audio_timestep: Bool
    var timestep_sampling: Int
    var shifted_logit_mode: Int
    var shifted_logit_shift: Float32
    var logit_std: Float32
    var shifted_logit_eps: Float32
    var shifted_logit_uniform_prob: Float32
    var min_timestep: Float32
    var max_timestep: Float32
    var audio_only_sequence_resolution: Int
    var fail_on_unready: Bool
    # feature #2: config-JSON lever set (loss/LR-schedule/optimizer). Default =
    # TrainConfig.default() = all levers off (C13 byte-identical behavior).
    var levers: TrainConfig
    # P2: val_loss (--val_cache_dir/--validate_every; 0/empty = off) + in-training
    # sampling side-process (--gemma_safetensors/--sample_prompt for the render cmd).
    var val_cache_dir: String
    var validate_every: Int
    var gemma_safetensors: String
    var sample_prompt: String

    @staticmethod
    def default() -> LTX2TrainerConfig:
        return LTX2TrainerConfig(
            String(""),
            String(""),
            String("output/ltx2_av_lora"),
            String(""),
            String(""),
            String(""),
            MODE_AV,
            VERSION_23,
            PRESET_T2V,
            32,
            Float32(32.0),
            Float32(1.0e-4),
            Float32(0.01),
            1,
            1,
            3000,
            500,
            250,
            UInt64(42),
            Float32(1.0),
            Float32(1.0),
            False,
            SIGMA_SHIFTED_LOGIT_NORMAL,
            SHIFT_STRETCHED,
            Float32(-1.0),
            Float32(1.0),
            Float32(1.0e-3),
            Float32(0.1),
            Float32(0.0),
            Float32(1000.0),
            64,
            False,
            TrainConfig.default(),
            String(""),                                            # val_cache_dir
            0,                                                     # validate_every
            String(""),                                            # gemma_safetensors
            String("a photorealistic red fox in autumn leaves"),   # sample_prompt
        )

    def __init__(
        out self,
        var ltx2_checkpoint: String,
        var dataset_cache_dir: String,
        var output_dir: String,
        var resume_from: String,
        var validation_prompts_cache: String,
        var sample_latents_cache: String,
        ltx_mode: Int,
        ltx_version: Int,
        lora_target_preset: Int,
        lora_rank: Int,
        lora_alpha: Float32,
        learning_rate: Float32,
        weight_decay: Float32,
        batch_size: Int,
        gradient_accumulation_steps: Int,
        max_steps: Int,
        save_every: Int,
        sample_every: Int,
        seed: UInt64,
        video_loss_weight: Float32,
        audio_loss_weight: Float32,
        independent_audio_timestep: Bool,
        timestep_sampling: Int,
        shifted_logit_mode: Int,
        shifted_logit_shift: Float32,
        logit_std: Float32,
        shifted_logit_eps: Float32,
        shifted_logit_uniform_prob: Float32,
        min_timestep: Float32,
        max_timestep: Float32,
        audio_only_sequence_resolution: Int,
        fail_on_unready: Bool,
        var levers: TrainConfig,
        var val_cache_dir: String,
        validate_every: Int,
        var gemma_safetensors: String,
        var sample_prompt: String,
    ):
        self.ltx2_checkpoint = ltx2_checkpoint^
        self.dataset_cache_dir = dataset_cache_dir^
        self.output_dir = output_dir^
        self.resume_from = resume_from^
        self.validation_prompts_cache = validation_prompts_cache^
        self.sample_latents_cache = sample_latents_cache^
        self.ltx_mode = ltx_mode
        self.ltx_version = ltx_version
        self.lora_target_preset = lora_target_preset
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_steps = max_steps
        self.save_every = save_every
        self.sample_every = sample_every
        self.seed = seed
        self.video_loss_weight = video_loss_weight
        self.audio_loss_weight = audio_loss_weight
        self.independent_audio_timestep = independent_audio_timestep
        self.timestep_sampling = timestep_sampling
        self.shifted_logit_mode = shifted_logit_mode
        self.shifted_logit_shift = shifted_logit_shift
        self.logit_std = logit_std
        self.shifted_logit_eps = shifted_logit_eps
        self.shifted_logit_uniform_prob = shifted_logit_uniform_prob
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.audio_only_sequence_resolution = audio_only_sequence_resolution
        self.fail_on_unready = fail_on_unready
        self.levers = levers^
        self.val_cache_dir = val_cache_dir^
        self.validate_every = validate_every
        self.gemma_safetensors = gemma_safetensors^
        self.sample_prompt = sample_prompt^

    def validate(self) raises:
        if self.lora_rank <= 0:
            raise Error("LTX2TrainerConfig: lora_rank must be > 0")
        if self.lora_alpha <= Float32(0.0):
            raise Error("LTX2TrainerConfig: lora_alpha must be > 0")
        if self.learning_rate <= Float32(0.0):
            raise Error("LTX2TrainerConfig: learning_rate must be > 0")
        if self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise Error("LTX2TrainerConfig: batch and grad accumulation must be > 0")
        if self.max_steps <= 0:
            raise Error("LTX2TrainerConfig: max_steps must be > 0")
        if self.min_timestep < Float32(0.0) or self.max_timestep > Float32(1000.0):
            raise Error("LTX2TrainerConfig: min/max timestep must be within 0..1000")
        if self.max_timestep < self.min_timestep:
            raise Error("LTX2TrainerConfig: max_timestep must be >= min_timestep")
        if self.audio_only_sequence_resolution != 0 and self.audio_only_sequence_resolution < 32:
            raise Error("LTX2TrainerConfig: audio_only_sequence_resolution must be 0 or >= 32")
        self._validate_p6_av()

    # ── LTX2 P6 AV-arm parse-time validation (P6.0). Torchref ranges
    #    (ltx2_train_network.py:2197-2218) + the lead-mandated XOR fail-loud.
    #    Unconditional (matches torchref's always-run block); every torchref default
    #    passes, so a default/video-only config never raises (C13). Consumption is
    #    P6.2 — this only rejects out-of-range knobs at load time.
    def _validate_p6_av(self) raises:
        ref lv = self.levers
        var m = lv.audio_loss_balance_mode
        if m != "none" and m != "inv_freq" and m != "ema_mag" and m != "uncertainty":
            raise Error(String("LTX2 P6: audio_loss_balance_mode must be none|inv_freq|ema_mag|uncertainty; got ") + m)
        if not (lv.audio_loss_balance_beta > Float32(0.0) and lv.audio_loss_balance_beta <= Float32(1.0)):
            raise Error("LTX2 P6: audio_loss_balance_beta must be in (0, 1]")
        if lv.audio_loss_balance_eps <= Float32(0.0):
            raise Error("LTX2 P6: audio_loss_balance_eps must be > 0")
        if lv.audio_loss_balance_min < Float32(0.0):
            raise Error("LTX2 P6: audio_loss_balance_min must be >= 0")
        if lv.audio_loss_balance_max <= Float32(0.0):
            raise Error("LTX2 P6: audio_loss_balance_max must be > 0")
        if lv.audio_loss_balance_max < lv.audio_loss_balance_min:
            raise Error("LTX2 P6: audio_loss_balance_max must be >= audio_loss_balance_min")
        if m == "inv_freq":
            if not (lv.audio_loss_balance_ema_init > Float32(0.0) and lv.audio_loss_balance_ema_init <= Float32(1.0)):
                raise Error("LTX2 P6: audio_loss_balance_ema_init must be in (0, 1] for inv_freq")
        else:
            if lv.audio_loss_balance_ema_init <= Float32(0.0):
                raise Error("LTX2 P6: audio_loss_balance_ema_init must be > 0")
        if lv.audio_loss_balance_target_ratio < Float32(0.0):
            raise Error("LTX2 P6: audio_loss_balance_target_ratio must be >= 0")
        if not (lv.audio_loss_balance_ema_decay > Float32(0.0) and lv.audio_loss_balance_ema_decay < Float32(1.0)):
            raise Error("LTX2 P6: audio_loss_balance_ema_decay must be in (0, 1)")
        # XOR (lead-mandated fail-loud): the quota sampler (min_audio_batches_per_
        # accum > 0) and the probability sampler (audio_batch_probability set, i.e.
        # >= 0 with the -1.0 unset sentinel) are mutually exclusive.
        if lv.min_audio_batches_per_accum > 0 and lv.audio_batch_probability >= Float32(0.0):
            raise Error("LTX2 P6: min_audio_batches_per_accum and audio_batch_probability are mutually exclusive (set only one)")
        # bucket strategy, when set (non-empty sentinel), must be pad|truncate.
        var bs = lv.audio_bucket_strategy
        if bs != "" and bs != "pad" and bs != "truncate":
            raise Error(String("LTX2 P6: audio_bucket_strategy must be pad|truncate; got ") + bs)

    @staticmethod
    def from_args(args: List[String]) raises -> LTX2TrainerConfig:
        var cfg = LTX2TrainerConfig.default()
        var i = 1
        while i < len(args):
            var a = String(args[i])
            var value: String
            if a == "--ltx2_checkpoint" and i + 1 < len(args):
                i += 1
                cfg.ltx2_checkpoint = String(args[i])
            elif (value := _strip_arg_value(a, String("--ltx2_checkpoint"))) != "":
                cfg.ltx2_checkpoint = value
            elif a == "--dataset_cache_dir" and i + 1 < len(args):
                i += 1
                cfg.dataset_cache_dir = String(args[i])
            elif (value := _strip_arg_value(a, String("--dataset_cache_dir"))) != "":
                cfg.dataset_cache_dir = value
            elif a == "--output_dir" and i + 1 < len(args):
                i += 1
                cfg.output_dir = String(args[i])
            elif (value := _strip_arg_value(a, String("--output_dir"))) != "":
                cfg.output_dir = value
            elif a == "--resume" or a == "--resume_from":
                if i + 1 >= len(args):
                    raise Error(String(a) + " requires a value")
                i += 1
                cfg.resume_from = String(args[i])
            elif (value := _strip_arg_value(a, String("--resume"))) != "":
                cfg.resume_from = value
            elif (value := _strip_arg_value(a, String("--resume_from"))) != "":
                cfg.resume_from = value
            elif a == "--validation_prompts_cache" and i + 1 < len(args):
                i += 1
                cfg.validation_prompts_cache = String(args[i])
            elif (value := _strip_arg_value(a, String("--validation_prompts_cache"))) != "":
                cfg.validation_prompts_cache = value
            elif a == "--sample_latents_cache" and i + 1 < len(args):
                i += 1
                cfg.sample_latents_cache = String(args[i])
            elif (value := _strip_arg_value(a, String("--sample_latents_cache"))) != "":
                cfg.sample_latents_cache = value
            elif a == "--ltx2_mode" or a == "--ltx_mode":
                if i + 1 >= len(args):
                    raise Error(String(a) + " requires a value")
                i += 1
                cfg.ltx_mode = mode_from_string(String(args[i]))
            elif (value := _strip_arg_value(a, String("--ltx2_mode"))) != "":
                cfg.ltx_mode = mode_from_string(value)
            elif (value := _strip_arg_value(a, String("--ltx_mode"))) != "":
                cfg.ltx_mode = mode_from_string(value)
            elif a == "--ltx_version" and i + 1 < len(args):
                i += 1
                cfg.ltx_version = version_from_string(String(args[i]))
            elif (value := _strip_arg_value(a, String("--ltx_version"))) != "":
                cfg.ltx_version = version_from_string(value)
            elif a == "--lora_target_preset" and i + 1 < len(args):
                i += 1
                cfg.lora_target_preset = preset_from_string(String(args[i]))
            elif (value := _strip_arg_value(a, String("--lora_target_preset"))) != "":
                cfg.lora_target_preset = preset_from_string(value)
            elif a == "--network_dim" or a == "--lora_rank":
                if i + 1 >= len(args):
                    raise Error(String(a) + " requires a value")
                i += 1
                cfg.lora_rank = _parse_nonnegative_int(String(args[i]))
            elif (value := _strip_arg_value(a, String("--network_dim"))) != "":
                cfg.lora_rank = _parse_nonnegative_int(value)
            elif (value := _strip_arg_value(a, String("--lora_rank"))) != "":
                cfg.lora_rank = _parse_nonnegative_int(value)
            elif a == "--network_alpha" or a == "--lora_alpha":
                if i + 1 >= len(args):
                    raise Error(String(a) + " requires a value")
                i += 1
                cfg.lora_alpha = _parse_float32(String(args[i]))
            elif (value := _strip_arg_value(a, String("--network_alpha"))) != "":
                cfg.lora_alpha = _parse_float32(value)
            elif (value := _strip_arg_value(a, String("--lora_alpha"))) != "":
                cfg.lora_alpha = _parse_float32(value)
            elif a == "--learning_rate" or a == "--lr":
                if i + 1 >= len(args):
                    raise Error(String(a) + " requires a value")
                i += 1
                cfg.learning_rate = _parse_float32(String(args[i]))
            elif (value := _strip_arg_value(a, String("--learning_rate"))) != "":
                cfg.learning_rate = _parse_float32(value)
            elif (value := _strip_arg_value(a, String("--lr"))) != "":
                cfg.learning_rate = _parse_float32(value)
            elif a == "--max_train_steps" or a == "--max_steps" or a == "--steps":
                if i + 1 >= len(args):
                    raise Error(String(a) + " requires a value")
                i += 1
                cfg.max_steps = _parse_nonnegative_int(String(args[i]))
            elif (value := _strip_arg_value(a, String("--max_train_steps"))) != "":
                cfg.max_steps = _parse_nonnegative_int(value)
            elif (value := _strip_arg_value(a, String("--max_steps"))) != "":
                cfg.max_steps = _parse_nonnegative_int(value)
            elif (value := _strip_arg_value(a, String("--steps"))) != "":
                cfg.max_steps = _parse_nonnegative_int(value)
            elif a == "--gradient_accumulation_steps" or a == "--grad_accum" or a == "--accum":
                if i + 1 >= len(args):
                    raise Error(String(a) + " requires a value")
                i += 1
                cfg.gradient_accumulation_steps = _parse_nonnegative_int(String(args[i]))
            elif (value := _strip_arg_value(a, String("--gradient_accumulation_steps"))) != "":
                cfg.gradient_accumulation_steps = _parse_nonnegative_int(value)
            elif (value := _strip_arg_value(a, String("--grad_accum"))) != "":
                cfg.gradient_accumulation_steps = _parse_nonnegative_int(value)
            elif (value := _strip_arg_value(a, String("--accum"))) != "":
                cfg.gradient_accumulation_steps = _parse_nonnegative_int(value)
            elif a == "--batch_size" or a == "--bs":
                if i + 1 >= len(args):
                    raise Error(String(a) + " requires a value")
                i += 1
                cfg.batch_size = _parse_nonnegative_int(String(args[i]))
            elif (value := _strip_arg_value(a, String("--batch_size"))) != "":
                cfg.batch_size = _parse_nonnegative_int(value)
            elif (value := _strip_arg_value(a, String("--bs"))) != "":
                cfg.batch_size = _parse_nonnegative_int(value)
            elif a == "--save_every_n_steps" or a == "--save_every":
                if i + 1 >= len(args):
                    raise Error(String(a) + " requires a value")
                i += 1
                cfg.save_every = _parse_nonnegative_int(String(args[i]))
            elif (value := _strip_arg_value(a, String("--save_every_n_steps"))) != "":
                cfg.save_every = _parse_nonnegative_int(value)
            elif (value := _strip_arg_value(a, String("--save_every"))) != "":
                cfg.save_every = _parse_nonnegative_int(value)
            elif a == "--sample_every_n_steps" or a == "--sample_every":
                if i + 1 >= len(args):
                    raise Error(String(a) + " requires a value")
                i += 1
                cfg.sample_every = _parse_nonnegative_int(String(args[i]))
            elif (value := _strip_arg_value(a, String("--sample_every_n_steps"))) != "":
                cfg.sample_every = _parse_nonnegative_int(value)
            elif (value := _strip_arg_value(a, String("--sample_every"))) != "":
                cfg.sample_every = _parse_nonnegative_int(value)
            elif a == "--seed":
                if i + 1 >= len(args):
                    raise Error("--seed requires a value")
                i += 1
                cfg.seed = UInt64(_parse_nonnegative_int(String(args[i])))
            elif (value := _strip_arg_value(a, String("--seed"))) != "":
                cfg.seed = UInt64(_parse_nonnegative_int(value))
            elif a == "--video_loss_weight":
                if i + 1 >= len(args):
                    raise Error("--video_loss_weight requires a value")
                i += 1
                cfg.video_loss_weight = _parse_float32(String(args[i]))
            elif (value := _strip_arg_value(a, String("--video_loss_weight"))) != "":
                cfg.video_loss_weight = _parse_float32(value)
            elif a == "--audio_loss_weight":
                if i + 1 >= len(args):
                    raise Error("--audio_loss_weight requires a value")
                i += 1
                cfg.audio_loss_weight = _parse_float32(String(args[i]))
            elif (value := _strip_arg_value(a, String("--audio_loss_weight"))) != "":
                cfg.audio_loss_weight = _parse_float32(value)
            elif a == "--independent_audio_timestep":
                cfg.independent_audio_timestep = True
            elif (value := _strip_arg_value(a, String("--independent_audio_timestep"))) != "":
                cfg.independent_audio_timestep = _parse_bool(value)
            elif a == "--timestep_sampling" and i + 1 < len(args):
                i += 1
                cfg.timestep_sampling = sigma_sampling_from_string(String(args[i]))
            elif (value := _strip_arg_value(a, String("--timestep_sampling"))) != "":
                cfg.timestep_sampling = sigma_sampling_from_string(value)
            elif a == "--shifted_logit_mode" and i + 1 < len(args):
                i += 1
                cfg.shifted_logit_mode = shift_mode_from_string(String(args[i]))
            elif (value := _strip_arg_value(a, String("--shifted_logit_mode"))) != "":
                cfg.shifted_logit_mode = shift_mode_from_string(value)
            elif a == "--shifted_logit_shift" and i + 1 < len(args):
                i += 1
                cfg.shifted_logit_shift = _parse_float32(String(args[i]))
            elif (value := _strip_arg_value(a, String("--shifted_logit_shift"))) != "":
                cfg.shifted_logit_shift = _parse_float32(value)
            elif a == "--logit_std" and i + 1 < len(args):
                i += 1
                cfg.logit_std = _parse_float32(String(args[i]))
            elif (value := _strip_arg_value(a, String("--logit_std"))) != "":
                cfg.logit_std = _parse_float32(value)
            elif a == "--shifted_logit_uniform_prob" and i + 1 < len(args):
                i += 1
                cfg.shifted_logit_uniform_prob = _parse_float32(String(args[i]))
            elif (value := _strip_arg_value(a, String("--shifted_logit_uniform_prob"))) != "":
                cfg.shifted_logit_uniform_prob = _parse_float32(value)
            elif a == "--shifted_logit_eps" and i + 1 < len(args):
                i += 1
                cfg.shifted_logit_eps = _parse_float32(String(args[i]))
            elif (value := _strip_arg_value(a, String("--shifted_logit_eps"))) != "":
                cfg.shifted_logit_eps = _parse_float32(value)
            elif a == "--audio_only_sequence_resolution" and i + 1 < len(args):
                i += 1
                cfg.audio_only_sequence_resolution = _parse_nonnegative_int(String(args[i]))
            elif (value := _strip_arg_value(a, String("--audio_only_sequence_resolution"))) != "":
                cfg.audio_only_sequence_resolution = _parse_nonnegative_int(value)
            elif a == "--fail_on_unready":
                cfg.fail_on_unready = True
            elif (value := _strip_arg_value(a, String("--fail_on_unready"))) != "":
                cfg.fail_on_unready = _parse_bool(value)
            elif a == "--min_timestep" and i + 1 < len(args):
                i += 1
                cfg.min_timestep = _parse_float32(String(args[i]))
            elif (value := _strip_arg_value(a, String("--min_timestep"))) != "":
                cfg.min_timestep = _parse_float32(value)
            elif a == "--max_timestep" and i + 1 < len(args):
                i += 1
                cfg.max_timestep = _parse_float32(String(args[i]))
            elif (value := _strip_arg_value(a, String("--max_timestep"))) != "":
                cfg.max_timestep = _parse_float32(value)
            elif a == "--val_cache_dir" and i + 1 < len(args):
                i += 1
                cfg.val_cache_dir = String(args[i])
            elif (value := _strip_arg_value(a, String("--val_cache_dir"))) != "":
                cfg.val_cache_dir = value
            elif a == "--validate_every" and i + 1 < len(args):
                i += 1
                cfg.validate_every = _parse_nonnegative_int(String(args[i]))
            elif (value := _strip_arg_value(a, String("--validate_every"))) != "":
                cfg.validate_every = _parse_nonnegative_int(value)
            elif a == "--gemma_safetensors" and i + 1 < len(args):
                i += 1
                cfg.gemma_safetensors = String(args[i])
            elif (value := _strip_arg_value(a, String("--gemma_safetensors"))) != "":
                cfg.gemma_safetensors = value
            elif a == "--sample_prompt" and i + 1 < len(args):
                i += 1
                cfg.sample_prompt = String(args[i])
            elif (value := _strip_arg_value(a, String("--sample_prompt"))) != "":
                cfg.sample_prompt = value
            # ── LTX2 P6 AV arm (P6.0) — thin argv onto the levers (space-form;
            #    the config JSON / UI seam is the primary surface). validate() at
            #    the end enforces the XOR + torchref ranges. A later --config
            #    wholesale-replaces the lever set (config-wins); full config-wins
            #    consumption is P6.2. ──────────────────────────────────────────
            elif a == "--audio_loss_balance_mode" and i + 1 < len(args):
                i += 1
                cfg.levers.audio_loss_balance_mode = String(args[i])
            elif a == "--audio_loss_balance_beta" and i + 1 < len(args):
                i += 1
                cfg.levers.audio_loss_balance_beta = _parse_float32(String(args[i]))
            elif a == "--audio_loss_balance_eps" and i + 1 < len(args):
                i += 1
                cfg.levers.audio_loss_balance_eps = _parse_float32(String(args[i]))
            elif a == "--audio_loss_balance_min" and i + 1 < len(args):
                i += 1
                cfg.levers.audio_loss_balance_min = _parse_float32(String(args[i]))
            elif a == "--audio_loss_balance_max" and i + 1 < len(args):
                i += 1
                cfg.levers.audio_loss_balance_max = _parse_float32(String(args[i]))
            elif a == "--audio_loss_balance_ema_init" and i + 1 < len(args):
                i += 1
                cfg.levers.audio_loss_balance_ema_init = _parse_float32(String(args[i]))
            elif a == "--audio_loss_balance_target_ratio" and i + 1 < len(args):
                i += 1
                cfg.levers.audio_loss_balance_target_ratio = _parse_float32(String(args[i]))
            elif a == "--audio_loss_balance_ema_decay" and i + 1 < len(args):
                i += 1
                cfg.levers.audio_loss_balance_ema_decay = _parse_float32(String(args[i]))
            elif a == "--uncertainty_lr" and i + 1 < len(args):
                i += 1
                cfg.levers.uncertainty_lr = _parse_float32(String(args[i]))
            elif a == "--video_caption_dropout_rate" and i + 1 < len(args):
                i += 1
                cfg.levers.video_caption_dropout_rate = _parse_float32(String(args[i]))
            elif a == "--audio_caption_dropout_rate" and i + 1 < len(args):
                i += 1
                cfg.levers.audio_caption_dropout_rate = _parse_float32(String(args[i]))
            elif a == "--separate_audio_buckets":
                cfg.levers.separate_audio_buckets = True
            elif a == "--audio_bucket_strategy" and i + 1 < len(args):
                i += 1
                cfg.levers.audio_bucket_strategy = String(args[i])
            elif a == "--audio_bucket_interval" and i + 1 < len(args):
                i += 1
                cfg.levers.audio_bucket_interval = _parse_float32(String(args[i]))
            elif a == "--min_audio_batches_per_accum" and i + 1 < len(args):
                i += 1
                cfg.levers.min_audio_batches_per_accum = _parse_nonnegative_int(String(args[i]))
            elif a == "--audio_batch_probability" and i + 1 < len(args):
                i += 1
                cfg.levers.audio_batch_probability = _parse_float32(String(args[i]))
            elif a == "--use_audio_length_mask":
                cfg.levers.use_audio_length_mask = True
            elif a == "--audio_ref_use_negative_positions":
                cfg.levers.audio_ref_use_negative_positions = True
            elif a == "--audio_ref_mask_cross_attention_to_reference":
                cfg.levers.audio_ref_mask_cross_attention_to_reference = True
            elif a == "--audio_ref_mask_reference_from_text_attention":
                cfg.levers.audio_ref_mask_reference_from_text_attention = True
            elif (a == "--config" or a == "--config_json") and i + 1 < len(args):
                i += 1
                cfg.levers = read_model_config(String(args[i]))
            elif (value := _strip_arg_value(a, String("--config"))) != "":
                cfg.levers = read_model_config(value)
            elif (value := _strip_arg_value(a, String("--config_json"))) != "":
                cfg.levers = read_model_config(value)
            i += 1
        # TORCHREF-PARITY TRANSLATION (skeptic S1): torchref's loss_type "huber" IS
        # F.smooth_l1_loss(beta=huber_delta) — hv_train_network.py:499-506 maps
        # huber|smooth_l1 to the SAME formula. A torchref config copied verbatim
        # must not silently get torch-huber (clamped-grad) semantics, so for
        # ltx2 a HUBER tag is remapped to SMOOTH_L1 with beta = huber_delta,
        # LOUDLY. (Other models keep the reader's torch-huber meaning.)
        if cfg.levers.loss_fn == LOSS_FN_HUBER:
            cfg.levers.loss_fn = LOSS_FN_SMOOTH_L1
            cfg.levers.smooth_l1_beta = cfg.levers.huber_delta
            print("  [levers] NOTE: loss_fn 'huber' remapped to smooth_l1(beta=",
                  cfg.levers.smooth_l1_beta,
                  ") — torchref's huber IS smooth_l1 (hv_train_network.py:499-506)")
        # ltx2 argv WINS for its own fields. NOTE (skeptic S4): the scheduler
        # (_ltx2_step_lr) reads cfg.learning_rate/cfg.max_steps DIRECTLY; this
        # copy only keeps cfg.levers self-consistent for print_levers_summary
        # and any future TrainConfig consumer — it is informational, not the
        # LR data path.
        cfg.levers.lr = cfg.learning_rate
        cfg.levers.max_steps = cfg.max_steps
        cfg.validate()
        return cfg^


def print_config_summary(cfg: LTX2TrainerConfig):
    print("LTX2 AV trainer config contract")
    print("  checkpoint:", cfg.ltx2_checkpoint if cfg.ltx2_checkpoint != "" else "(unset)")
    print("  cache:", cfg.dataset_cache_dir if cfg.dataset_cache_dir != "" else "(unset)")
    print("  output:", cfg.output_dir)
    print("  mode:", mode_name(cfg.ltx_mode), " version:", version_name(cfg.ltx_version))
    print("  lora:", preset_name(cfg.lora_target_preset), " rank:", cfg.lora_rank, " alpha:", cfg.lora_alpha)
    print("  steps:", cfg.max_steps, " save_every:", cfg.save_every, " sample_every:", cfg.sample_every)
    print_levers_summary(cfg)


# Report which lever keys deviate from the default-off TrainConfig, or announce
# that every lever is at its legacy default (byte-identical path).
def print_levers_summary(cfg: LTX2TrainerConfig):
    var any_active = False
    if cfg.levers.loss_fn != LOSS_FN_MSE:
        print("  [levers] loss_fn:", cfg.levers.loss_fn,
              " huber_delta:", cfg.levers.huber_delta,
              " smooth_l1_beta:", cfg.levers.smooth_l1_beta)
        any_active = True
    if cfg.levers.min_snr_gamma_flow > Float32(0.0):
        print("  [levers] min_snr_gamma_flow:", cfg.levers.min_snr_gamma_flow)
        any_active = True
    if cfg.levers.lr_scheduler != 0 or cfg.levers.lr_warmup_steps != 0:
        print("  [levers] lr_scheduler:", cfg.levers.lr_scheduler,
              " lr_warmup_steps:", cfg.levers.lr_warmup_steps,
              " (base lr", cfg.levers.lr, " over", cfg.levers.max_steps, "steps)")
        any_active = True
    if cfg.levers.optimizer != TRAIN_OPTIMIZER_ADAMW:
        print("  [levers] optimizer:", cfg.levers.optimizer,
              " (F32-master path, MJ-1112; --resume owed -> adamw only)")
        any_active = True
    if cfg.min_timestep != Float32(0.0) or cfg.max_timestep != Float32(1000.0):
        print("  [levers] sigma rescale min/max_timestep:",
              cfg.min_timestep, "/", cfg.max_timestep)
        any_active = True
    if cfg.levers.caption_dropout_prob > Float32(0.0):
        print("  [levers] caption_dropout_prob:", cfg.levers.caption_dropout_prob,
              " (MJ-1113: zero prompt-embeds row on drop; mask stays all-ones)")
        any_active = True
    if not any_active:
        print("  levers: all default-off (legacy byte-identical path)")
    # val_loss + sampling are OUTPUT features (no training-math effect), reported
    # separately so they never flip the levers-default line above.
    if cfg.val_cache_dir.byte_length() > 0 and cfg.validate_every > 0:
        print("  [val] val_loss every", cfg.validate_every, "steps @", cfg.val_cache_dir,
              " (deterministic per-val-index sigma+noise; documented deviation)")
    if cfg.sample_every > 0:
        print("  [sample] native-artifact + render command file every",
              cfg.sample_every, "steps (side-process, not auto-run)")
