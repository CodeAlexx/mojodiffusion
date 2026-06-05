# config.mojo -- LTX-2.3 AV trainer config and CLI contract.
#
# Runtime contract: Mojo parses the launch shape and carries concrete training
# policy. Python/TOML references are development oracles only.

from std.collections import List


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
            elif a == "--audio_only_sequence_resolution" and i + 1 < len(args):
                i += 1
                cfg.audio_only_sequence_resolution = _parse_nonnegative_int(String(args[i]))
            elif (value := _strip_arg_value(a, String("--audio_only_sequence_resolution"))) != "":
                cfg.audio_only_sequence_resolution = _parse_nonnegative_int(value)
            elif a == "--fail_on_unready":
                cfg.fail_on_unready = True
            elif (value := _strip_arg_value(a, String("--fail_on_unready"))) != "":
                cfg.fail_on_unready = _parse_bool(value)
            i += 1
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
