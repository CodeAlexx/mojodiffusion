# Shared validation/sample prompt config reader.
#
# Trainer configs point at one JSON file with any number of validation prompts.
# The trainer only reads precomputed cap-cache paths from this file; it must not
# load text encoders during training.

from std.collections import List

from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value
from serenitymojo.io.train_config_reader import _read_file_bytes, _read_scalar


comptime SAMPLE_UNIT_EPOCH = 0
comptime SAMPLE_UNIT_STEP = 1
comptime SAMPLE_UNIT_SECOND = 2
comptime SAMPLE_UNIT_MINUTE = 3
comptime SAMPLE_UNIT_HOUR = 4
comptime SAMPLE_UNIT_NEVER = 5
comptime SAMPLE_UNIT_ALWAYS = 6


@fieldwise_init
struct SamplePrompt(Copyable, Movable):
    var enabled: Bool
    var label: String
    var prompt: String
    var negative: String
    var width: Int
    var height: Int
    var frames: Int
    var length: Float32
    var fps: Float32
    var steps: Int
    var cfg: Float32
    var seed: UInt64
    var random_seed: Bool
    var noise_scheduler: String
    var sample_inpainting: Bool
    var base_image_path: String
    var mask_image_path: String
    var caps_pos: String
    var caps_neg: String


@fieldwise_init
struct SampleCadence(Copyable, Movable):
    var sample_after: Int
    var sample_after_unit: Int
    var sample_skip_first: Int
    var sample_at_start: Bool
    var save_before_sample: Bool
    var samples_to_tensorboard: Bool
    var non_ema_sampling: Bool
    var sample_definition_file_name: String

    def sample_every_steps(self, fallback: Int) -> Int:
        if self.sample_after_unit == SAMPLE_UNIT_STEP and self.sample_after > 0:
            return self.sample_after
        return fallback


struct SamplePromptConfig(Movable):
    var schema: String
    var every_steps: Int
    var sample_after: Int
    var sample_after_unit: Int
    var sample_skip_first: Int
    var sample_at_start: Bool
    var save_before_sample: Bool
    var precache_required: Bool
    var enforce_min_image_size: Bool
    var samples_to_tensorboard: Bool
    var non_ema_sampling: Bool
    var sample_definition_file_name: String
    var width: Int
    var height: Int
    var frames: Int
    var length: Float32
    var fps: Float32
    var steps: Int
    var cfg: Float32
    var seed: UInt64
    var random_seed: Bool
    var noise_scheduler: String
    var sample_inpainting: Bool
    var base_image_path: String
    var mask_image_path: String
    var negative: String
    var prompts: List[SamplePrompt]

    def __init__(out self):
        self.schema = String("serenity.sample_prompts.v1")
        self.every_steps = 500
        self.sample_after = 500
        self.sample_after_unit = SAMPLE_UNIT_STEP
        self.sample_skip_first = 0
        self.sample_at_start = True
        self.save_before_sample = True
        self.precache_required = True
        self.enforce_min_image_size = True
        self.samples_to_tensorboard = True
        self.non_ema_sampling = True
        self.sample_definition_file_name = String("")
        self.width = 1024
        self.height = 1024
        self.frames = 1
        self.length = Float32(10.0)
        self.fps = Float32(24.0)
        self.steps = 20
        self.cfg = Float32(4.0)
        self.seed = UInt64(42)
        self.random_seed = False
        self.noise_scheduler = String("")
        self.sample_inpainting = False
        self.base_image_path = String("")
        self.mask_image_path = String("")
        self.negative = String("")
        self.prompts = List[SamplePrompt]()


def sample_time_unit_from_string(unit: String) raises -> Int:
    if unit == String("EPOCH") or unit == String("epoch"):
        return SAMPLE_UNIT_EPOCH
    if unit == String("STEP") or unit == String("step"):
        return SAMPLE_UNIT_STEP
    if unit == String("SECOND") or unit == String("second"):
        return SAMPLE_UNIT_SECOND
    if unit == String("MINUTE") or unit == String("minute"):
        return SAMPLE_UNIT_MINUTE
    if unit == String("HOUR") or unit == String("hour"):
        return SAMPLE_UNIT_HOUR
    if unit == String("NEVER") or unit == String("never"):
        return SAMPLE_UNIT_NEVER
    if unit == String("ALWAYS") or unit == String("always"):
        return SAMPLE_UNIT_ALWAYS
    raise Error(String("sample cadence: unknown OneTrainer time unit ") + unit)


def sample_time_unit_name(unit: Int) -> String:
    if unit == SAMPLE_UNIT_EPOCH:
        return String("EPOCH")
    if unit == SAMPLE_UNIT_STEP:
        return String("STEP")
    if unit == SAMPLE_UNIT_SECOND:
        return String("SECOND")
    if unit == SAMPLE_UNIT_MINUTE:
        return String("MINUTE")
    if unit == SAMPLE_UNIT_HOUR:
        return String("HOUR")
    if unit == SAMPLE_UNIT_NEVER:
        return String("NEVER")
    if unit == SAMPLE_UNIT_ALWAYS:
        return String("ALWAYS")
    return String("UNKNOWN")


def default_sample_cadence(fallback_every_steps: Int = 500) -> SampleCadence:
    var every = fallback_every_steps
    if every <= 0:
        every = 500
    return SampleCadence(
        every,
        SAMPLE_UNIT_STEP,
        0,
        True,
        True,
        True,
        True,
        String(""),
    )


def _read_bool(mut cur: _Cursor) raises -> Bool:
    return _read_scalar(cur).num != 0.0


def _read_string(mut cur: _Cursor) raises -> String:
    var sc = _read_scalar(cur)
    if sc.is_string:
        return sc.s.copy()
    return String("")


def _apply_scalar_default(key: String, mut cur: _Cursor, mut cfg: SamplePromptConfig) raises -> Bool:
    if key == "every_steps" or key == "sample_every" or key == "sample_every_n_steps":
        cfg.every_steps = Int(_read_scalar(cur).num)
        cfg.sample_after = cfg.every_steps
        cfg.sample_after_unit = SAMPLE_UNIT_STEP
    elif key == "sample_after":
        cfg.sample_after = Int(_read_scalar(cur).num)
        cfg.every_steps = cfg.sample_after
    elif key == "sample_after_unit":
        cfg.sample_after_unit = sample_time_unit_from_string(_read_string(cur))
    elif key == "sample_skip_first":
        cfg.sample_skip_first = Int(_read_scalar(cur).num)
    elif key == "sample_definition_file_name":
        cfg.sample_definition_file_name = _read_string(cur)
    elif key == "sample_at_start" or key == "baseline_at_start":
        cfg.sample_at_start = _read_bool(cur)
    elif key == "save_before_sample":
        cfg.save_before_sample = _read_bool(cur)
    elif key == "precache_required":
        cfg.precache_required = _read_bool(cur)
    elif key == "enforce_min_image_size":
        cfg.enforce_min_image_size = _read_bool(cur)
    elif key == "samples_to_tensorboard":
        cfg.samples_to_tensorboard = _read_bool(cur)
    elif key == "non_ema_sampling":
        cfg.non_ema_sampling = _read_bool(cur)
    elif key == "width":
        cfg.width = Int(_read_scalar(cur).num)
    elif key == "height":
        cfg.height = Int(_read_scalar(cur).num)
    elif key == "frames":
        cfg.frames = Int(_read_scalar(cur).num)
    elif key == "length":
        cfg.length = Float32(_read_scalar(cur).num)
    elif key == "fps":
        cfg.fps = Float32(_read_scalar(cur).num)
    elif key == "steps" or key == "sample_steps" or key == "diffusion_steps":
        cfg.steps = Int(_read_scalar(cur).num)
    elif key == "cfg" or key == "guidance_scale" or key == "cfg_scale":
        cfg.cfg = Float32(_read_scalar(cur).num)
    elif key == "seed":
        cfg.seed = UInt64(Int(_read_scalar(cur).num))
    elif key == "random_seed":
        cfg.random_seed = _read_bool(cur)
    elif key == "noise_scheduler":
        cfg.noise_scheduler = _read_string(cur)
    elif key == "sample_inpainting":
        cfg.sample_inpainting = _read_bool(cur)
    elif key == "base_image_path":
        cfg.base_image_path = _read_string(cur)
    elif key == "mask_image_path":
        cfg.mask_image_path = _read_string(cur)
    elif key == "negative" or key == "negative_prompt":
        cfg.negative = _read_string(cur)
    else:
        return False
    return True


def _parse_defaults(mut cur: _Cursor, mut cfg: SamplePromptConfig) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if not _apply_scalar_default(key, cur, cfg):
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x7D:
            cur.advance()
            break
        raise Error("sample prompt config: expected ',' or '}' in defaults")


def _parse_caps(mut cur: _Cursor, mut prompt: SamplePrompt) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if key == "positive" or key == "pos":
            prompt.caps_pos = _read_string(cur)
        elif key == "negative" or key == "neg":
            prompt.caps_neg = _read_string(cur)
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
        raise Error("sample prompt config: expected ',' or '}' in caps")


def _parse_one_prompt(mut cur: _Cursor, cfg: SamplePromptConfig, idx: Int) raises -> SamplePrompt:
    var p = SamplePrompt(
        True,
        String("p") + String(idx + 1),
        String(""),
        cfg.negative.copy(),
        cfg.width,
        cfg.height,
        cfg.frames,
        cfg.length,
        cfg.fps,
        cfg.steps,
        cfg.cfg,
        cfg.seed,
        cfg.random_seed,
        cfg.noise_scheduler.copy(),
        cfg.sample_inpainting,
        cfg.base_image_path.copy(),
        cfg.mask_image_path.copy(),
        String(""),
        String(""),
    )
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return p^
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if key == "enabled":
            p.enabled = _read_bool(cur)
        elif key == "id" or key == "label":
            p.label = _read_string(cur)
        elif key == "prompt" or key == "text":
            p.prompt = _read_string(cur)
        elif key == "negative" or key == "negative_prompt":
            p.negative = _read_string(cur)
        elif key == "width":
            p.width = Int(_read_scalar(cur).num)
        elif key == "height":
            p.height = Int(_read_scalar(cur).num)
        elif key == "frames":
            p.frames = Int(_read_scalar(cur).num)
        elif key == "length":
            p.length = Float32(_read_scalar(cur).num)
        elif key == "fps":
            p.fps = Float32(_read_scalar(cur).num)
        elif key == "steps" or key == "sample_steps" or key == "diffusion_steps":
            p.steps = Int(_read_scalar(cur).num)
        elif key == "cfg" or key == "guidance_scale" or key == "cfg_scale":
            p.cfg = Float32(_read_scalar(cur).num)
        elif key == "seed":
            p.seed = UInt64(Int(_read_scalar(cur).num))
        elif key == "random_seed":
            p.random_seed = _read_bool(cur)
        elif key == "noise_scheduler":
            p.noise_scheduler = _read_string(cur)
        elif key == "sample_inpainting":
            p.sample_inpainting = _read_bool(cur)
        elif key == "base_image_path":
            p.base_image_path = _read_string(cur)
        elif key == "mask_image_path":
            p.mask_image_path = _read_string(cur)
        elif key == "caps":
            _parse_caps(cur, p)
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
        raise Error("sample prompt config: expected ',' or '}' in prompt")
    return p^


def _parse_prompts(mut cur: _Cursor, mut cfg: SamplePromptConfig) raises:
    cur.expect(0x5B)
    cur.skip_ws()
    if cur.peek() == 0x5D:
        cur.advance()
        return
    var idx = 0
    while True:
        var p = _parse_one_prompt(cur, cfg, idx)
        if p.enabled and p.prompt == String(""):
            raise Error(String("sample prompt config: prompt ") + String(idx + 1) + String(" has no text"))
        if p.enabled and cfg.precache_required and (p.caps_pos == String("") or p.caps_neg == String("")):
            raise Error(
                String("sample prompt config: prompt ") + p.label
                + String(" must provide caps.positive and caps.negative")
            )
        cfg.prompts.append(p^)
        idx += 1
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C:
            cur.advance()
            continue
        if ch == 0x5D:
            cur.advance()
            break
        raise Error("sample prompt config: expected ',' or ']' in prompts")


def _validate_prompt_sizes(cfg: SamplePromptConfig) raises:
    if not cfg.enforce_min_image_size:
        return
    for i in range(len(cfg.prompts)):
        var p = cfg.prompts[i].copy()
        if p.enabled and p.frames == 1 and (p.width < 1024 or p.height < 1024):
            raise Error(
                String("sample prompt config: image prompt ") + p.label
                + String(" is ") + String(p.width) + String("x") + String(p.height)
                + String("; image validation samples must be 1024x1024 or larger")
            )


def _parse_top_defaults(var bytes: List[UInt8]) raises -> SamplePromptConfig:
    var cur = _Cursor(bytes^)
    var cfg = SamplePromptConfig()
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return cfg^
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if key == "schema":
            cfg.schema = _read_string(cur)
        elif key == "defaults" or key == "params":
            _parse_defaults(cur, cfg)
        elif _apply_scalar_default(key, cur, cfg):
            pass
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
        raise Error("sample prompt config: expected ',' or '}' at top level")
    return cfg^


def _read_top_level_prompt_list(var bytes: List[UInt8]) raises -> SamplePromptConfig:
    var cfg = SamplePromptConfig()
    cfg.schema = String("onetrainer.samples.v1")
    cfg.precache_required = False
    cfg.enforce_min_image_size = False
    var cur = _Cursor(bytes^)
    _parse_prompts(cur, cfg)
    if len(cfg.prompts) == 0:
        raise Error("sample prompt config has no prompts")
    return cfg^


def read_sample_prompt_config(path: String) raises -> SamplePromptConfig:
    var bytes = _read_file_bytes(path)
    var probe = _Cursor(bytes.copy())
    probe.skip_ws()
    if probe.peek() == 0x5B:
        var list_cfg = _read_top_level_prompt_list(bytes^)
        return list_cfg^

    var cfg = _parse_top_defaults(bytes.copy())
    var cur = _Cursor(bytes^)
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return cfg^
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if key == "prompts":
            _parse_prompts(cur, cfg)
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
        raise Error("sample prompt config: expected ',' or '}' at top level")
    if len(cfg.prompts) == 0:
        raise Error(String("sample prompt config has no prompts: ") + path)
    _validate_prompt_sizes(cfg)
    return cfg^


def cadence_from_prompt_config(cfg: SamplePromptConfig) -> SampleCadence:
    return SampleCadence(
        cfg.sample_after,
        cfg.sample_after_unit,
        cfg.sample_skip_first,
        cfg.sample_at_start,
        cfg.save_before_sample,
        cfg.samples_to_tensorboard,
        cfg.non_ema_sampling,
        cfg.sample_definition_file_name.copy(),
    )


def _parse_cadence_object(mut cur: _Cursor, mut cadence: SampleCadence) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if key == "sample_after":
            cadence.sample_after = Int(_read_scalar(cur).num)
        elif key == "sample_every" or key == "sample_every_n_steps":
            cadence.sample_after = Int(_read_scalar(cur).num)
            cadence.sample_after_unit = SAMPLE_UNIT_STEP
        elif key == "sample_after_unit":
            cadence.sample_after_unit = sample_time_unit_from_string(_read_string(cur))
        elif key == "sample_skip_first":
            cadence.sample_skip_first = Int(_read_scalar(cur).num)
        elif key == "sample_at_start" or key == "baseline_at_start":
            cadence.sample_at_start = _read_bool(cur)
        elif key == "save_before_sample":
            cadence.save_before_sample = _read_bool(cur)
        elif key == "samples_to_tensorboard":
            cadence.samples_to_tensorboard = _read_bool(cur)
        elif key == "non_ema_sampling":
            cadence.non_ema_sampling = _read_bool(cur)
        elif key == "sample_definition_file_name" or key == "validation_prompts_file":
            cadence.sample_definition_file_name = _read_string(cur)
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
        raise Error("sample cadence: expected ',' or '}'")


def read_sample_cadence_config(path: String, fallback_every_steps: Int = 500) raises -> SampleCadence:
    var bytes = _read_file_bytes(path)
    var cur = _Cursor(bytes^)
    var cadence = default_sample_cadence(fallback_every_steps)
    cur.skip_ws()
    if cur.peek() == 0x5B:
        return cadence^
    _parse_cadence_object(cur, cadence)
    return cadence^


def validate_step_sample_cadence(cadence: SampleCadence) raises:
    if cadence.sample_after_unit == SAMPLE_UNIT_STEP:
        if cadence.sample_after <= 0:
            raise Error("sample cadence: STEP unit requires sample_after > 0")
        return
    if cadence.sample_after_unit == SAMPLE_UNIT_NEVER or cadence.sample_after_unit == SAMPLE_UNIT_ALWAYS:
        return
    raise Error(
        String("sample cadence: ")
        + sample_time_unit_name(cadence.sample_after_unit)
        + String(" needs epoch/time progress plumbing; only STEP/NEVER/ALWAYS is ready")
    )


def should_sample_completed_step(cadence: SampleCadence, completed_step: Int) raises -> Bool:
    validate_step_sample_cadence(cadence)
    if cadence.sample_after_unit == SAMPLE_UNIT_NEVER:
        return False
    if cadence.sample_after_unit == SAMPLE_UNIT_ALWAYS:
        if completed_step == 0:
            return cadence.sample_at_start
        return completed_step > cadence.sample_skip_first
    if completed_step == 0:
        return cadence.sample_at_start
    if completed_step <= cadence.sample_skip_first:
        return False
    return completed_step % cadence.sample_after == 0


def next_sample_completed_step(cadence: SampleCadence, current_step: Int, max_steps: Int) raises -> Int:
    validate_step_sample_cadence(cadence)
    if cadence.sample_after_unit == SAMPLE_UNIT_NEVER:
        return -1
    if cadence.sample_after_unit == SAMPLE_UNIT_ALWAYS:
        var n = current_step + 1
        if n > max_steps:
            return -1
        return n
    var every = cadence.sample_after
    var n = current_step + every - (current_step % every)
    if n <= current_step:
        n += every
    if n <= cadence.sample_skip_first:
        n = cadence.sample_skip_first + every - (cadence.sample_skip_first % every)
    if n > max_steps:
        return -1
    return n
