# Shared validation/sample prompt config reader.
#
# Trainer configs point at one JSON file with any number of validation prompts.
# The trainer only reads precomputed cap-cache paths from this file; it must not
# load text encoders during training.

from std.collections import List

from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value
from serenitymojo.io.train_config_reader import _read_file_bytes, _read_scalar


@fieldwise_init
struct SamplePrompt(Copyable, Movable):
    var label: String
    var prompt: String
    var negative: String
    var width: Int
    var height: Int
    var frames: Int
    var fps: Float32
    var steps: Int
    var cfg: Float32
    var seed: UInt64
    var caps_pos: String
    var caps_neg: String


struct SamplePromptConfig(Movable):
    var schema: String
    var every_steps: Int
    var sample_at_start: Bool
    var save_before_sample: Bool
    var precache_required: Bool
    var width: Int
    var height: Int
    var frames: Int
    var fps: Float32
    var steps: Int
    var cfg: Float32
    var seed: UInt64
    var negative: String
    var prompts: List[SamplePrompt]

    def __init__(out self):
        self.schema = String("serenity.sample_prompts.v1")
        self.every_steps = 500
        self.sample_at_start = True
        self.save_before_sample = True
        self.precache_required = True
        self.width = 1024
        self.height = 1024
        self.frames = 1
        self.fps = Float32(24.0)
        self.steps = 20
        self.cfg = Float32(4.0)
        self.seed = UInt64(42)
        self.negative = String("")
        self.prompts = List[SamplePrompt]()


def _read_bool(mut cur: _Cursor) raises -> Bool:
    return _read_scalar(cur).num != 0.0


def _read_string(mut cur: _Cursor) raises -> String:
    var sc = _read_scalar(cur)
    if sc.is_string:
        return sc.s.copy()
    return String("")


def _parse_defaults(mut cur: _Cursor, mut cfg: SamplePromptConfig) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if key == "every_steps" or key == "sample_every":
            cfg.every_steps = Int(_read_scalar(cur).num)
        elif key == "sample_at_start" or key == "baseline_at_start":
            cfg.sample_at_start = _read_bool(cur)
        elif key == "save_before_sample":
            cfg.save_before_sample = _read_bool(cur)
        elif key == "precache_required":
            cfg.precache_required = _read_bool(cur)
        elif key == "width":
            cfg.width = Int(_read_scalar(cur).num)
        elif key == "height":
            cfg.height = Int(_read_scalar(cur).num)
        elif key == "frames":
            cfg.frames = Int(_read_scalar(cur).num)
        elif key == "fps":
            cfg.fps = Float32(_read_scalar(cur).num)
        elif key == "steps" or key == "sample_steps":
            cfg.steps = Int(_read_scalar(cur).num)
        elif key == "cfg" or key == "guidance_scale":
            cfg.cfg = Float32(_read_scalar(cur).num)
        elif key == "seed":
            cfg.seed = UInt64(Int(_read_scalar(cur).num))
        elif key == "negative":
            cfg.negative = _read_string(cur)
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
        String("p") + String(idx + 1),
        String(""),
        cfg.negative.copy(),
        cfg.width,
        cfg.height,
        cfg.frames,
        cfg.fps,
        cfg.steps,
        cfg.cfg,
        cfg.seed,
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
        if key == "id" or key == "label":
            p.label = _read_string(cur)
        elif key == "prompt" or key == "text":
            p.prompt = _read_string(cur)
        elif key == "negative":
            p.negative = _read_string(cur)
        elif key == "width":
            p.width = Int(_read_scalar(cur).num)
        elif key == "height":
            p.height = Int(_read_scalar(cur).num)
        elif key == "frames":
            p.frames = Int(_read_scalar(cur).num)
        elif key == "fps":
            p.fps = Float32(_read_scalar(cur).num)
        elif key == "steps" or key == "sample_steps":
            p.steps = Int(_read_scalar(cur).num)
        elif key == "cfg" or key == "guidance_scale":
            p.cfg = Float32(_read_scalar(cur).num)
        elif key == "seed":
            p.seed = UInt64(Int(_read_scalar(cur).num))
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
        if p.prompt == String(""):
            raise Error(String("sample prompt config: prompt ") + String(idx + 1) + String(" has no text"))
        if cfg.precache_required and (p.caps_pos == String("") or p.caps_neg == String("")):
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


def read_sample_prompt_config(path: String) raises -> SamplePromptConfig:
    var bytes = _read_file_bytes(path)
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
    return cfg^
