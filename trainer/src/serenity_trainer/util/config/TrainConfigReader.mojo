# TrainConfigReader.mojo — read a Serenity preset JSON into a TrainConfig.
#
# PORTED from mojodiffusion serenitymojo/io/train_config_reader.mojo (the parser
# machinery copied verbatim — file-read FFI, _Cursor, _parse_number,
# _read_literal, _read_scalar, _parse_optimizer, the key-dispatch loop). ONLY the
# namespace and the key->field map are changed to Serenity's preset schema
# (configs/*.json) and this port's TrainConfig. Start from defaults and overwrite
# each field as its JSON key is parsed (mirrors Serenity BaseConfig.from_dict);
# unknown keys are skipped, missing keys keep their default.
#
# Mojo 1.0.0b1: `def` not `fn`; no Python.

from std.collections import List
from std.memory import alloc
from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, BytePtr, O_RDONLY
from serenity_trainer.util.config.TrainConfig import TrainConfig


# ── General JSON number parser (signed, fractional, scientific). VERBATIM. ───
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
        raise Error(String("JSON config: expected number at byte ") + String(start))
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
            raise Error(String("JSON config: malformed exponent at byte ") + String(cur.pos))
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


# Bool / null skip-or-read. Returns 1.0 for true, 0.0 for false/null. VERBATIM.
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
    raise Error(String("JSON config: expected true/false/null at byte ") + String(cur.pos))


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


# Serenity TimestepDistribution enum string -> port enum int.
# (TimestepDistribution.py: UNIFORM/SIGMOID/LOGIT_NORMAL/HEAVY_TAIL/COS_MAP/INVERTED_PARABOLA.)
def _timestep_distribution_int(s: String) raises -> Int:
    if s == "UNIFORM":
        return 0
    elif s == "SIGMOID":
        return 1
    elif s == "LOGIT_NORMAL":
        return 2
    elif s == "HEAVY_TAIL":
        return 3
    elif s == "COS_MAP":
        return 4
    elif s == "INVERTED_PARABOLA":
        return 5
    raise Error(String("JSON config: unknown timestep_distribution '") + s + "'")


def _adapter_algo_int(s: String) raises -> Int:
    if s == "lora" or s == "LORA":
        return 0
    elif s == "locon" or s == "LOCON" or s == "lycoris" or s == "LYCORIS":
        return 7
    elif s == "loha" or s == "LOHA":
        return 2
    elif s == "dora" or s == "DORA":
        return 3
    elif s == "lokr" or s == "LOKR":
        return 4
    elif s == "oft" or s == "OFT":
        return 5
    elif s == "boft" or s == "BOFT":
        raise Error(
            "JSON config: adapter algorithm 'boft' is intentionally unsupported; "
            + "expected lora|locon|loha|lokr|dora|oft"
        )
    raise Error(
        String("JSON config: unknown adapter algorithm '") + s
        + "' (expected lora|locon|loha|lokr|dora|oft)"
    )


# Serenity nests beta1/beta2/epsilon/weight_decay under "optimizer".
def _parse_optimizer(mut cur: _Cursor, mut cfg: TrainConfig) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        var field = _parse_string(cur)
        cur.expect(0x3A)
        if field == "beta1":
            cfg.beta1 = Float32(_read_scalar(cur).num)
        elif field == "beta2":
            cfg.beta2 = Float32(_read_scalar(cur).num)
        elif field == "epsilon" or field == "eps":
            cfg.eps = Float32(_read_scalar(cur).num)
        elif field == "weight_decay":
            cfg.weight_decay = Float32(_read_scalar(cur).num)
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
        raise Error(String("JSON config: expected ',' or '}' in optimizer at byte ") + String(cur.pos))


# ── Read a whole file's bytes via raw syscalls (pure-Mojo, no Python). VERBATIM.
def _read_file_bytes(path: String) raises -> List[UInt8]:
    var fd = sys_open(path, O_RDONLY, Int32(0))
    if fd < 0:
        raise Error(String("TrainConfigReader: cannot open ") + path)
    var out = List[UInt8]()
    comptime CHUNK = 65536
    var buf = alloc[UInt8](CHUNK)
    var offset = 0
    while True:
        var n = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, offset)
        if n < 0:
            buf.free()
            _ = sys_close(fd)
            raise Error("TrainConfigReader: read error")
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
# PUBLIC: read a Serenity preset JSON into a TrainConfig (recipe fields).
# Missing keys keep TrainConfig.adamw_lora_defaults() values.
# ─────────────────────────────────────────────────────────────────────────────
def read_train_config(json_path: String) raises -> TrainConfig:
    var bytes = _read_file_bytes(json_path)
    var cur = _Cursor(bytes^)
    var cfg = TrainConfig.adamw_lora_defaults()

    cur.expect(0x7B)  # top-level '{'
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return cfg^

    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)  # ':'

        if key == "learning_rate":
            cfg.learning_rate = Float32(_read_scalar(cur).num)
        elif key == "weight_decay":
            cfg.weight_decay = Float32(_read_scalar(cur).num)
        elif key == "epsilon" or key == "eps":
            cfg.eps = Float32(_read_scalar(cur).num)
        elif key == "epochs":
            cfg.epochs = Int(_read_scalar(cur).num)
        elif key == "batch_size":
            cfg.batch_size = Int(_read_scalar(cur).num)
        elif key == "gradient_accumulation_steps":
            cfg.gradient_accumulation_steps = Int(_read_scalar(cur).num)
        elif key == "clip_grad_norm":
            cfg.clip_grad_norm = Float32(_read_scalar(cur).num)
        elif key == "seed":
            cfg.seed = UInt32(Int(_read_scalar(cur).num))
        elif key == "lora_rank":
            cfg.lora_rank = Int(_read_scalar(cur).num)
        elif key == "lora_alpha":
            cfg.lora_alpha = Float32(_read_scalar(cur).num)
        elif key == "network_algorithm" or key == "adapter_algo" or key == "algo":
            var sc = _read_scalar(cur)
            if not sc.is_string:
                raise Error("JSON config: network_algorithm/adapter_algo/algo must be a string")
            cfg.adapter_algo = _adapter_algo_int(sc.s)
        elif key == "timestep_distribution":
            var sc = _read_scalar(cur)
            if not sc.is_string:
                raise Error("JSON config: timestep_distribution must be a string")
            cfg.timestep_distribution = _timestep_distribution_int(sc.s)
        elif key == "min_noising_strength":
            cfg.min_noising_strength = Float32(_read_scalar(cur).num)
        elif key == "max_noising_strength":
            cfg.max_noising_strength = Float32(_read_scalar(cur).num)
        elif key == "noising_weight":
            cfg.noising_weight = Float32(_read_scalar(cur).num)
        elif key == "noising_bias":
            cfg.noising_bias = Float32(_read_scalar(cur).num)
        elif key == "timestep_shift":
            cfg.timestep_shift = Float32(_read_scalar(cur).num)
        elif key == "dynamic_timestep_shifting":
            cfg.dynamic_timestep_shifting = _read_scalar(cur).num != 0.0
        elif key == "guidance_scale":
            cfg.guidance_scale = Float32(_read_scalar(cur).num)
        elif key == "min_snr_gamma":
            cfg.min_snr_gamma = Float32(_read_scalar(cur).num)
        elif key == "optimizer":
            _parse_optimizer(cur, cfg)
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
        raise Error(String("JSON config: expected ',' or '}' at top level at byte ") + String(cur.pos))

    return cfg^
