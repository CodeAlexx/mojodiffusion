# train_config_reader.mojo — read a OneTrainer-style JSON config into the
# project's TrainConfig (training/train_config.mojo).
#
# ── Why a NEW parser and not json_header.mojo's parse_header ─────────────────
# io/json_header.mojo is a FLAT-safetensors-header parser: its `_parse_int` only
# reads non-negative INTEGERS and its top-level loop expects every value to be a
# {dtype,shape,data_offsets} object. OneTrainer configs are general JSON —
# floats ("learning_rate":0.0004), scientific notation ("eps":1e-08), booleans,
# strings ("model_type":"FLUX_2"), and NESTED objects ("optimizer":{...}). So we
# REUSE json_header's proven cursor primitives (the `_Cursor` byte-cursor with
# peek/advance/skip_ws/expect and the escape-aware `_parse_string`) and ADD a
# general scalar reader (float + sci-notation + bool + null) plus a one-level
# descent into the `optimizer` object. We do NOT build a full DOM — Mojo 1.0.0b1
# Dict[String, <variant>] is awkward for heterogeneous values; instead we scan
# once and pull out exactly the fields TrainConfig needs, skipping the rest with
# json_header's `_skip_value` (balanced-brace scanner).
#
# ── Keys mapped (verified against /home/alex/OneTrainer/configs/
#    klein9b_loss_compare.json, 2026-05-30) ─────────────────────────────────
#   top-level "learning_rate"  (Float64, e.g. 0.0004) → TrainConfig.lr
#   top-level "lora_rank"      (Int, 16)              → TrainConfig.lora_rank
#   top-level "lora_alpha"     (Int/Float, 16)        → TrainConfig.lora_alpha
#   top-level "model_type"     (String, "FLUX_2")     → TrainConfig.name
#   nested   "optimizer"."eps" (Float64, 1e-08)       → TrainConfig.eps
#   nested   "optimizer"."weight_decay" (Float64, 0.01) → returned in TrainConfigExtra
#   nested   "optimizer"."beta1"/"beta2"             → returned in TrainConfigExtra
# NOTE: OneTrainer configs do NOT carry timestep_shift, nor the model DIMS
# (d_model/n_heads/head_dim/mlp_hidden/n_layers). Those come from the per-model
# config constructor in serenitymojo/models/<model>/config.mojo (see
# train_config.mojo header: "Per-model constructors ... return this type"). This
# reader fills the RECIPE scalars it CAN read from JSON and takes the model dims
# + timestep_shift as caller-supplied defaults, so the result is a complete,
# valid TrainConfig. timestep_shift defaults to 1.8 (the Klein9B empirical value
# — feedback_klein9b_timestep_shift_1.8).
#
# Mojo 1.0.0b1: `def` not `fn`; STDtype-style value structs; no Python.

from std.collections import List
from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value
from serenitymojo.training.train_config import TrainConfig


# Optimizer scalars that DON'T fit in TrainConfig but a trainer wants (passed to
# loop.mojo::apply_step / optim.adamw_step). Returned alongside TrainConfig.
@fieldwise_init
struct OptimExtra(Copyable, Movable):
    var weight_decay: Float32
    var beta1: Float32
    var beta2: Float32


@fieldwise_init
struct ReadConfigResult(Movable):
    var cfg: TrainConfig
    var optim: OptimExtra


# ── General JSON number parser (signed, fractional, scientific) ──────────────
# json_header._parse_int only does non-negative integers. We parse a full JSON
# number to Float64 BY HAND (no stdlib string→float dependency — `atof` is not
# verified present in this toolchain, so we avoid it). We accumulate the integer
# and fractional mantissa as a Float64 and apply a base-10 exponent built from
# the fractional length and any explicit e-notation exponent. Config numbers are
# small-magnitude (lr 4e-4, eps 1e-8, ranks/alphas) so Float64 accumulation is
# exact enough for the recipe (and lr/eps are stored as Float32 downstream).
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
    # integer part
    while not cur.at_end():
        var ch = cur.peek()
        if ch >= 0x30 and ch <= 0x39:
            mantissa = mantissa * 10.0 + Float64(ch - 0x30)
            any_digit = True
            cur.advance()
        else:
            break
    # fractional part
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
    # explicit exponent
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
    # net base-10 exponent = explicit exp - fractional digit count
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


# ── Bool / null skip-or-read. Returns 1.0 for true, 0.0 for false/null ───────
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


# Read a scalar VALUE generically; for strings returns "" via out_str, else the
# numeric value. We only need string for "model_type"; everything else numeric.
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
    # number
    return _Scalar(False, String(""), _parse_number(cur))


# ── Parse the nested "optimizer" object, pulling eps + weight_decay/beta1/beta2.
# eps goes into TrainConfig; weight_decay/beta1/beta2 ride in OptimExtra. The
# wrapper _OptimAll carries both out of the single pass.
@fieldwise_init
struct _OptimAll(Copyable, Movable):
    var eps: Float32
    var extra: OptimExtra


def _parse_optimizer_all(mut cur: _Cursor) raises -> _OptimAll:
    var eps = Float64(1e-8)
    var wd = Float64(0.0)
    var b1 = Float64(0.9)
    var b2 = Float64(0.999)
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return _OptimAll(Float32(eps), OptimExtra(Float32(wd), Float32(b1), Float32(b2)))
    while True:
        var field = _parse_string(cur)
        cur.expect(0x3A)
        if field == "eps":
            eps = _read_scalar(cur).num
        elif field == "weight_decay":
            wd = _read_scalar(cur).num
        elif field == "beta1":
            b1 = _read_scalar(cur).num
        elif field == "beta2":
            b2 = _read_scalar(cur).num
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
    return _OptimAll(Float32(eps), OptimExtra(Float32(wd), Float32(b1), Float32(b2)))


# ── Read a whole file's bytes. Configs are tiny; we read in 64 KiB chunks via
# io/ffi's raw syscalls (sys_open/sys_pread/sys_close) to stay pure-Mojo
# (PLAN.md: no Python in the runtime path). No mmap — mmap.mojo is sized for the
# multi-GB weight files. sys_pread is already exported by io/ffi (ffi.mojo:137).
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
# PUBLIC: read a OneTrainer JSON config into a TrainConfig + OptimExtra.
# Model DIMS + timestep_shift are caller-supplied (JSON doesn't carry them); the
# JSON supplies the RECIPE scalars (lr, lora_rank, lora_alpha, eps) + name.
# ─────────────────────────────────────────────────────────────────────────────
def read_train_config(
    json_path: String,
    d_model: Int,
    n_heads: Int,
    head_dim: Int,
    mlp_hidden: Int,
    n_layers: Int,
    timestep_shift: Float32 = Float32(1.8),
) raises -> ReadConfigResult:
    """Parse `json_path` (a OneTrainer config) and return a fully-populated
    TrainConfig plus the OptimExtra (weight_decay/beta1/beta2) a trainer needs.

    Recipe scalars read from JSON:  lr, lora_rank, lora_alpha, eps, name.
    Model dims + timestep_shift:    caller-supplied (from the per-model config).
    Defaults if a key is absent:    lr=1e-4, rank=16, alpha=16, eps=1e-8."""
    var bytes = _read_file_bytes(json_path)
    var cur = _Cursor(bytes^)

    # Recipe defaults (used when a key is absent from the file).
    var lr = Float64(1e-4)
    var lora_rank = 16
    var lora_alpha = Float64(16.0)
    var eps = Float64(1e-8)
    var name = String("unknown")
    var optim = OptimExtra(Float32(0.0), Float32(0.9), Float32(0.999))

    cur.expect(0x7B)  # top-level '{'
    cur.skip_ws()
    if cur.peek() == 0x7D:  # empty object
        cur.advance()
    else:
        while True:
            var key = _parse_string(cur)
            cur.expect(0x3A)  # ':'
            if key == "learning_rate":
                lr = _read_scalar(cur).num
            elif key == "lora_rank":
                lora_rank = Int(_read_scalar(cur).num)
            elif key == "lora_alpha":
                lora_alpha = _read_scalar(cur).num
            elif key == "model_type":
                var sc = _read_scalar(cur)
                if sc.is_string:
                    name = sc.s
            elif key == "optimizer":
                var oa = _parse_optimizer_all(cur)
                eps = Float64(oa.eps)
                optim = oa.extra.copy()
            else:
                _skip_value(cur)  # skip every other top-level key
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

    var cfg = TrainConfig(
        name^,
        d_model,
        n_heads,
        head_dim,
        mlp_hidden,
        n_layers,
        Float32(lr),
        timestep_shift,
        lora_rank,
        Float32(lora_alpha),
        Float32(eps),
    )
    return ReadConfigResult(cfg^, optim^)
