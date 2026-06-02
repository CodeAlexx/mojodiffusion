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
#
# Mojo 1.0.0b1: `def` not `fn`; no Python.

from std.collections import List
from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value
from serenitymojo.training.train_config import TrainConfig


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


# Parse the nested "optimizer" object, mutating eps/weight_decay/beta1/beta2 on cfg.
def _parse_optimizer(mut cur: _Cursor, mut cfg: TrainConfig) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return
    while True:
        var field = _parse_string(cur)
        cur.expect(0x3A)
        if field == "eps":
            cfg.eps = Float32(_read_scalar(cur).num)
        elif field == "weight_decay":
            cfg.weight_decay = Float32(_read_scalar(cur).num)
        elif field == "beta1":
            cfg.beta1 = Float32(_read_scalar(cur).num)
        elif field == "beta2":
            cfg.beta2 = Float32(_read_scalar(cur).num)
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
    if s == "constant":
        return 0
    elif s == "linear":
        return 1
    elif s == "cosine":
        return 2
    elif s == "cosine_with_restarts":
        return 3
    elif s == "polynomial":
        return 4
    elif s == "rex":
        return 5
    raise Error(
        String("JSON config: unknown lr_scheduler '") + s
        + "' (expected constant|linear|cosine|cosine_with_restarts|polynomial|rex)"
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
    if s == "uniform":
        return 0
    elif s == "sigmoid":
        return 1
    elif s == "logit_normal":
        return 2
    raise Error(
        String("JSON config: unknown timestep_distribution '") + s
        + "' (expected uniform|sigmoid|logit_normal)"
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
        elif key == "checkpoint":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.checkpoint = sc.s
        elif key == "vae":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.vae = sc.s
        elif key == "validation_prompts_file" or key == "sample_prompts_file":
            var sc = _read_scalar(cur)
            if sc.is_string:
                cfg.validation_prompts_file = sc.s
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
        elif key == "lora_rank":
            cfg.lora_rank = Int(_read_scalar(cur).num)
        elif key == "lora_alpha":
            cfg.lora_alpha = Float32(_read_scalar(cur).num)
        elif key == "timestep_shift":
            cfg.timestep_shift = Float32(_read_scalar(cur).num)
        elif key == "max_grad_norm":
            cfg.max_grad_norm = Float32(_read_scalar(cur).num)
        elif key == "max_steps":
            cfg.max_steps = Int(_read_scalar(cur).num)
        elif key == "save_every":
            cfg.save_every = Int(_read_scalar(cur).num)
        elif key == "sample_every":
            cfg.sample_every = Int(_read_scalar(cur).num)
        elif key == "optimizer":
            _parse_optimizer(cur, cfg)
        # ── Wave 2A: lr scheduler ──
        elif key == "lr_scheduler":
            var sc = _read_scalar(cur)
            if not sc.is_string:
                raise Error("JSON config: lr_scheduler must be a string")
            cfg.lr_scheduler = _lr_scheduler_int(sc.s)
        elif key == "lr_warmup_steps":
            cfg.lr_warmup_steps = Int(_read_scalar(cur).num)
        elif key == "lr_min_factor":
            cfg.lr_min_factor = Float32(_read_scalar(cur).num)
        elif key == "lr_cycles":
            cfg.lr_cycles = Float32(_read_scalar(cur).num)
        # ── Wave 2A: loss weighting ──
        elif key == "min_snr_gamma":
            cfg.min_snr_gamma = Float32(_read_scalar(cur).num)
        elif key == "debiased":
            cfg.debiased = _read_scalar(cur).num != 0.0
        # ── Wave 2A: combined loss strengths ──
        elif key == "loss_mse_strength":
            cfg.loss_mse_strength = Float32(_read_scalar(cur).num)
        elif key == "loss_mae_strength":
            cfg.loss_mae_strength = Float32(_read_scalar(cur).num)
        elif key == "loss_huber_strength":
            cfg.loss_huber_strength = Float32(_read_scalar(cur).num)
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
        elif key == "timestep_noising_weight":
            cfg.timestep_noising_weight = Float32(_read_scalar(cur).num)
        elif key == "timestep_noising_bias":
            cfg.timestep_noising_bias = Float32(_read_scalar(cur).num)
        # ── Wave 2B: caption dropout ──
        elif key == "caption_dropout_prob":
            cfg.caption_dropout_prob = Float32(_read_scalar(cur).num)
        # ── Wave 2B: noise modifiers ──
        elif key == "offset_noise_weight":
            cfg.offset_noise_weight = Float32(_read_scalar(cur).num)
        elif key == "offset_noise_prob":
            cfg.offset_noise_prob = Float32(_read_scalar(cur).num)
        elif key == "input_perturbation":
            cfg.input_perturbation = Float32(_read_scalar(cur).num)
        elif key == "multires_iterations":
            cfg.multires_iterations = Int(_read_scalar(cur).num)
        elif key == "multires_discount":
            cfg.multires_discount = Float32(_read_scalar(cur).num)
        # ── Wave 2B: gradient accumulation ──
        elif key == "grad_accum_steps":
            cfg.grad_accum_steps = Int(_read_scalar(cur).num)
        # ── Wave 2B: EMA ──
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
        # ── Wave 2B: adapter algo selector ──
        elif key == "algo" or key == "adapter_algo":
            var sc = _read_scalar(cur)
            if not sc.is_string:
                raise Error("JSON config: algo must be a string")
            cfg.adapter_algo = _adapter_algo_int(sc.s)
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

    return cfg^
