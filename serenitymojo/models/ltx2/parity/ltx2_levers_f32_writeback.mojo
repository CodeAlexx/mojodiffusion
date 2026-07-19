# ltx2_levers_f32_writeback.mojo — F32-MASTER levers optimizer no-rounding gate.
#
# The existing optimizer parity gates validate the *_step fns (F32 params) vs
# their torch/bnb/ai-toolkit oracles — but ALL of them drove the bf16-master
# wrapper (cast -> step -> RNE/SR bf16 writeback). None asserts the ONE property
# the F32-master path (MJ-1112, levers_optimizer_step_host_f32) exists for: the
# step introduces NO bf16 rounding anywhere. This gate seeds F32 params carrying
# sub-bf16 mantissa bits, runs ONE step of each of the 4 levers optimizers, and
# asserts the OUTPUT params STILL carry sub-bf16 bits (>90%). A bf16 hop anywhere
# would collapse that fraction to 0 (every value bf16-representable).
#
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/ltx2/parity/ltx2_levers_f32_writeback.mojo \
#     -o /tmp/ltx2_levers_f32_writeback && /tmp/ltx2_levers_f32_writeback

from std.math import sin, cos

from serenitymojo.training.levers import (
    F32AdapterView, LeversOptimizerState, levers_optimizer_step_host_f32,
)
from serenitymojo.training.train_config import (
    TrainConfig,
    TRAIN_OPTIMIZER_ADAFACTOR, TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW,
    TRAIN_OPTIMIZER_ADAMW_8BIT, TRAIN_OPTIMIZER_AUTOMAGIC3,
)


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


# NON-degenerate sinusoidal fill (never modular) — generic F32 values almost all
# carry sub-bf16 mantissa bits.
def _sinfill(n: Int, phase: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var fi = Float32(i)
        out.append(Float32(0.03) * (sin(Float32(0.7) * fi + phase)
                   + Float32(0.3) * cos(Float32(1.3) * fi)))
    return out^


def _frac_sub_bf16(v: List[Float32]) -> Float32:
    var cnt = 0
    for i in range(len(v)):
        if v[i] != Float32(BFloat16(v[i])):
            cnt += 1
    return Float32(cnt) / Float32(len(v))


def _changed(a: List[Float32], b: List[Float32]) -> Bool:
    for i in range(len(a)):
        if a[i] != b[i]:
            return True
    return False


def _test_opt(opt_tag: Int, name: String) raises:
    comptime R = 4
    comptime IN = 16
    comptime OUT = 16
    var n_ad = 3
    var cfg = TrainConfig.default()
    cfg.optimizer = opt_tag
    cfg.lr = Float32(1.0e-3)          # sizeable so params move
    cfg.weight_decay = Float32(0.0)   # isolate the step's own update

    var views = List[F32AdapterView]()
    var seed_a = List[List[Float32]]()
    var seed_b = List[List[Float32]]()
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    for j in range(n_ad):
        var fj = Float32(j) * Float32(0.41)
        var a0 = _sinfill(R * IN, fj + Float32(0.10))
        var b0 = _sinfill(OUT * R, fj + Float32(0.50))
        seed_a.append(a0.copy())
        seed_b.append(b0.copy())
        views.append(F32AdapterView(a0^, b0^, R, IN, OUT))
        d_a.append(_sinfill(R * IN, fj + Float32(0.90)))
        d_b.append(_sinfill(OUT * R, fj + Float32(1.30)))
    # seeds MUST be sub-bf16 or the tripwire is meaningless
    _require(_frac_sub_bf16(seed_a[0]) > Float32(0.9),
             name + ": seed A not sub-bf16 (test data invalid)")

    var st = LeversOptimizerState()
    levers_optimizer_step_host_f32(
        cfg, views, d_a, d_b, 1, Float32(1.0e-3), 0, n_ad, st)

    for j in range(n_ad):
        _require(_changed(views[j].a, seed_a[j]) or _changed(views[j].b, seed_b[j]),
                 name + ": step was a no-op (adapter " + String(j) + ")")
        var fa = _frac_sub_bf16(views[j].a)
        var fb = _frac_sub_bf16(views[j].b)
        _require(fa > Float32(0.9),
                 name + ": A lost sub-bf16 bits (bf16 hop!) frac=" + String(fa))
        _require(fb > Float32(0.9),
                 name + ": B lost sub-bf16 bits (bf16 hop!) frac=" + String(fb))
    print("  PASS", name, " (F32-in == F32-out, no bf16 rounding)")


def main() raises:
    print("=== LTX2 F32-master levers optimizer no-rounding gate ===")
    _test_opt(TRAIN_OPTIMIZER_ADAFACTOR, String("adafactor"))
    _test_opt(TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW, String("schedule_free"))
    _test_opt(TRAIN_OPTIMIZER_ADAMW_8BIT, String("adamw_8bit"))
    _test_opt(TRAIN_OPTIMIZER_AUTOMAGIC3, String("automagic3"))
    print("ALL PASS")
