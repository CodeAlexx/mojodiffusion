# optimizer_fixed_update_smoke.mojo — deterministic optimizer update gates.
#
# Covers the optimizer identifiers actually used by the local target OneTrainer
# presets/configs: ADAMW (including missing/null default) and ADAFACTOR. Also
# pins the cosine LR scalar used by target schedules. These are small fixed-input
# gates, not train-loop integration.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/optimizer_fixed_update_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.fused_adamw_multitensor import fused_adamw_step, TArc
from serenitymojo.training.opt_adafactor import (
    adafactor_step_factored,
    adafactor_eps_param,
)
from serenitymojo.training.lr_schedule import LR_COSINE, cosine_lr, lr_for_step


comptime N = 4
comptime ADAM_LR = Float32(0.003)
comptime ADAM_BETA1 = Float32(0.9)
comptime ADAM_BETA2 = Float32(0.999)
comptime ADAM_EPS = Float32(1.0e-8)
comptime ADAM_WD = Float32(0.01)
comptime ORDER_LR = Float32(0.1)
comptime ORDER_WD = Float32(0.5)
comptime ORDER_TOL = Float32(1.0e-6)
comptime UPDATE_TOL = Float32(2.0e-6)
comptime STATE_TOL = Float32(2.0e-6)


def _adam_p0() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.25)
    out.append(-0.5)
    out.append(0.75)
    out.append(-1.0)
    return out^


def _adam_g1() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.10)
    out.append(-0.20)
    out.append(0.30)
    out.append(-0.40)
    return out^


def _adam_g2() -> List[Float32]:
    var out = List[Float32]()
    out.append(-0.05)
    out.append(0.15)
    out.append(-0.25)
    out.append(0.35)
    return out^


def _adam_order_p0() -> List[Float32]:
    var out = List[Float32]()
    out.append(1.0)
    out.append(-2.0)
    out.append(0.5)
    out.append(-0.25)
    return out^


def _adam_order_g() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.10)
    out.append(-0.20)
    out.append(0.30)
    out.append(-0.40)
    return out^


def _adam_m0() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.01)
    out.append(-0.02)
    out.append(0.03)
    out.append(-0.04)
    return out^


def _adam_v0() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.001)
    out.append(0.002)
    out.append(0.003)
    out.append(0.004)
    return out^


def _expected_adam_p1() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.2496300042)
    out.append(-0.4994748831)
    out.append(0.7493557930)
    out.append(-0.9992555976)
    return out^


def _expected_adam_order_p1() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.8499999642372131)
    out.append(-1.7999999523162842)
    out.append(0.375)
    out.append(-0.13749998807907104)
    return out^


def _wrong_post_adam_decay_order_p1() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.8549999594688416)
    out.append(-1.8049999475479126)
    out.append(0.3799999952316284)
    out.append(-0.14249999821186066)
    return out^


def _expected_adam_m1() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.01900000125)
    out.append(-0.03800000250)
    out.append(0.05700000003)
    out.append(-0.07600000501)
    return out^


def _expected_adam_v1() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.0010090000)
    out.append(0.0020380002)
    out.append(0.0030869999)
    out.append(0.0041559999)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def _assert_close_list(
    name: String, actual: List[Float32], expected: List[Float32], tol: Float32
) raises:
    if len(actual) != len(expected):
        print(name, " length mismatch actual=", len(actual), " expected=", len(expected))
        raise Error(name + " length mismatch")
    for i in range(len(actual)):
        var d = actual[i] - expected[i]
        if d < 0.0:
            d = -d
        if d > tol:
            print(name, " mismatch at ", i, " actual=", actual[i], " expected=", expected[i], " diff=", d)
            raise Error(name + " mismatch")


def _assert_far_list(
    name: String, actual: List[Float32], forbidden: List[Float32], min_max_abs: Float32
) raises:
    if len(actual) != len(forbidden):
        print(name, " length mismatch actual=", len(actual), " forbidden=", len(forbidden))
        raise Error(name + " length mismatch")
    var max_abs = Float32(0.0)
    for i in range(len(actual)):
        var d = actual[i] - forbidden[i]
        if d < 0.0:
            d = -d
        if d > max_abs:
            max_abs = d
    if max_abs < min_max_abs:
        print(name, " max_abs=", max_abs, " min_required=", min_max_abs)
        raise Error(name + " did not distinguish wrong AdamW order")


def _assert_close_scalar(name: String, actual: Float32, expected: Float32, tol: Float32) raises:
    var d = actual - expected
    if d < 0.0:
        d = -d
    if d > tol:
        print(name, " actual=", actual, " expected=", expected, " diff=", d)
        raise Error(name + " mismatch")


def _fused_adamw_single(
    p_src: List[Float32],
    g_src: List[Float32],
    m_src: List[Float32],
    v_src: List[Float32],
    step: Int,
    dtype: STDtype,
    ctx: DeviceContext,
) raises -> List[List[Float32]]:
    return _fused_adamw_single_hparams(
        p_src, g_src, m_src, v_src, step, dtype,
        ADAM_LR, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, ADAM_WD, ctx,
    )


def _fused_adamw_single_hparams(
    p_src: List[Float32],
    g_src: List[Float32],
    m_src: List[Float32],
    v_src: List[Float32],
    step: Int,
    dtype: STDtype,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    ctx: DeviceContext,
) raises -> List[List[Float32]]:
    var ps = List[TArc]()
    var gs = List[TArc]()
    var ms = List[TArc]()
    var vs = List[TArc]()
    ps.append(TArc(Tensor.from_host(p_src.copy(), [N], dtype, ctx)))
    gs.append(TArc(Tensor.from_host(g_src.copy(), [N], dtype, ctx)))
    ms.append(TArc(Tensor.from_host(m_src.copy(), [N], STDtype.F32, ctx)))
    vs.append(TArc(Tensor.from_host(v_src.copy(), [N], STDtype.F32, ctx)))
    fused_adamw_step(
        ps,
        gs,
        ms,
        vs,
        step,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        ctx,
    )
    var out = List[List[Float32]]()
    out.append(ps[0][].to_host(ctx))
    out.append(ms[0][].to_host(ctx))
    out.append(vs[0][].to_host(ctx))
    return out^


def _test_adamw_fixed_update(ctx: DeviceContext) raises:
    var got = _fused_adamw_single(
        _adam_p0(), _adam_g1(), _adam_m0(), _adam_v0(), 3, STDtype.F32, ctx
    )
    _assert_close_list("adamw p after fixed step", got[0], _expected_adam_p1(), UPDATE_TOL)
    _assert_close_list("adamw m after fixed step", got[1], _expected_adam_m1(), STATE_TOL)
    _assert_close_list("adamw v after fixed step", got[2], _expected_adam_v1(), STATE_TOL)


def _test_adamw_order_sensitive_update(ctx: DeviceContext) raises:
    var got = _fused_adamw_single_hparams(
        _adam_order_p0(),
        _adam_order_g(),
        _zeros(N),
        _zeros(N),
        1,
        STDtype.F32,
        ORDER_LR,
        ADAM_BETA1,
        ADAM_BETA2,
        ADAM_EPS,
        ORDER_WD,
        ctx,
    )
    _assert_close_list(
        "adamw order-sensitive p after fixed step",
        got[0],
        _expected_adam_order_p1(),
        ORDER_TOL,
    )
    _assert_far_list(
        "adamw order-sensitive wrong post-decay reference",
        got[0],
        _wrong_post_adam_decay_order_p1(),
        Float32(0.004),
    )


def _test_adamw_resume_equivalence(ctx: DeviceContext) raises:
    var first = _fused_adamw_single(
        _adam_p0(), _adam_g1(), _adam_m0(), _adam_v0(), 3, STDtype.F32, ctx
    )
    var resumed = _fused_adamw_single(first[0], _adam_g2(), first[1], first[2], 4, STDtype.F32, ctx)

    var cont = _fused_adamw_single(
        _adam_p0(), _adam_g1(), _adam_m0(), _adam_v0(), 3, STDtype.F32, ctx
    )
    cont = _fused_adamw_single(cont[0], _adam_g2(), cont[1], cont[2], 4, STDtype.F32, ctx)

    _assert_close_list("adamw resumed p", resumed[0], cont[0], Float32(0.0))
    _assert_close_list("adamw resumed m", resumed[1], cont[1], Float32(0.0))
    _assert_close_list("adamw resumed v", resumed[2], cont[2], Float32(0.0))


def _test_adamw_storage_dtype(ctx: DeviceContext) raises:
    var ps = List[TArc]()
    var gs = List[TArc]()
    var ms = List[TArc]()
    var vs = List[TArc]()
    ps.append(TArc(Tensor.from_host(_adam_p0(), [N], STDtype.BF16, ctx)))
    gs.append(TArc(Tensor.from_host(_adam_g1(), [N], STDtype.BF16, ctx)))
    ms.append(TArc(Tensor.from_host(_zeros(N), [N], STDtype.F32, ctx)))
    vs.append(TArc(Tensor.from_host(_zeros(N), [N], STDtype.F32, ctx)))
    fused_adamw_step(
        ps,
        gs,
        ms,
        vs,
        1,
        Float32(0.05),
        ADAM_BETA1,
        ADAM_BETA2,
        ADAM_EPS,
        Float32(0.0),
        ctx,
    )
    if ps[0][].dtype() != STDtype.BF16 or gs[0][].dtype() != STDtype.BF16:
        raise Error("adamw BF16 param/grad storage dtype changed")
    if ms[0][].dtype() != STDtype.F32 or vs[0][].dtype() != STDtype.F32:
        raise Error("adamw moment storage dtype changed")
    var moved = ps[0][].to_host(ctx)
    var before = Tensor.from_host(_adam_p0(), [N], STDtype.BF16, ctx).to_host(ctx)
    var max_delta = Float32(0.0)
    for i in range(N):
        var d = moved[i] - before[i]
        if d < 0.0:
            d = -d
        if d > max_delta:
            max_delta = d
    if max_delta < Float32(0.01):
        raise Error("adamw BF16 dtype gate did not observe a visible update")


def _af_p0() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.1)
    out.append(-0.2)
    out.append(0.3)
    out.append(-0.4)
    return out^


def _af_g() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.05)
    out.append(-0.10)
    out.append(0.15)
    out.append(-0.20)
    return out^


def _expected_af_p1() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.09922540188)
    out.append(-0.1989045590)
    out.append(0.2989607751)
    out.append(-0.3990201950)
    return out^


def _expected_af_row1() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.0062499996)
    out.append(0.0312500000)
    return out^


def _expected_af_col1() -> List[Float32]:
    var out = List[Float32]()
    out.append(0.0125000002)
    out.append(0.0249999985)
    return out^


def _af_step_once(
    p_src: List[Float32], row_src: List[Float32], col_src: List[Float32], step: Int
) raises -> List[List[Float32]]:
    var p = p_src.copy()
    var row = row_src.copy()
    var col = col_src.copy()
    adafactor_step_factored(
        p,
        _af_g(),
        row,
        col,
        2,
        2,
        step,
        Float32(1.0e-3),
        adafactor_eps_param(Float32(1.0e-3)),
        Float32(0.0),
        False,
    )
    var out = List[List[Float32]]()
    out.append(p.copy())
    out.append(row.copy())
    out.append(col.copy())
    return out^


def _test_adafactor_fixed_update() raises:
    var got = _af_step_once(_af_p0(), _zeros(2), _zeros(2), 1)
    _assert_close_list("adafactor p after fixed step", got[0], _expected_af_p1(), UPDATE_TOL)
    _assert_close_list("adafactor row state", got[1], _expected_af_row1(), STATE_TOL)
    _assert_close_list("adafactor col state", got[2], _expected_af_col1(), STATE_TOL)


def _test_adafactor_resume_equivalence() raises:
    var first = _af_step_once(_af_p0(), _zeros(2), _zeros(2), 1)
    var resumed = _af_step_once(first[0], first[1], first[2], 2)
    var cont = _af_step_once(_af_p0(), _zeros(2), _zeros(2), 1)
    cont = _af_step_once(cont[0], cont[1], cont[2], 2)
    _assert_close_list("adafactor resumed p", resumed[0], cont[0], Float32(0.0))
    _assert_close_list("adafactor resumed row", resumed[1], cont[1], Float32(0.0))
    _assert_close_list("adafactor resumed col", resumed[2], cont[2], Float32(0.0))


def _test_cosine_lr_fixed_inputs() raises:
    var base = Float32(0.01)
    _assert_close_scalar("cosine warmup step0", cosine_lr(base, 0, 10, 2, 0.1), 0.005, Float32(1.0e-7))
    _assert_close_scalar("cosine warmup step1", cosine_lr(base, 1, 10, 2, 0.1), 0.010, Float32(1.0e-7))
    _assert_close_scalar("cosine start", cosine_lr(base, 2, 10, 2, 0.1), 0.010, Float32(1.0e-7))
    _assert_close_scalar("cosine midpoint", cosine_lr(base, 6, 10, 2, 0.1), 0.0055, Float32(1.0e-7))
    _assert_close_scalar("cosine end", cosine_lr(base, 10, 10, 2, 0.1), 0.001, Float32(1.0e-7))
    _assert_close_scalar(
        "cosine dispatch midpoint",
        lr_for_step(base, 6, 2, 10, LR_COSINE, 0.1, 1.0, 2.0),
        0.0055,
        Float32(1.0e-7),
    )


def main() raises:
    var ctx = DeviceContext()
    _test_adamw_fixed_update(ctx)
    _test_adamw_order_sensitive_update(ctx)
    _test_adamw_resume_equivalence(ctx)
    _test_adamw_storage_dtype(ctx)
    _test_adafactor_fixed_update()
    _test_adafactor_resume_equivalence()
    _test_cosine_lr_fixed_inputs()
    print("PASS: fixed optimizer update gates passed")
