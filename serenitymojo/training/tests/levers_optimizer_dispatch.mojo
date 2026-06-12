# training/tests/levers_optimizer_dispatch.mojo — T1.C dispatch unit gate.
#
# Proves the levers optimizer seam routes correctly (training/levers.mojo
# T1.C section), HOST-ONLY (levers_optimizer_step_host — no DeviceContext):
#   1. default TrainConfig (optimizer=ADAMW) => levers_optimizer_active is
#      False (the trainer's seam routes AROUND levers entirely, C13) and a
#      forced step call fails loud.
#   2. optimizer=ADAFACTOR routes to adafactor: state lazily created
#      (2 AdafactorState per adapter, A then B), params actually move.
#   3. optimizer=SCHEDULE_FREE_ADAMW routes to schedule-free: per-param
#      states + optimizer-level k advance; eval/train save bracket flips
#      train_mode (and only for SF).
#   4. resume guard: first call at k != 1 fails loud (no state sidecar yet).
#   5. levers_optimizer_validate rejects an unsupported tag (CAME).
#   6. T2.A: optimizer=ADAMW_8BIT routes to the bnb block-wise 8-bit AdamW
#      (training/adamw8bit.mojo): per-matrix Adam8bitState + the two 256-entry
#      dynamic LUTs lazily created, params actually move, per-state step
#      counters advance, validate passes.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/training/tests/levers_optimizer_dispatch.mojo \
#     -o /tmp/levers_optimizer_dispatch && /tmp/levers_optimizer_dispatch

from serenitymojo.training.levers import (
    LeversOptimizerState, levers_optimizer_active, levers_optimizer_step_host,
    levers_optimizer_validate, levers_optimizer_eval_for_save,
    levers_optimizer_train_after_save,
)
from serenitymojo.training.train_config import (
    TrainConfig, TRAIN_OPTIMIZER_ADAMW, TRAIN_OPTIMIZER_ADAFACTOR,
    TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW, TRAIN_OPTIMIZER_CAME,
    TRAIN_OPTIMIZER_ADAMW_8BIT,
)
from serenitymojo.training.train_step import LoraAdapter


comptime RANK = 2
comptime IN_F = 4
comptime OUT_F = 3


def _check(name: String, cond: Bool) raises:
    if cond:
        print("  PASS ", name)
    else:
        raise Error(String("FAIL ") + name)


def _const_list(n: Int, v: Float32) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for _ in range(n):
        out.append(v)
    return out^


def _ramp_list(n: Int, scale: Float32) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(scale * Float32(i + 1))
    return out^


def _make_adapter() -> LoraAdapter:
    # A [RANK, IN_F] nonzero, B [OUT_F, RANK] zero (the LoRA init shape).
    return LoraAdapter(
        _ramp_list(RANK * IN_F, Float32(0.01)),
        _const_list(OUT_F * RANK, Float32(0.0)),
        RANK, IN_F, OUT_F, Float32(1.0),
        _const_list(RANK * IN_F, Float32(0.0)),
        _const_list(RANK * IN_F, Float32(0.0)),
        _const_list(OUT_F * RANK, Float32(0.0)),
        _const_list(OUT_F * RANK, Float32(0.0)),
    )


def _make_set() -> List[LoraAdapter]:
    var ads = List[LoraAdapter]()
    ads.append(_make_adapter())
    ads.append(_make_adapter())
    return ads^


def _make_grads() -> List[List[Float32]]:
    var gs = List[List[Float32]]()
    gs.append(_ramp_list(RANK * IN_F, Float32(0.1)))
    gs.append(_ramp_list(RANK * IN_F, Float32(-0.05)))
    return gs^


def _make_grads_b() -> List[List[Float32]]:
    var gs = List[List[Float32]]()
    gs.append(_ramp_list(OUT_F * RANK, Float32(0.2)))
    gs.append(_ramp_list(OUT_F * RANK, Float32(-0.07)))
    return gs^


def _absum_bf16(p: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(p)):
        var v = p[i].cast[DType.float32]()
        s += v if v >= Float32(0.0) else -v
    return s


def main() raises:
    print("== levers optimizer dispatch gate ==")

    # 1. default routes around levers entirely.
    var cfg = TrainConfig.default()
    _check(String("default-not-active"), not levers_optimizer_active(cfg))
    _check(
        String("default-tag-is-adamw"),
        cfg.optimizer == TRAIN_OPTIMIZER_ADAMW,
    )
    var st0 = LeversOptimizerState()
    var ads0 = _make_set()
    var raised_default = False
    try:
        levers_optimizer_step_host(
            cfg, ads0, _make_grads(), _make_grads_b(), 1, Float32(1e-3),
            0, 2, st0,
        )
    except _e:
        raised_default = True
    _check(String("default-forced-step-fails-loud"), raised_default)
    _check(String("default-no-state"), not st0.initialized)

    # 2. ADAFACTOR routes to adafactor; state created; params move.
    cfg.optimizer = TRAIN_OPTIMIZER_ADAFACTOR
    _check(String("adafactor-active"), levers_optimizer_active(cfg))
    var st_a = LeversOptimizerState()
    var ads_a = _make_set()
    var b_before = _absum_bf16(ads_a[0].b)
    levers_optimizer_step_host(
        cfg, ads_a, _make_grads(), _make_grads_b(), 1, Float32(1e-3),
        0, 2, st_a,
    )
    _check(String("adafactor-state-created"), st_a.initialized)
    _check(
        String("adafactor-kind"), st_a.kind == TRAIN_OPTIMIZER_ADAFACTOR
    )
    _check(String("adafactor-ada-states-2-per-adapter"), len(st_a.ada) == 4)
    _check(String("adafactor-no-sf-states"), len(st_a.sf) == 0)
    _check(String("adafactor-step-counted"), st_a.ada[0].step == 1)
    _check(
        String("adafactor-A-shape"),
        st_a.ada[0].rows == RANK and st_a.ada[0].cols == IN_F,
    )
    _check(
        String("adafactor-B-shape"),
        st_a.ada[1].rows == OUT_F and st_a.ada[1].cols == RANK,
    )
    var b_after = _absum_bf16(ads_a[0].b)
    _check(String("adafactor-params-moved"), b_after != b_before)
    # second step advances per-matrix step counters
    levers_optimizer_step_host(
        cfg, ads_a, _make_grads(), _make_grads_b(), 2, Float32(1e-3),
        0, 2, st_a,
    )
    _check(String("adafactor-step-2"), st_a.ada[3].step == 2)

    # 3. SCHEDULE_FREE_ADAMW routes to schedule-free; ctl.k advances; the
    # eval/train save bracket flips train_mode only for SF.
    cfg.optimizer = TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW
    cfg.optimizer_warmup_steps = 5
    _check(String("sf-active"), levers_optimizer_active(cfg))
    var st_s = LeversOptimizerState()
    var ads_s = _make_set()
    levers_optimizer_step_host(
        cfg, ads_s, _make_grads(), _make_grads_b(), 1, Float32(1e-3),
        0, 2, st_s,
    )
    _check(String("sf-state-created"), st_s.initialized)
    _check(String("sf-states-2-per-adapter"), len(st_s.sf) == 4)
    _check(String("sf-no-ada-states"), len(st_s.ada) == 0)
    _check(String("sf-ctl-k-advanced"), st_s.sf_ctl.k == 1)
    _check(String("sf-b-moved"), _absum_bf16(ads_s[1].b) > Float32(0.0))
    _check(String("sf-train-mode-on"), st_s.sf_ctl.train_mode)
    levers_optimizer_eval_for_save(cfg, st_s)
    _check(String("sf-eval-flips-train-mode"), not st_s.sf_ctl.train_mode)
    levers_optimizer_train_after_save(cfg, st_s)
    _check(String("sf-train-restores-train-mode"), st_s.sf_ctl.train_mode)
    # the bracket is a no-op for non-SF state (default cfg + adafactor state)
    var cfg_ada = TrainConfig.default()
    cfg_ada.optimizer = TRAIN_OPTIMIZER_ADAFACTOR
    levers_optimizer_eval_for_save(cfg_ada, st_a)
    _check(String("bracket-noop-for-adafactor"), st_a.sf_ctl.train_mode)

    # 4. resume guard: first call at k != 1 fails loud.
    var st_r = LeversOptimizerState()
    var ads_r = _make_set()
    var raised_resume = False
    try:
        levers_optimizer_step_host(
            cfg, ads_r, _make_grads(), _make_grads_b(), 5, Float32(1e-3),
            0, 2, st_r,
        )
    except _e2:
        raised_resume = True
    _check(String("resume-at-k5-fails-loud"), raised_resume)

    # 5. validate rejects unsupported tags with the supported list.
    var cfg_bad = TrainConfig.default()
    cfg_bad.optimizer = TRAIN_OPTIMIZER_CAME
    var raised_came = False
    try:
        levers_optimizer_validate(cfg_bad, String("dispatch gate"))
    except _e3:
        raised_came = True
    _check(String("validate-rejects-came"), raised_came)
    levers_optimizer_validate(cfg, String("dispatch gate"))  # SF passes

    # 6. T2.A ADAMW_8BIT routes to the bnb 8-bit AdamW; lazy state + LUTs;
    # params move; per-state step counters advance; validate passes.
    var cfg8 = TrainConfig.default()
    cfg8.optimizer = TRAIN_OPTIMIZER_ADAMW_8BIT
    _check(String("a8-active"), levers_optimizer_active(cfg8))
    levers_optimizer_validate(cfg8, String("dispatch gate"))
    var st_8 = LeversOptimizerState()
    var ads_8 = _make_set()
    var a8_before = _absum_bf16(ads_8[0].a)
    levers_optimizer_step_host(
        cfg8, ads_8, _make_grads(), _make_grads_b(), 1, Float32(1e-3),
        0, 2, st_8,
    )
    _check(String("a8-state-created"), st_8.initialized)
    _check(String("a8-kind"), st_8.kind == TRAIN_OPTIMIZER_ADAMW_8BIT)
    _check(String("a8-states-2-per-adapter"), len(st_8.a8) == 4)
    _check(
        String("a8-no-other-states"),
        len(st_8.ada) == 0 and len(st_8.sf) == 0,
    )
    _check(
        String("a8-luts-256"),
        len(st_8.a8_qmap_signed) == 256 and len(st_8.a8_qmap_unsigned) == 256,
    )
    _check(String("a8-step-counted"), st_8.a8[0].step == 1)
    _check(String("a8-params-moved"), _absum_bf16(ads_8[0].a) != a8_before)
    levers_optimizer_step_host(
        cfg8, ads_8, _make_grads(), _make_grads_b(), 2, Float32(1e-3),
        0, 2, st_8,
    )
    _check(String("a8-step-2"), st_8.a8[3].step == 2)

    print("ALL PASS — levers optimizer dispatch OK")
