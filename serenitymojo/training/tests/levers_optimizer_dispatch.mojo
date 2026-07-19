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
    TRAIN_OPTIMIZER_ADAMW_8BIT, TRAIN_OPTIMIZER_AUTOMAGIC3,
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


# --- B1 bf16-stochastic-rounding "moves not stalled" gate helpers ---
# Realistic 2D LoRA shape [_SR_R, _SR_C]; init weight 0.1 (bf16 ULP there
# ~4.88e-4). Constant +grad -> consistent update sign. We keep the run SHORT
# (just past the H=8 warmup) so the self-adapting lr only rises modestly and the
# per-step update stays sub-ULP — isolating the WRITEBACK question (does a
# sub-ULP update accumulate?) from the controller's lr dynamics.
comptime _SR_R = 16
comptime _SR_C = 16
comptime _SR_N = _SR_R * _SR_C
comptime _SR_W0 = Float32(0.1)
comptime _SR_GRAD = Float32(0.05)  # constant POSITIVE grad -> p descends


def _sr_adapter() -> LoraAdapter:
    # A = [rank=_SR_R, in_f=_SR_C] all 0.1; B = [out_f=_SR_C, rank=_SR_R] zero.
    return LoraAdapter(
        _const_list(_SR_N, _SR_W0),       # a (f32 in; ctor casts to bf16)
        _const_list(_SR_N, Float32(0.0)), # b
        _SR_R, _SR_C, _SR_C, Float32(1.0),
        _const_list(_SR_N, Float32(0.0)),
        _const_list(_SR_N, Float32(0.0)),
        _const_list(_SR_N, Float32(0.0)),
        _const_list(_SR_N, Float32(0.0)),
    )


def _sr_grads_a() -> List[List[Float32]]:
    var gs = List[List[Float32]]()
    gs.append(_const_list(_SR_N, _SR_GRAD))  # constant +grad -> consistent sign
    return gs^


def _sr_grads_b() -> List[List[Float32]]:
    var gs = List[List[Float32]]()
    gs.append(_const_list(_SR_N, _SR_GRAD))
    return gs^


def _mean_bf16(p: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(p)):
        s += p[i].cast[DType.float32]()
    return s / Float32(len(p))


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

    # 7. AUTOMAGIC3 routes to the ai-toolkit adaptive optimizer; lazy state (2
    # factored Automagic3State per adapter); ONE shared adaptive lr in
    # auto3_ctl seeded from cfg.lr; params move; the scheduler step_lr is
    # IGNORED; during the H=8 warmup the controller abstains so lr stays at the
    # seed; validate passes.
    var cfg_am = TrainConfig.default()
    cfg_am.optimizer = TRAIN_OPTIMIZER_AUTOMAGIC3
    cfg_am.lr = Float32(1.0e-4)
    _check(String("am-active"), levers_optimizer_active(cfg_am))
    levers_optimizer_validate(cfg_am, String("dispatch gate"))
    var st_am = LeversOptimizerState()
    var ads_am = _make_set()
    var am_a_before = _absum_bf16(ads_am[0].a)
    # step_lr deliberately bogus (1.0) — automagic3 must ignore it and use the
    # ctl's seeded cfg.lr (1e-4), so params barely move (not by 1.0).
    levers_optimizer_step_host(
        cfg_am, ads_am, _make_grads(), _make_grads_b(), 1, Float32(1.0),
        0, 2, st_am,
    )
    _check(String("am-state-created"), st_am.initialized)
    _check(String("am-kind"), st_am.kind == TRAIN_OPTIMIZER_AUTOMAGIC3)
    _check(String("am-states-2-per-adapter"), len(st_am.auto3) == 4)
    _check(
        String("am-no-other-states"),
        len(st_am.ada) == 0 and len(st_am.sf) == 0 and len(st_am.a8) == 0,
    )
    _check(String("am-step-counted"), st_am.auto3[0].step == 1)
    _check(String("am-A-factored"), st_am.auto3[0].factored)
    _check(
        String("am-A-shape"),
        st_am.auto3[0].rows == RANK and st_am.auto3[0].cols == IN_F,
    )
    _check(
        String("am-B-shape"),
        st_am.auto3[1].rows == OUT_F and st_am.auto3[1].cols == RANK,
    )
    _check(String("am-lr-seeded-from-cfg"), st_am.auto3_ctl.lr == Float64(cfg_am.lr))
    _check(String("am-ctl-initialized"), st_am.auto3_ctl.initialized)
    # H=8 warmup: window not full after 1 step -> no vote -> lr unchanged
    # (still exactly the seeded cfg.lr).
    _check(
        String("am-warmup-lr-unchanged"),
        st_am.auto3_ctl.lr == Float64(cfg_am.lr),
    )
    _check(String("am-params-moved"), _absum_bf16(ads_am[0].a) != am_a_before)
    # step 2 advances the per-param step counter (B side too); still in warmup.
    levers_optimizer_step_host(
        cfg_am, ads_am, _make_grads(), _make_grads_b(), 2, Float32(1.0),
        0, 2, st_am,
    )
    _check(String("am-step-2"), st_am.auto3[3].step == 2)
    _check(
        String("am-warmup-lr-still-unchanged"),
        st_am.auto3_ctl.lr == Float64(cfg_am.lr),
    )

    # Run THROUGH the H=8 warmup (steps 3..9). The _make_grads/_make_grads_b
    # ramps carry a CONSTANT sign per element across every call, so each
    # element's update sign is consistent step-to-step -> once the 8-plane
    # window fills, every element votes "all agree" (step too small) -> the
    # GROUP-pooled vote drives the single shared lr UP. This proves the levers
    # path reaches automagic3's controller (not a silent AdamW fallback) and
    # nudges the lr exactly once per group per step.
    var lr_warmup_end = st_am.auto3_ctl.lr  # == seed (still warmup after step 2)
    for k in range(3, 10):  # steps 3,4,5,6,7,8,9
        levers_optimizer_step_host(
            cfg_am, ads_am, _make_grads(), _make_grads_b(), k, Float32(1.0),
            0, 2, st_am,
        )
    # After step 9 (>H=8), the window has been full for >=2 steps -> the lr has
    # been nudged and, with all-agree votes, has RISEN above the seed.
    _check(String("am-step-9-counted"), st_am.auto3[0].step == 9)
    _check(
        String("am-lr-nudged-after-warmup"),
        st_am.auto3_ctl.lr > lr_warmup_end,
    )
    _check(
        String("am-lr-rose-above-seed"),
        st_am.auto3_ctl.lr > Float64(cfg_am.lr),
    )
    # The single group lr stays in the overflow-guard band (never a control
    # rail), confirming the exp(clamp(signal,±1)) nudge is bounded.
    _check(
        String("am-lr-bounded"),
        st_am.auto3_ctl.lr > Float64(1.0e-30)
        and st_am.auto3_ctl.lr < Float64(1.0e3),
    )

    # 8. B1 — bf16 STOCHASTIC ROUNDING: the writeback must KEEP THE WEIGHT
    # MOVING when the per-step update is sub-bf16-ULP. Regime: a bf16 weight at
    # 0.1 (bf16 ULP there ~4.9e-4), lr=1e-4, constant +grad -> update ~+1 after
    # the trust-region clip -> per-step |Δw| ~1e-4 < ULP. Under plain RNE the
    # weight would round to no change every step and STALL at 0.1; under SR it
    # descends. Run the REAL levers automagic3 dispatch for 60 steps and assert
    # the mean A weight moved DOWN off 0.1 by more than one bf16 ULP.
    var cfg_sr = TrainConfig.default()
    cfg_sr.optimizer = TRAIN_OPTIMIZER_AUTOMAGIC3
    # lr 1e-4: each step's |Δw| ~ lr*|update| ~ 1e-4 is well under HALF a bf16
    # ULP at 0.1 (~2.44e-4), so plain RNE rounds every step back to the init and
    # STALLS; SR rounds DOWN with probability ~(step/ULP)~0.2 per step, so over
    # the 8 abstaining steps it accumulates ~1-2 ULP of real descent.
    cfg_sr.lr = Float32(1.0e-4)
    cfg_sr.seed = UInt64(1234)
    var st_sr = LeversOptimizerState()
    var ads_sr = List[LoraAdapter]()
    ads_sr.append(_sr_adapter())
    var w0 = _mean_bf16(ads_sr[0].a)
    _check(String("sr-init-is-0p1"),
           w0 > Float32(0.0999) and w0 < Float32(0.1001))
    # PROVE plain RNE stalls: subtracting one sub-half-ULP step from the ACTUAL
    # bf16 init value rounds straight back to it (no move).
    var rne_one_step = BFloat16(w0 - Float32(1.0e-4)).cast[DType.float32]()
    _check(String("sr-rne-would-stall"), rne_one_step == w0)
    # Run 7 steps (< H=8): the controller ABSTAINS entirely (its window fills
    # and first votes ON step 8), so the lr stays pinned at the 1e-4 seed and
    # the ONLY thing that can move the bf16 weight off 0.1 is the SR writeback
    # accumulating the sub-ULP steps. (Deliberately isolates the WRITEBACK from
    # the lr controller — past H the constant-sign votes amplify the lr.)
    var n_sr_steps = 7
    for k in range(1, n_sr_steps + 1):
        levers_optimizer_step_host(
            cfg_sr, ads_sr, _sr_grads_a(), _sr_grads_b(), k, Float32(1.0e-4),
            0, 1, st_sr,
        )
    var wN = _mean_bf16(ads_sr[0].a)
    var ulp_01 = Float32(4.88e-4)
    var moved = (w0 - wN) if (w0 - wN) >= Float32(0.0) else (wN - w0)
    print("  INFO  sr w0=", w0, " wN=", wN, " |Δ|=", moved,
          " lr=", st_sr.auto3_ctl.lr)
    # lr stayed at the seed (controller abstained through the H=8 warmup) — so
    # any movement is purely the SR writeback, not the lr rising.
    _check(String("sr-lr-stayed-at-seed"),
           st_sr.auto3_ctl.lr == Float64(cfg_sr.lr))
    # MOVED: at least one bf16 ULP off the init (RNE would still read exactly
    # w0). The mean over 256 elements, each independently SR-dithered, lands
    # cleanly above a single ULP.
    _check(String("sr-weight-moved"), moved >= ulp_01)
    # RIGHT DIRECTION: +grad -> p -= lr*update -> w descends below init.
    _check(String("sr-weight-descended"), wN < w0)
    # MAGNITUDE IN RANGE: neither stalled (>= 1 ULP) nor a blow-up (< 0.02).
    _check(String("sr-weight-magnitude-sane"),
           moved >= ulp_01 and moved < Float32(0.02))

    print("ALL PASS — levers optimizer dispatch OK")
