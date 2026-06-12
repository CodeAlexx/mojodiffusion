# training/levers.mojo — the ONE shared runtime-config lever module
# (TIER1_PARITY_CAMPAIGN_2026-06-11.md MODULARITY DIRECTIVE: one shared
# runtime-config module, each trainer wires ONE call — no per-trainer
# comptime forks).
#
# Phase T1.A: loss-fn selection (mse | huber | smooth_l1, torch semantics)
# + flow-match min-SNR-γ weighting, dispatched at RUNTIME off TrainConfig
# fields. Math lives in ops/loss_fns.mojo (torch-oracle gated by
# ops/tests/loss_fns_parity.mojo); this module is dispatch only.
#
# DEFAULT-OFF CONTRACT (C13): with the config keys absent —
#   loss_fn == LOSS_FN_MSE, min_snr_gamma_flow == 0.0 —
# levers_loss_grad IS mse_loss_grad, formula-identical to the trainers'
# existing inline MSE blocks:
#   loss   = F32( Σ F64(d_i)^2 / N ),   d_i = pred_i - target_i   (F32)
#   d_pred = (2/N) * d_i                                          (F32)
# (same element order, same F64 reduction, same F32 rounding points), so the
# default path moves no anchors.
#
# ── min-SNR-γ DIVISOR DIFFERENCE (two SEPARATE config keys) ──────────────────
# * cfg.min_snr_gamma (klein, Wave 2A): consumed via training/loss_weight.mojo
#   apply_loss_weight(..., is_v_prediction=True) ⇒ w = min(SNR,γ)/(SNR+1)
#   (EDv2 loss_weight.rs v-pred form; off sentinel is γ < 0).
# * cfg.min_snr_gamma_flow (T1.A, this module): SimpleTuner ε-style
#   w = min(SNR,γ)/SNR (ops/loss_fns.mojo min_snr_gamma_weight; off sentinel
#   is 0.0). Keeping them separate leaves klein's existing behavior untouched.
#
# ── Trainer wiring contract ──────────────────────────────────────────────────
# Each loss site computes its host pred/target F32 lists exactly as before
# (zimage: pred = -raw_out), calls levers_loss_grad with THAT STEP's
# flow-match sigma, then chains its own sign/padding:
#   d_out_i = -d_pred_i   (zimage pred = -out chain), seq tail padded 0.
# levers_loss_active() lets a call site keep a literal legacy block for the
# default path where the refactored arithmetic is not bit-provable (zimage
# B2: the old joint 2N-mean F64 accumulation vs per-sample means).
#
# LATER PHASES LAND HERE: EMA (T1.B), optimizer levers (T1.C), caption
# dropout (T1.D), masked loss (T1.E) — one shared entry per lever, each
# trainer wires one call. No per-trainer comptime forks.
#
# Mojo 1.0.0b1. The T1.A/T1.D/T1.E sections are host-only; the T1.C
# optimizer section (end of file) imports GPU host types for the resident
# dev_p sync (see levers_optimizer_sync_resident).

from serenitymojo.ops.loss_fns import (
    LossGrad, mse_loss_grad, huber_loss_grad, smooth_l1_loss_grad,
    min_snr_gamma_weight, mask_weights, masked_mse_loss_grad,
    masked_huber_loss_grad, masked_smooth_l1_loss_grad,
)
from serenitymojo.training.train_config import (
    TrainConfig, LOSS_FN_MSE, LOSS_FN_HUBER, LOSS_FN_SMOOTH_L1,
)


def levers_loss_active(cfg: TrainConfig) -> Bool:
    """True iff any T1.A loss lever deviates from the default MSE path.

    Call sites that must keep a literal legacy block for bit-exact default
    anchors (zimage B2's joint 2N-mean reduction) branch on this; plain
    per-sample sites just call levers_loss_grad unconditionally."""
    return cfg.loss_fn != LOSS_FN_MSE or cfg.min_snr_gamma_flow > Float32(0.0)


def levers_loss_grad(
    pred: List[Float32], target: List[Float32], sigma: Float32,
    cfg: TrainConfig,
) raises -> LossGrad:
    """Runtime-dispatched training loss + d/dpred (mean reduction).

    loss_fn: LOSS_FN_MSE (default; formula-identical to the trainers' inline
    MSE) | LOSS_FN_HUBER (torch huber_loss, cfg.huber_delta) |
    LOSS_FN_SMOOTH_L1 (torch smooth_l1_loss, cfg.smooth_l1_beta).
    min_snr_gamma_flow > 0 additionally scales loss AND d_pred by
    w = min(SNR(sigma), γ)/SNR(sigma)  — the FLOW (ε-style) divisor; see the
    header for how this differs from klein's cfg.min_snr_gamma."""
    var lg: LossGrad
    if cfg.loss_fn == LOSS_FN_HUBER:
        lg = huber_loss_grad(pred, target, cfg.huber_delta)
    elif cfg.loss_fn == LOSS_FN_SMOOTH_L1:
        lg = smooth_l1_loss_grad(pred, target, cfg.smooth_l1_beta)
    elif cfg.loss_fn == LOSS_FN_MSE:
        lg = mse_loss_grad(pred, target)
    else:
        raise Error(
            String("levers_loss_grad: invalid loss_fn tag ")
            + String(cfg.loss_fn)
        )
    if cfg.min_snr_gamma_flow > Float32(0.0):
        var w = min_snr_gamma_weight(sigma, cfg.min_snr_gamma_flow)
        lg.loss = w * lg.loss
        for i in range(len(lg.d_pred)):
            lg.d_pred[i] = w * lg.d_pred[i]
    return lg^


# ── T1.D caption dropout (one shared entry; each trainer wires ONE call) ─────
from serenitymojo.training.caption_dropout import should_drop_caption


def caption_dropout_pick(step: UInt64, seed_base: UInt64, p: Float32) -> Bool:
    """Per-step caption-dropout Bernoulli, the ONE shared seed derivation
    (klein Wave 2B's `seed_base * 31 + step`, distinct from the sigma
    `seed_base + step` and noise `seed_base * 7919 + step` streams).
    p <= 0 never draws (should_drop_caption default-off contract), so the
    default path is byte-identical to no caption dropout."""
    return should_drop_caption(seed_base * UInt64(31) + step, p)


# ── T1.E masked loss (one shared entry; each trainer wires ONE call) ─────────
def levers_masked_active(cfg: TrainConfig) -> Bool:
    """True iff the T1.E masked-loss lever is on (cfg.masked_training).

    DEFAULT-OFF CONTRACT (C13): masked_training defaults False; trainers
    route to levers_masked_loss_grad ONLY when this is True AND the sample
    has a staged mask, so the default path stays the untouched
    levers_loss_grad / legacy block — no anchors move."""
    return cfg.masked_training


def levers_masked_loss_grad(
    pred: List[Float32], target: List[Float32], mask: List[Float32],
    sigma: Float32, cfg: TrainConfig,
) raises -> LossGrad:
    """Masked sibling of levers_loss_grad (signature there is UNCHANGED, so
    existing call sites compile as-is). mask = per-element mask values in
    [0,1] expanded by the trainer to the pred/target element order (per-patch
    value repeated across each token's channel-minor values).

    Semantics (ops/loss_fns.mojo, torch-gated by masked_loss_parity.mojo):
      weights = clamp(mask, cfg.unmasked_weight, 1)   (OneTrainer
        masked_loss.py:11; == SimpleTuner common.py:4694 loss*mask when
        unmasked_weight == 0)
      loss = mean(w * loss_i), d_pred = w * dloss_i/dpred_i / N, both divided
        by mean(w) when cfg.normalize_masked_area_loss (masked_loss.py:15-16)
    then the same loss_fn dispatch + min_snr_gamma_flow scaling as
    levers_loss_grad."""
    var w = mask_weights(mask, cfg.unmasked_weight)
    var norm = cfg.normalize_masked_area_loss
    var lg: LossGrad
    if cfg.loss_fn == LOSS_FN_HUBER:
        lg = masked_huber_loss_grad(pred, target, w, cfg.huber_delta, norm)
    elif cfg.loss_fn == LOSS_FN_SMOOTH_L1:
        lg = masked_smooth_l1_loss_grad(
            pred, target, w, cfg.smooth_l1_beta, norm
        )
    elif cfg.loss_fn == LOSS_FN_MSE:
        lg = masked_mse_loss_grad(pred, target, w, norm)
    else:
        raise Error(
            String("levers_masked_loss_grad: invalid loss_fn tag ")
            + String(cfg.loss_fn)
        )
    if cfg.min_snr_gamma_flow > Float32(0.0):
        var ws = min_snr_gamma_weight(sigma, cfg.min_snr_gamma_flow)
        lg.loss = ws * lg.loss
        for i in range(len(lg.d_pred)):
            lg.d_pred[i] = ws * lg.d_pred[i]
    return lg^


# ══════════════════════════════════════════════════════════════════════════════
# T1.C OPTIMIZER LEVERS (one shared dispatch; each trainer wires ONE call)
#
# Math modules (both reference-parity gated by training/tests/
# optimizer_parity.mojo vs /tmp/optimizer_oracle.safetensors):
#   * training/adafactor.mojo          — torch.optim.Adafactor ("torch-adafactor")
#   * training/adamw_schedulefree.mojo — SimpleTuner AdamWScheduleFreeKahan
#
# DEFAULT-OFF CONTRACT (C13): levers_optimizer_active() is False for
# cfg.optimizer == TRAIN_OPTIMIZER_ADAMW (the config default), and the trainer
# seam is `if levers_optimizer_active(cfg): levers path else: <existing
# literal fused AdamW call>` — the default path never enters this section, so
# existing anchors cannot move.
#
# LR SEMANTICS (read this before adding a model):
#   * ADAFACTOR consumes the TRAINER-SCHEDULED lr (`step_lr` =
#     ot_lr_for_optimizer_step) — SimpleTuner runs its LR scheduler for
#     torch-adafactor; rho_t = min(lr, 1/sqrt(t)) clips it internally.
#   * SCHEDULE_FREE_ADAMW consumes the RAW cfg.lr and IGNORES step_lr — the
#     reference registers override_lr_scheduler=True (optimizer_param.py:
#     249-259): warmup lives INSIDE the optimizer via
#     cfg.optimizer_warmup_steps (:= args.lr_warmup_steps, :1114-1116).
#
# ── RESIDENT dev_p SYNC DECISION (the v2-engine seam) ────────────────────────
# zimage's v2 engine keeps the live bf16 LoRA params in
# LoraAdamWPlainDeviceState.dev_p; the model's device LoRA views are
# SUB-BUFFERS of it ("dev_p is THE live bf16 parameter buffer",
# training/lora_adamw_plain_fused.mojo:285-299). The levers optimizers step
# the HOST a/b mirrors (bf16 -> F32 -> step -> RNE bf16 writeback, the
# _adamw_host_list semantics of training/train_step.mojo:242), so after every
# host step levers_optimizer_sync_resident packs a/b into the pinned
# opt_state.host_p and uploads ONE H2D into dev_p — the exact inverse of the
# resident AdamW's per-step P readback (lora_adamw_plain_fused.mojo:483-502)
# and the same pack layout as lora_adamw_plain_device_state_init
# (:353-405). Correctness over speed: one ~70 MB H2D per step; a fused GPU
# path can land later behind this same dispatch. On the non-resident
# (ZIMAGE_V2_ENGINE=False) path the sync is a harmless dead store — that path
# re-uploads the host set every step via zimage_lora_set_to_device.
#
# SAVE CONTRACT (schedule-free): trainers MUST bracket every weight save /
# validation sample with levers_optimizer_eval_for_save(...) ...
# levers_optimizer_train_after_save(...) — the adamw_schedulefree.mojo header
# documents why (true schedule-free evaluates the x iterate; the reference's
# z is dead so today it only flips train_mode, but the seam keeps trainers
# unchanged if real-z math ever lands). No-op for every other optimizer.
#
# RESUME: levers optimizer state (Adafactor row/col vars, schedule-free
# m/v/k) has NO save/resume sidecar yet — levers_optimizer_step fails loud if
# the first call arrives at k != 1.
# ══════════════════════════════════════════════════════════════════════════════

from std.gpu.host import DeviceContext

from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState,
)
from serenitymojo.training.adafactor import AdafactorState, adafactor_step_2d
from serenitymojo.training.adamw_schedulefree import (
    AdamWScheduleFreeCtl, AdamWScheduleFreeState,
    adamw_schedulefree_adjusted_lr, adamw_schedulefree_step_param,
    adamw_schedulefree_eval, adamw_schedulefree_train,
)
from serenitymojo.training.train_config import (
    TRAIN_OPTIMIZER_ADAMW, TRAIN_OPTIMIZER_ADAFACTOR,
    TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW,
)

# SimpleTuner "torch-adafactor" default_settings (optimizer_param.py:153-162)
# — used when the config carries the 0-valued "unset" sentinels.
comptime _ADAFACTOR_BETA2_DECAY_DEFAULT = Float64(-0.8)
comptime _ADAFACTOR_EPS2_DEFAULT = Float64(1.0e-3)
comptime _ADAFACTOR_D_DEFAULT = Float64(1.0)


def levers_optimizer_active(cfg: TrainConfig) -> Bool:
    """True iff the T1.C optimizer lever deviates from the default AdamW path.

    Trainer seam contract: `if levers_optimizer_active(cfg):
    levers_optimizer_step(...) else: <the existing literal fused AdamW
    call>` — the default path routes AROUND this module entirely (C13)."""
    return cfg.optimizer != TRAIN_OPTIMIZER_ADAMW


def levers_optimizer_validate(cfg: TrainConfig, trainer_name: String) raises:
    """Fail-loud setup-time check for a trainer that wires the levers
    optimizer dispatch. Unsupported tags already failed at config load
    (io/train_config_reader.mojo _optimizer_int); this re-asserts the
    contract for configs built in code."""
    if cfg.optimizer == TRAIN_OPTIMIZER_ADAMW:
        return
    if cfg.optimizer == TRAIN_OPTIMIZER_ADAFACTOR:
        return
    if cfg.optimizer == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW:
        if cfg.optimizer_warmup_steps < 0:
            raise Error(
                trainer_name + String(" requires optimizer_warmup_steps >= 0")
            )
        return
    raise Error(
        trainer_name
        + String(": optimizer tag ")
        + String(cfg.optimizer)
        + String(" has no levers dispatch; supported: ADAMW (default fused")
        + String(" path), ADAFACTOR, SCHEDULE_FREE_ADAMW")
    )


struct LeversOptimizerState(Movable):
    """Per-run levers optimizer state, lazily initialized on the FIRST
    levers_optimizer_step call (so the default AdamW path allocates
    nothing). Layout: 2 entries per adapter in [start,end) — A then B —
    matching LoraAdamWPlainDeviceState.seg_len ordering."""

    var initialized: Bool
    var kind: Int                          # TRAIN_OPTIMIZER_* it was built for
    var start: Int
    var end: Int
    var ada: List[AdafactorState]          # ADAFACTOR: [2*(end-start)]
    var sf: List[AdamWScheduleFreeState]   # SCHEDULE_FREE: [2*(end-start)]
    var sf_ctl: AdamWScheduleFreeCtl       # SCHEDULE_FREE: optimizer-level k

    def __init__(out self):
        self.initialized = False
        self.kind = -1
        self.start = -1
        self.end = -1
        self.ada = List[AdafactorState]()
        self.sf = List[AdamWScheduleFreeState]()
        self.sf_ctl = AdamWScheduleFreeCtl()


def _levers_bf16_to_f32(p: List[BFloat16]) -> List[Float32]:
    var out = List[Float32](capacity=len(p))
    for i in range(len(p)):
        out.append(p[i].cast[DType.float32]())
    return out^


def _levers_writeback_bf16(mut p: List[BFloat16], src: List[Float32]):
    # Plain RNE bf16 writeback — the _adamw_host_list rounding point
    # (training/train_step.mojo:242 `p[i] = BFloat16(pv)`).
    for i in range(len(p)):
        p[i] = BFloat16(src[i])


def _levers_optimizer_lazy_init(
    cfg: TrainConfig,
    adapters: List[LoraAdapter],
    k: Int,
    start: Int,
    end: Int,
    mut st: LeversOptimizerState,
) raises:
    if st.initialized:
        if st.kind != cfg.optimizer or st.start != start or st.end != end:
            raise Error(
                "levers_optimizer_step: state kind/range changed mid-run"
            )
        return
    if k != 1:
        raise Error(
            String("levers_optimizer_step: first call at step ")
            + String(k)
            + String(" — levers optimizer state has no resume sidecar yet;")
            + String(" restart from step 1 or use optimizer=ADAMW to resume")
        )
    if start < 0 or end > len(adapters) or start >= end:
        raise Error("levers_optimizer_step: bad adapter range")
    if cfg.optimizer == TRAIN_OPTIMIZER_ADAFACTOR:
        for i in range(start, end):
            # A is [rank, in_f], B is [out_f, rank] (train_step.mojo
            # LoraAdapter) — row/col factored state per matrix.
            st.ada.append(AdafactorState(adapters[i].rank, adapters[i].in_f))
            st.ada.append(AdafactorState(adapters[i].out_f, adapters[i].rank))
    elif cfg.optimizer == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW:
        for i in range(start, end):
            st.sf.append(AdamWScheduleFreeState(len(adapters[i].a)))
            st.sf.append(AdamWScheduleFreeState(len(adapters[i].b)))
    else:
        raise Error(
            String("levers_optimizer_step: optimizer tag ")
            + String(cfg.optimizer)
            + String(" has no levers dispatch; supported: ADAMW (default")
            + String(" fused path), ADAFACTOR, SCHEDULE_FREE_ADAMW")
        )
    st.kind = cfg.optimizer
    st.start = start
    st.end = end
    st.initialized = True


def levers_optimizer_step_host(
    cfg: TrainConfig,
    mut adapters: List[LoraAdapter],
    d_a: List[List[Float32]],
    d_b: List[List[Float32]],
    k: Int,
    step_lr: Float32,
    start: Int,
    end: Int,
    mut st: LeversOptimizerState,
) raises:
    """HOST half of the levers optimizer step (GPU-free; the dispatch unit
    test runs this directly). One optimizer step over adapters[start:end)
    A+B at trainer step k (1-based, the same t the AdamW path passes).
    bf16 params: cast -> F32 step -> RNE bf16 writeback per matrix.
    d_a/d_b are indexed by ABSOLUTE adapter index, like the fused AdamW."""
    _levers_optimizer_lazy_init(cfg, adapters, k, start, end, st)
    if len(d_a) < end or len(d_b) < end:
        raise Error("levers_optimizer_step: grads shorter than adapter range")

    if cfg.optimizer == TRAIN_OPTIMIZER_ADAFACTOR:
        # SimpleTuner "torch-adafactor" hyperparams; cfg sentinels 0.0 mean
        # "unset" -> registered defaults (optimizer_param.py:153-162).
        var lr = Float64(step_lr)
        var beta2_decay = _ADAFACTOR_BETA2_DECAY_DEFAULT
        if cfg.optimizer_decay_rate != Float32(0.0):
            beta2_decay = Float64(cfg.optimizer_decay_rate)
        var eps2 = _ADAFACTOR_EPS2_DEFAULT
        if cfg.optimizer_eps2 > Float32(0.0):
            eps2 = Float64(cfg.optimizer_eps2)
        var d = _ADAFACTOR_D_DEFAULT
        if cfg.optimizer_clip_threshold > Float32(0.0):
            d = Float64(cfg.optimizer_clip_threshold)
        var wd = Float64(cfg.weight_decay)
        for i in range(start, end):
            var idx = 2 * (i - start)
            var pa = _levers_bf16_to_f32(adapters[i].a)
            adafactor_step_2d(
                pa, d_a[i], st.ada[idx], lr, beta2_decay,
                Float64(-1.0), eps2, d, wd,
            )
            _levers_writeback_bf16(adapters[i].a, pa)
            var pb = _levers_bf16_to_f32(adapters[i].b)
            adafactor_step_2d(
                pb, d_b[i], st.ada[idx + 1], lr, beta2_decay,
                Float64(-1.0), eps2, d, wd,
            )
            _levers_writeback_bf16(adapters[i].b, pb)
    elif cfg.optimizer == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW:
        # RAW cfg.lr, NOT step_lr (override_lr_scheduler=True — see section
        # header); warmup is internal via cfg.optimizer_warmup_steps.
        var k0 = st.sf_ctl.k
        if k0 != k - 1:
            raise Error(
                String("levers_optimizer_step: schedule-free k desync (ctl.k=")
                + String(k0) + String(", trainer step=") + String(k)
                + String(")")
            )
        var lr = Float64(cfg.lr)
        var beta1 = Float64(cfg.beta1)
        var beta2 = Float64(cfg.beta2)
        var eps = Float64(cfg.eps)
        var wd = Float64(cfg.weight_decay)
        var warmup = cfg.optimizer_warmup_steps
        for i in range(start, end):
            var idx = 2 * (i - start)
            var pa = _levers_bf16_to_f32(adapters[i].a)
            adamw_schedulefree_step_param(
                pa, d_a[i], st.sf[idx], k0, lr, beta1, beta2, eps, wd,
                warmup, True,
            )
            _levers_writeback_bf16(adapters[i].a, pa)
            var pb = _levers_bf16_to_f32(adapters[i].b)
            adamw_schedulefree_step_param(
                pb, d_b[i], st.sf[idx + 1], k0, lr, beta1, beta2, eps, wd,
                warmup, True,
            )
            _levers_writeback_bf16(adapters[i].b, pb)
        st.sf_ctl.end_step(
            adamw_schedulefree_adjusted_lr(k0, lr, beta2, warmup)
        )
    else:
        raise Error(
            String("levers_optimizer_step: optimizer tag ")
            + String(cfg.optimizer)
            + String(" has no levers dispatch")
        )


def levers_optimizer_sync_resident(
    mut opt_state: LoraAdamWPlainDeviceState,
    adapters: List[LoraAdapter],
    ctx: DeviceContext,
) raises:
    """Push the host-stepped bf16 a/b params into the resident optimizer
    state's LIVE dev_p buffer (the model's device LoRA views are sub-buffers
    of it — lora_adamw_plain_fused.mojo:285-299). Pack layout mirrors
    lora_adamw_plain_device_state_init (:353-405): A then B per adapter,
    flat, via the pinned host_p mirror; ONE H2D + sync."""
    var hp = Int(opt_state.host_p.unsafe_ptr())
    var off = 0
    for i in range(opt_state.start, opt_state.end):
        var n_a = opt_state.seg_len[2 * (i - opt_state.start)]
        var n_b = opt_state.seg_len[2 * (i - opt_state.start) + 1]
        if len(adapters[i].a) != n_a or len(adapters[i].b) != n_b:
            raise Error(
                String("levers_optimizer_sync_resident: adapter len mismatch")
                + String(" at ") + String(i)
            )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hp + off * 2),
            BytePtr(unsafe_from_address=Int(adapters[i].a.unsafe_ptr())),
            n_a * 2,
        )
        off += n_a
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hp + off * 2),
            BytePtr(unsafe_from_address=Int(adapters[i].b.unsafe_ptr())),
            n_b * 2,
        )
        off += n_b
    ctx.enqueue_copy(dst_buf=opt_state.dev_p, src_buf=opt_state.host_p)
    ctx.synchronize()


def levers_optimizer_step(
    cfg: TrainConfig,
    mut adapters: List[LoraAdapter],
    d_a: List[List[Float32]],
    d_b: List[List[Float32]],
    k: Int,
    step_lr: Float32,
    mut st: LeversOptimizerState,
    mut opt_state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
) raises:
    """The ONE trainer call for the T1.C optimizer lever: host optimizer
    step over opt_state's adapter range, then the resident dev_p sync so
    the device LoRA views see the new weights next step. Call ONLY when
    levers_optimizer_active(cfg) — the default AdamW path keeps its
    existing literal fused call (C13)."""
    levers_optimizer_step_host(
        cfg, adapters, d_a, d_b, k, step_lr,
        opt_state.start, opt_state.end, st,
    )
    levers_optimizer_sync_resident(opt_state, adapters, ctx)


def levers_optimizer_eval_for_save(cfg: TrainConfig, mut st: LeversOptimizerState):
    """Schedule-free save bracket, BEFORE any weight save / validation
    sample (adamw_schedulefree.mojo header: true schedule-free saves the x
    iterate; the reference's z is dead so this only flips train_mode today,
    but trainers wire the bracket NOW so real-z math is a drop-in). No-op
    for every other optimizer. NOTE: passes no params because the reference
    never creates z; a real-z implementation must route the live param
    lists through here."""
    if cfg.optimizer == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW and st.initialized:
        var no_params = List[List[Float32]]()
        adamw_schedulefree_eval(st.sf_ctl, no_params)


def levers_optimizer_train_after_save(cfg: TrainConfig, mut st: LeversOptimizerState):
    """Schedule-free save bracket, AFTER the save / validation sample and
    before the train loop resumes (pair of levers_optimizer_eval_for_save)."""
    if cfg.optimizer == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW and st.initialized:
        var no_params = List[List[Float32]]()
        adamw_schedulefree_train(st.sf_ctl, no_params)


# ══════════════════════════════════════════════════════════════════════════════
# KLEIN ADDITIVE HELPER (Tier-1 fan-out 2026-06-11; bounded addition — the
# campaign doc's "Lever fan-out" follow-up). Everything ABOVE this banner is
# the T1.A-T1.F module as landed; ONLY this OT-state resident sync is new.
#
# klein's v2 engine keeps its live bf16 LoRA params in TWO OT-semantics
# LoraAdamWOTDeviceState buffers (dbl + sgl; training/lora_adamw_ot_fused.mojo
# :395-441) — a DIFFERENT struct from zimage's LoraAdamWPlainDeviceState: it
# has NO [start,end) window (each state covers its WHOLE adapter list,
# nseg == 2*len(adapters)), but the SAME flat A-then-B pack layout, and the
# model's device LoRA views sub-buffer its dev_p (klein_stack_lora.mojo
# _klein_resident_adapter). After the host levers step on the dbl/sgl host
# a/b mirrors, this pushes the stepped params back into dev_p — the exact
# inverse of the resident OT step's P readback
# (fused_lora_adamw_ot_step_resident's host_p<-dev_p + memcpy-out,
# lora_adamw_ot_fused.mojo). Correctness over speed: one host_p pack + one
# H2D + sync per adapter list per step; a fused GPU path can land later
# behind the same dispatch (same contract as levers_optimizer_sync_resident).
# ══════════════════════════════════════════════════════════════════════════════
from serenitymojo.training.lora_adamw_ot_fused import LoraAdamWOTDeviceState


def levers_optimizer_sync_resident_ot(
    mut opt_state: LoraAdamWOTDeviceState,
    adapters: List[LoraAdapter],
    ctx: DeviceContext,
) raises:
    """OT-state sibling of levers_optimizer_sync_resident: pack the
    host-stepped bf16 a/b of the WHOLE adapter list into the pinned host_p
    mirror (A then B per adapter, flat — the lora_adamw_ot_device_state_init
    layout) and upload ONE H2D into the live dev_p that the device LoRA
    views sub-buffer."""
    if 2 * len(adapters) != opt_state.nseg:
        raise Error(
            String("levers_optimizer_sync_resident_ot: adapter list len ")
            + String(len(adapters)) + String(" != state nseg/2, nseg=")
            + String(opt_state.nseg)
        )
    var hp = Int(opt_state.host_p.unsafe_ptr())
    var off = 0
    for i in range(len(adapters)):
        var n_a = opt_state.seg_len[2 * i]
        var n_b = opt_state.seg_len[2 * i + 1]
        if len(adapters[i].a) != n_a or len(adapters[i].b) != n_b:
            raise Error(
                String("levers_optimizer_sync_resident_ot: adapter len")
                + String(" mismatch at ") + String(i)
            )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hp + off * 2),
            BytePtr(unsafe_from_address=Int(adapters[i].a.unsafe_ptr())),
            n_a * 2,
        )
        off += n_a
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hp + off * 2),
            BytePtr(unsafe_from_address=Int(adapters[i].b.unsafe_ptr())),
            n_b * 2,
        )
        off += n_b
    ctx.enqueue_copy(dst_buf=opt_state.dev_p, src_buf=opt_state.host_p)
    ctx.synchronize()
