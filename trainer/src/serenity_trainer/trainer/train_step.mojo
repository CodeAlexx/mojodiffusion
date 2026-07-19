# train_step.mojo — ONE training-step recipe, ported from Serenity
# GenericTrainer.train() L732-878 inner block (the per-batch body):
#   predict -> calculate_loss -> loss/=accum -> backward
#   -> [on update step] clip_grad_norm -> optimizer.step -> zero_grad
#
# This file owns the MODEL-AGNOSTIC step. It composes the existing pipeline
# pieces (tape backward, grad.clip_grad_norm/scale_grads, optim.adamw_step,
# loss.timestep_weight/loss_scale, lr_schedule factor) around a ModelSpec.
#
# Tape interaction (see loss.mojo header): the tape's mse_loss arm produces the
# FULL d_pred and IGNORES the scalar loss seed. So the per-step loss weight and
# the 1/accum factor are applied as a GRAD SCALE (chain rule: d(w·L)/dp = w·dL/dp),
# NOT by scaling the scalar loss. Serenity scales the *loss* before backward
# (L780 `loss = loss / accum`); with the seed-ignoring tape we get the identical
# parameter update by scaling the grads instead.
#
# Dtype policy: params / moments / grads are BF16 storage. F32 only for the host
# scalar grad-norm and the host LR factor.

from std.memory import ArcPointer
from std.collections.dict import Dict
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.autograd import Tape, backward
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.modelSetup.BaseModelSetup import StepOutput, ModelSpec
from serenity_trainer.modelSetup.mixin.ModelSetupDiffusionLossMixin import timestep_weight, loss_scale, LS_NONE
from serenity_trainer.util.grad_util import (
    clip_grad_norm,
    scale_grads,
    accumulate,
    TArc,
)
from serenity_trainer.util.optimizer.adamw_extensions import adamw_step


# ─────────────────────────────────────────────────────────────────────────────
# ParamSlot — a single trainable param plus its AdamW state.
#
# Tensor is move-only (cannot be a List element), so each tensor is boxed in an
# ArcPointer; that makes ParamSlot itself Copyable → it can live in a List. The
# model owns the underlying tensors; the slot keeps refcounted handles. `tid` is
# the param's tape id (stamped by Tape.track at the start of the step) — backward
# keys grads by this id.
#
# m (exp_avg) and v (exp_avg_sq) are zeros_like(p) = BF16 (Serenity keeps state
# in param dtype, adamw_extensions.py — no F32 master / moments). `accum` holds
# the running BF16 grad accumulation across micro-steps (= torch .grad).
struct ParamSlot(Copyable, Movable):
    var p: TArc       # param  [.](BF16, trained in place)
    var m: TArc       # exp_avg     (BF16)
    var v: TArc       # exp_avg_sq  (BF16)
    var accum: TArc   # accumulated grad across micro-steps (BF16)
    var tid: Int      # tape id of `p` (set by track_params)

    def __init__(out self, var p: TArc, var m: TArc, var v: TArc, var accum: TArc):
        self.p = p^
        self.m = m^
        self.v = v^
        self.accum = accum^
        self.tid = 0


# ─────────────────────────────────────────────────────────────────────────────
# StepResult — host diagnostics for one optimizer step (mirrors what Serenity
# logs to the board: the pre-clip grad norm; plus the loss for ema/board).
@fieldwise_init
struct StepResult(Copyable, Movable):
    var loss: Float32            # the (unscaled) host MSE for logging/ema
    var grad_norm: Float32       # pre-clip global grad norm (-1 if not an update step)
    var did_update: Bool         # True iff optimizer.step ran this micro-step


# Track every param on the tape (stamping ids) BEFORE predict, so the LoRA/model
# forward records grads against these ids. Mirrors the fact that Serenity's
# autograd graph references the same Parameter objects each step.
def track_params(mut tape: Tape, mut slots: List[ParamSlot]):
    for i in range(len(slots)):
        # We build a FRESH Tape per step, whose id counter restarts at 1. A param
        # carrying an id from a previous tape would collide with this tape's
        # freshly-minted op ids → reset to 0 first, then let track() re-stamp it
        # from this tape's counter (the smoke-test idiom: track() before record).
        # Mutate the boxed Tensor IN PLACE via the Arc deref (Tensor is move-only;
        # do NOT move it out of the box). `slots[i].p[]` yields a mutable ref.
        slots[i].p[].set_id(0)
        tape.track(slots[i].p[])
        slots[i].tid = slots[i].p[].id


# Zero the accumulators after an optimizer step (Serenity optimizer.zero_grad).
# Replaces the boxed grad with a fresh zeros_like; cheap and avoids an in-place
# kernel. We just drop the accum (set to a freshly-zeroed clone) by reusing the
# param shape via a device zeros — but to stay dependency-light we mark it stale
# and rebuild on first accumulate of the next cycle (see step()).


# ─────────────────────────────────────────────────────────────────────────────
# train_step — run ONE micro-step (one batch). Records forward on a FRESH tape,
# seeds the MSE arm, backs out grads, applies (loss_weight · loss_scale / accum)
# as a grad scale, accumulates into each slot, and — on the accumulation boundary
# — clips and runs AdamW + zero_grad.
#
# Args mirror the Serenity loop variables:
#   spec        : the ModelSpec (predict)               [BaseModelSetup.predict]
#   slots       : trainable params + AdamW state         [self.parameters / state]
#   cfg         : TrainConfig                             [self.config]
#   global_step : 0-based optimizer step count (for SR seed + bias-corr step)
#   micro_idx   : 0-based micro-step within the accumulation window
#   lr          : the resolved LR for this step (base_lr · lr_schedule.factor)
#   snr_or_sigma: the timestep weighting input from predict (StepOutput.timestep)
#
# `micro_idx == accum-1` is the update step (Serenity __is_update_step).
def train_step[S: ModelSpec](
    mut spec: S,
    mut slots: List[ParamSlot],
    cfg: TrainConfig,
    global_step: Int,
    micro_idx: Int,
    lr: Float32,
    ctx: DeviceContext,
) raises -> StepResult:
    var accum = cfg.gradient_accumulation_steps
    if accum < 1:
        accum = 1

    # ── forward on a fresh tape (Serenity: model_setup.predict) ─────────────
    var tape = Tape()
    track_params(tape, slots)
    var out = spec.predict(tape, cfg, global_step, ctx)   # StepOutput

    # ── loss: tape MSE arm seeds the chain (calculate_loss base = MSE) ────────
    # The scalar value is host-side diagnostics only; backward ignores its seed.
    var loss_scalar = tape.mse_loss(out.predicted, out.target, ctx)
    var host_loss = loss_scalar.to_host(ctx)[0]

    # ── backward (Serenity loss.backward) → grads keyed by tape id ──────────
    var gmap = backward(tape, loss_scalar, ctx)

    # ── resolve the per-step grad scale ───────────────────────────────────────
    # Serenity applies, before/at backward:
    #   * loss / gradient_accumulation_steps                  (L780)
    #   * LossScaler.get_scale (batch / accum factors)        (calculate_loss)
    #   * the timestep loss weight (min-SNR / debiased / p2 / sigma)
    # With the seed-ignoring MSE tape these all fold into ONE grad multiplier
    # (chain rule: d(w·L)/dp = w·dL/dp).
    #
    # SEAM NOTE (v_prediction): timestep_weight is called with v_prediction=False.
    # Serenity derives v_pred = (data['prediction_type']=='v_prediction') per
    # model (ModelSetupDiffusionLossMixin.py); for flow-matching (the default
    # target, flow_target.mojo) v_pred is False and this is exact. v-prediction
    # models (e.g. SD2.x-v) must EITHER pre-fold their loss weight upstream in
    # predict() and select cfg.loss_weight_kind == LW_CONSTANT, OR this seam must
    # be extended to carry a per-model v_prediction flag. StepOutput does not yet
    # carry that flag, so as written v-pred min-SNR/debiased/p2 weights are NOT
    # reproduced — flagged for the model unit.
    var w_t = timestep_weight(
        cfg.loss_weight_kind, out.timestep, cfg.min_snr_gamma, False
    )
    # LossScaler.get_scale: the driver/config does not expose a loss-scaler kind,
    # so it is fixed to LS_NONE (=1.0) here (the Serenity default unless a model
    # opts into batch/accum scaling). The 1/accum factor (L780) is applied
    # explicitly so this matches Serenity's effective grad scale exactly when
    # loss_scaler==NONE. Per-SAMPLE batch['loss_weight'] (if any) is upstream,
    # folded into pred/target by predict() — it is NOT applied here.
    var grad_scale = w_t * loss_scale(LS_NONE, cfg.batch_size, accum) / Float32(accum)

    # ── gather this micro-step's grads (in slot order), scale, accumulate ─────
    var micro_grads = List[TArc]()
    for i in range(len(slots)):
        var tid = slots[i].tid
        if gmap.__contains__(tid):
            micro_grads.append(gmap[tid])          # refcount-bump copy
        else:
            # param received no gradient this step. NOTE: this is NOT a skip —
            # the param still gets a zero grad and is AdamW-stepped below, so
            # decoupled weight-decay runs (Serenity's AdamW applies WD every
            # step regardless of grad). Build the zero grad by cloning p and
            # scaling by 0 (clone keeps shape/dtype; ×0 makes it a true zero).
            micro_grads.append(TArc(slots[i].p[].clone(ctx)))
            # mark with scale 0 by appending a sentinel handled below — simplest
            # correct path: multiply this entry by 0 individually.
            scale_grads_one(micro_grads, len(micro_grads) - 1, Float32(0.0), ctx)

    # apply the loss-weight/accum grad scale to ALL micro grads at once
    scale_grads(micro_grads, grad_scale, ctx)

    # accumulate into each slot (torch: grads sum into .grad across micro-steps).
    # First micro-step of a window seeds accum directly; later ones add in place.
    for i in range(len(slots)):
        if micro_idx == 0:
            slots[i].accum = micro_grads[i]                       # seed
        else:
            accumulate(slots[i].accum[], micro_grads[i][], ctx)   # += in place

    var is_update = (micro_idx == accum - 1)
    if not is_update:
        return StepResult(host_loss, Float32(-1.0), False)

    # ── update step (Serenity __is_update_step branch, L834-855) ────────────
    # 1) clip_grad_norm over ALL accumulated grads (pre-clip norm returned).
    var clip_list = List[TArc]()
    for i in range(len(slots)):
        clip_list.append(slots[i].accum)
    var pre_clip_norm = clip_grad_norm(clip_list, cfg.clip_grad_norm, ctx)
    # clip_grad_norm may have replaced the boxed grads (scale_grads rebinds) —
    # copy the (possibly-clipped) handles back into the slots.
    for i in range(len(slots)):
        slots[i].accum = clip_list[i]

    # 2) optimizer.step — AdamW per param (BF16 state, F32-register compute, SR).
    #    step counter is 1-based (Serenity state['step'] += 1 before update).
    var step_1based = global_step + 1
    # SR seed = bare global_step (Serenity GenericTrainer.py:732
    #   step_seed = train_progress.global_step; set_seed(step_seed)). NO XOR with
    # cfg.seed — Serenity does not fold a config seed here, so any nonzero offset
    # would break bf16 stochastic-rounding bit-parity with the reference.
    var sr_seed = UInt32(global_step)              # per-step SR seed (set_seed)
    for i in range(len(slots)):
        adamw_step(
            slots[i].p[],
            slots[i].m[],
            slots[i].v[],
            slots[i].accum[],
            step_1based,
            lr,
            cfg.beta1,
            cfg.beta2,
            cfg.eps,
            cfg.weight_decay,
            cfg.stochastic_rounding,
            sr_seed,
            ctx,
        )

    # 3) zero_grad — drop the accumulators (next window re-seeds at micro_idx 0).
    #    Handled implicitly: micro_idx==0 next window overwrites slot.accum.

    return StepResult(host_loss, pre_clip_norm, True)


# Scale a single boxed grad at `idx` in place (helper for the no-grad zero path).
def scale_grads_one(
    mut grads: List[TArc], idx: Int, factor: Float32, ctx: DeviceContext
) raises:
    grads[idx] = TArc(mul_scalar(grads[idx][], factor, ctx))
