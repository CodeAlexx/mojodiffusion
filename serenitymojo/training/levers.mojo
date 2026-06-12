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
# Mojo 1.0.0b1. Host-only — no GPU imports.

from serenitymojo.ops.loss_fns import (
    LossGrad, mse_loss_grad, huber_loss_grad, smooth_l1_loss_grad,
    min_snr_gamma_weight,
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
