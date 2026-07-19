# loss.mojo — loss weighting + scaling, ported from Serenity
# (modules/modelSetup/mixin/ModelSetupDiffusionLossMixin.py). The base MSE with
# autograd is the tape's `mse_loss` (serenitymojo.autograd); this module provides
# the timestep loss-weight scalars and the loss scaler.
#
# IMPORTANT (tape interaction): the autograd tape's OP_MSE arm computes the full
# d_pred and IGNORES the incoming loss seed. So a per-step loss weight CANNOT be
# applied by scaling the scalar loss after mse_loss — it must scale the GRADS
# (chain rule: d(w*L)/dp = w*dL/dp). The driver multiplies the resolved weight
# into the grads (via grad.scale_grads). This file computes that scalar weight.
#
# Serenity computes losses in F32 (predicted/target cast bf16->f32). That is
# compute-in-f32 producing a loss scalar (a host statistic), not stored model
# state — consistent with the bf16-storage policy.

from std.math import rsqrt

comptime LW_CONSTANT = 0
comptime LW_MIN_SNR_GAMMA = 1
comptime LW_DEBIASED_ESTIMATION = 2
comptime LW_P2 = 3
comptime LW_SIGMA = 4   # flow-matching sigma weighting


# --- timestep loss-weight scalars (exact ports; operate on a scalar SNR) -------
# The model spec computes `snr` for the sampled timestep; these map it to a
# multiplier. Faithful to ModelSetupDiffusionLossMixin.{__min_snr_weight,
# __debiased_estimation_weight, __p2_loss_weight}.

def min_snr_weight(snr_in: Float32, gamma: Float32, v_prediction: Bool) -> Float32:
    var min_snr_gamma = snr_in if snr_in < gamma else gamma
    var snr = snr_in
    if v_prediction:
        snr = snr + 1.0   # denom increased by 1 for v-pred (AFTER min computed)
    return min_snr_gamma / snr


def debiased_estimation_weight(snr_in: Float32, v_prediction: Bool) -> Float32:
    var w = snr_in
    if w > 1.0e3:        # torch.clip(weight, max=1e3) — Kohya stability fix
        w = 1.0e3
    if v_prediction:
        w = w + 1.0
    return rsqrt(w)


def p2_loss_weight(snr_in: Float32, gamma: Float32, v_prediction: Bool) -> Float32:
    var snr = snr_in
    if v_prediction:
        snr = snr + 1.0
    # (1 + snr) ** -gamma
    var base = 1.0 + snr
    return base ** (-gamma)


# Resolve the timestep weight for the active loss_weight_fn. `snr` is the SNR for
# diffusion weightings; for LW_SIGMA it is the sigma value directly (flow match).
def timestep_weight(
    kind: Int, snr_or_sigma: Float32, gamma: Float32, v_prediction: Bool
) -> Float32:
    if kind == LW_CONSTANT:
        return 1.0
    elif kind == LW_MIN_SNR_GAMMA:
        return min_snr_weight(snr_or_sigma, gamma, v_prediction)
    elif kind == LW_DEBIASED_ESTIMATION:
        return debiased_estimation_weight(snr_or_sigma, v_prediction)
    elif kind == LW_P2:
        return p2_loss_weight(snr_or_sigma, gamma, v_prediction)
    else:  # LW_SIGMA — flow-matching sigma weighting (sigma used directly)
        return snr_or_sigma


# --- loss scaler (LossScaler.get_scale) ---------------------------------------
# LossScaler enum: NONE/BATCH/GLOBAL_BATCH/GRADIENT_ACCUMULATION/BOTH/GLOBAL_BOTH.
# Local (non-global) scales by the LOCAL factors. GLOBAL_* also scale by world
# size (multi-GPU = out of scope → treated as local here). Verify against
# modules/util/enum/LossScaler.py in the skeptic pass.
comptime LS_NONE = 0
comptime LS_BATCH = 1
comptime LS_GLOBAL_BATCH = 2
comptime LS_GRADIENT_ACCUMULATION = 3
comptime LS_BOTH = 4
comptime LS_GLOBAL_BOTH = 5


def loss_scale(kind: Int, batch_size: Int, accumulation_steps: Int) -> Float32:
    var s = Float32(1.0)
    if kind == LS_BATCH or kind == LS_GLOBAL_BATCH or kind == LS_BOTH or kind == LS_GLOBAL_BOTH:
        s = s * Float32(batch_size)
    if kind == LS_GRADIENT_ACCUMULATION or kind == LS_BOTH or kind == LS_GLOBAL_BOTH:
        s = s * Float32(accumulation_steps)
    return s
