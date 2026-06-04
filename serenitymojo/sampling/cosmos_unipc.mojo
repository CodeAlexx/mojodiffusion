# sampling/cosmos_unipc.mojo — Cosmos-Predict2.5 UniPC multistep sampler (bh2).
#
# Pure-Mojo + MAX, inference-only, GPU-only. This is the Cosmos-Predict2.5
# `FlowUniPCMultistepScheduler` (bh2, solver_order=2, predict_x0=True,
# lower_order_final=True, final_sigmas_type="zero", shift=5.0 for V2_2B).
#
# REUSE: the predictor/corrector math, the bh2 coefficient pipeline, the
# linsolve, the ring buffer and step-state machine ALREADY live in
# sampling/unipc.mojo (`UniPcMultistepScheduler`) — that file is a line-for-line
# port of:
#   /home/alex/EriDiffusion/inference-flame/src/sampling/cosmos_unipc.rs
#   canonical: /home/alex/refs/cosmos-predict2.5/cosmos_predict2/_src/predict2/
#              models/fm_solvers_unipc.py
# The cosmos-specific delta is ONLY the named entrypoint + the wired V2_2B
# defaults (num_train=1000, shift=5, solver_order=2), so this module re-exports
# the shared scheduler and adds a thin convenience constructor. No new math.
#
# THE MATH (verbatim, all in sampling/unipc.mojo):
#   schedule:   sigmas = shift*linspace((N-1)/N,0,n+1)[:-1]/(1+(shift-1)*s) ⧺ [0]
#   convert:    x0_pred = sample - sigma_t * model_output            (predict_x0)
#   predictor (multistep_uni_p_bh_update, order≤2):
#     lambda = log(alpha) - log(sigma),  alpha = 1 - sigma,  h = lambda_t-lambda_s0
#     hh = -h ; h_phi_1 = expm1(hh) ; B_h = expm1(hh)          (bh2)
#     x_t_ = (sigma_t/sigma_s0)*x - alpha_t*h_phi_1*m0
#     order==2: rhos_p=[0.5]; pred_res = 0.5*(m1-m0)/rk ; x_t = x_t_ - alpha_t*B_h*pred_res
#   corrector (multistep_uni_c_bh_update):
#     order==1: rhos_c=[0.5]; order==2: solve 2x2 R·rhos=b (f64 Gauss-Jordan)
#     x_t = x_t_ - alpha_t*B_h*(corr_res + rhos_c[-1]*(model_t - m0))
#   step: use_corrector when step_index>0 & last_sample set; ring-buffer shift;
#         this_order = min(min(order, n-step_index), lower_order_nums+1); advance.
#
# Mojo 1.0.0b1. Inference-only. No autograd, no Python at runtime.

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
# Re-export the shared Cosmos UniPC machinery so callers can import everything
# from the cosmos-named module.
from serenitymojo.sampling.unipc import (
    UniPcMultistepScheduler,
    compute_bh2_coefficients,
    build_unipc_sigma_schedule,
    alpha_from_sigma,
)


comptime COSMOS_NUM_TRAIN_TIMESTEPS = 1000
comptime COSMOS_UNIPC_DEFAULT_SHIFT: Float64 = 5.0
comptime COSMOS_UNIPC_DEFAULT_NUM_STEPS = 35
comptime COSMOS_UNIPC_SOLVER_ORDER = 2


def build_cosmos_unipc_sampler(
    num_inference_steps: Int = COSMOS_UNIPC_DEFAULT_NUM_STEPS,
    shift: Float64 = COSMOS_UNIPC_DEFAULT_SHIFT,
) raises -> UniPcMultistepScheduler:
    """Construct the Cosmos-Predict2.5 UniPC sampler with V2_2B defaults.

    bh2 / solver_order=2 / predict_x0=True / lower_order_final=True /
    final_sigmas_type="zero" / num_train_timesteps=1000. Call `.step(model_out,
    sample, ctx)` once per inference step in order; `step_index` advances
    internally and the corrector engages from step 1 on.
    """
    return UniPcMultistepScheduler(
        COSMOS_NUM_TRAIN_TIMESTEPS,
        num_inference_steps,
        shift,
        COSMOS_UNIPC_SOLVER_ORDER,
    )
