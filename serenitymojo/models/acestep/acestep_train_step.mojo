# acestep_train_step.mojo — ACE-Step-1.5 LoRA training step (Tier-3 T3.C, #13).
#
# The per-step training logic wrapping the full-stack backward
# (autograd_v2/acestep_stack_lora_graph_backward): sample timestep t + noise x1,
# flow-match interpolate xt, CFG-dropout the condition, run the forward+backward,
# return (loss, 512 LoRA grads) for the driver's AdamW (#14).
#
# Faithful to acestep/training_v2/{fixed_lora_module.py::training_step,
# timestep_sampling.py} (READ IN FULL 2026-07-13):
#   t = max(sigmoid(N(mu,sigma)), sigmoid(N(mu,sigma))), r = t   (use_meanflow=False
#       → data_proportion=1.0 → r=t; mu=-0.4, sigma=1.0)
#   x1 = randn_like(x0)  (noise);  xt = t*x1 + (1-t)*x0;  flow = x1 - x0
#   CFG dropout: per-sample rand<cfg_ratio(0.15) → replace ehs with null_condition_emb
#       (null_condition_emb is a TOP-LEVEL model Parameter, NOT decoder.* → absent in
#       the decoder-only load → the oracle dump ran with CFG OFF; this path is wired
#       but UNTESTED by the oracle).
#   loss = plain F.mse_loss(out0, flow) (fp32); NOT masked in the fixed path.
#
# Mojo 1.0.0b1, NVIDIA.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.collections import Optional
from std.math import exp as _exp, max as _fmax
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random_torch import randn_torch
from serenitymojo.ops.random import rand_uniform
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar, mul, reshape
from serenitymojo.ops.reduce import reduce_mean_f32
from serenitymojo.models.dit.acestep_dit import AceStepDiTConfig
from serenitymojo.autograd_v2.acestep_block_graph import (
    AcestepStackLoraGrads,
    acestep_stack_lora_graph_backward,
)


# ── logit-normal timestep: t = max(sigmoid(a*σ+μ), sigmoid(b*σ+μ)), r=t ────────
def acestep_sample_t(
    seed: UInt64, mu: Float32, sigma: Float32, ctx: DeviceContext
) raises -> Float32:
    """sample_timesteps() at bs=1: two standard normals → logit-normal → max.
    r=t (use_meanflow=False). Faithful to timestep_sampling.sample_timesteps."""
    var a = randn_torch([1], seed, ctx).to_host(ctx)[0]
    var b = randn_torch([1], seed + 1, ctx).to_host(ctx)[0]
    var ta = _sigmoid(a * sigma + mu)
    var tb = _sigmoid(b * sigma + mu)
    return _fmax(ta, tb)


def _sigmoid(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + _exp(-x))


# ── flow-match interpolation: xt = t*x1 + (1-t)*x0 ; flow = x1 - x0 ────────────
struct AcestepNoise(Movable):
    var xt: Tensor
    var flow: Tensor

    def __init__(out self, var xt: Tensor, var flow: Tensor):
        self.xt = xt^
        self.flow = flow^


def acestep_flow_noise(
    x0: Tensor, x1: Tensor, t: Float32, ctx: DeviceContext
) raises -> AcestepNoise:
    """xt = t*x1 + (1-t)*x0 ; flow = x1 - x0 (x1=noise, x0=data). Same dtype as x0."""
    var xt = add(mul_scalar(x1, t, ctx), mul_scalar(x0, Float32(1.0) - t, ctx), ctx)
    var flow = sub(x1, x0, ctx)
    return AcestepNoise(xt^, flow^)


# ── CFG dropout (bs=1): coin flip → replace ehs with null_cond ────────────────
def acestep_apply_cfg_dropout(
    ehs: Tensor, null_cond: Tensor, cfg_ratio: Float32, seed: UInt64,
    ctx: DeviceContext,
) raises -> Tensor:
    """bs=1 CFG dropout: draw one U[0,1); if < cfg_ratio replace ehs with null_cond
    (must already be shaped == ehs), else keep ehs. Faithful to
    timestep_sampling.apply_cfg_dropout at B=1. UNTESTED by the oracle (CFG off)."""
    var b = ehs.shape()[0]
    if b != 1:
        raise Error("acestep_apply_cfg_dropout: only bs=1 wired (the recipe is bs=1)")
    var u = rand_uniform([1], seed, STDtype.F32, Float32(0.0), Float32(1.0), ctx).to_host(ctx)[0]
    if u < cfg_ratio:
        if null_cond.numel() != ehs.numel():
            raise Error("acestep cfg dropout: null_cond must be pre-expanded to ehs shape")
        return null_cond.clone(ctx)
    return ehs.clone(ctx)


# ── loss scalar (standalone; the backward computes its own internally) ────────
def acestep_loss_mse(out0: Tensor, flow: Tensor, ctx: DeviceContext) raises -> Float32:
    """mean((out0-flow)^2), F32-accumulated (~torch F.mse_loss on bf16 inputs)."""
    var diff = sub(out0, flow, ctx)
    var sq = mul(diff, diff, ctx)
    var n = out0.numel()
    var m = reduce_mean_f32(reshape(sq, [n], ctx), [0], False, ctx)
    return m.to_host(ctx)[0]


# ── the training step: sample → noise → CFG → forward+backward → (loss, grads) ─
def acestep_train_step[
    SP: Int, L: Int, NH: Int, LAYERS: Int
](
    target_latents: Tensor,       # x0 [1,T,acoustic]
    context: Tensor,              # [1,T,128]
    ehs: Tensor,                  # encoder_hidden_states [1,L,2048] (RAW; embedded inside)
    full: Dict[String, ArcPointer[Tensor]],   # decoder.* weights + 512 LoRA A/B
    cfg: AceStepDiTConfig,
    lora_scale: Float32,
    seed: UInt64,
    ctx: DeviceContext,
    mu: Float32 = -0.4, sigma: Float32 = 1.0, cfg_ratio: Float32 = 0.15,
    null_cond: Optional[Tensor] = None,
) raises -> AcestepStackLoraGrads:
    """One ACE-Step LoRA training step. Samples t + noise, interpolates xt, applies
    CFG dropout (if null_cond given), runs the full-stack forward+backward, returns
    the 512 LoRA grads + loss scalar (AcestepStackLoraGrads) for the driver's AdamW."""
    var t = acestep_sample_t(seed, mu, sigma, ctx)
    var x1 = cast_tensor(
        randn_torch(target_latents.shape(), seed + 100, ctx), target_latents.dtype(), ctx
    )
    var noise = acestep_flow_noise(target_latents, x1, t, ctx)

    var ehs2 = ehs.clone(ctx)
    if null_cond and cfg_ratio > Float32(0.0):
        ehs2 = acestep_apply_cfg_dropout(ehs, null_cond.value(), cfg_ratio, seed + 200, ctx)

    return acestep_stack_lora_graph_backward[SP, L, NH, LAYERS](
        noise.xt, context, ehs2, t, t, noise.flow, full, cfg, lora_scale, ctx
    )
