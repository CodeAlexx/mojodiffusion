# seedvr2_sampler.mojo — SeedVR2 SAMPLER math, pure Mojo + MAX (inference only).
#
# Faithful port of SeedVR's common/diffusion sampling primitives:
#   * cfg               : classifier-free guidance, rescale=0 (no rescale).
#   * euler_vlerp_step  : one euler v-lerp denoise step (v-prediction lerp form).
#   * sampler_timesteps : SD3-shift timestep schedule (host float compute).
#
# All tensor arithmetic is elementwise F32 via serenitymojo.ops.tensor_algebra
# (add / sub / mul_scalar). Tensors are [1024,16] latents in the parity oracle.
#
# Parity gate: serenitymojo/models/dit/tests/seedvr2_sampler_parity.mojo

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar


# ─────────────────────────────────────────────────────────────────────────────
# Classifier-free guidance.  out = neg + scale * (pos - neg)   (rescale = 0)
# pos/neg are [N, C] tensors, scale is a scalar (e.g. 7.5).
# ─────────────────────────────────────────────────────────────────────────────
def cfg(pos: Tensor, neg: Tensor, scale: Float32, ctx: DeviceContext) raises -> Tensor:
    var diff = sub(pos, neg, ctx)              # pos - neg
    var scaled = mul_scalar(diff, scale, ctx)  # scale * (pos - neg)
    return add(neg, scaled, ctx)               # neg + scale*(pos - neg)


# ─────────────────────────────────────────────────────────────────────────────
# One euler v-lerp step.  Given the model prediction `pred` at state `x` and
# current timestep `t`, produce the state at the next timestep `s`.  T = 1000.
#
#   A_t = 1 - t/T ;  B_t = t/T
#   px0 = (x - B_t*pred) / (A_t + B_t)     # clean-latent endpoint
#   pxT = (x + A_t*pred) / (A_t + B_t)     # noise endpoint
#   A_s = 1 - s/T ;  B_s = s/T
#   x_s = A_s*px0 + B_s*pxT
#   if s <= 0: x_s = px0                   # clamp at the final step
#
# (A_t + B_t == 1 always, but the divide is kept for faithfulness.)
# ─────────────────────────────────────────────────────────────────────────────
def euler_vlerp_step(
    pred: Tensor, x: Tensor, t: Float32, s: Float32, T: Float32, ctx: DeviceContext
) raises -> Tensor:
    var A_t = 1.0 - t / T
    var B_t = t / T
    var denom = A_t + B_t
    var inv_denom = 1.0 / denom

    # px0 = (x - B_t*pred) / denom
    var bt_pred = mul_scalar(pred, B_t, ctx)     # B_t * pred
    var x0_num = sub(x, bt_pred, ctx)            # x - B_t*pred
    var px0 = mul_scalar(x0_num, inv_denom, ctx)

    if s <= 0.0:
        return px0^

    # pxT = (x + A_t*pred) / denom
    var at_pred = mul_scalar(pred, A_t, ctx)     # A_t * pred
    var xT_num = add(x, at_pred, ctx)            # x + A_t*pred
    var pxT = mul_scalar(xT_num, inv_denom, ctx)

    var A_s = 1.0 - s / T
    var B_s = s / T
    var term0 = mul_scalar(px0, A_s, ctx)        # A_s * px0
    var termT = mul_scalar(pxT, B_s, ctx)        # B_s * pxT
    return add(term0, termT, ctx)                # A_s*px0 + B_s*pxT


# ─────────────────────────────────────────────────────────────────────────────
# Timestep schedule.  SD3-style shift applied to a linear arange(1, 0, -1/steps),
# scaled to T.  Returns (ts, s_all): ts[k] is the current timestep at step k,
# s_all[k] is the "next" timestep (ts[k+1], or 0.0 at the final step).
#
#   raw = 1 - k*(1/steps)                          # [1.0, .98, ...]
#   t   = shift*raw / (1 + (shift-1)*raw)          # SD3 shift transform
#   t   = t * T
# Host float compute (F64 accumulation for the divide, stored F32).
# ─────────────────────────────────────────────────────────────────────────────
def sampler_timesteps(
    T: Float32, steps: Int, shift: Float32
) -> Tuple[List[Float32], List[Float32]]:
    var ts = List[Float32]()
    var Td = Float64(T)
    var shd = Float64(shift)
    for k in range(steps):
        var raw = 1.0 - Float64(k) * (1.0 / Float64(steps))
        var t = shd * raw / (1.0 + (shd - 1.0) * raw)
        t = t * Td
        ts.append(Float32(t))

    var s_all = List[Float32]()
    for k in range(steps):
        if k < steps - 1:
            s_all.append(ts[k + 1])
        else:
            s_all.append(Float32(0.0))
    return (ts^, s_all^)
