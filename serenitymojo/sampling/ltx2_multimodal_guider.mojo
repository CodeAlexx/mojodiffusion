# sampling/ltx2_multimodal_guider.mojo — LTX-2 MultiModalGuider.calculate port.
#
# 1:1 port of `MultiModalGuider.calculate` from
# /home/alex/LTX-2/packages/ltx-core/src/ltx_core/components/guiders.py:244-268:
#
#     pred = cond
#          + (cfg_scale - 1)      * (cond - uncond_text)
#          + stg_scale            * (cond - uncond_perturbed)
#          + (modality_scale - 1) * (cond - uncond_modality)
#     if rescale_scale != 0:
#         factor = cond.std() / pred.std()
#         factor = rescale_scale * factor + (1 - rescale_scale)
#         pred = pred * factor
#
# HQ recipe params (ltx_pipelines/utils/constants.py LTX_2_3_HQ_PARAMS):
#   video: cfg=3.0  stg=0.0  rescale=0.45  modality=3.0
#   audio: cfg=7.0  stg=0.0  rescale=1.0   modality=3.0
#
# stg_scale is carried in the signature for spec fidelity but MUST be 0 here:
# the perturbed (ptb) pass tensor is not threaded through this entry point
# (HQ runs stg=0, so the reference never evaluates that pass either). Fail-loud
# if a non-zero stg sneaks in.
#
# dtype-contract: inputs may be BF16 or F32; all arithmetic (including the
# std() reductions) runs in an F32 workspace and the result is cast back to the
# input dtype. std() is the torch default: UNBIASED (N-1) sample std over the
# WHOLE tensor, accumulated in F64 on host for the two scalar reductions (the
# reference accumulates bf16->f32 on GPU; F64 host accumulation is a strict
# precision superset and is covered by the unit gate cos >= 0.99999).

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar


def _std_whole_tensor(x: Tensor, ctx: DeviceContext) raises -> Float64:
    """Unbiased (N-1) sample std over the whole tensor (torch `.std()`)."""
    var h = x.to_host(ctx)
    var n = len(h)
    if n < 2:
        raise Error("ltx2_multimodal_guider: std needs >= 2 elements")
    var s = 0.0
    var s2 = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
    var mean = s / Float64(n)
    var var_ = (s2 - Float64(n) * mean * mean) / Float64(n - 1)
    if var_ < 0.0:
        var_ = 0.0
    return sqrt(var_)


def guider_calculate(
    cond: Tensor,
    uncond: Tensor,
    mod: Tensor,
    cfg: Float32,
    stg_scale: Float32,
    rescale: Float32,
    mod_scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """MultiModalGuider.calculate for the HQ 3-pass case (cond, uncond, mod).

    Returns the guided prediction in `cond`'s dtype. See module header for the
    exact reference formula and the stg_scale=0 restriction."""
    if stg_scale != Float32(0.0):
        raise Error(
            "guider_calculate: stg_scale != 0 requires the perturbed (ptb) "
            "pass tensor, which is not threaded here (HQ runs stg=0) — "
            "fail-closed"
        )
    var out_dtype = cond.dtype()
    var c = cast_tensor(cond, STDtype.F32, ctx)
    var u = cast_tensor(uncond, STDtype.F32, ctx)
    var m = cast_tensor(mod, STDtype.F32, ctx)

    # pred = c + (cfg-1)*(c-u) + (mod_scale-1)*(c-m)
    var pred = add(
        c,
        add(
            mul_scalar(sub(c, u, ctx), cfg - Float32(1.0), ctx),
            mul_scalar(sub(c, m, ctx), mod_scale - Float32(1.0), ctx),
            ctx,
        ),
        ctx,
    )

    if rescale != Float32(0.0):
        var std_cond = _std_whole_tensor(c, ctx)
        var std_pred = _std_whole_tensor(pred, ctx)
        var factor = Float64(rescale) * (std_cond / std_pred) + (
            1.0 - Float64(rescale)
        )
        pred = mul_scalar(pred, Float32(factor), ctx)

    if out_dtype == STDtype.F32:
        return pred^
    return cast_tensor(pred, out_dtype, ctx)
