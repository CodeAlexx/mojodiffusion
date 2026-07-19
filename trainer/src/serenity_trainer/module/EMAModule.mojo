# ema.mojo — pure-Mojo port of Serenity's EMAModuleWrapper
# (modules/module/EMAModule.py).
#
# MEASURED semantics (EMAModule.py):
#   ema_parameters = [p.clone().detach() for p in parameters]            (:15)
#   get_current_decay(step) = min((1+step)/(10+step), decay)            (:31-35)
#   step(parameters, step):                                              (:38-53)
#     one_minus_decay = 1 - get_current_decay(step)                     (:41)
#     if (step + 1) % update_step_interval == 0:                        (:43)
#       for ema, p in zip(...): if p.requires_grad:                     (:44-45)
#         ema += one_minus_decay * (p - ema)   # lerp toward p          (:47)
#   copy_ema_to(parameters): parameter.data.copy_(ema)                  (:63-69)
#
# DTYPE POLICY: ema_parameters mirror the param dtype = BF16 STORAGE. The EMA
# update is computed in F32 REGISTERS and written back to BF16. Serenity keeps
# ema as bf16 (clone of a bf16 param) and never uses stochastic rounding for the
# EMA path (EMAModule.py uses a plain add_, no copy_stochastic_), so we round to
# nearest-even on write-back — matching torch's bf16 add_ accumulation.
#
# This is a per-tensor primitive: callers iterate their parameter list and call
# `ema_step` once per (ema, param) pair, exactly as EMAModule.step zips the two
# lists. The host owns the parameter↔ema correspondence (Serenity's zip).

from std.gpu.host import DeviceContext
from std.builtin.dtype import DType
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.torch_bf16 import torch_bf16_rne_value


# get_current_decay(step) = min((1+step)/(10+step), decay)  (EMAModule.py:31-35).
# Warms the effective decay up from ~0.1 at step 0 toward `decay`, so early
# updates track the params closely (the standard EMA warmup).
def ema_current_decay(step: Int, decay: Float32) -> Float32:
    var warm = Float32(1 + step) / Float32(10 + step)
    return warm if warm < decay else decay


# Whether the EMA should update on this step:
#   (step + 1) % update_step_interval == 0   (EMAModule.py:43).
def ema_should_update(step: Int, update_step_interval: Int) -> Bool:
    if update_step_interval <= 1:
        return True
    return (step + 1) % update_step_interval == 0


# Single-tensor EMA update, IN PLACE on `ema` (BF16). Mirrors
#   ema.add_(one_minus_decay * (param - ema))                 (EMAModule.py:47)
# i.e. ema += (1 - decay) * (param - ema), a lerp of ema toward param.
#
# Caller is responsible for the cadence / requires_grad gating (ema_should_update
# and the param.requires_grad check at EMAModule.py:43,:45); this kernel-like
# host op only performs the arithmetic for one tensor pair. F32 compute, BF16 out.
def ema_step(
    ema: Tensor,
    param: Tensor,
    step: Int,
    decay: Float32,
    ctx: DeviceContext,
) raises:
    if ema.dtype() != STDtype.BF16 or param.dtype() != STDtype.BF16:
        raise Error("ema_step: ema/param must be BF16 (port dtype policy)")
    var n = ema.numel()
    if param.numel() != n:
        raise Error("ema_step: numel mismatch")

    var one_minus_decay = Float32(1.0) - ema_current_decay(step, decay)

    var ef = ema.to_host(ctx)       # BF16 → F32 registers
    var pf = param.to_host(ctx)

    var nbytes = n * 2              # BF16 = 2 bytes/elem
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var bp = host.unsafe_ptr().bitcast[BFloat16]()
    for i in range(n):
        var new_ema = ef[i] + one_minus_decay * (pf[i] - ef[i])
        # RNE bf16 via CUDA-parity helper (no SR in EMA path). Mojo's native
        # cast[bfloat16] differs by one BF16 quantum on some values vs torch's
        # add_ accumulation (torch_bf16.mojo:4-7); EMA accumulates over many steps
        # so the per-write quantum error matters here.
        bp[i] = torch_bf16_rne_value(new_ema)
    ctx.enqueue_copy(dst_buf=ema.buf, src_buf=host)
    ctx.synchronize()


# copy_ema_to: parameter.data.copy_(ema)  (EMAModule.py:63-69). Writes the EMA
# weights into `param` IN PLACE (used at validation/save). Plain bf16 copy.
def ema_copy_to(
    param: Tensor,
    ema: Tensor,
    ctx: DeviceContext,
) raises:
    if param.dtype() != STDtype.BF16 or ema.dtype() != STDtype.BF16:
        raise Error("ema_copy_to: param/ema must be BF16 (port dtype policy)")
    var n = param.numel()
    if ema.numel() != n:
        raise Error("ema_copy_to: numel mismatch")
    ctx.enqueue_copy(dst_buf=param.buf, src_buf=ema.buf)
    ctx.synchronize()


# Initialize an EMA buffer from a param: ema = p.clone().detach()
# (EMAModule.py:15). Returns a fresh BF16 tensor (own device buffer, id=0).
def ema_init(param: Tensor, ctx: DeviceContext) raises -> Tensor:
    if param.dtype() != STDtype.BF16:
        raise Error("ema_init: param must be BF16 (port dtype policy)")
    return param.clone(ctx)
