# grad.mojo — gradient accumulation + global-norm clipping, ported from
# Serenity's GenericTrainer.train update path:
#   loss = loss / gradient_accumulation_steps   (scale BEFORE backward)
#   loss.backward()                             (grads accumulate in .grad)
#   on update step: clip_grad_norm_(params, clip_grad_norm); optimizer.step()
#
# Here grads come out of the tape per micro-step; the driver scales each by
# 1/accum (or scales the seed) and accumulates with `accumulate`. Clipping mirrors
# torch.nn.utils.clip_grad_norm_: total_norm = sqrt(Σ Σ g²) over ALL grads; if
# total_norm > max, multiply every grad by max/(total_norm + 1e-6).
#
# Dtype: grads are BF16 storage. Sum-of-squares is F32-ACCUMULATED (reduce_sum_f32)
# — a host statistic, not stored model state. The clip multiply stays BF16.

from std.math import sqrt
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import add_in_place, mul, mul_scalar
from serenitymojo.ops.reduce import reduce_sum_f32

# Tensor is move-only → cannot be a List element. Grads are boxed in ArcPointer
# (the same idiom the tape's backward() uses: Dict[Int, ArcPointer[Tensor]]).
comptime TArc = ArcPointer[Tensor]


# dst += src, in place (BF16). Matches torch grad accumulation.
def accumulate(dst: Tensor, src: Tensor, ctx: DeviceContext) raises:
    add_in_place(dst, src, ctx)


# Σ g² for one grad tensor, returned as a host F32 scalar (F32-accumulated).
def _sum_sq(g: Tensor, ctx: DeviceContext) raises -> Float32:
    var sq = mul(g, g, ctx)                       # BF16 storage, F32-accum inside
    var dims = List[Int]()
    for i in range(len(sq.shape())):
        dims.append(i)
    var s = reduce_sum_f32(sq, dims^, False, ctx)  # F32 scalar tensor
    var host = s.to_host(ctx)
    return host[0]


# Global L2 norm over ALL grads (sqrt of summed sum-of-squares).
def global_grad_norm(grads: List[TArc], ctx: DeviceContext) raises -> Float32:
    var total = Float32(0.0)
    for i in range(len(grads)):
        total = total + _sum_sq(grads[i][], ctx)
    return sqrt(total)


# Scale every grad by `factor` in place-returning (BF16). Used by clipping AND by
# loss weighting (the tape's MSE arm ignores the loss seed, so a per-step loss
# weight is applied here as a grad scale — see loss.mojo).
def scale_grads(mut grads: List[TArc], factor: Float32, ctx: DeviceContext) raises:
    for i in range(len(grads)):
        grads[i] = TArc(mul_scalar(grads[i][], factor, ctx))


# clip_grad_norm_: if total_norm > max_norm, scale all grads by
# max_norm/(total_norm+1e-6). Returns the pre-clip total norm (for logging, like
# Serenity's grad_norm board scalar). max_norm <= 0 disables clipping.
def clip_grad_norm(
    mut grads: List[TArc], max_norm: Float32, ctx: DeviceContext
) raises -> Float32:
    var total_norm = global_grad_norm(grads, ctx)
    if max_norm > 0.0 and total_norm > max_norm:
        var clip_coef = max_norm / (total_norm + Float32(1e-6))
        scale_grads(grads, clip_coef, ctx)
    return total_norm
