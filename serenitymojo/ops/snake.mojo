# ops/snake.mojo — SnakeBeta activation (BigVGAN vocoder), per-channel.
#
#   snake_beta(x, alpha_exp[1,C,1], inv_beta_eps[1,C,1])
#       = x + inv_beta_eps * sin²(alpha_exp * x)
#
# alpha / beta are stored LOG-SCALE and per-channel. The vocoder precomputes
# the broadcast-ready params ONCE at load (snake_beta_fast, ltx2_vocoder.rs:
# 157-168, 496-534):
#       alpha_exp     = exp(alpha)
#       inv_beta_eps  = 1 / (exp(beta) + 1e-9)
# stored in `[C,1,1]` (depthwise [B*C,1,L] layout) — here we expose them as
# `[1,C,1]` so they broadcast against an `[B,C,L]` activation under the
# NumPy right-aligned rule in tensor_algebra (size-1 axes get stride 0).
#
# The forward mirrors `snake_beta_fast` exactly (four pointwise launches):
#       ax     = x * alpha_exp        (broadcast mul)
#       sin_ax = sin(ax)              (P0 sin_op)
#       sin_sq = sin_ax * sin_ax      (elementwise mul)
#       scaled = sin_sq * inv_beta_eps(broadcast mul)
#       y      = x + scaled           (add)
# Built entirely on shipped, broadcast-correct kernels — ops/unary.sin_op and
# ops/tensor_algebra.{mul,add,add_scalar} — plus ops/unary.{exp_op,reciprocal_op}
# for the load-time precompute. No new GPU kernel is introduced; correctness
# rides on the already-gated P0 + tensor-algebra kernels.
#
# Gate: serenitymojo/ops/snake_smoke.mojo (host-F64 parity, BF16 storage).
# See serenitymojo/docs/LTX2_PORT_PLAN_2026-05-28.md §P-snake.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.unary import sin_op, exp_op, reciprocal_op
from serenitymojo.ops.tensor_algebra import mul, add, add_scalar


comptime _SNAKE_EPS = Float32(1e-9)


def snake_beta_precompute(
    alpha: Tensor, beta: Tensor, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    """Load-time precompute of the SnakeBeta broadcast params.

    `alpha` / `beta` are the raw LOG-SCALE per-channel weights (`[C]` or any
    shape that already broadcasts against the activation). Returns
        (alpha_exp, inv_beta_eps) = (exp(alpha), 1/(exp(beta) + 1e-9))
    in the SAME shape/dtype as the inputs. The +1e-9 lands INSIDE the
    reciprocal denominator (matches `beta_exp.add_scalar(1e-9).reciprocal()`,
    ltx2_vocoder.rs:524) so it caps the gain on near-zero exp(beta) and the
    op's own eps-clamp never engages.
    """
    var alpha_exp = exp_op(alpha, ctx)
    var beta_exp = exp_op(beta, ctx)
    var beta_exp_eps = add_scalar(beta_exp, _SNAKE_EPS, ctx)
    var inv_beta_eps = reciprocal_op(beta_exp_eps, ctx)
    return (alpha_exp^, inv_beta_eps^)


def snake_beta(
    x: Tensor, alpha_exp: Tensor, inv_beta_eps: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """SnakeBeta forward: x + inv_beta_eps · sin²(alpha_exp · x).

    `alpha_exp` / `inv_beta_eps` are the PRECOMPUTED params from
    `snake_beta_precompute` (exp already applied). They broadcast per-channel
    against `x` (e.g. x=[B,C,L] with params [1,C,1]). Mirrors `snake_beta_fast`
    (ltx2_vocoder.rs:162-168) launch-for-launch.
    """
    var ax = mul(x, alpha_exp, ctx)  # alpha_exp * x  (broadcast)
    var sin_ax = sin_op(ax, ctx)  # sin(alpha_exp * x)
    var sin_sq = mul(sin_ax, sin_ax, ctx)  # sin²
    var scaled = mul(sin_sq, inv_beta_eps, ctx)  # inv_beta_eps · sin²  (broadcast)
    var y = add(x, scaled, ctx)  # x + …
    return y^
