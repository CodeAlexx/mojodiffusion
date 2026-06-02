# models/sdxl/sampling.mojo — SDXL UNet Down/Up sampling fwd+bwd.
#
# ARCHITECTURE ONLY (Tenet 1): composes already-gated ops/ primitives —
# conv2d (fwd) / conv2d_backward (bwd), upsample_nearest2x_nhwc (fwd) /
# upsample_nearest2d_backward (bwd). No new primitive inline.
#
# ── Downsample (sdxl_unet.rs conv2d_forward `.op` branch: stride-2 pad-1 3x3) ──
#   FORWARD:  y = Conv3x3(x, op_w, op_b, stride=2, pad=1)          # H,W -> H/2,W/2
#   BACKWARD: (d_x, d_op_w, d_op_b) = conv2d_backward(x, op_w, go) # stride 2
#
# ── Upsample (sdxl_unet.rs: nearest-2x then `.conv` stride-1 pad-1 3x3) ────────
#   FORWARD:  u = UpsampleNearest2x(x)            # H,W -> 2H,2W
#             y = Conv3x3(u, conv_w, conv_b, stride=1, pad=1)
#   BACKWARD: (d_u, d_conv_w, d_conv_b) = conv2d_backward(u, conv_w, go)
#             d_x = upsample_nearest2d_backward(d_u)   # sum-pool the 2x2 grad
#
# All NHWC, F32. Conv filters RSCF [Kh,Kw,Cin,Cout].

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from std.collections import Optional

from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.conv2d_backward import conv2d_backward
from serenitymojo.models.vae.upsample import upsample_nearest2x_nhwc
from serenitymojo.ops.pool_backward import upsample_nearest2d_backward


# ── Downsample backward grads ─────────────────────────────────────────────────
struct SampleGrads(Movable):
    """d_x + conv weight/bias grads of a Down/Up sample step."""
    var d_x: Tensor
    var d_w: Tensor
    var d_b: Tensor

    def __init__(out self, var d_x: Tensor, var d_w: Tensor, var d_b: Tensor):
        self.d_x = d_x^
        self.d_w = d_w^
        self.d_b = d_b^


# ══════════════════════════ DOWNSAMPLE ════════════════════════════════════════
# FORWARD: stride-2 pad-1 Conv3x3. Hi must be even; Ho = Hi//2 (pad 1, k3, s2).
def downsample_forward[
    N: Int, Hi: Int, Wi: Int, C: Int,
](
    x: Tensor, op_w: Tensor, op_b: Tensor, ctx: DeviceContext,
) raises -> Tensor:
    return conv2d[N, Hi, Wi, C, 3, 3, C, 2, 2, 1, 1](
        x, op_w.clone(ctx), Optional[Tensor](op_b.clone(ctx)), ctx
    )


# BACKWARD: conv2d_backward at stride 2 (gated by conv2d_bwd_s2_parity).
# go: dL/dy [N, Hi/2, Wi/2, C].  x: forward input [N,Hi,Wi,C].
def downsample_backward[
    N: Int, Hi: Int, Wi: Int, C: Int,
](
    go: Tensor, x: Tensor, op_w: Tensor, ctx: DeviceContext,
) raises -> SampleGrads:
    var g = conv2d_backward[N, Hi, Wi, C, 3, 3, C, 2, 2, 1, 1](
        x, op_w, go, ctx
    )
    return SampleGrads(g.d_x.clone(ctx), g.d_w.clone(ctx), g.d_b.clone(ctx))


# ══════════════════════════ UPSAMPLE ══════════════════════════════════════════
# FORWARD: nearest-2x then stride-1 pad-1 Conv3x3.
#   input x [N,Hi,Wi,C] -> upsample [N,2Hi,2Wi,C] -> conv (same C) [N,2Hi,2Wi,C].
struct UpsampleFwd(Movable):
    var out: Tensor   # [N,2Hi,2Wi,C]
    var up: Tensor    # saved upsampled activation (conv input) for backward

    def __init__(out self, var out: Tensor, var up: Tensor):
        self.out = out^
        self.up = up^


def upsample_forward[
    N: Int, Hi: Int, Wi: Int, C: Int,
](
    x: Tensor, conv_w: Tensor, conv_b: Tensor, ctx: DeviceContext,
) raises -> UpsampleFwd:
    var up = upsample_nearest2x_nhwc(x, ctx)   # [N,2Hi,2Wi,C]
    var y = conv2d[N, 2 * Hi, 2 * Wi, C, 3, 3, C, 1, 1, 1, 1](
        up.clone(ctx), conv_w.clone(ctx), Optional[Tensor](conv_b.clone(ctx)), ctx
    )
    return UpsampleFwd(y^, up^)


# BACKWARD: conv bwd (stride 1) on the upsampled activation, then sum-pool the
# 2x2 grad back to the original resolution.
#   go: dL/dy [N,2Hi,2Wi,C].  up: saved upsampled conv-input [N,2Hi,2Wi,C].
def upsample_backward[
    N: Int, Hi: Int, Wi: Int, C: Int,
](
    go: Tensor, up: Tensor, conv_w: Tensor, ctx: DeviceContext,
) raises -> SampleGrads:
    var g = conv2d_backward[N, 2 * Hi, 2 * Wi, C, 3, 3, C, 1, 1, 1, 1](
        up, conv_w, go, ctx
    )
    var d_conv_w = g.d_w.clone(ctx)
    var d_conv_b = g.d_b.clone(ctx)
    # d_up [N,2Hi,2Wi,C] -> d_x [N,Hi,Wi,C] (inverse of nearest-2x replication)
    var d_x = upsample_nearest2d_backward[N, Hi, Wi, C, 2](g.d_x.clone(ctx), ctx)
    return SampleGrads(d_x^, d_conv_w^, d_conv_b^)
