# ops/activation1d.mojo — BigVGAN-v2 anti-aliased SnakeBeta activation
# (kaiser ratio-2 path), the LTX-2 vocoder critical-path op.
#
# LTX2_PORT_PLAN_2026-05-28 §P1. Ports the REFERENCE tensor-op path of the Rust
# `activation1d` (inference-flame/src/vae/ltx2_vocoder.rs:247-285), NOT the fused
# CUDA kernel. Bit-reproducible against the same Rust path because every step is
# a shipped, gated primitive (replicate_pad1d / zero_insert1d / conv1d from
# P-conv, snake_beta from P-snake).
#
# Algorithm (depthwise per-channel; folds C into batch via [B*C,1,L]):
#   x_bc      = x.reshape(B*C, 1, L)
#   # --- upsample 2x (ConvTranspose1d stride=2 kernel=12, decomposed) ---
#   x_pad     = replicate_pad1d(x_bc, 5, 5)
#   x_zi      = zero_insert1d(x_pad, 2)                 # (L+10-1)*2+1
#   x_padded  = pad1d(x_zi, 11, 11)        ZERO pad     # side = dil*(K-1)-pad = 11
#   y         = conv1d(x_padded, up_filter[1,1,12], stride=1, pad=0)
#   y         = y * 2.0                                 # ratio scale
#   y         = y[..., 15 : len-15]                     # slice pad_left/right
#   # --- snake (per-channel, params [C,1,1] broadcast over [C,1,L']) ---
#   y         = snake_beta(y, alpha_exp, inv_beta_eps)
#   # --- downsample 2x (regular conv1d stride=2) ---
#   y_pad     = replicate_pad1d(y, 5, 6)
#   out       = conv1d(y_pad, down_filter[1,1,12], stride=2, pad=0)
#   out       = out.reshape(B, C, L_out)
#
# The up/down filters are stored as [1,1,12]; here they are used as Conv1d
# weights [1, 1, 12] on the channel-folded [B*C,1,L] input (groups=1, single in
# channel). The ConvTranspose1d in Rust uses the SAME symmetric kaiser filter as
# a Conv1d weight after a no-op flip (symmetric) — so the reference path replaces
# the transpose with zero_insert + side-pad + plain conv1d (exactly conv1d's own
# conv_transpose1d decomposition, but inlined here because the filter is single-
# channel and used as-is, no flip/swap needed).
#
# B is assumed 1 (LTX-2 vocoder) so B*C == C and the [C,1,1] params broadcast
# directly over the folded [C,1,L'] tensor.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.ops.conv1d import conv1d, zero_insert1d, replicate_pad1d
from serenitymojo.ops.snake import snake_beta
from serenitymojo.ops.tensor_algebra import reshape, mul_scalar, slice


comptime _ACT_UP_REPLICATE_PAD = 5
comptime _ACT_UP_SIDE_PAD = 11  # dil*(K-1) - padding = 1*11 - 0
comptime _ACT_UP_SLICE_LEFT = 15
comptime _ACT_UP_SLICE_RIGHT = 15
comptime _ACT_DOWN_PAD_LEFT = 5
comptime _ACT_DOWN_PAD_RIGHT = 6
comptime _ACT_RATIO = 2


def activation1d(
    x: Tensor,
    alpha_exp: Tensor,
    inv_beta_eps: Tensor,
    up_filter: Tensor,
    down_filter: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """Anti-aliased SnakeBeta (kaiser ratio=2). Reference tensor-op path.

    x:            [B, C, L]                 (compute dtype; B assumed 1)
    alpha_exp:    [C, 1, 1]                 precomputed exp(alpha) per channel
    inv_beta_eps: [C, 1, 1]                 precomputed 1/(exp(beta)+1e-9)
    up_filter:    [1, 1, 12]                kaiser sinc upsample FIR
    down_filter:  [1, 1, 12]                kaiser sinc downsample FIR
    returns       [B, C, L]                 (anti-aliasing preserves length)
    """
    var xs = x.shape()
    if len(xs) != 3:
        raise Error("activation1d: x must be [B,C,L]")
    var B = xs[0]
    var C = xs[1]
    var L = xs[2]
    if B != 1:
        raise Error("activation1d: B must be 1 (depthwise fold assumes B=1)")

    # Fold channels into batch: [B,C,L] -> [B*C,1,L].
    var x_bc = reshape(x, [B * C, 1, L], ctx)

    # --- upsample 2x ---
    var x_pad = replicate_pad1d(
        x_bc, _ACT_UP_REPLICATE_PAD, _ACT_UP_REPLICATE_PAD, ctx
    )
    var x_zi = zero_insert1d(x_pad, _ACT_RATIO, ctx)
    # ZERO side-pad of 11 each side (conv1d's pad arg is symmetric zero pad).
    var y = conv1d(
        x_zi, up_filter, None, 1, _ACT_UP_SIDE_PAD, 1, 1, ctx
    )
    y = mul_scalar(y, Float32(_ACT_RATIO), ctx)
    # slice [15 : len-15]
    var y_len = y.shape()[2]
    var keep = y_len - _ACT_UP_SLICE_LEFT - _ACT_UP_SLICE_RIGHT
    if keep <= 0:
        raise Error("activation1d: upsample slice produced non-positive length")
    y = slice(y, 2, _ACT_UP_SLICE_LEFT, keep, ctx)

    # --- snake (per-channel; [C,1,1] broadcasts over [C,1,L']) ---
    y = snake_beta(y, alpha_exp, inv_beta_eps, ctx)

    # --- downsample 2x ---
    var y_pad = replicate_pad1d(y, _ACT_DOWN_PAD_LEFT, _ACT_DOWN_PAD_RIGHT, ctx)
    var out = conv1d(y_pad, down_filter, None, _ACT_RATIO, 0, 1, 1, ctx)

    # Unfold back to [B, C, L_out].
    var l_out = out.shape()[2]
    return reshape(out, [B, C, l_out], ctx)
