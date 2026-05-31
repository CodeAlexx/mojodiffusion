# serenitymojo/training/parity/train_skeleton.mojo
#
# WALKING-SKELETON TRAINING PROOF (NO tape / NO autograd.mojo).
#
# PURPOSE: prove the whole training stack descends — forward → loss → hand-
# chained backward → optimizer step — for N steps, with LOSS ACTUALLY
# DECREASING. This is NOT a single-step parity number; it is a real multi-step
# gradient-descent run on a tiny network. If the chained backward or the
# optimizer is wrong, the loss will stall or diverge and this file FAILS.
#
# TASK: a fixed tiny regression. Fixed random-ish input X[B,IN] and target
# T[B,OUT]; learn a 2-layer MLP to map X→T by minimizing MSE.
#
# MODEL (production ops, F32 throughout):
#   h1 = linear(X, W1, b1)       X:[B,IN]  W1:[HID,IN] b1:[HID] -> h1:[B,HID]   (pre-silu)
#   a  = silu(h1)                                                 -> a:[B,HID]
#   y  = linear(a, W2, b2)       W2:[OUT,HID] b2:[OUT]            -> y:[B,OUT]
#   L  = mse(y, T) = mean((y-T)^2)
#
# BACKWARD (real per-op backward kernels, hand-threaded, reverse order — NO tape):
#   dy   = 2*(y-T)/numel                              (inline MSE leaf grad)
#   (d_a, d_W2, d_b2) = linear_backward(dy, a, W2)    (ops/linalg_backward.mojo)
#   d_h1 = silu_backward(d_a, h1)                     (ops/activation_backward.mojo;
#                                                      takes the PRE-silu h1)
#   (d_X, d_W1, d_b1) = linear_backward(d_h1, X, W1)  (d_X unused — X is data)
#
# OPTIMIZER: AdamW (decoupled WD), one step per parameter (W1,b1,W2,b2), each
# with its own (m,v) state and the SHARED 1-based step counter t (training/optim.mojo).
#
# GATE (real descent — Tenet 4, measurement beats assertion):
#   * final loss < 0.5 * initial loss   (meaningful descent, not a single number)
#   * loss is monotone-ish decreasing   (no step rises by more than a small slack)
# Both must hold or the VERDICT is "stalls"/"diverges" and the trajectory is the
# evidence. There is NO faked curve: the printed losses come from the real ops.
#
# DETERMINISTIC INIT: weights/data are a sinusoidal fill  v(i) = scale*sin(0.1*i+phase)
# reproduced byte-for-byte by train_skeleton_oracle.py (torch), so the Mojo loss
# trajectory can be cross-checked against torch's (cos of the loss-vs-step curve
# >= 0.99) — proving not just "loss goes down" but "it descends like torch does".
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/training/parity/train_skeleton.mojo

from std.math import sin, sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.activation_backward import silu_backward
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.training.optim import adamw_step
from std.collections import List, Optional


comptime B = 8
comptime IN = 4
comptime HID = 8
comptime OUT = 3
comptime STEPS = 200

# AdamW hyperparameters (mirror train_skeleton_oracle.py).
comptime LR = Float32(1e-2)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime ADAM_EPS = Float32(1e-8)
comptime WD = Float32(0.0)


# Deterministic fill v(i) = scale*sin(0.1*i + phase). Mirrors the torch oracle.
def _fill(n: Int, scale: Float32, phase: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(scale * sin(Float32(0.1) * Float32(i) + phase))
    return out^


# Build a fresh F32 zero buffer of length n (for Adam m/v state).
def _zeros(n: Int, ctx: DeviceContext) raises -> Tensor:
    var z = List[Float32]()
    for _ in range(n):
        z.append(Float32(0.0))
    var sh = List[Int]()
    sh.append(n)
    return Tensor.from_host(z^, sh^, STDtype.F32, ctx)


# Forward pass over fresh tensors built from host param lists -> (y_host, h1_host).
# Tensors are move-only: each op consumes its inputs, so we pass from_host
# rvalues inline (the proven composed_chain_parity pattern). Returns the output
# y and the PRE-silu activation h1 (silu_backward needs the pre-silu input).
def forward(
    x_h: List[Float32],
    w1_h: List[Float32],
    b1_h: List[Float32],
    w2_h: List[Float32],
    b2_h: List[Float32],
    ctx: DeviceContext,
) raises -> Tuple[List[Float32], List[Float32]]:
    var bias1 = Optional[Tensor](
        Tensor.from_host(b1_h, [HID], STDtype.F32, ctx)
    )
    var h1 = linear(
        Tensor.from_host(x_h, [B, IN], STDtype.F32, ctx),
        Tensor.from_host(w1_h, [HID, IN], STDtype.F32, ctx),
        bias1^, ctx,
    )                                                   # [B, HID] pre-silu
    var h1_host = h1.to_host(ctx)
    var a = silu(h1^, ctx)                              # [B, HID]
    var bias2 = Optional[Tensor](
        Tensor.from_host(b2_h, [OUT], STDtype.F32, ctx)
    )
    var y = linear(a^, Tensor.from_host(w2_h, [OUT, HID], STDtype.F32, ctx), bias2^, ctx)
    var y_host = y.to_host(ctx)                         # [B, OUT]
    return (y_host^, h1_host^)


# MSE loss (host reduction; gradients use kernels).
def _mse(pred: List[Float32], tgt: List[Float32]) -> Float32:
    var acc = Float32(0.0)
    for i in range(len(pred)):
        var d = pred[i] - tgt[i]
        acc += d * d
    return acc / Float32(len(pred))


def main() raises:
    var ctx = DeviceContext()
    print("==== train_skeleton (walking-skeleton training proof, NO tape) ====")
    print("model: linear -> silu -> linear ; loss: mse ; optim: AdamW")
    print("B=", B, " IN=", IN, " HID=", HID, " OUT=", OUT, " STEPS=", STEPS, " LR=", LR)

    # ---- Fixed data (deterministic fills, mirrored by the torch oracle). ----
    var X_h = _fill(B * IN, Float32(1.0), Float32(0.0))
    var T_h = _fill(B * OUT, Float32(0.7), Float32(1.3))

    # ---- Parameters (host master copies, updated each step via readback). ----
    # Tensor is move-only and AdamW updates IN PLACE on device buffers, so each
    # parameter is a PERSISTENT device Tensor (var) re-passed every step. We keep
    # the host lists only for the initial upload + the forward rebuilds.
    var W1_h = _fill(HID * IN, Float32(0.5), Float32(0.5))
    var b1_h = _fill(HID, Float32(0.1), Float32(0.2))
    var W2_h = _fill(OUT * HID, Float32(0.5), Float32(0.9))
    var b2_h = _fill(OUT, Float32(0.1), Float32(0.4))

    # Persistent device parameters (AdamW mutates these in place).
    var W1 = Tensor.from_host(W1_h, [HID, IN], STDtype.F32, ctx)
    var b1 = Tensor.from_host(b1_h, [HID], STDtype.F32, ctx)
    var W2 = Tensor.from_host(W2_h, [OUT, HID], STDtype.F32, ctx)
    var b2 = Tensor.from_host(b2_h, [OUT], STDtype.F32, ctx)

    # Adam moment state (m, v) per parameter — persistent, zero-init.
    var mW1 = _zeros(HID * IN, ctx)
    var vW1 = _zeros(HID * IN, ctx)
    var mb1 = _zeros(HID, ctx)
    var vb1 = _zeros(HID, ctx)
    var mW2 = _zeros(OUT * HID, ctx)
    var vW2 = _zeros(OUT * HID, ctx)
    var mb2 = _zeros(OUT, ctx)
    var vb2 = _zeros(OUT, ctx)

    var losses = List[Float32]()

    for step in range(STEPS):
        # Read current params to host for the forward rebuild (move-only ops).
        var w1c = W1.to_host(ctx)
        var b1c = b1.to_host(ctx)
        var w2c = W2.to_host(ctx)
        var b2c = b2.to_host(ctx)

        # ---- FORWARD ----
        var fwd = forward(X_h, w1c, b1c, w2c, b2c, ctx)
        var y_h = fwd[0].copy()
        var h1_h = fwd[1].copy()

        var loss = _mse(y_h, T_h)
        losses.append(loss)

        # ---- BACKWARD (hand-chained, reverse order, NO tape) ----
        # dy = 2*(y-T)/numel  (MSE leaf grad)
        var numel = Float32(B * OUT)
        var dy_h = List[Float32]()
        for i in range(len(y_h)):
            dy_h.append(Float32(2.0) * (y_h[i] - T_h[i]) / numel)

        # Need `a = silu(h1)` on device for linear2 backward (it is layer-2's x).
        var a_dev = silu(Tensor.from_host(h1_h, [B, HID], STDtype.F32, ctx), ctx)
        var a_h = a_dev.to_host(ctx)

        # linear2 backward: grads wrt a, W2, b2.
        var lb2 = linear_backward(
            Tensor.from_host(dy_h, [B, OUT], STDtype.F32, ctx),   # grad_y
            Tensor.from_host(a_h, [B, HID], STDtype.F32, ctx),    # x = a
            Tensor.from_host(w2c, [OUT, HID], STDtype.F32, ctx),  # weight = W2
            B, HID, OUT, ctx,
        )
        var d_a_h = lb2.d_x.to_host(ctx)        # dL/da   [B, HID]
        var dW2_h = lb2.d_w.to_host(ctx)        # dL/dW2  [OUT, HID]
        var db2_h = lb2.d_b.to_host(ctx)        # dL/db2  [OUT]

        # silu backward: d_h1 = d_a * silu'(h1)  (takes the PRE-silu h1).
        var d_h1_dev = silu_backward(
            Tensor.from_host(d_a_h, [B, HID], STDtype.F32, ctx),  # grad_out = d_a
            Tensor.from_host(h1_h, [B, HID], STDtype.F32, ctx),   # x = pre-silu h1
            ctx,
        )
        var d_h1_h = d_h1_dev.to_host(ctx)      # dL/dh1  [B, HID]

        # linear1 backward: grads wrt W1, b1 (d_x wrt X is unused — X is data).
        var lb1 = linear_backward(
            Tensor.from_host(d_h1_h, [B, HID], STDtype.F32, ctx),  # grad_y
            Tensor.from_host(X_h, [B, IN], STDtype.F32, ctx),      # x = X
            Tensor.from_host(w1c, [HID, IN], STDtype.F32, ctx),    # weight = W1
            B, IN, HID, ctx,
        )
        var dW1_h = lb1.d_w.to_host(ctx)        # dL/dW1  [HID, IN]
        var db1_h = lb1.d_b.to_host(ctx)        # dL/db1  [HID]

        # ---- OPTIMIZER STEP (AdamW, in place, shared 1-based counter t) ----
        var t = step + 1
        var gW1 = Tensor.from_host(dW1_h, [HID, IN], STDtype.F32, ctx)
        adamw_step(W1, gW1, mW1, vW1, t, LR, BETA1, BETA2, ADAM_EPS, WD, ctx)
        var gb1 = Tensor.from_host(db1_h, [HID], STDtype.F32, ctx)
        adamw_step(b1, gb1, mb1, vb1, t, LR, BETA1, BETA2, ADAM_EPS, WD, ctx)
        var gW2 = Tensor.from_host(dW2_h, [OUT, HID], STDtype.F32, ctx)
        adamw_step(W2, gW2, mW2, vW2, t, LR, BETA1, BETA2, ADAM_EPS, WD, ctx)
        var gb2 = Tensor.from_host(db2_h, [OUT], STDtype.F32, ctx)
        adamw_step(b2, gb2, mb2, vb2, t, LR, BETA1, BETA2, ADAM_EPS, WD, ctx)

        if step % 10 == 0 or step == STEPS - 1:
            print("step", step, " loss", loss)

    # ---- TRAJECTORY + GATES ----
    var initial = losses[0]
    var final = losses[len(losses) - 1]
    var ratio = final / initial

    print("")
    print("---- trajectory (every 10 steps) ----")
    var traj = List[Float32]()
    var s = 0
    while s < STEPS:
        traj.append(losses[s])
        print("  step", s, " loss", losses[s])
        s += 10
    print("INITIAL", initial)
    print("FINAL", final)
    print("RATIO (final/initial)", ratio)

    # Gate 1: meaningful descent — final < 0.5 * initial.
    var pass_descent = final < Float32(0.5) * initial

    # Gate 2: monotone-ish — allow a small upward slack per step (Adam can bump
    # slightly near convergence). Flag a real divergence (a large sustained rise).
    var max_rise = Float32(0.0)
    for i in range(1, len(losses)):
        var rise = losses[i] - losses[i - 1]
        if rise > max_rise:
            max_rise = rise
    # "small": no single step may rise by more than 25% of the initial loss.
    var pass_monotone = max_rise < Float32(0.25) * initial

    # NaN/Inf guard (a diverged run shows up as a non-finite final loss).
    var finite = (final == final) and (final < Float32(1e30))

    print("")
    print("max single-step rise =", max_rise, " (slack =", Float32(0.25) * initial, ")")
    print("")
    if pass_descent and pass_monotone and finite:
        print("VERDICT: TRAINS (real multi-step descent — stack is sound)")
        print("  final loss is", ratio, "x the initial loss")
    else:
        print("VERDICT: DOES NOT TRAIN")
        if not finite:
            print("  -> loss is NaN/Inf — backward or optimizer diverged")
        if not pass_descent:
            print("  -> no meaningful descent (final >= 0.5*initial): STALLS")
        if not pass_monotone:
            print("  -> loss spikes upward (max rise", max_rise, "): DIVERGES")
