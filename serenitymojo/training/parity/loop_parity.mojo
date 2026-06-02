# loop_parity.mojo — proves the training-loop harness (training/loop.mojo)
# trains, checkpoints BYTE-EXACT, and resumes without divergence.
#
# Regime: F32 master weights + BF16 compute (the proven mixed-precision path,
# generalized to a multi-step loop). Task: a tiny linear y = x @ W^T + b, MSE
# against a fixed target map — driven ENTIRELY through the TrainState harness.
#
# Compute (per step), host-threaded (Tensor is move-only — same idiom as
# train_skeleton.mojo / zimage_train_step.mojo):
#   1. w_bf = state.compute_weight(0), b_bf = state.compute_weight(1)   (BF16)
#   2. forward (BF16):  pre = linear(x_bf, w_bf, b_bf) ; loss = mse(pre, y_bf)
#   3. backward (BF16): d_pre = mse_backward(pre, y_bf)
#                       lg = linear_backward(d_pre, x_bf, w_bf, M,in,out)
#                       grads = [lg.d_w, lg.d_b]  (BF16)
#   4. state.accumulate_grads(grads) ; state.apply_step(lr)   (F32 master AdamW)
#
# Gates (all must hold, with tool output — Tenet 4, no faked descent):
#   GATE 1  TRAINS:        final loss < 0.5 * initial loss over ~50 steps.
#   GATE 2  ROUND-TRIPS:   save_checkpoint -> load_checkpoint; all F32 masters
#                          byte-exact (max_abs == 0) AND opt step t preserved.
#   GATE 3a RESUME MATCHES: loss on the loaded state == pre-save last loss
#                          (BF16 floor) — the loaded weights reproduce the run.
#   GATE 3b RESUME DESCENDS: one more apply_step on the loaded state keeps loss
#                          going down.
#
# Toolchain: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo run -I . serenitymojo/training/parity/loop_parity.mojo

from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.loss_swiglu_backward import mse_backward
from serenitymojo.training.loop import (
    TrainState, save_checkpoint, load_checkpoint,
)

comptime TArc = ArcPointer[Tensor]


comptime M = 4        # batch rows
comptime IN = 16      # in_features
comptime OUT = 8      # out_features
comptime LR = Float32(1.0e-2)
comptime N_STEPS = 50


# deterministic xavier-ish fill (mirrors mixed_precision_parity style)
def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var s = seed
    for _ in range(n):
        s = s * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(s >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


# host MSE for the printed scalar (matches mixed_precision_parity's host loss).
def _mse(pred: List[Float32], tgt: List[Float32]) -> Float32:
    var acc = Float32(0.0)
    for i in range(len(pred)):
        var d = pred[i] - tgt[i]
        acc += d * d
    return acc / Float32(len(pred))


def _absf(x: Float32) -> Float32:
    return x if x >= 0.0 else -x


# Multi-return for the move-only grad list (Mojo tuples need Copyable elements;
# a List[ArcPointer[Tensor]] is move-only, so we wrap it in a Movable struct —
# the same pattern BlockGrads / LinearGrads / FlowMatchOut use).
struct StepResult(Movable):
    var loss: Float32
    var grads: List[ArcPointer[Tensor]]

    def __init__(out self, loss: Float32, var grads: List[ArcPointer[Tensor]]):
        self.loss = loss
        self.grads = grads^


# One BF16 forward+backward through the harness; returns (loss, [d_w, d_b] BF16).
# The caller (harness) owns the optimizer; this owns the compute — the grads-as-
# input design of loop.mojo. Tensors rebuilt fresh from host data (move-only).
def forward_backward(
    state: TrainState, x_h: List[Float32], y_h: List[Float32], ctx: DeviceContext
) raises -> StepResult:
    var w_bf = state.compute_weight(0, ctx)        # [OUT,IN] BF16
    var b_bf = state.compute_weight(1, ctx)        # [OUT]    BF16
    var x_bf = cast_tensor(
        Tensor.from_host(x_h.copy(), [M, IN], STDtype.F32, ctx), STDtype.BF16, ctx)
    var y_bf = cast_tensor(
        Tensor.from_host(y_h.copy(), [M, OUT], STDtype.F32, ctx), STDtype.BF16, ctx)

    # forward: pre = linear(x, W, b) = x @ W^T + b  [M,OUT]
    var bias_opt = Optional[Tensor](b_bf.clone(ctx))
    var pre = linear(x_bf, w_bf, bias_opt^, ctx)
    var pre_h = pre.to_host(ctx)
    var loss = _mse(pre_h, y_h)

    # backward: d_pre = mse_backward(pre, y)  then linear_backward -> d_w, d_b
    var d_pre = mse_backward(pre, y_bf, ctx)       # BF16 [M,OUT]
    var lg = linear_backward(d_pre, x_bf, w_bf, M, IN, OUT, ctx)
    var grads = List[ArcPointer[Tensor]]()
    grads.append(ArcPointer(lg.d_w.clone(ctx)))    # [OUT,IN] BF16
    grads.append(ArcPointer(lg.d_b.clone(ctx)))    # [OUT]    BF16
    return StepResult(loss, grads^)


def main() raises:
    var ctx = DeviceContext()
    print("=== Loop Harness Parity (F32 master / BF16 compute) ===")
    print("task: linear y = x @ W^T + b, MSE ; M=", M, " IN=", IN, " OUT=", OUT)

    # ── init masters: W xavier-ish (F32), b zeros (F32) ──
    var w0 = Tensor.from_host(_fill(OUT * IN, 42, 1.0), [OUT, IN], STDtype.F32, ctx)
    var b0 = Tensor.from_host(_zeros(OUT), [OUT], STDtype.F32, ctx)
    var init_masters = List[ArcPointer[Tensor]]()
    init_masters.append(ArcPointer(w0^))
    init_masters.append(ArcPointer(b0^))
    var state = TrainState(init_masters^, ctx)

    # ── fixed synthetic dataset: y = x @ W_true^T + 0.1 ──
    var x_h = _fill(M * IN, 7, 2.0)
    var wt_true = _fill(OUT * IN, 99, 1.0)
    var y_h = List[Float32]()
    for r in range(M):
        for o in range(OUT):
            var acc = Float32(0.0)
            for c in range(IN):
                acc += x_h[r * IN + c] * wt_true[o * IN + c]
            y_h.append(acc + Float32(0.1))

    # ========================================================================
    # GATE 1: TRAINS — 50 steps, one micro-batch each (accumulate then apply).
    # ========================================================================
    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    print("\n--- training", N_STEPS, "steps ---")
    for step in range(N_STEPS):
        var r = forward_backward(state, x_h, y_h, ctx)
        var lv = r.loss
        state.accumulate_grads(r.grads, ctx)
        state.apply_step(LR, ctx)
        if step == 0:
            first_loss = lv
        last_loss = lv
        if step % 10 == 0:
            print("  step", step, " loss", lv)
    print("first_loss:", first_loss, " last_loss:", last_loss)
    var trains = last_loss < first_loss * Float32(0.5)

    # ========================================================================
    # GATE 2: CHECKPOINT ROUND-TRIPS BYTE-EXACT.
    # ========================================================================
    var ckpt = String("/tmp/serenitymojo_loop_ckpt.safetensors")
    save_checkpoint(state, ckpt, ctx)
    print("\n--- saved checkpoint:", ckpt)
    var loaded = load_checkpoint(ckpt, ctx)
    print("--- loaded, num_params:", loaded.num_params())

    var max_abs = Float32(0.0)
    if loaded.num_params() != state.num_params():
        print("FAIL: param count changed across round-trip")
    for i in range(state.num_params()):
        var orig = state.master_host(i, ctx)
        var got = loaded.master_host(i, ctx)
        if len(orig) != len(got):
            print("FAIL: numel changed for param", i)
        for j in range(len(orig)):
            var d = _absf(orig[j] - got[j])
            if d > max_abs:
                max_abs = d
    print("master round-trip max_abs:", max_abs)
    print("opt_step before save:", state.opt_step(), " after load:", loaded.opt_step())
    var roundtrip_exact = (max_abs == Float32(0.0)) and (loaded.opt_step() == state.opt_step())

    # ========================================================================
    # GATE 3: RESUME MATCHES + CONTINUES DESCENT.
    #
    # 3a — the loaded state must reproduce the SAVED weights' loss. The correct
    # baseline is the loss of the in-memory `state` measured on its CURRENT
    # (post-step-50) weights — NOT `last_loss`, which was step 50's loss measured
    # BEFORE step 50's optimizer update (a different, pre-update weight set). The
    # checkpoint correctly persists the post-update weights (max_abs==0 already
    # proved that), so loaded-loss must equal a fresh state-loss to the BF16 floor.
    # 3b — one more apply_step on the loaded state keeps the loss descending.
    # ========================================================================
    var r_orig = forward_backward(state, x_h, y_h, ctx)   # state @ post-step-50
    var orig_post_loss = r_orig.loss
    var r_pre = forward_backward(loaded, x_h, y_h, ctx)   # loaded @ post-step-50
    var resume_pre_loss = r_pre.loss
    var r2 = forward_backward(loaded, x_h, y_h, ctx)      # one more resumed step
    loaded.accumulate_grads(r2.grads, ctx)
    loaded.apply_step(LR, ctx)
    var r_post = forward_backward(loaded, x_h, y_h, ctx)
    var resume_post_loss = r_post.loss
    var loss_diff = _absf(resume_pre_loss - orig_post_loss)
    var tol = Float32(0.0001)
    print("\n--- resume ---")
    print("orig state loss (post-step-50 weights):", orig_post_loss)
    print("resumed loss (loaded, same weights):    ", resume_pre_loss)
    print("resumed loss (after one more step):     ", resume_post_loss)
    print("|resumed - orig| =", loss_diff, " tol =", tol)
    # 3a: identical weights (max_abs==0) → identical loss to the BF16 floor.
    var resume_matches = loss_diff <= tol
    var resume_descends = resume_post_loss <= resume_pre_loss

    # ========================================================================
    # VERDICT
    # ========================================================================
    print("\n=== VERDICT ===")
    print("GATE 1 trains (final < 0.5*initial):", trains)
    print("GATE 2 round-trip byte-exact (max_abs==0 & t kept):", roundtrip_exact)
    print("GATE 3a resume reproduces trajectory (|d|<1e-4):", resume_matches)
    print("GATE 3b resume continues descent:", resume_descends)
    if trains and roundtrip_exact and resume_matches and resume_descends:
        print("\nPASS: harness trains + checkpoints byte-exact + resumes")
    else:
        print("\nFAIL: see gates above")
        raise Error("loop_parity gate failed")
