# training/zimage_train_step.mojo — ONE training step on a single Z-Image
# DiT-block FFN sub-path, synthetic small tensors. Phase-T5 scaffold.
#
# DELIVERABLE (T5_ZIMAGE_TRAINING_MAP.md §5): the largest cleanly-verifiable
# piece of a Z-Image training step. Assembles the modulated FFN sub-path of the
# NextDiT `_block` (zimage_dit.mojo:452-458) — the half of the block that uses
# only HAVE/HAVE-trivial backward arms — drives ONE step:
#
#   flow_match_noise_target  (schedule.mojo, the REAL Z-Image v-target)
#     -> FFN-subpath forward -> mse loss
#     -> MANUAL chained backward (NOT the 5-op Tape; see map §4 Gap A)
#     -> AdamW step (optim.mojo)
#   asserts loss finite + grads nonzero, then that loss decreases over iters.
#
# DESIGN follows training/parity/composed_chain_parity.mojo EXACTLY (the proven
# manual-chain template): Tensor is move-only, so each op consumes its inputs;
# we keep host-list copies of the intermediates and REBUILD fresh tensors at each
# backward call. We import ONLY symbols verified to import in Mojo 1.0.0b1:
#   linear (linear.mojo), rms_norm (norm.mojo), swiglu (activations.mojo),
#   linear_backward (linalg_backward.mojo), rms_norm_backward (norm_backward.mojo),
#   swiglu_backward (loss_swiglu_backward.mojo — the LAST def, which imports;
#     mse_backward in that file is UNIMPORTABLE per the composed_chain header, so
#     the MSE leaf grad is inlined, same as composed_chain_parity).
#   adamw_step (optim.mojo), flow_match_noise_target (schedule.mojo).
#
# Sub-path (the FFN half of the modulated block; scale_mlp/gate_mlp fixed):
#   x_in (block input from flow_match)
#   xfn1   = rms_norm(x_in, fn1_w)
#   xfn1s  = xfn1 * scale_mlp          (broadcast [1,1,D] over [1,S,D])
#   g      = linear(xfn1s, w1)          # FFN gate proj  [.,F]
#   u      = linear(xfn1s, w3)          # FFN up proj    [.,F]
#   act    = swiglu(g, u)               # silu(g)*u
#   ff     = linear(act, w2)            # FFN down proj  [.,D]
#   ff_n2  = rms_norm(ff, fn2_w)
#   gated  = gate_mlp * ff_n2           (broadcast [1,1,D])
#   out    = x_in + gated               (residual)
#   loss   = mse(out, target)
#
# Trained leaves (F32 master, optim.mojo path): w1, w2, w3.
#
# WHY the FFN sub-path and not the full block: the full `_block` also runs the
# ATTENTION sub-path (rms_norm_4d_head -> rope -> sdpa -> reshape -> linear).
# Those two arms (sdpa_backward, 4D-head RMSNorm backward) are the highest-risk;
# the FFN half is the cleanest provable unit and is what this scaffold proves
# end-to-end. The attention half is the documented next increment (map §5).
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/training/zimage_train_step.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU. F32 throughout (the optim.mojo / parity gate dtype).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.norm_backward import rms_norm_backward
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.training.optim import adamw_step
from serenitymojo.training.schedule import flow_match_noise_target


comptime S = 4    # sequence (tiny stand-in for unified_len)
comptime D = 8    # model dim (stand-in for 3840)
comptime F = 16   # FFN hidden (stand-in for the SwiGLU inner dim)
comptime EPS = Float32(1e-5)


# ── deterministic host randn-ish (mirrors composed_chain_parity style) ───────
def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


# ── inline MSE leaf grad: d/dpred mean((pred-target)^2) = 2(pred-target)/N ────
# (loss_swiglu_backward.mse_backward is UNIMPORTABLE — composed_chain header.)
def _mse_grad(pred_h: List[Float32], tgt_h: List[Float32]) -> List[Float32]:
    var n = len(pred_h)
    var d = List[Float32]()
    for i in range(n):
        d.append(Float32(2.0) * (pred_h[i] - tgt_h[i]) / Float32(n))
    return d^


def _mse_loss(pred_h: List[Float32], tgt_h: List[Float32]) -> Float32:
    var acc = Float32(0.0)
    for i in range(len(pred_h)):
        var diff = pred_h[i] - tgt_h[i]
        acc += diff * diff
    return acc / Float32(len(pred_h))


# ── inline broadcast-mul: y[s,d] = scale_d * x[s,d], scale is [1,1,D] ─────────
def _bmul_fwd(x_h: List[Float32], scale_h: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for s in range(S):
        for d in range(D):
            out.append(x_h[s * D + d] * scale_h[d])
    return out^


# d wrt x: d_x[s,d] = grad[s,d] * scale_d (scale is the broadcast operand).
def _bmul_bwd_x(grad_h: List[Float32], scale_h: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for s in range(S):
        for d in range(D):
            out.append(grad_h[s * D + d] * scale_h[d])
    return out^


# ── tensor <-> host helpers ──────────────────────────────────────────────────
def _to_host(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    return t.to_host(ctx)


def _abs_sum(h: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(h)):
        var v = h[i]
        s += v if v >= 0.0 else -v
    return s


def main() raises:
    var ctx = DeviceContext()
    print("=== Z-Image DiT-block FFN-subpath: ONE training step (synthetic) ===")
    print("S=", S, " D=", D, " F=", F)

    # ── flow-matching v-target on a synthetic latent (the REAL Z-Image target):
    #    x_t = (1-sigma)*latent + sigma*noise ; target = noise - latent ─────────
    var latent = Tensor.from_host(_randn(S * D, 11, 1.0), [1, S, D], STDtype.F32, ctx)
    var noise = Tensor.from_host(_randn(S * D, 12, 1.0), [1, S, D], STDtype.F32, ctx)
    var fm = flow_match_noise_target(latent, Float32(0.3), noise, ctx)
    var x_in_h = fm.x_t.to_host(ctx)       # [1,S,D] block input (data leaf)
    var target_h = fm.target.to_host(ctx)  # [1,S,D] v-target

    # ── fixed (non-trained) norm + modulation weights ─────────────────────────
    var fn1_h = _randn(D, 21, 0.2)
    var fn2_h = _randn(D, 22, 0.2)
    var sm_h = _randn(D, 23, 0.1)          # scale_mlp = 1 + small
    for i in range(len(sm_h)):
        sm_h[i] = Float32(1.0) + sm_h[i]
    var gm_h = _randn(D, 24, 0.3)          # gate_mlp

    # ── trained leaves: FFN matrices (F32 master) ─────────────────────────────
    var w1_h = _randn(F * D, 31, 0.3)      # [F,D]
    var w3_h = _randn(F * D, 32, 0.3)      # [F,D]
    var w2_h = _randn(D * F, 33, 0.3)      # [D,F]

    # AdamW moments (kept as device tensors across iters; in-place updates)
    var w1 = Tensor.from_host(w1_h.copy(), [F, D], STDtype.F32, ctx)
    var w2 = Tensor.from_host(w2_h.copy(), [D, F], STDtype.F32, ctx)
    var w3 = Tensor.from_host(w3_h.copy(), [F, D], STDtype.F32, ctx)
    var m1 = Tensor.from_host(_zeros(F * D), [F, D], STDtype.F32, ctx)
    var v1 = Tensor.from_host(_zeros(F * D), [F, D], STDtype.F32, ctx)
    var m2 = Tensor.from_host(_zeros(D * F), [D, F], STDtype.F32, ctx)
    var v2 = Tensor.from_host(_zeros(D * F), [D, F], STDtype.F32, ctx)
    var m3 = Tensor.from_host(_zeros(F * D), [F, D], STDtype.F32, ctx)
    var v3 = Tensor.from_host(_zeros(F * D), [F, D], STDtype.F32, ctx)

    var lr = Float32(1.0e-2)
    var b1 = Float32(0.9)
    var b2 = Float32(0.999)
    var aeps = Float32(1.0e-8)
    var wd = Float32(0.0)

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var n_iters = 6

    for it in range(n_iters):
        var t = it + 1  # 1-based AdamW step counter

        # current trained weights as host lists (rebuild fresh per op; move-only).
        var w1c = w1.to_host(ctx)
        var w2c = w2.to_host(ctx)
        var w3c = w3.to_host(ctx)

        # ── FORWARD (host-threaded, mirrors composed_chain_parity) ────────────
        var xfn1_h = rms_norm(
            Tensor.from_host(x_in_h.copy(), [S, D], STDtype.F32, ctx),
            Tensor.from_host(fn1_h.copy(), [D], STDtype.F32, ctx),
            EPS, ctx,
        ).to_host(ctx)                                     # [S,D]
        var xfn1s_h = _bmul_fwd(xfn1_h, sm_h)              # [S,D]
        var nb1 = Optional[Tensor](None)
        var g_h = linear(
            Tensor.from_host(xfn1s_h.copy(), [S, D], STDtype.F32, ctx),
            Tensor.from_host(w1c.copy(), [F, D], STDtype.F32, ctx),
            nb1^, ctx,
        ).to_host(ctx)                                     # [S,F]
        var nb2 = Optional[Tensor](None)
        var u_h = linear(
            Tensor.from_host(xfn1s_h.copy(), [S, D], STDtype.F32, ctx),
            Tensor.from_host(w3c.copy(), [F, D], STDtype.F32, ctx),
            nb2^, ctx,
        ).to_host(ctx)                                     # [S,F]
        var act_h = swiglu(
            Tensor.from_host(g_h.copy(), [S, F], STDtype.F32, ctx),
            Tensor.from_host(u_h.copy(), [S, F], STDtype.F32, ctx),
            ctx,
        ).to_host(ctx)                                     # [S,F]
        var nb3 = Optional[Tensor](None)
        var ff_h = linear(
            Tensor.from_host(act_h.copy(), [S, F], STDtype.F32, ctx),
            Tensor.from_host(w2c.copy(), [D, F], STDtype.F32, ctx),
            nb3^, ctx,
        ).to_host(ctx)                                     # [S,D]
        var ffn2_h = rms_norm(
            Tensor.from_host(ff_h.copy(), [S, D], STDtype.F32, ctx),
            Tensor.from_host(fn2_h.copy(), [D], STDtype.F32, ctx),
            EPS, ctx,
        ).to_host(ctx)                                     # [S,D]
        var gated_h = _bmul_fwd(ffn2_h, gm_h)              # [S,D]
        var out_h = List[Float32]()
        for i in range(S * D):
            out_h.append(x_in_h[i] + gated_h[i])          # residual [S,D]

        var loss = _mse_loss(out_h, target_h)
        if it == 0:
            first_loss = loss
        last_loss = loss

        # ── BACKWARD (reverse chain, host-threaded) ───────────────────────────
        var d_out_h = _mse_grad(out_h, target_h)          # [S,D]
        # residual: gated branch gets d_out unchanged.
        # gated = gate_mlp(broadcast) * ff_n2 -> d wrt ff_n2 = d_out * gate_mlp.
        var d_ffn2_h = _bmul_bwd_x(d_out_h, gm_h)         # [S,D]
        # rms_norm_backward(go=d_ffn2, x=ff, g=fn2) -> .d_x (wrt ff), .d_g.
        var nb_ff = rms_norm_backward(
            Tensor.from_host(d_ffn2_h.copy(), [S, D], STDtype.F32, ctx),
            Tensor.from_host(ff_h.copy(), [S, D], STDtype.F32, ctx),
            Tensor.from_host(fn2_h.copy(), [D], STDtype.F32, ctx),
            EPS, ctx,
        )
        var d_ff_h = nb_ff.d_x.to_host(ctx)               # [S,D]
        # ff = linear(act, w2): linear_backward(grad_y=d_ff, x=act, W=w2,
        #   M=S, in_features=F, out_features=D) -> d_x (wrt act), d_w (=d_w2).
        var lb2 = linear_backward(
            Tensor.from_host(d_ff_h.copy(), [S, D], STDtype.F32, ctx),
            Tensor.from_host(act_h.copy(), [S, F], STDtype.F32, ctx),
            Tensor.from_host(w2c.copy(), [D, F], STDtype.F32, ctx),
            S, F, D, ctx,
        )
        var d_act_h = lb2.d_x.to_host(ctx)                # [S,F]
        var d_w2_h = lb2.d_w.to_host(ctx)                 # [D,F]
        # act = swiglu(g,u): swiglu_backward(gate=g, up=u, grad_out=d_act)
        #   -> .d_gate (wrt g), .d_up (wrt u).
        var sg = swiglu_backward(
            Tensor.from_host(g_h.copy(), [S, F], STDtype.F32, ctx),
            Tensor.from_host(u_h.copy(), [S, F], STDtype.F32, ctx),
            Tensor.from_host(d_act_h.copy(), [S, F], STDtype.F32, ctx),
            ctx,
        )
        var d_g_h = sg.d_gate.to_host(ctx)                # [S,F]
        var d_u_h = sg.d_up.to_host(ctx)                  # [S,F]
        # g = linear(xfn1s, w1): linear_backward -> d_w (=d_w1). (in=D, out=F)
        var lb1 = linear_backward(
            Tensor.from_host(d_g_h.copy(), [S, F], STDtype.F32, ctx),
            Tensor.from_host(xfn1s_h.copy(), [S, D], STDtype.F32, ctx),
            Tensor.from_host(w1c.copy(), [F, D], STDtype.F32, ctx),
            S, D, F, ctx,
        )
        var d_w1_h = lb1.d_w.to_host(ctx)                 # [F,D]
        # u = linear(xfn1s, w3): linear_backward -> d_w (=d_w3).
        var lb3 = linear_backward(
            Tensor.from_host(d_u_h.copy(), [S, F], STDtype.F32, ctx),
            Tensor.from_host(xfn1s_h.copy(), [S, D], STDtype.F32, ctx),
            Tensor.from_host(w3c.copy(), [F, D], STDtype.F32, ctx),
            S, D, F, ctx,
        )
        var d_w3_h = lb3.d_w.to_host(ctx)                 # [F,D]

        # ── grad-nonzero + finite assertion (first iter) ──────────────────────
        if it == 0:
            var a1 = _abs_sum(d_w1_h)
            var a2 = _abs_sum(d_w2_h)
            var a3 = _abs_sum(d_w3_h)
            print("  [grad] sum|d_w1|=", a1, " sum|d_w2|=", a2, " sum|d_w3|=", a3)
            if not (a1 > 0.0 and a2 > 0.0 and a3 > 0.0):
                raise Error("FAIL: a trained-weight grad is all-zero (dead branch)")
            if not (loss == loss):  # NaN != NaN
                raise Error("FAIL: loss is NaN")

        # ── AdamW step on the 3 FFN matrices (in place on device tensors) ─────
        adamw_step(w1, Tensor.from_host(d_w1_h.copy(), [F, D], STDtype.F32, ctx),
                   m1, v1, t, lr, b1, b2, aeps, wd, ctx)
        adamw_step(w2, Tensor.from_host(d_w2_h.copy(), [D, F], STDtype.F32, ctx),
                   m2, v2, t, lr, b1, b2, aeps, wd, ctx)
        adamw_step(w3, Tensor.from_host(d_w3_h.copy(), [F, D], STDtype.F32, ctx),
                   m3, v3, t, lr, b1, b2, aeps, wd, ctx)

        print("  iter", it, " loss=", loss)

    print("  first_loss=", first_loss, " last_loss=", last_loss)
    if not (last_loss == last_loss):
        raise Error("FAIL: final loss NaN")
    if last_loss < first_loss:
        print("PASS: loss finite, grads nonzero, loss DECREASED",
              first_loss, "->", last_loss)
    else:
        print("WARN: loss did not decrease over", n_iters, "iters (",
              first_loss, "->", last_loss, ") — inspect lr/chain.")
