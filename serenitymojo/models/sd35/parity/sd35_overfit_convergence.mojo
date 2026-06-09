# serenitymojo/models/sd35/parity/sd35_overfit_convergence.mojo
#
# OVERFIT-CONVERGENCE smoke for the SD3.5 MMDiT joint block with all 8 LoRA slots
# (ctx + x × qkv/proj/fc1/fc2) — the practically-decisive "does it TRAIN" gate
# (mirrors Klein/Z-Image). Builds the REAL LoRA set (B=0 init), fixes a single
# target on BOTH streams, and runs N steps of the REAL pipeline:
#   sd35_joint_block_forward -> MSE(pred, target) -> d_ctx/d_x = dL/dpred
#   -> sd35_joint_block_backward -> global-norm clip -> sd35_lora_adamw_step
# overfitting one batch. If the bf16 LoRA grads are a correct descent direction,
# the MSE must fall sharply (running-min >=20x, no divergence).
#
# Self-contained: synthetic block weights at the SAME dim regime the block gate
# verifies (H=24, Dh=8, D=192), no checkpoint / offload loader needed.
#
# Run:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/sd35/parity/sd35_overfit_convergence.mojo \
#       -o /tmp/sd35_overfit
#   /tmp/sd35_overfit

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, sin

from serenitymojo.models.sd35.sd35_block import (
    JointBlockWeights, ModVecs, StreamWeights, StreamLoraGrads,
    sd35_joint_block_forward, sd35_joint_block_backward,
)
from serenitymojo.models.sd35.sd35_stack_lora import (
    SD35LoraSet, SD35LoraGradSet, build_sd35_lora_set, sd35_lora_adamw_step,
)
from serenitymojo.training.train_step import LoraAdapter


comptime H = 24
comptime Dh = 8
comptime D = H * Dh          # 192
comptime MLP = 64
comptime N_CTX = 3
comptime N_IMG = 5
comptime S = N_CTX + N_IMG   # 8
comptime RANK = 8
comptime ALPHA = Float32(16.0)
comptime EPS = Float32(1.0e-6)
comptime QK_EPS = Float32(1.0e-6)
comptime SCALE = Float32(1.0) / Float32(2.828427)   # 1/sqrt(Dh=8)

comptime STEPS = 70
comptime LR = Float32(2.0e-3)
comptime M_CTX = N_CTX * D
comptime M_IMG = N_IMG * D
comptime M_OUT = M_CTX + M_IMG


def _rand(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _ones(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(1.0))
    return out^


def _stream_weights(seed: UInt64) -> StreamWeights:
    return StreamWeights(
        _rand(3 * D * D, seed + 1, Float32(0.03)),
        _rand(3 * D, seed + 2, Float32(0.01)),
        _rand(D * D, seed + 3, Float32(0.03)),
        _rand(D, seed + 4, Float32(0.01)),
        _rand(MLP * D, seed + 5, Float32(0.03)),
        _rand(MLP, seed + 6, Float32(0.01)),
        _rand(D * MLP, seed + 7, Float32(0.03)),
        _rand(D, seed + 8, Float32(0.01)),
        _ones(Dh),
        _ones(Dh),
    )


def _mods(seed: UInt64) -> ModVecs:
    return ModVecs(
        _rand(D, seed + 1, Float32(0.03)),
        _rand(D, seed + 2, Float32(0.03)),
        _rand(D, seed + 3, Float32(0.30)),
        _rand(D, seed + 4, Float32(0.03)),
        _rand(D, seed + 5, Float32(0.03)),
        _rand(D, seed + 6, Float32(0.30)),
    )


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= Float32(0.0) else -x
    return s


def _gradset_from_block(g_ctx: StreamLoraGrads, g_x: StreamLoraGrads) -> SD35LoraGradSet:
    # slot order MUST match build_sd35_lora_set: ctx{qkv,proj,fc1,fc2}, x{qkv,proj,fc1,fc2}
    var d_a = List[List[Float32]]()
    var d_b = List[List[Float32]]()
    d_a.append(g_ctx.qkv_d_a.copy()); d_b.append(g_ctx.qkv_d_b.copy())
    d_a.append(g_ctx.proj_d_a.copy()); d_b.append(g_ctx.proj_d_b.copy())
    d_a.append(g_ctx.fc1_d_a.copy()); d_b.append(g_ctx.fc1_d_b.copy())
    d_a.append(g_ctx.fc2_d_a.copy()); d_b.append(g_ctx.fc2_d_b.copy())
    d_a.append(g_x.qkv_d_a.copy()); d_b.append(g_x.qkv_d_b.copy())
    d_a.append(g_x.proj_d_a.copy()); d_b.append(g_x.proj_d_b.copy())
    d_a.append(g_x.fc1_d_a.copy()); d_b.append(g_x.fc1_d_b.copy())
    d_a.append(g_x.fc2_d_a.copy()); d_b.append(g_x.fc2_d_b.copy())
    var nf = 0
    for i in range(len(d_a)):
        for j in range(len(d_a[i])):
            var x = d_a[i][j]
            if (x != x) or (x - x != Float32(0.0)):
                nf += 1
        for j in range(len(d_b[i])):
            var x = d_b[i][j]
            if (x != x) or (x - x != Float32(0.0)):
                nf += 1
    return SD35LoraGradSet(d_a^, d_b^, nf)


def _global_norm(grads: SD35LoraGradSet) -> Float32:
    var ss = Float32(0.0)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += grads.d_a[i][j] * grads.d_a[i][j]
        for j in range(len(grads.d_b[i])):
            ss += grads.d_b[i][j] * grads.d_b[i][j]
    return sqrt(ss)


def _clip(mut grads: SD35LoraGradSet, max_norm: Float32):
    var gn = _global_norm(grads)
    if gn <= max_norm or gn == Float32(0.0):
        return
    var s = max_norm / gn
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s


def _forward(
    context: List[Float32], x: List[Float32],
    w: JointBlockWeights, cm: ModVecs, xm: ModVecs, lora: SD35LoraSet,
    ctx: DeviceContext,
) raises -> List[List[Float32]]:
    var fwd = sd35_joint_block_forward[1, S, H, Dh](
        context.copy(), x.copy(), w, cm.copy(), xm.copy(),
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
        Optional[List[Float32]](None),
        Optional[LoraAdapter](lora.ad[0].copy()), Optional[LoraAdapter](lora.ad[1].copy()),
        Optional[LoraAdapter](lora.ad[2].copy()), Optional[LoraAdapter](lora.ad[3].copy()),
        Optional[LoraAdapter](lora.ad[4].copy()), Optional[LoraAdapter](lora.ad[5].copy()),
        Optional[LoraAdapter](lora.ad[6].copy()), Optional[LoraAdapter](lora.ad[7].copy()),
    )
    var out = List[List[Float32]]()
    out.append(fwd.ctx_out.copy())
    out.append(fwd.x_out.copy())
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("==== sd35 OVERFIT-CONVERGENCE smoke (joint block, 8 LoRA slots, trains) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " MLP=", MLP, " N_CTX=", N_CTX, " N_IMG=", N_IMG,
          " RANK=", RANK, " ALPHA=", ALPHA, " STEPS=", STEPS, " LR=", LR)

    var w = JointBlockWeights(_stream_weights(100), _stream_weights(200))
    var cm = _mods(300)
    var xm = _mods(400)
    var context = _rand(N_CTX * D, 500, Float32(0.5))
    var x = _rand(N_IMG * D, 600, Float32(0.5))

    var lora = build_sd35_lora_set(1, D, MLP, RANK, ALPHA)

    # seed a FIXED reachable target = pred0 + fixed bias, on BOTH streams
    var p0 = _forward(context, x, w, cm, xm, lora, ctx)
    var ctx_target = List[Float32]()
    for i in range(M_CTX):
        ctx_target.append(p0[0][i] + Float32(0.30) * sin(Float32(0.21) * Float32(i) + Float32(0.10)))
    var x_target = List[Float32]()
    for i in range(M_IMG):
        x_target.append(p0[1][i] + Float32(0.30) * sin(Float32(0.19) * Float32(i) + Float32(0.20)))

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)
    var min_loss = Float32(1.0e30)
    var max_after_warmup = Float32(0.0)
    var up_steps = 0
    var prev_loss = Float32(1.0e30)
    print("")
    print("step   MSE")

    for step in range(STEPS):
        var pred = _forward(context, x, w, cm, xm, lora, ctx)

        # MSE over both streams; d_ctx/d_x = (2/M)*(pred-target)
        var loss = Float32(0.0)
        var d_ctx = List[Float32]()
        for i in range(M_CTX):
            var diff = pred[0][i] - ctx_target[i]
            loss += diff * diff
            d_ctx.append((Float32(2.0) / Float32(M_OUT)) * diff)
        var d_x = List[Float32]()
        for i in range(M_IMG):
            var diff = pred[1][i] - x_target[i]
            loss += diff * diff
            d_x.append((Float32(2.0) / Float32(M_OUT)) * diff)
        loss = loss / Float32(M_OUT)

        if step == 0:
            first_loss = loss
        last_loss = loss
        if loss < min_loss:
            min_loss = loss
        if step >= 3 and loss > max_after_warmup:
            max_after_warmup = loss
        if step > 0 and loss >= prev_loss:
            up_steps += 1
        prev_loss = loss
        print(step, "  ", loss)

        # rebuild forward saved-state for backward (block backward needs `fwd`)
        var fwd = sd35_joint_block_forward[1, S, H, Dh](
            context.copy(), x.copy(), w, cm.copy(), xm.copy(),
            N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
            Optional[List[Float32]](None),
            Optional[LoraAdapter](lora.ad[0].copy()), Optional[LoraAdapter](lora.ad[1].copy()),
            Optional[LoraAdapter](lora.ad[2].copy()), Optional[LoraAdapter](lora.ad[3].copy()),
            Optional[LoraAdapter](lora.ad[4].copy()), Optional[LoraAdapter](lora.ad[5].copy()),
            Optional[LoraAdapter](lora.ad[6].copy()), Optional[LoraAdapter](lora.ad[7].copy()),
        )
        var g = sd35_joint_block_backward[1, S, H, Dh](
            d_ctx, d_x, w, cm.copy(), xm.copy(), fwd,
            N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
            Optional[LoraAdapter](lora.ad[0].copy()), Optional[LoraAdapter](lora.ad[1].copy()),
            Optional[LoraAdapter](lora.ad[2].copy()), Optional[LoraAdapter](lora.ad[3].copy()),
            Optional[LoraAdapter](lora.ad[4].copy()), Optional[LoraAdapter](lora.ad[5].copy()),
            Optional[LoraAdapter](lora.ad[6].copy()), Optional[LoraAdapter](lora.ad[7].copy()),
        )
        var grads = _gradset_from_block(g.ctx_lora, g.x_lora)
        if grads.nonfinite != 0:
            print("  !! nonfinite LoRA grads at step", step)
        _clip(grads, Float32(1.0))
        sd35_lora_adamw_step(lora, grads, step + 1, LR, ctx)

    var reduction_x = first_loss / (min_loss + Float32(1.0e-30))
    print("")
    print("first MSE   =", first_loss)
    print("min MSE     =", min_loss, "  (", reduction_x, "x reduction)")
    print("last MSE    =", last_loss)
    print("max MSE @step>=3 =", max_after_warmup, "  (start =", first_loss, ")")
    print("up-steps (bf16 noise-floor jitter) =", up_steps, "/", STEPS - 1)

    var converged = (min_loss < first_loss * Float32(0.05))
    var no_diverge = (max_after_warmup <= first_loss)
    if converged and no_diverge:
        print("VERDICT: PASS — SD3.5 joint block + 8 LoRA slots overfits one batch;",
              "MSE driven from", first_loss, "to", min_loss, "(", reduction_x,
              "x), no divergence => bf16 LoRA grads ARE a correct descent direction")
    else:
        print("VERDICT: FAIL — converged(<5% start)=", converged,
              " no_diverge=", no_diverge,
              " (first=", first_loss, " min=", min_loss, " max@>=3=", max_after_warmup, ")")
