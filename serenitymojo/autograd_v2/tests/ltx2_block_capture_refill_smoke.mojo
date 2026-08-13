# autograd_v2/tests/ltx2_block_capture_refill_smoke.mojo — rung-3(d) REFILL proof.
#
# The 48-block replay's core unknown: after capturing block-0's graph, if the
# conductor REFILLS the (fixed-address) weight buffers with block-1's weights
# between launches, does REPLAY produce block-1's result? This proves it
# same-process: capture block-0's fwd+bwd, replay (baseline == eager block-0),
# then enqueue_copy block-1's weights into block-0's weight buffers and replay
# again — the result must now byte-match an INDEPENDENT eager block-1 run
# (block-1 weights in their OWN buffers). LoRA + inputs are held constant (shared
# ArcPointers), so weights are the ONLY refilled variable; P5 (capture_smoke)
# already proved fixed-pointer re-read with new INPUT data. A meaningfulness
# guard asserts eager-block-0 != eager-block-1 (else the refill test is trivial).
#
#   rm -f serenitymojo.mojopkg && pixi run mojo build -O2 -I . \
#     -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib \
#     -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/autograd_v2/tests/ltx2_block_capture_refill_smoke.mojo -o /tmp/ltx2_block_capture_refill_smoke
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:\
#     $HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     /tmp/ltx2_block_capture_refill_smoke

from std.math import sin, cos
from std.memory import ArcPointer
from max.gpu.host import DeviceContext, DeviceBuffer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.models.ltx2.ltx2_video_stack import (
    LTX2VideoBlockSource, video_lora_names, _attach_block_lora,
)
from serenitymojo.models.ltx2.ltx2_video_backward import (
    ltx2_video_block_train_forward_slab, ltx2_video_block_backward_slab_dev,
)
from serenitymojo.models.ltx2.ltx2_av_backward import Ltx2LoraGradStore
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.autograd_v2.capture import (
    cuda_capture_begin, cuda_capture_end_instantiate, cuda_graph_launch,
)

comptime S_V = 256
comptime N_TXT = 1024
comptime VD = 4096
comptime H = 32
comptime DH = 128
comptime RANK = 8
comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
comptime EPS = Float32(1.0e-6)
comptime SCALE = Float32(0.5)
comptime SLAB_BYTES = 2 * 1024 * 1024 * 1024


def _fill(n: Int, phase: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        var fi = Float32(i)
        out.append(Float32(0.03) * (sin(Float32(0.7) * fi + phase)
                   + Float32(0.4) * cos(Float32(1.3) * fi + Float32(0.2))))
    return out^


def _t(shape: List[Int], phase: Float32, ctx: DeviceContext) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return Tensor.from_host(_fill(n, phase), shape.copy(), STDtype.BF16, ctx)


def _sh(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^
def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _cmp(a: List[Float32], b: List[Float32]) -> Tuple[Int, Bool]:
    if len(a) != len(b):
        return (-1, False)
    var nm = 0
    var nz = False
    for i in range(len(a)):
        if a[i] != b[i]:
            nm += 1
        if a[i] != Float32(0.0):
            nz = True
    return (nm, nz)


def _check(name: String, a: List[Float32], b: List[Float32], mut allok: Bool):
    var r = _cmp(a, b)
    var nm = r[0]
    var nz = r[1]
    if nm != 0 or not nz:
        allok = False
    var verdict = "PASS" if (nm == 0 and nz) else "FAIL"
    print("  ", verdict, name, " n_mismatch=", nm, " nonzero=", nz, " n=", len(a))


def _run_eager(
    w: LTX2AVBlockWeights, hidden: Tensor, enc: Tensor, v_temb: Tensor,
    v_prompt_ts: Tensor, v_cos: Tensor, v_sin: Tensor, d_video: Tensor,
    mut store: Ltx2LoraGradStore, mut slab: StepSlab, ctx: DeviceContext,
) raises -> List[Float32]:
    slab.reset()
    var fwd = ltx2_video_block_train_forward_slab[S_V, N_TXT](
        w, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab)
    var bg = ltx2_video_block_backward_slab_dev[S_V, N_TXT](
        w, fwd.acts, d_video, v_temb, v_cos, v_sin, EPS, ctx, slab, store, 0)
    return bg.d_hidden.to_host(ctx)


def main() raises:
    print("=== LTX2 VIDEO block CAPTURE + WEIGHT-REFILL smoke (replay refill==eager) ===")
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var src = LTX2VideoBlockSource.open(CKPT, cfg, False)
    var w0 = src.get_block(0, ctx)
    var w1 = src.get_block(1, ctx)

    # SHARED synthetic LoRA on BOTH blocks (same ArcPointers ⇒ LoRA is not a
    # refilled variable; only the base weights differ between the two blocks).
    var names = video_lora_names(0)
    var la = List[ArcPointer[Tensor]]()
    var lb = List[ArcPointer[Tensor]]()
    for s in range(len(names)):
        var ws = w0.weight_shape(names[s])
        la.append(ArcPointer[Tensor](_t(_sh2(RANK, ws[1]), Float32(s) * Float32(0.3) + Float32(1.0), ctx)))
        lb.append(ArcPointer[Tensor](_t(_sh2(ws[0], RANK), Float32(s) * Float32(0.3) + Float32(2.0), ctx)))
    _attach_block_lora(w0, 0, names, la, lb, SCALE)
    _attach_block_lora(w1, 0, names, la, lb, SCALE)

    var hidden = _t(_sh(1, S_V, VD), Float32(0.11), ctx)
    var enc = _t(_sh(1, N_TXT, VD), Float32(0.23), ctx)
    var v_temb = _t(_sh(1, S_V, 9 * VD), Float32(0.31), ctx)
    var v_prompt_ts = _t(_sh(1, N_TXT, 2 * VD), Float32(0.41), ctx)
    var v_cos = _t(_sh2(S_V * H, DH // 2), Float32(0.51), ctx)
    var v_sin = _t(_sh2(S_V * H, DH // 2), Float32(0.61), ctx)
    var d_video = _t(_sh(1, S_V, VD), Float32(0.71), ctx)
    var nb = d_video.nbytes()

    var store = Ltx2LoraGradStore.create(1, names, la, lb, ctx)
    var slab = StepSlab(ctx, SLAB_BYTES)
    var fixed_dh = ctx.enqueue_create_buffer[DType.uint8](nb)

    var allok = True

    # ── independent EAGER references for block-0 and block-1 weights. ───────────
    var dh_eager0 = _run_eager(w0, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, d_video, store, slab, ctx)
    var dh_eager1 = _run_eager(w1, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, d_video, store, slab, ctx)
    # meaningfulness: the two blocks' weights MUST produce different d_hidden.
    var diff = _cmp(dh_eager0, dh_eager1)
    if diff[0] == 0:
        print("   FAIL block-0 and block-1 eager d_hidden identical — refill test degenerate")
        allok = False
    else:
        print("   PASS eager block-0 != block-1 (", diff[0], "of", len(dh_eager0), "differ) — refill test meaningful")

    # ── WARMUP x2 (block-0). ────────────────────────────────────────────────────
    for _ in range(2):
        slab.reset()
        var fwm = ltx2_video_block_train_forward_slab[S_V, N_TXT](
            w0, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab)
        var bwm = ltx2_video_block_backward_slab_dev[S_V, N_TXT](
            w0, fwm.acts, d_video, v_temb, v_cos, v_sin, EPS, ctx, slab, store, 0)
        _ = bwm.d_hidden.to_host(ctx)
        ctx.synchronize()

    # ── CAPTURE block-0's fwd+bwd (bakes w0's weight/LoRA/input addresses). ──────
    slab.reset()
    cuda_capture_begin(ctx)
    var fc = ltx2_video_block_train_forward_slab[S_V, N_TXT](
        w0, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab)
    var bc = ltx2_video_block_backward_slab_dev[S_V, N_TXT](
        w0, fc.acts, d_video, v_temb, v_cos, v_sin, EPS, ctx, slab, store, 0)
    var dh_dst = fixed_dh.create_sub_buffer[DType.uint8](0, nb)
    ctx.enqueue_copy(dst_buf=dh_dst, src_buf=bc.d_hidden.buf)
    var graph = cuda_capture_end_instantiate(ctx)
    print("  captured graph nodes:", graph.nodes)

    # ── REPLAY-0 (baseline: block-0 weights still resident) == eager block-0. ────
    cuda_graph_launch(graph, ctx)
    ctx.synchronize()
    var dh_replay0 = Tensor(fixed_dh.create_sub_buffer[DType.uint8](0, nb),
                            d_video.shape(), d_video.dtype()).to_host(ctx)
    _check("replay-0 (no refill) == eager block-0", dh_eager0, dh_replay0, allok)

    # ── REFILL: overwrite block-0's weight buffers (the FIXED addresses the graph
    #    baked) with block-1's weights, in place (D2D enqueue_copy, no alloc). ────
    if len(w0.weights) != len(w1.weights):
        print("   FAIL weight-list length mismatch", len(w0.weights), "vs", len(w1.weights))
        allok = False
    else:
        var refilled = 0
        for k in range(len(w0.weights)):
            if w0.weights[k][].nbytes() != w1.weights[k][].nbytes():
                print("   FAIL weight", k, "byte mismatch", w0.weights[k][].nbytes(),
                      "vs", w1.weights[k][].nbytes())
                allok = False
            else:
                ctx.enqueue_copy(dst_buf=w0.weights[k][].buf, src_buf=w1.weights[k][].buf)
                refilled += 1
        print("  refilled", refilled, "weight tensors block-0 -> block-1 (in place)")

    # ── REPLAY-1 (block-1 weights now in block-0's buffers) == eager block-1. ────
    cuda_graph_launch(graph, ctx)
    ctx.synchronize()
    var dh_replay1 = Tensor(fixed_dh.create_sub_buffer[DType.uint8](0, nb),
                            d_video.shape(), d_video.dtype()).to_host(ctx)
    _check("replay-1 (weights refilled) == eager block-1", dh_eager1, dh_replay1, allok)

    if allok:
        print("GATE ltx2_block_capture_refill_smoke: ALL PASS",
              "(captured block-0 graph replays block-1's result after in-place weight refill)")
    else:
        print("GATE ltx2_block_capture_refill_smoke: FAIL")
        raise Error("ltx2_block_capture_refill_smoke gate FAILED")
