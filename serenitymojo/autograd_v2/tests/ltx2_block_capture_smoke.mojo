# autograd_v2/tests/ltx2_block_capture_smoke.mojo — rung-3(d) CAPTURE FEASIBILITY.
#
# Make-or-break for LTX2_V2_CAPTURE: is the ltx2 VIDEO block's slab fwd+bwd
# (recompute forward_slab + backward_slab_dev, resident LoRA grad store — the
# capture-safe path) CAPTURABLE, and does REPLAY reproduce the eager result
# BYTE-for-byte? Modeled on ideogram4_capture_bench.mojo: capture the RAW slab
# compute (NOT the graph engine — that host overhead is what capture eliminates),
# replay it, compare replayed d_hidden + LoRA grads to the eager run. If capture
# fails (an alloc/sync inside the region), cuda_capture_end_instantiate RAISES
# with the decoded CUresult — that is the STOP-and-report signal (the exact
# blocker). ltx2 is MATH-MODE recompute-softmax attention (no flash allocs), so a
# hard BIT gate on replay==eager.
#
#   rm -f serenitymojo.mojopkg && pixi run mojo build -O2 -I . \
#     -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib \
#     -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/autograd_v2/tests/ltx2_block_capture_smoke.mojo -o /tmp/ltx2_block_capture_smoke
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:\
#     $HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     /tmp/ltx2_block_capture_smoke

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
comptime N_REPLAY = 4


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


def main() raises:
    print("=== LTX2 VIDEO block CUDA-graph CAPTURE smoke (replay == eager, BIT) ===")
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var src = LTX2VideoBlockSource.open(CKPT, cfg, False)
    var w = src.get_block(0, ctx)

    # Attach NONZERO LoRA (preset 0) — keep la/lb to size the resident store.
    var names = video_lora_names(0)
    var la = List[ArcPointer[Tensor]]()
    var lb = List[ArcPointer[Tensor]]()
    for s in range(len(names)):
        var ws = w.weight_shape(names[s])   # [out, in]
        la.append(ArcPointer[Tensor](_t(_sh2(RANK, ws[1]), Float32(s) * Float32(0.3) + Float32(1.0), ctx)))
        lb.append(ArcPointer[Tensor](_t(_sh2(ws[0], RANK), Float32(s) * Float32(0.3) + Float32(2.0), ctx)))
    _attach_block_lora(w, 0, names, la, lb, SCALE)

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

    # ── EAGER reference (block-0 recompute fwd + dev bwd → store, d_hidden). ─────
    slab.reset()
    var fwd0 = ltx2_video_block_train_forward_slab[S_V, N_TXT](
        w, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab)
    var bg0 = ltx2_video_block_backward_slab_dev[S_V, N_TXT](
        w, fwd0.acts, d_video, v_temb, v_cos, v_sin, EPS, ctx, slab, store, 0)
    var dh_ref = bg0.d_hidden.to_host(ctx)
    var lora_ref = store.readback_block(0, ctx)
    print("  eager d_hidden n=", len(dh_ref), " lora_slots=", len(lora_ref))

    # ── WARMUP x2 (cuBLAS handles/workspace + MAX kernel JIT OUTSIDE capture). ──
    for _ in range(2):
        slab.reset()
        var fwm = ltx2_video_block_train_forward_slab[S_V, N_TXT](
            w, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab)
        var bwm = ltx2_video_block_backward_slab_dev[S_V, N_TXT](
            w, fwm.acts, d_video, v_temb, v_cos, v_sin, EPS, ctx, slab, store, 0)
        _ = bwm.d_hidden.to_host(ctx)   # force completion
        ctx.synchronize()

    # ── CAPTURE the raw slab fwd+bwd + the d_hidden copy-out into fixed_dh. ──────
    slab.reset()
    cuda_capture_begin(ctx)
    var fc = ltx2_video_block_train_forward_slab[S_V, N_TXT](
        w, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab)
    var bc = ltx2_video_block_backward_slab_dev[S_V, N_TXT](
        w, fc.acts, d_video, v_temb, v_cos, v_sin, EPS, ctx, slab, store, 0)
    var dh_dst = fixed_dh.create_sub_buffer[DType.uint8](0, nb)
    ctx.enqueue_copy(dst_buf=dh_dst, src_buf=bc.d_hidden.buf)
    var graph = cuda_capture_end_instantiate(ctx)
    print("  captured graph nodes:", graph.nodes)

    # ── REPLAY (re-runs recorded kernels against the recorded/fixed addresses). ─
    for _ in range(N_REPLAY):
        cuda_graph_launch(graph, ctx)
    ctx.synchronize()
    var dh_replay = Tensor(fixed_dh.create_sub_buffer[DType.uint8](0, nb),
                           d_video.shape(), d_video.dtype()).to_host(ctx)
    var lora_replay = store.readback_block(0, ctx)

    # ── COMPARE replay == eager (BIT). ──────────────────────────────────────────
    var allok = True
    _check("d_hidden replay==eager", dh_ref, dh_replay, allok)
    if len(lora_ref) != len(lora_replay):
        print("   FAIL lora slot count", len(lora_ref), "vs", len(lora_replay))
        allok = False
    else:
        for s in range(len(lora_ref)):
            _check("lora.d_a[" + String(s) + "]", lora_ref[s].d_a, lora_replay[s].d_a, allok)
            _check("lora.d_b[" + String(s) + "]", lora_ref[s].d_b, lora_replay[s].d_b, allok)

    if allok:
        print("GATE ltx2_block_capture_smoke: ALL PASS",
              "(block fwd+bwd CAPTURABLE;", graph.nodes,
              "nodes; replay == eager byte-exact, d_hidden + all LoRA grads)")
    else:
        print("GATE ltx2_block_capture_smoke: FAIL")
        raise Error("ltx2_block_capture_smoke gate FAILED")
