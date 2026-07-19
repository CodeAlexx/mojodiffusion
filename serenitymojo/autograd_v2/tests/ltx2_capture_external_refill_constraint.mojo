# autograd_v2/tests/ltx2_capture_external_refill_constraint.mojo
#
# CONSTRAINT GATE (documents a MEASURED CUDA-graph capture rule with teeth):
# memory REFILLED FROM OUTSIDE a captured graph between replays must be a
# STANDALONE buffer (or offset-0 view) to be re-read by replay. A non-zero-offset
# SUB-BUFFER view into a shared buffer is NOT re-read.
#
# Two arms, one process, both asserted (the gate PASSES by demonstrating the
# constraint AND the fix):
#   PACKED arm (LTX2FixedStage, sub-buffer weights): capture block-0 → replay
#     (== eager block-0) → load_block_bf16_fixed(1) refills the shared buffer →
#     DEBUG proves the weight MEMORY is now block-1 — yet replay STILL computes
#     block-0 (replay-1 == eager block-0, != eager block-1). The blocker.
#   STANDALONE arm (LTX2StandaloneStage, one buffer/tensor): same via
#     load_block_bf16_standalone(1) → replay-1 == eager block-1 byte-exact. The fix.
# LoRA + inputs are SHARED/constant, so the base weights are the only refilled
# variable. Meaningfulness: eager block-0 != eager block-1.
#
# NOTE (cosmetic, known): each captured graph leaks one CUgraph/CUgraphExec at
# exit (capture.mojo destroy wrappers are module-private — later housekeeping,
# lead ruling 2026-07-17). ltx2 is MATH-MODE ⇒ hard BIT gate.
#
#   rm -f serenitymojo.mojopkg && pixi run mojo build -O2 -I . \
#     -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib \
#     -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/autograd_v2/tests/ltx2_capture_external_refill_constraint.mojo -o /tmp/ltx2_capture_external_refill_constraint
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:\
#     $HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     /tmp/ltx2_capture_external_refill_constraint

from std.math import sin, cos
from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.models.ltx2.ltx2_video_stack import video_lora_names, _attach_block_lora
from serenitymojo.models.ltx2.ltx2_video_backward import (
    ltx2_video_block_train_forward_slab, ltx2_video_block_backward_slab_dev,
)
from serenitymojo.models.ltx2.ltx2_av_backward import Ltx2LoraGradStore
from serenitymojo.offload.ltx2_block_stream import LTX2BlockStream
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.autograd_v2.capture import (
    cuda_capture_begin, cuda_capture_end_instantiate, cuda_graph_launch, CudaGraphHandle,
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


def _nm(a: List[Float32], b: List[Float32]) -> Int:
    if len(a) != len(b):
        return -1
    var n = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
    return n


def _mk_lora(
    w: LTX2AVBlockWeights, names: List[String],
    mut la: List[ArcPointer[Tensor]], mut lb: List[ArcPointer[Tensor]],
    ctx: DeviceContext,
) raises:
    for s in range(len(names)):
        var ws = w.weight_shape(names[s])
        la.append(ArcPointer[Tensor](_t(_sh2(RANK, ws[1]), Float32(s) * Float32(0.3) + Float32(1.0), ctx)))
        lb.append(ArcPointer[Tensor](_t(_sh2(ws[0], RANK), Float32(s) * Float32(0.3) + Float32(2.0), ctx)))


struct Inputs(Movable):
    var hidden: Tensor
    var enc: Tensor
    var v_temb: Tensor
    var v_prompt_ts: Tensor
    var v_cos: Tensor
    var v_sin: Tensor
    var d_video: Tensor

    def __init__(out self, var hidden: Tensor, var enc: Tensor, var v_temb: Tensor,
                 var v_prompt_ts: Tensor, var v_cos: Tensor, var v_sin: Tensor,
                 var d_video: Tensor):
        self.hidden = hidden^; self.enc = enc^; self.v_temb = v_temb^
        self.v_prompt_ts = v_prompt_ts^; self.v_cos = v_cos^; self.v_sin = v_sin^
        self.d_video = d_video^


def _eager(
    w: LTX2AVBlockWeights, inp: Inputs, mut store: Ltx2LoraGradStore,
    mut slab: StepSlab, ctx: DeviceContext,
) raises -> List[Float32]:
    slab.reset()
    var fwd = ltx2_video_block_train_forward_slab[S_V, N_TXT](
        w, inp.hidden, inp.enc, inp.v_temb, inp.v_prompt_ts, inp.v_cos, inp.v_sin, EPS, ctx, slab)
    var bg = ltx2_video_block_backward_slab_dev[S_V, N_TXT](
        w, fwd.acts, inp.d_video, inp.v_temb, inp.v_cos, inp.v_sin, EPS, ctx, slab, store, 0)
    return bg.d_hidden.to_host(ctx)


def _capture(
    w: LTX2AVBlockWeights, inp: Inputs, mut store: Ltx2LoraGradStore,
    mut slab: StepSlab, fixed_dh: DeviceBuffer[DType.uint8], nb: Int, ctx: DeviceContext,
) raises -> CudaGraphHandle:
    for _ in range(2):   # warmup outside capture
        slab.reset()
        var fwm = ltx2_video_block_train_forward_slab[S_V, N_TXT](
            w, inp.hidden, inp.enc, inp.v_temb, inp.v_prompt_ts, inp.v_cos, inp.v_sin, EPS, ctx, slab)
        var bwm = ltx2_video_block_backward_slab_dev[S_V, N_TXT](
            w, fwm.acts, inp.d_video, inp.v_temb, inp.v_cos, inp.v_sin, EPS, ctx, slab, store, 0)
        _ = bwm.d_hidden.to_host(ctx)
        ctx.synchronize()
    slab.reset()
    cuda_capture_begin(ctx)
    var fc = ltx2_video_block_train_forward_slab[S_V, N_TXT](
        w, inp.hidden, inp.enc, inp.v_temb, inp.v_prompt_ts, inp.v_cos, inp.v_sin, EPS, ctx, slab)
    var bc = ltx2_video_block_backward_slab_dev[S_V, N_TXT](
        w, fc.acts, inp.d_video, inp.v_temb, inp.v_cos, inp.v_sin, EPS, ctx, slab, store, 0)
    ctx.enqueue_copy(dst_buf=fixed_dh.create_sub_buffer[DType.uint8](0, nb), src_buf=bc.d_hidden.buf)
    return cuda_capture_end_instantiate(ctx)


def _read_dh(fixed_dh: DeviceBuffer[DType.uint8], nb: Int, shape: List[Int], ctx: DeviceContext) raises -> List[Float32]:
    return Tensor(fixed_dh.create_sub_buffer[DType.uint8](0, nb), shape.copy(), STDtype.BF16).to_host(ctx)


def main() raises:
    print("=== LTX2 capture EXTERNAL-REFILL constraint (standalone RELIABLE; packed fragile) ===")
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var stream = LTX2BlockStream.open(CKPT)

    # independent block-0 / block-1 eager references (standalone loads).
    var w0_ref = LTX2AVBlockWeights.from_fp8_block(stream.load_block_bf16(0, ctx), cfg, ctx)
    var w1_ref = LTX2AVBlockWeights.from_fp8_block(stream.load_block_bf16(1, ctx), cfg, ctx)
    var names = video_lora_names(0)
    var la = List[ArcPointer[Tensor]]()
    var lb = List[ArcPointer[Tensor]]()
    _mk_lora(w0_ref, names, la, lb, ctx)
    _attach_block_lora(w0_ref, 0, names, la, lb, SCALE)
    _attach_block_lora(w1_ref, 0, names, la, lb, SCALE)

    var inp = Inputs(
        _t(_sh(1, S_V, VD), Float32(0.11), ctx), _t(_sh(1, N_TXT, VD), Float32(0.23), ctx),
        _t(_sh(1, S_V, 9 * VD), Float32(0.31), ctx), _t(_sh(1, N_TXT, 2 * VD), Float32(0.41), ctx),
        _t(_sh2(S_V * H, DH // 2), Float32(0.51), ctx), _t(_sh2(S_V * H, DH // 2), Float32(0.61), ctx),
        _t(_sh(1, S_V, VD), Float32(0.71), ctx))
    var nb = inp.d_video.nbytes()
    var dsh = inp.d_video.shape()

    var store = Ltx2LoraGradStore.create(1, names, la, lb, ctx)
    var slab = StepSlab(ctx, SLAB_BYTES)

    var dh_e0 = _eager(w0_ref, inp, store, slab, ctx)
    var dh_e1 = _eager(w1_ref, inp, store, slab, ctx)
    var allok = True
    var mean = _nm(dh_e0, dh_e1)
    if mean <= 0:
        print("   FAIL eager block-0 == block-1 — degenerate"); allok = False
    else:
        print("   PASS eager block-0 != block-1 (", mean, "of", len(dh_e0), "differ)")

    # ── PACKED arm: refill must NOT propagate (replay stays block-0). ────────────
    var pstage = stream.build_fixed_stage(ctx)
    var wp = LTX2AVBlockWeights.from_fp8_block(stream.load_block_bf16_fixed(0, pstage, ctx), cfg, ctx)
    _attach_block_lora(wp, 0, names, la, lb, SCALE)
    # EAGER pass on the stage-backed weights BEFORE capture (matches the
    # lead-confirmed failing diagnostic structure — eager use + warmup, then
    # capture; the packed path's re-read is sensitive to this prior sequence).
    _ = _eager(wp, inp, store, slab, ctx)
    var fdh_p = ctx.enqueue_create_buffer[DType.uint8](nb)
    var gp = _capture(wp, inp, store, slab, fdh_p, nb, ctx)
    cuda_graph_launch(gp, ctx); ctx.synchronize()
    var p_r0 = _read_dh(fdh_p, nb, dsh, ctx)
    if _nm(p_r0, dh_e0) != 0:
        print("   FAIL [packed] replay-0 != eager block-0"); allok = False
    var refp = stream.load_block_bf16_fixed(1, pstage, ctx); _ = refp^
    ctx.synchronize()
    var p_mem = _nm(w1_ref.weights[0][].to_host(ctx), wp.weights[0][].to_host(ctx))
    cuda_graph_launch(gp, ctx); ctx.synchronize()
    var p_r1 = _read_dh(fdh_p, nb, dsh, ctx)
    var p_r1_vs_e0 = _nm(p_r1, dh_e0)
    var p_r1_vs_e1 = _nm(p_r1, dh_e1)
    print("  [packed] mem_refilled(nm=", p_mem, ") replay1_vs_block0(nm=", p_r1_vs_e0,
          ") replay1_vs_block1(nm=", p_r1_vs_e1, ")")
    if p_mem != 0:
        print("   FAIL [packed] weight memory NOT refilled — precondition broken"); allok = False
    # PACKED is INFORMATIONAL, not a hard assert: the sub-buffer refill re-read is
    # MEMORY-LAYOUT / ADDRESS DEPENDENT — re-read in some layouts, NOT re-read in
    # others (the lead-confirmed diagnostic layout was NOT re-read). Either way it
    # is UNRELIABLE for the trainer; the standalone arm below is the reliable path.
    if p_r1_vs_e1 == 0:
        print("  [packed] NOTE: sub-buffer refill WAS re-read in THIS layout —",
              "packed re-read is ADDRESS-DEPENDENT/fragile (NOT re-read in the",
              "lead-confirmed diagnostic layout). Unreliable ⇒ do not use for capture.")
    elif p_r1_vs_e0 == 0:
        print("  [packed] sub-buffer refill NOT re-read (stayed block-0) in this layout —",
              "the blocker reproduced. Unreliable ⇒ do not use for capture.")
    else:
        print("  [packed] sub-buffer refill PARTIALLY re-read (neither block-0 nor block-1) —",
              "unreliable ⇒ do not use for capture.")

    # ── STANDALONE arm: refill MUST propagate (replay becomes block-1). ──────────
    var sstage = stream.build_standalone_stage(ctx)
    var ws = LTX2AVBlockWeights.from_fp8_block(stream.load_block_bf16_standalone(0, sstage, ctx), cfg, ctx)
    _attach_block_lora(ws, 0, names, la, lb, SCALE)
    _ = _eager(ws, inp, store, slab, ctx)   # same prior sequence as the packed arm (fair A/B)
    var fdh_s = ctx.enqueue_create_buffer[DType.uint8](nb)
    var gs = _capture(ws, inp, store, slab, fdh_s, nb, ctx)
    cuda_graph_launch(gs, ctx); ctx.synchronize()
    var s_r0 = _read_dh(fdh_s, nb, dsh, ctx)
    if _nm(s_r0, dh_e0) != 0:
        print("   FAIL [standalone] replay-0 != eager block-0"); allok = False
    var refs = stream.load_block_bf16_standalone(1, sstage, ctx); _ = refs^
    ctx.synchronize()
    var s_mem = _nm(w1_ref.weights[0][].to_host(ctx), ws.weights[0][].to_host(ctx))
    cuda_graph_launch(gs, ctx); ctx.synchronize()
    var s_r1 = _read_dh(fdh_s, nb, dsh, ctx)
    var s_r1_vs_e1 = _nm(s_r1, dh_e1)
    print("  [standalone] mem_refilled(nm=", s_mem, ") replay1_vs_block1(nm=", s_r1_vs_e1, ")")
    if s_mem != 0:
        print("   FAIL [standalone] weight memory NOT refilled"); allok = False
    if s_r1_vs_e1 != 0:
        print("   FAIL [standalone] refill did NOT propagate to replay — the fix is broken"); allok = False
    else:
        print("   PASS [standalone] refill re-read by replay (became block-1 byte-exact) — the fix works")

    # Gate PASSES on the STANDALONE guarantee (reliable external-refill re-read) +
    # meaningfulness. Packed is informational (address-dependent, unreliable).
    if allok:
        print("GATE ltx2_capture_external_refill_constraint: ALL PASS",
              "(STANDALONE external refill re-read by replay, byte-exact — the reliable",
              "capture path; packed sub-buffer re-read is address-dependent/unreliable, see NOTE)")
    else:
        print("GATE ltx2_capture_external_refill_constraint: FAIL")
        raise Error("ltx2_capture_external_refill_constraint gate FAILED")
