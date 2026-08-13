# autograd_v2/tests/ltx2_capture_wiring_parity.mojo — rung-3 CAPTURE WIRING gate.
#
# Proves the LTX2_V2_CAPTURE stack driver (ltx2_video_stack_lora_backward_graph_
# capture) is BIT-IDENTICAL to the committed LTX2_V2_SLAB driver
# (ltx2_video_stack_lora_backward_graph_slab) over a real NLAY-block stack:
#   • same synthetic-but-real-shaped stack inputs (d_pred/x_last/v_embedded +
#     enc/v_temb/v_prompt_ts/v_cos/v_sin + per-block saved inputs), NONZERO LoRA;
#   • ORACLE = one _slab call (engine apply -> raw slab pair);
#   • CAPTURE = three capture-driver calls (step0 WARMUP eager, step1 CAPTURE +
#     replay, step2 all REPLAYS) — each with the SAME inputs, so every step's
#     grads must byte-match the oracle (warmup produces real grads, capture +
#     replay reproduce them through refills against fixed addresses).
# Byte-compare d_input + the store's d_A/d_B (all NLAY*n_slots). n_mismatch must
# be 0. The captured graph node count must be > 1000 (the whole block is inside).
# Enumerate the stage's staging buffers (each a distinct standalone allocation /
# offset-0 view, MJ-1114) and assert distinct + stable addresses across steps.
# ltx2 is MATH-MODE (recompute-softmax attention) => a HARD BIT gate.
#
#   rm -f serenitymojo.mojopkg && pixi run mojo build -O2 -I . \
#     -Xlinker -lm -Xlinker -lcuda -Xlinker -L.pixi/envs/default/lib \
#     -Xlinker -lsqlite3 -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/autograd_v2/tests/ltx2_capture_wiring_parity.mojo -o /tmp/ltx2_capture_wiring_parity
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:\
#     $HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     /tmp/ltx2_capture_wiring_parity

from std.math import sin, cos
from std.memory import ArcPointer
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.ltx2_dit import LTX2Config
from serenitymojo.models.ltx2.ltx2_video_stack import (
    LTX2VideoBlockSource, LTX2VideoTail, video_lora_names, LTX2VideoStackGrads,
)
from serenitymojo.models.ltx2.ltx2_video_stack_graph import (
    ltx2_video_stack_lora_backward_graph_slab,
)
from serenitymojo.models.ltx2.ltx2_video_stack_capture import (
    ltx2_video_stack_lora_backward_graph_capture,
    LTX2CaptureStage,
)

comptime S_V = 256
comptime N_TXT = 1024
comptime VD = 4096
comptime H = 32
comptime DH = 128
comptime RANK = 8
comptime NLAY = 3
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


def _tf32(shape: List[Int], phase: Float32, ctx: DeviceContext) raises -> Tensor:
    """d_pred is F32 in the trainer (Tensor.from_host(..., STDtype.F32))."""
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return Tensor.from_host(_fill(n, phase), shape.copy(), STDtype.F32, ctx)


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


def _check_grads(
    tag: String, cap: LTX2VideoStackGrads, slab: LTX2VideoStackGrads, mut allok: Bool
):
    # d_input + all NLAY*n_slots d_A/d_B byte-compare.
    _check(tag + " d_input", slab.d_input, cap.d_input, allok)
    if len(cap.d_a) != len(slab.d_a) or len(cap.d_b) != len(slab.d_b):
        print("   FAIL", tag, "grad-list length mismatch")
        allok = False
        return
    var na = 0
    var nb = 0
    for i in range(len(cap.d_a)):
        var ra = _cmp(slab.d_a[i], cap.d_a[i])
        var rb = _cmp(slab.d_b[i], cap.d_b[i])
        na += ra[0] if ra[0] > 0 else 0
        nb += rb[0] if rb[0] > 0 else 0
        if ra[0] != 0 or rb[0] != 0:
            allok = False
    print("  ", "PASS" if (na == 0 and nb == 0) else "FAIL", tag,
          "d_A/d_B over", len(cap.d_a), "adapters: dA_mismatch=", na, " dB_mismatch=", nb)


def main() raises:
    print("=== LTX2 CAPTURE WIRING parity (capture driver == _slab driver, BIT) ===")
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var src = LTX2VideoBlockSource.open(CKPT, cfg, False)
    var tail = LTX2VideoTail.load(CKPT, False, ctx)
    var names = video_lora_names(0)
    var n_slots = len(names)

    # NONZERO LoRA for every (block, slot) — sized from block-0's weight shapes.
    var w0 = src.get_block(0, ctx)
    var la = List[ArcPointer[Tensor]]()
    var lb = List[ArcPointer[Tensor]]()
    for bi in range(NLAY):
        for s in range(n_slots):
            var ws = w0.weight_shape(names[s])   # [out, in]
            la.append(ArcPointer[Tensor](_t(_sh2(RANK, ws[1]),
                Float32(bi) * Float32(0.13) + Float32(s) * Float32(0.3) + Float32(1.0), ctx)))
            lb.append(ArcPointer[Tensor](_t(_sh2(ws[0], RANK),
                Float32(bi) * Float32(0.17) + Float32(s) * Float32(0.3) + Float32(2.0), ctx)))

    # Stack inputs (S_V geometry).
    var saved_inputs = List[ArcPointer[Tensor]]()
    for bi in range(NLAY):
        saved_inputs.append(ArcPointer[Tensor](_t(_sh(1, S_V, VD),
            Float32(0.11) + Float32(bi) * Float32(0.07), ctx)))
    var enc = _t(_sh(1, N_TXT, VD), Float32(0.23), ctx)
    var v_temb = _t(_sh(1, S_V, 9 * VD), Float32(0.31), ctx)
    var v_prompt_ts = _t(_sh(1, N_TXT, 2 * VD), Float32(0.41), ctx)
    var v_cos = _t(_sh2(S_V * H, DH // 2), Float32(0.51), ctx)
    var v_sin = _t(_sh2(S_V * H, DH // 2), Float32(0.61), ctx)
    var v_embedded = _t(_sh(1, 1, VD), Float32(0.29), ctx)
    var d_pred = _tf32(_sh(1, S_V, 128), Float32(0.71), ctx)
    var x_last = _t(_sh(1, S_V, VD), Float32(0.19), ctx)

    var allok = True

    # ── ORACLE: the committed LTX2_V2_SLAB driver. ─────────────────────────────
    var grads_slab = ltx2_video_stack_lora_backward_graph_slab[S_V, N_TXT](
        d_pred, saved_inputs, x_last, enc, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
        tail, src, la, lb, SCALE, NLAY, EPS, ctx, 0, 0)
    if grads_slab.nonfinite != 0:
        print("   FAIL oracle _slab produced non-finite grads"); allok = False
    print("  oracle _slab: d_input n=", len(grads_slab.d_input),
          " adapters=", len(grads_slab.d_a))

    # ── CAPTURE stage (NLAY blocks). protos = shape-only bf16 refs. ────────────
    var la_proto = List[ArcPointer[Tensor]]()
    var lb_proto = List[ArcPointer[Tensor]]()
    for s in range(n_slots):
        la_proto.append(la[s].copy())
        lb_proto.append(lb[s].copy())
    var bsz = 2   # bf16
    var vcos_nb = S_V * H * (DH // 2) * bsz
    var stage = LTX2CaptureStage.create(
        src, names, la_proto, lb_proto, NLAY, SCALE,
        N_TXT * VD * bsz, S_V * 9 * VD * bsz, N_TXT * 2 * VD * bsz,
        vcos_nb, vcos_nb, S_V * VD * bsz, SLAB_BYTES, ctx)

    # ── step 0 WARMUP -> step 1 CAPTURE -> step 2 REPLAY (same inputs). ────────
    var g0 = ltx2_video_stack_lora_backward_graph_capture[S_V, N_TXT](
        d_pred, saved_inputs, x_last, enc, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
        tail, src, la, lb, SCALE, NLAY, EPS, ctx, stage, 0, 0)
    print("  capture step0 (warmup) done; graph_set=", Bool(stage.graph), " step=", stage.step)
    var g1 = ltx2_video_stack_lora_backward_graph_capture[S_V, N_TXT](
        d_pred, saved_inputs, x_last, enc, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
        tail, src, la, lb, SCALE, NLAY, EPS, ctx, stage, 0, 0)
    print("  capture step1 (capture+replay) done; graph nodes=", stage.graph.value().nodes)
    var g2 = ltx2_video_stack_lora_backward_graph_capture[S_V, N_TXT](
        d_pred, saved_inputs, x_last, enc, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
        tail, src, la, lb, SCALE, NLAY, EPS, ctx, stage, 0, 0)
    print("  capture step2 (all replays) done; step=", stage.step)

    # ── BYTE-compare each capture step against the oracle. ─────────────────────
    _check_grads("step0(warmup)", g0, grads_slab, allok)
    _check_grads("step1(capture)", g1, grads_slab, allok)
    _check_grads("step2(replay)", g2, grads_slab, allok)

    # ── captured graph node count (whole block inside). ────────────────────────
    var g_nodes = stage.graph.value().nodes
    if g_nodes > 1000:
        print("   PASS captured graph nodes=", g_nodes, "> 1000 (whole block inside)")
    else:
        print("   FAIL captured graph nodes=", g_nodes, "<= 1000"); allok = False

    # ── ENUMERATION: staging inputs are distinct standalone allocations that stay
    #    at a stable address across the 3 steps (MJ-1114 offset-0 refill target). ─
    var addrs = List[Int]()
    addrs.append(Int(stage.enc_buf.unsafe_ptr()))
    addrs.append(Int(stage.vt_buf.unsafe_ptr()))
    addrs.append(Int(stage.vp_buf.unsafe_ptr()))
    addrs.append(Int(stage.vcos_buf.unsafe_ptr()))
    addrs.append(Int(stage.vsin_buf.unsafe_ptr()))
    addrs.append(Int(stage.bin_buf.unsafe_ptr()))
    addrs.append(Int(stage.dxA.unsafe_ptr()))
    addrs.append(Int(stage.dhB.unsafe_ptr()))
    for s in range(n_slots):
        addrs.append(Int(stage.la_slots[s][].buf.unsafe_ptr()))
        addrs.append(Int(stage.lb_slots[s][].buf.unsafe_ptr()))
    var enum_ok = True
    for i in range(len(addrs)):
        if addrs[i] == 0:
            enum_ok = False
        for k in range(i + 1, len(addrs)):
            if addrs[i] == addrs[k]:
                enum_ok = False
    if enum_ok:
        print("   PASS enumeration: ", len(addrs),
              "staging inputs are distinct non-zero standalone allocations (offset-0)")
    else:
        print("   FAIL enumeration: staging inputs aliased or null"); allok = False

    if allok:
        print("GATE ltx2_capture_wiring_parity: ALL PASS",
              "(capture warmup/capture/replay == _slab driver byte-exact;",
              g_nodes, "nodes; standalone staging)")
    else:
        print("GATE ltx2_capture_wiring_parity: FAIL")
        raise Error("ltx2_capture_wiring_parity gate FAILED")
