# autograd_v2/tests/ltx2_block_parity.mojo — L4 same-process per-block BIT gate
# for the LTX-2.3 VIDEO block graph backward (Option A coarse).
#
# ltx2 is MATH-MODE (recompute-softmax attention, ltx2_av_backward.mojo:377 — NO
# flash nondeterminism), so this gates BIT-EQUALITY (n_mismatch=0), not a variance
# class. A REAL block (src.get_block(0)) carries the production weights; synthetic
# NONZERO LoRA A AND B make every d_A non-degenerate (B=0 would gate vacuously);
# a degenerate all-zero compared tensor FAILS the gate.
#
#   oracle = the hand-chain per-block pair (ltx2_video_block_train_forward +
#            ltx2_video_block_backward, ltx2_video_stack.mojo:402-419)
#   graph  = ltx2_video_block_graph_backward (autograd_v2, L1)
# Both on the SAME w (LoRA attached once) + SAME inputs. Bar: n_mismatch=0 on
# d_hidden AND every LoRA d_a/d_b slot.
#
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/autograd_v2/tests/ltx2_block_parity.mojo -o /tmp/ltx2_block_parity
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/ltx2_block_parity

from std.math import sin, cos
from std.memory import ArcPointer
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.ltx2_dit import LTX2Config
from serenitymojo.models.ltx2.ltx2_video_stack import (
    LTX2VideoBlockSource, video_lora_names, _attach_block_lora,
)
from serenitymojo.models.ltx2.ltx2_video_backward import (
    ltx2_video_block_train_forward, ltx2_video_block_backward,
)
from serenitymojo.autograd_v2.node import TArc
from serenitymojo.autograd_v2.ltx2_video_block_graph import (
    ltx2_video_block_graph_backward,
    ltx2_video_block_graph_backward_slab,
    ltx2_video_block_graph_backward_slab_dev,
)
from serenitymojo.models.ltx2.ltx2_av_backward import ltx2_lora_dev_readback, Ltx2LoraGradStore
from serenitymojo.autograd_v2.step_slab import StepSlab

comptime SLAB_BYTES = 6 * 1024 * 1024 * 1024   # 6 GiB (eager); peak printed below

comptime S_V = 256          # image512 geometry
comptime N_TXT = 1024
comptime VD = 4096
comptime H = 32
comptime DH = 128
comptime RANK = 8
comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
comptime EPS = Float32(1.0e-6)
comptime SCALE = Float32(0.5)


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


# bit-compare two host F32 lists; returns (n_mismatch, any_nonzero).
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
    print("=== LTX2 VIDEO block graph BIT gate (engine vs hand-chain, same-process) ===")
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    # bf16 stack (run_f32=False) = the production default arm.
    var src = LTX2VideoBlockSource.open(CKPT, cfg, False)
    var w = src.get_block(0, ctx)

    # synthetic NONZERO LoRA (A and B) for the 8 t2v slots (all VDxVD).
    var names = video_lora_names(0)
    var n_slots = len(names)
    var lora_a = List[ArcPointer[Tensor]]()
    var lora_b = List[ArcPointer[Tensor]]()
    for s in range(n_slots):
        lora_a.append(ArcPointer[Tensor](_t(_sh2(RANK, VD), Float32(s) * Float32(0.3) + Float32(1.0), ctx)))
        lora_b.append(ArcPointer[Tensor](_t(_sh2(VD, RANK), Float32(s) * Float32(0.3) + Float32(2.0), ctx)))
    _attach_block_lora(w, 0, names, lora_a, lora_b, SCALE)

    # synthetic-but-real-shaped inputs (bf16, the block dtype).
    var hidden = _t(_sh(1, S_V, VD), Float32(0.11), ctx)
    var enc = _t(_sh(1, N_TXT, VD), Float32(0.23), ctx)
    var v_temb = _t(_sh(1, S_V, 9 * VD), Float32(0.31), ctx)
    var v_prompt_ts = _t(_sh(1, N_TXT, 2 * VD), Float32(0.41), ctx)
    var v_cos = _t(_sh2(S_V * H, DH // 2), Float32(0.51), ctx)
    var v_sin = _t(_sh2(S_V * H, DH // 2), Float32(0.61), ctx)
    var d_video = _t(_sh(1, S_V, VD), Float32(0.71), ctx)

    # ── HAND-CHAIN oracle (recompute forward + block backward) ───────────────
    var fwd_h = ltx2_video_block_train_forward[S_V, N_TXT](
        w, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx)
    var bg_h = ltx2_video_block_backward[S_V, N_TXT](
        w, fwd_h.acts, d_video, v_temb, v_cos, v_sin, EPS, ctx)

    # ── GRAPH engine (records the block, executes; recompute happens inside) ──
    var bg_g = ltx2_video_block_graph_backward[S_V, N_TXT](
        TArc(Tensor(d_video.buf.copy(), d_video.shape(), d_video.dtype())),
        w,
        TArc(Tensor(hidden.buf.copy(), hidden.shape(), hidden.dtype())),
        enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx)

    # ── GRAPH engine, StepSlab path (execute_ltx2v_block_slab; apply_ltx2v_slab
    #    mark/rewind per block) — the L5 slab engine arm ────────────────────────
    var slab = StepSlab(ctx, SLAB_BYTES)
    var bg_gs = ltx2_video_block_graph_backward_slab[S_V, N_TXT](
        TArc(Tensor(d_video.buf.copy(), d_video.shape(), d_video.dtype())),
        w,
        TArc(Tensor(hidden.buf.copy(), hidden.shape(), hidden.dtype())),
        enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab)

    # ── bit-compare d_hidden + every LoRA d_a/d_b slot ───────────────────────
    var allok = True
    print(" -- engine (non-slab) vs hand-chain --")
    _check("d_hidden", bg_h.d_hidden.to_host(ctx), bg_g.d_hidden.to_host(ctx), allok)
    if len(bg_h.lora) != len(bg_g.lora):
        print("   FAIL: lora slot count mismatch", len(bg_h.lora), "vs", len(bg_g.lora))
        allok = False
    else:
        for i in range(len(bg_h.lora)):
            _check("d_a[" + String(i) + "]", bg_h.lora[i].d_a, bg_g.lora[i].d_a, allok)
            _check("d_b[" + String(i) + "]", bg_h.lora[i].d_b, bg_g.lora[i].d_b, allok)

    print(" -- engine (StepSlab) vs hand-chain --")
    _check("slab.d_hidden", bg_h.d_hidden.to_host(ctx), bg_gs.d_hidden.to_host(ctx), allok)
    if len(bg_h.lora) != len(bg_gs.lora):
        print("   FAIL: slab lora slot count mismatch", len(bg_h.lora), "vs", len(bg_gs.lora))
        allok = False
    else:
        for i in range(len(bg_h.lora)):
            _check("slab.d_a[" + String(i) + "]", bg_h.lora[i].d_a, bg_gs.lora[i].d_a, allok)
            _check("slab.d_b[" + String(i) + "]", bg_h.lora[i].d_b, bg_gs.lora[i].d_b, allok)
    print("  slab peak_bytes=", slab.peak_bytes(), " used_bytes=", slab.used_bytes(),
          " n_allocs=", slab.n_allocs, " (rewound per block ⇒ used==0)")

    # ── GRAPH engine, StepSlab + DEVICE-GRAD path (rung 2: execute_ltx2v_block_slab_dev
    #    → apply_ltx2v_slab_dev; LoRA d_A/d_B device-resident, ONE boundary readback) ──
    var slab_dev = StepSlab(ctx, SLAB_BYTES)
    var bg_gd = ltx2_video_block_graph_backward_slab_dev[S_V, N_TXT](
        TArc(Tensor(d_video.buf.copy(), d_video.shape(), d_video.dtype())),
        w,
        TArc(Tensor(hidden.buf.copy(), hidden.shape(), hidden.dtype())),
        enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab_dev)
    var lora_gd = ltx2_lora_dev_readback(bg_gd.lora, ctx)   # boundary readback
    print(" -- engine (StepSlab+DEVICE grads) vs hand-chain --")
    _check("slabdev.d_hidden", bg_h.d_hidden.to_host(ctx), bg_gd.d_hidden.to_host(ctx), allok)
    if len(bg_h.lora) != len(lora_gd):
        print("   FAIL: slabdev lora slot count mismatch", len(bg_h.lora), "vs", len(lora_gd))
        allok = False
    else:
        for i in range(len(bg_h.lora)):
            _check("slabdev.d_a[" + String(i) + "]", bg_h.lora[i].d_a, lora_gd[i].d_a, allok)
            _check("slabdev.d_b[" + String(i) + "]", bg_h.lora[i].d_b, lora_gd[i].d_b, allok)
    print("  slabdev peak_bytes=", slab_dev.peak_bytes(), " used_bytes=", slab_dev.used_bytes(),
          " n_allocs=", slab_dev.n_allocs, " (rewound per block ⇒ used==0)")
    if slab_dev.used_bytes() != 0:
        print("   FAIL: slabdev used_bytes != 0 — apply_ltx2v_slab_dev did not rewind per block")
        allok = False
    if not (slab_dev.n_allocs > 0):
        print("   FAIL: slabdev n_allocs==0")
        allok = False

    # ── GRAPH engine, StepSlab + RESIDENT STORE (rung 3a): the engine dev path
    #    routes d_A/d_B into the pre-allocated store; read the store at the
    #    boundary. This is the ENGINE-level store gate (the block gate lives in
    #    ltx2_block_slab_parity). Compare BY NAME (store slot-order vs oracle
    #    exec-order); used==0 confirms apply_ltx2v_slab_dev rewinds per block. ────
    var store = Ltx2LoraGradStore.create(1, w.lora_names, w.lora_a, w.lora_b, ctx)
    var slab_str = StepSlab(ctx, SLAB_BYTES)
    var bg_gr = ltx2_video_block_graph_backward_slab_dev[S_V, N_TXT](
        TArc(Tensor(d_video.buf.copy(), d_video.shape(), d_video.dtype())),
        w,
        TArc(Tensor(hidden.buf.copy(), hidden.shape(), hidden.dtype())),
        enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab_str, store, 0)
    _ = bg_gr
    var store_host = store.readback_block(0, ctx)
    print(" -- engine (StepSlab+RESIDENT STORE) vs hand-chain --")
    if len(bg_h.lora) != len(store_host):
        print("   FAIL: store slot count mismatch", len(bg_h.lora), "vs", len(store_host))
        allok = False
    else:
        for i in range(len(bg_h.lora)):
            var nm = bg_h.lora[i].name
            var jf = -1
            for j in range(len(store_host)):
                if store_host[j].name == nm:
                    jf = j
            if jf < 0:
                print("   FAIL: store missing grad for", nm); allok = False
            else:
                _check("store.d_a[" + nm + "]", bg_h.lora[i].d_a, store_host[jf].d_a, allok)
                _check("store.d_b[" + nm + "]", bg_h.lora[i].d_b, store_host[jf].d_b, allok)
    print("  store slab used_bytes=", slab_str.used_bytes(), " n_allocs=", slab_str.n_allocs)
    if slab_str.used_bytes() != 0:
        print("   FAIL: store engine used_bytes != 0 — apply_ltx2v_slab_dev did not rewind")
        allok = False

    # W2 hard asserts (skeptic): the slab MUST have been used (no silent fallback
    # to enqueue_create_buffer), and apply_ltx2v_slab MUST have rewound per block.
    if not (slab.n_allocs > 0):
        print("   FAIL: slab n_allocs==0 — engine slab path fell back off-slab")
        allok = False
    if not (slab.peak_bytes() > 0):
        print("   FAIL: slab peak_bytes==0 — engine slab path fell back off-slab")
        allok = False
    if slab.used_bytes() != 0:
        print("   FAIL: slab used_bytes=", slab.used_bytes(), "!= 0 — apply_ltx2v_slab did not rewind per block")
        allok = False

    if allok:
        print("GATE ltx2_block_parity: ALL PASS (n_mismatch=0, all nonzero; non-slab + slab engine arms)")
    else:
        print("GATE ltx2_block_parity: FAIL")
        raise Error("ltx2_block_parity gate FAILED")
