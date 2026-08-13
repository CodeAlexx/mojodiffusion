# autograd_v2/tests/ltx2_block_slab_parity.mojo — L5 BLOCK-level slab BIT gate.
#
# The whole video-block fwd+bwd, StepSlab twins vs the non-slab hand-chain pair,
# on the REAL block-0 (production BF16 weights) with synthetic NONZERO LoRA A+B.
# This transitively gates EVERY helper twin (_ada_row_pertok / _modulate_bc /
# _kv_modulate / _rms_norm_opt / _linear_b / _av_attention_train / _av_attention_bwd
# / _lora_pair_bwd / _rms_bwd_opt / _one_plus _slab) plus the two block twins —
# a _slab twin is BYTE-IDENTICAL to its non-slab original (same kernels, only the
# alloc source differs), so the bar is n_mismatch=0 on video_out, d_hidden, and
# every LoRA d_a/d_b slot. A degenerate all-zero compared tensor FAILS.
#
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/autograd_v2/tests/ltx2_block_slab_parity.mojo -o /tmp/ltx2_block_slab_parity
#   env LD_LIBRARY_PATH=.pixi/envs/default/lib /tmp/ltx2_block_slab_parity

from std.math import sin, cos
from std.memory import ArcPointer
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.models.ltx2.ltx2_video_stack import (
    LTX2VideoBlockSource, video_lora_names, _attach_block_lora,
)
from serenitymojo.models.ltx2.ltx2_video_backward import (
    ltx2_video_block_train_forward, ltx2_video_block_backward,
    ltx2_video_block_train_forward_slab, ltx2_video_block_backward_slab,
    ltx2_video_block_backward_slab_dev,
)
from serenitymojo.models.ltx2.ltx2_av_backward import ltx2_lora_dev_readback, Ltx2LoraGradStore
from serenitymojo.autograd_v2.step_slab import StepSlab

comptime S_V = 256          # image512 geometry (matches ltx2_block_parity)
comptime N_TXT = 1024
comptime VD = 4096
comptime H = 32
comptime DH = 128
comptime RANK = 8
comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
comptime EPS = Float32(1.0e-6)
comptime SCALE = Float32(0.5)
comptime SLAB_BYTES = 6 * 1024 * 1024 * 1024   # 6 GiB (eager); peak printed below


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


def _attach_preset(mut w: LTX2AVBlockWeights, preset: Int, ctx: DeviceContext) raises:
    """Attach synthetic NONZERO LoRA A+B for every slot of `preset`, each shaped
    from the weight's own [out, in] (weight_shape) — preset 1 adds ff.net.0.proj
    [16384,4096] and ff.net.2 [4096,16384], whose shapes differ from the VDxVD
    attn slots, so shapes must come from the weight, not a fixed VD."""
    var names = video_lora_names(preset)
    var lora_a = List[ArcPointer[Tensor]]()
    var lora_b = List[ArcPointer[Tensor]]()
    for s in range(len(names)):
        var ws = w.weight_shape(names[s])   # [out, in]
        var out_f = ws[0]
        var in_f = ws[1]
        lora_a.append(ArcPointer[Tensor](_t(_sh2(RANK, in_f), Float32(s) * Float32(0.3) + Float32(1.0), ctx)))
        lora_b.append(ArcPointer[Tensor](_t(_sh2(out_f, RANK), Float32(s) * Float32(0.3) + Float32(2.0), ctx)))
    _attach_block_lora(w, 0, names, lora_a, lora_b, SCALE)


def _run_arm[S_V: Int, N_TXT: Int](
    label: String, w: LTX2AVBlockWeights,
    hidden: Tensor, enc: Tensor, v_temb: Tensor, v_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor, d_video: Tensor,
    ctx: DeviceContext, mut allok: Bool,
) raises:
    """One arm: block fwd+bwd on the currently-attached LoRA, hand-chain vs slab
    twin, bit-compare video_out + d_hidden + every LoRA d_a/d_b, then HARD-ASSERT
    the slab counters (W2: a twin silently falling back to enqueue_create_buffer
    would leave n_allocs / peak_bytes at 0 and must FAIL, not stay green)."""
    print(" -- arm:", label, "--")
    var fwd_h = ltx2_video_block_train_forward[S_V, N_TXT](
        w, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx)
    var bg_h = ltx2_video_block_backward[S_V, N_TXT](
        w, fwd_h.acts, d_video, v_temb, v_cos, v_sin, EPS, ctx)
    var slab = StepSlab(ctx, SLAB_BYTES)
    var fwd_s = ltx2_video_block_train_forward_slab[S_V, N_TXT](
        w, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab)
    var bg_s = ltx2_video_block_backward_slab[S_V, N_TXT](
        w, fwd_s.acts, d_video, v_temb, v_cos, v_sin, EPS, ctx, slab)
    _check(label + ".video_out", fwd_h.video_out.to_host(ctx), fwd_s.video_out.to_host(ctx), allok)
    _check(label + ".d_hidden", bg_h.d_hidden.to_host(ctx), bg_s.d_hidden.to_host(ctx), allok)
    if len(bg_h.lora) != len(bg_s.lora):
        print("   FAIL: lora slot count mismatch", len(bg_h.lora), "vs", len(bg_s.lora))
        allok = False
    else:
        for i in range(len(bg_h.lora)):
            _check(label + ".d_a[" + String(i) + "]", bg_h.lora[i].d_a, bg_s.lora[i].d_a, allok)
            _check(label + ".d_b[" + String(i) + "]", bg_h.lora[i].d_b, bg_s.lora[i].d_b, allok)
    print("  ", label, "slab n_allocs=", slab.n_allocs, " peak_bytes=", slab.peak_bytes(),
          " lora_slots=", len(bg_s.lora))
    if not (slab.n_allocs > 0):
        print("   FAIL:", label, "slab n_allocs==0 — twin fell back off-slab")
        allok = False
    if not (slab.peak_bytes() > 0):
        print("   FAIL:", label, "slab peak_bytes==0 — twin fell back off-slab")
        allok = False

    # ── RUNG 1: device-resident LoRA grads. Run forward+dev-backward inside ONE
    #    slab mark/rewind window (exactly as apply_ltx2v will), copy d_hidden +
    #    do the SINGLE boundary readback BEFORE rewind, then bit-compare the host
    #    F32 result vs the hand-chain oracle (same fold order). The dev d_A/d_B
    #    clones are OFF-SLAB by design — this arm confirms they DON'T corrupt the
    #    slab rewind accounting (used_bytes==0 after rewind). ───────────────────
    var slab_d = StepSlab(ctx, SLAB_BYTES)
    var m_d = slab_d.mark()
    var fwd_d = ltx2_video_block_train_forward_slab[S_V, N_TXT](
        w, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab_d)
    var bg_d = ltx2_video_block_backward_slab_dev[S_V, N_TXT](
        w, fwd_d.acts, d_video, v_temb, v_cos, v_sin, EPS, ctx, slab_d)
    # copy-out BEFORE rewind: d_hidden (slab) → host; LoRA d_A/d_B (off-slab dev
    # clones) → host via the ONE boundary readback.
    var dhid_host = bg_d.d_hidden.to_host(ctx)
    var lora_host = ltx2_lora_dev_readback(bg_d.lora, ctx)
    var dev_n_allocs = slab_d.n_allocs
    var dev_peak = slab_d.peak_bytes()
    slab_d.rewind(m_d)
    _check(label + ".dev.d_hidden", bg_h.d_hidden.to_host(ctx), dhid_host, allok)
    if len(bg_h.lora) != len(lora_host):
        print("   FAIL: dev lora slot count mismatch", len(bg_h.lora), "vs", len(lora_host))
        allok = False
    else:
        for i in range(len(bg_h.lora)):
            _check(label + ".dev.d_a[" + String(i) + "]", bg_h.lora[i].d_a, lora_host[i].d_a, allok)
            _check(label + ".dev.d_b[" + String(i) + "]", bg_h.lora[i].d_b, lora_host[i].d_b, allok)
    print("  ", label, "dev slab n_allocs=", dev_n_allocs, " peak_bytes=", dev_peak,
          " used_after_rewind=", slab_d.used_bytes())
    if not (dev_n_allocs > 0):
        print("   FAIL:", label, "dev slab n_allocs==0")
        allok = False
    if not (dev_peak > 0):
        print("   FAIL:", label, "dev slab peak_bytes==0")
        allok = False
    if slab_d.used_bytes() != 0:
        print("   FAIL:", label, "dev slab used_bytes=", slab_d.used_bytes(),
              "!= 0 — off-slab dev clones corrupted the rewind accounting")
        allok = False

    # ── RUNG 3 (a): RESIDENT grad store. Same dev backward, but LoRA d_A/d_B are
    #    enqueue_copy'd into the PRE-ALLOCATED store slots (no alloc — retires the
    #    .clone). Read the store at the boundary; bit-compare BY NAME (store is
    #    slot-order, the hand-chain oracle is backward-exec order). used==0 after
    #    rewind confirms the store writes (off-slab resident) don't touch the slab
    #    ledger. ────────────────────────────────────────────────────────────────
    var store = Ltx2LoraGradStore.create(1, w.lora_names, w.lora_a, w.lora_b, ctx)
    var slab_st = StepSlab(ctx, SLAB_BYTES)
    var m_st = slab_st.mark()
    var fwd_st = ltx2_video_block_train_forward_slab[S_V, N_TXT](
        w, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx, slab_st)
    var bg_st = ltx2_video_block_backward_slab_dev[S_V, N_TXT](
        w, fwd_st.acts, d_video, v_temb, v_cos, v_sin, EPS, ctx, slab_st, store, 0)
    var store_host = store.readback_block(0, ctx)   # slot order, host F32 (boundary)
    var st_n_allocs = slab_st.n_allocs
    slab_st.rewind(m_st)
    if len(bg_h.lora) != len(store_host):
        print("   FAIL:", label, "store slot count mismatch", len(bg_h.lora), "vs", len(store_host))
        allok = False
    else:
        for i in range(len(bg_h.lora)):
            var nm = bg_h.lora[i].name
            var jfound = -1
            for j in range(len(store_host)):
                if store_host[j].name == nm:
                    jfound = j
            if jfound < 0:
                print("   FAIL:", label, "store missing grad for", nm)
                allok = False
            else:
                _check(label + ".store.d_a[" + nm + "]", bg_h.lora[i].d_a, store_host[jfound].d_a, allok)
                _check(label + ".store.d_b[" + nm + "]", bg_h.lora[i].d_b, store_host[jfound].d_b, allok)
    print("  ", label, "store slab n_allocs=", st_n_allocs,
          " used_after_rewind=", slab_st.used_bytes(), " store_slots=", len(store_host))
    if slab_st.used_bytes() != 0:
        print("   FAIL:", label, "store slab used_bytes != 0 — store write touched the slab ledger")
        allok = False


def main() raises:
    print("=== LTX2 VIDEO block StepSlab BIT gate (slab twin vs non-slab hand-chain) ===")
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var src = LTX2VideoBlockSource.open(CKPT, cfg, False)
    var w = src.get_block(0, ctx)
    # Byte-math on record: the MEASURED per-block bf16 weight footprint =
    # sum of the loaded block tensors' bytes (the fixed-weight-staging buffer size
    # for rung-3 capture; it replaces the current fresh per-block alloc ⇒ NET ~0).
    var block_bytes = 0
    for i in range(len(w.weights)):
        block_bytes += w.weights[i][].nbytes()
    print("  [byte-math] per-block bf16 weight footprint (measured) =", block_bytes, "bytes (",
          block_bytes // (1024 * 1024), "MiB )")

    # synthetic-but-real-shaped inputs (bf16, the block dtype) — shared by both arms.
    var hidden = _t(_sh(1, S_V, VD), Float32(0.11), ctx)
    var enc = _t(_sh(1, N_TXT, VD), Float32(0.23), ctx)
    var v_temb = _t(_sh(1, S_V, 9 * VD), Float32(0.31), ctx)
    var v_prompt_ts = _t(_sh(1, N_TXT, 2 * VD), Float32(0.41), ctx)
    var v_cos = _t(_sh2(S_V * H, DH // 2), Float32(0.51), ctx)
    var v_sin = _t(_sh2(S_V * H, DH // 2), Float32(0.61), ctx)
    var d_video = _t(_sh(1, S_V, VD), Float32(0.71), ctx)

    var allok = True
    # ── ARM 1: PRESET 0 (t2v, 8 attn slots) — the ffn_lora branch stays OFF ────
    _attach_preset(w, 0, ctx)
    _run_arm[S_V, N_TXT](
        "p0_t2v", w, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, d_video, ctx, allok)

    # ── ARM 2: PRESET 1 (v2v; +ff.net.0.proj +ff.net.2) — EXERCISES the
    #    ffn_lora branch of ltx2_video_block_backward_slab (never gated before,
    #    skeptic W1). Re-uses the same block weights + inputs; only LoRA differs.
    w.clear_lora_factors()
    _attach_preset(w, 1, ctx)
    _run_arm[S_V, N_TXT](
        "p1_v2v_ffn", w, hidden, enc, v_temb, v_prompt_ts, v_cos, v_sin, d_video, ctx, allok)

    if allok:
        print("GATE ltx2_block_slab_parity: ALL PASS (bit-exact slab vs non-slab; t2v + v2v/ffn arms)")
    else:
        print("GATE ltx2_block_slab_parity: FAIL")
        raise Error("ltx2_block_slab_parity gate FAILED")
