# models/ltx2/ltx2_video_stack_capture.mojo — LTX2_V2_CAPTURE: per-block CUDA-graph
# capture/replay driver for the LTX-2.3 VIDEO stack backward (rung 3, the launch-
# storm kill). Sits BESIDE the LTX2_V2_SLAB loop (ltx2_video_stack_graph.mojo),
# C13: default OFF, existing paths untouched + reachable.
#
# Bit-proven primitives this integrates (do not re-derive):
#   autograd_v2/tests/ltx2_block_capture_smoke.mojo        (capture+replay==eager)
#   autograd_v2/tests/ltx2_block_capture_refill_smoke.mojo (in-place weight refill)
#   autograd_v2/tests/ltx2_capture_external_refill_constraint.mojo (MJ-1114:
#       externally-refilled memory MUST be a standalone buffer / offset-0 view)
#
# The captured graph = ONE block's RAW slab fwd+bwd+store-write+d_hidden->dhB
# (ltx2_video_block_train_forward_slab + ltx2_video_block_backward_slab_dev — the
# SAME pair the engine's apply_ltx2v_slab_dev calls, so grads are bit-identical to
# the LTX2_V2_SLAB path, C14). Everything the graph reads is a STANDALONE buffer
# or an offset-0 view, refilled in place per block/step (MJ-1114).
#
# KEY SIMPLIFICATION (design authority, VERIFIED): from_fp8_block SHARES buffers,
# so ONE w = from_fp8_block(load_block_bf16_standalone(0, wstage)) is built ONCE at
# stage-create; every later in-place refill of wstage is seen by that same w. The
# LoRA is attached ONCE to the n_slots standalone la_slots/lb_slots. No per-block
# weight-struct rebuild, no per-block re-attach.
#
# d_x rides FIXED buffers: the graph reads d_video from dxA, writes d_hidden to
# dhB; after each block dxA<-dhB (chain); d_input is dxA after the last block.
#
# DEVIATION (approved by lead + design authority 2026-07-17): v_cos/v_sin are
# STAGED into fixed buffers per step (like enc/v_temb/v_prompt_ts) — the spec said
# "geometry-fixed resident, passed as-is", but the trainer builds them FRESH every
# step (cast_tensor(ho.v_cos,...)), so their addresses drift; a captured graph
# would read freed memory. Staging applies the MJ-1114 fixed-address rule and is
# correct even if they ever become step-dependent. eps is a scalar (baked, fine).
#
# Mojo 1.0.0b1, NVIDIA.

from std.memory import ArcPointer
from std.collections import Optional
from std.gpu.host import DeviceContext, DeviceBuffer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.ltx2.ltx2_video_stack import (
    _tail_backward, video_lora_names, _attach_block_lora, _nonfinite,
    LTX2VideoTail, LTX2VideoBlockSource, LTX2VideoStackGrads,
)
from serenitymojo.models.ltx2.ltx2_video_backward import (
    ltx2_video_block_train_forward_slab, ltx2_video_block_backward_slab_dev,
)
from serenitymojo.models.ltx2.ltx2_av_backward import Ltx2LoraGradStore
from serenitymojo.models.dit.ltx2_dit import LTX2AVBlockWeights
from serenitymojo.offload.ltx2_block_stream import LTX2StandaloneStage
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.autograd_v2.capture import (
    cuda_capture_begin, cuda_capture_end_instantiate, cuda_graph_launch,
    CudaGraphHandle,
)


def _view(buf: DeviceBuffer[DType.uint8], shape: List[Int], dtype: STDtype) raises -> Tensor:
    """Offset-0 whole-buffer view (standalone, MJ-1114). The buffer is exactly
    sized to shape*dtype, so this wraps the whole allocation at offset 0."""
    return Tensor(buf.copy(), shape.copy(), dtype)


struct LTX2CaptureStage(Movable):
    """Persistent per-block CUDA-graph capture state. Created ONCE before the train
    loop, passed `mut`. `w` is built once (shares wstage buffers); la_slots/lb_slots
    are attached to it once. Everything the captured graph reads is a standalone
    buffer or an offset-0 view (MJ-1114)."""

    var wstage: LTX2StandaloneStage             # standalone weight stage (refilled per block)
    var w: LTX2AVBlockWeights                    # built ONCE; sees wstage refills (shared buffers)
    var la_slots: List[ArcPointer[Tensor]]       # n_slots fixed LoRA A (offset-0; attached to w)
    var lb_slots: List[ArcPointer[Tensor]]       # n_slots fixed LoRA B
    var enc_buf: DeviceBuffer[DType.uint8]       # per-step staging
    var vt_buf: DeviceBuffer[DType.uint8]
    var vp_buf: DeviceBuffer[DType.uint8]
    var vcos_buf: DeviceBuffer[DType.uint8]      # per-step staging (see DEVIATION note)
    var vsin_buf: DeviceBuffer[DType.uint8]
    var bin_buf: DeviceBuffer[DType.uint8]       # block_input, refilled per block
    var dxA: DeviceBuffer[DType.uint8]           # d_video in (graph reads); d_input out
    var dhB: DeviceBuffer[DType.uint8]           # d_hidden out (graph writes)
    var store: Ltx2LoraGradStore                # resident num_layers*n_slots grad store
    var slab: StepSlab                           # persistent (captured graph bakes slab addrs)
    var graph: Optional[CudaGraphHandle]         # None until captured (step 1)
    var step: Int
    var names: List[String]
    var n_slots: Int
    var lora_scale: Float32
    var num_layers: Int
    var enc_nb: Int
    var vt_nb: Int
    var vp_nb: Int
    var vcos_nb: Int
    var vsin_nb: Int
    var dh_nb: Int
    var addr_rec: Bool
    var enc_addr: Int
    var vt_addr: Int
    var vp_addr: Int
    var vcos_addr: Int
    var vsin_addr: Int
    var bin_addr: Int
    var a_addr: Int
    var b_addr: Int
    var pers_addrs: List[Int]   # wstage + store slots + la/lb slots (recorded 1st visit)

    def __init__(
        out self,
        var wstage: LTX2StandaloneStage,
        var w: LTX2AVBlockWeights,
        var la_slots: List[ArcPointer[Tensor]],
        var lb_slots: List[ArcPointer[Tensor]],
        var enc_buf: DeviceBuffer[DType.uint8],
        var vt_buf: DeviceBuffer[DType.uint8],
        var vp_buf: DeviceBuffer[DType.uint8],
        var vcos_buf: DeviceBuffer[DType.uint8],
        var vsin_buf: DeviceBuffer[DType.uint8],
        var bin_buf: DeviceBuffer[DType.uint8],
        var dxA: DeviceBuffer[DType.uint8],
        var dhB: DeviceBuffer[DType.uint8],
        var store: Ltx2LoraGradStore,
        var slab: StepSlab,
        var names: List[String],
        n_slots: Int, lora_scale: Float32, num_layers: Int,
        enc_nb: Int, vt_nb: Int, vp_nb: Int, vcos_nb: Int, vsin_nb: Int, dh_nb: Int,
    ):
        self.wstage = wstage^
        self.w = w^
        self.la_slots = la_slots^
        self.lb_slots = lb_slots^
        self.enc_buf = enc_buf^
        self.vt_buf = vt_buf^
        self.vp_buf = vp_buf^
        self.vcos_buf = vcos_buf^
        self.vsin_buf = vsin_buf^
        self.bin_buf = bin_buf^
        self.dxA = dxA^
        self.dhB = dhB^
        self.store = store^
        self.slab = slab^
        self.graph = None
        self.step = 0
        self.names = names^
        self.n_slots = n_slots
        self.lora_scale = lora_scale
        self.num_layers = num_layers
        self.enc_nb = enc_nb
        self.vt_nb = vt_nb
        self.vp_nb = vp_nb
        self.vcos_nb = vcos_nb
        self.vsin_nb = vsin_nb
        self.dh_nb = dh_nb
        self.addr_rec = False
        self.enc_addr = 0
        self.vt_addr = 0
        self.vp_addr = 0
        self.vcos_addr = 0
        self.vsin_addr = 0
        self.bin_addr = 0
        self.a_addr = 0
        self.b_addr = 0
        self.pers_addrs = List[Int]()

    @staticmethod
    def create(
        src: LTX2VideoBlockSource,
        names: List[String],
        la_proto: List[ArcPointer[Tensor]], lb_proto: List[ArcPointer[Tensor]],
        num_layers: Int, lora_scale: Float32,
        enc_nb: Int, vt_nb: Int, vp_nb: Int, vcos_nb: Int, vsin_nb: Int, dh_nb: Int,
        slab_bytes: Int,
        ctx: DeviceContext,
    ) raises -> LTX2CaptureStage:
        """Allocate all persistent capture buffers (MJ-1114 standalone/offset-0),
        build `w` ONCE from the standalone stage (shares buffers -> sees refills),
        attach the n_slots fixed LoRA slots to it, size the resident grad store from
        la_proto/lb_proto (shape+dtype only). Post-alloc VRAM fail-loud."""
        var n_slots = len(names)
        var wstage = src.stream.build_standalone_stage(ctx)

        # n_slots fixed LoRA A/B input buffers (offset-0 whole-buffer views).
        var la_slots = List[ArcPointer[Tensor]]()
        var lb_slots = List[ArcPointer[Tensor]]()
        for s in range(n_slots):
            var ab = ctx.enqueue_create_buffer[DType.uint8](la_proto[s][].nbytes())
            var bb = ctx.enqueue_create_buffer[DType.uint8](lb_proto[s][].nbytes())
            la_slots.append(ArcPointer[Tensor](
                Tensor(ab^, la_proto[s][].shape(), la_proto[s][].dtype())))
            lb_slots.append(ArcPointer[Tensor](
                Tensor(bb^, lb_proto[s][].shape(), lb_proto[s][].dtype())))

        # Build w ONCE: load block 0 into the standalone stage, build the weights
        # struct (SHARES the stage buffers), attach the fixed LoRA slots ONCE.
        var blk0 = src.stream.load_block_bf16_standalone(0, wstage, ctx, True)
        var w = LTX2AVBlockWeights.from_fp8_block(blk0^, src.cfg, ctx)
        _attach_block_lora(w, 0, names, la_slots, lb_slots, lora_scale)

        # Resident grad store: replicate the n_slots protos to num_layers (shared
        # ArcPointers; store.create allocs its own buffers).
        var la_ref = List[ArcPointer[Tensor]]()
        var lb_ref = List[ArcPointer[Tensor]]()
        for _bi in range(num_layers):
            for s in range(n_slots):
                la_ref.append(la_proto[s].copy())
                lb_ref.append(lb_proto[s].copy())
        var store = Ltx2LoraGradStore.create(num_layers, names, la_ref, lb_ref, ctx)

        var enc_buf = ctx.enqueue_create_buffer[DType.uint8](enc_nb)
        var vt_buf = ctx.enqueue_create_buffer[DType.uint8](vt_nb)
        var vp_buf = ctx.enqueue_create_buffer[DType.uint8](vp_nb)
        var vcos_buf = ctx.enqueue_create_buffer[DType.uint8](vcos_nb)
        var vsin_buf = ctx.enqueue_create_buffer[DType.uint8](vsin_nb)
        var bin_buf = ctx.enqueue_create_buffer[DType.uint8](dh_nb)
        var dxA = ctx.enqueue_create_buffer[DType.uint8](dh_nb)
        var dhB = ctx.enqueue_create_buffer[DType.uint8](dh_nb)
        var slab = StepSlab(ctx, slab_bytes)

        # Post-alloc VRAM fail-loud (baseline ~2.62GB free under SLAB=1; the stage
        # adds the standalone weight stage + fixed LoRA + staging + dxA/dhB + slab +
        # resident store).
        var mi = ctx.get_memory_info()
        var free_mb = Int(mi[0]) // (1024 * 1024)
        print("  [ltx2 capture] stage allocated; VRAM free=", free_mb, "MiB")
        if free_mb < 512:
            raise Error(
                String("LTX2_V2_CAPTURE: only ") + String(free_mb)
                + " MiB VRAM free after stage allocation (< 512 MiB floor) —"
                + " lower LTX2_RESIDENT_BLOCKS.")

        return LTX2CaptureStage(
            wstage^, w^, la_slots^, lb_slots^, enc_buf^, vt_buf^, vp_buf^, vcos_buf^,
            vsin_buf^, bin_buf^, dxA^, dhB^, store^, slab^, names.copy(), n_slots,
            lora_scale, num_layers, enc_nb, vt_nb, vp_nb, vcos_nb, vsin_nb, dh_nb)

    def store_active(self) -> Bool:
        return self.store.active()

    # ── per-step staging (values change each step; ONE D2D per step) + seed dxA ──
    def copy_per_step(
        mut self, enc: Tensor, v_temb: Tensor, v_prompt_ts: Tensor,
        v_cos: Tensor, v_sin: Tensor, d_seed: Tensor, ctx: DeviceContext,
    ) raises:
        if enc.nbytes() != self.enc_nb:
            raise Error("ltx2 capture: enc nbytes " + String(enc.nbytes()) + " != " + String(self.enc_nb))
        if v_temb.nbytes() != self.vt_nb:
            raise Error("ltx2 capture: v_temb nbytes " + String(v_temb.nbytes()) + " != " + String(self.vt_nb))
        if v_prompt_ts.nbytes() != self.vp_nb:
            raise Error("ltx2 capture: v_prompt_ts nbytes " + String(v_prompt_ts.nbytes()) + " != " + String(self.vp_nb))
        if v_cos.nbytes() != self.vcos_nb:
            raise Error("ltx2 capture: v_cos nbytes " + String(v_cos.nbytes()) + " != " + String(self.vcos_nb))
        if v_sin.nbytes() != self.vsin_nb:
            raise Error("ltx2 capture: v_sin nbytes " + String(v_sin.nbytes()) + " != " + String(self.vsin_nb))
        if d_seed.nbytes() != self.dh_nb:
            raise Error("ltx2 capture: d_seed nbytes " + String(d_seed.nbytes()) + " != " + String(self.dh_nb))
        var e = self.enc_buf.copy(); ctx.enqueue_copy(dst_buf=e, src_buf=enc.buf)
        var t = self.vt_buf.copy(); ctx.enqueue_copy(dst_buf=t, src_buf=v_temb.buf)
        var p = self.vp_buf.copy(); ctx.enqueue_copy(dst_buf=p, src_buf=v_prompt_ts.buf)
        var c = self.vcos_buf.copy(); ctx.enqueue_copy(dst_buf=c, src_buf=v_cos.buf)
        var s = self.vsin_buf.copy(); ctx.enqueue_copy(dst_buf=s, src_buf=v_sin.buf)
        var a = self.dxA.copy(); ctx.enqueue_copy(dst_buf=a, src_buf=d_seed.buf)  # seed dxA

    def check_addrs(mut self) raises:
        """Record staging-buffer addresses on the first visit; RAISE on any drift
        across visits (the captured graph baked these addresses)."""
        var e = Int(self.enc_buf.unsafe_ptr())
        var vt = Int(self.vt_buf.unsafe_ptr())
        var vp = Int(self.vp_buf.unsafe_ptr())
        var vc = Int(self.vcos_buf.unsafe_ptr())
        var vs = Int(self.vsin_buf.unsafe_ptr())
        var bn = Int(self.bin_buf.unsafe_ptr())
        var a = Int(self.dxA.unsafe_ptr())
        var b = Int(self.dhB.unsafe_ptr())
        # Also fingerprint the OTHER persistent buffers the captured graph baked:
        # the standalone weight-stage buffers, the resident grad-store slots, and the
        # fixed LoRA slots. All are refilled IN PLACE — a fresh-buffer re-point would
        # silently corrupt replay (the loader's own drift assert guards wstage on
        # refill; this is the stage-level belt-and-suspenders across every step).
        var cur = List[Int]()
        for i in range(len(self.wstage.bufs)):
            cur.append(Int(self.wstage.bufs[i].unsafe_ptr()))
        for i in range(len(self.store.d_a)):
            cur.append(Int(self.store.d_a[i][].buf.unsafe_ptr()))
        for i in range(len(self.store.d_b)):
            cur.append(Int(self.store.d_b[i][].buf.unsafe_ptr()))
        for s in range(self.n_slots):
            cur.append(Int(self.la_slots[s][].buf.unsafe_ptr()))
            cur.append(Int(self.lb_slots[s][].buf.unsafe_ptr()))
        if not self.addr_rec:
            self.enc_addr = e; self.vt_addr = vt; self.vp_addr = vp
            self.vcos_addr = vc; self.vsin_addr = vs; self.bin_addr = bn
            self.a_addr = a; self.b_addr = b
            self.pers_addrs = cur^
            self.addr_rec = True
        else:
            if (e != self.enc_addr or vt != self.vt_addr or vp != self.vp_addr
                or vc != self.vcos_addr or vs != self.vsin_addr or bn != self.bin_addr
                or a != self.a_addr or b != self.b_addr):
                raise Error("ltx2 capture: staging buffer address drift across steps"
                            " — captured graph would read stale memory")
            if len(cur) != len(self.pers_addrs):
                raise Error("ltx2 capture: persistent buffer COUNT drift across steps")
            for i in range(len(cur)):
                if cur[i] != self.pers_addrs[i]:
                    raise Error("ltx2 capture: persistent buffer address drift (wstage/"
                                "store/LoRA slot) across steps — captured graph would"
                                " read stale memory")

    # ── conductor ops (OUTSIDE the capture window; sync-free stream-ordered) ──────
    def refill_block(
        mut self, bi: Int, lora_a: List[ArcPointer[Tensor]],
        lora_b: List[ArcPointer[Tensor]], saved_inputs: List[ArcPointer[Tensor]],
        src: LTX2VideoBlockSource, ctx: DeviceContext,
    ) raises:
        """Refill everything the block reads, in place (sync-free): weights
        (standalone stage, seen by the persistent w), LoRA slots (D2D), block_input
        (D2D). MJ-1114: all targets are standalone / offset-0."""
        var blk = src.stream.load_block_bf16_standalone(bi, self.wstage, ctx, False)
        _ = blk^
        var base = bi * self.n_slots
        for s in range(self.n_slots):
            if lora_a[base + s][].nbytes() != self.la_slots[s][].nbytes():
                raise Error("ltx2 capture: LoRA A nbytes mismatch slot " + String(s))
            if lora_b[base + s][].nbytes() != self.lb_slots[s][].nbytes():
                raise Error("ltx2 capture: LoRA B nbytes mismatch slot " + String(s))
            var ad = self.la_slots[s][].buf.copy()
            ctx.enqueue_copy(dst_buf=ad, src_buf=lora_a[base + s][].buf)
            var bd = self.lb_slots[s][].buf.copy()
            ctx.enqueue_copy(dst_buf=bd, src_buf=lora_b[base + s][].buf)
        if saved_inputs[bi][].nbytes() != self.dh_nb:
            raise Error("ltx2 capture: block_input nbytes " + String(saved_inputs[bi][].nbytes())
                        + " != " + String(self.dh_nb))
        var bn = self.bin_buf.copy()
        ctx.enqueue_copy(dst_buf=bn, src_buf=saved_inputs[bi][].buf)

    def after_block(mut self, bi: Int, ctx: DeviceContext) raises:
        """dxA <- dhB (chain this block's d_hidden into the next block's d_video),
        then remap the captured write set store[0] -> store[bi] (skip bi==0: block-0
        IS the write set, processed last)."""
        var a = self.dxA.copy()
        var b = self.dhB.copy()
        ctx.enqueue_copy(dst_buf=a, src_buf=b)
        if bi == 0:
            return
        for s in range(self.n_slots):
            var dst_a = self.store.a_arc(bi, s)
            var src_a = self.store.a_arc(0, s)
            ctx.enqueue_copy(dst_buf=dst_a[].buf, src_buf=src_a[].buf)
            var dst_b = self.store.b_arc(bi, s)
            var src_b = self.store.b_arc(0, s)
            ctx.enqueue_copy(dst_buf=dst_b[].buf, src_buf=src_b[].buf)

    # ── the CAPTURE WINDOW: ONE block's raw slab fwd+bwd+store-write+d_hidden->dhB.
    #    `capture` records it (no execute — caller launches); else eager execute. ──
    def block_work[S_V: Int, N_TXT: Int](
        mut self, bin_st: Tensor, dxA_st: Tensor, enc_st: Tensor, vt_st: Tensor,
        vp_st: Tensor, vcos_st: Tensor, vsin_st: Tensor, eps: Float32,
        capture: Bool, ctx: DeviceContext,
    ) raises:
        self.slab.reset()
        if capture:
            cuda_capture_begin(ctx)
        var fwd = ltx2_video_block_train_forward_slab[S_V, N_TXT](
            self.w, bin_st, enc_st, vt_st, vp_st, vcos_st, vsin_st, eps, ctx, self.slab)
        var bg = ltx2_video_block_backward_slab_dev[S_V, N_TXT](
            self.w, fwd.acts, dxA_st, vt_st, vcos_st, vsin_st, eps, ctx, self.slab,
            self.store, 0)
        var b = self.dhB.copy()
        ctx.enqueue_copy(dst_buf=b, src_buf=bg.d_hidden.buf)   # d_hidden -> dhB
        if capture:
            self.graph = Optional[CudaGraphHandle](cuda_capture_end_instantiate(ctx))
            print("  [ltx2 capture] captured block graph nodes:", self.graph.value().nodes)
        _ = bg^
        _ = fwd^


def ltx2_video_stack_lora_backward_graph_capture[S_V: Int, N_TXT: Int](
    d_pred: Tensor, saved_inputs: List[ArcPointer[Tensor]], x_last: Tensor,
    enc: Tensor, v_temb: Tensor, v_embedded: Tensor, v_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    tail: LTX2VideoTail,
    src: LTX2VideoBlockSource,
    lora_a: List[ArcPointer[Tensor]], lora_b: List[ArcPointer[Tensor]],
    lora_scale: Float32, num_layers: Int, eps: Float32, ctx: DeviceContext,
    mut cap: LTX2CaptureStage,
    preset: Int = 0,
    save_acts_k: Int = 0,
) raises -> LTX2VideoStackGrads:
    """LTX2_V2_CAPTURE per-block CUDA-graph capture/replay driver. Same output as
    `ltx2_video_stack_lora_backward_graph_slab` (bit-identical grads, C14). Per call
    (= one train step): stage per-step inputs into fixed buffers, seed dxA, then
    step 0 = WARMUP (eager all blocks), step 1 = CAPTURE first block + REPLAY rest,
    step >=2 = all REPLAYS. d_input = dxA after the last block; boundary readback of
    the store, same as _slab."""
    if save_acts_k > 0:
        raise Error(
            "ltx2_video_stack_lora_backward_graph_capture: save_acts>0 not carried by"
            " the graph path (full-recompute only); unset LTX2_V2_CAPTURE/LTX2_SAVE_ACTS")
    if not cap.store_active():
        raise Error(
            "ltx2_video_stack_lora_backward_graph_capture: store inactive — the .clone"
            " LoRA path would alloc inside the captured region (impossible)")
    var names = video_lora_names(preset)
    var n_slots = len(names)

    # ── per-step staging (enc/v_temb/v_prompt_ts/v_cos/v_sin; v_cos/v_sin staged
    #    per the DEVIATION note) + seed dxA. ────────────────────────────────────
    var d_seed = _tail_backward[S_V](d_pred, x_last, v_embedded, tail, eps, ctx)
    cap.copy_per_step(enc, v_temb, v_prompt_ts, v_cos, v_sin, d_seed, ctx)
    cap.check_addrs()

    # ── views (reference the FIXED buffers; same handle every block/step). ──────
    var bin_st = _view(cap.bin_buf, saved_inputs[0][].shape(), saved_inputs[0][].dtype())
    var dxA_st = _view(cap.dxA, d_seed.shape(), d_seed.dtype())
    var enc_st = _view(cap.enc_buf, enc.shape(), enc.dtype())
    var vt_st = _view(cap.vt_buf, v_temb.shape(), v_temb.dtype())
    var vp_st = _view(cap.vp_buf, v_prompt_ts.shape(), v_prompt_ts.dtype())
    var vcos_st = _view(cap.vcos_buf, v_cos.shape(), v_cos.dtype())
    var vsin_st = _view(cap.vsin_buf, v_sin.shape(), v_sin.dtype())

    if cap.step == 0:
        # ── WARMUP: eager all blocks (JITs kernels + cuBLAS workspace OUTSIDE any
        #    capture; produces this step's real grads into the store). ──────────
        var bi = num_layers - 1
        while bi >= 0:
            cap.refill_block(bi, lora_a, lora_b, saved_inputs, src, ctx)
            cap.block_work[S_V, N_TXT](
                bin_st, dxA_st, enc_st, vt_st, vp_st, vcos_st, vsin_st, eps, False, ctx)
            cap.after_block(bi, ctx)
            bi -= 1
    elif not cap.graph:
        # ── CAPTURE the first-processed block, LAUNCH it (capture RECORDS, does not
        #    execute), then REPLAY the rest. ────────────────────────────────────
        var bi = num_layers - 1
        cap.refill_block(bi, lora_a, lora_b, saved_inputs, src, ctx)
        cap.block_work[S_V, N_TXT](
            bin_st, dxA_st, enc_st, vt_st, vp_st, vcos_st, vsin_st, eps, True, ctx)
        # cuStreamBeginCapture RECORDS the block, it does NOT execute it (MEASURED
        # 2026-07-17: without this launch step-1's first block's grads are never
        # computed — all 1048576 d_input elems + ~786k d_A/d_B mismatch the oracle).
        # So the captured block, like every replayed block, needs an explicit launch.
        cuda_graph_launch(cap.graph.value(), ctx)
        cap.after_block(bi, ctx)
        bi -= 1
        while bi >= 0:
            cap.refill_block(bi, lora_a, lora_b, saved_inputs, src, ctx)
            cuda_graph_launch(cap.graph.value(), ctx)
            cap.after_block(bi, ctx)
            bi -= 1
    else:
        # ── steady state: all REPLAYS. ────────────────────────────────────────
        var bi = num_layers - 1
        while bi >= 0:
            cap.refill_block(bi, lora_a, lora_b, saved_inputs, src, ctx)
            cuda_graph_launch(cap.graph.value(), ctx)
            cap.after_block(bi, ctx)
            bi -= 1
    cap.step += 1

    # ── boundary readback (store -> host F32) + d_input from dxA. ──────────────
    var n_ad = num_layers * n_slots
    var d_a_flat = List[List[Float32]]()
    var d_b_flat = List[List[Float32]]()
    for _ in range(n_ad):
        d_a_flat.append(List[Float32]())
        d_b_flat.append(List[Float32]())
    for block_bi in range(num_layers):
        var host = cap.store.readback_block(block_bi, ctx)
        var base = block_bi * n_slots
        for s in range(n_slots):
            d_a_flat[base + s] = host[s].d_a.copy()
            d_b_flat[base + s] = host[s].d_b.copy()

    var d_input = _view(cap.dxA, d_seed.shape(), d_seed.dtype()).to_host(ctx)
    var nbad = _nonfinite(d_input)
    for i in range(n_ad):
        nbad += _nonfinite(d_a_flat[i]) + _nonfinite(d_b_flat[i])
    return LTX2VideoStackGrads(d_a_flat^, d_b_flat^, d_input^, nbad)
