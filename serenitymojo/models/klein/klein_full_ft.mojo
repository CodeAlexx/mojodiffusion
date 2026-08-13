# serenitymojo/models/klein/klein_full_ft.mojo
#
# Klein-9B FULL FINETUNE P2+P3 (FULL_FINETUNE_ROLLOUT_PLAN_2026-07-07, klein
# card; the krea2 worked example `reference-mojo-full-finetune` adapted to klein's TWO
# block kinds):
#
#   P2 — KleinHostBf16: the pinned-host bf16 BOTH-WAYS store. All 8 double
#        blocks' 8 matmuls (per stream: qkv/proj/mlp.0/mlp.2 × img+txt) and all
#        24 single blocks' 2 matmuls (linear1/linear2) live as PINNED HOST BF16
#        bytes (~17.4GB host RAM — the store IS the live model). Two device
#        slot-sets PER KIND (double-slot = 8 mats, single-slot = 2 mats),
#        double-buffered on a dedicated copy stream with per-slot H2D-done
#        events (the krea2 Krea2HostBf16 shape). The q/k rms-norm scales stay
#        device-resident and FROZEN (v1 surface = the matmuls; documented delta
#        vs reference trainer Flux2FineTuneSetup which trains all transformer params).
#        klein_host_bf16_save writes the pinned bytes STRAIGHT to a safetensors
#        overlay (original checkpoint tensor names; load base ckpt then these)
#        with no GPU round-trip — an ~17GB save cannot round-trip a 16GB card.
#
#   P3 — klein_stack_ft_forward_streamed / klein_stack_ft_backward_streamed:
#        the streamed FT stack step. Forward walks doubles then singles loading
#        the LIVE weights from the store (block fwd = the parity-verified
#        LoRA-with-None-adapters device-resident forwards, which save the MATH
#        tape the P1 FT backwards consume — saved.q_rope/k_rope/v, NOT the
#        flash tape). Backward walks in reverse: load slot → recompute the
#        block forward from the saved block input → P1 FT `_dev` backward
#        (single_block_ft_backward_dev / double_block_ft_backward_dev, commit
#        553ac64, torch-gated) → fused per-matmul DEVICE Adafactor step
#        (training/adafactor_device.mojo, bit-gated; SR seed per step) mutating
#        the SLOT weights in place → D2H write-back into the pinned host bytes
#        → carry d_x. Per-block ctx.synchronize() (the krea2 discipline) makes
#        slot li%2 reuse safe with no compute-done events and bounds the
#        deferred frees to one block.
#
# Adafactor state flat-index scheme (af_states):
#   doubles first: bi*8 + wi, wi in {img wqkv, img wproj, img wgu, img wd,
#                                    txt wqkv, txt wproj, txt wgu, txt wd}
#   singles next : num_double*8 + bi*2 + wi, wi in {w1, w2}
# (matches DoubleBlockFTGrads.dw / SingleBlockFTGrads.dw order EXACTLY).
#
# TRAPS honored (krea2 skill): dW dtype comes in F32 from the P1 backwards
# (linear_backward_dw(..., STDtype.F32) — the BOOL sentinel means match-input);
# cfg.lr not learning_rate; Tensor is move-only (TArc carriers everywhere);
# "weights changed" checks must diff WHOLE tensors (sub-ulp SR statistics).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import (
    DeviceContext, HostBuffer, DeviceBuffer, DeviceStream, DeviceEvent,
)
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer, alloc
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_close, sys_pwrite, O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.offload.turbo_loader import _h2d_dma_copy
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.tensor_algebra import concat, slice, zeros_device, add
from serenitymojo.ops.linalg_backward import linear_backward_dw
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.norm_backward import layer_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.models.klein.weights import _load_tensor
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.klein.double_block import (
    StreamWeights, DoubleBlockWeights, ModVecsDevice,
    StreamLoraDevice, DoubleBlockLoraDevice,
    double_block_lora_forward_device_resident,
    double_block_ft_backward_dev, DoubleBlockFTGrads,
)
from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecsDevice, SingleBlockSaved,
    SingleBlockLoraDevice,
    single_block_lora_forward_device_resident,
    single_block_lora_recompute_saved_device_resident,
    single_block_ft_backward_dev, SingleBlockFTGrads,
)
from serenitymojo.models.klein.klein_stack import (
    KleinStackBase, KleinStackForward, _ones, _zeros, _t,
)
from serenitymojo.models.klein.double_block import DoubleBlockSaved
from serenitymojo.training.adafactor_device import (
    AdafactorDeviceState, adafactor_step_device, adafactor_step_device_1d,
)
# FULL-FT resume (the fleet sidecar): overlay the trained weights over the
# base-built pinned host store (byte-exact, dtype-checked).
from serenitymojo.training.full_ft_sidecar import full_ft_overlay_into_host_store

comptime TArc = ArcPointer[Tensor]
comptime HArc = ArcPointer[HostBuffer[DType.uint8]]

# matmuls per block kind (the trained v1 surface)
comptime KLEIN_DBL_FT_KEYS = 8   # img qkv/proj/mlp.0/mlp.2 + txt qkv/proj/mlp.0/mlp.2
comptime KLEIN_SGL_FT_KEYS = 2   # linear1, linear2


# ── per-block pinned-host records (LIVE weights) + frozen small tensors ───────
struct KleinDblBlockHostBf16(Copyable, Movable):
    var w_h: List[HArc]          # 8 pinned host BF16 (LIVE weights, slot order)
    var w_nbytes: List[Int]
    var w_shape: List[List[Int]]
    var img_qnorm: TArc          # device-resident, FROZEN
    var img_knorm: TArc
    var txt_qnorm: TArc
    var txt_knorm: TArc

    def __init__(
        out self,
        var w_h: List[HArc], var w_nbytes: List[Int], var w_shape: List[List[Int]],
        var img_qnorm: TArc, var img_knorm: TArc,
        var txt_qnorm: TArc, var txt_knorm: TArc,
    ):
        self.w_h = w_h^
        self.w_nbytes = w_nbytes^
        self.w_shape = w_shape^
        self.img_qnorm = img_qnorm^
        self.img_knorm = img_knorm^
        self.txt_qnorm = txt_qnorm^
        self.txt_knorm = txt_knorm^


struct KleinSglBlockHostBf16(Copyable, Movable):
    var w_h: List[HArc]          # 2 pinned host BF16 (w1, w2)
    var w_nbytes: List[Int]
    var w_shape: List[List[Int]]
    var qnorm: TArc              # device-resident, FROZEN
    var knorm: TArc

    def __init__(
        out self,
        var w_h: List[HArc], var w_nbytes: List[Int], var w_shape: List[List[Int]],
        var qnorm: TArc, var knorm: TArc,
    ):
        self.w_h = w_h^
        self.w_nbytes = w_nbytes^
        self.w_shape = w_shape^
        self.qnorm = qnorm^
        self.knorm = knorm^


struct KleinHostBf16(Copyable, Movable):
    var dbl: List[KleinDblBlockHostBf16]   # len == num_double
    var sgl: List[KleinSglBlockHostBf16]   # len == num_single
    # double-buffer device slots PER KIND + one copy stream + per-slot events.
    var slot_dbl: List[TArc]               # 16 = slot*8 + wi, BF16 device
    var slot_sgl: List[TArc]               # 4  = slot*2 + wi, BF16 device
    var copy_stream: List[ArcPointer[DeviceStream]]   # len 1 (Arc'd: Copyable)
    var ev_dbl: List[ArcPointer[DeviceEvent]]         # len 2: per-slot H2D-done
    var ev_sgl: List[ArcPointer[DeviceEvent]]         # len 2
    var num_double: Int
    var num_single: Int

    def __init__(
        out self,
        var dbl: List[KleinDblBlockHostBf16], var sgl: List[KleinSglBlockHostBf16],
        var slot_dbl: List[TArc], var slot_sgl: List[TArc],
        var copy_stream: List[ArcPointer[DeviceStream]],
        var ev_dbl: List[ArcPointer[DeviceEvent]],
        var ev_sgl: List[ArcPointer[DeviceEvent]],
        num_double: Int, num_single: Int,
    ):
        self.dbl = dbl^
        self.sgl = sgl^
        self.slot_dbl = slot_dbl^
        self.slot_sgl = slot_sgl^
        self.copy_stream = copy_stream^
        self.ev_dbl = ev_dbl^
        self.ev_sgl = ev_sgl^
        self.num_double = num_double
        self.num_single = num_single


# save-key tails, in the SAME order as the slots / the FT dw lists.
def _klein_dbl_ft_key_tails() -> List[String]:
    var t = List[String]()
    t.append(String("img_attn.qkv.weight"))
    t.append(String("img_attn.proj.weight"))
    t.append(String("img_mlp.0.weight"))
    t.append(String("img_mlp.2.weight"))
    t.append(String("txt_attn.qkv.weight"))
    t.append(String("txt_attn.proj.weight"))
    t.append(String("txt_mlp.0.weight"))
    t.append(String("txt_mlp.2.weight"))
    return t^


def _klein_sgl_ft_key_tails() -> List[String]:
    var t = List[String]()
    t.append(String("linear1.weight"))
    t.append(String("linear2.weight"))
    return t^


# D2H one checkpoint matrix into a fresh PINNED host buffer; fail loud on a
# non-BF16 store (the klein transformer mats are BF16 on disk — the reference trainer full-FT
# contract is bf16 weights, NO quantized linears).
def _pin_bf16_weight(
    st: SafeTensors, name: String, ctx: DeviceContext,
    mut w_h: List[HArc], mut w_nbytes: List[Int], mut w_shape: List[List[Int]],
) raises:
    var t = _load_tensor(st, name, ctx)
    if t.dtype() != STDtype.BF16:
        raise Error(
            String("klein full-FT store: ") + name
            + String(" is not BF16 on disk (dtype.tag ") + String(t.dtype().tag)
            + String(") — quantized/other-dtype full-FT is forbidden")
        )
    var sh = t.shape()
    if len(sh) != 2:
        raise Error(String("klein full-FT store: ") + name + String(" is not 2-D"))
    var bh = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
    ctx.enqueue_copy(bh, t.buf)   # D2H
    ctx.synchronize()
    w_h.append(HArc(bh^))
    w_nbytes.append(t.nbytes())
    w_shape.append(sh.copy())


def build_klein_host_bf16(
    st: SafeTensors, num_double: Int, num_single: Int, ctx: DeviceContext
) raises -> KleinHostBf16:
    """Load ALL blocks' trained matmuls as PINNED-HOST BF16 (the full-FT live
    model, ~17.4GB host RAM for klein-9B) + the q/k rms scales device-resident
    (frozen). BF16 bytes are the checkpoint bytes verbatim."""
    var dbl = List[KleinDblBlockHostBf16]()
    var dtails = _klein_dbl_ft_key_tails()
    for bi in range(num_double):
        var p = String("double_blocks.") + String(bi) + String(".")
        var w_h = List[HArc]()
        var w_nbytes = List[Int]()
        var w_shape = List[List[Int]]()
        for ki in range(KLEIN_DBL_FT_KEYS):
            _pin_bf16_weight(st, p + dtails[ki], ctx, w_h, w_nbytes, w_shape)
        dbl.append(KleinDblBlockHostBf16(
            w_h^, w_nbytes^, w_shape^,
            TArc(_load_tensor(st, p + String("img_attn.norm.query_norm.scale"), ctx)),
            TArc(_load_tensor(st, p + String("img_attn.norm.key_norm.scale"), ctx)),
            TArc(_load_tensor(st, p + String("txt_attn.norm.query_norm.scale"), ctx)),
            TArc(_load_tensor(st, p + String("txt_attn.norm.key_norm.scale"), ctx)),
        ))
        cu_mempool_trim_current(0)
        print("full-ft bf16 host store: pinned double block", bi + 1, "/", num_double)
    var sgl = List[KleinSglBlockHostBf16]()
    var stails = _klein_sgl_ft_key_tails()
    for bi in range(num_single):
        var p = String("single_blocks.") + String(bi) + String(".")
        var w_h = List[HArc]()
        var w_nbytes = List[Int]()
        var w_shape = List[List[Int]]()
        for ki in range(KLEIN_SGL_FT_KEYS):
            _pin_bf16_weight(st, p + stails[ki], ctx, w_h, w_nbytes, w_shape)
        sgl.append(KleinSglBlockHostBf16(
            w_h^, w_nbytes^, w_shape^,
            TArc(_load_tensor(st, p + String("norm.query_norm.scale"), ctx)),
            TArc(_load_tensor(st, p + String("norm.key_norm.scale"), ctx)),
        ))
        cu_mempool_trim_current(0)
        if (bi + 1) % 6 == 0 or bi + 1 == num_single:
            print("full-ft bf16 host store: pinned single block", bi + 1, "/", num_single)

    # device slots (shapes uniform within each kind — asserted off block 0)
    var slot_dbl = List[TArc]()
    ref d0 = dbl[0]
    for slot in range(2):
        _ = slot
        for i in range(KLEIN_DBL_FT_KEYS):
            var dbuf = ctx.enqueue_create_buffer[DType.uint8](d0.w_nbytes[i])
            var t = Tensor(dbuf^, d0.w_shape[i].copy(), STDtype.BF16)
            slot_dbl.append(TArc(t^))
    var slot_sgl = List[TArc]()
    ref s0 = sgl[0]
    for slot in range(2):
        _ = slot
        for i in range(KLEIN_SGL_FT_KEYS):
            var dbuf = ctx.enqueue_create_buffer[DType.uint8](s0.w_nbytes[i])
            var t = Tensor(dbuf^, s0.w_shape[i].copy(), STDtype.BF16)
            slot_sgl.append(TArc(t^))
    ctx.synchronize()
    # kind-uniformity fail-loud (slot reuse depends on it)
    for bi in range(num_double):
        for i in range(KLEIN_DBL_FT_KEYS):
            if dbl[bi].w_nbytes[i] != d0.w_nbytes[i]:
                raise Error("klein full-FT store: double-block weight shapes not uniform")
    for bi in range(num_single):
        for i in range(KLEIN_SGL_FT_KEYS):
            if sgl[bi].w_nbytes[i] != s0.w_nbytes[i]:
                raise Error("klein full-FT store: single-block weight shapes not uniform")

    var copy_stream = List[ArcPointer[DeviceStream]]()
    copy_stream.append(ArcPointer(ctx.create_stream()))
    var ev_dbl = List[ArcPointer[DeviceEvent]]()
    ev_dbl.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    ev_dbl.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    var ev_sgl = List[ArcPointer[DeviceEvent]]()
    ev_sgl.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    ev_sgl.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    return KleinHostBf16(
        dbl^, sgl^, slot_dbl^, slot_sgl^, copy_stream^, ev_dbl^, ev_sgl^,
        num_double, num_single,
    )


# ── prefetch (async H2D on the copy stream into slot bi%2, record event) ─────
def klein_host_bf16_prefetch_dbl(store: KleinHostBf16, bi: Int) raises:
    if bi < 0 or bi >= len(store.dbl):
        return
    var slot = bi % 2
    ref b = store.dbl[bi]
    for i in range(KLEIN_DBL_FT_KEYS):
        _h2d_dma_copy(
            UInt64(Int(store.slot_dbl[slot * KLEIN_DBL_FT_KEYS + i][].buf.unsafe_ptr())),
            b.w_h[i][].unsafe_ptr(),
            b.w_nbytes[i],
            store.copy_stream[0][],
        )
    store.copy_stream[0][].record_event(store.ev_dbl[slot][])


def klein_host_bf16_prefetch_sgl(store: KleinHostBf16, bi: Int) raises:
    if bi < 0 or bi >= len(store.sgl):
        return
    var slot = bi % 2
    ref b = store.sgl[bi]
    for i in range(KLEIN_SGL_FT_KEYS):
        _h2d_dma_copy(
            UInt64(Int(store.slot_sgl[slot * KLEIN_SGL_FT_KEYS + i][].buf.unsafe_ptr())),
            b.w_h[i][].unsafe_ptr(),
            b.w_nbytes[i],
            store.copy_stream[0][],
        )
    store.copy_stream[0][].record_event(store.ev_sgl[slot][])


# ── slot loaders (fence the H2D, wrap the PERSISTENT slot tensors) ───────────
def _load_klein_dbl_slot(
    store: KleinHostBf16, bi: Int, ctx: DeviceContext
) raises -> DoubleBlockWeights:
    """Fence slot bi%2's H2D and build DoubleBlockWeights over the SAME slot
    tensors (refcount) — the FT optimizer mutates these in place pre-writeback."""
    var slot = bi % 2
    ctx.stream().enqueue_wait_for(store.ev_dbl[slot][])
    ref b = store.dbl[bi]
    var base = slot * KLEIN_DBL_FT_KEYS
    var img = StreamWeights(
        store.slot_dbl[base + 0].copy(),
        store.slot_dbl[base + 1].copy(),
        store.slot_dbl[base + 2].copy(),
        store.slot_dbl[base + 3].copy(),
        b.img_qnorm.copy(),
        b.img_knorm.copy(),
    )
    var txt = StreamWeights(
        store.slot_dbl[base + 4].copy(),
        store.slot_dbl[base + 5].copy(),
        store.slot_dbl[base + 6].copy(),
        store.slot_dbl[base + 7].copy(),
        b.txt_qnorm.copy(),
        b.txt_knorm.copy(),
    )
    return DoubleBlockWeights(img^, txt^)


def _load_klein_sgl_slot(
    store: KleinHostBf16, bi: Int, D: Int, F: Int, ctx: DeviceContext
) raises -> SingleBlockWeights:
    """Fence slot bi%2's H2D and build SingleBlockWeights over the SAME slot
    tensors. The w2_att/w2_mlp packed views are sliced fresh AFTER the fence
    (enqueued on the default stream), so they see this block's bytes."""
    var slot = bi % 2
    ctx.stream().enqueue_wait_for(store.ev_sgl[slot][])
    ref b = store.sgl[bi]
    var base = slot * KLEIN_SGL_FT_KEYS
    return SingleBlockWeights(
        store.slot_sgl[base + 0].copy(),
        store.slot_sgl[base + 1].copy(),
        b.qnorm.copy(),
        b.knorm.copy(),
        D, F, ctx, True,
    )


def klein_host_bf16_overlay_resume(store: KleinHostBf16, overlay_path: String) raises:
    """RESUME store rebuild, step 2: the store already holds the BASE checkpoint
    (build_klein_host_bf16); copy the trained overlay's bytes over the pinned
    host bytes (byte-exact, BF16/size fail-loud). Key order == klein_host_
    bf16_save's (doubles bi-major then singles bi-major)."""
    var dtails = _klein_dbl_ft_key_tails()
    var stails = _klein_sgl_ft_key_tails()
    var names = List[String]()
    var bufs = List[HArc]()
    var nbytes = List[Int]()
    for bi in range(len(store.dbl)):
        ref b = store.dbl[bi]
        for wi in range(KLEIN_DBL_FT_KEYS):
            names.append(String("double_blocks.") + String(bi) + String(".") + dtails[wi])
            bufs.append(b.w_h[wi].copy())
            nbytes.append(b.w_nbytes[wi])
    for bi in range(len(store.sgl)):
        ref b = store.sgl[bi]
        for wi in range(KLEIN_SGL_FT_KEYS):
            names.append(String("single_blocks.") + String(bi) + String(".") + stails[wi])
            bufs.append(b.w_h[wi].copy())
            nbytes.append(b.w_nbytes[wi])
    full_ft_overlay_into_host_store(overlay_path, names, bufs, nbytes)


def klein_ft_state_shapes[SURF: Bool = False](
    store: KleinHostBf16, mut rows: List[Int], mut cols: List[Int]
) raises:
    """FLAT per-state [rows, cols] in the af_states order (build_klein_ft_
    adafactor_states) — the sidecar loader's expected-shape lists. cols==0 marks
    a 1D UNFACTORED state (the SURF q/k scales)."""
    for _bi in range(store.num_double):
        for wi in range(KLEIN_DBL_FT_KEYS):
            ref sh = store.dbl[0].w_shape[wi]
            rows.append(sh[0])
            cols.append(sh[1])
    for _bi in range(store.num_single):
        for wi in range(KLEIN_SGL_FT_KEYS):
            ref sh = store.sgl[0].w_shape[wi]
            rows.append(sh[0])
            cols.append(sh[1])
    comptime if SURF:
        var dims = _klein_dims_from_store(store)
        var D = dims[0]
        var Dh = dims[1]
        for _bi in range(store.num_double):
            for _wi in range(KLEIN_QK_PER_DBL):
                rows.append(Dh)
                cols.append(0)   # 1D unfactored sentinel
        for _bi in range(store.num_single):
            for _wi in range(KLEIN_QK_PER_SGL):
                rows.append(Dh)
                cols.append(0)
        rows.append(6 * D); cols.append(D)   # img_mod
        rows.append(6 * D); cols.append(D)   # txt_mod
        rows.append(3 * D); cols.append(D)   # single_mod


# ── write-back (D2H the optimizer-updated slot into the pinned host bytes) ───
def klein_host_bf16_writeback_dbl(store: KleinHostBf16, bi: Int, ctx: DeviceContext) raises:
    var slot = bi % 2
    ref b = store.dbl[bi]
    for i in range(KLEIN_DBL_FT_KEYS):
        ctx.enqueue_copy(b.w_h[i][], store.slot_dbl[slot * KLEIN_DBL_FT_KEYS + i][].buf)


def klein_host_bf16_writeback_sgl(store: KleinHostBf16, bi: Int, ctx: DeviceContext) raises:
    var slot = bi % 2
    ref b = store.sgl[bi]
    for i in range(KLEIN_SGL_FT_KEYS):
        ctx.enqueue_copy(b.w_h[i][], store.slot_sgl[slot * KLEIN_SGL_FT_KEYS + i][].buf)


# ── host-direct safetensors overlay save (no GPU round-trip) ─────────────────
def klein_host_bf16_save(store: KleinHostBf16, path: String) raises:
    """Write the trained matmuls (the pinned-host bf16 bytes — the live model)
    DIRECTLY to a safetensors file. Keys keep the ORIGINAL checkpoint names (an
    overlay: load the base ckpt, then these). v1 saves the trained surface only
    (8 mats x num_double + 2 mats x num_single)."""
    var dtails = _klein_dbl_ft_key_tails()
    var stails = _klein_sgl_ft_key_tails()
    # header JSON + offsets (doubles first, then singles — deterministic)
    var header = String("{")
    var off = 0
    var first = True
    for bi in range(len(store.dbl)):
        ref b = store.dbl[bi]
        for wi in range(KLEIN_DBL_FT_KEYS):
            var nm = String("double_blocks.") + String(bi) + String(".") + dtails[wi]
            var n = b.w_nbytes[wi]
            if not first:
                header += String(",")
            first = False
            header += String("\"") + nm + String("\":{\"dtype\":\"BF16\",\"shape\":[")
            header += String(b.w_shape[wi][0]) + String(",") + String(b.w_shape[wi][1])
            header += String("],\"data_offsets\":[") + String(off) + String(",")
            header += String(off + n) + String("]}")
            off += n
    for bi in range(len(store.sgl)):
        ref b = store.sgl[bi]
        for wi in range(KLEIN_SGL_FT_KEYS):
            var nm = String("single_blocks.") + String(bi) + String(".") + stails[wi]
            var n = b.w_nbytes[wi]
            if not first:
                header += String(",")
            first = False
            header += String("\"") + nm + String("\":{\"dtype\":\"BF16\",\"shape\":[")
            header += String(b.w_shape[wi][0]) + String(",") + String(b.w_shape[wi][1])
            header += String("],\"data_offsets\":[") + String(off) + String(",")
            header += String(off + n) + String("]}")
            off += n
    header += String("}")
    var hlen = header.byte_length()

    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("klein_host_bf16_save: cannot open ") + path)
    var lenbuf = alloc[UInt8](8)
    var hl = hlen
    for i in range(8):
        lenbuf[i] = UInt8(hl & 0xFF)
        hl >>= 8
    _ = sys_pwrite(fd, lenbuf, 8, 0)
    lenbuf.free()
    var hptr = BytePtr(unsafe_from_address=Int(header.unsafe_ptr()))
    _ = sys_pwrite(fd, hptr, hlen, 8)
    var base = 8 + hlen
    var woff = 0
    for bi in range(len(store.dbl)):
        ref b = store.dbl[bi]
        for wi in range(KLEIN_DBL_FT_KEYS):
            var src = BytePtr(unsafe_from_address=Int(b.w_h[wi][].unsafe_ptr()))
            var wrote = sys_pwrite(fd, src, b.w_nbytes[wi], base + woff)
            if wrote != b.w_nbytes[wi]:
                _ = sys_close(fd)
                raise Error("klein_host_bf16_save: short write (double)")
            woff += b.w_nbytes[wi]
    for bi in range(len(store.sgl)):
        ref b = store.sgl[bi]
        for wi in range(KLEIN_SGL_FT_KEYS):
            var src = BytePtr(unsafe_from_address=Int(b.w_h[wi][].unsafe_ptr()))
            var wrote = sys_pwrite(fd, src, b.w_nbytes[wi], base + woff)
            if wrote != b.w_nbytes[wi]:
                _ = sys_close(fd)
                raise Error("klein_host_bf16_save: short write (single)")
            woff += b.w_nbytes[wi]
    _ = sys_close(fd)
    print(
        "[klein-ft-save] wrote",
        len(store.dbl) * KLEIN_DBL_FT_KEYS + len(store.sgl) * KLEIN_SGL_FT_KEYS,
        "trained weights ->", path,
    )


# ── Phase B (SURF) save/resume: the q/k rms scales (device-resident in the
# store) + the 3 SHARED mod.lin weights (device-resident, passed in) join the
# overlay. Original checkpoint key names; a v1 (112-tensor) overlay fails loud
# against the 195 count and vice-versa (see full_ft_overlay_into_host_store). ──
def _klein_dbl_qk_key_tails() -> List[String]:
    var t = List[String]()
    t.append(String("img_attn.norm.query_norm.scale"))
    t.append(String("img_attn.norm.key_norm.scale"))
    t.append(String("txt_attn.norm.query_norm.scale"))
    t.append(String("txt_attn.norm.key_norm.scale"))
    return t^


def _klein_sgl_qk_key_tails() -> List[String]:
    var t = List[String]()
    t.append(String("norm.query_norm.scale"))
    t.append(String("norm.key_norm.scale"))
    return t^


def _klein_mod_key_names() -> List[String]:
    var t = List[String]()
    t.append(String("double_stream_modulation_img.lin.weight"))
    t.append(String("double_stream_modulation_txt.lin.weight"))
    t.append(String("single_stream_modulation.lin.weight"))
    return t^


def _klein_dbl_qk_tensor(store: KleinHostBf16, bi: Int, mi: Int) -> TArc:
    ref b = store.dbl[bi]
    if mi == 0:
        return b.img_qnorm.copy()
    if mi == 1:
        return b.img_knorm.copy()
    if mi == 2:
        return b.txt_qnorm.copy()
    return b.txt_knorm.copy()


def _klein_sgl_qk_tensor(store: KleinHostBf16, bi: Int, mi: Int) -> TArc:
    ref b = store.sgl[bi]
    if mi == 0:
        return b.qnorm.copy()
    return b.knorm.copy()


# append one safetensors header entry (BF16) to `header`, advancing `off`.
def _klein_hdr_entry(
    mut header: String, mut off: Int, mut first: Bool,
    nm: String, shape: List[Int], n: Int,
):
    if not first:
        header += String(",")
    first = False
    header += String("\"") + nm + String("\":{\"dtype\":\"BF16\",\"shape\":[")
    for di in range(len(shape)):
        if di > 0:
            header += String(",")
        header += String(shape[di])
    header += String("],\"data_offsets\":[") + String(off) + String(",")
    header += String(off + n) + String("]}")
    off += n


def klein_ft_surf_save(
    store: KleinHostBf16,
    img_mod_w: Tensor, txt_mod_w: Tensor, single_mod_w: Tensor,
    path: String, ctx: DeviceContext,
) raises:
    """Phase B overlay save: the 112 matmuls (pinned host bytes, direct) + the
    80 q/k rms scales (device-resident, D2H) + the 3 shared mod.lin weights
    (device-resident, D2H) = 195 tensors, original checkpoint key names."""
    var dtails = _klein_dbl_ft_key_tails()
    var stails = _klein_sgl_ft_key_tails()
    var dqk = _klein_dbl_qk_key_tails()
    var sqk = _klein_sgl_qk_key_tails()
    var mnames = _klein_mod_key_names()

    # ── header (offsets in save order: matmuls, dbl q/k, sgl q/k, mods) ──
    var header = String("{")
    var off = 0
    var first = True
    for bi in range(len(store.dbl)):
        for wi in range(KLEIN_DBL_FT_KEYS):
            _klein_hdr_entry(header, off, first,
                String("double_blocks.") + String(bi) + String(".") + dtails[wi],
                store.dbl[bi].w_shape[wi], store.dbl[bi].w_nbytes[wi])
    for bi in range(len(store.sgl)):
        for wi in range(KLEIN_SGL_FT_KEYS):
            _klein_hdr_entry(header, off, first,
                String("single_blocks.") + String(bi) + String(".") + stails[wi],
                store.sgl[bi].w_shape[wi], store.sgl[bi].w_nbytes[wi])
    for bi in range(len(store.dbl)):
        for mi in range(KLEIN_QK_PER_DBL):
            var t = _klein_dbl_qk_tensor(store, bi, mi)
            _klein_hdr_entry(header, off, first,
                String("double_blocks.") + String(bi) + String(".") + dqk[mi],
                t[].shape(), t[].nbytes())
    for bi in range(len(store.sgl)):
        for mi in range(KLEIN_QK_PER_SGL):
            var t = _klein_sgl_qk_tensor(store, bi, mi)
            _klein_hdr_entry(header, off, first,
                String("single_blocks.") + String(bi) + String(".") + sqk[mi],
                t[].shape(), t[].nbytes())
    _klein_hdr_entry(header, off, first, mnames[0], img_mod_w.shape(), img_mod_w.nbytes())
    _klein_hdr_entry(header, off, first, mnames[1], txt_mod_w.shape(), txt_mod_w.nbytes())
    _klein_hdr_entry(header, off, first, mnames[2], single_mod_w.shape(), single_mod_w.nbytes())
    header += String("}")
    var hlen = header.byte_length()

    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("klein_ft_surf_save: cannot open ") + path)
    var lenbuf = alloc[UInt8](8)
    var hl = hlen
    for i in range(8):
        lenbuf[i] = UInt8(hl & 0xFF)
        hl >>= 8
    _ = sys_pwrite(fd, lenbuf, 8, 0)
    lenbuf.free()
    var hptr = BytePtr(unsafe_from_address=Int(header.unsafe_ptr()))
    _ = sys_pwrite(fd, hptr, hlen, 8)
    var body_base = 8 + hlen
    var woff = 0

    # matmuls: pinned host bytes, direct pwrite.
    for bi in range(len(store.dbl)):
        for wi in range(KLEIN_DBL_FT_KEYS):
            var n = store.dbl[bi].w_nbytes[wi]
            var src = BytePtr(unsafe_from_address=Int(store.dbl[bi].w_h[wi][].unsafe_ptr()))
            if sys_pwrite(fd, src, n, body_base + woff) != n:
                _ = sys_close(fd); raise Error("klein_ft_surf_save: short write (double)")
            woff += n
    for bi in range(len(store.sgl)):
        for wi in range(KLEIN_SGL_FT_KEYS):
            var n = store.sgl[bi].w_nbytes[wi]
            var src = BytePtr(unsafe_from_address=Int(store.sgl[bi].w_h[wi][].unsafe_ptr()))
            if sys_pwrite(fd, src, n, body_base + woff) != n:
                _ = sys_close(fd); raise Error("klein_ft_surf_save: short write (single)")
            woff += n
    # q/k scales + mods: D2H one-at-a-time (bound host RAM), pwrite, free.
    var dev = List[TArc]()
    for bi in range(len(store.dbl)):
        for mi in range(KLEIN_QK_PER_DBL):
            dev.append(_klein_dbl_qk_tensor(store, bi, mi))
    for bi in range(len(store.sgl)):
        for mi in range(KLEIN_QK_PER_SGL):
            dev.append(_klein_sgl_qk_tensor(store, bi, mi))
    dev.append(TArc(img_mod_w.clone(ctx)))
    dev.append(TArc(txt_mod_w.clone(ctx)))
    dev.append(TArc(single_mod_w.clone(ctx)))
    for i in range(len(dev)):
        var n = dev[i][].nbytes()
        var hb = ctx.enqueue_create_host_buffer[DType.uint8](n)
        ctx.enqueue_copy(hb, dev[i][].buf)
        ctx.synchronize()
        var src = BytePtr(unsafe_from_address=Int(hb.unsafe_ptr()))
        if sys_pwrite(fd, src, n, body_base + woff) != n:
            _ = sys_close(fd); raise Error("klein_ft_surf_save: short write (surf)")
        woff += n
    _ = sys_close(fd)
    var total = (
        len(store.dbl) * KLEIN_DBL_FT_KEYS + len(store.sgl) * KLEIN_SGL_FT_KEYS
        + len(dev)
    )
    print("[klein-ft-save] SURF wrote", total, "trained tensors ->", path)


def klein_ft_surf_overlay_resume(
    store: KleinHostBf16,
    img_mod_w: Tensor, txt_mod_w: Tensor, single_mod_w: Tensor,
    overlay_path: String, ctx: DeviceContext,
) raises:
    """Phase B resume: overlay the 112 matmuls into the pinned host store, then
    H2D the 80 q/k scales + 3 mod.lin over their device-resident tensors."""
    var dtails = _klein_dbl_ft_key_tails()
    var stails = _klein_sgl_ft_key_tails()
    var names = List[String]()
    var bufs = List[HArc]()
    var nbytes = List[Int]()
    for bi in range(len(store.dbl)):
        ref b = store.dbl[bi]
        for wi in range(KLEIN_DBL_FT_KEYS):
            names.append(String("double_blocks.") + String(bi) + String(".") + dtails[wi])
            bufs.append(b.w_h[wi].copy())
            nbytes.append(b.w_nbytes[wi])
    for bi in range(len(store.sgl)):
        ref b = store.sgl[bi]
        for wi in range(KLEIN_SGL_FT_KEYS):
            names.append(String("single_blocks.") + String(bi) + String(".") + stails[wi])
            bufs.append(b.w_h[wi].copy())
            nbytes.append(b.w_nbytes[wi])
    var surf_total = (
        len(names)
        + len(store.dbl) * KLEIN_QK_PER_DBL + len(store.sgl) * KLEIN_QK_PER_SGL
        + KLEIN_NUM_MOD
    )
    full_ft_overlay_into_host_store(
        overlay_path, names, bufs, nbytes, expected_total=surf_total
    )

    var mst = SafeTensors.open(overlay_path)
    var dqk = _klein_dbl_qk_key_tails()
    var sqk = _klein_sgl_qk_key_tails()
    var mnames = _klein_mod_key_names()

    # q/k scales: dst is a refcount COPY of the store's TArc (SAME device
    # buffer) — H2D writes land in the store. Mods write DIRECTLY into the
    # borrowed weight buffers (a clone would be a throwaway copy).
    var onames = List[String]()
    var odst = List[TArc]()
    for bi in range(len(store.dbl)):
        for mi in range(KLEIN_QK_PER_DBL):
            onames.append(String("double_blocks.") + String(bi) + String(".") + dqk[mi])
            odst.append(_klein_dbl_qk_tensor(store, bi, mi))
    for bi in range(len(store.sgl)):
        for mi in range(KLEIN_QK_PER_SGL):
            onames.append(String("single_blocks.") + String(bi) + String(".") + sqk[mi])
            odst.append(_klein_sgl_qk_tensor(store, bi, mi))
    for i in range(len(onames)):
        var t = _load_tensor(mst, onames[i], ctx)
        if t.dtype() != STDtype.BF16 or t.nbytes() != odst[i][].nbytes():
            raise Error(String("klein full-FT resume: ") + onames[i] + String(" dtype/size mismatch"))
        ctx.enqueue_copy(odst[i][].buf, t.buf)
    # mods: direct H2D into the resident weight buffers.
    var mt0 = _load_tensor(mst, mnames[0], ctx)
    if mt0.nbytes() != img_mod_w.nbytes():
        raise Error(String("klein full-FT resume: ") + mnames[0] + String(" size mismatch"))
    ctx.enqueue_copy(img_mod_w.buf, mt0.buf)
    var mt1 = _load_tensor(mst, mnames[1], ctx)
    if mt1.nbytes() != txt_mod_w.nbytes():
        raise Error(String("klein full-FT resume: ") + mnames[1] + String(" size mismatch"))
    ctx.enqueue_copy(txt_mod_w.buf, mt1.buf)
    var mt2 = _load_tensor(mst, mnames[2], ctx)
    if mt2.nbytes() != single_mod_w.nbytes():
        raise Error(String("klein full-FT resume: ") + mnames[2] + String(" size mismatch"))
    ctx.enqueue_copy(single_mod_w.buf, mt2.buf)
    ctx.synchronize()


# ── NO-adapter LoRA shells (the FT forward is the base block) ────────────────
def _ft_dbl_none_lora() -> DoubleBlockLoraDevice:
    var img = StreamLoraDevice(
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
    )
    var txt = StreamLoraDevice(
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
    )
    return DoubleBlockLoraDevice(img^, txt^)


def _ft_sgl_none_lora() -> SingleBlockLoraDevice:
    return SingleBlockLoraDevice(
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
    )


# ── P3 forward: streamed base-block stack forward from the host store ─────────
def klein_stack_ft_forward_streamed[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens_t: TArc, txt_tokens_t: TArc,
    base: KleinStackBase,
    store: KleinHostBf16,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    D: Int, F: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> KleinStackForward:
    """Mirror of klein_stack_lora_forward_device_inputs_offload_turbo_* with the
    LIVE bf16 host store as the weight source and NO adapters. Saves the per-
    block inputs (the backward's recompute checkpoints); per-block
    ctx.synchronize() (slot-reuse + async-free discipline)."""
    var num_double = store.num_double
    var num_single = store.num_single

    klein_host_bf16_prefetch_dbl(store, 0)

    var no_bias = Optional[Tensor](None)
    var img = TArc(linear(img_tokens_t[], base.img_in[], no_bias^, ctx))
    var no_bias_txt = Optional[Tensor](None)
    var txt = TArc(linear(txt_tokens_t[], base.txt_in[], no_bias_txt^, ctx))
    var img_in_act = img.copy()
    var txt_in_act = txt.copy()
    var norm_ones = TArc(_t(_ones(D), [D], ctx))
    var norm_zeros = TArc(_t(_zeros(D), [D], ctx))

    var dbl_img_in = List[TArc]()
    var dbl_txt_in = List[TArc]()
    var dbl_saved = List[DoubleBlockSaved]()
    for bi in range(num_double):
        dbl_img_in.append(img.copy())
        dbl_txt_in.append(txt.copy())
        var w = _load_klein_dbl_slot(store, bi, ctx)
        if bi + 1 < num_double:
            klein_host_bf16_prefetch_dbl(store, bi + 1)
        else:
            klein_host_bf16_prefetch_sgl(store, 0)
        var bl = _ft_dbl_none_lora()
        var fwd = double_block_lora_forward_device_resident[H, Dh, N_IMG, N_TXT, S](
            img, txt, w, img_mod_dev, txt_mod_dev, bl,
            cos_t, sin_t, D, F, eps, ctx,
        )
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        ctx.synchronize()   # slot li%2 reuse safety + bound deferred frees

    var x = TArc(concat(0, ctx, txt[], img[]))

    var sgl_x_in = List[TArc]()
    var sgl_saved = List[SingleBlockSaved]()
    for bi in range(num_single):
        sgl_x_in.append(x.copy())
        var w = _load_klein_sgl_slot(store, bi, D, F, ctx)
        klein_host_bf16_prefetch_sgl(store, bi + 1)
        var sl = _ft_sgl_none_lora()
        var fwd = single_block_lora_forward_device_resident[H, Dh, S](
            x, w, single_mod_dev, sl, cos_t, sin_t, D, F, eps, ctx,
        )
        x = fwd.out.copy()
        ctx.synchronize()

    var img_out = TArc(slice(x[], 0, N_TXT, N_IMG, ctx))
    var ln_img_out = TArc(layer_norm(
        img_out[], norm_ones[], norm_zeros[], eps, ctx,
    ))
    var normed = modulate(
        ln_img_out[], base.final_scale[], base.final_shift[], ctx,
    )
    var no_bias_out = Optional[Tensor](None)
    var out = linear(normed, base.final_lin[], no_bias_out^, ctx).to_host(ctx)

    return KleinStackForward(
        out^, img_in_act^, txt_in_act^,
        dbl_img_in^, dbl_txt_in^, sgl_x_in^,
        dbl_saved^, sgl_saved^,
        img_out^, ln_img_out^,
    )


# ── P3 backward: streamed FT stack backward + fused device Adafactor ─────────
struct KleinStackFTWrite(Movable):
    var grad_count: Int
    var streaming_sync_count: Int

    def __init__(out self, grad_count: Int, streaming_sync_count: Int):
        self.grad_count = grad_count
        self.streaming_sync_count = streaming_sync_count


def klein_stack_ft_backward_streamed[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, SURF: Bool = False
](
    d_out: List[Float32],            # [N_IMG*out_ch] dL/d(velocity)
    base: KleinStackBase,
    store: KleinHostBf16,
    img_mod_dev: ModVecsDevice,
    txt_mod_dev: ModVecsDevice,
    single_mod_dev: SingleModVecsDevice,
    cos_t: Tensor, sin_t: Tensor,
    saved: KleinStackForward,
    D: Int, F: Int, out_ch: Int, eps: Float32,
    mut af_states: List[AdafactorDeviceState],   # flat scheme (module header)
    t_step: Int,                     # 1-based optimizer step
    lr: Float64, beta2_decay: Float64, eps2: Float64, d_thresh: Float64,
    weight_decay: Float64,
    sr_seed: UInt64,                 # per-step seed; 0 = RNE writes
    ctx: DeviceContext,
    # SURF (Phase B) — the SHARED modulation feature silu(vec) [1,D] F32 and the
    # 3 device-resident mod.lin weights (BF16, mutated in place after the walk).
    # Ignored (never read) under SURF=False; the trainer always supplies them.
    vec_silu_t: TArc,
    img_mod_w: Tensor,
    txt_mod_w: Tensor,
    single_mod_w: Tensor,
) raises -> KleinStackFTWrite:
    """DESCENDING walk (singles then doubles, deepest first): load slot →
    recompute the block forward from the saved input → P1 FT `_dev` backward →
    per-matmul fused device Adafactor mutating the slot in place (SR bf16
    write) → D2H write-back into the host store → carry d_x. The fused-back-
    pass shape: grads and updated weights never accumulate beyond one block."""
    var num_double = store.num_double
    var num_single = store.num_single
    var expected_states = num_double * KLEIN_DBL_FT_KEYS + num_single * KLEIN_SGL_FT_KEYS
    comptime if SURF:
        expected_states += (
            num_double * KLEIN_QK_PER_DBL + num_single * KLEIN_QK_PER_SGL + KLEIN_NUM_MOD
        )
    if len(af_states) != expected_states:
        raise Error("klein_stack_ft_backward_streamed: af_states length mismatch")

    var norm_ones = TArc(_t(_ones(D), [D], ctx))

    # SURF: flat-index bases + shared mod.lin grad accumulators (summed across
    # blocks — the krea2 shared-param pattern; klein's mods are SHARED, not
    # per-block). d_*_mod_acc start at zero and take each block's mod_flat.
    var qk_base = _klein_qk_state_base(store)
    var sgl_qk_base = qk_base + num_double * KLEIN_QK_PER_DBL
    var mod_base = _klein_mod_state_base(store)
    var d_img_mod_acc = TArc(zeros_device([1, 6 * D], STDtype.F32, ctx))
    var d_txt_mod_acc = TArc(zeros_device([1, 6 * D], STDtype.F32, ctx))
    var d_single_mod_acc = TArc(zeros_device([1, 3 * D], STDtype.F32, ctx))

    # Prime the DESCENDING walk: deepest single block first.
    klein_host_bf16_prefetch_sgl(store, num_single - 1)

    # frozen final-layer chain (identical to the LoRA turbo backward head)
    var d_normed_t = linear_backward_dx(
        _t(d_out, [N_IMG, out_ch], ctx), base.final_lin[],
        N_IMG, D, out_ch, ctx,
    )
    var mbf = modulate_backward(
        d_normed_t, saved.ln_img_out[], base.final_scale[], ctx, False,
    )
    var d_img_out_t = layer_norm_backward_dx(
        mbf.d_x, saved.img_out[], norm_ones[], eps, ctx,
    )
    var d_txt_zero = zeros_device([N_TXT, D], STDtype.F32, ctx)
    var d_x = TArc(concat(0, ctx, d_txt_zero, d_img_out_t))

    var grad_count = 0
    var streaming_sync_count = 0

    # ── singles, deepest → shallowest ─────────────────────────────────────────
    var bi = num_single - 1
    var sgl_state_base = num_double * KLEIN_DBL_FT_KEYS
    while bi >= 0:
        var w = _load_klein_sgl_slot(store, bi, D, F, ctx)
        if bi > 0:
            klein_host_bf16_prefetch_sgl(store, bi - 1)
        else:
            klein_host_bf16_prefetch_dbl(store, num_double - 1)
        var sl = _ft_sgl_none_lora()
        var block_saved = single_block_lora_recompute_saved_device_resident[H, Dh, S](
            saved.sgl_x_in[bi], w, single_mod_dev, sl,
            cos_t, sin_t, D, F, eps, ctx,
        )
        var bg = single_block_ft_backward_dev[H, Dh, S, SURF](
            d_x, w, single_mod_dev, block_saved, cos_t, sin_t, D, F, eps, ctx,
        )
        # fused back pass: update THIS block's 2 slot weights in place
        var slot = bi % 2
        for wi in range(KLEIN_SGL_FT_KEYS):
            var flat = sgl_state_base + bi * KLEIN_SGL_FT_KEYS + wi
            adafactor_step_device(
                store.slot_sgl[slot * KLEIN_SGL_FT_KEYS + wi][],
                bg.dw[wi][],
                af_states[flat],
                t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                weight_decay,
                sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)),
                ctx,
            )
        klein_host_bf16_writeback_sgl(store, bi, ctx)
        grad_count += KLEIN_SGL_FT_KEYS
        comptime if SURF:
            # q/k rms scales (1D unfactored, device-resident in the store —
            # updated in place, persisted at save). d_qk=[d_qnorm, d_knorm].
            var qf0 = sgl_qk_base + bi * KLEIN_QK_PER_SGL + 0
            adafactor_step_device_1d(
                store.sgl[bi].qnorm[], bg.d_qk[0][], af_states[qf0],
                t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                weight_decay, sr_seed ^ (UInt64(qf0 + 1) * UInt64(0x9E37)), ctx,
            )
            var qf1 = sgl_qk_base + bi * KLEIN_QK_PER_SGL + 1
            adafactor_step_device_1d(
                store.sgl[bi].knorm[], bg.d_qk[1][], af_states[qf1],
                t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                weight_decay, sr_seed ^ (UInt64(qf1 + 1) * UInt64(0x9E37)), ctx,
            )
            # accumulate the SHARED single mod.lin flat grad [1,3D].
            d_single_mod_acc = TArc(add(d_single_mod_acc[], bg.mod_flat[0][], ctx))
            grad_count += KLEIN_QK_PER_SGL
        d_x = bg.d_x.copy()
        ctx.synchronize()   # lands: grads freed, update done, write-back D2H'd
        streaming_sync_count += 1
        bi -= 1

    # split the joint carry back into the two double-block streams
    var d_to = TArc(slice(d_x[], 0, 0, N_TXT, ctx))
    var d_io = TArc(slice(d_x[], 0, N_TXT, N_IMG, ctx))

    # ── doubles, deepest → shallowest ─────────────────────────────────────────
    var di = num_double - 1
    while di >= 0:
        var w = _load_klein_dbl_slot(store, di, ctx)
        klein_host_bf16_prefetch_dbl(store, di - 1)
        var bl = _ft_dbl_none_lora()
        var fwd = double_block_lora_forward_device_resident[H, Dh, N_IMG, N_TXT, S](
            saved.dbl_img_in[di], saved.dbl_txt_in[di],
            w, img_mod_dev, txt_mod_dev, bl, cos_t, sin_t, D, F, eps, ctx,
        )
        var bg = double_block_ft_backward_dev[H, Dh, N_IMG, N_TXT, S, SURF](
            d_io, d_to, w, img_mod_dev, txt_mod_dev, fwd.saved,
            cos_t, sin_t, D, F, eps, ctx,
        )
        var slot = di % 2
        for wi in range(KLEIN_DBL_FT_KEYS):
            var flat = di * KLEIN_DBL_FT_KEYS + wi
            adafactor_step_device(
                store.slot_dbl[slot * KLEIN_DBL_FT_KEYS + wi][],
                bg.dw[wi][],
                af_states[flat],
                t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                weight_decay,
                sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)),
                ctx,
            )
        klein_host_bf16_writeback_dbl(store, di, ctx)
        grad_count += KLEIN_DBL_FT_KEYS
        comptime if SURF:
            # 4 q/k rms scales per double block (device-resident in the store).
            # d_qk=[img_q, img_k, txt_q, txt_k]; store fields in the same order.
            var qbase = qk_base + di * KLEIN_QK_PER_DBL
            adafactor_step_device_1d(
                store.dbl[di].img_qnorm[], bg.d_qk[0][], af_states[qbase + 0],
                t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                weight_decay, sr_seed ^ (UInt64(qbase + 1) * UInt64(0x9E37)), ctx,
            )
            adafactor_step_device_1d(
                store.dbl[di].img_knorm[], bg.d_qk[1][], af_states[qbase + 1],
                t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                weight_decay, sr_seed ^ (UInt64(qbase + 2) * UInt64(0x9E37)), ctx,
            )
            adafactor_step_device_1d(
                store.dbl[di].txt_qnorm[], bg.d_qk[2][], af_states[qbase + 2],
                t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                weight_decay, sr_seed ^ (UInt64(qbase + 3) * UInt64(0x9E37)), ctx,
            )
            adafactor_step_device_1d(
                store.dbl[di].txt_knorm[], bg.d_qk[3][], af_states[qbase + 3],
                t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                weight_decay, sr_seed ^ (UInt64(qbase + 4) * UInt64(0x9E37)), ctx,
            )
            # accumulate SHARED double mod.lin flat grads [1,6D] (img, txt).
            d_img_mod_acc = TArc(add(d_img_mod_acc[], bg.mod_flat[0][], ctx))
            d_txt_mod_acc = TArc(add(d_txt_mod_acc[], bg.mod_flat[1][], ctx))
            grad_count += KLEIN_QK_PER_DBL
        d_io = bg.d_img_x.copy()
        d_to = bg.d_txt_x.copy()
        ctx.synchronize()
        streaming_sync_count += 1
        di -= 1

    comptime if SURF:
        # SHARED mod.lin update ONCE per step: d_W = d_mod_accᵀ @ silu(vec).
        # mod = linear(vec_silu[1,D], W[out,D]) -> [1,out]; grad_y=[1,out] F32,
        # x=vec_silu[1,D] F32 -> linear_backward_dw -> [out,D] F32. Device-
        # resident BF16 weights updated in place (persisted at save).
        var d_img_mod_w = linear_backward_dw(
            d_img_mod_acc[], vec_silu_t[], 1, D, 6 * D, ctx, STDtype.F32,
        )
        adafactor_step_device(
            img_mod_w, d_img_mod_w, af_states[mod_base + 0],
            t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
            weight_decay, sr_seed ^ (UInt64(mod_base + 1) * UInt64(0x9E37)), ctx,
        )
        var d_txt_mod_w = linear_backward_dw(
            d_txt_mod_acc[], vec_silu_t[], 1, D, 6 * D, ctx, STDtype.F32,
        )
        adafactor_step_device(
            txt_mod_w, d_txt_mod_w, af_states[mod_base + 1],
            t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
            weight_decay, sr_seed ^ (UInt64(mod_base + 2) * UInt64(0x9E37)), ctx,
        )
        var d_single_mod_w = linear_backward_dw(
            d_single_mod_acc[], vec_silu_t[], 1, D, 3 * D, ctx, STDtype.F32,
        )
        adafactor_step_device(
            single_mod_w, d_single_mod_w, af_states[mod_base + 2],
            t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
            weight_decay, sr_seed ^ (UInt64(mod_base + 3) * UInt64(0x9E37)), ctx,
        )
        grad_count += KLEIN_NUM_MOD

    return KleinStackFTWrite(grad_count, streaming_sync_count)


# ── Phase B (SURF) flat-index scheme extension ───────────────────────────────
# v1 matmuls occupy [0 .. num_double*8 + num_single*2). SURF appends, in order:
#   q/k rms scales (1D unfactored, cols==0): per double block 4
#     (img_q, img_k, txt_q, txt_k), then per single block 2 (q, k).
#   shared mod.lin (2D factored): [img_mod, txt_mod, single_mod].
comptime KLEIN_MATMUL_STATES_PER_DBL = KLEIN_DBL_FT_KEYS   # 8
comptime KLEIN_MATMUL_STATES_PER_SGL = KLEIN_SGL_FT_KEYS   # 2
comptime KLEIN_QK_PER_DBL = 4
comptime KLEIN_QK_PER_SGL = 2
comptime KLEIN_NUM_MOD = 3   # img_mod, txt_mod, single_mod


def _klein_matmul_state_count(store: KleinHostBf16) -> Int:
    return store.num_double * KLEIN_DBL_FT_KEYS + store.num_single * KLEIN_SGL_FT_KEYS


def _klein_qk_state_base(store: KleinHostBf16) -> Int:
    return _klein_matmul_state_count(store)


def _klein_mod_state_base(store: KleinHostBf16) -> Int:
    return (
        _klein_qk_state_base(store)
        + store.num_double * KLEIN_QK_PER_DBL
        + store.num_single * KLEIN_QK_PER_SGL
    )


def _klein_dims_from_store(store: KleinHostBf16) raises -> Tuple[Int, Int]:
    # D from img qkv shape [3D, D] (w_shape[0][1]); Dh from a q_norm scale [Dh].
    var D = store.dbl[0].w_shape[0][1]
    var Dh = store.dbl[0].img_qnorm[].shape()[0]
    return (D, Dh)


# ── Adafactor factored-state builder (flat scheme, device-resident ~few MB) ──
def build_klein_ft_adafactor_states[SURF: Bool = False](
    store: KleinHostBf16, ctx: DeviceContext
) raises -> List[AdafactorDeviceState]:
    var states = List[AdafactorDeviceState]()
    for _bi in range(store.num_double):
        for wi in range(KLEIN_DBL_FT_KEYS):
            ref sh = store.dbl[0].w_shape[wi]
            states.append(AdafactorDeviceState(sh[0], sh[1], ctx))
    for _bi in range(store.num_single):
        for wi in range(KLEIN_SGL_FT_KEYS):
            ref sh = store.sgl[0].w_shape[wi]
            states.append(AdafactorDeviceState(sh[0], sh[1], ctx))
    comptime if SURF:
        var dims = _klein_dims_from_store(store)
        var D = dims[0]
        var Dh = dims[1]
        # q/k scales: 1D UNFACTORED (cols==0 sentinel via the (n, ctx) ctor).
        for _bi in range(store.num_double):
            for _wi in range(KLEIN_QK_PER_DBL):
                states.append(AdafactorDeviceState(Dh, ctx))
        for _bi in range(store.num_single):
            for _wi in range(KLEIN_QK_PER_SGL):
                states.append(AdafactorDeviceState(Dh, ctx))
        # shared mod.lin: 2D FACTORED. img_mod/txt_mod [6D,D]; single_mod [3D,D].
        states.append(AdafactorDeviceState(6 * D, D, ctx))
        states.append(AdafactorDeviceState(6 * D, D, ctx))
        states.append(AdafactorDeviceState(3 * D, D, ctx))
    return states^
