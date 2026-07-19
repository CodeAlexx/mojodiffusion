# serenitymojo/models/chroma/chroma_full_ft.mojo
#
# Chroma1-HD FULL FINETUNE phases (a)+(b) (FULL_FINETUNE_ROLLOUT_PLAN_2026-07-07
# chroma card; the klein worked example klein_full_ft.mojo adapted to chroma —
# chroma's blocks ARE flux's, P1 backwards live flux-side, chroma re-exports):
#
#   (a) ChromaHostBf16: the pinned-host bf16 BOTH-WAYS store. v2 FULL SURFACE
#       (FULL_SURFACE_PLAN_2026-07-08 Phase B row 3): all 19 double blocks'
#       20 trained per-block params (FT_DBL_SLOT order — slots 0-7 the 8
#       matmuls img/txt × qkv/proj/mlp.0/mlp.2; slots 8-15 the 8 linear
#       biases; slots 16-19 the 4 q/k rms-norm scales) and all 38 single
#       blocks' 6 trained params (FT_SGL_SLOT order — linear1/linear2 weights,
#       b1/b2 biases, q/k rms scales) live as PINNED HOST BF16 bytes
#       (~17.2GB host RAM — the store IS the live model; ~8.609B trained
#       params of the 8.9B model). The 1D params are bf16 on disk and pin
#       byte-exact, same as the matmuls (the BFL ckpt is all-BF16). Two device
#       slot-sets PER KIND (double-slot = 20 tensors, single-slot = 6),
#       double-buffered on a dedicated copy stream with per-slot H2D-done
#       events (the krea2/klein shape). The BIAS TRAP (plan traps list) is
#       resolved HERE: biases are forward-needed and now TRAINED, so they
#       moved from the old device-resident frozen fields INTO the pinned
#       store + streaming slots — the slot loaders below hand the forward
#       (StreamWeights/SingleBlockWeights) the SLOT tensors, so the optimizer-
#       updated bias values are what the next forward reads. Still frozen +
#       device-resident: the stack base (img_in/txt_in/final_layer) and the
#       distilled-guidance approximator (chroma's mod source — genuinely
#       frozen; Phase C candidate). chroma_host_bf16_save writes the pinned
#       bytes STRAIGHT to a safetensors overlay (ORIGINAL BFL checkpoint
#       tensor names, 608 tensors: double_blocks.N.{img,txt}_{attn.qkv,
#       attn.proj,mlp.0,mlp.2}.{weight,bias} + double_blocks.N.{img,txt}_attn.
#       norm.{query,key}_norm.scale, single_blocks.N.linear{1,2}.{weight,bias}
#       + single_blocks.N.norm.{query,key}_norm.scale) with no GPU round-trip.
#       v1 overlays (228 tensors) and v1 sidecars (228 states) FAIL LOUD on
#       resume (count mismatches) — the v2 surface REPLACES v1.
#
#   (b) chroma_stack_ft_forward_streamed / chroma_stack_ft_backward_streamed:
#       the streamed FT stack step. Forward walks doubles then singles loading
#       the LIVE weights from the store (block fwd = the parity-verified
#       chroma_*_lora_forward_device with NO adapters — saves the MATH tape
#       saved.q_rope/k_rope/v the P1 FT backwards consume; the same fwd the FT
#       parity gate chroma_block_ft_parity.mojo pairs with the backwards).
#       Per-block mod vecs are SLICED ROWS of the per-step pooled_temb table
#       (the frozen approximator output — chroma_stack_lora.mojo layout).
#       Backward walks in reverse: load slot → recompute the block forward
#       from the saved block input → chroma_*_ft_backward_dev[SURFACE_V2=True]
#       (flux_block_ft.mojo; torch-gated: v1 arms cos >= 0.9997, v2 bias/norm
#       arms at the 0.9999 Phase B bar) → fused per-param DEVICE Adafactor
#       step (training/adafactor_device.mojo, bit-gated; 2D factored / 1D
#       unfactored dispatch on state.factored; per-weight SR seed mix)
#       mutating the SLOT weights in place → D2H write-back into the pinned
#       host bytes → carry d_x (doubles carry d_img_x + d_txt_x). Per-block
#       ctx.synchronize() (the krea2 discipline) makes slot li%2 reuse safe
#       with no compute-done events and bounds the deferred frees to a block.
#
# BFL-KEY LOADERS: the on-box chroma1_hd_bf16.safetensors is BFL-format
# (double_blocks/single_blocks, PRE-FUSED qkv/linear1 — verified 643 keys all
# BF16). The in-repo diffusers-key loaders (weights.mojo / ChromaDitCache.load
# name lookups) do not match it, so this file owns BFL loaders for the store,
# the stack base (img_in/txt_in/final_layer.linear -> ChromaStackBase), and the
# approximator (load_chroma_dit_cache_bfl: BFL layers.N.{in,out}_layer /
# norms.N.scale registered under the diffusers alias names
# layers.N.linear_{1,2} / norms.N.weight that ChromaDitCache.approximator_
# forward looks up — the math is identical, only the names differ).
#
# Adafactor state flat-index scheme (af_states):
#   doubles first: bi*20 + wi, wi in FT_DBL_SLOT order (8 matmuls, 8 biases,
#                  4 q/k rms scales)
#   singles next : num_double*20 + bi*6 + wi, wi in FT_SGL_SLOT order
#                  (w1, w2, b1, b2, q_norm, k_norm)
# (matches FluxDoubleBlockFTGrads.dw / FluxSingleBlockFTGrads.dw SURFACE_V2
# order EXACTLY). Matmul slots are FACTORED (2D) states; bias/scale slots are
# UNFACTORED (1D, cols==0 sentinel — torch uses the unfactored second moment
# for <2D params).
#
# TRAPS honored (reference-mojo-full-finetune): dW arrives F32 from the P1 backwards
# (linear_backward_dw(..., STDtype.F32) — the BOOL sentinel means match-input);
# cfg.lr not learning_rate; Tensor is move-only (TArc carriers everywhere);
# "weights changed" checks must diff WHOLE tensors (sub-ulp SR statistics).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import (
    DeviceContext, HostBuffer, DeviceBuffer, DeviceStream, DeviceEvent,
)
from std.collections import List, Optional, Dict
from std.memory import ArcPointer, alloc
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_close, sys_pwrite, O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.offload.turbo_loader import _h2d_dma_copy
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.tensor_algebra import concat, slice, zeros_device
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.norm_backward import layer_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward

# flux-shaped block weight carriers (chroma's blocks ARE flux's; the BFL ckpt
# stores the fused layout these expect verbatim).
from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)
from serenitymojo.models.flux.flux_stack import _ones, _zeros, _t

# parity-verified device-resident block forwards (save the FT backward's tape)
# + the mod-vec device carriers.
from serenitymojo.models.chroma.chroma_block_device import (
    ModVecsDevice, SingleModVecsDevice,
    ChromaLoraAdapterDevice, StreamLoraDevice,
    DoubleBlockLoraDevice, SingleBlockLoraDevice,
    modvecs_to_device, single_modvecs_to_device,
    chroma_double_block_lora_forward_device,
    chroma_single_block_lora_forward_device,
)
# P1 FT block backwards (flux-side impl, chroma re-exports; torch-gated) +
# the v2 full-surface slot constants (shared flux/chroma FIXED order).
from serenitymojo.models.chroma.chroma_block import (
    chroma_double_block_ft_backward_dev, chroma_single_block_ft_backward_dev,
    FT_DBL_SLOT_IMG_BQKV, FT_DBL_SLOT_IMG_BPROJ,
    FT_DBL_SLOT_IMG_BMLP0, FT_DBL_SLOT_IMG_BMLP2,
    FT_DBL_SLOT_TXT_BQKV, FT_DBL_SLOT_TXT_BPROJ,
    FT_DBL_SLOT_TXT_BMLP0, FT_DBL_SLOT_TXT_BMLP2,
    FT_DBL_SLOT_IMG_QNORM, FT_DBL_SLOT_IMG_KNORM,
    FT_DBL_SLOT_TXT_QNORM, FT_DBL_SLOT_TXT_KNORM,
    FT_DBL_SLOTS_V2,
    FT_SGL_SLOT_B1, FT_SGL_SLOT_B2, FT_SGL_SLOT_QNORM, FT_SGL_SLOT_KNORM,
    FT_SGL_SLOTS_V2,
)
# pooled_temb row-slicing helpers + the frozen stack base carrier (shared with
# the LoRA stack — the modulation layout is chroma_dit.mojo's, gate-verified).
from serenitymojo.models.chroma.chroma_stack_lora import (
    ChromaStackBase,
    _dbl_img_mod_flat, _dbl_txt_mod_flat, _sgl_mod_flat, _final_shift_scale,
    _modvecs_from_flat6, _single_modvecs_from_flat3,
    _t_like, _dev_f32, _dev_bf16,
)
from serenitymojo.models.dit.chroma_dit import ChromaDitCache, ChromaConfig
from serenitymojo.training.adafactor_device import (
    AdafactorDeviceState, adafactor_step_device, adafactor_step_device_1d,
)
# FULL-FT resume (the fleet sidecar): overlay the trained weights over the
# base-built pinned host store (byte-exact, dtype-checked).
from serenitymojo.training.full_ft_sidecar import full_ft_overlay_into_host_store

comptime TArc = ArcPointer[Tensor]
comptime HArc = ArcPointer[HostBuffer[DType.uint8]]

# trained params per block kind (the v2 FULL surface; FT_*_SLOT order — the
# leading MM slots are the fp-checkpoint 2D matmuls, the rest are 1D params).
comptime CHROMA_DBL_FT_MM_KEYS = 8              # img/txt qkv/proj/mlp.0/mlp.2 weights
comptime CHROMA_DBL_FT_KEYS = FT_DBL_SLOTS_V2   # 20: + 8 biases + 4 q/k rms scales
comptime CHROMA_SGL_FT_MM_KEYS = 2              # linear1/linear2 weights
comptime CHROMA_SGL_FT_KEYS = FT_SGL_SLOTS_V2   # 6: + b1/b2 + q/k rms scales


# ── BFL-key device loader (storage dtype preserved) ──────────────────────────
def _load_tensor_bfl(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


# ── per-block pinned-host records (ALL LIVE trained tensors, slot order) ──────
struct ChromaDblBlockHostBf16(Copyable, Movable):
    var w_h: List[HArc]          # 20 pinned host BF16 (LIVE params, FT_DBL_SLOT order)
    var w_nbytes: List[Int]
    var w_shape: List[List[Int]] # 2-D for slots 0-7, 1-D for slots 8-19

    def __init__(
        out self,
        var w_h: List[HArc], var w_nbytes: List[Int], var w_shape: List[List[Int]],
    ):
        self.w_h = w_h^
        self.w_nbytes = w_nbytes^
        self.w_shape = w_shape^


struct ChromaSglBlockHostBf16(Copyable, Movable):
    var w_h: List[HArc]          # 6 pinned host BF16 (LIVE params, FT_SGL_SLOT order)
    var w_nbytes: List[Int]
    var w_shape: List[List[Int]] # 2-D for slots 0-1, 1-D for slots 2-5

    def __init__(
        out self,
        var w_h: List[HArc], var w_nbytes: List[Int], var w_shape: List[List[Int]],
    ):
        self.w_h = w_h^
        self.w_nbytes = w_nbytes^
        self.w_shape = w_shape^


struct ChromaHostBf16(Copyable, Movable):
    var dbl: List[ChromaDblBlockHostBf16]  # len == num_double
    var sgl: List[ChromaSglBlockHostBf16]  # len == num_single
    # double-buffer device slots PER KIND + one copy stream + per-slot events.
    var slot_dbl: List[TArc]               # 40 = slot*20 + wi, BF16 device
    var slot_sgl: List[TArc]               # 12 = slot*6 + wi, BF16 device
    var copy_stream: List[ArcPointer[DeviceStream]]   # len 1 (Arc'd: Copyable)
    var ev_dbl: List[ArcPointer[DeviceEvent]]         # len 2: per-slot H2D-done
    var ev_sgl: List[ArcPointer[DeviceEvent]]         # len 2
    var num_double: Int
    var num_single: Int

    def __init__(
        out self,
        var dbl: List[ChromaDblBlockHostBf16], var sgl: List[ChromaSglBlockHostBf16],
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


# save-key tails, in the SAME order as the slots / the FT dw lists (BFL names;
# slots 0-7 the v1 matmuls, 8-19 the v2 1D biases + q/k rms scales).
def _chroma_dbl_ft_key_tails() -> List[String]:
    var t = List[String]()
    t.append(String("img_attn.qkv.weight"))
    t.append(String("img_attn.proj.weight"))
    t.append(String("img_mlp.0.weight"))
    t.append(String("img_mlp.2.weight"))
    t.append(String("txt_attn.qkv.weight"))
    t.append(String("txt_attn.proj.weight"))
    t.append(String("txt_mlp.0.weight"))
    t.append(String("txt_mlp.2.weight"))
    t.append(String("img_attn.qkv.bias"))                # FT_DBL_SLOT_IMG_BQKV
    t.append(String("img_attn.proj.bias"))               # FT_DBL_SLOT_IMG_BPROJ
    t.append(String("img_mlp.0.bias"))                   # FT_DBL_SLOT_IMG_BMLP0
    t.append(String("img_mlp.2.bias"))                   # FT_DBL_SLOT_IMG_BMLP2
    t.append(String("txt_attn.qkv.bias"))                # FT_DBL_SLOT_TXT_BQKV
    t.append(String("txt_attn.proj.bias"))               # FT_DBL_SLOT_TXT_BPROJ
    t.append(String("txt_mlp.0.bias"))                   # FT_DBL_SLOT_TXT_BMLP0
    t.append(String("txt_mlp.2.bias"))                   # FT_DBL_SLOT_TXT_BMLP2
    t.append(String("img_attn.norm.query_norm.scale"))   # FT_DBL_SLOT_IMG_QNORM
    t.append(String("img_attn.norm.key_norm.scale"))     # FT_DBL_SLOT_IMG_KNORM
    t.append(String("txt_attn.norm.query_norm.scale"))   # FT_DBL_SLOT_TXT_QNORM
    t.append(String("txt_attn.norm.key_norm.scale"))     # FT_DBL_SLOT_TXT_KNORM
    return t^


def _chroma_sgl_ft_key_tails() -> List[String]:
    var t = List[String]()
    t.append(String("linear1.weight"))
    t.append(String("linear2.weight"))
    t.append(String("linear1.bias"))                     # FT_SGL_SLOT_B1
    t.append(String("linear2.bias"))                     # FT_SGL_SLOT_B2
    t.append(String("norm.query_norm.scale"))            # FT_SGL_SLOT_QNORM
    t.append(String("norm.key_norm.scale"))              # FT_SGL_SLOT_KNORM
    return t^


# D2H one checkpoint matrix into a fresh PINNED host buffer; fail loud on a
# non-BF16 store (the reference trainer full-FT contract is bf16 weights, NO quantized linears).
def _pin_bf16_weight(
    st: SafeTensors, name: String, ctx: DeviceContext,
    mut w_h: List[HArc], mut w_nbytes: List[Int], mut w_shape: List[List[Int]],
) raises:
    var t = _load_tensor_bfl(st, name, ctx)
    if t.dtype() != STDtype.BF16:
        raise Error(
            String("chroma full-FT store: ") + name
            + String(" is not BF16 on disk (dtype.tag ") + String(t.dtype().tag)
            + String(") — quantized/other-dtype full-FT is forbidden")
        )
    var sh = t.shape()
    if len(sh) != 2:
        raise Error(String("chroma full-FT store: ") + name + String(" is not 2-D"))
    var bh = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
    ctx.enqueue_copy(bh, t.buf)   # D2H
    ctx.synchronize()
    w_h.append(HArc(bh^))
    w_nbytes.append(t.nbytes())
    w_shape.append(sh.copy())


# bf16-on-disk 1D param (linear bias / q/k rms scale) -> byte-exact PINNED host
# copy (no conversion — the BFL ckpt is all-BF16). Fail loud on non-BF16 or
# non-1-D (quantized/other-dtype full-FT forbidden).
def _pin_bf16_1d_weight(
    st: SafeTensors, name: String, ctx: DeviceContext,
    mut w_h: List[HArc], mut w_nbytes: List[Int], mut w_shape: List[List[Int]],
) raises:
    var t = _load_tensor_bfl(st, name, ctx)
    if t.dtype() != STDtype.BF16:
        raise Error(
            String("chroma full-FT store: ") + name
            + String(" is not BF16 on disk (dtype.tag ") + String(t.dtype().tag)
            + String(") — quantized/other-dtype full-FT is forbidden")
        )
    var sh = t.shape()
    if len(sh) != 1:
        raise Error(String("chroma full-FT store: ") + name + String(" is not 1-D"))
    var bh = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
    ctx.enqueue_copy(bh, t.buf)   # D2H
    ctx.synchronize()
    w_h.append(HArc(bh^))
    w_nbytes.append(t.nbytes())
    w_shape.append(sh.copy())


def build_chroma_host_bf16(
    st: SafeTensors, num_double: Int, num_single: Int, ctx: DeviceContext
) raises -> ChromaHostBf16:
    """Load ALL blocks' trained params as PINNED-HOST BF16 (the full-FT live
    model, ~17.2GB host RAM for chroma1-hd): the leading MM slots the 2D
    matmuls, the rest the 1D biases + q/k rms scales (the v2 FULL surface —
    nothing per-block stays frozen). BF16 bytes are the checkpoint bytes
    verbatim."""
    var dbl = List[ChromaDblBlockHostBf16]()
    var dtails = _chroma_dbl_ft_key_tails()
    for bi in range(num_double):
        var p = String("double_blocks.") + String(bi) + String(".")
        var w_h = List[HArc]()
        var w_nbytes = List[Int]()
        var w_shape = List[List[Int]]()
        for ki in range(CHROMA_DBL_FT_MM_KEYS):
            _pin_bf16_weight(st, p + dtails[ki], ctx, w_h, w_nbytes, w_shape)
        for ki in range(CHROMA_DBL_FT_MM_KEYS, CHROMA_DBL_FT_KEYS):
            _pin_bf16_1d_weight(st, p + dtails[ki], ctx, w_h, w_nbytes, w_shape)
        dbl.append(ChromaDblBlockHostBf16(w_h^, w_nbytes^, w_shape^))
        cu_mempool_trim_current(0)
        if (bi + 1) % 4 == 0 or bi + 1 == num_double:
            print("full-ft bf16 host store: pinned double block", bi + 1, "/", num_double)
    var sgl = List[ChromaSglBlockHostBf16]()
    var stails = _chroma_sgl_ft_key_tails()
    for bi in range(num_single):
        var p = String("single_blocks.") + String(bi) + String(".")
        var w_h = List[HArc]()
        var w_nbytes = List[Int]()
        var w_shape = List[List[Int]]()
        for ki in range(CHROMA_SGL_FT_MM_KEYS):
            _pin_bf16_weight(st, p + stails[ki], ctx, w_h, w_nbytes, w_shape)
        for ki in range(CHROMA_SGL_FT_MM_KEYS, CHROMA_SGL_FT_KEYS):
            _pin_bf16_1d_weight(st, p + stails[ki], ctx, w_h, w_nbytes, w_shape)
        sgl.append(ChromaSglBlockHostBf16(w_h^, w_nbytes^, w_shape^))
        cu_mempool_trim_current(0)
        if (bi + 1) % 8 == 0 or bi + 1 == num_single:
            print("full-ft bf16 host store: pinned single block", bi + 1, "/", num_single)

    # device slots (shapes uniform within each kind — asserted off block 0)
    var slot_dbl = List[TArc]()
    ref d0 = dbl[0]
    for slot in range(2):
        _ = slot
        for i in range(CHROMA_DBL_FT_KEYS):
            var dbuf = ctx.enqueue_create_buffer[DType.uint8](d0.w_nbytes[i])
            var t = Tensor(dbuf^, d0.w_shape[i].copy(), STDtype.BF16)
            slot_dbl.append(TArc(t^))
    var slot_sgl = List[TArc]()
    ref s0 = sgl[0]
    for slot in range(2):
        _ = slot
        for i in range(CHROMA_SGL_FT_KEYS):
            var dbuf = ctx.enqueue_create_buffer[DType.uint8](s0.w_nbytes[i])
            var t = Tensor(dbuf^, s0.w_shape[i].copy(), STDtype.BF16)
            slot_sgl.append(TArc(t^))
    ctx.synchronize()
    # kind-uniformity fail-loud (slot reuse depends on it)
    for bi in range(num_double):
        for i in range(CHROMA_DBL_FT_KEYS):
            if dbl[bi].w_nbytes[i] != d0.w_nbytes[i]:
                raise Error("chroma full-FT store: double-block weight shapes not uniform")
    for bi in range(num_single):
        for i in range(CHROMA_SGL_FT_KEYS):
            if sgl[bi].w_nbytes[i] != s0.w_nbytes[i]:
                raise Error("chroma full-FT store: single-block weight shapes not uniform")

    var copy_stream = List[ArcPointer[DeviceStream]]()
    copy_stream.append(ArcPointer(ctx.create_stream()))
    var ev_dbl = List[ArcPointer[DeviceEvent]]()
    ev_dbl.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    ev_dbl.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    var ev_sgl = List[ArcPointer[DeviceEvent]]()
    ev_sgl.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    ev_sgl.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    return ChromaHostBf16(
        dbl^, sgl^, slot_dbl^, slot_sgl^, copy_stream^, ev_dbl^, ev_sgl^,
        num_double, num_single,
    )


# ── prefetch (async H2D on the copy stream into slot bi%2, record event) ─────
def chroma_host_bf16_prefetch_dbl(store: ChromaHostBf16, bi: Int) raises:
    if bi < 0 or bi >= len(store.dbl):
        return
    var slot = bi % 2
    ref b = store.dbl[bi]
    for i in range(CHROMA_DBL_FT_KEYS):
        _h2d_dma_copy(
            UInt64(Int(store.slot_dbl[slot * CHROMA_DBL_FT_KEYS + i][].buf.unsafe_ptr())),
            b.w_h[i][].unsafe_ptr(),
            b.w_nbytes[i],
            store.copy_stream[0][],
        )
    store.copy_stream[0][].record_event(store.ev_dbl[slot][])


def chroma_host_bf16_prefetch_sgl(store: ChromaHostBf16, bi: Int) raises:
    if bi < 0 or bi >= len(store.sgl):
        return
    var slot = bi % 2
    ref b = store.sgl[bi]
    for i in range(CHROMA_SGL_FT_KEYS):
        _h2d_dma_copy(
            UInt64(Int(store.slot_sgl[slot * CHROMA_SGL_FT_KEYS + i][].buf.unsafe_ptr())),
            b.w_h[i][].unsafe_ptr(),
            b.w_nbytes[i],
            store.copy_stream[0][],
        )
    store.copy_stream[0][].record_event(store.ev_sgl[slot][])


# ── slot loaders (fence the H2D, wrap the PERSISTENT slot tensors) ───────────
def _load_chroma_dbl_slot(
    store: ChromaHostBf16, bi: Int, ctx: DeviceContext
) raises -> DoubleBlockWeights:
    """Fence slot bi%2's H2D and build DoubleBlockWeights over the SAME slot
    tensors (refcount) — the FT optimizer mutates these in place pre-writeback.
    ALL 20 per-block params are trained (v2) and stream from the slots — the
    forward reads the BIASES and q/k rms scales from the slots too, so the
    trained values apply (the plan's bias trap). Slot map = FT_DBL_SLOT
    order."""
    var slot = bi % 2
    ctx.stream().enqueue_wait_for(store.ev_dbl[slot][])
    var base = slot * CHROMA_DBL_FT_KEYS
    var img = StreamWeights(
        store.slot_dbl[base + 0].copy(),
        store.slot_dbl[base + FT_DBL_SLOT_IMG_BQKV].copy(),
        store.slot_dbl[base + 1].copy(),
        store.slot_dbl[base + FT_DBL_SLOT_IMG_BPROJ].copy(),
        store.slot_dbl[base + 2].copy(),
        store.slot_dbl[base + FT_DBL_SLOT_IMG_BMLP0].copy(),
        store.slot_dbl[base + 3].copy(),
        store.slot_dbl[base + FT_DBL_SLOT_IMG_BMLP2].copy(),
        store.slot_dbl[base + FT_DBL_SLOT_IMG_QNORM].copy(),
        store.slot_dbl[base + FT_DBL_SLOT_IMG_KNORM].copy(),
    )
    var txt = StreamWeights(
        store.slot_dbl[base + 4].copy(),
        store.slot_dbl[base + FT_DBL_SLOT_TXT_BQKV].copy(),
        store.slot_dbl[base + 5].copy(),
        store.slot_dbl[base + FT_DBL_SLOT_TXT_BPROJ].copy(),
        store.slot_dbl[base + 6].copy(),
        store.slot_dbl[base + FT_DBL_SLOT_TXT_BMLP0].copy(),
        store.slot_dbl[base + 7].copy(),
        store.slot_dbl[base + FT_DBL_SLOT_TXT_BMLP2].copy(),
        store.slot_dbl[base + FT_DBL_SLOT_TXT_QNORM].copy(),
        store.slot_dbl[base + FT_DBL_SLOT_TXT_KNORM].copy(),
    )
    return DoubleBlockWeights(img^, txt^)


def _load_chroma_sgl_slot(
    store: ChromaHostBf16, bi: Int, ctx: DeviceContext
) raises -> SingleBlockWeights:
    """Fence slot bi%2's H2D and build SingleBlockWeights over the SAME slot
    tensors (refcount); ALL 6 per-block params trained (v2) and streamed —
    b1/b2 + q/k rms scales included (FT_SGL_SLOT order)."""
    var slot = bi % 2
    ctx.stream().enqueue_wait_for(store.ev_sgl[slot][])
    var base = slot * CHROMA_SGL_FT_KEYS
    return SingleBlockWeights(
        store.slot_sgl[base + 0].copy(),
        store.slot_sgl[base + FT_SGL_SLOT_B1].copy(),
        store.slot_sgl[base + 1].copy(),
        store.slot_sgl[base + FT_SGL_SLOT_B2].copy(),
        store.slot_sgl[base + FT_SGL_SLOT_QNORM].copy(),
        store.slot_sgl[base + FT_SGL_SLOT_KNORM].copy(),
    )


# ── write-back (D2H the optimizer-updated slot into the pinned host bytes) ───
def chroma_host_bf16_writeback_dbl(store: ChromaHostBf16, bi: Int, ctx: DeviceContext) raises:
    var slot = bi % 2
    ref b = store.dbl[bi]
    for i in range(CHROMA_DBL_FT_KEYS):
        ctx.enqueue_copy(b.w_h[i][], store.slot_dbl[slot * CHROMA_DBL_FT_KEYS + i][].buf)


def chroma_host_bf16_writeback_sgl(store: ChromaHostBf16, bi: Int, ctx: DeviceContext) raises:
    var slot = bi % 2
    ref b = store.sgl[bi]
    for i in range(CHROMA_SGL_FT_KEYS):
        ctx.enqueue_copy(b.w_h[i][], store.slot_sgl[slot * CHROMA_SGL_FT_KEYS + i][].buf)


# ── host-direct safetensors overlay save (no GPU round-trip) ─────────────────
def chroma_host_bf16_save(store: ChromaHostBf16, path: String) raises:
    """Write the trained surface (the pinned-host bf16 bytes — the live model)
    DIRECTLY to a safetensors file. Keys keep the ORIGINAL BFL checkpoint names
    (an overlay: load the base ckpt, then these). v2 saves the FULL per-block
    surface (20 x num_double + 6 x num_single = 608 tensors: matmuls + biases
    + q/k rms scales)."""
    var dtails = _chroma_dbl_ft_key_tails()
    var stails = _chroma_sgl_ft_key_tails()
    # header JSON + offsets (doubles first, then singles — deterministic)
    var header = String("{")
    var off = 0
    var first = True
    for bi in range(len(store.dbl)):
        ref b = store.dbl[bi]
        for wi in range(CHROMA_DBL_FT_KEYS):
            var nm = String("double_blocks.") + String(bi) + String(".") + dtails[wi]
            var n = b.w_nbytes[wi]
            if not first:
                header += String(",")
            first = False
            header += String("\"") + nm + String("\":{\"dtype\":\"BF16\",\"shape\":[")
            for di in range(len(b.w_shape[wi])):
                if di > 0:
                    header += String(",")
                header += String(b.w_shape[wi][di])
            header += String("],\"data_offsets\":[") + String(off) + String(",")
            header += String(off + n) + String("]}")
            off += n
    for bi in range(len(store.sgl)):
        ref b = store.sgl[bi]
        for wi in range(CHROMA_SGL_FT_KEYS):
            var nm = String("single_blocks.") + String(bi) + String(".") + stails[wi]
            var n = b.w_nbytes[wi]
            if not first:
                header += String(",")
            first = False
            header += String("\"") + nm + String("\":{\"dtype\":\"BF16\",\"shape\":[")
            for di in range(len(b.w_shape[wi])):
                if di > 0:
                    header += String(",")
                header += String(b.w_shape[wi][di])
            header += String("],\"data_offsets\":[") + String(off) + String(",")
            header += String(off + n) + String("]}")
            off += n
    header += String("}")
    var hlen = header.byte_length()

    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("chroma_host_bf16_save: cannot open ") + path)
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
        for wi in range(CHROMA_DBL_FT_KEYS):
            var src = BytePtr(unsafe_from_address=Int(b.w_h[wi][].unsafe_ptr()))
            var wrote = sys_pwrite(fd, src, b.w_nbytes[wi], base + woff)
            if wrote != b.w_nbytes[wi]:
                _ = sys_close(fd)
                raise Error("chroma_host_bf16_save: short write (double)")
            woff += b.w_nbytes[wi]
    for bi in range(len(store.sgl)):
        ref b = store.sgl[bi]
        for wi in range(CHROMA_SGL_FT_KEYS):
            var src = BytePtr(unsafe_from_address=Int(b.w_h[wi][].unsafe_ptr()))
            var wrote = sys_pwrite(fd, src, b.w_nbytes[wi], base + woff)
            if wrote != b.w_nbytes[wi]:
                _ = sys_close(fd)
                raise Error("chroma_host_bf16_save: short write (single)")
            woff += b.w_nbytes[wi]
    _ = sys_close(fd)
    print(
        "[chroma-ft-save] wrote",
        len(store.dbl) * CHROMA_DBL_FT_KEYS + len(store.sgl) * CHROMA_SGL_FT_KEYS,
        "trained tensors (v2 full surface) ->", path,
    )


def chroma_host_bf16_overlay_resume(store: ChromaHostBf16, overlay_path: String) raises:
    """RESUME store rebuild, step 2: the store already holds the BASE checkpoint
    (build_chroma_host_bf16); copy the trained overlay's bytes over the pinned
    host bytes (byte-exact, BF16/size fail-loud). Key order == chroma_host_
    bf16_save's (doubles bi-major then singles bi-major, BFL names, the v2
    20/6 tails); a v1 (228-tensor, matmuls-only) overlay FAILS LOUD on the
    count check."""
    var dtails = _chroma_dbl_ft_key_tails()
    var stails = _chroma_sgl_ft_key_tails()
    var names = List[String]()
    var bufs = List[HArc]()
    var nbytes = List[Int]()
    for bi in range(len(store.dbl)):
        ref b = store.dbl[bi]
        for wi in range(CHROMA_DBL_FT_KEYS):
            names.append(String("double_blocks.") + String(bi) + String(".") + dtails[wi])
            bufs.append(b.w_h[wi].copy())
            nbytes.append(b.w_nbytes[wi])
    for bi in range(len(store.sgl)):
        ref b = store.sgl[bi]
        for wi in range(CHROMA_SGL_FT_KEYS):
            names.append(String("single_blocks.") + String(bi) + String(".") + stails[wi])
            bufs.append(b.w_h[wi].copy())
            nbytes.append(b.w_nbytes[wi])
    full_ft_overlay_into_host_store(overlay_path, names, bufs, nbytes)


def chroma_ft_state_shapes(
    store: ChromaHostBf16, mut rows: List[Int], mut cols: List[Int]
) raises:
    """FLAT per-state [rows, cols] in the af_states order (build_chroma_ft_
    adafactor_states) — the sidecar loader's expected-shape lists. 1-D params
    (bias/scale slots) report cols == 0: the UNFACTORED sentinel the sidecar
    v2 loader dispatches on."""
    for _bi in range(store.num_double):
        for wi in range(CHROMA_DBL_FT_KEYS):
            ref sh = store.dbl[0].w_shape[wi]
            rows.append(sh[0])
            if len(sh) == 2:
                cols.append(sh[1])
            else:
                cols.append(0)
    for _bi in range(store.num_single):
        for wi in range(CHROMA_SGL_FT_KEYS):
            ref sh = store.sgl[0].w_shape[wi]
            rows.append(sh[0])
            if len(sh) == 2:
                cols.append(sh[1])
            else:
                cols.append(0)


# ── BFL stack base + approximator loaders (the on-box ckpt is BFL-format) ────
def load_chroma_stack_base_bfl(
    st: SafeTensors, num_double: Int, num_single: Int, ctx: DeviceContext
) raises -> ChromaStackBase:
    """ChromaStackBase from BFL keys: img_in -> x_embedder, txt_in ->
    context_embedder, final_layer.linear -> proj_out."""
    return ChromaStackBase(
        ArcPointer(_load_tensor_bfl(st, String("img_in.weight"), ctx)),
        ArcPointer(_load_tensor_bfl(st, String("img_in.bias"), ctx)),
        ArcPointer(_load_tensor_bfl(st, String("txt_in.weight"), ctx)),
        ArcPointer(_load_tensor_bfl(st, String("txt_in.bias"), ctx)),
        ArcPointer(_load_tensor_bfl(st, String("final_layer.linear.weight"), ctx)),
        ArcPointer(_load_tensor_bfl(st, String("final_layer.linear.bias"), ctx)),
        num_double, num_single,
    )


def _dit_alias(
    st: SafeTensors, bfl_name: String, alias_name: String,
    mut weights: List[ArcPointer[Tensor]], mut name_to_idx: Dict[String, Int],
    ctx: DeviceContext,
) raises:
    var t = _load_tensor_bfl(st, bfl_name, ctx)
    name_to_idx[alias_name] = len(weights)
    weights.append(ArcPointer(t^))


def load_chroma_dit_cache_bfl(path: String, ctx: DeviceContext) raises -> ChromaDitCache:
    """ChromaDitCache (the frozen distilled-guidance approximator) from the BFL
    checkpoint: BFL layers.N.{in,out}_layer / norms.N.scale registered under
    the diffusers alias names ChromaDitCache.approximator_forward looks up
    (linear_1/linear_2/norms.N.weight — identical math, different names)."""
    var st = SafeTensors.open(path)
    var cfg = ChromaConfig.chroma1_hd()
    var weights = List[ArcPointer[Tensor]]()
    var name_to_idx = Dict[String, Int]()
    var p = String("distilled_guidance_layer.")
    _dit_alias(st, p + String("in_proj.weight"), p + String("in_proj.weight"),
               weights, name_to_idx, ctx)
    _dit_alias(st, p + String("in_proj.bias"), p + String("in_proj.bias"),
               weights, name_to_idx, ctx)
    for i in range(cfg.approximator_num_layers):
        var lb = p + String("layers.") + String(i) + String(".")
        _dit_alias(st, lb + String("in_layer.weight"), lb + String("linear_1.weight"),
                   weights, name_to_idx, ctx)
        _dit_alias(st, lb + String("in_layer.bias"), lb + String("linear_1.bias"),
                   weights, name_to_idx, ctx)
        _dit_alias(st, lb + String("out_layer.weight"), lb + String("linear_2.weight"),
                   weights, name_to_idx, ctx)
        _dit_alias(st, lb + String("out_layer.bias"), lb + String("linear_2.bias"),
                   weights, name_to_idx, ctx)
        var nb = p + String("norms.") + String(i)
        _dit_alias(st, nb + String(".scale"), nb + String(".weight"),
                   weights, name_to_idx, ctx)
    _dit_alias(st, p + String("out_proj.weight"), p + String("out_proj.weight"),
               weights, name_to_idx, ctx)
    _dit_alias(st, p + String("out_proj.bias"), p + String("out_proj.bias"),
               weights, name_to_idx, ctx)
    return ChromaDitCache(weights^, name_to_idx^, cfg)


# ── NO-adapter LoRA shells (the FT forward is the base block) ────────────────
def _ft_dbl_none_lora() -> DoubleBlockLoraDevice:
    var img = StreamLoraDevice(
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
    )
    var txt = StreamLoraDevice(
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
    )
    return DoubleBlockLoraDevice(img^, txt^)


def _ft_sgl_none_lora() -> SingleBlockLoraDevice:
    return SingleBlockLoraDevice(
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
        Optional[ChromaLoraAdapterDevice](None), Optional[ChromaLoraAdapterDevice](None),
        Optional[ChromaLoraAdapterDevice](None),
    )


# ── FT forward tape (device input snapshots for recompute + host mod flats) ──
struct ChromaStackFTForward(Movable):
    var out: List[Float32]                 # [N_IMG, out_ch]  (host, for the loss)
    var dbl_img_in: List[TArc]             # num_double x [N_IMG, D]  recompute inputs
    var dbl_txt_in: List[TArc]             # num_double x [N_TXT, D]
    var sgl_x_in: List[TArc]               # num_single x [S, D]
    var img_out: TArc                      # [N_IMG, D]  final-layer backward input
    var ln_img_out: TArc                   # [N_IMG, D]
    var dbl_img_mod: List[List[Float32]]   # num_double x [6D]  host mod flats
    var dbl_txt_mod: List[List[Float32]]
    var sgl_mod_flat: List[List[Float32]]  # num_single x [3D]
    var final_shift: List[Float32]         # [D]
    var final_scale: List[Float32]         # [D]

    def __init__(
        out self,
        var out: List[Float32],
        var dbl_img_in: List[TArc], var dbl_txt_in: List[TArc], var sgl_x_in: List[TArc],
        var img_out: TArc, var ln_img_out: TArc,
        var dbl_img_mod: List[List[Float32]], var dbl_txt_mod: List[List[Float32]],
        var sgl_mod_flat: List[List[Float32]],
        var final_shift: List[Float32], var final_scale: List[Float32],
    ):
        self.out = out^
        self.dbl_img_in = dbl_img_in^
        self.dbl_txt_in = dbl_txt_in^
        self.sgl_x_in = sgl_x_in^
        self.img_out = img_out^
        self.ln_img_out = ln_img_out^
        self.dbl_img_mod = dbl_img_mod^
        self.dbl_txt_mod = dbl_txt_mod^
        self.sgl_mod_flat = sgl_mod_flat^
        self.final_shift = final_shift^
        self.final_scale = final_scale^


# ── phase (b) forward: streamed base-block stack forward from the host store ─
def chroma_stack_ft_forward_streamed[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    pooled: List[Float32], mod_index: Int,
    base: ChromaStackBase,
    store: ChromaHostBf16,
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ChromaStackFTForward:
    """Mirror of chroma_stack_lora_forward_device_offload with the LIVE bf16
    host store as the weight source and NO adapters. Saves the per-block inputs
    (the backward's recompute checkpoints) + the host mod flats; per-block
    ctx.synchronize() (slot-reuse + async-free discipline)."""
    var num_double = store.num_double
    var num_single = store.num_single

    chroma_host_bf16_prefetch_dbl(store, 0)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    # input projections (frozen base linears, WITH biases — chroma has them).
    var bi_img = Optional[Tensor](base.x_embedder_b[].clone(ctx))
    var img = TArc(linear(
        _t_like(img_tokens.copy(), [N_IMG, in_ch], base.x_embedder_w[], ctx),
        base.x_embedder_w[], bi_img, ctx,
    ))
    var bi_txt = Optional[Tensor](base.context_embedder_b[].clone(ctx))
    var txt = TArc(linear(
        _t_like(txt_tokens.copy(), [N_TXT, txt_ch], base.context_embedder_w[], ctx),
        base.context_embedder_w[], bi_txt, ctx,
    ))

    var dbl_img_in = List[TArc]()
    var dbl_txt_in = List[TArc]()
    var dbl_img_mod = List[List[Float32]]()
    var dbl_txt_mod = List[List[Float32]]()
    for bi in range(num_double):
        dbl_img_in.append(img.copy())
        dbl_txt_in.append(txt.copy())
        var w = _load_chroma_dbl_slot(store, bi, ctx)
        if bi + 1 < num_double:
            chroma_host_bf16_prefetch_dbl(store, bi + 1)
        else:
            chroma_host_bf16_prefetch_sgl(store, 0)
        var im_flat = _dbl_img_mod_flat(pooled, bi, num_double, num_single, D)
        var tm_flat = _dbl_txt_mod_flat(pooled, bi, num_double, num_single, D)
        var im_dev = modvecs_to_device(_modvecs_from_flat6(im_flat, D), D, ctx)
        var tm_dev = modvecs_to_device(_modvecs_from_flat6(tm_flat, D), D, ctx)
        var bl = _ft_dbl_none_lora()
        var fwd = chroma_double_block_lora_forward_device[H, Dh, N_IMG, N_TXT, S](
            img, txt, w, im_dev, tm_dev, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        dbl_img_mod.append(im_flat^)
        dbl_txt_mod.append(tm_flat^)
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        ctx.synchronize()   # slot li%2 reuse safety + bound deferred frees

    # joint sequence: txt FIRST then img (Chroma/Flux convention).
    var x = TArc(concat(0, ctx, txt[], img[]))

    var sgl_x_in = List[TArc]()
    var sgl_mod_flat = List[List[Float32]]()
    for bi in range(num_single):
        sgl_x_in.append(x.copy())
        var w = _load_chroma_sgl_slot(store, bi, ctx)
        chroma_host_bf16_prefetch_sgl(store, bi + 1)
        var sm_flat = _sgl_mod_flat(pooled, bi, D)
        var sm_dev = single_modvecs_to_device(_single_modvecs_from_flat3(sm_flat, D), D, ctx)
        var sl = _ft_sgl_none_lora()
        var fwd = chroma_single_block_lora_forward_device[H, Dh, S](
            x, w, sm_dev, sl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        sgl_mod_flat.append(sm_flat^)
        x = fwd.out.copy()
        ctx.synchronize()

    # final layer: layer_norm(no affine) -> modulate(scale,shift) -> proj_out.
    var img_out = TArc(slice(x[], 0, N_TXT, N_IMG, ctx))
    var ss = _final_shift_scale(pooled, mod_index, D)
    var final_shift = ss[0].copy()
    var final_scale = ss[1].copy()

    var ones_bf = _t(_ones(D), [D], ctx)
    var zeros_bf = _t(_zeros(D), [D], ctx)
    var ln_img_out = TArc(layer_norm(img_out[], ones_bf, zeros_bf, eps, ctx))
    var normed = modulate(
        ln_img_out[], _t(final_scale.copy(), [D], ctx), _t(final_shift.copy(), [D], ctx), ctx,
    )
    var pb = Optional[Tensor](base.proj_out_b[].clone(ctx))
    var out = linear(normed, base.proj_out_w[], pb, ctx).to_host(ctx)

    return ChromaStackFTForward(
        out^, dbl_img_in^, dbl_txt_in^, sgl_x_in^,
        img_out^, ln_img_out^,
        dbl_img_mod^, dbl_txt_mod^, sgl_mod_flat^,
        final_shift^, final_scale^,
    )


# ── phase (b) backward: streamed FT stack backward + fused device Adafactor ──
struct ChromaStackFTWrite(Movable):
    var grad_count: Int
    var streaming_sync_count: Int

    def __init__(out self, grad_count: Int, streaming_sync_count: Int):
        self.grad_count = grad_count
        self.streaming_sync_count = streaming_sync_count


def chroma_stack_ft_backward_streamed[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],            # [N_IMG*out_ch] dL/d(velocity)
    base: ChromaStackBase,
    store: ChromaHostBf16,
    cos: List[Float32], sin: List[Float32],
    saved: ChromaStackFTForward,
    D: Int, Fmlp: Int, out_ch: Int, eps: Float32,
    mut af_states: List[AdafactorDeviceState],   # flat scheme (module header)
    t_step: Int,                     # 1-based optimizer step
    lr: Float64, beta2_decay: Float64, eps2: Float64, d_thresh: Float64,
    weight_decay: Float64,
    sr_seed: UInt64,                 # per-step seed; 0 = RNE writes
    ctx: DeviceContext,
) raises -> ChromaStackFTWrite:
    """DESCENDING walk (singles then doubles, deepest first): load slot →
    recompute the block forward from the saved input → P1 FT `_dev` backward
    (SURFACE_V2=True: matmul dW + bias d_b + q/k rms d_g) → per-param fused
    device Adafactor (2D factored / 1D unfactored dispatch) mutating the slot
    in place (SR bf16 write) → D2H write-back into the host store → carry d_x.
    The fused-back-pass shape: grads and updated weights never accumulate
    beyond one block."""
    var num_double = store.num_double
    var num_single = store.num_single
    if len(af_states) != num_double * CHROMA_DBL_FT_KEYS + num_single * CHROMA_SGL_FT_KEYS:
        raise Error("chroma_stack_ft_backward_streamed: af_states length mismatch")

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    # Prime the DESCENDING walk: deepest single block first.
    chroma_host_bf16_prefetch_sgl(store, num_single - 1)

    # frozen final-layer chain (identical to the LoRA device backward head):
    # proj_out (frozen, dx only) -> modulate (frozen approximator rows) -> ln.
    var ones_bf = _t(_ones(D), [D], ctx)
    var fs_bf = _t(saved.final_scale.copy(), [D], ctx)
    var d_out_bf = _t_like(d_out.copy(), [N_IMG, out_ch], base.proj_out_w[], ctx)
    var d_normed_t = linear_backward_dx(d_out_bf, base.proj_out_w[], N_IMG, D, out_ch, ctx)
    var mbf = modulate_backward(_dev_bf16(d_normed_t, ctx), saved.ln_img_out[], fs_bf, ctx, False)
    var d_img_out_bf = layer_norm_backward_dx(mbf.d_x, saved.img_out[], ones_bf, eps, ctx)
    var d_txt_zero = zeros_device([N_TXT, D], STDtype.F32, ctx)
    var d_x = TArc(concat(0, ctx, d_txt_zero, _dev_f32(d_img_out_bf, ctx)))   # F32 [S,D]

    var grad_count = 0
    var streaming_sync_count = 0

    # ── singles, deepest → shallowest ─────────────────────────────────────────
    var bi = num_single - 1
    var sgl_state_base = num_double * CHROMA_DBL_FT_KEYS
    while bi >= 0:
        var w = _load_chroma_sgl_slot(store, bi, ctx)
        if bi > 0:
            chroma_host_bf16_prefetch_sgl(store, bi - 1)
        else:
            chroma_host_bf16_prefetch_dbl(store, num_double - 1)
        var sm_dev = single_modvecs_to_device(
            _single_modvecs_from_flat3(saved.sgl_mod_flat[bi].copy(), D), D, ctx)
        var sl = _ft_sgl_none_lora()
        var rfwd = chroma_single_block_lora_forward_device[H, Dh, S](
            saved.sgl_x_in[bi], w, sm_dev, sl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        var bg = chroma_single_block_ft_backward_dev[H, Dh, S, True, True](
            d_x, w, sm_dev, rfwd.saved, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        # fused back pass: update THIS block's 6 slot params in place —
        # FACTORED (2D matmuls) via adafactor_step_device, UNFACTORED
        # (1D biases/scales) via adafactor_step_device_1d (torch trains <2D
        # params with the unfactored second moment).
        var slot = bi % 2
        for wi in range(CHROMA_SGL_FT_KEYS):
            var flat = sgl_state_base + bi * CHROMA_SGL_FT_KEYS + wi
            if af_states[flat].factored:
                adafactor_step_device(
                    store.slot_sgl[slot * CHROMA_SGL_FT_KEYS + wi][],
                    bg.dw[wi][],
                    af_states[flat],
                    t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                    weight_decay,
                    sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)),
                    ctx,
                )
            else:
                adafactor_step_device_1d(
                    store.slot_sgl[slot * CHROMA_SGL_FT_KEYS + wi][],
                    bg.dw[wi][],
                    af_states[flat],
                    t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                    weight_decay,
                    sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)),
                    ctx,
                )
        chroma_host_bf16_writeback_sgl(store, bi, ctx)
        grad_count += CHROMA_SGL_FT_KEYS
        d_x = bg.d_x.copy()
        ctx.synchronize()   # lands: grads freed, update done, write-back D2H'd
        streaming_sync_count += 1
        bi -= 1

    # split the joint carry back into the two double-block streams (txt FIRST).
    var d_to = TArc(slice(d_x[], 0, 0, N_TXT, ctx))
    var d_io = TArc(slice(d_x[], 0, N_TXT, N_IMG, ctx))

    # ── doubles, deepest → shallowest ─────────────────────────────────────────
    var di = num_double - 1
    while di >= 0:
        var w = _load_chroma_dbl_slot(store, di, ctx)
        chroma_host_bf16_prefetch_dbl(store, di - 1)
        var im_dev = modvecs_to_device(
            _modvecs_from_flat6(saved.dbl_img_mod[di].copy(), D), D, ctx)
        var tm_dev = modvecs_to_device(
            _modvecs_from_flat6(saved.dbl_txt_mod[di].copy(), D), D, ctx)
        var bl = _ft_dbl_none_lora()
        var rfwd = chroma_double_block_lora_forward_device[H, Dh, N_IMG, N_TXT, S](
            saved.dbl_img_in[di], saved.dbl_txt_in[di],
            w, im_dev, tm_dev, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        var bg = chroma_double_block_ft_backward_dev[H, Dh, N_IMG, N_TXT, S, True, True](
            d_io, d_to, w, im_dev, tm_dev, rfwd.saved,
            cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        var slot = di % 2
        for wi in range(CHROMA_DBL_FT_KEYS):
            var flat = di * CHROMA_DBL_FT_KEYS + wi
            if af_states[flat].factored:
                adafactor_step_device(
                    store.slot_dbl[slot * CHROMA_DBL_FT_KEYS + wi][],
                    bg.dw[wi][],
                    af_states[flat],
                    t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                    weight_decay,
                    sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)),
                    ctx,
                )
            else:
                adafactor_step_device_1d(
                    store.slot_dbl[slot * CHROMA_DBL_FT_KEYS + wi][],
                    bg.dw[wi][],
                    af_states[flat],
                    t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                    weight_decay,
                    sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)),
                    ctx,
                )
        chroma_host_bf16_writeback_dbl(store, di, ctx)
        grad_count += CHROMA_DBL_FT_KEYS
        d_io = bg.d_img_x.copy()
        d_to = bg.d_txt_x.copy()
        ctx.synchronize()
        streaming_sync_count += 1
        di -= 1

    return ChromaStackFTWrite(grad_count, streaming_sync_count)


# ── Adafactor state builder (flat scheme, device-resident ~few MB):
# FACTORED [rows]+[cols] for the 2D matmuls, UNFACTORED exp_avg_sq [n]
# (cols==0 sentinel) for the 1D biases + q/k rms scales. ─────────────────────
def build_chroma_ft_adafactor_states(
    store: ChromaHostBf16, ctx: DeviceContext
) raises -> List[AdafactorDeviceState]:
    var states = List[AdafactorDeviceState]()
    for _bi in range(store.num_double):
        for wi in range(CHROMA_DBL_FT_KEYS):
            ref sh = store.dbl[0].w_shape[wi]
            if len(sh) == 2:
                states.append(AdafactorDeviceState(sh[0], sh[1], ctx))
            else:
                states.append(AdafactorDeviceState(sh[0], ctx))   # unfactored 1D
    for _bi in range(store.num_single):
        for wi in range(CHROMA_SGL_FT_KEYS):
            ref sh = store.sgl[0].w_shape[wi]
            if len(sh) == 2:
                states.append(AdafactorDeviceState(sh[0], sh[1], ctx))
            else:
                states.append(AdafactorDeviceState(sh[0], ctx))   # unfactored 1D
    return states^
