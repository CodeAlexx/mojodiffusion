# serenitymojo/models/flux/flux_full_ft.mojo
#
# FLUX.1 FULL FINETUNE phases (a)+(b) (FULL_FINETUNE_ROLLOUT_PLAN_2026-07-07
# flux card) — a NEAR-COPY of the shipped chroma worked example
# (models/chroma/chroma_full_ft.mojo, 827916d) with the flux deltas. The block
# math is byte-for-byte shared (chroma's blocks ARE flux's; the P1 FT backwards
# flux_double/single_block_ft_backward_dev live FLUX-side in flux_block_ft.mojo
# and were torch-gated at real dims, cos >= 0.9997).
#
#   (a) FluxHostBf16: the pinned-host bf16 BOTH-WAYS store. All 19 double
#       blocks' 8 matmuls (img/txt × qkv/proj/mlp.0/mlp.2) and all 38 single
#       blocks' 2 matmuls (linear1/linear2) live as PINNED HOST BF16 bytes
#       (~17.2GB host RAM — the trained matmul surface of the ~11.9B model;
#       the store IS the live model). Two device slot-sets PER KIND, double-
#       buffered on a dedicated copy stream with per-slot H2D-done events.
#       FROZEN-but-FORWARD-NEEDED tensors stay DEVICE-RESIDENT, loaded once:
#       every linear BIAS + the q/k rms-norm scales (per block), plus the
#       FluxStackBase (img_in/txt_in, time_in/guidance_in/vector_in embed
#       MLPs, the PER-BLOCK img_mod/txt_mod/modulation linears, final layer)
#       via the existing BFL loader models/flux/weights.load_flux_stack_base.
#       flux_host_bf16_save writes the pinned bytes STRAIGHT to a safetensors
#       overlay (ORIGINAL BFL tensor names) with no GPU round-trip.
#
#   (b) flux_stack_ft_forward_streamed / flux_stack_ft_backward_streamed:
#       the streamed FT stack step. FLUX DELTA vs chroma: modulation comes
#       from vec = time_in(t) + [guidance_in(g)] + vector_in(clip_pool)
#       (flux_stack.mojo _embed_vec_forward — the parity-gated composition),
#       then PER-BLOCK silu(vec) -> img_mod.lin/txt_mod.lin -> [6D] flats /
#       modulation.lin -> [3D] flats / final adaLN -> (shift,scale) — instead
#       of chroma's frozen approximator pooled_temb table rows. v1 freezes
#       the embedders + ALL mod.lin linears + final layer (reference trainer trains them —
#       documented delta, same class as chroma/krea2 v1); the block backward
#       therefore drops the mod-grad chain entirely (the frozen-mod FT
#       backwards already skip d_shift/d_scale/d_gate).
#
# CHECKPOINT NOTE (documented per the staging decision): the on-box
# /home/alex/.serenity/models/checkpoints/flux1-dev.safetensors is a SYMLINK to
# flux1-schnell.safetensors — SCHNELL weights under the dev name (23.8GB,
# all-BF16, BFL keys, 776 tensors, verified 2026-07-08). Schnell has NO
# guidance_in embedder (guidance-distilled the other way): the header carries
# no guidance_in.* keys, so the store detects that and loads the stack base
# with has_guidance=False; the guidance input is structurally IGNORED
# (flux_stack.mojo:469 `if base.has_guidance and guidance:`). A real flux1-dev
# checkpoint under the same name would auto-detect has_guidance=True and feed
# guidance*1000 through guidance_in — no code change.
#
# Adafactor state flat-index scheme (af_states) — matches chroma/krea2 and the
# FluxDoubleBlockFTGrads.dw / FluxSingleBlockFTGrads.dw order EXACTLY:
#   doubles first: bi*8 + wi, wi in {img wqkv, img wproj, img wmlp0, img wmlp2,
#                                    txt wqkv, txt wproj, txt wmlp0, txt wmlp2}
#   singles next : num_double*8 + bi*2 + wi, wi in {w1, w2}
#
# TRAPS honored (reference-mojo-full-finetune): dW arrives F32 from the P1 backwards
# (linear_backward_dw(..., STDtype.F32)); cfg.lr not learning_rate; Tensor is
# move-only (TArc carriers); "weights changed" checks diff WHOLE tensors.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import (
    DeviceContext, HostBuffer, DeviceBuffer, DeviceStream, DeviceEvent,
)
from std.collections import List, Optional
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
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import layer_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.tensor_algebra import concat, slice, zeros_device
from serenitymojo.ops.linalg_backward import (
    linear_backward_dx, linear_backward_dw, linear_backward_db,
)
from serenitymojo.ops.norm_backward import layer_norm_backward_dx
from serenitymojo.ops.elementwise_backward import modulate_backward

# flux block weight carriers (the BFL ckpt stores the fused layout verbatim).
from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)
# stack base (frozen residents) + the parity-gated embed/mod composition
# helpers (vec fwd, silu(vec) -> mod.lin flats, modvec chunking, host lifts).
from serenitymojo.models.flux.flux_stack import (
    FluxStackBase, _VecFwd, _embed_vec_forward, _mod_proj,
    _modvecs_from_flat, _single_modvecs_from_flat, _chunk,
    _ones, _zeros, _t,
)

# parity-verified device-resident block forwards (save the FT backward's tape)
# + the mod-vec device carriers (chroma's blocks ARE flux's — the same reuse
# direction flux_block_ft.mojo already established).
from serenitymojo.models.chroma.chroma_block_device import (
    ModVecsDevice, SingleModVecsDevice,
    ChromaLoraAdapterDevice, StreamLoraDevice,
    DoubleBlockLoraDevice, SingleBlockLoraDevice,
    modvecs_to_device, single_modvecs_to_device,
    chroma_double_block_lora_forward_device,
    chroma_single_block_lora_forward_device,
)
# P1 FT block backwards (flux-side impl, torch-gated) + the v2 full-surface
# slot constants (shared flux/chroma FIXED order).
from serenitymojo.models.flux.flux_block_ft import (
    flux_double_block_ft_backward_dev, flux_single_block_ft_backward_dev,
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
from serenitymojo.training.adafactor_device import (
    AdafactorDeviceState, adafactor_step_device, adafactor_step_device_1d,
)
# FULL-FT resume (the fleet sidecar): overlay the trained weights over the
# base-built pinned host store (byte-exact, dtype-checked).
from serenitymojo.training.full_ft_sidecar import full_ft_overlay_into_host_store

comptime TArc = ArcPointer[Tensor]
comptime HArc = ArcPointer[HostBuffer[DType.uint8]]

# trained params per block kind (v2 FULL surface; FT_*_SLOT order — leading MM
# slots are the fp-ckpt 2D matmuls, the rest are 1D biases + q/k rms scales).
comptime FLUX_DBL_FT_MM_KEYS = 8              # img/txt qkv/proj/mlp.0/mlp.2 weights
comptime FLUX_DBL_FT_KEYS = FT_DBL_SLOTS_V2   # 20: + 8 biases + 4 q/k rms scales
comptime FLUX_SGL_FT_MM_KEYS = 2              # linear1/linear2 weights
comptime FLUX_SGL_FT_KEYS = FT_SGL_SLOTS_V2   # 6: + b1/b2 + q/k rms scales

# per-block mod.lin params (FLUX delta; device-resident in FluxStackBase,
# trained IN PLACE — NOT in the store slots). af_states for these are appended
# AFTER the block states. Per double: img_mod.{w,b} + txt_mod.{w,b} = 4; per
# single: modulation.{w,b} = 2. Flat sub-index order below.
comptime FLUX_DBL_MOD_KEYS = 4   # img_mod.w, img_mod.b, txt_mod.w, txt_mod.b
comptime FLUX_SGL_MOD_KEYS = 2   # modulation.w, modulation.b


# ── BFL-key device loader (storage dtype preserved) ──────────────────────────
def _load_tensor_bfl(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


# schnell detection: guidance_in.* absent from the header => has_guidance=False.
def flux_ckpt_has_guidance(st: SafeTensors) -> Bool:
    var nm = st.names()
    for i in range(len(nm)):
        if nm[i] == String("guidance_in.in_layer.weight"):
            return True
    return False


# ── host lift helpers (local; mirror chroma_stack_lora's) ────────────────────
def _t_like(
    vals: List[Float32], var shape: List[Int], ref_weight: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var t = _t(vals.copy(), shape^, ctx)
    if t.dtype() == ref_weight.dtype():
        return t^
    return cast_tensor(t, ref_weight.dtype(), ctx)


def _dev_f32(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    if t.dtype() == STDtype.F32:
        return t.clone(ctx)
    return cast_tensor(t, STDtype.F32, ctx)


def _dev_bf16(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    if t.dtype() == STDtype.BF16:
        return t.clone(ctx)
    return cast_tensor(t, STDtype.BF16, ctx)


# ── per-block pinned-host records (LIVE weights) + frozen small tensors ───────
struct FluxDblBlockHostBf16(Copyable, Movable):
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


struct FluxSglBlockHostBf16(Copyable, Movable):
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


struct FluxHostBf16(Copyable, Movable):
    var dbl: List[FluxDblBlockHostBf16]    # len == num_double
    var sgl: List[FluxSglBlockHostBf16]    # len == num_single
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
        var dbl: List[FluxDblBlockHostBf16], var sgl: List[FluxSglBlockHostBf16],
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


# save-key tails, in the SAME order as the slots / the FT dw lists (BFL names —
# identical to chroma's: the two models share the BFL block key layout).
def _flux_dbl_ft_key_tails() -> List[String]:
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


def _flux_sgl_ft_key_tails() -> List[String]:
    var t = List[String]()
    t.append(String("linear1.weight"))
    t.append(String("linear2.weight"))
    t.append(String("linear1.bias"))                     # FT_SGL_SLOT_B1
    t.append(String("linear2.bias"))                     # FT_SGL_SLOT_B2
    t.append(String("norm.query_norm.scale"))            # FT_SGL_SLOT_QNORM
    t.append(String("norm.key_norm.scale"))              # FT_SGL_SLOT_KNORM
    return t^


# mod.lin BFL key tails (FLUX delta; device-resident, saved after the store).
def _flux_dbl_mod_key_tails() -> List[String]:
    var t = List[String]()
    t.append(String("img_mod.lin.weight"))
    t.append(String("img_mod.lin.bias"))
    t.append(String("txt_mod.lin.weight"))
    t.append(String("txt_mod.lin.bias"))
    return t^


def _flux_sgl_mod_key_tails() -> List[String]:
    var t = List[String]()
    t.append(String("modulation.lin.weight"))
    t.append(String("modulation.lin.bias"))
    return t^


# mod.lin device-tensor accessors by sub-index (FLUX_DBL_MOD_KEYS / _SGL order).
def _mod_tensor_dbl(base: FluxStackBase, bi: Int, mi: Int) raises -> TArc:
    if mi == 0:
        return base.dbl_mod[bi].img.w.copy()
    elif mi == 1:
        return base.dbl_mod[bi].img.b.copy()
    elif mi == 2:
        return base.dbl_mod[bi].txt.w.copy()
    else:
        return base.dbl_mod[bi].txt.b.copy()


def _mod_tensor_sgl(base: FluxStackBase, bi: Int, mi: Int) raises -> TArc:
    if mi == 0:
        return base.sgl_mod[bi].w.copy()
    else:
        return base.sgl_mod[bi].b.copy()


# D2H one checkpoint matrix into a fresh PINNED host buffer; fail loud on a
# non-BF16 store (the reference trainer full-FT contract is bf16 weights, NO quantized linears).
def _pin_bf16_weight(
    st: SafeTensors, name: String, ctx: DeviceContext,
    mut w_h: List[HArc], mut w_nbytes: List[Int], mut w_shape: List[List[Int]],
) raises:
    var t = _load_tensor_bfl(st, name, ctx)
    if t.dtype() != STDtype.BF16:
        raise Error(
            String("flux full-FT store: ") + name
            + String(" is not BF16 on disk (dtype.tag ") + String(t.dtype().tag)
            + String(") — quantized/other-dtype full-FT is forbidden")
        )
    var sh = t.shape()
    if len(sh) != 2:
        raise Error(String("flux full-FT store: ") + name + String(" is not 2-D"))
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
            String("flux full-FT store: ") + name
            + String(" is not BF16 on disk (dtype.tag ") + String(t.dtype().tag)
            + String(") — quantized/other-dtype full-FT is forbidden")
        )
    var sh = t.shape()
    if len(sh) != 1:
        raise Error(String("flux full-FT store: ") + name + String(" is not 1-D"))
    var bh = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
    ctx.enqueue_copy(bh, t.buf)   # D2H
    ctx.synchronize()
    w_h.append(HArc(bh^))
    w_nbytes.append(t.nbytes())
    w_shape.append(sh.copy())


def build_flux_host_bf16(
    st: SafeTensors, num_double: Int, num_single: Int, ctx: DeviceContext
) raises -> FluxHostBf16:
    """Load ALL blocks' trained params as PINNED-HOST BF16 (the full-FT live
    model): leading MM slots the 2D matmuls, the rest the 1D biases + q/k rms
    scales (the v2 FULL surface — nothing per-block stays frozen in the store;
    the mod.lins stay device-resident in FluxStackBase). BF16 = ckpt bytes."""
    var dbl = List[FluxDblBlockHostBf16]()
    var dtails = _flux_dbl_ft_key_tails()
    for bi in range(num_double):
        var p = String("double_blocks.") + String(bi) + String(".")
        var w_h = List[HArc]()
        var w_nbytes = List[Int]()
        var w_shape = List[List[Int]]()
        for ki in range(FLUX_DBL_FT_MM_KEYS):
            _pin_bf16_weight(st, p + dtails[ki], ctx, w_h, w_nbytes, w_shape)
        for ki in range(FLUX_DBL_FT_MM_KEYS, FLUX_DBL_FT_KEYS):
            _pin_bf16_1d_weight(st, p + dtails[ki], ctx, w_h, w_nbytes, w_shape)
        dbl.append(FluxDblBlockHostBf16(w_h^, w_nbytes^, w_shape^))
        cu_mempool_trim_current(0)
        if (bi + 1) % 4 == 0 or bi + 1 == num_double:
            print("full-ft bf16 host store: pinned double block", bi + 1, "/", num_double)
    var sgl = List[FluxSglBlockHostBf16]()
    var stails = _flux_sgl_ft_key_tails()
    for bi in range(num_single):
        var p = String("single_blocks.") + String(bi) + String(".")
        var w_h = List[HArc]()
        var w_nbytes = List[Int]()
        var w_shape = List[List[Int]]()
        for ki in range(FLUX_SGL_FT_MM_KEYS):
            _pin_bf16_weight(st, p + stails[ki], ctx, w_h, w_nbytes, w_shape)
        for ki in range(FLUX_SGL_FT_MM_KEYS, FLUX_SGL_FT_KEYS):
            _pin_bf16_1d_weight(st, p + stails[ki], ctx, w_h, w_nbytes, w_shape)
        sgl.append(FluxSglBlockHostBf16(w_h^, w_nbytes^, w_shape^))
        cu_mempool_trim_current(0)
        if (bi + 1) % 8 == 0 or bi + 1 == num_single:
            print("full-ft bf16 host store: pinned single block", bi + 1, "/", num_single)

    # device slots (shapes uniform within each kind — asserted off block 0)
    var slot_dbl = List[TArc]()
    ref d0 = dbl[0]
    for slot in range(2):
        _ = slot
        for i in range(FLUX_DBL_FT_KEYS):
            var dbuf = ctx.enqueue_create_buffer[DType.uint8](d0.w_nbytes[i])
            var t = Tensor(dbuf^, d0.w_shape[i].copy(), STDtype.BF16)
            slot_dbl.append(TArc(t^))
    var slot_sgl = List[TArc]()
    ref s0 = sgl[0]
    for slot in range(2):
        _ = slot
        for i in range(FLUX_SGL_FT_KEYS):
            var dbuf = ctx.enqueue_create_buffer[DType.uint8](s0.w_nbytes[i])
            var t = Tensor(dbuf^, s0.w_shape[i].copy(), STDtype.BF16)
            slot_sgl.append(TArc(t^))
    ctx.synchronize()
    # kind-uniformity fail-loud (slot reuse depends on it)
    for bi in range(num_double):
        for i in range(FLUX_DBL_FT_KEYS):
            if dbl[bi].w_nbytes[i] != d0.w_nbytes[i]:
                raise Error("flux full-FT store: double-block weight shapes not uniform")
    for bi in range(num_single):
        for i in range(FLUX_SGL_FT_KEYS):
            if sgl[bi].w_nbytes[i] != s0.w_nbytes[i]:
                raise Error("flux full-FT store: single-block weight shapes not uniform")

    var copy_stream = List[ArcPointer[DeviceStream]]()
    copy_stream.append(ArcPointer(ctx.create_stream()))
    var ev_dbl = List[ArcPointer[DeviceEvent]]()
    ev_dbl.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    ev_dbl.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    var ev_sgl = List[ArcPointer[DeviceEvent]]()
    ev_sgl.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    ev_sgl.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    return FluxHostBf16(
        dbl^, sgl^, slot_dbl^, slot_sgl^, copy_stream^, ev_dbl^, ev_sgl^,
        num_double, num_single,
    )


# ── prefetch (async H2D on the copy stream into slot bi%2, record event) ─────
def flux_host_bf16_prefetch_dbl(store: FluxHostBf16, bi: Int) raises:
    if bi < 0 or bi >= len(store.dbl):
        return
    var slot = bi % 2
    ref b = store.dbl[bi]
    for i in range(FLUX_DBL_FT_KEYS):
        _h2d_dma_copy(
            UInt64(Int(store.slot_dbl[slot * FLUX_DBL_FT_KEYS + i][].buf.unsafe_ptr())),
            b.w_h[i][].unsafe_ptr(),
            b.w_nbytes[i],
            store.copy_stream[0][],
        )
    store.copy_stream[0][].record_event(store.ev_dbl[slot][])


def flux_host_bf16_prefetch_sgl(store: FluxHostBf16, bi: Int) raises:
    if bi < 0 or bi >= len(store.sgl):
        return
    var slot = bi % 2
    ref b = store.sgl[bi]
    for i in range(FLUX_SGL_FT_KEYS):
        _h2d_dma_copy(
            UInt64(Int(store.slot_sgl[slot * FLUX_SGL_FT_KEYS + i][].buf.unsafe_ptr())),
            b.w_h[i][].unsafe_ptr(),
            b.w_nbytes[i],
            store.copy_stream[0][],
        )
    store.copy_stream[0][].record_event(store.ev_sgl[slot][])


# ── slot loaders (fence the H2D, wrap the PERSISTENT slot tensors) ───────────
def _load_flux_dbl_slot(
    store: FluxHostBf16, bi: Int, ctx: DeviceContext
) raises -> DoubleBlockWeights:
    """Fence slot bi%2's H2D and build DoubleBlockWeights over the SAME slot
    tensors (refcount) — the FT optimizer mutates these in place pre-writeback.
    Frozen biases/norms come from the block's device-resident record."""
    var slot = bi % 2
    ctx.stream().enqueue_wait_for(store.ev_dbl[slot][])
    var base = slot * FLUX_DBL_FT_KEYS
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


def _load_flux_sgl_slot(
    store: FluxHostBf16, bi: Int, ctx: DeviceContext
) raises -> SingleBlockWeights:
    """Fence slot bi%2's H2D and build SingleBlockWeights over the SAME slot
    tensors (refcount); frozen b1/b2/norms device-resident."""
    var slot = bi % 2
    ctx.stream().enqueue_wait_for(store.ev_sgl[slot][])
    var base = slot * FLUX_SGL_FT_KEYS
    return SingleBlockWeights(
        store.slot_sgl[base + 0].copy(),
        store.slot_sgl[base + FT_SGL_SLOT_B1].copy(),
        store.slot_sgl[base + 1].copy(),
        store.slot_sgl[base + FT_SGL_SLOT_B2].copy(),
        store.slot_sgl[base + FT_SGL_SLOT_QNORM].copy(),
        store.slot_sgl[base + FT_SGL_SLOT_KNORM].copy(),
    )


# ── write-back (D2H the optimizer-updated slot into the pinned host bytes) ───
def flux_host_bf16_writeback_dbl(store: FluxHostBf16, bi: Int, ctx: DeviceContext) raises:
    var slot = bi % 2
    ref b = store.dbl[bi]
    for i in range(FLUX_DBL_FT_KEYS):
        ctx.enqueue_copy(b.w_h[i][], store.slot_dbl[slot * FLUX_DBL_FT_KEYS + i][].buf)


def flux_host_bf16_writeback_sgl(store: FluxHostBf16, bi: Int, ctx: DeviceContext) raises:
    var slot = bi % 2
    ref b = store.sgl[bi]
    for i in range(FLUX_SGL_FT_KEYS):
        ctx.enqueue_copy(b.w_h[i][], store.slot_sgl[slot * FLUX_SGL_FT_KEYS + i][].buf)


def flux_host_bf16_overlay_resume(
    store: FluxHostBf16, base: FluxStackBase, overlay_path: String, ctx: DeviceContext
) raises:
    """RESUME store rebuild, step 2: the store already holds the BASE checkpoint
    (build_flux_host_bf16); copy the trained overlay's bytes over the pinned
    host store bytes (byte-exact, BF16/size fail-loud) AND over the device-
    resident mod.lin tensors (H2D). Key set == flux_host_bf16_save's (v2 20/6
    store tails + 4/2 mod.lin tails); a v1 (228-tensor) overlay FAILS LOUD via
    the store count check."""
    var dtails = _flux_dbl_ft_key_tails()
    var stails = _flux_sgl_ft_key_tails()
    var names = List[String]()
    var bufs = List[HArc]()
    var nbytes = List[Int]()
    for bi in range(len(store.dbl)):
        ref b = store.dbl[bi]
        for wi in range(FLUX_DBL_FT_KEYS):
            names.append(String("double_blocks.") + String(bi) + String(".") + dtails[wi])
            bufs.append(b.w_h[wi].copy())
            nbytes.append(b.w_nbytes[wi])
    for bi in range(len(store.sgl)):
        ref b = store.sgl[bi]
        for wi in range(FLUX_SGL_FT_KEYS):
            names.append(String("single_blocks.") + String(bi) + String(".") + stails[wi])
            bufs.append(b.w_h[wi].copy())
            nbytes.append(b.w_nbytes[wi])
    # The v2 overlay ALSO holds the 152 device-resident mod.lin tensors (loaded
    # below), so the file's total (608 store + mod.lin) exceeds the store names.
    # Declare the full expected count so the store overlay still validates its
    # 608 names while a v1 (228-tensor) overlay fails loud (228 != full total).
    var mod_total = (
        len(base.dbl_mod) * FLUX_DBL_MOD_KEYS + len(base.sgl_mod) * FLUX_SGL_MOD_KEYS
    )
    full_ft_overlay_into_host_store(
        overlay_path, names, bufs, nbytes, expected_total=len(names) + mod_total
    )

    # mod.lin overlay (device): load each trained tensor by BFL name from the
    # overlay and H2D into the resident FluxStackBase tensor (byte-exact bf16).
    var mst = SafeTensors.open(overlay_path)
    var dmtails = _flux_dbl_mod_key_tails()
    var smtails = _flux_sgl_mod_key_tails()
    for bi in range(len(base.dbl_mod)):
        var p = String("double_blocks.") + String(bi) + String(".")
        for mi in range(FLUX_DBL_MOD_KEYS):
            var dst = _mod_tensor_dbl(base, bi, mi)
            var t = _load_tensor_bfl(mst, p + dmtails[mi], ctx)
            if t.dtype() != STDtype.BF16 or t.nbytes() != dst[].nbytes():
                raise Error(
                    String("flux full-FT resume: mod.lin ") + p + dmtails[mi]
                    + String(" dtype/size mismatch")
                )
            ctx.enqueue_copy(dst[].buf, t.buf)   # H2D over the resident tensor
    for bi in range(len(base.sgl_mod)):
        var p = String("single_blocks.") + String(bi) + String(".")
        for mi in range(FLUX_SGL_MOD_KEYS):
            var dst = _mod_tensor_sgl(base, bi, mi)
            var t = _load_tensor_bfl(mst, p + smtails[mi], ctx)
            if t.dtype() != STDtype.BF16 or t.nbytes() != dst[].nbytes():
                raise Error(
                    String("flux full-FT resume: mod.lin ") + p + smtails[mi]
                    + String(" dtype/size mismatch")
                )
            ctx.enqueue_copy(dst[].buf, t.buf)
    ctx.synchronize()


def flux_ft_state_shapes(
    store: FluxHostBf16, base: FluxStackBase, mut rows: List[Int], mut cols: List[Int]
) raises:
    """FLAT per-state [rows, cols] in the af_states order (build_flux_ft_
    adafactor_states) — the sidecar loader's expected-shape lists. 1-D params
    (bias/scale slots + mod.lin biases) report cols == 0: the UNFACTORED
    sentinel the sidecar v2 loader dispatches on. mod.lin states are APPENDED
    after the block states (doubles then singles)."""
    for _bi in range(store.num_double):
        for wi in range(FLUX_DBL_FT_KEYS):
            ref sh = store.dbl[0].w_shape[wi]
            rows.append(sh[0])
            cols.append(sh[1] if len(sh) == 2 else 0)
    for _bi in range(store.num_single):
        for wi in range(FLUX_SGL_FT_KEYS):
            ref sh = store.sgl[0].w_shape[wi]
            rows.append(sh[0])
            cols.append(sh[1] if len(sh) == 2 else 0)
    # mod.lin (FLUX delta): img_mod.w/b, txt_mod.w/b per double; mod.w/b per single.
    for bi in range(store.num_double):
        ref iw = base.dbl_mod[bi].img.w[].shape()
        rows.append(iw[0]); cols.append(iw[1])
        ref ib = base.dbl_mod[bi].img.b[].shape()
        rows.append(ib[0]); cols.append(0)
        ref tw = base.dbl_mod[bi].txt.w[].shape()
        rows.append(tw[0]); cols.append(tw[1])
        ref tb = base.dbl_mod[bi].txt.b[].shape()
        rows.append(tb[0]); cols.append(0)
    for bi in range(store.num_single):
        ref sw = base.sgl_mod[bi].w[].shape()
        rows.append(sw[0]); cols.append(sw[1])
        ref sb = base.sgl_mod[bi].b[].shape()
        rows.append(sb[0]); cols.append(0)


# ── host-direct safetensors overlay save (no GPU round-trip for the store;
# mod.lin is device-resident -> D2H'd one-at-a-time and streamed) ─────────────
def flux_host_bf16_save(
    store: FluxHostBf16, base: FluxStackBase, path: String, ctx: DeviceContext
) raises:
    """Write the trained surface (the pinned-host bf16 store bytes + the
    device-resident mod.lin params) DIRECTLY to a safetensors file. Keys keep
    the ORIGINAL BFL checkpoint names (an overlay). v2 FULL surface: 20 x
    num_double + 6 x num_single store tensors + 4 x num_double + 2 x num_single
    mod.lin tensors."""
    var dtails = _flux_dbl_ft_key_tails()
    var stails = _flux_sgl_ft_key_tails()
    var dmtails = _flux_dbl_mod_key_tails()
    var smtails = _flux_sgl_mod_key_tails()
    # header JSON + offsets (store doubles, store singles, mod doubles, mod
    # singles — deterministic; the resume overlay reads by name so order is free
    # but the count/keys must match).
    var header = String("{")
    var off = 0
    var first = True
    for bi in range(len(store.dbl)):
        ref b = store.dbl[bi]
        for wi in range(FLUX_DBL_FT_KEYS):
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
        for wi in range(FLUX_SGL_FT_KEYS):
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
    # mod.lin header entries (shapes/nbytes read from the device tensors; no D2H).
    for bi in range(len(base.dbl_mod)):
        for mi in range(FLUX_DBL_MOD_KEYS):
            var mt = _mod_tensor_dbl(base, bi, mi)
            var nm = String("double_blocks.") + String(bi) + String(".") + dmtails[mi]
            var n = mt[].nbytes()
            if not first:
                header += String(",")
            first = False
            header += String("\"") + nm + String("\":{\"dtype\":\"BF16\",\"shape\":[")
            ref sh = mt[].shape()
            for di in range(len(sh)):
                if di > 0:
                    header += String(",")
                header += String(sh[di])
            header += String("],\"data_offsets\":[") + String(off) + String(",")
            header += String(off + n) + String("]}")
            off += n
    for bi in range(len(base.sgl_mod)):
        for mi in range(FLUX_SGL_MOD_KEYS):
            var mt = _mod_tensor_sgl(base, bi, mi)
            var nm = String("single_blocks.") + String(bi) + String(".") + smtails[mi]
            var n = mt[].nbytes()
            if not first:
                header += String(",")
            first = False
            header += String("\"") + nm + String("\":{\"dtype\":\"BF16\",\"shape\":[")
            ref sh = mt[].shape()
            for di in range(len(sh)):
                if di > 0:
                    header += String(",")
                header += String(sh[di])
            header += String("],\"data_offsets\":[") + String(off) + String(",")
            header += String(off + n) + String("]}")
            off += n
    header += String("}")
    var hlen = header.byte_length()

    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd < 0:
        raise Error(String("flux_host_bf16_save: cannot open ") + path)
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
    for bi in range(len(store.dbl)):
        ref b = store.dbl[bi]
        for wi in range(FLUX_DBL_FT_KEYS):
            var src = BytePtr(unsafe_from_address=Int(b.w_h[wi][].unsafe_ptr()))
            var wrote = sys_pwrite(fd, src, b.w_nbytes[wi], body_base + woff)
            if wrote != b.w_nbytes[wi]:
                _ = sys_close(fd)
                raise Error("flux_host_bf16_save: short write (double)")
            woff += b.w_nbytes[wi]
    for bi in range(len(store.sgl)):
        ref b = store.sgl[bi]
        for wi in range(FLUX_SGL_FT_KEYS):
            var src = BytePtr(unsafe_from_address=Int(b.w_h[wi][].unsafe_ptr()))
            var wrote = sys_pwrite(fd, src, b.w_nbytes[wi], body_base + woff)
            if wrote != b.w_nbytes[wi]:
                _ = sys_close(fd)
                raise Error("flux_host_bf16_save: short write (single)")
            woff += b.w_nbytes[wi]
    # mod.lin data: D2H one-at-a-time (bound host RAM), pwrite, free.
    for bi in range(len(base.dbl_mod)):
        for mi in range(FLUX_DBL_MOD_KEYS):
            var mt = _mod_tensor_dbl(base, bi, mi)
            var n = mt[].nbytes()
            var hb = ctx.enqueue_create_host_buffer[DType.uint8](n)
            ctx.enqueue_copy(hb, mt[].buf)
            ctx.synchronize()
            var src = BytePtr(unsafe_from_address=Int(hb.unsafe_ptr()))
            var wrote = sys_pwrite(fd, src, n, body_base + woff)
            if wrote != n:
                _ = sys_close(fd)
                raise Error("flux_host_bf16_save: short write (dbl mod.lin)")
            woff += n
    for bi in range(len(base.sgl_mod)):
        for mi in range(FLUX_SGL_MOD_KEYS):
            var mt = _mod_tensor_sgl(base, bi, mi)
            var n = mt[].nbytes()
            var hb = ctx.enqueue_create_host_buffer[DType.uint8](n)
            ctx.enqueue_copy(hb, mt[].buf)
            ctx.synchronize()
            var src = BytePtr(unsafe_from_address=Int(hb.unsafe_ptr()))
            var wrote = sys_pwrite(fd, src, n, body_base + woff)
            if wrote != n:
                _ = sys_close(fd)
                raise Error("flux_host_bf16_save: short write (sgl mod.lin)")
            woff += n
    _ = sys_close(fd)
    print(
        "[flux-ft-save] wrote",
        len(store.dbl) * FLUX_DBL_FT_KEYS + len(store.sgl) * FLUX_SGL_FT_KEYS
        + len(base.dbl_mod) * FLUX_DBL_MOD_KEYS + len(base.sgl_mod) * FLUX_SGL_MOD_KEYS,
        "trained tensors (v2 full surface + mod.lin) ->", path,
    )


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
struct FluxStackFTForward(Movable):
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
    # THE MOD.LIN BACKWARD INPUT: silu(vec) [1,D]. Every per-block mod.lin
    # (img/txt/single) is a Linear over this SAME vector; the backward routes
    # each block's [6D]/[3D] flat grad through mod.lin as d_W = flatᵀ@silu(vec),
    # d_b = colsum(flat). Saving it here is what makes the mod.lin arms trainable.
    var vec_silu: List[Float32]            # [1, D]

    def __init__(
        out self,
        var out: List[Float32],
        var dbl_img_in: List[TArc], var dbl_txt_in: List[TArc], var sgl_x_in: List[TArc],
        var img_out: TArc, var ln_img_out: TArc,
        var dbl_img_mod: List[List[Float32]], var dbl_txt_mod: List[List[Float32]],
        var sgl_mod_flat: List[List[Float32]],
        var final_shift: List[Float32], var final_scale: List[Float32],
        var vec_silu: List[Float32],
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
        self.vec_silu = vec_silu^


# ── phase (b) forward: streamed base-block stack forward from the host store ─
def flux_stack_ft_forward_streamed[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_tokens: List[Float32], txt_tokens: List[Float32],
    timestep: List[Float32],               # pre-scaled t*1000 (BFL time_factor)
    guidance: Optional[List[Float32]],     # pre-scaled g*1000; IGNORED when
                                           # base.has_guidance=False (schnell)
    vector: List[Float32],                 # CLIP-L pooled [VEC_DIM]
    base: FluxStackBase,
    store: FluxHostBf16,
    cos: List[Float32], sin: List[Float32],
    D: Int, Fmlp: Int, in_ch: Int, txt_ch: Int, out_ch: Int,
    T_DIM: Int, VEC_DIM: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> FluxStackFTForward:
    """Mirror of chroma_stack_ft_forward_streamed with the FLUX conditioning
    deltas: vec = time_in + [guidance_in] + vector_in (frozen embed MLPs),
    per-block mod flats from silu(vec) -> frozen mod.lin (flux_stack._mod_proj),
    final (shift,scale) from the frozen adaLN linear. Saves the per-block
    inputs (the backward's recompute checkpoints) + the host mod flats;
    per-block ctx.synchronize() (slot-reuse + async-free discipline)."""
    var num_double = store.num_double
    var num_single = store.num_single

    flux_host_bf16_prefetch_dbl(store, 0)

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    # ── embeds -> vec -> silu(vec) (frozen; flux_stack's gated composition) ──
    var vf = _embed_vec_forward(timestep, guidance, vector, base, D, T_DIM, VEC_DIM, ctx)
    var vec_silu = silu(_t(vf.vec.copy(), [1, D], ctx), ctx).to_host(ctx)   # [1,D]

    # input projections (frozen base linears, WITH biases).
    var bi_img = Optional[Tensor](base.img_in_b[].clone(ctx))
    var img = TArc(linear(
        _t_like(img_tokens, [N_IMG, in_ch], base.img_in[], ctx),
        base.img_in[], bi_img, ctx,
    ))
    var bi_txt = Optional[Tensor](base.txt_in_b[].clone(ctx))
    var txt = TArc(linear(
        _t_like(txt_tokens, [N_TXT, txt_ch], base.txt_in[], ctx),
        base.txt_in[], bi_txt, ctx,
    ))

    var dbl_img_in = List[TArc]()
    var dbl_txt_in = List[TArc]()
    var dbl_img_mod = List[List[Float32]]()
    var dbl_txt_mod = List[List[Float32]]()
    for bi in range(num_double):
        dbl_img_in.append(img.copy())
        dbl_txt_in.append(txt.copy())
        var w = _load_flux_dbl_slot(store, bi, ctx)
        if bi + 1 < num_double:
            flux_host_bf16_prefetch_dbl(store, bi + 1)
        else:
            flux_host_bf16_prefetch_sgl(store, 0)
        # FLUX delta: per-block mod flats from the frozen mod.lin projections.
        var im_flat = _mod_proj(vec_silu.copy(), base.dbl_mod[bi].img, 6 * D, D, ctx)
        var tm_flat = _mod_proj(vec_silu.copy(), base.dbl_mod[bi].txt, 6 * D, D, ctx)
        var im_dev = modvecs_to_device(_modvecs_from_flat(im_flat, D), D, ctx)
        var tm_dev = modvecs_to_device(_modvecs_from_flat(tm_flat, D), D, ctx)
        var bl = _ft_dbl_none_lora()
        var fwd = chroma_double_block_lora_forward_device[H, Dh, N_IMG, N_TXT, S](
            img, txt, w, im_dev, tm_dev, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        dbl_img_mod.append(im_flat^)
        dbl_txt_mod.append(tm_flat^)
        img = fwd.img_out.copy()
        txt = fwd.txt_out.copy()
        ctx.synchronize()   # slot li%2 reuse safety + bound deferred frees

    # joint sequence: txt FIRST then img (Flux convention).
    var x = TArc(concat(0, ctx, txt[], img[]))

    var sgl_x_in = List[TArc]()
    var sgl_mod_flat = List[List[Float32]]()
    for bi in range(num_single):
        sgl_x_in.append(x.copy())
        var w = _load_flux_sgl_slot(store, bi, ctx)
        flux_host_bf16_prefetch_sgl(store, bi + 1)
        var sm_flat = _mod_proj(vec_silu.copy(), base.sgl_mod[bi], 3 * D, D, ctx)
        var sm_dev = single_modvecs_to_device(_single_modvecs_from_flat(sm_flat, D), D, ctx)
        var sl = _ft_sgl_none_lora()
        var fwd = chroma_single_block_lora_forward_device[H, Dh, S](
            x, w, sm_dev, sl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        sgl_mod_flat.append(sm_flat^)
        x = fwd.out.copy()
        ctx.synchronize()

    # final layer: silu(vec) -> adaLN.1 -> (shift,scale); ln -> modulate -> lin.
    var img_out = TArc(slice(x[], 0, N_TXT, N_IMG, ctx))
    var fb = Optional[Tensor](base.final_adaln_b[].clone(ctx))
    var fmods = linear(
        _t(vec_silu.copy(), [1, D], ctx), base.final_adaln_w[], fb, ctx,
    ).to_host(ctx)   # [1, 2D]: chunk0=shift, chunk1=scale (flux_stack:565-567)
    var final_shift = _chunk(fmods, 0, D)
    var final_scale = _chunk(fmods, 1, D)

    var ones_bf = _t(_ones(D), [D], ctx)
    var zeros_bf = _t(_zeros(D), [D], ctx)
    var ln_img_out = TArc(layer_norm(img_out[], ones_bf, zeros_bf, eps, ctx))
    var normed = modulate(
        ln_img_out[], _t(final_scale.copy(), [D], ctx), _t(final_shift.copy(), [D], ctx), ctx,
    )
    var pb = Optional[Tensor](base.final_lin_b[].clone(ctx))
    var out = linear(normed, base.final_lin[], pb, ctx).to_host(ctx)

    return FluxStackFTForward(
        out^, dbl_img_in^, dbl_txt_in^, sgl_x_in^,
        img_out^, ln_img_out^,
        dbl_img_mod^, dbl_txt_mod^, sgl_mod_flat^,
        final_shift^, final_scale^,
        vec_silu^,
    )


# ── phase (b) backward: streamed FT stack backward + fused device Adafactor ──
struct FluxStackFTWrite(Movable):
    var grad_count: Int
    var streaming_sync_count: Int

    def __init__(out self, grad_count: Int, streaming_sync_count: Int):
        self.grad_count = grad_count
        self.streaming_sync_count = streaming_sync_count


# fused Adafactor dispatch: FACTORED (2D) vs UNFACTORED (1D, cols==0). Same
# scalar bookkeeping either way; mutates p (BF16, SR bf16 write) in place.
def _af_step(
    p: Tensor, g: Tensor, state: AdafactorDeviceState,
    t_step: Int, lr: Float64, beta2_decay: Float64, eps2: Float64,
    d_thresh: Float64, weight_decay: Float64, seed: UInt64, ctx: DeviceContext,
) raises:
    if state.factored:
        adafactor_step_device(
            p, g, state, t_step, lr, beta2_decay, Float64(-1.0), eps2,
            d_thresh, weight_decay, seed, ctx,
        )
    else:
        adafactor_step_device_1d(
            p, g, state, t_step, lr, beta2_decay, Float64(-1.0), eps2,
            d_thresh, weight_decay, seed, ctx,
        )


def flux_stack_ft_backward_streamed[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_out: List[Float32],            # [N_IMG*out_ch] dL/d(velocity)
    base: FluxStackBase,
    store: FluxHostBf16,
    cos: List[Float32], sin: List[Float32],
    saved: FluxStackFTForward,
    D: Int, Fmlp: Int, out_ch: Int, eps: Float32,
    mut af_states: List[AdafactorDeviceState],   # flat scheme (module header)
    t_step: Int,                     # 1-based optimizer step
    lr: Float64, beta2_decay: Float64, eps2: Float64, d_thresh: Float64,
    weight_decay: Float64,
    sr_seed: UInt64,                 # per-step seed; 0 = RNE writes
    ctx: DeviceContext,
) raises -> FluxStackFTWrite:
    """DESCENDING walk (singles then doubles, deepest first): load slot →
    recompute the block forward from the saved input → P1 flux FT `_dev`
    backward (SURFACE_V2 + MOD_GRADS: matmul dW + bias d_b + q/k rms d_g + the
    [6D]/[3D] mod.lin flat grad) → per-param fused device Adafactor (2D factored
    / 1D unfactored) mutating the slot (block params) or the resident mod.lin
    tensor in place (SR bf16 write) → D2H write-back → carry d_x. The mod.lin
    Linears (device-resident in FluxStackBase) are trained IN PLACE: their flat
    grad routes as d_W = flatᵀ@silu(vec) / d_b = colsum(flat), so the NEXT
    forward's _mod_proj reads the updated weights (the forward-needed trap)."""
    var num_double = store.num_double
    var num_single = store.num_single
    var block_states = num_double * FLUX_DBL_FT_KEYS + num_single * FLUX_SGL_FT_KEYS
    var mod_states = num_double * FLUX_DBL_MOD_KEYS + num_single * FLUX_SGL_MOD_KEYS
    if len(af_states) != block_states + mod_states:
        raise Error("flux_stack_ft_backward_streamed: af_states length mismatch")
    # mod.lin af_states are appended after the block states (doubles then singles).
    var dbl_mod_base = block_states
    var sgl_mod_base = block_states + num_double * FLUX_DBL_MOD_KEYS
    var silu_bf = _t(saved.vec_silu.copy(), [1, D], ctx)   # bf16, mod.lin bwd input

    var cos_t = Tensor.from_host(cos.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sin.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    # Prime the DESCENDING walk: deepest single block first.
    flux_host_bf16_prefetch_sgl(store, num_single - 1)

    # frozen final-layer chain: final_lin (dx only) -> modulate (frozen adaLN
    # shift/scale) -> ln (no affine).
    var ones_bf = _t(_ones(D), [D], ctx)
    var fs_bf = _t(saved.final_scale.copy(), [D], ctx)
    var d_out_bf = _t_like(d_out, [N_IMG, out_ch], base.final_lin[], ctx)
    var d_normed_t = linear_backward_dx(d_out_bf, base.final_lin[], N_IMG, D, out_ch, ctx)
    var mbf = modulate_backward(_dev_bf16(d_normed_t, ctx), saved.ln_img_out[], fs_bf, ctx, False)
    var d_img_out_bf = layer_norm_backward_dx(mbf.d_x, saved.img_out[], ones_bf, eps, ctx)
    var d_txt_zero = zeros_device([N_TXT, D], STDtype.F32, ctx)
    var d_x = TArc(concat(0, ctx, d_txt_zero, _dev_f32(d_img_out_bf, ctx)))   # F32 [S,D]

    var grad_count = 0
    var streaming_sync_count = 0

    # ── singles, deepest → shallowest ─────────────────────────────────────────
    var bi = num_single - 1
    var sgl_state_base = num_double * FLUX_DBL_FT_KEYS
    while bi >= 0:
        var w = _load_flux_sgl_slot(store, bi, ctx)
        if bi > 0:
            flux_host_bf16_prefetch_sgl(store, bi - 1)
        else:
            flux_host_bf16_prefetch_dbl(store, num_double - 1)
        var sm_dev = single_modvecs_to_device(
            _single_modvecs_from_flat(saved.sgl_mod_flat[bi].copy(), D), D, ctx)
        var sl = _ft_sgl_none_lora()
        var rfwd = chroma_single_block_lora_forward_device[H, Dh, S](
            saved.sgl_x_in[bi], w, sm_dev, sl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        var bg = flux_single_block_ft_backward_dev[H, Dh, S, True, True, True](
            d_x, w, sm_dev, rfwd.saved, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        # fused back pass: update THIS block's 6 slot params in place (2D
        # factored / 1D unfactored dispatch).
        var slot = bi % 2
        for wi in range(FLUX_SGL_FT_KEYS):
            var flat = sgl_state_base + bi * FLUX_SGL_FT_KEYS + wi
            _af_step(
                store.slot_sgl[slot * FLUX_SGL_FT_KEYS + wi][], bg.dw[wi][],
                af_states[flat], t_step, lr, beta2_decay, eps2, d_thresh,
                weight_decay, sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)), ctx,
            )
        flux_host_bf16_writeback_sgl(store, bi, ctx)
        # mod.lin (device-resident, in-place): route the [1,3D] flat grad.
        ref d_sm = bg.mod_flat[0][]                        # [1,3D] F32
        var dW_sm = linear_backward_dw(_dev_bf16(d_sm, ctx), silu_bf, 1, D, 3 * D, ctx, STDtype.F32)
        var dB_sm = linear_backward_db(d_sm, 1, 3 * D, ctx)   # [3D] F32
        var mw = sgl_mod_base + bi * FLUX_SGL_MOD_KEYS + 0
        var mb = sgl_mod_base + bi * FLUX_SGL_MOD_KEYS + 1
        _af_step(
            base.sgl_mod[bi].w[], dW_sm, af_states[mw], t_step, lr, beta2_decay,
            eps2, d_thresh, weight_decay,
            sr_seed ^ (UInt64(mw + 1) * UInt64(0x9E37)), ctx,
        )
        _af_step(
            base.sgl_mod[bi].b[], dB_sm, af_states[mb], t_step, lr, beta2_decay,
            eps2, d_thresh, weight_decay,
            sr_seed ^ (UInt64(mb + 1) * UInt64(0x9E37)), ctx,
        )
        grad_count += FLUX_SGL_FT_KEYS + FLUX_SGL_MOD_KEYS
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
        var w = _load_flux_dbl_slot(store, di, ctx)
        flux_host_bf16_prefetch_dbl(store, di - 1)
        var im_dev = modvecs_to_device(
            _modvecs_from_flat(saved.dbl_img_mod[di].copy(), D), D, ctx)
        var tm_dev = modvecs_to_device(
            _modvecs_from_flat(saved.dbl_txt_mod[di].copy(), D), D, ctx)
        var bl = _ft_dbl_none_lora()
        var rfwd = chroma_double_block_lora_forward_device[H, Dh, N_IMG, N_TXT, S](
            saved.dbl_img_in[di], saved.dbl_txt_in[di],
            w, im_dev, tm_dev, bl, cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        var bg = flux_double_block_ft_backward_dev[H, Dh, N_IMG, N_TXT, S, True, True, True](
            d_io, d_to, w, im_dev, tm_dev, rfwd.saved,
            cos_t, sin_t, D, Fmlp, eps, ctx,
        )
        var slot = di % 2
        for wi in range(FLUX_DBL_FT_KEYS):
            var flat = di * FLUX_DBL_FT_KEYS + wi
            _af_step(
                store.slot_dbl[slot * FLUX_DBL_FT_KEYS + wi][], bg.dw[wi][],
                af_states[flat], t_step, lr, beta2_decay, eps2, d_thresh,
                weight_decay, sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)), ctx,
            )
        flux_host_bf16_writeback_dbl(store, di, ctx)
        # mod.lin (device-resident, in-place): img flat [0], txt flat [1].
        ref d_im = bg.mod_flat[0][]                        # [1,6D] F32
        ref d_tm = bg.mod_flat[1][]
        var dW_im = linear_backward_dw(_dev_bf16(d_im, ctx), silu_bf, 1, D, 6 * D, ctx, STDtype.F32)
        var dB_im = linear_backward_db(d_im, 1, 6 * D, ctx)
        var dW_tm = linear_backward_dw(_dev_bf16(d_tm, ctx), silu_bf, 1, D, 6 * D, ctx, STDtype.F32)
        var dB_tm = linear_backward_db(d_tm, 1, 6 * D, ctx)
        var miw = dbl_mod_base + di * FLUX_DBL_MOD_KEYS + 0
        var mib = dbl_mod_base + di * FLUX_DBL_MOD_KEYS + 1
        var mtw = dbl_mod_base + di * FLUX_DBL_MOD_KEYS + 2
        var mtb = dbl_mod_base + di * FLUX_DBL_MOD_KEYS + 3
        _af_step(base.dbl_mod[di].img.w[], dW_im, af_states[miw], t_step, lr,
                 beta2_decay, eps2, d_thresh, weight_decay,
                 sr_seed ^ (UInt64(miw + 1) * UInt64(0x9E37)), ctx)
        _af_step(base.dbl_mod[di].img.b[], dB_im, af_states[mib], t_step, lr,
                 beta2_decay, eps2, d_thresh, weight_decay,
                 sr_seed ^ (UInt64(mib + 1) * UInt64(0x9E37)), ctx)
        _af_step(base.dbl_mod[di].txt.w[], dW_tm, af_states[mtw], t_step, lr,
                 beta2_decay, eps2, d_thresh, weight_decay,
                 sr_seed ^ (UInt64(mtw + 1) * UInt64(0x9E37)), ctx)
        _af_step(base.dbl_mod[di].txt.b[], dB_tm, af_states[mtb], t_step, lr,
                 beta2_decay, eps2, d_thresh, weight_decay,
                 sr_seed ^ (UInt64(mtb + 1) * UInt64(0x9E37)), ctx)
        grad_count += FLUX_DBL_FT_KEYS + FLUX_DBL_MOD_KEYS
        d_io = bg.d_img_x.copy()
        d_to = bg.d_txt_x.copy()
        ctx.synchronize()
        streaming_sync_count += 1
        di -= 1

    return FluxStackFTWrite(grad_count, streaming_sync_count)


# ── Adafactor factored-state builder (flat scheme, device-resident ~few MB) ──
def build_flux_ft_adafactor_states(
    store: FluxHostBf16, base: FluxStackBase, ctx: DeviceContext
) raises -> List[AdafactorDeviceState]:
    """FACTORED [rows]+[cols] for 2D matmuls + mod.lin weights; UNFACTORED
    exp_avg_sq [n] (cols==0) for 1D biases + q/k rms scales + mod.lin biases.
    Order: block states (doubles then singles) THEN mod.lin states (doubles
    then singles) — must match flux_ft_state_shapes + the backward's indexing."""
    var states = List[AdafactorDeviceState]()
    for _bi in range(store.num_double):
        for wi in range(FLUX_DBL_FT_KEYS):
            ref sh = store.dbl[0].w_shape[wi]
            if len(sh) == 2:
                states.append(AdafactorDeviceState(sh[0], sh[1], ctx))
            else:
                states.append(AdafactorDeviceState(sh[0], ctx))   # unfactored 1D
    for _bi in range(store.num_single):
        for wi in range(FLUX_SGL_FT_KEYS):
            ref sh = store.sgl[0].w_shape[wi]
            if len(sh) == 2:
                states.append(AdafactorDeviceState(sh[0], sh[1], ctx))
            else:
                states.append(AdafactorDeviceState(sh[0], ctx))
    # mod.lin (FLUX delta): per double [img.w 2D, img.b 1D, txt.w 2D, txt.b 1D],
    # per single [w 2D, b 1D].
    for bi in range(store.num_double):
        for mi in range(FLUX_DBL_MOD_KEYS):
            ref sh = _mod_tensor_dbl(base, bi, mi)[].shape()
            if len(sh) == 2:
                states.append(AdafactorDeviceState(sh[0], sh[1], ctx))
            else:
                states.append(AdafactorDeviceState(sh[0], ctx))
    for bi in range(store.num_single):
        for mi in range(FLUX_SGL_MOD_KEYS):
            ref sh = _mod_tensor_sgl(base, bi, mi)[].shape()
            if len(sh) == 2:
                states.append(AdafactorDeviceState(sh[0], sh[1], ctx))
            else:
                states.append(AdafactorDeviceState(sh[0], ctx))
    return states^
