# serenitymojo/models/ideogram4/ideogram4_full_ft.mojo
#
# Ideogram-4 FULL FINETUNE phases (a)+(b) (FULL_FINETUNE_ROLLOUT_PLAN_2026-07-07
# ideogram4 card; blueprint .claude/skills/reference-mojo-full-finetune — krea2 worked
# example; fleet shape = models/chroma/chroma_full_ft.mojo, adapted to
# ideogram4's UNIFORM 34-layer stack — ONE block kind, 6 trained mats/block):
#
#   (a) Ideogram4HostBf16: the pinned-host bf16 BOTH-WAYS store. All 34 layers'
#       13 trained per-block params (v2 FULL SURFACE, FULL_SURFACE_PLAN_
#       2026-07-08 Phase B row 1; I4_FT_SLOT order — slots 0-5 the 6 matmuls:
#       attention.qkv [13824,4608], attention.o [4608,4608],
#       feed_forward.w1 [12288,4608], feed_forward.w2 [4608,12288],
#       feed_forward.w3 [12288,4608], adaln_modulation [18432,512]; slots 6-12
#       the 1D params: adaln_modulation.bias [18432], attention_norm1/2 [4608],
#       ffn_norm1/2 [4608], attention.norm_q/norm_k [256]) live as PINNED HOST
#       BF16 bytes (~18.0GB host RAM — the store IS the live model; ~8.985B
#       trained params of the DiT). The checkpoint stores the matmuls FP8
#       (F8_E4M3 [out,in] + F32 per-row `<name>_scale`, ~9.3GB on disk) — the
#       store build CONVERTS each one via ops/fp8.load_fp8_dequant (the SAME
#       dequant the resident trainer path uses: fp8_e4m3_dequant_perrow_to_bf16,
#       per-row scale; models/dit/ideogram4_resident.mojo `_lin` and block.mojo
#       `load_ideogram4_block_weights` both go through it), then D2H-pins the
#       resulting bf16 bytes; the 1D params are bf16 on disk and pin byte-exact
#       (no dequant). Two device slot-sets (13 tensors each), double-
#       buffered on a dedicated copy stream with per-slot H2D-done events (the
#       krea2/chroma shape). The GLOBAL frozen tensors (embedders input_proj/
#       t_embedding/adaln_proj/llm_cond_*/embed_image_indicator + final_layer)
#       are held device-resident by the TRAINER arm (fp8-resident
#       Ideogram4Weights subset; Serenity Ideogram4LiveTrainer IDEOGRAM4_FULL_FT).
#       ideogram4_host_bf16_save writes the pinned bytes STRAIGHT to a
#       safetensors overlay (ORIGINAL checkpoint tensor names, 442 tensors:
#       the 6 matmul .weight keys + adaln_modulation.bias +
#       attention_norm1/2.weight + ffn_norm1/2.weight +
#       attention.norm_q/norm_k.weight per layer — note: saved BF16, an
#       overlay to apply OVER the fp8 base at load) with no GPU round-trip.
#       v1 overlays (204 tensors) and v1 sidecars (204 states) FAIL LOUD on
#       resume (count mismatches) — the v2 surface REPLACES v1.
#
#   (b) ideogram4_stack_ft_forward_streamed / ideogram4_stack_ft_backward_streamed:
#       the streamed FT stack step. Forward walks the 34 layers loading the
#       LIVE weights from the store (block fwd = the parity-verified
#       ideogram4_block_lora_forward with ZERO-B adapters — up = down@0 = 0,
#       add(base, 0*scale) = base, so the produced Ideogram4BlockActs tape is
#       the base block's bit-for-bit; EXACTLY what the P1 gate
#       models/ideogram4/parity/ideogram4_block_ft_parity.mojo pairs with the
#       backward). Saves the per-layer x inputs (the backward's recompute
#       checkpoints; adaln_input is shared by all 34 layers — one tensor).
#       Backward walks in reverse: load slot → recompute the block forward
#       from the saved input → ideogram4_block_ft_backward_dev (commit b75c04d,
#       aitk-oracle-gated: 13 dW/d_b/d_g arms on REAL block-0 weights) →
#       fused per-param DEVICE Adafactor step (training/adafactor_device.mojo,
#       bit-gated; 2D factored / 1D unfactored dispatch on state.factored;
#       per-weight SR seed mix, flat index li*13+wi) mutating the
#       SLOT weights in place → D2H write-back into the pinned host bytes →
#       carry d_x + FOLD d_adaln_input (mirrors the LoRA stack backward's
#       fold: block.mojo ideogram4_stack_lora_backward_resident
#       `d_adaln = add(d_adaln, bb.d_adaln_input)` — adaln_input is SHARED
#       across layers so its grads SUM; unused upstream in v1, the embed is
#       frozen, but the carry contract is mirrored). Per-block
#       ctx.synchronize() (the krea2 discipline) makes slot li%2 reuse safe
#       with no compute-done events and bounds the deferred frees to a block.
#
# Tensor shapes through the stack follow the P1 GATE exactly (2-D x [S,Hidden],
# adaln_input [1,Adaln] — ideogram4_block_ft_parity.mojo:134-141); the trainer
# arm reshapes its [1,SEQ,Hidden]/[1,1,Adaln] embed outputs at the seam.
#
# Adafactor state flat-index scheme (af_states): li * 13 + wi, wi in
# I4_FT_SLOT order (qkv, o, w1, w2, w3, adaln_w, adaln_b, attn_norm1/2,
# ffn_norm1/2, norm_q, norm_k) — matches Ideogram4BlockFTGrads.dw EXACTLY.
# Slots 0-5 are FACTORED (2D) states; slots 6-12 are UNFACTORED (1D, cols==0
# sentinel — torch uses the unfactored second moment for <2D params).
#
# TRAPS honored (reference-mojo-full-finetune): dW arrives F32 from the P1 backward
# (linear_backward_dw(..., STDtype.F32) explicit); "weights changed" checks
# must diff WHOLE tensors (sub-ulp SR statistics); quantized full-FT is
# forbidden (the store dequants to bf16 ONCE — training math is bf16, reference trainer
# contract; the fp8 bytes are never trained through).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import (
    DeviceContext, HostBuffer, DeviceStream, DeviceEvent,
)
from std.collections import List
from std.memory import ArcPointer, alloc
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_close, sys_pwrite, O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.offload.turbo_loader import _h2d_dma_copy
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.fp8 import load_fp8_dequant
from serenitymojo.ops.tensor_algebra import add, zeros_device

from serenitymojo.models.ideogram4.block import (
    I4_SLOTS_PER_BLOCK,
    Ideogram4BlockWeights,
    build_ideogram4_lora_set,
    ideogram4_block_lora_forward,
    LArc,
    _clone,
)
from serenitymojo.models.ideogram4.ideogram4_block_ft import (
    I4_FT_SLOTS_V2,
    ideogram4_block_ft_backward_dev,
)
from serenitymojo.training.adafactor_device import (
    AdafactorDeviceState, adafactor_step_device, adafactor_step_device_1d,
)
# FULL-FT resume (the fleet sidecar): overlay the trained weights over the
# base-built pinned host store (byte-exact, dtype-checked).
from serenitymojo.training.full_ft_sidecar import full_ft_overlay_into_host_store

comptime TArc = ArcPointer[Tensor]
comptime HArc = ArcPointer[HostBuffer[DType.uint8]]

# trained params per layer (the v2 FULL surface; I4_FT_SLOT order — slots 0-5
# the I4_SLOT matmuls, slots 6-12 the 1D adaln bias + rms scales).
comptime I4_FT_MM_KEYS = I4_SLOTS_PER_BLOCK   # 6 fp8-dequanted matmuls
comptime I4_FT_KEYS = I4_FT_SLOTS_V2          # 13 trained tensors per layer


# save-key tails, in the SAME order as the slots / Ideogram4BlockFTGrads.dw
# (ORIGINAL checkpoint names — layers.N.<tail>).
def _i4_ft_key_tails() -> List[String]:
    var t = List[String]()
    t.append(String("attention.qkv.weight"))       # I4_SLOT_QKV
    t.append(String("attention.o.weight"))         # I4_SLOT_O
    t.append(String("feed_forward.w1.weight"))     # I4_SLOT_W1
    t.append(String("feed_forward.w2.weight"))     # I4_SLOT_W2
    t.append(String("feed_forward.w3.weight"))     # I4_SLOT_W3
    t.append(String("adaln_modulation.weight"))    # I4_SLOT_ADALN
    t.append(String("adaln_modulation.bias"))      # I4_FT_SLOT_ADALN_B
    t.append(String("attention_norm1.weight"))     # I4_FT_SLOT_ATTN_NORM1
    t.append(String("attention_norm2.weight"))     # I4_FT_SLOT_ATTN_NORM2
    t.append(String("ffn_norm1.weight"))           # I4_FT_SLOT_FFN_NORM1
    t.append(String("ffn_norm2.weight"))           # I4_FT_SLOT_FFN_NORM2
    t.append(String("attention.norm_q.weight"))    # I4_FT_SLOT_NORM_Q
    t.append(String("attention.norm_k.weight"))    # I4_FT_SLOT_NORM_K
    return t^


# ── per-layer pinned-host record (ALL 13 LIVE trained tensors, slot order) ───
struct Ideogram4LayerHostBf16(Copyable, Movable):
    var w_h: List[HArc]          # 13 pinned host BF16 (LIVE weights, slot order)
    var w_nbytes: List[Int]
    var w_shape: List[List[Int]] # 2-D for slots 0-5, 1-D for slots 6-12

    def __init__(
        out self,
        var w_h: List[HArc], var w_nbytes: List[Int], var w_shape: List[List[Int]],
    ):
        self.w_h = w_h^
        self.w_nbytes = w_nbytes^
        self.w_shape = w_shape^


struct Ideogram4HostBf16(Copyable, Movable):
    var layers: List[Ideogram4LayerHostBf16]   # len == num_layers (34)
    # double-buffer device slots + one copy stream + per-slot H2D-done events.
    var slot: List[TArc]                        # 26 = slot_idx*13 + wi, BF16 device
    var copy_stream: List[ArcPointer[DeviceStream]]   # len 1 (Arc'd: Copyable)
    var ev: List[ArcPointer[DeviceEvent]]             # len 2: per-slot H2D-done
    var num_layers: Int

    def __init__(
        out self,
        var layers: List[Ideogram4LayerHostBf16],
        var slot: List[TArc],
        var copy_stream: List[ArcPointer[DeviceStream]],
        var ev: List[ArcPointer[DeviceEvent]],
        num_layers: Int,
    ):
        self.layers = layers^
        self.slot = slot^
        self.copy_stream = copy_stream^
        self.ev = ev^
        self.num_layers = num_layers


# fp8 checkpoint matrix -> bf16 (load_fp8_dequant: F8_E4M3 [out,in] + F32
# per-row `_scale` -> fp8_e4m3_dequant_perrow_to_bf16, the EXACT dequant the
# resident loader uses) -> D2H into a fresh PINNED host buffer. Fail loud if
# the dequant did not land BF16 2-D (quantized/other-dtype full-FT forbidden).
def _pin_dequant_weight(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext,
    mut w_h: List[HArc], mut w_nbytes: List[Int], mut w_shape: List[List[Int]],
) raises:
    var t = load_fp8_dequant(st, name, ctx)
    if t.dtype() != STDtype.BF16:
        raise Error(
            String("ideogram4 full-FT store: ") + name
            + String(" did not dequant to BF16 (dtype.tag ")
            + String(t.dtype().tag) + String(")")
        )
    var sh = t.shape()
    if len(sh) != 2:
        raise Error(String("ideogram4 full-FT store: ") + name + String(" is not 2-D"))
    var bh = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
    ctx.enqueue_copy(bh, t.buf)   # D2H
    ctx.synchronize()
    w_h.append(HArc(bh^))
    w_nbytes.append(t.nbytes())
    w_shape.append(sh.copy())


# bf16-on-disk 1D param (adaln bias / rms scales / norm_q/k) -> byte-exact
# PINNED host copy (no dequant — these are unquantized in the fp8 checkpoint).
# Fail loud on non-BF16 or non-1-D (quantized/other-dtype full-FT forbidden).
def _pin_bf16_1d_weight(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext,
    mut w_h: List[HArc], mut w_nbytes: List[Int], mut w_shape: List[List[Int]],
) raises:
    var t = Tensor.from_view(st.tensor_view(name), ctx)
    if t.dtype() != STDtype.BF16:
        raise Error(
            String("ideogram4 full-FT store: ") + name
            + String(" is not BF16 on disk (dtype.tag ")
            + String(t.dtype().tag) + String(")")
        )
    var sh = t.shape()
    if len(sh) != 1:
        raise Error(String("ideogram4 full-FT store: ") + name + String(" is not 1-D"))
    var bh = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
    ctx.enqueue_copy(bh, t.buf)   # D2H
    ctx.synchronize()
    w_h.append(HArc(bh^))
    w_nbytes.append(t.nbytes())
    w_shape.append(sh.copy())


def build_ideogram4_host_bf16(
    st: ShardedSafeTensors, num_layers: Int, ctx: DeviceContext
) raises -> Ideogram4HostBf16:
    """Convert ALL 34 layers' 13 trained per-block params to PINNED-HOST BF16
    (the full-FT live model, ~18GB host RAM): slots 0-5 fp8->bf16 dequanted
    matmuls, slots 6-12 the bf16-on-disk 1D adaln bias + rms scales
    (byte-exact pins — the v2 FULL surface, nothing per-block stays frozen)."""
    var layers = List[Ideogram4LayerHostBf16]()
    var tails = _i4_ft_key_tails()
    for li in range(num_layers):
        var p = String("layers.") + String(li) + String(".")
        var w_h = List[HArc]()
        var w_nbytes = List[Int]()
        var w_shape = List[List[Int]]()
        for ki in range(I4_FT_MM_KEYS):
            _pin_dequant_weight(st, p + tails[ki], ctx, w_h, w_nbytes, w_shape)
        for ki in range(I4_FT_MM_KEYS, I4_FT_KEYS):
            _pin_bf16_1d_weight(st, p + tails[ki], ctx, w_h, w_nbytes, w_shape)
        layers.append(Ideogram4LayerHostBf16(w_h^, w_nbytes^, w_shape^))
        cu_mempool_trim_current(0)
        if (li + 1) % 4 == 0 or li + 1 == num_layers:
            print("full-ft bf16 host store: pinned+converted layer", li + 1, "/", num_layers)

    # device slots (shapes uniform across layers — asserted off layer 0)
    var slot = List[TArc]()
    ref l0 = layers[0]
    for s in range(2):
        _ = s
        for i in range(I4_FT_KEYS):
            var dbuf = ctx.enqueue_create_buffer[DType.uint8](l0.w_nbytes[i])
            var t = Tensor(dbuf^, l0.w_shape[i].copy(), STDtype.BF16)
            slot.append(TArc(t^))
    ctx.synchronize()
    # layer-uniformity fail-loud (slot reuse depends on it)
    for li in range(num_layers):
        for i in range(I4_FT_KEYS):
            if layers[li].w_nbytes[i] != l0.w_nbytes[i]:
                raise Error("ideogram4 full-FT store: layer weight shapes not uniform")

    var copy_stream = List[ArcPointer[DeviceStream]]()
    copy_stream.append(ArcPointer(ctx.create_stream()))
    var ev = List[ArcPointer[DeviceEvent]]()
    ev.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    ev.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    return Ideogram4HostBf16(layers^, slot^, copy_stream^, ev^, num_layers)


# ── prefetch (async H2D on the copy stream into slot li%2, record event) ─────
def ideogram4_host_bf16_prefetch(store: Ideogram4HostBf16, li: Int) raises:
    if li < 0 or li >= len(store.layers):
        return
    var s = li % 2
    ref b = store.layers[li]
    for i in range(I4_FT_KEYS):
        _h2d_dma_copy(
            UInt64(Int(store.slot[s * I4_FT_KEYS + i][].buf.unsafe_ptr())),
            b.w_h[i][].unsafe_ptr(),
            b.w_nbytes[i],
            store.copy_stream[0][],
        )
    store.copy_stream[0][].record_event(store.ev[s][])


# alias view over a persistent device tensor: refcounted buffer copy, SAME
# device memory (the block.mojo `Tensor(t.buf.copy(), t.shape(), t.dtype())`
# idiom) — the optimizer's in-place slot mutation is visible through it.
def _alias_t(t: TArc) raises -> Tensor:
    return Tensor(t[].buf.copy(), t[].shape(), t[].dtype())


# ── slot loader (fence the H2D, wrap the PERSISTENT slot tensors) ─────────────
def _load_ideogram4_slot(
    store: Ideogram4HostBf16, li: Int, ctx: DeviceContext
) raises -> Ideogram4BlockWeights:
    """Fence slot li%2's H2D and build Ideogram4BlockWeights ALIASING the SAME
    slot device buffers (buf.copy() = refcount, same memory) — the FT optimizer
    mutates these in place pre-writeback. ALL 13 per-block params are trained
    (v2) and stream from the slots. Field order matches block.mojo's struct;
    slot map is I4_FT_SLOT order: qkv_w<-0, o_w<-1, w1<-2, w2<-3, w3<-4,
    adaln_w<-5, adaln_b<-6, attn_norm1<-7, attn_norm2<-8, ffn_norm1<-9,
    ffn_norm2<-10, norm_q<-11, norm_k<-12."""
    var s = li % 2
    ctx.stream().enqueue_wait_for(store.ev[s][])
    var base = s * I4_FT_KEYS
    return Ideogram4BlockWeights(
        _alias_t(store.slot[base + 5]),   # adaln_w    (I4_SLOT_ADALN)
        _alias_t(store.slot[base + 6]),   # adaln_b    (I4_FT_SLOT_ADALN_B)
        _alias_t(store.slot[base + 7]),   # attn_norm1 (I4_FT_SLOT_ATTN_NORM1)
        _alias_t(store.slot[base + 8]),   # attn_norm2 (I4_FT_SLOT_ATTN_NORM2)
        _alias_t(store.slot[base + 9]),   # ffn_norm1  (I4_FT_SLOT_FFN_NORM1)
        _alias_t(store.slot[base + 10]),  # ffn_norm2  (I4_FT_SLOT_FFN_NORM2)
        _alias_t(store.slot[base + 0]),   # qkv_w      (I4_SLOT_QKV)
        _alias_t(store.slot[base + 1]),   # o_w        (I4_SLOT_O)
        _alias_t(store.slot[base + 11]),  # norm_q     (I4_FT_SLOT_NORM_Q)
        _alias_t(store.slot[base + 12]),  # norm_k     (I4_FT_SLOT_NORM_K)
        _alias_t(store.slot[base + 2]),   # w1         (I4_SLOT_W1)
        _alias_t(store.slot[base + 3]),   # w2         (I4_SLOT_W2)
        _alias_t(store.slot[base + 4]),   # w3         (I4_SLOT_W3)
    )


# ── write-back (D2H the optimizer-updated slot into the pinned host bytes) ───
def ideogram4_host_bf16_writeback(
    store: Ideogram4HostBf16, li: Int, ctx: DeviceContext
) raises:
    var s = li % 2
    ref b = store.layers[li]
    for i in range(I4_FT_KEYS):
        ctx.enqueue_copy(b.w_h[i][], store.slot[s * I4_FT_KEYS + i][].buf)


# ── host-direct safetensors overlay save (no GPU round-trip) ─────────────────
def ideogram4_host_bf16_overlay_resume(
    store: Ideogram4HostBf16, overlay_path: String
) raises:
    """RESUME store rebuild, step 2: the store already holds the fp8-dequanted
    BASE checkpoint (build_ideogram4_host_bf16); copy the trained BF16 overlay's
    bytes over the pinned host bytes (byte-exact, BF16/size fail-loud). Key
    order == ideogram4_host_bf16_save's (li-major, the 13 tails); a v1
    (204-tensor, matmuls-only) overlay FAILS LOUD on the count check."""
    var tails = _i4_ft_key_tails()
    var names = List[String]()
    var bufs = List[HArc]()
    var nbytes = List[Int]()
    for li in range(len(store.layers)):
        ref b = store.layers[li]
        for wi in range(I4_FT_KEYS):
            names.append(String("layers.") + String(li) + String(".") + tails[wi])
            bufs.append(b.w_h[wi].copy())
            nbytes.append(b.w_nbytes[wi])
    full_ft_overlay_into_host_store(overlay_path, names, bufs, nbytes)


def ideogram4_ft_state_shapes(
    store: Ideogram4HostBf16, mut rows: List[Int], mut cols: List[Int]
) raises:
    """FLAT per-state [rows, cols] in the af_states order (build_ideogram4_ft_
    adafactor_states) — the sidecar loader's expected-shape lists. 1-D params
    (slots 6-12) report cols == 0: the UNFACTORED sentinel the sidecar v2
    loader dispatches on."""
    for _li in range(store.num_layers):
        for wi in range(I4_FT_KEYS):
            ref sh = store.layers[0].w_shape[wi]
            rows.append(sh[0])
            if len(sh) == 2:
                cols.append(sh[1])
            else:
                cols.append(0)


def ideogram4_host_bf16_save(store: Ideogram4HostBf16, path: String) raises:
    """Write the trained surface (the pinned-host bf16 bytes — the live model)
    DIRECTLY to a safetensors file. Keys keep the ORIGINAL checkpoint names
    (layers.N.<tail>); dtype is BF16 (an overlay: load the fp8 base, then
    substitute these trained tensors). v2 saves the FULL per-block surface
    (13 tensors x 34 layers = 442: 6 mats + adaln bias + 6 rms scales)."""
    var tails = _i4_ft_key_tails()
    var header = String("{")
    var off = 0
    var first = True
    for li in range(len(store.layers)):
        ref b = store.layers[li]
        for wi in range(I4_FT_KEYS):
            var nm = String("layers.") + String(li) + String(".") + tails[wi]
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
        raise Error(String("ideogram4_host_bf16_save: cannot open ") + path)
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
    for li in range(len(store.layers)):
        ref b = store.layers[li]
        for wi in range(I4_FT_KEYS):
            var src = BytePtr(unsafe_from_address=Int(b.w_h[wi][].unsafe_ptr()))
            var wrote = sys_pwrite(fd, src, b.w_nbytes[wi], base + woff)
            if wrote != b.w_nbytes[wi]:
                _ = sys_close(fd)
                raise Error("ideogram4_host_bf16_save: short write")
            woff += b.w_nbytes[wi]
    _ = sys_close(fd)
    print(
        "[ideogram4-ft-save] wrote",
        len(store.layers) * I4_FT_KEYS,
        "trained weights ->", path,
    )


# ── ZERO-B adapter shell (the FT forward is the base block) ──────────────────
# The P1 gate's exact recipe (ideogram4_block_ft_parity.mojo:127-131):
# build_ideogram4_lora_set's make_lora_adapter zero-inits B, so
# ideogram4_block_lora_forward with these IS the base forward bit-for-bit and
# its Ideogram4BlockActs tape is the base tape the FT backward consumes.
# One 6-adapter set is shared by all 34 layers (read-only, mathematically inert).
def _ft_zero_loras[
    Hidden: Int, FF: Int, Adaln: Int
](ctx: DeviceContext) raises -> List[LArc]:
    var set = build_ideogram4_lora_set[Hidden, FF, Adaln](4, Float32(4.0), ctx, 1)
    var bl = List[LArc]()
    for slot in range(I4_SLOTS_PER_BLOCK):   # 6 matmul adapters (zero-B)
        bl.append(set.ad[slot])
    return bl^


# ── FT forward tape (device input snapshots for the backward recompute) ──────
struct Ideogram4StackFTForward(Movable):
    var out: TArc                # [S, Hidden] bf16 — stack output (pre final-layer)
    var x_inputs: List[TArc]     # num_layers x [S, Hidden] recompute checkpoints

    def __init__(out self, var out: TArc, var x_inputs: List[TArc]):
        self.out = out^
        self.x_inputs = x_inputs^


# ── phase (b) forward: streamed base-block stack forward from the host store ─
def ideogram4_stack_ft_forward_streamed[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    x_in: Tensor,          # [S, Hidden] bf16 (embed output, 2-D — the gate shape)
    adaln_input: Tensor,   # [1, Adaln]  bf16 (shared by all layers)
    cosf: Tensor,          # [1, S, Dh]  bf16
    sinf: Tensor,          # [1, S, Dh]  bf16
    store: Ideogram4HostBf16,
    ctx: DeviceContext,
) raises -> Ideogram4StackFTForward:
    """Mirror of block.mojo ideogram4_stack_lora_forward_resident with the LIVE
    bf16 host store as the weight source and ZERO-B adapters (= base forward).
    Saves per-layer x inputs (backward recompute checkpoints); per-block
    ctx.synchronize() (slot-reuse + async-free discipline)."""
    ideogram4_host_bf16_prefetch(store, 0)
    var bl = _ft_zero_loras[Hidden, FF, Adaln](ctx)

    var x = _clone(x_in, ctx)
    var saved = List[TArc]()
    for li in range(store.num_layers):
        saved.append(TArc(_clone(x, ctx)))
        var w = _load_ideogram4_slot(store, li, ctx)
        ideogram4_host_bf16_prefetch(store, li + 1)
        var out = ideogram4_block_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
            x, adaln_input, cosf, sinf, w, bl, ctx
        )
        x = out^.take_out()
        ctx.synchronize()   # slot li%2 reuse safety + bound deferred frees
    return Ideogram4StackFTForward(TArc(x^), saved^)


# ── phase (b) backward: streamed FT stack backward + fused device Adafactor ──
struct Ideogram4StackFTWrite(Movable):
    var d_x_in: TArc             # [S, Hidden] grad into the (frozen) embed
    var d_adaln: TArc            # [1, Adaln]  summed adaln-input grad (frozen upstream)
    var grad_count: Int
    var streaming_sync_count: Int

    def __init__(
        out self, var d_x_in: TArc, var d_adaln: TArc,
        grad_count: Int, streaming_sync_count: Int,
    ):
        self.d_x_in = d_x_in^
        self.d_adaln = d_adaln^
        self.grad_count = grad_count
        self.streaming_sync_count = streaming_sync_count


def ideogram4_stack_ft_backward_streamed[
    S: Int, Hidden: Int, Heads: Int, Dh: Int, FF: Int, Adaln: Int,
](
    d_out: Tensor,               # [S, Hidden] bf16 dL/d(stack out)
    adaln_input: Tensor,         # [1, Adaln]  bf16 (the forward's, shared)
    cosf: Tensor, sinf: Tensor,
    store: Ideogram4HostBf16,
    saved: Ideogram4StackFTForward,
    mut af_states: List[AdafactorDeviceState],   # flat li*13+wi (module header)
    t_step: Int,                 # 1-based optimizer step
    lr: Float64, beta2_decay: Float64, eps2: Float64, d_thresh: Float64,
    weight_decay: Float64,
    sr_seed: UInt64,             # per-step seed; 0 = RNE writes
    ctx: DeviceContext,
) raises -> Ideogram4StackFTWrite:
    """DESCENDING walk (layer 33 -> 0): load slot → recompute the block forward
    from the saved input (zero-B adapters = base tape) → P1 FT backward →
    per-param fused device Adafactor (2D factored / 1D unfactored dispatch)
    mutating the slot in place (SR bf16
    write) → D2H write-back into the host store → carry d_x + fold d_adaln
    (the LoRA stack's add-fold). Fused-back-pass shape: grads and updated
    weights never accumulate beyond one layer."""
    if len(af_states) != store.num_layers * I4_FT_KEYS:
        raise Error("ideogram4_stack_ft_backward_streamed: af_states length mismatch")

    ideogram4_host_bf16_prefetch(store, store.num_layers - 1)
    var bl = _ft_zero_loras[Hidden, FF, Adaln](ctx)

    var d_x = _clone(d_out, ctx)
    # mirrors ideogram4_stack_lora_backward_resident: zeros + per-layer add.
    var d_adaln = zeros_device(adaln_input.shape(), adaln_input.dtype(), ctx)
    var grad_count = 0
    var streaming_sync_count = 0

    var li = store.num_layers - 1
    while li >= 0:
        var w = _load_ideogram4_slot(store, li, ctx)
        ideogram4_host_bf16_prefetch(store, li - 1)
        var rfwd = ideogram4_block_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
            saved.x_inputs[li][], adaln_input, cosf, sinf, w, bl, ctx
        )
        var bg = ideogram4_block_ft_backward_dev[S, Hidden, Heads, Dh, FF, Adaln](
            d_x, rfwd.acts, cosf, sinf, w, ctx
        )
        # fused back pass: update THIS layer's 13 slot params in place —
        # FACTORED (2D matmuls, slots 0-5) via adafactor_step_device,
        # UNFACTORED (1D bias/scales, slots 6-12) via adafactor_step_device_1d
        # (torch trains <2D params with the unfactored second moment).
        var s = li % 2
        for wi in range(I4_FT_KEYS):
            var flat = li * I4_FT_KEYS + wi
            if af_states[flat].factored:
                adafactor_step_device(
                    store.slot[s * I4_FT_KEYS + wi][],
                    bg.dw[wi][],
                    af_states[flat],
                    t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                    weight_decay,
                    sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)),
                    ctx,
                )
            else:
                adafactor_step_device_1d(
                    store.slot[s * I4_FT_KEYS + wi][],
                    bg.dw[wi][],
                    af_states[flat],
                    t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                    weight_decay,
                    sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)),
                    ctx,
                )
        ideogram4_host_bf16_writeback(store, li, ctx)
        grad_count += I4_FT_KEYS
        d_adaln = add(d_adaln, bg.d_adaln_input, ctx)
        d_x = _clone(bg.d_x, ctx)
        ctx.synchronize()   # lands: grads freed, update done, write-back D2H'd
        streaming_sync_count += 1
        li -= 1

    return Ideogram4StackFTWrite(
        TArc(d_x^), TArc(d_adaln^), grad_count, streaming_sync_count
    )


# ── Adafactor state builder (flat li*13+wi, device-resident ~few MB):
# FACTORED [rows]+[cols] for the 2D matmuls (slots 0-5), UNFACTORED
# exp_avg_sq [n] (cols==0 sentinel) for the 1D bias/scales (slots 6-12). ─────
def build_ideogram4_ft_adafactor_states(
    store: Ideogram4HostBf16, ctx: DeviceContext
) raises -> List[AdafactorDeviceState]:
    var states = List[AdafactorDeviceState]()
    for _li in range(store.num_layers):
        for wi in range(I4_FT_KEYS):
            ref sh = store.layers[0].w_shape[wi]
            if len(sh) == 2:
                states.append(AdafactorDeviceState(sh[0], sh[1], ctx))
            else:
                states.append(AdafactorDeviceState(sh[0], ctx))   # unfactored 1D
    return states^
