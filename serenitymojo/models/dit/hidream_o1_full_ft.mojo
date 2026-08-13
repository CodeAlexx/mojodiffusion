# serenitymojo/models/dit/hidream_o1_full_ft.mojo
#
# HiDream-O1 FULL FINETUNE phases (a)+(b) (FULL_FINETUNE_ROLLOUT_PLAN_2026-07-07
# hidream card; blueprint .claude/skills/reference-mojo-full-finetune — krea2 worked
# example; fleet shape = models/ideogram4/ideogram4_full_ft.mojo /
# models/chroma/chroma_full_ft.mojo, adapted to hidream's UNIFORM 36-layer
# Qwen3-VL decoder stack — ONE block kind, 7 trained mats/block):
#
#   (a) HiDreamO1HostBf16: the pinned-host bf16 BOTH-WAYS store. All 36 layers'
#       11 trained per-layer params (v2 FULL SURFACE, FULL_SURFACE_PLAN_
#       2026-07-08 Phase B row 2; HD_FT_SLOT order — slots 0-6 the 7 matmuls:
#       self_attn.q_proj [4096,4096], k_proj [1024,4096], v_proj [1024,4096],
#       o_proj [4096,4096], mlp.gate_proj [12288,4096], up_proj [12288,4096],
#       down_proj [4096,12288]; slots 7-10 the 1D rms scales:
#       input_layernorm [4096], self_attn.q_norm/k_norm [128],
#       post_attention_layernorm [4096]) live as PINNED HOST BF16 bytes
#       (~12.9GiB host RAM — the store IS the live model; ~6.946B trained
#       params of the DiT). The checkpoint stores everything F32
#       (/mnt/disk1/models/hidream-o1, 8 shards, 8.80B params / 35.2GB — index
#       metadata; every tensor F32) — the store build CONVERTS each one
#       (matmuls AND 1D scales) via Tensor.from_view_as_bf16 (the SAME
#       F32→bf16 conversion HiDreamO1Offloaded.load / load_block_as_bf16 use —
#       hidream_o1.mojo:745-760; the campaign doc's "15.2 GB converted once"
#       bf16 scheme, HIDREAM_O1_TRAINING_CAMPAIGN.md:55-57), then D2H-pins the
#       bf16 bytes. Two device slot-sets (11 tensors each), double-buffered on
#       a dedicated copy stream with per-slot H2D-done events (the krea2/
#       chroma/ideogram4 shape). RESIDENT-vs-STREAMED decision: bf16-RESIDENT
#       is ruled out by MEASURED evidence — the campaign doc records the
#       resident LoRA trainer at 18,255 MiB VRAM (HIDREAM_O1_TRAINING_
#       CAMPAIGN.md:90, on the 24GB 3090) > the 5080's 16,303 MiB before any
#       F32 dW; streamed is the only fit. The GLOBAL frozen tensors
#       (embed_tokens, model.language_model.norm, x_embedder, t_embedder1,
#       final_layer2) are held device-resident by the TRAINER arm
#       (HiDreamO1Offloaded.shared; Serenity train_hidream_o1_real.mojo
#       HIDREAM_FULL_FT). hidream_o1_host_bf16_save writes the pinned bytes
#       STRAIGHT to a safetensors overlay (ORIGINAL checkpoint tensor names,
#       396 tensors: model.language_model.layers.N.{self_attn.q/k/v/o_proj,
#       mlp.gate/up/down_proj,input_layernorm,self_attn.q_norm/k_norm,
#       post_attention_layernorm}.weight — saved BF16, an overlay to apply
#       OVER the F32 base at load) with no GPU round-trip. v1 overlays (252
#       tensors) and v1 sidecars (252 states) FAIL LOUD on resume (count
#       mismatches) — the v2 surface REPLACES v1.
#
#   (b) hidream_o1_stack_ft_forward_streamed / hidream_o1_stack_ft_backward_streamed:
#       the streamed FT stack step. Forward walks the 36 layers loading the
#       LIVE weights from the store (block fwd = the parity-verified
#       hidream_o1_block_lora_forward with ALL-None adapters — _maybe_lora_apply
#       passes base through untouched, so the produced HiDreamO1BlockSaved tape
#       is the base block's bit-for-bit; EXACTLY what the P1 gate
#       models/dit/tests/hidream_o1_block_ft_parity.mojo:110-149 pairs with the
#       backward). Saves the per-layer block inputs (the backward's recompute
#       checkpoints — the LoRA trainer's recompute-checkpoint discipline,
#       train_hidream_o1_real.mojo:949-962). Backward walks in reverse: load
#       slot → recompute the block forward from the saved input →
#       hidream_o1_block_ft_backward_dev (base-only-oracle gated: 7 dW cos >=
#       0.999999999998 + 4 d_g arms at the real head config) → fused per-param
#       DEVICE Adafactor step (training/adafactor_device.mojo, bit-gated;
#       2D factored / 1D unfactored dispatch on state.factored; per-weight SR
#       seed mix, flat index li*11+wi) mutating the SLOT weights in place →
#       D2H write-back into the pinned host bytes → carry d_hidden. Per-block
#       ctx.synchronize() (the krea2 discipline) makes slot li%2 reuse safe
#       with no compute-done events and bounds the deferred frees to a block.
#
# Tensor shapes through the stack follow the P1 GATE exactly (3-D x [1,S,D],
# mask4 [1,H,S,S] block-dtype forward / mask_f32 [H*S,S] F32 backward —
# hidream_o1_block_ft_parity.mojo:122-149 and the LoRA trainer's per-step mask
# build, train_hidream_o1_real.mojo:943-947).
#
# Adafactor state flat-index scheme (af_states): li * 11 + wi, wi in
# HD_FT_SLOT order (q,k,v,o,gate,up,down,in_ln,q_norm,k_norm,post_ln) —
# matches HiDreamO1BlockFTGrads.dw EXACTLY. Slots 0-6 are FACTORED (2D)
# states; slots 7-10 are UNFACTORED (1D, cols==0 sentinel — torch uses the
# unfactored second moment for <2D params).
#
# TRAPS honored (reference-mojo-full-finetune): dW arrives F32 from the P1 backward
# (linear_backward_dw(..., STDtype.F32) explicit); "weights changed" checks
# must diff WHOLE tensors (sub-ulp SR statistics); quantized full-FT is
# forbidden (the store converts to bf16 ONCE — training math is bf16, the reference trainer
# contract; no fp8/int8 anywhere on this arm).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from max.gpu.host import (
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

from serenitymojo.models.dit.hidream_o1_train_block import (
    HiDreamO1BlockWeights,
    HiDreamO1BlockLora,
    hidream_o1_block_lora_forward,
)
from serenitymojo.models.dit.hidream_o1_block_ft import (
    HD_FT_SLOTS_PER_BLOCK,
    HD_FT_SLOTS_V2,
    hidream_o1_block_ft_backward_dev,
)
from serenitymojo.training.adafactor_device import (
    AdafactorDeviceState, adafactor_step_device, adafactor_step_device_1d,
)
# FULL-FT resume (the fleet sidecar): overlay the trained weights over the
# base-built pinned host store (byte-exact, dtype-checked).
from serenitymojo.training.full_ft_sidecar import full_ft_overlay_into_host_store

comptime TArc = ArcPointer[Tensor]
comptime HArc = ArcPointer[HostBuffer[DType.uint8]]

# trained params per layer (the v2 FULL surface; HD_FT_SLOT order — slots 0-6
# the matmuls, slots 7-10 the 1D rms-norm scales).
comptime HD_FT_MM_KEYS = HD_FT_SLOTS_PER_BLOCK   # 7 F32→bf16 converted matmuls
comptime HD_FT_KEYS = HD_FT_SLOTS_V2             # 11 trained tensors per layer

comptime _HD_LAYER_PREFIX = "model.language_model.layers."


# save-key tails, in the SAME order as the slots / HiDreamO1BlockFTGrads.dw
# (ORIGINAL checkpoint names — model.language_model.layers.N.<tail>).
def _hd_ft_key_tails() -> List[String]:
    var t = List[String]()
    t.append(String("self_attn.q_proj.weight"))    # HD_FT_SLOT_Q
    t.append(String("self_attn.k_proj.weight"))    # HD_FT_SLOT_K
    t.append(String("self_attn.v_proj.weight"))    # HD_FT_SLOT_V
    t.append(String("self_attn.o_proj.weight"))    # HD_FT_SLOT_O
    t.append(String("mlp.gate_proj.weight"))       # HD_FT_SLOT_GATE
    t.append(String("mlp.up_proj.weight"))         # HD_FT_SLOT_UP
    t.append(String("mlp.down_proj.weight"))       # HD_FT_SLOT_DOWN
    t.append(String("input_layernorm.weight"))     # HD_FT_SLOT_IN_LN
    t.append(String("self_attn.q_norm.weight"))    # HD_FT_SLOT_Q_NORM
    t.append(String("self_attn.k_norm.weight"))    # HD_FT_SLOT_K_NORM
    t.append(String("post_attention_layernorm.weight"))  # HD_FT_SLOT_POST_LN
    return t^


# ── per-layer pinned-host record (ALL 11 LIVE trained tensors, slot order) ───
struct HiDreamO1LayerHostBf16(Copyable, Movable):
    var w_h: List[HArc]          # 11 pinned host BF16 (LIVE weights, slot order)
    var w_nbytes: List[Int]
    var w_shape: List[List[Int]] # 2-D for slots 0-6, 1-D for slots 7-10

    def __init__(
        out self,
        var w_h: List[HArc], var w_nbytes: List[Int], var w_shape: List[List[Int]],
    ):
        self.w_h = w_h^
        self.w_nbytes = w_nbytes^
        self.w_shape = w_shape^


struct HiDreamO1HostBf16(Copyable, Movable):
    var layers: List[HiDreamO1LayerHostBf16]   # len == num_layers (36)
    # double-buffer device slots + one copy stream + per-slot H2D-done events.
    var slot: List[TArc]                        # 22 = slot_idx*11 + wi, BF16 device
    var copy_stream: List[ArcPointer[DeviceStream]]   # len 1 (Arc'd: Copyable)
    var ev: List[ArcPointer[DeviceEvent]]             # len 2: per-slot H2D-done
    var num_layers: Int

    def __init__(
        out self,
        var layers: List[HiDreamO1LayerHostBf16],
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


# F32 checkpoint matrix -> bf16 (Tensor.from_view_as_bf16 — the offloaded
# loader's exact conversion, hidream_o1.mojo:754) -> D2H into a fresh PINNED
# host buffer. Fail loud if the load did not land BF16 2-D (quantized/
# other-dtype full-FT forbidden).
def _pin_bf16_weight(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext,
    mut w_h: List[HArc], mut w_nbytes: List[Int], mut w_shape: List[List[Int]],
) raises:
    var t = Tensor.from_view_as_bf16(st.tensor_view(name), ctx)
    if t.dtype() != STDtype.BF16:
        raise Error(
            String("hidream full-FT store: ") + name
            + String(" did not convert to BF16 (dtype.tag ")
            + String(t.dtype().tag) + String(")")
        )
    var sh = t.shape()
    if len(sh) != 2:
        raise Error(String("hidream full-FT store: ") + name + String(" is not 2-D"))
    var bh = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
    ctx.enqueue_copy(bh, t.buf)   # D2H
    ctx.synchronize()
    w_h.append(HArc(bh^))
    w_nbytes.append(t.nbytes())
    w_shape.append(sh.copy())


# F32-on-disk 1D rms scale (in_ln / q_norm / k_norm / post_ln) -> bf16 via the
# SAME Tensor.from_view_as_bf16 conversion as the matmul pin (hidream's base
# checkpoint is F32 EVERYWHERE — unlike ideogram4's bf16-on-disk 1D params,
# this pin CONVERTS, it does not byte-copy) -> PINNED host copy. Fail loud on
# non-BF16-after-convert or non-1-D.
def _pin_bf16_1d_weight(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext,
    mut w_h: List[HArc], mut w_nbytes: List[Int], mut w_shape: List[List[Int]],
) raises:
    var t = Tensor.from_view_as_bf16(st.tensor_view(name), ctx)
    if t.dtype() != STDtype.BF16:
        raise Error(
            String("hidream full-FT store: ") + name
            + String(" did not convert to BF16 (dtype.tag ")
            + String(t.dtype().tag) + String(")")
        )
    var sh = t.shape()
    if len(sh) != 1:
        raise Error(String("hidream full-FT store: ") + name + String(" is not 1-D"))
    var bh = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
    ctx.enqueue_copy(bh, t.buf)   # D2H
    ctx.synchronize()
    w_h.append(HArc(bh^))
    w_nbytes.append(t.nbytes())
    w_shape.append(sh.copy())


def build_hidream_o1_host_bf16(
    st: ShardedSafeTensors, num_layers: Int, ctx: DeviceContext
) raises -> HiDreamO1HostBf16:
    """Convert ALL 36 layers' 11 trained per-layer params to PINNED-HOST BF16
    (the full-FT live model, ~12.9GiB host RAM): slots 0-6 the F32→bf16
    converted matmuls, slots 7-10 the F32→bf16 converted 1D rms scales (the
    v2 FULL surface, nothing per-layer stays frozen)."""
    var layers = List[HiDreamO1LayerHostBf16]()
    var tails = _hd_ft_key_tails()
    for li in range(num_layers):
        var p = String(_HD_LAYER_PREFIX) + String(li) + String(".")
        var w_h = List[HArc]()
        var w_nbytes = List[Int]()
        var w_shape = List[List[Int]]()
        for ki in range(HD_FT_MM_KEYS):
            _pin_bf16_weight(st, p + tails[ki], ctx, w_h, w_nbytes, w_shape)
        for ki in range(HD_FT_MM_KEYS, HD_FT_KEYS):
            _pin_bf16_1d_weight(st, p + tails[ki], ctx, w_h, w_nbytes, w_shape)
        layers.append(HiDreamO1LayerHostBf16(w_h^, w_nbytes^, w_shape^))
        cu_mempool_trim_current(0)
        if (li + 1) % 4 == 0 or li + 1 == num_layers:
            print("full-ft bf16 host store: pinned+converted layer", li + 1, "/", num_layers)

    # device slots (per-slot shapes uniform across layers — asserted below)
    var slot = List[TArc]()
    ref l0 = layers[0]
    for s in range(2):
        _ = s
        for i in range(HD_FT_KEYS):
            var dbuf = ctx.enqueue_create_buffer[DType.uint8](l0.w_nbytes[i])
            var t = Tensor(dbuf^, l0.w_shape[i].copy(), STDtype.BF16)
            slot.append(TArc(t^))
    ctx.synchronize()
    # layer-uniformity fail-loud (slot reuse depends on it)
    for li in range(num_layers):
        for i in range(HD_FT_KEYS):
            if layers[li].w_nbytes[i] != l0.w_nbytes[i]:
                raise Error("hidream full-FT store: layer weight shapes not uniform")

    var copy_stream = List[ArcPointer[DeviceStream]]()
    copy_stream.append(ArcPointer(ctx.create_stream()))
    var ev = List[ArcPointer[DeviceEvent]]()
    ev.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    ev.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    return HiDreamO1HostBf16(layers^, slot^, copy_stream^, ev^, num_layers)


# ── prefetch (async H2D on the copy stream into slot li%2, record event) ─────
def hidream_o1_host_bf16_prefetch(store: HiDreamO1HostBf16, li: Int) raises:
    if li < 0 or li >= len(store.layers):
        return
    var s = li % 2
    ref b = store.layers[li]
    for i in range(HD_FT_KEYS):
        _h2d_dma_copy(
            UInt64(Int(store.slot[s * HD_FT_KEYS + i][].buf.unsafe_ptr())),
            b.w_h[i][].unsafe_ptr(),
            b.w_nbytes[i],
            store.copy_stream[0][],
        )
    store.copy_stream[0][].record_event(store.ev[s][])


# alias view over a persistent device tensor: refcounted buffer copy, SAME
# device memory (the block.mojo `Tensor(t.buf.copy(), t.shape(), t.dtype())`
# idiom) — the optimizer's in-place slot mutation is visible through it.
def _alias_t(t: TArc) raises -> TArc:
    return TArc(Tensor(t[].buf.copy(), t[].shape(), t[].dtype()))


# ── slot loader (fence the H2D, wrap the PERSISTENT slot tensors) ─────────────
def _load_hidream_slot(
    store: HiDreamO1HostBf16, li: Int, ctx: DeviceContext
) raises -> HiDreamO1BlockWeights:
    """Fence slot li%2's H2D and build HiDreamO1BlockWeights ALIASING the SAME
    slot device buffers (buf.copy() = refcount, same memory) — the FT optimizer
    mutates these in place pre-writeback. ALL 11 per-layer params are trained
    (v2) and stream from the slots. Field order matches the train-block struct
    (hidream_o1_train_block.mojo:68-100); slot map is HD_FT_SLOT order:
    qw<-0, kw<-1, vw<-2, ow<-3, gw<-4, uw<-5, dw<-6, in_ln<-7, q_norm<-8,
    k_norm<-9, post_ln<-10."""
    var s = li % 2
    ctx.stream().enqueue_wait_for(store.ev[s][])
    var base = s * HD_FT_KEYS
    return HiDreamO1BlockWeights(
        _alias_t(store.slot[base + 7]),   # in_ln   (HD_FT_SLOT_IN_LN)
        _alias_t(store.slot[base + 0]),   # qw      (HD_FT_SLOT_Q)
        _alias_t(store.slot[base + 1]),   # kw      (HD_FT_SLOT_K)
        _alias_t(store.slot[base + 2]),   # vw      (HD_FT_SLOT_V)
        _alias_t(store.slot[base + 8]),   # q_norm  (HD_FT_SLOT_Q_NORM)
        _alias_t(store.slot[base + 9]),   # k_norm  (HD_FT_SLOT_K_NORM)
        _alias_t(store.slot[base + 3]),   # ow      (HD_FT_SLOT_O)
        _alias_t(store.slot[base + 10]),  # post_ln (HD_FT_SLOT_POST_LN)
        _alias_t(store.slot[base + 4]),   # gw      (HD_FT_SLOT_GATE)
        _alias_t(store.slot[base + 5]),   # uw      (HD_FT_SLOT_UP)
        _alias_t(store.slot[base + 6]),   # dw      (HD_FT_SLOT_DOWN)
    )


# ── write-back (D2H the optimizer-updated slot into the pinned host bytes) ───
def hidream_o1_host_bf16_writeback(
    store: HiDreamO1HostBf16, li: Int, ctx: DeviceContext
) raises:
    var s = li % 2
    ref b = store.layers[li]
    for i in range(HD_FT_KEYS):
        ctx.enqueue_copy(b.w_h[i][], store.slot[s * HD_FT_KEYS + i][].buf)


# ── host-direct safetensors overlay save (no GPU round-trip) ─────────────────
def hidream_o1_host_bf16_overlay_resume(
    store: HiDreamO1HostBf16, overlay_path: String
) raises:
    """RESUME store rebuild, step 2: the store already holds the F32→bf16
    converted BASE checkpoint (build_hidream_o1_host_bf16); copy the trained
    BF16 overlay's bytes over the pinned host bytes (byte-exact, BF16/size
    fail-loud). Key order == hidream_o1_host_bf16_save's (li-major, the 11
    tails); a v1 (252-tensor, matmuls-only) overlay FAILS LOUD on the count
    check."""
    var tails = _hd_ft_key_tails()
    var names = List[String]()
    var bufs = List[HArc]()
    var nbytes = List[Int]()
    for li in range(len(store.layers)):
        ref b = store.layers[li]
        for wi in range(HD_FT_KEYS):
            names.append(String(_HD_LAYER_PREFIX) + String(li) + String(".") + tails[wi])
            bufs.append(b.w_h[wi].copy())
            nbytes.append(b.w_nbytes[wi])
    full_ft_overlay_into_host_store(overlay_path, names, bufs, nbytes)


def hidream_o1_ft_state_shapes(
    store: HiDreamO1HostBf16, mut rows: List[Int], mut cols: List[Int]
) raises:
    """FLAT per-state [rows, cols] in the af_states order (build_hidream_o1_ft_
    adafactor_states) — the sidecar loader's expected-shape lists. 1-D params
    (slots 7-10) report cols == 0: the UNFACTORED sentinel the sidecar v2
    loader dispatches on."""
    for _li in range(store.num_layers):
        for wi in range(HD_FT_KEYS):
            ref sh = store.layers[0].w_shape[wi]
            rows.append(sh[0])
            if len(sh) == 2:
                cols.append(sh[1])
            else:
                cols.append(0)


def hidream_o1_host_bf16_save(store: HiDreamO1HostBf16, path: String) raises:
    """Write the trained surface (the pinned-host bf16 bytes — the live model)
    DIRECTLY to a safetensors file. Keys keep the ORIGINAL checkpoint names
    (model.language_model.layers.N.<tail>); dtype is BF16 (converted-from-F32
    trained weights — an overlay: load the F32 base, then substitute these
    trained tensors). v2 saves the FULL per-layer surface (11 tensors x 36
    layers = 396: 7 mats + 4 rms scales)."""
    var tails = _hd_ft_key_tails()
    var header = String("{")
    var off = 0
    var first = True
    for li in range(len(store.layers)):
        ref b = store.layers[li]
        for wi in range(HD_FT_KEYS):
            var nm = String(_HD_LAYER_PREFIX) + String(li) + String(".") + tails[wi]
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
        raise Error(String("hidream_o1_host_bf16_save: cannot open ") + path)
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
        for wi in range(HD_FT_KEYS):
            var src = BytePtr(unsafe_from_address=Int(b.w_h[wi][].unsafe_ptr()))
            var wrote = sys_pwrite(fd, src, b.w_nbytes[wi], base + woff)
            if wrote != b.w_nbytes[wi]:
                _ = sys_close(fd)
                raise Error("hidream_o1_host_bf16_save: short write")
            woff += b.w_nbytes[wi]
    _ = sys_close(fd)
    print(
        "[hidream-ft-save] wrote",
        len(store.layers) * HD_FT_KEYS,
        "trained weights ->", path,
    )


# ── FT forward tape (per-layer input snapshots for the backward recompute) ───
struct HiDreamO1StackFTForward(Movable):
    var out: TArc                # [1,S,D] bf16 — stack output (pre final-norm)
    var x_inputs: List[TArc]     # num_layers x [1,S,D] recompute checkpoints

    def __init__(out self, var out: TArc, var x_inputs: List[TArc]):
        self.out = out^
        self.x_inputs = x_inputs^


# ── phase (b) forward: streamed base-block stack forward from the host store ─
def hidream_o1_stack_ft_forward_streamed[
    S: Int, H: Int, HKV: Int, Dh: Int
](
    x_in: TArc,            # [1, S, D] bf16 (embed output — the P1-gate shape)
    cos_q: Tensor, sin_q: Tensor,   # [S*H*(Dh/2)] F32 per-head tables
    cos_k: Tensor, sin_k: Tensor,   # [S*HKV*(Dh/2)]
    mask4: Tensor,                   # [1,H,S,S] additive (block dtype)
    store: HiDreamO1HostBf16,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> HiDreamO1StackFTForward:
    """Mirror of the LoRA trainer's recompute-checkpoint forward loop
    (train_hidream_o1_real.mojo:949-962) with the LIVE bf16 host store as the
    weight source and ALL-None adapters (= base forward, the P1-gate pairing).
    Saves per-layer x inputs (backward recompute checkpoints); per-block
    ctx.synchronize() (slot-reuse + async-free discipline)."""
    hidream_o1_host_bf16_prefetch(store, 0)
    var lora = HiDreamO1BlockLora()   # all-None = base block bit-for-bit

    var x = x_in.copy()
    var saved = List[TArc]()
    for li in range(store.num_layers):
        saved.append(x.copy())
        var w = _load_hidream_slot(store, li, ctx)
        hidream_o1_host_bf16_prefetch(store, li + 1)
        var f = hidream_o1_block_lora_forward[S, H, HKV, Dh](
            x, w, lora, cos_q, sin_q, cos_k, sin_k, mask4, D, F, eps, ctx
        )
        x = f.out.copy()
        # f.saved drops here — recompute rebuilds it in the bwd loop.
        ctx.synchronize()   # slot li%2 reuse safety + bound deferred frees
    return HiDreamO1StackFTForward(x^, saved^)


# ── phase (b) backward: streamed FT stack backward + fused device Adafactor ──
struct HiDreamO1StackFTWrite(Movable):
    var d_x_in: TArc             # [1,S,D] grad into the (frozen) embeds
    var grad_count: Int
    var streaming_sync_count: Int

    def __init__(
        out self, var d_x_in: TArc,
        grad_count: Int, streaming_sync_count: Int,
    ):
        self.d_x_in = d_x_in^
        self.grad_count = grad_count
        self.streaming_sync_count = streaming_sync_count


def hidream_o1_stack_ft_backward_streamed[
    S: Int, H: Int, HKV: Int, Dh: Int
](
    d_out: Tensor,               # [1,S,D] bf16 dL/d(stack out)
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    mask4: Tensor,               # [1,H,S,S] block dtype (recompute forward)
    mask_f32: Tensor,            # [H*S,S] F32 (sdpa_backward_masked contract)
    store: HiDreamO1HostBf16,
    saved: HiDreamO1StackFTForward,
    mut af_states: List[AdafactorDeviceState],   # flat li*11+wi (module header)
    t_step: Int,                 # 1-based optimizer step
    lr: Float64, beta2_decay: Float64, eps2: Float64, d_thresh: Float64,
    weight_decay: Float64,
    sr_seed: UInt64,             # per-step seed; 0 = RNE writes
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> HiDreamO1StackFTWrite:
    """DESCENDING walk (layer 35 -> 0): load slot → recompute the block forward
    from the saved input (all-None adapters = base tape) → P1 FT backward →
    per-param fused device Adafactor (2D factored / 1D unfactored dispatch)
    mutating the slot in place (SR bf16 write) → D2H write-back into the host
    store → carry d_hidden. Fused-back-pass shape: grads and updated weights
    never accumulate beyond one layer."""
    if len(af_states) != store.num_layers * HD_FT_KEYS:
        raise Error("hidream_o1_stack_ft_backward_streamed: af_states length mismatch")

    hidream_o1_host_bf16_prefetch(store, store.num_layers - 1)
    var lora = HiDreamO1BlockLora()

    var d_x = TArc(d_out.clone(ctx))
    var grad_count = 0
    var streaming_sync_count = 0

    var li = store.num_layers - 1
    while li >= 0:
        var w = _load_hidream_slot(store, li, ctx)
        hidream_o1_host_bf16_prefetch(store, li - 1)
        var rfwd = hidream_o1_block_lora_forward[S, H, HKV, Dh](
            saved.x_inputs[li], w, lora, cos_q, sin_q, cos_k, sin_k,
            mask4, D, F, eps, ctx
        )
        var bg = hidream_o1_block_ft_backward_dev[S, H, HKV, Dh](
            d_x[], w, rfwd.saved, cos_q, sin_q, cos_k, sin_k, mask_f32,
            D, F, eps, ctx
        )
        # fused back pass: update THIS layer's 11 slot params in place —
        # FACTORED (2D matmuls, slots 0-6) via adafactor_step_device,
        # UNFACTORED (1D rms scales, slots 7-10) via adafactor_step_device_1d
        # (torch trains <2D params with the unfactored second moment).
        var s = li % 2
        for wi in range(HD_FT_KEYS):
            var flat = li * HD_FT_KEYS + wi
            if af_states[flat].factored:
                adafactor_step_device(
                    store.slot[s * HD_FT_KEYS + wi][],
                    bg.dw[wi][],
                    af_states[flat],
                    t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                    weight_decay,
                    sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)),
                    ctx,
                )
            else:
                adafactor_step_device_1d(
                    store.slot[s * HD_FT_KEYS + wi][],
                    bg.dw[wi][],
                    af_states[flat],
                    t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                    weight_decay,
                    sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)),
                    ctx,
                )
        hidream_o1_host_bf16_writeback(store, li, ctx)
        grad_count += HD_FT_KEYS
        d_x = bg.d_hidden.copy()
        ctx.synchronize()   # lands: grads freed, update done, write-back D2H'd
        streaming_sync_count += 1
        li -= 1

    return HiDreamO1StackFTWrite(d_x^, grad_count, streaming_sync_count)


# ── Adafactor state builder (flat li*11+wi, device-resident ~few MB):
# FACTORED [rows]+[cols] for the 2D matmuls (slots 0-6), UNFACTORED
# exp_avg_sq [n] (cols==0 sentinel) for the 1D rms scales (slots 7-10). ──────
def build_hidream_o1_ft_adafactor_states(
    store: HiDreamO1HostBf16, ctx: DeviceContext
) raises -> List[AdafactorDeviceState]:
    var states = List[AdafactorDeviceState]()
    for _li in range(store.num_layers):
        for wi in range(HD_FT_KEYS):
            ref sh = store.layers[0].w_shape[wi]
            if len(sh) == 2:
                states.append(AdafactorDeviceState(sh[0], sh[1], ctx))
            else:
                states.append(AdafactorDeviceState(sh[0], ctx))   # unfactored 1D
    return states^
