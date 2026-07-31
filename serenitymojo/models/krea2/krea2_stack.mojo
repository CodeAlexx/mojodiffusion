# models/krea2/krea2_stack.mojo — Krea-2-Raw SINGLE-STREAM STACK *WITH LoRA*.
#
# Phase-2 of the krea2 LoRA-training port: the training FORWARD over the N
# single-stream blocks (saving each block's input for recompute) + the final-layer
# (`last`) forward, plus the LoRA STACK BACKWARD (final-layer bwd → single-stream
# bwd ×N → STOP). This file COMPOSES the Phase-1 unit
# (models/krea2/krea2_block.mojo: krea2_single_stream_block_lora /
# krea2_single_stream_block_lora_backward) — it rebuilds NO block math. It mirrors
# the ideogram4 stack template (serenitymojo/models/ideogram4/block.mojo:
# ideogram4_stack_lora_forward/backward) loop-for-loop.
#
# ── SCOPE (current Mojo LoRA backward path) ───────────────────────────────────
# Krea2 forward: first(img) → text-fusion (12-layer ctx) → single-stream ×N →
# last. This stack currently wires LoRA only on the N main single-stream blocks.
# So the LoRA backward implemented here is:
#   d_velocity → final-layer bwd (FROZEN, d_x only) → single-stream block bwd ×N
#   (Phase-1's backward: LoRA dA/dB + d_x carry) → STOP.
# The text-fusion blocks + first/embedders are BEFORE the single-stream blocks →
# frozen-skip in this file. ai-toolkit output also contains text-fusion LoRA
# under `diffusion_model.txtfusion.*`; that is a known product-parity blocker,
# not an architectural exclusion.
#
# The historical per-block gates used F32 oracle tensors, but the product trainer
# preserves BF16/F16/FP8 storage boundaries. F32 is limited to compute internals,
# reductions, optimizer grads/moments, scalar stats, and oracle/debug artifacts.
#
# ── full-recompute discipline (ideogram4/klein pattern) ──────────────────────
# The forward saves ONLY each block's INPUT (a [1,L,features] clone), not its
# internal acts. The backward, deepest→shallowest, RE-RUNS the Phase-1 block
# forward from the saved input to regenerate Krea2BlockSaved, then runs the
# Phase-1 block backward. This keeps peak memory at one block's acts (the 28-block
# real model fits) at the cost of one extra block forward per block in backward.
#
# Mojo 1.0.0b1: `def` only; Tensor move-only (never in a collection); ArcPointer
# [Tensor] is the Copyable device carrier; no-bias linear = linear(x,w,None,ctx).

from std.gpu.host import (
    DeviceContext, HostBuffer, DeviceBuffer, DeviceStream, DeviceEvent,
)
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer, alloc
from serenitymojo.io.ffi import (
    BytePtr, sys_open, sys_close, sys_pwrite, O_WRONLY, O_CREAT, O_TRUNC,
)
# Copy-stream H2D DMA (cuMemcpyHtoDAsync on an explicit stream): the int8
# host-offload prefetch overlaps next-block H2D with current-block compute
# (the proven turbo_loader double-buffer primitive).
from serenitymojo.offload.turbo_loader import _h2d_dma_copy
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
# T2.B fp8-quantized-resident base: encode side (bf16 ckpt → E4M3 bytes + per-row
# F32 scale, ONCE at load) + decode side (E4M3 → bf16 per block right before
# compute). Same per-output-row layout the ideogram4 resident base uses
# (models/ideogram4/block.mojo::_resident_fp8_weight).
from serenitymojo.ops.fp8_quant import fp8_e4m3_rowscale, fp8_e4m3_encode_perrow
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
from serenitymojo.ops.cast import cast_tensor
# int8 W8A8-resident base (cfg.quantized_resident=="int_w8a8"): encode side only —
# the block does int8 GEMM directly (NO per-step dequant, unlike fp8).
from serenitymojo.ops.int8_quant import (
    int8_tensorwise_scale, int8_encode_tensorwise,
)
# squareq W4-resident base (cfg.quantized_resident=="squareq_w4"): per-block bf16
# reconstruction W_hat = dequant4@H_bd + lora_up@lora_down^T from the prebuilt sidecar.
from serenitymojo.ops.squareq import squareq_reconstruct_weight
# Reclaim freed transient bf16 sources back to the OS between quantized weights so
# the load-once int8 quantize can hold all 28 blocks resident without the mempool
# retaining garbage (cuMemPoolTrimTo at an allocation boundary — after a sync).
from serenitymojo.offload.vmm_cuda import cu_mempool_trim_current

comptime TArc = ArcPointer[Tensor]

# ── reused forward ops (final layer = krea2_dit's last; tiling = krea2_dit) ────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.tensor_algebra import (
    reshape, slice, concat, add, zeros_device,
)
from serenitymojo.models.dit.krea2_dit import (
    krea2_last_layer, krea2_simple_modulation, _reshape_chunk_to_vec,
    _tile_rope_table, krea2_rmsnorm, krea2_rmsnorm_backward_dx,
    Krea2BlockResidentFp8, Krea2ResidentFp8,   # resident-fp8 structs moved here (no cycle)
    Krea2BlockResidentSquareq, Krea2ResidentSquareq, # resident-squareq structs (same reason)
    Krea2BlockResidentInt8, Krea2ResidentInt8, # resident-int8 structs (same reason)
    Krea2HostInt8Inf, _krea2_host_i8_block_dev,
)

# ── reused backward arms (final-layer chain; all pre-built + gated elsewhere) ──
from serenitymojo.ops.linalg_backward import linear_backward_dx, linear_backward_dw
from serenitymojo.ops.elementwise_backward import modulate_backward

# ── Phase-1 block unit (the composed primitive) ───────────────────────────────
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockInt8, Krea2BlockLora, Krea2LoraGrad, Krea2BlockGrads,
    Krea2BlockSaved,
    Krea2LoraGradT, Krea2BlockGradsT,
    Krea2DirectDoRAGradT, Krea2DirectOFTGradT,
    Krea2BlockDirectDoRAGradsT, Krea2BlockDirectOFTGradsT,
    krea2_single_stream_block_lora, krea2_single_stream_block_lora_backward,
    krea2_single_stream_block_lora_backward_dev,
    Krea2BlockFTGrads, krea2_single_stream_block_ft_backward_dev,
    KREA2_FT_SLOTS_V2,
    krea2_single_stream_block_lora_b2, krea2_single_stream_block_lora_backward_b2,
    krea2_single_stream_block_lora_backward_b2_dev,
    krea2_single_stream_block_dora, krea2_single_stream_block_dora_backward_dev,
    krea2_single_stream_block_oft, krea2_single_stream_block_oft_backward_dev,
)
from serenitymojo.models.krea2.krea2_direct_lycoris_stack import (
    Krea2StackDirectDoRA, Krea2StackDirectOFT,
)

# ── Phase 4b autograd_v2 engine arm (the COARSE per-block graph backward) ──────
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.autograd_v2.krea2_block_graph import (
    krea2_single_stream_block_graph_backward,
    krea2_single_stream_block_graph_backward_seg,
    krea2_single_stream_block_graph_backward_slab,
)
from serenitymojo.autograd_v2.step_slab import StepSlab
from serenitymojo.models.krea2.krea2_block import KREA2_SLAB_SEGMENTED
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState,
    lora_adamw_plain_device_state_copy_device_grad_pair,
)
from serenitymojo.training.automagic3_device import (
    Automagic3DeviceState,
    automagic3_device_state_copy_device_grad_pair,
)
# FULL-FT (FULL_FINETUNE_KREA2_PLAN P3): per-block fused device Adafactor on
# the streamed bf16 slots + SR write-back + D2H to the pinned host store.
from serenitymojo.training.adafactor_device import (
    AdafactorDeviceState, adafactor_step_device, adafactor_step_device_1d,
)
# FULL-FT resume (the fleet sidecar): overlay the trained weights over the
# base-built pinned host store (byte-exact, dtype-checked).
from serenitymojo.training.full_ft_sidecar import full_ft_overlay_into_host_store


# ══════════════════════════════════════════════════════════════════════════════
# CARRIERS
# ══════════════════════════════════════════════════════════════════════════════

# Per-block frozen weights + per-block LoRA, flat-indexed by block. Krea2BlockWeights
# / Krea2BlockLora are Copyable, so a List of them is fine (they hold TArc, not bare
# Tensor). N blocks of each.
struct Krea2StackWeights(Copyable, Movable):
    var blocks: List[Krea2BlockWeights]   # len == N
    # final-layer (last) frozen params.
    var last_norm: TArc        # [features] raw checkpoint dtype (last.norm.scale)
    var last_mod_lin: TArc     # [2, features]   (last.modulation.lin)
    var last_lin_w: TArc       # [out_ch, features]  ([64, 6144])
    var last_lin_b: TArc       # [out_ch]            ([64])

    def __init__(
        out self, var blocks: List[Krea2BlockWeights],
        var last_norm: TArc, var last_mod_lin: TArc,
        var last_lin_w: TArc, var last_lin_b: TArc,
    ):
        self.blocks = blocks^
        self.last_norm = last_norm^
        self.last_mod_lin = last_mod_lin^
        self.last_lin_w = last_lin_w^
        self.last_lin_b = last_lin_b^


struct Krea2StackLora(Copyable, Movable):
    var blocks: List[Krea2BlockLora]      # len == N (8 adapters each)

    def __init__(out self, var blocks: List[Krea2BlockLora]):
        self.blocks = blocks^


struct Krea2StackForward(Movable):
    var velocity: TArc                    # [1, imglen, out_ch] device-resident
    var block_inputs: List[TArc]          # len N: each block's [1,L,features] input
    # final-layer acts needed for its backward (cheap, kept; NOT recomputed).
    var x_blocks_out: TArc                # [1,L,features] last single-stream output
    var last_xn: TArc                     # [1,L,features] krea2_rmsnorm(x_blocks_out)
    var txtlen: Int                       # slice offset for the image tokens
    var imglen: Int
    # SAVED-TAPE hybrid (KREA2_SAVE_TAPES): the forward keeps the first
    # len(tapes) blocks' SLIMMED Krea2BlockSaved so the backward skips their
    # recompute-forward. Empty = the historical full-recompute discipline.
    var tapes: List[Krea2BlockSaved]

    def __init__(
        out self, var velocity: TArc, var block_inputs: List[TArc],
        var x_blocks_out: TArc, var last_xn: TArc, txtlen: Int, imglen: Int,
    ):
        self.velocity = velocity^
        self.block_inputs = block_inputs^
        self.x_blocks_out = x_blocks_out^
        self.last_xn = last_xn^
        self.txtlen = txtlen
        self.imglen = imglen
        self.tapes = List[Krea2BlockSaved]()


# Flat LoRA grads, parallel to Krea2StackLora.blocks: block bi slot s at bi*8+s.
# Slot order matches Krea2BlockLora field order: wq wk wv gate wo mlp_gate mlp_up mlp_down.
comptime KREA2_SLOTS_PER_BLOCK = 8

struct Krea2StackLoraGrads(Movable):
    var grads: List[Krea2LoraGrad]        # len N*8, flat (bi*8 + slot)
    var d_combined: TArc                  # [1,L,features] grad into the block-stack input
    # OPT-IN img-EDIT reference branch grad (models/krea2/krea2_img_in_ref_param.mojo,
    # ported from klein). The ONLY new trained param's grad, [D=features, in_ch].
    # None when no ref supplied (text-to-image path) — the C13 flags-off contract
    # keeps the whole pass byte-identical. NEW trailing defaulted field so every
    # existing 2-arg construction site is unchanged.
    var d_img_in_ref: Optional[TArc]      # [features, in_ch] (None when no ref)

    def __init__(
        out self, var grads: List[Krea2LoraGrad], var d_combined: TArc,
        var d_img_in_ref: Optional[TArc] = Optional[TArc](None),
    ):
        self.grads = grads^
        self.d_combined = d_combined^
        self.d_img_in_ref = d_img_in_ref^


# DEVICE-grad sibling: flat LoRA dA/dB stay on DEVICE as Krea2LoraGradT. The
# streamed_dev path copies one block's 8 pairs to host under the existing
# per-block streaming fence; the AdamW-preload path copies those pairs D2D into
# shared optimizer state instead. Same flat layout (len N*8, bi*8 + slot).
struct Krea2StackLoraGradsT(Movable):
    var grads: List[Krea2LoraGradT]       # len N*8, flat (bi*8 + slot), DEVICE
    var d_combined: TArc                  # [1,L,features] grad into the block-stack input

    def __init__(out self, var grads: List[Krea2LoraGradT], var d_combined: TArc):
        self.grads = grads^
        self.d_combined = d_combined^


struct Krea2StackDeviceGradWrite(Movable):
    var d_combined: TArc
    var grad_count: Int
    var streaming_sync_count: Int
    # OPT-IN img-EDIT reference branch grad (models/krea2/krea2_img_in_ref_param.mojo).
    # [features, in_ch]; None (default) on the text-to-image path — NEW trailing
    # defaulted field so every existing 3-arg construction site is unchanged (C13).
    var d_img_in_ref: Optional[TArc]

    def __init__(
        out self, var d_combined: TArc, grad_count: Int, streaming_sync_count: Int,
        var d_img_in_ref: Optional[TArc] = Optional[TArc](None),
    ):
        self.d_combined = d_combined^
        self.grad_count = grad_count
        self.streaming_sync_count = streaming_sync_count
        self.d_img_in_ref = d_img_in_ref^


struct Krea2GradCopyKeepalive(Movable):
    var grads: List[TArc]

    def __init__(out self, var grads: List[TArc]):
        self.grads = grads^


struct Krea2StackDirectDoRAGradsT(Movable):
    var grads: List[Krea2DirectDoRAGradT]  # len N*8, flat (bi*8 + slot), DEVICE
    var d_combined: TArc

    def __init__(
        out self, var grads: List[Krea2DirectDoRAGradT], var d_combined: TArc,
    ):
        self.grads = grads^
        self.d_combined = d_combined^


struct Krea2StackDirectOFTGradsT(Movable):
    var grads: List[Krea2DirectOFTGradT]  # len N*8, flat (bi*8 + slot), DEVICE
    var d_combined: TArc

    def __init__(
        out self, var grads: List[Krea2DirectOFTGradT], var d_combined: TArc,
    ):
        self.grads = grads^
        self.d_combined = d_combined^


# ══════════════════════════════════════════════════════════════════════════════
# OPT-IN img-EDIT reference-conditioning projection (PORT of klein's img_in_ref).
#
# krea2's `first` Linear ([D=features, in_ch]) is applied UPSTREAM of this stack —
# the block-stack input `combined` already carries first(img) in its image rows
# (oracle: combined = cat(txtfusion(ctx), first(img))). So the additive reference
# projection lands at the TOP of the stack forward, on the image rows of combined:
#     combined_img_rows += linear(ref_tokens, img_in_ref_w)
# Byte-identical to klein_stack's img_in_ref hook (same _linear + scatter + add);
# when img_in_ref_w == 0 (zero-init) the added term is exactly 0 -> the C13 flags-
# off contract. When ref_tokens/img_in_ref_w are absent the whole hook is SKIPPED.
#
# ref_tokens : [1, imglen, in_ch]  the reference image, VAE-encoded + packed (SAME
#                                   layout as the image tokens fed to `first`)
# img_in_ref_w : [features, in_ch] the NEW zero-init TRAINABLE projection
def _krea2_apply_img_in_ref_fwd(
    x: TArc, ref_tokens: TArc, img_in_ref_w: TArc,
    txtlen: Int, imglen: Int, features: Int, ctx: DeviceContext,
) raises -> TArc:
    var xdt = x[].dtype()
    var L = x[].shape()[1]
    # match the block-stack dtype (F32 in the parity gate; bf16 in the live path).
    var ref_x = cast_tensor(ref_tokens[], xdt, ctx)        # [1, imglen, in_ch]
    var w = cast_tensor(img_in_ref_w[], xdt, ctx)          # [features, in_ch]
    var ref_proj = linear(ref_x, w, None, ctx)             # [1, imglen, features]
    # scatter ref_proj into a full [1, L, features] (zeros on the txt head + tail),
    # mirroring krea2_final_layer_backward's un-slice.
    var head = txtlen
    var tail = L - txtlen - imglen
    var scatter: Tensor
    if head > 0 and tail > 0:
        var zh = zeros_device([1, head, features], xdt, ctx)
        var zt = zeros_device([1, tail, features], xdt, ctx)
        var part = concat(1, ctx, zh, ref_proj)
        scatter = concat(1, ctx, part, zt)
    elif head > 0:
        var zh = zeros_device([1, head, features], xdt, ctx)
        scatter = concat(1, ctx, zh, ref_proj)
    elif tail > 0:
        var zt = zeros_device([1, tail, features], xdt, ctx)
        scatter = concat(1, ctx, ref_proj, zt)
    else:
        scatter = ref_proj.clone(ctx)
    return TArc(add(x[], scatter, ctx))


# OPT-IN reference-branch backward — SHARED by the primary + every streamed backward.
# d_x IS dL/d(block-stack input) [1,L,features]; its image rows [txtlen:txtlen+imglen]
# are dL/d(first(img)) — the SAME `d_img_act` klein's img_in_ref_backward consumes.
# d_img_in_ref = d_img_actᵀ @ ref_tokens (a plain linear_backward_dw). None when no
# ref supplied. Uses the SAME slice + cast + linear_backward_dw as the primary path,
# so the emitted grad is bit-identical for a bit-identical d_x.
def _krea2_emit_img_in_ref_grad(
    d_x: Tensor, ref_tokens: Optional[TArc], img_in_ref_w: Optional[TArc],
    txtlen: Int, imglen: Int, features: Int, ctx: DeviceContext,
) raises -> Optional[TArc]:
    if not (Bool(ref_tokens) and Bool(img_in_ref_w)):
        return Optional[TArc](None)
    var in_ch = img_in_ref_w.value()[].shape()[1]
    var d_img = slice(d_x, 1, txtlen, imglen, ctx)                  # [1,imglen,features]
    var reft = cast_tensor(ref_tokens.value()[], d_img.dtype(), ctx)  # [1,imglen,in_ch]
    var dw = linear_backward_dw(
        d_img, reft, imglen, in_ch, features, ctx, STDtype.F32,
    )   # [features, in_ch]
    return Optional[TArc](TArc(dw^))


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD — N LoRA blocks (saving inputs) + final layer
# ══════════════════════════════════════════════════════════════════════════════
def krea2_stack_lora_forward[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    combined: TArc,            # [1, L, features]  block-stack input (post txtmlp cat)
    blk_vec: Tensor,           # [1, 6*features]   block modulation vec (tproj(t))
    tmlp_out: Tensor,          # [1, 1, features]  final-layer tvec (tmlp(temb))
    w: Krea2StackWeights, lora: Krea2StackLora,
    cos: Tensor, sin: Tensor,  # [L, HEADDIM/2] per-token RoPE table (untiled)
    eps: Float32,
    txtlen: Int, imglen: Int,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),  # one value per sample, shared
        # across all N blocks. None (or == L) = full attn (no pad). Present & < L =
        # length-bucket flash-padmask SDPA (see krea2_block); the valid contiguous
        # prefix is [0:real_len], [real_len:L] is pad.
    # OPT-IN img-EDIT reference conditioning (PORT of klein's img_in_ref). When BOTH
    # ref_tokens [1,imglen,in_ch] and img_in_ref_w [features,in_ch] are supplied, the
    # block-stack input image rows become first(img) + linear(ref, img_in_ref) — the
    # SAME summed input projection klein_stack computes. Absent (or img_in_ref_w==0)
    # => the EXACT existing text-to-image path, byte-for-byte (C13 flags-off).
    ref_tokens: Optional[TArc] = Optional[TArc](None),
    img_in_ref_w: Optional[TArc] = Optional[TArc](None),
) raises -> Krea2StackForward:
    """Single-stream STACK forward WITH LoRA, saving per-block inputs for recompute.

    Composes the verified Phase-1 block forward over N blocks (N = len(w.blocks)),
    then krea2_last_layer. cos/sin are the per-token rope table (built once by the
    caller from pos); tiled here to q (HEADS) and k (KVHEADS). real_len (optional):
    None = the gate's unpadded sequence (L == LFULL → block SDPA is the no-mask
    path); present = the length-bucket flash-padmask threaded into every block's
    SDPA. Returns velocity sliced to the image tokens [1, imglen, out_ch]."""
    comptime features = HEADS * HEADDIM
    var n = len(w.blocks)

    # tile the per-token rope table once for q/k (Phase-1 block contract).
    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var x = combined.copy()
    # OPT-IN additive reference projection on the image rows (img-EDIT). Skipped
    # entirely (byte-identical text-to-image path) when no ref is supplied. The
    # block_inputs saved below therefore carry first(img)+linear(ref,img_in_ref).
    if Bool(ref_tokens) and Bool(img_in_ref_w):
        x = _krea2_apply_img_in_ref_fwd(
            x, ref_tokens.value(), img_in_ref_w.value(),
            txtlen, imglen, features, ctx,
        )
    var block_inputs = List[TArc]()
    for bi in range(n):
        block_inputs.append(x.copy())                          # SAVE the block input
        var fwd = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            x, blk_vec, w.blocks[bi], lora.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        x = fwd.out.copy()

    var x_blocks_out = x.copy()                                 # last single-stream output

    # final = last_layer(x, tmlp_out): krea2_rmsnorm applies (scale.float()+1)
    # inside the op and returns x dtype; modulate + linear keep product boundaries.
    # We need the normed x saved for the final-layer backward (modulate_backward
    # needs the pre-modulate normed acts).
    var last_xn = krea2_rmsnorm(x[], w.last_norm[], eps, ctx)  # [1,L,features]
    var final = krea2_last_layer(
        x[], tmlp_out, w.last_norm[], w.last_mod_lin[],
        w.last_lin_w[], w.last_lin_b[], features, ctx,
    )                                                          # [1, L, out_ch]

    # velocity = final[:, txtlen : txtlen+imglen, :]  (the image tokens).
    var velocity = slice(final, 1, txtlen, imglen, ctx)        # [1, imglen, out_ch]

    return Krea2StackForward(
        TArc(velocity^), block_inputs^,
        x_blocks_out^, TArc(last_xn^), txtlen, imglen,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FINAL-LAYER BACKWARD (frozen `last`, d_x only) — the new frozen arm
# ══════════════════════════════════════════════════════════════════════════════
def krea2_final_layer_backward[
    L: Int, HEADS: Int, HEADDIM: Int
](
    d_velocity: Tensor,        # [1, imglen, out_ch] upstream grad on the image tokens
    fwd: Krea2StackForward,    # carries x_blocks_out, last_xn, txtlen, imglen
    tmlp_out: Tensor,          # [1,1,features] final-layer tvec (for the mod chunks)
    w: Krea2StackWeights,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Backward of krea2_last_layer (FROZEN — base weights not trained; we want only
    d_x into the single-stream stack output). Exact reverse of:
        scale,shift = SimpleModulation(tmlp_out, last.modulation.lin)
        xn          = krea2_rmsnorm(x, last.norm.scale)
        xm          = (1+scale)*xn + shift
        velocity    = Linear(xm)[:, txtlen:txtlen+imglen]

    Steps: scatter d_velocity into a full [1,L,out_ch] (zeros on txt+pad rows) →
    linear_backward_dx → modulate_backward (drop param grads) → krea2_rmsnorm_backward_dx."""
    comptime features = HEADS * HEADDIM
    var out_ch = w.last_lin_w[].shape()[0]
    var L_full = fwd.x_blocks_out[].shape()[1]

    # 1) un-slice d_velocity → d_final [1, L, out_ch]: image rows [txtlen:txtlen+imglen]
    # get d_velocity, the txt rows [0:txtlen] and tail rows [txtlen+imglen:L] are zero
    # (only image tokens feed the velocity loss). Build via concat of the zero pads.
    var head = fwd.txtlen
    var tail = L_full - fwd.txtlen - fwd.imglen
    var d_final2: Tensor
    if head > 0 and tail > 0:
        var zh = zeros_device([1, head, out_ch], d_velocity.dtype(), ctx)
        var zt = zeros_device([1, tail, out_ch], d_velocity.dtype(), ctx)
        var part = concat(1, ctx, zh, d_velocity)
        d_final2 = concat(1, ctx, part, zt)
    elif head > 0:
        var zh = zeros_device([1, head, out_ch], d_velocity.dtype(), ctx)
        d_final2 = concat(1, ctx, zh, d_velocity)
    elif tail > 0:
        var zt = zeros_device([1, tail, out_ch], d_velocity.dtype(), ctx)
        d_final2 = concat(1, ctx, d_velocity, zt)
    else:
        d_final2 = d_velocity.clone(ctx)

    # 2) velocity = Linear(xm): d_xm = linear_backward_dx (base weight frozen).
    var M = L_full
    var d_xm = linear_backward_dx(d_final2, w.last_lin_w[], M, features, out_ch, ctx)  # [1*L, features]
    var d_xm3 = reshape(d_xm, [1, L_full, features], ctx)

    # 3) xm = (1+scale)*xn + shift → d_xn via modulate_backward (drop param grads).
    # scale = SimpleModulation(tmlp_out).scale, reshaped [features].
    var mods = krea2_simple_modulation(tmlp_out, w.last_mod_lin[], ctx)  # (scale,shift) [1,1,features]
    var scale = _reshape_chunk_to_vec(mods[0], features, ctx)            # [features]
    # Keep modulate_backward dtype-consistent with the saved normed activation.
    var mb = modulate_backward(cast_tensor(d_xm3, fwd.last_xn[].dtype(), ctx), fwd.last_xn[], cast_tensor(scale, fwd.last_xn[].dtype(), ctx), ctx, compute_param_grads=False)

    # 4) xn = krea2_rmsnorm(x, last.norm.scale) (FROZEN weight) → d_x.
    # krea2_rmsnorm_backward_dx casts the raw scale to F32 inside the op for
    # scale.float()+1 math and returns d_x in x dtype.
    var rb_dx = krea2_rmsnorm_backward_dx(
        mb.d_x, fwd.x_blocks_out[], w.last_norm[], eps, ctx,
    )
    var d_x_out = rb_dx.clone(ctx)
    return d_x_out^


# ══════════════════════════════════════════════════════════════════════════════
# STACK LoRA BACKWARD — final-layer bwd → single-stream bwd ×N (deepest→shallowest)
# ══════════════════════════════════════════════════════════════════════════════
def krea2_stack_lora_backward[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_velocity: Tensor,        # [1, imglen, out_ch] upstream grad on the image tokens
    blk_vec: Tensor,           # [1, 6*features]  block modulation vec
    tmlp_out: Tensor,          # [1, 1, features] final-layer tvec
    w: Krea2StackWeights, lora: Krea2StackLora,
    fwd: Krea2StackForward,
    cos: Tensor, sin: Tensor,  # [L, HEADDIM/2] per-token RoPE table (untiled)
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),  # MUST match the forward call —
        # threaded into BOTH the recompute forward and the block backward.
    # OPT-IN img-EDIT reference conditioning (see krea2_stack_lora_forward). When
    # supplied, d_img_in_ref [features,in_ch] is emitted: a plain linear_backward_dw
    # on the ref branch with grad_y = d_combined's image rows (= dL/d(first(img))).
    # Absent => d_img_in_ref stays None and the pass is byte-identical.
    ref_tokens: Optional[TArc] = Optional[TArc](None),
    img_in_ref_w: Optional[TArc] = Optional[TArc](None),
) raises -> Krea2StackLoraGrads:
    """LoRA stack backward. Runs the final-layer backward (frozen) to get d into the
    last block's output, then walks the N single-stream blocks deepest→shallowest:
    for each, RE-RUN the Phase-1 block forward from the saved input (regenerate its
    acts), run the Phase-1 block backward (LoRA dA/dB + d_x carry), scatter the 8
    dA/dB at bi*8+slot, carry d_x to the shallower block. Returns the flat LoRA grads
    + d_combined (grad into the block-stack input — load-bearing, proves the chain).
    real_len (optional): same length-bucket prefix the forward used; threaded into
    the recompute forward (so its acts match) AND the flash-padmask block backward."""
    comptime features = HEADS * HEADDIM
    var n = len(w.blocks)

    # tile rope once (same as forward).
    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    # pre-fill the flat grad list (one entry per block-slot).
    var grads = List[Krea2LoraGrad]()
    for _ in range(n * KREA2_SLOTS_PER_BLOCK):
        grads.append(Krea2LoraGrad(None, None))

    # ── final-layer backward (frozen) → d into the last single-stream output ─────
    var d_x = krea2_final_layer_backward[L, HEADS, HEADDIM](
        d_velocity, fwd, tmlp_out, w, eps, ctx,
    )

    # ── single-stream block backward, deepest → shallowest ───────────────────────
    var bi = n - 1
    while bi >= 0:
        # recompute the block forward from the SAVED input to regenerate acts.
        var rb = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            fwd.block_inputs[bi].copy(), blk_vec, w.blocks[bi], lora.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        # Phase-1 block backward: LoRA dA/dB (8) + d_x carry.
        var bg = krea2_single_stream_block_lora_backward[L, HEADS, KVHEADS, HEADDIM](
            d_x, blk_vec, w.blocks[bi], lora.blocks[bi], rb.saved,
            cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        var base = bi * KREA2_SLOTS_PER_BLOCK
        grads[base + 0] = bg.wq.copy()
        grads[base + 1] = bg.wk.copy()
        grads[base + 2] = bg.wv.copy()
        grads[base + 3] = bg.gate_w.copy()
        grads[base + 4] = bg.wo.copy()
        grads[base + 5] = bg.mlp_gate_w.copy()
        grads[base + 6] = bg.mlp_up_w.copy()
        grads[base + 7] = bg.mlp_down_w.copy()
        d_x = bg.d_x[].clone(ctx)
        bi -= 1

    # ── OPT-IN reference-branch backward (img-EDIT) ──
    # d_x here IS dL/d(block-stack input) [1,L,features]; its image rows are
    # dL/d(first(img)) — the SAME `d_img_act` klein's img_in_ref_backward consumes.
    # d_img_in_ref = d_img_actᵀ @ ref_tokens (a plain linear_backward_dw). Skipped
    # when no ref (d_img_in_ref stays None). Uses fwd.txtlen/fwd.imglen for the slice.
    var d_img_in_ref = _krea2_emit_img_in_ref_grad(
        d_x, ref_tokens, img_in_ref_w, fwd.txtlen, fwd.imglen, features, ctx,
    )

    return Krea2StackLoraGrads(grads^, TArc(d_x^), d_img_in_ref^)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING variants — load each block's FROZEN weights per-iteration from the
# checkpoint (ShardedSafeTensors), use them, FREE at iteration end. This is the
# inference krea2_forward streaming discipline (krea2_dit.mojo:1304 — "each
# `_wb`/`_scale` load copies only the active block H2D and frees them when the
# loop iteration ends, so the 28-block real model never goes fully GPU-resident").
# The non-streaming Krea2StackWeights path above holds all 28 blocks resident
# (~24GB bf16) → OOM at real depth; the trainer (train_krea2.mojo) uses these.
#
# The frozen final-layer (`last.*`) params are small and loaded ONCE by the
# caller into Krea2StreamFinal (passed to both fwd + bwd). Only the 28 per-block
# bundles stream. Block weights mirror krea2_forward's per-block load EXACTLY:
# matmul weights bf16 (= reference v.to(bf16)); norm scales stay bf16 at tensor
# boundaries and krea2_rmsnorm casts internally for scale.float()+1.
# ══════════════════════════════════════════════════════════════════════════════

# bf16-resident matmul weight (real disk dtype; bf16-round if F32/F16 on disk).
def _stream_wb(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_view_as_bf16(st.tensor_view(key), ctx))


# norm scale: bf16 checkpoint storage. Krea2 RMSNorm casts scale to
# F32 internally for reference scale.float()+1 compute.


def _stream_scale(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_view_as_bf16(st.tensor_view(key), ctx))


def _load_krea2_block_streamed(
    st: ShardedSafeTensors, bi: Int, key_prefix: String, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    """Load block `bi`'s FROZEN weights H2D for one fwd/bwd iteration (caller frees
    by dropping the returned struct at loop end). Keys == krea2_forward's per-block
    load (krea2_dit.mojo:1469-1481). Matmul weights and norm scales keep bf16
    storage boundaries."""
    var p = key_prefix + "blocks." + String(bi) + "."
    return Krea2BlockWeights(
        _stream_wb(st, p + "attn.wq.weight", ctx),
        _stream_wb(st, p + "attn.wk.weight", ctx),
        _stream_wb(st, p + "attn.wv.weight", ctx),
        _stream_wb(st, p + "attn.gate.weight", ctx),
        _stream_wb(st, p + "attn.wo.weight", ctx),
        _stream_wb(st, p + "mlp.gate.weight", ctx),
        _stream_wb(st, p + "mlp.up.weight", ctx),
        _stream_wb(st, p + "mlp.down.weight", ctx),
        _stream_scale(st, p + "attn.qknorm.qnorm.scale", ctx),
        _stream_scale(st, p + "attn.qknorm.knorm.scale", ctx),
        _stream_scale(st, p + "prenorm.scale", ctx),
        _stream_scale(st, p + "postnorm.scale", ctx),
        _stream_wb(st, p + "mod.lin", ctx),
    )


# ══════════════════════════════════════════════════════════════════════════════
# FP8-QUANTIZED-RESIDENT base (T2.B; cfg.quantized_resident == "fp8_e4m3").
#
# The streamed path above re-reads all 28 blocks' bf16 weights from the mmap'd
# checkpoint H2D on EVERY step (in BOTH fwd and bwd recompute). Under GPU pressure
# the 26GB checkpoint is not held in page cache → ~48GB/step disk read = ~150s/step
# (the measured per-step disk re-read). The base is FROZEN (LoRA training), so it
# must load ONCE. bf16-resident is ~24GB → doesn't fit with the ~8GB working set;
# fp8-resident is ~12GB → fits (12 + 8 = 20GB on the 24GB card).
#
# Scheme (the proven ideogram4 layout, models/ideogram4/block.mojo:195-221):
#   load-once: each block's 8 matmul weights (the large [out,in] BF16 tensors) are
#   quantized ONCE to E4M3 bytes + a per-output-row F32 scale
#   (ops/fp8_quant.mojo fp8_e4m3_rowscale + fp8_e4m3_encode_perrow) and held
#   resident (~1 byte/param + 4 bytes/row). The 5 small per-block tensors
#   (qnorm/knorm/prenorm/postnorm scales F32, mod.lin bf16) are tiny → held
#   resident at their existing dtype (NOT quantized — exact, matches the streamed
#   path bit-for-bit).
#   per-block: fp8_e4m3_dequant_perrow_to_bf16 reconstructs the bf16 [out,in]
#   weight right before compute (cheap on-GPU kernel, no disk), into the SAME
#   Krea2BlockWeights the existing block fwd/bwd consume — only the matmul-weight
#   SOURCE changes. LoRA + optimizer state + activations are untouched.
#
# Lossy: fp8 e4m3 has 3 mantissa bits → decode(encode(w)) ≈ cos 0.99+ vs the bf16
# original. This is a DIFFERENT-trajectory numerics class (never mix-resume across
# the flag); the trainer gate is "loss still drops", not bit-exactness.
# ══════════════════════════════════════════════════════════════════════════════

# 8 matmul-weight keys per block (the large [out,in] tensors that get fp8'd),
# in Krea2BlockWeights field order: wq wk wv gate wo mlp_gate mlp_up mlp_down.
comptime KREA2_FP8_KEYS = 8

# Per-block resident bundle: 8 (fp8 bytes, per-row F32 scale) pairs + the 5 small
# resident tensors (qnorm/knorm/prenorm/postnorm F32, mod.lin bf16). Tensor is
# Krea2BlockResidentFp8 / Krea2ResidentFp8 MOVED to krea2_dit.mojo (the lower
# module — krea2_stack imports krea2_dit, so the inference krea2_forward can take
# the resident store WITHOUT a circular import). Re-imported at the top of this file.


# Quantize one bf16 [out,in] matmul weight ONCE: rowscale → fp8 bytes. Returns
# (E4M3 bytes [out,in], F32 per-row scale [out]). The source view is read bf16
# (== _stream_wb / krea2_forward's per-block load: reference v.to(bf16)).
def _quantize_wb_fp8(
    st: ShardedSafeTensors, key: String, ctx: DeviceContext
) raises -> List[TArc]:
    var w_bf = Tensor.from_view_as_bf16(st.tensor_view(key), ctx)   # [out,in] BF16
    var scale = fp8_e4m3_rowscale(w_bf, ctx)                        # F32 [out]
    var bytes = fp8_e4m3_encode_perrow(w_bf, scale, ctx)           # E4M3 [out,in]
    var out = List[TArc]()
    out.append(TArc(bytes^))
    out.append(TArc(scale^))
    return out^   # [bytes, scale]


# Quantize ALL `nblocks` blocks' 8 matmul weights ONCE at trainer startup; hold
# the resident store (~12GB E4M3 + per-row scales). The 5 small per-block tensors
# stay resident at their existing dtype (exact). NO per-step disk access after this.
def build_krea2_resident_fp8(
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    resident_blocks: Int, ctx: DeviceContext
) raises -> Krea2ResidentFp8:
    """Load-once-quantize the FIRST `resident_blocks` (≤ nblocks) frozen krea2 blocks
    to fp8-resident. Each block's 8 matmul weights (wq wk wv gate wo mlp_gate mlp_up
    mlp_down) → E4M3 bytes + per-output-row F32 scale, held resident; the 5 small
    scales/mod.lin stay bf16/F32 resident. Mirrors the ideogram4 resident base, but
    quantizes the bf16 source (ideogram4's ckpt is already fp8). Sync after each block
    bounds peak transient memory (the bf16 source frees before the next block's H2D).
    Blocks [resident_blocks:nblocks] are NOT stored here — the caller streams them per
    step from the page-cached safetensors (reference trainer-style partial offload for 16GB fit)."""
    var n_res = resident_blocks if resident_blocks < nblocks else nblocks
    var blocks = List[Krea2BlockResidentFp8]()
    for bi in range(n_res):
        var p = key_prefix + "blocks." + String(bi) + "."
        var fp8 = List[TArc]()
        var scale = List[TArc]()
        # 8 matmul weights, Krea2BlockWeights field order.
        var keys = List[String]()
        keys.append(p + "attn.wq.weight")
        keys.append(p + "attn.wk.weight")
        keys.append(p + "attn.wv.weight")
        keys.append(p + "attn.gate.weight")
        keys.append(p + "attn.wo.weight")
        keys.append(p + "mlp.gate.weight")
        keys.append(p + "mlp.up.weight")
        keys.append(p + "mlp.down.weight")
        for ki in range(KREA2_FP8_KEYS):
            var q = _quantize_wb_fp8(st, keys[ki], ctx)
            fp8.append(q[0].copy())
            scale.append(q[1].copy())
        blocks.append(Krea2BlockResidentFp8(
            fp8^, scale^,
            _stream_scale(st, p + "attn.qknorm.qnorm.scale", ctx),
            _stream_scale(st, p + "attn.qknorm.knorm.scale", ctx),
            _stream_scale(st, p + "prenorm.scale", ctx),
            _stream_scale(st, p + "postnorm.scale", ctx),
            _stream_wb(st, p + "mod.lin", ctx),
        ))
        ctx.synchronize()   # free the per-block bf16 source before the next H2D so the
        # load-once quantize never holds two blocks' bf16 weights at once (~868MB ea).
        # progress every 7 blocks: this phase is minutes-long and used to be silent
        # ("looks dead" report 2026-07-05) — the UI tails these lines live.
        if (bi + 1) % 7 == 0 or bi + 1 == n_res:
            print("fp8_e4m3 resident base: quantized block", bi + 1, "/", n_res,
                  "(", nblocks - n_res, "streamed)")
    return Krea2ResidentFp8(blocks^)


# Per-block resident reconstruction: dequant the 8 fp8 weights → bf16, clone the 5
# small resident tensors → the SAME Krea2BlockWeights the block fwd/bwd consume.
# NO disk access (the drop-in replacement for _load_krea2_block_streamed when the
# fp8-resident store is present). The transient bf16 block (~868MB) frees when the
# caller drops the returned struct (the existing per-block ctx.synchronize reclaims).
def _load_krea2_block_resident(
    store: Krea2ResidentFp8, bi: Int, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    # Krea2BlockResidentFp8 is not ImplicitlyCopyable (holds Lists) → reference the
    # bundle in place via a Pointer (no copy of the 8 fp8 pairs; the TArc carriers
    # stay owned by the store). The dequant outputs are fresh transient bf16 tensors.
    ref b = store.blocks[bi]
    return Krea2BlockWeights(
        TArc(fp8_e4m3_dequant_perrow_to_bf16(b.fp8[0][], b.scale[0][], ctx)),
        TArc(fp8_e4m3_dequant_perrow_to_bf16(b.fp8[1][], b.scale[1][], ctx)),
        TArc(fp8_e4m3_dequant_perrow_to_bf16(b.fp8[2][], b.scale[2][], ctx)),
        TArc(fp8_e4m3_dequant_perrow_to_bf16(b.fp8[3][], b.scale[3][], ctx)),
        TArc(fp8_e4m3_dequant_perrow_to_bf16(b.fp8[4][], b.scale[4][], ctx)),
        TArc(fp8_e4m3_dequant_perrow_to_bf16(b.fp8[5][], b.scale[5][], ctx)),
        TArc(fp8_e4m3_dequant_perrow_to_bf16(b.fp8[6][], b.scale[6][], ctx)),
        TArc(fp8_e4m3_dequant_perrow_to_bf16(b.fp8[7][], b.scale[7][], ctx)),
        b.qnorm_scale.copy(),
        b.knorm_scale.copy(),
        b.prenorm_scale.copy(),
        b.postnorm_scale.copy(),
        b.mod_lin.copy(),
    )


# ── SquareQ W4-RESIDENT base (cfg.quantized_resident == "squareq_w4") ──────────
# Loads the PREBUILT sidecar slab (scripts/squareq_build_slab.py output dir —
# never quantizes at startup): per matmul weight the packed int4 qweight of the
# H256-rotated rank-R residual + group-64 BF16 scales + BF16 low-rank factors
# (~0.28x bf16, vs fp8's ~0.5x). Per-block load reconstructs the bf16 weight
# W_hat = dequant4@H_bd + lora_up@lora_down^T (ops/squareq.mojo) into the SAME
# Krea2BlockWeights the block fwd/bwd consume — a drop-in sibling of the fp8
# dequant path. The 5 small per-block tensors are read from the sidecar's
# passthrough copies (bit-identical to the base checkpoint by builder contract).


def build_krea2_resident_squareq(
    sc: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    resident_blocks: Int, ctx: DeviceContext
) raises -> Krea2ResidentSquareq:
    """Pin the FIRST `resident_blocks` (≤ nblocks) blocks' squareq payloads
    device-resident VERBATIM from the sidecar. ~0.28x bf16 → all 28 krea2
    blocks fit in ~4.8GB. Sync per block bounds transients."""
    var n_res = resident_blocks if resident_blocks < nblocks else nblocks
    var blocks = List[Krea2BlockResidentSquareq]()
    for bi in range(n_res):
        var p = key_prefix + "blocks." + String(bi) + "."
        var bases = List[String]()
        bases.append(p + "attn.wq")
        bases.append(p + "attn.wk")
        bases.append(p + "attn.wv")
        bases.append(p + "attn.gate")
        bases.append(p + "attn.wo")
        bases.append(p + "mlp.gate")
        bases.append(p + "mlp.up")
        bases.append(p + "mlp.down")
        var qw = List[TArc]()
        var ws = List[TArc]()
        var ld = List[TArc]()
        var lu = List[TArc]()
        for ki in range(KREA2_FP8_KEYS):
            qw.append(TArc(Tensor.from_view_raw(
                sc.tensor_view(bases[ki] + ".qweight"), ctx)))
            ws.append(TArc(Tensor.from_view(
                sc.tensor_view(bases[ki] + ".wscales"), ctx)))
            ld.append(TArc(Tensor.from_view(
                sc.tensor_view(bases[ki] + ".lora_down"), ctx)))
            lu.append(TArc(Tensor.from_view(
                sc.tensor_view(bases[ki] + ".lora_up"), ctx)))
        blocks.append(Krea2BlockResidentSquareq(
            qw^, ws^, ld^, lu^,
            _stream_scale(sc, p + "attn.qknorm.qnorm.scale", ctx),
            _stream_scale(sc, p + "attn.qknorm.knorm.scale", ctx),
            _stream_scale(sc, p + "prenorm.scale", ctx),
            _stream_scale(sc, p + "postnorm.scale", ctx),
            _stream_wb(sc, p + "mod.lin", ctx),
        ))
        ctx.synchronize()
        if (bi + 1) % 7 == 0 or bi + 1 == n_res:
            print("squareq_w4 resident base: pinned block", bi + 1, "/", n_res,
                  "(", nblocks - n_res, "streamed)")
    return Krea2ResidentSquareq(blocks^)


def _squareq_rw(
    b: Krea2BlockResidentSquareq, ki: Int, ctx: DeviceContext
) raises -> TArc:
    """Reconstruct ONE packed squareq weight `ki` of block bundle `b` → bf16 TArc."""
    var out_f = b.qweight[ki][].shape()[0]
    var in_f = b.qweight[ki][].shape()[1] * 2
    return TArc(squareq_reconstruct_weight(
        b.qweight[ki][], b.wscales[ki][], b.lora_down[ki][], b.lora_up[ki][],
        in_f, out_f, ctx,
    ))


def _load_krea2_block_squareq(
    store: Krea2ResidentSquareq, bi: Int, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    """Reconstruct block `bi`'s 8 bf16 matmul weights from the resident packed
    payload (NO disk). Drop-in sibling of _load_krea2_block_resident (fp8)."""
    ref b = store.blocks[bi]
    return Krea2BlockWeights(
        _squareq_rw(b, 0, ctx), _squareq_rw(b, 1, ctx),
        _squareq_rw(b, 2, ctx), _squareq_rw(b, 3, ctx),
        _squareq_rw(b, 4, ctx), _squareq_rw(b, 5, ctx),
        _squareq_rw(b, 6, ctx), _squareq_rw(b, 7, ctx),
        b.qnorm_scale.copy(),
        b.knorm_scale.copy(),
        b.prenorm_scale.copy(),
        b.postnorm_scale.copy(),
        b.mod_lin.copy(),
    )


# ── int8 W8A8-RESIDENT base (cfg.quantized_resident == "int_w8a8") ─────────────
# The SPEED path: quantize each frozen matmul weight TENSORWISE (one scalar scale,
# == reference trainer LinearW8A8) ONCE at startup, hold int8 resident, and do int8×int8→int32
# GEMM per step with NO weight dequant (vs fp8's per-step decode-to-bf16 — the
# 18s/step hotspot). Per weight we hold w_8[N,K] (fwd contracts K), w_8T[K,N]
# (bwd contracts N), and the scalar scale.

# Quantize one bf16 [out,in]=[N,K] matmul weight ONCE → (w_8[N,K], scale[1]).
# Tensorwise (scalar) scale so it factors out of BOTH fwd and bwd. ONE orientation
# only — the backward transposes w8→w8T on the fly (16GB fit; == reference trainer).
def _quantize_wb_int8(
    st: ShardedSafeTensors, key: String, ctx: DeviceContext
) raises -> List[TArc]:
    var w_bf = Tensor.from_view_as_bf16(st.tensor_view(key), ctx)   # [N,K] BF16
    var scale = int8_tensorwise_scale(w_bf, ctx)                    # F32 [1]
    var w8 = int8_encode_tensorwise(w_bf, scale, ctx)              # I8 [N,K]
    var out = List[TArc]()
    out.append(TArc(w8^))
    out.append(TArc(scale^))
    return out^   # [w8, scale]


# Load-once-quantize the FIRST `resident_blocks` frozen blocks to int8-resident.
# Mirror of build_krea2_resident_fp8 (same 8 keys, same 5 small tensors, same
# per-block sync to bound peak transient bf16). ~6.5GB for 28 blocks (1 byte/param
# ×2 orientations + a scalar) → fits 16GB with the ~8GB working set, no offload.
def build_krea2_resident_int8(
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    resident_blocks: Int, ctx: DeviceContext
) raises -> Krea2ResidentInt8:
    """int8 W8A8 analogue of build_krea2_resident_fp8. Each block's 8 matmul
    weights → int8 [N,K] + int8 [K,N] + scalar F32 scale, held resident; the 5
    small scales/mod.lin stay bf16/F32. NO per-step disk access after this."""
    var n_res = resident_blocks if resident_blocks < nblocks else nblocks
    var blocks = List[Krea2BlockResidentInt8]()
    for bi in range(n_res):
        var p = key_prefix + "blocks." + String(bi) + "."
        var w8 = List[TArc]()
        var scale = List[TArc]()
        # 8 matmul weights, Krea2BlockWeights field order.
        var keys = List[String]()
        keys.append(p + "attn.wq.weight")
        keys.append(p + "attn.wk.weight")
        keys.append(p + "attn.wv.weight")
        keys.append(p + "attn.gate.weight")
        keys.append(p + "attn.wo.weight")
        keys.append(p + "mlp.gate.weight")
        keys.append(p + "mlp.up.weight")
        keys.append(p + "mlp.down.weight")
        for ki in range(KREA2_FP8_KEYS):
            var q = _quantize_wb_int8(st, keys[ki], ctx)
            w8.append(q[0].copy())
            scale.append(q[1].copy())
            # Free this weight's transient bf16 source BEFORE loading the next so the
            # 28-block quantize never holds more than one bf16 [out,in] (~200MB) on
            # top of the growing int8 store. One-time load cost; slow is fine.
            ctx.synchronize()
            cu_mempool_trim_current(0)
        blocks.append(Krea2BlockResidentInt8(
            w8^, scale^,
            _stream_scale(st, p + "attn.qknorm.qnorm.scale", ctx),
            _stream_scale(st, p + "attn.qknorm.knorm.scale", ctx),
            _stream_scale(st, p + "prenorm.scale", ctx),
            _stream_scale(st, p + "postnorm.scale", ctx),
            _stream_wb(st, p + "mod.lin", ctx),
        ))
        ctx.synchronize()   # free the per-block bf16 source before the next H2D.
        if (bi + 1) % 7 == 0 or bi + 1 == n_res:
            print("int8_w8a8 resident base: quantized block", bi + 1, "/", n_res,
                  "(", nblocks - n_res, "streamed)")
    return Krea2ResidentInt8(blocks^)


# Per-block resident bundle: NO dequant (the whole point). Build a Krea2BlockWeights
# whose 8 bf16 matmul fields are tiny DUMMIES (never read on the int8 path) and
# whose `int8` payload carries the resident int8 weight refs (refcount copies, no
# device copy). The 5 small resident tensors are cloned as usual. The block's base
# matmuls dispatch on `w.int8` presence → int8_linear_fwd/bwd.
def _krea2_int8_bundle_to_weights(
    b: Krea2BlockResidentInt8, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    var w8 = List[TArc]()
    var scale = List[TArc]()
    for i in range(KREA2_FP8_KEYS):
        w8.append(b.w8[i].copy())
        scale.append(b.scale[i].copy())
    var payload = Krea2BlockInt8(w8^, scale^)
    # One tiny bf16 placeholder reused for all 8 unused matmul fields (refcount).
    var dummy = TArc(zeros_device([1], STDtype.BF16, ctx))
    return Krea2BlockWeights(
        dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy(),
        dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy(),
        b.qnorm_scale.copy(),
        b.knorm_scale.copy(),
        b.prenorm_scale.copy(),
        b.postnorm_scale.copy(),
        b.mod_lin.copy(),
        Optional[Krea2BlockInt8](payload^),
    )


def _load_krea2_block_int8(
    store: Krea2ResidentInt8, bi: Int, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    return _krea2_int8_bundle_to_weights(store.blocks[bi], ctx)


def _load_krea2_block_host_int8_inf(
    store: Krea2HostInt8Inf, bi: Int, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    if bi < store.first or bi - store.first >= len(store.blocks):
        raise Error("Krea2HostInt8Inf block index is outside its covered range")
    var bundle = _krea2_host_i8_block_dev(store, bi - store.first, ctx)
    return _krea2_int8_bundle_to_weights(bundle^, ctx)


# ── int8 W8A8 PINNED-HOST offload for the NON-resident blocks [K:nblocks] ──────
# The device-resident store above (build_krea2_resident_int8) holds only the first
# K blocks; the remainder used to be re-read from DISK per step
# (_load_krea2_block_streamed) — a MJ-1065 policy violation AND the measured
# ~2.1s/block tax (disk read + bf16 decode). This holds those blocks' int8 weights
# PINNED IN HOST RAM instead: quantize ONCE at load, then per step H2D the raw int8
# bytes (~450MB/block over PCIe, no disk, no decode) and int8-GEMM directly. This is
# == SerenityTrainer offload 0.3 (int8 blocks in CPU RAM, H2D per step). Modeled 1:1 on
# offload/turbo_planned_loader.mojo::pin_residents_fp8_host, minus the dequant (int8
# feeds the GEMM directly). The scalar scale + 5 small per-block tensors stay
# device-resident (tiny, ~100KB/block) — only the 8 big int8 weights are H2D'd.
comptime HArc = ArcPointer[HostBuffer[DType.uint8]]


struct Krea2BlockHostInt8(Copyable, Movable):
    """One NON-resident block held as PINNED-HOST int8. The 8 matmul weights are
    int8 [N,K] bytes pinned in host RAM (H2D'd per step); the scalar scale + 5 small
    tensors stay device-resident. Host analogue of Krea2BlockResidentInt8."""
    var w8_h: List[HArc]                  # len 8: pinned host int8 bytes [N,K]
    var w8_nbytes: List[Int]              # len 8: byte count per weight
    var w8_shape: List[List[Int]]         # len 8: [N,K] per weight
    var scale: List[ArcPointer[Tensor]]   # len 8: F32 scalar tensorwise scale [1]
    var qnorm_scale: ArcPointer[Tensor]   # raw checkpoint dtype [HEADDIM]
    var knorm_scale: ArcPointer[Tensor]   # raw checkpoint dtype [HEADDIM]
    var prenorm_scale: ArcPointer[Tensor]    # raw checkpoint dtype [features]
    var postnorm_scale: ArcPointer[Tensor]   # raw checkpoint dtype [features]
    var mod_lin: ArcPointer[Tensor]          # bf16 [6*features]

    def __init__(
        out self,
        var w8_h: List[HArc], var w8_nbytes: List[Int],
        var w8_shape: List[List[Int]], var scale: List[ArcPointer[Tensor]],
        var qnorm_scale: ArcPointer[Tensor], var knorm_scale: ArcPointer[Tensor],
        var prenorm_scale: ArcPointer[Tensor], var postnorm_scale: ArcPointer[Tensor],
        var mod_lin: ArcPointer[Tensor],
    ):
        self.w8_h = w8_h^
        self.w8_nbytes = w8_nbytes^
        self.w8_shape = w8_shape^
        self.scale = scale^
        self.qnorm_scale = qnorm_scale^
        self.knorm_scale = knorm_scale^
        self.prenorm_scale = prenorm_scale^
        self.postnorm_scale = postnorm_scale^
        self.mod_lin = mod_lin^


struct Krea2HostInt8(Copyable, Movable):
    var blocks: List[Krea2BlockHostInt8]   # len == nblocks - resident_blocks
    # ── double-buffered H2D prefetch (turbo_loader pattern) ──────────────────
    # All host blocks share the same 8 weight shapes, so TWO persistent slot
    # sets of 8 device I8 tensors suffice: prefetch block li into slot li%2 on
    # the explicit copy stream while the default stream computes block li∓1.
    # ev[slot] = H2D-done event; the consumer fences the default stream with
    # enqueue_wait_for(ev[slot]) before reading (turbo_loader's load-bearing
    # ordering guarantee). Slot overwrite safety needs NO compute-done event
    # here: the training loops ctx.synchronize() after EVERY block, so the
    # previous user of a slot has fully completed before the next prefetch
    # into it is issued (one iteration later). Arc carriers keep the struct
    # Copyable (Tensor/DeviceStream/DeviceEvent are Movable-only).
    var slot_w8: List[ArcPointer[Tensor]]        # len 16: slot*8 + weight_idx
    var copy_stream: List[ArcPointer[DeviceStream]]  # len 1 (Arc'd for Copyable)
    var ev: List[ArcPointer[DeviceEvent]]        # len 2: per-slot H2D-done

    def __init__(
        out self, var blocks: List[Krea2BlockHostInt8],
        var slot_w8: List[ArcPointer[Tensor]],
        var copy_stream: List[ArcPointer[DeviceStream]],
        var ev: List[ArcPointer[DeviceEvent]],
    ):
        self.blocks = blocks^
        self.slot_w8 = slot_w8^
        self.copy_stream = copy_stream^
        self.ev = ev^


# Quantize blocks [resident_blocks:nblocks] ONCE to int8 and hold the 8 matmul
# weights PINNED IN HOST RAM. Same tensorwise int8 encode as build_krea2_resident_int8
# (== reference trainer), but D2H the int8 bytes into pinned host buffers and drop the device copy —
# so this store costs ~0 device VRAM (only the tiny scale + 5 small tensors stay
# resident). Blocks [0:resident_blocks] are the caller's device-resident store.
def build_krea2_host_int8(
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    resident_blocks: Int, ctx: DeviceContext
) raises -> Krea2HostInt8:
    """int8 pinned-HOST offload for the NON-resident remainder. Mirror of
    build_krea2_resident_int8 for blocks [resident_blocks:nblocks], but the int8
    weights live host-side (H2D per step, no disk). Replaces _load_krea2_block_streamed."""
    var n_res = resident_blocks if resident_blocks < nblocks else nblocks
    var blocks = List[Krea2BlockHostInt8]()
    for bi in range(n_res, nblocks):
        var p = key_prefix + "blocks." + String(bi) + "."
        var keys = List[String]()
        keys.append(p + "attn.wq.weight")
        keys.append(p + "attn.wk.weight")
        keys.append(p + "attn.wv.weight")
        keys.append(p + "attn.gate.weight")
        keys.append(p + "attn.wo.weight")
        keys.append(p + "mlp.gate.weight")
        keys.append(p + "mlp.up.weight")
        keys.append(p + "mlp.down.weight")
        var w8_h = List[HArc]()
        var w8_nbytes = List[Int]()
        var w8_shape = List[List[Int]]()
        var scale = List[TArc]()
        for ki in range(KREA2_FP8_KEYS):
            var w_bf = Tensor.from_view_as_bf16(st.tensor_view(keys[ki]), ctx)  # [N,K] BF16
            var sc = int8_tensorwise_scale(w_bf, ctx)                           # F32 [1]
            var w8 = int8_encode_tensorwise(w_bf, sc, ctx)                      # I8 [N,K] device
            # D2H the int8 bytes into a PINNED host buffer; keep the scalar scale
            # device-resident (4 bytes — an H2D of a scalar per step is wasteful).
            var bh = ctx.enqueue_create_host_buffer[DType.uint8](w8.nbytes())
            ctx.enqueue_copy(bh, w8.buf)   # D2H (== pin_residents_fp8_host)
            ctx.synchronize()          # D2H done + fence the transient device bf16/int8
            cu_mempool_trim_current(0)  # reclaim the transient bf16 source between weights
            w8_h.append(HArc(bh^))
            w8_nbytes.append(w8.nbytes())
            w8_shape.append(w8.shape().copy())
            scale.append(TArc(sc^))
            # w_bf, w8 drop here → device buffers free (sync already ran above).
        blocks.append(Krea2BlockHostInt8(
            w8_h^, w8_nbytes^, w8_shape^, scale^,
            _stream_scale(st, p + "attn.qknorm.qnorm.scale", ctx),
            _stream_scale(st, p + "attn.qknorm.knorm.scale", ctx),
            _stream_scale(st, p + "prenorm.scale", ctx),
            _stream_scale(st, p + "postnorm.scale", ctx),
            _stream_wb(st, p + "mod.lin", ctx),
        ))
        if (bi + 1 - n_res) % 7 == 0 or bi + 1 == nblocks:
            print("int8_w8a8 host offload: pinned block", bi + 1, "/", nblocks,
                  "(", bi + 1 - n_res, "of", nblocks - n_res, "host)")

    # ── double-buffer slots: 2 × 8 persistent device I8 tensors sized from
    # block 0's shapes (all krea2 blocks share the same 8 weight shapes —
    # verified below, fail-loud). Plus the copy stream + 2 H2D-done events.
    var slot_w8 = List[ArcPointer[Tensor]]()
    var copy_stream = List[ArcPointer[DeviceStream]]()
    var ev = List[ArcPointer[DeviceEvent]]()
    if len(blocks) > 0:
        ref b0 = blocks[0]
        for li in range(1, len(blocks)):
            for i in range(KREA2_FP8_KEYS):
                if blocks[li].w8_nbytes[i] != b0.w8_nbytes[i]:
                    raise Error(
                        String("build_krea2_host_int8: weight ") + String(i)
                        + " nbytes mismatch between host blocks (double-buffer"
                        + " slots assume uniform shapes)"
                    )
        for slot in range(2):
            _ = slot
            for i in range(KREA2_FP8_KEYS):
                var dbuf = ctx.enqueue_create_buffer[DType.uint8](b0.w8_nbytes[i])
                var t = Tensor(dbuf^, b0.w8_shape[i].copy(), STDtype.I8)
                slot_w8.append(TArc(t^))
        ctx.synchronize()
        copy_stream.append(ArcPointer(ctx.create_stream()))
        ev.append(ArcPointer(ctx.create_event[disable_timing=True]()))
        ev.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    return Krea2HostInt8(blocks^, slot_w8^, copy_stream^, ev^)


# Issue the async H2D prefetch of host block `li`'s 8 int8 weights into slot
# li%2 on the COPY stream, then record the slot's H2D-done event. NON-blocking:
# the DMA overlaps whatever compute is executing on the default stream. The
# consumer must fence with the slot event before reading (see
# _load_krea2_block_host_int8_slot).
def krea2_host_i8_prefetch(store: Krea2HostInt8, li: Int) raises:
    if li < 0 or li >= len(store.blocks):
        return
    var slot = li % 2
    ref b = store.blocks[li]
    for i in range(KREA2_FP8_KEYS):
        _h2d_dma_copy(
            UInt64(Int(store.slot_w8[slot * KREA2_FP8_KEYS + i][].buf.unsafe_ptr())),
            b.w8_h[i][].unsafe_ptr(),
            b.w8_nbytes[i],
            store.copy_stream[0][],
        )
    store.copy_stream[0][].record_event(store.ev[slot][])


# Slot-consuming variant of _load_krea2_block_host_int8: fence the default
# stream on slot li%2's H2D-done event, then build the block payload from the
# PERSISTENT slot tensors (refcount copies — no allocation, no H2D here; the
# bytes were prefetched by krea2_host_i8_prefetch during the previous block's
# compute). The caller must prefetch li before consuming li, and blocks must be
# consumed in a strictly monotonic li order (asc fwd / desc bwd) so slot li%2
# alternates — the per-block ctx.synchronize() then guarantees the slot's
# previous user finished before its next overwrite.
def _load_krea2_block_host_int8_slot(
    store: Krea2HostInt8, li: Int, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    var slot = li % 2
    ctx.stream().enqueue_wait_for(store.ev[slot][])
    ref b = store.blocks[li]
    var w8 = List[TArc]()
    var scale = List[TArc]()
    for i in range(KREA2_FP8_KEYS):
        w8.append(store.slot_w8[slot * KREA2_FP8_KEYS + i].copy())
        scale.append(b.scale[i].copy())
    var payload = Krea2BlockInt8(w8^, scale^)
    var dummy = TArc(zeros_device([1], STDtype.BF16, ctx))
    return Krea2BlockWeights(
        dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy(),
        dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy(),
        b.qnorm_scale.copy(),
        b.knorm_scale.copy(),
        b.prenorm_scale.copy(),
        b.postnorm_scale.copy(),
        b.mod_lin.copy(),
        Optional[Krea2BlockInt8](payload^),
    )


# Per-step H2D reconstruction of a HOST-offloaded int8 block (local index `li` =
# global bi - resident_blocks). H2D the 8 int8 weight byte buffers from pinned host
# → device I8 tensors and build the SAME Krea2BlockWeights + Krea2BlockInt8 payload
# _load_krea2_block_int8 produces (dummy bf16 fields; int8 dispatch). NO disk, NO
# decode. The H2D runs on the default stream (ordered before the block's compute;
# the caller's per-block ctx.synchronize() covers it). Drop-in for the `else`
# disk-stream branch. The device buffers free when the caller drops the returned
# struct (same lifetime as the disk-streamed / resident paths).
def _load_krea2_block_host_int8(
    store: Krea2HostInt8, li: Int, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    ref b = store.blocks[li]
    var w8 = List[TArc]()
    var scale = List[TArc]()
    for i in range(KREA2_FP8_KEYS):
        var dbuf = ctx.enqueue_create_buffer[DType.uint8](b.w8_nbytes[i])
        ctx.enqueue_copy(dbuf, b.w8_h[i][])   # H2D pinned int8 (== pin_residents_fp8_host)
        var wt = Tensor(dbuf^, b.w8_shape[i].copy(), STDtype.I8)
        w8.append(TArc(wt^))
        scale.append(b.scale[i].copy())
    var payload = Krea2BlockInt8(w8^, scale^)
    var dummy = TArc(zeros_device([1], STDtype.BF16, ctx))
    return Krea2BlockWeights(
        dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy(),
        dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy(),
        b.qnorm_scale.copy(),
        b.knorm_scale.copy(),
        b.prenorm_scale.copy(),
        b.postnorm_scale.copy(),
        b.mod_lin.copy(),
        Optional[Krea2BlockInt8](payload^),
    )


# ── SAVED-TAPE hybrid (KREA2_SAVE_TAPES): skip the backward's recompute ───────
# The historical discipline recomputes each block's forward in the backward
# (full-recompute checkpointing, a 24GB-era memory tradeoff) — one extra
# forward-equivalent per step (~30% of step time). With the int8 base the
# training state is small enough to SAVE the per-block backward tape for the
# first N blocks instead (the SerenityTrainer structure: VRAM spent on activations,
# not recompute). The tape is SLIMMED: q_rope/k_rope/k_full/v_full/v are only
# read by the MATH-mode (non-flash) attention backward, and the trainer always
# runs the cuDNN flash-padmask path (length-bucketed real_len < L), so those 5
# get a shared tiny dummy — ~115MB/block less at b2. Fail-loud if the flash
# tape is absent (the slim tape cannot serve the math-mode backward).
def _slim_tape(saved: Krea2BlockSaved, ctx: DeviceContext) raises -> Krea2BlockSaved:
    if not saved.flash_stats:
        raise Error(
            "krea2 _slim_tape: block ran the MATH-mode attention (no flash tape);"
            " the slimmed saved-tape path requires the flash-padmask regime"
            " (real_len < L). Build without KREA2_SAVE_TAPES for math-mode runs."
        )
    var dummy = TArc(zeros_device([1], STDtype.BF16, ctx))
    return Krea2BlockSaved(
        saved.x.copy(), saved.xm.copy(),
        saved.q_pre.copy(), saved.k_pre.copy(), dummy.copy(),   # v: flash path unused
        dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy(), # q_rope k_rope k_full v_full
        saved.attn_flat.copy(), saved.gate_pre.copy(), saved.sg.copy(), saved.gated.copy(),
        saved.a.copy(), saved.x1.copy(), saved.xm2.copy(),
        saved.mlp_gate.copy(), saved.mlp_up.copy(), saved.sw.copy(), saved.m.copy(),
        saved.xn.copy(), saved.xn2.copy(),
        saved.flash_q.copy(), saved.flash_k.copy(), saved.flash_v.copy(),
        saved.flash_o.copy(), saved.flash_stats.copy(),
    )


# ── FULL-FT: pinned-host BF16 block store, BOTH-WAYS (FULL_FINETUNE_KREA2_PLAN
# P2; v2 FULL SURFACE 2026-07-08, FULL_SURFACE_PLAN Phase B row krea2). The
# host store IS the live model: H2D per block for fwd/bwd (the int8
# double-buffer pattern) and, after the per-block optimizer step mutates the
# device slot, D2H WRITE-BACK into the pinned bytes — the next step's H2D reads
# the updated weights. v2 trains ALL 13 per-block params: slots 0-7 the 8
# matmuls (bf16 on disk), slots 8-12 the 1D params (F32 on disk, pinned
# bf16-CONVERTED via from_view_as_bf16 — the SAME values the v1 frozen
# device-resident copies held): attn.qknorm.qnorm.scale [Dh],
# attn.qknorm.knorm.scale [Dh], prenorm.scale [F], postnorm.scale [F],
# mod.lin [6F] (a 1D Parameter, NOT a Linear — mmdit.py:122-133). The block
# forward READS the 1D params from the SLOTS (the chroma forward-needed trap:
# trained values must feed the forward).
comptime KREA2_FT_KEYS = KREA2_FT_SLOTS_V2   # 13 trained tensors per block

struct Krea2BlockHostBf16(Copyable, Movable):
    var w_h: List[HArc]                   # 13 pinned host BF16 (LIVE params, slot order)
    var w_nbytes: List[Int]
    var w_shape: List[List[Int]]          # 2-D for slots 0-7, 1-D for slots 8-12

    def __init__(
        out self,
        var w_h: List[HArc], var w_nbytes: List[Int], var w_shape: List[List[Int]],
    ):
        self.w_h = w_h^
        self.w_nbytes = w_nbytes^
        self.w_shape = w_shape^


struct Krea2HostBf16(Copyable, Movable):
    var blocks: List[Krea2BlockHostBf16]  # len == nblocks (ALL blocks host)
    # double-buffer slots + copy stream + per-slot H2D-done events (== the
    # int8 store; per-block ctx.synchronize() makes slot reuse safe).
    var slot_w: List[TArc]                # 26 = slot*13 + weight_idx, BF16 device
    var copy_stream: List[ArcPointer[DeviceStream]]
    var ev: List[ArcPointer[DeviceEvent]]

    def __init__(
        out self, var blocks: List[Krea2BlockHostBf16],
        var slot_w: List[TArc],
        var copy_stream: List[ArcPointer[DeviceStream]],
        var ev: List[ArcPointer[DeviceEvent]],
    ):
        self.blocks = blocks^
        self.slot_w = slot_w^
        self.copy_stream = copy_stream^
        self.ev = ev^


def build_krea2_host_bf16(
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int, ctx: DeviceContext
) raises -> Krea2HostBf16:
    """Load ALL `nblocks` blocks' 13 trained params as PINNED-HOST BF16 (the
    full-FT live model, ~24GB host RAM for krea2): slots 0-7 the matmuls (bf16
    read == the reference v.to(bf16) load), slots 8-12 the 1D norm scales +
    mod.lin (F32 on disk -> bf16-converted, the SAME from_view_as_bf16 values
    the v1 frozen device-resident copies held). Fail-loud on rank mismatches
    (a quantized/other-layout checkpoint is not a krea2 full-FT base)."""
    var blocks = List[Krea2BlockHostBf16]()
    var tails = krea2_ft_key_tails()
    for bi in range(nblocks):
        var p = key_prefix + "blocks." + String(bi) + "."
        var w_h = List[HArc]()
        var w_nbytes = List[Int]()
        var w_shape = List[List[Int]]()
        for ki in range(KREA2_FT_KEYS):
            var w_bf = Tensor.from_view_as_bf16(st.tensor_view(p + tails[ki]), ctx)
            var want_rank = 2 if ki < KREA2_FP8_KEYS else 1
            if len(w_bf.shape()) != want_rank:
                raise Error(
                    String("krea2 full-FT store: ") + p + tails[ki]
                    + String(" rank ") + String(len(w_bf.shape()))
                    + String(" != expected ") + String(want_rank)
                )
            var bh = ctx.enqueue_create_host_buffer[DType.uint8](w_bf.nbytes())
            ctx.enqueue_copy(bh, w_bf.buf)   # D2H
            ctx.synchronize()
            w_h.append(HArc(bh^))
            w_nbytes.append(w_bf.nbytes())
            w_shape.append(w_bf.shape().copy())
        blocks.append(Krea2BlockHostBf16(w_h^, w_nbytes^, w_shape^))
        cu_mempool_trim_current(0)
        if (bi + 1) % 7 == 0 or bi + 1 == nblocks:
            print("full-ft bf16 host store: pinned block", bi + 1, "/", nblocks)
    # slots (shapes uniform across blocks — verified for int8; same weights)
    var slot_w = List[TArc]()
    var copy_stream = List[ArcPointer[DeviceStream]]()
    var ev = List[ArcPointer[DeviceEvent]]()
    ref b0 = blocks[0]
    for slot in range(2):
        _ = slot
        for i in range(KREA2_FT_KEYS):
            var dbuf = ctx.enqueue_create_buffer[DType.uint8](b0.w_nbytes[i])
            var t = Tensor(dbuf^, b0.w_shape[i].copy(), STDtype.BF16)
            slot_w.append(TArc(t^))
    ctx.synchronize()
    copy_stream.append(ArcPointer(ctx.create_stream()))
    ev.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    ev.append(ArcPointer(ctx.create_event[disable_timing=True]()))
    return Krea2HostBf16(blocks^, slot_w^, copy_stream^, ev^)


def krea2_host_bf16_prefetch(store: Krea2HostBf16, bi: Int) raises:
    """Async H2D of block bi's 13 bf16 params into slot bi%2 (copy stream)."""
    if bi < 0 or bi >= len(store.blocks):
        return
    var slot = bi % 2
    ref b = store.blocks[bi]
    for i in range(KREA2_FT_KEYS):
        _h2d_dma_copy(
            UInt64(Int(store.slot_w[slot * KREA2_FT_KEYS + i][].buf.unsafe_ptr())),
            b.w_h[i][].unsafe_ptr(),
            b.w_nbytes[i],
            store.copy_stream[0][],
        )
    store.copy_stream[0][].record_event(store.ev[slot][])


def _load_krea2_block_host_bf16_slot(
    store: Krea2HostBf16, bi: Int, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    """Fence slot bi%2's H2D and build a Krea2BlockWeights over the PERSISTENT
    slot tensors (refcount) — the block runs plain bf16 matmuls (int8=None).
    ALL 13 per-block params are trained (v2) and stream from the slots — the
    forward reads the LIVE trained norm scales + mod.lin. The FT optimizer
    mutates these SAME slot tensors in place before write-back."""
    var slot = bi % 2
    ctx.stream().enqueue_wait_for(store.ev[slot][])
    return Krea2BlockWeights(
        store.slot_w[slot * KREA2_FT_KEYS + 0].copy(),
        store.slot_w[slot * KREA2_FT_KEYS + 1].copy(),
        store.slot_w[slot * KREA2_FT_KEYS + 2].copy(),
        store.slot_w[slot * KREA2_FT_KEYS + 3].copy(),
        store.slot_w[slot * KREA2_FT_KEYS + 4].copy(),
        store.slot_w[slot * KREA2_FT_KEYS + 5].copy(),
        store.slot_w[slot * KREA2_FT_KEYS + 6].copy(),
        store.slot_w[slot * KREA2_FT_KEYS + 7].copy(),
        store.slot_w[slot * KREA2_FT_KEYS + 8].copy(),   # qnorm_scale
        store.slot_w[slot * KREA2_FT_KEYS + 9].copy(),   # knorm_scale
        store.slot_w[slot * KREA2_FT_KEYS + 10].copy(),  # prenorm_scale
        store.slot_w[slot * KREA2_FT_KEYS + 11].copy(),  # postnorm_scale
        store.slot_w[slot * KREA2_FT_KEYS + 12].copy(),  # mod_lin
    )


# save/overlay key tails, in the SAME order as the slots / the FT dw lists
# (KREA2_FT_SLOT_* — slots 8-12 are the v2 1D params).
def krea2_ft_key_tails() -> List[String]:
    var tails = List[String]()
    tails.append(String("attn.wq.weight"))
    tails.append(String("attn.wk.weight"))
    tails.append(String("attn.wv.weight"))
    tails.append(String("attn.gate.weight"))
    tails.append(String("attn.wo.weight"))
    tails.append(String("mlp.gate.weight"))
    tails.append(String("mlp.up.weight"))
    tails.append(String("mlp.down.weight"))
    tails.append(String("attn.qknorm.qnorm.scale"))   # KREA2_FT_SLOT_QNORM
    tails.append(String("attn.qknorm.knorm.scale"))   # KREA2_FT_SLOT_KNORM
    tails.append(String("prenorm.scale"))             # KREA2_FT_SLOT_PRENORM
    tails.append(String("postnorm.scale"))            # KREA2_FT_SLOT_POSTNORM
    tails.append(String("mod.lin"))                   # KREA2_FT_SLOT_MOD_LIN
    return tails^


def krea2_host_bf16_save(
    store: Krea2HostBf16, key_prefix: String, path: String
) raises:
    """Write the trained surface (the pinned-host bf16 bytes — the live model)
    DIRECTLY to a safetensors file, no GPU round-trip. Keys keep the ORIGINAL
    checkpoint names (an overlay: load base ckpt, then these). v2 saves the
    FULL per-block surface (13 x nblocks = 364 tensors: 8 matmuls + 4 norm
    scales + mod.lin; the 1D params are saved BF16 — the store's live dtype —
    over an F32 base, like hidream's converted pins)."""
    var tails = krea2_ft_key_tails()
    # header JSON + offsets
    var header = String("{")
    var off = 0
    var first = True
    for bi in range(len(store.blocks)):
        ref b = store.blocks[bi]
        for wi in range(KREA2_FT_KEYS):
            var nm = key_prefix + "blocks." + String(bi) + "." + tails[wi]
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
        raise Error(String("krea2_host_bf16_save: cannot open ") + path)
    # 8-byte LE header length
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
    for bi in range(len(store.blocks)):
        ref b = store.blocks[bi]
        for wi in range(KREA2_FT_KEYS):
            var src = BytePtr(unsafe_from_address=Int(b.w_h[wi][].unsafe_ptr()))
            var wrote = sys_pwrite(fd, src, b.w_nbytes[wi], base + woff)
            if wrote != b.w_nbytes[wi]:
                _ = sys_close(fd)
                raise Error("krea2_host_bf16_save: short write")
            woff += b.w_nbytes[wi]
    _ = sys_close(fd)
    print("[krea2-ft-save] wrote", len(store.blocks) * KREA2_FT_KEYS,
          "trained weights ->", path)


def krea2_host_bf16_overlay_resume(
    store: Krea2HostBf16, key_prefix: String, overlay_path: String
) raises:
    """RESUME store rebuild, step 2: the store already holds the BASE checkpoint
    (build_krea2_host_bf16); copy the trained overlay's bytes over the pinned
    host bytes (byte-exact, BF16/size fail-loud). Key order == krea2_host_
    bf16_save's (bi-major, the 13 tails); a v1 (224-tensor, matmuls-only)
    overlay FAILS LOUD on the count check."""
    var tails = krea2_ft_key_tails()
    var names = List[String]()
    var bufs = List[HArc]()
    var nbytes = List[Int]()
    for bi in range(len(store.blocks)):
        ref b = store.blocks[bi]
        for wi in range(KREA2_FT_KEYS):
            names.append(key_prefix + "blocks." + String(bi) + "." + tails[wi])
            bufs.append(b.w_h[wi].copy())
            nbytes.append(b.w_nbytes[wi])
    full_ft_overlay_into_host_store(overlay_path, names, bufs, nbytes)


def krea2_host_bf16_writeback(store: Krea2HostBf16, bi: Int, ctx: DeviceContext) raises:
    """D2H the (optimizer-updated) slot weights back into the pinned host bytes
    — the host store stays the authoritative model. Enqueued on the default
    stream AFTER the update kernels (stream order); the caller's per-block
    ctx.synchronize() lands it before the slot's next reuse."""
    var slot = bi % 2
    ref b = store.blocks[bi]
    for i in range(KREA2_FT_KEYS):
        ctx.enqueue_copy(b.w_h[i][], store.slot_w[slot * KREA2_FT_KEYS + i][].buf)


# Small one-time frozen `last.*` params (loaded ONCE by the caller; not streamed).
struct Krea2StreamFinal(Copyable, Movable):
    var last_norm: TArc        # [features] raw checkpoint dtype (last.norm.scale)
    var last_mod_lin: TArc     # [2, features]   bf16 (last.modulation.lin)
    var last_lin_w: TArc       # [out_ch, features] bf16
    var last_lin_b: TArc       # [out_ch] bf16

    def __init__(
        out self, var last_norm: TArc, var last_mod_lin: TArc,
        var last_lin_w: TArc, var last_lin_b: TArc,
    ):
        self.last_norm = last_norm^
        self.last_mod_lin = last_mod_lin^
        self.last_lin_w = last_lin_w^
        self.last_lin_b = last_lin_b^

    @staticmethod
    def load(st: ShardedSafeTensors, key_prefix: String, ctx: DeviceContext) raises -> Krea2StreamFinal:
        return Krea2StreamFinal(
            _stream_scale(st, key_prefix + "last.norm.scale", ctx),
            _stream_wb(st, key_prefix + "last.modulation.lin", ctx),
            _stream_wb(st, key_prefix + "last.linear.weight", ctx),
            _stream_wb(st, key_prefix + "last.linear.bias", ctx),
        )

    # build a transient Krea2StackWeights pointing at this final-layer + EMPTY blocks
    # list (the final-layer backward only reads last_*; blocks is never indexed there).
    def as_stack_weights(self) -> Krea2StackWeights:
        return Krea2StackWeights(
            List[Krea2BlockWeights](),
            self.last_norm.copy(), self.last_mod_lin.copy(),
            self.last_lin_w.copy(), self.last_lin_b.copy(),
        )


def krea2_stack_lora_forward_streamed[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    combined: TArc,            # [1, L, features]  block-stack input (post txtmlp cat)
    blk_vec: Tensor,           # [1, 6*features]   block modulation vec (tproj(t))
    tmlp_out: Tensor,          # [1, 1, features]  final-layer tvec (tmlp(temb))
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    cos: Tensor, sin: Tensor,  # [L, HEADDIM/2] per-token RoPE table (untiled)
    eps: Float32,
    txtlen: Int, imglen: Int,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),  # one value per sample, shared
        # across all nblocks; None (or == L) = full attn, present & < L = the
        # length-bucket flash-padmask prefix [0:real_len].
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
        # T2.B fp8-quantized-resident base. None (default, C13) = the per-step disk
        # stream from `st` (UNCHANGED, byte-identical). Present = dequant each
        # block's 8 matmul weights from the resident fp8 store (NO disk).
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
        # squareq_w4-resident base: reconstruct each block's 8 matmul weights from
        # the resident packed sidecar payload (NO disk). Priority just above the
        # fp8 `resident` arm, below the host_bf16/int8 arms.
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
        # int8 W8A8-resident base (SPEED path): the block does int8 GEMM with NO
        # per-step dequant. Takes priority over `resident` when present.
    host_i8: Optional[Krea2HostInt8] = Optional[Krea2HostInt8](None),
        # int8 PINNED-HOST offload for the NON-resident blocks [K:nblocks]: H2D the
        # int8 bytes per step (no disk, no decode; MJ-1065). Replaces the disk stream
        # for bi >= K. Local store index = bi - (resident int8 block count).
    host_i8_inf: Optional[Krea2HostInt8Inf] = Optional[Krea2HostInt8Inf](None),
        # Inference-side pinned-host int8 store loaded verbatim from the shared
        # Krea2 .int8cache sidecar. This is the same payload used by FlowEdit and
        # prevents non-resident LanPaint blocks from falling back to BF16 disk
        # reloads on every inner-model evaluation.
    save_tapes: Int = 0,
        # SAVED-TAPE hybrid: keep the first N blocks' slimmed backward tapes so
        # the backward skips their recompute-forward (see _slim_tape).
    host_bf16: Optional[Krea2HostBf16] = Optional[Krea2HostBf16](None),
        # FULL-FT: stream the LIVE bf16 weights from the pinned host store
        # (takes priority over every other weight source when present).
    # OPT-IN img-EDIT reference conditioning (PORT of klein's img_in_ref). SAME hook
    # as the primary krea2_stack_lora_forward: when BOTH ref_tokens [1,imglen,in_ch]
    # and img_in_ref_w [features,in_ch] are supplied, the block-stack input image rows
    # become first(img) + linear(ref, img_in_ref) BEFORE the streaming block loop.
    # The ref weight + ref_tokens are CONSTANT resident carriers — applied once here,
    # NEVER pushed onto any per-block stream/tape. Absent (or img_in_ref_w==0) => the
    # EXACT existing text-to-image streamed path, byte-for-byte (C13 flags-off).
    ref_tokens: Optional[TArc] = Optional[TArc](None),
    img_in_ref_w: Optional[TArc] = Optional[TArc](None),
    # ── OminiControl EDIT (C6 trainer seam) ───────────────────────────────────
    # The SAME three optional switches krea2_single_stream_block_lora took in
    # C2/C3, threaded UNCHANGED into every block of the streaming loop:
    #   vec_cond  = tproj(tmlp(temb(t=0))) [1, 6*features] for the COND rows
    #   cond_off  = the per-segment split row (Krea2OminiLayout.cond_off())
    #   cond_len  = the COND segment length (Krea2OminiLayout.cond_len())
    # ALL THREE ABSENT (the default) => every block call below is the pre-C6 call
    # with the pre-C6 argument list — the CONDLEN=0 bit-equal contract. The block
    # itself also guards on 0 < cond_off < L, so a -1 split from
    # krea2_omini_mod_split() falls through to the uniform path.
    # NOTE the ROW LAYOUT this implies for `txtlen`/`imglen`: the velocity slice
    # stays [txtlen : txtlen+imglen], i.e. the IMAGE rows only. With the EDIT
    # layout [TXT_real | IMG | COND | TXT_pad] that slice is EXACTLY the image
    # segment — cond rows and pad rows never reach the loss (C6 gate c).
    vec_cond: Optional[TArc] = Optional[TArc](None),
    cond_off: Optional[Int] = Optional[Int](None),
    cond_len: Optional[Int] = Optional[Int](None),
    # ── OminiControl `condition_scale` (C8 inference seam) ────────────────────
    # An additive [1, HEADS, L, L] score bias in `combined`'s dtype, built ONCE
    # per forward by krea2_cache_reader.krea2_build_edit_attn_bias and handed to
    # EVERY block unchanged. Absent (the default, and all the trainer ever
    # passes) => the block's SDPA dispatch is untouched. Present => every block
    # takes the masked math SDPA and `real_len` stops mattering for masking (the
    # bias carries the pad columns). There is NO backward for this arm.
    attn_bias: Optional[TArc] = Optional[TArc](None),
) raises -> Krea2StackForward:
    """STREAMING single-stream stack forward WITH LoRA. Identical math to
    krea2_stack_lora_forward, but each block's FROZEN weights are loaded H2D from
    `st` inside the loop and freed at iteration end (peak = one block's weights +
    acts, NOT all 28 resident). `lora` is the small trainable LoRA set (kept
    resident). Saves per-block inputs for the backward's recompute. real_len
    (optional): the length-bucket flash-padmask prefix threaded into every block."""
    comptime features = HEADS * HEADDIM

    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var res_i8_count = len(resident_i8.value().blocks) if resident_i8 else 0
    # Prime the host-offload double-buffer: start block K's H2D on the copy
    # stream NOW — it lands while the resident blocks [0:K) compute.
    if host_i8 and len(host_i8.value().blocks) > 0:
        krea2_host_i8_prefetch(host_i8.value(), 0)
    if host_bf16:
        krea2_host_bf16_prefetch(host_bf16.value(), 0)
    var x = combined.copy()
    # OPT-IN additive reference projection on the image rows (img-EDIT). Applied ONCE
    # here at the top of the stack — BEFORE the block streaming loop — exactly as the
    # primary path does; the ref carriers stay OUT of the per-block stream/tape. The
    # block_inputs saved below therefore carry first(img)+linear(ref,img_in_ref).
    # Skipped entirely (byte-identical text-to-image path) when no ref is supplied.
    if Bool(ref_tokens) and Bool(img_in_ref_w):
        x = _krea2_apply_img_in_ref_fwd(
            x, ref_tokens.value(), img_in_ref_w.value(),
            txtlen, imglen, features, ctx,
        )
    var block_inputs = List[TArc]()
    var tapes = List[Krea2BlockSaved]()
    for bi in range(nblocks):
        block_inputs.append(x.copy())                          # SAVE the block input
        # full-FT bf16 host store > int8-resident (NO dequant) > int8
        # host-offload (prefetched H2D, no disk) > fp8-resident > disk stream.
        var wbi: Krea2BlockWeights
        if host_bf16:
            wbi = _load_krea2_block_host_bf16_slot(host_bf16.value(), bi, ctx)
            krea2_host_bf16_prefetch(host_bf16.value(), bi + 1)
        elif resident_i8 and bi < res_i8_count:
            wbi = _load_krea2_block_int8(resident_i8.value(), bi, ctx)
        elif (
            host_i8_inf and bi >= host_i8_inf.value().first
            and bi - host_i8_inf.value().first < len(host_i8_inf.value().blocks)
        ):
            wbi = _load_krea2_block_host_int8_inf(
                host_i8_inf.value(), bi, ctx
            )
        elif host_i8 and bi - res_i8_count < len(host_i8.value().blocks):
            var li = bi - res_i8_count
            wbi = _load_krea2_block_host_int8_slot(host_i8.value(), li, ctx)
            # Overlap: stage the NEXT host block while this one computes.
            krea2_host_i8_prefetch(host_i8.value(), li + 1)
        elif resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)  # H2D this block
        var fwd = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            x, blk_vec, wbi, lora.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
            vec_cond=vec_cond.copy(), cond_off=cond_off, cond_len=cond_len,
            attn_bias=attn_bias.copy(),
        )
        x = fwd.out.copy()
        if bi < save_tapes:
            tapes.append(_slim_tape(fwd.saved, ctx))   # refcount copies, no D2D
        ctx.synchronize()   # async free discipline: reclaim this block's saved-acts (incl. ~2GB SDPA
        # scores) + streamed weights before the next H2D — else the deferred frees accumulate -> OOM.
        # wbi drops here → its device weights free before the next block loads.
        # (tape-saved blocks keep their slimmed acts alive by design.)

    var x_blocks_out = x.copy()

    var last_xn = krea2_rmsnorm(x[], fin.last_norm[], eps, ctx)
    var final = krea2_last_layer(
        x[], tmlp_out, fin.last_norm[], fin.last_mod_lin[],
        fin.last_lin_w[], fin.last_lin_b[], features, ctx,
    )
    var velocity = slice(final, 1, txtlen, imglen, ctx)        # [1, imglen, out_ch]

    var out = Krea2StackForward(
        TArc(velocity^), block_inputs^,
        x_blocks_out^, TArc(last_xn^), txtlen, imglen,
    )
    out.tapes = tapes^
    return out^


def krea2_stack_dora_forward_streamed[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    combined: TArc,
    blk_vec: Tensor,
    tmlp_out: Tensor,
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    dora: Krea2StackDirectDoRA, fin: Krea2StreamFinal,
    cos: Tensor, sin: Tensor,
    eps: Float32,
    txtlen: Int, imglen: Int,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> Krea2StackForward:
    """STREAMING single-stream stack forward with direct DoRA W_eff projections."""
    comptime features = HEADS * HEADDIM

    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var x = combined.copy()
    var block_inputs = List[TArc]()
    for bi in range(nblocks):
        block_inputs.append(x.copy())
        var wbi: Krea2BlockWeights
        if resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
        var fwd = krea2_single_stream_block_dora[L, HEADS, KVHEADS, HEADDIM](
            x, blk_vec, wbi, dora.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        x = fwd.out.copy()
        ctx.synchronize()

    var x_blocks_out = x.copy()
    var last_xn = krea2_rmsnorm(x[], fin.last_norm[], eps, ctx)
    var final = krea2_last_layer(
        x[], tmlp_out, fin.last_norm[], fin.last_mod_lin[],
        fin.last_lin_w[], fin.last_lin_b[], features, ctx,
    )
    var velocity = slice(final, 1, txtlen, imglen, ctx)

    return Krea2StackForward(
        TArc(velocity^), block_inputs^,
        x_blocks_out^, TArc(last_xn^), txtlen, imglen,
    )


def krea2_stack_oft_forward_streamed[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    combined: TArc,
    blk_vec: Tensor,
    tmlp_out: Tensor,
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    oft: Krea2StackDirectOFT, fin: Krea2StreamFinal,
    cos: Tensor, sin: Tensor,
    eps: Float32,
    txtlen: Int, imglen: Int,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> Krea2StackForward:
    """STREAMING single-stream stack forward with direct OFT W_eff projections."""
    comptime features = HEADS * HEADDIM

    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var x = combined.copy()
    var block_inputs = List[TArc]()
    for bi in range(nblocks):
        block_inputs.append(x.copy())
        var wbi: Krea2BlockWeights
        if resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
        var fwd = krea2_single_stream_block_oft[L, HEADS, KVHEADS, HEADDIM](
            x, blk_vec, wbi, oft.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        x = fwd.out.copy()
        ctx.synchronize()

    var x_blocks_out = x.copy()
    var last_xn = krea2_rmsnorm(x[], fin.last_norm[], eps, ctx)
    var final = krea2_last_layer(
        x[], tmlp_out, fin.last_norm[], fin.last_mod_lin[],
        fin.last_lin_w[], fin.last_lin_b[], features, ctx,
    )
    var velocity = slice(final, 1, txtlen, imglen, ctx)

    return Krea2StackForward(
        TArc(velocity^), block_inputs^,
        x_blocks_out^, TArc(last_xn^), txtlen, imglen,
    )


def krea2_stack_lora_backward_streamed[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_velocity: Tensor,        # [1, imglen, out_ch] upstream grad on the image tokens
    blk_vec: Tensor,           # [1, 6*features]  block modulation vec
    tmlp_out: Tensor,          # [1, 1, features] final-layer tvec
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    fwd: Krea2StackForward,
    cos: Tensor, sin: Tensor,  # [L, HEADDIM/2] per-token RoPE table (untiled)
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),  # MUST match the forward call —
        # threaded into BOTH the re-run forward and the flash-padmask block backward.
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
        # T2.B fp8-quantized-resident base. MUST match the forward call: None
        # (default, C13) = per-step disk stream from `st` (UNCHANGED); present =
        # dequant each block's 8 matmul weights from the resident fp8 store (NO disk).
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
        # squareq_w4-resident base. MUST match the forward call (same weight source
        # fwd + bwd). Priority just above the fp8 `resident` arm.
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
        # int8 W8A8-resident base (SPEED path). MUST match the forward call. Takes
        # priority over `resident` when present (int8 GEMM, no per-step dequant).
    # OPT-IN img-EDIT reference conditioning (SAME as krea2_stack_lora_backward). When
    # supplied, d_img_in_ref [features,in_ch] is emitted from d_x's image rows (=
    # dL/d(first(img))) via linear_backward_dw on the ref branch. The ref carriers are
    # CONSTANT — used only AFTER the block streaming loop, never in it. Absent =>
    # d_img_in_ref stays None and the pass is byte-identical.
    ref_tokens: Optional[TArc] = Optional[TArc](None),
    img_in_ref_w: Optional[TArc] = Optional[TArc](None),
) raises -> Krea2StackLoraGrads:
    """STREAMING LoRA stack backward. Identical to krea2_stack_lora_backward but
    the final-layer + every per-block FROZEN weight load streams from `st` (peak =
    one block resident). final-layer bwd (frozen) → walk N single-stream blocks
    deepest→shallowest, RE-LOAD + RE-RUN each block's forward from its saved input,
    run the Phase-1 block backward, scatter 8 dA/dB, carry d_x. Returns flat LoRA
    grads + d_combined. real_len (optional): same length-bucket prefix the forward
    used, threaded into the re-run forward (acts must match) AND the block backward."""
    comptime features = HEADS * HEADDIM

    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var grads = List[Krea2LoraGrad]()
    for _ in range(nblocks * KREA2_SLOTS_PER_BLOCK):
        grads.append(Krea2LoraGrad(None, None))

    # final-layer backward (frozen) → d into the last single-stream output. The
    # final-layer bwd only reads fin.last_* (via as_stack_weights — blocks unused).
    var fin_w = fin.as_stack_weights()
    var d_x = krea2_final_layer_backward[L, HEADS, HEADDIM](
        d_velocity, fwd, tmlp_out, fin_w, eps, ctx,
    )

    var bi = nblocks - 1
    while bi >= 0:
        # int8-resident (NO dequant) > fp8-resident (dequant) > disk stream.
        var wbi: Krea2BlockWeights
        if resident_i8 and bi < len(resident_i8.value().blocks):
            wbi = _load_krea2_block_int8(resident_i8.value(), bi, ctx)
        elif resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)  # H2D this block
        # recompute the block forward from the SAVED input to regenerate acts.
        var rb = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            fwd.block_inputs[bi].copy(), blk_vec, wbi, lora.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        var bg = krea2_single_stream_block_lora_backward[L, HEADS, KVHEADS, HEADDIM](
            d_x, blk_vec, wbi, lora.blocks[bi], rb.saved,
            cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
            dbg_block=(bi if bi == nblocks - 1 else -1),
        )
        var base = bi * KREA2_SLOTS_PER_BLOCK
        grads[base + 0] = bg.wq.copy()
        grads[base + 1] = bg.wk.copy()
        grads[base + 2] = bg.wv.copy()
        grads[base + 3] = bg.gate_w.copy()
        grads[base + 4] = bg.wo.copy()
        grads[base + 5] = bg.mlp_gate_w.copy()
        grads[base + 6] = bg.mlp_up_w.copy()
        grads[base + 7] = bg.mlp_down_w.copy()
        d_x = bg.d_x[].clone(ctx)
        comptime if KREA2_B2_DXDEBUG:
            print("[dxdbg-b1] block", bi, " d_x_norm=", _dxdbg_norm(d_x, ctx))
        ctx.synchronize()   # async free discipline: reclaim this block's recompute-acts + backward
        # grads + streamed weights before the next H2D — else the deferred frees accumulate -> OOM.
        bi -= 1
        # wbi drops here → device weights free before the next block loads.

    # OPT-IN reference-branch backward (img-EDIT) — d_x's image rows are the ref
    # grad_y. SAME emission as the primary path; skipped (None) when no ref. The ref
    # carriers were NEVER touched inside the streaming loop above.
    var d_img_in_ref = _krea2_emit_img_in_ref_grad(
        d_x, ref_tokens, img_in_ref_w, fwd.txtlen, fwd.imglen, features, ctx,
    )
    return Krea2StackLoraGrads(grads^, TArc(d_x^), d_img_in_ref^)


# ══════════════════════════════════════════════════════════════════════════════
# TRUE BATCH-2 streamed stack (row-stacked [seq0 | seq1] = [1, 2L, features]).
# Two samples processed as one stacked forward/backward: the token-parallel GEMMs
# (the ~63% of the step) batch over 2L rows; per-sample adaLN via [2, F] mod packs;
# attention runs two cuDNN flash-padmask calls per block (own real_len). RoPE tables
# are PER-SAMPLE (krea2 text positions are all-zero, but the image block sits at
# sequence offset lt, which differs per sample when lt0 != lt1 — so the two samples'
# per-token tables are NOT identical): build each sample's tiled table and CONCAT to
# [2L*HEADS, half] / [2L*KVHEADS, half]. LoRA grads accumulate over both halves (=
# the batch gradient). GATE: parity/krea2_batch2_parity.mojo — loss_B2 vs
# mean(loss_B1(s0), loss_B1(s1)) and grad_B2 vs mean of the two B1 grads.
# ══════════════════════════════════════════════════════════════════════════════
struct Krea2StackForwardB2(Movable):
    var velocity0: TArc                   # [1, imglen, out_ch] sample-0 image velocity
    var velocity1: TArc                   # [1, imglen, out_ch] sample-1
    var block_inputs: List[TArc]          # len N: each block's [1, 2L, features] input
    var x_blocks_out: TArc                # [1, 2L, features] last single-stream output
    var last_xn: TArc                     # [1, 2L, features] krea2_rmsnorm(x_blocks_out)
    var txtlen0: Int                      # sample-0 image-slice offset (== lt0)
    var txtlen1: Int                      # sample-1 image-slice offset (== lt1)
    var imglen: Int
    # SAVED-TAPE hybrid (KREA2_SAVE_TAPES): the forward keeps the first
    # len(tapes) blocks' SLIMMED Krea2BlockSaved so the backward skips their
    # recompute-forward (the reference trainer-parity structure: spend VRAM on activations
    # instead of a 3rd forward-equivalent per step). Empty = full recompute.
    var tapes: List[Krea2BlockSaved]

    def __init__(
        out self, var velocity0: TArc, var velocity1: TArc,
        var block_inputs: List[TArc], var x_blocks_out: TArc, var last_xn: TArc,
        txtlen0: Int, txtlen1: Int, imglen: Int,
    ):
        self.velocity0 = velocity0^
        self.velocity1 = velocity1^
        self.block_inputs = block_inputs^
        self.x_blocks_out = x_blocks_out^
        self.last_xn = last_xn^
        self.txtlen0 = txtlen0
        self.txtlen1 = txtlen1
        self.imglen = imglen
        self.tapes = List[Krea2BlockSaved]()


# Build the two samples' concatenated q/k rope tables [2L*heads, half] from the
# per-sample untiled tables (cos0/cos1 each [L, half]).
# Debug: per-block d_x norm prints in BOTH stack backwards (b1 + b2). For the
# identical-samples parity probe: d_x_b2[0:L] must equal 0.5*d_x_b1 per block, so
# the first block (deep end first) where 2*||d_x_b2[0:L]|| != ||d_x_b1|| is the
# error-injection point. Flip to True ONLY for a debug gate build.
comptime KREA2_B2_DXDEBUG = False


def _dxdbg_norm(t: Tensor, ctx: DeviceContext) raises -> Float64:
    var h = t.to_host(ctx)
    var s = Float64(0.0)
    for i in range(len(h)):
        s += Float64(h[i]) * Float64(h[i])
    return sqrt(s)


def _tile_rope_b2(
    tbl0: Tensor, tbl1: Tensor, L: Int, heads: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    return concat(
        0, ctx,
        _tile_rope_table(tbl0, L, heads, half, ctx),
        _tile_rope_table(tbl1, L, heads, half, ctx),
    )


# FINAL LAYER (per-sample tmlp modulation) → [1, 2L, out_ch].
def krea2_final_layer_b2(
    x: Tensor,             # [1, 2L, features]
    tmlp2: Tensor,         # [2, 1, features]  per-sample tvec
    norm_scale: Tensor,    # [features]
    mod_lin: Tensor,       # [2, features]     (SimpleModulation.lin)
    lin_w: Tensor, lin_b: Tensor,
    features: Int, ctx: DeviceContext,
) raises -> Tensor:
    var mods = krea2_simple_modulation(tmlp2, mod_lin, ctx)  # (scale, shift) each [2,1,features]
    var scale = reshape(mods[0], [2, features], ctx)         # [2, features]
    var shift = reshape(mods[1], [2, features], ctx)
    var xn = krea2_rmsnorm(x, norm_scale, Float32(1.0e-5), ctx)   # [1,2L,features]
    var xm = modulate(xn, scale, shift, ctx)                      # per-sample adaLN
    return linear(xm, lin_w, Optional[Tensor](lin_b.clone(ctx)), ctx)   # [1, 2L, out_ch]


# Scatter one sample's image-token velocity grad into a full [1, L, out_ch] (zeros
# on the txt-head [0:txtlen] and pad-tail [txtlen+imglen:L] rows).
def _scatter_img_half(
    d_vel: Tensor, txtlen: Int, imglen: Int, L: Int, out_ch: Int, ctx: DeviceContext
) raises -> Tensor:
    var tail = L - txtlen - imglen
    if txtlen > 0 and tail > 0:
        var zh = zeros_device([1, txtlen, out_ch], d_vel.dtype(), ctx)
        var zt = zeros_device([1, tail, out_ch], d_vel.dtype(), ctx)
        return concat(1, ctx, concat(1, ctx, zh, d_vel), zt)
    elif txtlen > 0:
        var zh = zeros_device([1, txtlen, out_ch], d_vel.dtype(), ctx)
        return concat(1, ctx, zh, d_vel)
    elif tail > 0:
        var zt = zeros_device([1, tail, out_ch], d_vel.dtype(), ctx)
        return concat(1, ctx, d_vel, zt)
    return d_vel.clone(ctx)


# FINAL-LAYER BACKWARD (FROZEN `last`; d_x only) — per-sample scatter + per-sample scale.
def krea2_final_layer_backward_b2[
    L: Int, HEADS: Int, HEADDIM: Int
](
    d_velocity0: Tensor,   # [1, imglen, out_ch]
    d_velocity1: Tensor,
    fwd: Krea2StackForwardB2,
    tmlp2: Tensor,         # [2, 1, features]
    w: Krea2StackWeights,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime features = HEADS * HEADDIM
    comptime L2 = 2 * L
    var out_ch = w.last_lin_w[].shape()[0]

    var half0 = _scatter_img_half(d_velocity0, fwd.txtlen0, fwd.imglen, L, out_ch, ctx)
    var half1 = _scatter_img_half(d_velocity1, fwd.txtlen1, fwd.imglen, L, out_ch, ctx)
    var d_final2 = concat(1, ctx, half0, half1)              # [1, 2L, out_ch]

    var d_xm = linear_backward_dx(d_final2, w.last_lin_w[], L2, features, out_ch, ctx)
    var d_xm3 = reshape(d_xm, [1, L2, features], ctx)

    var mods = krea2_simple_modulation(tmlp2, w.last_mod_lin[], ctx)  # scale [2,1,features]
    var scale = reshape(mods[0], [2, features], ctx)                  # [2, features]
    var mb = modulate_backward(cast_tensor(d_xm3, fwd.last_xn[].dtype(), ctx), fwd.last_xn[], cast_tensor(scale, fwd.last_xn[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb_dx = krea2_rmsnorm_backward_dx(
        mb.d_x, fwd.x_blocks_out[], w.last_norm[], eps, ctx,
    )
    return rb_dx.clone(ctx)


def krea2_stack_lora_forward_streamed_b2[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    combined: TArc,            # [1, 2L, features]  row-stacked block-stack input
    blk_vec2: Tensor,          # [2, 6*features]    per-sample block modulation vec
    tmlp2: Tensor,             # [2, 1, features]   per-sample final-layer tvec
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    cos0: Tensor, sin0: Tensor,   # [L, HEADDIM/2] sample-0 untiled RoPE table
    cos1: Tensor, sin1: Tensor,   # [L, HEADDIM/2] sample-1
    eps: Float32,
    txtlen0: Int, txtlen1: Int, imglen: Int,
    ctx: DeviceContext,
    real_len0: Int, real_len1: Int,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8] = Optional[Krea2HostInt8](None),
    save_tapes: Int = 0,
) raises -> Krea2StackForwardB2:
    """STREAMING batch-2 stack forward. Same weight-streaming discipline as the B=1
    krea2_stack_lora_forward_streamed (per-block H2D from `st` or dequant from the
    resident fp8 store, freed at iteration end), but the block is the row-stacked b2
    block. Saves per-block [1, 2L, features] inputs for the backward recompute.
    int8 dispatch identical to the B=1 loop (resident int8 > host int8 H2D > fp8 >
    disk); the b2 block's base matmuls run int8 via the shared _linear_lora sites.
    save_tapes: keep the first N blocks' slimmed backward tapes (skip their
    recompute in the backward) — see _slim_tape."""
    comptime features = HEADS * HEADDIM
    comptime half = HEADDIM // 2

    var cos_q = _tile_rope_b2(cos0, cos1, L, HEADS, half, ctx)
    var sin_q = _tile_rope_b2(sin0, sin1, L, HEADS, half, ctx)
    var cos_k = _tile_rope_b2(cos0, cos1, L, KVHEADS, half, ctx)
    var sin_k = _tile_rope_b2(sin0, sin1, L, KVHEADS, half, ctx)

    var res_i8_count = len(resident_i8.value().blocks) if resident_i8 else 0
    if host_i8 and len(host_i8.value().blocks) > 0:
        krea2_host_i8_prefetch(host_i8.value(), 0)
    var x = combined.copy()
    var block_inputs = List[TArc]()
    var tapes = List[Krea2BlockSaved]()
    for bi in range(nblocks):
        block_inputs.append(x.copy())
        var wbi: Krea2BlockWeights
        if resident_i8 and bi < res_i8_count:
            wbi = _load_krea2_block_int8(resident_i8.value(), bi, ctx)
        elif host_i8 and bi - res_i8_count < len(host_i8.value().blocks):
            var li = bi - res_i8_count
            wbi = _load_krea2_block_host_int8_slot(host_i8.value(), li, ctx)
            krea2_host_i8_prefetch(host_i8.value(), li + 1)
        elif resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
        var fwd = krea2_single_stream_block_lora_b2[L, HEADS, KVHEADS, HEADDIM](
            x, blk_vec2, wbi, lora.blocks[bi],
            cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len0, real_len1,
        )
        x = fwd.out.copy()
        if bi < save_tapes:
            tapes.append(_slim_tape(fwd.saved, ctx))   # refcount copies, no D2D
        ctx.synchronize()

    var x_blocks_out = x.copy()
    var last_xn = krea2_rmsnorm(x[], fin.last_norm[], eps, ctx)
    var final = krea2_final_layer_b2(
        x[], tmlp2, fin.last_norm[], fin.last_mod_lin[],
        fin.last_lin_w[], fin.last_lin_b[], features, ctx,
    )                                                          # [1, 2L, out_ch]
    var v0 = slice(final, 1, txtlen0, imglen, ctx)             # sample-0 image tokens
    var v1 = slice(final, 1, L + txtlen1, imglen, ctx)         # sample-1 (offset L)

    var out = Krea2StackForwardB2(
        TArc(v0^), TArc(v1^), block_inputs^,
        x_blocks_out^, TArc(last_xn^), txtlen0, txtlen1, imglen,
    )
    out.tapes = tapes^
    return out^


def krea2_stack_lora_backward_streamed_b2[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_velocity0: Tensor,       # [1, imglen, out_ch]
    d_velocity1: Tensor,
    blk_vec2: Tensor,          # [2, 6*features]
    tmlp2: Tensor,             # [2, 1, features]
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    fwd: Krea2StackForwardB2,
    cos0: Tensor, sin0: Tensor,
    cos1: Tensor, sin1: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len0: Int, real_len1: Int,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> Krea2StackLoraGrads:
    """STREAMING batch-2 stack backward. final-layer bwd (frozen, per-sample scatter)
    → walk N b2 blocks deepest→shallowest: RE-LOAD + RE-RUN each block's b2 forward
    from its saved [1,2L,F] input, run the b2 block backward, scatter the 8 dA/dB,
    carry d_x. The 8 grads are already summed over BOTH sample halves (the batch
    gradient). real_len0/1 + resident MUST match the forward call."""
    comptime features = HEADS * HEADDIM
    comptime half = HEADDIM // 2

    var cos_q = _tile_rope_b2(cos0, cos1, L, HEADS, half, ctx)
    var sin_q = _tile_rope_b2(sin0, sin1, L, HEADS, half, ctx)
    var cos_k = _tile_rope_b2(cos0, cos1, L, KVHEADS, half, ctx)
    var sin_k = _tile_rope_b2(sin0, sin1, L, KVHEADS, half, ctx)

    var grads = List[Krea2LoraGrad]()
    for _ in range(nblocks * KREA2_SLOTS_PER_BLOCK):
        grads.append(Krea2LoraGrad(None, None))

    var fin_w = fin.as_stack_weights()
    var d_x = krea2_final_layer_backward_b2[L, HEADS, HEADDIM](
        d_velocity0, d_velocity1, fwd, tmlp2, fin_w, eps, ctx,
    )

    var bi = nblocks - 1
    while bi >= 0:
        var wbi: Krea2BlockWeights
        if resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
        var rb = krea2_single_stream_block_lora_b2[L, HEADS, KVHEADS, HEADDIM](
            fwd.block_inputs[bi].copy(), blk_vec2, wbi, lora.blocks[bi],
            cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len0, real_len1,
        )
        var bg = krea2_single_stream_block_lora_backward_b2[L, HEADS, KVHEADS, HEADDIM](
            d_x, blk_vec2, wbi, lora.blocks[bi], rb.saved,
            cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len0, real_len1,
            dbg_block=(bi if bi == nblocks - 1 else -1),
        )
        var base = bi * KREA2_SLOTS_PER_BLOCK
        grads[base + 0] = bg.wq.copy()
        grads[base + 1] = bg.wk.copy()
        grads[base + 2] = bg.wv.copy()
        grads[base + 3] = bg.gate_w.copy()
        grads[base + 4] = bg.wo.copy()
        grads[base + 5] = bg.mlp_gate_w.copy()
        grads[base + 6] = bg.mlp_up_w.copy()
        grads[base + 7] = bg.mlp_down_w.copy()
        d_x = bg.d_x[].clone(ctx)
        comptime if KREA2_B2_DXDEBUG:
            var h0 = slice(d_x, 1, 0, L, ctx)
            print("[dxdbg-b2] block", bi,
                  " d_x2L_norm=", _dxdbg_norm(d_x, ctx),
                  " half0x2=", Float64(2.0) * _dxdbg_norm(h0, ctx))
        ctx.synchronize()
        bi -= 1

    return Krea2StackLoraGrads(grads^, TArc(d_x^))


# ── per-block batched-D2H staging (option B) ─────────────────────────────────
# Holds ONE block's 8 dA/dB host buffers + metadata between the enqueue and the
# decode (the single per-block fence sits in between). _PB = pair-of-bufs.
struct _PBufs(Movable):
    var buf: List[HostBuffer[DType.uint8]]   # 16 = 8 adapters × (d_a, d_b)
    var n: List[Int]                          # numel each
    var dt: List[STDtype]                     # dtype each
    var present: List[Bool]                   # adapter slot has a grad

    def __init__(out self, var buf: List[HostBuffer[DType.uint8]],
                 var n: List[Int], var dt: List[STDtype], var present: List[Bool]):
        self.buf = buf^
        self.n = n^
        self.dt = dt^
        self.present = present^


def _enq_one(
    mut buf: List[HostBuffer[DType.uint8]], mut n: List[Int],
    mut dt: List[STDtype], mut present: List[Bool],
    g: Krea2LoraGradT, ctx: DeviceContext,
) raises:
    # d_a then d_b for one adapter; absent → a 1-byte placeholder (present=False).
    if g.d_a:
        var t = g.d_a.value()
        var hb = ctx.enqueue_create_host_buffer[DType.uint8](t[].nbytes())
        ctx.enqueue_copy(dst_buf=hb, src_buf=t[].buf)
        buf.append(hb^); n.append(t[].numel()); dt.append(t[].dtype()); present.append(True)
    else:
        buf.append(ctx.enqueue_create_host_buffer[DType.uint8](1)); n.append(0); dt.append(STDtype.F32); present.append(False)
    if g.d_b:
        var t = g.d_b.value()
        var hb = ctx.enqueue_create_host_buffer[DType.uint8](t[].nbytes())
        ctx.enqueue_copy(dst_buf=hb, src_buf=t[].buf)
        buf.append(hb^); n.append(t[].numel()); dt.append(t[].dtype()); present.append(True)
    else:
        buf.append(ctx.enqueue_create_host_buffer[DType.uint8](1)); n.append(0); dt.append(STDtype.F32); present.append(False)


def _block_grads_d2h_enqueue(bg: Krea2BlockGradsT, ctx: DeviceContext) raises -> _PBufs:
    """Enqueue this block's 8 dA/dB device→host copies (NO fence; the caller's single
    per-block ctx.synchronize covers them). Order matches the Krea2BlockLora slot
    order: wq wk wv gate wo mlp_gate mlp_up mlp_down."""
    var buf = List[HostBuffer[DType.uint8]]()
    var n = List[Int]()
    var dt = List[STDtype]()
    var present = List[Bool]()
    _enq_one(buf, n, dt, present, bg.wq, ctx)
    _enq_one(buf, n, dt, present, bg.wk, ctx)
    _enq_one(buf, n, dt, present, bg.wv, ctx)
    _enq_one(buf, n, dt, present, bg.gate_w, ctx)
    _enq_one(buf, n, dt, present, bg.wo, ctx)
    _enq_one(buf, n, dt, present, bg.mlp_gate_w, ctx)
    _enq_one(buf, n, dt, present, bg.mlp_up_w, ctx)
    _enq_one(buf, n, dt, present, bg.mlp_down_w, ctx)
    return _PBufs(buf^, n^, dt^, present^)


def _decode_buf_f32(buf: HostBuffer[DType.uint8], n: Int, dt: STDtype) raises -> List[Float32]:
    var out = List[Float32]()
    var mdt = dt.to_mojo_dtype()
    if mdt == DType.float32:
        var fp = buf.unsafe_ptr().bitcast[Float32]()
        for i in range(n):
            out.append(fp[i])
    elif mdt == DType.bfloat16:
        var bp = buf.unsafe_ptr().bitcast[BFloat16]()
        for i in range(n):
            out.append(bp[i].cast[DType.float32]())
    else:
        var hp = buf.unsafe_ptr().bitcast[Float16]()
        for i in range(n):
            out.append(hp[i].cast[DType.float32]())
    return out^


def _block_grads_decode_into(mut grads: List[Krea2LoraGrad], base: Int, dh: _PBufs) raises:
    """Decode the 8 adapters' host buffers (already fenced) into grads[base..base+8].
    Buffer layout: adapter k → buf[2k]=d_a, buf[2k+1]=d_b."""
    for k in range(KREA2_SLOTS_PER_BLOCK):
        var ia = 2 * k
        var ib = 2 * k + 1
        var da = Optional[List[Float32]](None)
        var db = Optional[List[Float32]](None)
        if dh.present[ia]:
            da = Optional[List[Float32]](_decode_buf_f32(dh.buf[ia], dh.n[ia], dh.dt[ia]))
        if dh.present[ib]:
            db = Optional[List[Float32]](_decode_buf_f32(dh.buf[ib], dh.n[ib], dh.dt[ib]))
        grads[base + k] = Krea2LoraGrad(da^, db^)


def _copy_krea2_grad_to_adamw_state(
    mut state: LoraAdamWPlainDeviceState,
    adapter_idx: Int,
    g: Krea2LoraGradT,
    mut keepalive: List[TArc],
    ctx: DeviceContext,
) raises:
    if not g.d_a:
        raise Error(
            "krea2_stack_lora_backward_streamed_adamw_device_grads: missing dA at adapter "
            + String(adapter_idx)
        )
    if not g.d_b:
        raise Error(
            "krea2_stack_lora_backward_streamed_adamw_device_grads: missing dB at adapter "
            + String(adapter_idx)
        )
    var d_a = _krea2_device_grad_for_adamw_state(state, g.d_a.value(), keepalive, ctx)
    var d_b = _krea2_device_grad_for_adamw_state(state, g.d_b.value(), keepalive, ctx)
    lora_adamw_plain_device_state_copy_device_grad_pair(state, adapter_idx, d_a, d_b, ctx)


def _copy_krea2_grad_to_a3_state(
    mut state: Automagic3DeviceState,
    adapter_idx: Int,
    g: Krea2LoraGradT,
    mut keepalive: List[TArc],
    ctx: DeviceContext,
) raises:
    if not g.d_a:
        raise Error(
            "krea2_stack_lora_backward_streamed_automagic3_device_grads: missing dA at adapter "
            + String(adapter_idx)
        )
    if not g.d_b:
        raise Error(
            "krea2_stack_lora_backward_streamed_automagic3_device_grads: missing dB at adapter "
            + String(adapter_idx)
        )
    automagic3_device_state_copy_device_grad_pair(
        state, adapter_idx, g.d_a.value(), g.d_b.value(), keepalive, ctx
    )


def _krea2_device_grad_for_adamw_state(
    state: LoraAdamWPlainDeviceState,
    t: TArc,
    mut keepalive: List[TArc],
    ctx: DeviceContext,
) raises -> TArc:
    if t[].dtype() == state.grad_dtype:
        return t.copy()
    # The resident optimizer state defines grad storage. Krea2/ai-toolkit keeps
    # LoRA grads BF16; old F32-state callers still get an explicit cast here.
    var tg = TArc(cast_tensor(t[], state.grad_dtype, ctx))
    keepalive.append(tg.copy())
    return tg^


def _copy_krea2_block_grads_to_adamw_state(
    bg: Krea2BlockGradsT,
    base: Int,
    mut state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
) raises -> Krea2GradCopyKeepalive:
    var keepalive = List[TArc]()
    _copy_krea2_grad_to_adamw_state(state, base + 0, bg.wq, keepalive, ctx)
    _copy_krea2_grad_to_adamw_state(state, base + 1, bg.wk, keepalive, ctx)
    _copy_krea2_grad_to_adamw_state(state, base + 2, bg.wv, keepalive, ctx)
    _copy_krea2_grad_to_adamw_state(state, base + 3, bg.gate_w, keepalive, ctx)
    _copy_krea2_grad_to_adamw_state(state, base + 4, bg.wo, keepalive, ctx)
    _copy_krea2_grad_to_adamw_state(state, base + 5, bg.mlp_gate_w, keepalive, ctx)
    _copy_krea2_grad_to_adamw_state(state, base + 6, bg.mlp_up_w, keepalive, ctx)
    _copy_krea2_grad_to_adamw_state(state, base + 7, bg.mlp_down_w, keepalive, ctx)
    return Krea2GradCopyKeepalive(keepalive^)


def _copy_krea2_block_grads_to_a3_state(
    bg: Krea2BlockGradsT,
    base: Int,
    mut state: Automagic3DeviceState,
    ctx: DeviceContext,
) raises -> Krea2GradCopyKeepalive:
    var keepalive = List[TArc]()
    _copy_krea2_grad_to_a3_state(state, base + 0, bg.wq, keepalive, ctx)
    _copy_krea2_grad_to_a3_state(state, base + 1, bg.wk, keepalive, ctx)
    _copy_krea2_grad_to_a3_state(state, base + 2, bg.wv, keepalive, ctx)
    _copy_krea2_grad_to_a3_state(state, base + 3, bg.gate_w, keepalive, ctx)
    _copy_krea2_grad_to_a3_state(state, base + 4, bg.wo, keepalive, ctx)
    _copy_krea2_grad_to_a3_state(state, base + 5, bg.mlp_gate_w, keepalive, ctx)
    _copy_krea2_grad_to_a3_state(state, base + 6, bg.mlp_up_w, keepalive, ctx)
    _copy_krea2_grad_to_a3_state(state, base + 7, bg.mlp_down_w, keepalive, ctx)
    return Krea2GradCopyKeepalive(keepalive^)


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE-GRAD streamed backward (option B) — IDENTICAL conductor to
# krea2_stack_lora_backward_streamed, but the per-block backward keeps its 8 LoRA
# dA/dB on DEVICE (krea2_single_stream_block_lora_backward_dev → Krea2BlockGradsT),
# then we batch-copy that ONE block's 8 grads to host under the single per-block
# fence and decode into the host grad lists; the device `bg` drops at loop end so
# ≤8 device grads (~7.7MB) are ever resident. 28 fences/step (vs hand-chain 224),
# fits the 24GB edge (holding all 224 = 215MB tipped the OOM). Returns the SAME
# host Krea2StackLoraGrads the hand-chain returns.
# ══════════════════════════════════════════════════════════════════════════════
def krea2_stack_lora_backward_streamed_dev[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_velocity: Tensor,        # [1, imglen, out_ch] upstream grad on the image tokens
    blk_vec: Tensor,           # [1, 6*features]  block modulation vec
    tmlp_out: Tensor,          # [1, 1, features] final-layer tvec
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    fwd: Krea2StackForward,
    cos: Tensor, sin: Tensor,  # [L, HEADDIM/2] per-token RoPE table (untiled)
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    # OPT-IN img-EDIT reference conditioning (SAME as krea2_stack_lora_backward). The
    # ref carriers are CONSTANT — used only AFTER the streaming loop. Absent =>
    # d_img_in_ref stays None and the pass is byte-identical.
    ref_tokens: Optional[TArc] = Optional[TArc](None),
    img_in_ref_w: Optional[TArc] = Optional[TArc](None),
) raises -> Krea2StackLoraGrads:
    """Device-grad streamed backward, PER-BLOCK batched D2H (option B). Same recompute
    + block backward math as krea2_stack_lora_backward_streamed (bit-identical), but
    the per-block backward keeps its 8 LoRA dA/dB on DEVICE (no per-adapter to_host),
    then we batch-copy that ONE block's 8 grads to host under the SINGLE per-block
    fence already present (the async-free seam), decode into the host grad lists, and
    the device `bg` drops at loop end → those 8 device grads free before the next
    block. So: 28 fences/step (1/block, vs the hand-chain's 224), and ≤8 device LoRA
    grads (~7.7MB) ever resident (vs holding all 224 = 215MB). This fits the ~1GB
    24GB-edge headroom where holding all 224 tipped the OOM. Returns the SAME host
    Krea2StackLoraGrads the hand-chain returns. real_len/resident MUST match the
    forward call (same contract as the streamed path)."""
    comptime features = HEADS * HEADDIM

    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var grads = List[Krea2LoraGrad]()
    for _ in range(nblocks * KREA2_SLOTS_PER_BLOCK):
        grads.append(Krea2LoraGrad(None, None))

    var fin_w = fin.as_stack_weights()
    var d_x = krea2_final_layer_backward[L, HEADS, HEADDIM](
        d_velocity, fwd, tmlp_out, fin_w, eps, ctx,
    )

    var bi = nblocks - 1
    while bi >= 0:
        var wbi: Krea2BlockWeights
        if resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
        var rb = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            fwd.block_inputs[bi].copy(), blk_vec, wbi, lora.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        var bg = krea2_single_stream_block_lora_backward_dev[L, HEADS, KVHEADS, HEADDIM](
            d_x, blk_vec, wbi, lora.blocks[bi], rb.saved,
            cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        var base = bi * KREA2_SLOTS_PER_BLOCK
        # PER-BLOCK batched D2H: enqueue this block's 8 dA/dB copies (no fence yet),
        # then the single block fence below covers them, then decode into host lists.
        var dh = _block_grads_d2h_enqueue(bg, ctx)
        d_x = bg.d_x[].clone(ctx)
        ctx.synchronize()   # ONE fence/block: covers the 8 D2H copies above AND reclaims
        # this block's recompute-acts + backward intermediates + streamed weights before
        # the next H2D. The 8 device grads in `bg` drop at the end of this iteration.
        # decode the now-resident host buffers into the 8 host grad slots.
        _block_grads_decode_into(grads, base, dh)
        bi -= 1
        # bg drops here → the 8 device LoRA grads free; wbi drops → device weights free.

    # OPT-IN reference-branch backward (img-EDIT) — SAME emission as the primary path;
    # skipped (None) when no ref. Ref carriers untouched by the streaming loop above.
    var d_img_in_ref = _krea2_emit_img_in_ref_grad(
        d_x, ref_tokens, img_in_ref_w, fwd.txtlen, fwd.imglen, features, ctx,
    )
    return Krea2StackLoraGrads(grads^, TArc(d_x^), d_img_in_ref^)


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE-GRAD → RESIDENT ADAMW G BUFFER
#
# Same streamed hand-chain as krea2_stack_lora_backward_streamed_dev, but each
# block's transient Krea2LoraGradT is copied D2D into opt_state.dev_g instead of
# decoding to host Krea2StackLoraGrads. The caller runs
# fused_lora_adamw_plain_step_resident_preloaded_grads after this returns.
# ══════════════════════════════════════════════════════════════════════════════
def krea2_stack_lora_backward_streamed_adamw_device_grads[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_velocity: Tensor,
    blk_vec: Tensor,
    tmlp_out: Tensor,
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    fwd: Krea2StackForward,
    cos: Tensor, sin: Tensor,
    eps: Float32,
    mut opt_state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8] = Optional[Krea2HostInt8](None),
    # OPT-IN img-EDIT reference conditioning. The ref carriers are CONSTANT — used
    # only AFTER the streaming loop, never copied into opt_state.dev_g. Absent =>
    # d_img_in_ref stays None (returned in Krea2StackDeviceGradWrite) and the LoRA
    # device-grad write path is byte-identical.
    ref_tokens: Optional[TArc] = Optional[TArc](None),
    img_in_ref_w: Optional[TArc] = Optional[TArc](None),
    # ── OminiControl EDIT (C6 trainer seam) — the mirror of the streamed
    # forward's three switches. They MUST match the forward call exactly: a
    # per-segment forward with a uniform backward is silently wrong on the
    # condition rows (krea2_block.mojo C2/C3/C4 contract). Threaded into BOTH
    # the recompute forward and the block backward. All three absent (default)
    # => every call below is the pre-C6 call with the pre-C6 arguments.
    #
    # d_vec_t / d_vec_cond (the two temb chains' modulation-vector grads the C4
    # backward emits per segment) are DISCARDED here, exactly as the uniform
    # path already discards its d_vec: krea2's temb->tmlp->tproj chain is FROZEN
    # (Krea2ResidentCond, no adapters), so neither chain has a trainable
    # parameter for those grads to reach. The only consumer of a block d_vec in
    # this trainer is the OFT carrier path (train_krea2.mojo:3185), which does
    # not support the EDIT layout.
    vec_cond: Optional[TArc] = Optional[TArc](None),
    cond_off: Optional[Int] = Optional[Int](None),
    cond_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2StackDeviceGradWrite:
    comptime features = HEADS * HEADDIM

    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var fin_w = fin.as_stack_weights()
    var d_x = krea2_final_layer_backward[L, HEADS, HEADDIM](
        d_velocity, fwd, tmlp_out, fin_w, eps, ctx,
    )

    var res_i8_count = len(resident_i8.value().blocks) if resident_i8 else 0
    # Prime the host-offload double-buffer for the DESCENDING walk: the deepest
    # host block (the last one) is consumed first.
    if host_i8 and len(host_i8.value().blocks) > 0:
        krea2_host_i8_prefetch(host_i8.value(), len(host_i8.value().blocks) - 1)
    var grad_count = 0
    var streaming_sync_count = 0
    var bi = nblocks - 1
    while bi >= 0:
        var wbi: Krea2BlockWeights
        if resident_i8 and bi < res_i8_count:
            wbi = _load_krea2_block_int8(resident_i8.value(), bi, ctx)
        elif host_i8 and bi - res_i8_count < len(host_i8.value().blocks):
            var li = bi - res_i8_count
            wbi = _load_krea2_block_host_int8_slot(host_i8.value(), li, ctx)
            # Overlap: stage the next-shallower host block during this compute.
            krea2_host_i8_prefetch(host_i8.value(), li - 1)
        elif resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
        # SAVED-TAPE hybrid: blocks with a forward-saved tape skip the recompute.
        var bg: Krea2BlockGradsT
        if bi < len(fwd.tapes):
            bg = krea2_single_stream_block_lora_backward_dev[L, HEADS, KVHEADS, HEADDIM](
                d_x, blk_vec, wbi, lora.blocks[bi], fwd.tapes[bi],
                cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
                vec_cond=vec_cond.copy(), cond_off=cond_off, cond_len=cond_len,
            )
        else:
            var rb = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
                fwd.block_inputs[bi].copy(), blk_vec, wbi, lora.blocks[bi],
                cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
                vec_cond=vec_cond.copy(), cond_off=cond_off, cond_len=cond_len,
            )
            bg = krea2_single_stream_block_lora_backward_dev[L, HEADS, KVHEADS, HEADDIM](
                d_x, blk_vec, wbi, lora.blocks[bi], rb.saved,
                cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
                vec_cond=vec_cond.copy(), cond_off=cond_off, cond_len=cond_len,
            )
        var base = bi * KREA2_SLOTS_PER_BLOCK
        var grad_keepalive = _copy_krea2_block_grads_to_adamw_state(
            bg, base, opt_state, ctx
        )
        grad_count += KREA2_SLOTS_PER_BLOCK
        d_x = bg.d_x[].clone(ctx)
        ctx.synchronize()   # preserve the streaming async-free discipline per block
        streaming_sync_count += 1
        _ = len(grad_keepalive.grads)
        bi -= 1

    # OPT-IN reference-branch backward (img-EDIT) — SAME emission as the primary path;
    # skipped (None) when no ref. Ref carriers never entered opt_state.dev_g. The
    # caller AdamW-steps this d_img_in_ref on the standalone Krea2ImgInRefParam.
    var d_img_in_ref = _krea2_emit_img_in_ref_grad(
        d_x, ref_tokens, img_in_ref_w, fwd.txtlen, fwd.imglen, features, ctx,
    )
    return Krea2StackDeviceGradWrite(
        TArc(d_x^), grad_count, streaming_sync_count, d_img_in_ref^,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRUE BATCH-2 DEVICE-GRAD → RESIDENT ADAMW G BUFFER
#
# Merge of krea2_stack_lora_backward_streamed_b2 (the b2 recompute + b2 block backward
# math: _tile_rope_b2 tables, krea2_final_layer_backward_b2, per-block b2 forward
# recompute + b2 device block backward) and krea2_stack_lora_backward_streamed_adamw_
# device_grads (the D2D copy of each block's transient Krea2BlockGradsT into
# opt_state.dev_g via _copy_krea2_block_grads_to_adamw_state — no per-adapter to_host,
# no host grad lists). The 8 dA/dB are already summed over BOTH sample halves (the
# batch gradient). real_len0/1 + resident MUST match the forward call. Returns the
# SAME Krea2StackDeviceGradWrite the b1 device-grad AdamW backward uses; the caller
# runs fused_lora_adamw_plain_step_resident_preloaded_grads after this returns.
# ══════════════════════════════════════════════════════════════════════════════
def krea2_stack_lora_backward_streamed_b2_adamw_device_grads[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_velocity0: Tensor,       # [1, imglen, out_ch]
    d_velocity1: Tensor,
    blk_vec2: Tensor,          # [2, 6*features]
    tmlp2: Tensor,             # [2, 1, features]
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    fwd: Krea2StackForwardB2,
    cos0: Tensor, sin0: Tensor,
    cos1: Tensor, sin1: Tensor,
    eps: Float32,
    mut opt_state: LoraAdamWPlainDeviceState,
    ctx: DeviceContext,
    real_len0: Int, real_len1: Int,
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
    resident_i8: Optional[Krea2ResidentInt8] = Optional[Krea2ResidentInt8](None),
    host_i8: Optional[Krea2HostInt8] = Optional[Krea2HostInt8](None),
) raises -> Krea2StackDeviceGradWrite:
    comptime features = HEADS * HEADDIM
    comptime half = HEADDIM // 2

    var cos_q = _tile_rope_b2(cos0, cos1, L, HEADS, half, ctx)
    var sin_q = _tile_rope_b2(sin0, sin1, L, HEADS, half, ctx)
    var cos_k = _tile_rope_b2(cos0, cos1, L, KVHEADS, half, ctx)
    var sin_k = _tile_rope_b2(sin0, sin1, L, KVHEADS, half, ctx)

    var fin_w = fin.as_stack_weights()
    var d_x = krea2_final_layer_backward_b2[L, HEADS, HEADDIM](
        d_velocity0, d_velocity1, fwd, tmlp2, fin_w, eps, ctx,
    )

    var res_i8_count = len(resident_i8.value().blocks) if resident_i8 else 0
    if host_i8 and len(host_i8.value().blocks) > 0:
        krea2_host_i8_prefetch(host_i8.value(), len(host_i8.value().blocks) - 1)
    var grad_count = 0
    var streaming_sync_count = 0
    var bi = nblocks - 1
    while bi >= 0:
        var wbi: Krea2BlockWeights
        if resident_i8 and bi < res_i8_count:
            wbi = _load_krea2_block_int8(resident_i8.value(), bi, ctx)
        elif host_i8 and bi - res_i8_count < len(host_i8.value().blocks):
            var li = bi - res_i8_count
            wbi = _load_krea2_block_host_int8_slot(host_i8.value(), li, ctx)
            krea2_host_i8_prefetch(host_i8.value(), li - 1)
        elif resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
        # SAVED-TAPE hybrid: blocks with a forward-saved tape skip the recompute
        # (the extra forward-equivalent); the rest re-run the block forward.
        var bg: Krea2BlockGradsT
        if bi < len(fwd.tapes):
            bg = krea2_single_stream_block_lora_backward_b2_dev[L, HEADS, KVHEADS, HEADDIM](
                d_x, blk_vec2, wbi, lora.blocks[bi], fwd.tapes[bi],
                cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len0, real_len1,
            )
        else:
            var rb = krea2_single_stream_block_lora_b2[L, HEADS, KVHEADS, HEADDIM](
                fwd.block_inputs[bi].copy(), blk_vec2, wbi, lora.blocks[bi],
                cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len0, real_len1,
            )
            bg = krea2_single_stream_block_lora_backward_b2_dev[L, HEADS, KVHEADS, HEADDIM](
                d_x, blk_vec2, wbi, lora.blocks[bi], rb.saved,
                cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len0, real_len1,
            )
        var base = bi * KREA2_SLOTS_PER_BLOCK
        var grad_keepalive = _copy_krea2_block_grads_to_adamw_state(
            bg, base, opt_state, ctx
        )
        grad_count += KREA2_SLOTS_PER_BLOCK
        d_x = bg.d_x[].clone(ctx)
        ctx.synchronize()   # preserve the streaming async-free discipline per block
        streaming_sync_count += 1
        _ = len(grad_keepalive.grads)
        bi -= 1

    return Krea2StackDeviceGradWrite(TArc(d_x^), grad_count, streaming_sync_count)


# ── FULL-FT streamed stack backward (FULL_FINETUNE_KREA2_PLAN P3) ─────────────
# Descending walk over the N blocks: H2D the LIVE bf16 weights from the pinned
# host store into the slot → (tape or recompute) → FT block backward (8 dW F32
# + d_x) → per-weight fused DEVICE Adafactor step mutating the SLOT weights in
# place (SR bf16 write) → D2H write-back into the host store → carry d_x.
# The fused-back-pass shape: grads and updated weights never accumulate beyond
# one block. States: af_states flat bi*13+wi (slots 0-7 factored row/col,
# slots 8-12 unfactored 1D exp_avg_sq; device-resident, ~tens of MB total).
# The per-block ctx.synchronize() lands the write-back before the slot's next
# reuse (bi-2's prefetch).
struct Krea2StackFTWrite(Movable):
    var d_combined: TArc
    var grad_count: Int
    var streaming_sync_count: Int

    def __init__(out self, var d_combined: TArc, grad_count: Int, streaming_sync_count: Int):
        self.d_combined = d_combined^
        self.grad_count = grad_count
        self.streaming_sync_count = streaming_sync_count


def krea2_stack_ft_backward_streamed[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_velocity: Tensor,
    blk_vec: Tensor,
    tmlp_out: Tensor,
    nblocks: Int,
    host_bf16: Krea2HostBf16,
    lora: Krea2StackLora,             # EMPTY adapters (base-only recompute fwd)
    fin: Krea2StreamFinal,
    fwd: Krea2StackForward,
    cos: Tensor, sin: Tensor,
    eps: Float32,
    mut af_states: List[AdafactorDeviceState],   # len nblocks*13, flat bi*13+wi
        # (KREA2_FT_SLOT_* order: 0-7 factored 2D matmuls, 8-12 unfactored 1D)
    t_step: Int,                      # 1-based optimizer step
    lr: Float64, beta2_decay: Float64, eps2: Float64, d_thresh: Float64,
    weight_decay: Float64,
    sr_seed: UInt64,                  # per-step seed; 0 = RNE writes
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2StackFTWrite:
    comptime features = HEADS * HEADDIM

    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var fin_w = fin.as_stack_weights()
    var d_x = krea2_final_layer_backward[L, HEADS, HEADDIM](
        d_velocity, fwd, tmlp_out, fin_w, eps, ctx,
    )

    if len(af_states) != nblocks * KREA2_FT_KEYS:
        raise Error("krea2_stack_ft_backward_streamed: af_states length mismatch")

    # Prime the DESCENDING walk: deepest block first.
    krea2_host_bf16_prefetch(host_bf16, nblocks - 1)
    var grad_count = 0
    var streaming_sync_count = 0
    var bi = nblocks - 1
    while bi >= 0:
        var wbi = _load_krea2_block_host_bf16_slot(host_bf16, bi, ctx)
        # Stage the next-shallower block's H2D behind this block's compute.
        krea2_host_bf16_prefetch(host_bf16, bi - 1)
        var bg: Krea2BlockFTGrads
        if bi < len(fwd.tapes):
            bg = krea2_single_stream_block_ft_backward_dev[L, HEADS, KVHEADS, HEADDIM](
                d_x, blk_vec, wbi, fwd.tapes[bi],
                cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
            )
        else:
            var rb = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
                fwd.block_inputs[bi].copy(), blk_vec, wbi, lora.blocks[bi],
                cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
            )
            bg = krea2_single_stream_block_ft_backward_dev[L, HEADS, KVHEADS, HEADDIM](
                d_x, blk_vec, wbi, rb.saved,
                cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
            )
        # ── fused back pass: update THIS block's 13 slot params in place —
        # FACTORED (2D matmuls, slots 0-7) via adafactor_step_device,
        # UNFACTORED (1D scales + mod.lin, slots 8-12) via
        # adafactor_step_device_1d (torch trains <2D params with the
        # unfactored second moment — the Phase A 1D path). ───────────────────
        var slot = bi % 2
        for wi in range(KREA2_FT_KEYS):
            var flat = bi * KREA2_FT_KEYS + wi
            if af_states[flat].factored:
                adafactor_step_device(
                    host_bf16.slot_w[slot * KREA2_FT_KEYS + wi][],
                    bg.dw[wi][],
                    af_states[flat],
                    t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                    weight_decay,
                    sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)),
                    ctx,
                )
            else:
                adafactor_step_device_1d(
                    host_bf16.slot_w[slot * KREA2_FT_KEYS + wi][],
                    bg.dw[wi][],
                    af_states[flat],
                    t_step, lr, beta2_decay, Float64(-1.0), eps2, d_thresh,
                    weight_decay,
                    sr_seed ^ (UInt64(flat + 1) * UInt64(0x9E37)),
                    ctx,
                )
        krea2_host_bf16_writeback(host_bf16, bi, ctx)
        grad_count += KREA2_FT_KEYS
        d_x = bg.d_x[].clone(ctx)
        ctx.synchronize()   # lands: grads freed, update done, write-back D2H'd
        streaming_sync_count += 1
        bi -= 1

    return Krea2StackFTWrite(TArc(d_x^), grad_count, streaming_sync_count)


def krea2_stack_lora_backward_streamed_automagic3_device_grads[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_velocity: Tensor,
    blk_vec: Tensor,
    tmlp_out: Tensor,
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    fwd: Krea2StackForward,
    cos: Tensor, sin: Tensor,
    eps: Float32,
    mut opt_state: Automagic3DeviceState,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> Krea2StackDeviceGradWrite:
    comptime features = HEADS * HEADDIM

    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var fin_w = fin.as_stack_weights()
    var d_x = krea2_final_layer_backward[L, HEADS, HEADDIM](
        d_velocity, fwd, tmlp_out, fin_w, eps, ctx,
    )

    var grad_count = 0
    var streaming_sync_count = 0
    var bi = nblocks - 1
    while bi >= 0:
        var wbi: Krea2BlockWeights
        if resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
        var rb = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            fwd.block_inputs[bi].copy(), blk_vec, wbi, lora.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        var bg = krea2_single_stream_block_lora_backward_dev[L, HEADS, KVHEADS, HEADDIM](
            d_x, blk_vec, wbi, lora.blocks[bi], rb.saved,
            cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        var base = bi * KREA2_SLOTS_PER_BLOCK
        var grad_keepalive = _copy_krea2_block_grads_to_a3_state(
            bg, base, opt_state, ctx
        )
        grad_count += KREA2_SLOTS_PER_BLOCK
        d_x = bg.d_x[].clone(ctx)
        ctx.synchronize()
        streaming_sync_count += 1
        _ = len(grad_keepalive.grads)
        bi -= 1

    return Krea2StackDeviceGradWrite(TArc(d_x^), grad_count, streaming_sync_count)


def krea2_stack_dora_backward_streamed_dev[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_velocity: Tensor,
    blk_vec: Tensor,
    tmlp_out: Tensor,
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    dora: Krea2StackDirectDoRA, fin: Krea2StreamFinal,
    fwd: Krea2StackForward,
    cos: Tensor, sin: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> Krea2StackDirectDoRAGradsT:
    """Device-grad streamed backward for direct DoRA Krea2 blocks."""
    comptime features = HEADS * HEADDIM

    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var grads = List[Krea2DirectDoRAGradT]()
    for _ in range(nblocks * KREA2_SLOTS_PER_BLOCK):
        grads.append(Krea2DirectDoRAGradT(None, None, None))

    var fin_w = fin.as_stack_weights()
    var d_x = krea2_final_layer_backward[L, HEADS, HEADDIM](
        d_velocity, fwd, tmlp_out, fin_w, eps, ctx,
    )

    var bi = nblocks - 1
    while bi >= 0:
        var wbi: Krea2BlockWeights
        if resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
        var rb = krea2_single_stream_block_dora[L, HEADS, KVHEADS, HEADDIM](
            fwd.block_inputs[bi].copy(), blk_vec, wbi, dora.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        var bg = krea2_single_stream_block_dora_backward_dev[L, HEADS, KVHEADS, HEADDIM](
            d_x, blk_vec, wbi, dora.blocks[bi], rb.saved,
            cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        var base = bi * KREA2_SLOTS_PER_BLOCK
        grads[base + 0] = bg.wq.copy()
        grads[base + 1] = bg.wk.copy()
        grads[base + 2] = bg.wv.copy()
        grads[base + 3] = bg.gate_w.copy()
        grads[base + 4] = bg.wo.copy()
        grads[base + 5] = bg.mlp_gate_w.copy()
        grads[base + 6] = bg.mlp_up_w.copy()
        grads[base + 7] = bg.mlp_down_w.copy()
        d_x = bg.d_x[].clone(ctx)
        ctx.synchronize()
        bi -= 1

    return Krea2StackDirectDoRAGradsT(grads^, TArc(d_x^))


def krea2_stack_oft_backward_streamed_dev[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_velocity: Tensor,
    blk_vec: Tensor,
    tmlp_out: Tensor,
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    oft: Krea2StackDirectOFT, fin: Krea2StreamFinal,
    fwd: Krea2StackForward,
    cos: Tensor, sin: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> Krea2StackDirectOFTGradsT:
    """Device-grad streamed backward for direct OFT Krea2 blocks."""
    comptime features = HEADS * HEADDIM

    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var grads = List[Krea2DirectOFTGradT]()
    for _ in range(nblocks * KREA2_SLOTS_PER_BLOCK):
        grads.append(Krea2DirectOFTGradT(None))

    var fin_w = fin.as_stack_weights()
    var d_x = krea2_final_layer_backward[L, HEADS, HEADDIM](
        d_velocity, fwd, tmlp_out, fin_w, eps, ctx,
    )

    var bi = nblocks - 1
    while bi >= 0:
        var wbi: Krea2BlockWeights
        if resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
        var rb = krea2_single_stream_block_oft[L, HEADS, KVHEADS, HEADDIM](
            fwd.block_inputs[bi].copy(), blk_vec, wbi, oft.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        var bg = krea2_single_stream_block_oft_backward_dev[L, HEADS, KVHEADS, HEADDIM](
            d_x, blk_vec, wbi, oft.blocks[bi], rb.saved,
            cos_q, sin_q, cos_k, sin_k, eps, ctx, real_len,
        )
        var base = bi * KREA2_SLOTS_PER_BLOCK
        grads[base + 0] = bg.wq.copy()
        grads[base + 1] = bg.wk.copy()
        grads[base + 2] = bg.wv.copy()
        grads[base + 3] = bg.gate_w.copy()
        grads[base + 4] = bg.wo.copy()
        grads[base + 5] = bg.mlp_gate_w.copy()
        grads[base + 6] = bg.mlp_up_w.copy()
        grads[base + 7] = bg.mlp_down_w.copy()
        d_x = bg.d_x[].clone(ctx)
        ctx.synchronize()
        bi -= 1

    return Krea2StackDirectOFTGradsT(grads^, TArc(d_x^))


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4b — autograd_v2 ENGINE backward (the COARSE per-block graph arm)
# ══════════════════════════════════════════════════════════════════════════════
def krea2_stack_lora_backward_graph[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_velocity: Tensor,        # [1, imglen, out_ch] upstream grad on the image tokens
    blk_vec: Tensor,           # [1, 6*features]  block modulation vec
    tmlp_out: Tensor,          # [1, 1, features] final-layer tvec
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    fwd: Krea2StackForward,
    cos: Tensor, sin: Tensor,  # [L, HEADDIM/2] per-token RoPE table (untiled)
    eps: Float32,
    ctx: DeviceContext,
    mut scratch: ScratchRingAllocator,
    real_len: Optional[Int] = Optional[Int](None),
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> Krea2StackLoraGrads:
    """autograd_v2 ENGINE replacement for krea2_stack_lora_backward_streamed
    ([[feedback_all_trainers_autograd_v2]] mandate; Phase 4b). IDENTICAL conductor
    loop (final-layer frozen bwd → walk N single-stream blocks deepest→shallowest,
    RE-LOAD each block's frozen weights from the resident fp8 store / disk stream,
    scatter 8 dA/dB, carry d_x, per-block ctx.synchronize() async-free seam) — but
    each block's (recompute forward + block backward) pair is driven through the
    per-block COARSE graph (krea2_single_stream_block_graph_backward), which records
    ONE composite OPK_KREA2_SINGLE_BLOCK node and calls the WHOLE block backward
    oracle via execute_krea2_block. Returns the SAME flat LoRA grads + d_combined the
    streamed conductor returns, BIT-identical (the recompute is folded INTO the
    coarse arm; the conductor no longer recomputes here). real_len/resident MUST
    match the forward call (threaded into the re-run forward inside the arm AND the
    flash-padmask block backward), exactly as the streamed path requires."""
    comptime features = HEADS * HEADDIM

    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var grads = List[Krea2LoraGrad]()
    for _ in range(nblocks * KREA2_SLOTS_PER_BLOCK):
        grads.append(Krea2LoraGrad(None, None))

    # final-layer backward (frozen) → d into the last single-stream output. Identical
    # to the streamed path (the engine arm covers ONLY the per-block stack; the
    # final-layer chain is frozen and stays the hand-chain call).
    var fin_w = fin.as_stack_weights()
    var d_x = krea2_final_layer_backward[L, HEADS, HEADDIM](
        d_velocity, fwd, tmlp_out, fin_w, eps, ctx,
    )

    var bi = nblocks - 1
    while bi >= 0:
        # fp8-resident: dequant from the resident store (NO disk). Else stream H2D.
        var wbi: Krea2BlockWeights
        if resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)  # H2D this block
        # COARSE per-block graph: records ONE composite node from the SAVED input,
        # recomputes the forward + calls the WHOLE block backward oracle inside the
        # engine apply arm (bit-identical to the streamed conductor's recompute +
        # block backward pair). d_out for this block is the carried d_x.
        var bg = krea2_single_stream_block_graph_backward[L, HEADS, KVHEADS, HEADDIM](
            TArc(Tensor(d_x.buf.copy(), d_x.shape(), d_x.dtype())),
            fwd.block_inputs[bi].copy(), blk_vec, wbi, lora.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, scratch, real_len,
        )
        var base = bi * KREA2_SLOTS_PER_BLOCK
        grads[base + 0] = bg.wq.copy()
        grads[base + 1] = bg.wk.copy()
        grads[base + 2] = bg.wv.copy()
        grads[base + 3] = bg.gate_w.copy()
        grads[base + 4] = bg.wo.copy()
        grads[base + 5] = bg.mlp_gate_w.copy()
        grads[base + 6] = bg.mlp_up_w.copy()
        grads[base + 7] = bg.mlp_down_w.copy()
        d_x = bg.d_x[].clone(ctx)
        ctx.synchronize()   # async free discipline (same seam as the streamed path)
        bi -= 1
        # wbi drops here → device weights free before the next block loads.

    return Krea2StackLoraGrads(grads^, TArc(d_x^))


# ══════════════════════════════════════════════════════════════════════════════
# StepSlab SEGMENTED engine backward — the alloc-free engine+slab+FLASH arm.
# IDENTICAL conductor loop, but each block's backward is the 2-segment activation-
# checkpointed slab recorder (krea2_single_stream_block_graph_backward_seg), which
# bounds the per-block slab to ONE segment (~6.65GB, fits the 12GB fp8 base on 24GB
# — the whole-block slab was 12.2GB, didn't fit). One StepSlab is allocated ONCE and
# RESET per block (the segments rewind internally; reset returns it to base so every
# block sees the identical allocation sequence). The returned grads are host lists
# (carrier b: the segmented arm does the batched to_host) → the conductor scatters
# them exactly as the graph arm does. Bit-identical to the hand-chain in MATH mode
# (KREA2_SLAB_FLASH=False, the gate); FLASH (trainer) grads are value-tolerance.
# ══════════════════════════════════════════════════════════════════════════════
def krea2_stack_lora_backward_graph_slab[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_velocity: Tensor,
    blk_vec: Tensor,
    tmlp_out: Tensor,
    st: ShardedSafeTensors, key_prefix: String, nblocks: Int,
    lora: Krea2StackLora, fin: Krea2StreamFinal,
    fwd: Krea2StackForward,
    cos: Tensor, sin: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    mut slab: StepSlab,
    real_len: Optional[Int] = Optional[Int](None),
    resident: Optional[Krea2ResidentFp8] = Optional[Krea2ResidentFp8](None),
    resident_sq: Optional[Krea2ResidentSquareq] = Optional[Krea2ResidentSquareq](None),
) raises -> Krea2StackLoraGrads:
    comptime features = HEADS * HEADDIM
    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var grads = List[Krea2LoraGrad]()
    for _ in range(nblocks * KREA2_SLOTS_PER_BLOCK):
        grads.append(Krea2LoraGrad(None, None))

    var fin_w = fin.as_stack_weights()
    var d_x = krea2_final_layer_backward[L, HEADS, HEADDIM](
        d_velocity, fwd, tmlp_out, fin_w, eps, ctx,
    )

    var bi = nblocks - 1
    while bi >= 0:
        var wbi: Krea2BlockWeights
        if resident_sq and bi < len(resident_sq.value().blocks):
            wbi = _load_krea2_block_squareq(resident_sq.value(), bi, ctx)
        elif resident and bi < len(resident.value().blocks):
            wbi = _load_krea2_block_resident(resident.value(), bi, ctx)
        else:
            wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)
        slab.reset()   # block starts from the slab base (segments rewind internally)
        # SEGMENTED (option 2, ~6.65GB/segment, fits) vs WHOLE-BLOCK (option 1,
        # ~12.2GB, expected setup-OOM) — comptime-selected by KREA2_SLAB_SEGMENTED.
        var bg: Krea2BlockGrads
        comptime if KREA2_SLAB_SEGMENTED:
            bg = krea2_single_stream_block_graph_backward_seg[L, HEADS, KVHEADS, HEADDIM](
                TArc(Tensor(d_x.buf.copy(), d_x.shape(), d_x.dtype())),
                fwd.block_inputs[bi].copy(), blk_vec, wbi, lora.blocks[bi],
                cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, slab, real_len,
            )
        else:
            bg = krea2_single_stream_block_graph_backward_slab[L, HEADS, KVHEADS, HEADDIM](
                TArc(Tensor(d_x.buf.copy(), d_x.shape(), d_x.dtype())),
                fwd.block_inputs[bi].copy(), blk_vec, wbi, lora.blocks[bi],
                cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx, slab, real_len,
            )
        var base = bi * KREA2_SLOTS_PER_BLOCK
        grads[base + 0] = bg.wq.copy()
        grads[base + 1] = bg.wk.copy()
        grads[base + 2] = bg.wv.copy()
        grads[base + 3] = bg.gate_w.copy()
        grads[base + 4] = bg.wo.copy()
        grads[base + 5] = bg.mlp_gate_w.copy()
        grads[base + 6] = bg.mlp_up_w.copy()
        grads[base + 7] = bg.mlp_down_w.copy()
        d_x = bg.d_x[].clone(ctx)
        ctx.synchronize()
        bi -= 1

    return Krea2StackLoraGrads(grads^, TArc(d_x^))
