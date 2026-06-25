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
# ── SCOPE (LoRA backward path — from the architecture) ────────────────────────
# Krea2 forward: first(img) → text-fusion (12-layer ctx) → single-stream ×N →
# last. LoRA lives ONLY on the N single-stream blocks. So the LoRA backward is:
#   d_velocity → final-layer bwd (FROZEN, d_x only) → single-stream block bwd ×N
#   (Phase-1's backward: LoRA dA/dB + d_x carry) → STOP.
# The text-fusion blocks + first/embedders are BEFORE the single-stream blocks →
# frozen-skip (no LoRA there; their d_x is not needed). NO text-fusion backward.
#
# Because the whole gated span runs F32 (Phase-1's discipline; the per-block gate
# proves the chain rule at cos≥0.999), this stack carries F32 activations and the
# parity gate (krea2_stack_parity.mojo) compares vs an F32 ai-toolkit oracle.
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

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors

comptime TArc = ArcPointer[Tensor]

# ── reused forward ops (final layer = krea2_dit's last; tiling = krea2_dit) ────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.tensor_algebra import (
    reshape, slice, concat, add, zeros_device,
)
from serenitymojo.models.dit.krea2_dit import (
    krea2_last_layer, krea2_simple_modulation, _reshape_chunk_to_vec,
    _tile_rope_table,
)

# ── reused backward arms (final-layer chain; all pre-built + gated elsewhere) ──
from serenitymojo.ops.linalg_backward import linear_backward_dx
from serenitymojo.ops.norm_backward import rms_norm_backward
from serenitymojo.ops.elementwise_backward import modulate_backward

# ── Phase-1 block unit (the composed primitive) ───────────────────────────────
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, Krea2LoraGrad, Krea2BlockGrads,
    krea2_single_stream_block_lora, krea2_single_stream_block_lora_backward,
    _add_scale_one,
)


# ══════════════════════════════════════════════════════════════════════════════
# CARRIERS
# ══════════════════════════════════════════════════════════════════════════════

# Per-block frozen weights + per-block LoRA, flat-indexed by block. Krea2BlockWeights
# / Krea2BlockLora are Copyable, so a List of them is fine (they hold TArc, not bare
# Tensor). N blocks of each.
struct Krea2StackWeights(Copyable, Movable):
    var blocks: List[Krea2BlockWeights]   # len == N
    # final-layer (last) frozen params.
    var last_norm: TArc        # [features] F32  (last.norm.scale)
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
    var last_xn: TArc                     # [1,L,features] rms_norm(x_blocks_out)
    var txtlen: Int                       # slice offset for the image tokens
    var imglen: Int

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


# Flat LoRA grads, parallel to Krea2StackLora.blocks: block bi slot s at bi*8+s.
# Slot order matches Krea2BlockLora field order: wq wk wv gate wo mlp_gate mlp_up mlp_down.
comptime KREA2_SLOTS_PER_BLOCK = 8

struct Krea2StackLoraGrads(Movable):
    var grads: List[Krea2LoraGrad]        # len N*8, flat (bi*8 + slot)
    var d_combined: TArc                  # [1,L,features] grad into the block-stack input

    def __init__(out self, var grads: List[Krea2LoraGrad], var d_combined: TArc):
        self.grads = grads^
        self.d_combined = d_combined^


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
) raises -> Krea2StackForward:
    """Single-stream STACK forward WITH LoRA, saving per-block inputs for recompute.

    Composes the verified Phase-1 block forward over N blocks (N = len(w.blocks)),
    then krea2_last_layer. cos/sin are the per-token rope table (built once by the
    caller from pos); tiled here to q (HEADS) and k (KVHEADS). NO mask: the gate's
    sequence is unpadded (L == LFULL, mult of 256 in the reference → block SDPA is
    the all-true-mask == no-mask path the Phase-1 block computes). Returns velocity
    sliced to the image tokens [1, imglen, out_ch]."""
    comptime features = HEADS * HEADDIM
    var n = len(w.blocks)

    # tile the per-token rope table once for q/k (Phase-1 block contract).
    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var x = combined.copy()
    var block_inputs = List[TArc]()
    for bi in range(n):
        block_inputs.append(x.copy())                          # SAVE the block input
        var fwd = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            x, blk_vec, w.blocks[bi], lora.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx,
        )
        x = fwd.out.copy()

    var x_blocks_out = x.copy()                                 # last single-stream output

    # final = last_layer(x, tmlp_out): (1+scale)*rms_norm(x) + shift → linear.
    # We need rms_norm(x) saved for the final-layer backward (modulate_backward
    # needs the pre-modulate normed acts).
    var last_xn = rms_norm(
        x[], _add_scale_one(w.last_norm[], ctx), eps, ctx,
    )                                                          # [1,L,features]
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
        xn          = rms_norm(x, last.norm.scale + 1)
        xm          = (1+scale)*xn + shift
        velocity    = Linear(xm)[:, txtlen:txtlen+imglen]

    Steps: scatter d_velocity into a full [1,L,out_ch] (zeros on txt+pad rows) →
    linear_backward_dx → modulate_backward (drop param grads) → rms_norm_backward."""
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
        var zh = zeros_device([1, head, out_ch], STDtype.F32, ctx)
        var zt = zeros_device([1, tail, out_ch], STDtype.F32, ctx)
        var part = concat(1, ctx, zh, d_velocity)
        d_final2 = concat(1, ctx, part, zt)
    elif head > 0:
        var zh = zeros_device([1, head, out_ch], STDtype.F32, ctx)
        d_final2 = concat(1, ctx, zh, d_velocity)
    elif tail > 0:
        var zt = zeros_device([1, tail, out_ch], STDtype.F32, ctx)
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
    # Mixed precision: cast grad-in + scale to the (F32) acts dtype so
    # modulate_backward is dtype-consistent (the fwd modulate was F32 here).
    var mb = modulate_backward(cast_tensor(d_xm3, fwd.last_xn[].dtype(), ctx), fwd.last_xn[], cast_tensor(scale, fwd.last_xn[].dtype(), ctx), ctx, compute_param_grads=False)

    # 4) xn = rms_norm(x, last.norm+1) (FROZEN weight) → d_x. (clone out of the
    # Movable RmsNormBackward to avoid a partial-move of its d_x field.)
    # Mixed precision: x_blocks_out (last single-stream output) is bf16, last.norm
    # scale is F32. The forward last-layer rms_norm casts the scale down to the act
    # dtype (norm.mojo:173-174); mirror it here so rms_norm_backward runs the
    # all-bf16 path (go=mb.d_x bf16, x bf16, weight bf16) instead of the F32-acts
    # mixed path that would raise. F32 gate: cast is F32→F32 (no-op).
    var rb = rms_norm_backward(
        mb.d_x, fwd.x_blocks_out[], cast_tensor(_add_scale_one(w.last_norm[], ctx), fwd.x_blocks_out[].dtype(), ctx), eps, ctx,
    )
    var d_x_out = rb.d_x.clone(ctx)
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
) raises -> Krea2StackLoraGrads:
    """LoRA stack backward. Runs the final-layer backward (frozen) to get d into the
    last block's output, then walks the N single-stream blocks deepest→shallowest:
    for each, RE-RUN the Phase-1 block forward from the saved input (regenerate its
    acts), run the Phase-1 block backward (LoRA dA/dB + d_x carry), scatter the 8
    dA/dB at bi*8+slot, carry d_x to the shallower block. Returns the flat LoRA grads
    + d_combined (grad into the block-stack input — load-bearing, proves the chain)."""
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
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx,
        )
        # Phase-1 block backward: LoRA dA/dB (8) + d_x carry.
        var bg = krea2_single_stream_block_lora_backward[L, HEADS, KVHEADS, HEADDIM](
            d_x, blk_vec, w.blocks[bi], lora.blocks[bi], rb.saved,
            cos_q, sin_q, cos_k, sin_k, eps, ctx,
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

    return Krea2StackLoraGrads(grads^, TArc(d_x^))


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
# matmul weights bf16 (= reference v.to(bf16)); norm/mod scales bf16-rounded then
# F32 (= reference bf16(scale).float()), consumed by the F32-internal rms_norm.
# ══════════════════════════════════════════════════════════════════════════════

# bf16-resident matmul weight (real disk dtype; bf16-round if F32/F16 on disk).
def _stream_wb(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_view_as_bf16(st.tensor_view(key), ctx))


# norm/mod scale: bf16-rounded then upcast F32 (= reference bf16(scale).float();
# krea2_rmsnorm/modulate need F32 and add +1.0 in F32 via _add_scale_one).
from serenitymojo.ops.cast import cast_tensor


def _stream_scale(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> TArc:
    return TArc(cast_tensor(
        Tensor.from_view_as_bf16(st.tensor_view(key), ctx), STDtype.F32, ctx
    ))


def _load_krea2_block_streamed(
    st: ShardedSafeTensors, bi: Int, key_prefix: String, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    """Load block `bi`'s FROZEN weights H2D for one fwd/bwd iteration (caller frees
    by dropping the returned struct at loop end). Keys == krea2_forward's per-block
    load (krea2_dit.mojo:1469-1481). Matmul weights bf16, norm/mod scales bf16->F32."""
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


# Small one-time frozen `last.*` params (loaded ONCE by the caller; not streamed).
struct Krea2StreamFinal(Copyable, Movable):
    var last_norm: TArc        # [features] F32  (last.norm.scale, bf16-rounded->F32)
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
) raises -> Krea2StackForward:
    """STREAMING single-stream stack forward WITH LoRA. Identical math to
    krea2_stack_lora_forward, but each block's FROZEN weights are loaded H2D from
    `st` inside the loop and freed at iteration end (peak = one block's weights +
    acts, NOT all 28 resident). `lora` is the small trainable LoRA set (kept
    resident). Saves per-block inputs for the backward's recompute."""
    comptime features = HEADS * HEADDIM

    var cos_q = _tile_rope_table(cos, L, HEADS, HEADDIM // 2, ctx)
    var sin_q = _tile_rope_table(sin, L, HEADS, HEADDIM // 2, ctx)
    var cos_k = _tile_rope_table(cos, L, KVHEADS, HEADDIM // 2, ctx)
    var sin_k = _tile_rope_table(sin, L, KVHEADS, HEADDIM // 2, ctx)

    var x = combined.copy()
    var block_inputs = List[TArc]()
    for bi in range(nblocks):
        block_inputs.append(x.copy())                          # SAVE the block input
        var wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)  # H2D this block
        var fwd = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            x, blk_vec, wbi, lora.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx,
        )
        x = fwd.out.copy()
        # wbi drops here → its device weights free before the next block loads.

    var x_blocks_out = x.copy()

    var last_xn = rms_norm(
        x[], _add_scale_one(fin.last_norm[], ctx), eps, ctx,
    )
    var final = krea2_last_layer(
        x[], tmlp_out, fin.last_norm[], fin.last_mod_lin[],
        fin.last_lin_w[], fin.last_lin_b[], features, ctx,
    )
    var velocity = slice(final, 1, txtlen, imglen, ctx)        # [1, imglen, out_ch]

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
) raises -> Krea2StackLoraGrads:
    """STREAMING LoRA stack backward. Identical to krea2_stack_lora_backward but
    the final-layer + every per-block FROZEN weight load streams from `st` (peak =
    one block resident). final-layer bwd (frozen) → walk N single-stream blocks
    deepest→shallowest, RE-LOAD + RE-RUN each block's forward from its saved input,
    run the Phase-1 block backward, scatter 8 dA/dB, carry d_x. Returns flat LoRA
    grads + d_combined."""
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
        var wbi = _load_krea2_block_streamed(st, bi, key_prefix, ctx)  # H2D this block
        # recompute the block forward from the SAVED input to regenerate acts.
        var rb = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
            fwd.block_inputs[bi].copy(), blk_vec, wbi, lora.blocks[bi],
            cos, sin, cos_q, sin_q, cos_k, sin_k, eps, ctx,
        )
        var bg = krea2_single_stream_block_lora_backward[L, HEADS, KVHEADS, HEADDIM](
            d_x, blk_vec, wbi, lora.blocks[bi], rb.saved,
            cos_q, sin_q, cos_k, sin_k, eps, ctx,
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
        # wbi drops here → device weights free before the next block loads.

    return Krea2StackLoraGrads(grads^, TArc(d_x^))
