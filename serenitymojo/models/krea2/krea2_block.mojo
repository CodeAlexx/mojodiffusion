# models/krea2/krea2_block.mojo — Krea-2-Raw SingleStreamBlock TRAINING unit.
#
# Forward (saving activations) + hand-chained backward for ONE krea2
# SingleStreamBlock, with LoRA on the 8 block nn.Linears. This MIRRORS the
# verified inference forward in models/dit/krea2_dit.mojo
# (`krea2_single_stream_block`, line 812) op-for-op; it does NOT re-derive the
# math. It is the Klein single_block.mojo save-acts + hand-chain template
# adapted to krea2's three differences from a FLUX single block:
#   (1) GQA  — k/v have KVHEADS<HEADS heads, repeat_kv'd before SDPA (and
#       sum-reduced in backward via repeat_kv_backward).
#   (2) a SIGMOID GATE on the attention output: a = wo(sdpa(...) * sigmoid(gate)),
#       where gate = gate_proj(xm). New backward path: mul + sigmoid_backward.
#   (3) RMSNorm weight = scale + 1.0 (the F32 reparam, mmdit.py:172-177). We pass
#       `scale + 1` as the rms weight; the prenorm/postnorm/qknorm scales are
#       FROZEN (not LoRA targets), so their grads aren't needed.
#
# AdaLN-Zero double branch (mmdit.py SingleStreamBlock.forward, 328-337):
#   prescale,preshift,pregate,postscale,postshift,postgate = mod(vec)   # raw chunks
#   x1 = x + pregate  * attn ((1+prescale )*prenorm (x)  + preshift)
#   x2 = x1 + postgate * mlp ((1+postscale)*postnorm(x1) + postshift)
#   attn(y) = wo( sdpa(QKNorm+RoPE+GQA on wq/wk/wv(y)) * sigmoid(gate(y)) )
#   mlp(y)  = down( silu(gate(y)) * up(y) )
#
# LoRA on all 8 Linears (krea2.py:148 target_lora_modules=["SingleStreamDiT"],
# resolved to the per-block nn.Linears by lora_special.py): wq wk wv gate wo
# (attention) + mlp_gate mlp_up mlp_down. LoRA math == the Klein lora_block
# helper: y' = linear(x,W) + scale*((x@Aᵀ)@Bᵀ), A=[rank,in], B=[out,rank].
# (We REUSE LoraAdapterDevice + the klein_lora unfused fwd/bwd — they are
# model-agnostic LoRA-on-one-Linear primitives.)
#
# Historical parity gates also exercise F32 oracle tensors, but product training
# preserves BF16/F16/FP8 storage boundaries. This block may use F32 internally for
# reductions, score math, and optimizer-bound gradients; it must return/store
# model activations and LoRA params in their boundary dtype.
#
# Mojo 1.0.0b1: `def` only; Tensor move-only (Movable structs, no Tensor in a
# collection); no-bias linear = linear(x, w, Optional[Tensor](None), ctx).

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

comptime TArc = ArcPointer[Tensor]

# autograd_v2 slab-block attn sdpa switch (krea2_block_graph.mojo slab recorder):
# False = MATH sdpa_nomask (deterministic, the BIT GATE path — bit-exact vs the
# hand-chain). True = cuDNN FLASH (O(L), the PRODUCTION/TRAINER path; flash dQ is
# nondeterministic → value-tolerance grads, NOT bit). DEFAULT False so the block bit
# gate proves the slab block math-exact; the trainer flips it for the flash fit/speed
# path (the math O(L²) scores make the math attn 13.4GB — doesn't fit with the 12GB
# fp8 base on 24GB; flash O(L) ~2.2GB fits). Only the slab recorder reads it; the
# hand-chain is untouched.
comptime KREA2_SLAB_FLASH = False

# autograd_v2 slab-conductor path: True = the SEGMENTED (2-segment activation
# checkpoint) per-block backward (per-segment slab ~6.65GB, MEASURED to fit ~22GB/24GB
# — the safe path). False = the WHOLE-BLOCK slab recorder (no segmentation; MEASURED
# slab.peak_bytes = 12.23GB at L=4864 flash → 12GB fp8 + 12.2GB > 24GB, EXPECTED to
# OOM at setup — kept selectable so the trainer run can confirm/refute the fit
# directly, per the lead's "the trainer run IS the fit test"). DEFAULT True (the
# fitting path); flip to False to test the whole-block path.
comptime KREA2_SLAB_SEGMENTED = True

# Hand-chain LoRA launch reduction: batch the shared-input LoRA down-projection
# GEMMs for groups that all see the same x. Enabled for krea2 after the torch
# block fixture stayed at 11-nines and the 512px sync smoke improved 4.8-4.9 -> 4.7 s/step.
comptime KREA2_BATCH_LORA_GROUPS = True

# ── forward ops ──────────────────────────────────────────────────────────────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.int8_linear import int8_linear_fwd, int8_linear_bwd
from serenitymojo.ops.int8_quant import int8_transpose
from serenitymojo.ops.activations import swiglu, sigmoid
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask, sdpa_chunked
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd_padmask_bf16, sdpa_flash_backward_padmask_bf16,
)
from serenitymojo.ops.gqa_backward import repeat_kv_f32, repeat_kv_backward
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, slice, concat, add, mul, mul_scalar, zeros_device,
)
from serenitymojo.ops.cast import cast_tensor

# ── backward arms (all pre-built + gated elsewhere) ──────────────────────────
from serenitymojo.ops.linalg_backward import (
    linear_backward_dx, linear_backward_dw,
)
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward, SwigluGrads
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.norm_backward import rms_norm_backward_dg
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, GateResidualGrads, rope_backward,
)
from serenitymojo.ops.activation_backward import sigmoid_backward_from_output
from serenitymojo.models.dit.krea2_dit import (
    krea2_rmsnorm,
    krea2_rmsnorm_backward_dx,
)

# ── LoRA primitive (model-agnostic LoRA-on-one-Linear) ───────────────────────
from serenitymojo.models.klein.lora_block import (
    LoraAdapterDevice,
    klein_lora_fwd_device_resident_unfused,
    klein_lora_bwd_device_resident_unfused,
    klein_lora_bwd_device_resident_tensors_unfused,
    KleinLoraDeviceGrads,
    KleinLoraDeviceGradTensors,
)
from serenitymojo.training.dora_substitution_device import (
    DoRAAdapterDevice, DoRADeviceGrads,
)
from serenitymojo.training.oft_serenity_trainer_device import OFTOTDeviceGrads
from serenitymojo.models.krea2.krea2_direct_lycoris_stack import (
    Krea2BlockDirectDoRA, Krea2BlockDirectOFT, Krea2DirectOFTDeviceSlot,
    krea2_direct_dora_projection_forward_resident,
    krea2_direct_dora_projection_backward_resident,
    krea2_direct_oft_projection_forward_resident,
    krea2_direct_oft_projection_backward_resident,
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _no_bias() -> Optional[Tensor]:
    return Optional[Tensor](None)


def _add_scale_one(scale: Tensor, ctx: DeviceContext) raises -> Tensor:
    """RMSNorm weight reparam: weight = scale + 1.0 (mmdit.py:175). scale is the
    raw [D] parameter in the checkpoint/storage dtype; we materialize
    (scale+1) as the rms_norm weight without forcing an F32 storage boundary.
    The [1] one broadcasts against the [D]/[Dh] scale."""
    var o = List[Float32]()
    o.append(Float32(1.0))
    var one = Tensor.from_host(o^, [1], scale.dtype(), ctx)
    return add(scale, one, ctx)


def _lora_fwd(
    x: Tensor, lo: Optional[LoraAdapterDevice], M: Int, ctx: DeviceContext
) raises -> Optional[Tensor]:
    """LoRA delta scale*((x@Aᵀ)@Bᵀ) for one Linear, or None if no adapter."""
    if lo:
        return Optional[Tensor](
            klein_lora_fwd_device_resident_unfused(x, lo.value(), M, ctx)
        )
    return Optional[Tensor](None)


# ══════════════════════════════════════════════════════════════════════════════
# OminiControl EDIT — COND-ROW LoRA ROUTING (seam B.2 of
# training/krea2_omini_layout.mojo; intake §1.3/§3.4)
#
# OminiControl attaches the adapter to the CONDITION branch only (trainer.py:57
# `adapter_names=[None, None, "default"]`, confirmed by their own comment at
# trainer.py:370). Text and image rows run the FROZEN BASE. Concretely, at every
# one of the 8 block Linears:
#     y = x @ Wᵀ                                   over the FULL sequence
#     y[:, c_off : c_off+c_len] += scale * ((x_c @ Aᵀ) @ Bᵀ),  x_c = x[:, c_off:…]
# i.e. the base matmul shape is unchanged and only the low-rank delta shrinks
# from L rows to c_len rows (a strict FLOP saving, not a cost).
#
# The two helpers below carry a SENTINEL: c_off < 0 means "no cond routing", and
# both then reduce EXACTLY to the pre-existing `_lora_fwd` + `add` — the same
# kernels with the same arguments, so every existing caller stays bit-for-bit
# unchanged (this is what keeps the CONDLEN=0 build bit-equal).
#
# NOTE the coupling this deliberately does NOT prevent: the img rows' OUTPUT does
# change even though no delta is added to them, because the cond rows' K/V feed
# bidirectional attention (intake §1.4). That is the method working — the oracle
# measures it (`[coupling] block-out max|delta| img rows`) and asserts it nonzero.
# ══════════════════════════════════════════════════════════════════════════════
def _lora_delta_rows(
    x: Tensor, lo: Optional[LoraAdapterDevice], M: Int,
    c_off: Int, c_len: Int, ctx: DeviceContext,
) raises -> Optional[Tensor]:
    """LoRA delta for one Linear. c_off < 0 -> the whole-sequence delta [1,M,out]
    (identical to _lora_fwd). c_off >= 0 -> the delta of the COND ROW SLICE only,
    returned as [1, c_len, out] for _add_delta_rows to scatter back."""
    if lo:
        if c_off < 0:
            return Optional[Tensor](
                klein_lora_fwd_device_resident_unfused(x, lo.value(), M, ctx)
            )
        var xc = slice(x, 1, c_off, c_len, ctx)
        return Optional[Tensor](
            klein_lora_fwd_device_resident_unfused(xc, lo.value(), c_len, ctx)
        )
    return Optional[Tensor](None)


def _row_dim(t: Tensor) raises -> Int:
    """The ROW axis of a block tensor: dim 1 for the rank-3 [1, L, C] forward
    activations, dim 0 for the rank-2 [L, C] tensors the backward's
    ops/linalg_backward.linear_backward_dx allocates ([M, in_features], :631).
    C4 added this so the same scatter/slice helpers serve both."""
    return 0 if len(t.shape()) == 2 else 1


def _rows(t: Tensor, off: Int, n: Int, ctx: DeviceContext) raises -> Tensor:
    """Row slice [off, off+n) on whichever axis `_row_dim` names."""
    return slice(t, _row_dim(t), off, n, ctx)


def _add_delta_rows(
    y: Tensor, d: Tensor, c_off: Int, c_len: Int, ctx: DeviceContext
) raises -> Tensor:
    """c_off < 0 -> plain y + d (unchanged path). c_off >= 0 -> SCATTER-ADD: d is
    [1,c_len,out] and lands on rows [c_off, c_off+c_len) only; the head and tail
    rows are the frozen base output copied through byte-for-byte.
    The row axis comes from `_row_dim` — for the rank-3 forward tensors this is
    dim 1, i.e. BYTE-FOR-BYTE the pre-C4 helper; rank 2 is the C4 backward's
    linear_backward_dx output."""
    if c_off < 0:
        return add(y, d, ctx)
    var rdim = _row_dim(y)
    var rows = y.shape()[rdim]
    var mid = add(slice(y, rdim, c_off, c_len, ctx), d, ctx)
    var tail_len = rows - c_off - c_len
    if c_off == 0:
        if tail_len == 0:
            return mid^
        return concat(rdim, ctx, mid, slice(y, rdim, c_off + c_len, tail_len, ctx))
    if tail_len == 0:
        return concat(rdim, ctx, slice(y, rdim, 0, c_off, ctx), mid)
    return concat(
        rdim, ctx, slice(y, rdim, 0, c_off, ctx), mid,
        slice(y, rdim, c_off + c_len, tail_len, ctx),
    )


def _linear_lora(
    x: Tensor, w: Tensor, lo: Optional[LoraAdapterDevice], M: Int, ctx: DeviceContext,
    w8: Optional[TArc] = Optional[TArc](None),
    w8_scale: Optional[TArc] = Optional[TArc](None),
    c_off: Int = -1,          # OminiControl EDIT cond-row LoRA routing (C3):
    c_len: Int = 0,           # -1 (default) = the pre-existing full-sequence path.
) raises -> Tensor:
    """y = linear(x,W) [no bias] + LoRA delta (if present). When w8/w8_scale are
    present the FROZEN base runs int8 W8A8 (int8_linear_fwd, x cast to BF16 ==
    reference trainer's bf16 activations) instead of the bf16 `linear`; the LoRA delta stays bf16.
    c_off >= 0 restricts the delta to rows [c_off, c_off+c_len) — the base matmul
    is untouched and still runs the FULL sequence."""
    if w8:
        var base: Tensor
        if x.dtype() == STDtype.BF16:
            base = int8_linear_fwd(x, w8.value()[], w8_scale.value()[], ctx)
        else:
            var xb = cast_tensor(x, STDtype.BF16, ctx)
            base = int8_linear_fwd(xb, w8.value()[], w8_scale.value()[], ctx)
        var d = _lora_delta_rows(x, lo, M, c_off, c_len, ctx)
        if d:
            base = _add_delta_rows(base, d.value(), c_off, c_len, ctx)
        return base^
    var nb = _no_bias()
    var y = linear(x, w, nb^, ctx)
    var d2 = _lora_delta_rows(x, lo, M, c_off, c_len, ctx)
    if d2:
        y = _add_delta_rows(y, d2.value(), c_off, c_len, ctx)
    return y^


# ── int8 W8A8 base-matmul dispatch (cfg.quantized_resident=="int_w8a8") ─────────
# When blk.int8 is present, the 8 frozen base matmuls run int8×int8→int32 GEMM
# (no per-step fp8 dequant) via the parity-gated ops/int8_linear primitives; else
# the unchanged bf16 `linear`. idx = weight position in Krea2BlockWeights field
# order (wq=0 wk=1 wv=2 gate_w=3 wo=4 mlp_gate_w=5 mlp_up_w=6 mlp_down_w=7).
def _base_fwd(
    x: Tensor, w_bf: Tensor, blk: Krea2BlockWeights, idx: Int, ctx: DeviceContext
) raises -> Tensor:
    """Frozen base forward y = x @ W[idx]ᵀ (no bias): int8 W8A8 or bf16."""
    if blk.int8:
        ref p = blk.int8.value()
        if x.dtype() == STDtype.BF16:
            return int8_linear_fwd(x, p.w8[idx][], p.scale[idx][], ctx)
        var xb = cast_tensor(x, STDtype.BF16, ctx)
        return int8_linear_fwd(xb, p.w8[idx][], p.scale[idx][], ctx)
    var nb = _no_bias()
    return linear(x, w_bf, nb^, ctx)


# int8 weight / scale accessors for the _linear_lora sites (None when bf16).
def _i8w(blk: Krea2BlockWeights, idx: Int) raises -> Optional[TArc]:
    if blk.int8:
        return Optional[TArc](blk.int8.value().w8[idx].copy())
    return Optional[TArc](None)


def _i8s(blk: Krea2BlockWeights, idx: Int) raises -> Optional[TArc]:
    if blk.int8:
        return Optional[TArc](blk.int8.value().scale[idx].copy())
    return Optional[TArc](None)


# Frozen base backward dX = d_y @ W (contract N): int8 W8A8 when the int8 weight
# w8[N,K] is present, else the bf16 linear_backward_dx. Transposes w8 → w8T[K,N]
# on the fly with the TILED byte transpose (32x32 shared tile, ~0.2ms) and runs
# the fast NT IMMA GEMM (i16832, 387µs). MEASURED faster than the no-transpose
# NN GEMM (ops/int8_linear.int8_linear_bwd_nn — cuBLAS 13 accepts int8 NN but
# picks a forwardCompat wmma kernel at 1348µs, 3.5× slower; nsys 2026-07-07).
# Both paths are bit-identical (gated in ops/tests/int8_linear_parity.mojo).
# Base is FROZEN → dX only (no d_w), == reference trainer. d_y cast BF16 (== reference trainer bf16 grads).
def _base_dx(
    d_y: Tensor, w_bf: Tensor, M: Int, in_f: Int, out_f: Int,
    w8: Optional[TArc], w8_scale: Optional[TArc], ctx: DeviceContext,
) raises -> Tensor:
    if w8:
        var w8t = int8_transpose(w8.value()[], ctx)   # [N,K] → [K,N], tiled, transient
        if d_y.dtype() == STDtype.BF16:
            return int8_linear_bwd(d_y, w8t, w8_scale.value()[], ctx)
        var db = cast_tensor(d_y, STDtype.BF16, ctx)
        return int8_linear_bwd(db, w8t, w8_scale.value()[], ctx)
    return linear_backward_dx(d_y, w_bf, M, in_f, out_f, ctx)


def _tensor_to_host_f32_local(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    if t.dtype() == STDtype.F32:
        return t.to_host(ctx)
    var t32 = cast_tensor(t, STDtype.F32, ctx)
    return t32.to_host(ctx)


# ── trainable weights (FROZEN base + per-Linear LoRA adapters) ───────────────
# Base projection matrices are torch Linear weight layout [out, in]; rmsnorm/mod
# params are the RAW [D]/[Dh]/[6D] vectors (we add +1 to the rms scales inside).
# int8 W8A8 payload for one block's 8 frozen matmul weights (field order:
# wq wk wv gate_w wo mlp_gate_w mlp_up_w mlp_down_w). Carried OPTIONALLY inside
# Krea2BlockWeights: when present, the block's base matmuls dispatch to
# int8_linear_fwd/bwd (no per-step fp8 dequant) instead of the bf16 `linear`.
# Per weight i: w8[i]=int8 [N,K] (fwd B), w8t[i]=int8 [K,N] (bwd B), scale[i]=F32
# scalar tensorwise scale [1]. Base weight is FROZEN → grad_input only (== reference trainer).
struct Krea2BlockInt8(Copyable, Movable):
    var w8: List[TArc]     # len 8: int8 [out,in]=[N,K] (ONE orientation)
    var scale: List[TArc]  # len 8: F32 scalar [1]

    def __init__(
        out self, var w8: List[TArc], var scale: List[TArc],
    ):
        self.w8 = w8^
        self.scale = scale^


struct Krea2BlockWeights(Copyable, Movable):
    var wq: TArc          # [HEADS*HEADDIM, features]
    var wk: TArc          # [KVHEADS*HEADDIM, features]
    var wv: TArc          # [KVHEADS*HEADDIM, features]
    var gate_w: TArc      # [features, features]
    var wo: TArc          # [features, features]
    var mlp_gate_w: TArc  # [mlpdim, features]
    var mlp_up_w: TArc    # [mlpdim, features]
    var mlp_down_w: TArc  # [features, mlpdim]
    var qnorm_scale: TArc # [HEADDIM] raw
    var knorm_scale: TArc # [HEADDIM] raw
    var prenorm_scale: TArc   # [features] raw
    var postnorm_scale: TArc  # [features] raw
    var mod_lin: TArc     # [6*features] (DoubleSharedModulation.lin)
    # int8 W8A8 base (cfg.quantized_resident=="int_w8a8"). None (default, C13) =
    # bf16 base via `linear` (UNCHANGED). Present = the 8 base matmuls run int8;
    # the 8 bf16 wX fields above are then unused dummies (the int8 loader fills
    # them with tiny placeholders — only these int8 tensors carry the real base).
    var int8: Optional[Krea2BlockInt8]

    def __init__(
        out self,
        var wq: TArc, var wk: TArc, var wv: TArc, var gate_w: TArc, var wo: TArc,
        var mlp_gate_w: TArc, var mlp_up_w: TArc, var mlp_down_w: TArc,
        var qnorm_scale: TArc, var knorm_scale: TArc,
        var prenorm_scale: TArc, var postnorm_scale: TArc, var mod_lin: TArc,
        var int8: Optional[Krea2BlockInt8] = Optional[Krea2BlockInt8](None),
    ):
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^
        self.qnorm_scale = qnorm_scale^
        self.knorm_scale = knorm_scale^
        self.prenorm_scale = prenorm_scale^
        self.postnorm_scale = postnorm_scale^
        self.mod_lin = mod_lin^
        self.int8 = int8^


# The 8 LoRA adapters (each Optional so the block reduces to the frozen base
# when all are absent). Order matches the 8 target Linears.
struct Krea2BlockLora(Copyable, Movable):
    var wq: Optional[LoraAdapterDevice]
    var wk: Optional[LoraAdapterDevice]
    var wv: Optional[LoraAdapterDevice]
    var gate_w: Optional[LoraAdapterDevice]
    var wo: Optional[LoraAdapterDevice]
    var mlp_gate_w: Optional[LoraAdapterDevice]
    var mlp_up_w: Optional[LoraAdapterDevice]
    var mlp_down_w: Optional[LoraAdapterDevice]

    def __init__(
        out self,
        var wq: Optional[LoraAdapterDevice], var wk: Optional[LoraAdapterDevice],
        var wv: Optional[LoraAdapterDevice], var gate_w: Optional[LoraAdapterDevice],
        var wo: Optional[LoraAdapterDevice],
        var mlp_gate_w: Optional[LoraAdapterDevice],
        var mlp_up_w: Optional[LoraAdapterDevice],
        var mlp_down_w: Optional[LoraAdapterDevice],
    ):
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^


# ── saved activations (device-resident) ──────────────────────────────────────
struct Krea2BlockSaved(Copyable, Movable):
    var x: TArc          # [1,L,features] block input
    var xm: TArc         # [1,L,features] modulate(prenorm(x))  — attn-proj input
    var q_pre: TArc      # [1,L,HEADS,HEADDIM]   wq(xm) reshaped (pre-QKNorm)
    var k_pre: TArc      # [1,L,KVHEADS,HEADDIM] wk(xm)
    var v: TArc          # [1,L,KVHEADS,HEADDIM] wv(xm) (untouched by QKNorm)
    var q_rope: TArc     # [1,L,HEADS,HEADDIM]   rope(qnorm(q_pre))
    var k_rope: TArc     # [1,L,KVHEADS,HEADDIM] rope(knorm(k_pre))
    var k_full: TArc     # [1,L,HEADS,HEADDIM]   repeat_kv(k_rope)
    var v_full: TArc     # [1,L,HEADS,HEADDIM]   repeat_kv(v)
    var attn_flat: TArc  # [1,L,features]        sdpa(...) merged
    var gate_pre: TArc   # [1,L,features]        gate_w(xm) (pre-sigmoid)
    var sg: TArc         # [1,L,features]        sigmoid(gate_pre)
    var gated: TArc      # [1,L,features]        attn_flat * sg  (wo input)
    var a: TArc          # [1,L,features]        wo(gated) — the attn branch output (gate_residual y)
    var x1: TArc         # [1,L,features]        x + pregate*attn
    var xm2: TArc        # [1,L,features]        modulate(postnorm(x1)) — mlp input
    var mlp_gate: TArc   # [1,L,mlpdim]          mlp_gate_w(xm2)
    var mlp_up: TArc     # [1,L,mlpdim]          mlp_up_w(xm2)
    var sw: TArc         # [1,L,mlpdim]          swiglu(mlp_gate, mlp_up) — down input
    var m: TArc          # [1,L,features]        mlp_down(sw) — mlp branch output (gate_residual y)
    # the rms-normed (pre-modulate) activations needed for modulate_backward
    var xn: TArc         # [1,L,features] prenorm(x)
    var xn2: TArc        # [1,L,features] postnorm(x1)
    # ── flash-padmask saved set (Phase: length-bucket flash training) ──────────
    # Present ONLY when the masked/padded SDPA ran the cuDNN flash-padmask path
    # (real_len < L). The flash backward consumes the bf16 q/k/v/o + F32 stats
    # WITHOUT recompute (= klein KLEIN_SDPA_FLASH tape pattern). On the no-pad
    # (full-attn) path these are None and the backward uses sdpa_backward
    # (BIT-IDENTICAL to the pre-flash block — the F32 parity gate guard).
    var flash_q: Optional[TArc]   # [1,L,HEADS,Dh] bf16
    var flash_k: Optional[TArc]   # [1,L,HEADS,Dh] bf16 (post-GQA k_full)
    var flash_v: Optional[TArc]   # [1,L,HEADS,Dh] bf16 (post-GQA v_full)
    var flash_o: Optional[TArc]   # [1,L,HEADS,Dh] bf16 padded SDPA output
    var flash_stats: Optional[TArc]  # [1,HEADS,L,1] F32 softmax LSE

    def __init__(
        out self,
        var x: TArc, var xm: TArc,
        var q_pre: TArc, var k_pre: TArc, var v: TArc,
        var q_rope: TArc, var k_rope: TArc, var k_full: TArc, var v_full: TArc,
        var attn_flat: TArc, var gate_pre: TArc, var sg: TArc, var gated: TArc,
        var a: TArc, var x1: TArc, var xm2: TArc,
        var mlp_gate: TArc, var mlp_up: TArc, var sw: TArc, var m: TArc,
        var xn: TArc, var xn2: TArc,
        var flash_q: Optional[TArc] = Optional[TArc](None),
        var flash_k: Optional[TArc] = Optional[TArc](None),
        var flash_v: Optional[TArc] = Optional[TArc](None),
        var flash_o: Optional[TArc] = Optional[TArc](None),
        var flash_stats: Optional[TArc] = Optional[TArc](None),
    ):
        self.x = x^
        self.xm = xm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.v = v^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.k_full = k_full^
        self.v_full = v_full^
        self.attn_flat = attn_flat^
        self.gate_pre = gate_pre^
        self.sg = sg^
        self.gated = gated^
        self.a = a^
        self.x1 = x1^
        self.xm2 = xm2^
        self.mlp_gate = mlp_gate^
        self.mlp_up = mlp_up^
        self.sw = sw^
        self.m = m^
        self.xn = xn^
        self.xn2 = xn2^
        self.flash_q = flash_q^
        self.flash_k = flash_k^
        self.flash_v = flash_v^
        self.flash_o = flash_o^
        self.flash_stats = flash_stats^


struct Krea2BlockForward(Movable):
    var out: TArc            # [1,L,features] device-resident block output
    var saved: Krea2BlockSaved

    def __init__(out self, var out: TArc, var saved: Krea2BlockSaved):
        self.out = out^
        self.saved = saved^


# ── per-Linear LoRA grad pair (host F32; None when adapter absent) ────────────
struct Krea2LoraGrad(Copyable, Movable):
    var d_a: Optional[List[Float32]]
    var d_b: Optional[List[Float32]]

    def __init__(
        out self, var d_a: Optional[List[Float32]], var d_b: Optional[List[Float32]]
    ):
        self.d_a = d_a^
        self.d_b = d_b^


# ── backward result ──────────────────────────────────────────────────────────
struct Krea2BlockGrads(Movable):
    var d_x: TArc                 # input grad [1,L,features]
    # the 8 LoRA dA/dB (None when the adapter is absent)
    var wq: Krea2LoraGrad
    var wk: Krea2LoraGrad
    var wv: Krea2LoraGrad
    var gate_w: Krea2LoraGrad
    var wo: Krea2LoraGrad
    var mlp_gate_w: Krea2LoraGrad
    var mlp_up_w: Krea2LoraGrad
    var mlp_down_w: Krea2LoraGrad
    # ── OminiControl EDIT per-segment modulation grads (C4) ──────────────────
    # PRESENT ONLY when the backward ran the per-segment path (vec_cond + a valid
    # cond_off). F32 [6*features], in `_mod6` chunk order. d_vec_t is the grad of
    # the mods(t) vector, accumulated over the TXT_real+IMG rows [0, cond_off);
    # d_vec_cond is the grad of the mods(t=0) vector, over the COND rows
    # [cond_off, real_len) — the TXT_pad tail is deliberately excluded, see the
    # PER-SEGMENT MODULATION BACKWARD box. They feed the trainer's two SEPARATE
    # temb chains. Both are None on the uniform path, where nothing downstream
    # consumes a modulation grad (LoRA training freezes mod.lin) and the backward
    # runs the pre-C4 kernels with compute_param_grads/compute_gate_grad False.
    var d_vec_t: Optional[TArc]
    var d_vec_cond: Optional[TArc]

    def __init__(
        out self, var d_x: TArc,
        var wq: Krea2LoraGrad, var wk: Krea2LoraGrad, var wv: Krea2LoraGrad,
        var gate_w: Krea2LoraGrad, var wo: Krea2LoraGrad,
        var mlp_gate_w: Krea2LoraGrad, var mlp_up_w: Krea2LoraGrad,
        var mlp_down_w: Krea2LoraGrad,
        var d_vec_t: Optional[TArc] = Optional[TArc](None),
        var d_vec_cond: Optional[TArc] = Optional[TArc](None),
    ):
        self.d_x = d_x^
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^
        self.d_vec_t = d_vec_t^
        self.d_vec_cond = d_vec_cond^


# ── DEVICE-resident per-Linear LoRA grad pair (TArc; None when adapter absent) ─
# Sibling of Krea2LoraGrad. The d_A/d_B stay on device (no per-adapter to_host),
# and the streamed stack either copies a block's 8 pairs to host under the
# per-block streaming fence or copies them D2D into shared AdamW state in the
# krea2devicegrad path. The HOST Krea2LoraGrad above is the bit-gate oracle and
# is left untouched.
struct Krea2LoraGradT(Copyable, Movable):
    var d_a: Optional[TArc]
    var d_b: Optional[TArc]

    def __init__(
        out self, var d_a: Optional[TArc], var d_b: Optional[TArc]
    ):
        self.d_a = d_a^
        self.d_b = d_b^


struct Krea2BlockGradsT(Movable):
    var d_x: TArc                 # input grad [1,L,features]
    var wq: Krea2LoraGradT
    var wk: Krea2LoraGradT
    var wv: Krea2LoraGradT
    var gate_w: Krea2LoraGradT
    var wo: Krea2LoraGradT
    var mlp_gate_w: Krea2LoraGradT
    var mlp_up_w: Krea2LoraGradT
    var mlp_down_w: Krea2LoraGradT
    # OminiControl EDIT per-segment modulation grads (C4) — same contract as
    # Krea2BlockGrads.d_vec_t / .d_vec_cond above (F32 [6*features], present
    # ONLY on the per-segment path).
    var d_vec_t: Optional[TArc]
    var d_vec_cond: Optional[TArc]

    def __init__(
        out self, var d_x: TArc,
        var wq: Krea2LoraGradT, var wk: Krea2LoraGradT, var wv: Krea2LoraGradT,
        var gate_w: Krea2LoraGradT, var wo: Krea2LoraGradT,
        var mlp_gate_w: Krea2LoraGradT, var mlp_up_w: Krea2LoraGradT,
        var mlp_down_w: Krea2LoraGradT,
        var d_vec_t: Optional[TArc] = Optional[TArc](None),
        var d_vec_cond: Optional[TArc] = Optional[TArc](None),
    ):
        self.d_x = d_x^
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^
        self.d_vec_t = d_vec_t^
        self.d_vec_cond = d_vec_cond^


# ── modulation: out = vec + lin; chunk 6 along last dim → 6 raw [features] ────
# (mmdit.py DoubleSharedModulation.forward). vec [1,6F], lin [6F]; b==1 so each
# chunk is [features]; reshape to a clean [features] param for modulate/gate.
def _mod6(
    vec: Tensor, mod_lin: Tensor, features: Int, ctx: DeviceContext
) raises -> List[TArc]:
    var s = add(vec, mod_lin, ctx)          # [1, 6F] (lin [6F] broadcasts)
    var out = List[TArc]()
    for i in range(6):
        var c = slice(s, 1, i * features, features, ctx)   # [1, features]
        out.append(TArc(reshape_owned(c^, [features])))    # [features]
    return out^


# ── BATCH-2 modulation: vec [2, 6F], lin [6F] broadcasts. Returns 6 [2, features]
# per-sample chunks (adaLN-per-sample: modulate/residual_gate split the 2L rows into
# two contiguous L ranges, one per sample — see ops/elementwise modulate [B,D]). ────
def _mod6_b2(
    vec: Tensor, mod_lin: Tensor, features: Int, ctx: DeviceContext
) raises -> List[TArc]:
    var s = add(vec, mod_lin, ctx)          # [2, 6F] (lin [6F] broadcasts over rows)
    var out = List[TArc]()
    for i in range(6):
        var c = slice(s, 1, i * features, features, ctx)   # [2, features]
        out.append(TArc(c^))                               # KEEP [2, features]
    return out^


# ══════════════════════════════════════════════════════════════════════════════
# OminiControl EDIT — PER-SEGMENT MODULATION (seam B.1 of
# training/krea2_omini_layout.mojo; intake §3.3)
#
# The EDIT row layout is [TXT_real(lt) | IMG | COND | TXT_pad] and the CONDITION
# rows ride temb(t=0) while text+image ride temb(t) (OminiControl trainer.py:232
# `timesteps=[t, t] + [zeros]*len(conditions)`). That is ONE row boundary
# `split` == Krea2OminiLayout.cond_off(): rows [0, split) use the mods(t) chunks,
# rows [split, L) use the mods(t=0) chunks. The TXT_pad tail therefore sits in
# the t=0 span — matching the ai-toolkit krea2 reference exactly (mmdit.py
# SingleStreamBlock.forward tuple-vec branch `(vec, refvec, split)`); pad-row
# values are never read downstream, so this is a free, divergence-removing
# choice.
#
# Both helpers below are the SAME kernels as the whole-sequence `modulate` /
# `residual_gate`, run once per span. With identical chunks on both spans they
# are bit-equal to the unsegmented call (the slice/concat are pure byte moves).
# ══════════════════════════════════════════════════════════════════════════════
def _modulate_seg2(
    x: Tensor, scale_t: Tensor, shift_t: Tensor,
    scale_c: Tensor, shift_c: Tensor, split: Int, ctx: DeviceContext,
) raises -> Tensor:
    """(1+scale)*x + shift with (scale_t,shift_t) on rows [0,split) and
    (scale_c,shift_c) on rows [split,L). x is [1, L, features]."""
    var rows = x.shape()[1]
    var head = slice(x, 1, 0, split, ctx)
    var tail = slice(x, 1, split, rows - split, ctx)
    var mh = modulate(head, scale_t, shift_t, ctx)
    var mt = modulate(tail, scale_c, shift_c, ctx)
    return concat(1, ctx, mh, mt)


def _residual_gate_seg2(
    x: Tensor, gate_t: Tensor, gate_c: Tensor, y: Tensor,
    split: Int, ctx: DeviceContext,
) raises -> Tensor:
    """x + gate*y with gate_t on rows [0,split) and gate_c on rows [split,L)."""
    var rows = x.shape()[1]
    var xh = slice(x, 1, 0, split, ctx)
    var xt = slice(x, 1, split, rows - split, ctx)
    var yh = slice(y, 1, 0, split, ctx)
    var yt = slice(y, 1, split, rows - split, ctx)
    var rh = residual_gate(xh, gate_t, yh, ctx)
    var rt = residual_gate(xt, gate_c, yt, ctx)
    return concat(1, ctx, rh, rt)


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD (saves activations) — mirrors krea2_dit.mojo:812-859 + LoRA on 8 Linears
# ══════════════════════════════════════════════════════════════════════════════
def krea2_single_stream_block_lora[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    x_t: TArc,            # [1, L, features] F32
    vec: Tensor,          # [1, 6*features] F32  (timestep modulation vec)
    w: Krea2BlockWeights, lora: Krea2BlockLora,
    cos: Tensor, sin: Tensor,   # [L, HEADDIM/2] per-token RoPE table
    cos_q: Tensor, sin_q: Tensor,   # [L*HEADS, HEADDIM/2]   tiled for BSHD q
    cos_k: Tensor, sin_k: Tensor,   # [L*KVHEADS, HEADDIM/2] tiled for BSHD k
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),  # length-bucket pad: the VALID
        # contiguous-prefix length of the [0:real_len] real tokens; [real_len:L] is
        # text-pad. None (or real_len == L) = full attention via sdpa_nomask
        # (BIT-IDENTICAL to the pre-mask block — the F32 parity gate guard). Present
        # & < L = cuDNN flash-padmask SDPA: cuDNN masks the [real_len:L] tail rows
        # internally (NO materialized [1,H,L,L] mask, NO materialized scores). The
        # token order MUST be [valid(0:real_len) | pad(real_len:L)] — see the
        # trainer's [TXT_real | IMG | TXT_pad] reorder. real_len threads to bwd too.
    vec_cond: Optional[TArc] = Optional[TArc](None),  # OminiControl EDIT (C2):
        # the SECOND modulation vector, tproj(tmlp(temb(t=0))) [1, 6*features],
        # for the CONDITION rows. Absent = the unchanged uniform-modulation path.
    cond_off: Optional[Int] = Optional[Int](None),    # the per-segment split row
        # == Krea2OminiLayout.cond_off(); pass
        # training/krea2_omini_layout.krea2_omini_mod_split(lay), which returns
        # -1 when the layout has NO condition segment so the guard below falls
        # through to the pre-existing code path BIT-FOR-BIT (the CONDLEN=0
        # regression contract). Only 0 < cond_off < L enables segmentation.
        #
        # ⚠ SCOPE (updated by C4). The BATCH-1 backwards —
        # `krea2_single_stream_block_lora_backward` and
        # `..._backward_dev` — now take the SAME three switches and implement the
        # per-segment / cond-row backward (per-span d_mod, cond-row dA/dB,
        # full-sequence dX). Pass them there too: a per-segment FORWARD with a
        # uniform BACKWARD is silently wrong on the condition rows.
        # The BATCH-2 entry points (`..._b2`, `..._backward_b2`,
        # `..._backward_b2_dev`) do NOT support the EDIT layout in either
        # direction — the b2 forward has no vec_cond/cond_off/cond_len at all,
        # and its [2, features] per-sample modulation would need a 4-way
        # (sample x segment) split that ops/elementwise `modulate` does not
        # express. b2 + EDIT is not a reachable configuration today; building it
        # starts with the b2 FORWARD.
        # Nothing in the trainer passes these arguments yet.
    cond_len: Optional[Int] = Optional[Int](None),    # OminiControl EDIT (C3):
        # the CONDITION segment length == Krea2OminiLayout.cond_len() (S_COND).
        # Present together with a valid cond_off it turns on COND-ROW LoRA
        # ROUTING at all 8 Linears: the frozen base still runs the FULL sequence
        # and only the low-rank delta is computed on rows
        # [cond_off, cond_off+cond_len) and scatter-added back there
        # (OminiControl trainer.py:57 adapter_names [None, None, "default"]).
        # ABSENT — or cond_len <= 0, or cond_off <= 0, or cond_off+cond_len > L
        # — leaves the pre-existing full-sequence LoRA path BIT-FOR-BIT intact
        # (the CONDLEN=0 regression contract). Note cond_len is INDEPENDENT of
        # vec_cond: modulation segmentation and LoRA routing are separate
        # switches, so each can be gated on its own.
    attn_bias: Optional[TArc] = Optional[TArc](None),  # OminiControl
        # `condition_scale` (C8, INFERENCE ONLY): an additive [1, HEADS, L, L]
        # score bias in x's dtype, built ONCE per forward by
        # krea2_cache_reader.krea2_build_edit_attn_bias and shared by all blocks.
        # It carries BOTH the TXT_pad key-column mask AND the log(condition_scale)
        # cross-bias between COND and non-COND tokens (flux_omini.py:280-341).
        #
        # ABSENT (the default, and the ONLY thing the trainer ever passes) => not
        # one instruction of the SDPA dispatch below changes: flash-padmask when
        # real_len < L, sdpa_nomask otherwise. PRESENT => the masked math SDPA
        # (sdpa_chunked) runs INSTEAD of both, because neither existing arm has a
        # per-element logit hook. `real_len` is then IGNORED for masking (the bias
        # tensor already contains the pad columns) — passing both is not an error,
        # the bias simply wins.
        #
        # ⚠ NO BACKWARD. The biased arm saves no flash tape and nothing in this
        # repo differentiates through it; `krea2_single_stream_block_lora_backward`
        # has no attn_bias argument. Training must never set this.
        # ⚠ UNVERIFIED ON DEVICE as of C8 (authored while the GPU was busy). See
        # the C8 verification plan's condition_scale gate.
) raises -> Krea2BlockForward:
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var M = L                              # rows for LoRA (b==1, [L, features])
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    # mod(vec) → 6 raw chunks.
    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var preshift = mods[1]
    var pregate = mods[2]
    var postscale = mods[3]
    var postshift = mods[4]
    var postgate = mods[5]

    # OminiControl EDIT per-segment modulation (opt-in). Enabled ONLY when both
    # vec_cond and a split strictly inside (0, L) are supplied; otherwise every
    # line below runs exactly as it did before this chunk.
    var seg_mod = False
    var split = 0
    var mods_c = List[TArc]()
    if vec_cond:
        if cond_off:
            split = cond_off.value()
            if split > 0 and split < L:
                mods_c = _mod6(vec_cond.value()[], w.mod_lin[], features, ctx)
                seg_mod = True

    # OminiControl EDIT COND-ROW LoRA ROUTING (opt-in, C3). c_off < 0 is the
    # sentinel for "not routed": every LoRA site below then calls exactly the
    # kernels it called before this chunk, on the full sequence. Enabled only
    # when a cond_len is supplied AND the [cond_off, cond_off+cond_len) window is
    # a strict interior slice of the sequence. The FROZEN BASE is never
    # restricted — it always runs all L rows.
    var c_off = -1
    var c_len = 0
    if cond_off:
        if cond_len:
            var co = cond_off.value()
            var cl = cond_len.value()
            if co > 0 and cl > 0 and co + cl <= L:
                c_off = co
                c_len = cl

    # ── ATTENTION branch ─────────────────────────────────────────────────────
    # xm = (1+prescale)*prenorm(x) + preshift
    var xn = krea2_rmsnorm(x_t[], w.prenorm_scale[], eps, ctx)
    var xm: Tensor
    if seg_mod:
        xm = _modulate_seg2(
            xn, prescale[], preshift[], mods_c[0][], mods_c[1][], split, ctx
        )                                                       # [1,L,features]
    else:
        xm = modulate(xn, prescale[], preshift[], ctx)          # [1,L,features]

    # projections (+ LoRA). xm is [1,L,features]; linear treats leading dims as rows.
    var q = _base_fwd(xm, w.wq[], w, 0, ctx)             # [1,L,HEADS*HEADDIM]
    var k = _base_fwd(xm, w.wk[], w, 1, ctx)             # [1,L,KVHEADS*HEADDIM]
    var v_lin = _base_fwd(xm, w.wv[], w, 2, ctx)         # [1,L,KVHEADS*HEADDIM]
    var gate_pre = _base_fwd(xm, w.gate_w[], w, 3, ctx)  # [1,L,features]
    var qkvg_grouped = False
    comptime if KREA2_BATCH_LORA_GROUPS:
        if lora.wq:
            if lora.wk:
                if lora.wv:
                    if lora.gate_w:
                        var lq = lora.wq.value().copy()
                        var lk = lora.wk.value().copy()
                        var lv = lora.wv.value().copy()
                        var lg = lora.gate_w.value().copy()
                        if (
                            lq.rank == lk.rank and lq.rank == lv.rank and lq.rank == lg.rank
                            and lq.in_f == lk.in_f and lq.in_f == lv.in_f and lq.in_f == lg.in_f
                        ):
                            var a_stack = concat(0, ctx, lq.a[], lk.a[], lv.a[], lg.a[])
                            # Cond-row routing shrinks the SHARED down-projection
                            # input from L rows to c_len rows; grouping is kept.
                            var t_stack: Tensor
                            if c_off >= 0:
                                var xc = slice(xm, 1, c_off, c_len, ctx)
                                var nb_ac = _no_bias()
                                t_stack = linear(xc, a_stack, nb_ac^, ctx)
                            else:
                                var nb_a = _no_bias()
                                t_stack = linear(xm, a_stack, nb_a^, ctx)
                            var last_dim = len(t_stack.shape()) - 1
                            var tq = slice(t_stack, last_dim, 0, lq.rank, ctx)
                            var tk = slice(t_stack, last_dim, lq.rank, lk.rank, ctx)
                            var tv = slice(t_stack, last_dim, 2 * lq.rank, lv.rank, ctx)
                            var tg = slice(t_stack, last_dim, 3 * lq.rank, lg.rank, ctx)

                            var nb_bq = _no_bias()
                            q = _add_delta_rows(q, mul_scalar(linear(tq, lq.b[], nb_bq^, ctx), lq.scale, ctx), c_off, c_len, ctx)
                            var nb_bk = _no_bias()
                            k = _add_delta_rows(k, mul_scalar(linear(tk, lk.b[], nb_bk^, ctx), lk.scale, ctx), c_off, c_len, ctx)
                            var nb_bv = _no_bias()
                            v_lin = _add_delta_rows(v_lin, mul_scalar(linear(tv, lv.b[], nb_bv^, ctx), lv.scale, ctx), c_off, c_len, ctx)
                            var nb_bg = _no_bias()
                            gate_pre = _add_delta_rows(gate_pre, mul_scalar(linear(tg, lg.b[], nb_bg^, ctx), lg.scale, ctx), c_off, c_len, ctx)
                            qkvg_grouped = True
    if not qkvg_grouped:
        var dq = _lora_delta_rows(xm, lora.wq, M, c_off, c_len, ctx)
        if dq:
            q = _add_delta_rows(q, dq.value(), c_off, c_len, ctx)
        var dk = _lora_delta_rows(xm, lora.wk, M, c_off, c_len, ctx)
        if dk:
            k = _add_delta_rows(k, dk.value(), c_off, c_len, ctx)
        var dv = _lora_delta_rows(xm, lora.wv, M, c_off, c_len, ctx)
        if dv:
            v_lin = _add_delta_rows(v_lin, dv.value(), c_off, c_len, ctx)
        var dg = _lora_delta_rows(xm, lora.gate_w, M, c_off, c_len, ctx)
        if dg:
            gate_pre = _add_delta_rows(gate_pre, dg.value(), c_off, c_len, ctx)

    # reshape BSHD.
    var q_pre = reshape_owned(q^, [1, L, HEADS, HEADDIM])
    var k_pre = reshape_owned(k^, [1, L, KVHEADS, HEADDIM])
    var v = reshape_owned(v_lin^, [1, L, KVHEADS, HEADDIM])

    # QKNorm over HEADDIM (weight = scale+1); v untouched.
    var q_rms = krea2_rmsnorm(q_pre, w.qnorm_scale[], eps, ctx)
    var k_rms = krea2_rmsnorm(k_pre, w.knorm_scale[], eps, ctx)

    # RoPE on q,k (per-head tiled tables).
    var q_rope = rope_interleaved(q_rms, cos_q, sin_q, ctx)
    var k_rope = rope_interleaved(k_rms, cos_k, sin_k, ctx)

    # GQA: repeat_kv to HEADS.
    var k_full = repeat_kv_f32(k_rope, L, KVHEADS, n_rep, HEADDIM, ctx)
    var v_full = repeat_kv_f32(v, L, KVHEADS, n_rep, HEADDIM, ctx)

    # SDPA. No pad (default, or real_len == L) = full attention (sdpa_nomask) — the
    # per-block gate path, BIT-IDENTICAL to the pre-flash block. Length-bucket pad
    # (real_len present & < L) = cuDNN flash-padmask SDPA: cuDNN masks the
    # [real_len:L] tail rows internally (the token order is [valid | pad]); NO
    # materialized [1,H,L,L] mask (the 4.5GB resident the old sdpa_chunked path
    # needed), NO materialized [L,L] scores. The flash bf16 q/k/v/o + F32 stats go
    # to the tape for the flash backward (no recompute, no re-cast).
    var att: Tensor
    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    var use_flash = real_len and real_len.value() < L
    if attn_bias:
        # OminiControl condition_scale arm (C8, inference only). The bias tensor
        # already encodes the TXT_pad key columns, so flash/real_len is bypassed.
        # sdpa_chunked is bit-identical to sdpa but holds the F32 scores for ONE
        # head at a time ([L,L]) instead of all HEADS.
        var bias = attn_bias.value()
        if bias[].dtype() != q_rope.dtype():
            raise Error(
                "krea2_single_stream_block_lora: attn_bias dtype must equal q's"
                " (ops.attention.sdpa requires q.dtype == mask.dtype)"
            )
        att = sdpa_chunked[1, L, HEADS, HEADDIM](
            q_rope, k_full, v_full, bias[], scale, ctx
        )
    elif use_flash:
        var rl = real_len.value()
        var ff = sdpa_flash_train_fwd_padmask_bf16[1, L, HEADS, HEADDIM](
            q_rope, k_full, v_full, rl, scale, ctx
        )
        # ff.att is BF16 [1,L,HEADS,Dh] (pad-tail rows are masked-out garbage the
        # downstream gate zeroes via the pad d_out). Save the BF16 set for bwd.
        att = Tensor(ff.att.buf.copy(), ff.att.shape(), ff.att.dtype())
        flash_q = Optional[TArc](ff.q_bf.copy())
        flash_k = Optional[TArc](ff.k_bf.copy())
        flash_v = Optional[TArc](ff.v_bf.copy())
        flash_o = Optional[TArc](ff.o_bf.copy())
        flash_stats = Optional[TArc](ff.stats.copy())
    else:
        att = sdpa_nomask[1, L, HEADS, HEADDIM](q_rope, k_full, v_full, scale, ctx)
    var attn_flat = reshape_owned(att^, [1, L, features])

    # sigmoid gate + product, then wo.
    var sg = sigmoid(gate_pre, ctx)                             # [1,L,features]
    var gated = mul(attn_flat, sg, ctx)                        # [1,L,features]
    var a = _linear_lora(gated, w.wo[], lora.wo, M, ctx, _i8w(w, 4), _i8s(w, 4), c_off, c_len)  # [1,L,features]

    # x1 = x + pregate * a
    var x1: Tensor
    if seg_mod:
        x1 = _residual_gate_seg2(x_t[], pregate[], mods_c[2][], a, split, ctx)
    else:
        x1 = residual_gate(x_t[], pregate[], a, ctx)

    # ── MLP branch ───────────────────────────────────────────────────────────
    var xn2 = krea2_rmsnorm(x1, w.postnorm_scale[], eps, ctx)
    var xm2: Tensor
    if seg_mod:
        xm2 = _modulate_seg2(
            xn2, postscale[], postshift[], mods_c[3][], mods_c[4][], split, ctx
        )                                                      # [1,L,features]
    else:
        xm2 = modulate(xn2, postscale[], postshift[], ctx)     # [1,L,features]

    var mg = _base_fwd(xm2, w.mlp_gate_w[], w, 5, ctx)  # [1,L,mlpdim]
    var mu = _base_fwd(xm2, w.mlp_up_w[], w, 6, ctx)  # [1,L,mlpdim]
    var mlp_grouped = False
    comptime if KREA2_BATCH_LORA_GROUPS:
        if lora.mlp_gate_w:
            if lora.mlp_up_w:
                var lmg = lora.mlp_gate_w.value().copy()
                var lmu = lora.mlp_up_w.value().copy()
                if lmg.rank == lmu.rank and lmg.in_f == lmu.in_f:
                    var a_stack = concat(0, ctx, lmg.a[], lmu.a[])
                    var t_stack: Tensor
                    if c_off >= 0:
                        var xc2 = slice(xm2, 1, c_off, c_len, ctx)
                        var nb_ac = _no_bias()
                        t_stack = linear(xc2, a_stack, nb_ac^, ctx)
                    else:
                        var nb_a = _no_bias()
                        t_stack = linear(xm2, a_stack, nb_a^, ctx)
                    var last_dim = len(t_stack.shape()) - 1
                    var tmg = slice(t_stack, last_dim, 0, lmg.rank, ctx)
                    var tmu = slice(t_stack, last_dim, lmg.rank, lmu.rank, ctx)

                    var nb_bmg = _no_bias()
                    mg = _add_delta_rows(mg, mul_scalar(linear(tmg, lmg.b[], nb_bmg^, ctx), lmg.scale, ctx), c_off, c_len, ctx)
                    var nb_bmu = _no_bias()
                    mu = _add_delta_rows(mu, mul_scalar(linear(tmu, lmu.b[], nb_bmu^, ctx), lmu.scale, ctx), c_off, c_len, ctx)
                    mlp_grouped = True
    if not mlp_grouped:
        var dmg = _lora_delta_rows(xm2, lora.mlp_gate_w, M, c_off, c_len, ctx)
        if dmg:
            mg = _add_delta_rows(mg, dmg.value(), c_off, c_len, ctx)
        var dmu = _lora_delta_rows(xm2, lora.mlp_up_w, M, c_off, c_len, ctx)
        if dmu:
            mu = _add_delta_rows(mu, dmu.value(), c_off, c_len, ctx)
    var sw = swiglu(mg, mu, ctx)                                # silu(mg)*mu [1,L,mlpdim]
    var m = _linear_lora(sw, w.mlp_down_w[], lora.mlp_down_w, M, ctx, _i8w(w, 7), _i8s(w, 7), c_off, c_len)    # [1,L,features]

    var x2: Tensor
    if seg_mod:
        x2 = _residual_gate_seg2(x1, postgate[], mods_c[5][], m, split, ctx)
    else:
        x2 = residual_gate(x1, postgate[], m, ctx)             # x1 + postgate*m

    var saved = Krea2BlockSaved(
        x_t.copy(), TArc(xm^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(k_full^), TArc(v_full^),
        TArc(attn_flat^), TArc(gate_pre^), TArc(sg^), TArc(gated^),
        TArc(a^), TArc(x1^), TArc(xm2^),
        TArc(mg^), TArc(mu^), TArc(sw^), TArc(m^),
        TArc(xn^), TArc(xn2^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )
    return Krea2BlockForward(TArc(x2^), saved^)


def krea2_single_stream_block_dora[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    x_t: TArc,
    vec: Tensor,
    w: Krea2BlockWeights, dora: Krea2BlockDirectDoRA,
    cos: Tensor, sin: Tensor,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2BlockForward:
    """Krea2 SingleStreamBlock forward with direct DoRA W_eff projection hooks."""
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var preshift = mods[1]
    var pregate = mods[2]
    var postscale = mods[3]
    var postshift = mods[4]
    var postgate = mods[5]

    var xn = krea2_rmsnorm(x_t[], w.prenorm_scale[], eps, ctx)
    var xm = modulate(xn, prescale[], preshift[], ctx)

    var q = krea2_block_direct_dora_projection_forward(xm, w.wq[], dora.wq, M, ctx)
    var k = krea2_block_direct_dora_projection_forward(xm, w.wk[], dora.wk, M, ctx)
    var v_lin = krea2_block_direct_dora_projection_forward(xm, w.wv[], dora.wv, M, ctx)
    var gate_pre = krea2_block_direct_dora_projection_forward(xm, w.gate_w[], dora.gate_w, M, ctx)

    var q_pre = reshape_owned(q^, [1, L, HEADS, HEADDIM])
    var k_pre = reshape_owned(k^, [1, L, KVHEADS, HEADDIM])
    var v = reshape_owned(v_lin^, [1, L, KVHEADS, HEADDIM])

    var q_rms = krea2_rmsnorm(q_pre, w.qnorm_scale[], eps, ctx)
    var k_rms = krea2_rmsnorm(k_pre, w.knorm_scale[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos_q, sin_q, ctx)
    var k_rope = rope_interleaved(k_rms, cos_k, sin_k, ctx)

    var k_full = repeat_kv_f32(k_rope, L, KVHEADS, n_rep, HEADDIM, ctx)
    var v_full = repeat_kv_f32(v, L, KVHEADS, n_rep, HEADDIM, ctx)

    var att: Tensor
    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    var use_flash = real_len and real_len.value() < L
    if use_flash:
        var rl = real_len.value()
        var ff = sdpa_flash_train_fwd_padmask_bf16[1, L, HEADS, HEADDIM](
            q_rope, k_full, v_full, rl, scale, ctx
        )
        att = Tensor(ff.att.buf.copy(), ff.att.shape(), ff.att.dtype())
        flash_q = Optional[TArc](ff.q_bf.copy())
        flash_k = Optional[TArc](ff.k_bf.copy())
        flash_v = Optional[TArc](ff.v_bf.copy())
        flash_o = Optional[TArc](ff.o_bf.copy())
        flash_stats = Optional[TArc](ff.stats.copy())
    else:
        att = sdpa_nomask[1, L, HEADS, HEADDIM](q_rope, k_full, v_full, scale, ctx)
    var attn_flat = reshape_owned(att^, [1, L, features])

    var sg = sigmoid(gate_pre, ctx)
    var gated = mul(attn_flat, sg, ctx)
    var a = krea2_block_direct_dora_projection_forward(gated, w.wo[], dora.wo, M, ctx)

    var x1 = residual_gate(x_t[], pregate[], a, ctx)

    var xn2 = krea2_rmsnorm(x1, w.postnorm_scale[], eps, ctx)
    var xm2 = modulate(xn2, postscale[], postshift[], ctx)

    var mg = krea2_block_direct_dora_projection_forward(xm2, w.mlp_gate_w[], dora.mlp_gate_w, M, ctx)
    var mu = krea2_block_direct_dora_projection_forward(xm2, w.mlp_up_w[], dora.mlp_up_w, M, ctx)
    var sw = swiglu(mg, mu, ctx)
    var m = krea2_block_direct_dora_projection_forward(sw, w.mlp_down_w[], dora.mlp_down_w, M, ctx)

    var x2 = residual_gate(x1, postgate[], m, ctx)

    var saved = Krea2BlockSaved(
        x_t.copy(), TArc(xm^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(k_full^), TArc(v_full^),
        TArc(attn_flat^), TArc(gate_pre^), TArc(sg^), TArc(gated^),
        TArc(a^), TArc(x1^), TArc(xm2^),
        TArc(mg^), TArc(mu^), TArc(sw^), TArc(m^),
        TArc(xn^), TArc(xn2^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )
    return Krea2BlockForward(TArc(x2^), saved^)


def krea2_single_stream_block_oft[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    x_t: TArc,
    vec: Tensor,
    w: Krea2BlockWeights, oft: Krea2BlockDirectOFT,
    cos: Tensor, sin: Tensor,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2BlockForward:
    """Krea2 SingleStreamBlock forward with direct OFT projection hooks."""
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var preshift = mods[1]
    var pregate = mods[2]
    var postscale = mods[3]
    var postshift = mods[4]
    var postgate = mods[5]

    var xn = krea2_rmsnorm(x_t[], w.prenorm_scale[], eps, ctx)
    var xm = modulate(xn, prescale[], preshift[], ctx)

    var q = krea2_block_direct_oft_projection_forward(xm, w.wq[], oft.wq, M, ctx)
    var k = krea2_block_direct_oft_projection_forward(xm, w.wk[], oft.wk, M, ctx)
    var v_lin = krea2_block_direct_oft_projection_forward(xm, w.wv[], oft.wv, M, ctx)
    var gate_pre = krea2_block_direct_oft_projection_forward(xm, w.gate_w[], oft.gate_w, M, ctx)

    var q_pre = reshape_owned(q^, [1, L, HEADS, HEADDIM])
    var k_pre = reshape_owned(k^, [1, L, KVHEADS, HEADDIM])
    var v = reshape_owned(v_lin^, [1, L, KVHEADS, HEADDIM])

    var q_rms = krea2_rmsnorm(q_pre, w.qnorm_scale[], eps, ctx)
    var k_rms = krea2_rmsnorm(k_pre, w.knorm_scale[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos_q, sin_q, ctx)
    var k_rope = rope_interleaved(k_rms, cos_k, sin_k, ctx)

    var k_full = repeat_kv_f32(k_rope, L, KVHEADS, n_rep, HEADDIM, ctx)
    var v_full = repeat_kv_f32(v, L, KVHEADS, n_rep, HEADDIM, ctx)

    var att: Tensor
    var flash_q = Optional[TArc](None)
    var flash_k = Optional[TArc](None)
    var flash_v = Optional[TArc](None)
    var flash_o = Optional[TArc](None)
    var flash_stats = Optional[TArc](None)
    var use_flash = real_len and real_len.value() < L
    if use_flash:
        var rl = real_len.value()
        var ff = sdpa_flash_train_fwd_padmask_bf16[1, L, HEADS, HEADDIM](
            q_rope, k_full, v_full, rl, scale, ctx
        )
        att = Tensor(ff.att.buf.copy(), ff.att.shape(), ff.att.dtype())
        flash_q = Optional[TArc](ff.q_bf.copy())
        flash_k = Optional[TArc](ff.k_bf.copy())
        flash_v = Optional[TArc](ff.v_bf.copy())
        flash_o = Optional[TArc](ff.o_bf.copy())
        flash_stats = Optional[TArc](ff.stats.copy())
    else:
        att = sdpa_nomask[1, L, HEADS, HEADDIM](q_rope, k_full, v_full, scale, ctx)
    var attn_flat = reshape_owned(att^, [1, L, features])

    var sg = sigmoid(gate_pre, ctx)
    var gated = mul(attn_flat, sg, ctx)
    var a = krea2_block_direct_oft_projection_forward(gated, w.wo[], oft.wo, M, ctx)

    var x1 = residual_gate(x_t[], pregate[], a, ctx)

    var xn2 = krea2_rmsnorm(x1, w.postnorm_scale[], eps, ctx)
    var xm2 = modulate(xn2, postscale[], postshift[], ctx)

    var mg = krea2_block_direct_oft_projection_forward(xm2, w.mlp_gate_w[], oft.mlp_gate_w, M, ctx)
    var mu = krea2_block_direct_oft_projection_forward(xm2, w.mlp_up_w[], oft.mlp_up_w, M, ctx)
    var sw = swiglu(mg, mu, ctx)
    var m = krea2_block_direct_oft_projection_forward(sw, w.mlp_down_w[], oft.mlp_down_w, M, ctx)

    var x2 = residual_gate(x1, postgate[], m, ctx)

    var saved = Krea2BlockSaved(
        x_t.copy(), TArc(xm^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(k_full^), TArc(v_full^),
        TArc(attn_flat^), TArc(gate_pre^), TArc(sg^), TArc(gated^),
        TArc(a^), TArc(x1^), TArc(xm2^),
        TArc(mg^), TArc(mu^), TArc(sw^), TArc(m^),
        TArc(xn^), TArc(xn2^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )
    return Krea2BlockForward(TArc(x2^), saved^)


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD (hand-chained) — exact reverse of the forward graph above
# ══════════════════════════════════════════════════════════════════════════════
# d_x for a LoRA Linear + the adapter's dA/dB, in one Movable struct (Mojo Tuple
# element transfer-out is fragile, so we use an explicit struct like the rest of
# the codebase).
struct _LinBwd(Movable):
    var d_x: Tensor
    var lora: Krea2LoraGrad

    def __init__(out self, var d_x: Tensor, var lora: Krea2LoraGrad):
        self.d_x = d_x^
        self.lora = lora^


def _linear_bwd_dx(
    d_y: Tensor, x: Tensor, w: Tensor, lo: Optional[LoraAdapterDevice],
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
    w8t: Optional[TArc] = Optional[TArc](None),
    w8_scale: Optional[TArc] = Optional[TArc](None),
    c_off: Int = -1,          # OminiControl EDIT cond-row LoRA routing (C4):
    c_len: Int = 0,           # -1 (default) = the pre-existing full-sequence path.
) raises -> _LinBwd:
    """d_x for a LoRA Linear = base backward + LoRA branch d_x. Base W FROZEN (no
    d_w). When w8t/w8_scale present the base dX runs int8 W8A8; LoRA dA/dB (bf16)
    returned in the pair (None when the adapter is absent).

    C4 — COND-ROW ROUTING (c_off >= 0): the exact mirror of the C3 forward. The
    forward added the low-rank delta to rows [c_off, c_off+c_len) ONLY, computed
    from those rows' input, so the adapter's dA/dB see ONLY those rows and the
    LoRA branch's d_x lands ONLY on those rows. The FROZEN BASE dX is untouched
    and still runs the FULL sequence (the base saw every row). c_off < 0 runs the
    pre-existing kernels with the pre-existing arguments, bit-for-bit."""
    var d_x = _base_dx(d_y, w, M, in_f, out_f, w8t, w8_scale, ctx)
    if lo:
        if c_off >= 0:
            var dyc = _rows(d_y, c_off, c_len, ctx)
            var xc = _rows(x, c_off, c_len, ctx)
            var gc = klein_lora_bwd_device_resident_unfused(
                dyc, xc, lo.value(), c_len, ctx
            )
            var dxc = _add_delta_rows(d_x, gc.d_x, c_off, c_len, ctx)
            var pairc = Krea2LoraGrad(
                Optional[List[Float32]](gc.d_a.copy()),
                Optional[List[Float32]](gc.d_b.copy()),
            )
            return _LinBwd(dxc^, pairc^)
        var g = klein_lora_bwd_device_resident_unfused(d_y, x, lo.value(), M, ctx)
        d_x = add(d_x, g.d_x, ctx)
        var pair = Krea2LoraGrad(
            Optional[List[Float32]](g.d_a.copy()),
            Optional[List[Float32]](g.d_b.copy()),
        )
        return _LinBwd(d_x^, pair^)
    return _LinBwd(d_x^, Krea2LoraGrad(None, None))


# ── DEVICE-grad sibling of _LinBwd / _linear_bwd_dx ──────────────────────────
# Identical GEMM math, but the LoRA dA/dB stay on DEVICE (TArc) — no internal
# to_host. klein_lora_bwd_device_resident_tensors_unfused is the SAME unfused
# chain as klein_lora_bwd_device_resident_unfused minus the _to_host_pair_f32
# fence, so the device-grad path is bit-identical to the host path (the trainer
# proves this by the bit-identical loss gate).
struct _LinBwdT(Movable):
    var d_x: Tensor
    var lora: Krea2LoraGradT

    def __init__(out self, var d_x: Tensor, var lora: Krea2LoraGradT):
        self.d_x = d_x^
        self.lora = lora^


def _linear_bwd_dx_dev(
    d_y: Tensor, x: Tensor, w: Tensor, lo: Optional[LoraAdapterDevice],
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
    w8t: Optional[TArc] = Optional[TArc](None),
    w8_scale: Optional[TArc] = Optional[TArc](None),
    c_off: Int = -1,          # OminiControl EDIT cond-row LoRA routing (C4);
    c_len: Int = 0,           # see _linear_bwd_dx for the contract.
) raises -> _LinBwdT:
    """Device-grad d_x for a LoRA Linear. Base W FROZEN (no d_w). LoRA dA/dB stay
    on device (no per-adapter to_host) — the SAME GEMM math as _linear_bwd_dx.
    int8 W8A8 base when w8t/w8_scale present."""
    var d_x = _base_dx(d_y, w, M, in_f, out_f, w8t, w8_scale, ctx)
    if lo:
        if c_off >= 0:
            var dyc = _rows(d_y, c_off, c_len, ctx)
            var xc = _rows(x, c_off, c_len, ctx)
            var gc = klein_lora_bwd_device_resident_tensors_unfused(
                dyc, xc, lo.value(), c_len, ctx
            )
            var dxc = _add_delta_rows(d_x, gc.d_x[], c_off, c_len, ctx)
            var pairc = Krea2LoraGradT(
                Optional[TArc](gc.d_a.copy()), Optional[TArc](gc.d_b.copy()),
            )
            return _LinBwdT(dxc^, pairc^)
        var g = klein_lora_bwd_device_resident_tensors_unfused(d_y, x, lo.value(), M, ctx)
        d_x = add(d_x, g.d_x[], ctx)
        var pair = Krea2LoraGradT(
            Optional[TArc](g.d_a.copy()),
            Optional[TArc](g.d_b.copy()),
        )
        return _LinBwdT(d_x^, pair^)
    return _LinBwdT(d_x^, Krea2LoraGradT(None, None))


struct _LinBwdGroup2(Movable):
    var d_x: Tensor
    var g0: Krea2LoraGrad
    var g1: Krea2LoraGrad

    def __init__(
        out self, var d_x: Tensor, var g0: Krea2LoraGrad, var g1: Krea2LoraGrad
    ):
        self.d_x = d_x^
        self.g0 = g0^
        self.g1 = g1^


struct _LinBwdGroup4(Movable):
    var d_x: Tensor
    var g0: Krea2LoraGrad
    var g1: Krea2LoraGrad
    var g2: Krea2LoraGrad
    var g3: Krea2LoraGrad

    def __init__(
        out self, var d_x: Tensor,
        var g0: Krea2LoraGrad, var g1: Krea2LoraGrad,
        var g2: Krea2LoraGrad, var g3: Krea2LoraGrad,
    ):
        self.d_x = d_x^
        self.g0 = g0^
        self.g1 = g1^
        self.g2 = g2^
        self.g3 = g3^


struct _LinBwdGroup2T(Movable):
    var d_x: Tensor
    var g0: Krea2LoraGradT
    var g1: Krea2LoraGradT

    def __init__(
        out self, var d_x: Tensor, var g0: Krea2LoraGradT, var g1: Krea2LoraGradT
    ):
        self.d_x = d_x^
        self.g0 = g0^
        self.g1 = g1^


struct _LinBwdGroup4T(Movable):
    var d_x: Tensor
    var g0: Krea2LoraGradT
    var g1: Krea2LoraGradT
    var g2: Krea2LoraGradT
    var g3: Krea2LoraGradT

    def __init__(
        out self, var d_x: Tensor,
        var g0: Krea2LoraGradT, var g1: Krea2LoraGradT,
        var g2: Krea2LoraGradT, var g3: Krea2LoraGradT,
    ):
        self.d_x = d_x^
        self.g0 = g0^
        self.g1 = g1^
        self.g2 = g2^
        self.g3 = g3^


def _linear_bwd_dx_group2(
    d0: Tensor, d1: Tensor, x: Tensor,
    w0: Tensor, lo0: Optional[LoraAdapterDevice], out0: Int,
    w1: Tensor, lo1: Optional[LoraAdapterDevice], out1: Int,
    M: Int, in_f: Int, ctx: DeviceContext,
    w8t0: Optional[TArc] = Optional[TArc](None), s0: Optional[TArc] = Optional[TArc](None),
    w8t1: Optional[TArc] = Optional[TArc](None), s1: Optional[TArc] = Optional[TArc](None),
) raises -> _LinBwdGroup2:
    comptime if not KREA2_BATCH_LORA_GROUPS:
        var b0 = _linear_bwd_dx(d0, x, w0, lo0, M, in_f, out0, ctx, w8t0, s0)
        var b1 = _linear_bwd_dx(d1, x, w1, lo1, M, in_f, out1, ctx, w8t1, s1)
        var g0 = b0.lora.copy()
        var g1 = b1.lora.copy()
        var d_x = add(b0.d_x, b1.d_x, ctx)
        return _LinBwdGroup2(d_x^, g0^, g1^)
    if not lo0 or not lo1:
        var b0 = _linear_bwd_dx(d0, x, w0, lo0, M, in_f, out0, ctx, w8t0, s0)
        var b1 = _linear_bwd_dx(d1, x, w1, lo1, M, in_f, out1, ctx, w8t1, s1)
        var g0 = b0.lora.copy()
        var g1 = b1.lora.copy()
        var d_x = add(b0.d_x, b1.d_x, ctx)
        return _LinBwdGroup2(d_x^, g0^, g1^)

    var l0 = lo0.value().copy()
    var l1 = lo1.value().copy()
    if l0.rank != l1.rank or l0.in_f != l1.in_f:
        var b0 = _linear_bwd_dx(d0, x, w0, lo0, M, in_f, out0, ctx, w8t0, s0)
        var b1 = _linear_bwd_dx(d1, x, w1, lo1, M, in_f, out1, ctx, w8t1, s1)
        var g0 = b0.lora.copy()
        var g1 = b1.lora.copy()
        var d_x = add(b0.d_x, b1.d_x, ctx)
        return _LinBwdGroup2(d_x^, g0^, g1^)

    var dx0 = _base_dx(d0, w0, M, in_f, out0, w8t0, s0, ctx)
    var dx1 = _base_dx(d1, w1, M, in_f, out1, w8t1, s1, ctx)
    var dx_base = add(dx0, dx1, ctx)

    var a_stack = concat(0, ctx, l0.a[], l1.a[])
    var nb = _no_bias()
    var t_stack = linear(x, a_stack, nb^, ctx)
    var last_dim = len(t_stack.shape()) - 1
    var t0 = slice(t_stack, last_dim, 0, l0.rank, ctx)
    var t1 = slice(t_stack, last_dim, l0.rank, l1.rank, ctx)

    var ddy0 = mul_scalar(d0, l0.scale, ctx)
    var dt0 = linear_backward_dx(ddy0, l0.b[], M, l0.rank, out0, ctx)
    var db0 = linear_backward_dw(ddy0, t0, M, l0.rank, out0, ctx)
    var ddy1 = mul_scalar(d1, l1.scale, ctx)
    var dt1 = linear_backward_dx(ddy1, l1.b[], M, l1.rank, out1, ctx)
    var db1 = linear_backward_dw(ddy1, t1, M, l1.rank, out1, ctx)

    var dt_stack = concat(1, ctx, dt0, dt1)
    var dx_lora = linear_backward_dx(dt_stack, a_stack, M, in_f, 2 * l0.rank, ctx)
    var da_stack = linear_backward_dw(dt_stack, x, M, in_f, 2 * l0.rank, ctx)
    var da0 = slice(da_stack, 0, 0, l0.rank, ctx)
    var da1 = slice(da_stack, 0, l0.rank, l1.rank, ctx)
    var d_x = add(dx_base, dx_lora, ctx)

    return _LinBwdGroup2(
        d_x^,
        Krea2LoraGrad(
            Optional[List[Float32]](_tensor_to_host_f32_local(da0, ctx)),
            Optional[List[Float32]](_tensor_to_host_f32_local(db0, ctx)),
        ),
        Krea2LoraGrad(
            Optional[List[Float32]](_tensor_to_host_f32_local(da1, ctx)),
            Optional[List[Float32]](_tensor_to_host_f32_local(db1, ctx)),
        ),
    )


def _linear_bwd_dx_group4(
    d0: Tensor, d1: Tensor, d2: Tensor, d3: Tensor, x: Tensor,
    w0: Tensor, lo0: Optional[LoraAdapterDevice], out0: Int,
    w1: Tensor, lo1: Optional[LoraAdapterDevice], out1: Int,
    w2: Tensor, lo2: Optional[LoraAdapterDevice], out2: Int,
    w3: Tensor, lo3: Optional[LoraAdapterDevice], out3: Int,
    M: Int, in_f: Int, ctx: DeviceContext,
    w8t0: Optional[TArc] = Optional[TArc](None), s0: Optional[TArc] = Optional[TArc](None),
    w8t1: Optional[TArc] = Optional[TArc](None), s1: Optional[TArc] = Optional[TArc](None),
    w8t2: Optional[TArc] = Optional[TArc](None), s2: Optional[TArc] = Optional[TArc](None),
    w8t3: Optional[TArc] = Optional[TArc](None), s3: Optional[TArc] = Optional[TArc](None),
) raises -> _LinBwdGroup4:
    comptime if not KREA2_BATCH_LORA_GROUPS:
        var b0 = _linear_bwd_dx(d0, x, w0, lo0, M, in_f, out0, ctx, w8t0, s0)
        var b1 = _linear_bwd_dx(d1, x, w1, lo1, M, in_f, out1, ctx, w8t1, s1)
        var b2 = _linear_bwd_dx(d2, x, w2, lo2, M, in_f, out2, ctx, w8t2, s2)
        var b3 = _linear_bwd_dx(d3, x, w3, lo3, M, in_f, out3, ctx, w8t3, s3)
        var g0 = b0.lora.copy()
        var g1 = b1.lora.copy()
        var g2 = b2.lora.copy()
        var g3 = b3.lora.copy()
        var d_x = add(add(b0.d_x, b1.d_x, ctx), add(b2.d_x, b3.d_x, ctx), ctx)
        return _LinBwdGroup4(d_x^, g0^, g1^, g2^, g3^)
    if not lo0 or not lo1 or not lo2 or not lo3:
        var b0 = _linear_bwd_dx(d0, x, w0, lo0, M, in_f, out0, ctx, w8t0, s0)
        var b1 = _linear_bwd_dx(d1, x, w1, lo1, M, in_f, out1, ctx, w8t1, s1)
        var b2 = _linear_bwd_dx(d2, x, w2, lo2, M, in_f, out2, ctx, w8t2, s2)
        var b3 = _linear_bwd_dx(d3, x, w3, lo3, M, in_f, out3, ctx, w8t3, s3)
        var g0 = b0.lora.copy()
        var g1 = b1.lora.copy()
        var g2 = b2.lora.copy()
        var g3 = b3.lora.copy()
        var d_x = add(add(b0.d_x, b1.d_x, ctx), add(b2.d_x, b3.d_x, ctx), ctx)
        return _LinBwdGroup4(d_x^, g0^, g1^, g2^, g3^)

    var l0 = lo0.value().copy()
    var l1 = lo1.value().copy()
    var l2 = lo2.value().copy()
    var l3 = lo3.value().copy()
    if (
        l0.rank != l1.rank or l0.rank != l2.rank or l0.rank != l3.rank
        or l0.in_f != l1.in_f or l0.in_f != l2.in_f or l0.in_f != l3.in_f
    ):
        var b0 = _linear_bwd_dx(d0, x, w0, lo0, M, in_f, out0, ctx, w8t0, s0)
        var b1 = _linear_bwd_dx(d1, x, w1, lo1, M, in_f, out1, ctx, w8t1, s1)
        var b2 = _linear_bwd_dx(d2, x, w2, lo2, M, in_f, out2, ctx, w8t2, s2)
        var b3 = _linear_bwd_dx(d3, x, w3, lo3, M, in_f, out3, ctx, w8t3, s3)
        var g0 = b0.lora.copy()
        var g1 = b1.lora.copy()
        var g2 = b2.lora.copy()
        var g3 = b3.lora.copy()
        var d_x = add(add(b0.d_x, b1.d_x, ctx), add(b2.d_x, b3.d_x, ctx), ctx)
        return _LinBwdGroup4(d_x^, g0^, g1^, g2^, g3^)

    var dx0 = _base_dx(d0, w0, M, in_f, out0, w8t0, s0, ctx)
    var dx1 = _base_dx(d1, w1, M, in_f, out1, w8t1, s1, ctx)
    var dx2 = _base_dx(d2, w2, M, in_f, out2, w8t2, s2, ctx)
    var dx3 = _base_dx(d3, w3, M, in_f, out3, w8t3, s3, ctx)
    var dx_base = add(add(dx0, dx1, ctx), add(dx2, dx3, ctx), ctx)

    var a_stack = concat(0, ctx, l0.a[], l1.a[], l2.a[], l3.a[])
    var nb = _no_bias()
    var t_stack = linear(x, a_stack, nb^, ctx)
    var last_dim = len(t_stack.shape()) - 1
    var t0 = slice(t_stack, last_dim, 0, l0.rank, ctx)
    var t1 = slice(t_stack, last_dim, l0.rank, l1.rank, ctx)
    var t2 = slice(t_stack, last_dim, 2 * l0.rank, l2.rank, ctx)
    var t3 = slice(t_stack, last_dim, 3 * l0.rank, l3.rank, ctx)

    var ddy0 = mul_scalar(d0, l0.scale, ctx)
    var dt0 = linear_backward_dx(ddy0, l0.b[], M, l0.rank, out0, ctx)
    var db0 = linear_backward_dw(ddy0, t0, M, l0.rank, out0, ctx)
    var ddy1 = mul_scalar(d1, l1.scale, ctx)
    var dt1 = linear_backward_dx(ddy1, l1.b[], M, l1.rank, out1, ctx)
    var db1 = linear_backward_dw(ddy1, t1, M, l1.rank, out1, ctx)
    var ddy2 = mul_scalar(d2, l2.scale, ctx)
    var dt2 = linear_backward_dx(ddy2, l2.b[], M, l2.rank, out2, ctx)
    var db2 = linear_backward_dw(ddy2, t2, M, l2.rank, out2, ctx)
    var ddy3 = mul_scalar(d3, l3.scale, ctx)
    var dt3 = linear_backward_dx(ddy3, l3.b[], M, l3.rank, out3, ctx)
    var db3 = linear_backward_dw(ddy3, t3, M, l3.rank, out3, ctx)

    var dt_stack = concat(1, ctx, dt0, dt1, dt2, dt3)
    var dx_lora = linear_backward_dx(dt_stack, a_stack, M, in_f, 4 * l0.rank, ctx)
    var da_stack = linear_backward_dw(dt_stack, x, M, in_f, 4 * l0.rank, ctx)
    var da0 = slice(da_stack, 0, 0, l0.rank, ctx)
    var da1 = slice(da_stack, 0, l0.rank, l1.rank, ctx)
    var da2 = slice(da_stack, 0, 2 * l0.rank, l2.rank, ctx)
    var da3 = slice(da_stack, 0, 3 * l0.rank, l3.rank, ctx)
    var d_x = add(dx_base, dx_lora, ctx)

    return _LinBwdGroup4(
        d_x^,
        Krea2LoraGrad(
            Optional[List[Float32]](_tensor_to_host_f32_local(da0, ctx)),
            Optional[List[Float32]](_tensor_to_host_f32_local(db0, ctx)),
        ),
        Krea2LoraGrad(
            Optional[List[Float32]](_tensor_to_host_f32_local(da1, ctx)),
            Optional[List[Float32]](_tensor_to_host_f32_local(db1, ctx)),
        ),
        Krea2LoraGrad(
            Optional[List[Float32]](_tensor_to_host_f32_local(da2, ctx)),
            Optional[List[Float32]](_tensor_to_host_f32_local(db2, ctx)),
        ),
        Krea2LoraGrad(
            Optional[List[Float32]](_tensor_to_host_f32_local(da3, ctx)),
            Optional[List[Float32]](_tensor_to_host_f32_local(db3, ctx)),
        ),
    )


def _linear_bwd_dx_group2_dev(
    d0: Tensor, d1: Tensor, x: Tensor,
    w0: Tensor, lo0: Optional[LoraAdapterDevice], out0: Int,
    w1: Tensor, lo1: Optional[LoraAdapterDevice], out1: Int,
    M: Int, in_f: Int, ctx: DeviceContext,
    w8t0: Optional[TArc] = Optional[TArc](None), s0: Optional[TArc] = Optional[TArc](None),
    w8t1: Optional[TArc] = Optional[TArc](None), s1: Optional[TArc] = Optional[TArc](None),
) raises -> _LinBwdGroup2T:
    comptime if not KREA2_BATCH_LORA_GROUPS:
        var b0 = _linear_bwd_dx_dev(d0, x, w0, lo0, M, in_f, out0, ctx, w8t0, s0)
        var b1 = _linear_bwd_dx_dev(d1, x, w1, lo1, M, in_f, out1, ctx, w8t1, s1)
        var g0 = b0.lora.copy()
        var g1 = b1.lora.copy()
        var d_x = add(b0.d_x, b1.d_x, ctx)
        return _LinBwdGroup2T(d_x^, g0^, g1^)
    if not lo0 or not lo1:
        var b0 = _linear_bwd_dx_dev(d0, x, w0, lo0, M, in_f, out0, ctx, w8t0, s0)
        var b1 = _linear_bwd_dx_dev(d1, x, w1, lo1, M, in_f, out1, ctx, w8t1, s1)
        var g0 = b0.lora.copy()
        var g1 = b1.lora.copy()
        var d_x = add(b0.d_x, b1.d_x, ctx)
        return _LinBwdGroup2T(d_x^, g0^, g1^)

    var l0 = lo0.value().copy()
    var l1 = lo1.value().copy()
    if l0.rank != l1.rank or l0.in_f != l1.in_f:
        var b0 = _linear_bwd_dx_dev(d0, x, w0, lo0, M, in_f, out0, ctx, w8t0, s0)
        var b1 = _linear_bwd_dx_dev(d1, x, w1, lo1, M, in_f, out1, ctx, w8t1, s1)
        var g0 = b0.lora.copy()
        var g1 = b1.lora.copy()
        var d_x = add(b0.d_x, b1.d_x, ctx)
        return _LinBwdGroup2T(d_x^, g0^, g1^)

    var dx0 = _base_dx(d0, w0, M, in_f, out0, w8t0, s0, ctx)
    var dx1 = _base_dx(d1, w1, M, in_f, out1, w8t1, s1, ctx)
    var dx_base = add(dx0, dx1, ctx)

    var a_stack = concat(0, ctx, l0.a[], l1.a[])
    var nb = _no_bias()
    var t_stack = linear(x, a_stack, nb^, ctx)
    var last_dim = len(t_stack.shape()) - 1
    var t0 = slice(t_stack, last_dim, 0, l0.rank, ctx)
    var t1 = slice(t_stack, last_dim, l0.rank, l1.rank, ctx)

    var ddy0 = mul_scalar(d0, l0.scale, ctx)
    var dt0 = linear_backward_dx(ddy0, l0.b[], M, l0.rank, out0, ctx)
    var db0 = linear_backward_dw(ddy0, t0, M, l0.rank, out0, ctx)
    var ddy1 = mul_scalar(d1, l1.scale, ctx)
    var dt1 = linear_backward_dx(ddy1, l1.b[], M, l1.rank, out1, ctx)
    var db1 = linear_backward_dw(ddy1, t1, M, l1.rank, out1, ctx)

    var dt_stack = concat(1, ctx, dt0, dt1)
    var dx_lora = linear_backward_dx(dt_stack, a_stack, M, in_f, 2 * l0.rank, ctx)
    var da_stack = linear_backward_dw(dt_stack, x, M, in_f, 2 * l0.rank, ctx)
    var da0 = slice(da_stack, 0, 0, l0.rank, ctx)
    var da1 = slice(da_stack, 0, l0.rank, l1.rank, ctx)
    var d_x = add(dx_base, dx_lora, ctx)

    return _LinBwdGroup2T(
        d_x^,
        Krea2LoraGradT(Optional[TArc](TArc(da0^)), Optional[TArc](TArc(db0^))),
        Krea2LoraGradT(Optional[TArc](TArc(da1^)), Optional[TArc](TArc(db1^))),
    )


def _linear_bwd_dx_group4_dev(
    d0: Tensor, d1: Tensor, d2: Tensor, d3: Tensor, x: Tensor,
    w0: Tensor, lo0: Optional[LoraAdapterDevice], out0: Int,
    w1: Tensor, lo1: Optional[LoraAdapterDevice], out1: Int,
    w2: Tensor, lo2: Optional[LoraAdapterDevice], out2: Int,
    w3: Tensor, lo3: Optional[LoraAdapterDevice], out3: Int,
    M: Int, in_f: Int, ctx: DeviceContext,
    w8t0: Optional[TArc] = Optional[TArc](None), s0: Optional[TArc] = Optional[TArc](None),
    w8t1: Optional[TArc] = Optional[TArc](None), s1: Optional[TArc] = Optional[TArc](None),
    w8t2: Optional[TArc] = Optional[TArc](None), s2: Optional[TArc] = Optional[TArc](None),
    w8t3: Optional[TArc] = Optional[TArc](None), s3: Optional[TArc] = Optional[TArc](None),
) raises -> _LinBwdGroup4T:
    comptime if not KREA2_BATCH_LORA_GROUPS:
        var b0 = _linear_bwd_dx_dev(d0, x, w0, lo0, M, in_f, out0, ctx, w8t0, s0)
        var b1 = _linear_bwd_dx_dev(d1, x, w1, lo1, M, in_f, out1, ctx, w8t1, s1)
        var b2 = _linear_bwd_dx_dev(d2, x, w2, lo2, M, in_f, out2, ctx, w8t2, s2)
        var b3 = _linear_bwd_dx_dev(d3, x, w3, lo3, M, in_f, out3, ctx, w8t3, s3)
        var g0 = b0.lora.copy()
        var g1 = b1.lora.copy()
        var g2 = b2.lora.copy()
        var g3 = b3.lora.copy()
        var d_x = add(add(b0.d_x, b1.d_x, ctx), add(b2.d_x, b3.d_x, ctx), ctx)
        return _LinBwdGroup4T(d_x^, g0^, g1^, g2^, g3^)
    if not lo0 or not lo1 or not lo2 or not lo3:
        var b0 = _linear_bwd_dx_dev(d0, x, w0, lo0, M, in_f, out0, ctx, w8t0, s0)
        var b1 = _linear_bwd_dx_dev(d1, x, w1, lo1, M, in_f, out1, ctx, w8t1, s1)
        var b2 = _linear_bwd_dx_dev(d2, x, w2, lo2, M, in_f, out2, ctx, w8t2, s2)
        var b3 = _linear_bwd_dx_dev(d3, x, w3, lo3, M, in_f, out3, ctx, w8t3, s3)
        var g0 = b0.lora.copy()
        var g1 = b1.lora.copy()
        var g2 = b2.lora.copy()
        var g3 = b3.lora.copy()
        var d_x = add(add(b0.d_x, b1.d_x, ctx), add(b2.d_x, b3.d_x, ctx), ctx)
        return _LinBwdGroup4T(d_x^, g0^, g1^, g2^, g3^)

    var l0 = lo0.value().copy()
    var l1 = lo1.value().copy()
    var l2 = lo2.value().copy()
    var l3 = lo3.value().copy()
    if (
        l0.rank != l1.rank or l0.rank != l2.rank or l0.rank != l3.rank
        or l0.in_f != l1.in_f or l0.in_f != l2.in_f or l0.in_f != l3.in_f
    ):
        var b0 = _linear_bwd_dx_dev(d0, x, w0, lo0, M, in_f, out0, ctx, w8t0, s0)
        var b1 = _linear_bwd_dx_dev(d1, x, w1, lo1, M, in_f, out1, ctx, w8t1, s1)
        var b2 = _linear_bwd_dx_dev(d2, x, w2, lo2, M, in_f, out2, ctx, w8t2, s2)
        var b3 = _linear_bwd_dx_dev(d3, x, w3, lo3, M, in_f, out3, ctx, w8t3, s3)
        var g0 = b0.lora.copy()
        var g1 = b1.lora.copy()
        var g2 = b2.lora.copy()
        var g3 = b3.lora.copy()
        var d_x = add(add(b0.d_x, b1.d_x, ctx), add(b2.d_x, b3.d_x, ctx), ctx)
        return _LinBwdGroup4T(d_x^, g0^, g1^, g2^, g3^)

    var dx0 = _base_dx(d0, w0, M, in_f, out0, w8t0, s0, ctx)
    var dx1 = _base_dx(d1, w1, M, in_f, out1, w8t1, s1, ctx)
    var dx2 = _base_dx(d2, w2, M, in_f, out2, w8t2, s2, ctx)
    var dx3 = _base_dx(d3, w3, M, in_f, out3, w8t3, s3, ctx)
    var dx_base = add(add(dx0, dx1, ctx), add(dx2, dx3, ctx), ctx)

    var a_stack = concat(0, ctx, l0.a[], l1.a[], l2.a[], l3.a[])
    var nb = _no_bias()
    var t_stack = linear(x, a_stack, nb^, ctx)
    var last_dim = len(t_stack.shape()) - 1
    var t0 = slice(t_stack, last_dim, 0, l0.rank, ctx)
    var t1 = slice(t_stack, last_dim, l0.rank, l1.rank, ctx)
    var t2 = slice(t_stack, last_dim, 2 * l0.rank, l2.rank, ctx)
    var t3 = slice(t_stack, last_dim, 3 * l0.rank, l3.rank, ctx)

    var ddy0 = mul_scalar(d0, l0.scale, ctx)
    var dt0 = linear_backward_dx(ddy0, l0.b[], M, l0.rank, out0, ctx)
    var db0 = linear_backward_dw(ddy0, t0, M, l0.rank, out0, ctx)
    var ddy1 = mul_scalar(d1, l1.scale, ctx)
    var dt1 = linear_backward_dx(ddy1, l1.b[], M, l1.rank, out1, ctx)
    var db1 = linear_backward_dw(ddy1, t1, M, l1.rank, out1, ctx)
    var ddy2 = mul_scalar(d2, l2.scale, ctx)
    var dt2 = linear_backward_dx(ddy2, l2.b[], M, l2.rank, out2, ctx)
    var db2 = linear_backward_dw(ddy2, t2, M, l2.rank, out2, ctx)
    var ddy3 = mul_scalar(d3, l3.scale, ctx)
    var dt3 = linear_backward_dx(ddy3, l3.b[], M, l3.rank, out3, ctx)
    var db3 = linear_backward_dw(ddy3, t3, M, l3.rank, out3, ctx)

    var dt_stack = concat(1, ctx, dt0, dt1, dt2, dt3)
    var dx_lora = linear_backward_dx(dt_stack, a_stack, M, in_f, 4 * l0.rank, ctx)
    var da_stack = linear_backward_dw(dt_stack, x, M, in_f, 4 * l0.rank, ctx)
    var da0 = slice(da_stack, 0, 0, l0.rank, ctx)
    var da1 = slice(da_stack, 0, l0.rank, l1.rank, ctx)
    var da2 = slice(da_stack, 0, 2 * l0.rank, l2.rank, ctx)
    var da3 = slice(da_stack, 0, 3 * l0.rank, l3.rank, ctx)
    var d_x = add(dx_base, dx_lora, ctx)

    return _LinBwdGroup4T(
        d_x^,
        Krea2LoraGradT(Optional[TArc](TArc(da0^)), Optional[TArc](TArc(db0^))),
        Krea2LoraGradT(Optional[TArc](TArc(da1^)), Optional[TArc](TArc(db1^))),
        Krea2LoraGradT(Optional[TArc](TArc(da2^)), Optional[TArc](TArc(db2^))),
        Krea2LoraGradT(Optional[TArc](TArc(da3^)), Optional[TArc](TArc(db3^))),
    )


struct Krea2DirectDoRAGradT(Copyable, Movable):
    var d_a: Optional[TArc]
    var d_b: Optional[TArc]
    var d_m: Optional[TArc]

    def __init__(
        out self, var d_a: Optional[TArc], var d_b: Optional[TArc],
        var d_m: Optional[TArc],
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_m = d_m^


struct Krea2DirectOFTGradT(Copyable, Movable):
    var d_vec: Optional[TArc]

    def __init__(out self, var d_vec: Optional[TArc]):
        self.d_vec = d_vec^


struct Krea2BlockDirectDoRAGradsT(Movable):
    var d_x: TArc
    var wq: Krea2DirectDoRAGradT
    var wk: Krea2DirectDoRAGradT
    var wv: Krea2DirectDoRAGradT
    var gate_w: Krea2DirectDoRAGradT
    var wo: Krea2DirectDoRAGradT
    var mlp_gate_w: Krea2DirectDoRAGradT
    var mlp_up_w: Krea2DirectDoRAGradT
    var mlp_down_w: Krea2DirectDoRAGradT

    def __init__(
        out self, var d_x: TArc,
        var wq: Krea2DirectDoRAGradT, var wk: Krea2DirectDoRAGradT,
        var wv: Krea2DirectDoRAGradT, var gate_w: Krea2DirectDoRAGradT,
        var wo: Krea2DirectDoRAGradT,
        var mlp_gate_w: Krea2DirectDoRAGradT,
        var mlp_up_w: Krea2DirectDoRAGradT,
        var mlp_down_w: Krea2DirectDoRAGradT,
    ):
        self.d_x = d_x^
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^


struct Krea2BlockDirectOFTGradsT(Movable):
    var d_x: TArc
    var wq: Krea2DirectOFTGradT
    var wk: Krea2DirectOFTGradT
    var wv: Krea2DirectOFTGradT
    var gate_w: Krea2DirectOFTGradT
    var wo: Krea2DirectOFTGradT
    var mlp_gate_w: Krea2DirectOFTGradT
    var mlp_up_w: Krea2DirectOFTGradT
    var mlp_down_w: Krea2DirectOFTGradT

    def __init__(
        out self, var d_x: TArc,
        var wq: Krea2DirectOFTGradT, var wk: Krea2DirectOFTGradT,
        var wv: Krea2DirectOFTGradT, var gate_w: Krea2DirectOFTGradT,
        var wo: Krea2DirectOFTGradT,
        var mlp_gate_w: Krea2DirectOFTGradT,
        var mlp_up_w: Krea2DirectOFTGradT,
        var mlp_down_w: Krea2DirectOFTGradT,
    ):
        self.d_x = d_x^
        self.wq = wq^
        self.wk = wk^
        self.wv = wv^
        self.gate_w = gate_w^
        self.wo = wo^
        self.mlp_gate_w = mlp_gate_w^
        self.mlp_up_w = mlp_up_w^
        self.mlp_down_w = mlp_down_w^


struct _DirectDoRALinBwdT(Movable):
    var d_x: Tensor
    var dora: Krea2DirectDoRAGradT

    def __init__(out self, var d_x: Tensor, var dora: Krea2DirectDoRAGradT):
        self.d_x = d_x^
        self.dora = dora^


struct _DirectOFTLinBwdT(Movable):
    var d_x: Tensor
    var oft: Krea2DirectOFTGradT

    def __init__(out self, var d_x: Tensor, var oft: Krea2DirectOFTGradT):
        self.d_x = d_x^
        self.oft = oft^


def krea2_block_direct_dora_projection_forward(
    x: Tensor, w: Tensor, ad: Optional[DoRAAdapterDevice],
    M: Int, ctx: DeviceContext,
) raises -> Tensor:
    """Direct DoRA projection hook for Krea2 block lowering.

    Present adapter path returns x @ W_eff^T. It is not an additive LoRA delta.
    """
    if ad:
        return krea2_direct_dora_projection_forward_resident(ad.value(), x, w, M, ctx)
    var nb = _no_bias()
    return linear(x, w, nb^, ctx)


def krea2_block_direct_dora_projection_backward_dev(
    d_y: Tensor, x: Tensor, w: Tensor, ad: Optional[DoRAAdapterDevice],
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> _DirectDoRALinBwdT:
    """Direct DoRA projection backward hook for Krea2 block lowering.

    Present adapter path returns the full W_eff d_x from the DoRA primitive. Do
    not add a separate frozen-W base d_x.
    """
    if ad:
        var g = krea2_direct_dora_projection_backward_resident(
            ad.value(), d_y, x, w, M, ctx,
        )
        return _DirectDoRALinBwdT(
            g.d_x.clone(ctx),
            Krea2DirectDoRAGradT(
                Optional[TArc](TArc(g.d_a.clone(ctx))),
                Optional[TArc](TArc(g.d_b.clone(ctx))),
                Optional[TArc](TArc(g.d_m.clone(ctx))),
            ),
        )
    var d_x = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    return _DirectDoRALinBwdT(d_x^, Krea2DirectDoRAGradT(None, None, None))


def krea2_block_direct_oft_projection_forward(
    x: Tensor, w: Tensor, ad: Optional[Krea2DirectOFTDeviceSlot],
    M: Int, ctx: DeviceContext,
) raises -> Tensor:
    """Direct OFT projection hook for Krea2 block lowering."""
    if ad:
        return krea2_direct_oft_projection_forward_resident(ad.value(), x, w, M, ctx)
    var nb = _no_bias()
    return linear(x, w, nb^, ctx)


def krea2_block_direct_oft_projection_backward_dev(
    d_y: Tensor, x: Tensor, w: Tensor, ad: Optional[Krea2DirectOFTDeviceSlot],
    M: Int, in_f: Int, out_f: Int, ctx: DeviceContext,
) raises -> _DirectOFTLinBwdT:
    """Direct OFT projection backward hook for Krea2 block lowering."""
    if ad:
        var g = krea2_direct_oft_projection_backward_resident(
            ad.value(), d_y, x, w, M, ctx,
        )
        return _DirectOFTLinBwdT(
            g.d_x.clone(ctx),
            Krea2DirectOFTGradT(Optional[TArc](TArc(g.d_vec.clone(ctx)))),
        )
    var d_x = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    return _DirectOFTLinBwdT(d_x^, Krea2DirectOFTGradT(None))


# ── op-by-op backward divergence probe (team-lead b2-block-diff task) ─────────
# Set True + rebuild ONLY the krea2_b2_blockdiff_gate probe. Prints the norm of
# every backward INTERMEDIATE for the block whose dbg_block>=0 (the stack passes
# the deepest block index). For the b2 backward it also prints half0x2 = 2*||[0:L]||
# so that, for identical samples, half0x2 must equal the b1 norm at every step —
# the FIRST intermediate where they part names the divergent op. Behind dbg_block
# (default -1) so ALL shipped callers are untouched at runtime.
comptime KREA2_OPDBG = False


def _opdbg(
    tag: String, blk: Int, name: String, t: Tensor, is_b2: Bool, L: Int,
    rl: Int, ctx: DeviceContext,
) raises:
    # Work on an L-row view: b1 = the [.,L,..] tensor directly; b2 = the [0:L]
    # (sample-0) half. Split each into VALID rows [0:rl] and PAD rows [rl:L] so
    # a divergence can be attributed to real tokens vs masked padding. For b2 the
    # sub-norms carry the *2 loss-scale unscale, so they compare directly to b1.
    # The row axis is dim 0 for a rank-2 [2L,C] linear-backward output, else dim 1
    # ([1,2L,C] / [1,2L,H,Dh]) — slicing the wrong axis corrupts the split.
    var rowdim = 0 if len(t.shape()) == 2 else 1
    var view = slice(t, rowdim, 0, L, ctx) if is_b2 else t.clone(ctx)
    var h = view.to_host(ctx)
    var n = len(h)
    var stride = n // L
    var vs = Float64(0.0)
    var ps = Float64(0.0)
    for r in range(L):
        var acc = Float64(0.0)
        var b = r * stride
        for c in range(stride):
            var v = Float64(h[b + c])
            acc += v * v
        if r < rl:
            vs += acc
        else:
            ps += acc
    var k = Float64(2.0) if is_b2 else Float64(1.0)
    print("[opdbg-", tag, "] blk", blk, " ", name,
          " valid=", k * sqrt(vs), " pad=", k * sqrt(ps))


# ══════════════════════════════════════════════════════════════════════════════
# OminiControl EDIT — PER-SEGMENT MODULATION BACKWARD (C4; seam B.1's reverse)
#
# The forward runs rows [0, split) on the mods(t) chunks and rows [split, L) on
# the mods(t=0) chunks (`_modulate_seg2` / `_residual_gate_seg2`). Its backward
# therefore has to do TWO things the uniform backward never did:
#   (1) d_x must be differentiated against the chunk that actually modulated that
#       row — the same 2-span split, run through the SAME kernels once per span.
#   (2) the CHUNK grads (d_scale/d_shift/d_gate) must be accumulated PER SPAN:
#       rows [0, split) (TXT_real + IMG) sum into d_mod(t), rows [split, …) (COND)
#       into d_mod(t=0). Those are the two separate temb chains the EDIT trainer
#       carries, so mixing them is not a rounding difference, it is a wrong
#       gradient.
#
# ⚠ THE PAD TAIL IS EXCLUDED FROM (2), AND THAT IS A MEASURED DECISION.
# The TXT_pad tail sits in the t=0 span, so a naive "rows [split, L)" reduction
# would add pad-row gradient to d_mod(t=0). Two facts, both measured on the CUDA
# fixture (scripts/krea2_omini_torch_oracle.py, "[pad-row evidence]"):
#   * pad rows CANNOT reach the real rows — with the pad KEY columns masked, the
#     softmax weight on them is exp(-inf) = 0 EXACTLY, so dK/dV on pad rows are
#     exactly zero. The oracle confirms it: d_x[0:real_len] and all 16 LoRA dA/dB
#     are BIT-EQUAL whether or not the pad tail carries an upstream gradient.
#   * they DO reach d_mod(t=0), and hard: cos(d_blk_vec_cond with vs without the
#     pad-tail gradient) = 0.679 at block 0 and 0.994 at block 27. d_blk_vec_t is
#     BIT-EQUAL (the t span has no pad rows). So the pad tail's ONLY effect is on
#     the condition chunk grads.
# In the Mojo production path those rows are worse than merely different: the
# cuDNN flash TAIL-padmask kernel masks only KEY columns, so pad QUERY rows carry
# masked-out garbage BY DESIGN (see the SDPA comment in the forward). Feeding
# that garbage into the condition temb chain is not a defensible gradient. The
# reduction therefore stops at `real_rows`; the pad rows still get their d_x
# (same chunks, param grads off) so the returned d_x is complete.
# When real_rows == L (no pad) the extra span disappears and the helper is the
# plain 2-span form.
# ══════════════════════════════════════════════════════════════════════════════
struct _ModBwdSeg2(Movable):
    var d_x: Tensor
    var d_scale_t: Tensor
    var d_shift_t: Tensor
    var d_scale_c: Tensor
    var d_shift_c: Tensor

    def __init__(
        out self, var d_x: Tensor, var d_scale_t: Tensor, var d_shift_t: Tensor,
        var d_scale_c: Tensor, var d_shift_c: Tensor,
    ):
        self.d_x = d_x^
        self.d_scale_t = d_scale_t^
        self.d_shift_t = d_shift_t^
        self.d_scale_c = d_scale_c^
        self.d_shift_c = d_shift_c^


def _modulate_backward_seg2(
    go: Tensor, x: Tensor, scale_t: Tensor, scale_c: Tensor,
    split: Int, real_rows: Int, ctx: DeviceContext,
) raises -> _ModBwdSeg2:
    """Backward of `_modulate_seg2`. d_x over ALL rows; param grads from rows
    [0,split) (-> t) and [split, real_rows) (-> t=0) only."""
    var rows = x.shape()[1]
    if split <= 0 or split >= rows or real_rows < split or real_rows > rows:
        raise Error("krea2 modulate_backward_seg2: bad split/real_rows")
    var mh = modulate_backward(
        slice(go, 1, 0, split, ctx), slice(x, 1, 0, split, ctx), scale_t, ctx,
        compute_param_grads=True,
    )
    var n_real = real_rows - split
    var mt = modulate_backward(
        slice(go, 1, split, n_real, ctx), slice(x, 1, split, n_real, ctx),
        scale_c, ctx, compute_param_grads=True,
    )
    if real_rows == rows:
        return _ModBwdSeg2(
            concat(1, ctx, mh.d_x, mt.d_x),
            mh.d_scale.clone(ctx), mh.d_shift.clone(ctx),
            mt.d_scale.clone(ctx), mt.d_shift.clone(ctx),
        )
    var n_pad = rows - real_rows
    var mp = modulate_backward(
        slice(go, 1, real_rows, n_pad, ctx), slice(x, 1, real_rows, n_pad, ctx),
        scale_c, ctx, compute_param_grads=False,
    )
    return _ModBwdSeg2(
        concat(1, ctx, mh.d_x, mt.d_x, mp.d_x),
        mh.d_scale.clone(ctx), mh.d_shift.clone(ctx),
        mt.d_scale.clone(ctx), mt.d_shift.clone(ctx),
    )


struct _GateBwdSeg2(Movable):
    var d_x: Tensor
    var d_y: Tensor
    var d_g_t: Tensor
    var d_g_c: Tensor

    def __init__(
        out self, var d_x: Tensor, var d_y: Tensor,
        var d_g_t: Tensor, var d_g_c: Tensor,
    ):
        self.d_x = d_x^
        self.d_y = d_y^
        self.d_g_t = d_g_t^
        self.d_g_c = d_g_c^


def _gate_residual_backward_seg2(
    go: Tensor, x: Tensor, gate_t: Tensor, gate_c: Tensor, y: Tensor,
    split: Int, real_rows: Int, ctx: DeviceContext,
) raises -> _GateBwdSeg2:
    """Backward of `_residual_gate_seg2`. d_x/d_y over ALL rows; d_g from rows
    [0,split) (-> t) and [split, real_rows) (-> t=0) only."""
    var rows = go.shape()[1]
    if split <= 0 or split >= rows or real_rows < split or real_rows > rows:
        raise Error("krea2 gate_residual_backward_seg2: bad split/real_rows")
    var rh = gate_residual_backward(
        slice(go, 1, 0, split, ctx), slice(x, 1, 0, split, ctx), gate_t,
        slice(y, 1, 0, split, ctx), ctx, compute_gate_grad=True,
    )
    var n_real = real_rows - split
    var rt = gate_residual_backward(
        slice(go, 1, split, n_real, ctx), slice(x, 1, split, n_real, ctx), gate_c,
        slice(y, 1, split, n_real, ctx), ctx, compute_gate_grad=True,
    )
    if real_rows == rows:
        return _GateBwdSeg2(
            concat(1, ctx, rh.d_x, rt.d_x), concat(1, ctx, rh.d_y, rt.d_y),
            rh.d_g.clone(ctx), rt.d_g.clone(ctx),
        )
    var n_pad = rows - real_rows
    var rp = gate_residual_backward(
        slice(go, 1, real_rows, n_pad, ctx), slice(x, 1, real_rows, n_pad, ctx),
        gate_c, slice(y, 1, real_rows, n_pad, ctx), ctx, compute_gate_grad=False,
    )
    return _GateBwdSeg2(
        concat(1, ctx, rh.d_x, rt.d_x, rp.d_x),
        concat(1, ctx, rh.d_y, rt.d_y, rp.d_y),
        rh.d_g.clone(ctx), rt.d_g.clone(ctx),
    )


def _pack_d_vec(g: List[TArc], ctx: DeviceContext) raises -> Tensor:
    """The [6*features] grad of ONE modulation vector. `g` is appended in the
    order the backward produces the chunks — [postgate, postscale, postshift,
    pregate, prescale, preshift] — and the packed layout is `_mod6`'s chunk
    order [prescale, preshift, pregate, postscale, postshift, postgate]. Each
    chunk is cast F32 (the optimizer/oracle boundary; the reductions themselves
    already accumulate in F32 and round once at store, exactly like the torch
    bf16 grad the oracle dumps). Since mods = vec + mod_lin, this IS the grad
    w.r.t. `vec` AND w.r.t. mod.lin — the two differ by nothing."""
    if len(g) != 6:
        raise Error(
            String("krea2 _pack_d_vec: expected 6 chunk grads, got ")
            + String(len(g))
        )
    return concat(
        0, ctx,
        cast_tensor(g[4][], STDtype.F32, ctx),   # prescale
        cast_tensor(g[5][], STDtype.F32, ctx),   # preshift
        cast_tensor(g[3][], STDtype.F32, ctx),   # pregate
        cast_tensor(g[1][], STDtype.F32, ctx),   # postscale
        cast_tensor(g[2][], STDtype.F32, ctx),   # postshift
        cast_tensor(g[0][], STDtype.F32, ctx),   # postgate
    )


def krea2_single_stream_block_lora_backward[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out: Tensor,        # [1, L, features] upstream grad of the block output
    vec: Tensor,          # [1, 6*features]  (for the raw mod chunks)
    w: Krea2BlockWeights, lora: Krea2BlockLora, saved: Krea2BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),  # MUST match the forward call:
        # None (or real_len == L) = sdpa_backward (full attn, BIT-IDENTICAL to the
        # pre-flash block). Present & < L = cuDNN flash-padmask backward, consuming
        # the saved flash bf16 q/k/v/o + F32 stats (set in the forward), passing the
        # SAME real_len so the bwd respects the same [real_len:L] pad masking.
        # FAIL-LOUD if real_len < L but the saved tape has no flash set (fwd/bwd
        # real_len mismatch).
    dbg_block: Int = -1,   # op-by-op probe: >=0 prints backward intermediates.
    vec_cond: Optional[TArc] = Optional[TArc](None),  # OminiControl EDIT (C4):
        # the SECOND modulation vector tproj(tmlp(temb(t=0))) [1, 6*features] —
        # MUST be the same tensor the forward was given. Present together with a
        # valid cond_off turns on the PER-SEGMENT backward: d_x differentiated
        # against the chunk that modulated each row, and d_vec_t / d_vec_cond
        # accumulated per span (see the PER-SEGMENT MODULATION BACKWARD box).
    cond_off: Optional[Int] = Optional[Int](None),    # the split row; pass
        # training/krea2_omini_layout.krea2_omini_mod_split(lay), which is -1 for
        # a layout with no condition segment so the guard falls through to the
        # pre-C4 code path BIT-FOR-BIT (the CONDLEN=0 regression contract).
    cond_len: Optional[Int] = Optional[Int](None),    # the condition segment
        # length; present with a valid cond_off it routes the 8 adapters' dA/dB
        # (and their d_x contribution) to rows [cond_off, cond_off+cond_len) —
        # the exact mirror of the C3 forward. Independent of vec_cond.
) raises -> Krea2BlockGrads:
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var mlpdim = saved.mlp_gate[].shape()[2]
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    # raw mod chunks (pregate/postgate/prescale/postscale needed in backward).
    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var pregate = mods[2]
    var postscale = mods[3]
    var postgate = mods[5]

    # ── OminiControl EDIT switches (C4). Both default OFF, and with both off
    # every line below is the pre-C4 call with the pre-C4 arguments. ─────────
    var seg_mod = False
    var split = 0
    var mods_c = List[TArc]()
    if vec_cond:
        if cond_off:
            split = cond_off.value()
            if split > 0 and split < L:
                mods_c = _mod6(vec_cond.value()[], w.mod_lin[], features, ctx)
                seg_mod = True
    var c_off = -1
    var c_len = 0
    if cond_off:
        if cond_len:
            var co = cond_off.value()
            var cl = cond_len.value()
            if co > 0 and cl > 0 and co + cl <= L:
                c_off = co
                c_len = cl
    # rows the per-span CHUNK reductions may read (the pad tail is excluded —
    # measured decision, see the PER-SEGMENT MODULATION BACKWARD box).
    var rl_rows = real_len.value() if real_len else L
    if rl_rows > L:
        rl_rows = L
    if seg_mod and rl_rows < split:
        raise Error("krea2 block bwd: real_len below the modulation split")
    # chunk grads in production order [postgate, postscale, postshift, pregate,
    # prescale, preshift]; empty on the uniform path.
    var dv_t = List[TArc]()
    var dv_c = List[TArc]()

    # ── MLP branch backward: x2 = residual_gate(x1, postgate, m) ──────────────
    # o = x1 + postgate*m  → d_x1(res)=d_out ; d_m = d_out*postgate (per-channel).
    # (`m` is saved; gate_residual_backward only needs y's value for d_g, which we
    # skip with compute_gate_grad=False — but it still shape-checks y.)
    var grg2_dx: Tensor
    var d_m: Tensor
    if seg_mod:
        var s2 = _gate_residual_backward_seg2(
            d_out, saved.x1[], postgate[], mods_c[5][], saved.m[],
            split, rl_rows, ctx,
        )
        grg2_dx = s2.d_x.clone(ctx)
        d_m = s2.d_y.clone(ctx)
        dv_t.append(TArc(s2.d_g_t.clone(ctx)))
        dv_c.append(TArc(s2.d_g_c.clone(ctx)))
    else:
        var grg2 = gate_residual_backward(d_out, saved.x1[], postgate[], saved.m[], ctx, compute_gate_grad=False)
        # grg2.d_x = passthrough to x1 ; grg2.d_y = grad into m (= d_out*postgate)
        grg2_dx = grg2.d_x.clone(ctx)
        d_m = grg2.d_y.clone(ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "d_out", d_out, False, L, (real_len.value() if real_len else L), ctx)
            _opdbg("b1", dbg_block, "grg2.d_x", grg2_dx, False, L, (real_len.value() if real_len else L), ctx)
            _opdbg("b1", dbg_block, "d_m", d_m, False, L, (real_len.value() if real_len else L), ctx)

    # m = mlp_down(sw)  → d_sw (base dx + LoRA dx) + mlp_down dA/dB
    var bw_down = _linear_bwd_dx(
        d_m, saved.sw[], w.mlp_down_w[], lora.mlp_down_w, M, mlpdim, features, ctx,
        _i8w(w,7), _i8s(w, 7), c_off, c_len,
    )
    var d_sw = bw_down.d_x.clone(ctx)
    var g_down = bw_down.lora.copy()
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "d_sw", d_sw, False, L, (real_len.value() if real_len else L), ctx)

    # sw = swiglu(mlp_gate, mlp_up) → d_mlp_gate, d_mlp_up
    var sgb = swiglu_backward(d_sw, saved.mlp_gate[], saved.mlp_up[], ctx)
    # mlp_gate/mlp_up share xm2, so their LoRA down-projection can batch.
    # COND-ROW ROUTING (C4) takes the ungrouped pair instead: grouping is a
    # launch-count optimization whose shared down-projection GEMM would have to
    # be re-derived for the c_len-row window, and it is not on this chunk's
    # critical path. The math is the same two backwards the grouped helper falls
    # back to when an adapter is missing.
    var g_mg: Krea2LoraGrad
    var g_mu: Krea2LoraGrad
    var d_xm2: Tensor                             # [1,L,features] / [L,features]
    if c_off >= 0:
        var b_mg = _linear_bwd_dx(
            sgb.d_gate, saved.xm2[], w.mlp_gate_w[], lora.mlp_gate_w,
            M, features, mlpdim, ctx, _i8w(w,5), _i8s(w, 5), c_off, c_len,
        )
        var b_mu = _linear_bwd_dx(
            sgb.d_up, saved.xm2[], w.mlp_up_w[], lora.mlp_up_w,
            M, features, mlpdim, ctx, _i8w(w,6), _i8s(w, 6), c_off, c_len,
        )
        g_mg = b_mg.lora.copy()
        g_mu = b_mu.lora.copy()
        d_xm2 = add(b_mg.d_x, b_mu.d_x, ctx)
    else:
        var bw_mlp = _linear_bwd_dx_group2(
            sgb.d_gate, sgb.d_up, saved.xm2[],
            w.mlp_gate_w[], lora.mlp_gate_w, mlpdim,
            w.mlp_up_w[], lora.mlp_up_w, mlpdim,
            M, features, ctx,
            _i8w(w,5), _i8s(w, 5), _i8w(w,6), _i8s(w, 6),
        )
        g_mg = bw_mlp.g0.copy()
        g_mu = bw_mlp.g1.copy()
        d_xm2 = bw_mlp.d_x.clone(ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "d_xm2", d_xm2, False, L, (real_len.value() if real_len else L), ctx)

    # xm2 = modulate(xn2, postscale, postshift) → d_xn2 (drop d_scale/d_shift)
    # modulate_backward requires scale to match the (bf16) acts dtype — the F32-scale
    # production path needs the cast (forward modulate casts internally; backward raises).
    # Mixed precision: saved.xn2 is F32 (rms_norm w/ F32 scale) but d_xm2 is bf16
    # (matmul-backward grad). modulate operated in F32 in the fwd → cast both grad-in
    # and scale to the F32 acts dtype so modulate_backward is dtype-consistent.
    var mb2_dx: Tensor
    if seg_mod:
        var sm2 = _modulate_backward_seg2(
            reshape(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx),
                    [1, L, features], ctx),
            saved.xn2[],
            cast_tensor(postscale[], saved.xn2[].dtype(), ctx),
            cast_tensor(mods_c[3][], saved.xn2[].dtype(), ctx),
            split, rl_rows, ctx,
        )
        mb2_dx = sm2.d_x.clone(ctx)
        dv_t.append(TArc(sm2.d_scale_t.clone(ctx)))
        dv_t.append(TArc(sm2.d_shift_t.clone(ctx)))
        dv_c.append(TArc(sm2.d_scale_c.clone(ctx)))
        dv_c.append(TArc(sm2.d_shift_c.clone(ctx)))
    else:
        var mb2 = modulate_backward(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx), saved.xn2[], cast_tensor(postscale[], saved.xn2[].dtype(), ctx), ctx, compute_param_grads=False)
        mb2_dx = mb2.d_x.clone(ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "mb2.d_x", mb2_dx, False, L, (real_len.value() if real_len else L), ctx)
    # xn2 = postnorm(x1) (weight=postnorm+1, FROZEN) → d_x1 via rms_norm_backward.
    # Mixed precision: the saved acts are bf16 (block input feeds bf16 through the
    # norm→modulate→matmul chain) but the postnorm scale is F32. The FORWARD
    # rms_norm casts the F32 scale DOWN to the act dtype and computes in bf16
    # (norm.mojo:173-174); mirror that here — cast (scale+1) to the act dtype so
    # rms_norm_backward runs the all-bf16 path (go/x/weight matched), not the
    # F32-acts-only mixed path. In the F32 gate this cast is F32→F32 (no-op).
    # FROZEN norm scale → d_x only (rms_norm_backward_dx skips the O(cols²) discarded
    # d_g kernel that was 89% of the step; see norm_backward.mojo:374).
    var rb2_dx = krea2_rmsnorm_backward_dx(mb2_dx, saved.x1[], w.postnorm_scale[], eps, ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "rb2_dx", rb2_dx, False, L, (real_len.value() if real_len else L), ctx)
    # x1 feeds the residual (grg2.d_x) AND postnorm(x1) → SUM.
    var d_x1 = add(grg2_dx, rb2_dx, ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "d_x1", d_x1, False, L, (real_len.value() if real_len else L), ctx)

    # ── ATTENTION branch backward: x1 = residual_gate(x, pregate, a) ──────────
    var grg1_dx: Tensor
    var d_a: Tensor
    if seg_mod:
        var s1 = _gate_residual_backward_seg2(
            d_x1, saved.x[], pregate[], mods_c[2][], saved.a[],
            split, rl_rows, ctx,
        )
        grg1_dx = s1.d_x.clone(ctx)
        d_a = s1.d_y.clone(ctx)
        dv_t.append(TArc(s1.d_g_t.clone(ctx)))
        dv_c.append(TArc(s1.d_g_c.clone(ctx)))
    else:
        var grg1 = gate_residual_backward(d_x1, saved.x[], pregate[], saved.a[], ctx, compute_gate_grad=False)
        grg1_dx = grg1.d_x.clone(ctx)
        d_a = grg1.d_y.clone(ctx)               # grad into a (=d_x1*pregate)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "grg1.d_x", grg1_dx, False, L, (real_len.value() if real_len else L), ctx)
            _opdbg("b1", dbg_block, "d_a", d_a, False, L, (real_len.value() if real_len else L), ctx)

    # a = wo(gated) → d_gated (base dx + LoRA dx) + wo dA/dB
    var bw_wo = _linear_bwd_dx(
        d_a, saved.gated[], w.wo[], lora.wo, M, features, features, ctx,
        _i8w(w,4), _i8s(w, 4), c_off, c_len,
    )
    var d_gated = bw_wo.d_x.clone(ctx)
    var g_wo = bw_wo.lora.copy()
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "d_gated", d_gated, False, L, (real_len.value() if real_len else L), ctx)

    # gated = attn_flat * sg  → d_attn_flat = d_gated*sg ; d_sg = d_gated*attn_flat
    var d_attn_flat = mul(d_gated, saved.sg[], ctx)
    var d_sg = mul(d_gated, saved.attn_flat[], ctx)
    # Differentiate from the saved sigmoid output; PyTorch autograd saves y.
    var d_gate_pre = sigmoid_backward_from_output(d_sg, saved.sg[], ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "d_attn_flat", d_attn_flat, False, L, (real_len.value() if real_len else L), ctx)
            _opdbg("b1", dbg_block, "d_gate_pre", d_gate_pre, False, L, (real_len.value() if real_len else L), ctx)

    # attn_flat = reshape(sdpa(q_rope, k_full, v_full)) → sdpa backward. Length-bucket
    # pad (real_len present & < L): the cuDNN flash-padmask backward from the saved
    # bf16 q/k/v/o + F32 stats (no recompute), passing the SAME real_len so the bwd
    # respects the [real_len:L] pad masking. No pad: the math sdpa_backward
    # (BIT-IDENTICAL to the pre-flash block). FLASH dQ is NONDETERMINISTIC run-to-run
    # (cuDNN atomics on the dQ accumulation) → flash-path grads are value-tolerance,
    # NOT bit-exact (see krea2_mask_pad_gate's documented tolerance).
    var d_att = reshape(d_attn_flat, [1, L, HEADS, HEADDIM], ctx)
    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    var bwd_use_flash = real_len and real_len.value() < L
    if bwd_use_flash:
        if not saved.flash_stats:
            raise Error(
                "krea2 block bwd: real_len < L but saved tape has no flash set"
                " (forward/backward real_len mismatch)"
            )
        var rl = real_len.value()
        # BF16 flash bwd consumes BF16 d_att and returns BF16 dQ/dK/dV. F32 stays
        # inside cuDNN stats/score math, not at the model/activation boundary.
        var fb = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
            saved.flash_q.value(), saved.flash_k.value(), saved.flash_v.value(),
            saved.flash_o.value(), saved.flash_stats.value(), d_att, rl, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward[1, L, HEADS, HEADDIM](
            saved.q_rope[], saved.k_full[], saved.v_full[], d_att, scale, ctx
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())
    # d_q_sb [1,L,HEADS,Dh] ; d_k_sb [1,L,HEADS,Dh] ; d_v_sb [1,L,HEADS,Dh]
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "d_q_sb", d_q_sb, False, L, (real_len.value() if real_len else L), ctx)
            _opdbg("b1", dbg_block, "d_k_sb", d_k_sb, False, L, (real_len.value() if real_len else L), ctx)
            _opdbg("b1", dbg_block, "d_v_sb", d_v_sb, False, L, (real_len.value() if real_len else L), ctx)

    # GQA backward: repeat_kv sum-reduce HEADS → KVHEADS for k and v.
    var d_k_rope = repeat_kv_backward(d_k_sb, L, KVHEADS, n_rep, HEADDIM, ctx)
    var d_v = repeat_kv_backward(d_v_sb, L, KVHEADS, n_rep, HEADDIM, ctx)

    # RoPE backward (cos/sin non-learnable → only d_x).
    var d_q_rms = rope_backward(d_q_sb, cos_q, sin_q, True, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, True, ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "d_q_rms", d_q_rms, False, L, (real_len.value() if real_len else L), ctx)

    # QKNorm backward (weight=qnorm/knorm+1, FROZEN) → d_q_pre, d_k_pre. Same
    # mixed-precision mirror as the rb2 call above: q_pre/k_pre are bf16 acts, the
    # qnorm/knorm scales are F32 → cast (scale+1) down to the act dtype so the
    # all-bf16 rms_norm_backward path runs (matches the bf16 forward QKNorm).
    var rbq_dx = krea2_rmsnorm_backward_dx(d_q_rms, saved.q_pre[], w.qnorm_scale[], eps, ctx)
    var rbk_dx = krea2_rmsnorm_backward_dx(d_k_rms, saved.k_pre[], w.knorm_scale[], eps, ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "rbq_dx", rbq_dx, False, L, (real_len.value() if real_len else L), ctx)

    # flatten BSHD grads back to [1,L,*] for the projection backward.
    var d_q = reshape(rbq_dx, [1, L, HEADS * HEADDIM], ctx)
    var d_k = reshape(rbk_dx, [1, L, KVHEADS * HEADDIM], ctx)
    var d_v_flat = reshape(d_v, [1, L, KVHEADS * HEADDIM], ctx)

    # q/k/v/gate share xm, so their LoRA down-projection can batch.
    # COND-ROW ROUTING (C4) takes the ungrouped four instead — same reason as the
    # MLP pair above; this is exactly the fallback the grouped helper itself uses
    # when one of the four adapters is absent.
    var g_wq: Krea2LoraGrad
    var g_wk: Krea2LoraGrad
    var g_wv: Krea2LoraGrad
    var g_gate: Krea2LoraGrad
    var d_xm: Tensor
    if c_off >= 0:
        var b_q = _linear_bwd_dx(
            d_q, saved.xm[], w.wq[], lora.wq, M, features, HEADS * HEADDIM, ctx,
            _i8w(w,0), _i8s(w, 0), c_off, c_len,
        )
        var b_k = _linear_bwd_dx(
            d_k, saved.xm[], w.wk[], lora.wk, M, features, KVHEADS * HEADDIM, ctx,
            _i8w(w,1), _i8s(w, 1), c_off, c_len,
        )
        var b_v = _linear_bwd_dx(
            d_v_flat, saved.xm[], w.wv[], lora.wv, M, features, KVHEADS * HEADDIM,
            ctx, _i8w(w,2), _i8s(w, 2), c_off, c_len,
        )
        var b_g = _linear_bwd_dx(
            d_gate_pre, saved.xm[], w.gate_w[], lora.gate_w, M, features, features,
            ctx, _i8w(w,3), _i8s(w, 3), c_off, c_len,
        )
        g_wq = b_q.lora.copy()
        g_wk = b_k.lora.copy()
        g_wv = b_v.lora.copy()
        g_gate = b_g.lora.copy()
        d_xm = add(add(b_q.d_x, b_k.d_x, ctx), add(b_v.d_x, b_g.d_x, ctx), ctx)
    else:
        var bw_qkvg = _linear_bwd_dx_group4(
            d_q, d_k, d_v_flat, d_gate_pre, saved.xm[],
            w.wq[], lora.wq, HEADS * HEADDIM,
            w.wk[], lora.wk, KVHEADS * HEADDIM,
            w.wv[], lora.wv, KVHEADS * HEADDIM,
            w.gate_w[], lora.gate_w, features,
            M, features, ctx,
            _i8w(w,0), _i8s(w, 0), _i8w(w,1), _i8s(w, 1),
            _i8w(w,2), _i8s(w, 2), _i8w(w,3), _i8s(w, 3),
        )
        g_wq = bw_qkvg.g0.copy()
        g_wk = bw_qkvg.g1.copy()
        g_wv = bw_qkvg.g2.copy()
        g_gate = bw_qkvg.g3.copy()
        d_xm = bw_qkvg.d_x.clone(ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "d_xm", d_xm, False, L, (real_len.value() if real_len else L), ctx)

    # xm = modulate(xn, prescale, preshift) → d_xn (drop param grads).
    var mb1_dx: Tensor
    if seg_mod:
        var sm1 = _modulate_backward_seg2(
            reshape(cast_tensor(d_xm, saved.xn[].dtype(), ctx),
                    [1, L, features], ctx),
            saved.xn[],
            cast_tensor(prescale[], saved.xn[].dtype(), ctx),
            cast_tensor(mods_c[0][], saved.xn[].dtype(), ctx),
            split, rl_rows, ctx,
        )
        mb1_dx = sm1.d_x.clone(ctx)
        dv_t.append(TArc(sm1.d_scale_t.clone(ctx)))
        dv_t.append(TArc(sm1.d_shift_t.clone(ctx)))
        dv_c.append(TArc(sm1.d_scale_c.clone(ctx)))
        dv_c.append(TArc(sm1.d_shift_c.clone(ctx)))
    else:
        var mb1 = modulate_backward(cast_tensor(d_xm, saved.xn[].dtype(), ctx), saved.xn[], cast_tensor(prescale[], saved.xn[].dtype(), ctx), ctx, compute_param_grads=False)
        mb1_dx = mb1.d_x.clone(ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "mb1.d_x", mb1_dx, False, L, (real_len.value() if real_len else L), ctx)
    # xn = prenorm(x) (weight=prenorm+1, FROZEN) → d_x via rms_norm_backward. Same
    # mixed-precision mirror: saved.x is the bf16 block input, prenorm scale is F32
    # → cast (scale+1) down to the act dtype for the all-bf16 path (matches fwd).
    var rb1_dx = krea2_rmsnorm_backward_dx(mb1_dx, saved.x[], w.prenorm_scale[], eps, ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "rb1_dx", rb1_dx, False, L, (real_len.value() if real_len else L), ctx)

    # x feeds: residual (grg1.d_x), prenorm(x) (rb1.d_x). SUM.
    var d_x = add(grg1_dx, rb1_dx, ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b1", dbg_block, "d_x", d_x, False, L, (real_len.value() if real_len else L), ctx)

    var dvt = Optional[TArc](None)
    var dvc = Optional[TArc](None)
    if seg_mod:
        dvt = Optional[TArc](TArc(_pack_d_vec(dv_t, ctx)))
        dvc = Optional[TArc](TArc(_pack_d_vec(dv_c, ctx)))

    return Krea2BlockGrads(
        TArc(d_x^),
        g_wq^, g_wk^, g_wv^, g_gate^, g_wo^, g_mg^, g_mu^, g_down^,
        dvt^, dvc^,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRUE BATCH-2 single-stream block (row-stacked [seq0 | seq1] = [1, 2L, features]).
# Every token-parallel op (GEMMs incl. LoRA, RMS norms, swiglu) runs over the 2L
# rows unchanged — GEMM is ~63% of the step, so this is where the batching win is.
# Per-sample adaLN via [2, features] modulation chunks (modulate/residual_gate split
# the 2L rows into two contiguous L ranges). Attention runs TWO separate cuDNN
# flash-padmask calls (one per sample slice with its own real_len) → NO cross-sample
# attention; the cuDNN tail-mask machinery is byte-unchanged. Requires BOTH samples
# in the flash-padmask regime (real_len < L, the bucketed/padded production regime)
# — FAIL LOUD otherwise. The per-sample flash tensors are CONCATENATED back into the
# reused Krea2BlockSaved (q/k/v/o [1,2L,H,Dh]; stats [1,H,2L,1]); the backward slices
# them per sample. LoRA grads accumulate over BOTH sample halves (that IS the batch
# gradient). Keep in lockstep with krea2_single_stream_block_lora{,_backward}.
# ══════════════════════════════════════════════════════════════════════════════
def krea2_single_stream_block_lora_b2[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    x_t: TArc,            # [1, 2L, features]  row-stacked [seq0 | seq1]
    vec: Tensor,          # [2, 6*features]    per-sample timestep modulation vec
    w: Krea2BlockWeights, lora: Krea2BlockLora,
    cos_q: Tensor, sin_q: Tensor,   # [2L*HEADS, HEADDIM/2]   per-sample-tiled + concat
    cos_k: Tensor, sin_k: Tensor,   # [2L*KVHEADS, HEADDIM/2]
    eps: Float32,
    ctx: DeviceContext,
    real_len0: Int, real_len1: Int,   # per-sample valid-prefix lengths (1 <= rl < L)
) raises -> Krea2BlockForward:
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    comptime L2 = 2 * L
    if real_len0 >= L or real_len0 < 1 or real_len1 >= L or real_len1 < 1:
        raise Error(
            String("krea2 b2 block: requires both samples in the flash-padmask")
            + " regime (1 <= real_len < L); got real_len0=" + String(real_len0)
            + " real_len1=" + String(real_len1) + " L=" + String(L)
            + " (raise KREA2_LTMAX above the batch's max caption length)"
        )
    var M = L2
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    # per-sample raw mod chunks (each [2, features]).
    var mods = _mod6_b2(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var preshift = mods[1]
    var pregate = mods[2]
    var postscale = mods[3]
    var postshift = mods[4]
    var postgate = mods[5]

    # ── ATTENTION branch (per-token ops over 2L rows; per-sample adaLN) ──────────
    var xn = krea2_rmsnorm(x_t[], w.prenorm_scale[], eps, ctx)      # [1,2L,features]
    var xm = modulate(xn, prescale[], preshift[], ctx)             # per-sample [2,F]

    # int8 W8A8 dispatch (== the b1 sites): the frozen BASE runs int8 when
    # w.int8 is present; LoRA delta stays bf16.
    var q = _linear_lora(xm, w.wq[], lora.wq, M, ctx, _i8w(w, 0), _i8s(w, 0))      # [1,2L,HEADS*HEADDIM]
    var k = _linear_lora(xm, w.wk[], lora.wk, M, ctx, _i8w(w, 1), _i8s(w, 1))     # [1,2L,KVHEADS*HEADDIM]
    var v_lin = _linear_lora(xm, w.wv[], lora.wv, M, ctx, _i8w(w, 2), _i8s(w, 2))
    var gate_pre = _linear_lora(xm, w.gate_w[], lora.gate_w, M, ctx, _i8w(w, 3), _i8s(w, 3))  # [1,2L,features]

    var q_pre = reshape_owned(q^, [1, L2, HEADS, HEADDIM])
    var k_pre = reshape_owned(k^, [1, L2, KVHEADS, HEADDIM])
    var v = reshape_owned(v_lin^, [1, L2, KVHEADS, HEADDIM])

    var q_rms = krea2_rmsnorm(q_pre, w.qnorm_scale[], eps, ctx)
    var k_rms = krea2_rmsnorm(k_pre, w.knorm_scale[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos_q, sin_q, ctx)         # [1,2L,HEADS,HEADDIM]
    var k_rope = rope_interleaved(k_rms, cos_k, sin_k, ctx)
    var k_full = repeat_kv_f32(k_rope, L2, KVHEADS, n_rep, HEADDIM, ctx)  # [1,2L,HEADS,HEADDIM]
    var v_full = repeat_kv_f32(v, L2, KVHEADS, n_rep, HEADDIM, ctx)

    # SDPA: TWO cuDNN flash-padmask calls (one per sample slice, own real_len).
    var q0 = slice(q_rope, 1, 0, L, ctx)     # [1,L,HEADS,HEADDIM]
    var q1 = slice(q_rope, 1, L, L, ctx)
    var k0 = slice(k_full, 1, 0, L, ctx)
    var k1 = slice(k_full, 1, L, L, ctx)
    var v0 = slice(v_full, 1, 0, L, ctx)
    var v1 = slice(v_full, 1, L, L, ctx)
    var ff0 = sdpa_flash_train_fwd_padmask_bf16[1, L, HEADS, HEADDIM](
        q0, k0, v0, real_len0, scale, ctx
    )
    var ff1 = sdpa_flash_train_fwd_padmask_bf16[1, L, HEADS, HEADDIM](
        q1, k1, v1, real_len1, scale, ctx
    )
    var att0 = Tensor(ff0.att.buf.copy(), ff0.att.shape(), ff0.att.dtype())
    var att1 = Tensor(ff1.att.buf.copy(), ff1.att.shape(), ff1.att.dtype())
    var att = concat(1, ctx, att0, att1)     # [1,2L,HEADS,HEADDIM]
    var flash_q = Optional[TArc](TArc(concat(1, ctx, ff0.q_bf[], ff1.q_bf[])))
    var flash_k = Optional[TArc](TArc(concat(1, ctx, ff0.k_bf[], ff1.k_bf[])))
    var flash_v = Optional[TArc](TArc(concat(1, ctx, ff0.v_bf[], ff1.v_bf[])))
    var flash_o = Optional[TArc](TArc(concat(1, ctx, ff0.o_bf[], ff1.o_bf[])))
    var flash_stats = Optional[TArc](TArc(concat(2, ctx, ff0.stats[], ff1.stats[])))  # [1,H,2L,1]
    var attn_flat = reshape_owned(att^, [1, L2, features])

    var sg = sigmoid(gate_pre, ctx)
    var gated = mul(attn_flat, sg, ctx)
    var a = _linear_lora(gated, w.wo[], lora.wo, M, ctx, _i8w(w, 4), _i8s(w, 4))

    var x1 = residual_gate(x_t[], pregate[], a, ctx)               # per-sample gate

    # ── MLP branch ──────────────────────────────────────────────────────────────
    var xn2 = krea2_rmsnorm(x1, w.postnorm_scale[], eps, ctx)
    var xm2 = modulate(xn2, postscale[], postshift[], ctx)

    var mg = _linear_lora(xm2, w.mlp_gate_w[], lora.mlp_gate_w, M, ctx, _i8w(w, 5), _i8s(w, 5))
    var mu = _linear_lora(xm2, w.mlp_up_w[], lora.mlp_up_w, M, ctx, _i8w(w, 6), _i8s(w, 6))
    var sw = swiglu(mg, mu, ctx)
    var m = _linear_lora(sw, w.mlp_down_w[], lora.mlp_down_w, M, ctx, _i8w(w, 7), _i8s(w, 7))

    var x2 = residual_gate(x1, postgate[], m, ctx)

    var saved = Krea2BlockSaved(
        x_t.copy(), TArc(xm^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rope^), TArc(k_rope^), TArc(k_full^), TArc(v_full^),
        TArc(attn_flat^), TArc(gate_pre^), TArc(sg^), TArc(gated^),
        TArc(a^), TArc(x1^), TArc(xm2^),
        TArc(mg^), TArc(mu^), TArc(sw^), TArc(m^),
        TArc(xn^), TArc(xn2^),
        flash_q^, flash_k^, flash_v^, flash_o^, flash_stats^,
    )
    return Krea2BlockForward(TArc(x2^), saved^)


def krea2_single_stream_block_lora_backward_b2[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out: Tensor,        # [1, 2L, features]  upstream grad of the stacked block output
    vec: Tensor,          # [2, 6*features]
    w: Krea2BlockWeights, lora: Krea2BlockLora, saved: Krea2BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len0: Int, real_len1: Int,   # MUST match the forward call (per sample).
    dbg_block: Int = -1,   # op-by-op probe: >=0 prints backward intermediates.
) raises -> Krea2BlockGrads:
    """TRUE batch-2 block backward. Mirrors krea2_single_stream_block_lora_backward,
    but modulate/gate use per-sample [2, features] chunks and the SDPA backward runs
    TWO cuDNN flash-padmask backwards (per sample slice, own real_len), concatenating
    dQ/dK/dV. The 8 LoRA dA/dB accumulate over BOTH sample halves (the batch grad)."""
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    comptime L2 = 2 * L
    var mlpdim = saved.mlp_gate[].shape()[2]
    var M = L2
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6_b2(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var pregate = mods[2]
    var postscale = mods[3]
    var postgate = mods[5]

    # ── MLP branch backward ─────────────────────────────────────────────────────
    # Op-probe dumps are placed IMMEDIATELY after each assignment (before the
    # consuming op) so the tensor is still live — a late/batched dump can read a
    # buffer Mojo already ASAP-freed and reused (stale-read artifact).
    var grg2 = gate_residual_backward(d_out, saved.x1[], postgate[], saved.m[], ctx, compute_gate_grad=False)
    var d_m = grg2.d_y.clone(ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "d_out", d_out, True, L, real_len0, ctx)
            _opdbg("b2", dbg_block, "grg2.d_x", grg2.d_x, True, L, real_len0, ctx)
            _opdbg("b2", dbg_block, "d_m", d_m, True, L, real_len0, ctx)
    var bw_down = _linear_bwd_dx(
        d_m, saved.sw[], w.mlp_down_w[], lora.mlp_down_w, M, mlpdim, features, ctx,
        _i8w(w,7), _i8s(w, 7),
    )
    var d_sw = bw_down.d_x.clone(ctx)
    var g_down = bw_down.lora.copy()
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "d_sw", d_sw, True, L, real_len0, ctx)

    var sgb = swiglu_backward(d_sw, saved.mlp_gate[], saved.mlp_up[], ctx)
    var bw_mlp = _linear_bwd_dx_group2(
        sgb.d_gate, sgb.d_up, saved.xm2[],
        w.mlp_gate_w[], lora.mlp_gate_w, mlpdim,
        w.mlp_up_w[], lora.mlp_up_w, mlpdim,
        M, features, ctx,
        _i8w(w,5), _i8s(w, 5), _i8w(w,6), _i8s(w, 6),
    )
    var g_mg = bw_mlp.g0.copy()
    var g_mu = bw_mlp.g1.copy()
    var d_xm2 = bw_mlp.d_x.clone(ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "d_xm2", d_xm2, True, L, real_len0, ctx)

    var mb2 = modulate_backward(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx), saved.xn2[], cast_tensor(postscale[], saved.xn2[].dtype(), ctx), ctx, compute_param_grads=False)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "mb2.d_x", mb2.d_x, True, L, real_len0, ctx)
    var rb2_dx = krea2_rmsnorm_backward_dx(mb2.d_x, saved.x1[], w.postnorm_scale[], eps, ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "rb2_dx", rb2_dx, True, L, real_len0, ctx)
    var d_x1 = add(grg2.d_x, rb2_dx, ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "d_x1", d_x1, True, L, real_len0, ctx)

    # ── ATTENTION branch backward ───────────────────────────────────────────────
    var grg1 = gate_residual_backward(d_x1, saved.x[], pregate[], saved.a[], ctx, compute_gate_grad=False)
    var d_a = grg1.d_y.clone(ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "grg1.d_x", grg1.d_x, True, L, real_len0, ctx)
            _opdbg("b2", dbg_block, "d_a", d_a, True, L, real_len0, ctx)
    var bw_wo = _linear_bwd_dx(
        d_a, saved.gated[], w.wo[], lora.wo, M, features, features, ctx,
        _i8w(w,4), _i8s(w, 4),
    )
    var d_gated = bw_wo.d_x.clone(ctx)
    var g_wo = bw_wo.lora.copy()
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "d_gated", d_gated, True, L, real_len0, ctx)

    var d_attn_flat = mul(d_gated, saved.sg[], ctx)
    var d_sg = mul(d_gated, saved.attn_flat[], ctx)
    var d_gate_pre = sigmoid_backward_from_output(d_sg, saved.sg[], ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "d_attn_flat", d_attn_flat, True, L, real_len0, ctx)
            _opdbg("b2", dbg_block, "d_gate_pre", d_gate_pre, True, L, real_len0, ctx)

    # attn_flat = reshape(concat(sdpa0, sdpa1)) → TWO cuDNN flash-padmask backwards
    # from the per-sample slices of the saved (concatenated) flash tape. FLASH dQ is
    # NONDETERMINISTIC (cuDNN atomics) so dQ-derived grads are value-tolerance.
    if not saved.flash_stats:
        raise Error(
            "krea2 b2 block bwd: saved tape has no flash set"
            " (forward/backward real_len mismatch)"
        )
    var fq = saved.flash_q.value()
    var fk = saved.flash_k.value()
    var fv = saved.flash_v.value()
    var fo = saved.flash_o.value()
    var fs = saved.flash_stats.value()
    var d_att = reshape(d_attn_flat, [1, L2, HEADS, HEADDIM], ctx)
    var d_att0 = slice(d_att, 1, 0, L, ctx)
    var d_att1 = slice(d_att, 1, L, L, ctx)
    var fb0 = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
        TArc(slice(fq[], 1, 0, L, ctx)), TArc(slice(fk[], 1, 0, L, ctx)),
        TArc(slice(fv[], 1, 0, L, ctx)), TArc(slice(fo[], 1, 0, L, ctx)),
        TArc(slice(fs[], 2, 0, L, ctx)), d_att0, real_len0, scale, ctx,
    )
    var fb1 = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
        TArc(slice(fq[], 1, L, L, ctx)), TArc(slice(fk[], 1, L, L, ctx)),
        TArc(slice(fv[], 1, L, L, ctx)), TArc(slice(fo[], 1, L, L, ctx)),
        TArc(slice(fs[], 2, L, L, ctx)), d_att1, real_len1, scale, ctx,
    )
    var d_q_sb = concat(1, ctx, fb0.d_q, fb1.d_q)   # [1,2L,HEADS,Dh]
    var d_k_sb = concat(1, ctx, fb0.d_k, fb1.d_k)
    var d_v_sb = concat(1, ctx, fb0.d_v, fb1.d_v)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "d_q_sb", d_q_sb, True, L, real_len0, ctx)
            _opdbg("b2", dbg_block, "d_k_sb", d_k_sb, True, L, real_len0, ctx)
            _opdbg("b2", dbg_block, "d_v_sb", d_v_sb, True, L, real_len0, ctx)

    var d_k_rope = repeat_kv_backward(d_k_sb, L2, KVHEADS, n_rep, HEADDIM, ctx)
    var d_v = repeat_kv_backward(d_v_sb, L2, KVHEADS, n_rep, HEADDIM, ctx)
    var d_q_rms = rope_backward(d_q_sb, cos_q, sin_q, True, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, True, ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "d_q_rms", d_q_rms, True, L, real_len0, ctx)

    var rbq_dx = krea2_rmsnorm_backward_dx(d_q_rms, saved.q_pre[], w.qnorm_scale[], eps, ctx)
    var rbk_dx = krea2_rmsnorm_backward_dx(d_k_rms, saved.k_pre[], w.knorm_scale[], eps, ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "rbq_dx", rbq_dx, True, L, real_len0, ctx)

    var d_q = reshape(rbq_dx, [1, L2, HEADS * HEADDIM], ctx)
    var d_k = reshape(rbk_dx, [1, L2, KVHEADS * HEADDIM], ctx)
    var d_v_flat = reshape(d_v, [1, L2, KVHEADS * HEADDIM], ctx)

    var bw_qkvg = _linear_bwd_dx_group4(
        d_q, d_k, d_v_flat, d_gate_pre, saved.xm[],
        w.wq[], lora.wq, HEADS * HEADDIM,
        w.wk[], lora.wk, KVHEADS * HEADDIM,
        w.wv[], lora.wv, KVHEADS * HEADDIM,
        w.gate_w[], lora.gate_w, features,
        M, features, ctx,
        _i8w(w,0), _i8s(w, 0), _i8w(w,1), _i8s(w, 1),
        _i8w(w,2), _i8s(w, 2), _i8w(w,3), _i8s(w, 3),
    )
    var g_wq = bw_qkvg.g0.copy()
    var g_wk = bw_qkvg.g1.copy()
    var g_wv = bw_qkvg.g2.copy()
    var g_gate = bw_qkvg.g3.copy()
    var d_xm = bw_qkvg.d_x.clone(ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "d_xm", d_xm, True, L, real_len0, ctx)

    var mb1 = modulate_backward(cast_tensor(d_xm, saved.xn[].dtype(), ctx), saved.xn[], cast_tensor(prescale[], saved.xn[].dtype(), ctx), ctx, compute_param_grads=False)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "mb1.d_x", mb1.d_x, True, L, real_len0, ctx)
    var rb1_dx = krea2_rmsnorm_backward_dx(mb1.d_x, saved.x[], w.prenorm_scale[], eps, ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "rb1_dx", rb1_dx, True, L, real_len0, ctx)
    var d_x = add(grg1.d_x, rb1_dx, ctx)
    comptime if KREA2_OPDBG:
        if dbg_block >= 0:
            _opdbg("b2", dbg_block, "d_x", d_x, True, L, real_len0, ctx)

    return Krea2BlockGrads(
        TArc(d_x^),
        g_wq^, g_wk^, g_wv^, g_gate^, g_wo^, g_mg^, g_mu^, g_down^,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TRUE BATCH-2 DEVICE-GRAD BACKWARD — byte-for-byte the batch-2 block backward math
# (krea2_single_stream_block_lora_backward_b2 above), but the four projection
# backwards use the _dev helper variants so the 8 LoRA dA/dB stay on DEVICE
# (Krea2BlockGradsT, no per-adapter to_host). ONLY the grad packaging differs — the
# two flash-padmask backwards, the per-sample slicing, the row-stacked GEMMs and the
# [2,F] modulation are UNCHANGED. Like the B1 _dev backward it drops the OPDBG probes
# (the host b2 backward is the bit-gate oracle). Keep in lockstep with the b2 host bwd.
# ══════════════════════════════════════════════════════════════════════════════
def krea2_single_stream_block_lora_backward_b2_dev[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out: Tensor,        # [1, 2L, features]  upstream grad of the stacked block output
    vec: Tensor,          # [2, 6*features]
    w: Krea2BlockWeights, lora: Krea2BlockLora, saved: Krea2BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len0: Int, real_len1: Int,   # MUST match the forward call (per sample).
) raises -> Krea2BlockGradsT:
    """TRUE batch-2 device-grad block backward. Identical math to
    krea2_single_stream_block_lora_backward_b2 — modulate/gate use per-sample
    [2, features] chunks and the SDPA backward runs TWO cuDNN flash-padmask backwards
    (per sample slice, own real_len), concatenating dQ/dK/dV. The ONLY difference is
    the four projection backwards use the _dev helpers so the 8 LoRA dA/dB stay on
    DEVICE (Krea2BlockGradsT). The grads still accumulate over BOTH sample halves."""
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    comptime L2 = 2 * L
    var mlpdim = saved.mlp_gate[].shape()[2]
    var M = L2
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6_b2(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var pregate = mods[2]
    var postscale = mods[3]
    var postgate = mods[5]

    # ── MLP branch backward ─────────────────────────────────────────────────────
    var grg2 = gate_residual_backward(d_out, saved.x1[], postgate[], saved.m[], ctx, compute_gate_grad=False)
    var d_m = grg2.d_y.clone(ctx)
    var bw_down = _linear_bwd_dx_dev(
        d_m, saved.sw[], w.mlp_down_w[], lora.mlp_down_w, M, mlpdim, features, ctx,
        _i8w(w,7), _i8s(w, 7),
    )
    var d_sw = bw_down.d_x.clone(ctx)
    var g_down = bw_down.lora.copy()

    var sgb = swiglu_backward(d_sw, saved.mlp_gate[], saved.mlp_up[], ctx)
    var bw_mlp = _linear_bwd_dx_group2_dev(
        sgb.d_gate, sgb.d_up, saved.xm2[],
        w.mlp_gate_w[], lora.mlp_gate_w, mlpdim,
        w.mlp_up_w[], lora.mlp_up_w, mlpdim,
        M, features, ctx,
        _i8w(w,5), _i8s(w, 5), _i8w(w,6), _i8s(w, 6),
    )
    var g_mg = bw_mlp.g0.copy()
    var g_mu = bw_mlp.g1.copy()
    var d_xm2 = bw_mlp.d_x.clone(ctx)

    var mb2 = modulate_backward(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx), saved.xn2[], cast_tensor(postscale[], saved.xn2[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb2_dx = krea2_rmsnorm_backward_dx(mb2.d_x, saved.x1[], w.postnorm_scale[], eps, ctx)
    var d_x1 = add(grg2.d_x, rb2_dx, ctx)

    # ── ATTENTION branch backward ───────────────────────────────────────────────
    var grg1 = gate_residual_backward(d_x1, saved.x[], pregate[], saved.a[], ctx, compute_gate_grad=False)
    var d_a = grg1.d_y.clone(ctx)
    var bw_wo = _linear_bwd_dx_dev(
        d_a, saved.gated[], w.wo[], lora.wo, M, features, features, ctx,
        _i8w(w,4), _i8s(w, 4),
    )
    var d_gated = bw_wo.d_x.clone(ctx)
    var g_wo = bw_wo.lora.copy()

    var d_attn_flat = mul(d_gated, saved.sg[], ctx)
    var d_sg = mul(d_gated, saved.attn_flat[], ctx)
    var d_gate_pre = sigmoid_backward_from_output(d_sg, saved.sg[], ctx)

    # attn_flat = reshape(concat(sdpa0, sdpa1)) → TWO cuDNN flash-padmask backwards
    # from the per-sample slices of the saved (concatenated) flash tape. FLASH dQ is
    # NONDETERMINISTIC (cuDNN atomics) so dQ-derived grads are value-tolerance.
    if not saved.flash_stats:
        raise Error(
            "krea2 b2 block bwd (dev): saved tape has no flash set"
            " (forward/backward real_len mismatch)"
        )
    var fq = saved.flash_q.value()
    var fk = saved.flash_k.value()
    var fv = saved.flash_v.value()
    var fo = saved.flash_o.value()
    var fs = saved.flash_stats.value()
    var d_att = reshape(d_attn_flat, [1, L2, HEADS, HEADDIM], ctx)
    var d_att0 = slice(d_att, 1, 0, L, ctx)
    var d_att1 = slice(d_att, 1, L, L, ctx)
    var fb0 = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
        TArc(slice(fq[], 1, 0, L, ctx)), TArc(slice(fk[], 1, 0, L, ctx)),
        TArc(slice(fv[], 1, 0, L, ctx)), TArc(slice(fo[], 1, 0, L, ctx)),
        TArc(slice(fs[], 2, 0, L, ctx)), d_att0, real_len0, scale, ctx,
    )
    var fb1 = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
        TArc(slice(fq[], 1, L, L, ctx)), TArc(slice(fk[], 1, L, L, ctx)),
        TArc(slice(fv[], 1, L, L, ctx)), TArc(slice(fo[], 1, L, L, ctx)),
        TArc(slice(fs[], 2, L, L, ctx)), d_att1, real_len1, scale, ctx,
    )
    var d_q_sb = concat(1, ctx, fb0.d_q, fb1.d_q)   # [1,2L,HEADS,Dh]
    var d_k_sb = concat(1, ctx, fb0.d_k, fb1.d_k)
    var d_v_sb = concat(1, ctx, fb0.d_v, fb1.d_v)

    var d_k_rope = repeat_kv_backward(d_k_sb, L2, KVHEADS, n_rep, HEADDIM, ctx)
    var d_v = repeat_kv_backward(d_v_sb, L2, KVHEADS, n_rep, HEADDIM, ctx)
    var d_q_rms = rope_backward(d_q_sb, cos_q, sin_q, True, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, True, ctx)

    var rbq_dx = krea2_rmsnorm_backward_dx(d_q_rms, saved.q_pre[], w.qnorm_scale[], eps, ctx)
    var rbk_dx = krea2_rmsnorm_backward_dx(d_k_rms, saved.k_pre[], w.knorm_scale[], eps, ctx)

    var d_q = reshape(rbq_dx, [1, L2, HEADS * HEADDIM], ctx)
    var d_k = reshape(rbk_dx, [1, L2, KVHEADS * HEADDIM], ctx)
    var d_v_flat = reshape(d_v, [1, L2, KVHEADS * HEADDIM], ctx)

    var bw_qkvg = _linear_bwd_dx_group4_dev(
        d_q, d_k, d_v_flat, d_gate_pre, saved.xm[],
        w.wq[], lora.wq, HEADS * HEADDIM,
        w.wk[], lora.wk, KVHEADS * HEADDIM,
        w.wv[], lora.wv, KVHEADS * HEADDIM,
        w.gate_w[], lora.gate_w, features,
        M, features, ctx,
        _i8w(w,0), _i8s(w, 0), _i8w(w,1), _i8s(w, 1),
        _i8w(w,2), _i8s(w, 2), _i8w(w,3), _i8s(w, 3),
    )
    var g_wq = bw_qkvg.g0.copy()
    var g_wk = bw_qkvg.g1.copy()
    var g_wv = bw_qkvg.g2.copy()
    var g_gate = bw_qkvg.g3.copy()
    var d_xm = bw_qkvg.d_x.clone(ctx)

    var mb1 = modulate_backward(cast_tensor(d_xm, saved.xn[].dtype(), ctx), saved.xn[], cast_tensor(prescale[], saved.xn[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb1_dx = krea2_rmsnorm_backward_dx(mb1.d_x, saved.x[], w.prenorm_scale[], eps, ctx)
    var d_x = add(grg1.d_x, rb1_dx, ctx)

    return Krea2BlockGradsT(
        TArc(d_x^),
        g_wq^, g_wk^, g_wv^, g_gate^, g_wo^, g_mg^, g_mu^, g_down^,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE-GRAD BACKWARD — bit-identical math to krea2_single_stream_block_lora_backward
# above, but the 8 LoRA dA/dB stay on DEVICE (_linear_bwd_dx_dev, no per-adapter
# to_host). Returns Krea2BlockGradsT. The streamed stack consumes those transient
# grads block-by-block: either per-block D2H for host-list compatibility or D2D
# preload into AdamW state. The body is otherwise a verbatim clone of the host
# backward — keep the two in lockstep if the block math changes.
# ══════════════════════════════════════════════════════════════════════════════
def krea2_single_stream_block_lora_backward_dev[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out: Tensor,        # [1, L, features] upstream grad of the block output
    vec: Tensor,          # [1, 6*features]  (for the raw mod chunks)
    w: Krea2BlockWeights, lora: Krea2BlockLora, saved: Krea2BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),  # MUST match the forward call
        # (same contract as the host backward).
    vec_cond: Optional[TArc] = Optional[TArc](None),  # OminiControl EDIT (C4) —
    cond_off: Optional[Int] = Optional[Int](None),    # same three switches, same
    cond_len: Optional[Int] = Optional[Int](None),    # contract, as the host
        # backward above. Absent (the default) = the pre-C4 path, bit-for-bit.
) raises -> Krea2BlockGradsT:
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var mlpdim = saved.mlp_gate[].shape()[2]
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var pregate = mods[2]
    var postscale = mods[3]
    var postgate = mods[5]

    # ── OminiControl EDIT switches (C4); see the host backward for the contract.
    var seg_mod = False
    var split = 0
    var mods_c = List[TArc]()
    if vec_cond:
        if cond_off:
            split = cond_off.value()
            if split > 0 and split < L:
                mods_c = _mod6(vec_cond.value()[], w.mod_lin[], features, ctx)
                seg_mod = True
    var c_off = -1
    var c_len = 0
    if cond_off:
        if cond_len:
            var co = cond_off.value()
            var cl = cond_len.value()
            if co > 0 and cl > 0 and co + cl <= L:
                c_off = co
                c_len = cl
    var rl_rows = real_len.value() if real_len else L
    if rl_rows > L:
        rl_rows = L
    if seg_mod and rl_rows < split:
        raise Error("krea2 block bwd (dev): real_len below the modulation split")
    var dv_t = List[TArc]()
    var dv_c = List[TArc]()

    # ── MLP branch backward ──────────────────────────────────────────────────
    var grg2_dx: Tensor
    var d_m: Tensor
    if seg_mod:
        var s2 = _gate_residual_backward_seg2(
            d_out, saved.x1[], postgate[], mods_c[5][], saved.m[],
            split, rl_rows, ctx,
        )
        grg2_dx = s2.d_x.clone(ctx)
        d_m = s2.d_y.clone(ctx)
        dv_t.append(TArc(s2.d_g_t.clone(ctx)))
        dv_c.append(TArc(s2.d_g_c.clone(ctx)))
    else:
        var grg2 = gate_residual_backward(d_out, saved.x1[], postgate[], saved.m[], ctx, compute_gate_grad=False)
        grg2_dx = grg2.d_x.clone(ctx)
        d_m = grg2.d_y.clone(ctx)

    var bw_down = _linear_bwd_dx_dev(
        d_m, saved.sw[], w.mlp_down_w[], lora.mlp_down_w, M, mlpdim, features, ctx,
        _i8w(w,7), _i8s(w, 7), c_off, c_len,
    )
    var d_sw = bw_down.d_x.clone(ctx)
    var g_down = bw_down.lora.copy()

    var sgb = swiglu_backward(d_sw, saved.mlp_gate[], saved.mlp_up[], ctx)
    var g_mg: Krea2LoraGradT
    var g_mu: Krea2LoraGradT
    var d_xm2: Tensor
    if c_off >= 0:
        var b_mg = _linear_bwd_dx_dev(
            sgb.d_gate, saved.xm2[], w.mlp_gate_w[], lora.mlp_gate_w,
            M, features, mlpdim, ctx, _i8w(w,5), _i8s(w, 5), c_off, c_len,
        )
        var b_mu = _linear_bwd_dx_dev(
            sgb.d_up, saved.xm2[], w.mlp_up_w[], lora.mlp_up_w,
            M, features, mlpdim, ctx, _i8w(w,6), _i8s(w, 6), c_off, c_len,
        )
        g_mg = b_mg.lora.copy()
        g_mu = b_mu.lora.copy()
        d_xm2 = add(b_mg.d_x, b_mu.d_x, ctx)
    else:
        var bw_mlp = _linear_bwd_dx_group2_dev(
            sgb.d_gate, sgb.d_up, saved.xm2[],
            w.mlp_gate_w[], lora.mlp_gate_w, mlpdim,
            w.mlp_up_w[], lora.mlp_up_w, mlpdim,
            M, features, ctx,
            _i8w(w,5), _i8s(w, 5), _i8w(w,6), _i8s(w, 6),
        )
        g_mg = bw_mlp.g0.copy()
        g_mu = bw_mlp.g1.copy()
        d_xm2 = bw_mlp.d_x.clone(ctx)

    var mb2_dx: Tensor
    if seg_mod:
        var sm2 = _modulate_backward_seg2(
            reshape(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx),
                    [1, L, features], ctx),
            saved.xn2[],
            cast_tensor(postscale[], saved.xn2[].dtype(), ctx),
            cast_tensor(mods_c[3][], saved.xn2[].dtype(), ctx),
            split, rl_rows, ctx,
        )
        mb2_dx = sm2.d_x.clone(ctx)
        dv_t.append(TArc(sm2.d_scale_t.clone(ctx)))
        dv_t.append(TArc(sm2.d_shift_t.clone(ctx)))
        dv_c.append(TArc(sm2.d_scale_c.clone(ctx)))
        dv_c.append(TArc(sm2.d_shift_c.clone(ctx)))
    else:
        var mb2 = modulate_backward(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx), saved.xn2[], cast_tensor(postscale[], saved.xn2[].dtype(), ctx), ctx, compute_param_grads=False)
        mb2_dx = mb2.d_x.clone(ctx)
    var rb2_dx = krea2_rmsnorm_backward_dx(mb2_dx, saved.x1[], w.postnorm_scale[], eps, ctx)
    var d_x1 = add(grg2_dx, rb2_dx, ctx)

    # ── ATTENTION branch backward ────────────────────────────────────────────
    var grg1_dx: Tensor
    var d_a: Tensor
    if seg_mod:
        var s1 = _gate_residual_backward_seg2(
            d_x1, saved.x[], pregate[], mods_c[2][], saved.a[],
            split, rl_rows, ctx,
        )
        grg1_dx = s1.d_x.clone(ctx)
        d_a = s1.d_y.clone(ctx)
        dv_t.append(TArc(s1.d_g_t.clone(ctx)))
        dv_c.append(TArc(s1.d_g_c.clone(ctx)))
    else:
        var grg1 = gate_residual_backward(d_x1, saved.x[], pregate[], saved.a[], ctx, compute_gate_grad=False)
        grg1_dx = grg1.d_x.clone(ctx)
        d_a = grg1.d_y.clone(ctx)

    var bw_wo = _linear_bwd_dx_dev(
        d_a, saved.gated[], w.wo[], lora.wo, M, features, features, ctx,
        _i8w(w,4), _i8s(w, 4), c_off, c_len,
    )
    var d_gated = bw_wo.d_x.clone(ctx)
    var g_wo = bw_wo.lora.copy()

    var d_attn_flat = mul(d_gated, saved.sg[], ctx)
    var d_sg = mul(d_gated, saved.attn_flat[], ctx)
    var d_gate_pre = sigmoid_backward_from_output(d_sg, saved.sg[], ctx)

    var d_att = reshape(d_attn_flat, [1, L, HEADS, HEADDIM], ctx)
    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    var bwd_use_flash = real_len and real_len.value() < L
    if bwd_use_flash:
        if not saved.flash_stats:
            raise Error(
                "krea2 block bwd (dev): real_len < L but saved tape has no flash set"
                " (forward/backward real_len mismatch)"
            )
        var rl = real_len.value()
        var fb = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
            saved.flash_q.value(), saved.flash_k.value(), saved.flash_v.value(),
            saved.flash_o.value(), saved.flash_stats.value(), d_att, rl, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward[1, L, HEADS, HEADDIM](
            saved.q_rope[], saved.k_full[], saved.v_full[], d_att, scale, ctx
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_k_rope = repeat_kv_backward(d_k_sb, L, KVHEADS, n_rep, HEADDIM, ctx)
    var d_v = repeat_kv_backward(d_v_sb, L, KVHEADS, n_rep, HEADDIM, ctx)

    var d_q_rms = rope_backward(d_q_sb, cos_q, sin_q, True, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, True, ctx)

    var rbq_dx = krea2_rmsnorm_backward_dx(d_q_rms, saved.q_pre[], w.qnorm_scale[], eps, ctx)
    var rbk_dx = krea2_rmsnorm_backward_dx(d_k_rms, saved.k_pre[], w.knorm_scale[], eps, ctx)

    var d_q = reshape(rbq_dx, [1, L, HEADS * HEADDIM], ctx)
    var d_k = reshape(rbk_dx, [1, L, KVHEADS * HEADDIM], ctx)
    var d_v_flat = reshape(d_v, [1, L, KVHEADS * HEADDIM], ctx)

    var g_wq: Krea2LoraGradT
    var g_wk: Krea2LoraGradT
    var g_wv: Krea2LoraGradT
    var g_gate: Krea2LoraGradT
    var d_xm: Tensor
    if c_off >= 0:
        var b_q = _linear_bwd_dx_dev(
            d_q, saved.xm[], w.wq[], lora.wq, M, features, HEADS * HEADDIM, ctx,
            _i8w(w,0), _i8s(w, 0), c_off, c_len,
        )
        var b_k = _linear_bwd_dx_dev(
            d_k, saved.xm[], w.wk[], lora.wk, M, features, KVHEADS * HEADDIM, ctx,
            _i8w(w,1), _i8s(w, 1), c_off, c_len,
        )
        var b_v = _linear_bwd_dx_dev(
            d_v_flat, saved.xm[], w.wv[], lora.wv, M, features, KVHEADS * HEADDIM,
            ctx, _i8w(w,2), _i8s(w, 2), c_off, c_len,
        )
        var b_g = _linear_bwd_dx_dev(
            d_gate_pre, saved.xm[], w.gate_w[], lora.gate_w, M, features, features,
            ctx, _i8w(w,3), _i8s(w, 3), c_off, c_len,
        )
        g_wq = b_q.lora.copy()
        g_wk = b_k.lora.copy()
        g_wv = b_v.lora.copy()
        g_gate = b_g.lora.copy()
        d_xm = add(add(b_q.d_x, b_k.d_x, ctx), add(b_v.d_x, b_g.d_x, ctx), ctx)
    else:
        var bw_qkvg = _linear_bwd_dx_group4_dev(
            d_q, d_k, d_v_flat, d_gate_pre, saved.xm[],
            w.wq[], lora.wq, HEADS * HEADDIM,
            w.wk[], lora.wk, KVHEADS * HEADDIM,
            w.wv[], lora.wv, KVHEADS * HEADDIM,
            w.gate_w[], lora.gate_w, features,
            M, features, ctx,
            _i8w(w,0), _i8s(w, 0), _i8w(w,1), _i8s(w, 1),
            _i8w(w,2), _i8s(w, 2), _i8w(w,3), _i8s(w, 3),
        )
        g_wq = bw_qkvg.g0.copy()
        g_wk = bw_qkvg.g1.copy()
        g_wv = bw_qkvg.g2.copy()
        g_gate = bw_qkvg.g3.copy()
        d_xm = bw_qkvg.d_x.clone(ctx)

    var mb1_dx: Tensor
    if seg_mod:
        var sm1 = _modulate_backward_seg2(
            reshape(cast_tensor(d_xm, saved.xn[].dtype(), ctx),
                    [1, L, features], ctx),
            saved.xn[],
            cast_tensor(prescale[], saved.xn[].dtype(), ctx),
            cast_tensor(mods_c[0][], saved.xn[].dtype(), ctx),
            split, rl_rows, ctx,
        )
        mb1_dx = sm1.d_x.clone(ctx)
        dv_t.append(TArc(sm1.d_scale_t.clone(ctx)))
        dv_t.append(TArc(sm1.d_shift_t.clone(ctx)))
        dv_c.append(TArc(sm1.d_scale_c.clone(ctx)))
        dv_c.append(TArc(sm1.d_shift_c.clone(ctx)))
    else:
        var mb1 = modulate_backward(cast_tensor(d_xm, saved.xn[].dtype(), ctx), saved.xn[], cast_tensor(prescale[], saved.xn[].dtype(), ctx), ctx, compute_param_grads=False)
        mb1_dx = mb1.d_x.clone(ctx)
    var rb1_dx = krea2_rmsnorm_backward_dx(mb1_dx, saved.x[], w.prenorm_scale[], eps, ctx)

    var d_x = add(grg1_dx, rb1_dx, ctx)

    var dvt = Optional[TArc](None)
    var dvc = Optional[TArc](None)
    if seg_mod:
        dvt = Optional[TArc](TArc(_pack_d_vec(dv_t, ctx)))
        dvc = Optional[TArc](TArc(_pack_d_vec(dv_c, ctx)))

    return Krea2BlockGradsT(
        TArc(d_x^),
        g_wq^, g_wk^, g_wv^, g_gate^, g_wo^, g_mg^, g_mu^, g_down^,
        dvt^, dvc^,
    )


# ── FULL-FINETUNE block backward (krea2 full-FT campaign, 2026-07-07;
# v2 FULL SURFACE 2026-07-08, FULL_SURFACE_PLAN Phase B row krea2) ────────────
# Same hand-chain as the LoRA device-grad backward above, but the 8 base matmul
# weights are TRAINABLE: each projection site emits d_W = d_yᵀ @ x_saved
# (ops/linalg_backward.linear_backward_dw, F32 accumulate — the optimizer input)
# alongside the plain bf16 d_x carry. NO LoRA, NO int8 (reference trainer full-FT forbids
# quantized linears — Krea2FineTuneSetup trains bf16 weights).
#
# v2 trainable surface = ALL 13 per-block params, in KREA2_FT_SLOT_* order
# (0-7 the v1 matmuls: wq wk wv gate wo mlp_gate mlp_up mlp_down):
#   [8]  qnorm.scale    [Dh]  full rms d_g (rms_norm_backward_dg — the weight
#   [9]  knorm.scale    [Dh]  never enters d_g, so the krea2 scale+1 convention
#   [10] prenorm.scale  [F]   differentiates to the SAME formula; F32 out)
#   [11] postnorm.scale [F]
#   [12] mod.lin        [6F]  DoubleSharedModulation.lin (mmdit.py:122-133) is
#        a 1D Parameter ADDED to vec (out = vec + lin) — NOT a Linear (no 2D
#        weight, no bias). d_mod_lin = concat of the 6 chunk grads in chunk
#        order [prescale, preshift, pregate, postscale, postshift, postgate]:
#        scale/shift from modulate_backward param grads, gates from
#        gate_residual_backward d_g. All cast F32 for the optimizer (the
#        linear_backward_dw dtype trap).
# The d_x carry math is UNCHANGED from v1 (the param-grad kernels are separate
# arms of the same ops) — step-1 forward/loss stays in the v1 byte class.
# Oracle: the SAME krea2_block_oracle.py (KREA2_FT_ORACLE=1) — dumps
# kref_ft_{qnorm,knorm,prenorm,postnorm,mod_lin}_dW for the new arms.

# v2 full-surface slot indices 8-12 (0-7 are the matmuls in the dw-list order
# above). FIXED order — the host store tails, af_states flat index, sidecar
# and the parity gate all key off it.
comptime KREA2_FT_SLOT_QNORM = 8
comptime KREA2_FT_SLOT_KNORM = 9
comptime KREA2_FT_SLOT_PRENORM = 10
comptime KREA2_FT_SLOT_POSTNORM = 11
comptime KREA2_FT_SLOT_MOD_LIN = 12
comptime KREA2_FT_SLOTS_V2 = 13


struct Krea2BlockFTGrads(Movable):
    var d_x: TArc          # [1,L,features] input grad (the inter-block carry)
    # len 13, F32 device, FIXED KREA2_FT_SLOT_* order:
    #   [0-7] matmul dW [out,in]: wq wk wv gate wo mlp_gate mlp_up mlp_down
    #   [8] d_qnorm [Dh]  [9] d_knorm [Dh]  [10] d_prenorm [F]
    #   [11] d_postnorm [F]  [12] d_mod_lin [6F]
    var dw: List[TArc]

    def __init__(out self, var d_x: TArc, var dw: List[TArc]):
        self.d_x = d_x^
        self.dw = dw^


def krea2_single_stream_block_ft_backward_dev[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out: Tensor,        # [1, L, features] upstream grad of the block output
    vec: Tensor,          # [1, 6*features]  (for the raw mod chunks)
    w: Krea2BlockWeights, saved: Krea2BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),  # MUST match the forward call
) raises -> Krea2BlockFTGrads:
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var mlpdim = saved.mlp_gate[].shape()[2]
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var pregate = mods[2]
    var postscale = mods[3]
    var postgate = mods[5]

    # ── MLP branch backward ──────────────────────────────────────────────────
    # postgate TRAINABLE (v2): d_postgate = sum_rows d_out*m (F32-accumulated
    # d_g kernel, grad dtype out) — cast F32 for the optimizer.
    var grg2 = gate_residual_backward(d_out, saved.x1[], postgate[], saved.m[], ctx, compute_gate_grad=True)
    var d_m = grg2.d_y.clone(ctx)

    var dw_down = linear_backward_dw(d_m, saved.sw[], M, mlpdim, features, ctx, STDtype.F32)
    var d_sw = linear_backward_dx(d_m, w.mlp_down_w[], M, mlpdim, features, ctx)

    var sgb = swiglu_backward(d_sw, saved.mlp_gate[], saved.mlp_up[], ctx)
    var dw_mg = linear_backward_dw(sgb.d_gate, saved.xm2[], M, features, mlpdim, ctx, STDtype.F32)
    var dw_mu = linear_backward_dw(sgb.d_up, saved.xm2[], M, features, mlpdim, ctx, STDtype.F32)
    var d_xm2_g = linear_backward_dx(sgb.d_gate, w.mlp_gate_w[], M, features, mlpdim, ctx)
    var d_xm2_u = linear_backward_dx(sgb.d_up, w.mlp_up_w[], M, features, mlpdim, ctx)
    var d_xm2 = add(d_xm2_g, d_xm2_u, ctx)

    # postscale/postshift TRAINABLE (v2) -> modulate param grads ([F], xn2 dtype).
    var mb2 = modulate_backward(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx), saved.xn2[], cast_tensor(postscale[], saved.xn2[].dtype(), ctx), ctx, compute_param_grads=True)
    # postnorm scale TRAINABLE (v2): d_g via the weight-free dg arm (F32 out);
    # d_x stays the UNCHANGED krea2 scale+1 dx kernel (v1 byte class).
    var d_postnorm = rms_norm_backward_dg(mb2.d_x, saved.x1[], eps, ctx)
    var rb2_dx = krea2_rmsnorm_backward_dx(mb2.d_x, saved.x1[], w.postnorm_scale[], eps, ctx)
    var d_x1 = add(grg2.d_x, rb2_dx, ctx)

    # ── ATTENTION branch backward ────────────────────────────────────────────
    # pregate TRAINABLE (v2): d_pregate = sum_rows d_x1*a.
    var grg1 = gate_residual_backward(d_x1, saved.x[], pregate[], saved.a[], ctx, compute_gate_grad=True)
    var d_a = grg1.d_y.clone(ctx)

    var dw_wo = linear_backward_dw(d_a, saved.gated[], M, features, features, ctx, STDtype.F32)
    var d_gated = linear_backward_dx(d_a, w.wo[], M, features, features, ctx)

    var d_attn_flat = mul(d_gated, saved.sg[], ctx)
    var d_sg = mul(d_gated, saved.attn_flat[], ctx)
    var d_gate_pre = sigmoid_backward_from_output(d_sg, saved.sg[], ctx)

    var d_att = reshape(d_attn_flat, [1, L, HEADS, HEADDIM], ctx)
    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    var bwd_use_flash = real_len and real_len.value() < L
    if bwd_use_flash:
        if not saved.flash_stats:
            raise Error(
                "krea2 block ft-bwd (dev): real_len < L but saved tape has no flash set"
                " (forward/backward real_len mismatch)"
            )
        var rl = real_len.value()
        var fb = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
            saved.flash_q.value(), saved.flash_k.value(), saved.flash_v.value(),
            saved.flash_o.value(), saved.flash_stats.value(), d_att, rl, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward[1, L, HEADS, HEADDIM](
            saved.q_rope[], saved.k_full[], saved.v_full[], d_att, scale, ctx
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_k_rope = repeat_kv_backward(d_k_sb, L, KVHEADS, n_rep, HEADDIM, ctx)
    var d_v = repeat_kv_backward(d_v_sb, L, KVHEADS, n_rep, HEADDIM, ctx)

    var d_q_rms = rope_backward(d_q_sb, cos_q, sin_q, True, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, True, ctx)

    # qnorm/knorm scales TRAINABLE (v2): weight-free d_g over q_pre/k_pre
    # [1,L,H,Dh] (rows = L*H, d_g [Dh] F32); d_x stays the v1 dx kernel.
    var d_qnorm = rms_norm_backward_dg(d_q_rms, saved.q_pre[], eps, ctx)
    var d_knorm = rms_norm_backward_dg(d_k_rms, saved.k_pre[], eps, ctx)
    var rbq_dx = krea2_rmsnorm_backward_dx(d_q_rms, saved.q_pre[], w.qnorm_scale[], eps, ctx)
    var rbk_dx = krea2_rmsnorm_backward_dx(d_k_rms, saved.k_pre[], w.knorm_scale[], eps, ctx)

    var d_q = reshape(rbq_dx, [1, L, HEADS * HEADDIM], ctx)
    var d_k = reshape(rbk_dx, [1, L, KVHEADS * HEADDIM], ctx)
    var d_v_flat = reshape(d_v, [1, L, KVHEADS * HEADDIM], ctx)

    var dw_wq = linear_backward_dw(d_q, saved.xm[], M, features, HEADS * HEADDIM, ctx, STDtype.F32)
    var dw_wk = linear_backward_dw(d_k, saved.xm[], M, features, KVHEADS * HEADDIM, ctx, STDtype.F32)
    var dw_wv = linear_backward_dw(d_v_flat, saved.xm[], M, features, KVHEADS * HEADDIM, ctx, STDtype.F32)
    var dw_gate = linear_backward_dw(d_gate_pre, saved.xm[], M, features, features, ctx, STDtype.F32)
    var dx_q = linear_backward_dx(d_q, w.wq[], M, features, HEADS * HEADDIM, ctx)
    var dx_k = linear_backward_dx(d_k, w.wk[], M, features, KVHEADS * HEADDIM, ctx)
    var dx_v = linear_backward_dx(d_v_flat, w.wv[], M, features, KVHEADS * HEADDIM, ctx)
    var dx_g = linear_backward_dx(d_gate_pre, w.gate_w[], M, features, features, ctx)
    var d_xm = add(add(dx_q, dx_k, ctx), add(dx_v, dx_g, ctx), ctx)

    # prescale/preshift TRAINABLE (v2) -> modulate param grads ([F], xn dtype).
    var mb1 = modulate_backward(cast_tensor(d_xm, saved.xn[].dtype(), ctx), saved.xn[], cast_tensor(prescale[], saved.xn[].dtype(), ctx), ctx, compute_param_grads=True)
    # prenorm scale TRAINABLE (v2).
    var d_prenorm = rms_norm_backward_dg(mb1.d_x, saved.x[], eps, ctx)
    var rb1_dx = krea2_rmsnorm_backward_dx(mb1.d_x, saved.x[], w.prenorm_scale[], eps, ctx)

    var d_x = add(grg1.d_x, rb1_dx, ctx)

    # mod.lin TRAINABLE (v2): a 1D [6F] Parameter added to vec (out = vec +
    # lin) — d_mod_lin = the 6 chunk grads concatenated in _mod6 chunk order
    # [prescale, preshift, pregate, postscale, postshift, postgate]. The
    # scale/shift grads carry the acts dtype, the gate grads the grad dtype
    # (both F32-accumulated) — cast each F32 for the optimizer before concat.
    var d_mod_lin = concat(
        0, ctx,
        cast_tensor(mb1.d_scale, STDtype.F32, ctx),
        cast_tensor(mb1.d_shift, STDtype.F32, ctx),
        cast_tensor(grg1.d_g, STDtype.F32, ctx),
        cast_tensor(mb2.d_scale, STDtype.F32, ctx),
        cast_tensor(mb2.d_shift, STDtype.F32, ctx),
        cast_tensor(grg2.d_g, STDtype.F32, ctx),
    )

    var dw = List[TArc]()
    dw.append(TArc(dw_wq^))
    dw.append(TArc(dw_wk^))
    dw.append(TArc(dw_wv^))
    dw.append(TArc(dw_gate^))
    dw.append(TArc(dw_wo^))
    dw.append(TArc(dw_mg^))
    dw.append(TArc(dw_mu^))
    dw.append(TArc(dw_down^))
    dw.append(TArc(d_qnorm^))      # KREA2_FT_SLOT_QNORM
    dw.append(TArc(d_knorm^))      # KREA2_FT_SLOT_KNORM
    dw.append(TArc(d_prenorm^))    # KREA2_FT_SLOT_PRENORM
    dw.append(TArc(d_postnorm^))   # KREA2_FT_SLOT_POSTNORM
    dw.append(TArc(d_mod_lin^))    # KREA2_FT_SLOT_MOD_LIN
    return Krea2BlockFTGrads(TArc(d_x^), dw^)


# Direct DoRA device-grad backward. This is the same block chain as the LoRA
# device-grad backward above, but each projection backward is full W_eff
# substitution: when a direct adapter is present, the helper returns the full
# d_x and DoRA d_A/d_B/d_m. Base W is frozen.
def krea2_single_stream_block_dora_backward_dev[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out: Tensor,
    vec: Tensor,
    w: Krea2BlockWeights, dora: Krea2BlockDirectDoRA, saved: Krea2BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2BlockDirectDoRAGradsT:
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var mlpdim = saved.mlp_gate[].shape()[2]
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var pregate = mods[2]
    var postscale = mods[3]
    var postgate = mods[5]

    var grg2 = gate_residual_backward(d_out, saved.x1[], postgate[], saved.m[], ctx, compute_gate_grad=False)
    var d_m = grg2.d_y.clone(ctx)

    var bw_down = krea2_block_direct_dora_projection_backward_dev(
        d_m, saved.sw[], w.mlp_down_w[], dora.mlp_down_w,
        M, mlpdim, features, ctx,
    )
    var d_sw = bw_down.d_x.clone(ctx)
    var g_down = bw_down.dora.copy()

    var sgb = swiglu_backward(d_sw, saved.mlp_gate[], saved.mlp_up[], ctx)
    var bw_mg = krea2_block_direct_dora_projection_backward_dev(
        sgb.d_gate, saved.xm2[], w.mlp_gate_w[], dora.mlp_gate_w,
        M, features, mlpdim, ctx,
    )
    var bw_mu = krea2_block_direct_dora_projection_backward_dev(
        sgb.d_up, saved.xm2[], w.mlp_up_w[], dora.mlp_up_w,
        M, features, mlpdim, ctx,
    )
    var g_mg = bw_mg.dora.copy()
    var g_mu = bw_mu.dora.copy()
    var d_xm2 = add(bw_mg.d_x, bw_mu.d_x, ctx)

    var mb2 = modulate_backward(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx), saved.xn2[], cast_tensor(postscale[], saved.xn2[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb2_dx = krea2_rmsnorm_backward_dx(mb2.d_x, saved.x1[], w.postnorm_scale[], eps, ctx)
    var d_x1 = add(grg2.d_x, rb2_dx, ctx)

    var grg1 = gate_residual_backward(d_x1, saved.x[], pregate[], saved.a[], ctx, compute_gate_grad=False)
    var d_a = grg1.d_y.clone(ctx)

    var bw_wo = krea2_block_direct_dora_projection_backward_dev(
        d_a, saved.gated[], w.wo[], dora.wo, M, features, features, ctx
    )
    var d_gated = bw_wo.d_x.clone(ctx)
    var g_wo = bw_wo.dora.copy()

    var d_attn_flat = mul(d_gated, saved.sg[], ctx)
    var d_sg = mul(d_gated, saved.attn_flat[], ctx)
    var d_gate_pre = sigmoid_backward_from_output(d_sg, saved.sg[], ctx)

    var d_att = reshape(d_attn_flat, [1, L, HEADS, HEADDIM], ctx)
    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    var bwd_use_flash = real_len and real_len.value() < L
    if bwd_use_flash:
        if not saved.flash_stats:
            raise Error(
                "krea2 direct DoRA bwd: real_len < L but saved tape has no flash set"
                " (forward/backward real_len mismatch)"
            )
        var rl = real_len.value()
        var fb = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
            saved.flash_q.value(), saved.flash_k.value(), saved.flash_v.value(),
            saved.flash_o.value(), saved.flash_stats.value(), d_att, rl, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward[1, L, HEADS, HEADDIM](
            saved.q_rope[], saved.k_full[], saved.v_full[], d_att, scale, ctx
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_k_rope = repeat_kv_backward(d_k_sb, L, KVHEADS, n_rep, HEADDIM, ctx)
    var d_v = repeat_kv_backward(d_v_sb, L, KVHEADS, n_rep, HEADDIM, ctx)

    var d_q_rms = rope_backward(d_q_sb, cos_q, sin_q, True, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, True, ctx)

    var rbq_dx = krea2_rmsnorm_backward_dx(d_q_rms, saved.q_pre[], w.qnorm_scale[], eps, ctx)
    var rbk_dx = krea2_rmsnorm_backward_dx(d_k_rms, saved.k_pre[], w.knorm_scale[], eps, ctx)

    var d_q = reshape(rbq_dx, [1, L, HEADS * HEADDIM], ctx)
    var d_k = reshape(rbk_dx, [1, L, KVHEADS * HEADDIM], ctx)
    var d_v_flat = reshape(d_v, [1, L, KVHEADS * HEADDIM], ctx)

    var bw_q = krea2_block_direct_dora_projection_backward_dev(
        d_q, saved.xm[], w.wq[], dora.wq, M, features, HEADS * HEADDIM, ctx,
    )
    var bw_k = krea2_block_direct_dora_projection_backward_dev(
        d_k, saved.xm[], w.wk[], dora.wk, M, features, KVHEADS * HEADDIM, ctx,
    )
    var bw_v = krea2_block_direct_dora_projection_backward_dev(
        d_v_flat, saved.xm[], w.wv[], dora.wv, M, features, KVHEADS * HEADDIM, ctx,
    )
    var bw_g = krea2_block_direct_dora_projection_backward_dev(
        d_gate_pre, saved.xm[], w.gate_w[], dora.gate_w, M, features, features, ctx,
    )
    var g_wq = bw_q.dora.copy()
    var g_wk = bw_k.dora.copy()
    var g_wv = bw_v.dora.copy()
    var g_gate = bw_g.dora.copy()

    var d_xm = add(add(bw_q.d_x, bw_k.d_x, ctx), add(bw_v.d_x, bw_g.d_x, ctx), ctx)

    var mb1 = modulate_backward(cast_tensor(d_xm, saved.xn[].dtype(), ctx), saved.xn[], cast_tensor(prescale[], saved.xn[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb1_dx = krea2_rmsnorm_backward_dx(mb1.d_x, saved.x[], w.prenorm_scale[], eps, ctx)

    var d_x = add(grg1.d_x, rb1_dx, ctx)

    return Krea2BlockDirectDoRAGradsT(
        TArc(d_x^),
        g_wq^, g_wk^, g_wv^, g_gate^, g_wo^, g_mg^, g_mu^, g_down^,
    )


# Direct OFT device-grad backward. Same chain as the LoRA device-grad backward,
# but each projection consumes the current frozen W_orig plus resident OFT vec
# and returns direct d_vec/d_x without a dense full-delta carrier.
def krea2_single_stream_block_oft_backward_dev[
    L: Int, HEADS: Int, KVHEADS: Int, HEADDIM: Int
](
    d_out: Tensor,
    vec: Tensor,
    w: Krea2BlockWeights, oft: Krea2BlockDirectOFT, saved: Krea2BlockSaved,
    cos_q: Tensor, sin_q: Tensor,
    cos_k: Tensor, sin_k: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    real_len: Optional[Int] = Optional[Int](None),
) raises -> Krea2BlockDirectOFTGradsT:
    comptime features = HEADS * HEADDIM
    comptime n_rep = HEADS // KVHEADS
    var mlpdim = saved.mlp_gate[].shape()[2]
    var M = L
    var scale = Float32(1.0) / sqrt(Float32(HEADDIM))

    var mods = _mod6(vec, w.mod_lin[], features, ctx)
    var prescale = mods[0]
    var pregate = mods[2]
    var postscale = mods[3]
    var postgate = mods[5]

    var grg2 = gate_residual_backward(d_out, saved.x1[], postgate[], saved.m[], ctx, compute_gate_grad=False)
    var d_m = grg2.d_y.clone(ctx)

    var bw_down = krea2_block_direct_oft_projection_backward_dev(
        d_m, saved.sw[], w.mlp_down_w[], oft.mlp_down_w,
        M, mlpdim, features, ctx,
    )
    var d_sw = bw_down.d_x.clone(ctx)
    var g_down = bw_down.oft.copy()

    var sgb = swiglu_backward(d_sw, saved.mlp_gate[], saved.mlp_up[], ctx)
    var bw_mg = krea2_block_direct_oft_projection_backward_dev(
        sgb.d_gate, saved.xm2[], w.mlp_gate_w[], oft.mlp_gate_w,
        M, features, mlpdim, ctx,
    )
    var bw_mu = krea2_block_direct_oft_projection_backward_dev(
        sgb.d_up, saved.xm2[], w.mlp_up_w[], oft.mlp_up_w,
        M, features, mlpdim, ctx,
    )
    var g_mg = bw_mg.oft.copy()
    var g_mu = bw_mu.oft.copy()
    var d_xm2 = add(bw_mg.d_x, bw_mu.d_x, ctx)

    var mb2 = modulate_backward(cast_tensor(d_xm2, saved.xn2[].dtype(), ctx), saved.xn2[], cast_tensor(postscale[], saved.xn2[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb2_dx = krea2_rmsnorm_backward_dx(mb2.d_x, saved.x1[], w.postnorm_scale[], eps, ctx)
    var d_x1 = add(grg2.d_x, rb2_dx, ctx)

    var grg1 = gate_residual_backward(d_x1, saved.x[], pregate[], saved.a[], ctx, compute_gate_grad=False)
    var d_a = grg1.d_y.clone(ctx)

    var bw_wo = krea2_block_direct_oft_projection_backward_dev(
        d_a, saved.gated[], w.wo[], oft.wo, M, features, features, ctx
    )
    var d_gated = bw_wo.d_x.clone(ctx)
    var g_wo = bw_wo.oft.copy()

    var d_attn_flat = mul(d_gated, saved.sg[], ctx)
    var d_sg = mul(d_gated, saved.attn_flat[], ctx)
    var d_gate_pre = sigmoid_backward_from_output(d_sg, saved.sg[], ctx)

    var d_att = reshape(d_attn_flat, [1, L, HEADS, HEADDIM], ctx)
    var d_q_sb: Tensor
    var d_k_sb: Tensor
    var d_v_sb: Tensor
    var bwd_use_flash = real_len and real_len.value() < L
    if bwd_use_flash:
        if not saved.flash_stats:
            raise Error(
                "krea2 direct OFT bwd: real_len < L but saved tape has no flash set"
                " (forward/backward real_len mismatch)"
            )
        var rl = real_len.value()
        var fb = sdpa_flash_backward_padmask_bf16[1, L, HEADS, HEADDIM](
            saved.flash_q.value(), saved.flash_k.value(), saved.flash_v.value(),
            saved.flash_o.value(), saved.flash_stats.value(), d_att, rl, scale, ctx,
        )
        d_q_sb = Tensor(fb.d_q.buf.copy(), fb.d_q.shape(), fb.d_q.dtype())
        d_k_sb = Tensor(fb.d_k.buf.copy(), fb.d_k.shape(), fb.d_k.dtype())
        d_v_sb = Tensor(fb.d_v.buf.copy(), fb.d_v.shape(), fb.d_v.dtype())
    else:
        var sb = sdpa_backward[1, L, HEADS, HEADDIM](
            saved.q_rope[], saved.k_full[], saved.v_full[], d_att, scale, ctx
        )
        d_q_sb = Tensor(sb.d_q.buf.copy(), sb.d_q.shape(), sb.d_q.dtype())
        d_k_sb = Tensor(sb.d_k.buf.copy(), sb.d_k.shape(), sb.d_k.dtype())
        d_v_sb = Tensor(sb.d_v.buf.copy(), sb.d_v.shape(), sb.d_v.dtype())

    var d_k_rope = repeat_kv_backward(d_k_sb, L, KVHEADS, n_rep, HEADDIM, ctx)
    var d_v = repeat_kv_backward(d_v_sb, L, KVHEADS, n_rep, HEADDIM, ctx)

    var d_q_rms = rope_backward(d_q_sb, cos_q, sin_q, True, ctx)
    var d_k_rms = rope_backward(d_k_rope, cos_k, sin_k, True, ctx)

    var rbq_dx = krea2_rmsnorm_backward_dx(d_q_rms, saved.q_pre[], w.qnorm_scale[], eps, ctx)
    var rbk_dx = krea2_rmsnorm_backward_dx(d_k_rms, saved.k_pre[], w.knorm_scale[], eps, ctx)

    var d_q = reshape(rbq_dx, [1, L, HEADS * HEADDIM], ctx)
    var d_k = reshape(rbk_dx, [1, L, KVHEADS * HEADDIM], ctx)
    var d_v_flat = reshape(d_v, [1, L, KVHEADS * HEADDIM], ctx)

    var bw_q = krea2_block_direct_oft_projection_backward_dev(
        d_q, saved.xm[], w.wq[], oft.wq, M, features, HEADS * HEADDIM, ctx,
    )
    var bw_k = krea2_block_direct_oft_projection_backward_dev(
        d_k, saved.xm[], w.wk[], oft.wk, M, features, KVHEADS * HEADDIM, ctx,
    )
    var bw_v = krea2_block_direct_oft_projection_backward_dev(
        d_v_flat, saved.xm[], w.wv[], oft.wv, M, features, KVHEADS * HEADDIM, ctx,
    )
    var bw_g = krea2_block_direct_oft_projection_backward_dev(
        d_gate_pre, saved.xm[], w.gate_w[], oft.gate_w, M, features, features, ctx,
    )
    var g_wq = bw_q.oft.copy()
    var g_wk = bw_k.oft.copy()
    var g_wv = bw_v.oft.copy()
    var g_gate = bw_g.oft.copy()

    var d_xm = add(add(bw_q.d_x, bw_k.d_x, ctx), add(bw_v.d_x, bw_g.d_x, ctx), ctx)

    var mb1 = modulate_backward(cast_tensor(d_xm, saved.xn[].dtype(), ctx), saved.xn[], cast_tensor(prescale[], saved.xn[].dtype(), ctx), ctx, compute_param_grads=False)
    var rb1_dx = krea2_rmsnorm_backward_dx(mb1.d_x, saved.x[], w.prenorm_scale[], eps, ctx)

    var d_x = add(grg1.d_x, rb1_dx, ctx)

    return Krea2BlockDirectOFTGradsT(
        TArc(d_x^),
        g_wq^, g_wk^, g_wv^, g_gate^, g_wo^, g_mg^, g_mu^, g_down^,
    )
