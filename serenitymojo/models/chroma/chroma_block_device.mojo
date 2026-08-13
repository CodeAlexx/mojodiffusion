# serenitymojo/models/chroma/chroma_block_device.mojo
#
# DEVICE-RESIDENT Chroma (= Flux) DOUBLE + SINGLE LoRA block, fwd + bwd.
#
# This is a device-resident transcription of the HOST flux LoRA block that
# chroma re-exports (models/flux/lora_block.mojo — the torch-gated math oracle).
# Activations stay on the GPU as `TArc` (ArcPointer[Tensor]); there are NO host
# `List[Float32]` carriers in the hot path and NO per-op to_host/from_host — the
# saved activations are device Tensors. It is NUMERICALLY IDENTICAL to the host
# block: every op is the same shared op with the same dtypes, and the LoRA /
# projection helpers reproduce the host's EXACT bf16 rounding (base linear in
# bf16, LoRA apply accumulated in F32 then rounded once — matching
# flux_lora_apply's `_add_lists` + `_t`). Result: cos ~ 1.0, rel-L2 ~ 0 vs host.
#
# CHROMA FREEZE CONTRACT: chroma trains ONLY the LoRA adapters (frozen base
# weights + frozen distilled-guidance approximator). So this backward returns
# ONLY d_x (device TArc) + per-slot LoRA d_a/d_b (device TArc). Base weight grads
# (d_wqkv/d_wproj/d_wmlp*/d_w1/d_w2/d_q_norm/d_k_norm) and mod-vec grads
# (d_shift/d_scale/d_gate) are DISCARDED — we skip computing them entirely and
# use linear_backward_dx (d_x-only) for the frozen base linears. We STILL compute
# d_x correctly: the base weight contribution to d_x (d_y @ W) is required and
# included; only the d_W / d_b OUTPUTS are dropped.
#
# MLP is PLAIN GELU (flux): linear(wmlp0 [Fmlp,D]) -> gelu -> linear(wmlp2 [D,Fmlp]).
# NOT klein's gated-GLU MLP. Joint attention order is txt FIRST then img.
#
# Base weights / saved-activation dtypes: bf16 (flux StreamWeights/SingleBlockWeights
# store bf16 TArc). We REUSE those weight structs (no new device weight struct).
#
# Mojo current beta: `def` (not fn), `comptime` (not alias), `var`, explicit `raises`.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward_dx, linear_backward_dw

from serenitymojo.training.train_step import LoraAdapter

# shared forward + backward ops (Tenet 1: nothing new here)
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd_padmask_bf16, sdpa_flash_backward_padmask_bf16,
)
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, reshape_in_place, slice, concat, add, mul_scalar,
)
from serenitymojo.ops.norm_backward import (
    rms_norm_backward_dx, layer_norm_backward_dx,
)
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import gate_residual_backward, rope_backward
from serenitymojo.ops.shape_backward import cat_backward

# base weights + host modulation carriers + host LoRA carriers (to convert to device)
from serenitymojo.models.flux.block import (
    ModVecs, SingleModVecs,
    StreamWeights, DoubleBlockWeights, SingleBlockWeights,
)
from serenitymojo.models.flux.lora_block import (
    StreamLora, DoubleBlockLora, SingleBlockLora,
    DBL_STREAM_SLOTS, D_SQ, D_SK, D_SV, D_PROJ, D_MLP0, D_MLP2,
    SGL_SLOTS, S_SQ, S_SK, S_SV, S_PMLP, S_L2,
)


comptime TArc = ArcPointer[Tensor]


# ── local host->device upload / cast helpers (device versions live HERE) ──────
def _ones(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(1.0)
    return o^


def _zeros(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(0.0)
    return o^


def _t_bf16(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


def _t_f32(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


def _f32(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    if t.dtype() == STDtype.F32:
        return t.clone(ctx)
    return cast_tensor(t, STDtype.F32, ctx)


def _bf16(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    if t.dtype() == STDtype.BF16:
        return t.clone(ctx)
    return cast_tensor(t, STDtype.BF16, ctx)


# ═══════════════════════════════════════════════════════════════════════════
# Device LoRA adapter + carriers (bf16 a/b, mirrors flux StreamLora slots).
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct ChromaLoraAdapterDevice(Copyable, Movable):
    var a: TArc        # [rank, in_f]  bf16
    var b: TArc        # [out_f, rank] bf16
    var rank: Int
    var in_f: Int
    var out_f: Int
    var scale: Float32


def chroma_lora_adapter_to_device(
    lo: LoraAdapter, ctx: DeviceContext
) raises -> ChromaLoraAdapterDevice:
    # LoraAdapter stores a/b as List[BFloat16] (model storage dtype) — upload
    # verbatim to bf16 device buffers (no F32 detour), like zimage.
    return ChromaLoraAdapterDevice(
        TArc(Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx)),
        TArc(Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx)),
        lo.rank, lo.in_f, lo.out_f, lo.scale,
    )


def _opt_lora_to_device(
    lo: Optional[LoraAdapter], ctx: DeviceContext
) raises -> Optional[ChromaLoraAdapterDevice]:
    if lo:
        return Optional[ChromaLoraAdapterDevice](chroma_lora_adapter_to_device(lo.value(), ctx))
    return Optional[ChromaLoraAdapterDevice](None)


@fieldwise_init
struct StreamLoraDevice(Copyable, Movable):
    var to_q: Optional[ChromaLoraAdapterDevice]
    var to_k: Optional[ChromaLoraAdapterDevice]
    var to_v: Optional[ChromaLoraAdapterDevice]
    var proj: Optional[ChromaLoraAdapterDevice]
    var mlp0: Optional[ChromaLoraAdapterDevice]
    var mlp2: Optional[ChromaLoraAdapterDevice]


def stream_lora_to_device(lo: StreamLora, ctx: DeviceContext) raises -> StreamLoraDevice:
    return StreamLoraDevice(
        _opt_lora_to_device(lo.to_q, ctx), _opt_lora_to_device(lo.to_k, ctx),
        _opt_lora_to_device(lo.to_v, ctx), _opt_lora_to_device(lo.proj, ctx),
        _opt_lora_to_device(lo.mlp0, ctx), _opt_lora_to_device(lo.mlp2, ctx),
    )


@fieldwise_init
struct DoubleBlockLoraDevice(Copyable, Movable):
    var img: StreamLoraDevice
    var txt: StreamLoraDevice


def double_block_lora_to_device(
    lora: DoubleBlockLora, ctx: DeviceContext
) raises -> DoubleBlockLoraDevice:
    return DoubleBlockLoraDevice(
        stream_lora_to_device(lora.img, ctx), stream_lora_to_device(lora.txt, ctx),
    )


@fieldwise_init
struct SingleBlockLoraDevice(Copyable, Movable):
    var to_q: Optional[ChromaLoraAdapterDevice]
    var to_k: Optional[ChromaLoraAdapterDevice]
    var to_v: Optional[ChromaLoraAdapterDevice]
    var proj_mlp: Optional[ChromaLoraAdapterDevice]
    var linear2: Optional[ChromaLoraAdapterDevice]


def single_block_lora_to_device(
    lora: SingleBlockLora, ctx: DeviceContext
) raises -> SingleBlockLoraDevice:
    return SingleBlockLoraDevice(
        _opt_lora_to_device(lora.to_q, ctx), _opt_lora_to_device(lora.to_k, ctx),
        _opt_lora_to_device(lora.to_v, ctx), _opt_lora_to_device(lora.proj_mlp, ctx),
        _opt_lora_to_device(lora.linear2, ctx),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Device modulation vectors. Forward modulate/residual_gate use bf16 (host `_t`);
# backward gate_residual_backward uses FULL-F32 gate (host `_tf32(mv.gate)`), so
# we keep a bf16 copy (fwd) AND an F32 copy of the gate (bwd) from the same list.
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct ModVecsDevice(Copyable, Movable):
    var shift1: TArc      # bf16
    var scale1: TArc      # bf16
    var gate1: TArc       # bf16 (forward residual_gate)
    var shift2: TArc      # bf16
    var scale2: TArc      # bf16
    var gate2: TArc       # bf16 (forward residual_gate)
    var gate1_f32: TArc   # F32  (backward gate_residual_backward)
    var gate2_f32: TArc   # F32


def modvecs_to_device(mv: ModVecs, D: Int, ctx: DeviceContext) raises -> ModVecsDevice:
    return ModVecsDevice(
        TArc(_t_bf16(mv.shift1.copy(), [D], ctx)),
        TArc(_t_bf16(mv.scale1.copy(), [D], ctx)),
        TArc(_t_bf16(mv.gate1.copy(), [D], ctx)),
        TArc(_t_bf16(mv.shift2.copy(), [D], ctx)),
        TArc(_t_bf16(mv.scale2.copy(), [D], ctx)),
        TArc(_t_bf16(mv.gate2.copy(), [D], ctx)),
        TArc(_t_f32(mv.gate1.copy(), [D], ctx)),
        TArc(_t_f32(mv.gate2.copy(), [D], ctx)),
    )


@fieldwise_init
struct SingleModVecsDevice(Copyable, Movable):
    var shift: TArc      # bf16
    var scale: TArc      # bf16
    var gate: TArc       # bf16 (forward residual_gate)
    var gate_f32: TArc   # F32  (backward gate_residual_backward)


def single_modvecs_to_device(mv: SingleModVecs, D: Int, ctx: DeviceContext) raises -> SingleModVecsDevice:
    return SingleModVecsDevice(
        TArc(_t_bf16(mv.shift.copy(), [D], ctx)),
        TArc(_t_bf16(mv.scale.copy(), [D], ctx)),
        TArc(_t_bf16(mv.gate.copy(), [D], ctx)),
        TArc(_t_f32(mv.gate.copy(), [D], ctx)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Device saved activations (TArc; mirror flux StreamSaved / SingleBlockSaved).
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct StreamSavedDevice(Copyable, Movable):
    var x: TArc         # [N,D]
    var ln1: TArc       # [N,D]
    var norm: TArc      # [N,D]
    var q_pre: TArc     # [1,N,H,Dh]
    var k_pre: TArc     # [1,N,H,Dh]
    var att: TArc       # [N,D]  per-stream attention slice
    var attn_res: TArc  # [N,D]
    var ln2: TArc       # [N,D]
    var mlp_in: TArc    # [N,D]
    var mlp_pre: TArc   # [N,Fmlp]  gelu input
    var mlp_h: TArc     # [N,Fmlp]  gelu output


@fieldwise_init
struct DoubleBlockSavedDevice(Copyable, Movable):
    var img: StreamSavedDevice
    var txt: StreamSavedDevice
    var q_rope: TArc    # [1,S,H,Dh]
    var k_rope: TArc    # [1,S,H,Dh]
    var v_joint: TArc   # [1,S,H,Dh]


@fieldwise_init
struct SingleBlockSavedDevice(Copyable, Movable):
    var x: TArc         # [S,D]
    var ln: TArc        # [S,D]
    var norm: TArc      # [S,D]
    var q_pre: TArc     # [1,S,H,Dh]
    var k_pre: TArc     # [1,S,H,Dh]
    var q_rope: TArc    # [1,S,H,Dh]
    var k_rope: TArc    # [1,S,H,Dh]
    var v: TArc         # [1,S,H,Dh]
    var att_flat: TArc  # [S,D]
    var mlp_in: TArc    # [S,Fmlp]  gelu input
    var mlp_h: TArc     # [S,Fmlp]  gelu output
    var out_in: TArc    # [S, D+Fmlp]  concat(att_flat, mlp_h)


@fieldwise_init
struct DoubleBlockDeviceForward(Copyable, Movable):
    var img_out: TArc   # [N_IMG, D]  device-resident
    var txt_out: TArc   # [N_TXT, D]
    var saved: DoubleBlockSavedDevice


@fieldwise_init
struct SingleBlockDeviceForward(Copyable, Movable):
    var out: TArc       # [S, D]
    var saved: SingleBlockSavedDevice


# ── backward LoRA grads (per-slot d_a/d_b as device TArc; d_x device F32) ─────
@fieldwise_init
struct StreamLoraDeviceGrads(Copyable, Movable):
    var d_x: TArc                    # [N,D]  F32
    var d_a: List[Optional[TArc]]    # DBL_STREAM_SLOTS entries (None when adapter absent)
    var d_b: List[Optional[TArc]]


@fieldwise_init
struct DoubleBlockLoraDeviceGrads(Copyable, Movable):
    var img: StreamLoraDeviceGrads
    var txt: StreamLoraDeviceGrads


@fieldwise_init
struct SingleBlockLoraDeviceGrads(Copyable, Movable):
    var d_x: TArc                    # [S,D]  F32
    var d_a: List[Optional[TArc]]    # SGL_SLOTS entries
    var d_b: List[Optional[TArc]]


# ═══════════════════════════════════════════════════════════════════════════
# Device LoRA fwd/bwd primitives — reproduce host flux_lora_apply/flux_lora_bwd
# EXACTLY (base in bf16, LoRA accumulated in F32 then rounded to bf16 ONCE).
# ═══════════════════════════════════════════════════════════════════════════
# base_y + scale*((x@Aᵀ)@Bᵀ), or base_y unchanged when adapter absent.
def _lora_apply_device(
    var base_y: Tensor, x: Tensor, lo: Optional[ChromaLoraAdapterDevice],
    M: Int, ctx: DeviceContext,
) raises -> Tensor:
    if not lo:
        return base_y^
    ref l = lo.value()
    var nb1 = Optional[Tensor](None)
    var t = linear(x, l.a[], nb1^, ctx)                 # [M,rank] bf16
    var nb2 = Optional[Tensor](None)
    var dy = linear(t, l.b[], nb2^, ctx)                # [M,out] bf16
    # host: out = round_bf16( f32(base) + scale*f32(dy) )  (single rounding)
    var contrib = mul_scalar(_f32(dy, ctx), l.scale, ctx)   # F32
    var out_f32 = add(_f32(base_y, ctx), contrib, ctx)      # F32
    return _bf16(out_f32, ctx)


@fieldwise_init
struct _ChromaLoraBwd(Copyable, Movable):
    var d_a: TArc      # [rank,in_f] bf16
    var d_b: TArc      # [out_f,rank] bf16
    var d_x_lo: TArc   # [M,in_f]  bf16 (LoRA branch contribution to proj input grad)


# d_contrib is the projection-output grad (bf16, = host `_to_bf16(d_y)`).
def _lora_bwd_device(
    d_contrib: Tensor, x: Tensor, lo: ChromaLoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> _ChromaLoraBwd:
    var nb = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb^, ctx)                         # [M,rank] bf16
    # host: d_dy = round_bf16( scale * f32(d_contrib) )
    var d_dy = _bf16(mul_scalar(_f32(d_contrib, ctx), lo.scale, ctx), ctx)   # bf16
    var d_t = linear_backward_dx(d_dy, lo.b[], M, lo.rank, lo.out_f, ctx)    # [M,rank] bf16
    var d_b = linear_backward_dw(d_dy, t, M, lo.rank, lo.out_f, ctx)         # [out_f,rank] bf16
    var d_x_lo = linear_backward_dx(d_t, lo.a[], M, lo.in_f, lo.rank, ctx)   # [M,in_f] bf16
    var d_a = linear_backward_dw(d_t, x, M, lo.in_f, lo.rank, ctx)           # [rank,in_f] bf16
    return _ChromaLoraBwd(TArc(d_a^), TArc(d_b^), TArc(d_x_lo^))


# projection backward with optional LoRA: frozen base d_x (linear_backward_dx)
# + LoRA d_x contribution; d_a/d_b collected into the slot lists. Returns d_x bf16.
def _proj_bwd_with_lora_device(
    d_y: Tensor, x_in: Tensor, w: Tensor,
    lo: Optional[ChromaLoraAdapterDevice], slot: Int,
    M: Int, in_f: Int, out_f: Int,
    mut d_a_slots: List[Optional[TArc]], mut d_b_slots: List[Optional[TArc]],
    ctx: DeviceContext,
) raises -> Tensor:
    var base_dx = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)   # bf16
    if lo:
        var lg = _lora_bwd_device(d_y, x_in, lo.value(), M, ctx)
        d_a_slots[slot] = Optional[TArc](lg.d_a)
        d_b_slots[slot] = Optional[TArc](lg.d_b)
        # host: round_bf16( f32(base_dx) + f32(d_x_lo) ) — `add` does F32 math, store bf16
        return add(base_dx, lg.d_x_lo[], ctx)
    return base_dx^


# ═══════════════════════════════════════════════════════════════════════════
# FLASH vs MATH attention dispatch (2026-07-07). MEASURED: math SDPA is ~23% of
# chroma's 3.4s step / ~20-23% of flux's step. FLASH=True (DEFAULT): cuDNN
# padmask flash on the (already bf16) q/k/v — pad S to the next 128-multiple with
# zero rows, real_len=S masks the pad; backward REGENERATES o/stats by re-running
# the (deterministic-input) flash fwd, so the saved-tape contract (only
# q_rope/k_rope/v saved) is UNCHANGED. Returns bf16 (== the math sdpa_nomask /
# sdpa_backward output dtype on bf16 inputs), so the rest of the bf16 chain is
# untouched. FLASH=False = the existing math arm, BIT-IDENTICAL to the host
# oracle (the device-block parity bit bars). VALUE-CLASS numerics under FLASH
# (bf16 flash vs bf16 math — same class as the klein/zimage flash sign-off).
def _pad_seq_zeros[H: Int, Dh: Int](t_bf: Tensor, S: Int, S_PAD: Int, ctx: DeviceContext) raises -> Tensor:
    if S_PAD == S:
        return t_bf.clone(ctx)
    var zeros_h = List[Float32]()
    for _ in range((S_PAD - S) * H * Dh):
        zeros_h.append(0.0)
    var zpad = cast_tensor(
        Tensor.from_host(zeros_h^, [1, S_PAD - S, H, Dh], STDtype.F32, ctx),
        STDtype.BF16, ctx,
    )
    return concat(1, ctx, t_bf, zpad)


@fieldwise_init
struct _ChromaSdpaGrads(Movable):
    var d_q: Tensor
    var d_k: Tensor
    var d_v: Tensor


# ── SerenityTrainer T5 pad-KEY masking (2026-07-16) ───────────────────────────────
# reference trainer/diffusers pass an outer-product attention mask when text has padding
# (BaseChromaSetup: cat(text_mask, ones); processor: mask[:,None,None,:] *
# mask[:,None,:,None]) — pad rows are masked as KEYS everywhere, so for the
# image loss the valid rows compute EXACTLY the attention of a sequence with
# the pad rows REMOVED. Implementation mirrors krea2_infer.mojo: permute the
# joint rows to [txt_valid(lt) | img | txt_pad(tail)] — Chroma txt RoPE ids are
# all zeros (BaseChromaSetup text_ids = zeros [T,3]) so rope is applied BEFORE
# and rows carry their rotation through the permutation — then the EXISTING
# cuDNN padmask suffix-masks real_len = lt + n_img. Pad-row outputs and grads
# are rebuilt as EXACT zeros (loss-disconnected rows; reference trainer hides them behind its
# mask). Mask active iff 0 <= real_lt < n_txt; the default real_lt = -1 keeps
# the legacy unmasked path byte-identical. Gate:
# parity/chroma_padmask_parity.mojo (masked flash vs truncated math).
def _perm_txt_pad_to_tail(
    t: Tensor, lt: Int, n_txt: Int, S: Int, ctx: DeviceContext
) raises -> Tensor:
    var head = slice(t, 1, 0, lt, ctx)
    var img = slice(t, 1, n_txt, S - n_txt, ctx)
    var pad = slice(t, 1, lt, n_txt - lt, ctx)
    return concat(1, ctx, head, img, pad)


def _perm_txt_pad_to_tail_zeroed(
    t: Tensor, lt: Int, n_txt: Int, S: Int, ctx: DeviceContext
) raises -> Tensor:
    # permute like _perm_txt_pad_to_tail but force the pad tail to exact zeros
    # (used for d_att entering the masked backward).
    var head = slice(t, 1, 0, lt, ctx)
    var img = slice(t, 1, n_txt, S - n_txt, ctx)
    var pad = mul_scalar(slice(t, 1, lt, n_txt - lt, ctx), Float32(0.0), ctx)
    return concat(1, ctx, head, img, pad)


def _unperm_zero_pad(
    o: Tensor, lt: Int, n_txt: Int, S: Int, ctx: DeviceContext
) raises -> Tensor:
    # o rows: [txt_valid(lt) | img(S-n_txt) | masked tail] -> original layout
    # [txt_valid | ZERO pad rows | img]; the masked tail is never read.
    if n_txt - lt > S - n_txt:
        raise Error("_unperm_zero_pad: pad segment larger than img segment")
    var txt_valid = slice(o, 1, 0, lt, ctx)
    var img = slice(o, 1, lt, S - n_txt, ctx)
    var pad = mul_scalar(slice(o, 1, lt, n_txt - lt, ctx), Float32(0.0), ctx)
    return concat(1, ctx, txt_valid, pad, img)


def _chroma_sdpa_fwd[
    H: Int, Dh: Int, S: Int, FLASH: Bool
](q: Tensor, k: Tensor, v: Tensor, scale: Float32, ctx: DeviceContext,
  real_lt: Int = -1, n_txt: Int = 0) raises -> Tensor:
    var masked = real_lt >= 0 and n_txt > 0 and real_lt < n_txt
    comptime if FLASH:
        comptime S_PAD = ((S + 127) // 128) * 128
        if masked:
            var qm = _perm_txt_pad_to_tail(_bf16(q, ctx), real_lt, n_txt, S, ctx)
            var km = _perm_txt_pad_to_tail(_bf16(k, ctx), real_lt, n_txt, S, ctx)
            var vm = _perm_txt_pad_to_tail(_bf16(v, ctx), real_lt, n_txt, S, ctx)
            var qp = _pad_seq_zeros[H, Dh](qm, S, S_PAD, ctx)
            var kp = _pad_seq_zeros[H, Dh](km, S, S_PAD, ctx)
            var vp = _pad_seq_zeros[H, Dh](vm, S, S_PAD, ctx)
            var real_len = real_lt + (S - n_txt)
            var f = sdpa_flash_train_fwd_padmask_bf16[1, S_PAD, H, Dh](
                qp, kp, vp, real_len, scale, ctx
            )
            var o = slice(f.att, 1, 0, S, ctx)
            return _unperm_zero_pad(o, real_lt, n_txt, S, ctx)
        var qp = _pad_seq_zeros[H, Dh](_bf16(q, ctx), S, S_PAD, ctx)
        var kp = _pad_seq_zeros[H, Dh](_bf16(k, ctx), S, S_PAD, ctx)
        var vp = _pad_seq_zeros[H, Dh](_bf16(v, ctx), S, S_PAD, ctx)
        var f = sdpa_flash_train_fwd_padmask_bf16[1, S_PAD, H, Dh](qp, kp, vp, S, scale, ctx)
        return slice(f.att, 1, 0, S, ctx)      # bf16 (== math sdpa_nomask output)
    else:
        if masked:
            raise Error("_chroma_sdpa_fwd: T5 pad masking requires the FLASH arm")
        return sdpa_nomask[1, S, H, Dh](q, k, v, scale, ctx)


def _chroma_sdpa_bwd[
    H: Int, Dh: Int, S: Int, FLASH: Bool
](q: Tensor, k: Tensor, v: Tensor, d_att: Tensor, scale: Float32, ctx: DeviceContext,
  real_lt: Int = -1, n_txt: Int = 0) raises -> _ChromaSdpaGrads:
    var masked = real_lt >= 0 and n_txt > 0 and real_lt < n_txt
    comptime if FLASH:
        comptime S_PAD = ((S + 127) // 128) * 128
        if masked:
            var qm = _perm_txt_pad_to_tail(_bf16(q, ctx), real_lt, n_txt, S, ctx)
            var km = _perm_txt_pad_to_tail(_bf16(k, ctx), real_lt, n_txt, S, ctx)
            var vm = _perm_txt_pad_to_tail(_bf16(v, ctx), real_lt, n_txt, S, ctx)
            var qp = _pad_seq_zeros[H, Dh](qm, S, S_PAD, ctx)
            var kp = _pad_seq_zeros[H, Dh](km, S, S_PAD, ctx)
            var vp = _pad_seq_zeros[H, Dh](vm, S, S_PAD, ctx)
            var real_len = real_lt + (S - n_txt)
            # regenerate o/stats (flash fwd on identical permuted+masked inputs)
            var f = sdpa_flash_train_fwd_padmask_bf16[1, S_PAD, H, Dh](
                qp, kp, vp, real_len, scale, ctx
            )
            var dm = _perm_txt_pad_to_tail_zeroed(_bf16(d_att, ctx), real_lt, n_txt, S, ctx)
            var dop = _pad_seq_zeros[H, Dh](dm, S, S_PAD, ctx)
            var g = sdpa_flash_backward_padmask_bf16[1, S_PAD, H, Dh](
                f.q_bf, f.k_bf, f.v_bf, f.o_bf, f.stats, dop, real_len, scale, ctx,
            )
            var dq_m = _unperm_zero_pad(slice(g.d_q, 1, 0, S, ctx), real_lt, n_txt, S, ctx)
            var dk_m = _unperm_zero_pad(slice(g.d_k, 1, 0, S, ctx), real_lt, n_txt, S, ctx)
            var dv_m = _unperm_zero_pad(slice(g.d_v, 1, 0, S, ctx), real_lt, n_txt, S, ctx)
            return _ChromaSdpaGrads(dq_m^, dk_m^, dv_m^)
        var qp = _pad_seq_zeros[H, Dh](_bf16(q, ctx), S, S_PAD, ctx)
        var kp = _pad_seq_zeros[H, Dh](_bf16(k, ctx), S, S_PAD, ctx)
        var vp = _pad_seq_zeros[H, Dh](_bf16(v, ctx), S, S_PAD, ctx)
        # regenerate o/stats (flash fwd on identical padded inputs)
        var f = sdpa_flash_train_fwd_padmask_bf16[1, S_PAD, H, Dh](qp, kp, vp, S, scale, ctx)
        var dop = _pad_seq_zeros[H, Dh](_bf16(d_att, ctx), S, S_PAD, ctx)
        var g = sdpa_flash_backward_padmask_bf16[1, S_PAD, H, Dh](
            f.q_bf, f.k_bf, f.v_bf, f.o_bf, f.stats, dop, S, scale, ctx,
        )
        var dq_r = slice(g.d_q, 1, 0, S, ctx)
        var dk_r = slice(g.d_k, 1, 0, S, ctx)
        var dv_r = slice(g.d_v, 1, 0, S, ctx)
        return _ChromaSdpaGrads(dq_r^, dk_r^, dv_r^)   # bf16 (== math sdpa_backward output)
    else:
        if masked:
            raise Error("_chroma_sdpa_bwd: T5 pad masking requires the FLASH arm")
        var sb = sdpa_backward[1, S, H, Dh](q, k, v, d_att, scale, ctx)
        return _ChromaSdpaGrads(sb.d_q.clone(ctx), sb.d_k.clone(ctx), sb.d_v.clone(ctx))


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLE block — device forward
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct _StreamPreDevice(Copyable, Movable):
    var q_rms: TArc
    var k_rms: TArc
    var v: TArc
    var ln1: TArc
    var norm: TArc
    var q_pre: TArc
    var k_pre: TArc


def _stream_pre_lora_device[
    H: Int, Dh: Int
](
    x: TArc, w: StreamWeights, mv: ModVecsDevice, lo: StreamLoraDevice,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPreDevice:
    var ln1 = layer_norm(x[], ones, zeros, eps, ctx)              # bf16
    var norm = modulate(ln1, mv.scale1[], mv.shift1[], ctx)       # bf16
    var b = Optional[Tensor](w.bqkv[].clone(ctx))
    var qkv = linear(norm, w.wqkv[], b, ctx)                      # [N,3D] bf16
    var q_base = slice(qkv, 1, 0, D, ctx)
    var k_base = slice(qkv, 1, D, D, ctx)
    var v_base = slice(qkv, 1, 2 * D, D, ctx)
    var q_h = _lora_apply_device(q_base^, norm, lo.to_q, N, ctx)  # [N,D] bf16
    var k_h = _lora_apply_device(k_base^, norm, lo.to_k, N, ctx)
    var v_h = _lora_apply_device(v_base^, norm, lo.to_v, N, ctx)
    var q_pre = reshape_owned(q_h^, [1, N, H, Dh])
    var k_pre = reshape_owned(k_h^, [1, N, H, Dh])
    var v = reshape_owned(v_h^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPreDevice(
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
        TArc(ln1^), TArc(norm^), TArc(q_pre^), TArc(k_pre^),
    )


@fieldwise_init
struct _StreamPostDevice(Copyable, Movable):
    var out: TArc
    var attn_res: TArc
    var ln2: TArc
    var mlp_in: TArc
    var mlp_pre: TArc
    var mlp_h: TArc


def _stream_post_lora_device(
    x: TArc, att: TArc, w: StreamWeights, mv: ModVecsDevice, lo: StreamLoraDevice,
    N: Int, D: Int, Fmlp: Int, eps: Float32, ones: Tensor, zeros: Tensor,
    ctx: DeviceContext,
) raises -> _StreamPostDevice:
    var bp = Optional[Tensor](w.bproj[].clone(ctx))
    var out_base = linear(att[], w.wproj[], bp, ctx)             # [N,D]
    var out = _lora_apply_device(out_base^, att[], lo.proj, N, ctx)
    var attn_res = residual_gate(x[], mv.gate1[], out, ctx)      # [N,D]
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var mlp_in = modulate(ln2, mv.scale2[], mv.shift2[], ctx)
    var b0 = Optional[Tensor](w.bmlp0[].clone(ctx))
    var mlp_pre_base = linear(mlp_in, w.wmlp0[], b0, ctx)        # [N,Fmlp]
    var mlp_pre = _lora_apply_device(mlp_pre_base^, mlp_in, lo.mlp0, N, ctx)
    var mlp_h = gelu(mlp_pre, ctx)                               # [N,Fmlp]
    var b2 = Optional[Tensor](w.bmlp2[].clone(ctx))
    var mlp_base = linear(mlp_h, w.wmlp2[], b2, ctx)             # [N,D]
    var mlp_out = _lora_apply_device(mlp_base^, mlp_h, lo.mlp2, N, ctx)
    var final = residual_gate(attn_res, mv.gate2[], mlp_out, ctx)
    return _StreamPostDevice(
        TArc(final^), TArc(attn_res^), TArc(ln2^), TArc(mlp_in^), TArc(mlp_pre^), TArc(mlp_h^),
    )


def chroma_double_block_lora_forward_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, FLASH: Bool = True
](
    img_x: TArc, txt_x: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice,
    lora: DoubleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
    real_lt: Int = -1,
) raises -> DoubleBlockDeviceForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)
    var zeros_t = _t_bf16(_zeros(D), [D], ctx)
    var cos_b = _bf16(cos, ctx)
    var sin_b = _bf16(sin, ctx)

    var ip = _stream_pre_lora_device[H, Dh](
        img_x, w.img, img_mod, lora.img, N_IMG, D, eps, ones_t, zeros_t, ctx)
    var tp = _stream_pre_lora_device[H, Dh](
        txt_x, w.txt, txt_mod, lora.txt, N_TXT, D, eps, ones_t, zeros_t, ctx)

    var q = concat(1, ctx, tp.q_rms[], ip.q_rms[])   # [1,S,H,Dh] txt FIRST
    var k = concat(1, ctx, tp.k_rms[], ip.k_rms[])
    var v = concat(1, ctx, tp.v[], ip.v[])

    var q_rope = rope_interleaved(q, cos_b, sin_b, ctx)
    var k_rope = rope_interleaved(k, cos_b, sin_b, ctx)
    var att = _chroma_sdpa_fwd[H, Dh, S, FLASH](q_rope, k_rope, v, scale, ctx, real_lt, N_TXT)

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = TArc(reshape_owned(txt_att_4d^, [N_TXT, D]))
    var img_att = TArc(reshape_owned(img_att_4d^, [N_IMG, D]))

    var ipost = _stream_post_lora_device(
        img_x, img_att, w.img, img_mod, lora.img, N_IMG, D, Fmlp, eps, ones_t, zeros_t, ctx)
    var tpost = _stream_post_lora_device(
        txt_x, txt_att, w.txt, txt_mod, lora.txt, N_TXT, D, Fmlp, eps, ones_t, zeros_t, ctx)

    var img_saved = StreamSavedDevice(
        img_x.copy(), ip.ln1.copy(), ip.norm.copy(), ip.q_pre.copy(), ip.k_pre.copy(),
        img_att.copy(), ipost.attn_res.copy(), ipost.ln2.copy(), ipost.mlp_in.copy(),
        ipost.mlp_pre.copy(), ipost.mlp_h.copy(),
    )
    var txt_saved = StreamSavedDevice(
        txt_x.copy(), tp.ln1.copy(), tp.norm.copy(), tp.q_pre.copy(), tp.k_pre.copy(),
        txt_att.copy(), tpost.attn_res.copy(), tpost.ln2.copy(), tpost.mlp_in.copy(),
        tpost.mlp_pre.copy(), tpost.mlp_h.copy(),
    )
    var saved = DoubleBlockSavedDevice(
        img_saved^, txt_saved^, TArc(q_rope^), TArc(k_rope^), TArc(v^),
    )
    return DoubleBlockDeviceForward(ipost.out.copy(), tpost.out.copy(), saved^)


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLE block — device backward (LoRA d_a/d_b + d_x ONLY; base/mod grads dropped)
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct _StreamPostBackDevice(Copyable, Movable):
    var d_x: TArc      # [N,D] F32 (stream input grad via gate1 residual)
    var d_att: TArc    # [N,D] bf16 (grad into joint-attention slice)


def _stream_post_backward_lora_device(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: ModVecsDevice, sv: StreamSavedDevice, lo: StreamLoraDevice,
    N: Int, D: Int, Fmlp: Int, eps: Float32, ones: Tensor,
    mut d_a_slots: List[Optional[TArc]], mut d_b_slots: List[Optional[TArc]],
    ctx: DeviceContext,
    b2: Bool = False,
) raises -> _StreamPostBackDevice:
    # b2=True: mods are [2,D] (per-sample adaLN over the 2N row-stack). chroma
    # DISCARDS gate grads (frozen approximator), and gate_residual_backward's d_g
    # reduction is unsupported for [B,C] gates — so pass compute_gate_grad=False.
    # d_x/d_y are byte-identical either way; b1 (b2=False) keeps its default path.
    var comp_g = not b2
    # recompute mlp output WITH LoRA(mlp2) so gate_residual_backward y matches fwd.
    var bm2 = Optional[Tensor](w.bmlp2[].clone(ctx))
    var mlp_base = linear(sv.mlp_h[], w.wmlp2[], bm2, ctx)
    var mlp_y = _lora_apply_device(mlp_base^, sv.mlp_h[], lo.mlp2, N, ctx)   # bf16
    # gate2: FULL-F32 (host `_tf32`); d_out F32.
    var grg2 = gate_residual_backward(d_out[], sv.attn_res[], mv.gate2_f32[], mlp_y, ctx, compute_gate_grad=comp_g)

    # mlp = linear(mlp_h, Wmlp2)[+LoRA(mlp2)]  W [D,Fmlp]. grg2.d_y F32 -> bf16.
    var pm2_dx = _proj_bwd_with_lora_device(
        _bf16(grg2.d_y, ctx), sv.mlp_h[], w.wmlp2[], lo.mlp2, D_MLP2, N, Fmlp, D,
        d_a_slots, d_b_slots, ctx,
    )                                                                       # [N,Fmlp] bf16
    var d_mlp_pre = gelu_backward(pm2_dx, sv.mlp_pre[], ctx)                # bf16

    # mlp_pre = linear(mlp_in, Wmlp0)[+LoRA(mlp0)]  W [Fmlp,D]
    var pm0_dx = _proj_bwd_with_lora_device(
        d_mlp_pre, sv.mlp_in[], w.wmlp0[], lo.mlp0, D_MLP0, N, D, Fmlp,
        d_a_slots, d_b_slots, ctx,
    )                                                                       # [N,D] bf16

    # mlp_in = modulate(ln2, scale2, shift2); frozen mod -> d_x only.
    var mb2 = modulate_backward(pm0_dx, sv.ln2[], mv.scale2[], ctx, False)
    # ln2 weight frozen ones -> d_x only.
    var lnb2_dx = layer_norm_backward_dx(mb2.d_x, sv.attn_res[], ones, eps, ctx)   # bf16
    # attn_res feeds residual (grg2.d_x F32) AND ln2 (bf16): host sums bf16, then F32.
    var d_attn_res_total = add(_bf16(grg2.d_x, ctx), lnb2_dx, ctx)          # bf16
    var d_attn_res_f32 = _f32(d_attn_res_total, ctx)

    # attn_res = residual_gate(x, gate1, proj_out): recompute proj WITH LoRA(proj).
    var bp = Optional[Tensor](w.bproj[].clone(ctx))
    var proj_base = linear(att[], w.wproj[], bp, ctx)
    var proj_out = _lora_apply_device(proj_base^, att[], lo.proj, N, ctx)   # bf16
    var grg1 = gate_residual_backward(d_attn_res_f32, x[], mv.gate1_f32[], proj_out, ctx, compute_gate_grad=comp_g)

    # proj_out = linear(att, Wproj)[+LoRA(proj)]  W [D,D]. grg1.d_y F32 -> bf16.
    var d_att = _proj_bwd_with_lora_device(
        _bf16(grg1.d_y, ctx), att[], w.wproj[], lo.proj, D_PROJ, N, D, D,
        d_a_slots, d_b_slots, ctx,
    )                                                                       # [N,D] bf16
    return _StreamPostBackDevice(TArc(grg1.d_x.clone(ctx)), TArc(d_att^))


def _stream_pre_backward_lora_device[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    w: StreamWeights, mv: ModVecsDevice, sv: StreamSavedDevice, lo: StreamLoraDevice,
    N: Int, D: Int, eps: Float32, ones: Tensor,
    mut d_a_slots: List[Optional[TArc]], mut d_b_slots: List[Optional[TArc]],
    ctx: DeviceContext,
) raises -> Tensor:
    # incoming grads (cat_backward outputs) F32 -> bf16 for the bf16-native chain.
    var dq_b = _bf16(d_q_rms, ctx)
    var dk_b = _bf16(d_k_rms, ctx)
    var dv_b = _bf16(d_v, ctx)
    # rms_norm frozen weight -> d_x only.
    var rb_q_dx = rms_norm_backward_dx(dq_b, sv.q_pre[], w.q_norm[], eps, ctx)   # [1,N,H,Dh]
    var rb_k_dx = rms_norm_backward_dx(dk_b, sv.k_pre[], w.k_norm[], eps, ctx)
    reshape_in_place(rb_q_dx, [N, D])
    reshape_in_place(rb_k_dx, [N, D])
    var d_v_flat = reshape(dv_b, [N, D], ctx)

    # fused qkv base linear: d_norm base = linear_backward_dx(d_qkv, wqkv) [N,D].
    var d_qkv = concat(1, ctx, rb_q_dx, rb_k_dx, d_v_flat)   # [N,3D] bf16
    var base_d_norm = linear_backward_dx(d_qkv, w.wqkv[], N, D, 3 * D, ctx)   # [N,D] bf16
    var d_norm_f32 = _f32(base_d_norm, ctx)

    # to_q/to_k/to_v LoRA share the `norm` input; each d_x_lo sums into d_norm.
    if lo.to_q:
        var lg = _lora_bwd_device(rb_q_dx, sv.norm[], lo.to_q.value(), N, ctx)
        d_a_slots[D_SQ] = Optional[TArc](lg.d_a)
        d_b_slots[D_SQ] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lo.to_k:
        var lg = _lora_bwd_device(rb_k_dx, sv.norm[], lo.to_k.value(), N, ctx)
        d_a_slots[D_SK] = Optional[TArc](lg.d_a)
        d_b_slots[D_SK] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lo.to_v:
        var lg = _lora_bwd_device(d_v_flat, sv.norm[], lo.to_v.value(), N, ctx)
        d_a_slots[D_SV] = Optional[TArc](lg.d_a)
        d_b_slots[D_SV] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)

    # norm = modulate(ln1, scale1, shift1); frozen mod -> d_x only.
    var mb1 = modulate_backward(_bf16(d_norm_f32, ctx), sv.ln1[], mv.scale1[], ctx, False)
    # ln1 weight frozen ones -> d_x only. Return F32 (stream input grad via norm path).
    var lnb1_dx = layer_norm_backward_dx(mb1.d_x, sv.x[], ones, eps, ctx)   # bf16
    return _f32(lnb1_dx, ctx)


def chroma_double_block_lora_backward_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, FLASH: Bool = True
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice,
    lora: DoubleBlockLoraDevice,
    saved: DoubleBlockSavedDevice,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
    real_lt: Int = -1,
) raises -> DoubleBlockLoraDeviceGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)

    var ia = List[Optional[TArc]]()
    var ib = List[Optional[TArc]]()
    var ta = List[Optional[TArc]]()
    var tb = List[Optional[TArc]]()
    for _ in range(DBL_STREAM_SLOTS):
        ia.append(Optional[TArc](None)); ib.append(Optional[TArc](None))
        ta.append(Optional[TArc](None)); tb.append(Optional[TArc](None))

    var ipb = _stream_post_backward_lora_device(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, saved.img, lora.img,
        N_IMG, D, Fmlp, eps, ones_t, ia, ib, ctx,
    )
    var tpb = _stream_post_backward_lora_device(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT, D, Fmlp, eps, ones_t, ta, tb, ctx,
    )

    # join per-stream d_att into joint d_att (txt FIRST); reshape [N,D]->[1,N,H,Dh].
    var d_tatt_4d = reshape(tpb.d_att[], [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = reshape(ipb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)   # [1,S,H,Dh] bf16

    var sb = _chroma_sdpa_bwd[H, Dh, S, FLASH](
        saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx,
        real_lt, N_TXT)

    # F32-ONLY: rope_backward grad + cos/sin F32. sb.d_q/d_k bf16 -> cast up.
    var d_q_joint = rope_backward(_f32(sb.d_q, ctx), cos, sin, True, ctx)
    var d_k_joint = rope_backward(_f32(sb.d_k, ctx), cos, sin, True, ctx)

    # F32-ONLY: cat_backward grad F32. sb.d_v bf16 -> cast up.
    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(_f32(sb.d_v, ctx), N_TXT, N_IMG, 1, ctx)

    var iprb_dx = _stream_pre_backward_lora_device[H, Dh](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, saved.img, lora.img,
        N_IMG, D, eps, ones_t, ia, ib, ctx,
    )
    var tprb_dx = _stream_pre_backward_lora_device[H, Dh](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT, D, eps, ones_t, ta, tb, ctx,
    )

    var d_img_x = add(ipb.d_x[], iprb_dx, ctx)   # F32
    var d_txt_x = add(tpb.d_x[], tprb_dx, ctx)

    var img_g = StreamLoraDeviceGrads(TArc(d_img_x^), ia^, ib^)
    var txt_g = StreamLoraDeviceGrads(TArc(d_txt_x^), ta^, tb^)
    return DoubleBlockLoraDeviceGrads(img_g^, txt_g^)


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE block — device forward + backward
# ═══════════════════════════════════════════════════════════════════════════
def chroma_single_block_lora_forward_device[
    H: Int, Dh: Int, S: Int, FLASH: Bool = True
](
    x: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
    real_lt: Int = -1,
    n_txt: Int = 0,
) raises -> SingleBlockDeviceForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)
    var zeros_t = _t_bf16(_zeros(D), [D], ctx)
    var cos_b = _bf16(cos, ctx)
    var sin_b = _bf16(sin, ctx)

    var ln = layer_norm(x[], ones_t, zeros_t, eps, ctx)          # bf16
    var norm = modulate(ln, mv.scale[], mv.shift[], ctx)         # bf16

    var b1 = Optional[Tensor](w.b1[].clone(ctx))
    var fused = linear(norm, w.w1[], b1, ctx)                    # [S, 3D+Fmlp]
    var q_base = slice(fused, 1, 0, D, ctx)
    var k_base = slice(fused, 1, D, D, ctx)
    var v_base = slice(fused, 1, 2 * D, D, ctx)
    var mlp_base = slice(fused, 1, 3 * D, Fmlp, ctx)             # [S,Fmlp]

    var q_h = _lora_apply_device(q_base^, norm, lora.to_q, S, ctx)
    var k_h = _lora_apply_device(k_base^, norm, lora.to_k, S, ctx)
    var v_h = _lora_apply_device(v_base^, norm, lora.to_v, S, ctx)
    var mlp_in = _lora_apply_device(mlp_base^, norm, lora.proj_mlp, S, ctx)   # [S,Fmlp]

    var q_pre = reshape_owned(q_h^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_h^, [1, S, H, Dh])
    var v = reshape_owned(v_h^, [1, S, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos_b, sin_b, ctx)
    var k_rope = rope_interleaved(k_rms, cos_b, sin_b, ctx)
    var att = _chroma_sdpa_fwd[H, Dh, S, FLASH](q_rope, k_rope, v, scale, ctx, real_lt, n_txt)
    var att_flat = reshape_owned(att^, [S, D])

    var mlp_h = gelu(mlp_in, ctx)                               # [S,Fmlp]
    var out_in = concat(1, ctx, att_flat, mlp_h)               # [S, D+Fmlp]

    var b2 = Optional[Tensor](w.b2[].clone(ctx))
    var out_base = linear(out_in, w.w2[], b2, ctx)             # [S,D]
    var out_proj = _lora_apply_device(out_base^, out_in, lora.linear2, S, ctx)
    var result = residual_gate(x[], mv.gate[], out_proj, ctx)

    var saved = SingleBlockSavedDevice(
        x.copy(), TArc(ln^), TArc(norm^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rope^), TArc(k_rope^), TArc(v^), TArc(att_flat^),
        TArc(mlp_in^), TArc(mlp_h^), TArc(out_in^),
    )
    return SingleBlockDeviceForward(TArc(result^), saved^)


def chroma_single_block_lora_backward_device[
    H: Int, Dh: Int, S: Int, FLASH: Bool = True
](
    d_out: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    saved: SingleBlockSavedDevice,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
    real_lt: Int = -1,
    n_txt: Int = 0,
) raises -> SingleBlockLoraDeviceGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)

    var d_a_slots = List[Optional[TArc]]()
    var d_b_slots = List[Optional[TArc]]()
    for _ in range(SGL_SLOTS):
        d_a_slots.append(Optional[TArc](None)); d_b_slots.append(Optional[TArc](None))

    # result = residual_gate(x, gate, out): recompute out WITH LoRA(linear2).
    var b2 = Optional[Tensor](w.b2[].clone(ctx))
    var out_base = linear(saved.out_in[], w.w2[], b2, ctx)
    var out_y = _lora_apply_device(out_base^, saved.out_in[], lora.linear2, S, ctx)   # bf16
    var grg = gate_residual_backward(d_out[], saved.x[], mv.gate_f32[], out_y, ctx)

    # out = linear(out_in, W2)[+LoRA(linear2)]  W2 [D, D+Fmlp]. grg.d_y F32 -> bf16.
    var pl2_dx = _proj_bwd_with_lora_device(
        _bf16(grg.d_y, ctx), saved.out_in[], w.w2[], lora.linear2, S_L2,
        S, D + Fmlp, D, d_a_slots, d_b_slots, ctx,
    )                                                            # [S, D+Fmlp] bf16

    # out_in = concat(att_flat, mlp_h). F32-ONLY cat_backward.
    var dx_w2_f32 = _f32(pl2_dx, ctx)
    reshape_in_place(dx_w2_f32, [1, S, D + Fmlp])
    var cb = cat_backward(dx_w2_f32, D, Fmlp, 2, ctx)
    reshape_in_place(cb.d_0, [1, S, H, Dh])
    reshape_in_place(cb.d_1, [S, Fmlp])
    var d_att_flat_b = _bf16(cb.d_0, ctx)
    var d_mlp_h_b = _bf16(cb.d_1, ctx)

    var d_mlp_in = gelu_backward(d_mlp_h_b, saved.mlp_in[], ctx)   # [S,Fmlp] bf16

    var sb = _chroma_sdpa_bwd[H, Dh, S, FLASH](
        saved.q_rope[], saved.k_rope[], saved.v[], d_att_flat_b, scale, ctx,
        real_lt, n_txt)

    var d_q_rms = rope_backward(_f32(sb.d_q, ctx), cos, sin, True, ctx)
    var d_k_rms = rope_backward(_f32(sb.d_k, ctx), cos, sin, True, ctx)

    var rb_q_dx = rms_norm_backward_dx(_bf16(d_q_rms, ctx), saved.q_pre[], w.q_norm[], eps, ctx)
    var rb_k_dx = rms_norm_backward_dx(_bf16(d_k_rms, ctx), saved.k_pre[], w.k_norm[], eps, ctx)
    reshape_in_place(rb_q_dx, [S, D])
    reshape_in_place(rb_k_dx, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    # join per-slice d_y into d_fused [S, 3D+Fmlp] (all bf16).
    var d_qkv = concat(1, ctx, rb_q_dx, rb_k_dx, sb.d_v)   # [S,3D]
    var d_fused = concat(1, ctx, d_qkv, d_mlp_in)          # [S,3D+Fmlp]

    # fused = linear(norm, W1, b1): d_norm base = linear_backward_dx(d_fused, w1).
    var base_d_norm = linear_backward_dx(d_fused, w.w1[], S, D, 3 * D + Fmlp, ctx)   # [S,D]
    var d_norm_f32 = _f32(base_d_norm, ctx)

    # LoRA to_q/to_k/to_v (input=norm) + proj_mlp.
    if lora.to_q:
        var lg = _lora_bwd_device(rb_q_dx, saved.norm[], lora.to_q.value(), S, ctx)
        d_a_slots[S_SQ] = Optional[TArc](lg.d_a); d_b_slots[S_SQ] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lora.to_k:
        var lg = _lora_bwd_device(rb_k_dx, saved.norm[], lora.to_k.value(), S, ctx)
        d_a_slots[S_SK] = Optional[TArc](lg.d_a); d_b_slots[S_SK] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lora.to_v:
        var lg = _lora_bwd_device(sb.d_v, saved.norm[], lora.to_v.value(), S, ctx)
        d_a_slots[S_SV] = Optional[TArc](lg.d_a); d_b_slots[S_SV] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lora.proj_mlp:
        var lg = _lora_bwd_device(d_mlp_in, saved.norm[], lora.proj_mlp.value(), S, ctx)
        d_a_slots[S_PMLP] = Optional[TArc](lg.d_a); d_b_slots[S_PMLP] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)

    # norm = modulate(ln, scale, shift); frozen mod -> d_x only.
    var mb = modulate_backward(_bf16(d_norm_f32, ctx), saved.ln[], mv.scale[], ctx, False)
    var lnb_dx = layer_norm_backward_dx(mb.d_x, saved.x[], ones_t, eps, ctx)   # bf16
    var d_x = add(grg.d_x, _f32(lnb_dx, ctx), ctx)   # F32
    return SingleBlockLoraDeviceGrads(TArc(d_x^), d_a_slots^, d_b_slots^)


# ═══════════════════════════════════════════════════════════════════════════
# TRUE BATCH-2 (row-stacked) — device fwd/bwd for BOTH double and single blocks.
#
# DESIGN (mirrors krea2's row-stacked b2). Two samples are laid out ROW-STACKED
# on the token dim: sample0's rows first, then sample1's ([2N, D] streams, [2S,D]
# single). Mod vecs are packed [2, D] (sample0 on row 0, sample1 on row 1); the
# shared modulate/residual_gate ops split the 2N rows into two contiguous
# N-ranges (per-sample adaLN). LoRA/linear GEMMs run at M=2N, so the shared
# adapters' d_a/d_b SUM over both samples inside the GEMM = the batch gradient
# (mean once the caller scales each sample's d_out by 0.5).
#
# ATTENTION never mixes samples. chroma has a UNIFORM comptime S, so both samples
# share one real_len (= S; the padmask flash entry takes a SCALAR real_len, NOT a
# per-batch array). So attention runs PER SAMPLE via TWO reused _chroma_sdpa_fwd/
# _bwd calls (each bit-identical to the b1 attention) — NOT a B=2 flash. The
# per-sample joint q/k/v are assembled, flashed independently, then re-stacked.
#
# The forward stream helpers (_stream_pre_lora_device / _stream_post_lora_device)
# and the pre-backward helper (_stream_pre_backward_lora_device) are REUSED
# UNCHANGED at 2N with [2,D] mods (modulate / modulate_backward(...,False) /
# residual_gate all accept [B,D]). Only _stream_post_backward_lora_device needs
# b2=True (compute_gate_grad=False; chroma discards gate grads).
# ═══════════════════════════════════════════════════════════════════════════

def _stack2_bf16(a: List[Float32], b: List[Float32], D: Int, ctx: DeviceContext) raises -> Tensor:
    var v = List[Float32]()
    for i in range(D):
        v.append(a[i])
    for i in range(D):
        v.append(b[i])
    return Tensor.from_host(v^, [2, D], STDtype.BF16, ctx)


def _stack2_f32(a: List[Float32], b: List[Float32], D: Int, ctx: DeviceContext) raises -> Tensor:
    var v = List[Float32]()
    for i in range(D):
        v.append(a[i])
    for i in range(D):
        v.append(b[i])
    return Tensor.from_host(v^, [2, D], STDtype.F32, ctx)


# pack two host mod-vec sets into [2, D] device tensors (row 0 = sample0). bf16
# for the forward modulate/residual_gate, plus an F32 gate copy for the backward.
def modvecs_pack_b2(mv0: ModVecs, mv1: ModVecs, D: Int, ctx: DeviceContext) raises -> ModVecsDevice:
    return ModVecsDevice(
        TArc(_stack2_bf16(mv0.shift1, mv1.shift1, D, ctx)),
        TArc(_stack2_bf16(mv0.scale1, mv1.scale1, D, ctx)),
        TArc(_stack2_bf16(mv0.gate1, mv1.gate1, D, ctx)),
        TArc(_stack2_bf16(mv0.shift2, mv1.shift2, D, ctx)),
        TArc(_stack2_bf16(mv0.scale2, mv1.scale2, D, ctx)),
        TArc(_stack2_bf16(mv0.gate2, mv1.gate2, D, ctx)),
        TArc(_stack2_f32(mv0.gate1, mv1.gate1, D, ctx)),
        TArc(_stack2_f32(mv0.gate2, mv1.gate2, D, ctx)),
    )


def single_modvecs_pack_b2(mv0: SingleModVecs, mv1: SingleModVecs, D: Int, ctx: DeviceContext) raises -> SingleModVecsDevice:
    return SingleModVecsDevice(
        TArc(_stack2_bf16(mv0.shift, mv1.shift, D, ctx)),
        TArc(_stack2_bf16(mv0.scale, mv1.scale, D, ctx)),
        TArc(_stack2_bf16(mv0.gate, mv1.gate, D, ctx)),
        TArc(_stack2_f32(mv0.gate, mv1.gate, D, ctx)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLE block — device forward (row-stacked b2). img_x [2*N_IMG, D], txt_x
# [2*N_TXT, D]; img_mod/txt_mod packed [2, D]. Saves the per-sample joint tapes
# concatenated on the seq dim ([1, 2S, H, Dh]) for the backward.
# ═══════════════════════════════════════════════════════════════════════════
def chroma_double_block_lora_forward_device_b2[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, FLASH: Bool = True
](
    img_x: TArc, txt_x: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice,
    lora: DoubleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
    real_lt0: Int = -1,
    real_lt1: Int = -1,
) raises -> DoubleBlockDeviceForward:
    comptime N_IMG2 = 2 * N_IMG
    comptime N_TXT2 = 2 * N_TXT
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)
    var zeros_t = _t_bf16(_zeros(D), [D], ctx)
    var cos_b = _bf16(cos, ctx)
    var sin_b = _bf16(sin, ctx)

    var ip = _stream_pre_lora_device[H, Dh](
        img_x, w.img, img_mod, lora.img, N_IMG2, D, eps, ones_t, zeros_t, ctx)
    var tp = _stream_pre_lora_device[H, Dh](
        txt_x, w.txt, txt_mod, lora.txt, N_TXT2, D, eps, ones_t, zeros_t, ctx)

    # per-sample joint attention (txt FIRST then img). q/k/v pre: [1, 2N, H, Dh].
    var tq0 = slice(tp.q_rms[], 1, 0, N_TXT, ctx)
    var tq1 = slice(tp.q_rms[], 1, N_TXT, N_TXT, ctx)
    var iq0 = slice(ip.q_rms[], 1, 0, N_IMG, ctx)
    var iq1 = slice(ip.q_rms[], 1, N_IMG, N_IMG, ctx)
    var tk0 = slice(tp.k_rms[], 1, 0, N_TXT, ctx)
    var tk1 = slice(tp.k_rms[], 1, N_TXT, N_TXT, ctx)
    var ik0 = slice(ip.k_rms[], 1, 0, N_IMG, ctx)
    var ik1 = slice(ip.k_rms[], 1, N_IMG, N_IMG, ctx)
    var tv0 = slice(tp.v[], 1, 0, N_TXT, ctx)
    var tv1 = slice(tp.v[], 1, N_TXT, N_TXT, ctx)
    var iv0 = slice(ip.v[], 1, 0, N_IMG, ctx)
    var iv1 = slice(ip.v[], 1, N_IMG, N_IMG, ctx)

    var q0 = concat(1, ctx, tq0, iq0)   # [1, S, H, Dh]
    var q1 = concat(1, ctx, tq1, iq1)
    var k0 = concat(1, ctx, tk0, ik0)
    var k1 = concat(1, ctx, tk1, ik1)
    var v0 = concat(1, ctx, tv0, iv0)
    var v1 = concat(1, ctx, tv1, iv1)

    var q0r = rope_interleaved(q0, cos_b, sin_b, ctx)
    var q1r = rope_interleaved(q1, cos_b, sin_b, ctx)
    var k0r = rope_interleaved(k0, cos_b, sin_b, ctx)
    var k1r = rope_interleaved(k1, cos_b, sin_b, ctx)

    var att0 = _chroma_sdpa_fwd[H, Dh, S, FLASH](q0r, k0r, v0, scale, ctx, real_lt0, N_TXT)   # [1, S, H, Dh]
    var att1 = _chroma_sdpa_fwd[H, Dh, S, FLASH](q1r, k1r, v1, scale, ctx, real_lt1, N_TXT)

    var ta0 = slice(att0, 1, 0, N_TXT, ctx)
    var ia0 = slice(att0, 1, N_TXT, N_IMG, ctx)
    var ta1 = slice(att1, 1, 0, N_TXT, ctx)
    var ia1 = slice(att1, 1, N_TXT, N_IMG, ctx)
    var txt_att_4d = concat(1, ctx, ta0, ta1)   # [1, N_TXT2, H, Dh]
    var img_att_4d = concat(1, ctx, ia0, ia1)   # [1, N_IMG2, H, Dh]
    var txt_att = TArc(reshape_owned(txt_att_4d^, [N_TXT2, D]))
    var img_att = TArc(reshape_owned(img_att_4d^, [N_IMG2, D]))

    var ipost = _stream_post_lora_device(
        img_x, img_att, w.img, img_mod, lora.img, N_IMG2, D, Fmlp, eps, ones_t, zeros_t, ctx)
    var tpost = _stream_post_lora_device(
        txt_x, txt_att, w.txt, txt_mod, lora.txt, N_TXT2, D, Fmlp, eps, ones_t, zeros_t, ctx)

    var q_rope_st = concat(1, ctx, q0r, q1r)   # [1, 2S, H, Dh]
    var k_rope_st = concat(1, ctx, k0r, k1r)
    var v_st = concat(1, ctx, v0, v1)

    var img_saved = StreamSavedDevice(
        img_x.copy(), ip.ln1.copy(), ip.norm.copy(), ip.q_pre.copy(), ip.k_pre.copy(),
        img_att.copy(), ipost.attn_res.copy(), ipost.ln2.copy(), ipost.mlp_in.copy(),
        ipost.mlp_pre.copy(), ipost.mlp_h.copy(),
    )
    var txt_saved = StreamSavedDevice(
        txt_x.copy(), tp.ln1.copy(), tp.norm.copy(), tp.q_pre.copy(), tp.k_pre.copy(),
        txt_att.copy(), tpost.attn_res.copy(), tpost.ln2.copy(), tpost.mlp_in.copy(),
        tpost.mlp_pre.copy(), tpost.mlp_h.copy(),
    )
    var saved = DoubleBlockSavedDevice(
        img_saved^, txt_saved^, TArc(q_rope_st^), TArc(k_rope_st^), TArc(v_st^),
    )
    return DoubleBlockDeviceForward(ipost.out.copy(), tpost.out.copy(), saved^)


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLE block — device backward (row-stacked b2). d_io_t [2*N_IMG, D], d_to_t
# [2*N_TXT, D]. LoRA d_a/d_b sum over both samples (M=2N GEMM). d_x per stream.
# ═══════════════════════════════════════════════════════════════════════════
def chroma_double_block_lora_backward_device_b2[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int, FLASH: Bool = True
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: ModVecsDevice, txt_mod: ModVecsDevice,
    lora: DoubleBlockLoraDevice,
    saved: DoubleBlockSavedDevice,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
    real_lt0: Int = -1,
    real_lt1: Int = -1,
) raises -> DoubleBlockLoraDeviceGrads:
    comptime N_IMG2 = 2 * N_IMG
    comptime N_TXT2 = 2 * N_TXT
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)

    var ia = List[Optional[TArc]]()
    var ib = List[Optional[TArc]]()
    var ta = List[Optional[TArc]]()
    var tb = List[Optional[TArc]]()
    for _ in range(DBL_STREAM_SLOTS):
        ia.append(Optional[TArc](None)); ib.append(Optional[TArc](None))
        ta.append(Optional[TArc](None)); tb.append(Optional[TArc](None))

    var ipb = _stream_post_backward_lora_device(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, saved.img, lora.img,
        N_IMG2, D, Fmlp, eps, ones_t, ia, ib, ctx, True,
    )
    var tpb = _stream_post_backward_lora_device(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT2, D, Fmlp, eps, ones_t, ta, tb, ctx, True,
    )

    # per-stream d_att [2N, D] -> 4d -> split per sample -> per-sample joint d_att.
    var d_iatt_4d = reshape(ipb.d_att[], [1, N_IMG2, H, Dh], ctx)
    var d_tatt_4d = reshape(tpb.d_att[], [1, N_TXT2, H, Dh], ctx)
    var d_ta0 = slice(d_tatt_4d, 1, 0, N_TXT, ctx)
    var d_ta1 = slice(d_tatt_4d, 1, N_TXT, N_TXT, ctx)
    var d_ia0 = slice(d_iatt_4d, 1, 0, N_IMG, ctx)
    var d_ia1 = slice(d_iatt_4d, 1, N_IMG, N_IMG, ctx)
    var d_att0 = concat(1, ctx, d_ta0, d_ia0)   # [1, S, H, Dh] bf16
    var d_att1 = concat(1, ctx, d_ta1, d_ia1)

    # split saved joint tapes per sample.
    var q0r = slice(saved.q_rope[], 1, 0, S, ctx)
    var q1r = slice(saved.q_rope[], 1, S, S, ctx)
    var k0r = slice(saved.k_rope[], 1, 0, S, ctx)
    var k1r = slice(saved.k_rope[], 1, S, S, ctx)
    var v0 = slice(saved.v_joint[], 1, 0, S, ctx)
    var v1 = slice(saved.v_joint[], 1, S, S, ctx)

    var sb0 = _chroma_sdpa_bwd[H, Dh, S, FLASH](q0r, k0r, v0, d_att0, scale, ctx, real_lt0, N_TXT)
    var sb1 = _chroma_sdpa_bwd[H, Dh, S, FLASH](q1r, k1r, v1, d_att1, scale, ctx, real_lt1, N_TXT)

    var dq0 = rope_backward(_f32(sb0.d_q, ctx), cos, sin, True, ctx)   # [1, S, H, Dh] F32
    var dq1 = rope_backward(_f32(sb1.d_q, ctx), cos, sin, True, ctx)
    var dk0 = rope_backward(_f32(sb0.d_k, ctx), cos, sin, True, ctx)
    var dk1 = rope_backward(_f32(sb1.d_k, ctx), cos, sin, True, ctx)
    var dv0 = _f32(sb0.d_v, ctx)
    var dv1 = _f32(sb1.d_v, ctx)

    # cat_backward per sample: split joint S -> txt N_TXT (d_0), img N_IMG (d_1).
    var cq0 = cat_backward(dq0, N_TXT, N_IMG, 1, ctx)
    var cq1 = cat_backward(dq1, N_TXT, N_IMG, 1, ctx)
    var ck0 = cat_backward(dk0, N_TXT, N_IMG, 1, ctx)
    var ck1 = cat_backward(dk1, N_TXT, N_IMG, 1, ctx)
    var cv0 = cat_backward(dv0, N_TXT, N_IMG, 1, ctx)
    var cv1 = cat_backward(dv1, N_TXT, N_IMG, 1, ctx)

    # re-stack row-wise per stream: txt [1, N_TXT2], img [1, N_IMG2].
    var d_q_txt = concat(1, ctx, cq0.d_0, cq1.d_0)
    var d_q_img = concat(1, ctx, cq0.d_1, cq1.d_1)
    var d_k_txt = concat(1, ctx, ck0.d_0, ck1.d_0)
    var d_k_img = concat(1, ctx, ck0.d_1, ck1.d_1)
    var d_v_txt = concat(1, ctx, cv0.d_0, cv1.d_0)
    var d_v_img = concat(1, ctx, cv0.d_1, cv1.d_1)

    var iprb_dx = _stream_pre_backward_lora_device[H, Dh](
        d_q_img, d_k_img, d_v_img, w.img, img_mod, saved.img, lora.img,
        N_IMG2, D, eps, ones_t, ia, ib, ctx,
    )
    var tprb_dx = _stream_pre_backward_lora_device[H, Dh](
        d_q_txt, d_k_txt, d_v_txt, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT2, D, eps, ones_t, ta, tb, ctx,
    )

    var d_img_x = add(ipb.d_x[], iprb_dx, ctx)   # F32
    var d_txt_x = add(tpb.d_x[], tprb_dx, ctx)

    var img_g = StreamLoraDeviceGrads(TArc(d_img_x^), ia^, ib^)
    var txt_g = StreamLoraDeviceGrads(TArc(d_txt_x^), ta^, tb^)
    return DoubleBlockLoraDeviceGrads(img_g^, txt_g^)


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE block — device forward + backward (row-stacked b2). x [2S, D]; mv [2,D].
# ═══════════════════════════════════════════════════════════════════════════
def chroma_single_block_lora_forward_device_b2[
    H: Int, Dh: Int, S: Int, FLASH: Bool = True
](
    x: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
    real_lt0: Int = -1,
    real_lt1: Int = -1,
    n_txt: Int = 0,
) raises -> SingleBlockDeviceForward:
    comptime S2 = 2 * S
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)
    var zeros_t = _t_bf16(_zeros(D), [D], ctx)
    var cos_b = _bf16(cos, ctx)
    var sin_b = _bf16(sin, ctx)

    var ln = layer_norm(x[], ones_t, zeros_t, eps, ctx)          # [2S, D]
    var norm = modulate(ln, mv.scale[], mv.shift[], ctx)         # [2,D] mods split

    var b1 = Optional[Tensor](w.b1[].clone(ctx))
    var fused = linear(norm, w.w1[], b1, ctx)                    # [2S, 3D+Fmlp]
    var q_base = slice(fused, 1, 0, D, ctx)
    var k_base = slice(fused, 1, D, D, ctx)
    var v_base = slice(fused, 1, 2 * D, D, ctx)
    var mlp_base = slice(fused, 1, 3 * D, Fmlp, ctx)

    var q_h = _lora_apply_device(q_base^, norm, lora.to_q, S2, ctx)
    var k_h = _lora_apply_device(k_base^, norm, lora.to_k, S2, ctx)
    var v_h = _lora_apply_device(v_base^, norm, lora.to_v, S2, ctx)
    var mlp_in = _lora_apply_device(mlp_base^, norm, lora.proj_mlp, S2, ctx)   # [2S, Fmlp]

    var q_pre = reshape_owned(q_h^, [1, S2, H, Dh])
    var k_pre = reshape_owned(k_h^, [1, S2, H, Dh])
    var v = reshape_owned(v_h^, [1, S2, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    # per-sample rope + attention (each [1, S, H, Dh]).
    var q0 = slice(q_rms, 1, 0, S, ctx)
    var q1 = slice(q_rms, 1, S, S, ctx)
    var k0 = slice(k_rms, 1, 0, S, ctx)
    var k1 = slice(k_rms, 1, S, S, ctx)
    var v0 = slice(v, 1, 0, S, ctx)
    var v1 = slice(v, 1, S, S, ctx)
    var q0r = rope_interleaved(q0, cos_b, sin_b, ctx)
    var q1r = rope_interleaved(q1, cos_b, sin_b, ctx)
    var k0r = rope_interleaved(k0, cos_b, sin_b, ctx)
    var k1r = rope_interleaved(k1, cos_b, sin_b, ctx)
    var att0 = _chroma_sdpa_fwd[H, Dh, S, FLASH](q0r, k0r, v0, scale, ctx, real_lt0, n_txt)
    var att1 = _chroma_sdpa_fwd[H, Dh, S, FLASH](q1r, k1r, v1, scale, ctx, real_lt1, n_txt)
    var att = concat(1, ctx, att0, att1)                        # [1, 2S, H, Dh]
    var att_flat = reshape_owned(att^, [S2, D])

    var q_rope_st = concat(1, ctx, q0r, q1r)   # [1, 2S, H, Dh]
    var k_rope_st = concat(1, ctx, k0r, k1r)

    var mlp_h = gelu(mlp_in, ctx)                               # [2S, Fmlp]
    var out_in = concat(1, ctx, att_flat, mlp_h)               # [2S, D+Fmlp]

    var b2t = Optional[Tensor](w.b2[].clone(ctx))
    var out_base = linear(out_in, w.w2[], b2t, ctx)            # [2S, D]
    var out_proj = _lora_apply_device(out_base^, out_in, lora.linear2, S2, ctx)
    var result = residual_gate(x[], mv.gate[], out_proj, ctx)  # [2,D] gate splits

    var saved = SingleBlockSavedDevice(
        x.copy(), TArc(ln^), TArc(norm^), TArc(q_pre^), TArc(k_pre^),
        TArc(q_rope_st^), TArc(k_rope_st^), TArc(v^), TArc(att_flat^),
        TArc(mlp_in^), TArc(mlp_h^), TArc(out_in^),
    )
    return SingleBlockDeviceForward(TArc(result^), saved^)


def chroma_single_block_lora_backward_device_b2[
    H: Int, Dh: Int, S: Int, FLASH: Bool = True
](
    d_out: TArc,
    w: SingleBlockWeights, mv: SingleModVecsDevice, lora: SingleBlockLoraDevice,
    saved: SingleBlockSavedDevice,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
    real_lt0: Int = -1,
    real_lt1: Int = -1,
    n_txt: Int = 0,
) raises -> SingleBlockLoraDeviceGrads:
    comptime S2 = 2 * S
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)

    var d_a_slots = List[Optional[TArc]]()
    var d_b_slots = List[Optional[TArc]]()
    for _ in range(SGL_SLOTS):
        d_a_slots.append(Optional[TArc](None)); d_b_slots.append(Optional[TArc](None))

    # result = residual_gate(x, gate, out): recompute out WITH LoRA(linear2).
    var b2t = Optional[Tensor](w.b2[].clone(ctx))
    var out_base = linear(saved.out_in[], w.w2[], b2t, ctx)
    var out_y = _lora_apply_device(out_base^, saved.out_in[], lora.linear2, S2, ctx)   # bf16
    var grg = gate_residual_backward(d_out[], saved.x[], mv.gate_f32[], out_y, ctx, compute_gate_grad=False)

    var pl2_dx = _proj_bwd_with_lora_device(
        _bf16(grg.d_y, ctx), saved.out_in[], w.w2[], lora.linear2, S_L2,
        S2, D + Fmlp, D, d_a_slots, d_b_slots, ctx,
    )                                                            # [2S, D+Fmlp] bf16

    var dx_w2_f32 = _f32(pl2_dx, ctx)
    reshape_in_place(dx_w2_f32, [1, S2, D + Fmlp])
    var cb = cat_backward(dx_w2_f32, D, Fmlp, 2, ctx)
    reshape_in_place(cb.d_0, [1, S2, H, Dh])
    reshape_in_place(cb.d_1, [S2, Fmlp])
    var d_att_flat_b = _bf16(cb.d_0, ctx)
    var d_mlp_h_b = _bf16(cb.d_1, ctx)

    var d_mlp_in = gelu_backward(d_mlp_h_b, saved.mlp_in[], ctx)   # [2S, Fmlp]

    # per-sample attention backward.
    var d_att0 = slice(d_att_flat_b, 1, 0, S, ctx)
    var d_att1 = slice(d_att_flat_b, 1, S, S, ctx)
    var q0r = slice(saved.q_rope[], 1, 0, S, ctx)
    var q1r = slice(saved.q_rope[], 1, S, S, ctx)
    var k0r = slice(saved.k_rope[], 1, 0, S, ctx)
    var k1r = slice(saved.k_rope[], 1, S, S, ctx)
    var v0 = slice(saved.v[], 1, 0, S, ctx)
    var v1 = slice(saved.v[], 1, S, S, ctx)
    var sb0 = _chroma_sdpa_bwd[H, Dh, S, FLASH](q0r, k0r, v0, d_att0, scale, ctx, real_lt0, n_txt)
    var sb1 = _chroma_sdpa_bwd[H, Dh, S, FLASH](q1r, k1r, v1, d_att1, scale, ctx, real_lt1, n_txt)

    var dq0 = rope_backward(_f32(sb0.d_q, ctx), cos, sin, True, ctx)
    var dq1 = rope_backward(_f32(sb1.d_q, ctx), cos, sin, True, ctx)
    var dk0 = rope_backward(_f32(sb0.d_k, ctx), cos, sin, True, ctx)
    var dk1 = rope_backward(_f32(sb1.d_k, ctx), cos, sin, True, ctx)
    var d_q_rms = concat(1, ctx, dq0, dq1)   # [1, 2S, H, Dh] F32
    var d_k_rms = concat(1, ctx, dk0, dk1)
    var d_v_st = concat(1, ctx, sb0.d_v, sb1.d_v)   # [1, 2S, H, Dh] bf16

    var rb_q_dx = rms_norm_backward_dx(_bf16(d_q_rms, ctx), saved.q_pre[], w.q_norm[], eps, ctx)
    var rb_k_dx = rms_norm_backward_dx(_bf16(d_k_rms, ctx), saved.k_pre[], w.k_norm[], eps, ctx)
    reshape_in_place(rb_q_dx, [S2, D])
    reshape_in_place(rb_k_dx, [S2, D])
    var d_v_flat = reshape(d_v_st, [S2, D], ctx)   # bf16

    var d_qkv = concat(1, ctx, rb_q_dx, rb_k_dx, d_v_flat)   # [2S, 3D] bf16
    var d_fused = concat(1, ctx, d_qkv, d_mlp_in)           # [2S, 3D+Fmlp]

    var base_d_norm = linear_backward_dx(d_fused, w.w1[], S2, D, 3 * D + Fmlp, ctx)   # [2S, D]
    var d_norm_f32 = _f32(base_d_norm, ctx)

    if lora.to_q:
        var lg = _lora_bwd_device(rb_q_dx, saved.norm[], lora.to_q.value(), S2, ctx)
        d_a_slots[S_SQ] = Optional[TArc](lg.d_a); d_b_slots[S_SQ] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lora.to_k:
        var lg = _lora_bwd_device(rb_k_dx, saved.norm[], lora.to_k.value(), S2, ctx)
        d_a_slots[S_SK] = Optional[TArc](lg.d_a); d_b_slots[S_SK] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lora.to_v:
        var lg = _lora_bwd_device(d_v_flat, saved.norm[], lora.to_v.value(), S2, ctx)
        d_a_slots[S_SV] = Optional[TArc](lg.d_a); d_b_slots[S_SV] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)
    if lora.proj_mlp:
        var lg = _lora_bwd_device(d_mlp_in, saved.norm[], lora.proj_mlp.value(), S2, ctx)
        d_a_slots[S_PMLP] = Optional[TArc](lg.d_a); d_b_slots[S_PMLP] = Optional[TArc](lg.d_b)
        d_norm_f32 = add(d_norm_f32, _f32(lg.d_x_lo[], ctx), ctx)

    var mb = modulate_backward(_bf16(d_norm_f32, ctx), saved.ln[], mv.scale[], ctx, False)
    var lnb_dx = layer_norm_backward_dx(mb.d_x, saved.x[], ones_t, eps, ctx)   # bf16
    var d_x = add(grg.d_x, _f32(lnb_dx, ctx), ctx)   # F32
    return SingleBlockLoraDeviceGrads(TArc(d_x^), d_a_slots^, d_b_slots^)
