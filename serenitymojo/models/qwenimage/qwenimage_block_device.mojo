# serenitymojo/models/qwenimage/qwenimage_block_device.mojo
#
# DEVICE-RESIDENT QwenImage DOUBLE LoRA block, fwd + bwd (MJ-1084 port).
#
# Device transcription of the HOST qwen LoRA block (qwenimage_block.mojo —
# the torch-gated math oracle), chroma_block_device.mojo template (MJ-1068 /
# flux 238c089 doctrine: the disease is host-activation ROUND-TRIPS — the host
# block to_hosts every projection input to compute LoRA deltas on host lists,
# uploads per-block mod vecs from host lists, and to_hosts the block outputs;
# 95 sites, dmon: SM never >=50%). Activations stay on GPU as TArc; NO host
# List carriers in the hot path.
#
# NUMERICS: bit-faithful to the host block —
#   * activations bf16 (host `_t` uploads BF16; same shared ops, same dtypes);
#   * LoRA apply reproduces the host's EXACT double rounding:
#     host: delta = klein_lora_fwd (scale inside, F32 host math) -> _t BF16
#     (round 1) -> add(base bf16, delta bf16) (F32 math, round 2).
#     device: dy=linear(linear(x,a),b) bf16; delta_bf=bf16(scale*f32(dy))
#     (round 1); add(base, delta_bf) (round 2).
#   * frozen base/mod/norm: backward returns ONLY d_x + per-slot LoRA d_a/d_b
#     (linear_backward_dx for frozen bases; modulate/ln/rms `_dx` variants).
#
# QWEN vs the chroma template (QWEN_DEVICE_PORT_MAP_2026-07-06.md):
#   * THREE separate biased q/k/v linears (wq/bq, wk/bk, wv/bv) instead of the
#     fused wqkv+slice; backward sums three linear_backward_dx into d_norm.
#   * renames: wproj/bproj->wout/bout, wmlp0/bmlp0->wup/bup, wmlp2/bmlp2->wdn/bdn.
#   * 60 DOUBLE blocks only (num_single=0) — no single-block sections.
#   * StreamLora slots: q,k,v,out,ff_up,ff_down (order == flux to_q..mlp2).
#   * MLP is PLAIN GELU. Joint attention txt FIRST then img (host block :1571).
#
# Mojo current beta: `def` (not fn), `comptime` (not alias), `var`, explicit raises.

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

from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
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

# base weights + host modulation carriers + host LoRA carriers (to convert)
from serenitymojo.models.qwenimage.qwenimage_block import (
    ModVecs, StreamWeights, DoubleBlockWeights, StreamLora, DoubleBlockLora,
)


comptime TArc = ArcPointer[Tensor]

# per-stream LoRA slot order (mirrors flux D_SQ..D_MLP2 semantics)
comptime QWEN_STREAM_SLOTS = 6
comptime Q_SQ = 0    # q
comptime Q_SK = 1    # k
comptime Q_SV = 2    # v
comptime Q_OUT = 3   # attn out proj
comptime Q_UP = 4    # mlp up   (ff_up)
comptime Q_DN = 5    # mlp down (ff_down)


# ── local host->device upload / cast helpers ─────────────────────────────────
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
# Device LoRA adapter + carriers (bf16 a/b; LoraAdapter stores List[BFloat16]).
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct QwenLoraAdapterDevice(Copyable, Movable):
    var a: TArc        # [rank, in_f]  bf16
    var b: TArc        # [out_f, rank] bf16
    var rank: Int
    var in_f: Int
    var out_f: Int
    var scale: Float32


def qwen_lora_adapter_to_device(
    lo: LoraAdapter, ctx: DeviceContext
) raises -> QwenLoraAdapterDevice:
    return QwenLoraAdapterDevice(
        TArc(Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx)),
        TArc(Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx)),
        lo.rank, lo.in_f, lo.out_f, lo.scale,
    )


def _opt_lora_to_device(
    lo: Optional[LoraAdapter], ctx: DeviceContext
) raises -> Optional[QwenLoraAdapterDevice]:
    if lo:
        return Optional[QwenLoraAdapterDevice](qwen_lora_adapter_to_device(lo.value(), ctx))
    return Optional[QwenLoraAdapterDevice](None)


@fieldwise_init
struct QwenStreamLoraDevice(Copyable, Movable):
    var q: Optional[QwenLoraAdapterDevice]
    var k: Optional[QwenLoraAdapterDevice]
    var v: Optional[QwenLoraAdapterDevice]
    var out: Optional[QwenLoraAdapterDevice]
    var ff_up: Optional[QwenLoraAdapterDevice]
    var ff_down: Optional[QwenLoraAdapterDevice]


def qwen_stream_lora_to_device(lo: StreamLora, ctx: DeviceContext) raises -> QwenStreamLoraDevice:
    return QwenStreamLoraDevice(
        _opt_lora_to_device(lo.q, ctx), _opt_lora_to_device(lo.k, ctx),
        _opt_lora_to_device(lo.v, ctx), _opt_lora_to_device(lo.out, ctx),
        _opt_lora_to_device(lo.ff_up, ctx), _opt_lora_to_device(lo.ff_down, ctx),
    )


@fieldwise_init
struct QwenDoubleBlockLoraDevice(Copyable, Movable):
    var img: QwenStreamLoraDevice
    var txt: QwenStreamLoraDevice


def qwen_double_block_lora_to_device(
    lora: DoubleBlockLora, ctx: DeviceContext
) raises -> QwenDoubleBlockLoraDevice:
    return QwenDoubleBlockLoraDevice(
        qwen_stream_lora_to_device(lora.img, ctx),
        qwen_stream_lora_to_device(lora.txt, ctx),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Device modulation vectors (uploaded ONCE per step, not per block call —
# the host block re-uploads all 6 vecs from host lists at EVERY block).
# qwen's host uses BF16 gates in BOTH forward residual_gate AND backward
# gate_residual_backward (`_t(mv.gate2)` — qwenimage_block.mojo:1692; unlike
# chroma's host which used F32 backward gates). All six vecs bf16.
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct QwenModVecsDevice(Copyable, Movable):
    var shift1: TArc      # bf16
    var scale1: TArc      # bf16
    var gate1: TArc       # bf16
    var shift2: TArc      # bf16
    var scale2: TArc      # bf16
    var gate2: TArc       # bf16


def qwen_modvecs_to_device(mv: ModVecs, D: Int, ctx: DeviceContext) raises -> QwenModVecsDevice:
    return QwenModVecsDevice(
        TArc(_t_bf16(mv.shift1.copy(), [D], ctx)),
        TArc(_t_bf16(mv.scale1.copy(), [D], ctx)),
        TArc(_t_bf16(mv.gate1.copy(), [D], ctx)),
        TArc(_t_bf16(mv.shift2.copy(), [D], ctx)),
        TArc(_t_bf16(mv.scale2.copy(), [D], ctx)),
        TArc(_t_bf16(mv.gate2.copy(), [D], ctx)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Device saved activations
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct QwenStreamSavedDevice(Copyable, Movable):
    var x: TArc         # [N,D]
    var ln1: TArc       # [N,D]
    var norm: TArc      # [N,D]
    var q_pre: TArc     # [1,N,H,Dh]
    var k_pre: TArc     # [1,N,H,Dh]
    var att: TArc       # [N,D]  per-stream attention slice
    var attn_res: TArc  # [N,D]
    var ln2: TArc       # [N,D]
    var ff_in: TArc     # [N,D]
    var ff_up: TArc     # [N,F]  gelu input
    var ff_act: TArc    # [N,F]  gelu output


@fieldwise_init
struct QwenDoubleBlockSavedDevice(Copyable, Movable):
    var img: QwenStreamSavedDevice
    var txt: QwenStreamSavedDevice
    var q_rope: TArc    # [1,S,H,Dh]
    var k_rope: TArc    # [1,S,H,Dh]
    var v_joint: TArc   # [1,S,H,Dh]


@fieldwise_init
struct QwenDoubleBlockDeviceForward(Copyable, Movable):
    var img_out: TArc   # [N_IMG, D]  device-resident
    var txt_out: TArc   # [N_TXT, D]
    var saved: QwenDoubleBlockSavedDevice


@fieldwise_init
struct QwenStreamLoraDeviceGrads(Copyable, Movable):
    var d_x: TArc                    # [N,D]  F32
    var d_a: List[Optional[TArc]]    # QWEN_STREAM_SLOTS entries
    var d_b: List[Optional[TArc]]


@fieldwise_init
struct QwenDoubleBlockLoraDeviceGrads(Copyable, Movable):
    var img: QwenStreamLoraDeviceGrads
    var txt: QwenStreamLoraDeviceGrads


# ═══════════════════════════════════════════════════════════════════════════
# Device LoRA fwd/bwd primitives — reproduce the host's EXACT rounding:
# host fwd: delta = klein_lora_fwd (scale inside, host F32) -> _t BF16 (round 1)
#           -> add(base bf16, delta bf16) (F32 math, round 2).
# ═══════════════════════════════════════════════════════════════════════════
def _lora_apply_device(
    var base_y: Tensor, x: Tensor, lo: Optional[QwenLoraAdapterDevice],
    M: Int, ctx: DeviceContext,
) raises -> Tensor:
    if not lo:
        return base_y^
    ref l = lo.value()
    var nb1 = Optional[Tensor](None)
    var t = linear(x, l.a[], nb1^, ctx)                 # [M,rank] bf16
    var nb2 = Optional[Tensor](None)
    var dy = linear(t, l.b[], nb2^, ctx)                # [M,out] bf16
    # round 1: bf16(scale * f32(dy)) — mirrors the host delta upload `_t`.
    var delta_bf = _bf16(mul_scalar(_f32(dy, ctx), l.scale, ctx), ctx)
    # round 2: add (F32 math, bf16 store) — mirrors host add(y, delta).
    return add(base_y, delta_bf, ctx)


def qwen_lora_apply_device(
    var base_y: Tensor,
    x: Tensor,
    lo: Optional[QwenLoraAdapterDevice],
    M: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Public inference seam for the parity-gated Qwen device LoRA primitive."""
    return _lora_apply_device(base_y^, x, lo, M, ctx)


@fieldwise_init
struct _QwenLoraBwd(Copyable, Movable):
    var d_a: TArc      # [rank,in_f] bf16
    var d_b: TArc      # [out_f,rank] bf16
    var d_x_lo: TArc   # [M,in_f]  bf16


def _lora_bwd_device(
    d_contrib: Tensor, x: Tensor, lo: QwenLoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> _QwenLoraBwd:
    var nb = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb^, ctx)                                      # [M,rank] bf16
    var d_dy = _bf16(mul_scalar(_f32(d_contrib, ctx), lo.scale, ctx), ctx)   # bf16
    var d_t = linear_backward_dx(d_dy, lo.b[], M, lo.rank, lo.out_f, ctx)    # [M,rank]
    var d_b = linear_backward_dw(d_dy, t, M, lo.rank, lo.out_f, ctx)         # [out_f,rank]
    var d_x_lo = linear_backward_dx(d_t, lo.a[], M, lo.in_f, lo.rank, ctx)   # [M,in_f]
    var d_a = linear_backward_dw(d_t, x, M, lo.in_f, lo.rank, ctx)           # [rank,in_f]
    return _QwenLoraBwd(TArc(d_a^), TArc(d_b^), TArc(d_x_lo^))


def _proj_bwd_with_lora_device(
    d_y: Tensor, x_in: Tensor, w: Tensor,
    lo: Optional[QwenLoraAdapterDevice], slot: Int,
    M: Int, in_f: Int, out_f: Int,
    mut d_a_slots: List[Optional[TArc]], mut d_b_slots: List[Optional[TArc]],
    ctx: DeviceContext,
) raises -> Tensor:
    var base_dx = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)   # bf16
    if lo:
        var lg = _lora_bwd_device(d_y, x_in, lo.value(), M, ctx)
        d_a_slots[slot] = Optional[TArc](lg.d_a)
        d_b_slots[slot] = Optional[TArc](lg.d_b)
        return add(base_dx, lg.d_x_lo[], ctx)
    return base_dx^


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLE block — device forward
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct _QwenStreamPreDevice(Copyable, Movable):
    var q_rms: TArc
    var k_rms: TArc
    var v: TArc
    var ln1: TArc
    var norm: TArc
    var q_pre: TArc
    var k_pre: TArc


def _qwen_stream_pre_lora_device[
    H: Int, Dh: Int
](
    x: TArc, w: StreamWeights, mv: QwenModVecsDevice, lo: QwenStreamLoraDevice,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _QwenStreamPreDevice:
    var ln1 = layer_norm(x[], ones, zeros, eps, ctx)              # bf16
    var norm = modulate(ln1, mv.scale1[], mv.shift1[], ctx)       # bf16
    # THREE separate biased linears on the SAME norm (qwen delta vs chroma).
    var bq = Optional[Tensor](w.bq[].clone(ctx))
    var q_base = linear(norm, w.wq[], bq, ctx)
    var bk = Optional[Tensor](w.bk[].clone(ctx))
    var k_base = linear(norm, w.wk[], bk, ctx)
    var bv = Optional[Tensor](w.bv[].clone(ctx))
    var v_base = linear(norm, w.wv[], bv, ctx)
    var q_h = _lora_apply_device(q_base^, norm, lo.q, N, ctx)     # [N,D] bf16
    var k_h = _lora_apply_device(k_base^, norm, lo.k, N, ctx)
    var v_h = _lora_apply_device(v_base^, norm, lo.v, N, ctx)
    var q_pre = reshape_owned(q_h^, [1, N, H, Dh])
    var k_pre = reshape_owned(k_h^, [1, N, H, Dh])
    var v = reshape_owned(v_h^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _QwenStreamPreDevice(
        TArc(q_rms^), TArc(k_rms^), TArc(v^),
        TArc(ln1^), TArc(norm^), TArc(q_pre^), TArc(k_pre^),
    )


@fieldwise_init
struct _QwenStreamPostDevice(Copyable, Movable):
    var out: TArc
    var attn_res: TArc
    var ln2: TArc
    var ff_in: TArc
    var ff_up: TArc
    var ff_act: TArc


def _qwen_stream_post_lora_device(
    x: TArc, att: TArc, w: StreamWeights, mv: QwenModVecsDevice, lo: QwenStreamLoraDevice,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor, zeros: Tensor,
    ctx: DeviceContext,
) raises -> _QwenStreamPostDevice:
    var bo = Optional[Tensor](w.bout[].clone(ctx))
    var out_base = linear(att[], w.wout[], bo, ctx)              # [N,D]
    var out = _lora_apply_device(out_base^, att[], lo.out, N, ctx)
    var attn_res = residual_gate(x[], mv.gate1[], out, ctx)      # [N,D]
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var ff_in = modulate(ln2, mv.scale2[], mv.shift2[], ctx)
    var bu = Optional[Tensor](w.bup[].clone(ctx))
    var ff_up_base = linear(ff_in, w.wup[], bu, ctx)             # [N,F]
    var ff_up = _lora_apply_device(ff_up_base^, ff_in, lo.ff_up, N, ctx)
    var ff_act = gelu(ff_up, ctx)                                # [N,F]
    var bd = Optional[Tensor](w.bdn[].clone(ctx))
    var ff_down_base = linear(ff_act, w.wdn[], bd, ctx)          # [N,D]
    var ff_down = _lora_apply_device(ff_down_base^, ff_act, lo.ff_down, N, ctx)
    var final = residual_gate(attn_res, mv.gate2[], ff_down, ctx)
    return _QwenStreamPostDevice(
        TArc(final^), TArc(attn_res^), TArc(ln2^), TArc(ff_in^),
        TArc(ff_up^), TArc(ff_act^),
    )


def qwen_double_block_lora_forward_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_x: TArc, txt_x: TArc,
    w: DoubleBlockWeights, img_mod: QwenModVecsDevice, txt_mod: QwenModVecsDevice,
    lora: QwenDoubleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> QwenDoubleBlockDeviceForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)
    var zeros_t = _t_bf16(_zeros(D), [D], ctx)

    var ip = _qwen_stream_pre_lora_device[H, Dh](
        img_x, w.img, img_mod, lora.img, N_IMG, D, eps, ones_t, zeros_t, ctx)
    var tp = _qwen_stream_pre_lora_device[H, Dh](
        txt_x, w.txt, txt_mod, lora.txt, N_TXT, D, eps, ones_t, zeros_t, ctx)

    # JOINT concat TXT FIRST then IMG (host block :1571).
    var q = concat(1, ctx, tp.q_rms[], ip.q_rms[])   # [1,S,H,Dh]
    var k = concat(1, ctx, tp.k_rms[], ip.k_rms[])
    var v = concat(1, ctx, tp.v[], ip.v[])

    # host passes cos/sin AS GIVEN into rope (no bf16 cast — unlike chroma).
    var q_rope = rope_interleaved(q, cos, sin, ctx)
    var k_rope = rope_interleaved(k, cos, sin, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = TArc(reshape_owned(txt_att_4d^, [N_TXT, D]))
    var img_att = TArc(reshape_owned(img_att_4d^, [N_IMG, D]))

    var ipost = _qwen_stream_post_lora_device(
        img_x, img_att, w.img, img_mod, lora.img, N_IMG, D, F, eps, ones_t, zeros_t, ctx)
    var tpost = _qwen_stream_post_lora_device(
        txt_x, txt_att, w.txt, txt_mod, lora.txt, N_TXT, D, F, eps, ones_t, zeros_t, ctx)

    var img_saved = QwenStreamSavedDevice(
        img_x.copy(), ip.ln1.copy(), ip.norm.copy(), ip.q_pre.copy(), ip.k_pre.copy(),
        img_att.copy(), ipost.attn_res.copy(), ipost.ln2.copy(), ipost.ff_in.copy(),
        ipost.ff_up.copy(), ipost.ff_act.copy(),
    )
    var txt_saved = QwenStreamSavedDevice(
        txt_x.copy(), tp.ln1.copy(), tp.norm.copy(), tp.q_pre.copy(), tp.k_pre.copy(),
        txt_att.copy(), tpost.attn_res.copy(), tpost.ln2.copy(), tpost.ff_in.copy(),
        tpost.ff_up.copy(), tpost.ff_act.copy(),
    )
    var saved = QwenDoubleBlockSavedDevice(
        img_saved^, txt_saved^, TArc(q_rope^), TArc(k_rope^), TArc(v^),
    )
    return QwenDoubleBlockDeviceForward(ipost.out.copy(), tpost.out.copy(), saved^)


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLE block — device backward (LoRA d_a/d_b + d_x ONLY; base/mod grads dropped)
# ═══════════════════════════════════════════════════════════════════════════
@fieldwise_init
struct _QwenStreamPostBackDevice(Copyable, Movable):
    var d_x: TArc      # [N,D] bf16 (stream input grad via gate1 residual)
    var d_att: TArc    # [N,D] bf16 (grad into joint-attention slice)


def _qwen_stream_post_backward_lora_device(
    d_out: TArc, x: TArc, att: TArc,
    w: StreamWeights, mv: QwenModVecsDevice, sv: QwenStreamSavedDevice, lo: QwenStreamLoraDevice,
    N: Int, D: Int, F: Int, eps: Float32, ones: Tensor,
    mut d_a_slots: List[Optional[TArc]], mut d_b_slots: List[Optional[TArc]],
    ctx: DeviceContext,
    compute_gate_grad: Bool = True,
) raises -> _QwenStreamPostBackDevice:
    # `compute_gate_grad` mirrors the chroma b2 template: the frozen gate grad is
    # DISCARDED by qwen either way, but gate_residual_backward's d_g reduction is
    # unsupported for packed [B, D] gates (row-stacked b2) — the b2 caller passes
    # False; b1 keeps the default True (byte-identical to the landed b1 path).
    # recompute ff_down WITH LoRA(ff_down) so gate_residual_backward y matches fwd.
    var bd = Optional[Tensor](w.bdn[].clone(ctx))
    var ff_down_base = linear(sv.ff_act[], w.wdn[], bd, ctx)
    var ff_down_y = _lora_apply_device(ff_down_base^, sv.ff_act[], lo.ff_down, N, ctx)
    # host: gate_residual_backward(d_out bf16, attn_res, bf16 gate2, ff_down)
    var grg2 = gate_residual_backward(
        d_out[], sv.attn_res[], mv.gate2[], ff_down_y, ctx, compute_gate_grad=compute_gate_grad)

    var pdn_dx = _proj_bwd_with_lora_device(
        _bf16(grg2.d_y, ctx), sv.ff_act[], w.wdn[], lo.ff_down, Q_DN, N, F, D,
        d_a_slots, d_b_slots, ctx,
    )                                                                       # [N,F] bf16
    var d_ff_up = gelu_backward(pdn_dx, sv.ff_up[], ctx)                    # bf16

    var pup_dx = _proj_bwd_with_lora_device(
        d_ff_up, sv.ff_in[], w.wup[], lo.ff_up, Q_UP, N, D, F,
        d_a_slots, d_b_slots, ctx,
    )                                                                       # [N,D] bf16

    var mb2 = modulate_backward(pup_dx, sv.ln2[], mv.scale2[], ctx, False)
    var lnb2_dx = layer_norm_backward_dx(mb2.d_x, sv.attn_res[], ones, eps, ctx)
    # host: add(grg2.d_x, d_ln2) — both bf16, F32 math, bf16 store.
    var d_attn_res_total = add(grg2.d_x, lnb2_dx, ctx)                      # bf16

    # attn_res = residual_gate(x, gate1, out): recompute out WITH LoRA(out).
    var bo = Optional[Tensor](w.bout[].clone(ctx))
    var out_base = linear(att[], w.wout[], bo, ctx)
    var out_y = _lora_apply_device(out_base^, att[], lo.out, N, ctx)
    var grg1 = gate_residual_backward(
        d_attn_res_total, x[], mv.gate1[], out_y, ctx, compute_gate_grad=compute_gate_grad)

    var d_att = _proj_bwd_with_lora_device(
        _bf16(grg1.d_y, ctx), att[], w.wout[], lo.out, Q_OUT, N, D, D,
        d_a_slots, d_b_slots, ctx,
    )                                                                       # [N,D] bf16
    return _QwenStreamPostBackDevice(TArc(grg1.d_x.clone(ctx)), TArc(d_att^))  # both bf16


def _qwen_stream_pre_backward_lora_device[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    w: StreamWeights, mv: QwenModVecsDevice, sv: QwenStreamSavedDevice, lo: QwenStreamLoraDevice,
    N: Int, D: Int, eps: Float32, ones: Tensor,
    mut d_a_slots: List[Optional[TArc]], mut d_b_slots: List[Optional[TArc]],
    ctx: DeviceContext,
) raises -> Tensor:
    var dq_b = _bf16(d_q_rms, ctx)
    var dk_b = _bf16(d_k_rms, ctx)
    var dv_b = _bf16(d_v, ctx)
    var rb_q_dx = rms_norm_backward_dx(dq_b, sv.q_pre[], w.q_norm[], eps, ctx)   # [1,N,H,Dh]
    var rb_k_dx = rms_norm_backward_dx(dk_b, sv.k_pre[], w.k_norm[], eps, ctx)
    reshape_in_place(rb_q_dx, [N, D])
    reshape_in_place(rb_k_dx, [N, D])
    var d_v_flat = reshape(dv_b, [N, D], ctx)

    # THREE separate frozen base linears (qwen delta). Host fold order
    # (qwenimage_block.mojo:1817-1861, all F32 host, ONE bf16 round at _t):
    #   d_nq = dxq (+ lora_q.d_x); d_nk = dxk (+ lora_k.d_x); d_nv = dxv
    #   (+ lora_v.d_x); d_normed = (d_nq + d_nk) + d_nv.
    var dxq = linear_backward_dx(rb_q_dx, w.wq[], N, D, D, ctx)   # [N,D] bf16
    var dxk = linear_backward_dx(rb_k_dx, w.wk[], N, D, D, ctx)
    var dxv = linear_backward_dx(d_v_flat, w.wv[], N, D, D, ctx)
    var d_nq = _f32(dxq, ctx)
    var d_nk = _f32(dxk, ctx)
    var d_nv = _f32(dxv, ctx)
    if lo.q:
        var lg = _lora_bwd_device(rb_q_dx, sv.norm[], lo.q.value(), N, ctx)
        d_a_slots[Q_SQ] = Optional[TArc](lg.d_a)
        d_b_slots[Q_SQ] = Optional[TArc](lg.d_b)
        d_nq = add(d_nq, _f32(lg.d_x_lo[], ctx), ctx)
    if lo.k:
        var lg = _lora_bwd_device(rb_k_dx, sv.norm[], lo.k.value(), N, ctx)
        d_a_slots[Q_SK] = Optional[TArc](lg.d_a)
        d_b_slots[Q_SK] = Optional[TArc](lg.d_b)
        d_nk = add(d_nk, _f32(lg.d_x_lo[], ctx), ctx)
    if lo.v:
        var lg = _lora_bwd_device(d_v_flat, sv.norm[], lo.v.value(), N, ctx)
        d_a_slots[Q_SV] = Optional[TArc](lg.d_a)
        d_b_slots[Q_SV] = Optional[TArc](lg.d_b)
        d_nv = add(d_nv, _f32(lg.d_x_lo[], ctx), ctx)
    var d_norm_f32 = add(add(d_nq, d_nk, ctx), d_nv, ctx)

    var mb1 = modulate_backward(_bf16(d_norm_f32, ctx), sv.ln1[], mv.scale1[], ctx, False)
    var lnb1_dx = layer_norm_backward_dx(mb1.d_x, sv.x[], ones, eps, ctx)   # bf16
    return lnb1_dx^


def qwen_double_block_lora_backward_device[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: QwenModVecsDevice, txt_mod: QwenModVecsDevice,
    lora: QwenDoubleBlockLoraDevice,
    saved: QwenDoubleBlockSavedDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> QwenDoubleBlockLoraDeviceGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)

    var ia = List[Optional[TArc]]()
    var ib = List[Optional[TArc]]()
    var ta = List[Optional[TArc]]()
    var tb = List[Optional[TArc]]()
    for _ in range(QWEN_STREAM_SLOTS):
        ia.append(Optional[TArc](None)); ib.append(Optional[TArc](None))
        ta.append(Optional[TArc](None)); tb.append(Optional[TArc](None))

    var ipb = _qwen_stream_post_backward_lora_device(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, saved.img, lora.img,
        N_IMG, D, F, eps, ones_t, ia, ib, ctx,
    )
    var tpb = _qwen_stream_post_backward_lora_device(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT, D, F, eps, ones_t, ta, tb, ctx,
    )

    # join per-stream d_att into joint d_att (txt FIRST); [N,D] -> [1,N,H,Dh].
    var d_tatt_4d = reshape(tpb.d_att[], [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = reshape(ipb.d_att[], [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)   # [1,S,H,Dh] bf16

    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v_joint[], d_att_joint, scale, ctx)

    # host driver feeds sb.d_q/d_k/d_v DIRECTLY (bf16) — no F32 detour.
    var d_q_joint = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_joint = rope_backward(sb.d_k, cos, sin, True, ctx)

    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(sb.d_v, N_TXT, N_IMG, 1, ctx)

    var iprb_dx = _qwen_stream_pre_backward_lora_device[H, Dh](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, saved.img, lora.img,
        N_IMG, D, eps, ones_t, ia, ib, ctx,
    )
    var tprb_dx = _qwen_stream_pre_backward_lora_device[H, Dh](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT, D, eps, ones_t, ta, tb, ctx,
    )

    # host: _add_lists of the two F32 host lists (exact upcasts of bf16) —
    # F32 result, NO re-round. Device: F32 add of the bf16 tensors.
    var d_img_x = add(_f32(ipb.d_x[], ctx), _f32(iprb_dx, ctx), ctx)   # F32
    var d_txt_x = add(_f32(tpb.d_x[], ctx), _f32(tprb_dx, ctx), ctx)

    var img_g = QwenStreamLoraDeviceGrads(TArc(d_img_x^), ia^, ib^)
    var txt_g = QwenStreamLoraDeviceGrads(TArc(d_txt_x^), ta^, tb^)
    return QwenDoubleBlockLoraDeviceGrads(img_g^, txt_g^)


# ═══════════════════════════════════════════════════════════════════════════
# TRUE BATCH-2 (row-stacked) — device fwd/bwd for the qwen DOUBLE block.
#
# DESIGN (mirrors chroma_block_device.mojo's b2). Two samples are ROW-STACKED on
# the token dim: sample0's rows first, then sample1's (img [2*N_IMG, D], txt
# [2*N_TXT, D]). Per-sample mod vecs are packed [2, D] (sample0 on row 0,
# sample1 on row 1) because the two samples' sigmas -> silu(temb) -> mods differ
# (per-sample adaLN). The shared modulate / residual_gate / modulate_backward /
# gate_residual_backward ops split the 2N rows into two contiguous N-ranges.
# LoRA/linear GEMMs run at M=2N, so the shared adapters' d_a/d_b SUM over both
# samples in-GEMM = the batch gradient (mean once the caller scales each sample's
# d_out by 0.5).
#
# ATTENTION never mixes samples. qwen uses a UNIFORM comptime S per sample (S =
# N_TXT + N_IMG); the per-sample joint q/k/v are assembled (TXT FIRST then IMG,
# matching the b1 join at :411), roped, and flashed independently via TWO reused
# sdpa_nomask[1, S, H, Dh] calls (each bit-identical to the b1 attention) — NOT a
# B=2 flash (qwen's device path is math sdpa; flash is NOT wired).
#
# The forward/backward stream helpers (_qwen_stream_pre_lora_device /
# _qwen_stream_post_lora_device / _qwen_stream_pre_backward_lora_device) are
# REUSED UNCHANGED at 2N with [2, D] packed mods. Only the post-backward helper
# needs compute_gate_grad=False (packed [B, D] gate -> d_g reduction unsupported;
# the frozen gate grad is discarded anyway).
# ═══════════════════════════════════════════════════════════════════════════

def _qwen_stack2_bf16(a: List[Float32], b: List[Float32], D: Int, ctx: DeviceContext) raises -> Tensor:
    var v = List[Float32]()
    for i in range(D):
        v.append(a[i])
    for i in range(D):
        v.append(b[i])
    return Tensor.from_host(v^, [2, D], STDtype.BF16, ctx)


# pack two host mod-vec sets into [2, D] bf16 device tensors (row 0 = sample0).
# qwen uses bf16 gates in BOTH forward and backward -> no F32 gate copy needed
# (unlike chroma's ModVecsDevice which carried extra F32 gate tensors).
def qwen_modvecs_pack_b2(mv0: ModVecs, mv1: ModVecs, D: Int, ctx: DeviceContext) raises -> QwenModVecsDevice:
    return QwenModVecsDevice(
        TArc(_qwen_stack2_bf16(mv0.shift1, mv1.shift1, D, ctx)),
        TArc(_qwen_stack2_bf16(mv0.scale1, mv1.scale1, D, ctx)),
        TArc(_qwen_stack2_bf16(mv0.gate1, mv1.gate1, D, ctx)),
        TArc(_qwen_stack2_bf16(mv0.shift2, mv1.shift2, D, ctx)),
        TArc(_qwen_stack2_bf16(mv0.scale2, mv1.scale2, D, ctx)),
        TArc(_qwen_stack2_bf16(mv0.gate2, mv1.gate2, D, ctx)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLE block — device forward (row-stacked b2). img_x [2*N_IMG, D], txt_x
# [2*N_TXT, D]; img_mod/txt_mod packed [2, D]. Saves per-sample joint tapes
# concatenated on the seq dim ([1, 2S, H, Dh]) for the backward.
# ═══════════════════════════════════════════════════════════════════════════
def qwen_double_block_lora_forward_device_b2[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img_x: TArc, txt_x: TArc,
    w: DoubleBlockWeights, img_mod: QwenModVecsDevice, txt_mod: QwenModVecsDevice,
    lora: QwenDoubleBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> QwenDoubleBlockDeviceForward:
    comptime N_IMG2 = 2 * N_IMG
    comptime N_TXT2 = 2 * N_TXT
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)
    var zeros_t = _t_bf16(_zeros(D), [D], ctx)

    var ip = _qwen_stream_pre_lora_device[H, Dh](
        img_x, w.img, img_mod, lora.img, N_IMG2, D, eps, ones_t, zeros_t, ctx)
    var tp = _qwen_stream_pre_lora_device[H, Dh](
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

    var q0 = concat(1, ctx, tq0, iq0)   # [1, S, H, Dh]  txt FIRST
    var q1 = concat(1, ctx, tq1, iq1)
    var k0 = concat(1, ctx, tk0, ik0)
    var k1 = concat(1, ctx, tk1, ik1)
    var v0 = concat(1, ctx, tv0, iv0)
    var v1 = concat(1, ctx, tv1, iv1)

    # host passes cos/sin AS GIVEN into rope (no bf16 cast — matches b1 :416).
    var q0r = rope_interleaved(q0, cos, sin, ctx)
    var q1r = rope_interleaved(q1, cos, sin, ctx)
    var k0r = rope_interleaved(k0, cos, sin, ctx)
    var k1r = rope_interleaved(k1, cos, sin, ctx)

    var att0 = sdpa_nomask[1, S, H, Dh](q0r, k0r, v0, scale, ctx)   # [1, S, H, Dh]
    var att1 = sdpa_nomask[1, S, H, Dh](q1r, k1r, v1, scale, ctx)

    var ta0 = slice(att0, 1, 0, N_TXT, ctx)
    var ia0 = slice(att0, 1, N_TXT, N_IMG, ctx)
    var ta1 = slice(att1, 1, 0, N_TXT, ctx)
    var ia1 = slice(att1, 1, N_TXT, N_IMG, ctx)
    var txt_att_4d = concat(1, ctx, ta0, ta1)   # [1, N_TXT2, H, Dh]
    var img_att_4d = concat(1, ctx, ia0, ia1)   # [1, N_IMG2, H, Dh]
    var txt_att = TArc(reshape_owned(txt_att_4d^, [N_TXT2, D]))
    var img_att = TArc(reshape_owned(img_att_4d^, [N_IMG2, D]))

    var ipost = _qwen_stream_post_lora_device(
        img_x, img_att, w.img, img_mod, lora.img, N_IMG2, D, F, eps, ones_t, zeros_t, ctx)
    var tpost = _qwen_stream_post_lora_device(
        txt_x, txt_att, w.txt, txt_mod, lora.txt, N_TXT2, D, F, eps, ones_t, zeros_t, ctx)

    var q_rope_st = concat(1, ctx, q0r, q1r)   # [1, 2S, H, Dh]
    var k_rope_st = concat(1, ctx, k0r, k1r)
    var v_st = concat(1, ctx, v0, v1)

    var img_saved = QwenStreamSavedDevice(
        img_x.copy(), ip.ln1.copy(), ip.norm.copy(), ip.q_pre.copy(), ip.k_pre.copy(),
        img_att.copy(), ipost.attn_res.copy(), ipost.ln2.copy(), ipost.ff_in.copy(),
        ipost.ff_up.copy(), ipost.ff_act.copy(),
    )
    var txt_saved = QwenStreamSavedDevice(
        txt_x.copy(), tp.ln1.copy(), tp.norm.copy(), tp.q_pre.copy(), tp.k_pre.copy(),
        txt_att.copy(), tpost.attn_res.copy(), tpost.ln2.copy(), tpost.ff_in.copy(),
        tpost.ff_up.copy(), tpost.ff_act.copy(),
    )
    var saved = QwenDoubleBlockSavedDevice(
        img_saved^, txt_saved^, TArc(q_rope_st^), TArc(k_rope_st^), TArc(v_st^),
    )
    return QwenDoubleBlockDeviceForward(ipost.out.copy(), tpost.out.copy(), saved^)


# ═══════════════════════════════════════════════════════════════════════════
# DOUBLE block — device backward (row-stacked b2). d_io_t [2*N_IMG, D], d_to_t
# [2*N_TXT, D]. LoRA d_a/d_b sum over both samples (M=2N GEMM). d_x per stream.
# ═══════════════════════════════════════════════════════════════════════════
def qwen_double_block_lora_backward_device_b2[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_io_t: TArc, d_to_t: TArc,
    w: DoubleBlockWeights, img_mod: QwenModVecsDevice, txt_mod: QwenModVecsDevice,
    lora: QwenDoubleBlockLoraDevice,
    saved: QwenDoubleBlockSavedDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> QwenDoubleBlockLoraDeviceGrads:
    comptime N_IMG2 = 2 * N_IMG
    comptime N_TXT2 = 2 * N_TXT
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t_bf16(_ones(D), [D], ctx)

    var ia = List[Optional[TArc]]()
    var ib = List[Optional[TArc]]()
    var ta = List[Optional[TArc]]()
    var tb = List[Optional[TArc]]()
    for _ in range(QWEN_STREAM_SLOTS):
        ia.append(Optional[TArc](None)); ib.append(Optional[TArc](None))
        ta.append(Optional[TArc](None)); tb.append(Optional[TArc](None))

    # packed [B, D] gates -> compute_gate_grad=False (frozen grad discarded).
    var ipb = _qwen_stream_post_backward_lora_device(
        d_io_t, saved.img.x, saved.img.att, w.img, img_mod, saved.img, lora.img,
        N_IMG2, D, F, eps, ones_t, ia, ib, ctx, False,
    )
    var tpb = _qwen_stream_post_backward_lora_device(
        d_to_t, saved.txt.x, saved.txt.att, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT2, D, F, eps, ones_t, ta, tb, ctx, False,
    )

    # per-stream d_att [2N, D] -> [1, 2N, H, Dh] -> split per sample -> per-sample
    # joint d_att (txt FIRST, matching the forward join).
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

    var sb0 = sdpa_backward[1, S, H, Dh](q0r, k0r, v0, d_att0, scale, ctx)
    var sb1 = sdpa_backward[1, S, H, Dh](q1r, k1r, v1, d_att1, scale, ctx)

    # rope backward per sample (qwen feeds sb.d_q/d_k directly, bf16 — matches b1).
    var d_q0 = rope_backward(sb0.d_q, cos, sin, True, ctx)
    var d_q1 = rope_backward(sb1.d_q, cos, sin, True, ctx)
    var d_k0 = rope_backward(sb0.d_k, cos, sin, True, ctx)
    var d_k1 = rope_backward(sb1.d_k, cos, sin, True, ctx)

    # cat_backward per sample: split joint S -> txt N_TXT (d_0), img N_IMG (d_1).
    var cq0 = cat_backward(d_q0, N_TXT, N_IMG, 1, ctx)
    var cq1 = cat_backward(d_q1, N_TXT, N_IMG, 1, ctx)
    var ck0 = cat_backward(d_k0, N_TXT, N_IMG, 1, ctx)
    var ck1 = cat_backward(d_k1, N_TXT, N_IMG, 1, ctx)
    var cv0 = cat_backward(sb0.d_v, N_TXT, N_IMG, 1, ctx)
    var cv1 = cat_backward(sb1.d_v, N_TXT, N_IMG, 1, ctx)

    # re-stack row-wise per stream: txt [1, N_TXT2], img [1, N_IMG2].
    var d_q_txt = concat(1, ctx, cq0.d_0, cq1.d_0)
    var d_q_img = concat(1, ctx, cq0.d_1, cq1.d_1)
    var d_k_txt = concat(1, ctx, ck0.d_0, ck1.d_0)
    var d_k_img = concat(1, ctx, ck0.d_1, ck1.d_1)
    var d_v_txt = concat(1, ctx, cv0.d_0, cv1.d_0)
    var d_v_img = concat(1, ctx, cv0.d_1, cv1.d_1)

    var iprb_dx = _qwen_stream_pre_backward_lora_device[H, Dh](
        d_q_img, d_k_img, d_v_img, w.img, img_mod, saved.img, lora.img,
        N_IMG2, D, eps, ones_t, ia, ib, ctx,
    )
    var tprb_dx = _qwen_stream_pre_backward_lora_device[H, Dh](
        d_q_txt, d_k_txt, d_v_txt, w.txt, txt_mod, saved.txt, lora.txt,
        N_TXT2, D, eps, ones_t, ta, tb, ctx,
    )

    var d_img_x = add(_f32(ipb.d_x[], ctx), _f32(iprb_dx, ctx), ctx)   # F32
    var d_txt_x = add(_f32(tpb.d_x[], ctx), _f32(tprb_dx, ctx), ctx)

    var img_g = QwenStreamLoraDeviceGrads(TArc(d_img_x^), ia^, ib^)
    var txt_g = QwenStreamLoraDeviceGrads(TArc(d_txt_x^), ta^, tb^)
    return QwenDoubleBlockLoraDeviceGrads(img_g^, txt_g^)
