# serenitymojo/models/wan22/wan22_block.mojo
#
# Wan2.2 WanAttentionBlock: forward (saving activations) + hand-chained backward
# (training) + LoRA variants. Mirrors the inference forward in
# models/dit/wan22_dit.mojo::wan22_block_forward (lines 175-254), which read
# /home/alex/Wan2.2/wan/modules/model.py WanAttentionBlock (183-259) line-by-line.
# Recipe + LoRA targets cited in models/wan22/config.mojo (EDv2 wan22.rs).
#
# WHAT MAKES THIS A NEW COMPUTE (not a copy of the double-stream blocks):
#   - SINGLE image stream (no two-stream join). The x [S,dim] stream carries the
#     image tokens; text enters ONLY through cross-attn as context [TXT,dim].
#   - PER-TOKEN AdaLN: scale/shift/gate are [1,S,dim] tensors (one per token), NOT
#     per-channel [D] vectors. So the modulate/gate-residual backward param grads
#     are ELEMENTWISE (d_scale = go*x, d_shift = go, d_gate = go*y) — NOT the
#     cross-row column reductions the [D]-vector modulate_backward/
#     gate_residual_backward kernels do. Built here as wan_modulate_backward /
#     wan_gate_residual_backward (pure elementwise composition of mul/add — no
#     new GPU kernel, gated by its own elementwise identity).
#   - SELF-ATTN: LN-no-affine -> per-token mod_pre -> q/k/v biased linears ->
#     qk-rms-norm -> 3-axis INTERLEAVED RoPE -> SQUARE sdpa (S×S) -> o linear ->
#     PER-TOKEN gated residual.
#   - CROSS-ATTN (to text, distinct q-len S vs kv-len TXT): affine LN (norm3) ->
#     q biased linear + q-rms ; k/v biased linears on context + k-rms -> RECT
#     sdpa (sdpa_backward_rect[Sq=S,Skv=TXT]) -> o linear -> UNGATED residual
#     (plain add). This is the cross-attn-distinct-length backward path.
#   - FFN: LN-no-affine -> per-token mod_pre -> ffn.0 biased linear -> GELU(tanh)
#     -> ffn.2 biased linear -> PER-TOKEN gated residual.
#
# BACKWARD ARMS REUSED (all pre-built + gated): linear_backward (biased linears,
# gives d_w/d_b/d_x), layer_norm_backward (affine norm3), layer_norm_backward_dx
# (no-affine LN), rms_norm_backward (qk-norm), gelu_backward, sdpa_backward
# (self), sdpa_backward_rect (cross, Sq!=Skv), rope_backward(interleaved),
# cat/slice/reshape backward via tensor ops, mul/add (per-token AdaLN).
#
# ACTIVATION CARRIER: saved activations are stored as host List[BFloat16] (half
# the bytes of F32). flame-core (Rust reference) is BF16 in/out. Weights are
# uploaded to device as BF16. Grads and modulation vectors stay F32.
#
# API boundary: x/context enter + x_out + every grad leave as host List[Float32]
# (so the stack + parity gate use it unchanged). Saved activations host BF16.
#
# Mojo 1.0.0b1, NVIDIA GPU. `def` not `fn`; Tensor move-only (TArc carriers);
# biased linear = linear(x, w, Optional[Tensor](b), ctx). Block runs BF16 weights
# + BF16 saved activations (matches flame-core bf16 contract).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

# ── forward ops (GPU) ────────────────────────────────────────────────────────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.softmax import softmax_lastdim
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, reshape_in_place, slice, concat, add, sub, mul,
    add_scalar, mul_scalar, permute, transpose,
)

# ── backward arms (GPU; all pre-built + gated) ───────────────────────────────
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import (
    rms_norm_backward, RmsNormBackward,
    layer_norm_backward, layer_norm_backward_dx, LayerNormBackward,
)
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.attention_backward import (
    sdpa_backward, sdpa_backward_rect, SdpaGrads,
)
from serenitymojo.ops.rope_struct_backward import rope_backward


comptime TArc = ArcPointer[Tensor]


# ── host helpers ─────────────────────────────────────────────────────────────
def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


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


# BF16 device upload (the native bf16 compute path: weights, activations, mod
# vectors). The public API still hands us List[Float32]; we cast to bf16 on
# upload so every Linear runs bf16·bf16 with F32-accumulate inside the GEMM —
# matching flame-core's bf16 contract.
def _t16(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


def _ta16(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, shape^, STDtype.BF16, ctx))


# Re-upload a BF16-stored activation (saved by to_host_bf16) to a BF16 device
# tensor (NO widen to F32) so backward matmuls also run native bf16.
def _tbf16(vals: List[BFloat16], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host_bf16(vals, shape^, ctx)


def _clone_t(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# ── PER-TOKEN AdaLN forward primitives ───────────────────────────────────────
# mod_pre: o = LN_no_affine(x) * (1 + scale) + shift   (scale/shift [S,dim] tensors).
# Returns (o, ln) where ln = LN_no_affine(x) is saved for backward.
struct ModPre(Copyable, Movable):
    var o: TArc
    var ln: TArc

    def __init__(out self, var o: TArc, var ln: TArc):
        self.o = o^
        self.ln = ln^


def wan_mod_pre(
    x: Tensor, scale: Tensor, shift: Tensor, ones: Tensor, zeros: Tensor,
    eps: Float32, ctx: DeviceContext,
) raises -> ModPre:
    var ln = layer_norm(x, ones, zeros, eps, ctx)
    var sc1 = add_scalar(scale, 1.0, ctx)        # (1 + scale)
    var prod = mul(ln, sc1, ctx)
    var o = add(prod, shift, ctx)
    return ModPre(TArc(o^), TArc(ln^))


# PER-TOKEN gated residual forward: o = x + gate * y   (gate [S,dim]).
def wan_gated_residual(x: Tensor, y: Tensor, gate: Tensor, ctx: DeviceContext) raises -> Tensor:
    var gy = mul(y, gate, ctx)
    return add(x, gy, ctx)


# ── PER-TOKEN AdaLN backward primitives (elementwise; NO column reduction) ────
# modulate: o = (1+scale)*x + shift  (all [S,dim]). Given go = dL/do:
#   d_ln    = go * (1 + scale)
#   d_scale = go * ln            (per-token, elementwise — NOT summed over rows)
#   d_shift = go                 (per-token)
struct WanModulateBack(Movable):
    var d_ln: Tensor
    var d_scale: Tensor
    var d_shift: Tensor

    def __init__(out self, var d_ln: Tensor, var d_scale: Tensor, var d_shift: Tensor):
        self.d_ln = d_ln^
        self.d_scale = d_scale^
        self.d_shift = d_shift^


def wan_modulate_backward(
    go: Tensor, ln: Tensor, scale: Tensor, ctx: DeviceContext
) raises -> WanModulateBack:
    var sc1 = add_scalar(scale, 1.0, ctx)
    var d_ln = mul(go, sc1, ctx)
    var d_scale = mul(go, ln, ctx)
    var d_shift = _clone_t(go, ctx)
    return WanModulateBack(d_ln^, d_scale^, d_shift^)


# gated residual: o = x + gate*y  (all [S,dim]). Given go = dL/do:
#   d_x    = go
#   d_y    = go * gate
#   d_gate = go * y              (per-token, elementwise)
struct WanGateBack(Movable):
    var d_x: Tensor
    var d_y: Tensor
    var d_gate: Tensor

    def __init__(out self, var d_x: Tensor, var d_y: Tensor, var d_gate: Tensor):
        self.d_x = d_x^
        self.d_y = d_y^
        self.d_gate = d_gate^


def wan_gate_residual_backward(
    go: Tensor, y: Tensor, gate: Tensor, ctx: DeviceContext
) raises -> WanGateBack:
    var d_x = _clone_t(go, ctx)
    var d_y = mul(go, gate, ctx)
    var d_gate = mul(go, y, ctx)
    return WanGateBack(d_x^, d_y^, d_gate^)


# ── per-token AdaLN modulation vectors (each [S,dim] FLAT host list, S*dim) ───
# order matches model.py chunk(6): shift_sa, scale_sa, gate_sa, shift_ffn,
# scale_ffn, gate_ffn.
struct WanModVecs(Copyable, Movable):
    var shift_sa: List[Float32]
    var scale_sa: List[Float32]
    var gate_sa: List[Float32]
    var shift_ffn: List[Float32]
    var scale_ffn: List[Float32]
    var gate_ffn: List[Float32]

    def __init__(
        out self,
        var shift_sa: List[Float32], var scale_sa: List[Float32], var gate_sa: List[Float32],
        var shift_ffn: List[Float32], var scale_ffn: List[Float32], var gate_ffn: List[Float32],
    ):
        self.shift_sa = shift_sa^
        self.scale_sa = scale_sa^
        self.gate_sa = gate_sa^
        self.shift_ffn = shift_ffn^
        self.scale_ffn = scale_ffn^
        self.gate_ffn = gate_ffn^


# ── block trainable weights (DEVICE-RESIDENT TArc, uploaded ONCE) ─────────────
#   self_attn q/k/v/o: [dim,dim] + bias [dim] ; norm_q/norm_k [head_dim]
#   cross_attn q/k/v/o: [dim,dim] + bias [dim] ; norm_q/norm_k [head_dim]
#   norm3 weight/bias [dim]  (affine LN)
#   ffn.0: [ffn,dim] + bias [ffn] ; ffn.2: [dim,ffn] + bias [dim]
struct WanBlockWeights(Copyable, Movable):
    # self attention
    var sa_wq: TArc
    var sa_wk: TArc
    var sa_wv: TArc
    var sa_wo: TArc
    var sa_bq: TArc
    var sa_bk: TArc
    var sa_bv: TArc
    var sa_bo: TArc
    var sa_qn: TArc
    var sa_kn: TArc
    # cross attention
    var ca_wq: TArc
    var ca_wk: TArc
    var ca_wv: TArc
    var ca_wo: TArc
    var ca_bq: TArc
    var ca_bk: TArc
    var ca_bv: TArc
    var ca_bo: TArc
    var ca_qn: TArc
    var ca_kn: TArc
    # norm3 (affine LN before cross-attn)
    var n3_w: TArc
    var n3_b: TArc
    # ffn
    var ffn0_w: TArc
    var ffn0_b: TArc
    var ffn2_w: TArc
    var ffn2_b: TArc

    def __init__(
        out self,
        var sa_wq: List[Float32], var sa_wk: List[Float32], var sa_wv: List[Float32], var sa_wo: List[Float32],
        var sa_bq: List[Float32], var sa_bk: List[Float32], var sa_bv: List[Float32], var sa_bo: List[Float32],
        var sa_qn: List[Float32], var sa_kn: List[Float32],
        var ca_wq: List[Float32], var ca_wk: List[Float32], var ca_wv: List[Float32], var ca_wo: List[Float32],
        var ca_bq: List[Float32], var ca_bk: List[Float32], var ca_bv: List[Float32], var ca_bo: List[Float32],
        var ca_qn: List[Float32], var ca_kn: List[Float32],
        var n3_w: List[Float32], var n3_b: List[Float32],
        var ffn0_w: List[Float32], var ffn0_b: List[Float32],
        var ffn2_w: List[Float32], var ffn2_b: List[Float32],
        dim: Int, ffn: Int, hd: Int, ctx: DeviceContext,
    ) raises:
        self.sa_wq = TArc(Tensor.from_host(sa_wq^, [dim, dim], STDtype.BF16, ctx))
        self.sa_wk = TArc(Tensor.from_host(sa_wk^, [dim, dim], STDtype.BF16, ctx))
        self.sa_wv = TArc(Tensor.from_host(sa_wv^, [dim, dim], STDtype.BF16, ctx))
        self.sa_wo = TArc(Tensor.from_host(sa_wo^, [dim, dim], STDtype.BF16, ctx))
        self.sa_bq = TArc(Tensor.from_host(sa_bq^, [dim], STDtype.BF16, ctx))
        self.sa_bk = TArc(Tensor.from_host(sa_bk^, [dim], STDtype.BF16, ctx))
        self.sa_bv = TArc(Tensor.from_host(sa_bv^, [dim], STDtype.BF16, ctx))
        self.sa_bo = TArc(Tensor.from_host(sa_bo^, [dim], STDtype.BF16, ctx))
        self.sa_qn = TArc(Tensor.from_host(sa_qn^, [hd], STDtype.BF16, ctx))
        self.sa_kn = TArc(Tensor.from_host(sa_kn^, [hd], STDtype.BF16, ctx))
        self.ca_wq = TArc(Tensor.from_host(ca_wq^, [dim, dim], STDtype.BF16, ctx))
        self.ca_wk = TArc(Tensor.from_host(ca_wk^, [dim, dim], STDtype.BF16, ctx))
        self.ca_wv = TArc(Tensor.from_host(ca_wv^, [dim, dim], STDtype.BF16, ctx))
        self.ca_wo = TArc(Tensor.from_host(ca_wo^, [dim, dim], STDtype.BF16, ctx))
        self.ca_bq = TArc(Tensor.from_host(ca_bq^, [dim], STDtype.BF16, ctx))
        self.ca_bk = TArc(Tensor.from_host(ca_bk^, [dim], STDtype.BF16, ctx))
        self.ca_bv = TArc(Tensor.from_host(ca_bv^, [dim], STDtype.BF16, ctx))
        self.ca_bo = TArc(Tensor.from_host(ca_bo^, [dim], STDtype.BF16, ctx))
        self.ca_qn = TArc(Tensor.from_host(ca_qn^, [hd], STDtype.BF16, ctx))
        self.ca_kn = TArc(Tensor.from_host(ca_kn^, [hd], STDtype.BF16, ctx))
        self.n3_w = TArc(Tensor.from_host(n3_w^, [dim], STDtype.BF16, ctx))
        self.n3_b = TArc(Tensor.from_host(n3_b^, [dim], STDtype.BF16, ctx))
        self.ffn0_w = TArc(Tensor.from_host(ffn0_w^, [ffn, dim], STDtype.BF16, ctx))
        self.ffn0_b = TArc(Tensor.from_host(ffn0_b^, [ffn], STDtype.BF16, ctx))
        self.ffn2_w = TArc(Tensor.from_host(ffn2_w^, [dim, ffn], STDtype.BF16, ctx))
        self.ffn2_b = TArc(Tensor.from_host(ffn2_b^, [dim], STDtype.BF16, ctx))


# ── saved activations (HOST-RESIDENT BF16 — half the bytes of F32 device TArc) ───
# flame-core (Rust reference) holds saved activations in BF16; we match that.
# Forward: tensor.to_host_bf16(ctx) → List[BFloat16] stored here.
# Backward: _tbf16(saved.FIELD.copy(), shape, ctx) → BF16 device Tensor.
struct WanSaved(Copyable, Movable):
    var x: List[BFloat16]         # [S,dim]   block input
    # self-attn
    var sa_ln: List[BFloat16]     # [S,dim]   LN_no_affine(x)
    var sa_in: List[BFloat16]     # [S,dim]   mod_pre output (self-attn input)
    var sa_q_pre: List[BFloat16]  # [1,S,H,Dh]  q pre-rms
    var sa_k_pre: List[BFloat16]  # [1,S,H,Dh]
    var sa_v: List[BFloat16]      # [1,S,H,Dh]
    var sa_q_rope: List[BFloat16] # [1,S,H,Dh]  rope(rms(q))
    var sa_k_rope: List[BFloat16] # [1,S,H,Dh]
    var sa_att: List[BFloat16]    # [S,dim]   attention out (flattened, pre-o-linear)
    var x_sa: List[BFloat16]      # [S,dim]   gated residual after self-attn
    # cross-attn
    var ca_n3: List[BFloat16]     # [S,dim]   affine LN(x_sa)
    var ca_q_pre: List[BFloat16]  # [1,S,H,Dh]
    var ca_k_pre: List[BFloat16]  # [1,TXT,H,Dh]
    var ca_q_rms: List[BFloat16]  # [1,S,H,Dh]
    var ca_k_rms: List[BFloat16]  # [1,TXT,H,Dh]
    var ca_v: List[BFloat16]      # [1,TXT,H,Dh]
    var ca_att: List[BFloat16]    # [S,dim]   cross-attn out (flattened, pre-o-linear)
    var context: List[BFloat16]   # [TXT,dim] saved context (cross k/v input)
    var x_ca: List[BFloat16]      # [S,dim]   x_sa + cross_out (ungated residual)
    # ffn
    var ffn_ln: List[BFloat16]    # [S,dim]   LN_no_affine(x_ca)
    var ffn_in: List[BFloat16]    # [S,dim]   mod_pre output
    var ffn_h: List[BFloat16]     # [S,ffn]   ffn.0 out (pre-gelu)
    var ffn_act: List[BFloat16]   # [S,ffn]   gelu(ffn_h)

    def __init__(
        out self, var x: List[BFloat16],
        var sa_ln: List[BFloat16], var sa_in: List[BFloat16],
        var sa_q_pre: List[BFloat16], var sa_k_pre: List[BFloat16],
        var sa_v: List[BFloat16], var sa_q_rope: List[BFloat16],
        var sa_k_rope: List[BFloat16], var sa_att: List[BFloat16],
        var x_sa: List[BFloat16],
        var ca_n3: List[BFloat16], var ca_q_pre: List[BFloat16],
        var ca_k_pre: List[BFloat16],
        var ca_q_rms: List[BFloat16], var ca_k_rms: List[BFloat16],
        var ca_v: List[BFloat16], var ca_att: List[BFloat16],
        var context: List[BFloat16], var x_ca: List[BFloat16],
        var ffn_ln: List[BFloat16], var ffn_in: List[BFloat16],
        var ffn_h: List[BFloat16], var ffn_act: List[BFloat16],
    ):
        self.x = x^
        self.sa_ln = sa_ln^
        self.sa_in = sa_in^
        self.sa_q_pre = sa_q_pre^
        self.sa_k_pre = sa_k_pre^
        self.sa_v = sa_v^
        self.sa_q_rope = sa_q_rope^
        self.sa_k_rope = sa_k_rope^
        self.sa_att = sa_att^
        self.x_sa = x_sa^
        self.ca_n3 = ca_n3^
        self.ca_q_pre = ca_q_pre^
        self.ca_k_pre = ca_k_pre^
        self.ca_q_rms = ca_q_rms^
        self.ca_k_rms = ca_k_rms^
        self.ca_v = ca_v^
        self.ca_att = ca_att^
        self.context = context^
        self.x_ca = x_ca^
        self.ffn_ln = ffn_ln^
        self.ffn_in = ffn_in^
        self.ffn_h = ffn_h^
        self.ffn_act = ffn_act^


struct WanBlockForward(Copyable, Movable):
    var x_out: List[Float32]   # [S,dim]
    var saved: WanSaved

    def __init__(out self, var x_out: List[Float32], var saved: WanSaved):
        self.x_out = x_out^
        self.saved = saved^


# ── reshape/cross-attn helpers ────────────────────────────────────────────────
def _to_bshd(x: Tensor, S: Int, H: Int, Dh: Int, ctx: DeviceContext) raises -> Tensor:
    return reshape(x, [1, S, H, Dh], ctx)


def _from_bshd(x: Tensor, S: Int, dim: Int, ctx: DeviceContext) raises -> Tensor:
    return reshape(x, [S, dim], ctx)


# Expand a [rows, half] interleaved RoPE table to [rows*H, half] (each token row
# repeated H times contiguously) — mirrors wan22_dit.mojo::_expand_rope_per_head.
def _expand_rope_per_head(
    tbl: Tensor, S: Int, H: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    # Output dtype matches `tbl` (BF16 for the bf16 forward rope; F32 when the
    # backward feeds an F32 table to the F32-only rope_backward).
    var t3 = reshape(tbl, [S, 1, half], ctx)     # [S,1,half]
    var n = S * H * half
    var zh = List[Float32]()
    for _ in range(n):
        zh.append(0.0)
    var zeros = Tensor.from_host(zh^, [S, H, half], tbl.dtype(), ctx)
    var bc = add(t3, zeros, ctx)                  # broadcast [S,1,half] over H
    return reshape(bc, [S * H, half], ctx)


# ── FORWARD of one Wan2.2 block ───────────────────────────────────────────────
# cos/sin: precomputed 3-axis interleaved rope tables [S, Dh/2] (per-token, the
# self-attn axes — text cross-attn is NOT roped). H, Dh, S, TXT comptime.
def wan22_block_forward[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    x_h: List[Float32], context_h: List[Float32], mv: WanModVecs,
    w: WanBlockWeights, cos: Tensor, sin: Tensor,
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
) raises -> WanBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t16(_ones(dim), [dim], ctx)
    var zeros_t = _t16(_zeros(dim), [dim], ctx)

    var x = _ta16(x_h, [S, dim], ctx)
    var context = _ta16(context_h, [TXT, dim], ctx)

    # per-token AdaLN vectors as bf16 device tensors [S,dim]
    var shift_sa = _t16(mv.shift_sa.copy(), [S, dim], ctx)
    var scale_sa = _t16(mv.scale_sa.copy(), [S, dim], ctx)
    var gate_sa = _t16(mv.gate_sa.copy(), [S, dim], ctx)
    var shift_ffn = _t16(mv.shift_ffn.copy(), [S, dim], ctx)
    var scale_ffn = _t16(mv.scale_ffn.copy(), [S, dim], ctx)
    var gate_ffn = _t16(mv.gate_ffn.copy(), [S, dim], ctx)

    # rope tables: cast to bf16 so rope_interleaved runs on bf16 q/k.
    var cos16 = cast_tensor(cos, STDtype.BF16, ctx)
    var sin16 = cast_tensor(sin, STDtype.BF16, ctx)

    # ── self-attention ──
    var sa_mp = wan_mod_pre(x[], scale_sa, shift_sa, ones_t, zeros_t, eps, ctx)
    var q_flat = linear(sa_mp.o[], w.sa_wq[], Optional[Tensor](_clone_t(w.sa_bq[], ctx)), ctx)
    var k_flat = linear(sa_mp.o[], w.sa_wk[], Optional[Tensor](_clone_t(w.sa_bk[], ctx)), ctx)
    var v_flat = linear(sa_mp.o[], w.sa_wv[], Optional[Tensor](_clone_t(w.sa_bv[], ctx)), ctx)
    var q_pre = _to_bshd(q_flat^, S, H, Dh, ctx)
    var k_pre = _to_bshd(k_flat^, S, H, Dh, ctx)
    var v4 = _to_bshd(v_flat^, S, H, Dh, ctx)
    var q_rms = rms_norm(q_pre, w.sa_qn[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.sa_kn[], eps, ctx)
    var cos_e = _expand_rope_per_head(cos16, S, H, Dh // 2, ctx)
    var sin_e = _expand_rope_per_head(sin16, S, H, Dh // 2, ctx)
    var q_rope = rope_interleaved(q_rms, cos_e, sin_e, ctx)
    var k_rope = rope_interleaved(k_rms, cos_e, sin_e, ctx)
    var att4 = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v4, scale, ctx)
    var sa_att = _from_bshd(att4^, S, dim, ctx)
    var sa_out = linear(sa_att, w.sa_wo[], Optional[Tensor](_clone_t(w.sa_bo[], ctx)), ctx)
    var x_sa = wan_gated_residual(x[], sa_out, gate_sa, ctx)

    # ── cross-attention (to text, distinct q-len S vs kv-len TXT) ──
    var n3 = layer_norm(x_sa, w.n3_w[], w.n3_b[], eps, ctx)
    var caq_flat = linear(n3, w.ca_wq[], Optional[Tensor](_clone_t(w.ca_bq[], ctx)), ctx)
    var cak_flat = linear(context[], w.ca_wk[], Optional[Tensor](_clone_t(w.ca_bk[], ctx)), ctx)
    var cav_flat = linear(context[], w.ca_wv[], Optional[Tensor](_clone_t(w.ca_bv[], ctx)), ctx)
    var caq_pre = _to_bshd(caq_flat^, S, H, Dh, ctx)
    var cak_pre = reshape(cak_flat^, [1, TXT, H, Dh], ctx)
    var cav4 = reshape(cav_flat^, [1, TXT, H, Dh], ctx)
    var caq_rms = rms_norm(caq_pre, w.ca_qn[], eps, ctx)
    var cak_rms = rms_norm(cak_pre, w.ca_kn[], eps, ctx)
    var ca_att4 = _cross_attention[S, TXT, H, Dh](caq_rms, cak_rms, cav4, scale, ctx)
    var ca_att = _from_bshd(ca_att4^, S, dim, ctx)
    var ca_out = linear(ca_att, w.ca_wo[], Optional[Tensor](_clone_t(w.ca_bo[], ctx)), ctx)
    var x_ca = add(x_sa, ca_out, ctx)            # UNGATED residual

    # ── FFN ──
    var ffn_mp = wan_mod_pre(x_ca, scale_ffn, shift_ffn, ones_t, zeros_t, eps, ctx)
    var ffn_h = linear(ffn_mp.o[], w.ffn0_w[], Optional[Tensor](_clone_t(w.ffn0_b[], ctx)), ctx)
    var ffn_act = gelu(ffn_h, ctx)
    var ffn_out = linear(ffn_act, w.ffn2_w[], Optional[Tensor](_clone_t(w.ffn2_b[], ctx)), ctx)
    var x_final = wan_gated_residual(x_ca, ffn_out, gate_ffn, ctx)

    var x_out = x_final.to_host(ctx)
    var saved = WanSaved(
        x[].to_host_bf16(ctx),
        sa_mp.ln[].to_host_bf16(ctx), sa_mp.o[].to_host_bf16(ctx),
        q_pre.to_host_bf16(ctx), k_pre.to_host_bf16(ctx),
        v4.to_host_bf16(ctx), q_rope.to_host_bf16(ctx), k_rope.to_host_bf16(ctx),
        sa_att.to_host_bf16(ctx), x_sa.to_host_bf16(ctx),
        n3.to_host_bf16(ctx), caq_pre.to_host_bf16(ctx), cak_pre.to_host_bf16(ctx),
        caq_rms.to_host_bf16(ctx), cak_rms.to_host_bf16(ctx),
        cav4.to_host_bf16(ctx), ca_att.to_host_bf16(ctx),
        context[].to_host_bf16(ctx), x_ca.to_host_bf16(ctx),
        ffn_mp.ln[].to_host_bf16(ctx), ffn_mp.o[].to_host_bf16(ctx),
        ffn_h.to_host_bf16(ctx), ffn_act.to_host_bf16(ctx),
    )
    return WanBlockForward(x_out^, saved^)


# Cross-attention forward: distinct q-len S vs kv-len TXT, per-head matmul
# (mirrors wan22_dit.mojo::_cross_attention). Inputs [1,S,H,Dh]/[1,TXT,H,Dh].
def _cross_attention[S: Int, TXT: Int, H: Int, Dh: Int](
    q: Tensor, k: Tensor, v: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    var dim = H * Dh
    var q3 = reshape(q, [S, H, Dh], ctx)
    var k3 = reshape(k, [TXT, H, Dh], ctx)
    var v3 = reshape(v, [TXT, H, Dh], ctx)
    var qh = permute(q3, [1, 0, 2], ctx)    # [H,S,Dh]
    var kh = permute(k3, [1, 0, 2], ctx)    # [H,TXT,Dh]
    var vh = permute(v3, [1, 0, 2], ctx)    # [H,TXT,Dh]
    var out_parts = List[ArcPointer[Tensor]]()
    for h in range(H):
        var qh_h = reshape(slice(qh, 0, h, 1, ctx), [S, Dh], ctx)
        var kh_h = reshape(slice(kh, 0, h, 1, ctx), [TXT, Dh], ctx)
        var vh_h = reshape(slice(vh, 0, h, 1, ctx), [TXT, Dh], ctx)
        var nb = Optional[Tensor](None)
        var scores = linear(qh_h, kh_h, nb^, ctx)        # [S,TXT] (q@kᵀ)
        scores = mul_scalar(scores, scale, ctx)
        var p = softmax_lastdim(scores, ctx)             # [S,TXT]
        var v_t = transpose(vh_h, 0, 1, ctx)             # [Dh,TXT]
        var nb2 = Optional[Tensor](None)
        var out_h = linear(p, v_t, nb2^, ctx)            # [S,Dh]
        out_parts.append(ArcPointer(out_h^))
    var stacked = _stack_heads(out_parts, H, S, Dh, ctx)  # [H,S,Dh]
    var sh = permute(stacked, [1, 0, 2], ctx)             # [S,H,Dh]
    return reshape(sh, [1, S, dim], ctx)


def _stack_heads(
    parts: List[ArcPointer[Tensor]], H: Int, S: Int, Dh: Int, ctx: DeviceContext
) raises -> Tensor:
    var acc = reshape(parts[0][], [1, S, Dh], ctx)
    for h in range(1, H):
        var r = reshape(parts[h][], [1, S, Dh], ctx)
        acc = concat(0, ctx, acc, r)
    return acc^


# ── backward result: input grads (img x + txt context) + every weight/bias grad
#    + per-token modulation-vector grads ─────────────────────────────────────
struct WanBlockGrads(Copyable, Movable):
    var d_x: List[Float32]        # [S,dim]   img stream input grad
    var d_context: List[Float32]  # [TXT,dim] text stream input grad (cross-attn kv)
    # self-attn weight/bias grads
    var d_sa_wq: List[Float32]
    var d_sa_wk: List[Float32]
    var d_sa_wv: List[Float32]
    var d_sa_wo: List[Float32]
    var d_sa_bq: List[Float32]
    var d_sa_bk: List[Float32]
    var d_sa_bv: List[Float32]
    var d_sa_bo: List[Float32]
    var d_sa_qn: List[Float32]
    var d_sa_kn: List[Float32]
    # cross-attn weight/bias grads
    var d_ca_wq: List[Float32]
    var d_ca_wk: List[Float32]
    var d_ca_wv: List[Float32]
    var d_ca_wo: List[Float32]
    var d_ca_bq: List[Float32]
    var d_ca_bk: List[Float32]
    var d_ca_bv: List[Float32]
    var d_ca_bo: List[Float32]
    var d_ca_qn: List[Float32]
    var d_ca_kn: List[Float32]
    # norm3 affine grads
    var d_n3_w: List[Float32]
    var d_n3_b: List[Float32]
    # ffn grads
    var d_ffn0_w: List[Float32]
    var d_ffn0_b: List[Float32]
    var d_ffn2_w: List[Float32]
    var d_ffn2_b: List[Float32]
    # per-token modulation-vector grads (each [S,dim])
    var d_shift_sa: List[Float32]
    var d_scale_sa: List[Float32]
    var d_gate_sa: List[Float32]
    var d_shift_ffn: List[Float32]
    var d_scale_ffn: List[Float32]
    var d_gate_ffn: List[Float32]

    def __init__(
        out self,
        var d_x: List[Float32], var d_context: List[Float32],
        var d_sa_wq: List[Float32], var d_sa_wk: List[Float32], var d_sa_wv: List[Float32], var d_sa_wo: List[Float32],
        var d_sa_bq: List[Float32], var d_sa_bk: List[Float32], var d_sa_bv: List[Float32], var d_sa_bo: List[Float32],
        var d_sa_qn: List[Float32], var d_sa_kn: List[Float32],
        var d_ca_wq: List[Float32], var d_ca_wk: List[Float32], var d_ca_wv: List[Float32], var d_ca_wo: List[Float32],
        var d_ca_bq: List[Float32], var d_ca_bk: List[Float32], var d_ca_bv: List[Float32], var d_ca_bo: List[Float32],
        var d_ca_qn: List[Float32], var d_ca_kn: List[Float32],
        var d_n3_w: List[Float32], var d_n3_b: List[Float32],
        var d_ffn0_w: List[Float32], var d_ffn0_b: List[Float32],
        var d_ffn2_w: List[Float32], var d_ffn2_b: List[Float32],
        var d_shift_sa: List[Float32], var d_scale_sa: List[Float32], var d_gate_sa: List[Float32],
        var d_shift_ffn: List[Float32], var d_scale_ffn: List[Float32], var d_gate_ffn: List[Float32],
    ):
        self.d_x = d_x^
        self.d_context = d_context^
        self.d_sa_wq = d_sa_wq^
        self.d_sa_wk = d_sa_wk^
        self.d_sa_wv = d_sa_wv^
        self.d_sa_wo = d_sa_wo^
        self.d_sa_bq = d_sa_bq^
        self.d_sa_bk = d_sa_bk^
        self.d_sa_bv = d_sa_bv^
        self.d_sa_bo = d_sa_bo^
        self.d_sa_qn = d_sa_qn^
        self.d_sa_kn = d_sa_kn^
        self.d_ca_wq = d_ca_wq^
        self.d_ca_wk = d_ca_wk^
        self.d_ca_wv = d_ca_wv^
        self.d_ca_wo = d_ca_wo^
        self.d_ca_bq = d_ca_bq^
        self.d_ca_bk = d_ca_bk^
        self.d_ca_bv = d_ca_bv^
        self.d_ca_bo = d_ca_bo^
        self.d_ca_qn = d_ca_qn^
        self.d_ca_kn = d_ca_kn^
        self.d_n3_w = d_n3_w^
        self.d_n3_b = d_n3_b^
        self.d_ffn0_w = d_ffn0_w^
        self.d_ffn0_b = d_ffn0_b^
        self.d_ffn2_w = d_ffn2_w^
        self.d_ffn2_b = d_ffn2_b^
        self.d_shift_sa = d_shift_sa^
        self.d_scale_sa = d_scale_sa^
        self.d_gate_sa = d_gate_sa^
        self.d_shift_ffn = d_shift_ffn^
        self.d_scale_ffn = d_scale_ffn^
        self.d_gate_ffn = d_gate_ffn^


# ── BACKWARD of one Wan2.2 block (hand-chained reverse of the forward graph) ──
# d_out_h: upstream grad of x_final ([S,dim]). All weight/mod grads leave host.
def wan22_block_backward[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    d_out_h: List[Float32], mv: WanModVecs, w: WanBlockWeights, saved: WanSaved,
    cos: Tensor, sin: Tensor,
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
) raises -> WanBlockGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t16(_ones(dim), [dim], ctx)
    # rope_backward is F32-only -> keep the expanded tables F32 (cos/sin arrive
    # F32 from the public API); local-cast the bf16 grads to F32 at that one call.
    var cos_e = _expand_rope_per_head(cos, S, H, Dh // 2, ctx)
    var sin_e = _expand_rope_per_head(sin, S, H, Dh // 2, ctx)

    var d_out = _ta16(d_out_h, [S, dim], ctx)

    # ════════════════ FFN backward ════════════════
    # x_final = x_ca + gate_ffn * ffn_out   (per-token gated residual)
    var gate_ffn_t = _t16(mv.gate_ffn.copy(), [S, dim], ctx)
    # d_gate_ffn = go*ffn_out, d_y(=d_ffn_out)=go*gate_ffn, d_x(branch to x_ca)=go.
    # ffn_out is not saved -> recompute ffn_out = linear(ffn_act, ffn2_w)+ffn2_b.
    var sv_ffn_act = _tbf16(saved.ffn_act.copy(), [S, ffn], ctx)
    var ffn_out_rc = linear(sv_ffn_act, w.ffn2_w[], Optional[Tensor](_clone_t(w.ffn2_b[], ctx)), ctx)
    var gb_ffn2 = wan_gate_residual_backward(d_out[], ffn_out_rc, gate_ffn_t, ctx)
    var d_gate_ffn = gb_ffn2.d_gate.to_host(ctx)
    var d_x_ca_resid = TArc(gb_ffn2.d_x.clone(ctx))  # branch into x_ca via residual

    # ffn_out = linear(ffn_act, ffn2_w, ffn2_b) — d_ffn_out borrowed inline.
    var lb_ffn2 = linear_backward(gb_ffn2.d_y, sv_ffn_act, w.ffn2_w[], S, ffn, dim, ctx)
    var d_ffn2_w = lb_ffn2.d_w.to_host(ctx)
    var d_ffn2_b = lb_ffn2.d_b.to_host(ctx)

    # ffn_act = gelu(ffn_h)
    var sv_ffn_h = _tbf16(saved.ffn_h.copy(), [S, ffn], ctx)
    var d_ffn_h = gelu_backward(lb_ffn2.d_x, sv_ffn_h, ctx)

    # ffn_h = linear(ffn_in, ffn0_w, ffn0_b)
    var sv_ffn_in = _tbf16(saved.ffn_in.copy(), [S, dim], ctx)
    var lb_ffn0 = linear_backward(d_ffn_h, sv_ffn_in, w.ffn0_w[], S, dim, ffn, ctx)
    var d_ffn0_w = lb_ffn0.d_w.to_host(ctx)
    var d_ffn0_b = lb_ffn0.d_b.to_host(ctx)

    # ffn_in = mod_pre(x_ca): ffn_in = ffn_ln*(1+scale_ffn)+shift_ffn
    var scale_ffn_t = _t16(mv.scale_ffn.copy(), [S, dim], ctx)
    var sv_ffn_ln = _tbf16(saved.ffn_ln.copy(), [S, dim], ctx)
    var mb_ffn = wan_modulate_backward(lb_ffn0.d_x, sv_ffn_ln, scale_ffn_t, ctx)
    var d_scale_ffn = mb_ffn.d_scale.to_host(ctx)
    var d_shift_ffn = mb_ffn.d_shift.to_host(ctx)

    # ffn_ln = LN_no_affine(x_ca) (layer_norm_backward_dx returns d_x Tensor)
    var sv_x_ca = _tbf16(saved.x_ca.copy(), [S, dim], ctx)
    var lnb_ffn = layer_norm_backward_dx(mb_ffn.d_ln, sv_x_ca, ones_t, eps, ctx)
    # x_ca feeds BOTH the residual branch AND ffn LN -> SUM
    var d_x_ca = TArc(add(d_x_ca_resid[], lnb_ffn, ctx))

    # ════════════════ Cross-attention backward ════════════════
    # x_ca = x_sa + ca_out  (ungated): d_x_sa_branch = d_x_ca ; d_ca_out = d_x_ca.
    # ca_out = linear(ca_att, ca_wo, ca_bo) — d_ca_out (== d_x_ca) borrowed inline.
    var sv_ca_att = _tbf16(saved.ca_att.copy(), [S, dim], ctx)
    var lb_cao = linear_backward(d_x_ca[], sv_ca_att, w.ca_wo[], S, dim, dim, ctx)
    var d_ca_wo = lb_cao.d_w.to_host(ctx)
    var d_ca_bo = lb_cao.d_b.to_host(ctx)
    var d_ca_att = reshape(lb_cao.d_x, [1, S, H, Dh], ctx)   # byte no-op

    # cross-attn SDPA (rect Sq=S, Skv=TXT) on saved q_rms/k_rms/v -> d_q_rms, d_k_rms, d_v
    var sv_ca_q_rms = _tbf16(saved.ca_q_rms.copy(), [1, S, H, Dh], ctx)
    var sv_ca_k_rms = _tbf16(saved.ca_k_rms.copy(), [1, TXT, H, Dh], ctx)
    var sv_ca_v = _tbf16(saved.ca_v.copy(), [1, TXT, H, Dh], ctx)
    var csb = sdpa_backward_rect[1, S, TXT, H, Dh](
        sv_ca_q_rms, sv_ca_k_rms, sv_ca_v, d_ca_att, scale, ctx,
    )
    # caq_rms = rms_norm(caq_pre, ca_qn)
    var sv_ca_q_pre = _tbf16(saved.ca_q_pre.copy(), [1, S, H, Dh], ctx)
    var rb_caq = rms_norm_backward(csb.d_q, sv_ca_q_pre, w.ca_qn[], eps, ctx)
    var d_ca_qn = rb_caq.d_g.to_host(ctx)
    var sv_ca_k_pre = _tbf16(saved.ca_k_pre.copy(), [1, TXT, H, Dh], ctx)
    var rb_cak = rms_norm_backward(csb.d_k, sv_ca_k_pre, w.ca_kn[], eps, ctx)
    var d_ca_kn = rb_cak.d_g.to_host(ctx)

    # reshape [1,S,H,Dh]->[S,dim], [1,TXT,H,Dh]->[TXT,dim] (byte no-ops)
    var d_caq_flat = reshape(rb_caq.d_x, [S, dim], ctx)
    var d_cak_flat = reshape(rb_cak.d_x, [TXT, dim], ctx)
    var d_cav_flat = reshape(csb.d_v, [TXT, dim], ctx)

    # caq = linear(n3, ca_wq, ca_bq) ; cak = linear(context, ca_wk, ca_bk) ;
    # cav = linear(context, ca_wv, ca_bv)
    var sv_ca_n3 = _tbf16(saved.ca_n3.copy(), [S, dim], ctx)
    var sv_context = _tbf16(saved.context.copy(), [TXT, dim], ctx)
    var lb_caq = linear_backward(d_caq_flat, sv_ca_n3, w.ca_wq[], S, dim, dim, ctx)
    var lb_cak = linear_backward(d_cak_flat, sv_context, w.ca_wk[], TXT, dim, dim, ctx)
    var lb_cav = linear_backward(d_cav_flat, sv_context, w.ca_wv[], TXT, dim, dim, ctx)
    var d_ca_wq = lb_caq.d_w.to_host(ctx)
    var d_ca_bq = lb_caq.d_b.to_host(ctx)
    var d_ca_wk = lb_cak.d_w.to_host(ctx)
    var d_ca_bk = lb_cak.d_b.to_host(ctx)
    var d_ca_wv = lb_cav.d_w.to_host(ctx)
    var d_ca_bv = lb_cav.d_b.to_host(ctx)
    # context feeds BOTH cak and cav -> SUM (the text stream input grad)
    var d_context_t = TArc(add(lb_cak.d_x, lb_cav.d_x, ctx))

    # n3 = layer_norm(x_sa, n3_w, n3_b)  (affine; layer_norm_backward gives d_x/d_g/d_b)
    var sv_x_sa = _tbf16(saved.x_sa.copy(), [S, dim], ctx)
    var lnb_n3 = layer_norm_backward(lb_caq.d_x, sv_x_sa, w.n3_w[], eps, ctx)
    var d_n3_w = lnb_n3.d_g.to_host(ctx)
    var d_n3_b = lnb_n3.d_b.to_host(ctx)
    # x_sa feeds: cross-attn n3 (lnb_n3.d_x) AND ungated residual (d_x_ca) -> SUM
    var d_x_sa = TArc(add(lnb_n3.d_x, d_x_ca[], ctx))

    # ════════════════ Self-attention backward ════════════════
    # x_sa = x + gate_sa * sa_out  (per-token gated residual)
    var gate_sa_t = _t16(mv.gate_sa.copy(), [S, dim], ctx)
    var sv_sa_att = _tbf16(saved.sa_att.copy(), [S, dim], ctx)
    var sa_out_rc = linear(sv_sa_att, w.sa_wo[], Optional[Tensor](_clone_t(w.sa_bo[], ctx)), ctx)
    var gb_sa = wan_gate_residual_backward(d_x_sa[], sa_out_rc, gate_sa_t, ctx)
    var d_gate_sa = gb_sa.d_gate.to_host(ctx)
    var d_x_resid = TArc(gb_sa.d_x.clone(ctx))   # branch into block input x

    # sa_out = linear(sa_att, sa_wo, sa_bo) — d_sa_out borrowed inline.
    var lb_sao = linear_backward(gb_sa.d_y, sv_sa_att, w.sa_wo[], S, dim, dim, ctx)
    var d_sa_wo = lb_sao.d_w.to_host(ctx)
    var d_sa_bo = lb_sao.d_b.to_host(ctx)
    var d_sa_att = reshape(lb_sao.d_x, [1, S, H, Dh], ctx)

    # self-attn SDPA (square) on saved q_rope/k_rope/v -> d_q_rope, d_k_rope, d_v
    var sv_sa_q_rope = _tbf16(saved.sa_q_rope.copy(), [1, S, H, Dh], ctx)
    var sv_sa_k_rope = _tbf16(saved.sa_k_rope.copy(), [1, S, H, Dh], ctx)
    var sv_sa_v = _tbf16(saved.sa_v.copy(), [1, S, H, Dh], ctx)
    var ssb = sdpa_backward[1, S, H, Dh](
        sv_sa_q_rope, sv_sa_k_rope, sv_sa_v, d_sa_att, scale, ctx,
    )
    # rope backward (interleaved; cos/sin non-learnable -> d_x only).
    # LOCAL F32 CAST: rope_backward is F32-only; sdpa grads come back bf16, so
    # cast them up for this call (cos_e/sin_e are already F32). Result d_q_rms is
    # F32 -> the next rms_norm_backward up-casts internally (no further cast).
    var ssb_dq_f32 = cast_tensor(ssb.d_q, STDtype.F32, ctx)
    var ssb_dk_f32 = cast_tensor(ssb.d_k, STDtype.F32, ctx)
    var d_q_rms = rope_backward(ssb_dq_f32, cos_e, sin_e, True, ctx)
    var d_k_rms = rope_backward(ssb_dk_f32, cos_e, sin_e, True, ctx)
    # q_rms = rms_norm(q_pre, sa_qn) ; k_rms = rms_norm(k_pre, sa_kn)
    var sv_sa_q_pre = _tbf16(saved.sa_q_pre.copy(), [1, S, H, Dh], ctx)
    var rb_saq = rms_norm_backward(d_q_rms, sv_sa_q_pre, w.sa_qn[], eps, ctx)
    var d_sa_qn = rb_saq.d_g.to_host(ctx)
    var sv_sa_k_pre = _tbf16(saved.sa_k_pre.copy(), [1, S, H, Dh], ctx)
    var rb_sak = rms_norm_backward(d_k_rms, sv_sa_k_pre, w.sa_kn[], eps, ctx)
    var d_sa_kn = rb_sak.d_g.to_host(ctx)

    var d_saq_flat = reshape(rb_saq.d_x, [S, dim], ctx)
    var d_sak_flat = reshape(rb_sak.d_x, [S, dim], ctx)
    var d_sav_flat = reshape(ssb.d_v, [S, dim], ctx)

    # q/k/v = linear(sa_in, sa_w{q,k,v}, sa_b{q,k,v}) — all on the SAME sa_in
    var sv_sa_in = _tbf16(saved.sa_in.copy(), [S, dim], ctx)
    var lb_saq = linear_backward(d_saq_flat, sv_sa_in, w.sa_wq[], S, dim, dim, ctx)
    var lb_sak = linear_backward(d_sak_flat, sv_sa_in, w.sa_wk[], S, dim, dim, ctx)
    var lb_sav = linear_backward(d_sav_flat, sv_sa_in, w.sa_wv[], S, dim, dim, ctx)
    var d_sa_wq = lb_saq.d_w.to_host(ctx)
    var d_sa_bq = lb_saq.d_b.to_host(ctx)
    var d_sa_wk = lb_sak.d_w.to_host(ctx)
    var d_sa_bk = lb_sak.d_b.to_host(ctx)
    var d_sa_wv = lb_sav.d_w.to_host(ctx)
    var d_sa_bv = lb_sav.d_b.to_host(ctx)
    # sa_in feeds all three q/k/v -> SUM
    var d_sa_in = TArc(add(add(lb_saq.d_x, lb_sak.d_x, ctx), lb_sav.d_x, ctx))

    # sa_in = mod_pre(x): sa_in = sa_ln*(1+scale_sa)+shift_sa
    var scale_sa_t = _t16(mv.scale_sa.copy(), [S, dim], ctx)
    var sv_sa_ln = _tbf16(saved.sa_ln.copy(), [S, dim], ctx)
    var mb_sa = wan_modulate_backward(d_sa_in[], sv_sa_ln, scale_sa_t, ctx)
    var d_scale_sa = mb_sa.d_scale.to_host(ctx)
    var d_shift_sa = mb_sa.d_shift.to_host(ctx)

    # sa_ln = LN_no_affine(x) (returns d_x Tensor)
    var sv_x = _tbf16(saved.x.copy(), [S, dim], ctx)
    var lnb_sa = layer_norm_backward_dx(mb_sa.d_ln, sv_x, ones_t, eps, ctx)
    # x feeds: self-attn LN (lnb_sa) AND gated residual (d_x_resid) -> SUM
    var d_x = add(lnb_sa, d_x_resid[], ctx)
    var d_x_h = d_x.to_host(ctx)
    var d_context_h = d_context_t[].to_host(ctx)

    return WanBlockGrads(
        d_x_h^, d_context_h^,
        d_sa_wq^, d_sa_wk^, d_sa_wv^, d_sa_wo^,
        d_sa_bq^, d_sa_bk^, d_sa_bv^, d_sa_bo^,
        d_sa_qn^, d_sa_kn^,
        d_ca_wq^, d_ca_wk^, d_ca_wv^, d_ca_wo^,
        d_ca_bq^, d_ca_bk^, d_ca_bv^, d_ca_bo^,
        d_ca_qn^, d_ca_kn^,
        d_n3_w^, d_n3_b^,
        d_ffn0_w^, d_ffn0_b^, d_ffn2_w^, d_ffn2_b^,
        d_shift_sa^, d_scale_sa^, d_gate_sa^,
        d_shift_ffn^, d_scale_ffn^, d_gate_ffn^,
    )


# ═══════════════════════════════════════════════════════════════════════════
# LoRA-ON-PROJECTION VARIANT
#
# Targets (EDv2 wan22.rs:199-206 LoraTarget, 8/block):
#   self_attn.{q,k,v,o} + cross_attn.{q,k,v,o}  (in=out=dim each).
# Forward adds the LoRA delta at each projection's linear output; backward
# returns d_A/d_B for each and folds the LoRA d_x contribution back into the
# projection-input grad. REUSES klein_lora_fwd / klein_lora_bwd (the model-
# agnostic y=linear(x,W) LoRA math = train_step._lora_fwd/_lora_bwd).
# ═══════════════════════════════════════════════════════════════════════════

from serenitymojo.models.klein.lora_block import (
    LoraAdapter, klein_lora_fwd, klein_lora_bwd, KleinLoraGrads,
)


# Optional LoRA adapters for the block's 8 trained attention projections.
struct WanBlockLora(Copyable, Movable):
    var sa_q: Optional[LoraAdapter]
    var sa_k: Optional[LoraAdapter]
    var sa_v: Optional[LoraAdapter]
    var sa_o: Optional[LoraAdapter]
    var ca_q: Optional[LoraAdapter]
    var ca_k: Optional[LoraAdapter]
    var ca_v: Optional[LoraAdapter]
    var ca_o: Optional[LoraAdapter]

    def __init__(
        out self,
        var sa_q: Optional[LoraAdapter], var sa_k: Optional[LoraAdapter],
        var sa_v: Optional[LoraAdapter], var sa_o: Optional[LoraAdapter],
        var ca_q: Optional[LoraAdapter], var ca_k: Optional[LoraAdapter],
        var ca_v: Optional[LoraAdapter], var ca_o: Optional[LoraAdapter],
    ):
        self.sa_q = sa_q^
        self.sa_k = sa_k^
        self.sa_v = sa_v^
        self.sa_o = sa_o^
        self.ca_q = ca_q^
        self.ca_k = ca_k^
        self.ca_v = ca_v^
        self.ca_o = ca_o^


# d_A/d_B for the 8 adapters (empty when absent).
struct WanBlockLoraGrads(Copyable, Movable):
    var base: WanBlockGrads
    var sa_q_da: List[Float32]
    var sa_q_db: List[Float32]
    var sa_k_da: List[Float32]
    var sa_k_db: List[Float32]
    var sa_v_da: List[Float32]
    var sa_v_db: List[Float32]
    var sa_o_da: List[Float32]
    var sa_o_db: List[Float32]
    var ca_q_da: List[Float32]
    var ca_q_db: List[Float32]
    var ca_k_da: List[Float32]
    var ca_k_db: List[Float32]
    var ca_v_da: List[Float32]
    var ca_v_db: List[Float32]
    var ca_o_da: List[Float32]
    var ca_o_db: List[Float32]

    def __init__(
        out self, var base: WanBlockGrads,
        var sa_q_da: List[Float32], var sa_q_db: List[Float32],
        var sa_k_da: List[Float32], var sa_k_db: List[Float32],
        var sa_v_da: List[Float32], var sa_v_db: List[Float32],
        var sa_o_da: List[Float32], var sa_o_db: List[Float32],
        var ca_q_da: List[Float32], var ca_q_db: List[Float32],
        var ca_k_da: List[Float32], var ca_k_db: List[Float32],
        var ca_v_da: List[Float32], var ca_v_db: List[Float32],
        var ca_o_da: List[Float32], var ca_o_db: List[Float32],
    ):
        self.base = base^
        self.sa_q_da = sa_q_da^
        self.sa_q_db = sa_q_db^
        self.sa_k_da = sa_k_da^
        self.sa_k_db = sa_k_db^
        self.sa_v_da = sa_v_da^
        self.sa_v_db = sa_v_db^
        self.sa_o_da = sa_o_da^
        self.sa_o_db = sa_o_db^
        self.ca_q_da = ca_q_da^
        self.ca_q_db = ca_q_db^
        self.ca_k_da = ca_k_da^
        self.ca_k_db = ca_k_db^
        self.ca_v_da = ca_v_da^
        self.ca_v_db = ca_v_db^
        self.ca_o_da = ca_o_da^
        self.ca_o_db = ca_o_db^


def _empty() -> List[Float32]:
    return List[Float32]()


# Add the LoRA contribution of a projection (host input x_h [M,in]) into a device
# output Tensor y [M,out]; returns y + delta as a fresh device Tensor.
def _add_lora_delta(
    y: Tensor, x_h: List[Float32], lo: Optional[LoraAdapter], M: Int, ctx: DeviceContext,
) raises -> Tensor:
    if not lo:
        return _clone_t(y, ctx)
    var delta_h = klein_lora_fwd(x_h, lo.value(), M, ctx)
    # klein_lora_fwd returns a host F32 delta; upload in y's dtype (bf16) so the
    # add stays in the native bf16 compute path.
    var delta = Tensor.from_host(delta_h^, y.shape().copy(), y.dtype(), ctx)
    return add(y, delta, ctx)


# Run the LoRA backward for one projection (if present): given the projection's
# output grad d_y_h [M,out] and saved input x_h [M,in], returns (d_A,d_B,d_x_lo).
# d_x_lo is the LoRA branch's contribution to the projection input grad (must be
# summed into the base path's d_x by the caller).
def _lora_bwd_opt(
    lo: Optional[LoraAdapter], d_y_h: List[Float32], x_h: List[Float32],
    M: Int, in_f: Int, ctx: DeviceContext,
) raises -> KleinLoraGrads:
    if not lo:
        var z = List[Float32]()
        for _ in range(M * in_f):
            z.append(0.0)
        return KleinLoraGrads(_empty(), _empty(), z^)
    return klein_lora_bwd(d_y_h, x_h, lo.value(), M, ctx)


# ── FORWARD of one Wan2.2 block WITH LoRA ─────────────────────────────────────
# Identical to wan22_block_forward but adds LoRA deltas at the 8 attention
# projections. Saves the SAME WanSaved (the base saved activations) — but note
# the saved q/k/v/att activations now INCLUDE the LoRA delta (the forward graph),
# which is exactly what the base backward consumes.
def wan22_block_lora_forward[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    x_h: List[Float32], context_h: List[Float32], mv: WanModVecs,
    w: WanBlockWeights, lora: WanBlockLora, cos: Tensor, sin: Tensor,
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
) raises -> WanBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t16(_ones(dim), [dim], ctx)
    var zeros_t = _t16(_zeros(dim), [dim], ctx)

    var x = _ta16(x_h, [S, dim], ctx)
    var context = _ta16(context_h, [TXT, dim], ctx)
    var context_host = context_h.copy()

    var shift_sa = _t16(mv.shift_sa.copy(), [S, dim], ctx)
    var scale_sa = _t16(mv.scale_sa.copy(), [S, dim], ctx)
    var gate_sa = _t16(mv.gate_sa.copy(), [S, dim], ctx)
    var shift_ffn = _t16(mv.shift_ffn.copy(), [S, dim], ctx)
    var scale_ffn = _t16(mv.scale_ffn.copy(), [S, dim], ctx)
    var gate_ffn = _t16(mv.gate_ffn.copy(), [S, dim], ctx)

    var cos16 = cast_tensor(cos, STDtype.BF16, ctx)
    var sin16 = cast_tensor(sin, STDtype.BF16, ctx)

    # ── self-attention (LoRA on q/k/v/o) ──
    var sa_mp = wan_mod_pre(x[], scale_sa, shift_sa, ones_t, zeros_t, eps, ctx)
    var sa_in_h = sa_mp.o[].to_host(ctx)
    var q_base = linear(sa_mp.o[], w.sa_wq[], Optional[Tensor](_clone_t(w.sa_bq[], ctx)), ctx)
    var k_base = linear(sa_mp.o[], w.sa_wk[], Optional[Tensor](_clone_t(w.sa_bk[], ctx)), ctx)
    var v_base = linear(sa_mp.o[], w.sa_wv[], Optional[Tensor](_clone_t(w.sa_bv[], ctx)), ctx)
    var q_flat = _add_lora_delta(q_base, sa_in_h, lora.sa_q, S, ctx)
    var k_flat = _add_lora_delta(k_base, sa_in_h, lora.sa_k, S, ctx)
    var v_flat = _add_lora_delta(v_base, sa_in_h, lora.sa_v, S, ctx)
    var q_pre = _to_bshd(q_flat^, S, H, Dh, ctx)
    var k_pre = _to_bshd(k_flat^, S, H, Dh, ctx)
    var v4 = _to_bshd(v_flat^, S, H, Dh, ctx)
    var q_rms = rms_norm(q_pre, w.sa_qn[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.sa_kn[], eps, ctx)
    var cos_e = _expand_rope_per_head(cos16, S, H, Dh // 2, ctx)
    var sin_e = _expand_rope_per_head(sin16, S, H, Dh // 2, ctx)
    var q_rope = rope_interleaved(q_rms, cos_e, sin_e, ctx)
    var k_rope = rope_interleaved(k_rms, cos_e, sin_e, ctx)
    var att4 = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v4, scale, ctx)
    var sa_att = _from_bshd(att4^, S, dim, ctx)
    var sa_att_h = sa_att.to_host(ctx)
    var sa_out_base = linear(sa_att, w.sa_wo[], Optional[Tensor](_clone_t(w.sa_bo[], ctx)), ctx)
    var sa_out = _add_lora_delta(sa_out_base, sa_att_h, lora.sa_o, S, ctx)
    var x_sa = wan_gated_residual(x[], sa_out, gate_sa, ctx)

    # ── cross-attention (LoRA on q/k/v/o) ──
    var n3 = layer_norm(x_sa, w.n3_w[], w.n3_b[], eps, ctx)
    var n3_h = n3.to_host(ctx)
    var caq_base = linear(n3, w.ca_wq[], Optional[Tensor](_clone_t(w.ca_bq[], ctx)), ctx)
    var cak_base = linear(context[], w.ca_wk[], Optional[Tensor](_clone_t(w.ca_bk[], ctx)), ctx)
    var cav_base = linear(context[], w.ca_wv[], Optional[Tensor](_clone_t(w.ca_bv[], ctx)), ctx)
    var caq_flat = _add_lora_delta(caq_base, n3_h, lora.ca_q, S, ctx)
    var cak_flat = _add_lora_delta(cak_base, context_host, lora.ca_k, TXT, ctx)
    var cav_flat = _add_lora_delta(cav_base, context_host, lora.ca_v, TXT, ctx)
    var caq_pre = _to_bshd(caq_flat^, S, H, Dh, ctx)
    var cak_pre = reshape(cak_flat^, [1, TXT, H, Dh], ctx)
    var cav4 = reshape(cav_flat^, [1, TXT, H, Dh], ctx)
    var caq_rms = rms_norm(caq_pre, w.ca_qn[], eps, ctx)
    var cak_rms = rms_norm(cak_pre, w.ca_kn[], eps, ctx)
    var ca_att4 = _cross_attention[S, TXT, H, Dh](caq_rms, cak_rms, cav4, scale, ctx)
    var ca_att = _from_bshd(ca_att4^, S, dim, ctx)
    var ca_att_h = ca_att.to_host(ctx)
    var ca_out_base = linear(ca_att, w.ca_wo[], Optional[Tensor](_clone_t(w.ca_bo[], ctx)), ctx)
    var ca_out = _add_lora_delta(ca_out_base, ca_att_h, lora.ca_o, S, ctx)
    var x_ca = add(x_sa, ca_out, ctx)

    # ── FFN (no LoRA) ──
    var ffn_mp = wan_mod_pre(x_ca, scale_ffn, shift_ffn, ones_t, zeros_t, eps, ctx)
    var ffn_h = linear(ffn_mp.o[], w.ffn0_w[], Optional[Tensor](_clone_t(w.ffn0_b[], ctx)), ctx)
    var ffn_act = gelu(ffn_h, ctx)
    var ffn_out = linear(ffn_act, w.ffn2_w[], Optional[Tensor](_clone_t(w.ffn2_b[], ctx)), ctx)
    var x_final = wan_gated_residual(x_ca, ffn_out, gate_ffn, ctx)

    var x_out = x_final.to_host(ctx)
    var saved = WanSaved(
        x[].to_host_bf16(ctx),
        sa_mp.ln[].to_host_bf16(ctx), sa_mp.o[].to_host_bf16(ctx),
        q_pre.to_host_bf16(ctx), k_pre.to_host_bf16(ctx),
        v4.to_host_bf16(ctx), q_rope.to_host_bf16(ctx), k_rope.to_host_bf16(ctx),
        sa_att.to_host_bf16(ctx), x_sa.to_host_bf16(ctx),
        n3.to_host_bf16(ctx), caq_pre.to_host_bf16(ctx), cak_pre.to_host_bf16(ctx),
        caq_rms.to_host_bf16(ctx), cak_rms.to_host_bf16(ctx),
        cav4.to_host_bf16(ctx), ca_att.to_host_bf16(ctx),
        context[].to_host_bf16(ctx), x_ca.to_host_bf16(ctx),
        ffn_mp.ln[].to_host_bf16(ctx), ffn_mp.o[].to_host_bf16(ctx),
        ffn_h.to_host_bf16(ctx), ffn_act.to_host_bf16(ctx),
    )
    return WanBlockForward(x_out^, saved^)


# ── BACKWARD of one Wan2.2 block WITH LoRA ────────────────────────────────────
# Mirrors wan22_block_backward but ALSO runs the 8 attention-projection LoRA
# backwards and folds each LoRA d_x contribution into the base projection-input
# grad (so d_x/d_context include the LoRA branch). The base weight grads are the
# FROZEN-W grads (still reported for completeness). The LoRA backward needs each
# projection's saved input (host) + its output grad (host); both are available
# from the base reverse chain.
def wan22_block_lora_backward[
    H: Int, Dh: Int, S: Int, TXT: Int
](
    d_out_h: List[Float32], mv: WanModVecs, w: WanBlockWeights,
    lora: WanBlockLora, saved: WanSaved, cos: Tensor, sin: Tensor,
    dim: Int, ffn: Int, eps: Float32, ctx: DeviceContext,
) raises -> WanBlockLoraGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t16(_ones(dim), [dim], ctx)
    # rope_backward is F32-only -> expanded tables stay F32 here.
    var cos_e = _expand_rope_per_head(cos, S, H, Dh // 2, ctx)
    var sin_e = _expand_rope_per_head(sin, S, H, Dh // 2, ctx)

    # projection-input host activations re-uploaded as BF16 device tensors for
    # the LoRA backward (need F32 host copies for klein_lora_bwd).
    var sv_lora_sa_in = _tbf16(saved.sa_in.copy(), [S, dim], ctx)
    var sv_lora_sa_att = _tbf16(saved.sa_att.copy(), [S, dim], ctx)
    var sv_lora_ca_n3 = _tbf16(saved.ca_n3.copy(), [S, dim], ctx)
    var sv_lora_context = _tbf16(saved.context.copy(), [TXT, dim], ctx)
    var sv_lora_ca_att = _tbf16(saved.ca_att.copy(), [S, dim], ctx)
    # LoRA backward needs F32 host inputs (klein_lora_bwd takes List[Float32])
    var sa_in_h = sv_lora_sa_in.to_host(ctx)
    var sa_att_h = sv_lora_sa_att.to_host(ctx)
    var n3_h = sv_lora_ca_n3.to_host(ctx)
    var context_h = sv_lora_context.to_host(ctx)
    var ca_att_h = sv_lora_ca_att.to_host(ctx)

    var d_out = _ta16(d_out_h, [S, dim], ctx)

    # ════════════════ FFN backward (no LoRA) ════════════════
    var gate_ffn_t = _t16(mv.gate_ffn.copy(), [S, dim], ctx)
    var lsv_ffn_act = _tbf16(saved.ffn_act.copy(), [S, ffn], ctx)
    var ffn_out_rc = linear(lsv_ffn_act, w.ffn2_w[], Optional[Tensor](_clone_t(w.ffn2_b[], ctx)), ctx)
    var gb_ffn2 = wan_gate_residual_backward(d_out[], ffn_out_rc, gate_ffn_t, ctx)
    var d_gate_ffn = gb_ffn2.d_gate.to_host(ctx)
    var d_x_ca_resid = TArc(gb_ffn2.d_x.clone(ctx))
    var lb_ffn2 = linear_backward(gb_ffn2.d_y, lsv_ffn_act, w.ffn2_w[], S, ffn, dim, ctx)
    var d_ffn2_w = lb_ffn2.d_w.to_host(ctx)
    var d_ffn2_b = lb_ffn2.d_b.to_host(ctx)
    var lsv_ffn_h = _tbf16(saved.ffn_h.copy(), [S, ffn], ctx)
    var d_ffn_h = gelu_backward(lb_ffn2.d_x, lsv_ffn_h, ctx)
    var lsv_ffn_in = _tbf16(saved.ffn_in.copy(), [S, dim], ctx)
    var lb_ffn0 = linear_backward(d_ffn_h, lsv_ffn_in, w.ffn0_w[], S, dim, ffn, ctx)
    var d_ffn0_w = lb_ffn0.d_w.to_host(ctx)
    var d_ffn0_b = lb_ffn0.d_b.to_host(ctx)
    var scale_ffn_t = _t16(mv.scale_ffn.copy(), [S, dim], ctx)
    var lsv_ffn_ln = _tbf16(saved.ffn_ln.copy(), [S, dim], ctx)
    var mb_ffn = wan_modulate_backward(lb_ffn0.d_x, lsv_ffn_ln, scale_ffn_t, ctx)
    var d_scale_ffn = mb_ffn.d_scale.to_host(ctx)
    var d_shift_ffn = mb_ffn.d_shift.to_host(ctx)
    var lsv_x_ca = _tbf16(saved.x_ca.copy(), [S, dim], ctx)
    var lnb_ffn = layer_norm_backward_dx(mb_ffn.d_ln, lsv_x_ca, ones_t, eps, ctx)
    var d_x_ca = TArc(add(d_x_ca_resid[], lnb_ffn, ctx))

    # ════════════════ Cross-attention backward (LoRA q/k/v/o) ════════════════
    # ca_out = linear(ca_att, ca_wo, ca_bo) + LoRA(ca_o, ca_att)
    var d_ca_out_h = d_x_ca[].to_host(ctx)          # output grad of ca_out proj
    var lb_cao = linear_backward(d_x_ca[], sv_lora_ca_att, w.ca_wo[], S, dim, dim, ctx)
    var d_ca_wo = lb_cao.d_w.to_host(ctx)
    var d_ca_bo = lb_cao.d_b.to_host(ctx)
    var ca_o_g = _lora_bwd_opt(lora.ca_o, d_ca_out_h, ca_att_h, S, dim, ctx)
    # ca_att input grad = base (lb_cao.d_x bf16) + LoRA d_x (host F32 -> bf16)
    var d_ca_att = add(lb_cao.d_x, _t16(ca_o_g.d_x.copy(), [S, dim], ctx), ctx)
    var d_ca_att4 = reshape(d_ca_att, [1, S, H, Dh], ctx)

    var lsv_ca_q_rms = _tbf16(saved.ca_q_rms.copy(), [1, S, H, Dh], ctx)
    var lsv_ca_k_rms = _tbf16(saved.ca_k_rms.copy(), [1, TXT, H, Dh], ctx)
    var lsv_ca_v = _tbf16(saved.ca_v.copy(), [1, TXT, H, Dh], ctx)
    var csb = sdpa_backward_rect[1, S, TXT, H, Dh](
        lsv_ca_q_rms, lsv_ca_k_rms, lsv_ca_v, d_ca_att4, scale, ctx,
    )
    var lsv_ca_q_pre = _tbf16(saved.ca_q_pre.copy(), [1, S, H, Dh], ctx)
    var rb_caq = rms_norm_backward(csb.d_q, lsv_ca_q_pre, w.ca_qn[], eps, ctx)
    var d_ca_qn = rb_caq.d_g.to_host(ctx)
    var lsv_ca_k_pre = _tbf16(saved.ca_k_pre.copy(), [1, TXT, H, Dh], ctx)
    var rb_cak = rms_norm_backward(csb.d_k, lsv_ca_k_pre, w.ca_kn[], eps, ctx)
    var d_ca_kn = rb_cak.d_g.to_host(ctx)
    var d_caq_flat = reshape(rb_caq.d_x, [S, dim], ctx)
    var d_cak_flat = reshape(rb_cak.d_x, [TXT, dim], ctx)
    var d_cav_flat = reshape(csb.d_v, [TXT, dim], ctx)
    var d_caq_h = d_caq_flat.to_host(ctx)
    var d_cak_h = d_cak_flat.to_host(ctx)
    var d_cav_h = d_cav_flat.to_host(ctx)

    var lb_caq = linear_backward(d_caq_flat, sv_lora_ca_n3, w.ca_wq[], S, dim, dim, ctx)
    var lb_cak = linear_backward(d_cak_flat, sv_lora_context, w.ca_wk[], TXT, dim, dim, ctx)
    var lb_cav = linear_backward(d_cav_flat, sv_lora_context, w.ca_wv[], TXT, dim, dim, ctx)
    var d_ca_wq = lb_caq.d_w.to_host(ctx)
    var d_ca_bq = lb_caq.d_b.to_host(ctx)
    var d_ca_wk = lb_cak.d_w.to_host(ctx)
    var d_ca_bk = lb_cak.d_b.to_host(ctx)
    var d_ca_wv = lb_cav.d_w.to_host(ctx)
    var d_ca_bv = lb_cav.d_b.to_host(ctx)
    var ca_q_g = _lora_bwd_opt(lora.ca_q, d_caq_h, n3_h, S, dim, ctx)
    var ca_k_g = _lora_bwd_opt(lora.ca_k, d_cak_h, context_h, TXT, dim, ctx)
    var ca_v_g = _lora_bwd_opt(lora.ca_v, d_cav_h, context_h, TXT, dim, ctx)
    # n3 grad = base d_x(caq) + LoRA d_x(ca_q)
    var d_n3_in = add(lb_caq.d_x, _t16(ca_q_g.d_x.copy(), [S, dim], ctx), ctx)
    # context grad = base(cak+cav) + LoRA(ca_k + ca_v)
    var d_ctx_base = add(lb_cak.d_x, lb_cav.d_x, ctx)
    var d_ctx_lora = add(
        _t16(ca_k_g.d_x.copy(), [TXT, dim], ctx), _t16(ca_v_g.d_x.copy(), [TXT, dim], ctx), ctx
    )
    var d_context_t = TArc(add(d_ctx_base, d_ctx_lora, ctx))

    var lsv_x_sa = _tbf16(saved.x_sa.copy(), [S, dim], ctx)
    var lnb_n3 = layer_norm_backward(d_n3_in, lsv_x_sa, w.n3_w[], eps, ctx)
    var d_n3_w = lnb_n3.d_g.to_host(ctx)
    var d_n3_b = lnb_n3.d_b.to_host(ctx)
    var d_x_sa = TArc(add(lnb_n3.d_x, d_x_ca[], ctx))

    # ════════════════ Self-attention backward (LoRA q/k/v/o) ════════════════
    var gate_sa_t = _t16(mv.gate_sa.copy(), [S, dim], ctx)
    var sa_out_rc = linear(sv_lora_sa_att, w.sa_wo[], Optional[Tensor](_clone_t(w.sa_bo[], ctx)), ctx)
    var gb_sa = wan_gate_residual_backward(d_x_sa[], sa_out_rc, gate_sa_t, ctx)
    var d_gate_sa = gb_sa.d_gate.to_host(ctx)
    var d_x_resid = TArc(gb_sa.d_x.clone(ctx))
    var d_sa_out_h = gb_sa.d_y.to_host(ctx)
    var lb_sao = linear_backward(gb_sa.d_y, sv_lora_sa_att, w.sa_wo[], S, dim, dim, ctx)
    var d_sa_wo = lb_sao.d_w.to_host(ctx)
    var d_sa_bo = lb_sao.d_b.to_host(ctx)
    var sa_o_g = _lora_bwd_opt(lora.sa_o, d_sa_out_h, sa_att_h, S, dim, ctx)
    var d_sa_att = add(lb_sao.d_x, _t16(sa_o_g.d_x.copy(), [S, dim], ctx), ctx)
    var d_sa_att4 = reshape(d_sa_att, [1, S, H, Dh], ctx)

    var lsv_sa_q_rope = _tbf16(saved.sa_q_rope.copy(), [1, S, H, Dh], ctx)
    var lsv_sa_k_rope = _tbf16(saved.sa_k_rope.copy(), [1, S, H, Dh], ctx)
    var lsv_sa_v = _tbf16(saved.sa_v.copy(), [1, S, H, Dh], ctx)
    var ssb = sdpa_backward[1, S, H, Dh](
        lsv_sa_q_rope, lsv_sa_k_rope, lsv_sa_v, d_sa_att4, scale, ctx,
    )
    # LOCAL F32 CAST for the F32-only rope_backward (sdpa grads are bf16).
    var ssb_dq_f32 = cast_tensor(ssb.d_q, STDtype.F32, ctx)
    var ssb_dk_f32 = cast_tensor(ssb.d_k, STDtype.F32, ctx)
    var d_q_rms = rope_backward(ssb_dq_f32, cos_e, sin_e, True, ctx)
    var d_k_rms = rope_backward(ssb_dk_f32, cos_e, sin_e, True, ctx)
    var lsv_sa_q_pre = _tbf16(saved.sa_q_pre.copy(), [1, S, H, Dh], ctx)
    var rb_saq = rms_norm_backward(d_q_rms, lsv_sa_q_pre, w.sa_qn[], eps, ctx)
    var d_sa_qn = rb_saq.d_g.to_host(ctx)
    var lsv_sa_k_pre = _tbf16(saved.sa_k_pre.copy(), [1, S, H, Dh], ctx)
    var rb_sak = rms_norm_backward(d_k_rms, lsv_sa_k_pre, w.sa_kn[], eps, ctx)
    var d_sa_kn = rb_sak.d_g.to_host(ctx)
    var d_saq_flat = reshape(rb_saq.d_x, [S, dim], ctx)
    var d_sak_flat = reshape(rb_sak.d_x, [S, dim], ctx)
    var d_sav_flat = reshape(ssb.d_v, [S, dim], ctx)
    var d_saq_h = d_saq_flat.to_host(ctx)
    var d_sak_h = d_sak_flat.to_host(ctx)
    var d_sav_h = d_sav_flat.to_host(ctx)

    var lb_saq = linear_backward(d_saq_flat, sv_lora_sa_in, w.sa_wq[], S, dim, dim, ctx)
    var lb_sak = linear_backward(d_sak_flat, sv_lora_sa_in, w.sa_wk[], S, dim, dim, ctx)
    var lb_sav = linear_backward(d_sav_flat, sv_lora_sa_in, w.sa_wv[], S, dim, dim, ctx)
    var d_sa_wq = lb_saq.d_w.to_host(ctx)
    var d_sa_bq = lb_saq.d_b.to_host(ctx)
    var d_sa_wk = lb_sak.d_w.to_host(ctx)
    var d_sa_bk = lb_sak.d_b.to_host(ctx)
    var d_sa_wv = lb_sav.d_w.to_host(ctx)
    var d_sa_bv = lb_sav.d_b.to_host(ctx)
    var sa_q_g = _lora_bwd_opt(lora.sa_q, d_saq_h, sa_in_h, S, dim, ctx)
    var sa_k_g = _lora_bwd_opt(lora.sa_k, d_sak_h, sa_in_h, S, dim, ctx)
    var sa_v_g = _lora_bwd_opt(lora.sa_v, d_sav_h, sa_in_h, S, dim, ctx)
    # sa_in grad = base(q+k+v) + LoRA(q+k+v)
    var d_sa_in_base = add(add(lb_saq.d_x, lb_sak.d_x, ctx), lb_sav.d_x, ctx)
    var d_sa_in_lora = add(
        add(_t16(sa_q_g.d_x.copy(), [S, dim], ctx), _t16(sa_k_g.d_x.copy(), [S, dim], ctx), ctx),
        _t16(sa_v_g.d_x.copy(), [S, dim], ctx), ctx,
    )
    var d_sa_in = TArc(add(d_sa_in_base, d_sa_in_lora, ctx))

    var scale_sa_t = _t16(mv.scale_sa.copy(), [S, dim], ctx)
    var lsv_sa_ln = _tbf16(saved.sa_ln.copy(), [S, dim], ctx)
    var mb_sa = wan_modulate_backward(d_sa_in[], lsv_sa_ln, scale_sa_t, ctx)
    var d_scale_sa = mb_sa.d_scale.to_host(ctx)
    var d_shift_sa = mb_sa.d_shift.to_host(ctx)
    var lsv_x = _tbf16(saved.x.copy(), [S, dim], ctx)
    var lnb_sa = layer_norm_backward_dx(mb_sa.d_ln, lsv_x, ones_t, eps, ctx)
    var d_x = add(lnb_sa, d_x_resid[], ctx)
    var d_x_h = d_x.to_host(ctx)
    var d_context_h = d_context_t[].to_host(ctx)

    var base = WanBlockGrads(
        d_x_h^, d_context_h^,
        d_sa_wq^, d_sa_wk^, d_sa_wv^, d_sa_wo^,
        d_sa_bq^, d_sa_bk^, d_sa_bv^, d_sa_bo^,
        d_sa_qn^, d_sa_kn^,
        d_ca_wq^, d_ca_wk^, d_ca_wv^, d_ca_wo^,
        d_ca_bq^, d_ca_bk^, d_ca_bv^, d_ca_bo^,
        d_ca_qn^, d_ca_kn^,
        d_n3_w^, d_n3_b^,
        d_ffn0_w^, d_ffn0_b^, d_ffn2_w^, d_ffn2_b^,
        d_shift_sa^, d_scale_sa^, d_gate_sa^,
        d_shift_ffn^, d_scale_ffn^, d_gate_ffn^,
    )
    return WanBlockLoraGrads(
        base^,
        sa_q_g.d_a.copy(), sa_q_g.d_b.copy(),
        sa_k_g.d_a.copy(), sa_k_g.d_b.copy(),
        sa_v_g.d_a.copy(), sa_v_g.d_b.copy(),
        sa_o_g.d_a.copy(), sa_o_g.d_b.copy(),
        ca_q_g.d_a.copy(), ca_q_g.d_b.copy(),
        ca_k_g.d_a.copy(), ca_k_g.d_b.copy(),
        ca_v_g.d_a.copy(), ca_v_g.d_b.copy(),
        ca_o_g.d_a.copy(), ca_o_g.d_b.copy(),
    )
