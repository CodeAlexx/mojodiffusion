# models/anima/block.mojo — Anima MiniTrainDIT block: forward (save activations)
# + hand-chained backward (training), in the proven style of
# serenitymojo/models/klein/single_block.mojo.
#
# This is the ONE genuinely-new compute unit Anima needs in the training path.
# The architecture (self-attn 3D-RoPE + cross-attn no-RoPE + GELU MLP, each with
# AdaLN-LoRA modulation, F32 gated residuals) is copied — NOT cross-imported —
# from the proven inference forward models/dit/anima_dit.mojo::_transformer_block
# and the Rust oracle inference-flame/src/models/anima.rs::transformer_block
# (read line-by-line 2026-06-01) per the ANIMA INDEPENDENCE rule.
#
# FORWARD GRAPH (F32 residual stream; mirrors anima.rs:458-511):
#   For each of the 3 sub-blocks (self_attn, cross_attn, mlp):
#     (shift, scale, gate) = adaln_mod(t_cond, base_adaln, mod1, mod2)
#         = chunk3( linear(linear(silu(t_cond), W1[256,2048]), W2[6144,256]) + base_adaln )
#     x_mod = (1+scale)*LayerNorm(x_bf16, eps=1e-6, no-affine) + shift   # AdaLN-pre
#     sub_out = self_attn(x_mod) | cross_attn(x_mod, ctx) | mlp(x_mod)
#     x_f32 = x_f32 + gate * sub_out      (gate broadcast over seq)
#   return bf16(x_f32)
#
#   self_attn: q,k,v = linear(x,Wq/Wk/Wv) [B,S,2048]; reshape [B,S,H,Dh];
#              q,k = rms_norm_per_head(.,q_norm/k_norm,1e-6);
#              q,k = rope_HALFSPLIT(q,k, cos[S,Dh/2], sin)  (3D-RoPE, NON-full,
#                    cos/sin broadcast over B and H — matches anima.rs:379-380
#                    rope_halfsplit_bf16 with cos/sin [1,1,S,Dh/2]);
#              out = sdpa(q,k,v, 1/sqrt(Dh)); out = linear(flat(out), Wout)
#   cross_attn: q = linear(x_mod,Wq) [B,S_img,2048]; k,v = linear(ctx,Wk/Wv)
#              [B,S_txt,1024->2048]; reshape; q,k = rms_norm_per_head; NO RoPE;
#              out = RECTANGULAR sdpa (S_q != S_kv, NO MASK — anima.rs:433
#              sdpa(...,None)); out = linear(flat, Wout)
#   mlp: linear(x,W1[8192,2048]) -> GELU(tanh) -> linear(.,W2[2048,8192])
#
# RoPE convention (VERIFIED anima.rs:378-380): "matches standalone trainer's
# interleaved=False" → HALF-SPLIT pairing (i, i+Dh/2), NON-full single-angle
# table (cos[i]==cos[i+half]). So the matching backward is
# rope_backward(interleaved=False), NOT rope_halfsplit_full_backward.
#
# AdaLN convention (VERIFIED flame-core bf16_ops.rs:1665-1673 needs_grad path):
# the pre-modulation uses layer_norm(x,[dim],None,None,eps) — LayerNorm with NO
# affine — THEN (1+scale)*normed+shift. The AdaLN-pre BACKWARD therefore composes
# from existing ops: modulate_backward (d_x,d_scale,d_shift) -> layer_norm_backward_dx
# (LN weight=1,bias=0; discard d_g/d_b). d_scale/d_shift then backprop through the
# per-block adaln_modulation Linear chain (linear_backward W2 -> linear_backward W1
# -> silu_backward), accumulating d_W into the modulation Linears.
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor move-only -> save structs Movable-only.
# This block is F32 throughout (parity-clean, matches the Klein single_block
# gate's F32 path). BF16 storage is the trainer's concern (the backward ops all
# cast up to F32 internally), not the parity unit's.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt as fsqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import gelu, silu
from serenitymojo.ops.tensor_algebra import reshape, reshape_owned, slice, add, mul, concat
from serenitymojo.ops.reduce import reduce_sum
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.models.dit.sdxl_attention import sdxl_sdpa
from serenitymojo.models.anima.weights import AnimaBlockWeights

# Backward arms (all pre-built + parity-gated in ops/).
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import (
    rms_norm_backward, layer_norm_backward, layer_norm_backward_dx,
    RmsNormBackward, LayerNormBackward,
)
from serenitymojo.ops.activation_backward import silu_backward, gelu_backward
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.rope_struct_backward import rope_backward
from serenitymojo.ops.attention_backward import (
    sdpa_backward, sdpa_backward_rect, SdpaGrads,
)

from serenitymojo.models.dit.anima_contract import (
    ANIMA_HIDDEN,        # 2048
    ANIMA_NUM_HEADS,     # 16
    ANIMA_HEAD_DIM,      # 128
)


comptime TArc = ArcPointer[Tensor]


# ── small host helpers ────────────────────────────────────────────────────────
def _ones(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(1.0)
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


# ═══════════════════════════════════════════════════════════════════════════
# FORWARD PRIMITIVES (each saves nothing; the block fwd threads activations)
# ═══════════════════════════════════════════════════════════════════════════

# ── AdaLN-pre modulation: out = (1+scale)*LayerNorm(x,no-affine,eps)+shift ────
# x [B,S,D] F32; shift/scale [B,D] F32. ln_ones/ln_zeros [D] resident.
def _adaln_pre(
    x_f32: Tensor, shift: Tensor, scale: Tensor,
    ln_ones: Tensor, ln_zeros: Tensor, eps: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var normed = layer_norm(x_f32, ln_ones, ln_zeros, eps, ctx)   # [B,S,D]
    var xsh = x_f32.shape()
    var B = xsh[0]; var D = xsh[2]
    var s3 = List[Int](); s3.append(B); s3.append(1); s3.append(D)
    var scale_3d = reshape(scale, s3.copy(), ctx)
    var shift_3d = reshape(shift, s3.copy(), ctx)
    var one = Tensor.from_host(_ones(B * D), [B, 1, D], STDtype.F32, ctx)
    var factor = add(scale_3d, one, ctx)              # (1+scale) [B,1,D]
    var scaled = mul(normed, factor, ctx)             # broadcast over S
    return add(scaled, shift_3d, ctx)


# ── AdaLN-LoRA modulation vectors: chunk3 of (W2(W1(silu(t_cond))) + base) ────
struct _Mods(Movable):
    var shift: Tensor
    var scale: Tensor
    var gate: Tensor

    def __init__(out self, var shift: Tensor, var scale: Tensor, var gate: Tensor):
        self.shift = shift^
        self.scale = scale^
        self.gate = gate^


def _adaln_mod(
    t_silu: Tensor,       # [B, 2048] = silu(t_cond) — passed precomputed
    base_adaln: Tensor,   # [B, 6144]
    mod1: Tensor,         # [256, 2048]
    mod2: Tensor,         # [6144, 256]
    ctx: DeviceContext,
) raises -> _Mods:
    var no_bias = Optional[Tensor](None)
    var h = linear(t_silu, mod1, no_bias, ctx)             # [B, 256]
    var no_bias2 = Optional[Tensor](None)
    var mod_out = linear(h, mod2, no_bias2, ctx)           # [B, 6144]
    var mod_added = add(mod_out, base_adaln, ctx)
    var shift = slice(mod_added, 1, 0, ANIMA_HIDDEN, ctx)
    var scale = slice(mod_added, 1, ANIMA_HIDDEN, ANIMA_HIDDEN, ctx)
    var gate = slice(mod_added, 1, 2 * ANIMA_HIDDEN, ANIMA_HIDDEN, ctx)
    return _Mods(shift^, scale^, gate^)


# ── rms_norm per head over last dim ([B,S,H,Dh] -> flatten [B*S*H, Dh]) ───────
def _rms_per_head(x: Tensor, norm_w: Tensor, ctx: DeviceContext) raises -> Tensor:
    var xsh = x.shape()
    var B = xsh[0]; var S = xsh[1]; var H = xsh[2]; var D = xsh[3]
    var flat = List[Int](); flat.append(B * S * H); flat.append(D)
    var f = reshape(x, flat.copy(), ctx)
    var n = rms_norm(f, norm_w, Float32(1e-6), ctx)
    var osh = List[Int](); osh.append(B); osh.append(S); osh.append(H); osh.append(D)
    return reshape(n, osh.copy(), ctx)


# ═══════════════════════════════════════════════════════════════════════════
# SAVED ACTIVATIONS + GRADS
# ═══════════════════════════════════════════════════════════════════════════

# Per-sub-block saved activations for the hand-chained backward.
struct _SubSaved(Copyable, Movable):
    var x_in: TArc       # [B,S,D] F32   residual-stream input to this sub-block
    var ln: TArc         # [B,S,D] F32   layer_norm(x_in) (no affine)
    var x_mod: TArc      # [B,S,D] F32   adaln-pre output  (sub-block input)
    var shift: TArc      # [B,D]
    var scale: TArc      # [B,D]
    var gate: TArc       # [B,D]
    var t_silu: TArc     # [B,2048]  silu(t_cond) (shared, but stored per sub for chain)
    var mod_h: TArc      # [B,256]   linear(t_silu, mod1)
    var sub_out: TArc    # [B,Sq,D]  sub-block output (pre-gate)

    def __init__(
        out self, var x_in: TArc, var ln: TArc, var x_mod: TArc,
        var shift: TArc, var scale: TArc, var gate: TArc,
        var t_silu: TArc, var mod_h: TArc, var sub_out: TArc,
    ):
        self.x_in = x_in^; self.ln = ln^; self.x_mod = x_mod^
        self.shift = shift^; self.scale = scale^; self.gate = gate^
        self.t_silu = t_silu^; self.mod_h = mod_h^; self.sub_out = sub_out^


# self/cross attention internal activations (for the attention backward).
#   q_sdpa/k_sdpa: the q/k that ENTER sdpa (post-rope for self, post-rms for cross)
#   v4:           value [B,Skv,H,Dh]
#   q_pre/k_pre:  q/k BEFORE rms_norm (post-proj reshape) — rms backward needs these
#   q_rms/k_rms:  rms_norm output (self-attn rope backward feeds into rms backward;
#                 for self these are the rope INPUT, for cross == q_sdpa/k_sdpa)
#   attn_flat:    [B,Sq,D] reshape(sdpa(...)) — input to output_proj
#   q_ctx_in:     attention q-side input (sa_xmod for self, ca_xmod for cross)
#   kv_ctx_in:    attention k/v-side input (sa_xmod for self, context for cross)
struct _AttnSaved(Copyable, Movable):
    var q_sdpa: TArc     # [B,Sq,H,Dh]
    var k_sdpa: TArc     # [B,Skv,H,Dh]
    var v4: TArc         # [B,Skv,H,Dh]
    var q_pre: TArc      # [B,Sq,H,Dh]
    var k_pre: TArc      # [B,Skv,H,Dh]
    var q_rms: TArc      # [B,Sq,H,Dh]
    var k_rms: TArc      # [B,Skv,H,Dh]
    var attn_flat: TArc  # [B,Sq,D]
    var q_ctx_in: TArc   # [B,Sq,Din_q]
    var kv_ctx_in: TArc  # [B,Skv,Din_kv]

    def __init__(
        out self, var q_sdpa: TArc, var k_sdpa: TArc, var v4: TArc,
        var q_pre: TArc, var k_pre: TArc, var q_rms: TArc, var k_rms: TArc,
        var attn_flat: TArc, var q_ctx_in: TArc, var kv_ctx_in: TArc,
    ):
        self.q_sdpa = q_sdpa^; self.k_sdpa = k_sdpa^; self.v4 = v4^
        self.q_pre = q_pre^; self.k_pre = k_pre^
        self.q_rms = q_rms^; self.k_rms = k_rms^
        self.attn_flat = attn_flat^; self.q_ctx_in = q_ctx_in^; self.kv_ctx_in = kv_ctx_in^


struct AnimaBlockSaved(Copyable, Movable):
    var sa: _SubSaved
    var ca: _SubSaved
    var mlp: _SubSaved
    var sa_attn: _AttnSaved
    var ca_attn: _AttnSaved
    var mlp_h: TArc      # [B,S,F]   pre-GELU hidden  (mlp.layer1 output)
    var mlp_ha: TArc     # [B,S,F]   GELU(h)

    def __init__(
        out self, var sa: _SubSaved, var ca: _SubSaved, var mlp: _SubSaved,
        var sa_attn: _AttnSaved, var ca_attn: _AttnSaved,
        var mlp_h: TArc, var mlp_ha: TArc,
    ):
        self.sa = sa^; self.ca = ca^; self.mlp = mlp^
        self.sa_attn = sa_attn^; self.ca_attn = ca_attn^
        self.mlp_h = mlp_h^; self.mlp_ha = mlp_ha^


struct AnimaBlockForward(Movable):
    var out: List[Float32]   # [B,S,D] F32 (boundary readback)
    var saved: AnimaBlockSaved

    def __init__(out self, var out: List[Float32], var saved: AnimaBlockSaved):
        self.out = out^
        self.saved = saved^


# Per-block grads: d_x (input), all 20 weight grads, and the t_silu grad
# (so the caller can chain into t_cond). LoRA grads are a separate phase.
struct AnimaBlockGrads(Copyable, Movable):
    var d_x: List[Float32]          # [B,S,D]
    var d_t_silu: List[Float32]     # [B,2048] grad into silu(t_cond)
    # self-attn weights
    var d_sa_q: List[Float32]
    var d_sa_k: List[Float32]
    var d_sa_v: List[Float32]
    var d_sa_out: List[Float32]
    var d_sa_qn: List[Float32]
    var d_sa_kn: List[Float32]
    # cross-attn weights
    var d_ca_q: List[Float32]
    var d_ca_k: List[Float32]
    var d_ca_v: List[Float32]
    var d_ca_out: List[Float32]
    var d_ca_qn: List[Float32]
    var d_ca_kn: List[Float32]
    # mlp
    var d_mlp1: List[Float32]
    var d_mlp2: List[Float32]
    # AdaLN-LoRA-256 modulation weights (the Anima-specific arm)
    var d_sa_mod1: List[Float32]
    var d_sa_mod2: List[Float32]
    var d_ca_mod1: List[Float32]
    var d_ca_mod2: List[Float32]
    var d_mlp_mod1: List[Float32]
    var d_mlp_mod2: List[Float32]

    def __init__(
        out self,
        var d_x: List[Float32], var d_t_silu: List[Float32],
        var d_sa_q: List[Float32], var d_sa_k: List[Float32], var d_sa_v: List[Float32],
        var d_sa_out: List[Float32], var d_sa_qn: List[Float32], var d_sa_kn: List[Float32],
        var d_ca_q: List[Float32], var d_ca_k: List[Float32], var d_ca_v: List[Float32],
        var d_ca_out: List[Float32], var d_ca_qn: List[Float32], var d_ca_kn: List[Float32],
        var d_mlp1: List[Float32], var d_mlp2: List[Float32],
        var d_sa_mod1: List[Float32], var d_sa_mod2: List[Float32],
        var d_ca_mod1: List[Float32], var d_ca_mod2: List[Float32],
        var d_mlp_mod1: List[Float32], var d_mlp_mod2: List[Float32],
    ):
        self.d_x = d_x^; self.d_t_silu = d_t_silu^
        self.d_sa_q = d_sa_q^; self.d_sa_k = d_sa_k^; self.d_sa_v = d_sa_v^
        self.d_sa_out = d_sa_out^; self.d_sa_qn = d_sa_qn^; self.d_sa_kn = d_sa_kn^
        self.d_ca_q = d_ca_q^; self.d_ca_k = d_ca_k^; self.d_ca_v = d_ca_v^
        self.d_ca_out = d_ca_out^; self.d_ca_qn = d_ca_qn^; self.d_ca_kn = d_ca_kn^
        self.d_mlp1 = d_mlp1^; self.d_mlp2 = d_mlp2^
        self.d_sa_mod1 = d_sa_mod1^; self.d_sa_mod2 = d_sa_mod2^
        self.d_ca_mod1 = d_ca_mod1^; self.d_ca_mod2 = d_ca_mod2^
        self.d_mlp_mod1 = d_mlp_mod1^; self.d_mlp_mod2 = d_mlp_mod2^


# ═══════════════════════════════════════════════════════════════════════════
# FULL BLOCK FORWARD (F32, saves activations) — parity unit
# ═══════════════════════════════════════════════════════════════════════════
# cos/sin: 3D-RoPE half-split tables, ALREADY EXPANDED to [B*S_img*H, Dh/2] F32
#   (one row per (b,s,h); broadcast of the per-position [S_img,Dh/2] table over H,
#   matching anima.rs rope_halfsplit_bf16 cos/sin [1,1,S,Dh/2]).
# t_cond [B,2048] F32; base_adaln [B,6144] F32; context [B,S_txt,1024] F32.
def anima_block_forward[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    x: List[Float32],          # [B,S_img,D] F32 residual-stream input
    t_cond: List[Float32],     # [B,2048] F32
    base_adaln: List[Float32], # [B,6144] F32
    context: List[Float32],    # [B,S_txt,1024] F32  (frozen text context)
    w: AnimaBlockWeights,
    cos: Tensor, sin: Tensor,  # [B*S_img*H, Dh/2] F32
    B: Int, D: Int, JOINT: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> AnimaBlockForward:
    var ln_ones = Tensor.from_host(_ones(D), [D], STDtype.F32, ctx)
    var ln_zeros = Tensor.from_host(_zeros(D), [D], STDtype.F32, ctx)
    var sa_scale = Float32(1.0) / fsqrt(Float32(Dh))

    var x_f32 = Tensor.from_host(x, [B, S_IMG, D], STDtype.F32, ctx)
    var t_cond_t = Tensor.from_host(t_cond, [B, ANIMA_HIDDEN], STDtype.F32, ctx)
    var base_t = Tensor.from_host(base_adaln, [B, 3 * ANIMA_HIDDEN], STDtype.F32, ctx)
    var ctx_t = Tensor.from_host(context, [B, S_TXT, JOINT], STDtype.F32, ctx)
    var t_silu = silu(t_cond_t, ctx)   # [B,2048], shared across the 3 sub-blocks

    # ── sub-block 1: SELF-ATTENTION ──────────────────────────────────────────
    var sa_h = linear(t_silu, w.sa_mod1[], Optional[Tensor](None), ctx)   # [B,256]
    var sa_modout = linear(sa_h, w.sa_mod2[], Optional[Tensor](None), ctx)
    var sa_added = add(sa_modout, base_t, ctx)
    var sa_shift = slice(sa_added, 1, 0, D, ctx)
    var sa_scalev = slice(sa_added, 1, D, D, ctx)
    var sa_gate = slice(sa_added, 1, 2 * D, D, ctx)

    var sa_ln = layer_norm(x_f32, ln_ones, ln_zeros, eps, ctx)
    var sa_xmod = _adaln_pre(x_f32, sa_shift, sa_scalev, ln_ones, ln_zeros, eps, ctx)

    # self-attn body
    var sa_q = linear(sa_xmod, w.sa_q[], Optional[Tensor](None), ctx)
    var sa_k = linear(sa_xmod, w.sa_k[], Optional[Tensor](None), ctx)
    var sa_v = linear(sa_xmod, w.sa_v[], Optional[Tensor](None), ctx)
    var sh4 = List[Int](); sh4.append(B); sh4.append(S_IMG); sh4.append(H); sh4.append(Dh)
    var sa_q4 = reshape(sa_q, sh4.copy(), ctx)
    var sa_k4 = reshape(sa_k, sh4.copy(), ctx)
    var sa_v4 = reshape(sa_v, sh4.copy(), ctx)
    var sa_qrms = _rms_per_head(sa_q4, w.sa_qn[], ctx)
    var sa_krms = _rms_per_head(sa_k4, w.sa_kn[], ctx)
    var sa_qrope = rope_halfsplit(sa_qrms, cos, sin, ctx)
    var sa_krope = rope_halfsplit(sa_krms, cos, sin, ctx)
    var sa_att = sdpa_nomask[1, S_IMG, H, Dh](sa_qrope, sa_krope, sa_v4, sa_scale, ctx)
    var saf = List[Int](); saf.append(B); saf.append(S_IMG); saf.append(D)
    var sa_attflat = reshape(sa_att, saf.copy(), ctx)
    var sa_out = linear(sa_attflat, w.sa_out[], Optional[Tensor](None), ctx)  # [B,S_img,D]

    # gated residual: x_f32 += gate * sa_out  (gate [B,D] broadcast over seq)
    var g3 = List[Int](); g3.append(B); g3.append(1); g3.append(D)
    var sa_gate3 = reshape(sa_gate, g3.copy(), ctx)
    var sa_gated = mul(sa_out, sa_gate3, ctx)
    var x_after_sa = add(x_f32, sa_gated, ctx)

    # ── sub-block 2: CROSS-ATTENTION (no RoPE, rectangular SDPA, no mask) ─────
    var ca_h = linear(t_silu, w.ca_mod1[], Optional[Tensor](None), ctx)
    var ca_modout = linear(ca_h, w.ca_mod2[], Optional[Tensor](None), ctx)
    var ca_added = add(ca_modout, base_t, ctx)
    var ca_shift = slice(ca_added, 1, 0, D, ctx)
    var ca_scalev = slice(ca_added, 1, D, D, ctx)
    var ca_gate = slice(ca_added, 1, 2 * D, D, ctx)

    var ca_ln = layer_norm(x_after_sa, ln_ones, ln_zeros, eps, ctx)
    var ca_xmod = _adaln_pre(x_after_sa, ca_shift, ca_scalev, ln_ones, ln_zeros, eps, ctx)

    var ca_q = linear(ca_xmod, w.ca_q[], Optional[Tensor](None), ctx)      # [B,S_img,2048]
    var ca_k = linear(ctx_t, w.ca_k[], Optional[Tensor](None), ctx)        # [B,S_txt,2048]
    var ca_v = linear(ctx_t, w.ca_v[], Optional[Tensor](None), ctx)        # [B,S_txt,2048]
    var caq4s = List[Int](); caq4s.append(B); caq4s.append(S_IMG); caq4s.append(H); caq4s.append(Dh)
    var cak4s = List[Int](); cak4s.append(B); cak4s.append(S_TXT); cak4s.append(H); cak4s.append(Dh)
    var ca_q4 = reshape(ca_q, caq4s.copy(), ctx)
    var ca_k4 = reshape(ca_k, cak4s.copy(), ctx)
    var ca_v4 = reshape(ca_v, cak4s.copy(), ctx)
    var ca_qrms = _rms_per_head(ca_q4, w.ca_qn[], ctx)
    var ca_krms = _rms_per_head(ca_k4, w.ca_kn[], ctx)
    var ca_att = sdxl_sdpa[1, S_IMG, S_TXT, H, Dh](ca_qrms, ca_krms, ca_v4, sa_scale, ctx)
    var caf = List[Int](); caf.append(B); caf.append(S_IMG); caf.append(D)
    var ca_attflat = reshape(ca_att, caf.copy(), ctx)
    var ca_out = linear(ca_attflat, w.ca_out[], Optional[Tensor](None), ctx)

    var ca_gate3 = reshape(ca_gate, g3.copy(), ctx)
    var ca_gated = mul(ca_out, ca_gate3, ctx)
    var x_after_ca = add(x_after_sa, ca_gated, ctx)

    # ── sub-block 3: MLP (GELU) ──────────────────────────────────────────────
    var mlp_h_ = linear(t_silu, w.mlp_mod1[], Optional[Tensor](None), ctx)
    var mlp_modout = linear(mlp_h_, w.mlp_mod2[], Optional[Tensor](None), ctx)
    var mlp_added = add(mlp_modout, base_t, ctx)
    var mlp_shift = slice(mlp_added, 1, 0, D, ctx)
    var mlp_scalev = slice(mlp_added, 1, D, D, ctx)
    var mlp_gate = slice(mlp_added, 1, 2 * D, D, ctx)

    var mlp_ln = layer_norm(x_after_ca, ln_ones, ln_zeros, eps, ctx)
    var mlp_xmod = _adaln_pre(x_after_ca, mlp_shift, mlp_scalev, ln_ones, ln_zeros, eps, ctx)

    var mlp_h1 = linear(mlp_xmod, w.mlp1[], Optional[Tensor](None), ctx)   # [B,S,F]
    var mlp_ha = gelu(mlp_h1, ctx)
    var mlp_out = linear(mlp_ha, w.mlp2[], Optional[Tensor](None), ctx)    # [B,S,D]

    var mlp_gate3 = reshape(mlp_gate, g3.copy(), ctx)
    var mlp_gated = mul(mlp_out, mlp_gate3, ctx)
    var x_final = add(x_after_ca, mlp_gated, ctx)

    var out_host = x_final.to_host(ctx)

    # TArc handles (refcount bumps; the same device buffer is shared between the
    # _SubSaved x_mod slot and the _AttnSaved q/kv ctx_in slots — both BORROW it).
    var sa_xmod_a = TArc(sa_xmod^)
    var ca_xmod_a = TArc(ca_xmod^)
    var mlp_xmod_a = TArc(mlp_xmod^)
    var t_silu_a = TArc(t_silu^)

    var sa_sub = _SubSaved(
        TArc(x_f32^), TArc(sa_ln^), sa_xmod_a.copy(),
        TArc(sa_shift^), TArc(sa_scalev^), TArc(sa_gate^),
        t_silu_a.copy(), TArc(sa_h^), TArc(sa_out^),
    )
    var ca_sub = _SubSaved(
        TArc(x_after_sa^), TArc(ca_ln^), ca_xmod_a.copy(),
        TArc(ca_shift^), TArc(ca_scalev^), TArc(ca_gate^),
        t_silu_a.copy(), TArc(ca_h^), TArc(ca_out^),
    )
    var mlp_sub = _SubSaved(
        TArc(x_after_ca^), TArc(mlp_ln^), mlp_xmod_a.copy(),
        TArc(mlp_shift^), TArc(mlp_scalev^), TArc(mlp_gate^),
        t_silu_a.copy(), TArc(mlp_h_^), TArc(mlp_out^),
    )
    # self-attn: q/kv ctx are BOTH sa_xmod; q_sdpa/k_sdpa are post-rope.
    var ctx_t_a = TArc(ctx_t^)
    var sa_attn = _AttnSaved(
        TArc(sa_qrope^), TArc(sa_krope^), TArc(sa_v4^),
        TArc(sa_q4^), TArc(sa_k4^), TArc(sa_qrms^), TArc(sa_krms^),
        TArc(sa_attflat^), sa_xmod_a.copy(), sa_xmod_a.copy(),
    )
    # cross-attn: q ctx is ca_xmod, kv ctx is the (frozen) text context. NO rope,
    # so q_sdpa == q_rms and k_sdpa == k_rms.
    var ca_qrms_a = TArc(ca_qrms^)
    var ca_krms_a = TArc(ca_krms^)
    var ca_attn = _AttnSaved(
        ca_qrms_a.copy(), ca_krms_a.copy(), TArc(ca_v4^),
        TArc(ca_q4^), TArc(ca_k4^), ca_qrms_a.copy(), ca_krms_a.copy(),
        TArc(ca_attflat^), ca_xmod_a.copy(), ctx_t_a.copy(),
    )
    var saved = AnimaBlockSaved(
        sa_sub^, ca_sub^, mlp_sub^, sa_attn^, ca_attn^, TArc(mlp_h1^), TArc(mlp_ha^),
    )
    return AnimaBlockForward(out_host^, saved^)


# ═══════════════════════════════════════════════════════════════════════════
# BACKWARD HELPERS
# ═══════════════════════════════════════════════════════════════════════════
# Mojo grad structs are Movable with Tensor fields; a Tensor field CANNOT be
# moved out (partial-move of a destructible struct is rejected). So the backward
# is INLINED in anima_block_backward: each grad struct stays a live local and its
# fields are only ever BORROWED into the next op (which returns a fresh owned
# Tensor) or read via .to_host(). This mirrors the Klein single_block backward.
#
# d_gate / d_sub_out helpers return op-produced OWNED tensors (no field moves):
struct _GateGrads(Movable):
    var d_gate: Tensor       # [B,D]
    var d_sub_out: Tensor    # [B,Sq,D]

    def __init__(out self, var d_gate: Tensor, var d_sub_out: Tensor):
        self.d_gate = d_gate^; self.d_sub_out = d_sub_out^


def _gate_residual_bwd(
    d_x_out: Tensor, gate: Tensor, sub_out: Tensor,
    B: Int, Sq: Int, D: Int, ctx: DeviceContext,
) raises -> _GateGrads:
    # d_gate = sum_s (d_x_out ⊙ sub_out)  -> [B,D]
    var prod = mul(d_x_out, sub_out, ctx)            # [B,Sq,D]
    var d_gate = reduce_sum(prod, _dim1(), False, ctx)  # [B,D]
    # d_sub_out = d_x_out ⊙ gate (gate [B,D] broadcast over S via [B,1,D]).
    var g3 = List[Int](); g3.append(B); g3.append(1); g3.append(D)
    var gate3 = reshape(gate, g3.copy(), ctx)
    var d_sub_out = mul(d_x_out, gate3, ctx)
    return _GateGrads(d_gate^, d_sub_out^)


def _dim1() -> List[Int]:
    var d = List[Int](); d.append(1); return d^


# ═══════════════════════════════════════════════════════════════════════════
# FULL BLOCK BACKWARD (hand-chained, F32) — parity unit
# ═══════════════════════════════════════════════════════════════════════════
# d_out: upstream grad of the block output [B,S_img,D] (host list).
# Returns d_x (into the block input), d_t_silu (into shared silu(t_cond)), and
# all 20 weight grads + 6 AdaLN-LoRA-256 mod-weight grads.
def anima_block_backward[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    d_out: List[Float32],      # [B,S_img,D]
    saved: AnimaBlockSaved,
    w: AnimaBlockWeights,
    cos: Tensor, sin: Tensor,  # [B*S_img*H, Dh/2] F32 (same tables as forward)
    B: Int, D: Int, JOINT: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> AnimaBlockGrads:
    var ln_ones = Tensor.from_host(_ones(D), [D], STDtype.F32, ctx)
    var sa_scale = Float32(1.0) / fsqrt(Float32(Dh))

    var d_x = Tensor.from_host(d_out, [B, S_IMG, D], STDtype.F32, ctx)
    # d_t_silu accumulates over the 3 adaln-mod chains.
    var d_t_silu_acc = Tensor.from_host(_zeros(B * ANIMA_HIDDEN), [B, ANIMA_HIDDEN], STDtype.F32, ctx)

    # ─────────────────────────────── MLP sub-block (last in forward) ──────────
    var mlp_gg = _gate_residual_bwd(d_x, saved.mlp.gate[], saved.mlp.sub_out[], B, S_IMG, D, ctx)
    # d_x_in for this sub-block (residual passthrough) starts as d_x (== d_x_out)
    var d_x_after_ca = d_x^   # residual branch d_x passes straight through
    # mlp body backward: mlp_out = linear(gelu(linear(x_mod,W1)),W2)
    var mlp_lb2 = linear_backward(mlp_gg.d_sub_out, saved.mlp_ha[], w.mlp2[], B * S_IMG, F, D, ctx)
    var d_mlp2 = mlp_lb2.d_w.to_host(ctx)
    var d_mlp_ha = gelu_backward(mlp_lb2.d_x, saved.mlp_h[], ctx)
    var mlp_lb1 = linear_backward(d_mlp_ha, saved.mlp.x_mod[], w.mlp1[], B * S_IMG, D, F, ctx)
    var d_mlp1 = mlp_lb1.d_w.to_host(ctx)
    # adaln-pre backward: x_mod=(1+scale)*LN(x_in)+shift. mlp_lb1.d_x = d_x_mod.
    var mlp_mb = modulate_backward(mlp_lb1.d_x, saved.mlp.ln[], saved.mlp.scale[], ctx)
    var mlp_dxin = layer_norm_backward_dx(mlp_mb.d_x, saved.mlp.x_in[], ln_ones, eps, ctx)
    d_x_after_ca = add(d_x_after_ca, mlp_dxin, ctx)
    # adaln-mod backward: concat(d_shift, d_scale, d_gate) -> W2 bwd -> W1 bwd.
    var mlp_dadd = concat(1, ctx, mlp_mb.d_shift, mlp_mb.d_scale, mlp_gg.d_gate)
    var mlp_mlb2 = linear_backward(mlp_dadd, saved.mlp.mod_h[], w.mlp_mod2[], B, 256, 3 * D, ctx)
    var d_mlp_mod2 = mlp_mlb2.d_w.to_host(ctx)
    var mlp_mlb1 = linear_backward(mlp_mlb2.d_x, saved.mlp.t_silu[], w.mlp_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    var d_mlp_mod1 = mlp_mlb1.d_w.to_host(ctx)
    d_t_silu_acc = add(d_t_silu_acc, mlp_mlb1.d_x, ctx)

    # ─────────────────────────────── CROSS-ATTENTION sub-block ────────────────
    var ca_gg = _gate_residual_bwd(d_x_after_ca, saved.ca.gate[], saved.ca.sub_out[], B, S_IMG, D, ctx)
    var d_x_after_sa = d_x_after_ca^   # residual passthrough
    # ca body: out = linear(attn_flat, W_out); attn = sdxl_sdpa(q_rms,k_rms,v)
    var ca_lbout = linear_backward(ca_gg.d_sub_out, saved.ca_attn.attn_flat[], w.ca_out[], B * S_IMG, D, D, ctx)
    var d_ca_out = ca_lbout.d_w.to_host(ctx)
    # d_attn_flat [B,S_img,D] -> [B,S_img,H,Dh]
    var ca_af4 = List[Int](); ca_af4.append(B); ca_af4.append(S_IMG); ca_af4.append(H); ca_af4.append(Dh)
    var d_ca_att4 = reshape(ca_lbout.d_x, ca_af4.copy(), ctx)
    var ca_sb = sdpa_backward_rect[1, S_IMG, S_TXT, H, Dh](
        saved.ca_attn.q_sdpa[], saved.ca_attn.k_sdpa[], saved.ca_attn.v4[],
        d_ca_att4, sa_scale, ctx,
    )
    # NO rope on cross-attn: d_q_sdpa == d_q_rms, d_k_sdpa == d_k_rms.
    # rms backward for q (S_img) and k (S_txt).
    var ca_q_flat = List[Int](); ca_q_flat.append(B * S_IMG * H); ca_q_flat.append(Dh)
    var ca_k_flat = List[Int](); ca_k_flat.append(B * S_TXT * H); ca_k_flat.append(Dh)
    var d_ca_q_rms_f = reshape(ca_sb.d_q, ca_q_flat.copy(), ctx)
    var d_ca_k_rms_f = reshape(ca_sb.d_k, ca_k_flat.copy(), ctx)
    var ca_qpre_f = reshape(saved.ca_attn.q_pre[], ca_q_flat.copy(), ctx)
    var ca_kpre_f = reshape(saved.ca_attn.k_pre[], ca_k_flat.copy(), ctx)
    var ca_rbq = rms_norm_backward(d_ca_q_rms_f, ca_qpre_f, w.ca_qn[], Float32(1e-6), ctx)
    var ca_rbk = rms_norm_backward(d_ca_k_rms_f, ca_kpre_f, w.ca_kn[], Float32(1e-6), ctx)
    var d_ca_qn = ca_rbq.d_g.to_host(ctx)
    var d_ca_kn = ca_rbk.d_g.to_host(ctx)
    # reshape d_q_pre [B*S*H,Dh] -> [B,S,D] for q_proj backward; same for k/v.
    var ca_q3 = List[Int](); ca_q3.append(B); ca_q3.append(S_IMG); ca_q3.append(D)
    var ca_kv3 = List[Int](); ca_kv3.append(B); ca_kv3.append(S_TXT); ca_kv3.append(D)
    var d_ca_q_proj = reshape(ca_rbq.d_x, ca_q3.copy(), ctx)
    var d_ca_k_proj = reshape(ca_rbk.d_x, ca_kv3.copy(), ctx)
    # d_v: sdpa d_v is [B,S_txt,H,Dh] -> [B,S_txt,D]
    var d_ca_v_proj = reshape(ca_sb.d_v, ca_kv3.copy(), ctx)
    # q_proj: ca_q = linear(ca_xmod, W_q). k/v_proj: linear(context, W_k/W_v).
    var ca_lbq = linear_backward(d_ca_q_proj, saved.ca_attn.q_ctx_in[], w.ca_q[], B * S_IMG, D, D, ctx)
    var ca_lbk = linear_backward(d_ca_k_proj, saved.ca_attn.kv_ctx_in[], w.ca_k[], B * S_TXT, JOINT, D, ctx)
    var ca_lbv = linear_backward(d_ca_v_proj, saved.ca_attn.kv_ctx_in[], w.ca_v[], B * S_TXT, JOINT, D, ctx)
    var d_ca_q = ca_lbq.d_w.to_host(ctx)
    var d_ca_k = ca_lbk.d_w.to_host(ctx)
    var d_ca_v = ca_lbv.d_w.to_host(ctx)
    # d_ca_xmod = ca_lbq.d_x (q-side only; k/v come from frozen context -> d_x
    # there flows into context which is not a block input/leaf -> discard).
    var ca_mb = modulate_backward(ca_lbq.d_x, saved.ca.ln[], saved.ca.scale[], ctx)
    var ca_dxin = layer_norm_backward_dx(ca_mb.d_x, saved.ca.x_in[], ln_ones, eps, ctx)
    d_x_after_sa = add(d_x_after_sa, ca_dxin, ctx)
    var ca_dadd = concat(1, ctx, ca_mb.d_shift, ca_mb.d_scale, ca_gg.d_gate)
    var ca_mlb2 = linear_backward(ca_dadd, saved.ca.mod_h[], w.ca_mod2[], B, 256, 3 * D, ctx)
    var d_ca_mod2 = ca_mlb2.d_w.to_host(ctx)
    var ca_mlb1 = linear_backward(ca_mlb2.d_x, saved.ca.t_silu[], w.ca_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    var d_ca_mod1 = ca_mlb1.d_w.to_host(ctx)
    d_t_silu_acc = add(d_t_silu_acc, ca_mlb1.d_x, ctx)

    # ─────────────────────────────── SELF-ATTENTION sub-block ─────────────────
    var sa_gg = _gate_residual_bwd(d_x_after_sa, saved.sa.gate[], saved.sa.sub_out[], B, S_IMG, D, ctx)
    var d_x_in = d_x_after_sa^   # residual passthrough into block input
    var sa_lbout = linear_backward(sa_gg.d_sub_out, saved.sa_attn.attn_flat[], w.sa_out[], B * S_IMG, D, D, ctx)
    var d_sa_out = sa_lbout.d_w.to_host(ctx)
    var sa_af4 = List[Int](); sa_af4.append(B); sa_af4.append(S_IMG); sa_af4.append(H); sa_af4.append(Dh)
    var d_sa_att4 = reshape(sa_lbout.d_x, sa_af4.copy(), ctx)
    var sa_sb = sdpa_backward[1, S_IMG, H, Dh](
        saved.sa_attn.q_sdpa[], saved.sa_attn.k_sdpa[], saved.sa_attn.v4[],
        d_sa_att4, sa_scale, ctx,
    )
    # rope backward (halfsplit, NON-full -> interleaved=False). d_q_sdpa is the
    # grad of post-rope q; rope_backward returns grad of pre-rope (== rms output).
    var sa_q_flat = List[Int](); sa_q_flat.append(B * S_IMG * H); sa_q_flat.append(Dh)
    var d_sa_q_rope_f = reshape(sa_sb.d_q, sa_q_flat.copy(), ctx)
    var d_sa_k_rope_f = reshape(sa_sb.d_k, sa_q_flat.copy(), ctx)
    var d_sa_q_rms_f = rope_backward(d_sa_q_rope_f, cos, sin, False, ctx)
    var d_sa_k_rms_f = rope_backward(d_sa_k_rope_f, cos, sin, False, ctx)
    var sa_qpre_f = reshape(saved.sa_attn.q_pre[], sa_q_flat.copy(), ctx)
    var sa_kpre_f = reshape(saved.sa_attn.k_pre[], sa_q_flat.copy(), ctx)
    var sa_rbq = rms_norm_backward(d_sa_q_rms_f, sa_qpre_f, w.sa_qn[], Float32(1e-6), ctx)
    var sa_rbk = rms_norm_backward(d_sa_k_rms_f, sa_kpre_f, w.sa_kn[], Float32(1e-6), ctx)
    var d_sa_qn = sa_rbq.d_g.to_host(ctx)
    var d_sa_kn = sa_rbk.d_g.to_host(ctx)
    var sa3 = List[Int](); sa3.append(B); sa3.append(S_IMG); sa3.append(D)
    var d_sa_q_proj = reshape(sa_rbq.d_x, sa3.copy(), ctx)
    var d_sa_k_proj = reshape(sa_rbk.d_x, sa3.copy(), ctx)
    var d_sa_v_proj = reshape(sa_sb.d_v, sa3.copy(), ctx)
    # q/k/v all from sa_xmod -> sum the 3 d_x contributions.
    var sa_lbq = linear_backward(d_sa_q_proj, saved.sa_attn.q_ctx_in[], w.sa_q[], B * S_IMG, D, D, ctx)
    var sa_lbk = linear_backward(d_sa_k_proj, saved.sa_attn.q_ctx_in[], w.sa_k[], B * S_IMG, D, D, ctx)
    var sa_lbv = linear_backward(d_sa_v_proj, saved.sa_attn.q_ctx_in[], w.sa_v[], B * S_IMG, D, D, ctx)
    var d_sa_q = sa_lbq.d_w.to_host(ctx)
    var d_sa_k = sa_lbk.d_w.to_host(ctx)
    var d_sa_v = sa_lbv.d_w.to_host(ctx)
    var d_sa_xmod = add(add(sa_lbq.d_x, sa_lbk.d_x, ctx), sa_lbv.d_x, ctx)
    var sa_mb = modulate_backward(d_sa_xmod, saved.sa.ln[], saved.sa.scale[], ctx)
    var sa_dxin = layer_norm_backward_dx(sa_mb.d_x, saved.sa.x_in[], ln_ones, eps, ctx)
    d_x_in = add(d_x_in, sa_dxin, ctx)
    var sa_dadd = concat(1, ctx, sa_mb.d_shift, sa_mb.d_scale, sa_gg.d_gate)
    var sa_mlb2 = linear_backward(sa_dadd, saved.sa.mod_h[], w.sa_mod2[], B, 256, 3 * D, ctx)
    var d_sa_mod2 = sa_mlb2.d_w.to_host(ctx)
    var sa_mlb1 = linear_backward(sa_mlb2.d_x, saved.sa.t_silu[], w.sa_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    var d_sa_mod1 = sa_mlb1.d_w.to_host(ctx)
    d_t_silu_acc = add(d_t_silu_acc, sa_mlb1.d_x, ctx)

    var d_x_host = d_x_in.to_host(ctx)
    var d_t_silu_host = d_t_silu_acc.to_host(ctx)

    return AnimaBlockGrads(
        d_x_host^, d_t_silu_host^,
        d_sa_q^, d_sa_k^, d_sa_v^, d_sa_out^, d_sa_qn^, d_sa_kn^,
        d_ca_q^, d_ca_k^, d_ca_v^, d_ca_out^, d_ca_qn^, d_ca_kn^,
        d_mlp1^, d_mlp2^,
        d_sa_mod1^, d_sa_mod2^, d_ca_mod1^, d_ca_mod2^, d_mlp_mod1^, d_mlp_mod2^,
    )
