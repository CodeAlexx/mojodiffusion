# serenitymojo/models/acestep/acestep_block.mojo
#
# ACE-Step-1.5 DiT layer: forward (saving activations) + hand-chained backward
# (training) + LoRA variants. Mirrors the inference forward in
# models/dit/acestep_dit.mojo::acestep_block0_forward, which read the canonical
# modeling_acestep_v15_turbo.py + EriDiffusion-v2 acestep.rs::dit_layer_forward
# (843-913) line-by-line. Recipe + LoRA targets cited in models/acestep/config.mojo.
#
# WHAT MAKES THIS A NEW COMPUTE vs the wan22 block (the nearest template):
#   - GQA (16 q heads, 8 kv heads, n_rep=2): k/v projections emit [S, kv_dim] and
#     are repeat_kv'd to 16 heads after qk-rms/rope; backward = repeat_kv_backward
#     (grouped SUM over the n_rep repeated dst heads -> src kv head). New arm:
#     ops/gqa_backward.{repeat_kv_f32, repeat_kv_backward} (gated cos 1.0).
#   - PER-SAMPLE AdaLN ([H] vectors, NOT wan22's per-token [S,dim]): the 6 modvecs
#     shift_msa/scale_msa/gate_msa/c_shift/c_scale/c_gate are [H] (broadcast over
#     seq). So modulate/residual_gate use the [D]-vector kernels (ops/elementwise.
#     modulate + ops/elementwise_backward.modulate_backward COLUMN-reduce d_scale/
#     d_shift over rows; residual_gate + ops/rope_struct_backward.gate_residual_
#     backward COLUMN-reduce d_gate over rows).
#   - NO BIAS on any linear (q/k/v/o, mlp gate/up/down) — matches the safetensors
#     header (only .weight tensors). linear(x, w, None, ctx).
#   - SELF-ATTN RoPE is HALFSPLIT (Qwen3 rotate_half), cross-attn has NO RoPE.
#   - 3 affine RMSNorms (self_attn_norm/cross_attn_norm/mlp_norm [hidden]); the
#     qk-norms are per-head RMS [head_dim]. SwiGLU MLP (gate/up/down).
#   - CROSS-ATTN to the condition encoder (q-len S, kv-len L) -> rect SDPA;
#     UNGATED (plain add) residual.
#
# BACKWARD ARMS REUSED (all pre-built + gated): linear_backward (no-bias linears,
# d_w/d_x), rms_norm_backward (affine + per-head qk), modulate_backward (per-sample
# AdaLN), gate_residual_backward (per-sample gate), sdpa_backward (self, square),
# sdpa_backward_rect (cross, Sq!=Skv), rope_backward(halfsplit), swiglu_backward,
# repeat_kv_backward (GQA, NEW). All hand-chained in reverse (no tape).
#
# API boundary: hidden/enc enter + x_out + every grad leave as host List[Float32].
# Interior device-resident (TArc). F32 interior (training/parity contract).
#
# Mojo 1.0.0b1, NVIDIA GPU. `def` not `fn`; Tensor move-only (TArc carriers).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

# ── forward ops (GPU) ────────────────────────────────────────────────────────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.softmax import softmax_lastdim
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.tensor_algebra import (
    reshape, slice, add, mul, mul_scalar, permute, transpose, concat,
)
from serenitymojo.ops.gqa_backward import repeat_kv_f32, repeat_kv_backward

# ── backward arms (GPU; all pre-built + gated) ───────────────────────────────
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import rms_norm_backward, RmsNormBackward
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.rope_struct_backward import (
    rope_backward, gate_residual_backward, GateResidualGrads,
)
from serenitymojo.ops.attention_backward import (
    sdpa_backward, sdpa_backward_rect, SdpaGrads,
)
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward, SwigluGrads
from serenitymojo.ops.cast import cast_tensor


comptime TArc = ArcPointer[Tensor]


# ── host helpers ─────────────────────────────────────────────────────────────
# F32 host → F32 device (modulation vectors, d_out, grads — keep F32).
def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


# F32 host → BF16 device (block activations: hidden, enc, and intermediates).
def _ta(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, shape^, STDtype.BF16, ctx))


def _clone_t(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _nob() -> Optional[Tensor]:
    return Optional[Tensor](None)


# Expand a per-token rope table cos[seq, half] to [seq*heads, half] (each token
# row repeated `heads` times contiguously, matching reshape [seq,heads,Dh] row
# order). Mirrors acestep_dit._tile_rows (F32 here for training/parity).
def _tile_rows(
    tbl: Tensor, seq: Int, heads: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    var t3 = reshape(tbl, [seq, 1, half], ctx)        # [seq,1,half]
    var n = seq * heads * half
    var zh = List[Float32]()
    for _ in range(n):
        zh.append(0.0)
    var zeros = Tensor.from_host(zh^, [seq, heads, half], STDtype.F32, ctx)
    var bc = add(t3, zeros, ctx)                       # broadcast [seq,1,half] over heads
    return reshape(bc, [seq * heads, half], ctx)


# Rect cross-attn forward: q [1,S,H,Dh], k/v [1,L,H,Dh] (already GQA-expanded).
# Per-head softmax(q@kᵀ·scale)@v — the exact math sdpa_backward_rect inverts.
def _cross_sdpa[S: Int, L: Int, H: Int, Dh: Int](
    q: Tensor, k: Tensor, v: Tensor, scale: Float32, ctx: DeviceContext
) raises -> Tensor:
    var dim = H * Dh
    var q3 = reshape(q, [S, H, Dh], ctx)
    var k3 = reshape(k, [L, H, Dh], ctx)
    var v3 = reshape(v, [L, H, Dh], ctx)
    var qh = permute(q3, [1, 0, 2], ctx)    # [H,S,Dh]
    var kh = permute(k3, [1, 0, 2], ctx)    # [H,L,Dh]
    var vh = permute(v3, [1, 0, 2], ctx)    # [H,L,Dh]
    var acc = Tensor.from_host(List[Float32](), [0], STDtype.F32, ctx)
    for h in range(H):
        var qh_h = reshape(slice(qh, 0, h, 1, ctx), [S, Dh], ctx)
        var kh_h = reshape(slice(kh, 0, h, 1, ctx), [L, Dh], ctx)
        var vh_h = reshape(slice(vh, 0, h, 1, ctx), [L, Dh], ctx)
        var nb = Optional[Tensor](None)
        var scores = linear(qh_h, kh_h, nb^, ctx)        # [S,L] (q@kᵀ)
        scores = mul_scalar(scores, scale, ctx)
        var p = softmax_lastdim(scores, ctx)             # [S,L]
        var v_t = transpose(vh_h, 0, 1, ctx)             # [Dh,L]
        var nb2 = Optional[Tensor](None)
        var out_h = linear(p, v_t, nb2^, ctx)            # [S,Dh]
        var out_h3 = reshape(out_h, [1, S, Dh], ctx)
        if h == 0:
            acc = out_h3^
        else:
            acc = concat(0, ctx, acc, out_h3)            # [H,S,Dh]
    var sh = permute(acc, [1, 0, 2], ctx)                # [S,H,Dh]
    return reshape(sh, [1, S, dim], ctx)


# ── block trainable weights (DEVICE-RESIDENT TArc, uploaded ONCE) ─────────────
# self/cross attn: q/o [hidden,hidden]; k/v [kv_dim,hidden] (NO bias); q/k_norm
# [head_dim]. 3 affine RMS norms [hidden]. mlp gate/up [inter,hidden], down
# [hidden,inter] (NO bias).
struct AceBlockWeights(Copyable, Movable):
    var san_w: TArc        # self_attn_norm.weight [hidden]
    var can_w: TArc        # cross_attn_norm.weight
    var mn_w: TArc         # mlp_norm.weight
    # self attn
    var sa_wq: TArc
    var sa_wk: TArc
    var sa_wv: TArc
    var sa_wo: TArc
    var sa_qn: TArc
    var sa_kn: TArc
    # cross attn
    var ca_wq: TArc
    var ca_wk: TArc
    var ca_wv: TArc
    var ca_wo: TArc
    var ca_qn: TArc
    var ca_kn: TArc
    # mlp (SwiGLU)
    var mlp_gate: TArc     # [inter,hidden]
    var mlp_up: TArc       # [inter,hidden]
    var mlp_down: TArc     # [hidden,inter]

    def __init__(
        out self,
        var san_w: List[Float32], var can_w: List[Float32], var mn_w: List[Float32],
        var sa_wq: List[Float32], var sa_wk: List[Float32], var sa_wv: List[Float32], var sa_wo: List[Float32],
        var sa_qn: List[Float32], var sa_kn: List[Float32],
        var ca_wq: List[Float32], var ca_wk: List[Float32], var ca_wv: List[Float32], var ca_wo: List[Float32],
        var ca_qn: List[Float32], var ca_kn: List[Float32],
        var mlp_gate: List[Float32], var mlp_up: List[Float32], var mlp_down: List[Float32],
        hidden: Int, kv_dim: Int, inter: Int, hd: Int, ctx: DeviceContext,
    ) raises:
        self.san_w = TArc(Tensor.from_host(san_w^, [hidden], STDtype.BF16, ctx))
        self.can_w = TArc(Tensor.from_host(can_w^, [hidden], STDtype.BF16, ctx))
        self.mn_w = TArc(Tensor.from_host(mn_w^, [hidden], STDtype.BF16, ctx))
        self.sa_wq = TArc(Tensor.from_host(sa_wq^, [hidden, hidden], STDtype.BF16, ctx))
        self.sa_wk = TArc(Tensor.from_host(sa_wk^, [kv_dim, hidden], STDtype.BF16, ctx))
        self.sa_wv = TArc(Tensor.from_host(sa_wv^, [kv_dim, hidden], STDtype.BF16, ctx))
        self.sa_wo = TArc(Tensor.from_host(sa_wo^, [hidden, hidden], STDtype.BF16, ctx))
        self.sa_qn = TArc(Tensor.from_host(sa_qn^, [hd], STDtype.BF16, ctx))
        self.sa_kn = TArc(Tensor.from_host(sa_kn^, [hd], STDtype.BF16, ctx))
        self.ca_wq = TArc(Tensor.from_host(ca_wq^, [hidden, hidden], STDtype.BF16, ctx))
        self.ca_wk = TArc(Tensor.from_host(ca_wk^, [kv_dim, hidden], STDtype.BF16, ctx))
        self.ca_wv = TArc(Tensor.from_host(ca_wv^, [kv_dim, hidden], STDtype.BF16, ctx))
        self.ca_wo = TArc(Tensor.from_host(ca_wo^, [hidden, hidden], STDtype.BF16, ctx))
        self.ca_qn = TArc(Tensor.from_host(ca_qn^, [hd], STDtype.BF16, ctx))
        self.ca_kn = TArc(Tensor.from_host(ca_kn^, [hd], STDtype.BF16, ctx))
        self.mlp_gate = TArc(Tensor.from_host(mlp_gate^, [inter, hidden], STDtype.BF16, ctx))
        self.mlp_up = TArc(Tensor.from_host(mlp_up^, [inter, hidden], STDtype.BF16, ctx))
        self.mlp_down = TArc(Tensor.from_host(mlp_down^, [hidden, inter], STDtype.BF16, ctx))


# ── per-sample AdaLN modulation vectors (each [hidden] FLAT host list) ─────────
# order matches the modulation chunk6 (acestep.rs:861-866): shift_msa, scale_msa,
# gate_msa, c_shift, c_scale, c_gate.
struct AceModVecs(Copyable, Movable):
    var shift_msa: List[Float32]
    var scale_msa: List[Float32]
    var gate_msa: List[Float32]
    var c_shift: List[Float32]
    var c_scale: List[Float32]
    var c_gate: List[Float32]

    def __init__(
        out self,
        var shift_msa: List[Float32], var scale_msa: List[Float32], var gate_msa: List[Float32],
        var c_shift: List[Float32], var c_scale: List[Float32], var c_gate: List[Float32],
    ):
        self.shift_msa = shift_msa^
        self.scale_msa = scale_msa^
        self.gate_msa = gate_msa^
        self.c_shift = c_shift^
        self.c_scale = c_scale^
        self.c_gate = c_gate^


# ── saved activations (HOST-RESIDENT List[BFloat16], half the bytes vs F32) ────
# BF16 carrier: flame-core contract (bf16 in/out, F32 only inside GEMM).
# Re-upload in backward via Tensor.from_host_bf16(sv.FIELD.copy(), [shape], ctx).
# gate_residual_backward is F32-only → cast after re-upload (see backward).
struct AceSaved(Copyable, Movable):
    var hidden: List[BFloat16]      # [S,hidden]  block input
    # self-attn
    var sa_norm: List[BFloat16]     # [S,hidden]  rms(hidden, san_w)
    var sa_in: List[BFloat16]       # [S,hidden]  modulate(sa_norm) = self-attn input
    var sa_q_pre: List[BFloat16]    # [1,S,H,Dh]  q pre-rms
    var sa_k_pre: List[BFloat16]    # [1,S,Hkv,Dh]
    var sa_q_rms: List[BFloat16]    # [1,S,H,Dh]  rms(q_pre)
    var sa_k_rms: List[BFloat16]    # [1,S,Hkv,Dh]
    var sa_q_rope: List[BFloat16]   # [1,S,H,Dh]  rope(q_rms)
    var sa_k_rope: List[BFloat16]   # [1,S,Hkv,Dh]
    var sa_k_full: List[BFloat16]   # [1,S,H,Dh]  repeat_kv(k_rope)
    var sa_v_full: List[BFloat16]   # [1,S,H,Dh]  repeat_kv(v)
    var sa_att: List[BFloat16]      # [S,hidden]  attn out (flattened, pre o-linear)
    var x_sa: List[BFloat16]        # [S,hidden]  hidden + gate_msa*attn_o
    # cross-attn
    var ca_norm: List[BFloat16]     # [S,hidden]  rms(x_sa, can_w)
    var ca_q_pre: List[BFloat16]    # [1,S,H,Dh]
    var ca_k_pre: List[BFloat16]    # [1,L,Hkv,Dh]
    var ca_q_rms: List[BFloat16]    # [1,S,H,Dh]
    var ca_k_rms: List[BFloat16]    # [1,L,Hkv,Dh]
    var ca_k_full: List[BFloat16]   # [1,L,H,Dh]
    var ca_v_full: List[BFloat16]   # [1,L,H,Dh]
    var ca_att: List[BFloat16]      # [S,hidden]
    var enc: List[BFloat16]         # [L,hidden]  cross k/v input
    var x_ca: List[BFloat16]        # [S,hidden]  x_sa + cross_o (ungated)
    # mlp
    var mlp_norm: List[BFloat16]    # [S,hidden]  rms(x_ca, mn_w)
    var mlp_in: List[BFloat16]      # [S,hidden]  modulate(mlp_norm)
    var mlp_gate_h: List[BFloat16]  # [S,inter]   linear(mlp_in, gate)  (pre-silu)
    var mlp_up_h: List[BFloat16]    # [S,inter]   linear(mlp_in, up)
    var mlp_gu: List[BFloat16]      # [S,inter]   silu(gate)*up

    def __init__(
        out self, var hidden: List[BFloat16],
        var sa_norm: List[BFloat16], var sa_in: List[BFloat16],
        var sa_q_pre: List[BFloat16], var sa_k_pre: List[BFloat16],
        var sa_q_rms: List[BFloat16], var sa_k_rms: List[BFloat16],
        var sa_q_rope: List[BFloat16], var sa_k_rope: List[BFloat16],
        var sa_k_full: List[BFloat16], var sa_v_full: List[BFloat16],
        var sa_att: List[BFloat16], var x_sa: List[BFloat16],
        var ca_norm: List[BFloat16], var ca_q_pre: List[BFloat16],
        var ca_k_pre: List[BFloat16],
        var ca_q_rms: List[BFloat16], var ca_k_rms: List[BFloat16],
        var ca_k_full: List[BFloat16], var ca_v_full: List[BFloat16],
        var ca_att: List[BFloat16], var enc: List[BFloat16],
        var x_ca: List[BFloat16],
        var mlp_norm: List[BFloat16], var mlp_in: List[BFloat16],
        var mlp_gate_h: List[BFloat16],
        var mlp_up_h: List[BFloat16], var mlp_gu: List[BFloat16],
    ):
        self.hidden = hidden^
        self.sa_norm = sa_norm^
        self.sa_in = sa_in^
        self.sa_q_pre = sa_q_pre^
        self.sa_k_pre = sa_k_pre^
        self.sa_q_rms = sa_q_rms^
        self.sa_k_rms = sa_k_rms^
        self.sa_q_rope = sa_q_rope^
        self.sa_k_rope = sa_k_rope^
        self.sa_k_full = sa_k_full^
        self.sa_v_full = sa_v_full^
        self.sa_att = sa_att^
        self.x_sa = x_sa^
        self.ca_norm = ca_norm^
        self.ca_q_pre = ca_q_pre^
        self.ca_k_pre = ca_k_pre^
        self.ca_q_rms = ca_q_rms^
        self.ca_k_rms = ca_k_rms^
        self.ca_k_full = ca_k_full^
        self.ca_v_full = ca_v_full^
        self.ca_att = ca_att^
        self.enc = enc^
        self.x_ca = x_ca^
        self.mlp_norm = mlp_norm^
        self.mlp_in = mlp_in^
        self.mlp_gate_h = mlp_gate_h^
        self.mlp_up_h = mlp_up_h^
        self.mlp_gu = mlp_gu^


struct AceBlockForward(Copyable, Movable):
    var x_out: List[Float32]   # [S,hidden]
    var saved: AceSaved

    def __init__(out self, var x_out: List[Float32], var saved: AceSaved):
        self.x_out = x_out^
        self.saved = saved^


# ── FORWARD of one ACE-Step DiT block ─────────────────────────────────────────
# H = num_heads (q), HKV = num_kv_heads, S = q-seq, L = cross kv-seq (enc len).
def acestep_block_forward[
    H: Int, HKV: Int, Dh: Int, S: Int, L: Int
](
    hidden_h: List[Float32], enc_h: List[Float32], mv: AceModVecs,
    w: AceBlockWeights, cos: Tensor, sin: Tensor,
    hidden: Int, inter: Int, eps: Float32, ctx: DeviceContext,
) raises -> AceBlockForward:
    var nrep = H // HKV
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var x = _ta(hidden_h, [S, hidden], ctx)
    var enc = _ta(enc_h, [L, hidden], ctx)

    # Modulation vectors: host F32 → device BF16 (matching activation dtype).
    var shift_msa = Tensor.from_host(mv.shift_msa.copy(), [hidden], STDtype.BF16, ctx)
    var scale_msa = Tensor.from_host(mv.scale_msa.copy(), [hidden], STDtype.BF16, ctx)
    var gate_msa = Tensor.from_host(mv.gate_msa.copy(), [hidden], STDtype.BF16, ctx)
    var c_shift = Tensor.from_host(mv.c_shift.copy(), [hidden], STDtype.BF16, ctx)
    var c_scale = Tensor.from_host(mv.c_scale.copy(), [hidden], STDtype.BF16, ctx)
    var c_gate = Tensor.from_host(mv.c_gate.copy(), [hidden], STDtype.BF16, ctx)

    # rope tables expanded per head (q: H heads ; k: HKV heads); cast to BF16
    # so that rope_halfsplit dtype matches the BF16 q/k activations.
    var cos_q = cast_tensor(_tile_rows(cos, S, H, Dh // 2, ctx), STDtype.BF16, ctx)
    var sin_q = cast_tensor(_tile_rows(sin, S, H, Dh // 2, ctx), STDtype.BF16, ctx)
    var cos_k = cast_tensor(_tile_rows(cos, S, HKV, Dh // 2, ctx), STDtype.BF16, ctx)
    var sin_k = cast_tensor(_tile_rows(sin, S, HKV, Dh // 2, ctx), STDtype.BF16, ctx)

    # ── self-attention with per-sample AdaLN ──
    var sa_norm = rms_norm(x[], w.san_w[], eps, ctx)
    var sa_in = modulate(sa_norm, scale_msa, shift_msa, ctx)
    var q_flat = linear(sa_in, w.sa_wq[], _nob(), ctx)          # [S,hidden]
    var k_flat = linear(sa_in, w.sa_wk[], _nob(), ctx)          # [S,kv_dim]
    var v_flat = linear(sa_in, w.sa_wv[], _nob(), ctx)          # [S,kv_dim]
    var q_pre = reshape(q_flat^, [1, S, H, Dh], ctx)
    var k_pre = reshape(k_flat^, [1, S, HKV, Dh], ctx)
    var v4 = reshape(v_flat^, [1, S, HKV, Dh], ctx)
    var q_rms = rms_norm(q_pre, w.sa_qn[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.sa_kn[], eps, ctx)
    # rope (halfsplit) on [S*heads, Dh]
    var q_rope = reshape(
        rope_halfsplit(reshape(q_rms, [S * H, Dh], ctx), cos_q, sin_q, ctx),
        [1, S, H, Dh], ctx)
    var k_rope = reshape(
        rope_halfsplit(reshape(k_rms, [S * HKV, Dh], ctx), cos_k, sin_k, ctx),
        [1, S, HKV, Dh], ctx)
    # repeat_kv_f32 is F32-only → cast BF16 in, cast BF16 out.
    var k_full = cast_tensor(repeat_kv_f32(cast_tensor(k_rope, STDtype.F32, ctx), S, HKV, nrep, Dh, ctx), STDtype.BF16, ctx)
    var v_full = cast_tensor(repeat_kv_f32(cast_tensor(v4, STDtype.F32, ctx), S, HKV, nrep, Dh, ctx), STDtype.BF16, ctx)
    var att4 = sdpa_nomask[1, S, H, Dh](q_rope, k_full, v_full, scale, ctx)
    var sa_att = reshape(att4^, [S, hidden], ctx)
    var sa_o = linear(sa_att, w.sa_wo[], _nob(), ctx)
    var x_sa = residual_gate(x[], gate_msa, sa_o, ctx)          # hidden + gate*attn

    # ── cross-attention (q from x_sa, kv from enc, NO rope; ungated residual) ──
    var ca_norm = rms_norm(x_sa, w.can_w[], eps, ctx)
    var caq_flat = linear(ca_norm, w.ca_wq[], _nob(), ctx)      # [S,hidden]
    var cak_flat = linear(enc[], w.ca_wk[], _nob(), ctx)        # [L,kv_dim]
    var cav_flat = linear(enc[], w.ca_wv[], _nob(), ctx)        # [L,kv_dim]
    var caq_pre = reshape(caq_flat^, [1, S, H, Dh], ctx)
    var cak_pre = reshape(cak_flat^, [1, L, HKV, Dh], ctx)
    var cav4 = reshape(cav_flat^, [1, L, HKV, Dh], ctx)
    var caq_rms = rms_norm(caq_pre, w.ca_qn[], eps, ctx)
    var cak_rms = rms_norm(cak_pre, w.ca_kn[], eps, ctx)
    # repeat_kv_f32 is F32-only → cast BF16 in, cast BF16 out.
    var cak_full = cast_tensor(repeat_kv_f32(cast_tensor(cak_rms, STDtype.F32, ctx), L, HKV, nrep, Dh, ctx), STDtype.BF16, ctx)
    var cav_full = cast_tensor(repeat_kv_f32(cast_tensor(cav4, STDtype.F32, ctx), L, HKV, nrep, Dh, ctx), STDtype.BF16, ctx)
    var ca_att4 = _cross_sdpa[S, L, H, Dh](caq_rms, cak_full, cav_full, scale, ctx)
    var ca_att = reshape(ca_att4^, [S, hidden], ctx)
    var ca_o = linear(ca_att, w.ca_wo[], _nob(), ctx)
    var x_ca = add(x_sa, ca_o, ctx)                            # UNGATED residual

    # ── SwiGLU MLP with per-sample AdaLN ──
    var mlp_norm = rms_norm(x_ca, w.mn_w[], eps, ctx)
    var mlp_in = modulate(mlp_norm, c_scale, c_shift, ctx)
    var gate_h = linear(mlp_in, w.mlp_gate[], _nob(), ctx)      # [S,inter]
    var up_h = linear(mlp_in, w.mlp_up[], _nob(), ctx)          # [S,inter]
    var gate_act = silu(gate_h, ctx)
    var gu = mul(gate_act, up_h, ctx)                          # [S,inter]
    var mlp_o = linear(gu, w.mlp_down[], _nob(), ctx)          # [S,hidden]
    var x_final = residual_gate(x_ca, c_gate, mlp_o, ctx)

    var x_out = x_final.to_host(ctx)
    # Save activations as BF16 host lists (half the resident bytes vs F32).
    var saved = AceSaved(
        x[].to_host_bf16(ctx),
        sa_norm.to_host_bf16(ctx), sa_in.to_host_bf16(ctx),
        q_pre.to_host_bf16(ctx), k_pre.to_host_bf16(ctx),
        q_rms.to_host_bf16(ctx), k_rms.to_host_bf16(ctx),
        q_rope.to_host_bf16(ctx), k_rope.to_host_bf16(ctx),
        k_full.to_host_bf16(ctx), v_full.to_host_bf16(ctx),
        sa_att.to_host_bf16(ctx), x_sa.to_host_bf16(ctx),
        ca_norm.to_host_bf16(ctx), caq_pre.to_host_bf16(ctx),
        cak_pre.to_host_bf16(ctx),
        caq_rms.to_host_bf16(ctx), cak_rms.to_host_bf16(ctx),
        cak_full.to_host_bf16(ctx), cav_full.to_host_bf16(ctx),
        ca_att.to_host_bf16(ctx), enc[].to_host_bf16(ctx), x_ca.to_host_bf16(ctx),
        mlp_norm.to_host_bf16(ctx), mlp_in.to_host_bf16(ctx),
        gate_h.to_host_bf16(ctx), up_h.to_host_bf16(ctx), gu.to_host_bf16(ctx),
    )
    return AceBlockForward(x_out^, saved^)


# ── backward result: input grads (hidden + enc) + every weight grad + the 6
#    per-sample modulation-vector grads ────────────────────────────────────────
struct AceBlockGrads(Copyable, Movable):
    var d_hidden: List[Float32]   # [S,hidden]
    var d_enc: List[Float32]      # [L,hidden]  (cross-attn kv input grad)
    var d_san_w: List[Float32]
    var d_can_w: List[Float32]
    var d_mn_w: List[Float32]
    var d_sa_wq: List[Float32]
    var d_sa_wk: List[Float32]
    var d_sa_wv: List[Float32]
    var d_sa_wo: List[Float32]
    var d_sa_qn: List[Float32]
    var d_sa_kn: List[Float32]
    var d_ca_wq: List[Float32]
    var d_ca_wk: List[Float32]
    var d_ca_wv: List[Float32]
    var d_ca_wo: List[Float32]
    var d_ca_qn: List[Float32]
    var d_ca_kn: List[Float32]
    var d_mlp_gate: List[Float32]
    var d_mlp_up: List[Float32]
    var d_mlp_down: List[Float32]
    # 6 per-sample modulation-vector grads (each [hidden])
    var d_shift_msa: List[Float32]
    var d_scale_msa: List[Float32]
    var d_gate_msa: List[Float32]
    var d_c_shift: List[Float32]
    var d_c_scale: List[Float32]
    var d_c_gate: List[Float32]

    def __init__(
        out self,
        var d_hidden: List[Float32], var d_enc: List[Float32],
        var d_san_w: List[Float32], var d_can_w: List[Float32], var d_mn_w: List[Float32],
        var d_sa_wq: List[Float32], var d_sa_wk: List[Float32], var d_sa_wv: List[Float32], var d_sa_wo: List[Float32],
        var d_sa_qn: List[Float32], var d_sa_kn: List[Float32],
        var d_ca_wq: List[Float32], var d_ca_wk: List[Float32], var d_ca_wv: List[Float32], var d_ca_wo: List[Float32],
        var d_ca_qn: List[Float32], var d_ca_kn: List[Float32],
        var d_mlp_gate: List[Float32], var d_mlp_up: List[Float32], var d_mlp_down: List[Float32],
        var d_shift_msa: List[Float32], var d_scale_msa: List[Float32], var d_gate_msa: List[Float32],
        var d_c_shift: List[Float32], var d_c_scale: List[Float32], var d_c_gate: List[Float32],
    ):
        self.d_hidden = d_hidden^
        self.d_enc = d_enc^
        self.d_san_w = d_san_w^
        self.d_can_w = d_can_w^
        self.d_mn_w = d_mn_w^
        self.d_sa_wq = d_sa_wq^
        self.d_sa_wk = d_sa_wk^
        self.d_sa_wv = d_sa_wv^
        self.d_sa_wo = d_sa_wo^
        self.d_sa_qn = d_sa_qn^
        self.d_sa_kn = d_sa_kn^
        self.d_ca_wq = d_ca_wq^
        self.d_ca_wk = d_ca_wk^
        self.d_ca_wv = d_ca_wv^
        self.d_ca_wo = d_ca_wo^
        self.d_ca_qn = d_ca_qn^
        self.d_ca_kn = d_ca_kn^
        self.d_mlp_gate = d_mlp_gate^
        self.d_mlp_up = d_mlp_up^
        self.d_mlp_down = d_mlp_down^
        self.d_shift_msa = d_shift_msa^
        self.d_scale_msa = d_scale_msa^
        self.d_gate_msa = d_gate_msa^
        self.d_c_shift = d_c_shift^
        self.d_c_scale = d_c_scale^
        self.d_c_gate = d_c_gate^


# ── BACKWARD of one ACE-Step DiT block (hand-chained reverse of the forward) ──
# d_out_h: upstream grad of x_final [S,hidden].
def acestep_block_backward[
    H: Int, HKV: Int, Dh: Int, S: Int, L: Int
](
    d_out_h: List[Float32], mv: AceModVecs, w: AceBlockWeights, saved: AceSaved,
    cos: Tensor, sin: Tensor,
    hidden: Int, inter: Int, eps: Float32, ctx: DeviceContext,
) raises -> AceBlockGrads:
    var nrep = H // HKV
    var kv_dim = HKV * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    # F32 rope tables for rope_backward (grad dtype is F32; sv_* cast above).
    var cos_q = _tile_rows(cos, S, H, Dh // 2, ctx)
    var sin_q = _tile_rows(sin, S, H, Dh // 2, ctx)
    var cos_k = _tile_rows(cos, S, HKV, Dh // 2, ctx)
    var sin_k = _tile_rows(sin, S, HKV, Dh // 2, ctx)

    # Re-upload BF16 saved activations and immediately cast to F32.
    # All backward ops and grad accumulation run in F32 (recipe: "KEEP F32: grads").
    # The BF16→F32 cast here is a precision upcasting (no loss beyond the bf16 save).
    var sv_hidden   = cast_tensor(Tensor.from_host_bf16(saved.hidden.copy(),   [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_sa_norm  = cast_tensor(Tensor.from_host_bf16(saved.sa_norm.copy(),  [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_sa_in    = cast_tensor(Tensor.from_host_bf16(saved.sa_in.copy(),    [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_sa_q_pre = cast_tensor(Tensor.from_host_bf16(saved.sa_q_pre.copy(), [1, S, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_sa_k_pre = cast_tensor(Tensor.from_host_bf16(saved.sa_k_pre.copy(), [1, S, HKV, Dh], ctx), STDtype.F32, ctx)
    var sv_sa_q_rope= cast_tensor(Tensor.from_host_bf16(saved.sa_q_rope.copy(),[1, S, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_sa_k_full= cast_tensor(Tensor.from_host_bf16(saved.sa_k_full.copy(),[1, S, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_sa_v_full= cast_tensor(Tensor.from_host_bf16(saved.sa_v_full.copy(),[1, S, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_sa_att   = cast_tensor(Tensor.from_host_bf16(saved.sa_att.copy(),   [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_x_sa     = cast_tensor(Tensor.from_host_bf16(saved.x_sa.copy(),     [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_ca_norm  = cast_tensor(Tensor.from_host_bf16(saved.ca_norm.copy(),  [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_ca_q_pre = cast_tensor(Tensor.from_host_bf16(saved.ca_q_pre.copy(), [1, S, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_ca_k_pre = cast_tensor(Tensor.from_host_bf16(saved.ca_k_pre.copy(), [1, L, HKV, Dh], ctx), STDtype.F32, ctx)
    var sv_ca_q_rms = cast_tensor(Tensor.from_host_bf16(saved.ca_q_rms.copy(), [1, S, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_ca_k_full= cast_tensor(Tensor.from_host_bf16(saved.ca_k_full.copy(),[1, L, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_ca_v_full= cast_tensor(Tensor.from_host_bf16(saved.ca_v_full.copy(),[1, L, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_ca_att   = cast_tensor(Tensor.from_host_bf16(saved.ca_att.copy(),   [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_enc      = cast_tensor(Tensor.from_host_bf16(saved.enc.copy(),      [L, hidden],      ctx), STDtype.F32, ctx)
    var sv_x_ca     = cast_tensor(Tensor.from_host_bf16(saved.x_ca.copy(),     [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_mlp_norm = cast_tensor(Tensor.from_host_bf16(saved.mlp_norm.copy(), [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_mlp_in   = cast_tensor(Tensor.from_host_bf16(saved.mlp_in.copy(),   [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_mlp_gate = cast_tensor(Tensor.from_host_bf16(saved.mlp_gate_h.copy(),[S, inter],      ctx), STDtype.F32, ctx)
    var sv_mlp_up   = cast_tensor(Tensor.from_host_bf16(saved.mlp_up_h.copy(), [S, inter],       ctx), STDtype.F32, ctx)
    var sv_mlp_gu   = cast_tensor(Tensor.from_host_bf16(saved.mlp_gu.copy(),   [S, inter],       ctx), STDtype.F32, ctx)
    # All sv_* are now F32 — gate_residual_backward uses sv_x_ca and sv_hidden directly.

    var d_out = TArc(_t(d_out_h, [S, hidden], ctx))

    # ════════════════ MLP backward ════════════════
    # x_final = x_ca + c_gate * mlp_o  (per-sample gated residual)
    var c_gate_t = _t(mv.c_gate.copy(), [hidden], ctx)
    var mlp_o_rc = linear(sv_mlp_gu, w.mlp_down[], _nob(), ctx)
    var gb_mlp = gate_residual_backward(d_out[], sv_x_ca, c_gate_t, mlp_o_rc, ctx)
    var d_c_gate = gb_mlp.d_g.to_host(ctx)
    var d_x_ca_resid = TArc(_clone_t(gb_mlp.d_x, ctx))

    # mlp_o = linear(gu, mlp_down)
    var lb_down = linear_backward(gb_mlp.d_y, sv_mlp_gu, w.mlp_down[], S, inter, hidden, ctx)
    var d_mlp_down = lb_down.d_w.to_host(ctx)
    # gu = silu(gate_h) * up_h  (swiglu)
    var sg = swiglu_backward(lb_down.d_x, sv_mlp_gate, sv_mlp_up, ctx)
    # gate_h = linear(mlp_in, mlp_gate) ; up_h = linear(mlp_in, mlp_up)
    var lb_gate = linear_backward(sg.d_gate, sv_mlp_in, w.mlp_gate[], S, hidden, inter, ctx)
    var lb_up = linear_backward(sg.d_up, sv_mlp_in, w.mlp_up[], S, hidden, inter, ctx)
    var d_mlp_gate = lb_gate.d_w.to_host(ctx)
    var d_mlp_up = lb_up.d_w.to_host(ctx)
    var d_mlp_in = TArc(add(lb_gate.d_x, lb_up.d_x, ctx))
    # mlp_in = modulate(mlp_norm, c_scale, c_shift)
    var c_scale_t = _t(mv.c_scale.copy(), [hidden], ctx)
    var mb_mlp = modulate_backward(d_mlp_in[], sv_mlp_norm, c_scale_t, ctx)
    var d_c_scale = mb_mlp.d_scale.to_host(ctx)
    var d_c_shift = mb_mlp.d_shift.to_host(ctx)
    # mlp_norm = rms(x_ca, mn_w)
    var rb_mn = rms_norm_backward(mb_mlp.d_x, sv_x_ca, w.mn_w[], eps, ctx)
    var d_mn_w = rb_mn.d_g.to_host(ctx)
    # x_ca feeds the residual branch AND the mlp norm -> SUM
    var d_x_ca = TArc(add(d_x_ca_resid[], rb_mn.d_x, ctx))

    # ════════════════ Cross-attention backward ════════════════
    # x_ca = x_sa + ca_o  (ungated): d_x_sa_branch = d_x_ca ; d_ca_o = d_x_ca
    var lb_cao = linear_backward(d_x_ca[], sv_ca_att, w.ca_wo[], S, hidden, hidden, ctx)
    var d_ca_wo = lb_cao.d_w.to_host(ctx)
    var d_ca_att = reshape(lb_cao.d_x, [1, S, H, Dh], ctx)
    # rect SDPA on q_rms, k_full, v_full -> d_q_rms, d_k_full, d_v_full
    var csb = sdpa_backward_rect[1, S, L, H, Dh](
        sv_ca_q_rms, sv_ca_k_full, sv_ca_v_full, d_ca_att, scale, ctx,
    )
    # repeat_kv backward (GQA): d_k_full[1,L,H,Dh] -> d_k_rms[1,L,HKV,Dh] (grouped sum)
    var d_cak_rms = repeat_kv_backward(csb.d_k, L, HKV, nrep, Dh, ctx)
    var d_cav4 = repeat_kv_backward(csb.d_v, L, HKV, nrep, Dh, ctx)
    # caq_rms = rms(caq_pre, ca_qn) ; cak_rms = rms(cak_pre, ca_kn)
    var rb_caq = rms_norm_backward(csb.d_q, sv_ca_q_pre, w.ca_qn[], eps, ctx)
    var d_ca_qn = rb_caq.d_g.to_host(ctx)
    var rb_cak = rms_norm_backward(d_cak_rms, sv_ca_k_pre, w.ca_kn[], eps, ctx)
    var d_ca_kn = rb_cak.d_g.to_host(ctx)
    var d_caq_flat = reshape(rb_caq.d_x, [S, hidden], ctx)
    var d_cak_flat = reshape(rb_cak.d_x, [L, kv_dim], ctx)
    var d_cav_flat = reshape(d_cav4, [L, kv_dim], ctx)
    # caq = linear(ca_norm, ca_wq) ; cak = linear(enc, ca_wk) ; cav = linear(enc, ca_wv)
    var lb_caq = linear_backward(d_caq_flat, sv_ca_norm, w.ca_wq[], S, hidden, hidden, ctx)
    var lb_cak = linear_backward(d_cak_flat, sv_enc, w.ca_wk[], L, hidden, kv_dim, ctx)
    var lb_cav = linear_backward(d_cav_flat, sv_enc, w.ca_wv[], L, hidden, kv_dim, ctx)
    var d_ca_wq = lb_caq.d_w.to_host(ctx)
    var d_ca_wk = lb_cak.d_w.to_host(ctx)
    var d_ca_wv = lb_cav.d_w.to_host(ctx)
    # enc feeds cak AND cav -> SUM (cross kv input grad)
    var d_enc_t = TArc(add(lb_cak.d_x, lb_cav.d_x, ctx))
    # ca_norm = rms(x_sa, can_w)
    var rb_can = rms_norm_backward(lb_caq.d_x, sv_x_sa, w.can_w[], eps, ctx)
    var d_can_w = rb_can.d_g.to_host(ctx)
    # x_sa feeds cross norm AND the ungated residual (d_x_ca) -> SUM
    var d_x_sa = TArc(add(rb_can.d_x, d_x_ca[], ctx))

    # ════════════════ Self-attention backward ════════════════
    # x_sa = hidden + gate_msa * sa_o  (per-sample gated residual)
    var gate_msa_t = _t(mv.gate_msa.copy(), [hidden], ctx)
    var sa_o_rc = linear(sv_sa_att, w.sa_wo[], _nob(), ctx)
    var gb_sa = gate_residual_backward(d_x_sa[], sv_hidden, gate_msa_t, sa_o_rc, ctx)
    var d_gate_msa = gb_sa.d_g.to_host(ctx)
    var d_hidden_resid = TArc(_clone_t(gb_sa.d_x, ctx))
    # sa_o = linear(sa_att, sa_wo)
    var lb_sao = linear_backward(gb_sa.d_y, sv_sa_att, w.sa_wo[], S, hidden, hidden, ctx)
    var d_sa_wo = lb_sao.d_w.to_host(ctx)
    var d_sa_att = reshape(lb_sao.d_x, [1, S, H, Dh], ctx)
    # self SDPA (square) on q_rope, k_full, v_full
    var ssb = sdpa_backward[1, S, H, Dh](
        sv_sa_q_rope, sv_sa_k_full, sv_sa_v_full, d_sa_att, scale, ctx,
    )
    # repeat_kv backward (GQA) for k and v -> [1,S,HKV,Dh]
    var d_sak_rope = repeat_kv_backward(ssb.d_k, S, HKV, nrep, Dh, ctx)
    var d_sav4 = repeat_kv_backward(ssb.d_v, S, HKV, nrep, Dh, ctx)
    # rope backward (halfsplit; cos/sin non-learnable -> d_x only)
    var d_q_rms = rope_backward(
        reshape(ssb.d_q, [S * H, Dh], ctx), cos_q, sin_q, False, ctx)
    var d_k_rms = rope_backward(
        reshape(d_sak_rope, [S * HKV, Dh], ctx), cos_k, sin_k, False, ctx)
    var d_q_rms4 = reshape(d_q_rms, [1, S, H, Dh], ctx)
    var d_k_rms4 = reshape(d_k_rms, [1, S, HKV, Dh], ctx)
    # q_rms = rms(q_pre, sa_qn) ; k_rms = rms(k_pre, sa_kn)
    var rb_saq = rms_norm_backward(d_q_rms4, sv_sa_q_pre, w.sa_qn[], eps, ctx)
    var d_sa_qn = rb_saq.d_g.to_host(ctx)
    var rb_sak = rms_norm_backward(d_k_rms4, sv_sa_k_pre, w.sa_kn[], eps, ctx)
    var d_sa_kn = rb_sak.d_g.to_host(ctx)
    var d_saq_flat = reshape(rb_saq.d_x, [S, hidden], ctx)
    var d_sak_flat = reshape(rb_sak.d_x, [S, kv_dim], ctx)
    var d_sav_flat = reshape(d_sav4, [S, kv_dim], ctx)
    # q/k/v = linear(sa_in, sa_w{q,k,v}) — all on the SAME sa_in
    var lb_saq = linear_backward(d_saq_flat, sv_sa_in, w.sa_wq[], S, hidden, hidden, ctx)
    var lb_sak = linear_backward(d_sak_flat, sv_sa_in, w.sa_wk[], S, hidden, kv_dim, ctx)
    var lb_sav = linear_backward(d_sav_flat, sv_sa_in, w.sa_wv[], S, hidden, kv_dim, ctx)
    var d_sa_wq = lb_saq.d_w.to_host(ctx)
    var d_sa_wk = lb_sak.d_w.to_host(ctx)
    var d_sa_wv = lb_sav.d_w.to_host(ctx)
    var d_sa_in = TArc(add(add(lb_saq.d_x, lb_sak.d_x, ctx), lb_sav.d_x, ctx))
    # sa_in = modulate(sa_norm, scale_msa, shift_msa)
    var scale_msa_t = _t(mv.scale_msa.copy(), [hidden], ctx)
    var mb_sa = modulate_backward(d_sa_in[], sv_sa_norm, scale_msa_t, ctx)
    var d_scale_msa = mb_sa.d_scale.to_host(ctx)
    var d_shift_msa = mb_sa.d_shift.to_host(ctx)
    # sa_norm = rms(hidden, san_w)
    var rb_san = rms_norm_backward(mb_sa.d_x, sv_hidden, w.san_w[], eps, ctx)
    var d_san_w = rb_san.d_g.to_host(ctx)
    # hidden feeds self norm AND gated residual -> SUM
    var d_hidden = add(rb_san.d_x, d_hidden_resid[], ctx)
    var d_hidden_h = d_hidden.to_host(ctx)
    var d_enc_h = d_enc_t[].to_host(ctx)

    return AceBlockGrads(
        d_hidden_h^, d_enc_h^,
        d_san_w^, d_can_w^, d_mn_w^,
        d_sa_wq^, d_sa_wk^, d_sa_wv^, d_sa_wo^,
        d_sa_qn^, d_sa_kn^,
        d_ca_wq^, d_ca_wk^, d_ca_wv^, d_ca_wo^,
        d_ca_qn^, d_ca_kn^,
        d_mlp_gate^, d_mlp_up^, d_mlp_down^,
        d_shift_msa^, d_scale_msa^, d_gate_msa^,
        d_c_shift^, d_c_scale^, d_c_gate^,
    )


# ═══════════════════════════════════════════════════════════════════════════
# LoRA-ON-PROJECTION VARIANT
#
# Targets (acestep.rs:38-66 AceStepLoraTarget, 8/block): self_attn.{q,k,v,o} +
# cross_attn.{q,k,v,o}. q/o: in=hidden out=hidden ; k/v: in=hidden out=kv_dim
# (GQA). Forward adds the LoRA delta at each projection's linear output; backward
# returns d_A/d_B for each and folds the LoRA d_x contribution back into the
# projection-input grad. REUSES klein_lora_fwd / klein_lora_bwd (model-agnostic
# y=linear(x,W) LoRA math = train_step._lora_fwd/_lora_bwd).
# ═══════════════════════════════════════════════════════════════════════════

from serenitymojo.models.klein.lora_block import (
    LoraAdapter, klein_lora_fwd, klein_lora_bwd, KleinLoraGrads,
)


struct AceBlockLora(Copyable, Movable):
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


struct AceBlockLoraGrads(Copyable, Movable):
    var base: AceBlockGrads
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
        out self, var base: AceBlockGrads,
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


# Add LoRA delta of a projection (host input x_h [M,in]) into device output y.
# delta is cast to match y's dtype before the elementwise add.
def _add_lora_delta(
    y: Tensor, x_h: List[Float32], lo: Optional[LoraAdapter], M: Int, ctx: DeviceContext,
) raises -> Tensor:
    if not lo:
        return _clone_t(y, ctx)
    var delta_h = klein_lora_fwd(x_h, lo.value(), M, ctx)
    var delta_f32 = _t(delta_h^, y.shape().copy(), ctx)
    var delta = cast_tensor(delta_f32, y.dtype(), ctx)
    return add(y, delta, ctx)


# LoRA backward for one projection (if present); returns (d_A,d_B,d_x_lo). d_x_lo
# is the LoRA branch's contribution to the projection-input grad (summed by caller).
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


# ── FORWARD of one ACE-Step block WITH LoRA ───────────────────────────────────
def acestep_block_lora_forward[
    H: Int, HKV: Int, Dh: Int, S: Int, L: Int
](
    hidden_h: List[Float32], enc_h: List[Float32], mv: AceModVecs,
    w: AceBlockWeights, lora: AceBlockLora, cos: Tensor, sin: Tensor,
    hidden: Int, inter: Int, eps: Float32, ctx: DeviceContext,
) raises -> AceBlockForward:
    var nrep = H // HKV
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var x = _ta(hidden_h, [S, hidden], ctx)
    var enc = _ta(enc_h, [L, hidden], ctx)
    var enc_host = enc_h.copy()

    # Modulation vectors: host F32 → device BF16 (matching activation dtype).
    var shift_msa = Tensor.from_host(mv.shift_msa.copy(), [hidden], STDtype.BF16, ctx)
    var scale_msa = Tensor.from_host(mv.scale_msa.copy(), [hidden], STDtype.BF16, ctx)
    var gate_msa = Tensor.from_host(mv.gate_msa.copy(), [hidden], STDtype.BF16, ctx)
    var c_shift = Tensor.from_host(mv.c_shift.copy(), [hidden], STDtype.BF16, ctx)
    var c_scale = Tensor.from_host(mv.c_scale.copy(), [hidden], STDtype.BF16, ctx)
    var c_gate = Tensor.from_host(mv.c_gate.copy(), [hidden], STDtype.BF16, ctx)

    # rope tables cast to BF16 to match BF16 activations.
    var cos_q = cast_tensor(_tile_rows(cos, S, H, Dh // 2, ctx), STDtype.BF16, ctx)
    var sin_q = cast_tensor(_tile_rows(sin, S, H, Dh // 2, ctx), STDtype.BF16, ctx)
    var cos_k = cast_tensor(_tile_rows(cos, S, HKV, Dh // 2, ctx), STDtype.BF16, ctx)
    var sin_k = cast_tensor(_tile_rows(sin, S, HKV, Dh // 2, ctx), STDtype.BF16, ctx)

    # ── self-attention (LoRA on q/k/v/o) ──
    var sa_norm = rms_norm(x[], w.san_w[], eps, ctx)
    var sa_in = modulate(sa_norm, scale_msa, shift_msa, ctx)
    var sa_in_h = sa_in.to_host(ctx)
    var q_base = linear(sa_in, w.sa_wq[], _nob(), ctx)
    var k_base = linear(sa_in, w.sa_wk[], _nob(), ctx)
    var v_base = linear(sa_in, w.sa_wv[], _nob(), ctx)
    var q_flat = _add_lora_delta(q_base, sa_in_h, lora.sa_q, S, ctx)
    var k_flat = _add_lora_delta(k_base, sa_in_h, lora.sa_k, S, ctx)
    var v_flat = _add_lora_delta(v_base, sa_in_h, lora.sa_v, S, ctx)
    var q_pre = reshape(q_flat^, [1, S, H, Dh], ctx)
    var k_pre = reshape(k_flat^, [1, S, HKV, Dh], ctx)
    var v4 = reshape(v_flat^, [1, S, HKV, Dh], ctx)
    var q_rms = rms_norm(q_pre, w.sa_qn[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.sa_kn[], eps, ctx)
    var q_rope = reshape(
        rope_halfsplit(reshape(q_rms, [S * H, Dh], ctx), cos_q, sin_q, ctx),
        [1, S, H, Dh], ctx)
    var k_rope = reshape(
        rope_halfsplit(reshape(k_rms, [S * HKV, Dh], ctx), cos_k, sin_k, ctx),
        [1, S, HKV, Dh], ctx)
    # repeat_kv_f32 is F32-only → cast BF16 in, cast BF16 out.
    var k_full = cast_tensor(repeat_kv_f32(cast_tensor(k_rope, STDtype.F32, ctx), S, HKV, nrep, Dh, ctx), STDtype.BF16, ctx)
    var v_full = cast_tensor(repeat_kv_f32(cast_tensor(v4, STDtype.F32, ctx), S, HKV, nrep, Dh, ctx), STDtype.BF16, ctx)
    var att4 = sdpa_nomask[1, S, H, Dh](q_rope, k_full, v_full, scale, ctx)
    var sa_att = reshape(att4^, [S, hidden], ctx)
    var sa_att_h = sa_att.to_host(ctx)
    var sa_o_base = linear(sa_att, w.sa_wo[], _nob(), ctx)
    var sa_o = _add_lora_delta(sa_o_base, sa_att_h, lora.sa_o, S, ctx)
    var x_sa = residual_gate(x[], gate_msa, sa_o, ctx)

    # ── cross-attention (LoRA on q/k/v/o) ──
    var ca_norm = rms_norm(x_sa, w.can_w[], eps, ctx)
    var ca_norm_h = ca_norm.to_host(ctx)
    var caq_base = linear(ca_norm, w.ca_wq[], _nob(), ctx)
    var cak_base = linear(enc[], w.ca_wk[], _nob(), ctx)
    var cav_base = linear(enc[], w.ca_wv[], _nob(), ctx)
    var caq_flat = _add_lora_delta(caq_base, ca_norm_h, lora.ca_q, S, ctx)
    var cak_flat = _add_lora_delta(cak_base, enc_host, lora.ca_k, L, ctx)
    var cav_flat = _add_lora_delta(cav_base, enc_host, lora.ca_v, L, ctx)
    var caq_pre = reshape(caq_flat^, [1, S, H, Dh], ctx)
    var cak_pre = reshape(cak_flat^, [1, L, HKV, Dh], ctx)
    var cav4 = reshape(cav_flat^, [1, L, HKV, Dh], ctx)
    var caq_rms = rms_norm(caq_pre, w.ca_qn[], eps, ctx)
    var cak_rms = rms_norm(cak_pre, w.ca_kn[], eps, ctx)
    # repeat_kv_f32 is F32-only → cast BF16 in, cast BF16 out.
    var cak_full = cast_tensor(repeat_kv_f32(cast_tensor(cak_rms, STDtype.F32, ctx), L, HKV, nrep, Dh, ctx), STDtype.BF16, ctx)
    var cav_full = cast_tensor(repeat_kv_f32(cast_tensor(cav4, STDtype.F32, ctx), L, HKV, nrep, Dh, ctx), STDtype.BF16, ctx)
    var ca_att4 = _cross_sdpa[S, L, H, Dh](caq_rms, cak_full, cav_full, scale, ctx)
    var ca_att = reshape(ca_att4^, [S, hidden], ctx)
    var ca_att_h = ca_att.to_host(ctx)
    var ca_o_base = linear(ca_att, w.ca_wo[], _nob(), ctx)
    var ca_o = _add_lora_delta(ca_o_base, ca_att_h, lora.ca_o, S, ctx)
    var x_ca = add(x_sa, ca_o, ctx)

    # ── SwiGLU MLP (no LoRA) ──
    var mlp_norm = rms_norm(x_ca, w.mn_w[], eps, ctx)
    var mlp_in = modulate(mlp_norm, c_scale, c_shift, ctx)
    var gate_h = linear(mlp_in, w.mlp_gate[], _nob(), ctx)
    var up_h = linear(mlp_in, w.mlp_up[], _nob(), ctx)
    var gate_act = silu(gate_h, ctx)
    var gu = mul(gate_act, up_h, ctx)
    var mlp_o = linear(gu, w.mlp_down[], _nob(), ctx)
    var x_final = residual_gate(x_ca, c_gate, mlp_o, ctx)

    var x_out = x_final.to_host(ctx)
    # Save activations as BF16 host lists (half the resident bytes vs F32).
    var saved = AceSaved(
        x[].to_host_bf16(ctx),
        sa_norm.to_host_bf16(ctx), sa_in.to_host_bf16(ctx),
        q_pre.to_host_bf16(ctx), k_pre.to_host_bf16(ctx),
        q_rms.to_host_bf16(ctx), k_rms.to_host_bf16(ctx),
        q_rope.to_host_bf16(ctx), k_rope.to_host_bf16(ctx),
        k_full.to_host_bf16(ctx), v_full.to_host_bf16(ctx),
        sa_att.to_host_bf16(ctx), x_sa.to_host_bf16(ctx),
        ca_norm.to_host_bf16(ctx), caq_pre.to_host_bf16(ctx),
        cak_pre.to_host_bf16(ctx),
        caq_rms.to_host_bf16(ctx), cak_rms.to_host_bf16(ctx),
        cak_full.to_host_bf16(ctx), cav_full.to_host_bf16(ctx),
        ca_att.to_host_bf16(ctx), enc[].to_host_bf16(ctx), x_ca.to_host_bf16(ctx),
        mlp_norm.to_host_bf16(ctx), mlp_in.to_host_bf16(ctx),
        gate_h.to_host_bf16(ctx), up_h.to_host_bf16(ctx), gu.to_host_bf16(ctx),
    )
    return AceBlockForward(x_out^, saved^)


# ── BACKWARD of one ACE-Step block WITH LoRA ──────────────────────────────────
def acestep_block_lora_backward[
    H: Int, HKV: Int, Dh: Int, S: Int, L: Int
](
    d_out_h: List[Float32], mv: AceModVecs, w: AceBlockWeights,
    lora: AceBlockLora, saved: AceSaved, cos: Tensor, sin: Tensor,
    hidden: Int, inter: Int, eps: Float32, ctx: DeviceContext,
) raises -> AceBlockLoraGrads:
    var nrep = H // HKV
    var kv_dim = HKV * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var cos_q = _tile_rows(cos, S, H, Dh // 2, ctx)
    var sin_q = _tile_rows(sin, S, H, Dh // 2, ctx)
    var cos_k = _tile_rows(cos, S, HKV, Dh // 2, ctx)
    var sin_k = _tile_rows(sin, S, HKV, Dh // 2, ctx)

    # Re-upload BF16 saved activations and cast to F32. All backward ops and grad
    # accumulation run in F32 (recipe: "KEEP F32: grads"). LoRA helpers also need F32.
    var sv_sa_in    = cast_tensor(Tensor.from_host_bf16(saved.sa_in.copy(),    [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_sa_att   = cast_tensor(Tensor.from_host_bf16(saved.sa_att.copy(),   [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_ca_norm  = cast_tensor(Tensor.from_host_bf16(saved.ca_norm.copy(),  [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_enc      = cast_tensor(Tensor.from_host_bf16(saved.enc.copy(),      [L, hidden],      ctx), STDtype.F32, ctx)
    var sv_ca_att   = cast_tensor(Tensor.from_host_bf16(saved.ca_att.copy(),   [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_hidden   = cast_tensor(Tensor.from_host_bf16(saved.hidden.copy(),   [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_sa_norm  = cast_tensor(Tensor.from_host_bf16(saved.sa_norm.copy(),  [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_sa_q_pre = cast_tensor(Tensor.from_host_bf16(saved.sa_q_pre.copy(), [1, S, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_sa_k_pre = cast_tensor(Tensor.from_host_bf16(saved.sa_k_pre.copy(), [1, S, HKV, Dh], ctx), STDtype.F32, ctx)
    var sv_sa_q_rope= cast_tensor(Tensor.from_host_bf16(saved.sa_q_rope.copy(),[1, S, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_sa_k_full= cast_tensor(Tensor.from_host_bf16(saved.sa_k_full.copy(),[1, S, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_sa_v_full= cast_tensor(Tensor.from_host_bf16(saved.sa_v_full.copy(),[1, S, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_x_sa     = cast_tensor(Tensor.from_host_bf16(saved.x_sa.copy(),     [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_ca_q_pre = cast_tensor(Tensor.from_host_bf16(saved.ca_q_pre.copy(), [1, S, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_ca_k_pre = cast_tensor(Tensor.from_host_bf16(saved.ca_k_pre.copy(), [1, L, HKV, Dh], ctx), STDtype.F32, ctx)
    var sv_ca_q_rms = cast_tensor(Tensor.from_host_bf16(saved.ca_q_rms.copy(), [1, S, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_ca_k_full= cast_tensor(Tensor.from_host_bf16(saved.ca_k_full.copy(),[1, L, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_ca_v_full= cast_tensor(Tensor.from_host_bf16(saved.ca_v_full.copy(),[1, L, H, Dh],   ctx), STDtype.F32, ctx)
    var sv_x_ca     = cast_tensor(Tensor.from_host_bf16(saved.x_ca.copy(),     [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_mlp_norm = cast_tensor(Tensor.from_host_bf16(saved.mlp_norm.copy(), [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_mlp_in   = cast_tensor(Tensor.from_host_bf16(saved.mlp_in.copy(),   [S, hidden],      ctx), STDtype.F32, ctx)
    var sv_mlp_gate = cast_tensor(Tensor.from_host_bf16(saved.mlp_gate_h.copy(),[S, inter],      ctx), STDtype.F32, ctx)
    var sv_mlp_up   = cast_tensor(Tensor.from_host_bf16(saved.mlp_up_h.copy(), [S, inter],       ctx), STDtype.F32, ctx)
    var sv_mlp_gu   = cast_tensor(Tensor.from_host_bf16(saved.mlp_gu.copy(),   [S, inter],       ctx), STDtype.F32, ctx)
    # All sv_* are now F32 — gate_residual_backward can use them directly.
    # LoRA backward helpers need host F32 — extract from F32 device tensors.
    var sa_in_h  = sv_sa_in.to_host(ctx)
    var sa_att_h = sv_sa_att.to_host(ctx)
    var ca_norm_h= sv_ca_norm.to_host(ctx)
    var enc_h    = sv_enc.to_host(ctx)
    var ca_att_h = sv_ca_att.to_host(ctx)

    var d_out = TArc(_t(d_out_h, [S, hidden], ctx))

    # ════════════════ MLP backward (no LoRA) ════════════════
    var c_gate_t = _t(mv.c_gate.copy(), [hidden], ctx)
    var mlp_o_rc = linear(sv_mlp_gu, w.mlp_down[], _nob(), ctx)
    var gb_mlp = gate_residual_backward(d_out[], sv_x_ca, c_gate_t, mlp_o_rc, ctx)
    var d_c_gate = gb_mlp.d_g.to_host(ctx)
    var d_x_ca_resid = TArc(_clone_t(gb_mlp.d_x, ctx))
    var lb_down = linear_backward(gb_mlp.d_y, sv_mlp_gu, w.mlp_down[], S, inter, hidden, ctx)
    var d_mlp_down = lb_down.d_w.to_host(ctx)
    var sg = swiglu_backward(lb_down.d_x, sv_mlp_gate, sv_mlp_up, ctx)
    var lb_gate = linear_backward(sg.d_gate, sv_mlp_in, w.mlp_gate[], S, hidden, inter, ctx)
    var lb_up = linear_backward(sg.d_up, sv_mlp_in, w.mlp_up[], S, hidden, inter, ctx)
    var d_mlp_gate = lb_gate.d_w.to_host(ctx)
    var d_mlp_up = lb_up.d_w.to_host(ctx)
    var d_mlp_in = TArc(add(lb_gate.d_x, lb_up.d_x, ctx))
    var c_scale_t = _t(mv.c_scale.copy(), [hidden], ctx)
    var mb_mlp = modulate_backward(d_mlp_in[], sv_mlp_norm, c_scale_t, ctx)
    var d_c_scale = mb_mlp.d_scale.to_host(ctx)
    var d_c_shift = mb_mlp.d_shift.to_host(ctx)
    var rb_mn = rms_norm_backward(mb_mlp.d_x, sv_x_ca, w.mn_w[], eps, ctx)
    var d_mn_w = rb_mn.d_g.to_host(ctx)
    var d_x_ca = TArc(add(d_x_ca_resid[], rb_mn.d_x, ctx))

    # ════════════════ Cross-attention backward (LoRA q/k/v/o) ════════════════
    var d_ca_o_h = d_x_ca[].to_host(ctx)
    var lb_cao = linear_backward(d_x_ca[], sv_ca_att, w.ca_wo[], S, hidden, hidden, ctx)
    var d_ca_wo = lb_cao.d_w.to_host(ctx)
    var ca_o_g = _lora_bwd_opt(lora.ca_o, d_ca_o_h, ca_att_h, S, hidden, ctx)
    var d_ca_att = add(lb_cao.d_x, _t(ca_o_g.d_x.copy(), [S, hidden], ctx), ctx)
    var d_ca_att4 = reshape(d_ca_att, [1, S, H, Dh], ctx)
    var csb = sdpa_backward_rect[1, S, L, H, Dh](
        sv_ca_q_rms, sv_ca_k_full, sv_ca_v_full, d_ca_att4, scale, ctx,
    )
    var d_cak_rms = repeat_kv_backward(csb.d_k, L, HKV, nrep, Dh, ctx)
    var d_cav4 = repeat_kv_backward(csb.d_v, L, HKV, nrep, Dh, ctx)
    var rb_caq = rms_norm_backward(csb.d_q, sv_ca_q_pre, w.ca_qn[], eps, ctx)
    var d_ca_qn = rb_caq.d_g.to_host(ctx)
    var rb_cak = rms_norm_backward(d_cak_rms, sv_ca_k_pre, w.ca_kn[], eps, ctx)
    var d_ca_kn = rb_cak.d_g.to_host(ctx)
    var d_caq_flat = reshape(rb_caq.d_x, [S, hidden], ctx)
    var d_cak_flat = reshape(rb_cak.d_x, [L, kv_dim], ctx)
    var d_cav_flat = reshape(d_cav4, [L, kv_dim], ctx)
    var d_caq_h = d_caq_flat.to_host(ctx)
    var d_cak_h = d_cak_flat.to_host(ctx)
    var d_cav_h = d_cav_flat.to_host(ctx)
    var lb_caq = linear_backward(d_caq_flat, sv_ca_norm, w.ca_wq[], S, hidden, hidden, ctx)
    var lb_cak = linear_backward(d_cak_flat, sv_enc, w.ca_wk[], L, hidden, kv_dim, ctx)
    var lb_cav = linear_backward(d_cav_flat, sv_enc, w.ca_wv[], L, hidden, kv_dim, ctx)
    var d_ca_wq = lb_caq.d_w.to_host(ctx)
    var d_ca_wk = lb_cak.d_w.to_host(ctx)
    var d_ca_wv = lb_cav.d_w.to_host(ctx)
    var ca_q_g = _lora_bwd_opt(lora.ca_q, d_caq_h, ca_norm_h, S, hidden, ctx)
    var ca_k_g = _lora_bwd_opt(lora.ca_k, d_cak_h, enc_h, L, hidden, ctx)
    var ca_v_g = _lora_bwd_opt(lora.ca_v, d_cav_h, enc_h, L, hidden, ctx)
    var d_ca_norm_in = add(lb_caq.d_x, _t(ca_q_g.d_x.copy(), [S, hidden], ctx), ctx)
    var d_enc_base = add(lb_cak.d_x, lb_cav.d_x, ctx)
    var d_enc_lora = add(
        _t(ca_k_g.d_x.copy(), [L, hidden], ctx), _t(ca_v_g.d_x.copy(), [L, hidden], ctx), ctx
    )
    var d_enc_t = TArc(add(d_enc_base, d_enc_lora, ctx))
    var rb_can = rms_norm_backward(d_ca_norm_in, sv_x_sa, w.can_w[], eps, ctx)
    var d_can_w = rb_can.d_g.to_host(ctx)
    var d_x_sa = TArc(add(rb_can.d_x, d_x_ca[], ctx))

    # ════════════════ Self-attention backward (LoRA q/k/v/o) ════════════════
    var gate_msa_t = _t(mv.gate_msa.copy(), [hidden], ctx)
    var sa_o_rc = linear(sv_sa_att, w.sa_wo[], _nob(), ctx)
    var gb_sa = gate_residual_backward(d_x_sa[], sv_hidden, gate_msa_t, sa_o_rc, ctx)
    var d_gate_msa = gb_sa.d_g.to_host(ctx)
    var d_hidden_resid = TArc(_clone_t(gb_sa.d_x, ctx))
    var d_sa_o_h = gb_sa.d_y.to_host(ctx)
    var lb_sao = linear_backward(gb_sa.d_y, sv_sa_att, w.sa_wo[], S, hidden, hidden, ctx)
    var d_sa_wo = lb_sao.d_w.to_host(ctx)
    var sa_o_g = _lora_bwd_opt(lora.sa_o, d_sa_o_h, sa_att_h, S, hidden, ctx)
    var d_sa_att = add(lb_sao.d_x, _t(sa_o_g.d_x.copy(), [S, hidden], ctx), ctx)
    var d_sa_att4 = reshape(d_sa_att, [1, S, H, Dh], ctx)
    var ssb = sdpa_backward[1, S, H, Dh](
        sv_sa_q_rope, sv_sa_k_full, sv_sa_v_full, d_sa_att4, scale, ctx,
    )
    var d_sak_rope = repeat_kv_backward(ssb.d_k, S, HKV, nrep, Dh, ctx)
    var d_sav4 = repeat_kv_backward(ssb.d_v, S, HKV, nrep, Dh, ctx)
    var d_q_rms = rope_backward(
        reshape(ssb.d_q, [S * H, Dh], ctx), cos_q, sin_q, False, ctx)
    var d_k_rms = rope_backward(
        reshape(d_sak_rope, [S * HKV, Dh], ctx), cos_k, sin_k, False, ctx)
    var d_q_rms4 = reshape(d_q_rms, [1, S, H, Dh], ctx)
    var d_k_rms4 = reshape(d_k_rms, [1, S, HKV, Dh], ctx)
    var rb_saq = rms_norm_backward(d_q_rms4, sv_sa_q_pre, w.sa_qn[], eps, ctx)
    var d_sa_qn = rb_saq.d_g.to_host(ctx)
    var rb_sak = rms_norm_backward(d_k_rms4, sv_sa_k_pre, w.sa_kn[], eps, ctx)
    var d_sa_kn = rb_sak.d_g.to_host(ctx)
    var d_saq_flat = reshape(rb_saq.d_x, [S, hidden], ctx)
    var d_sak_flat = reshape(rb_sak.d_x, [S, kv_dim], ctx)
    var d_sav_flat = reshape(d_sav4, [S, kv_dim], ctx)
    var d_saq_h = d_saq_flat.to_host(ctx)
    var d_sak_h = d_sak_flat.to_host(ctx)
    var d_sav_h = d_sav_flat.to_host(ctx)
    var lb_saq = linear_backward(d_saq_flat, sv_sa_in, w.sa_wq[], S, hidden, hidden, ctx)
    var lb_sak = linear_backward(d_sak_flat, sv_sa_in, w.sa_wk[], S, hidden, kv_dim, ctx)
    var lb_sav = linear_backward(d_sav_flat, sv_sa_in, w.sa_wv[], S, hidden, kv_dim, ctx)
    var d_sa_wq = lb_saq.d_w.to_host(ctx)
    var d_sa_wk = lb_sak.d_w.to_host(ctx)
    var d_sa_wv = lb_sav.d_w.to_host(ctx)
    var sa_q_g = _lora_bwd_opt(lora.sa_q, d_saq_h, sa_in_h, S, hidden, ctx)
    var sa_k_g = _lora_bwd_opt(lora.sa_k, d_sak_h, sa_in_h, S, hidden, ctx)
    var sa_v_g = _lora_bwd_opt(lora.sa_v, d_sav_h, sa_in_h, S, hidden, ctx)
    var d_sa_in_base = add(add(lb_saq.d_x, lb_sak.d_x, ctx), lb_sav.d_x, ctx)
    var d_sa_in_lora = add(
        add(_t(sa_q_g.d_x.copy(), [S, hidden], ctx), _t(sa_k_g.d_x.copy(), [S, hidden], ctx), ctx),
        _t(sa_v_g.d_x.copy(), [S, hidden], ctx), ctx,
    )
    var d_sa_in = TArc(add(d_sa_in_base, d_sa_in_lora, ctx))
    var scale_msa_t = _t(mv.scale_msa.copy(), [hidden], ctx)
    var mb_sa = modulate_backward(d_sa_in[], sv_sa_norm, scale_msa_t, ctx)
    var d_scale_msa = mb_sa.d_scale.to_host(ctx)
    var d_shift_msa = mb_sa.d_shift.to_host(ctx)
    var rb_san = rms_norm_backward(mb_sa.d_x, sv_hidden, w.san_w[], eps, ctx)
    var d_san_w = rb_san.d_g.to_host(ctx)
    var d_hidden = add(rb_san.d_x, d_hidden_resid[], ctx)
    var d_hidden_h = d_hidden.to_host(ctx)
    var d_enc_h_out = d_enc_t[].to_host(ctx)

    var base = AceBlockGrads(
        d_hidden_h^, d_enc_h_out^,
        d_san_w^, d_can_w^, d_mn_w^,
        d_sa_wq^, d_sa_wk^, d_sa_wv^, d_sa_wo^,
        d_sa_qn^, d_sa_kn^,
        d_ca_wq^, d_ca_wk^, d_ca_wv^, d_ca_wo^,
        d_ca_qn^, d_ca_kn^,
        d_mlp_gate^, d_mlp_up^, d_mlp_down^,
        d_shift_msa^, d_scale_msa^, d_gate_msa^,
        d_c_shift^, d_c_scale^, d_c_gate^,
    )
    return AceBlockLoraGrads(
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
