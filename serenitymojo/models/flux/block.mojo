# serenitymojo/models/flux/block.mojo
#
# Flux (flux1-dev) DiT blocks: DOUBLE-STREAM + SINGLE-STREAM, forward (saving
# activations) + hand-chained backward (training). Packaged in the EXACT style
# proven by serenitymojo/models/klein/{double_block,single_block}.mojo (Klein's
# blocks gated 28/28 and 8/8 vs torch). Flux is Klein's architecture FAMILY but
# the blocks are NOT byte-identical, so this is a Flux-specific block built from
# the SAME composition primitives.
#
# WHY A FLUX-SPECIFIC BLOCK (not a direct Klein reuse) — measured from
# inference-flame/src/models/flux1_dit.rs (the parity oracle), lines 1-34, 729-1008:
#   (1) BIASES EVERYWHERE. Every Flux linear (qkv, proj, mlp.0, mlp.2 for the
#       double block; linear1, linear2 for the single block) carries a bias.
#       Klein's blocks are no-bias (linear(x, w, Optional(None))). So every Flux
#       linear here passes Optional[Tensor](bias) and the backward returns the
#       bias column-sum grad d_b (linear_backward already computes d_b).
#   (2) GELU MLP, not SwiGLU. Flux double-block MLP = linear(D->4D, bias) -> GELU
#       (tanh approx) -> linear(4D->D, bias). Flux single-block fuses the MLP-up
#       into linear1 (the 4D mlp_in slice) then GELU then fuses mlp-down into
#       linear2. Klein uses swiglu(gate, up) with a 2F gate_up split. So the MLP
#       arm here is gelu / gelu_backward, NOT swiglu / swiglu_backward.
#   (3) Everything else MATCHES Klein:
#       - modulate_pre = (1+scale)*layer_norm(x,1,0,eps) + shift   (eps 1e-6)
#       - residual_gate(x, gate, y) = x + gate*y
#       - q/k rms_norm over Dh (eps 1e-6); v un-normed
#       - joint concat txt FIRST then img (double); rope INTERLEAVED-pair
#         (the SAME rope_interleaved op Klein uses — flame-core's rope kernel
#          comment says "Used by Klein/Flux"). The Flux RoPE table is built 3-axis
#         (axes 16,56,56) at the STACK level; the BLOCK just consumes (cos,sin).
#       - sdpa non-causal scale = 1/sqrt(Dh).
#
# FORWARD GRAPH — DOUBLE block (flux1_dit.rs::double_block_forward 784-887)
#   For stream s in {img, txt}, with precomputed AdaLN vectors
#   (shift1,scale1,gate1,shift2,scale2,gate2) each [D]:
#     s_norm  = modulate(layer_norm(s,1,0,eps), scale1, shift1)
#     s_qkv   = linear(s_norm, Wqkv_s, bqkv_s)              # [N,3D]
#     s_q/k/v = split -> reshape [1,N,H,Dh]
#     s_q     = rms_norm(s_q, q_norm_s) ; s_k = rms_norm(s_k, k_norm_s)
#   JOINT (txt FIRST, then img):
#     q = concat(axis=1, txt_q, img_q) ; k,v likewise       # [1,S,H,Dh]
#     qr = rope_interleaved(q,cos,sin) ; kr = rope_interleaved(k,cos,sin)
#     att = sdpa(qr,kr,v,1/sqrt(Dh))                        # [1,S,H,Dh]
#     txt_att = slice(att,1,0,N_TXT) ; img_att = slice(att,1,N_TXT,N_IMG)
#   Per stream:
#     s_out      = linear(s_att, Wproj_s, bproj_s)
#     s_attn_res = s + gate1 * s_out
#     s_mlp_in   = modulate(layer_norm(s_attn_res,1,0,eps), scale2, shift2)
#     s_mlp_h    = gelu(linear(s_mlp_in, Wmlp0_s, bmlp0_s))  # [N,4D]
#     s_mlp      = linear(s_mlp_h, Wmlp2_s, bmlp2_s)         # [N,D]
#     s_final    = s_attn_res + gate2 * s_mlp
#
# FORWARD GRAPH — SINGLE block (flux1_dit.rs::single_block_forward 942-1007)
#   With (shift, scale, gate) each [D]:
#     x_norm   = modulate(layer_norm(x,1,0,eps), scale, shift)
#     fused    = linear(x_norm, W1, b1)                      # [S, 3D + Fmlp]
#     qkv      = fused[:, :3D]   ; mlp_in = fused[:, 3D:3D+Fmlp]   (Fmlp = 4D)
#     q,k,v    = split qkv -> [1,S,H,Dh]
#     q        = rms_norm(q, q_norm) ; k = rms_norm(k, k_norm)
#     att      = sdpa(rope(q), rope(k), v) -> reshape [S,D]
#     mlp_h    = gelu(mlp_in)                                # [S,Fmlp]
#     out_in   = concat(axis=1, att_flat, mlp_h)             # [S, D+Fmlp]
#     out      = linear(out_in, W2, b2)                      # W2 [D, D+Fmlp]
#     result   = residual_gate(x, gate, out)
#
# BACKWARD: every arm is an EXISTING, VERIFIED kernel (the SAME arms Klein's
# blocks compose) — this file only composes them, with bias grads added and the
# swiglu arm swapped for gelu_backward. The joint coupling means d for txt and
# img both flow OUT of the SAME sdpa_backward, then split via cat_backward
# (txt FIRST).
#
# Mojo 1.0.0b1+: `def` not `fn`; Tensor move-only; bias linear = linear(x, w,
# Optional[Tensor](b), ctx). Host List[Float32] at the API boundary (matches the
# parity gates + the Klein block boundary contract).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor

# TArc = Copyable device carrier (ArcPointer[Tensor]); a copy is a refcount bump
# of the SAME device buffer (mirrors klein/double_block.mojo:127). Lets the
# weight structs stay Copyable while holding move-only device Tensors.
comptime TArc = ArcPointer[Tensor]

# ── forward ops (GPU) ────────────────────────────────────────────────────────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, reshape_in_place, slice, concat, add,
)

# ── backward arms (GPU) ──────────────────────────────────────────────────────
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import (
    rms_norm_backward, RmsNormBackward,
    layer_norm_backward, LayerNormBackward,
)
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.attention_backward import sdpa_backward, SdpaGrads
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, GateResidualGrads, rope_backward,
)
from serenitymojo.ops.shape_backward import cat_backward, CatGrads2


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


# NATIVE BF16 COMPUTE: every F32-host carrier (block input activations, weights,
# biases, modulation vectors) is uploaded as a BF16 *device* tensor so the
# matmuls/elementwise ops hit linear's `dt==bfloat16` native bf16·bf16 path
# (F32 accumulate inside the GEMM, exactly like flame-core's cuBLAS). from_host
# casts F32-host → BF16-device.
def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


# F32 upload — used ONLY at the F32-only op boundaries (rope_backward cos/sin,
# gate_residual_backward grad_out/gate, cat_backward grad). Keeps the public
# List[Float32] verbatim on the device as F32.
def _tf32(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


# Re-upload a saved BF16 host activation back to a BF16 *device* tensor (native,
# no F32 detour) so the backward matmuls (linear_backward) run on bf16 saved acts.
def _tb16(vals: List[BFloat16], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host_bf16(vals, shape^, ctx)


# Convert a F32 host list to BFloat16 (for saving the x/input copy as BF16).
def _f32_to_bf16(v: List[Float32]) -> List[BFloat16]:
    var o = List[BFloat16]()
    for i in range(len(v)):
        o.append(BFloat16(v[i]))
    return o^


# Local bf16→F32 cast for a single op call that hard-requires F32 (rope_backward,
# cat_backward). Does NOT edit any shared op — casts on device for that one call.
def _to_f32(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(t, STDtype.F32, ctx)


# Local F32→bf16 cast to re-enter the bf16 chain after an F32-only op.
def _to_bf16(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(t, STDtype.BF16, ctx)


# ═══════════════════════════════════════════════════════════════════════════
# Per-stream modulation vectors (each [D]) — same shape as Klein's ModVecs.
# ═══════════════════════════════════════════════════════════════════════════
struct ModVecs(Copyable, Movable):
    var shift1: List[Float32]
    var scale1: List[Float32]
    var gate1: List[Float32]
    var shift2: List[Float32]
    var scale2: List[Float32]
    var gate2: List[Float32]

    def __init__(
        out self,
        var shift1: List[Float32], var scale1: List[Float32], var gate1: List[Float32],
        var shift2: List[Float32], var scale2: List[Float32], var gate2: List[Float32],
    ):
        self.shift1 = shift1^
        self.scale1 = scale1^
        self.gate1 = gate1^
        self.shift2 = shift2^
        self.scale2 = scale2^
        self.gate2 = gate2^


struct SingleModVecs(Copyable, Movable):
    var shift: List[Float32]
    var scale: List[Float32]
    var gate: List[Float32]

    def __init__(
        out self,
        var shift: List[Float32], var scale: List[Float32], var gate: List[Float32],
    ):
        self.shift = shift^
        self.scale = scale^
        self.gate = gate^


# ═══════════════════════════════════════════════════════════════════════════
# Per-stream trainable weights for the DOUBLE block (Flux: WITH biases).
#   wqkv: [3D, D]  bqkv: [3D]      wproj: [D, D]  bproj: [D]
#   wmlp0: [Fmlp, D] bmlp0: [Fmlp] wmlp2: [D, Fmlp] bmlp2: [D]   (Fmlp = 4D)
#   q_norm/k_norm: [Dh]
# Uploaded ONCE at construction (device-resident, borrowed by use-sites).
# ═══════════════════════════════════════════════════════════════════════════
struct StreamWeights(Copyable, Movable):
    var wqkv: TArc
    var bqkv: TArc
    var wproj: TArc
    var bproj: TArc
    var wmlp0: TArc
    var bmlp0: TArc
    var wmlp2: TArc
    var bmlp2: TArc
    var q_norm: TArc
    var k_norm: TArc

    def __init__(
        out self,
        var wqkv: List[Float32], var bqkv: List[Float32],
        var wproj: List[Float32], var bproj: List[Float32],
        var wmlp0: List[Float32], var bmlp0: List[Float32],
        var wmlp2: List[Float32], var bmlp2: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
        D: Int, Fmlp: Int, Dh: Int, ctx: DeviceContext,
    ) raises:
        self.wqkv = TArc(Tensor.from_host(wqkv^, [3 * D, D], STDtype.BF16, ctx))
        self.bqkv = TArc(Tensor.from_host(bqkv^, [3 * D], STDtype.BF16, ctx))
        self.wproj = TArc(Tensor.from_host(wproj^, [D, D], STDtype.BF16, ctx))
        self.bproj = TArc(Tensor.from_host(bproj^, [D], STDtype.BF16, ctx))
        self.wmlp0 = TArc(Tensor.from_host(wmlp0^, [Fmlp, D], STDtype.BF16, ctx))
        self.bmlp0 = TArc(Tensor.from_host(bmlp0^, [Fmlp], STDtype.BF16, ctx))
        self.wmlp2 = TArc(Tensor.from_host(wmlp2^, [D, Fmlp], STDtype.BF16, ctx))
        self.bmlp2 = TArc(Tensor.from_host(bmlp2^, [D], STDtype.BF16, ctx))
        self.q_norm = TArc(Tensor.from_host(q_norm^, [Dh], STDtype.BF16, ctx))
        self.k_norm = TArc(Tensor.from_host(k_norm^, [Dh], STDtype.BF16, ctx))

    def __init__(
        out self,
        var wqkv: TArc, var bqkv: TArc,
        var wproj: TArc, var bproj: TArc,
        var wmlp0: TArc, var bmlp0: TArc,
        var wmlp2: TArc, var bmlp2: TArc,
        var q_norm: TArc, var k_norm: TArc,
    ):
        self.wqkv = wqkv^
        self.bqkv = bqkv^
        self.wproj = wproj^
        self.bproj = bproj^
        self.wmlp0 = wmlp0^
        self.bmlp0 = bmlp0^
        self.wmlp2 = wmlp2^
        self.bmlp2 = bmlp2^
        self.q_norm = q_norm^
        self.k_norm = k_norm^


struct DoubleBlockWeights(Copyable, Movable):
    var img: StreamWeights
    var txt: StreamWeights

    def __init__(out self, var img: StreamWeights, var txt: StreamWeights):
        self.img = img^
        self.txt = txt^


# ── DOUBLE block saved activations (host BF16; half the bytes of F32) ────────
struct StreamSaved(Copyable, Movable):
    var x: List[BFloat16]        # [N,D]
    var ln1: List[BFloat16]      # [N,D]   layer_norm(x)
    var norm: List[BFloat16]     # [N,D]   modulate(ln1, scale1, shift1)
    var q_pre: List[BFloat16]    # [1,N,H,Dh]
    var k_pre: List[BFloat16]    # [1,N,H,Dh]
    var att: List[BFloat16]      # [N,D]   per-stream attention slice
    var attn_res: List[BFloat16] # [N,D]
    var ln2: List[BFloat16]      # [N,D]   layer_norm(attn_res)
    var mlp_in: List[BFloat16]   # [N,D]   modulate(ln2, scale2, shift2)
    var mlp_pre: List[BFloat16]  # [N,Fmlp]  linear(mlp_in, Wmlp0)+b  (gelu input)
    var mlp_h: List[BFloat16]    # [N,Fmlp]  gelu(mlp_pre)

    def __init__(
        out self,
        var x: List[BFloat16], var ln1: List[BFloat16], var norm: List[BFloat16],
        var q_pre: List[BFloat16], var k_pre: List[BFloat16],
        var att: List[BFloat16], var attn_res: List[BFloat16],
        var ln2: List[BFloat16], var mlp_in: List[BFloat16],
        var mlp_pre: List[BFloat16], var mlp_h: List[BFloat16],
    ):
        self.x = x^
        self.ln1 = ln1^
        self.norm = norm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.att = att^
        self.attn_res = attn_res^
        self.ln2 = ln2^
        self.mlp_in = mlp_in^
        self.mlp_pre = mlp_pre^
        self.mlp_h = mlp_h^


struct DoubleBlockSaved(Copyable, Movable):
    var img: StreamSaved
    var txt: StreamSaved
    var q_rope: List[BFloat16]   # [1,S,H,Dh]
    var k_rope: List[BFloat16]
    var v_joint: List[BFloat16]  # [1,S,H,Dh]

    def __init__(
        out self, var img: StreamSaved, var txt: StreamSaved,
        var q_rope: List[BFloat16], var k_rope: List[BFloat16], var v_joint: List[BFloat16],
    ):
        self.img = img^
        self.txt = txt^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.v_joint = v_joint^


struct DoubleBlockForward(Copyable, Movable):
    var img_out: List[Float32]
    var txt_out: List[Float32]
    var saved: DoubleBlockSaved

    def __init__(
        out self, var img_out: List[Float32], var txt_out: List[Float32],
        var saved: DoubleBlockSaved,
    ):
        self.img_out = img_out^
        self.txt_out = txt_out^
        self.saved = saved^


# ── DOUBLE block per-stream grads (input + all trainable weight/bias + mod) ───
struct StreamGrads(Copyable, Movable):
    var d_x: List[Float32]
    var d_wqkv: List[Float32]
    var d_bqkv: List[Float32]
    var d_wproj: List[Float32]
    var d_bproj: List[Float32]
    var d_wmlp0: List[Float32]
    var d_bmlp0: List[Float32]
    var d_wmlp2: List[Float32]
    var d_bmlp2: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]

    def __init__(
        out self,
        var d_x: List[Float32],
        var d_wqkv: List[Float32], var d_bqkv: List[Float32],
        var d_wproj: List[Float32], var d_bproj: List[Float32],
        var d_wmlp0: List[Float32], var d_bmlp0: List[Float32],
        var d_wmlp2: List[Float32], var d_bmlp2: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_shift1: List[Float32], var d_scale1: List[Float32], var d_gate1: List[Float32],
        var d_shift2: List[Float32], var d_scale2: List[Float32], var d_gate2: List[Float32],
    ):
        self.d_x = d_x^
        self.d_wqkv = d_wqkv^
        self.d_bqkv = d_bqkv^
        self.d_wproj = d_wproj^
        self.d_bproj = d_bproj^
        self.d_wmlp0 = d_wmlp0^
        self.d_bmlp0 = d_bmlp0^
        self.d_wmlp2 = d_wmlp2^
        self.d_bmlp2 = d_bmlp2^
        self.d_q_norm = d_q_norm^
        self.d_k_norm = d_k_norm^
        self.d_shift1 = d_shift1^
        self.d_scale1 = d_scale1^
        self.d_gate1 = d_gate1^
        self.d_shift2 = d_shift2^
        self.d_scale2 = d_scale2^
        self.d_gate2 = d_gate2^


struct DoubleBlockGrads(Copyable, Movable):
    var img: StreamGrads
    var txt: StreamGrads

    def __init__(out self, var img: StreamGrads, var txt: StreamGrads):
        self.img = img^
        self.txt = txt^


# ── per-stream pre-attention forward result (q_rms, k_rms, v for the join) ────
# Movable-only: holds move-only Tensor fields, consumed by-borrow then moved.
struct _StreamPre(Movable):
    var ln1: Tensor
    var norm: Tensor
    var q_pre: Tensor       # [1,N,H,Dh]  before rms
    var k_pre: Tensor
    var q_rms: Tensor       # [1,N,H,Dh]
    var k_rms: Tensor
    var v: Tensor

    def __init__(
        out self, var ln1: Tensor, var norm: Tensor,
        var q_pre: Tensor, var k_pre: Tensor,
        var q_rms: Tensor, var k_rms: Tensor, var v: Tensor,
    ):
        self.ln1 = ln1^
        self.norm = norm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.v = v^


def _stream_pre[
    H: Int, Dh: Int
](
    x: Tensor, w: StreamWeights, mv: ModVecs,
    N: Int, D: Int, eps: Float32, ones: Tensor, zeros: Tensor, ctx: DeviceContext,
) raises -> _StreamPre:
    var ln1 = layer_norm(x, ones, zeros, eps, ctx)
    var norm = modulate(ln1, _t(mv.scale1.copy(), [D], ctx), _t(mv.shift1.copy(), [D], ctx), ctx)
    var b = Optional[Tensor](w.bqkv[].clone(ctx))
    var qkv = linear(norm, w.wqkv[], b, ctx)   # [N,3D]
    var q_pre_flat = slice(qkv, 1, 0, D, ctx)
    var k_pre_flat = slice(qkv, 1, D, D, ctx)
    var v_flat = slice(qkv, 1, 2 * D, D, ctx)
    var q_pre = reshape_owned(q_pre_flat^, [1, N, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, N, H, Dh])
    var v = reshape_owned(v_flat^, [1, N, H, Dh])
    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    return _StreamPre(ln1^, norm^, q_pre^, k_pre^, q_rms^, k_rms^, v^)


# ── per-stream post-attention forward result ─────────────────────────────────
struct _StreamPost(Movable):
    var out: Tensor         # [N,D]  the stream final
    var attn_res: Tensor
    var ln2: Tensor
    var mlp_in: Tensor
    var mlp_pre: Tensor     # [N,Fmlp]  gelu input
    var mlp_h: Tensor       # [N,Fmlp]  gelu output

    def __init__(
        out self, var out: Tensor, var attn_res: Tensor,
        var ln2: Tensor, var mlp_in: Tensor,
        var mlp_pre: Tensor, var mlp_h: Tensor,
    ):
        self.out = out^
        self.attn_res = attn_res^
        self.ln2 = ln2^
        self.mlp_in = mlp_in^
        self.mlp_pre = mlp_pre^
        self.mlp_h = mlp_h^


def _stream_post(
    x: Tensor, att: Tensor, w: StreamWeights, mv: ModVecs,
    N: Int, D: Int, Fmlp: Int, eps: Float32, ones: Tensor, zeros: Tensor,
    ctx: DeviceContext,
) raises -> _StreamPost:
    var bp = Optional[Tensor](w.bproj[].clone(ctx))
    var out = linear(att, w.wproj[], bp, ctx)   # [N,D]
    var attn_res = residual_gate(x, _t(mv.gate1.copy(), [D], ctx), out, ctx)
    var ln2 = layer_norm(attn_res, ones, zeros, eps, ctx)
    var mlp_in = modulate(ln2, _t(mv.scale2.copy(), [D], ctx), _t(mv.shift2.copy(), [D], ctx), ctx)
    var b0 = Optional[Tensor](w.bmlp0[].clone(ctx))
    var mlp_pre = linear(mlp_in, w.wmlp0[], b0, ctx)   # [N,Fmlp]
    var mlp_h = gelu(mlp_pre, ctx)                   # [N,Fmlp]
    var b2 = Optional[Tensor](w.bmlp2[].clone(ctx))
    var mlp = linear(mlp_h, w.wmlp2[], b2, ctx)        # [N,D]
    var final = residual_gate(attn_res, _t(mv.gate2.copy(), [D], ctx), mlp, ctx)
    return _StreamPost(final^, attn_res^, ln2^, mlp_in^, mlp_pre^, mlp_h^)


# ── FORWARD of one DOUBLE block ──────────────────────────────────────────────
# cos/sin: precomputed JOINT-sequence rope tables [S*H, Dh/2] (3-axis Flux build,
# but the block just consumes them).
def double_block_forward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    img: List[Float32], txt: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)
    # rope tables in bf16 to match the bf16 q/k for the native bf16 rope/sdpa path.
    var cos_b = _to_bf16(cos, ctx)
    var sin_b = _to_bf16(sin, ctx)

    var img_x = _t(img, [N_IMG, D], ctx)
    var txt_x = _t(txt, [N_TXT, D], ctx)

    var ip = _stream_pre[H, Dh](img_x, w.img, img_mod, N_IMG, D, eps, ones_t, zeros_t, ctx)
    var tp = _stream_pre[H, Dh](txt_x, w.txt, txt_mod, N_TXT, D, eps, ones_t, zeros_t, ctx)

    # JOINT concat (txt FIRST, then img) along axis=1.
    var q = concat(1, ctx, tp.q_rms, ip.q_rms)   # [1,S,H,Dh]
    var k = concat(1, ctx, tp.k_rms, ip.k_rms)
    var v = concat(1, ctx, tp.v, ip.v)

    var q_rope = rope_interleaved(q, cos_b, sin_b, ctx)
    var k_rope = rope_interleaved(k, cos_b, sin_b, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)   # [1,S,H,Dh]

    var txt_att_4d = slice(att, 1, 0, N_TXT, ctx)
    var img_att_4d = slice(att, 1, N_TXT, N_IMG, ctx)
    var txt_att = reshape_owned(txt_att_4d^, [N_TXT, D])
    var img_att = reshape_owned(img_att_4d^, [N_IMG, D])

    var ipost = _stream_post(img_x, img_att, w.img, img_mod, N_IMG, D, Fmlp, eps, ones_t, zeros_t, ctx)
    var tpost = _stream_post(txt_x, txt_att, w.txt, txt_mod, N_TXT, D, Fmlp, eps, ones_t, zeros_t, ctx)

    var img_saved = StreamSaved(
        _f32_to_bf16(img), ip.ln1.to_host_bf16(ctx), ip.norm.to_host_bf16(ctx),
        ip.q_pre.to_host_bf16(ctx), ip.k_pre.to_host_bf16(ctx),
        img_att.to_host_bf16(ctx), ipost.attn_res.to_host_bf16(ctx),
        ipost.ln2.to_host_bf16(ctx), ipost.mlp_in.to_host_bf16(ctx),
        ipost.mlp_pre.to_host_bf16(ctx), ipost.mlp_h.to_host_bf16(ctx),
    )
    var txt_saved = StreamSaved(
        _f32_to_bf16(txt), tp.ln1.to_host_bf16(ctx), tp.norm.to_host_bf16(ctx),
        tp.q_pre.to_host_bf16(ctx), tp.k_pre.to_host_bf16(ctx),
        txt_att.to_host_bf16(ctx), tpost.attn_res.to_host_bf16(ctx),
        tpost.ln2.to_host_bf16(ctx), tpost.mlp_in.to_host_bf16(ctx),
        tpost.mlp_pre.to_host_bf16(ctx), tpost.mlp_h.to_host_bf16(ctx),
    )
    var saved = DoubleBlockSaved(
        img_saved^, txt_saved^,
        q_rope.to_host_bf16(ctx), k_rope.to_host_bf16(ctx), v.to_host_bf16(ctx),
    )

    var img_out = ipost.out.to_host(ctx)
    var txt_out = tpost.out.to_host(ctx)
    return DoubleBlockForward(img_out^, txt_out^, saved^)


# ── per-stream post-attention BACKWARD ───────────────────────────────────────
struct _StreamPostBack(Copyable, Movable):
    var d_x: List[Float32]      # grad into stream input via gate1 residual [N,D]
    var d_att: List[Float32]    # grad into the joint-attention slice [N,D]
    var d_wproj: List[Float32]
    var d_bproj: List[Float32]
    var d_wmlp0: List[Float32]
    var d_bmlp0: List[Float32]
    var d_wmlp2: List[Float32]
    var d_bmlp2: List[Float32]
    var d_gate1: List[Float32]
    var d_shift2: List[Float32]
    var d_scale2: List[Float32]
    var d_gate2: List[Float32]

    def __init__(
        out self, var d_x: List[Float32], var d_att: List[Float32],
        var d_wproj: List[Float32], var d_bproj: List[Float32],
        var d_wmlp0: List[Float32], var d_bmlp0: List[Float32],
        var d_wmlp2: List[Float32], var d_bmlp2: List[Float32],
        var d_gate1: List[Float32],
        var d_shift2: List[Float32], var d_scale2: List[Float32], var d_gate2: List[Float32],
    ):
        self.d_x = d_x^
        self.d_att = d_att^
        self.d_wproj = d_wproj^
        self.d_bproj = d_bproj^
        self.d_wmlp0 = d_wmlp0^
        self.d_bmlp0 = d_bmlp0^
        self.d_wmlp2 = d_wmlp2^
        self.d_bmlp2 = d_bmlp2^
        self.d_gate1 = d_gate1^
        self.d_shift2 = d_shift2^
        self.d_scale2 = d_scale2^
        self.d_gate2 = d_gate2^


def _stream_post_backward(
    d_out: Tensor, x: Tensor, att: Tensor,
    w: StreamWeights, mv: ModVecs, sv: StreamSaved,
    N: Int, D: Int, Fmlp: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
) raises -> _StreamPostBack:
    # All saved acts re-uploaded as BF16 device → bf16-native backward matmuls.
    # d_out is F32 (the upstream grad); gate_residual_backward needs grad_out + gate F32.
    var attn_res_t = _tb16(sv.attn_res.copy(), [N, D], ctx)
    var mlp_h_t = _tb16(sv.mlp_h.copy(), [N, Fmlp], ctx)
    var b2 = Optional[Tensor](w.bmlp2[].clone(ctx))
    var mlp_y = linear(mlp_h_t, w.wmlp2[], b2, ctx)            # bf16
    # F32-ONLY: gate_residual_backward grad_out/gate must be F32 (x/y may be bf16).
    var grg2 = gate_residual_backward(
        d_out, attn_res_t, _tf32(mv.gate2.copy(), [D], ctx), mlp_y, ctx
    )
    var d_gate2 = grg2.d_g.to_host(ctx)
    # grg2.d_y is F32 → cast to bf16 to feed the bf16 backward matmul chain.
    var d_y2_b = _to_bf16(grg2.d_y, ctx)

    # mlp = linear(mlp_h, Wmlp2, bmlp2)
    var lb_mlp2 = linear_backward(d_y2_b, mlp_h_t, w.wmlp2[], N, Fmlp, D, ctx)
    var d_wmlp2 = lb_mlp2.d_w.to_host(ctx)
    var d_bmlp2 = lb_mlp2.d_b.to_host(ctx)

    # mlp_h = gelu(mlp_pre)
    var mlp_pre_t = _tb16(sv.mlp_pre.copy(), [N, Fmlp], ctx)
    var d_mlp_pre = gelu_backward(lb_mlp2.d_x, mlp_pre_t, ctx)  # bf16

    # mlp_pre = linear(mlp_in, Wmlp0, bmlp0)
    var mlp_in_t = _tb16(sv.mlp_in.copy(), [N, D], ctx)
    var lb_mlp0 = linear_backward(d_mlp_pre, mlp_in_t, w.wmlp0[], N, D, Fmlp, ctx)
    var d_wmlp0 = lb_mlp0.d_w.to_host(ctx)
    var d_bmlp0 = lb_mlp0.d_b.to_host(ctx)

    # mlp_in = modulate(ln2, scale2, shift2)
    var ln2_t = _tb16(sv.ln2.copy(), [N, D], ctx)
    var mb2 = modulate_backward(lb_mlp0.d_x, ln2_t, _t(mv.scale2.copy(), [D], ctx), ctx)
    var d_scale2 = mb2.d_scale.to_host(ctx)
    var d_shift2 = mb2.d_shift.to_host(ctx)

    # ln2 = layer_norm(attn_res, 1, 0)
    var lnb2 = layer_norm_backward(mb2.d_x, attn_res_t, ones, eps, ctx)  # bf16
    # attn_res feeds BOTH the residual (grg2.d_x F32) AND ln2 (bf16) -> SUM (bf16).
    var d_attn_res_total = add(_to_bf16(grg2.d_x, ctx), lnb2.d_x, ctx)
    # gate_residual_backward grad_out must be F32 again.
    var d_attn_res_f32 = _to_f32(d_attn_res_total, ctx)

    # attn_res = residual_gate(x, gate1, proj_out): o = x + gate1*proj_out
    var bp = Optional[Tensor](w.bproj[].clone(ctx))
    var proj_out = linear(att, w.wproj[], bp, ctx)   # recompute proj output (bf16)
    var grg1 = gate_residual_backward(
        d_attn_res_f32, x, _tf32(mv.gate1.copy(), [D], ctx), proj_out, ctx
    )
    var d_gate1 = grg1.d_g.to_host(ctx)
    var d_x_res = grg1.d_x.to_host(ctx)

    # proj_out = linear(att, Wproj, bproj)
    var lb_p = linear_backward(_to_bf16(grg1.d_y, ctx), att, w.wproj[], N, D, D, ctx)
    var d_wproj = lb_p.d_w.to_host(ctx)
    var d_bproj = lb_p.d_b.to_host(ctx)
    var d_att = lb_p.d_x.to_host(ctx)

    return _StreamPostBack(
        d_x_res^, d_att^, d_wproj^, d_bproj^,
        d_wmlp0^, d_bmlp0^, d_wmlp2^, d_bmlp2^,
        d_gate1=d_gate1^, d_shift2=d_shift2^, d_scale2=d_scale2^, d_gate2=d_gate2^,
    )


# ── per-stream pre-attention BACKWARD ────────────────────────────────────────
struct _StreamPreBack(Copyable, Movable):
    var d_x: List[Float32]
    var d_wqkv: List[Float32]
    var d_bqkv: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    var d_shift1: List[Float32]
    var d_scale1: List[Float32]

    def __init__(
        out self, var d_x: List[Float32],
        var d_wqkv: List[Float32], var d_bqkv: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_shift1: List[Float32], var d_scale1: List[Float32],
    ):
        self.d_x = d_x^
        self.d_wqkv = d_wqkv^
        self.d_bqkv = d_bqkv^
        self.d_q_norm = d_q_norm^
        self.d_k_norm = d_k_norm^
        self.d_shift1 = d_shift1^
        self.d_scale1 = d_scale1^


def _stream_pre_backward[
    H: Int, Dh: Int
](
    d_q_rms: Tensor, d_k_rms: Tensor, d_v: Tensor,
    w: StreamWeights, mv: ModVecs, sv: StreamSaved,
    N: Int, D: Int, eps: Float32, ones: Tensor, ctx: DeviceContext,
) raises -> _StreamPreBack:
    # Incoming grads (cat_backward outputs) are F32; cast to bf16 to keep the
    # pre-attention backward bf16-native and dtype-consistent with the bf16 saved acts.
    var dq_b = _to_bf16(d_q_rms, ctx)
    var dk_b = _to_bf16(d_k_rms, ctx)
    var dv_b = _to_bf16(d_v, ctx)
    var q_pre_t = _tb16(sv.q_pre.copy(), [1, N, H, Dh], ctx)
    var k_pre_t = _tb16(sv.k_pre.copy(), [1, N, H, Dh], ctx)
    var rb_q = rms_norm_backward(dq_b, q_pre_t, w.q_norm[], eps, ctx)  # bf16
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(dk_b, k_pre_t, w.k_norm[], eps, ctx)  # bf16
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [N, D])
    reshape_in_place(rb_k.d_x, [N, D])
    var d_v_flat = reshape(dv_b, [N, D], ctx)
    var d_qkv = concat(1, ctx, rb_q.d_x, rb_k.d_x, d_v_flat)   # [N,3D] bf16

    # qkv = linear(norm, Wqkv, bqkv)
    var norm_t = _tb16(sv.norm.copy(), [N, D], ctx)
    var lb_qkv = linear_backward(d_qkv, norm_t, w.wqkv[], N, D, 3 * D, ctx)
    var d_wqkv = lb_qkv.d_w.to_host(ctx)
    var d_bqkv = lb_qkv.d_b.to_host(ctx)

    # norm = modulate(ln1, scale1, shift1)
    var ln1_t = _tb16(sv.ln1.copy(), [N, D], ctx)
    var mb1 = modulate_backward(lb_qkv.d_x, ln1_t, _t(mv.scale1.copy(), [D], ctx), ctx)
    var d_scale1 = mb1.d_scale.to_host(ctx)
    var d_shift1 = mb1.d_shift.to_host(ctx)

    # ln1 = layer_norm(x, 1, 0)
    var x_t = _tb16(sv.x.copy(), [N, D], ctx)
    var lnb1 = layer_norm_backward(mb1.d_x, x_t, ones, eps, ctx)
    var d_x_norm = lnb1.d_x.to_host(ctx)
    return _StreamPreBack(d_x_norm^, d_wqkv^, d_bqkv^, d_q_norm^, d_k_norm^, d_shift1^, d_scale1^)


# ── BACKWARD of one DOUBLE block (hand-chained; joint coupling) ──────────────
def double_block_backward[
    H: Int, Dh: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    d_img_out: List[Float32], d_txt_out: List[Float32],
    w: DoubleBlockWeights, img_mod: ModVecs, txt_mod: ModVecs, saved: DoubleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> DoubleBlockGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)

    # Upstream output grads stay F32 — gate_residual_backward needs grad_out F32.
    var d_io_t = _tf32(d_img_out, [N_IMG, D], ctx)
    var d_to_t = _tf32(d_txt_out, [N_TXT, D], ctx)

    var img_x = _tb16(saved.img.x.copy(), [N_IMG, D], ctx)
    var txt_x = _tb16(saved.txt.x.copy(), [N_TXT, D], ctx)
    var img_att = _tb16(saved.img.att.copy(), [N_IMG, D], ctx)
    var txt_att = _tb16(saved.txt.att.copy(), [N_TXT, D], ctx)

    var ipb = _stream_post_backward(
        d_io_t, img_x, img_att, w.img, img_mod, saved.img, N_IMG, D, Fmlp, eps, ones_t, ctx
    )
    var tpb = _stream_post_backward(
        d_to_t, txt_x, txt_att, w.txt, txt_mod, saved.txt, N_TXT, D, Fmlp, eps, ones_t, ctx
    )

    # join per-stream attention-slice grads back into joint d_att (txt FIRST).
    # _t uploads bf16 → matches the bf16 q_rope/k_rope/v for sdpa_backward.
    var d_tatt_4d = _t(tpb.d_att.copy(), [1, N_TXT, H, Dh], ctx)
    var d_iatt_4d = _t(ipb.d_att.copy(), [1, N_IMG, H, Dh], ctx)
    var d_att_joint = concat(1, ctx, d_tatt_4d, d_iatt_4d)   # [1,S,H,Dh] bf16

    # sdpa backward (JOINT) — bf16-native (q/k/v/d_out all bf16).
    var q_rope_t = _tb16(saved.q_rope.copy(), [1, S, H, Dh], ctx)
    var k_rope_t = _tb16(saved.k_rope.copy(), [1, S, H, Dh], ctx)
    var v_joint_t = _tb16(saved.v_joint.copy(), [1, S, H, Dh], ctx)
    var sb = sdpa_backward[1, S, H, Dh](q_rope_t, k_rope_t, v_joint_t, d_att_joint, scale, ctx)

    # F32-ONLY: rope_backward needs grad + cos/sin F32. sb.d_q/d_k are bf16 → cast up.
    var d_q_joint = rope_backward(_to_f32(sb.d_q, ctx), cos, sin, True, ctx)
    var d_k_joint = rope_backward(_to_f32(sb.d_k, ctx), cos, sin, True, ctx)

    # F32-ONLY: cat_backward needs F32 grad. d_q_joint/d_k_joint already F32;
    # sb.d_v is bf16 → cast up.
    var cq = cat_backward(d_q_joint, N_TXT, N_IMG, 1, ctx)
    var ck = cat_backward(d_k_joint, N_TXT, N_IMG, 1, ctx)
    var cv = cat_backward(_to_f32(sb.d_v, ctx), N_TXT, N_IMG, 1, ctx)

    var iprb = _stream_pre_backward[H, Dh](
        cq.d_1, ck.d_1, cv.d_1, w.img, img_mod, saved.img, N_IMG, D, eps, ones_t, ctx
    )
    var tprb = _stream_pre_backward[H, Dh](
        cq.d_0, ck.d_0, cv.d_0, w.txt, txt_mod, saved.txt, N_TXT, D, eps, ones_t, ctx
    )

    var d_img_x = _add_lists(ipb.d_x, iprb.d_x)
    var d_txt_x = _add_lists(tpb.d_x, tprb.d_x)

    var img_grads = StreamGrads(
        d_img_x^,
        iprb.d_wqkv.copy(), iprb.d_bqkv.copy(),
        ipb.d_wproj.copy(), ipb.d_bproj.copy(),
        ipb.d_wmlp0.copy(), ipb.d_bmlp0.copy(),
        ipb.d_wmlp2.copy(), ipb.d_bmlp2.copy(),
        iprb.d_q_norm.copy(), iprb.d_k_norm.copy(),
        iprb.d_shift1.copy(), iprb.d_scale1.copy(), ipb.d_gate1.copy(),
        ipb.d_shift2.copy(), ipb.d_scale2.copy(), ipb.d_gate2.copy(),
    )
    var txt_grads = StreamGrads(
        d_txt_x^,
        tprb.d_wqkv.copy(), tprb.d_bqkv.copy(),
        tpb.d_wproj.copy(), tpb.d_bproj.copy(),
        tpb.d_wmlp0.copy(), tpb.d_bmlp0.copy(),
        tpb.d_wmlp2.copy(), tpb.d_bmlp2.copy(),
        tprb.d_q_norm.copy(), tprb.d_k_norm.copy(),
        tprb.d_shift1.copy(), tprb.d_scale1.copy(), tpb.d_gate1.copy(),
        tpb.d_shift2.copy(), tpb.d_scale2.copy(), tpb.d_gate2.copy(),
    )
    return DoubleBlockGrads(img_grads^, txt_grads^)


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE block — Flux: WITH biases, GELU MLP fused into linear1/linear2.
#   w1: [3D+Fmlp, D]  b1: [3D+Fmlp]     (qkv + mlp_up; Fmlp = 4D)
#   w2: [D, D+Fmlp]   b2: [D]           (attn_proj + mlp_down fused on input)
#   q_norm/k_norm: [Dh]
# ═══════════════════════════════════════════════════════════════════════════
struct SingleBlockWeights(Copyable, Movable):
    var w1: TArc
    var b1: TArc
    var w2: TArc
    var b2: TArc
    var q_norm: TArc
    var k_norm: TArc

    def __init__(
        out self,
        var w1: List[Float32], var b1: List[Float32],
        var w2: List[Float32], var b2: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
        D: Int, Fmlp: Int, Dh: Int, ctx: DeviceContext,
    ) raises:
        self.w1 = TArc(Tensor.from_host(w1^, [3 * D + Fmlp, D], STDtype.BF16, ctx))
        self.b1 = TArc(Tensor.from_host(b1^, [3 * D + Fmlp], STDtype.BF16, ctx))
        self.w2 = TArc(Tensor.from_host(w2^, [D, D + Fmlp], STDtype.BF16, ctx))
        self.b2 = TArc(Tensor.from_host(b2^, [D], STDtype.BF16, ctx))
        self.q_norm = TArc(Tensor.from_host(q_norm^, [Dh], STDtype.BF16, ctx))
        self.k_norm = TArc(Tensor.from_host(k_norm^, [Dh], STDtype.BF16, ctx))

    def __init__(
        out self,
        var w1: TArc, var b1: TArc,
        var w2: TArc, var b2: TArc,
        var q_norm: TArc, var k_norm: TArc,
    ):
        self.w1 = w1^
        self.b1 = b1^
        self.w2 = w2^
        self.b2 = b2^
        self.q_norm = q_norm^
        self.k_norm = k_norm^


struct SingleBlockSaved(Copyable, Movable):
    var x: List[BFloat16]        # [S,D]
    var ln: List[BFloat16]       # [S,D]
    var norm: List[BFloat16]     # [S,D]
    var q_pre: List[BFloat16]    # [1,S,H,Dh]
    var k_pre: List[BFloat16]
    var q_rope: List[BFloat16]   # [1,S,H,Dh]
    var k_rope: List[BFloat16]
    var v: List[BFloat16]        # [1,S,H,Dh]
    var att_flat: List[BFloat16] # [S,D]
    var mlp_in: List[BFloat16]   # [S,Fmlp]  gelu input (= fused[:, 3D:])
    var mlp_h: List[BFloat16]    # [S,Fmlp]  gelu output
    var out_in: List[BFloat16]   # [S, D+Fmlp]  concat(att_flat, mlp_h)

    def __init__(
        out self,
        var x: List[BFloat16], var ln: List[BFloat16], var norm: List[BFloat16],
        var q_pre: List[BFloat16], var k_pre: List[BFloat16],
        var q_rope: List[BFloat16], var k_rope: List[BFloat16], var v: List[BFloat16],
        var att_flat: List[BFloat16],
        var mlp_in: List[BFloat16], var mlp_h: List[BFloat16], var out_in: List[BFloat16],
    ):
        self.x = x^
        self.ln = ln^
        self.norm = norm^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.v = v^
        self.att_flat = att_flat^
        self.mlp_in = mlp_in^
        self.mlp_h = mlp_h^
        self.out_in = out_in^


struct SingleBlockForward(Copyable, Movable):
    var out: List[Float32]
    var saved: SingleBlockSaved

    def __init__(out self, var out: List[Float32], var saved: SingleBlockSaved):
        self.out = out^
        self.saved = saved^


struct SingleBlockGrads(Copyable, Movable):
    var d_x: List[Float32]
    var d_w1: List[Float32]
    var d_b1: List[Float32]
    var d_w2: List[Float32]
    var d_b2: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    var d_shift: List[Float32]
    var d_scale: List[Float32]
    var d_gate: List[Float32]

    def __init__(
        out self,
        var d_x: List[Float32],
        var d_w1: List[Float32], var d_b1: List[Float32],
        var d_w2: List[Float32], var d_b2: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_shift: List[Float32], var d_scale: List[Float32], var d_gate: List[Float32],
    ):
        self.d_x = d_x^
        self.d_w1 = d_w1^
        self.d_b1 = d_b1^
        self.d_w2 = d_w2^
        self.d_b2 = d_b2^
        self.d_q_norm = d_q_norm^
        self.d_k_norm = d_k_norm^
        self.d_shift = d_shift^
        self.d_scale = d_scale^
        self.d_gate = d_gate^


# ── FORWARD of one SINGLE block ──────────────────────────────────────────────
def single_block_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: SingleBlockWeights, mv: SingleModVecs,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var zeros_t = _t(_zeros(D), [D], ctx)
    var cos_b = _to_bf16(cos, ctx)
    var sin_b = _to_bf16(sin, ctx)

    var x_t = _t(x, [S, D], ctx)
    var ln_t = layer_norm(x_t, ones_t, zeros_t, eps, ctx)
    var norm_t = modulate(ln_t, _t(mv.scale.copy(), [D], ctx), _t(mv.shift.copy(), [D], ctx), ctx)

    var b1 = Optional[Tensor](w.b1[].clone(ctx))
    var fused = linear(norm_t, w.w1[], b1, ctx)   # [S, 3D+Fmlp]

    var qkv = slice(fused, 1, 0, 3 * D, ctx)
    var mlp_in = slice(fused, 1, 3 * D, Fmlp, ctx)   # [S,Fmlp]

    var q_pre_flat = slice(qkv, 1, 0, D, ctx)
    var k_pre_flat = slice(qkv, 1, D, D, ctx)
    var v_flat = slice(qkv, 1, 2 * D, D, ctx)
    var q_pre = reshape_owned(q_pre_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_pre_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos_b, sin_b, ctx)
    var k_rope = rope_interleaved(k_rms, cos_b, sin_b, ctx)
    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var mlp_h = gelu(mlp_in, ctx)   # [S,Fmlp]

    var out_in = concat(1, ctx, att_flat, mlp_h)   # [S, D+Fmlp]

    var b2 = Optional[Tensor](w.b2[].clone(ctx))
    var out_proj = linear(out_in, w.w2[], b2, ctx)   # [S,D]

    var result = residual_gate(x_t, _t(mv.gate.copy(), [D], ctx), out_proj, ctx)

    var saved = SingleBlockSaved(
        _f32_to_bf16(x), ln_t.to_host_bf16(ctx), norm_t.to_host_bf16(ctx),
        q_pre.to_host_bf16(ctx), k_pre.to_host_bf16(ctx),
        q_rope.to_host_bf16(ctx), k_rope.to_host_bf16(ctx), v.to_host_bf16(ctx),
        att_flat.to_host_bf16(ctx),
        mlp_in.to_host_bf16(ctx), mlp_h.to_host_bf16(ctx), out_in.to_host_bf16(ctx),
    )
    return SingleBlockForward(result.to_host(ctx), saved^)


# ── BACKWARD of one SINGLE block (hand-chained) ──────────────────────────────
def single_block_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: SingleBlockWeights, mv: SingleModVecs, saved: SingleBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, Fmlp: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> SingleBlockGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ones_t = _t(_ones(D), [D], ctx)
    var scale_t = _t(mv.scale.copy(), [D], ctx)       # bf16 for modulate_backward
    var gate_t_f32 = _tf32(mv.gate.copy(), [D], ctx)  # F32 for gate_residual_backward

    var d_out_t = _tf32(d_out, [S, D], ctx)           # F32 (gate_residual_backward grad_out)
    var x_t = _tb16(saved.x.copy(), [S, D], ctx)      # bf16 saved act
    var out_in_t = _tb16(saved.out_in.copy(), [S, D + Fmlp], ctx)

    # result = residual_gate(x, gate, out): o = x + gate*out
    # F32-ONLY: gate_residual_backward grad_out/gate F32 (x/y may be bf16).
    var b2 = Optional[Tensor](w.b2[].clone(ctx))
    var out_y = linear(out_in_t, w.w2[], b2, ctx)     # bf16
    var grg = gate_residual_backward(d_out_t, x_t, gate_t_f32, out_y, ctx)
    var d_gate = grg.d_g.to_host(ctx)

    # out = linear(out_in, W2, b2) — grg.d_y is F32 → bf16 for the matmul chain.
    var lb_w2 = linear_backward(_to_bf16(grg.d_y, ctx), out_in_t, w.w2[], S, D + Fmlp, D, ctx)
    var d_w2 = lb_w2.d_w.to_host(ctx)
    var d_b2 = lb_w2.d_b.to_host(ctx)

    # out_in = concat(axis=2, att_flat, mlp_h). F32-ONLY: cat_backward needs F32 grad.
    var dx_w2_f32 = _to_f32(lb_w2.d_x, ctx)
    reshape_in_place(dx_w2_f32, [1, S, D + Fmlp])
    var cb = cat_backward(dx_w2_f32, D, Fmlp, 2, ctx)   # F32 outputs
    reshape_in_place(cb.d_0, [1, S, H, Dh])   # [1,S,D] == [1,S,H,Dh]
    reshape_in_place(cb.d_1, [S, Fmlp])       # [1,S,Fmlp] == [S,Fmlp]
    # back to bf16 for the bf16-native arms.
    var d_att_flat_b = _to_bf16(cb.d_0, ctx)
    var d_mlp_h_b = _to_bf16(cb.d_1, ctx)

    # mlp_h = gelu(mlp_in)
    var mlp_in_t = _tb16(saved.mlp_in.copy(), [S, Fmlp], ctx)
    var d_mlp_in = gelu_backward(d_mlp_h_b, mlp_in_t, ctx)   # [S,Fmlp] bf16

    # att branch: d_att_flat [1,S,H,Dh] -> sdpa backward (bf16-native).
    var q_rope_t = _tb16(saved.q_rope.copy(), [1, S, H, Dh], ctx)
    var k_rope_t = _tb16(saved.k_rope.copy(), [1, S, H, Dh], ctx)
    var v_t = _tb16(saved.v.copy(), [1, S, H, Dh], ctx)
    var sb = sdpa_backward[1, S, H, Dh](q_rope_t, k_rope_t, v_t, d_att_flat_b, scale, ctx)

    # F32-ONLY: rope_backward grad + cos/sin F32. sb.d_q/d_k bf16 → cast up.
    var d_q_rms = rope_backward(_to_f32(sb.d_q, ctx), cos, sin, True, ctx)  # F32
    var d_k_rms = rope_backward(_to_f32(sb.d_k, ctx), cos, sin, True, ctx)  # F32

    # rms_norm backward for q and k (back to bf16 to keep the chain bf16-native).
    var q_pre_t = _tb16(saved.q_pre.copy(), [1, S, H, Dh], ctx)
    var k_pre_t = _tb16(saved.k_pre.copy(), [1, S, H, Dh], ctx)
    var rb_q = rms_norm_backward(_to_bf16(d_q_rms, ctx), q_pre_t, w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(_to_bf16(d_k_rms, ctx), k_pre_t, w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    # join d_q_pre|d_k_pre|d_v into d_qkv [S,3D] (all bf16). sb.d_v is bf16.
    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])
    var d_qkv = concat(1, ctx, rb_q.d_x, rb_k.d_x, sb.d_v)   # [S,3D] bf16

    # join the qkv grad and mlp_in grad back into d_fused [S, 3D+Fmlp]
    var d_fused = concat(1, ctx, d_qkv, d_mlp_in)            # bf16

    # fused = linear(norm, W1, b1)
    var norm_t = _tb16(saved.norm.copy(), [S, D], ctx)
    var lb_w1 = linear_backward(d_fused, norm_t, w.w1[], S, D, 3 * D + Fmlp, ctx)
    var d_w1 = lb_w1.d_w.to_host(ctx)
    var d_b1 = lb_w1.d_b.to_host(ctx)

    # norm = modulate(ln, scale, shift)  (lb_w1.d_x, ln_t, scale_t all bf16)
    var ln_t = _tb16(saved.ln.copy(), [S, D], ctx)
    var mb = modulate_backward(lb_w1.d_x, ln_t, scale_t, ctx)
    var d_scale = mb.d_scale.to_host(ctx)
    var d_shift = mb.d_shift.to_host(ctx)

    # ln = layer_norm(x, 1, 0)
    var lnb = layer_norm_backward(mb.d_x, x_t, ones_t, eps, ctx)

    # x feeds BOTH the residual (grg.d_x) AND layer_norm(x) -> SUM (host F32).
    var d_x_res = grg.d_x.to_host(ctx)
    var d_x_norm = lnb.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    return SingleBlockGrads(
        d_x^, d_w1^, d_b1^, d_w2^, d_b2^, d_q_norm^, d_k_norm^,
        d_shift^, d_scale^, d_gate^,
    )
