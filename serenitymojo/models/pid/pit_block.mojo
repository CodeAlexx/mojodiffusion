# models/pid/pit_block.mojo — PiD PiTBlock (per-patch pixel transformer block).
#
# Pure-Mojo + MAX 26.3, NVIDIA GPU. Port of PiTBlock from the PiD repo
# (pid/_src/networks/pixeldit_official.py:416-509), no-context-parallel
# (cp_size=1), mask=None path. pixel_depth blocks stack this module.
#
# Reference forward (BL = B*L, C = pixel_dim, P2 = patch_size^2):
#   cond = adaLN(s_cond)                          # [BL, 6*C*P2]
#   cond = cond.view(BL, P2, 6*C); chunk6 ->      shift/scale/gate {msa,mlp}
#   x_norm = RMSNorm1(x) * (1+scale_msa) + shift_msa          # [BL,P2,C]
#   x_comp = compress(x_norm.view(BL, P2*C)).view(B, L, attn_dim)
#   attn   = RotaryAttention(x_comp, pos)                     # [B, L, attn_dim]
#   attn_e = expand(attn.view(B*L, attn_dim)).view(BL, P2, C)
#   x = x + gate_msa * attn_e
#   mlp = MLP(RMSNorm2(x) * (1+scale_mlp) + shift_mlp)        # exact-erf GELU
#   x = x + gate_mlp * mlp
#
# RotaryAttention (qkv_bias=False, qk_norm=True, head RMSNorm eps=1e-6):
#   qkv = qkv(x).reshape(B,N,3,H,Hc).permute -> q,k,v [B,N,H,Hc]
#   q = q_norm(q); k = k_norm(k)   (RMSNorm over Hc)
#   q,k = apply_rotary_emb(q,k, pos[N, Hc/2])   (interleaved complex, bcast B,H)
#   SDPA(q,k,v) over N, scale = Hc^-0.5
#   out = proj(out.reshape(B,N,attn_dim))
#
# apply_adaln scale/shift are PER-TOKEN [BL,P2,C] (not per-channel), so we use
# broadcasting elementwise mul/add — NOT ops/elementwise.modulate (which is
# per-channel [D]). The MLP uses nn.GELU() (EXACT erf gelu), not the tanh
# approximation in ops/activations.gelu, so we provide an exact-erf kernel here.
#
# All compute F32 (parity gate cos>=0.999). Mojo 1.0.0b1.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from std.math import erf, sqrt
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.tensor_algebra import add, mul, reshape, slice, add_scalar
from serenitymojo.models.pid.pid_ops import ntk_rope_tables_2d, RopeTables


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _INV_SQRT2 = Float32(0.7071067811865476)


def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Device-to-device byte copy (bias needs an owned Tensor for Optional)."""
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


# ════════════════════════════════════════════════════════════════════════════
# exact-erf GELU  (matches nn.GELU() default): 0.5*x*(1+erf(x/sqrt(2)))
# ════════════════════════════════════════════════════════════════════════════
def _gelu_erf_kernel(
    x: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        var v = rebind[Scalar[DType.float32]](x[i])
        o[i] = Float32(0.5) * v * (Float32(1.0) + erf(v * _INV_SQRT2))


def gelu_erf(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Exact erf GELU (PyTorch nn.GELU() default). F32 only."""
    if x.dtype() != STDtype.F32:
        raise Error("gelu_erf: F32 only")
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var X = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_gelu_erf_kernel, _gelu_erf_kernel](
        X, O, n, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), STDtype.F32)


# ════════════════════════════════════════════════════════════════════════════
# RoPE table expand: pos [L, half] -> [B*L*H, half] broadcasting over B and H.
# Layout of rope_interleaved rows: (b,l,h) flattened as ((b*L + l)*H + h).
# apply_rotary_emb broadcasts freqs over batch and head: freqs_cis[None,:,None,:]
# so row (b,l,h) maps to pos table row l.
# ════════════════════════════════════════════════════════════════════════════
def _rope_expand_kernel(
    pos: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [L*half]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],     # [B*L*H*half]
    B: Int, L: Int, H: Int, half: Int,
):
    var idx = Int(global_idx.x)
    var total = B * L * H * half
    if idx < total:
        var j = idx % half
        var rest = idx // half
        var h = rest % H
        rest = rest // H
        var l = rest % L
        # broadcast over b and h: source row is l
        o[idx] = pos[l * half + j]


def _expand_rope_table(
    pos: Tensor, B: Int, L: Int, H: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    """Expand a [L, half] cos/sin table to [B*L*H, half] for rope_interleaved
    (rows ordered (b,l,h))."""
    var n = B * L * H * half
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl_in = RuntimeLayout[_DYN1].row_major(IndexList[1](L * half))
    var rl_out = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        pos.buf.unsafe_ptr().bitcast[Float32](), rl_in
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl_out
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_rope_expand_kernel, _rope_expand_kernel](
        P, O, B, L, H, half, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [B * L * H, half], STDtype.F32)


# ════════════════════════════════════════════════════════════════════════════
# PiTBlock weights (all F32 device tensors; PyTorch row-major [out,in]).
# ════════════════════════════════════════════════════════════════════════════
struct PiTBlockWeights(Movable):
    var compress_w: Tensor   # [attn_dim, P2*pixel_dim]
    var compress_b: Tensor   # [attn_dim]
    var expand_w: Tensor     # [P2*pixel_dim, attn_dim]
    var expand_b: Tensor     # [P2*pixel_dim]
    var qkv_w: Tensor        # [3*attn_dim, attn_dim]  (qkv_bias=False)
    var proj_w: Tensor       # [attn_dim, attn_dim]
    var proj_b: Tensor       # [attn_dim]
    var qnorm_w: Tensor      # [head_dim]
    var knorm_w: Tensor      # [head_dim]
    var norm1_w: Tensor      # [pixel_dim]
    var norm2_w: Tensor      # [pixel_dim]
    var fc1_w: Tensor        # [mlp_hidden, pixel_dim]
    var fc1_b: Tensor        # [mlp_hidden]
    var fc2_w: Tensor        # [pixel_dim, mlp_hidden]
    var fc2_b: Tensor        # [pixel_dim]
    var adaln_w: Tensor      # [6*pixel_dim*P2, context_dim]
    var adaln_b: Tensor      # [6*pixel_dim*P2]

    def __init__(
        out self,
        var compress_w: Tensor, var compress_b: Tensor,
        var expand_w: Tensor, var expand_b: Tensor,
        var qkv_w: Tensor, var proj_w: Tensor, var proj_b: Tensor,
        var qnorm_w: Tensor, var knorm_w: Tensor,
        var norm1_w: Tensor, var norm2_w: Tensor,
        var fc1_w: Tensor, var fc1_b: Tensor,
        var fc2_w: Tensor, var fc2_b: Tensor,
        var adaln_w: Tensor, var adaln_b: Tensor,
    ):
        self.compress_w = compress_w^; self.compress_b = compress_b^
        self.expand_w = expand_w^; self.expand_b = expand_b^
        self.qkv_w = qkv_w^; self.proj_w = proj_w^; self.proj_b = proj_b^
        self.qnorm_w = qnorm_w^; self.knorm_w = knorm_w^
        self.norm1_w = norm1_w^; self.norm2_w = norm2_w^
        self.fc1_w = fc1_w^; self.fc1_b = fc1_b^
        self.fc2_w = fc2_w^; self.fc2_b = fc2_b^
        self.adaln_w = adaln_w^; self.adaln_b = adaln_b^


# ════════════════════════════════════════════════════════════════════════════
# RotaryAttention forward (no-CP, mask=None). x_comp [B, L, attn_dim].
# ════════════════════════════════════════════════════════════════════════════
def _rotary_attention[
    Bp: Int, Lp: Int, Hp: Int, Dhp: Int
](
    x_comp: Tensor,            # [B, L, attn_dim]
    w: PiTBlockWeights,
    cos_exp: Tensor,           # [B*L*H, head_dim/2]
    sin_exp: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var attn_dim = Hp * Dhp
    # qkv = x_comp @ qkv_wᵀ  -> [B, L, 3*attn_dim]
    var qkv = linear(x_comp, w.qkv_w, Optional[Tensor](None), ctx)
    # reshape [B*L, 3, H, Dh] then split q,k,v each [B,L,H,Dh].
    # PyTorch: reshape(B,N,3,H,Hc).permute(2,0,1,3,4) -> [3,B,N,H,Hc].
    # qkv flat row-major is [B, L, 3*H*Dh] with last dim ordered (s, h, dh),
    # s in {q=0,k=1,v=2}. So q = slice along last dim [0, attn_dim), etc.
    # First reshape to [B, L, 3, H*Dh] is the same bytes; slice dim=2.
    var qkv_r = reshape(qkv, [Bp * Lp, 3, attn_dim], ctx)
    var q = slice(qkv_r, 1, 0, 1, ctx)   # [B*L, 1, attn_dim]
    var k = slice(qkv_r, 1, 1, 1, ctx)
    var v = slice(qkv_r, 1, 2, 1, ctx)
    # reshape each to [B*L*H, Dh] for head-wise RMSNorm over Dh.
    var q_hd = reshape(q, [Bp * Lp * Hp, Dhp], ctx)
    var k_hd = reshape(k, [Bp * Lp * Hp, Dhp], ctx)
    var q_n = rms_norm(q_hd, w.qnorm_w, Float32(1e-6), ctx)   # [B*L*H, Dh]
    var k_n = rms_norm(k_hd, w.knorm_w, Float32(1e-6), ctx)
    # RoPE interleaved on q,k. cos/sin already [B*L*H, Dh/2] (rows (b,l,h)).
    var q_rope = rope_interleaved(q_n, cos_exp, sin_exp, ctx)  # [B*L*H, Dh]
    var k_rope = rope_interleaved(k_n, cos_exp, sin_exp, ctx)
    # SDPA needs [B, S, H, Dh] (BSHD). q_rope rows are (b,l,h) -> already
    # [B, L, H, Dh] in row-major. v is [B*L, 1, attn_dim] = [B,L,H,Dh] bytes.
    var q_bshd = reshape(q_rope, [Bp, Lp, Hp, Dhp], ctx)
    var k_bshd = reshape(k_rope, [Bp, Lp, Hp, Dhp], ctx)
    var v_bshd = reshape(v, [Bp, Lp, Hp, Dhp], ctx)
    var scale = Float32(1.0) / sqrt(Float32(Dhp))
    var attn = sdpa_nomask[Bp, Lp, Hp, Dhp](q_bshd, k_bshd, v_bshd, scale, ctx)
    # attn [B, L, H, Dh] -> reshape [B, L, attn_dim] (transpose(1,2).reshape in
    # the ref undoes the BHSD; our BSHD attn output is already [B,L,H,Dh] so a
    # plain reshape to [B,L,attn_dim] is correct).
    var attn_flat = reshape(attn, [Bp * Lp, attn_dim], ctx)
    var out = linear(attn_flat, w.proj_w, Optional[Tensor](_clone(w.proj_b, ctx)), ctx)
    return reshape(out, [Bp, Lp, attn_dim], ctx)


# ════════════════════════════════════════════════════════════════════════════
# PiTBlock forward.
# ════════════════════════════════════════════════════════════════════════════
def pit_block_forward[
    Bp: Int, Lp: Int, Hp: Int, Dhp: Int
](
    x: Tensor,            # [BL, P2, pixel_dim]
    s_cond: Tensor,       # [BL, context_dim]
    w: PiTBlockWeights,
    pixel_dim: Int,
    context_dim: Int,
    attn_dim: Int,
    P2: Int,
    img_h: Int,
    img_w: Int,
    patch_size: Int,
    rope_ref: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """PiTBlock.forward (no-CP, mask=None). BL = B*L, L = (img_h/ps)*(img_w/ps).
    Compile-time params Bp/Lp/Hp/Dhp pin the SDPA shapes (B, L, num_heads,
    head_dim). All F32."""
    var sh = x.shape()
    if len(sh) != 3:
        raise Error("pit_block: x must be [BL, P2, C]")
    var BL = sh[0]
    var C = sh[2]
    if C != pixel_dim:
        raise Error("pit_block: pixel_dim mismatch")
    var Hs = img_h // patch_size
    var Ws = img_w // patch_size
    var head_dim = Dhp
    var half = head_dim // 2

    # ── adaLN: cond = adaLN(s_cond) -> [BL, 6*C*P2], view [BL, P2, 6C] ────────
    var cond = linear(s_cond, w.adaln_w, Optional[Tensor](_clone(w.adaln_b, ctx)), ctx)
    var cond_v = reshape(cond, [BL, P2, 6 * pixel_dim], ctx)  # [BL,P2,6C]
    # chunk 6 along last dim -> each [BL, P2, C]
    var shift_msa = slice(cond_v, 2, 0 * pixel_dim, pixel_dim, ctx)
    var scale_msa = slice(cond_v, 2, 1 * pixel_dim, pixel_dim, ctx)
    var gate_msa = slice(cond_v, 2, 2 * pixel_dim, pixel_dim, ctx)
    var shift_mlp = slice(cond_v, 2, 3 * pixel_dim, pixel_dim, ctx)
    var scale_mlp = slice(cond_v, 2, 4 * pixel_dim, pixel_dim, ctx)
    var gate_mlp = slice(cond_v, 2, 5 * pixel_dim, pixel_dim, ctx)

    # ── x_norm = RMSNorm1(x) * (1+scale_msa) + shift_msa  (per-token) ─────────
    var n1 = rms_norm(x, w.norm1_w, Float32(1e-6), ctx)        # [BL,P2,C]
    var one_plus_sc1 = add_scalar(scale_msa, Float32(1.0), ctx)
    var x_norm = add(mul(n1, one_plus_sc1, ctx), shift_msa, ctx)  # [BL,P2,C]

    # ── compress -> [B, L, attn_dim] ─────────────────────────────────────────
    var x_flat = reshape(x_norm, [BL, P2 * pixel_dim], ctx)
    var x_comp_flat = linear(
        x_flat, w.compress_w, Optional[Tensor](_clone(w.compress_b, ctx)), ctx
    )  # [BL, attn_dim]
    var x_comp = reshape(x_comp_flat, [Bp, Lp, attn_dim], ctx)

    # ── RoPE tables (NTK 2D) at grid (Hs, Ws), expand over B,H ───────────────
    var tables = ntk_rope_tables_2d(head_dim, Hs, Ws, rope_ref, rope_ref, ctx)
    var cos_exp = _expand_rope_table(tables.cos, Bp, Lp, Hp, half, ctx)
    var sin_exp = _expand_rope_table(tables.sin, Bp, Lp, Hp, half, ctx)

    # ── attention ────────────────────────────────────────────────────────────
    var attn_out = _rotary_attention[Bp, Lp, Hp, Dhp](x_comp, w, cos_exp, sin_exp, ctx)
    # ── expand -> [BL, P2, C] ────────────────────────────────────────────────
    var attn_2d = reshape(attn_out, [Bp * Lp, attn_dim], ctx)
    var attn_flat = linear(
        attn_2d, w.expand_w, Optional[Tensor](_clone(w.expand_b, ctx)), ctx
    )  # [B*L, P2*C]
    var attn_exp = reshape(attn_flat, [BL, P2, pixel_dim], ctx)

    # ── x = x + gate_msa * attn_exp ──────────────────────────────────────────
    var x1 = add(x, mul(gate_msa, attn_exp, ctx), ctx)         # [BL,P2,C]

    # ── MLP(RMSNorm2(x)*(1+scale_mlp)+shift_mlp), exact GELU ─────────────────
    var n2 = rms_norm(x1, w.norm2_w, Float32(1e-6), ctx)
    var one_plus_sc2 = add_scalar(scale_mlp, Float32(1.0), ctx)
    var mlp_in = add(mul(n2, one_plus_sc2, ctx), shift_mlp, ctx)  # [BL,P2,C]
    var h1 = linear(mlp_in, w.fc1_w, Optional[Tensor](_clone(w.fc1_b, ctx)), ctx)
    var hg = gelu_erf(h1, ctx)
    var mlp_out = linear(hg, w.fc2_w, Optional[Tensor](_clone(w.fc2_b, ctx)), ctx)

    # ── x = x + gate_mlp * mlp_out ───────────────────────────────────────────
    var y = add(x1, mul(gate_mlp, mlp_out, ctx), ctx)          # [BL,P2,C]
    return y^
