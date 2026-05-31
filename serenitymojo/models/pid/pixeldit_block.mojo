# models/pid/pixeldit_block.mojo — PiD PixelDiT MMDiT joint-attention block.
#
# Pure-Mojo + MAX 26.3, NVIDIA GPU. Port of MMDiTBlockT2I + MMDiTJointAttention
# from the PiD repo (pixeldit_official.py:517-682). The patch-stream backbone
# stacks `patch_depth` (=14) of these dual-stream blocks.
#
# Architecture (verbatim from the repo; B=1 fixed here — `c` is [B,1,C] so the
# AdaLN modulation is per-channel and broadcasts over the whole sequence):
#
#   per-stream AdaLN (image=x, text=y), each a single Linear(C -> 6C) on `c`:
#     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk6
#   1) joint attention:
#       x_n = (1+scale_msa_x)*RMSNorm(x) + shift_msa_x          (apply_adaln)
#       y_n likewise
#       attn_x, attn_y = MMDiTJointAttention(x_n, y_n, pos_img, pos_txt)
#       x = x + gate_msa_x * attn_x ;  y = y + gate_msa_y * attn_y
#   2) per-stream SwiGLU FFN:
#       x = x + gate_mlp_x * mlp_x((1+scale_mlp_x)*RMSNorm(x)+shift_mlp_x)
#       y likewise
#
# MMDiTJointAttention (the key PiD difference vs sd3_mmdit — SEPARATE qkv_x /
# qkv_y projections, joint [text,image] SDPA, per-stream output proj):
#   qkv_x = Linear(x_n, qkv_x_w)  -> [1,Nx,3C]; split q,k,v [1,Nx,C]
#   reshape each to [1,N,H,hd]; per-head RMSNorm (q_norm_x / k_norm_x) over hd
#   image RoPE (interleaved complex view_as_complex on (q[2i],q[2i+1])) on qx,kx
#   text RoPE on qy,ky iff pos_txt given (this block-gate uses pos_txt=None)
#   joint seq = cat([text, image]) along the token axis -> SDPA (no mask)
#   split back; merge heads; out_x = proj_x(out_x), out_y = proj_y(out_y)
#
# SwiGLU FeedForward (no bias): w2( silu(w1(x)) * w3(x) ).
#
# All compute F32 (the block-parity gate vs the PyTorch F32 oracle expects
# cos>=0.999). Reuses ops/{linear,norm,rope,attention,activations,elementwise}
# and the NTK RoPE table from models/pid/pid_ops.mojo.
#
# Mojo 1.0.0b1.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_interleaved
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.tensor_algebra import reshape, slice, concat


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime F32 = STDtype.F32


# ════════════════════════════════════════════════════════════════════════════
# Weight container for one MMDiTBlockT2I.
# ════════════════════════════════════════════════════════════════════════════
struct StreamPair(Movable):
    """A pair of (image, text) stream tensors. Move-only return container since
    Tensor is not Copyable (so List[Tensor] is unavailable)."""

    var x: Tensor
    var y: Tensor

    def __init__(out self, var x: Tensor, var y: Tensor):
        self.x = x^
        self.y = y^


struct MMDiTBlockWeights(Movable):
    """All learned tensors for one PiD PixelDiT MMDiTBlockT2I (F32, GPU-resident).

    Shapes (C = hidden_size, hd = C/groups, FF = SwiGLU inner dim):
      adaln_img_w [6C, C]  adaln_img_b [6C]     (same for txt)
      norm_{x1,y1,x2,y2}_w [C]
      qkv_{x,y}_w [3C, C]  (no bias)
      {q,k}_norm_{x,y}_w [hd]
      proj_{x,y}_w [C, C]  proj_{x,y}_b [C]
      mlp_{x,y}_{w1,w3} [FF, C]  mlp_{x,y}_w2 [C, FF]  (no bias)
    """

    var adaln_img_w: Tensor
    var adaln_img_b: Tensor
    var adaln_txt_w: Tensor
    var adaln_txt_b: Tensor
    var norm_x1_w: Tensor
    var norm_y1_w: Tensor
    var norm_x2_w: Tensor
    var norm_y2_w: Tensor
    var qkv_x_w: Tensor
    var qkv_y_w: Tensor
    var q_norm_x_w: Tensor
    var k_norm_x_w: Tensor
    var q_norm_y_w: Tensor
    var k_norm_y_w: Tensor
    var proj_x_w: Tensor
    var proj_x_b: Tensor
    var proj_y_w: Tensor
    var proj_y_b: Tensor
    var mlp_x_w1: Tensor
    var mlp_x_w3: Tensor
    var mlp_x_w2: Tensor
    var mlp_y_w1: Tensor
    var mlp_y_w3: Tensor
    var mlp_y_w2: Tensor

    def __init__(
        out self,
        var adaln_img_w: Tensor, var adaln_img_b: Tensor,
        var adaln_txt_w: Tensor, var adaln_txt_b: Tensor,
        var norm_x1_w: Tensor, var norm_y1_w: Tensor,
        var norm_x2_w: Tensor, var norm_y2_w: Tensor,
        var qkv_x_w: Tensor, var qkv_y_w: Tensor,
        var q_norm_x_w: Tensor, var k_norm_x_w: Tensor,
        var q_norm_y_w: Tensor, var k_norm_y_w: Tensor,
        var proj_x_w: Tensor, var proj_x_b: Tensor,
        var proj_y_w: Tensor, var proj_y_b: Tensor,
        var mlp_x_w1: Tensor, var mlp_x_w3: Tensor, var mlp_x_w2: Tensor,
        var mlp_y_w1: Tensor, var mlp_y_w3: Tensor, var mlp_y_w2: Tensor,
    ):
        self.adaln_img_w = adaln_img_w^
        self.adaln_img_b = adaln_img_b^
        self.adaln_txt_w = adaln_txt_w^
        self.adaln_txt_b = adaln_txt_b^
        self.norm_x1_w = norm_x1_w^
        self.norm_y1_w = norm_y1_w^
        self.norm_x2_w = norm_x2_w^
        self.norm_y2_w = norm_y2_w^
        self.qkv_x_w = qkv_x_w^
        self.qkv_y_w = qkv_y_w^
        self.q_norm_x_w = q_norm_x_w^
        self.k_norm_x_w = k_norm_x_w^
        self.q_norm_y_w = q_norm_y_w^
        self.k_norm_y_w = k_norm_y_w^
        self.proj_x_w = proj_x_w^
        self.proj_x_b = proj_x_b^
        self.proj_y_w = proj_y_w^
        self.proj_y_b = proj_y_b^
        self.mlp_x_w1 = mlp_x_w1^
        self.mlp_x_w3 = mlp_x_w3^
        self.mlp_x_w2 = mlp_x_w2^
        self.mlp_y_w1 = mlp_y_w1^
        self.mlp_y_w3 = mlp_y_w3^
        self.mlp_y_w2 = mlp_y_w2^


# ════════════════════════════════════════════════════════════════════════════
# Helpers.
# ════════════════════════════════════════════════════════════════════════════
def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Device-to-device byte copy (bias needs an owned copy for Optional[Tensor])."""
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _broadcast_rope_kernel(
    cos: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [N*half]
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],     # [N*H*half]
    N: Int, H: Int, half: Int,
):
    # Tile the per-token RoPE table [N, half] across H heads, producing a
    # row-major [N, H, half] table whose flattened rows align with q/k laid out
    # as [N*H, head_dim] (token-major, then head, then pair).
    var idx = Int(global_idx.x)
    var total = N * H * half
    if idx < total:
        var j = idx % half
        var nh = idx // half
        var n = nh // H
        o[idx] = cos[n * half + j]


def _broadcast_rope_to_heads(
    tbl: Tensor, N: Int, H: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    """Tile a [N, half] RoPE table to [N*H, half] (one copy per head) so it lines
    up with q/k reshaped to rows [N*H, head_dim]."""
    var n = N * H * half
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    var rl_in = RuntimeLayout[_DYN1].row_major(IndexList[1](N * half))
    var rl_out = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var C = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        tbl.buf.unsafe_ptr().bitcast[Float32](), rl_in
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl_out
    )
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_broadcast_rope_kernel, _broadcast_rope_kernel](
        C, O, N, H, half, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, [N * H, half], F32)


def _chunk6_vec(c_params: Tensor, C: Int, idx: Int, ctx: DeviceContext) raises -> Tensor:
    """Extract chunk `idx` (in [0,6)) of a [1,1,6C] AdaLN parameter tensor as a
    flat [C] per-channel vector (torch.chunk(6, dim=-1)[idx])."""
    var flat = reshape(c_params, [6 * C], ctx)
    var seg = slice(flat, 0, idx * C, C, ctx)   # [C]
    return seg^


# ════════════════════════════════════════════════════════════════════════════
# Joint attention: separate qkv_x / qkv_y, per-head QK-RMSNorm, image RoPE,
# joint [text, image] SDPA, per-stream output proj. F32, B=1.
# ════════════════════════════════════════════════════════════════════════════
def _qk_norm_per_head(
    t: Tensor, weight: Tensor, N: Int, H: Int, hd: Int, ctx: DeviceContext
) raises -> Tensor:
    """RMSNorm over head_dim for q/k laid out [1,N,C]. Views as [N*H, hd] (the
    per-head channel groups are contiguous since C = H*hd row-major), runs
    rms_norm (normalizes last dim), returns [1,N,C]."""
    var rows = reshape(t, [N * H, hd], ctx)
    var normed = rms_norm(rows, weight, Float32(1e-6), ctx)
    return reshape(normed, [1, N, H * hd], ctx)


def _joint_attention[
    H: Int, hd: Int, Nx: Int, Ny: Int
](
    x_n: Tensor, y_n: Tensor,            # [1,Nx,C], [1,Ny,C]
    w: MMDiTBlockWeights,
    rope_cos_h: Tensor, rope_sin_h: Tensor,  # [Nx*H, hd/2] broadcast image RoPE
    C: Int,
    ctx: DeviceContext,
) raises -> StreamPair:
    """Returns StreamPair(out_x [1,Nx,C], out_y [1,Ny,C])."""
    # ── QKV projections (no bias) ─────────────────────────────────────────────
    var qkv_x = linear(x_n, w.qkv_x_w, None, ctx)   # [1,Nx,3C]
    var qkv_y = linear(y_n, w.qkv_y_w, None, ctx)   # [1,Ny,3C]
    var qx = slice(qkv_x, 2, 0, C, ctx)             # [1,Nx,C]
    var kx = slice(qkv_x, 2, C, C, ctx)
    var vx = slice(qkv_x, 2, 2 * C, C, ctx)
    var qy = slice(qkv_y, 2, 0, C, ctx)             # [1,Ny,C]
    var ky = slice(qkv_y, 2, C, C, ctx)
    var vy = slice(qkv_y, 2, 2 * C, C, ctx)

    # ── per-head QK RMSNorm (over head_dim) ───────────────────────────────────
    var qxn = _qk_norm_per_head(qx, w.q_norm_x_w, Nx, H, hd, ctx)
    var kxn = _qk_norm_per_head(kx, w.k_norm_x_w, Nx, H, hd, ctx)
    var qyn = _qk_norm_per_head(qy, w.q_norm_y_w, Ny, H, hd, ctx)
    var kyn = _qk_norm_per_head(ky, w.k_norm_y_w, Ny, H, hd, ctx)

    # ── image RoPE (interleaved complex). Rows = [Nx*H, hd]; cos/sin broadcast
    #    over heads to [Nx*H, hd/2]. apply_rotary_emb pairs (q[2i],q[2i+1]). ───
    var qx_rows = reshape(qxn, [Nx * H, hd], ctx)
    var kx_rows = reshape(kxn, [Nx * H, hd], ctx)
    var qx_rot = rope_interleaved(qx_rows, rope_cos_h, rope_sin_h, ctx)  # [Nx*H, hd]
    var kx_rot = rope_interleaved(kx_rows, rope_cos_h, rope_sin_h, ctx)
    # text RoPE: pos_txt=None in this block path → qy/ky pass through unrotated.

    # ── assemble joint [text, image] sequence in BSHD = [1, S, H, hd] ─────────
    # Each stream reshaped to [1, N, H, hd]; concat along the token axis (dim=1).
    var qx_bshd = reshape(qx_rot, [1, Nx, H, hd], ctx)
    var kx_bshd = reshape(kx_rot, [1, Nx, H, hd], ctx)
    var vx_bshd = reshape(vx, [1, Nx, H, hd], ctx)
    var qy_bshd = reshape(qyn, [1, Ny, H, hd], ctx)
    var ky_bshd = reshape(kyn, [1, Ny, H, hd], ctx)
    var vy_bshd = reshape(vy, [1, Ny, H, hd], ctx)

    var q_joint = concat(1, ctx, qy_bshd, qx_bshd)   # [1, Ny+Nx, H, hd]
    var k_joint = concat(1, ctx, ky_bshd, kx_bshd)
    var v_joint = concat(1, ctx, vy_bshd, vx_bshd)

    comptime S = Ny + Nx
    var scale = Float32(1.0) / Float32(hd) ** Float32(0.5)
    var out_joint = sdpa_nomask[1, S, H, hd](q_joint, k_joint, v_joint, scale, ctx)  # [1,S,H,hd]

    # ── split back to [text, image]; merge heads; per-stream output proj ──────
    var out_y_bshd = slice(out_joint, 1, 0, Ny, ctx)    # [1,Ny,H,hd]
    var out_x_bshd = slice(out_joint, 1, Ny, Nx, ctx)   # [1,Nx,H,hd]
    var out_y_flat = reshape(out_y_bshd, [1, Ny, C], ctx)
    var out_x_flat = reshape(out_x_bshd, [1, Nx, C], ctx)
    var proj_x = linear(out_x_flat, w.proj_x_w, Optional[Tensor](_clone(w.proj_x_b, ctx)), ctx)
    var proj_y = linear(out_y_flat, w.proj_y_w, Optional[Tensor](_clone(w.proj_y_b, ctx)), ctx)
    return StreamPair(proj_x^, proj_y^)


# ════════════════════════════════════════════════════════════════════════════
# Joint attention WITH text RoPE (use_text_rope=True path). Identical to
# _joint_attention but also rotates qy/ky with the broadcast text RoPE tables.
# ════════════════════════════════════════════════════════════════════════════
def _joint_attention_textrope[
    H: Int, hd: Int, Nx: Int, Ny: Int
](
    x_n: Tensor, y_n: Tensor,            # [1,Nx,C], [1,Ny,C]
    w: MMDiTBlockWeights,
    rope_cos_h: Tensor, rope_sin_h: Tensor,    # [Nx*H, hd/2] image RoPE
    trope_cos_h: Tensor, trope_sin_h: Tensor,  # [Ny*H, hd/2] text RoPE
    C: Int,
    ctx: DeviceContext,
) raises -> StreamPair:
    var qkv_x = linear(x_n, w.qkv_x_w, None, ctx)
    var qkv_y = linear(y_n, w.qkv_y_w, None, ctx)
    var qx = slice(qkv_x, 2, 0, C, ctx)
    var kx = slice(qkv_x, 2, C, C, ctx)
    var vx = slice(qkv_x, 2, 2 * C, C, ctx)
    var qy = slice(qkv_y, 2, 0, C, ctx)
    var ky = slice(qkv_y, 2, C, C, ctx)
    var vy = slice(qkv_y, 2, 2 * C, C, ctx)

    var qxn = _qk_norm_per_head(qx, w.q_norm_x_w, Nx, H, hd, ctx)
    var kxn = _qk_norm_per_head(kx, w.k_norm_x_w, Nx, H, hd, ctx)
    var qyn = _qk_norm_per_head(qy, w.q_norm_y_w, Ny, H, hd, ctx)
    var kyn = _qk_norm_per_head(ky, w.k_norm_y_w, Ny, H, hd, ctx)

    # image RoPE
    var qx_rows = reshape(qxn, [Nx * H, hd], ctx)
    var kx_rows = reshape(kxn, [Nx * H, hd], ctx)
    var qx_rot = rope_interleaved(qx_rows, rope_cos_h, rope_sin_h, ctx)
    var kx_rot = rope_interleaved(kx_rows, rope_cos_h, rope_sin_h, ctx)
    # text RoPE (1D)
    var qy_rows = reshape(qyn, [Ny * H, hd], ctx)
    var ky_rows = reshape(kyn, [Ny * H, hd], ctx)
    var qy_rot = rope_interleaved(qy_rows, trope_cos_h, trope_sin_h, ctx)
    var ky_rot = rope_interleaved(ky_rows, trope_cos_h, trope_sin_h, ctx)

    var qx_bshd = reshape(qx_rot, [1, Nx, H, hd], ctx)
    var kx_bshd = reshape(kx_rot, [1, Nx, H, hd], ctx)
    var vx_bshd = reshape(vx, [1, Nx, H, hd], ctx)
    var qy_bshd = reshape(qy_rot, [1, Ny, H, hd], ctx)
    var ky_bshd = reshape(ky_rot, [1, Ny, H, hd], ctx)
    var vy_bshd = reshape(vy, [1, Ny, H, hd], ctx)

    var q_joint = concat(1, ctx, qy_bshd, qx_bshd)
    var k_joint = concat(1, ctx, ky_bshd, kx_bshd)
    var v_joint = concat(1, ctx, vy_bshd, vx_bshd)

    comptime S = Ny + Nx
    var scale = Float32(1.0) / Float32(hd) ** Float32(0.5)
    var out_joint = sdpa_nomask[1, S, H, hd](q_joint, k_joint, v_joint, scale, ctx)

    var out_y_bshd = slice(out_joint, 1, 0, Ny, ctx)
    var out_x_bshd = slice(out_joint, 1, Ny, Nx, ctx)
    var out_y_flat = reshape(out_y_bshd, [1, Ny, C], ctx)
    var out_x_flat = reshape(out_x_bshd, [1, Nx, C], ctx)
    var proj_x = linear(out_x_flat, w.proj_x_w, Optional[Tensor](_clone(w.proj_x_b, ctx)), ctx)
    var proj_y = linear(out_y_flat, w.proj_y_w, Optional[Tensor](_clone(w.proj_y_b, ctx)), ctx)
    return StreamPair(proj_x^, proj_y^)


def _swiglu_ffn(
    x: Tensor, w1: Tensor, w3: Tensor, w2: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """FeedForward: w2( silu(w1(x)) * w3(x) ). All Linears bias-free."""
    var g = linear(x, w1, None, ctx)
    var u = linear(x, w3, None, ctx)
    var act = swiglu(g, u, ctx)
    return linear(act, w2, None, ctx)


# ════════════════════════════════════════════════════════════════════════════
# Full MMDiTBlockT2I.forward (B=1, pos_txt=None).
# ════════════════════════════════════════════════════════════════════════════
def mmdit_block_forward[
    H: Int, hd: Int, Nx: Int, Ny: Int
](
    x: Tensor,           # [1, Nx, C] image stream
    y: Tensor,           # [1, Ny, C] text stream
    c: Tensor,           # [1, 1, C]  conditioning (broadcast per-channel)
    rope_cos: Tensor,    # [Nx, hd/2] image NTK RoPE cos (real of freqs_cis)
    rope_sin: Tensor,    # [Nx, hd/2] image NTK RoPE sin (imag of freqs_cis)
    w: MMDiTBlockWeights,
    C: Int,
    ctx: DeviceContext,
) raises -> StreamPair:
    """One PiD PixelDiT MMDiTBlockT2I forward. Returns StreamPair(x_out [1,Nx,C],
    y_out [1,Ny,C]). H = groups (compile-time), hd = C/H, Nx/Ny token counts."""
    var half = hd // 2

    # ── per-stream AdaLN (Linear C -> 6C on c), chunk into 6 per-channel [C] ──
    var ci = linear(c, w.adaln_img_w, Optional[Tensor](_clone(w.adaln_img_b, ctx)), ctx)  # [1,1,6C]
    var ct = linear(c, w.adaln_txt_w, Optional[Tensor](_clone(w.adaln_txt_b, ctx)), ctx)  # [1,1,6C]
    var shift_msa_x = _chunk6_vec(ci, C, 0, ctx)
    var scale_msa_x = _chunk6_vec(ci, C, 1, ctx)
    var gate_msa_x = _chunk6_vec(ci, C, 2, ctx)
    var shift_mlp_x = _chunk6_vec(ci, C, 3, ctx)
    var scale_mlp_x = _chunk6_vec(ci, C, 4, ctx)
    var gate_mlp_x = _chunk6_vec(ci, C, 5, ctx)
    var shift_msa_y = _chunk6_vec(ct, C, 0, ctx)
    var scale_msa_y = _chunk6_vec(ct, C, 1, ctx)
    var gate_msa_y = _chunk6_vec(ct, C, 2, ctx)
    var shift_mlp_y = _chunk6_vec(ct, C, 3, ctx)
    var scale_mlp_y = _chunk6_vec(ct, C, 4, ctx)
    var gate_mlp_y = _chunk6_vec(ct, C, 5, ctx)

    # ── broadcast image RoPE tables across heads: [Nx,half] -> [Nx*H,half] ────
    var rope_cos_h = _broadcast_rope_to_heads(rope_cos, Nx, H, half, ctx)
    var rope_sin_h = _broadcast_rope_to_heads(rope_sin, Nx, H, half, ctx)

    # ── 1) joint attention with dual-stream AdaLN-modulated inputs ────────────
    var x_norm = modulate(rms_norm(x, w.norm_x1_w, Float32(1e-6), ctx), scale_msa_x, shift_msa_x, ctx)
    var y_norm = modulate(rms_norm(y, w.norm_y1_w, Float32(1e-6), ctx), scale_msa_y, shift_msa_y, ctx)
    var attn = _joint_attention[H, hd, Nx, Ny](
        x_norm, y_norm, w, rope_cos_h, rope_sin_h, C, ctx
    )
    var x1 = residual_gate(x, gate_msa_x, attn.x, ctx)
    var y1 = residual_gate(y, gate_msa_y, attn.y, ctx)

    # ── 2) per-stream SwiGLU FFN with AdaLN ───────────────────────────────────
    var x_mlp_in = modulate(rms_norm(x1, w.norm_x2_w, Float32(1e-6), ctx), scale_mlp_x, shift_mlp_x, ctx)
    var y_mlp_in = modulate(rms_norm(y1, w.norm_y2_w, Float32(1e-6), ctx), scale_mlp_y, shift_mlp_y, ctx)
    var x_mlp = _swiglu_ffn(x_mlp_in, w.mlp_x_w1, w.mlp_x_w3, w.mlp_x_w2, ctx)
    var y_mlp = _swiglu_ffn(y_mlp_in, w.mlp_y_w1, w.mlp_y_w3, w.mlp_y_w2, ctx)
    var x_out = residual_gate(x1, gate_mlp_x, x_mlp, ctx)
    var y_out = residual_gate(y1, gate_mlp_y, y_mlp, ctx)
    return StreamPair(x_out^, y_out^)


# ════════════════════════════════════════════════════════════════════════════
# MMDiTBlockT2I.forward WITH text RoPE (use_text_rope=True; the released PiD
# config). Takes precomputed image (rope_*) and text (trope_*) [N, hd/2] tables.
# ════════════════════════════════════════════════════════════════════════════
def mmdit_block_forward_textrope[
    H: Int, hd: Int, Nx: Int, Ny: Int
](
    x: Tensor, y: Tensor, c: Tensor,
    rope_cos: Tensor, rope_sin: Tensor,    # [Nx, hd/2] image RoPE
    trope_cos: Tensor, trope_sin: Tensor,  # [Ny, hd/2] text RoPE
    w: MMDiTBlockWeights,
    C: Int,
    ctx: DeviceContext,
) raises -> StreamPair:
    var half = hd // 2
    var ci = linear(c, w.adaln_img_w, Optional[Tensor](_clone(w.adaln_img_b, ctx)), ctx)
    var ct = linear(c, w.adaln_txt_w, Optional[Tensor](_clone(w.adaln_txt_b, ctx)), ctx)
    var shift_msa_x = _chunk6_vec(ci, C, 0, ctx)
    var scale_msa_x = _chunk6_vec(ci, C, 1, ctx)
    var gate_msa_x = _chunk6_vec(ci, C, 2, ctx)
    var shift_mlp_x = _chunk6_vec(ci, C, 3, ctx)
    var scale_mlp_x = _chunk6_vec(ci, C, 4, ctx)
    var gate_mlp_x = _chunk6_vec(ci, C, 5, ctx)
    var shift_msa_y = _chunk6_vec(ct, C, 0, ctx)
    var scale_msa_y = _chunk6_vec(ct, C, 1, ctx)
    var gate_msa_y = _chunk6_vec(ct, C, 2, ctx)
    var shift_mlp_y = _chunk6_vec(ct, C, 3, ctx)
    var scale_mlp_y = _chunk6_vec(ct, C, 4, ctx)
    var gate_mlp_y = _chunk6_vec(ct, C, 5, ctx)

    var rope_cos_h = _broadcast_rope_to_heads(rope_cos, Nx, H, half, ctx)
    var rope_sin_h = _broadcast_rope_to_heads(rope_sin, Nx, H, half, ctx)
    var trope_cos_h = _broadcast_rope_to_heads(trope_cos, Ny, H, half, ctx)
    var trope_sin_h = _broadcast_rope_to_heads(trope_sin, Ny, H, half, ctx)

    var x_norm = modulate(rms_norm(x, w.norm_x1_w, Float32(1e-6), ctx), scale_msa_x, shift_msa_x, ctx)
    var y_norm = modulate(rms_norm(y, w.norm_y1_w, Float32(1e-6), ctx), scale_msa_y, shift_msa_y, ctx)
    var attn = _joint_attention_textrope[H, hd, Nx, Ny](
        x_norm, y_norm, w, rope_cos_h, rope_sin_h, trope_cos_h, trope_sin_h, C, ctx
    )
    var x1 = residual_gate(x, gate_msa_x, attn.x, ctx)
    var y1 = residual_gate(y, gate_msa_y, attn.y, ctx)

    var x_mlp_in = modulate(rms_norm(x1, w.norm_x2_w, Float32(1e-6), ctx), scale_mlp_x, shift_mlp_x, ctx)
    var y_mlp_in = modulate(rms_norm(y1, w.norm_y2_w, Float32(1e-6), ctx), scale_mlp_y, shift_mlp_y, ctx)
    var x_mlp = _swiglu_ffn(x_mlp_in, w.mlp_x_w1, w.mlp_x_w3, w.mlp_x_w2, ctx)
    var y_mlp = _swiglu_ffn(y_mlp_in, w.mlp_y_w1, w.mlp_y_w3, w.mlp_y_w2, ctx)
    var x_out = residual_gate(x1, gate_mlp_x, x_mlp, ctx)
    var y_out = residual_gate(y1, gate_mlp_y, y_mlp, ctx)
    return StreamPair(x_out^, y_out^)
