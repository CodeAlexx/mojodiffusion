# seedvr2_dit.mojo — SeedVR2-3B NaDiT (video diffusion transformer) — pure Mojo + MAX.
#
# INFERENCE ONLY. GPU, BF16 (the DiT runs bf16; linear F32-accumulates internally).
# No Rust / no autograd / no Python-in-runtime.
#
# Chunk 1 — the two input embeddings:
#   * vid_in — NaPatchIn  (SeedVR models/dit_v2/patch/patch_v1.py)
#   * emb_in — TimeEmbedding (SeedVR models/dit_v2/embedding.py)
#
# Each gates vs a real torch bf16 oracle at cos >= 0.999 (see tests/).

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from math import exp, log, sin, cos, sqrt
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear_bias, linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.tensor_algebra import reshape, permute, slice, concat, add, mul, mul_scalar, full_device, gather_rows
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.softmax import softmax_lastdim
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ═══════════════════════════════════════════════════════════════════════════════
# (A) vid_in — NaPatchIn: patchify (1,2,2) + Linear proj  [.,132] -> [.,2560]
# ═══════════════════════════════════════════════════════════════════════════════
def vid_in(
    vid_flat: Tensor,   # [L, C] = [T*H*W, 33]  (bf16), grid channels-last
    proj_w: Tensor,     # vid_in.proj.weight  [2560, 132]
    proj_b: Tensor,     # vid_in.proj.bias    [2560]
    T: Int,             # grid T (=4)   — pre-patchify
    H: Int,             # grid H (=16)
    W: Int,             # grid W (=16)
    ctx: DeviceContext,
) raises -> Tensor:
    """SeedVR NaPatchIn: rearrange "(T t)(H h)(W w) c -> T H W (t h w c)" with
    patch (t,h,w)=(1,2,2), then flatten grid tokens and apply Linear(vid_in.proj).

    Input  vid_flat : [T*H*W, C]                 (grid flattened, channels-last)
    Output          : [T*(H/2)*(W/2), 2560]      = [256, 2560]

    Patchify fold: t=1, so each output token gathers a 2x2 (h,w) spatial block
    and concatenates the 4 pixels' C channels in (h w c) order:
        out_channel = ((h_sub*2 + w_sub) * C) + c        (h_sub slow, w_sub mid, c fast)

    Implemented as reshape -> permute -> reshape (no explicit gather):
      1. The contiguous [T*H*W, C] buffer is the grid [T, H, W, C] row-major.
         Reshape it to [T, H/2, 2, W/2, 2, C] = axes (T, Ho, h_sub, Wo, w_sub, C)
         — a pure reinterpret: hfull = Ho*2 + h_sub, wfull = Wo*2 + w_sub, so
         row-major position is bit-identical to the [T,H,W,C] layout.
      2. permute [0,1,3,2,4,5] -> (T, Ho, Wo, h_sub, w_sub, C): moves the two
         sub-axes to the end while keeping the (h_sub, w_sub, C) fold order.
      3. Reshape to [T*Ho*Wo, 2*2*C] = [256, 132]; the trailing 3 dims flatten
         row-major to ((h_sub*2 + w_sub)*C + c) — exactly the einops fold.
    Then Linear: y = x @ proj_wᵀ + proj_b.
    """
    var xs = vid_flat.shape()
    if len(xs) != 2:
        raise Error("vid_in: expected vid_flat rank-2 [L, C]")
    var C = xs[1]
    if xs[0] != T * H * W:
        raise Error("vid_in: L != T*H*W")
    if H % 2 != 0 or W % 2 != 0:
        raise Error("vid_in: H and W must be even for patch (1,2,2)")
    var Ho = H // 2
    var Wo = W // 2

    # 1. reinterpret [T*H*W, C] -> [T, Ho, 2, Wo, 2, C]
    var g6 = List[Int]()
    g6.append(T)
    g6.append(Ho)
    g6.append(2)
    g6.append(Wo)
    g6.append(2)
    g6.append(C)
    var grid = reshape(vid_flat, g6^, ctx)

    # 2. permute (T, Ho, h, Wo, w, C) -> (T, Ho, Wo, h, w, C)
    var perm = List[Int]()
    perm.append(0)
    perm.append(1)
    perm.append(3)
    perm.append(2)
    perm.append(4)
    perm.append(5)
    var permd = permute(grid, perm, ctx)

    # 3. flatten to tokens [T*Ho*Wo, (2*2*C)]
    var t2 = List[Int]()
    t2.append(T * Ho * Wo)
    t2.append(2 * 2 * C)
    var tokens = reshape(permd, t2^, ctx)

    # Linear proj -> [256, 2560]
    return linear_bias(tokens, proj_w, proj_b, ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# (B) emb_in — TimeEmbedding: sinusoidal(256) -> MLP proj_in/hid/out (SiLU) -> [1,15360]
# ═══════════════════════════════════════════════════════════════════════════════
def _sinusoidal_timestep(
    timestep: Float32, dim: Int, ctx: DeviceContext
) raises -> Tensor:
    """diffusers get_timestep_embedding(t, dim, flip_sin_to_cos=False,
    downscale_freq_shift=0, scale=1, max_period=10000).

        half   = dim // 2                      (= 128)
        freq_i = exp(-log(10000) * i / half)   for i in [0, half)
        ang_i  = t * freq_i
        emb    = concat([sin(ang), cos(ang)])  (SIN FIRST — flip_sin_to_cos=False)

    Computed on host (F32) then uploaded as a BF16 [1, dim] tensor so the
    downstream Linear runs bf16 (matches the torch bf16 reference)."""
    var half = dim // 2
    var vals = List[Float32]()
    var ln = Float32(log(Float64(10000.0)))
    # sin(ang) block
    for i in range(half):
        var freq = Float32(exp(Float64(-ln * (Float32(i) / Float32(half)))))
        vals.append(Float32(sin(Float64(timestep * freq))))
    # cos(ang) block
    for i in range(half):
        var freq = Float32(exp(Float64(-ln * (Float32(i) / Float32(half)))))
        vals.append(Float32(cos(Float64(timestep * freq))))
    var shp = List[Int]()
    shp.append(1)
    shp.append(dim)
    return Tensor.from_host(vals, shp^, STDtype.BF16, ctx)


def emb_in(
    timestep: Float32,   # scalar (= 500.0)
    pin_w: Tensor,       # emb_in.proj_in.weight  [2560, 256]
    pin_b: Tensor,       # emb_in.proj_in.bias    [2560]
    phid_w: Tensor,      # emb_in.proj_hid.weight [2560, 2560]
    phid_b: Tensor,      # emb_in.proj_hid.bias   [2560]
    pout_w: Tensor,      # emb_in.proj_out.weight [15360, 2560]
    pout_b: Tensor,      # emb_in.proj_out.bias   [15360]
    ctx: DeviceContext,
) raises -> Tensor:
    """SeedVR TimeEmbedding: sinusoidal(256) -> proj_in -> SiLU -> proj_hid ->
    SiLU -> proj_out. Output [1, 15360] (bf16)."""
    var sinu = _sinusoidal_timestep(timestep, 256, ctx)     # [1, 256] bf16
    var h0 = linear_bias(sinu, pin_w, pin_b, ctx)           # [1, 2560]
    var a0 = silu(h0, ctx)
    var h1 = linear_bias(a0, phid_w, phid_b, ctx)           # [1, 2560]
    var a1 = silu(h1, ctx)
    return linear_bias(a1, pout_w, pout_b, ctx)             # [1, 15360]


# ═══════════════════════════════════════════════════════════════════════════════
# (C) attn_qkv_norm — NaMMAttention entry (per stream): QKV proj + QK-RMSNorm.
#   SeedVR models/dit_v2/nablocks/attention/mmattn.py:76-108, one stream (vid|txt).
# ═══════════════════════════════════════════════════════════════════════════════
def attn_qkv_norm(
    x: Tensor,            # [L, 2560]  stream hidden (bf16)
    proj_qkv_w: Tensor,   # proj_qkv.{vid|txt}.weight  [7680, 2560]  (NO bias)
    norm_q_w: Tensor,     # norm_q.{vid|txt}.weight     [128]
    norm_k_w: Tensor,     # norm_k.{vid|txt}.weight     [128]
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor, Tensor]:
    """SeedVR NaMMAttention QKV+QK-norm for ONE stream (vid or txt).

        qkv = linear(x, proj_qkv_w)                 # NO bias -> [L, 7680]
        qkv = reshape "l (o h d) -> l o h d"        # o=3 (outermost), h=20, d=128
        q,k,v = qkv[:,0], qkv[:,1], qkv[:,2]        # each [L, 20, 128]
        q = rms_norm(q, norm_q_w, eps=1e-5)         # over last dim (128), per (L,head)
        k = rms_norm(k, norm_k_w, eps=1e-5)

    Packed layout is (o h d) row-major with o OUTERMOST, so column index
    = o*(20*128) + h*128 + d = o*2560 + h*128 + d.  Extracting stream `part`
    is therefore a contiguous column slice [part*2560, part*2560+2560), which
    reshapes [L,2560] -> [L,20,128] as exactly (h,d).

    Returns (q_norm, k_norm, v), each [L, 20, 128] (bf16). v is NOT normed
    (the later full-attention chunk consumes it as-is)."""
    var L = x.shape()[0]
    var HEADS = 20
    var HD = 128
    var INNER = HEADS * HD  # 2560

    # qkv projection (no bias): [L,2560] @ [7680,2560]ᵀ -> [L,7680]
    var qkv = linear(x, proj_qkv_w, None, ctx)

    # q = qkv[:,0] -> [L,20,128]
    var q_cols = slice(qkv, 1, 0 * INNER, INNER, ctx)   # [L,2560]
    var q_hs = List[Int]()
    q_hs.append(L)
    q_hs.append(HEADS)
    q_hs.append(HD)
    var q = reshape(q_cols, q_hs^, ctx)                 # [L,20,128]

    # k = qkv[:,1] -> [L,20,128]
    var k_cols = slice(qkv, 1, 1 * INNER, INNER, ctx)
    var k_hs = List[Int]()
    k_hs.append(L)
    k_hs.append(HEADS)
    k_hs.append(HD)
    var k = reshape(k_cols, k_hs^, ctx)

    # v = qkv[:,2] -> [L,20,128]  (not normed)
    var v_cols = slice(qkv, 1, 2 * INNER, INNER, ctx)
    var v_hs = List[Int]()
    v_hs.append(L)
    v_hs.append(HEADS)
    v_hs.append(HD)
    var v = reshape(v_cols, v_hs^, ctx)

    # RMSNorm over the last dim (128), affine, eps=1e-5, per (token,head) row.
    var q_norm = rms_norm(q, norm_q_w, Float32(1e-5), ctx)
    var k_norm = rms_norm(k, norm_k_w, Float32(1e-5), ctx)

    return (q_norm^, k_norm^, v^)


# ═══════════════════════════════════════════════════════════════════════════════
# (D) apply_rope3d — apply 3D rotary embedding (rotate-half) to normed q/k.
#   Convention = rotary_embedding_torch apply_rotary_emb(freqs, t), with
#   rotate_half over ADJACENT pairs ((d r) r=2). freqs are precomputed ANGLES
#   [L, FREQW] (F32); the FIRST FREQW dims are rotated, trailing dims pass through.
#
#   For adjacent pair i (dims 2i, 2i+1), f0=freqs[.,2i], f1=freqs[.,2i+1]:
#       out[2i]   = x[2i]  * cos(f0) - x[2i+1] * sin(f0)
#       out[2i+1] = x[2i+1]* cos(f1) + x[2i]   * sin(f1)
#   Head axis broadcasts (same freqs for every head at a given token).
#   cos/sin computed in F32; store bf16.
# ═══════════════════════════════════════════════════════════════════════════════
def _rope3d_kernel(
    q: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [L*H*D] flat
    fr: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],   # [L*FREQW] flat
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],   # [L*H*D] flat
    total: Int,      # L*H*(D//2) threads (one per adjacent pair)
    H: Int,          # head count
    D: Int,          # head_dim (=128)
    FREQW: Int,      # rotary width (=126)
):
    var i = Int(global_idx.x)
    if i < total:
        var pairs = D // 2
        var pair = i % pairs
        var row = i // pairs            # row = token*H + head
        var token = row // H
        var base = row * D + pair * 2   # element index of dim 2*pair
        var x0 = rebind[Scalar[DType.bfloat16]](q[base]).cast[DType.float32]()
        var x1 = rebind[Scalar[DType.bfloat16]](q[base + 1]).cast[DType.float32]()
        if pair * 2 < FREQW:
            var fbase = token * FREQW + pair * 2
            var f0 = rebind[Scalar[DType.float32]](fr[fbase])
            var f1 = rebind[Scalar[DType.float32]](fr[fbase + 1])
            var o0 = x0 * cos(f0) - x1 * sin(f0)
            var o1 = x1 * cos(f1) + x0 * sin(f1)
            o[base] = rebind[o.element_type](o0.cast[DType.bfloat16]())
            o[base + 1] = rebind[o.element_type](o1.cast[DType.bfloat16]())
        else:
            # trailing dims (e.g. 126,127) pass through unchanged
            o[base] = rebind[o.element_type](x0.cast[DType.bfloat16]())
            o[base + 1] = rebind[o.element_type](x1.cast[DType.bfloat16]())


def apply_rope3d(q: Tensor, freqs: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Apply 3D rotary embedding to q [L,H,D] using precomputed freqs [L,FREQW]
    (F32 angles). Rotates the first FREQW dims (adjacent-pair rotate_half);
    dims [FREQW..D) pass through. Returns [L,H,D] bf16."""
    var qs = q.shape()
    if len(qs) != 3:
        raise Error("apply_rope3d: expected q rank-3 [L,H,D]")
    var L = qs[0]
    var H = qs[1]
    var D = qs[2]
    if D % 2 != 0:
        raise Error("apply_rope3d: head_dim must be even")
    var fs = freqs.shape()
    if len(fs) != 2 or fs[0] != L:
        raise Error("apply_rope3d: freqs must be [L, FREQW] matching q token axis")
    var FREQW = fs[1]
    if FREQW % 2 != 0 or FREQW > D:
        raise Error("apply_rope3d: FREQW must be even and <= head_dim")
    if q.dtype() != STDtype.BF16:
        raise Error("apply_rope3d: q must be bf16")
    if freqs.dtype() != STDtype.F32:
        raise Error("apply_rope3d: freqs must be F32")

    var pairs = D // 2
    var total = L * H * pairs
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](q.nbytes())
    var q_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](q.numel()))
    var f_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](freqs.numel()))
    var Q = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        q.buf.unsafe_ptr().bitcast[BFloat16](), q_rl
    )
    var F = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        freqs.buf.unsafe_ptr().bitcast[Float32](), f_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), q_rl
    )
    var grid = (total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_rope3d_kernel, _rope3d_kernel](
        Q, F, O, total, H, D, FREQW, grid_dim=grid, block_dim=_BLOCK
    )
    return Tensor(out_buf^, q.shape(), q.dtype())


# ═══════════════════════════════════════════════════════════════════════════════
# (E) window_attn_projout — NaSwinAttention window-joint attention + proj_out.
#   SeedVR models/dit_v2/nablocks/attention/mmattn.py:238-266 (forward tail).
#
#   CONCRETE block-0 window structure (hardcoded): 4 temporal windows, each =
#   one frame's 64 vid tokens (8x8) + all 58 txt tokens => S=122 per window.
#   window w gathers vid[w*64:(w+1)*64] and txt[w*58:(w+1)*58].
#
#   Per window w: attn_w = softmax(q_w @ k_wᵀ / sqrt(128)) @ v_w over the joint
#   [vid_win(64) ; txt(58)] sequence, batched B=4,S=122,H=20,Dh=128.
#   Split each window output -> vid part (first 64) + txt part (last 58):
#     * vid_out = concat of the 4 windows' vid parts -> [256,20,128]
#                 (window_reverse is IDENTITY for block-0 — tokens already
#                  frame-ordered) -> [256,2560].
#     * txt_out = the 4 windows' txt parts [4,58,20,128] MEAN-POOLED over the
#                 window axis -> [58,20,128] -> [58,2560]  (na "text pooling").
#   Then per-stream proj_out (Linear 2560->2560 + bias, separate vid/txt weights).
#   gather_heads_scatter_seq is IDENTITY (single-GPU) — skipped.
# ═══════════════════════════════════════════════════════════════════════════════
def window_attn_projout(
    vid_q: Tensor,          # [256,20,128] bf16  (post-rope vid query)
    vid_k: Tensor,          # [256,20,128] bf16  (post-rope vid key)
    vid_v: Tensor,          # [256,20,128] bf16  (un-normed vid value)
    txt_q: Tensor,          # [232,20,128] bf16  (post-rope txt query, 4 windows x 58, window-major)
    txt_k: Tensor,          # [232,20,128] bf16  (post-rope txt key,   window-major)
    txt_v: Tensor,          # [232,20,128] bf16  (un-normed txt value, tiled 4x, window-major)
    projout_vid_w: Tensor,  # blocks.0.attn.proj_out.vid.weight [2560,2560]
    projout_vid_b: Tensor,  # blocks.0.attn.proj_out.vid.bias   [2560]
    projout_txt_w: Tensor,  # blocks.0.attn.proj_out.txt.weight [2560,2560]
    projout_txt_b: Tensor,  # blocks.0.attn.proj_out.txt.bias   [2560]
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    """Block-0 NaSwinAttention window-joint attention + proj_out.

    Returns (vid_out [256,2560], txt_out [58,2560]) bf16."""
    comptime NW = 4        # windows
    comptime VW = 64       # vid tokens per window
    comptime TW = 58       # txt tokens per window
    comptime SW = VW + TW  # 122 joint tokens per window
    comptime HEADS = 20
    comptime HD = 128
    comptime INNER = HEADS * HD  # 2560

    # Reshape each stream to [NW, tokens, H, Dh] (window-major reinterpret) then
    # concat along the token axis -> per-window joint sequences [4,122,20,128].
    var vq4 = reshape(vid_q, [NW, VW, HEADS, HD], ctx)
    var tq4 = reshape(txt_q, [NW, TW, HEADS, HD], ctx)
    var qwin = concat(1, ctx, vq4, tq4)                      # [4,122,20,128]

    var vk4 = reshape(vid_k, [NW, VW, HEADS, HD], ctx)
    var tk4 = reshape(txt_k, [NW, TW, HEADS, HD], ctx)
    var kwin = concat(1, ctx, vk4, tk4)

    var vv4 = reshape(vid_v, [NW, VW, HEADS, HD], ctx)
    var tv4 = reshape(txt_v, [NW, TW, HEADS, HD], ctx)
    var vwin = concat(1, ctx, vv4, tv4)

    # Batched joint attention: B=4, S=122, H=20, Dh=128, scale=1/sqrt(128).
    var scale = Float32(1.0) / Float32(sqrt(Float64(HD)))
    var att = sdpa_nomask[NW, SW, HEADS, HD](qwin, kwin, vwin, scale, ctx)  # [4,122,20,128]

    # vid part = att[:, 0:64] -> [4,64,20,128] -> [256,2560]. window_reverse is
    # IDENTITY for block-0, so the row-major flatten already yields frame order.
    var vid_part = slice(att, 1, 0, VW, ctx)                 # [4,64,20,128]
    var vid_2d = reshape(vid_part, [NW * VW, INNER], ctx)    # [256,2560]

    # txt part = att[:, 64:122] -> [4,58,20,128], mean over the window axis
    # (window-major: axis 0 = window copy) -> [58,20,128].
    var txt_part = slice(att, 1, VW, TW, ctx)                # [4,58,20,128]
    var acc = reshape(slice(txt_part, 0, 0, 1, ctx), [TW, HEADS, HD], ctx)
    for w in range(1, NW):
        var wslice = reshape(slice(txt_part, 0, w, 1, ctx), [TW, HEADS, HD], ctx)
        acc = add(acc, wslice, ctx)
    var txt_mean = mul_scalar(acc, Float32(1.0) / Float32(NW), ctx)  # [58,20,128]
    var txt_2d = reshape(txt_mean, [TW, INNER], ctx)                 # [58,2560]

    # Per-stream proj_out (Linear 2560->2560 + bias).
    var vid_out = linear_bias(vid_2d, projout_vid_w, projout_vid_b, ctx)  # [256,2560]
    var txt_out = linear_bias(txt_2d, projout_txt_w, projout_txt_b, ctx)  # [58,2560]
    return (vid_out^, txt_out^)


# ═══════════════════════════════════════════════════════════════════════════════
# (E') window_attn_projout_g[TW] — TW-generalized window-joint attention + proj_out.
#   Identical to window_attn_projout but the per-window txt count is the comptime
#   param TW (58 for the positive CFG stream, 64 for the negative). Windows=4,
#   each packs 64 vid + TW txt = (64+TW) joint tokens; sdpa_nomask[4,64+TW,20,128];
#   txt mean-pool over the 4 windows -> [TW,2560]. window_attn_projout (TW=58)
#   is left byte-identical so the verified winattn gate is untouched.
# ═══════════════════════════════════════════════════════════════════════════════
def window_attn_projout_g[TW: Int](
    vid_q: Tensor,          # [256,20,128] bf16
    vid_k: Tensor,          # [256,20,128] bf16
    vid_v: Tensor,          # [256,20,128] bf16
    txt_q: Tensor,          # [TW*4,20,128] bf16 (window-major)
    txt_k: Tensor,          # [TW*4,20,128] bf16
    txt_v: Tensor,          # [TW*4,20,128] bf16
    projout_vid_w: Tensor,  # [2560,2560]
    projout_vid_b: Tensor,  # [2560]
    projout_txt_w: Tensor,  # [2560,2560]
    projout_txt_b: Tensor,  # [2560]
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    """Window-joint attention + proj_out with comptime per-window txt count TW.
    Returns (vid_out [256,2560], txt_out [TW,2560]) bf16."""
    comptime NW = 4        # windows
    comptime VW = 64       # vid tokens per window
    comptime SW = VW + TW  # joint tokens per window (122 for TW=58, 128 for TW=64)
    comptime HEADS = 20
    comptime HD = 128
    comptime INNER = HEADS * HD  # 2560

    var vq4 = reshape(vid_q, [NW, VW, HEADS, HD], ctx)
    var tq4 = reshape(txt_q, [NW, TW, HEADS, HD], ctx)
    var qwin = concat(1, ctx, vq4, tq4)                      # [4,SW,20,128]

    var vk4 = reshape(vid_k, [NW, VW, HEADS, HD], ctx)
    var tk4 = reshape(txt_k, [NW, TW, HEADS, HD], ctx)
    var kwin = concat(1, ctx, vk4, tk4)

    var vv4 = reshape(vid_v, [NW, VW, HEADS, HD], ctx)
    var tv4 = reshape(txt_v, [NW, TW, HEADS, HD], ctx)
    var vwin = concat(1, ctx, vv4, tv4)

    var scale = Float32(1.0) / Float32(sqrt(Float64(HD)))
    var att = sdpa_nomask[NW, SW, HEADS, HD](qwin, kwin, vwin, scale, ctx)  # [4,SW,20,128]

    # vid part = att[:, 0:64] -> [4,64,20,128] -> [256,2560] (window_reverse identity).
    var vid_part = slice(att, 1, 0, VW, ctx)                 # [4,64,20,128]
    var vid_2d = reshape(vid_part, [NW * VW, INNER], ctx)    # [256,2560]

    # txt part = att[:, 64:64+TW] -> [4,TW,20,128], mean over the window axis.
    var txt_part = slice(att, 1, VW, TW, ctx)                # [4,TW,20,128]
    var acc = reshape(slice(txt_part, 0, 0, 1, ctx), [TW, HEADS, HD], ctx)
    for w in range(1, NW):
        var wslice = reshape(slice(txt_part, 0, w, 1, ctx), [TW, HEADS, HD], ctx)
        acc = add(acc, wslice, ctx)
    var txt_mean = mul_scalar(acc, Float32(1.0) / Float32(NW), ctx)  # [TW,20,128]
    var txt_2d = reshape(txt_mean, [TW, INNER], ctx)                 # [TW,2560]

    var vid_out = linear_bias(vid_2d, projout_vid_w, projout_vid_b, ctx)  # [256,2560]
    var txt_out = linear_bias(txt_2d, projout_txt_w, projout_txt_b, ctx)  # [TW,2560]
    return (vid_out^, txt_out^)


# ═══════════════════════════════════════════════════════════════════════════════
# (F) Full MMDiT block — SeedVR NaMMSRTransformerBlock.forward (dual-stream,
#     pre-norm DiT), composing the 3 verified attention sub-pieces (C,D,E) plus a
#     SwiGLU MLP and AdaSingle modulation.  models/dit_v2/nablocks/mmsr_block.py.
#
#   AdaSingle: emb [1,15360] reshaped row-major to [2560,6]; column index d*6+col.
#   Columns (the "A" terms, shared by vid & txt, broadcast over the L token axis):
#     0=attn_shift 1=attn_scale 2=attn_gate 3=mlp_shift 4=mlp_scale 5=mlp_gate.
#   Per stream the learnable "B" terms ada.{vid,txt}.{attn,mlp}_{scale,shift,gate}
#   [2560] add to the A terms. Modulation multiplies/adds per-CHANNEL (2560),
#   broadcast over L:  x*(scaleA+scaleB) + (shiftA+shiftB), gate: x*(gateA+gateB).
#   RMSNorm is NO-affine (eps=1e-5 over 2560): implemented via a ones[2560] weight.
# ═══════════════════════════════════════════════════════════════════════════════
struct Block0Weights(Movable):
    """All learnable tensors for one SeedVR MMDiT block (block 0), bf16."""

    var proj_qkv_vid_w: Tensor
    var proj_qkv_txt_w: Tensor
    var norm_q_vid_w: Tensor
    var norm_q_txt_w: Tensor
    var norm_k_vid_w: Tensor
    var norm_k_txt_w: Tensor
    var proj_out_vid_w: Tensor
    var proj_out_vid_b: Tensor
    var proj_out_txt_w: Tensor
    var proj_out_txt_b: Tensor
    # ada vid
    var ada_vid_attn_scale: Tensor
    var ada_vid_attn_shift: Tensor
    var ada_vid_attn_gate: Tensor
    var ada_vid_mlp_scale: Tensor
    var ada_vid_mlp_shift: Tensor
    var ada_vid_mlp_gate: Tensor
    # ada txt
    var ada_txt_attn_scale: Tensor
    var ada_txt_attn_shift: Tensor
    var ada_txt_attn_gate: Tensor
    var ada_txt_mlp_scale: Tensor
    var ada_txt_mlp_shift: Tensor
    var ada_txt_mlp_gate: Tensor
    # mlp
    var mlp_vid_proj_in_w: Tensor
    var mlp_vid_proj_in_gate_w: Tensor
    var mlp_vid_proj_out_w: Tensor
    var mlp_txt_proj_in_w: Tensor
    var mlp_txt_proj_in_gate_w: Tensor
    var mlp_txt_proj_out_w: Tensor

    def __init__(
        out self,
        var proj_qkv_vid_w: Tensor,
        var proj_qkv_txt_w: Tensor,
        var norm_q_vid_w: Tensor,
        var norm_q_txt_w: Tensor,
        var norm_k_vid_w: Tensor,
        var norm_k_txt_w: Tensor,
        var proj_out_vid_w: Tensor,
        var proj_out_vid_b: Tensor,
        var proj_out_txt_w: Tensor,
        var proj_out_txt_b: Tensor,
        var ada_vid_attn_scale: Tensor,
        var ada_vid_attn_shift: Tensor,
        var ada_vid_attn_gate: Tensor,
        var ada_vid_mlp_scale: Tensor,
        var ada_vid_mlp_shift: Tensor,
        var ada_vid_mlp_gate: Tensor,
        var ada_txt_attn_scale: Tensor,
        var ada_txt_attn_shift: Tensor,
        var ada_txt_attn_gate: Tensor,
        var ada_txt_mlp_scale: Tensor,
        var ada_txt_mlp_shift: Tensor,
        var ada_txt_mlp_gate: Tensor,
        var mlp_vid_proj_in_w: Tensor,
        var mlp_vid_proj_in_gate_w: Tensor,
        var mlp_vid_proj_out_w: Tensor,
        var mlp_txt_proj_in_w: Tensor,
        var mlp_txt_proj_in_gate_w: Tensor,
        var mlp_txt_proj_out_w: Tensor,
    ):
        self.proj_qkv_vid_w = proj_qkv_vid_w^
        self.proj_qkv_txt_w = proj_qkv_txt_w^
        self.norm_q_vid_w = norm_q_vid_w^
        self.norm_q_txt_w = norm_q_txt_w^
        self.norm_k_vid_w = norm_k_vid_w^
        self.norm_k_txt_w = norm_k_txt_w^
        self.proj_out_vid_w = proj_out_vid_w^
        self.proj_out_vid_b = proj_out_vid_b^
        self.proj_out_txt_w = proj_out_txt_w^
        self.proj_out_txt_b = proj_out_txt_b^
        self.ada_vid_attn_scale = ada_vid_attn_scale^
        self.ada_vid_attn_shift = ada_vid_attn_shift^
        self.ada_vid_attn_gate = ada_vid_attn_gate^
        self.ada_vid_mlp_scale = ada_vid_mlp_scale^
        self.ada_vid_mlp_shift = ada_vid_mlp_shift^
        self.ada_vid_mlp_gate = ada_vid_mlp_gate^
        self.ada_txt_attn_scale = ada_txt_attn_scale^
        self.ada_txt_attn_shift = ada_txt_attn_shift^
        self.ada_txt_attn_gate = ada_txt_attn_gate^
        self.ada_txt_mlp_scale = ada_txt_mlp_scale^
        self.ada_txt_mlp_shift = ada_txt_mlp_shift^
        self.ada_txt_mlp_gate = ada_txt_mlp_gate^
        self.mlp_vid_proj_in_w = mlp_vid_proj_in_w^
        self.mlp_vid_proj_in_gate_w = mlp_vid_proj_in_gate_w^
        self.mlp_vid_proj_out_w = mlp_vid_proj_out_w^
        self.mlp_txt_proj_in_w = mlp_txt_proj_in_w^
        self.mlp_txt_proj_in_gate_w = mlp_txt_proj_in_gate_w^
        self.mlp_txt_proj_out_w = mlp_txt_proj_out_w^


def _load_block_separate(st: SafeTensors, P: String, ctx: DeviceContext) raises -> Block0Weights:
    """Blocks 0-9: SEPARATE `.vid`/`.txt` weights (28 tensors)."""
    return Block0Weights(
        load_bias(st, P + "attn.proj_qkv.vid.weight", ctx),
        load_bias(st, P + "attn.proj_qkv.txt.weight", ctx),
        load_bias(st, P + "attn.norm_q.vid.weight", ctx),
        load_bias(st, P + "attn.norm_q.txt.weight", ctx),
        load_bias(st, P + "attn.norm_k.vid.weight", ctx),
        load_bias(st, P + "attn.norm_k.txt.weight", ctx),
        load_bias(st, P + "attn.proj_out.vid.weight", ctx),
        load_bias(st, P + "attn.proj_out.vid.bias", ctx),
        load_bias(st, P + "attn.proj_out.txt.weight", ctx),
        load_bias(st, P + "attn.proj_out.txt.bias", ctx),
        load_bias(st, P + "ada.vid.attn_scale", ctx),
        load_bias(st, P + "ada.vid.attn_shift", ctx),
        load_bias(st, P + "ada.vid.attn_gate", ctx),
        load_bias(st, P + "ada.vid.mlp_scale", ctx),
        load_bias(st, P + "ada.vid.mlp_shift", ctx),
        load_bias(st, P + "ada.vid.mlp_gate", ctx),
        load_bias(st, P + "ada.txt.attn_scale", ctx),
        load_bias(st, P + "ada.txt.attn_shift", ctx),
        load_bias(st, P + "ada.txt.attn_gate", ctx),
        load_bias(st, P + "ada.txt.mlp_scale", ctx),
        load_bias(st, P + "ada.txt.mlp_shift", ctx),
        load_bias(st, P + "ada.txt.mlp_gate", ctx),
        load_bias(st, P + "mlp.vid.proj_in.weight", ctx),
        load_bias(st, P + "mlp.vid.proj_in_gate.weight", ctx),
        load_bias(st, P + "mlp.vid.proj_out.weight", ctx),
        load_bias(st, P + "mlp.txt.proj_in.weight", ctx),
        load_bias(st, P + "mlp.txt.proj_in_gate.weight", ctx),
        load_bias(st, P + "mlp.txt.proj_out.weight", ctx),
    )


def _load_block_shared(st: SafeTensors, P: String, ctx: DeviceContext) raises -> Block0Weights:
    """Blocks 10-31: SHARED `.all` weights (14 tensors). vid AND txt use the SAME
    `.all` tensor; since Block0Weights holds separate vid/txt fields (each consumed
    by the constructor), clone each `.all` tensor once for the txt field. mmdit_block
    then runs unchanged — both streams read identical weights."""
    var qkv = load_bias(st, P + "attn.proj_qkv.all.weight", ctx)
    var nq = load_bias(st, P + "attn.norm_q.all.weight", ctx)
    var nk = load_bias(st, P + "attn.norm_k.all.weight", ctx)
    var pow = load_bias(st, P + "attn.proj_out.all.weight", ctx)
    var pob = load_bias(st, P + "attn.proj_out.all.bias", ctx)
    var a_as = load_bias(st, P + "ada.all.attn_scale", ctx)
    var a_ash = load_bias(st, P + "ada.all.attn_shift", ctx)
    var a_ag = load_bias(st, P + "ada.all.attn_gate", ctx)
    var a_ms = load_bias(st, P + "ada.all.mlp_scale", ctx)
    var a_msh = load_bias(st, P + "ada.all.mlp_shift", ctx)
    var a_mg = load_bias(st, P + "ada.all.mlp_gate", ctx)
    var m_in = load_bias(st, P + "mlp.all.proj_in.weight", ctx)
    var m_ing = load_bias(st, P + "mlp.all.proj_in_gate.weight", ctx)
    var m_out = load_bias(st, P + "mlp.all.proj_out.weight", ctx)
    # Args evaluate left-to-right: each `.clone(ctx)` (vid field) is read BEFORE
    # the corresponding `^` move (txt field), so ordering is safe.
    return Block0Weights(
        qkv.clone(ctx), qkv^,                              # proj_qkv vid, txt
        nq.clone(ctx), nq^,                                # norm_q vid, txt
        nk.clone(ctx), nk^,                                # norm_k vid, txt
        pow.clone(ctx), pob.clone(ctx),                    # proj_out vid weight, bias
        pow^, pob^,                                        # proj_out txt weight, bias
        a_as.clone(ctx), a_ash.clone(ctx), a_ag.clone(ctx),  # ada vid attn s/sh/g
        a_ms.clone(ctx), a_msh.clone(ctx), a_mg.clone(ctx),  # ada vid mlp s/sh/g
        a_as^, a_ash^, a_ag^,                              # ada txt attn s/sh/g
        a_ms^, a_msh^, a_mg^,                              # ada txt mlp s/sh/g
        m_in.clone(ctx), m_ing.clone(ctx), m_out.clone(ctx),  # mlp vid in/gate/out
        m_in^, m_ing^, m_out^,                             # mlp txt in/gate/out
    )


def load_block(st: SafeTensors, i: Int, ctx: DeviceContext) raises -> Block0Weights:
    """Generalized block loader. i<10 -> separate `.vid`/`.txt`; i>=10 -> shared
    `.all` (cloned into both vid/txt fields so mmdit_block is unchanged)."""
    var P = String("blocks.") + String(i) + String(".")
    if i < 10:
        return _load_block_separate(st, P, ctx)
    return _load_block_shared(st, P, ctx)


def load_block0(st: SafeTensors, ctx: DeviceContext) raises -> Block0Weights:
    """Stream all `blocks.0.` weights (bf16) into a Block0Weights."""
    return load_block(st, 0, ctx)


def dit_stack(
    vid: Tensor,          # [256,2560] bf16  (block-0 vid input)
    txt: Tensor,          # [58,2560]  bf16  (block-0 txt input)
    emb: Tensor,          # [1,15360]  bf16  (AdaSingle emb, shared by all blocks)
    vid_freqs: Tensor,    # [256,126]  F32   (IDENTICAL rope angles for all blocks)
    txt_freqs: Tensor,    # [232,126]  F32   (IDENTICAL rope angles for all blocks)
    blocks: List[ArcPointer[Block0Weights]],
    ctx: DeviceContext,
) raises -> Tensor:
    """Run all 32 SeedVR2-3B MMDiT blocks in sequence, returning the final vid
    [256,2560]. The SAME emb + vid_freqs/txt_freqs feed every block (window
    structure is uniform for this input); mmdit_block/apply_rope3d BORROW these,
    so no per-block cloning of freqs is needed. vid/txt are re-bound each block
    via a cheap clone of the returned tuple element (256/58 x 2560 bf16). Blocks
    are held as ArcPointer (Block0Weights is Movable-not-Copyable, so a plain
    List of it is rejected); deref with `[]` to borrow into mmdit_block."""
    var cur_vid = vid.clone(ctx)
    var cur_txt = txt.clone(ctx)
    var n = len(blocks)
    for i in range(n):
        var pair = mmdit_block(cur_vid, cur_txt, emb, vid_freqs, txt_freqs, blocks[i][], ctx)
        cur_vid = pair[0].clone(ctx)
        cur_txt = pair[1].clone(ctx)
    return cur_vid^


def _emb_col(emb2d: Tensor, c: Int, ctx: DeviceContext) raises -> Tensor:
    """Column c of the [2560,6] AdaSingle emb -> [2560] (bf16)."""
    var col = slice(emb2d, 1, c, 1, ctx)   # [2560,1]
    return reshape(col, [2560], ctx)       # [2560]


def _mod_in(
    x: Tensor, scaleA: Tensor, scaleB: Tensor, shiftA: Tensor, shiftB: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """AdaSingle "in": x*(scaleA+scaleB) + (shiftA+shiftB), per-channel (2560),
    broadcast over the L token axis."""
    var scale = add(scaleA, scaleB, ctx)   # [2560]
    var shift = add(shiftA, shiftB, ctx)   # [2560]
    var xs = mul(x, scale, ctx)            # [L,2560] broadcast over L
    return add(xs, shift, ctx)


def _mod_gate(
    x: Tensor, gateA: Tensor, gateB: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """AdaSingle "out": x*(gateA+gateB), per-channel broadcast over L."""
    var gate = add(gateA, gateB, ctx)      # [2560]
    return mul(x, gate, ctx)


def _tile4(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Concat 4 identical copies of x along the token axis (58 -> 232)."""
    var a = x.clone(ctx)
    var b = x.clone(ctx)
    var c = x.clone(ctx)
    return concat(0, ctx, x, a, b, c)


def _swiglu_mlp(
    x: Tensor, proj_in_w: Tensor, proj_in_gate_w: Tensor, proj_out_w: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    """SeedVR SwiGLU MLP (NO bias): gate=x@proj_in_gateᵀ, up=x@proj_inᵀ,
    h=silu(gate)*up, out=h@proj_outᵀ."""
    var gate = linear(x, proj_in_gate_w, None, ctx)  # [L,6912]
    var up = linear(x, proj_in_w, None, ctx)         # [L,6912]
    var h = swiglu(gate, up, ctx)                    # silu(gate)*up [L,6912]
    return linear(h, proj_out_w, None, ctx)          # [L,2560]


def mmdit_block(
    vid: Tensor,          # [256,2560] bf16
    txt: Tensor,          # [58,2560]  bf16
    emb: Tensor,          # [1,15360]  bf16 (AdaSingle emb)
    vid_freqs: Tensor,    # [256,126]  F32 rope angles
    txt_freqs: Tensor,    # [232,126]  F32 rope angles (4 windows x 58)
    w: Block0Weights,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    """One SeedVR NaMMSRTransformerBlock forward (dual-stream vid+txt, pre-norm).
    Returns (vid_out [256,2560], txt_out [58,2560]) bf16."""
    # no-affine RMSNorm weight (ones over the 2560 channel dim)
    var ones = full_device([2560], Float32(1.0), STDtype.BF16, ctx)

    # AdaSingle A-terms (shared vid & txt): emb [1,15360] -> [2560,6], col d*6+c.
    var emb2d = reshape(emb, [2560, 6], ctx)
    var attn_shiftA = _emb_col(emb2d, 0, ctx)
    var attn_scaleA = _emb_col(emb2d, 1, ctx)
    var attn_gateA = _emb_col(emb2d, 2, ctx)
    var mlp_shiftA = _emb_col(emb2d, 3, ctx)
    var mlp_scaleA = _emb_col(emb2d, 4, ctx)
    var mlp_gateA = _emb_col(emb2d, 5, ctx)

    # ─── ATTN sub-block ────────────────────────────────────────────────────────
    var va = rms_norm(vid, ones, Float32(1e-5), ctx)   # no-affine RMSNorm
    var ta = rms_norm(txt, ones, Float32(1e-5), ctx)
    va = _mod_in(va, attn_scaleA, w.ada_vid_attn_scale, attn_shiftA, w.ada_vid_attn_shift, ctx)
    ta = _mod_in(ta, attn_scaleA, w.ada_txt_attn_scale, attn_shiftA, w.ada_txt_attn_shift, ctx)

    # attention (compose the 3 verified pieces)
    var vqkv = attn_qkv_norm(va, w.proj_qkv_vid_w, w.norm_q_vid_w, w.norm_k_vid_w, ctx)
    var tqkv = attn_qkv_norm(ta, w.proj_qkv_txt_w, w.norm_q_txt_w, w.norm_k_txt_w, ctx)
    var vq = apply_rope3d(vqkv[0], vid_freqs, ctx)     # [256,20,128]
    var vk = apply_rope3d(vqkv[1], vid_freqs, ctx)
    var tq4 = apply_rope3d(_tile4(tqkv[0], ctx), txt_freqs, ctx)  # [232,20,128]
    var tk4 = apply_rope3d(_tile4(tqkv[1], ctx), txt_freqs, ctx)
    var tv4 = _tile4(tqkv[2], ctx)                     # [232,20,128]
    var attn = window_attn_projout(
        vq, vk, vqkv[2], tq4, tk4, tv4,
        w.proj_out_vid_w, w.proj_out_vid_b, w.proj_out_txt_w, w.proj_out_txt_b, ctx,
    )
    var va2 = _mod_gate(attn[0], attn_gateA, w.ada_vid_attn_gate, ctx)  # [256,2560]
    var ta2 = _mod_gate(attn[1], attn_gateA, w.ada_txt_attn_gate, ctx)  # [58,2560]
    var vid_a = add(va2, vid, ctx)                     # residual
    var txt_a = add(ta2, txt, ctx)

    # ─── MLP sub-block ─────────────────────────────────────────────────────────
    var vm = rms_norm(vid_a, ones, Float32(1e-5), ctx)
    var tm = rms_norm(txt_a, ones, Float32(1e-5), ctx)
    vm = _mod_in(vm, mlp_scaleA, w.ada_vid_mlp_scale, mlp_shiftA, w.ada_vid_mlp_shift, ctx)
    tm = _mod_in(tm, mlp_scaleA, w.ada_txt_mlp_scale, mlp_shiftA, w.ada_txt_mlp_shift, ctx)
    vm = _swiglu_mlp(vm, w.mlp_vid_proj_in_w, w.mlp_vid_proj_in_gate_w, w.mlp_vid_proj_out_w, ctx)
    tm = _swiglu_mlp(tm, w.mlp_txt_proj_in_w, w.mlp_txt_proj_in_gate_w, w.mlp_txt_proj_out_w, ctx)
    vm = _mod_gate(vm, mlp_gateA, w.ada_vid_mlp_gate, ctx)
    tm = _mod_gate(tm, mlp_gateA, w.ada_txt_mlp_gate, ctx)
    var vid_o = add(vm, vid_a, ctx)                    # residual
    var txt_o = add(tm, txt_a, ctx)
    return (vid_o^, txt_o^)


# ═══════════════════════════════════════════════════════════════════════════════
# (F') mmdit_block_g[TW] / dit_stack_g[TW] — TW-generalized block + stack.
#   Identical to mmdit_block / dit_stack except the txt stream carries TW tokens
#   per window and the joint attention uses window_attn_projout_g[TW]. txt_freqs
#   must be [TW*4,126]. mmdit_block/dit_stack (TW=58) are left untouched so the
#   fixed-58 full_dit_forward gate is unaffected.
# ═══════════════════════════════════════════════════════════════════════════════
def mmdit_block_g[TW: Int](
    vid: Tensor,          # [256,2560] bf16
    txt: Tensor,          # [TW,2560]  bf16
    emb: Tensor,          # [1,15360]  bf16
    vid_freqs: Tensor,    # [256,126]  F32
    txt_freqs: Tensor,    # [TW*4,126] F32
    w: Block0Weights,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    """One SeedVR NaMMSRTransformerBlock forward with per-window txt count TW."""
    var ones = full_device([2560], Float32(1.0), STDtype.BF16, ctx)

    var emb2d = reshape(emb, [2560, 6], ctx)
    var attn_shiftA = _emb_col(emb2d, 0, ctx)
    var attn_scaleA = _emb_col(emb2d, 1, ctx)
    var attn_gateA = _emb_col(emb2d, 2, ctx)
    var mlp_shiftA = _emb_col(emb2d, 3, ctx)
    var mlp_scaleA = _emb_col(emb2d, 4, ctx)
    var mlp_gateA = _emb_col(emb2d, 5, ctx)

    # ─── ATTN sub-block ────────────────────────────────────────────────────────
    var va = rms_norm(vid, ones, Float32(1e-5), ctx)
    var ta = rms_norm(txt, ones, Float32(1e-5), ctx)
    va = _mod_in(va, attn_scaleA, w.ada_vid_attn_scale, attn_shiftA, w.ada_vid_attn_shift, ctx)
    ta = _mod_in(ta, attn_scaleA, w.ada_txt_attn_scale, attn_shiftA, w.ada_txt_attn_shift, ctx)

    var vqkv = attn_qkv_norm(va, w.proj_qkv_vid_w, w.norm_q_vid_w, w.norm_k_vid_w, ctx)
    var tqkv = attn_qkv_norm(ta, w.proj_qkv_txt_w, w.norm_q_txt_w, w.norm_k_txt_w, ctx)
    var vq = apply_rope3d(vqkv[0], vid_freqs, ctx)     # [256,20,128]
    var vk = apply_rope3d(vqkv[1], vid_freqs, ctx)
    var tq4 = apply_rope3d(_tile4(tqkv[0], ctx), txt_freqs, ctx)  # [TW*4,20,128]
    var tk4 = apply_rope3d(_tile4(tqkv[1], ctx), txt_freqs, ctx)
    var tv4 = _tile4(tqkv[2], ctx)                     # [TW*4,20,128]
    var attn = window_attn_projout_g[TW](
        vq, vk, vqkv[2], tq4, tk4, tv4,
        w.proj_out_vid_w, w.proj_out_vid_b, w.proj_out_txt_w, w.proj_out_txt_b, ctx,
    )
    var va2 = _mod_gate(attn[0], attn_gateA, w.ada_vid_attn_gate, ctx)  # [256,2560]
    var ta2 = _mod_gate(attn[1], attn_gateA, w.ada_txt_attn_gate, ctx)  # [TW,2560]
    var vid_a = add(va2, vid, ctx)                     # residual
    var txt_a = add(ta2, txt, ctx)

    # ─── MLP sub-block ─────────────────────────────────────────────────────────
    var vm = rms_norm(vid_a, ones, Float32(1e-5), ctx)
    var tm = rms_norm(txt_a, ones, Float32(1e-5), ctx)
    vm = _mod_in(vm, mlp_scaleA, w.ada_vid_mlp_scale, mlp_shiftA, w.ada_vid_mlp_shift, ctx)
    tm = _mod_in(tm, mlp_scaleA, w.ada_txt_mlp_scale, mlp_shiftA, w.ada_txt_mlp_shift, ctx)
    vm = _swiglu_mlp(vm, w.mlp_vid_proj_in_w, w.mlp_vid_proj_in_gate_w, w.mlp_vid_proj_out_w, ctx)
    tm = _swiglu_mlp(tm, w.mlp_txt_proj_in_w, w.mlp_txt_proj_in_gate_w, w.mlp_txt_proj_out_w, ctx)
    vm = _mod_gate(vm, mlp_gateA, w.ada_vid_mlp_gate, ctx)
    tm = _mod_gate(tm, mlp_gateA, w.ada_txt_mlp_gate, ctx)
    var vid_o = add(vm, vid_a, ctx)                    # residual
    var txt_o = add(tm, txt_a, ctx)
    return (vid_o^, txt_o^)


def dit_stack_g[TW: Int](
    vid: Tensor,          # [256,2560] bf16
    txt: Tensor,          # [TW,2560]  bf16
    emb: Tensor,          # [1,15360]  bf16
    vid_freqs: Tensor,    # [256,126]  F32
    txt_freqs: Tensor,    # [TW*4,126] F32
    blocks: List[ArcPointer[Block0Weights]],
    ctx: DeviceContext,
) raises -> Tensor:
    """Run all 32 MMDiT blocks with per-window txt count TW; returns final vid [256,2560]."""
    var cur_vid = vid.clone(ctx)
    var cur_txt = txt.clone(ctx)
    var n = len(blocks)
    for i in range(n):
        var pair = mmdit_block_g[TW](cur_vid, cur_txt, emb, vid_freqs, txt_freqs, blocks[i][], ctx)
        cur_vid = pair[0].clone(ctx)
        cur_txt = pair[1].clone(ctx)
    return cur_vid^


# ═══════════════════════════════════════════════════════════════════════════════
# (G) dit_out_tail — the NaDiT OUTPUT TAIL (SeedVR nadit.forward:232-236 + NaPatchOut).
#   vid_out_norm -> vid_out_ada -> vid_out.proj -> unpatchify -> latent [1024,16].
#
#   vid_out_norm : RMSNorm WITH affine weight [2560], eps=1e-5, over dim 2560.
#   vid_out_ada  : AdaSingle single-layer modulation. The shipped single-layer ada
#     is shape-inconsistent for this config, so the SELF-CONSISTENT block-style rule
#     is used (identical to the block AdaSingle already implemented, and the rule the
#     oracle was generated with): emb [1,15360] -> [2560,6], col 0=shift,1=scale,2=gate
#     of the ATTN slot; mode "in": vid*(scaleA+out_scale)+(shiftA+out_shift).
#   vid_out.proj : Linear 2560 -> 64 (+ bias).
#   unpatchify   : NaPatchOut, the EXACT inverse of vid_in's (1,2,2) fold.
#     Input [256,64] is the grid [T=4, Ho=8, Wo=8, 64]; the 64 chans = (t=1,h=2,w=2,c=16),
#     index = ((h_sub*2 + w_sub)*16 + c). rearrange "T H W (t h w c) -> (T t)(H h)(W w) c"
#     -> [4,16,16,16] -> [1024,16]. Mirrors vid_in inverted:
#       1. reshape [256,64] -> [4,8,8,2,2,16]   (T, Ho, Wo, h_sub, w_sub, c) — inverse
#          of vid_in step-3: the trailing 64 splits row-major into (h_sub, w_sub, c).
#       2. permute [0,1,3,2,4,5] -> [4,8,2,8,2,16] = (T, Ho, h_sub, Wo, w_sub, c). This
#          swap of positions 2<->3 is an involution — the SAME perm vid_in used to move
#          the sub-axes out; here it moves them back interleaved with Ho/Wo.
#       3. reshape -> [1024,16]: flattening (T,Ho,h_sub,Wo,w_sub) yields token index
#          T*256 + (Ho*2+h_sub)*16 + (Wo*2+w_sub) = T*Hfull*Wfull + hfull*Wfull + wfull,
#          i.e. the [(T t)(H h)(W w)] frame-major grid, EXACTLY inverting vid_in's fold.
# ═══════════════════════════════════════════════════════════════════════════════
def dit_out_tail(
    vid: Tensor,        # [256,2560] bf16  (block-31 vid output)
    emb: Tensor,        # [1,15360]  bf16  (AdaSingle emb)
    norm_w: Tensor,     # vid_out_norm.weight   [2560]
    ada_shift: Tensor,  # vid_out_ada.out_shift [2560]
    ada_scale: Tensor,  # vid_out_ada.out_scale [2560]
    proj_w: Tensor,     # vid_out.proj.weight   [64,2560]
    proj_b: Tensor,     # vid_out.proj.bias     [64]
    ctx: DeviceContext,
) raises -> Tensor:
    """SeedVR NaDiT output tail. Returns the final latent [1024,16] (bf16)."""
    # 1. RMSNorm WITH affine weight (eps=1e-5, over 2560).
    var normed = rms_norm(vid, norm_w, Float32(1e-5), ctx)          # [256,2560]

    # 2. AdaSingle "in": attn-slot shift(col0)/scale(col1); B terms = out_shift/out_scale.
    var emb2d = reshape(emb, [2560, 6], ctx)
    var shiftA = _emb_col(emb2d, 0, ctx)                            # [2560]
    var scaleA = _emb_col(emb2d, 1, ctx)                            # [2560]
    var moded = _mod_in(normed, scaleA, ada_scale, shiftA, ada_shift, ctx)  # [256,2560]

    # 3. proj 2560 -> 64 (+ bias).
    var proj = linear_bias(moded, proj_w, proj_b, ctx)             # [256,64]

    # 4. unpatchify [256,64] -> [1024,16] (inverse of vid_in's (1,2,2) fold).
    var g6 = reshape(proj, [4, 8, 8, 2, 2, 16], ctx)               # T,Ho,Wo,h,w,c
    var perm = List[Int]()
    perm.append(0)
    perm.append(1)
    perm.append(3)
    perm.append(2)
    perm.append(4)
    perm.append(5)
    var permd = permute(g6, perm, ctx)                            # T,Ho,h,Wo,w,c
    return reshape(permd, [1024, 16], ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# (H) compute_dit_freqs — axial-3D RoPE angle tables for the fixed gated grid.
#   Post-patch grid T=4, H=8, W=8 (256 vid tokens), 4 temporal windows, txt_len=58.
#   Computed on HOST in F32 (Float64 math for the inverse freqs), uploaded as F32
#   tensors (apply_rope3d consumes F32 freqs). Matches the model's freqs exactly
#   (validated vs freqs_ref: vid/txt max err ~2.7e-6 — F32 rounding only).
#
#   inv[i] = 1.0 / 10000^((2i)/42)         for i in 0..20        (21 inverse freqs)
#   ax1(pos) -> [42]: for i in 0..20: v = pos*inv[i]; out[2i]=v, out[2i+1]=v (repeat x2)
#   VID [256,126]: token = win*64 + h*8 + w (win 0..3, h 0..7, w 0..7)
#       [0:42]  = ax1(58)   (temporal pos = txt_len offset 58, const for all vid)
#       [42:84] = ax1(h)
#       [84:126]= ax1(w)
#   TXT [232,126]: token = win*58 + l (win 0..3, l 0..57)
#       a = ax1(l); [0:42]=a, [42:84]=a, [84:126]=a   (tile x3)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_dit_freqs(ctx: DeviceContext) raises -> Tuple[Tensor, Tensor]:
    """(vid_freqs [256,126], txt_freqs [232,126]) F32 axial-3D RoPE angle tables."""
    # 21 inverse freqs (Float64 accuracy): inv[i] = 1/10000^((2i)/42)
    var lnbase = log(Float64(10000.0))
    var inv = List[Float64]()
    for i in range(21):
        inv.append(1.0 / exp((2.0 * Float64(i) / 42.0) * lnbase))

    # VID [256,126]: win(4) x h(8) x w(8), row = ax1(58) ++ ax1(h) ++ ax1(w)
    var vid_vals = List[Float32]()
    for _win in range(4):
        for h in range(8):
            for w in range(8):
                for i in range(21):
                    var v = Float32(58.0 * inv[i])
                    vid_vals.append(v)
                    vid_vals.append(v)
                for i in range(21):
                    var v = Float32(Float64(h) * inv[i])
                    vid_vals.append(v)
                    vid_vals.append(v)
                for i in range(21):
                    var v = Float32(Float64(w) * inv[i])
                    vid_vals.append(v)
                    vid_vals.append(v)

    # TXT [232,126]: win(4) x l(58), row = ax1(l) tiled x3
    var txt_vals = List[Float32]()
    for _win in range(4):
        for l in range(58):
            for _rep in range(3):
                for i in range(21):
                    var v = Float32(Float64(l) * inv[i])
                    txt_vals.append(v)
                    txt_vals.append(v)

    var vshp = List[Int]()
    vshp.append(256)
    vshp.append(126)
    var tshp = List[Int]()
    tshp.append(232)
    tshp.append(126)
    var vid_freqs = Tensor.from_host(vid_vals, vshp^, STDtype.F32, ctx)
    var txt_freqs = Tensor.from_host(txt_vals, tshp^, STDtype.F32, ctx)
    return (vid_freqs^, txt_freqs^)


# ═══════════════════════════════════════════════════════════════════════════════
# (H') compute_dit_freqs(txt_len) — GENERALIZED axial-3D RoPE tables for CFG.
#   Same math as compute_dit_freqs(ctx) but the txt length varies (pos=58, neg=64).
#   The VID temporal position == txt_len (the vid temporal pos IS the txt-length
#   offset), so ax1(txt_len) drives the vid [0:42] block. For txt_len=58 this
#   reproduces compute_dit_freqs(ctx) exactly.
#     VID [256,126]: token = win*64 + h*8 + w, row = ax1(txt_len) ++ ax1(h) ++ ax1(w)
#     TXT [txt_len*4,126]: token = win*txt_len + l, row = ax1(l) tiled x3
# ═══════════════════════════════════════════════════════════════════════════════
def compute_dit_freqs(txt_len: Int, ctx: DeviceContext) raises -> Tuple[Tensor, Tensor]:
    """(vid_freqs [256,126], txt_freqs [txt_len*4,126]) F32 axial-3D RoPE tables.
    vid temporal position = txt_len; txt uses per-window length txt_len."""
    var lnbase = log(Float64(10000.0))
    var inv = List[Float64]()
    for i in range(21):
        inv.append(1.0 / exp((2.0 * Float64(i) / 42.0) * lnbase))

    # VID [256,126]: win(4) x h(8) x w(8), row = ax1(txt_len) ++ ax1(h) ++ ax1(w)
    var vid_vals = List[Float32]()
    for _win in range(4):
        for h in range(8):
            for w in range(8):
                for i in range(21):
                    var v = Float32(Float64(txt_len) * inv[i])
                    vid_vals.append(v)
                    vid_vals.append(v)
                for i in range(21):
                    var v = Float32(Float64(h) * inv[i])
                    vid_vals.append(v)
                    vid_vals.append(v)
                for i in range(21):
                    var v = Float32(Float64(w) * inv[i])
                    vid_vals.append(v)
                    vid_vals.append(v)

    # TXT [txt_len*4,126]: win(4) x l(txt_len), row = ax1(l) tiled x3
    var txt_vals = List[Float32]()
    for _win in range(4):
        for l in range(txt_len):
            for _rep in range(3):
                for i in range(21):
                    var v = Float32(Float64(l) * inv[i])
                    txt_vals.append(v)
                    txt_vals.append(v)

    var vshp = List[Int]()
    vshp.append(256)
    vshp.append(126)
    var tshp = List[Int]()
    tshp.append(txt_len * 4)
    tshp.append(126)
    var vid_freqs = Tensor.from_host(vid_vals, vshp^, STDtype.F32, ctx)
    var txt_freqs = Tensor.from_host(txt_vals, tshp^, STDtype.F32, ctx)
    return (vid_freqs^, txt_freqs^)


# ═══════════════════════════════════════════════════════════════════════════════
# (H'') GENERAL arbitrary-resolution window path — SeedVR make_720Pwindows_bysize.
#   Replicates NaSwinAttention window partitioning for ANY post-patch grid (T,H,W),
#   producing windows of UNEVEN count/size (vs the hardcoded 4x(1,8,8) fast path).
#   num_windows = (4,3,3).  See window_gen_oracle.py.
# ═══════════════════════════════════════════════════════════════════════════════
comptime _SCALE_NUM = 45 * 80  # 3600 (720P reference token budget H*W)


def _ceil_div(a: Int, b: Int) -> Int:
    """ceil(a/b) for a>=0, b>0."""
    return (a + b - 1) // b


def _round_half_even(x: Float64) -> Int:
    """Python round(): round-half-to-even. x assumed >= 0."""
    var fl = Int(x)                 # trunc toward zero == floor for x>=0
    var diff = x - Float64(fl)
    if diff < 0.5:
        return fl
    if diff > 0.5:
        return fl + 1
    # exactly .5 -> nearest even
    if fl % 2 == 0:
        return fl
    return fl + 1


struct WindowPlan(Copyable, Movable):
    """Host-computed window partition for the SeedVR general attention path.
    wf/wh/ww hold per-window (f,h,w) shapes; counts[w] = f*h*w. partition maps
    windowed index -> original flat index (windowed[i] = vid[partition[i]]);
    reverse is argsort(partition) so out[j] = windowed_out[reverse[j]]."""

    var wf: List[Int]
    var wh: List[Int]
    var ww: List[Int]
    var counts: List[Int]
    var partition: List[Int]
    var reverse: List[Int]
    var win_count: Int
    var L: Int

    def __init__(
        out self,
        var wf: List[Int],
        var wh: List[Int],
        var ww: List[Int],
        var counts: List[Int],
        var partition: List[Int],
        var reverse: List[Int],
        win_count: Int,
        L: Int,
    ):
        self.wf = wf^
        self.wh = wh^
        self.ww = ww^
        self.counts = counts^
        self.partition = partition^
        self.reverse = reverse^
        self.win_count = win_count
        self.L = L


def compute_windows(T: Int, H: Int, W: Int) raises -> WindowPlan:
    """SeedVR make_720Pwindows_bysize(num_windows=(4,3,3)) — NON-shifted.
    Returns a WindowPlan with per-window shapes, counts, and the flat
    partition/reverse index maps (all host Int math)."""
    var rnt = 4
    var rnh = 3
    var rnw = 3
    var scale = sqrt(Float64(_SCALE_NUM) / Float64(H * W))
    var rH = _round_half_even(Float64(H) * scale)
    var rW = _round_half_even(Float64(W) * scale)
    var wh = _ceil_div(rH, rnh)
    var ww = _ceil_div(rW, rnw)
    var tmin = T
    if tmin > 30:
        tmin = 30
    var wt = _ceil_div(tmin, rnt)
    var nt = _ceil_div(T, wt)
    var nh = _ceil_div(H, wh)
    var nw = _ceil_div(W, ww)

    var wf = List[Int]()
    var whs = List[Int]()
    var wws = List[Int]()
    var counts = List[Int]()
    var partition = List[Int]()
    # ORDER: for iw, for ih, for it (matches oracle make_windows loop nesting).
    for iw in range(nw):
        for ih in range(nh):
            for it in range(nt):
                var t0 = it * wt
                var t1 = (it + 1) * wt
                if t1 > T:
                    t1 = T
                var h0 = ih * wh
                var h1 = (ih + 1) * wh
                if h1 > H:
                    h1 = H
                var w0 = iw * ww
                var w1 = (iw + 1) * ww
                if w1 > W:
                    w1 = W
                if t1 > t0 and h1 > h0 and w1 > w0:
                    var f = t1 - t0
                    var hh = h1 - h0
                    var wwd = w1 - w0
                    wf.append(f)
                    whs.append(hh)
                    wws.append(wwd)
                    counts.append(f * hh * wwd)
                    for tt in range(t0, t1):
                        for hp in range(h0, h1):
                            for wp in range(w0, w1):
                                partition.append(tt * H * W + hp * W + wp)

    var L = len(partition)
    # reverse = argsort(partition). partition is a permutation of 0..L-1, so
    # reverse[partition[i]] = i  ->  reverse[j] = i s.t. partition[i] = j.
    var reverse = List[Int]()
    for _j in range(L):
        reverse.append(0)
    for i in range(L):
        reverse[partition[i]] = i

    var win_count = len(counts)
    return WindowPlan(wf^, whs^, wws^, counts^, partition^, reverse^, win_count, L)


# ═══════════════════════════════════════════════════════════════════════════════
# (H''') compute_dit_freqs_general — axial-3D RoPE tables for the GENERAL windows.
#   vid: L=sum(f*h*w) tokens in WINDOWED order; per token in window (f,h,w) at
#        local (ft,fh,fw): row = ax1(txt_len+ft) ++ ax1(fh) ++ ax1(fw)  (each 42).
#   txt: Ntxt=win_count*txt_len; per window per l in 0..txt_len: ax1(l) tiled x3.
#   ax1 identical to compute_dit_freqs (inv[i]=1/10000^(2i/42), repeat-interleave).
#   For the fixed 4x(1,8,8) grid + txt_len=58 this reduces to compute_dit_freqs.
# ═══════════════════════════════════════════════════════════════════════════════
def compute_dit_freqs_general(
    plan: WindowPlan, txt_len: Int, ctx: DeviceContext
) raises -> Tuple[Tensor, Tensor]:
    """(vid_freqs [L,126], txt_freqs [win_count*txt_len,126]) F32 axial tables."""
    var lnbase = log(Float64(10000.0))
    var inv = List[Float64]()
    for i in range(21):
        inv.append(1.0 / exp((2.0 * Float64(i) / 42.0) * lnbase))

    # VID [L,126] windowed order: window (f,h,w) -> ft,fh,fw (t outer, h, w inner)
    var vid_vals = List[Float32]()
    for w in range(plan.win_count):
        var f = plan.wf[w]
        var hh = plan.wh[w]
        var wwd = plan.ww[w]
        for ft in range(f):
            for fh in range(hh):
                for fw in range(wwd):
                    var tp = Float64(txt_len + ft)
                    for i in range(21):
                        var v = Float32(tp * inv[i])
                        vid_vals.append(v)
                        vid_vals.append(v)
                    for i in range(21):
                        var v = Float32(Float64(fh) * inv[i])
                        vid_vals.append(v)
                        vid_vals.append(v)
                    for i in range(21):
                        var v = Float32(Float64(fw) * inv[i])
                        vid_vals.append(v)
                        vid_vals.append(v)

    # TXT [win_count*txt_len,126]: per window, l in 0..txt_len, ax1(l) tiled x3
    var txt_vals = List[Float32]()
    for _w in range(plan.win_count):
        for l in range(txt_len):
            for _rep in range(3):
                for i in range(21):
                    var v = Float32(Float64(l) * inv[i])
                    txt_vals.append(v)
                    txt_vals.append(v)

    var vshp = List[Int]()
    vshp.append(plan.L)
    vshp.append(126)
    var tshp = List[Int]()
    tshp.append(plan.win_count * txt_len)
    tshp.append(126)
    var vid_freqs = Tensor.from_host(vid_vals, vshp^, STDtype.F32, ctx)
    var txt_freqs = Tensor.from_host(txt_vals, tshp^, STDtype.F32, ctx)
    return (vid_freqs^, txt_freqs^)


# ═══════════════════════════════════════════════════════════════════════════════
# (E'') window_attn_projout_general — arbitrary-resolution NaSwinAttention tail.
#   vid q/k/v arrive in ORIGINAL grid order [L,20,128]; gathered to windowed order
#   by plan.partition, roped with the WINDOWED vid_freqs, then attended PER WINDOW
#   (windows have DIFFERENT sizes -> runtime manual multi-head attention, like the
#   VAE mid_attention but 20 heads). txt is tiled win_count x and roped with
#   txt_freqs. Each window's joint sequence = [c_w vid ; txt_len txt]; split -> vid
#   scattered back (gather by reverse), txt mean-pooled over windows. Per-stream
#   proj_out (+bias). Returns (vid_out [L,2560], txt_out [txt_len,2560]) bf16.
# ═══════════════════════════════════════════════════════════════════════════════
def window_attn_projout_general(
    vid_q: Tensor,          # [L,20,128] bf16  (ORIGINAL grid order, post qk-norm)
    vid_k: Tensor,          # [L,20,128] bf16
    vid_v: Tensor,          # [L,20,128] bf16  (un-normed value)
    txt_q: Tensor,          # [txt_len,20,128] bf16  (single copy, post qk-norm)
    txt_k: Tensor,          # [txt_len,20,128] bf16
    txt_v: Tensor,          # [txt_len,20,128] bf16  (un-normed value)
    plan: WindowPlan,
    txt_len: Int,
    projout_vid_w: Tensor,  # [2560,2560]
    projout_vid_b: Tensor,  # [2560]
    projout_txt_w: Tensor,  # [2560,2560]
    projout_txt_b: Tensor,  # [2560]
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    var HEADS = 20
    var HD = 128
    var INNER = HEADS * HD  # 2560
    var L = plan.L
    var NW = plan.win_count

    # ── window_partition: gather vid q/k/v ORIGINAL -> WINDOWED order ────────────
    # windowed[i] = vid[partition[i]]. gather_rows works on rank-2, so flatten the
    # (H,D) axes, gather rows, reshape back to [L,20,128].
    var vq2 = reshape(vid_q, [L, INNER], ctx)
    var vk2 = reshape(vid_k, [L, INNER], ctx)
    var vv2 = reshape(vid_v, [L, INNER], ctx)
    var wq = reshape(gather_rows(vq2, plan.partition, ctx), [L, HEADS, HD], ctx)
    var wk = reshape(gather_rows(vk2, plan.partition, ctx), [L, HEADS, HD], ctx)
    var wv = reshape(gather_rows(vv2, plan.partition, ctx), [L, HEADS, HD], ctx)

    # ── freqs (windowed vid order + per-window txt) ─────────────────────────────
    var freqs = compute_dit_freqs_general(plan, txt_len, ctx)  # (vid[L,126], txt[NW*txt_len,126])

    # ── rope vid (windowed) ─────────────────────────────────────────────────────
    var wq_r = apply_rope3d(wq, freqs[0], ctx)
    var wk_r = apply_rope3d(wk, freqs[0], ctx)
    # wv is NOT roped (value).

    # ── txt tiled win_count x, then rope with txt_freqs ─────────────────────────
    # All windows share identical txt_freqs, so the win_count copies are identical
    # after rope; we build the tiled buffer explicitly to match the freqs layout.
    var tq_tiled = txt_q.clone(ctx)
    var tk_tiled = txt_k.clone(ctx)
    var tv_tiled = txt_v.clone(ctx)
    for _w in range(1, NW):
        tq_tiled = concat(0, ctx, tq_tiled, txt_q)
        tk_tiled = concat(0, ctx, tk_tiled, txt_k)
        tv_tiled = concat(0, ctx, tv_tiled, txt_v)
    var tq_r = apply_rope3d(tq_tiled, freqs[1], ctx)  # [NW*txt_len,20,128]
    var tk_r = apply_rope3d(tk_tiled, freqs[1], ctx)
    # tv_tiled is the (tiled) un-roped value.

    var scale = Float32(1.0) / Float32(sqrt(Float64(HD)))

    # ── PER-WINDOW runtime multi-head attention ─────────────────────────────────
    var windowed_out = vid_q.clone(ctx)   # placeholder [L,20,128], overwritten
    var txt_acc = txt_q.clone(ctx)        # placeholder [txt_len,20,128]
    var have_txt_acc = False
    var offset = 0
    for w in range(NW):
        var c_w = plan.counts[w]
        var S_w = c_w + txt_len

        # joint q/k/v for this window: [vid c_w ; txt txt_len] -> [S_w,20,128]
        var vqw = slice(wq_r, 0, offset, c_w, ctx)             # [c_w,20,128]
        var vkw = slice(wk_r, 0, offset, c_w, ctx)
        var vvw = slice(wv, 0, offset, c_w, ctx)
        var tqw = slice(tq_r, 0, w * txt_len, txt_len, ctx)    # [txt_len,20,128]
        var tkw = slice(tk_r, 0, w * txt_len, txt_len, ctx)
        var tvw = slice(tv_tiled, 0, w * txt_len, txt_len, ctx)
        var qw = concat(0, ctx, vqw, tqw)                      # [S_w,20,128]
        var kw = concat(0, ctx, vkw, tkw)
        var vw = concat(0, ctx, vvw, tvw)

        # per-head attention, assembled into [S_w,20,128]
        var head_out = qw.clone(ctx)   # placeholder, overwritten at h==0
        for h in range(HEADS):
            var qh = reshape(slice(qw, 1, h, 1, ctx), [S_w, HD], ctx)   # [S_w,128]
            var kh = reshape(slice(kw, 1, h, 1, ctx), [S_w, HD], ctx)
            var vh = reshape(slice(vw, 1, h, 1, ctx), [S_w, HD], ctx)
            var scores = linear(qh, kh, None, ctx, True)               # qh @ khᵀ [S_w,S_w]
            scores = mul_scalar(scores, scale, ctx)
            var attn = softmax_lastdim(scores, ctx)                    # [S_w,S_w]
            var oh = linear(attn, vh, None, ctx, False)                # attn @ vh [S_w,128]
            var oh3 = reshape(oh, [S_w, 1, HD], ctx)                   # [S_w,1,128]
            if h == 0:
                head_out = oh3^
            else:
                head_out = concat(1, ctx, head_out, oh3)              # [S_w,h+1,128]

        # split window output: vid part [c_w,20,128], txt part [txt_len,20,128]
        var vid_part = slice(head_out, 0, 0, c_w, ctx)                 # [c_w,20,128]
        var txt_part = slice(head_out, 0, c_w, txt_len, ctx)          # [txt_len,20,128]

        # scatter vid part into windowed_out (windowed order is contiguous per win)
        if w == 0:
            windowed_out = vid_part^
        else:
            windowed_out = concat(0, ctx, windowed_out, vid_part)

        # accumulate txt part for mean pool over windows
        if not have_txt_acc:
            txt_acc = txt_part^
            have_txt_acc = True
        else:
            txt_acc = add(txt_acc, txt_part, ctx)

        offset += c_w

    # ── vid: windowed_out -> original order via reverse gather -> proj_out ───────
    var wout2 = reshape(windowed_out, [L, INNER], ctx)                # [L,2560]
    var vid_orig = gather_rows(wout2, plan.reverse, ctx)             # [L,2560] orig order
    var vid_out = linear_bias(vid_orig, projout_vid_w, projout_vid_b, ctx)  # [L,2560]

    # ── txt: mean over windows -> proj_out ──────────────────────────────────────
    var txt_mean = mul_scalar(txt_acc, Float32(1.0) / Float32(NW), ctx)  # [txt_len,20,128]
    var txt_2d = reshape(txt_mean, [txt_len, INNER], ctx)                 # [txt_len,2560]
    var txt_out = linear_bias(txt_2d, projout_txt_w, projout_txt_b, ctx)  # [txt_len,2560]

    return (vid_out^, txt_out^)


# ═══════════════════════════════════════════════════════════════════════════════
# (I) txt_in — Linear text-embedding entry:  txt_raw [58,5120] -> [58,2560].
#   weight txt_in.weight [2560,5120], bias txt_in.bias [2560].
# ═══════════════════════════════════════════════════════════════════════════════
def txt_in(
    txt_raw: Tensor,   # [58, 5120]  (bf16)
    proj_w: Tensor,    # txt_in.weight  [2560, 5120]
    proj_b: Tensor,    # txt_in.bias    [2560]
    ctx: DeviceContext,
) raises -> Tensor:
    """SeedVR NaDiT txt_in Linear (+bias): [58,5120] -> [58,2560] (bf16)."""
    return linear_bias(txt_raw, proj_w, proj_b, ctx)


# ═══════════════════════════════════════════════════════════════════════════════
# (J) full_dit_forward — the WHOLE SeedVR2-3B NaDiT forward with COMPUTED freqs.
#   vid_in + txt_in + emb_in -> compute_dit_freqs -> 32-block dit_stack -> tail.
#   Loads all DiT weights (non-block + 32 blocks) from an open SafeTensors handle.
#   Returns the final latent [1024,16] (bf16).
# ═══════════════════════════════════════════════════════════════════════════════
def full_dit_forward(
    vid_raw: Tensor,   # [T*H*W, 33] = [1024,33] (bf16), grid channels-last
    txt_raw: Tensor,   # [58, 5120]  (bf16)
    timestep: Float32, # scalar (= 500.0)
    T: Int,            # pre-patch grid T (=4)
    H: Int,            # pre-patch grid H (=16)
    W: Int,            # pre-patch grid W (=16)
    st: SafeTensors,   # open seedvr2_dit.safetensors (bf16)
    ctx: DeviceContext,
) raises -> Tensor:
    """SeedVR2-3B NaDiT full forward -> latent [1024,16] (bf16)."""
    # ── input embeddings ────────────────────────────────────────────────────────
    var vid = vid_in(
        vid_raw,
        load_bias(st, "vid_in.proj.weight", ctx),
        load_bias(st, "vid_in.proj.bias", ctx),
        T, H, W, ctx,
    )                                                          # [256,2560]
    var txt = txt_in(
        txt_raw,
        load_bias(st, "txt_in.weight", ctx),
        load_bias(st, "txt_in.bias", ctx),
        ctx,
    )                                                          # [58,2560]
    var emb = emb_in(
        timestep,
        load_bias(st, "emb_in.proj_in.weight", ctx),
        load_bias(st, "emb_in.proj_in.bias", ctx),
        load_bias(st, "emb_in.proj_hid.weight", ctx),
        load_bias(st, "emb_in.proj_hid.bias", ctx),
        load_bias(st, "emb_in.proj_out.weight", ctx),
        load_bias(st, "emb_in.proj_out.bias", ctx),
        ctx,
    )                                                          # [1,15360]

    # ── computed axial-3D RoPE angle tables (F32) ───────────────────────────────
    var freqs = compute_dit_freqs(ctx)                         # (vid[256,126], txt[232,126])

    # ── 32 MMDiT blocks ─────────────────────────────────────────────────────────
    var blocks = List[ArcPointer[Block0Weights]]()
    for i in range(32):
        blocks.append(ArcPointer(load_block(st, i, ctx)))
    var vid_stack = dit_stack(vid, txt, emb, freqs[0], freqs[1], blocks, ctx)  # [256,2560]

    # ── output tail -> latent [1024,16] ─────────────────────────────────────────
    return dit_out_tail(
        vid_stack,
        emb,
        load_bias(st, "vid_out_norm.weight", ctx),
        load_bias(st, "vid_out_ada.out_shift", ctx),
        load_bias(st, "vid_out_ada.out_scale", ctx),
        load_bias(st, "vid_out.proj.weight", ctx),
        load_bias(st, "vid_out.proj.bias", ctx),
        ctx,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# (K) DiTWeights + load_dit_weights + full_dit_forward_pre — PRELOADED weights.
#   The sampler runs the DiT 16 times (8 steps x pos/neg CFG); reloading 6.8GB of
#   weights per call is intolerable, so load_dit_weights streams ALL weights
#   (non-block heads/tail + 32 blocks) ONCE into a DiTWeights, and
#   full_dit_forward_pre reuses them (borrowed) for every call. Same math as
#   full_dit_forward but with the TW-generalized freqs + block path so the CFG
#   pos (txt_len=58) and neg (txt_len=64) streams share one code path.
# ═══════════════════════════════════════════════════════════════════════════════
struct DiTWeights(Movable):
    """All non-block DiT weights (input heads + output tail) + the 32 blocks."""

    var vin_w: Tensor        # vid_in.proj.weight  [2560,132]
    var vin_b: Tensor        # vid_in.proj.bias    [2560]
    var txt_w: Tensor        # txt_in.weight       [2560,5120]
    var txt_b: Tensor        # txt_in.bias         [2560]
    var ein_pin_w: Tensor    # emb_in.proj_in.weight  [2560,256]
    var ein_pin_b: Tensor    # emb_in.proj_in.bias    [2560]
    var ein_phid_w: Tensor   # emb_in.proj_hid.weight [2560,2560]
    var ein_phid_b: Tensor   # emb_in.proj_hid.bias   [2560]
    var ein_pout_w: Tensor   # emb_in.proj_out.weight [15360,2560]
    var ein_pout_b: Tensor   # emb_in.proj_out.bias   [15360]
    var tail_norm_w: Tensor  # vid_out_norm.weight   [2560]
    var tail_ada_shift: Tensor  # vid_out_ada.out_shift [2560]
    var tail_ada_scale: Tensor  # vid_out_ada.out_scale [2560]
    var tail_proj_w: Tensor  # vid_out.proj.weight   [64,2560]
    var tail_proj_b: Tensor  # vid_out.proj.bias     [64]
    var blocks: List[ArcPointer[Block0Weights]]  # 32 MMDiT blocks

    def __init__(
        out self,
        var vin_w: Tensor,
        var vin_b: Tensor,
        var txt_w: Tensor,
        var txt_b: Tensor,
        var ein_pin_w: Tensor,
        var ein_pin_b: Tensor,
        var ein_phid_w: Tensor,
        var ein_phid_b: Tensor,
        var ein_pout_w: Tensor,
        var ein_pout_b: Tensor,
        var tail_norm_w: Tensor,
        var tail_ada_shift: Tensor,
        var tail_ada_scale: Tensor,
        var tail_proj_w: Tensor,
        var tail_proj_b: Tensor,
        var blocks: List[ArcPointer[Block0Weights]],
    ):
        self.vin_w = vin_w^
        self.vin_b = vin_b^
        self.txt_w = txt_w^
        self.txt_b = txt_b^
        self.ein_pin_w = ein_pin_w^
        self.ein_pin_b = ein_pin_b^
        self.ein_phid_w = ein_phid_w^
        self.ein_phid_b = ein_phid_b^
        self.ein_pout_w = ein_pout_w^
        self.ein_pout_b = ein_pout_b^
        self.tail_norm_w = tail_norm_w^
        self.tail_ada_shift = tail_ada_shift^
        self.tail_ada_scale = tail_ada_scale^
        self.tail_proj_w = tail_proj_w^
        self.tail_proj_b = tail_proj_b^
        self.blocks = blocks^


def load_dit_weights(st: SafeTensors, ctx: DeviceContext) raises -> DiTWeights:
    """Stream the entire SeedVR2-3B DiT (heads + tail + 32 blocks) ONCE."""
    var blocks = List[ArcPointer[Block0Weights]]()
    for i in range(32):
        blocks.append(ArcPointer(load_block(st, i, ctx)))
    return DiTWeights(
        load_bias(st, "vid_in.proj.weight", ctx),
        load_bias(st, "vid_in.proj.bias", ctx),
        load_bias(st, "txt_in.weight", ctx),
        load_bias(st, "txt_in.bias", ctx),
        load_bias(st, "emb_in.proj_in.weight", ctx),
        load_bias(st, "emb_in.proj_in.bias", ctx),
        load_bias(st, "emb_in.proj_hid.weight", ctx),
        load_bias(st, "emb_in.proj_hid.bias", ctx),
        load_bias(st, "emb_in.proj_out.weight", ctx),
        load_bias(st, "emb_in.proj_out.bias", ctx),
        load_bias(st, "vid_out_norm.weight", ctx),
        load_bias(st, "vid_out_ada.out_shift", ctx),
        load_bias(st, "vid_out_ada.out_scale", ctx),
        load_bias(st, "vid_out.proj.weight", ctx),
        load_bias(st, "vid_out.proj.bias", ctx),
        blocks^,
    )


def full_dit_forward_pre(
    vid_raw: Tensor,   # [T*H*W,33] = [1024,33] (bf16), grid channels-last
    txt_raw: Tensor,   # [txt_len,5120] (bf16)
    txt_len: Int,      # 58 (CFG pos) or 64 (CFG neg)
    timestep: Float32, # scalar
    w: DiTWeights,     # PRELOADED weights (borrowed, reused across all calls)
    ctx: DeviceContext,
) raises -> Tensor:
    """SeedVR2-3B NaDiT forward using PRELOADED weights + TW-generalized freqs/blocks.
    Grid fixed at T=4,H=16,W=16 (vid_in patchifies -> 256 tokens). Returns [1024,16] bf16."""
    var vid = vid_in(vid_raw, w.vin_w, w.vin_b, 4, 16, 16, ctx)   # [256,2560]
    var txt = txt_in(txt_raw, w.txt_w, w.txt_b, ctx)              # [txt_len,2560]
    var emb = emb_in(
        timestep,
        w.ein_pin_w, w.ein_pin_b,
        w.ein_phid_w, w.ein_phid_b,
        w.ein_pout_w, w.ein_pout_b,
        ctx,
    )                                                             # [1,15360]
    var freqs = compute_dit_freqs(txt_len, ctx)  # (vid[256,126], txt[txt_len*4,126])

    if txt_len == 58:
        var vs = dit_stack_g[58](vid, txt, emb, freqs[0], freqs[1], w.blocks, ctx)
        return dit_out_tail(
            vs, emb, w.tail_norm_w, w.tail_ada_shift, w.tail_ada_scale,
            w.tail_proj_w, w.tail_proj_b, ctx,
        )
    elif txt_len == 64:
        var vs = dit_stack_g[64](vid, txt, emb, freqs[0], freqs[1], w.blocks, ctx)
        return dit_out_tail(
            vs, emb, w.tail_norm_w, w.tail_ada_shift, w.tail_ada_scale,
            w.tail_proj_w, w.tail_proj_b, ctx,
        )
    else:
        raise Error("full_dit_forward_pre: unsupported txt_len (want 58 or 64)")
