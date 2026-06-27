# acestep_dit.mojo — ACE-Step-1.5 DiT (audio diffusion transformer), inference.
#
# Pure Mojo + MAX, GPU-only, BF16 weights+activations / F32-accumulate.
# Reference (READ-ONLY): EriDiffusion/inference-flame/src/models/acestep_dit.rs
# Canonical oracle: acestep-v15-turbo/modeling_acestep_v15_turbo.py
#   (AceStepDiTLayer.forward, AceStepDiTModel.forward, Qwen3 rope/rmsnorm).
#
# AUDIO SEQ LAYOUT: 1-D sequence (NOT 3D spatial). Latent x_t is [B, T, 64];
# context_latents [B, T, 128] cat'd on last dim -> [B,T,192]; pad T to multiple
# of patch_size=2; Conv1d(k=2,s=2) patch-embed -> [B, T/2, 2048]. Transformer
# runs over the T/2 token sequence. RoPE = 1-D positions 0..T/2-1.
#
# ARCH (turbo): 24 DiT layers, hidden=2048, heads=16, kv_heads=8 (GQA n_rep=2),
#   head_dim=128, intermediate=6144, in_ch=192, acoustic=64, patch=2,
#   rope_theta=1e6, eps=1e-6. Each layer: self-attn (RoPE + per-head QK-RMSNorm,
#   GQA) with AdaLN modulation (gate); cross-attn (no RoPE, per-head QK-RMSNorm,
#   GQA, KV from condition encoder) plain residual; SwiGLU MLP with AdaLN (gate).
#   6-way scale_shift_table[1,6,H] + timestep_proj. Final: AdaLN(norm_out, 2-way)
#   then ConvTranspose1d(k=2,s=2) -> [B,T,64], crop to original length.
#
# RoPE scheme: Qwen3 rotate_half == HALFSPLIT (ops/rope.rope_halfsplit). cos/sin
#   = cos/sin(pos * theta^(-2i/Dh)), i in [0,Dh/2). qk-norm BEFORE rope.
#
# SELF-ATTN MASK: layer_types alternate sliding(window=128)/full, both
#   BIDIRECTIONAL (is_causal=False). layer i is "sliding_attention" when
#   (i+1)%2==1 (i.e. i even: 0,2,...,22 = 12 sliding layers), else "full".
#   At T/2 <= 128 the sliding mask is all-zeros == full == no mask, so
#   sdpa_nomask is exact (block-0 gate S=64; full gate SP=100). At T/2 > 128 the
#   sliding layers build the |i-j|<=window mask [1,H,S,S] in Q's storage dtype
#   and run sdpa_tiled (full-mask, online softmax, no [S,S] OOM); full layers
#   stay sdpa_nomask.
#   Verified: long-seq gate SP=300 cos=0.99976 vs canonical (forcing all-global
#   diverges to 0.9919 — the mask is load-bearing). See _self_attn /
#   _build_sliding_mask + acestep_full_longseq_gate.mojo.
#
# REUSE: ops/linear.linear, ops/norm.rms_norm, ops/attention.sdpa_nomask,
#   ops/rope.rope_halfsplit, ops/activations.silu, ops/tensor_algebra.{reshape,
#   slice,transpose}. New local kernel: _repeat_kv (GQA), copied from hidream_o1.
#
# DEFERRED (follow-ons, NOT built this pass): the acoustic VAE
#   (checkpoints/vae/), the condition encoder (lyric/timbre/text -> 2048 ctx,
#   Qwen3-Embedding-0.6B text), and the rectified-flow sampler. The block gate
#   feeds a fixed random conditioning tensor of the right shape [1,L,2048] for
#   the cross-attn inputs.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.memory import ArcPointer
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import silu
from serenitymojo.ops.attention import sdpa_nomask, sdpa_tiled, sdpa_nomask_tiled
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.tensor_algebra import reshape, slice, transpose, concat
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.embeddings import timestep_embedding
from std.math import cos as _cos64, sin as _sin64


@fieldwise_init
struct AceStepDiTConfig(Copyable, Movable, ImplicitlyCopyable):
    var hidden_size: Int      # 2048
    var num_heads: Int        # 16
    var num_kv_heads: Int     # 8
    var head_dim: Int         # 128
    var intermediate: Int     # 6144
    var num_layers: Int       # 24
    var in_channels: Int      # 192
    var acoustic_dim: Int     # 64
    var patch_size: Int       # 2
    var rope_theta: Float64   # 1e6
    var rms_norm_eps: Float32 # 1e-6
    var sliding_window: Int   # 128

    @staticmethod
    def turbo() -> AceStepDiTConfig:
        return AceStepDiTConfig(
            2048, 16, 8, 128, 6144, 24, 192, 64, 2, 1_000_000.0, 1e-6, 128
        )


# ── GQA repeat_kv (BSHD [1,S,H_kv,Dh] -> [1,S,H,Dh]) ─────────────────────────
# Grouped order: dst head reads src kv-head head//n_rep (PyTorch repeat_kv).
# Copied verbatim from hidream_o1._repeat_kv (a verified faithful port), not a
# foundation-op reimpl.
comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


def _repeat_kv_kernel_bf16(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    seq: Int, h: Int, h_kv: Int, dh: Int, n_rep: Int,
):
    var idx = Int(global_idx.x)
    var total = seq * h * dh
    if idx < total:
        var dh_i = idx % dh
        var rest = idx // dh
        var head = rest % h
        var t = rest // h
        var kvh = head // n_rep
        var src_idx = (t * h_kv + kvh) * dh + dh_i
        dst[idx] = rebind[dst.element_type](src[src_idx])


def _repeat_kv(
    x: Tensor, s: Int, h_kv: Int, n_rep: Int, dh: Int, ctx: DeviceContext
) raises -> Tensor:
    if n_rep == 1:
        var dev0 = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
        ctx.enqueue_copy(dst_buf=dev0, src_buf=x.buf)
        ctx.synchronize()
        return Tensor(dev0^, x.shape(), x.dtype())
    var h = h_kv * n_rep
    var out_n = s * h * dh
    var src_n = s * h_kv * dh
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * x.dtype().byte_size()
    )
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](src_n))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    var SRC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
    )
    var DST = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )
    ctx.enqueue_function[_repeat_kv_kernel_bf16, _repeat_kv_kernel_bf16](
        SRC, DST, s, h, h_kv, dh, n_rep, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var sh = [1, s, h, dh]
    return Tensor(out_buf^, sh^, x.dtype())


def _clone(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    return t.clone(ctx)


def _w(
    bw: Dict[String, ArcPointer[Tensor]], name: String, ctx: DeviceContext
) raises -> Tensor:
    return _clone(bw[name][], ctx)


# Per-head RMSNorm: x is [1,S,heads,Dh]; weight [Dh]; normalize over last dim.
def _qk_norm(
    x: Tensor, weight: Tensor, s: Int, heads: Int, dh: Int, eps: Float32,
    ctx: DeviceContext
) raises -> Tensor:
    var flat = reshape(x, [s * heads, dh], ctx)
    var normed = rms_norm(flat, weight, eps, ctx)
    return reshape(normed, [1, s, heads, dh], ctx)


# Slice the [1,6,H] modulation table into the k-th [H] vector.
def _mod_chunk(mod6: Tensor, k: Int, h: Int, ctx: DeviceContext) raises -> Tensor:
    # mod6 is [1,6,H]; take row k -> [H]
    var row = slice(mod6, 1, k, 1, ctx)   # [1,1,H]
    return reshape(row, [h], ctx)


# Tile cos/sin rows [pos, Dh/2] -> [pos*heads, Dh/2] (each position repeated
# across `heads` consecutive rows, matching reshape [S,heads,Dh] row-major).
def _tile_rows(
    x: Tensor, s: Int, heads: Int, half: Int, ctx: DeviceContext
) raises -> Tensor:
    var out_n = s * heads * half
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * x.dtype().byte_size()
    )
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](s * half))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    var SRC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
    )
    var DST = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )
    ctx.enqueue_function[_tile_rows_kernel, _tile_rows_kernel](
        SRC, DST, s, heads, half, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var sh = [s * heads, half]
    return Tensor(out_buf^, sh^, x.dtype())


def _tile_rows_kernel(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    s: Int, heads: Int, half: Int,
):
    var idx = Int(global_idx.x)
    var total = s * heads * half
    if idx < total:
        var d_i = idx % half
        var rest = idx // half
        var pos = rest // heads   # row = pos*heads + head
        var src_idx = pos * half + d_i
        dst[idx] = rebind[dst.element_type](src[src_idx])


# ── bidirectional sliding-window additive mask [1,H,S,S] storage dtype ───────
# Matches canonical create_4d_mask(is_sliding_window=True, is_causal=False):
#   keep (0.0) where |i-j| <= window, else large-negative (-1e9, == MLX/finfo).
# The tensor boundary matches q/k/v storage; sdpa_tiled casts the mask scalar to
# F32 internally for score math.
def _sliding_mask_kernel[dtype: DType](
    dst: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    h: Int, s: Int, window: Int,
):
    var idx = Int(global_idx.x)
    var total = h * s * s
    if idx < total:
        var j = idx % s
        var rest = idx // s
        var i = rest % s
        var d = i - j
        if d < 0:
            d = -d
        if d <= window:
            dst[idx] = rebind[dst.element_type](Float32(0.0).cast[dtype]())
        else:
            var blocked = Float32(-30000.0)
            comptime if dtype == DType.float32:
                blocked = Float32(-1.0e9)
            dst[idx] = rebind[dst.element_type](blocked.cast[dtype]())


def _build_sliding_mask(
    h: Int, s: Int, window: Int, dtype: STDtype, ctx: DeviceContext
) raises -> Tensor:
    var n = h * s * s
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * dtype.byte_size())
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = dtype.to_mojo_dtype()
    if dt == DType.float32:
        var DST = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[
            _sliding_mask_kernel[DType.float32],
            _sliding_mask_kernel[DType.float32],
        ](DST, h, s, window, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var DST = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[
            _sliding_mask_kernel[DType.bfloat16],
            _sliding_mask_kernel[DType.bfloat16],
        ](DST, h, s, window, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.float16:
        var DST = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[
            _sliding_mask_kernel[DType.float16],
            _sliding_mask_kernel[DType.float16],
        ](DST, h, s, window, grid_dim=grid, block_dim=_BLOCK)
    else:
        raise Error("_build_sliding_mask: expected F32/BF16/F16 dtype")
    ctx.synchronize()
    var sh = [1, h, s, s]
    return Tensor(out_buf^, sh^, dtype)


# Self-attention (S==SKV, with RoPE). layer_type: 0=full, 1=sliding(window).
def _self_attn[S: Int](
    hidden: Tensor, prefix: String,
    bw: Dict[String, ArcPointer[Tensor]], cfg: AceStepDiTConfig,
    cos_b: Tensor, sin_b: Tensor, layer_type: Int, window: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var nh = cfg.num_heads
    var nkv = cfg.num_kv_heads
    var dh = cfg.head_dim
    var nrep = nh // nkv
    var eps = cfg.rms_norm_eps
    var scale = Float32(1.0) / Float32(dh) ** 0.5

    var q = linear(hidden, _w(bw, prefix + ".q_proj.weight", ctx), None, ctx)
    var k = linear(hidden, _w(bw, prefix + ".k_proj.weight", ctx), None, ctx)
    var v = linear(hidden, _w(bw, prefix + ".v_proj.weight", ctx), None, ctx)
    var qh = _qk_norm(reshape(q, [1, S, nh, dh], ctx),
                      _w(bw, prefix + ".q_norm.weight", ctx), S, nh, dh, eps, ctx)
    var kh = _qk_norm(reshape(k, [1, S, nkv, dh], ctx),
                      _w(bw, prefix + ".k_norm.weight", ctx), S, nkv, dh, eps, ctx)
    var vh = reshape(v, [1, S, nkv, dh], ctx)

    var cos_q = _tile_rows(cos_b, S, nh, dh // 2, ctx)
    var sin_q = _tile_rows(sin_b, S, nh, dh // 2, ctx)
    qh = reshape(rope_halfsplit(reshape(qh, [S * nh, dh], ctx), cos_q, sin_q, ctx),
                 [1, S, nh, dh], ctx)
    var cos_k = _tile_rows(cos_b, S, nkv, dh // 2, ctx)
    var sin_k = _tile_rows(sin_b, S, nkv, dh // 2, ctx)
    kh = reshape(rope_halfsplit(reshape(kh, [S * nkv, dh], ctx), cos_k, sin_k, ctx),
                 [1, S, nkv, dh], ctx)

    var kfull = _repeat_kv(kh, S, nkv, nrep, dh, ctx)
    var vfull = _repeat_kv(vh, S, nkv, nrep, dh, ctx)
    # Sliding layer with S > window: the |i-j|<=window mask is non-trivial, so
    # we must materialize it and run sdpa_tiled (full-mask). For full layers OR
    # when S <= window (mask is all-zeros == no mask), sdpa_nomask is exact.
    var attn: Tensor
    if layer_type == 1 and S > window:
        var smask = _build_sliding_mask(nh, S, window, qh.dtype(), ctx)
        attn = sdpa_tiled[1, S, 16, 128](qh, kfull, vfull, smask, scale, ctx)
    else:
        attn = sdpa_nomask[1, S, 16, 128](qh, kfull, vfull, scale, ctx)
    var flat = reshape(attn, [1, S, nh * dh], ctx)
    return linear(flat, _w(bw, prefix + ".o_proj.weight", ctx), None, ctx)


# Cross-attention (q from hidden [1,S,H], kv from enc [1,L,H], NO RoPE).
def _cross_attn[S: Int, L: Int](
    hidden: Tensor, enc: Tensor, prefix: String,
    bw: Dict[String, ArcPointer[Tensor]], cfg: AceStepDiTConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    var nh = cfg.num_heads
    var nkv = cfg.num_kv_heads
    var dh = cfg.head_dim
    var nrep = nh // nkv
    var eps = cfg.rms_norm_eps
    var scale = Float32(1.0) / Float32(dh) ** 0.5

    var q = linear(hidden, _w(bw, prefix + ".q_proj.weight", ctx), None, ctx)
    var k = linear(enc, _w(bw, prefix + ".k_proj.weight", ctx), None, ctx)
    var v = linear(enc, _w(bw, prefix + ".v_proj.weight", ctx), None, ctx)
    var qh = _qk_norm(reshape(q, [1, S, nh, dh], ctx),
                      _w(bw, prefix + ".q_norm.weight", ctx), S, nh, dh, eps, ctx)
    var kh = _qk_norm(reshape(k, [1, L, nkv, dh], ctx),
                      _w(bw, prefix + ".k_norm.weight", ctx), L, nkv, dh, eps, ctx)
    var vh = reshape(v, [1, L, nkv, dh], ctx)

    var kfull = _repeat_kv(kh, L, nkv, nrep, dh, ctx)   # [1,L,nh,dh]
    var vfull = _repeat_kv(vh, L, nkv, nrep, dh, ctx)
    # sdpa over q-seq S, kv-seq L -> use SKV-parameterized math sdpa.
    var attn = _sdpa_cross[S, L](qh, kfull, vfull, nh, dh, scale, ctx)
    var flat = reshape(attn, [1, S, nh * dh], ctx)
    return linear(flat, _w(bw, prefix + ".o_proj.weight", ctx), None, ctx)


# Cross sdpa S!=SKV: sdpa_nomask requires q/k/v same S. Pad k/v to S if L<S, or
# crop q. To keep it exact we tile k/v seq into an S-length buffer is wrong; use
# the math path that accepts kv-seq via a [1,SKV,...] -> here L<=S so we run the
# generic math by zero-extending K/V scores via mask. Simpler: use the
# foundation sdpa with an explicit mask over S keys where extra keys are -inf.
from serenitymojo.ops.attention import sdpa
from serenitymojo.ops.cast import cast_tensor


def _sdpa_cross[S: Int, L: Int](
    q: Tensor, k: Tensor, v: Tensor, nh: Int, dh: Int, scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    # Build padded K,V to length S and an additive mask [1,nh,S,S] masking keys >= L.
    var kpad = _pad_seq(k, S, L, nh, dh, ctx)
    var vpad = _pad_seq(v, S, L, nh, dh, ctx)
    var mask = _cross_mask(S, L, nh, ctx)
    return sdpa[1, S, 16, 128](q, kpad, vpad, mask, scale, ctx)


# Pad [1,L,nh,dh] -> [1,S,nh,dh] with zeros on the seq axis.
def _pad_seq(
    x: Tensor, s: Int, l: Int, nh: Int, dh: Int, ctx: DeviceContext
) raises -> Tensor:
    var out_n = s * nh * dh
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        out_n * x.dtype().byte_size()
    )
    # zero-init
    var src_n = l * nh * dh
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](src_n))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_n))
    var grid = (out_n + _BLOCK - 1) // _BLOCK
    var SRC = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
    )
    var DST = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )
    ctx.enqueue_function[_pad_seq_kernel, _pad_seq_kernel](
        SRC, DST, s, l, nh, dh, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var sh = [1, s, nh, dh]
    return Tensor(out_buf^, sh^, x.dtype())


def _pad_seq_kernel(
    src: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    s: Int, l: Int, nh: Int, dh: Int,
):
    var idx = Int(global_idx.x)
    var total = s * nh * dh
    if idx < total:
        var d_i = idx % dh
        var rest = idx // dh
        var head = rest % nh
        var pos = rest // nh
        if pos < l:
            var src_idx = (pos * nh + head) * dh + d_i
            dst[idx] = rebind[dst.element_type](src[src_idx])
        else:
            dst[idx] = rebind[dst.element_type](BFloat16(0.0))


# Additive mask [1,nh,S,S]: 0 for key<L, -inf (large negative) for key>=L.
def _cross_mask(s: Int, l: Int, nh: Int, ctx: DeviceContext) raises -> Tensor:
    var n = nh * s * s
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * 2)  # bf16
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var DST = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )
    ctx.enqueue_function[_cross_mask_kernel, _cross_mask_kernel](
        DST, s, l, nh, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    var sh = [1, nh, s, s]
    return Tensor(out_buf^, sh^, STDtype.BF16)


def _cross_mask_kernel(
    dst: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    s: Int, l: Int, nh: Int,
):
    var idx = Int(global_idx.x)
    var total = nh * s * s
    if idx < total:
        var key = idx % s
        if key < l:
            dst[idx] = rebind[dst.element_type](BFloat16(0.0))
        else:
            dst[idx] = rebind[dst.element_type](BFloat16(-30000.0))


# ── single DiT block forward (block 0 for the gate; generic over weights) ────
def acestep_block0_forward[S: Int, L: Int](
    hidden: Tensor,        # [1,S,H]
    temb: Tensor,          # [1,6,H]  (timestep_proj)
    enc: Tensor,           # [1,L,H]  (condition_embedder output)
    cos_b: Tensor,         # [S,Dh/2]
    sin_b: Tensor,
    bw: Dict[String, ArcPointer[Tensor]],
    cfg: AceStepDiTConfig,
    layer_type: Int,       # 0=full_attention, 1=sliding_attention
    window: Int,           # sliding_window (128)
    ctx: DeviceContext,
) raises -> Tensor:
    var h = cfg.hidden_size
    var eps = cfg.rms_norm_eps

    # modulation = scale_shift_table[1,6,H] + temb[1,6,H]; chunk6 -> [H] each
    var sst = _w(bw, "scale_shift_table", ctx)     # [1,6,H]
    var mod6 = _add(sst, temb, ctx)           # [1,6,H]
    var shift_msa = _mod_chunk(mod6, 0, h, ctx)
    var scale_msa = _mod_chunk(mod6, 1, h, ctx)
    var gate_msa  = _mod_chunk(mod6, 2, h, ctx)
    var c_shift   = _mod_chunk(mod6, 3, h, ctx)
    var c_scale   = _mod_chunk(mod6, 4, h, ctx)
    var c_gate    = _mod_chunk(mod6, 5, h, ctx)

    # 1) self-attn with AdaLN
    var x_norm = rms_norm(hidden, _w(bw, "self_attn_norm.weight", ctx), eps, ctx)
    var norm_hs = modulate(x_norm, scale_msa, shift_msa, ctx)  # (1+scale)*x+shift
    var attn = _self_attn[S](
        norm_hs, "self_attn", bw, cfg, cos_b, sin_b, layer_type, window, ctx
    )
    var x = residual_gate(hidden, gate_msa, attn, ctx)         # x + gate*attn

    # 2) cross-attn (plain residual)
    var cross_norm = rms_norm(x, _w(bw, "cross_attn_norm.weight", ctx), eps, ctx)
    var cross = _cross_attn[S, L](cross_norm, enc, "cross_attn", bw, cfg, ctx)
    x = _add(x, cross, ctx)

    # 3) SwiGLU MLP with AdaLN
    var mlp_norm = rms_norm(x, _w(bw, "mlp_norm.weight", ctx), eps, ctx)
    var mlp_in = modulate(mlp_norm, c_scale, c_shift, ctx)
    var gate = silu(linear(mlp_in, _w(bw, "mlp.gate_proj.weight", ctx), None, ctx), ctx)
    var up = linear(mlp_in, _w(bw, "mlp.up_proj.weight", ctx), None, ctx)
    var gu = _mul(gate, up, ctx)
    var mlp_out = linear(gu, _w(bw, "mlp.down_proj.weight", ctx), None, ctx)
    x = residual_gate(x, c_gate, mlp_out, ctx)
    return x^


# ── small elementwise helpers (add/mul, same-shape) ──────────────────────────
def _add(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return _ew[0](a, b, ctx)


def _mul(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    return _ew[1](a, b, ctx)


def _ew[OP: Int](a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    var n = a.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](a.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var A = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var B = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), rl
    )
    ctx.enqueue_function[_ew_kernel[OP], _ew_kernel[OP]](
        A, B, O, n, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, a.shape(), a.dtype())


def _ew_kernel[OP: Int](
    a: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var av = Float32(rebind[Scalar[DType.bfloat16]](a[idx]))
        var bv = Float32(rebind[Scalar[DType.bfloat16]](b[idx]))
        var r: Float32
        comptime if OP == 0:
            r = av + bv
        else:
            r = av * bv
        o[idx] = rebind[o.element_type](BFloat16(r))


# ── add a per-channel bias [D] to x [...,D] (broadcast over rows) ─────────────
def _add_bias(x: Tensor, bias: Tensor, ctx: DeviceContext) raises -> Tensor:
    var xshape = x.shape()
    var d = xshape[len(xshape) - 1]
    var rows = x.numel() // d
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](rows * d))
    var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d))
    var grid = (rows * d + _BLOCK - 1) // _BLOCK
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var Bv = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        bias.buf.unsafe_ptr().bitcast[BFloat16](), b_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    ctx.enqueue_function[_add_bias_kernel, _add_bias_kernel](
        X, Bv, O, rows * d, d, grid_dim=grid, block_dim=_BLOCK
    )
    ctx.synchronize()
    return Tensor(out_buf^, x.shape(), x.dtype())


def _add_bias_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    n: Int, d: Int,
):
    var idx = Int(global_idx.x)
    if idx < n:
        var c = idx % d
        var xv = Float32(rebind[Scalar[DType.bfloat16]](x[idx]))
        var bv = Float32(rebind[Scalar[DType.bfloat16]](b[c]))
        o[idx] = rebind[o.element_type](BFloat16(xv + bv))


# ── timestep embedding (sinusoidal cos-first, scale 1000) + MLP + 6-way proj ──
# Returns timestep_proj [1,6,H]. (temb [1,H] is unused for the layer modulation;
# the FINAL AdaLN needs temb, returned separately.) Matches TimestepEmbedding.
def _time_embed(
    t_val: Float32, prefix: String,
    bw: Dict[String, ArcPointer[Tensor]], cfg: AceStepDiTConfig,
    ctx: DeviceContext,
    mut out_temb: Tensor, mut out_proj: Tensor,
) raises:
    var h = cfg.hidden_size
    # sinusoidal: dim=256, t scaled by 1000, max_period=10000 (cos-first)
    var t_scaled = List[Float32]()
    t_scaled.append(t_val * 1000.0)
    var t_t = Tensor.from_host(t_scaled, [1], STDtype.F32, ctx)
    var emb = timestep_embedding(
        t_t, 256, ctx, Float32(10000.0), STDtype.BF16
    )
    # linear_1 -> silu -> linear_2
    var l1 = linear(emb, _w(bw, prefix + ".linear_1.weight", ctx),
                    Optional[Tensor](_w(bw, prefix + ".linear_1.bias", ctx)), ctx)
    var a1 = silu(l1, ctx)
    var temb = linear(a1, _w(bw, prefix + ".linear_2.weight", ctx),
                      Optional[Tensor](_w(bw, prefix + ".linear_2.bias", ctx)), ctx)  # [1,H]
    # time_proj(silu(temb)) -> [1,6H] -> [1,6,H]
    var a2 = silu(temb, ctx)
    var proj = linear(a2, _w(bw, prefix + ".time_proj.weight", ctx),
                      Optional[Tensor](_w(bw, prefix + ".time_proj.bias", ctx)), ctx)  # [1,6H]
    var proj6 = reshape(proj, [1, 6, h], ctx)
    out_temb = temb^
    out_proj = proj6^


# ── Conv1d (k=s=patch): [1,T,Cin] -> [1,T/ps, Cout] ──────────────────────────
# Reshape [1,T,Cin]->[1,T/ps, ps*Cin]; weight [Cout,Cin,ps] permute to
# [Cout, ps, Cin]->[Cout, ps*Cin]; matmul via linear (weight [out,in]).
def _conv1d_patch(
    x: Tensor, weight: Tensor, bias: Tensor, t: Int, cin: Int, cout: Int,
    ps: Int, ctx: DeviceContext,
) raises -> Tensor:
    var tout = t // ps
    var xr = reshape(x, [1, tout, ps * cin], ctx)
    var wperm = transpose(weight, 1, 2, ctx)            # [Cout, ps, Cin]
    var w2d = reshape(wperm, [cout, ps * cin], ctx)     # [Cout, ps*Cin] = [out,in]
    var out = linear(xr, w2d, None, ctx)                # [1,tout,Cout]
    return _add_bias(out, bias, ctx)


# ── ConvTranspose1d (k=s=patch): [1,T,Cin] -> [1,T*ps, Cout] ─────────────────
# weight [Cin, Cout, ps] permute to [Cin, ps, Cout]->[Cin, ps*Cout]; matmul
# (need weight as [out,in] for linear => out=ps*Cout, in=Cin => transpose).
def _conv_transpose1d_patch(
    x: Tensor, weight: Tensor, bias: Tensor, t: Int, cin: Int, cout: Int,
    ps: Int, ctx: DeviceContext,
) raises -> Tensor:
    var wperm = transpose(weight, 1, 2, ctx)            # [Cin, ps, Cout]
    var w2d = reshape(wperm, [cin, ps * cout], ctx)     # [Cin, ps*Cout]
    var w2d_t = transpose(w2d, 0, 1, ctx)               # [ps*Cout, Cin] = [out,in]
    var out = linear(x, w2d_t, None, ctx)               # [1,T, ps*Cout]
    var outr = reshape(out, [1, t * ps, cout], ctx)     # [1,T*ps,Cout]
    return _add_bias(outr, bias, ctx)


# ── build rope cos/sin tables [S, Dh/2] (BF16) for 1-D positions ─────────────
def _build_rope(
    s: Int, dh: Int, theta: Float64, ctx: DeviceContext,
    mut out_cos: Tensor, mut out_sin: Tensor,
) raises:
    var half = dh // 2
    var cos_v = List[Float32]()
    var sin_v = List[Float32]()
    for pos in range(s):
        for i in range(half):
            var inv = 1.0 / (theta ** (Float64(2 * i) / Float64(dh)))
            var ang = Float64(pos) * inv
            cos_v.append(Float32(_cos64(ang)))
            sin_v.append(Float32(_sin64(ang)))
    out_cos = Tensor.from_host(cos_v, [s, half], STDtype.BF16, ctx)
    out_sin = Tensor.from_host(sin_v, [s, half], STDtype.BF16, ctx)


# ── load one layer's weights into an unprefixed dict for acestep_block0_forward
def _layer_bw(
    full: Dict[String, ArcPointer[Tensor]], layer: Int, ctx: DeviceContext
) raises -> Dict[String, ArcPointer[Tensor]]:
    var bw = Dict[String, ArcPointer[Tensor]]()
    var pfx = String("decoder.layers.") + String(layer) + "."
    var suffixes = [
        "scale_shift_table",
        "self_attn_norm.weight", "cross_attn_norm.weight", "mlp_norm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "cross_attn.q_proj.weight", "cross_attn.k_proj.weight",
        "cross_attn.v_proj.weight", "cross_attn.o_proj.weight",
        "cross_attn.q_norm.weight", "cross_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    ]
    for sfx in suffixes:
        var s = String(sfx)
        bw[s] = ArcPointer(_clone(full[pfx + s][], ctx))
    return bw^


# ── FULL forward: x_t [1,T,acoustic], context [1,T,128], enc_in [1,L,H],
#    timestep/timestep_r scalars. Returns velocity [1,T,acoustic].
#    SP = patched seq (T_padded/ps); must be <= 128 so masks are no-ops.
def acestep_forward[SP: Int, L: Int](
    x_t: Tensor, context: Tensor, enc_in: Tensor,
    timestep: Float32, timestep_r: Float32,
    full: Dict[String, ArcPointer[Tensor]], cfg: AceStepDiTConfig,
    ctx: DeviceContext,
) raises -> Tensor:
    var h = cfg.hidden_size
    var eps = cfg.rms_norm_eps
    var ps = cfg.patch_size
    var t = x_t.shape()[1]
    var acoustic = cfg.acoustic_dim

    # timestep embeddings: temb = temb_t + temb_r ; proj = proj_t + proj_r
    var dummy1 = List[Float32]()
    dummy1.append(0.0)
    var temb_t = Tensor.from_host(dummy1, [1], STDtype.BF16, ctx)
    var proj_t = _clone(temb_t, ctx)
    var temb_r = _clone(temb_t, ctx)
    var proj_r = _clone(temb_t, ctx)
    _time_embed(timestep, "decoder.time_embed", full, cfg, ctx, temb_t, proj_t)
    _time_embed(
        timestep - timestep_r, "decoder.time_embed_r", full, cfg, ctx,
        temb_r, proj_r,
    )
    var temb = _add(temb_t, temb_r, ctx)            # [1,H]
    var timestep_proj = _add(proj_t, proj_r, ctx)   # [1,6,H]

    # cat(context, x_t) on last dim -> [1,T,192]; T even here (no pad needed)
    var xin = concat(2, ctx, context, x_t)         # [1,T,in_channels]
    # proj_in conv1d -> [1,T/ps,H]
    var x = _conv1d_patch(xin, _w(full, "decoder.proj_in.1.weight", ctx),
                          _w(full, "decoder.proj_in.1.bias", ctx),
                          t, cfg.in_channels, h, ps, ctx)

    # condition_embedder: linear + bias -> [1,L,H]
    var enc = linear(enc_in, _w(full, "decoder.condition_embedder.weight", ctx),
                     Optional[Tensor](_w(full, "decoder.condition_embedder.bias", ctx)), ctx)

    # rope tables for patched seq
    var rope_cos = _clone(temb_t, ctx)
    var rope_sin = _clone(temb_t, ctx)
    _build_rope(SP, cfg.head_dim, cfg.rope_theta, ctx, rope_cos, rope_sin)

    # 24 layers (reuse the verified block forward). layer_types alternate:
    # canonical `("sliding" if (i+1)%2 else "full")` -> sliding when i is even.
    var window = cfg.sliding_window  # 128 (canonical config default)
    for li in range(cfg.num_layers):
        var lbw = _layer_bw(full, li, ctx)
        var ltype = 1 if ((li + 1) % 2 == 1) else 0  # 1=sliding, 0=full
        x = acestep_block0_forward[SP, L](
            x, timestep_proj, enc, rope_cos, rope_sin, lbw, cfg,
            ltype, window, ctx
        )

    # final AdaLN: shift,scale = (scale_shift_table[1,2,H] + temb.unsqueeze(1)).chunk2
    var out_sst = _w(full, "decoder.scale_shift_table", ctx)   # [1,2,H]
    var temb3 = reshape(temb, [1, 1, h], ctx)
    # broadcast temb over the 2 rows then add: build [1,2,H] = sst + temb
    var temb2 = concat(1, ctx, temb3, temb3)                    # [1,2,H]
    var sst_t = _add(out_sst, temb2, ctx)                       # [1,2,H]
    var shift = _mod_chunk(sst_t, 0, h, ctx)
    var scale = _mod_chunk(sst_t, 1, h, ctx)
    var xn = rms_norm(x, _w(full, "decoder.norm_out.weight", ctx), eps, ctx)
    var xmod = modulate(xn, scale, shift, ctx)                  # (1+scale)x+shift

    # proj_out conv_transpose1d -> [1, SP*ps, acoustic]; crop to original T
    var xout = _conv_transpose1d_patch(
        xmod, _w(full, "decoder.proj_out.1.weight", ctx),
        _w(full, "decoder.proj_out.1.bias", ctx),
        SP, h, acoustic, ps, ctx
    )
    if xout.shape()[1] > t:
        return slice(xout, 1, 0, t, ctx)
    return xout^
