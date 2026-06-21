# models/dit/ideogram4_dit.mojo — Ideogram-4 DiT (pure Mojo, inference).
# 1:1 port of /home/alex/ideogram4-ref/src/ideogram4/modeling_ideogram4.py.
# Reuses foundation ops; fp8 weights are dequantized to BF16 at load.
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.math import cos as fcos, sin as fsin, exp, log, floor
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.norm import rms_norm, layer_norm_no_affine
from serenitymojo.ops.unary import tanh_op
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_flash import sdpa_flash_train_fwd
from serenitymojo.ops.tensor_algebra import mul, add, add_scalar, reshape, slice, gather_rows
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.fp8 import load_fp8_dequant
from serenitymojo.autograd_v2.step_slab import StepSlab

comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime IDEOGRAM4_SDPA_FLASH = True


def ideogram4_sdpa_product_fwd[
    B: Int, S: Int, H: Int, Dh: Int
](
    q: Tensor, k: Tensor, v: Tensor,
    scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime if IDEOGRAM4_SDPA_FLASH:
        # Ideogram4 product/runtime attention is BF16 at the model boundary.
        # Dh=256 is forward-only gated in sdpa_flash_parity; no backward claim.
        if q.dtype() == STDtype.BF16 and k.dtype() == STDtype.BF16 and v.dtype() == STDtype.BF16:
            var ff = sdpa_flash_train_fwd[B, S, H, Dh](q, k, v, scale, ctx)
            return ff.o.clone(ctx)
        var q_bf16 = cast_tensor(q, STDtype.BF16, ctx, False)
        var k_bf16 = cast_tensor(k, STDtype.BF16, ctx, False)
        var v_bf16 = cast_tensor(v, STDtype.BF16, ctx, False)
        var ff = sdpa_flash_train_fwd[B, S, H, Dh](q_bf16, k_bf16, v_bf16, scale, ctx)
        var out = ff.o.clone(ctx)
        return cast_tensor(out, q.dtype(), ctx, False)
    else:
        return sdpa_nomask[B, S, H, Dh](q, k, v, scale, ctx)


# ── weight load helpers ──────────────────────────────────────────────────────
def load_w_fp8(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    """Linear .weight: F8_E4M3 + per-row .weight_scale -> BF16 [out,in]."""
    return load_fp8_dequant(st, name, ctx)


def load_w_bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    """BF16 tensor (bias / norm / embedding) loaded preserving dtype."""
    return Tensor.from_view(st.tensor_view(name), ctx)


# ── Ideogram4EmbedScalar sinusoidal (sin-first, scale 1e4, /(half-1)) ─────────
# Reference _sinusoidal_embedding (modeling_ideogram4.py:218-229) + EmbedScalar
# pre-scale (241-245): scaled = 1e4*(x-0)/(1-0); freq_i = exp(-i*ln(1e4)/(half-1));
# emb = [sin(scaled*freq), cos(scaled*freq)]; cast to bf16.
def _embedscalar_sinusoid_kernel(
    t: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    dim: Int,
    half: Int,
    log_scale_over_halfm1: Float32,  # ln(1e4)/(half-1)
    prescale: Float32,               # 1e4
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx >= n:
        return
    var row = idx // dim
    var d = idx % dim
    var i = d if d < half else d - half
    var tv = rebind[Scalar[DType.float32]](t[row]) * prescale
    var freq = exp(-Float32(i) * log_scale_over_halfm1)
    var angle_f32 = tv * freq
    comptime TWO_PI = Float64(6.283185307179586476925286766559)
    var a = Float64(angle_f32)
    var k = floor(a / TWO_PI + 0.5)
    var reduced = Float32(a - k * TWO_PI)
    var v = fsin(reduced) if d < half else fcos(reduced)
    o[idx] = rebind[o.element_type](v.cast[DType.bfloat16]())


def ideogram4_embedscalar_sinusoid(
    t: Tensor, dim: Int, ctx: DeviceContext
) raises -> Tensor:
    """t: F32 [N] -> BF16 [N, dim] sinusoidal (sin-first, scale 1e4)."""
    var N = t.numel()
    var half = dim // 2
    var n = N * dim
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * STDtype.BF16.byte_size())
    var t_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](N))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](t.buf.unsafe_ptr().bitcast[Float32](), t_rl)
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl)
    var lsf = Float32(log(Float64(10000.0)) / Float64(half - 1))
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_embedscalar_sinusoid_kernel, _embedscalar_sinusoid_kernel](
        T, O, dim, half, lsf, Float32(10000.0), n, grid_dim=grid, block_dim=_BLOCK)
    # No synchronize: single-stream ordering; downstream same-stream reads see
    # the result, host syncs only at .to_host(). (was a per-op pipeline drain)
    var out_shape = [N, dim]
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


# ── t_embedding: EmbedScalar -> Linear -> SiLU -> Linear (modeling 241-250) ───
def ideogram4_t_embedding(
    t: Tensor,
    dim: Int,
    mlp_in_w: Tensor, mlp_in_b: Tensor,
    mlp_out_w: Tensor, mlp_out_b: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var emb = ideogram4_embedscalar_sinusoid(t, dim, ctx)
    var h = linear(emb, mlp_in_w, Optional[Tensor](mlp_in_b.clone(ctx)), ctx)
    var a = silu(h, ctx)
    return linear(a, mlp_out_w, Optional[Tensor](mlp_out_b.clone(ctx)), ctx)


# ── Ideogram halfsplit RoPE applied per (b,l,h) with token-level cos/sin ──────
# Reference _apply_rotary_pos_emb + _rotate_half (modeling 44-62): q*cos +
# rotate_half(q)*sin, cos/sin full-width [.,L,Dh] (duplicated halves). For d<half:
# out=x*cos[d]-x[d+half]*sin[d]; d>=half: out=x*cos[d]+x[d-half]*sin[d].
def _rope_kernel[dt: DType](
    x: LayoutTensor[dt, _DYN1, MutAnyOrigin],
    cosx: LayoutTensor[dt, _DYN1, MutAnyOrigin],
    sinx: LayoutTensor[dt, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dt, _DYN1, MutAnyOrigin],
    L: Int, H: Int, Dh: Int, half: Int, n: Int,
):
    var idx = Int(global_idx.x)
    if idx >= n:
        return
    var d = idx % Dh
    var tok_h = idx // Dh
    var l = (tok_h // H) % L
    var cbase = l * Dh
    var xv = Float32(rebind[Scalar[dt]](x[idx]))
    var c = Float32(rebind[Scalar[dt]](cosx[cbase + d]))
    var s = Float32(rebind[Scalar[dt]](sinx[cbase + d]))
    var partner: Float32
    if d < half:
        partner = Float32(rebind[Scalar[dt]](x[idx + half]))
        o[idx] = rebind[o.element_type]((xv * c - partner * s).cast[dt]())
    else:
        partner = Float32(rebind[Scalar[dt]](x[idx - half]))
        o[idx] = rebind[o.element_type]((xv * c + partner * s).cast[dt]())


def apply_rope_ideogram(x: Tensor, cosf: Tensor, sinf: Tensor, ctx: DeviceContext) raises -> Tensor:
    """x: [B,L,H,Dh] bf16, cos/sin: [1,L,Dh] bf16 (B=1). Returns [B,L,H,Dh]."""
    var sh = x.shape()
    var L = sh[1]; var H = sh[2]; var Dh = sh[3]
    var half = Dh // 2
    var n = x.numel()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n * STDtype.BF16.byte_size())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var c_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](L * Dh))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
    var C = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](cosf.buf.unsafe_ptr().bitcast[BFloat16](), c_rl)
    var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](sinf.buf.unsafe_ptr().bitcast[BFloat16](), c_rl)
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_rope_kernel[DType.bfloat16], _rope_kernel[DType.bfloat16]](
        X, C, S, O, L, H, Dh, half, n, grid_dim=grid, block_dim=_BLOCK)
    # No synchronize: single-stream ordering; this rope output feeds SDPA on the
    # same stream. Per-call sync here drained the pipeline 2x/layer (q,k).
    var os = sh.copy()
    return Tensor(out_buf^, os^, STDtype.BF16)


def apply_rope_ideogram_slab(x: Tensor, cosf: Tensor, sinf: Tensor, ctx: DeviceContext, mut slab: StepSlab) raises -> Tensor:
    """StepSlab (contract C8) variant of `apply_rope_ideogram`: byte-identical
    except the output buffer comes from the step slab instead of the MAX pool."""
    var sh = x.shape()
    var L = sh[1]; var H = sh[2]; var Dh = sh[3]
    var half = Dh // 2
    var n = x.numel()
    var out_buf = slab.alloc(n * STDtype.BF16.byte_size())
    var x_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var c_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](L * Dh))
    var X = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
    var C = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](cosf.buf.unsafe_ptr().bitcast[BFloat16](), c_rl)
    var S = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](sinf.buf.unsafe_ptr().bitcast[BFloat16](), c_rl)
    var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl)
    var grid = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_rope_kernel[DType.bfloat16], _rope_kernel[DType.bfloat16]](
        X, C, S, O, L, H, Dh, half, n, grid_dim=grid, block_dim=_BLOCK)
    # No synchronize: single-stream ordering; this rope output feeds SDPA on the
    # same stream. Per-call sync here drained the pipeline 2x/layer (q,k).
    var os = sh.copy()
    return Tensor(out_buf^, os^, STDtype.BF16)


# ── attention (fused qkv, per-head q/k RMSNorm, rope, SDPA-nomask, o) ─────────
def ideogram4_attention[S: Int](
    x: Tensor,                      # [1,L,hidden] bf16
    qkv_w: Tensor, o_w: Tensor,
    normq_w: Tensor, normk_w: Tensor,
    cosf: Tensor, sinf: Tensor,
    num_heads: Int, head_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var sh = x.shape()
    var L = sh[1]
    var hidden = sh[2]
    var qkv = linear(x, qkv_w, None, ctx)                # [1,L,3*hidden]
    var qkv5 = reshape(qkv, [1, L, 3, num_heads, head_dim], ctx)
    var q = reshape(slice(qkv5, 2, 0, 1, ctx), [1, L, num_heads, head_dim], ctx)
    var k = reshape(slice(qkv5, 2, 1, 1, ctx), [1, L, num_heads, head_dim], ctx)
    var v = reshape(slice(qkv5, 2, 2, 1, ctx), [1, L, num_heads, head_dim], ctx)
    q = rms_norm(q, normq_w, Float32(1.0e-5), ctx)       # over head_dim
    k = rms_norm(k, normk_w, Float32(1.0e-5), ctx)
    q = apply_rope_ideogram(q, cosf, sinf, ctx)
    k = apply_rope_ideogram(k, cosf, sinf, ctx)
    var scale = Float32(1.0 / (Float32(head_dim) ** 0.5))
    var attn = ideogram4_sdpa_product_fwd[1, S, 18, 256](q, k, v, scale, ctx)  # [1,L,H,Dh]
    var merged = reshape(attn, [1, L, hidden], ctx)
    return linear(merged, o_w, None, ctx)


# ── transformer block (modeling 192-215) ─────────────────────────────────────
def ideogram4_block[S: Int](
    x: Tensor,                      # [1,L,hidden]
    adaln_input: Tensor,            # [1,1,adaln_dim]
    cosf: Tensor, sinf: Tensor,
    adaln_mod_w: Tensor, adaln_mod_b: Tensor,
    an1_w: Tensor, an2_w: Tensor, fn1_w: Tensor, fn2_w: Tensor,
    qkv_w: Tensor, o_w: Tensor, normq_w: Tensor, normk_w: Tensor,
    w1_w: Tensor, w2_w: Tensor, w3_w: Tensor,
    num_heads: Int, head_dim: Int, hidden: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var mod = linear(adaln_input, adaln_mod_w, Optional[Tensor](adaln_mod_b.clone(ctx)), ctx)  # [1,1,4*hidden]
    var scale_msa = add_scalar(slice(mod, 2, 0 * hidden, hidden, ctx), Float32(1.0), ctx)
    var gate_msa = tanh_op(slice(mod, 2, 1 * hidden, hidden, ctx), ctx)
    var scale_mlp = add_scalar(slice(mod, 2, 2 * hidden, hidden, ctx), Float32(1.0), ctx)
    var gate_mlp = tanh_op(slice(mod, 2, 3 * hidden, hidden, ctx), ctx)

    var an1 = rms_norm(x, an1_w, Float32(1.0e-5), ctx)
    var attn_in = mul(an1, scale_msa, ctx)               # broadcast [1,1,H]*[1,L,H]
    var attn_out = ideogram4_attention[S](attn_in, qkv_w, o_w, normq_w, normk_w, cosf, sinf, num_heads, head_dim, ctx)
    var an2 = rms_norm(attn_out, an2_w, Float32(1.0e-5), ctx)
    var x1 = add(x, mul(gate_msa, an2, ctx), ctx)

    var fn1 = rms_norm(x1, fn1_w, Float32(1.0e-5), ctx)
    var mlp_in = mul(fn1, scale_mlp, ctx)
    var g = linear(mlp_in, w1_w, None, ctx)              # feed_forward.w1
    var u = linear(mlp_in, w3_w, None, ctx)              # feed_forward.w3
    var act = swiglu(g, u, ctx)                          # silu(w1 x)*w3 x
    var ff = linear(act, w2_w, None, ctx)               # feed_forward.w2
    var fn2 = rms_norm(ff, fn2_w, Float32(1.0e-5), ctx)
    return add(x1, mul(gate_mlp, fn2, ctx), ctx)


# ── full Ideogram-4 DiT forward (inference). modeling forward 311-379. ────────
# Loads weights from `st` (fp8->bf16), blocks loaded per-layer to bound VRAM.
# Returns the tensor immediately before final_layer.linear:
#   LayerNorm(no-affine,1e-6)(h) * (1 + final_layer.adaln_modulation(silu(c)))
# This is the frozen-trunk seam used by onetrainer-mojo's Ideogram4 final-linear
# LoRA trainer.
def ideogram4_forward_prefinal_hidden[S: Int](
    st: ShardedSafeTensors,
    x_in: Tensor,            # [1,L,128] bf16   (noise tokens)
    llm_in: Tensor,          # [1,L,53248] bf16 (Qwen features)
    t_in: Tensor,            # [1] f32
    indicator: Tensor,       # [1,L] f32 (values 0/2/3)
    cosf: Tensor, sinf: Tensor,   # [1,L,256] bf16
    num_layers: Int, num_heads: Int, head_dim: Int, hidden: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var L = x_in.shape()[1]

    # masks from indicator (host-built): llm=3, image=2
    var ind_h = indicator.to_host(ctx)
    var llm_mask_v = List[Float32]()
    var img_mask_v = List[Float32]()
    var img_ids = List[Int]()
    for i in range(L):
        var vi = ind_h[i]
        llm_mask_v.append(Float32(1.0) if (vi > 2.5 and vi < 3.5) else Float32(0.0))
        var is_img = (vi > 1.5 and vi < 2.5)
        img_mask_v.append(Float32(1.0) if is_img else Float32(0.0))
        img_ids.append(1 if is_img else 0)
    var llm_mask = Tensor.from_host(llm_mask_v, [1, L, 1], STDtype.BF16, ctx)
    var img_mask = Tensor.from_host(img_mask_v, [1, L, 1], STDtype.BF16, ctx)

    var llm = mul(llm_in, llm_mask, ctx)                      # zero non-text
    var x = mul(x_in, img_mask, ctx)
    var ipw = load_w_fp8(st, "input_proj.weight", ctx)
    var ipb = load_w_bf16(st, "input_proj.bias", ctx)
    x = mul(linear(x, ipw, Optional[Tensor](ipb.clone(ctx)), ctx), img_mask, ctx)

    # t -> adaln_input
    var miw = load_w_fp8(st, "t_embedding.mlp_in.weight", ctx)
    var mib = load_w_bf16(st, "t_embedding.mlp_in.bias", ctx)
    var mow = load_w_fp8(st, "t_embedding.mlp_out.weight", ctx)
    var mob = load_w_bf16(st, "t_embedding.mlp_out.bias", ctx)
    var t_cond = reshape(ideogram4_t_embedding(t_in, hidden, miw, mib, mow, mob, ctx), [1, 1, hidden], ctx)
    var apw = load_w_fp8(st, "adaln_proj.weight", ctx)
    var apb = load_w_bf16(st, "adaln_proj.bias", ctx)
    var adaln_input = silu(linear(t_cond, apw, Optional[Tensor](apb.clone(ctx)), ctx), ctx)  # [1,1,512]

    # llm conditioning: RMSNorm(eps 1e-6) -> proj -> mask
    var lcn = load_w_bf16(st, "llm_cond_norm.weight", ctx)
    llm = rms_norm(llm, lcn, Float32(1.0e-6), ctx)
    var lcpw = load_w_fp8(st, "llm_cond_proj.weight", ctx)
    var lcpb = load_w_bf16(st, "llm_cond_proj.bias", ctx)
    llm = mul(linear(llm, lcpw, Optional[Tensor](lcpb.clone(ctx)), ctx), llm_mask, ctx)

    var h = add(x, llm, ctx)
    # image-indicator embedding
    var eii = load_w_bf16(st, "embed_image_indicator.weight", ctx)  # [2,hidden]
    var iemb = reshape(gather_rows(eii, img_ids, ctx), [1, L, hidden], ctx)
    h = add(h, iemb, ctx)

    # 34 blocks, loaded per-layer
    for li in range(num_layers):
        var p = String("layers.") + String(li) + "."
        var amw = load_w_fp8(st, p + "adaln_modulation.weight", ctx)
        var amb = load_w_bf16(st, p + "adaln_modulation.bias", ctx)
        var an1 = load_w_bf16(st, p + "attention_norm1.weight", ctx)
        var an2 = load_w_bf16(st, p + "attention_norm2.weight", ctx)
        var fn1 = load_w_bf16(st, p + "ffn_norm1.weight", ctx)
        var fn2 = load_w_bf16(st, p + "ffn_norm2.weight", ctx)
        var qkv = load_w_fp8(st, p + "attention.qkv.weight", ctx)
        var ow = load_w_fp8(st, p + "attention.o.weight", ctx)
        var nq = load_w_bf16(st, p + "attention.norm_q.weight", ctx)
        var nk = load_w_bf16(st, p + "attention.norm_k.weight", ctx)
        var w1 = load_w_fp8(st, p + "feed_forward.w1.weight", ctx)
        var w2 = load_w_fp8(st, p + "feed_forward.w2.weight", ctx)
        var w3 = load_w_fp8(st, p + "feed_forward.w3.weight", ctx)
        h = ideogram4_block[S](
            h, adaln_input, cosf, sinf, amw, amb, an1, an2, fn1, fn2,
            qkv, ow, nq, nk, w1, w2, w3, num_heads, head_dim, hidden, ctx)

    # final layer: LayerNorm(no-affine,1e-6) * (1 + adaln_mod(silu(c))) -> linear
    var fmw = load_w_fp8(st, "final_layer.adaln_modulation.weight", ctx)
    var fmb = load_w_bf16(st, "final_layer.adaln_modulation.bias", ctx)
    var fscale = add_scalar(linear(silu(adaln_input, ctx), fmw, Optional[Tensor](fmb.clone(ctx)), ctx), Float32(1.0), ctx)
    var hn = mul(layer_norm_no_affine(h, Float32(1.0e-6), ctx), fscale, ctx)
    return hn^


def ideogram4_forward[S: Int](
    st: ShardedSafeTensors,
    x_in: Tensor,            # [1,L,128] bf16   (noise tokens)
    llm_in: Tensor,          # [1,L,53248] bf16 (Qwen features)
    t_in: Tensor,            # [1] f32
    indicator: Tensor,       # [1,L] f32 (values 0/2/3)
    cosf: Tensor, sinf: Tensor,   # [1,L,256] bf16
    num_layers: Int, num_heads: Int, head_dim: Int, hidden: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var hn = ideogram4_forward_prefinal_hidden[S](
        st, x_in, llm_in, t_in, indicator, cosf, sinf,
        num_layers, num_heads, head_dim, hidden, ctx,
    )
    var flw = load_w_fp8(st, "final_layer.linear.weight", ctx)
    var flb = load_w_bf16(st, "final_layer.linear.bias", ctx)
    var out = linear(hn, flw, Optional[Tensor](flb.clone(ctx)), ctx)  # [1,L,128] bf16
    return cast_tensor(out, STDtype.F32, ctx)
