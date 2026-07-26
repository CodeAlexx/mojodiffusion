# Pure-Mojo, layer-streamed Gemma-3-12B text encoder for LTX-2.
#
# Product contract:
#   * no Python/PyTorch/Rust inference dependency;
#   * tokenizer IDs are supplied by tokenizer.Qwen3Tokenizer.encode_gemma;
#   * FP8 E4M3 Gemma linears are dequantized one layer at a time;
#   * both positive and negative prompts advance while a layer is resident;
#   * all 49 Hugging Face hidden states are retained (48 pre-layer states plus
#     the final Gemma RMS-normalized state);
#   * compact valid-prefix attention buckets preserve the reference LEFT-pad
#     position IDs by starting RoPE at (1024 - real_len).
#
# Architecture (Gemma3TextConfig for the installed 12B tower):
#   hidden=3840, intermediate=15360, layers=48, heads=16, kv_heads=8,
#   head_dim=256, eps=1e-6, local theta=10000, global theta=1e6 with linear
#   factor 8, and every sixth layer global.  The 1024-token local window spans
#   the complete LTX prompt window, so both layer types share the same causal
#   valid-prefix attention graph and differ only in RoPE.

from std.gpu import barrier, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.math import cos as fcos, exp as fexp, log as flog, sin as fsin, sqrt
from std.memory import ArcPointer, stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.text_encoder.qwen3_encoder import (
    Qwen3Config,
    Qwen3Encoder,
    _add,
    _repeat_kv,
    _reshape,
)
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.attention_flash import sdpa_flash_infer_fwd_causal_padmask
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_to_bf16
from serenitymojo.ops.linear import linear
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.tensor_algebra import mul, mul_scalar
from serenitymojo.tensor import Tensor


comptime TArc = ArcPointer[Tensor]
comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _TPB = 256
comptime GEMMA_HIDDEN = 3840
comptime GEMMA_LAYERS = 48
comptime GEMMA_HEADS = 16
comptime GEMMA_KV_HEADS = 8
comptime GEMMA_HEAD_DIM = 256
comptime GEMMA_MAX_TOKENS = 1024


@fieldwise_init
struct GemmaRope(Copyable, Movable):
    var local_cos_q: TArc
    var local_sin_q: TArc
    var local_cos_k: TArc
    var local_sin_k: TArc
    var global_cos_q: TArc
    var global_sin_q: TArc
    var global_cos_k: TArc
    var global_sin_k: TArc


@fieldwise_init
struct GemmaLayerWeights(Movable):
    var input_ln: Tensor
    var post_attention_ln: Tensor
    var pre_feedforward_ln: Tensor
    var post_feedforward_ln: Tensor
    var q_norm: Tensor
    var k_norm: Tensor
    var q_proj: Tensor
    var k_proj: Tensor
    var v_proj: Tensor
    var o_proj: Tensor
    var gate_proj: Tensor
    var up_proj: Tensor
    var down_proj: Tensor


@fieldwise_init
struct GemmaHiddenBatch(Movable):
    # states[p] contains the exact 49-state HF output_hidden_states ordering.
    var states: List[List[TArc]]
    var lengths: List[Int]
    var bucket: Int


def _alias(x: Tensor) -> Tensor:
    return Tensor(x.buf.copy(), x.shape(), x.dtype())


def _load_bf16(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var tv = st.tensor_view(name)
    if tv.dtype == STDtype.F8_E4M3:
        var raw = Tensor.from_view_raw(tv, ctx)
        return fp8_e4m3_dequant_to_bf16(raw, Float32(1.0), ctx)
    return Tensor.from_view_as_bf16(tv, ctx)


def _load_layer(
    st: ShardedSafeTensors, layer_idx: Int, ctx: DeviceContext
) raises -> GemmaLayerWeights:
    var p = String("model.layers.") + String(layer_idx) + "."
    return GemmaLayerWeights(
        _load_bf16(st, p + "input_layernorm.weight", ctx),
        _load_bf16(st, p + "post_attention_layernorm.weight", ctx),
        _load_bf16(st, p + "pre_feedforward_layernorm.weight", ctx),
        _load_bf16(st, p + "post_feedforward_layernorm.weight", ctx),
        _load_bf16(st, p + "self_attn.q_norm.weight", ctx),
        _load_bf16(st, p + "self_attn.k_norm.weight", ctx),
        _load_bf16(st, p + "self_attn.q_proj.weight", ctx),
        _load_bf16(st, p + "self_attn.k_proj.weight", ctx),
        _load_bf16(st, p + "self_attn.v_proj.weight", ctx),
        _load_bf16(st, p + "self_attn.o_proj.weight", ctx),
        _load_bf16(st, p + "mlp.gate_proj.weight", ctx),
        _load_bf16(st, p + "mlp.up_proj.weight", ctx),
        _load_bf16(st, p + "mlp.down_proj.weight", ctx),
    )


# Gemma RMSNorm is not the Llama/Qwen formula implemented by ops.rms_norm.
# The checkpoint stores a zero-centered delta and HF computes in F32:
#   y = (x * rsqrt(mean(x^2)+eps)) * (1 + weight.float()), then casts to BF16.
def _gemma_rms_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    w: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    cols: Int,
    eps: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local: Float32 = 0.0
    var c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        local += v * v
        c += _TPB
    shared[tid] = local
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var inv = 1.0 / sqrt(shared[0] / Float32(cols) + eps)
    c = tid
    while c < cols:
        var v = rebind[Scalar[DType.bfloat16]](x[row, c]).cast[DType.float32]()
        var ww = rebind[Scalar[DType.bfloat16]](w[c]).cast[DType.float32]()
        o[row, c] = rebind[o.element_type](
            (v * inv * (Float32(1.0) + ww)).cast[DType.bfloat16]()
        )
        c += _TPB


def gemma_rms_norm(
    x: Tensor, weight: Tensor, ctx: DeviceContext
) raises -> Tensor:
    if x.dtype() != STDtype.BF16 or weight.dtype() != STDtype.BF16:
        raise Error("gemma_rms_norm: x and weight must be BF16")
    var shape = x.shape()
    var cols = shape[len(shape) - 1]
    if weight.numel() != cols:
        raise Error("gemma_rms_norm: weight width mismatch")
    var rows = x.numel() // cols
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var w_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](cols))
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var W = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        weight.buf.unsafe_ptr().bitcast[BFloat16](), w_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    ctx.enqueue_function[_gemma_rms_kernel, _gemma_rms_kernel](
        X, W, O, cols, Float32(1.0e-6),
        grid_dim=rows, block_dim=_TPB,
    )
    return Tensor(out_buf^, shape^, STDtype.BF16)


def _rope_host(
    seq: Int,
    heads: Int,
    position_offset: Int,
    theta: Float64,
    position_scale: Float32,
) raises -> List[List[Float32]]:
    var half = GEMMA_HEAD_DIM // 2
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(Float32(theta))
    for t in range(seq):
        var position = Float32(position_offset + t) * position_scale
        for _h in range(heads):
            for i in range(half):
                var exponent = -log_theta * Float32(2 * i) / Float32(GEMMA_HEAD_DIM)
                var angle = position * fexp(exponent)
                cos_vals.append(fcos(angle))
                sin_vals.append(fsin(angle))
    var out = List[List[Float32]]()
    out.append(cos_vals^)
    out.append(sin_vals^)
    return out^


def _rope_tensor(
    values: List[Float32], seq: Int, heads: Int, ctx: DeviceContext
) raises -> Tensor:
    var shape = List[Int]()
    shape.append(seq * heads * (GEMMA_HEAD_DIM // 2))
    return Tensor.from_host(values, shape^, STDtype.BF16, ctx)


def _build_rope(
    seq: Int, real_len: Int, ctx: DeviceContext
) raises -> GemmaRope:
    var offset = GEMMA_MAX_TOKENS - real_len
    var lq = _rope_host(seq, GEMMA_HEADS, offset, Float64(10000.0), Float32(1.0))
    var lk = _rope_host(seq, GEMMA_KV_HEADS, offset, Float64(10000.0), Float32(1.0))
    # Global Gemma RoPE uses theta=1e6 and linear scaling factor 8.
    var gq = _rope_host(seq, GEMMA_HEADS, offset, Float64(1000000.0), Float32(0.125))
    var gk = _rope_host(seq, GEMMA_KV_HEADS, offset, Float64(1000000.0), Float32(0.125))
    return GemmaRope(
        TArc(_rope_tensor(lq[0], seq, GEMMA_HEADS, ctx)),
        TArc(_rope_tensor(lq[1], seq, GEMMA_HEADS, ctx)),
        TArc(_rope_tensor(lk[0], seq, GEMMA_KV_HEADS, ctx)),
        TArc(_rope_tensor(lk[1], seq, GEMMA_KV_HEADS, ctx)),
        TArc(_rope_tensor(gq[0], seq, GEMMA_HEADS, ctx)),
        TArc(_rope_tensor(gq[1], seq, GEMMA_HEADS, ctx)),
        TArc(_rope_tensor(gk[0], seq, GEMMA_KV_HEADS, ctx)),
        TArc(_rope_tensor(gk[1], seq, GEMMA_KV_HEADS, ctx)),
    )


def _attention_dispatch(
    q: Tensor, k: Tensor, v: Tensor,
    real_len: Int, seq: Int, ctx: DeviceContext,
) raises -> Tensor:
    var scale = Float32(1.0 / 16.0)  # query_pre_attn_scalar(256)^-0.5
    if seq == 128:
        return sdpa_flash_infer_fwd_causal_padmask[1, 128, 16, 256](
            q, k, v, real_len, scale, ctx
        )
    if seq == 256:
        return sdpa_flash_infer_fwd_causal_padmask[1, 256, 16, 256](
            q, k, v, real_len, scale, ctx
        )
    if seq == 384:
        return sdpa_flash_infer_fwd_causal_padmask[1, 384, 16, 256](
            q, k, v, real_len, scale, ctx
        )
    if seq == 512:
        return sdpa_flash_infer_fwd_causal_padmask[1, 512, 16, 256](
            q, k, v, real_len, scale, ctx
        )
    if seq == 640:
        return sdpa_flash_infer_fwd_causal_padmask[1, 640, 16, 256](
            q, k, v, real_len, scale, ctx
        )
    if seq == 768:
        return sdpa_flash_infer_fwd_causal_padmask[1, 768, 16, 256](
            q, k, v, real_len, scale, ctx
        )
    if seq == 896:
        return sdpa_flash_infer_fwd_causal_padmask[1, 896, 16, 256](
            q, k, v, real_len, scale, ctx
        )
    if seq == 1024:
        return sdpa_flash_infer_fwd_causal_padmask[1, 1024, 16, 256](
            q, k, v, real_len, scale, ctx
        )
    raise Error("Gemma attention bucket must be 128..1024 in 128-token steps")


def _layer_forward(
    w: GemmaLayerWeights,
    hidden: Tensor,
    rope: GemmaRope,
    global_rope: Bool,
    real_len: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var seq = hidden.shape()[1]
    var residual = _alias(hidden)
    var normed = gemma_rms_norm(hidden, w.input_ln, ctx)
    var q = linear(normed, w.q_proj, None, ctx)
    var k = linear(normed, w.k_proj, None, ctx)
    var v = linear(normed, w.v_proj, None, ctx)

    var q_shape: List[Int] = [1, seq, GEMMA_HEADS, GEMMA_HEAD_DIM]
    var kv_shape: List[Int] = [1, seq, GEMMA_KV_HEADS, GEMMA_HEAD_DIM]
    q = _reshape(q, q_shape^, ctx)
    k = _reshape(k, kv_shape.copy(), ctx)
    v = _reshape(v, kv_shape^, ctx)
    q = gemma_rms_norm(q, w.q_norm, ctx)
    k = gemma_rms_norm(k, w.k_norm, ctx)
    if global_rope:
        q = rope_halfsplit(q, rope.global_cos_q[], rope.global_sin_q[], ctx)
        k = rope_halfsplit(k, rope.global_cos_k[], rope.global_sin_k[], ctx)
    else:
        q = rope_halfsplit(q, rope.local_cos_q[], rope.local_sin_q[], ctx)
        k = rope_halfsplit(k, rope.local_cos_k[], rope.local_sin_k[], ctx)

    var k_rep = _repeat_kv(k^, GEMMA_HEADS, GEMMA_KV_HEADS, ctx)
    var v_rep = _repeat_kv(v^, GEMMA_HEADS, GEMMA_KV_HEADS, ctx)
    var attn = _attention_dispatch(q, k_rep, v_rep, real_len, seq, ctx)
    var flat_shape: List[Int] = [1, seq, GEMMA_HEADS * GEMMA_HEAD_DIM]
    attn = _reshape(attn, flat_shape^, ctx)
    var attn_out = linear(attn, w.o_proj, None, ctx)
    attn_out = gemma_rms_norm(attn_out, w.post_attention_ln, ctx)
    var hidden2 = _add(residual, attn_out, ctx)

    residual = _alias(hidden2)
    normed = gemma_rms_norm(hidden2, w.pre_feedforward_ln, ctx)
    var gate = gelu(linear(normed, w.gate_proj, None, ctx), ctx)
    var up = linear(normed, w.up_proj, None, ctx)
    var ff = linear(mul(gate, up, ctx), w.down_proj, None, ctx)
    ff = gemma_rms_norm(ff, w.post_feedforward_ln, ctx)
    return _add(residual, ff, ctx)


def _bucket_for(max_len: Int) raises -> Int:
    if max_len < 1 or max_len > GEMMA_MAX_TOKENS:
        raise Error("Gemma token count must be in [1, 1024]")
    return ((max_len + 127) // 128) * 128


def encode_gemma3_hidden_states_streamed(
    checkpoint_path: String,
    ids_unpadded: List[List[Int]],
    ctx: DeviceContext,
) raises -> GemmaHiddenBatch:
    """Encode one or more prompts while each FP8 layer is resident once."""
    if len(ids_unpadded) < 1:
        raise Error("Gemma encode: no prompts")
    var max_len = 0
    var lengths = List[Int]()
    for p in range(len(ids_unpadded)):
        var n = len(ids_unpadded[p])
        if n < 1 or n > GEMMA_MAX_TOKENS:
            raise Error("Gemma encode: prompt token count outside [1, 1024]")
        lengths.append(n)
        if n > max_len:
            max_len = n
    var bucket = _bucket_for(max_len)
    var ids = List[List[Int]]()
    for p in range(len(ids_unpadded)):
        var padded = ids_unpadded[p].copy()
        while len(padded) < bucket:
            padded.append(0)
        ids.append(padded^)

    var st = ShardedSafeTensors.open(checkpoint_path)
    # Reuse the admitted embedding gather implementation with a transient
    # mini encoder holding only model.embed_tokens.weight.
    var ew = List[TArc]()
    var en = Dict[String, Int]()
    ew.append(TArc(_load_bf16(st, String("model.embed_tokens.weight"), ctx)))
    en[String("model.embed_tokens.weight")] = 0
    var ecfg = Qwen3Config(
        GEMMA_HIDDEN, GEMMA_LAYERS, GEMMA_HEADS, GEMMA_KV_HEADS,
        GEMMA_HEAD_DIM, Float32(1.0e-6), Float64(1000000.0),
    )
    var embedder = Qwen3Encoder(ew^, en^, ecfg)
    var hiddens = List[TArc]()
    for p in range(len(ids)):
        # Gemma's sqrt(3840) embedding scale rounds to exactly 62 in BF16.
        hiddens.append(TArc(mul_scalar(embedder._embed(ids[p], ctx), Float32(62.0), ctx)))
    ctx.synchronize()

    var ropes = List[GemmaRope]()
    for p in range(len(ids)):
        ropes.append(_build_rope(bucket, lengths[p], ctx))

    var states = List[List[TArc]]()
    for _p in range(len(ids)):
        states.append(List[TArc]())

    for li in range(GEMMA_LAYERS):
        for p in range(len(ids)):
            ref prompt_states = states[p]
            prompt_states.append(TArc(_alias(hiddens[p][])))
        var layer = _load_layer(st, li, ctx)
        var is_global = ((li + 1) % 6) == 0
        for p in range(len(ids)):
            hiddens[p] = TArc(_layer_forward(
                layer, hiddens[p][], ropes[p], is_global, lengths[p], ctx
            ))
        ctx.synchronize()
        print(
            String("LTX2_ACTIVITY encoding Gemma layer ")
            + String(li + 1) + String("/") + String(GEMMA_LAYERS)
        )

    var final_norm = _load_bf16(st, String("model.norm.weight"), ctx)
    for p in range(len(ids)):
        var final_hidden = gemma_rms_norm(hiddens[p][], final_norm, ctx)
        ref prompt_states = states[p]
        prompt_states.append(TArc(final_hidden^))
    ctx.synchronize()
    return GemmaHiddenBatch(states^, lengths^, bucket)
