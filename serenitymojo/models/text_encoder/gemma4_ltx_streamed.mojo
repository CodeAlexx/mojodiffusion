# Pure-Mojo, layer-streamed Gemma-4-12B text encoder for LTX-2.5.
#
# Sibling of `gemma3_ltx_streamed.mojo`; same product contract:
#   * no Python/PyTorch/Rust inference dependency;
#   * tokenizer IDs are supplied by the caller;
#   * linears are loaded (and FP8-dequantized, should a quantized checkpoint
#     ever ship) one layer at a time;
#   * both positive and negative prompts advance while a layer is resident;
#   * all 49 Hugging Face hidden states are retained (48 pre-layer states plus
#     the final Gemma RMS-normalized state);
#   * compact valid-prefix attention buckets preserve the reference LEFT-pad
#     position IDs by starting RoPE at (1024 - real_len).
#
# Architecture (Gemma4UnifiedTextConfig of the installed 12B tower — every
# figure below verified against the checkpoint config.json AND the safetensors
# header, not derived):
#   hidden=3840, intermediate=15360, layers=48, heads=16, eps=1e-6,
#   gelu_pytorch_tanh MLP, the same four-norm sandwich, sliding_window=1024,
#   and `full_attention` at layer_types indices 5,11,...,47 (i.e. (li+1)%6==0).
#   The 1024-token window spans the complete LTX prompt window, so both layer
#   types share one causal valid-prefix attention graph.
#
# Deltas vs the Gemma-3 tower (all measured, all load-bearing):
#   1. RMSNorm is a PLAIN weight multiply, `x*rsqrt(mean(x^2)+eps)*w`, not
#      Gemma-3's zero-centered `(1 + w)`. A scale-less variant also exists.
#   2. V passes through that scale-less RMSNorm on every layer (`v_norm`); no
#      weight tensor exists for it.
#   3. Attention scale is 1.0 on BOTH layer types (`self.scaling = 1.0`), not
#      Gemma-3's query_pre_attn_scalar^-0.5.
#   4. Global (every sixth) layers use head_dim 512 with ONE kv head (MQA,
#      repeated 16x) and carry NO v_proj: `attention_k_eq_v` makes V the raw
#      k_proj view taken BEFORE k_norm.
#   5. Sliding layers keep the Gemma-3 shape: head_dim 256, 8 kv heads.
#   6. Global RoPE is "proportional" partial rotary: theta 1e6, no linear
#      position factor, and only the first int(0.25*512//2)=64 frequency pairs
#      rotate (the remaining 192 pairs get inv_freq 0 -> cos 1, sin 0, i.e.
#      identity). Sliding RoPE is the plain full-width 256-dim theta-1e4 table.
#   7. Every layer carries a `layer_scalar` [1] buffer applied to the layer
#      output AFTER the feed-forward residual add.
#   8. Checkpoint keys are prefixed `model.language_model.` .

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
from serenitymojo.ops.attention_flash import (
    sdpa_flash_infer_fwd_causal_padmask_dynamic,
)
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_to_bf16
from serenitymojo.ops.linear import linear
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.tensor_algebra import mul, mul_scalar
from serenitymojo.tensor import Tensor


comptime TArc = ArcPointer[Tensor]
comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _TPB = 256
comptime GEMMA4_HIDDEN = 3840
comptime GEMMA4_LAYERS = 48
comptime GEMMA4_HEADS = 16
comptime GEMMA4_KV_HEADS = 8
comptime GEMMA4_HEAD_DIM = 256
comptime GEMMA4_GLOBAL_KV_HEADS = 1
comptime GEMMA4_GLOBAL_HEAD_DIM = 512
comptime GEMMA4_MAX_TOKENS = 1024
# int(partial_rotary_factor * global_head_dim // 2) = int(0.25 * 512 // 2).
comptime GEMMA4_GLOBAL_ROPE_ANGLES = 64
comptime GEMMA4_EPS = Float32(1.0e-6)


@fieldwise_init
struct Gemma4Rope(Copyable, Movable):
    var local_cos_q: TArc
    var local_sin_q: TArc
    var local_cos_k: TArc
    var local_sin_k: TArc
    var global_cos_q: TArc
    var global_sin_q: TArc
    var global_cos_k: TArc
    var global_sin_k: TArc


@fieldwise_init
struct Gemma4LayerWeights(Movable):
    var input_ln: Tensor
    var post_attention_ln: Tensor
    var pre_feedforward_ln: Tensor
    var post_feedforward_ln: Tensor
    var q_norm: Tensor
    var k_norm: Tensor
    var q_proj: Tensor
    var k_proj: Tensor
    # Placeholder on global layers, where `attention_k_eq_v` removes the
    # tensor entirely; read only when `is_global` is False.
    var v_proj: Tensor
    var o_proj: Tensor
    var gate_proj: Tensor
    var up_proj: Tensor
    var down_proj: Tensor
    var layer_scalar: Float32
    var is_global: Bool


@fieldwise_init
struct Gemma4HiddenBatch(Movable):
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


def detect_gemma4_prefix(st: ShardedSafeTensors) raises -> String:
    """Resolve the checkpoint's tensor-name prefix.

    Two shipping layouts carry the SAME 48x3840 tower under different names:
      * `google/gemma-4-12B-it`                  -> `model.language_model.*`
      * LTX-2.5 `gemma4-12b-with-proj-...`       -> `model.*`
    Both were read off the real files (safetensors headers), not assumed. The
    LTX file additionally carries `vision_model.*` / `audio_projector.*` /
    `multi_modal_projector.*` / `text_embedding_projection.*`; the text path
    ignores those.
    """
    if st.has_tensor(String("model.language_model.embed_tokens.weight")):
        return String("model.language_model.")
    if st.has_tensor(String("model.embed_tokens.weight")):
        return String("model.")
    raise Error(
        "gemma4: checkpoint has neither model.language_model.embed_tokens.weight"
        " nor model.embed_tokens.weight — not a recognized Gemma-4 tower"
    )


def _load_layer(
    st: ShardedSafeTensors, layer_idx: Int, is_global: Bool, prefix: String,
    ctx: DeviceContext,
) raises -> Gemma4LayerWeights:
    var p = prefix + "layers." + String(layer_idx) + "."
    # Global layers have no v_proj key at all — never request it.
    var v_proj: Tensor
    if is_global:
        var dummy_vals: List[Float32] = [Float32(0.0)]
        var dummy_shape: List[Int] = [1]
        v_proj = Tensor.from_host(dummy_vals, dummy_shape^, STDtype.BF16, ctx)
    else:
        v_proj = _load_bf16(st, p + "self_attn.v_proj.weight", ctx)
    var scalar_t = _load_bf16(st, p + "layer_scalar", ctx)
    var scalar_host = scalar_t.to_host(ctx)
    return Gemma4LayerWeights(
        _load_bf16(st, p + "input_layernorm.weight", ctx),
        _load_bf16(st, p + "post_attention_layernorm.weight", ctx),
        _load_bf16(st, p + "pre_feedforward_layernorm.weight", ctx),
        _load_bf16(st, p + "post_feedforward_layernorm.weight", ctx),
        _load_bf16(st, p + "self_attn.q_norm.weight", ctx),
        _load_bf16(st, p + "self_attn.k_norm.weight", ctx),
        _load_bf16(st, p + "self_attn.q_proj.weight", ctx),
        _load_bf16(st, p + "self_attn.k_proj.weight", ctx),
        v_proj^,
        _load_bf16(st, p + "self_attn.o_proj.weight", ctx),
        _load_bf16(st, p + "mlp.gate_proj.weight", ctx),
        _load_bf16(st, p + "mlp.up_proj.weight", ctx),
        _load_bf16(st, p + "mlp.down_proj.weight", ctx),
        scalar_host[0],
        is_global,
    )


# Gemma-4 RMSNorm (Gemma4UnifiedRMSNorm, with_scale=True) is neither the
# Llama/Qwen formula implemented by ops.rms_norm nor the Gemma-3 delta form.
# HF computes in F32 and casts back:
#   y = (x.float() * (mean(x^2)+eps)^-0.5) * weight.float()
def _gemma4_rms_kernel(
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
        o[row, c] = rebind[o.element_type]((v * inv * ww).cast[DType.bfloat16]())
        c += _TPB


# Scale-less variant (Gemma4UnifiedRMSNorm(with_scale=False)) — `v_norm` holds
# no weight tensor, so the normalized value is stored directly.
def _gemma4_rms_noscale_kernel(
    x: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
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
        o[row, c] = rebind[o.element_type]((v * inv).cast[DType.bfloat16]())
        c += _TPB


def gemma4_rms_norm(
    x: Tensor, weight: Tensor, ctx: DeviceContext
) raises -> Tensor:
    if x.dtype() != STDtype.BF16 or weight.dtype() != STDtype.BF16:
        raise Error("gemma4_rms_norm: x and weight must be BF16")
    var shape = x.shape()
    var cols = shape[len(shape) - 1]
    if weight.numel() != cols:
        raise Error("gemma4_rms_norm: weight width mismatch")
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
    ctx.enqueue_function[_gemma4_rms_kernel, _gemma4_rms_kernel](
        X, W, O, cols, GEMMA4_EPS,
        grid_dim=rows, block_dim=_TPB,
    )
    return Tensor(out_buf^, shape^, STDtype.BF16)


def gemma4_rms_norm_noscale(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Weight-free Gemma-4 RMSNorm — the `v_norm` applied to V every layer."""
    if x.dtype() != STDtype.BF16:
        raise Error("gemma4_rms_norm_noscale: x must be BF16")
    var shape = x.shape()
    var cols = shape[len(shape) - 1]
    var rows = x.numel() // cols
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    var x_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rows, cols))
    var X = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        x.buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), x_rl
    )
    ctx.enqueue_function[
        _gemma4_rms_noscale_kernel, _gemma4_rms_noscale_kernel
    ](
        X, O, cols, GEMMA4_EPS,
        grid_dim=rows, block_dim=_TPB,
    )
    return Tensor(out_buf^, shape^, STDtype.BF16)


def _rope_host(
    seq: Int,
    heads: Int,
    position_offset: Int,
    theta: Float64,
    position_scale: Float32,
    head_dim: Int,
    rope_angles: Int,
) raises -> List[List[Float32]]:
    """Half-split cos/sin rows in (position, head) order.

    `rope_angles` is the count of LEADING frequency pairs that rotate; the
    remaining `head_dim/2 - rope_angles` pairs carry inv_freq 0, which is the
    `proportional` RoPE partial-rotary identity (cos 1, sin 0).
    """
    var half = head_dim // 2
    var cos_vals = List[Float32]()
    var sin_vals = List[Float32]()
    var log_theta = flog(Float32(theta))
    for t in range(seq):
        var position = Float32(position_offset + t) * position_scale
        for _h in range(heads):
            for i in range(half):
                if i < rope_angles:
                    var exponent = (
                        -log_theta * Float32(2 * i) / Float32(head_dim)
                    )
                    var angle = position * fexp(exponent)
                    cos_vals.append(fcos(angle))
                    sin_vals.append(fsin(angle))
                else:
                    cos_vals.append(Float32(1.0))
                    sin_vals.append(Float32(0.0))
    var out = List[List[Float32]]()
    out.append(cos_vals^)
    out.append(sin_vals^)
    return out^


def _rope_tensor(
    values: List[Float32], seq: Int, heads: Int, head_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    # HF casts cos/sin to the activation dtype before applying them
    # (`cos.to(dtype=x.dtype)`), so a BF16 table is the faithful carrier.
    var shape = List[Int]()
    shape.append(seq * heads * (head_dim // 2))
    return Tensor.from_host(values, shape^, STDtype.BF16, ctx)


def _build_rope(
    seq: Int, real_len: Int, ctx: DeviceContext
) raises -> Gemma4Rope:
    var offset = GEMMA4_MAX_TOKENS - real_len
    comptime LOCAL_HALF_ANGLES = GEMMA4_HEAD_DIM // 2
    var lq = _rope_host(
        seq, GEMMA4_HEADS, offset, Float64(10000.0), Float32(1.0),
        GEMMA4_HEAD_DIM, LOCAL_HALF_ANGLES,
    )
    var lk = _rope_host(
        seq, GEMMA4_KV_HEADS, offset, Float64(10000.0), Float32(1.0),
        GEMMA4_HEAD_DIM, LOCAL_HALF_ANGLES,
    )
    # Global RoPE: theta 1e6, NO linear position factor, partial rotary over
    # the first 64 of 256 pairs (partial_rotary_factor 0.25 at head_dim 512).
    var gq = _rope_host(
        seq, GEMMA4_HEADS, offset, Float64(1000000.0), Float32(1.0),
        GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_ROPE_ANGLES,
    )
    var gk = _rope_host(
        seq, GEMMA4_GLOBAL_KV_HEADS, offset, Float64(1000000.0), Float32(1.0),
        GEMMA4_GLOBAL_HEAD_DIM, GEMMA4_GLOBAL_ROPE_ANGLES,
    )
    return Gemma4Rope(
        TArc(_rope_tensor(lq[0], seq, GEMMA4_HEADS, GEMMA4_HEAD_DIM, ctx)),
        TArc(_rope_tensor(lq[1], seq, GEMMA4_HEADS, GEMMA4_HEAD_DIM, ctx)),
        TArc(_rope_tensor(lk[0], seq, GEMMA4_KV_HEADS, GEMMA4_HEAD_DIM, ctx)),
        TArc(_rope_tensor(lk[1], seq, GEMMA4_KV_HEADS, GEMMA4_HEAD_DIM, ctx)),
        TArc(_rope_tensor(
            gq[0], seq, GEMMA4_HEADS, GEMMA4_GLOBAL_HEAD_DIM, ctx
        )),
        TArc(_rope_tensor(
            gq[1], seq, GEMMA4_HEADS, GEMMA4_GLOBAL_HEAD_DIM, ctx
        )),
        TArc(_rope_tensor(
            gk[0], seq, GEMMA4_GLOBAL_KV_HEADS, GEMMA4_GLOBAL_HEAD_DIM, ctx
        )),
        TArc(_rope_tensor(
            gk[1], seq, GEMMA4_GLOBAL_KV_HEADS, GEMMA4_GLOBAL_HEAD_DIM, ctx
        )),
    )


# Gemma-4 sets `self.scaling = 1.0` for every layer type — the softmax logits
# are NOT divided by sqrt(head_dim).
comptime GEMMA4_ATTN_SCALE = Float32(1.0)


def _attention(
    q: Tensor, k: Tensor, v: Tensor,
    real_len: Int, seq: Int, ctx: DeviceContext,
) raises -> Tensor:
    """Causal valid-prefix SDPA for BOTH layer types.

    The gemma3 template enumerates the eight 128-step buckets against the
    `sdpa_flash_infer_fwd_causal_padmask[B,S,H,Dh]` wrapper as if the comptime
    parameters selected a specialization. They do not: that wrapper's whole body
    forwards to `..._dynamic`, which reads B/S/H/Dh off the tensor shapes
    (ops/attention_flash.mojo:283). The bucket table was therefore eight
    identical calls, so it is not reproduced here — and dropping it is what lets
    one code path serve head_dim 256 (sliding) and 512 (global) alike.

    The 128-step bucketing itself still happens, in `_bucket_for`; it bounds how
    many distinct shapes the cuDNN plan cache sees. Only the fake dispatch goes.
    """
    if seq < 128 or seq > GEMMA4_MAX_TOKENS or (seq % 128) != 0:
        raise Error("Gemma4 attention bucket must be 128..1024 in 128-token steps")
    return sdpa_flash_infer_fwd_causal_padmask_dynamic(
        q, k, v, real_len, GEMMA4_ATTN_SCALE, ctx
    )


def _layer_forward(
    w: Gemma4LayerWeights,
    hidden: Tensor,
    rope: Gemma4Rope,
    real_len: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var seq = hidden.shape()[1]
    var head_dim = GEMMA4_GLOBAL_HEAD_DIM if w.is_global else GEMMA4_HEAD_DIM
    var kv_heads = (
        GEMMA4_GLOBAL_KV_HEADS if w.is_global else GEMMA4_KV_HEADS
    )
    var residual = _alias(hidden)
    var normed = gemma4_rms_norm(hidden, w.input_ln, ctx)

    var q_shape: List[Int] = [1, seq, GEMMA4_HEADS, head_dim]
    var kv_shape: List[Int] = [1, seq, kv_heads, head_dim]
    var q = _reshape(linear(normed, w.q_proj, None, ctx), q_shape^, ctx)
    var k_view = _reshape(
        linear(normed, w.k_proj, None, ctx), kv_shape.copy(), ctx
    )

    # `attention_k_eq_v` global layers take V from the RAW k_proj view, i.e.
    # before k_norm; sliding layers project V separately. Either way v_norm
    # (weight-free) is applied and NO RoPE touches V.
    var v: Tensor
    if w.is_global:
        v = gemma4_rms_norm_noscale(k_view, ctx)
    else:
        v = gemma4_rms_norm_noscale(
            _reshape(linear(normed, w.v_proj, None, ctx), kv_shape.copy(), ctx),
            ctx,
        )

    q = gemma4_rms_norm(q, w.q_norm, ctx)
    var k = gemma4_rms_norm(k_view, w.k_norm, ctx)
    if w.is_global:
        q = rope_halfsplit(q, rope.global_cos_q[], rope.global_sin_q[], ctx)
        k = rope_halfsplit(k, rope.global_cos_k[], rope.global_sin_k[], ctx)
    else:
        q = rope_halfsplit(q, rope.local_cos_q[], rope.local_sin_q[], ctx)
        k = rope_halfsplit(k, rope.local_cos_k[], rope.local_sin_k[], ctx)

    var k_rep = _repeat_kv(k^, GEMMA4_HEADS, kv_heads, ctx)
    var v_rep = _repeat_kv(v^, GEMMA4_HEADS, kv_heads, ctx)
    var attn = _attention(q, k_rep, v_rep, real_len, seq, ctx)
    var flat_shape: List[Int] = [1, seq, GEMMA4_HEADS * head_dim]
    attn = _reshape(attn, flat_shape^, ctx)
    var attn_out = linear(attn, w.o_proj, None, ctx)
    attn_out = gemma4_rms_norm(attn_out, w.post_attention_ln, ctx)
    var hidden2 = _add(residual, attn_out, ctx)

    residual = _alias(hidden2)
    normed = gemma4_rms_norm(hidden2, w.pre_feedforward_ln, ctx)
    var gate = gelu(linear(normed, w.gate_proj, None, ctx), ctx)
    var up = linear(normed, w.up_proj, None, ctx)
    var ff = linear(mul(gate, up, ctx), w.down_proj, None, ctx)
    ff = gemma4_rms_norm(ff, w.post_feedforward_ln, ctx)
    var out = _add(residual, ff, ctx)
    # `hidden_states *= self.layer_scalar` closes the decoder layer.
    return mul_scalar(out, w.layer_scalar, ctx)


def _bucket_for(max_len: Int) raises -> Int:
    if max_len < 1 or max_len > GEMMA4_MAX_TOKENS:
        raise Error("Gemma4 token count must be in [1, 1024]")
    return ((max_len + 127) // 128) * 128


def encode_gemma4_hidden_states_streamed(
    checkpoint_path: String,
    ids_unpadded: List[List[Int]],
    ctx: DeviceContext,
) raises -> Gemma4HiddenBatch:
    """Encode one or more prompts while each layer is resident once."""
    if len(ids_unpadded) < 1:
        raise Error("Gemma4 encode: no prompts")
    var max_len = 0
    var lengths = List[Int]()
    for p in range(len(ids_unpadded)):
        var n = len(ids_unpadded[p])
        if n < 1 or n > GEMMA4_MAX_TOKENS:
            raise Error("Gemma4 encode: prompt token count outside [1, 1024]")
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
    # `model.language_model.*` (google) vs `model.*` (LTX-2.5) — resolved from
    # the checkpoint, never assumed.
    var prefix = detect_gemma4_prefix(st)
    print(String("LTX2_ACTIVITY gemma4 key prefix '") + prefix + "'")
    # Reuse the admitted embedding gather with a transient mini encoder holding
    # only the embedding table. `_embed` looks the table up under the Qwen3
    # name, so the Gemma-4 checkpoint key is registered under that alias here —
    # the checkpoint itself is read by its real name.
    var ew = List[TArc]()
    var en = Dict[String, Int]()
    ew.append(TArc(_load_bf16(st, prefix + "embed_tokens.weight", ctx)))
    en[String("model.embed_tokens.weight")] = 0
    var ecfg = Qwen3Config(
        GEMMA4_HIDDEN, GEMMA4_LAYERS, GEMMA4_HEADS, GEMMA4_KV_HEADS,
        GEMMA4_HEAD_DIM, GEMMA4_EPS, Float64(1000000.0),
    )
    var embedder = Qwen3Encoder(ew^, en^, ecfg)
    var hiddens = List[TArc]()
    for p in range(len(ids)):
        # embed_scale = hidden_size**0.5 = sqrt(3840), which rounds to exactly
        # 62 when HF casts it to the BF16 weight dtype.
        hiddens.append(TArc(
            mul_scalar(embedder._embed(ids[p], ctx), Float32(62.0), ctx)
        ))
    ctx.synchronize()
    st.release_to_os()

    var ropes = List[Gemma4Rope]()
    for p in range(len(ids)):
        ropes.append(_build_rope(bucket, lengths[p], ctx))

    var states = List[List[TArc]]()
    for _p in range(len(ids)):
        states.append(List[TArc]())

    for li in range(GEMMA4_LAYERS):
        for p in range(len(ids)):
            ref prompt_states = states[p]
            prompt_states.append(TArc(_alias(hiddens[p][])))
        var is_global = ((li + 1) % 6) == 0
        var layer = _load_layer(st, li, is_global, prefix, ctx)
        for p in range(len(ids)):
            hiddens[p] = TArc(_layer_forward(
                layer, hiddens[p][], ropes[p], lengths[p], ctx
            ))
        ctx.synchronize()
        st.release_to_os()
        print(
            String("LTX2_ACTIVITY encoding Gemma4 layer ")
            + String(li + 1) + String("/") + String(GEMMA4_LAYERS)
        )

    var final_norm = _load_bf16(st, prefix + "norm.weight", ctx)
    for p in range(len(ids)):
        var final_hidden = gemma4_rms_norm(hiddens[p][], final_norm, ctx)
        ref prompt_states = states[p]
        prompt_states.append(TArc(final_hidden^))
    ctx.synchronize()
    st.release_to_os()
    return Gemma4HiddenBatch(states^, lengths^, bucket)
