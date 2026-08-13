# Pure-Mojo LTX-2 prompt conditioner.
#
# Converts positive + negative prompt text into the exact pre-connector
# safetensors consumed by ltx2_t2v_av_hq:
#   video_context / audio_context / neg_video_context / neg_audio_context
#   video_len / neg_video_len
#
# FeatureExtractorV2 contract:
#   stack 49 Gemma hidden states as [B,T,D,L], RMS-normalize over D for each
#   token/layer, reshape with L as the INNER dimension to [B,T,D*L], rescale,
#   then apply the checkpoint's video/audio aggregate projections. Padding is
#   right-sided after encoding; zero features become projection-bias rows.

from std.sys import argv
from std.gpu import block_idx, thread_idx
from max.gpu import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from std.math import sqrt
from std.memory import ArcPointer, stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.text_encoder.gemma3_ltx_streamed import (
    GEMMA_HIDDEN,
    GEMMA_LAYERS,
    GEMMA_MAX_TOKENS,
    GemmaHiddenBatch,
    encode_gemma3_hidden_states_streamed,
)
from serenitymojo.ops.linear import linear_bias
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.tensor import Tensor
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


comptime TArc = ArcPointer[Tensor]
comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _TPB = 256
comptime _N_STATES = GEMMA_LAYERS + 1
comptime _FEATURES = GEMMA_HIDDEN * _N_STATES


def _alias(x: Tensor) -> Tensor:
    return Tensor(x.buf.copy(), x.shape(), x.dtype())


def _pack_state_kernel(
    src: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    dst: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    layer_idx_w: Int32,
):
    var layer_idx = Int(layer_idx_w)
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var local: Float32 = 0.0
    var d = tid
    while d < GEMMA_HIDDEN:
        var v = rebind[Scalar[DType.bfloat16]](src[row, d]).cast[DType.float32]()
        local += v * v
        d += _TPB
    shared[tid] = local
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var inv = Float32(1.0) / sqrt(
        shared[0] / Float32(GEMMA_HIDDEN) + Float32(1.0e-6)
    )
    d = tid
    while d < GEMMA_HIDDEN:
        var v = rebind[Scalar[DType.bfloat16]](src[row, d]).cast[DType.float32]()
        # torch.stack(hidden_states, dim=-1).reshape(D*L) makes L the
        # innermost dimension: column = hidden_dim_index * 49 + layer_index.
        dst[row, d * _N_STATES + layer_idx] = rebind[dst.element_type](
            (v * inv).cast[DType.bfloat16]()
        )
        d += _TPB


def _pack_features(
    states: List[TArc], real_len: Int, bucket: Int, ctx: DeviceContext
) raises -> Tensor:
    if len(states) != _N_STATES:
        raise Error("LTX2 feature extractor requires exactly 49 Gemma states")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](bucket * _FEATURES * 2)
    ctx.enqueue_memset[DType.uint8](out_buf, 0)
    var out_shape: List[Int] = [1, bucket, _FEATURES]
    var dst_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](bucket, _FEATURES))
    var DST = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=dst_rl,
    )
    var src_rl = RuntimeLayout[_DYN2].row_major(
        IndexList[2](bucket, GEMMA_HIDDEN)
    )
    for li in range(_N_STATES):
        var SRC = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(states[li][].buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=src_rl,
    )
        ctx.enqueue_function[_pack_state_kernel](
            SRC, DST, Int32(li), grid_dim=real_len, block_dim=_TPB
        )
    ctx.synchronize()
    return Tensor(out_buf^, out_shape^, STDtype.BF16)


def _fill_bias_rows_kernel(
    dst: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],
    bias: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    first_row_w: Int32,
    rows_w: Int32,
    cols_w: Int32,
):
    var first_row = Int(first_row_w)
    var rows = Int(rows_w)
    var cols = Int(cols_w)
    var idx = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    var n = (rows - first_row) * cols
    if idx < n:
        var row = first_row + idx // cols
        var col = idx % cols
        dst[row, col] = rebind[dst.element_type](bias[col])


def _pad_projection_with_bias(
    var projected: Tensor, bias: Tensor, bucket: Int, out_dim: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    if bucket == GEMMA_MAX_TOKENS:
        return projected^
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        GEMMA_MAX_TOKENS * out_dim * 2
    )
    ctx.enqueue_memset[DType.uint8](out_buf, 0)
    var prefix = projected.buf.create_sub_buffer[DType.uint8](
        0, bucket * out_dim * 2
    )
    var dest_prefix = out_buf.create_sub_buffer[DType.uint8](
        0, bucket * out_dim * 2
    )
    ctx.enqueue_copy(dst_buf=dest_prefix, src_buf=prefix)
    var dst_rl = RuntimeLayout[_DYN2].row_major(
        IndexList[2](GEMMA_MAX_TOKENS, out_dim)
    )
    var bias_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](out_dim))
    var DST = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(out_buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=dst_rl,
    )
    var BIAS = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.bfloat16], MutAnyOrigin](
            unsafe_from_address=Int(bias.buf.unsafe_ptr().bitcast[BFloat16]())
        ),
        runtime_layout=bias_rl,
    )
    var n = (GEMMA_MAX_TOKENS - bucket) * out_dim
    ctx.enqueue_function[_fill_bias_rows_kernel](
        DST, BIAS, Int32(bucket), Int32(GEMMA_MAX_TOKENS), Int32(out_dim),
        grid_dim=(n + _TPB - 1) // _TPB, block_dim=_TPB,
    )
    var shape: List[Int] = [1, GEMMA_MAX_TOKENS, out_dim]
    return Tensor(out_buf^, shape^, STDtype.BF16)


@fieldwise_init
struct _ProjectedPair(Movable):
    var positive: Tensor
    var negative: Tensor


def _project_pair(
    checkpoint: ShardedSafeTensors,
    positive_features: Tensor,
    negative_features: Tensor,
    stem: String,
    out_dim: Int,
    bucket: Int,
    ctx: DeviceContext,
) raises -> _ProjectedPair:
    # Function scope is intentional: the large aggregate matrix is released
    # before the other modality's matrix is loaded.
    var weight = Tensor.from_view_as_bf16(
        checkpoint.tensor_view(stem + ".weight"), ctx
    )
    var bias = Tensor.from_view_as_bf16(
        checkpoint.tensor_view(stem + ".bias"), ctx
    )
    var scale = sqrt(Float32(out_dim) / Float32(GEMMA_HIDDEN))
    var pos_scaled = mul_scalar(positive_features, scale, ctx)
    var neg_scaled = mul_scalar(negative_features, scale, ctx)
    var pos = linear_bias(pos_scaled, weight, bias, ctx)
    var neg = linear_bias(neg_scaled, weight, bias, ctx)
    pos = _pad_projection_with_bias(pos^, bias, bucket, out_dim, ctx)
    neg = _pad_projection_with_bias(neg^, bias, bucket, out_dim, ctx)
    ctx.synchronize()
    return _ProjectedPair(pos^, neg^)


def encode_ltx2_prompts(
    gemma_checkpoint: String,
    tokenizer_json: String,
    ltx_checkpoint: String,
    output_path: String,
    prompt: String,
    negative_prompt: String,
) raises:
    print("LTX2_ACTIVITY tokenizing prompt")
    var tokenizer = Qwen3Tokenizer(tokenizer_json)
    var ids = List[List[Int]]()
    ids.append(tokenizer.encode_gemma(prompt))
    ids.append(tokenizer.encode_gemma(negative_prompt))

    var ctx = DeviceContext()
    print("LTX2_ACTIVITY loading Gemma text encoder")
    var hidden = encode_gemma3_hidden_states_streamed(
        gemma_checkpoint, ids^, ctx
    )
    print("LTX2_ACTIVITY projecting video and audio conditioning")
    var positive_features = _pack_features(
        hidden.states[0], hidden.lengths[0], hidden.bucket, ctx
    )
    var negative_features = _pack_features(
        hidden.states[1], hidden.lengths[1], hidden.bucket, ctx
    )
    var ltx = ShardedSafeTensors.open(ltx_checkpoint)
    var video = _project_pair(
        ltx, positive_features, negative_features,
        String("text_embedding_projection.video_aggregate_embed"),
        4096, hidden.bucket, ctx,
    )
    ltx.release_to_os()
    var audio = _project_pair(
        ltx, positive_features, negative_features,
        String("text_embedding_projection.audio_aggregate_embed"),
        2048, hidden.bucket, ctx,
    )
    ltx.release_to_os()

    var pos_len_host = List[Float32]()
    pos_len_host.append(Float32(hidden.lengths[0]))
    var neg_len_host = List[Float32]()
    neg_len_host.append(Float32(hidden.lengths[1]))
    var len_shape = List[Int]()
    len_shape.append(1)
    var pos_len = Tensor.from_host(pos_len_host, len_shape.copy(), STDtype.F32, ctx)
    var neg_len = Tensor.from_host(neg_len_host, len_shape^, STDtype.F32, ctx)

    var names = List[String]()
    var tensors = List[TArc]()
    names.append(String("video_context"))
    tensors.append(TArc(_alias(video.positive)))
    names.append(String("audio_context"))
    tensors.append(TArc(_alias(audio.positive)))
    names.append(String("neg_video_context"))
    tensors.append(TArc(_alias(video.negative)))
    names.append(String("neg_audio_context"))
    tensors.append(TArc(_alias(audio.negative)))
    names.append(String("video_len"))
    tensors.append(TArc(pos_len^))
    names.append(String("neg_video_len"))
    tensors.append(TArc(neg_len^))
    print("LTX2_ACTIVITY saving prompt conditioning")
    save_safetensors(names, tensors, output_path, ctx)
    print("LTX2_ACTIVITY conditioning complete")


def main() raises:
    var args = argv()
    if len(args) != 7:
        raise Error(
            "usage: ltx2_encode_prompt GEMMA_FP8 TOKENIZER_JSON "
            "LTX_CHECKPOINT OUTPUT PROMPT NEGATIVE_PROMPT"
        )
    encode_ltx2_prompts(
        String(args[1]), String(args[2]), String(args[3]),
        String(args[4]), String(args[5]), String(args[6]),
    )
