# MiniMax-H3 Qwen3-VL-32B INT8 streamed conditioner.
#
# The native checkpoint is BF16 (~62 GiB).  Loading even one full embedding
# tensor through MAX's pinned staging can charge the desktop cgroup by many
# GiB and OOM the whole user session.  This path therefore has two strict
# contracts:
#
#   1. CACHE BUILD: BF16 source rows are transferred in a fixed 64 MiB slab,
#      quantized PER OUTPUT ROW on the GPU, and immediately written as raw I8
#      plus F32 row-scale sidecars.  CPU code performs file I/O only; it never
#      computes a scale or quantized value.
#   2. ENCODE: linears stream the cached I8 weights and execute direct
#      INT8xINT8->INT32 cuBLAS GEMMs.  A full BF16 weight is never uploaded or
#      dequantized.  Only each small GEMM output returns to BF16.
#
# The cache is resumable and per-tensor atomic.  A weight is accepted only when
# BOTH raw files exist at the exact byte sizes implied by the source header.
# Interrupted builds simply resume at the first missing pair.

from std.gpu.host import DeviceContext, HostBuffer
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor, BatchedTensorUploader
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import (
    BytePtr,
    O_CREAT,
    O_RDONLY,
    O_TRUNC,
    O_WRONLY,
    file_size,
    sys_close,
    sys_memcpy,
    sys_mkdirs,
    sys_open,
    sys_pread,
    sys_pwrite,
    sys_remove,
    sys_rename,
)
from serenitymojo.models.text_encoder.qwen3_encoder import (
    _add,
    _build_causal_mask,
    _build_rope_tables,
    _repeat_kv,
    _reshape,
    _sdpa_dispatch,
)
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_streamed import (
    H3_EPS,
    H3_EXTRACT_LAYER,
    H3_HEADS,
    H3_HEAD_DIM,
    H3_HIDDEN,
    H3_KV_HEADS,
    H3_THETA,
    _detect_layer_prefix,
    _h3_embed_prefix,
    _h3_open_available_shards,
)
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.int8_linear import int8_linear_fwd_rowscale
from serenitymojo.ops.int8_quant import (
    int8_dequant_perrow_to_bf16,
    int8_encode_perrow,
    int8_rowscale,
)
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.rope import rope_halfsplit


comptime H3_ENCODER_INT8_CACHE_VERSION = 1
comptime H3_ENCODER_INT8_STAGE_BYTES = 64 * 1024 * 1024
comptime H3_ENCODER_INT8_SMALL_STAGE_BYTES = 1024 * 1024
comptime _TArc = ArcPointer[Tensor]


def minimax_h3_int8_cache_dir(text_encoder_dir: String) -> String:
    return text_encoder_dir + String("/serenity_int8_rowscale_v1")


def _h3_i8_weight_path(cache_dir: String, name: String) -> String:
    return cache_dir + String("/") + name + String(".i8")


def _h3_i8_scale_path(cache_dir: String, name: String) -> String:
    return cache_dir + String("/") + name + String(".scale.f32")


def _h3_i8_file_size(path: String) -> Int:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return -1
    var n = file_size(fd)
    _ = sys_close(fd)
    return n


def _h3_i8_write_exact(
    fd: Int, ptr: BytePtr, count: Int, offset: Int
) raises:
    var done = 0
    while done < count:
        var n = sys_pwrite(fd, ptr + done, count - done, offset + done)
        if n <= 0:
            raise Error("MiniMax-H3 INT8 cache pwrite failed")
        done += n


def _h3_i8_read_exact(
    fd: Int, ptr: BytePtr, count: Int, offset: Int
) raises:
    var done = 0
    while done < count:
        var n = sys_pread(fd, ptr + done, count - done, offset + done)
        if n <= 0:
            raise Error("MiniMax-H3 INT8 cache pread failed")
        done += n


struct _H3Int8IO(Movable):
    """One fixed pinned slab shared by conversion and runtime streaming."""

    var stage: HostBuffer[DType.uint8]

    def __init__(out self, ctx: DeviceContext) raises:
        self.stage = ctx.enqueue_create_host_buffer[DType.uint8](
            H3_ENCODER_INT8_STAGE_BYTES
        )


def _h3_i8_cache_valid(
    st: ShardedSafeTensors, cache_dir: String, name: String
) raises -> Bool:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.BF16 or len(info.shape) != 2:
        return False
    var rows = info.shape[0]
    var cols = info.shape[1]
    return _h3_i8_file_size(_h3_i8_weight_path(cache_dir, name)) \
        == rows * cols and _h3_i8_file_size(
            _h3_i8_scale_path(cache_dir, name)
        ) == rows * 4


def _h3_i8_quantize_matrix(
    st: ShardedSafeTensors,
    cache_dir: String,
    name: String,
    mut io: _H3Int8IO,
    ctx: DeviceContext,
) raises -> Bool:
    """GPU-quantize one BF16 [rows,cols] matrix in row-aligned chunks.

    Returns True when this call built the files, False for an exact-size cache
    hit.  The only host operations are mmap->pinned memcpy and pwrite.
    """
    if _h3_i8_cache_valid(st, cache_dir, name):
        return False

    var tv = st.tensor_view(name)
    if tv.dtype != STDtype.BF16 or len(tv.shape) != 2:
        raise Error(
            String("MiniMax-H3 INT8 cache requires BF16 rank-2 matrix: ")
            + name
        )
    var rows = tv.shape[0]
    var cols = tv.shape[1]
    var row_bytes = cols * 2
    var rows_per_chunk = H3_ENCODER_INT8_STAGE_BYTES // row_bytes
    if rows_per_chunk <= 0:
        raise Error(
            String("MiniMax-H3 INT8 cache row exceeds staging slab: ") + name
        )

    _ = sys_mkdirs(cache_dir)
    var w_path = _h3_i8_weight_path(cache_dir, name)
    var s_path = _h3_i8_scale_path(cache_dir, name)
    var w_tmp = w_path + String(".tmp")
    var s_tmp = s_path + String(".tmp")
    var wfd = sys_open(w_tmp, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if wfd < 0:
        raise Error(String("failed to open INT8 cache weight: ") + w_tmp)
    var sfd = sys_open(s_tmp, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    if sfd < 0:
        _ = sys_close(wfd)
        _ = sys_remove(w_tmp)
        raise Error(String("failed to open INT8 cache scale: ") + s_tmp)

    var row0 = 0
    try:
        while row0 < rows:
            var nr = rows - row0
            if nr > rows_per_chunk:
                nr = rows_per_chunk
            var in_bytes = nr * row_bytes
            var q_bytes = nr * cols
            var s_bytes = nr * 4
            var s_off = (q_bytes + 255) & ~255
            if s_off + s_bytes > H3_ENCODER_INT8_STAGE_BYTES:
                raise Error("MiniMax-H3 INT8 conversion staging arithmetic overflow")

            # mmap -> fixed pinned transport slab -> device BF16 chunk.
            var hp = BytePtr(unsafe_from_address=Int(io.stage.unsafe_ptr()))
            var src = BytePtr(
                unsafe_from_address=Int(tv.data.unsafe_ptr()) + row0 * row_bytes
            )
            _ = sys_memcpy(hp, src, in_bytes)
            var in_dev = ctx.enqueue_create_buffer[DType.uint8](in_bytes)
            var hsrc = io.stage.create_sub_buffer[DType.uint8](0, in_bytes)
            ctx.enqueue_copy(dst_buf=in_dev, src_buf=hsrc)
            var chunk_shape: List[Int] = [nr, cols]
            var w = Tensor(in_dev^, chunk_shape^, STDtype.BF16)

            # All quantization arithmetic is GPU-resident.
            var scale = int8_rowscale(w, ctx)
            var q = int8_encode_perrow(w, scale, ctx)

            # Device results -> same fixed slab -> raw atomic sidecar files.
            var hq = io.stage.create_sub_buffer[DType.uint8](0, q_bytes)
            var hs = io.stage.create_sub_buffer[DType.uint8](s_off, s_bytes)
            ctx.enqueue_copy(dst_buf=hq, src_buf=q.buf)
            ctx.enqueue_copy(dst_buf=hs, src_buf=scale.buf)
            ctx.synchronize()
            _h3_i8_write_exact(wfd, hp, q_bytes, row0 * cols)
            _h3_i8_write_exact(sfd, hp + s_off, s_bytes, row0 * 4)
            row0 += nr
            st.release_to_os()
    except e:
        _ = sys_close(wfd)
        _ = sys_close(sfd)
        _ = sys_remove(w_tmp)
        _ = sys_remove(s_tmp)
        raise Error(String("MiniMax-H3 INT8 cache build failed for ") + name + ": " + String(e))

    _ = sys_close(wfd)
    _ = sys_close(sfd)
    # Publish scale first: a weight without its scale is never a valid pair.
    if sys_rename(s_tmp, s_path) != 0:
        _ = sys_remove(w_tmp)
        _ = sys_remove(s_tmp)
        raise Error(String("failed to publish INT8 scale: ") + s_path)
    if sys_rename(w_tmp, w_path) != 0:
        _ = sys_remove(w_tmp)
        raise Error(String("failed to publish INT8 weight: ") + w_path)
    return True


def _h3_i8_matrix_suffixes() -> List[String]:
    var out = List[String]()
    out.append(String("self_attn.q_proj.weight"))
    out.append(String("self_attn.k_proj.weight"))
    out.append(String("self_attn.v_proj.weight"))
    out.append(String("self_attn.o_proj.weight"))
    out.append(String("mlp.gate_proj.weight"))
    out.append(String("mlp.up_proj.weight"))
    out.append(String("mlp.down_proj.weight"))
    return out^


def minimax_h3_build_int8_encoder_cache(
    text_encoder_dir: String,
    num_layers: Int,
    ctx: DeviceContext,
) raises -> Int:
    """Build/resume the GPU-quantized cache. Returns matrices newly built."""
    if num_layers <= 0 or num_layers > H3_EXTRACT_LAYER:
        raise Error("MiniMax-H3 INT8 cache layer count must be in 1..50")
    var st = _h3_open_available_shards(text_encoder_dir)
    var layer_prefix = _detect_layer_prefix(st)
    var embed_prefix = _h3_embed_prefix(layer_prefix)
    var cache_dir = minimax_h3_int8_cache_dir(text_encoder_dir)
    var io = _H3Int8IO(ctx)
    var built = 0
    if _h3_i8_quantize_matrix(
        st, cache_dir, embed_prefix + "embed_tokens.weight", io, ctx
    ):
        built += 1
        print("  H3 encoder INT8 cache: embedding complete")
    var suffixes = _h3_i8_matrix_suffixes()
    for li in range(num_layers):
        var prefix = layer_prefix + String(li) + "."
        var layer_built = 0
        for ki in range(len(suffixes)):
            if _h3_i8_quantize_matrix(
                st, cache_dir, prefix + suffixes[ki], io, ctx
            ):
                built += 1
                layer_built += 1
        if layer_built > 0 or (li + 1) % 5 == 0 or li + 1 == num_layers:
            print(
                "  H3 encoder INT8 cache: layer", li + 1, "/", num_layers,
                "new matrices", layer_built,
            )
    print(
        "  H3 encoder INT8 cache ready:", cache_dir,
        "new matrices", built,
    )
    return built


def _h3_i8_load_raw(
    path: String,
    var shape: List[Int],
    dtype: STDtype,
    mut io: _H3Int8IO,
    ctx: DeviceContext,
) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    var nbytes = n * dtype.byte_size()
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        raise Error(String("MiniMax-H3 INT8 cache missing: ") + path)
    if file_size(fd) != nbytes:
        _ = sys_close(fd)
        raise Error(String("MiniMax-H3 INT8 cache size mismatch: ") + path)
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var off = 0
    try:
        while off < nbytes:
            var take = nbytes - off
            if take > H3_ENCODER_INT8_STAGE_BYTES:
                take = H3_ENCODER_INT8_STAGE_BYTES
            var hp = BytePtr(unsafe_from_address=Int(io.stage.unsafe_ptr()))
            _h3_i8_read_exact(fd, hp, take, off)
            var hs = io.stage.create_sub_buffer[DType.uint8](0, take)
            var ds = dev.create_sub_buffer[DType.uint8](off, take)
            ctx.enqueue_copy(dst_buf=ds, src_buf=hs)
            ctx.synchronize()
            off += take
    except e:
        _ = sys_close(fd)
        raise Error(String("MiniMax-H3 INT8 cache load failed: ") + String(e))
    _ = sys_close(fd)
    return Tensor(dev^, shape^, dtype)


def _h3_i8_load_matrix(
    st: ShardedSafeTensors,
    cache_dir: String,
    name: String,
    mut io: _H3Int8IO,
    ctx: DeviceContext,
) raises -> List[_TArc]:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.BF16 or len(info.shape) != 2:
        raise Error(String("bad source metadata for cached INT8 matrix: ") + name)
    var w = _h3_i8_load_raw(
        _h3_i8_weight_path(cache_dir, name), info.shape.copy(), STDtype.I8,
        io, ctx,
    )
    var scale_shape: List[Int] = [info.shape[0]]
    var s = _h3_i8_load_raw(
        _h3_i8_scale_path(cache_dir, name), scale_shape^, STDtype.F32,
        io, ctx,
    )
    var out = List[_TArc]()
    out.append(_TArc(w^))
    out.append(_TArc(s^))
    return out^


def _h3_i8_embed(
    st: ShardedSafeTensors,
    cache_dir: String,
    name: String,
    ids: List[Int],
    mut io: _H3Int8IO,
    ctx: DeviceContext,
) raises -> Tensor:
    """Gather cached INT8 token rows, then dequantize those rows on GPU."""
    var info = st.tensor_info(name)
    if info.dtype != STDtype.BF16 or len(info.shape) != 2:
        raise Error("MiniMax-H3 INT8 embedding source metadata is not BF16 [V,H]")
    var vocab = info.shape[0]
    var hidden = info.shape[1]
    var seq = len(ids)
    var q_bytes = seq * hidden
    var s_off = (q_bytes + 255) & ~255
    var s_bytes = seq * 4
    if s_off + s_bytes > H3_ENCODER_INT8_STAGE_BYTES:
        raise Error("MiniMax-H3 INT8 selected embedding rows exceed staging slab")

    var wfd = sys_open(_h3_i8_weight_path(cache_dir, name), O_RDONLY, 0)
    var sfd = sys_open(_h3_i8_scale_path(cache_dir, name), O_RDONLY, 0)
    if wfd < 0 or sfd < 0:
        if wfd >= 0:
            _ = sys_close(wfd)
        if sfd >= 0:
            _ = sys_close(sfd)
        raise Error("MiniMax-H3 INT8 embedding cache is missing")
    var hp = BytePtr(unsafe_from_address=Int(io.stage.unsafe_ptr()))
    try:
        for i in range(seq):
            var tok = ids[i]
            if tok < 0 or tok >= vocab:
                raise Error("MiniMax-H3 token id outside embedding vocabulary")
            _h3_i8_read_exact(wfd, hp + i * hidden, hidden, tok * hidden)
            _h3_i8_read_exact(sfd, hp + s_off + i * 4, 4, tok * 4)
    except e:
        _ = sys_close(wfd)
        _ = sys_close(sfd)
        raise Error(String("MiniMax-H3 INT8 embedding gather failed: ") + String(e))
    _ = sys_close(wfd)
    _ = sys_close(sfd)

    var qdev = ctx.enqueue_create_buffer[DType.uint8](q_bytes)
    var sdev = ctx.enqueue_create_buffer[DType.uint8](s_bytes)
    ctx.enqueue_copy(
        dst_buf=qdev,
        src_buf=io.stage.create_sub_buffer[DType.uint8](0, q_bytes),
    )
    ctx.enqueue_copy(
        dst_buf=sdev,
        src_buf=io.stage.create_sub_buffer[DType.uint8](s_off, s_bytes),
    )
    ctx.synchronize()
    var qshape: List[Int] = [seq, hidden]
    var sshape: List[Int] = [seq]
    var q = Tensor(qdev^, qshape^, STDtype.I8)
    var s = Tensor(sdev^, sshape^, STDtype.F32)
    var flat = int8_dequant_perrow_to_bf16(q, s, ctx)
    var outshape: List[Int] = [1, seq, hidden]
    print(
        "  H3 encoder INT8 embedding rows:", flat.shape(), "->",
        outshape,
    )
    return _reshape(flat, outshape^, ctx)


struct _H3Int8Layer(Movable):
    # Fixed order: input_layernorm, q_norm, k_norm,
    # post_attention_layernorm.
    var norms: List[_TArc]
    var w8: List[_TArc]
    var scale: List[_TArc]

    def __init__(
        out self,
        var norms: List[_TArc],
        var w8: List[_TArc],
        var scale: List[_TArc],
    ):
        self.norms = norms^
        self.w8 = w8^
        self.scale = scale^


def _h3_i8_load_layer(
    st: ShardedSafeTensors,
    cache_dir: String,
    layer_prefix: String,
    li: Int,
    mut io: _H3Int8IO,
    mut small_uploader: BatchedTensorUploader,
    ctx: DeviceContext,
) raises -> _H3Int8Layer:
    var weights = List[_TArc]()
    var scales = List[_TArc]()
    var suffixes = _h3_i8_matrix_suffixes()
    var ps = layer_prefix + String(li) + "."
    for ki in range(len(suffixes)):
        var pair = _h3_i8_load_matrix(
            st, cache_dir, ps + suffixes[ki], io, ctx
        )
        weights.append(pair[0].copy())
        scales.append(pair[1].copy())

    var small = List[_TArc]()
    small.append(_TArc(small_uploader.from_view(
        st.tensor_view(ps + "input_layernorm.weight"), ctx
    )))
    small.append(_TArc(small_uploader.from_view(
        st.tensor_view(ps + "self_attn.q_norm.weight"), ctx
    )))
    small.append(_TArc(small_uploader.from_view(
        st.tensor_view(ps + "self_attn.k_norm.weight"), ctx
    )))
    small.append(_TArc(small_uploader.from_view(
        st.tensor_view(ps + "post_attention_layernorm.weight"), ctx
    )))
    small_uploader.finish(ctx)
    return _H3Int8Layer(small^, weights^, scales^)


def _h3_i8_layer_forward(
    layer: _H3Int8Layer,
    li: Int,
    hidden: Tensor,
    cos_q: Tensor,
    sin_q: Tensor,
    cos_k: Tensor,
    sin_k: Tensor,
    mask: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var seq = hidden.shape()[1]
    ref in_ln = layer.norms[0][]
    var normed = rms_norm(hidden, in_ln, H3_EPS, ctx)
    var q = int8_linear_fwd_rowscale(
        normed, layer.w8[0][], layer.scale[0][], ctx
    )
    var k = int8_linear_fwd_rowscale(
        normed, layer.w8[1][], layer.scale[1][], ctx
    )
    var v = int8_linear_fwd_rowscale(
        normed, layer.w8[2][], layer.scale[2][], ctx
    )
    if li == 0:
        print(
            "  H3 encoder INT8 layer 0 qkv:", q.shape(), k.shape(), v.shape()
        )
    q = _reshape(q, [1, seq, H3_HEADS, H3_HEAD_DIM], ctx)
    k = _reshape(k, [1, seq, H3_KV_HEADS, H3_HEAD_DIM], ctx)
    v = _reshape(v, [1, seq, H3_KV_HEADS, H3_HEAD_DIM], ctx)
    ref qn = layer.norms[1][]
    ref kn = layer.norms[2][]
    q = rms_norm(q, qn, H3_EPS, ctx)
    k = rms_norm(k, kn, H3_EPS, ctx)
    q = rope_halfsplit(q, cos_q, sin_q, ctx)
    k = rope_halfsplit(k, cos_k, sin_k, ctx)
    var k_rep = _repeat_kv(k^, H3_HEADS, H3_KV_HEADS, ctx)
    var v_rep = _repeat_kv(v^, H3_HEADS, H3_KV_HEADS, ctx)
    var attn = _sdpa_dispatch(
        q, k_rep, v_rep, mask,
        Float32(1.0) / sqrt(Float32(H3_HEAD_DIM)),
        seq, H3_HEADS, H3_HEAD_DIM, ctx,
    )
    if li == 0:
        print("  H3 encoder INT8 layer 0 attention:", attn.shape())
    attn = _reshape(
        attn, [1, seq, H3_HEADS * H3_HEAD_DIM], ctx
    )
    var attn_out = int8_linear_fwd_rowscale(
        attn, layer.w8[3][], layer.scale[3][], ctx
    )
    var hidden2 = _add(hidden, attn_out, ctx)
    ref post_ln = layer.norms[3][]
    var normed2 = rms_norm(hidden2, post_ln, H3_EPS, ctx)
    var gate = int8_linear_fwd_rowscale(
        normed2, layer.w8[4][], layer.scale[4][], ctx
    )
    var up = int8_linear_fwd_rowscale(
        normed2, layer.w8[5][], layer.scale[5][], ctx
    )
    if li == 0:
        print("  H3 encoder INT8 layer 0 mlp gate/up:", gate.shape(), up.shape())
    var act = swiglu(gate, up, ctx)
    var mlp_out = int8_linear_fwd_rowscale(
        act, layer.w8[6][], layer.scale[6][], ctx
    )
    return _add(hidden2, mlp_out, ctx)


def minimax_h3_encode_conditioning_int8_streamed_depth(
    text_encoder_dir: String,
    ids: List[Int],
    num_layers: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Direct W8A8 GPU conditioner. No BF16 matrix enters the runtime path."""
    if len(ids) == 0:
        raise Error("MiniMax-H3 INT8 conditioner received an empty prompt")
    if num_layers <= 0 or num_layers > H3_EXTRACT_LAYER:
        raise Error("MiniMax-H3 INT8 conditioner layer count must be in 1..50")

    _ = minimax_h3_build_int8_encoder_cache(text_encoder_dir, num_layers, ctx)
    var st = _h3_open_available_shards(text_encoder_dir)
    var layer_prefix = _detect_layer_prefix(st)
    var embed_prefix = _h3_embed_prefix(layer_prefix)
    var cache_dir = minimax_h3_int8_cache_dir(text_encoder_dir)
    var io = _H3Int8IO(ctx)
    var small_uploader = BatchedTensorUploader(
        H3_ENCODER_INT8_SMALL_STAGE_BYTES, ctx
    )
    var hidden = _h3_i8_embed(
        st, cache_dir, embed_prefix + "embed_tokens.weight", ids, io, ctx
    )

    var seq = len(ids)
    var q_tables = _build_rope_tables(
        seq, H3_HEADS, H3_HEAD_DIM, H3_THETA
    )
    var k_tables = _build_rope_tables(
        seq, H3_KV_HEADS, H3_HEAD_DIM, H3_THETA
    )
    comptime half = H3_HEAD_DIM // 2
    var cos_q = Tensor.from_host(
        q_tables[0], [seq * H3_HEADS * half], STDtype.BF16, ctx
    )
    var sin_q = Tensor.from_host(
        q_tables[1], [seq * H3_HEADS * half], STDtype.BF16, ctx
    )
    var cos_k = Tensor.from_host(
        k_tables[0], [seq * H3_KV_HEADS * half], STDtype.BF16, ctx
    )
    var sin_k = Tensor.from_host(
        k_tables[1], [seq * H3_KV_HEADS * half], STDtype.BF16, ctx
    )
    var mask_data = _build_causal_mask(seq, H3_HEADS, seq)
    var mask = Tensor.from_host(
        mask_data, [1, H3_HEADS, seq, seq], STDtype.BF16, ctx
    )

    for li in range(num_layers):
        var layer = _h3_i8_load_layer(
            st, cache_dir, layer_prefix, li, io, small_uploader, ctx
        )
        hidden = _h3_i8_layer_forward(
            layer, li, hidden, cos_q, sin_q, cos_k, sin_k, mask, ctx
        )
        ctx.synchronize()
        st.release_to_os()
        if (li + 1) % 5 == 0 or li + 1 == num_layers:
            print("  H3 encoder INT8: layer", li + 1, "/", num_layers)
    print(
        "  H3 encoder INT8 runtime: direct W8A8 GEMM, fixed pinned slab",
        H3_ENCODER_INT8_STAGE_BYTES, "bytes",
    )
    return hidden^


def minimax_h3_encode_conditioning_int8_streamed(
    text_encoder_dir: String,
    ids: List[Int],
    ctx: DeviceContext,
) raises -> Tensor:
    return minimax_h3_encode_conditioning_int8_streamed_depth(
        text_encoder_dir, ids, H3_EXTRACT_LAYER, ctx
    )
