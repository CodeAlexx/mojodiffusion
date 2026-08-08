# MiniMax-H3 byte-exact runtime sidecars.
#
# The released UI profile previously rebuilt three deterministic products on
# every request: the 40-block groupwise-INT8 resident store, the prompt
# conditioning, and the step-owned AdaLN modulation cache.  This module owns
# the two DiT-side caches.  Payloads are saved in their exact storage dtype and
# reloaded by verbatim mmap -> pinned host -> GPU copies; no CPU inference and
# no requantization occur on a cache hit.

from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.models.minimax_h3.fp8_policy import (
    H3_FP8_BF16_KEEP,
    H3_FP8_ROW,
    minimax_h3_fp8_class,
)
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    minimax_h3_block_tensor_names,
    minimax_h3_expected_shape,
)
from serenitymojo.models.dit.minimax_h3_fp8_resident import (
    MINIMAX_H3_INT8_FC1_GROUP_SIZE,
    MINIMAX_H3_INT8_FC2_GROUP_SIZE,
    MINIMAX_H3_INT8_OUT_GROUP_SIZE,
    MINIMAX_H3_INT8_QKV_GROUP_SIZE,
    MINIMAX_H3_RESIDENT_INT8,
    MINIMAX_H3_RESIDENT_INT8_W8A8,
    MiniMaxH3BlockResidentFp8,
    MiniMaxH3ResidentFp8,
    minimax_h3_allocate_resident_scratch,
    minimax_h3_int8_group_size,
)
from serenitymojo.models.dit.minimax_h3_modcache import MiniMaxH3ModCache


comptime MINIMAX_H3_RUNTIME_CACHE_VERSION = 1


def _stat_size_mtime(path: String) -> List[Int]:
    """Return [size, mtime seconds], or [-1, -1] when stat(2) fails."""
    var n = path.byte_length()
    var cbuf = alloc[UInt8](n + 1)
    var src = path.as_bytes()
    for i in range(n):
        cbuf[i] = src[i]
    cbuf[n] = 0
    var statbuf = alloc[UInt8](160)
    var rc = Int(
        external_call["stat", Int32](
            BytePtr(unsafe_from_address=Int(cbuf)),
            BytePtr(unsafe_from_address=Int(statbuf)),
        )
    )
    var size = -1
    var mtime = -1
    if rc == 0:
        var q = statbuf.bitcast[Int64]()
        size = Int(q[6])
        mtime = Int(q[11])
    cbuf.free()
    statbuf.free()
    var out: List[Int] = [size, mtime]
    return out^


def _i64_tensor(value: Int, ctx: DeviceContext) raises -> Tensor:
    var host = ctx.enqueue_create_host_buffer[DType.uint8](8)
    host.unsafe_ptr().bitcast[Int64]()[0] = Int64(value)
    var dev = ctx.enqueue_create_buffer[DType.uint8](8)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    var shape: List[Int] = [1]
    return Tensor(dev^, shape^, STDtype.I64)


def _u8_tensor(value: String, ctx: DeviceContext) raises -> Tensor:
    var n = value.byte_length()
    if n == 0:
        raise Error("MiniMax-H3 runtime cache metadata cannot be empty")
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
    var src = value.as_bytes()
    for i in range(n):
        host.unsafe_ptr()[i] = src[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    var shape: List[Int] = [n]
    return Tensor(dev^, shape^, STDtype.U8)


def _meta_i64(st: SafeTensors, name: String, default: Int) raises -> Int:
    if not st.has_tensor(name):
        return default
    var bytes = st.tensor_bytes(name)
    if len(bytes) != 8:
        return default
    var value = 0
    for i in range(8):
        value |= Int(bytes[i]) << (8 * i)
    return value


def _meta_string(st: SafeTensors, name: String) raises -> String:
    if not st.has_tensor(name):
        return String("")
    var bytes = st.tensor_bytes(name)
    var value = String("")
    for i in range(len(bytes)):
        value += chr(Int(bytes[i]))
    return value^


def _append_common_metadata(
    mut names: List[String],
    mut tensors: List[ArcPointer[Tensor]],
    kind: String,
    source_index: String,
    ctx: DeviceContext,
) raises:
    var sm = _stat_size_mtime(source_index)
    if sm[0] < 0:
        raise Error(
            String("MiniMax-H3 runtime cache cannot stat source index: ")
            + source_index
        )
    names.append(String("__meta__.version"))
    tensors.append(ArcPointer(_i64_tensor(MINIMAX_H3_RUNTIME_CACHE_VERSION, ctx)))
    names.append(String("__meta__.kind"))
    tensors.append(ArcPointer(_u8_tensor(kind, ctx)))
    names.append(String("__meta__.src_path"))
    tensors.append(ArcPointer(_u8_tensor(source_index, ctx)))
    names.append(String("__meta__.src_size"))
    tensors.append(ArcPointer(_i64_tensor(sm[0], ctx)))
    names.append(String("__meta__.src_mtime"))
    tensors.append(ArcPointer(_i64_tensor(sm[1], ctx)))


def _check_common_metadata(
    st: SafeTensors, kind: String, source_index: String
) raises:
    var sm = _stat_size_mtime(source_index)
    if sm[0] < 0:
        raise Error(
            String("MiniMax-H3 runtime cache cannot stat source index: ")
            + source_index
        )
    if _meta_i64(st, String("__meta__.version"), -1) \
        != MINIMAX_H3_RUNTIME_CACHE_VERSION:
        raise Error("MiniMax-H3 runtime cache version mismatch")
    if _meta_string(st, String("__meta__.kind")) != kind:
        raise Error("MiniMax-H3 runtime cache kind mismatch")
    if _meta_string(st, String("__meta__.src_path")) != source_index:
        raise Error("MiniMax-H3 runtime cache source path mismatch")
    if _meta_i64(st, String("__meta__.src_size"), -1) != sm[0] \
        or _meta_i64(st, String("__meta__.src_mtime"), -1) != sm[1]:
        raise Error("MiniMax-H3 runtime cache source index changed")


def _load_raw_h2d(
    st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var view = from_parts(info.dtype, info.shape.copy(), bytes)
    var nbytes = view.nbytes()
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var dst = BytePtr(unsafe_from_address=Int(host.unsafe_ptr()))
    var src = BytePtr(unsafe_from_address=Int(view.data.unsafe_ptr()))
    _ = sys_memcpy(dst, src, nbytes)
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, view.shape.copy(), view.dtype)


def _load_raw_h2d_into(
    st: SafeTensors, name: String, mut dst: Tensor, ctx: DeviceContext
) raises:
    """Reload one cached tensor into an existing GPU allocation."""
    var info = st.tensor_info(name)
    if info.dtype != dst.dtype() or info.shape != dst.shape():
        raise Error(
            String("MiniMax-H3 reusable resident tensor mismatch: ") + name
        )
    var bytes = st.tensor_bytes(name)
    var view = from_parts(info.dtype, info.shape.copy(), bytes)
    var nbytes = view.nbytes()
    if nbytes != dst.nbytes():
        raise Error(
            String("MiniMax-H3 reusable resident byte mismatch: ") + name
        )
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var host_dst = BytePtr(unsafe_from_address=Int(host.unsafe_ptr()))
    var src = BytePtr(unsafe_from_address=Int(view.data.unsafe_ptr()))
    _ = sys_memcpy(host_dst, src, nbytes)
    ctx.enqueue_copy(dst_buf=dst.buf, src_buf=host)
    ctx.synchronize()


def _block_prefix(layer: Int) -> String:
    return String("block.") + String(layer) + String(".")


def save_minimax_h3_resident_cache(
    resident: MiniMaxH3ResidentFp8,
    source_index: String,
    cache_path: String,
    ctx: DeviceContext,
) raises:
    if (
        resident.scheme != MINIMAX_H3_RESIDENT_INT8
        and resident.scheme != MINIMAX_H3_RESIDENT_INT8_W8A8
    ):
        raise Error("MiniMax-H3 runtime cache only accepts INT8 resident schemes")
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    var kind = String("resident-int8-groupwise")
    if resident.scheme == MINIMAX_H3_RESIDENT_INT8_W8A8:
        kind = String("resident-int8-w8a8-row")
    _append_common_metadata(names, tensors, kind, source_index, ctx)
    names.append(String("__meta__.start_layer"))
    tensors.append(ArcPointer(_i64_tensor(resident.start_layer, ctx)))
    names.append(String("__meta__.nblocks"))
    tensors.append(ArcPointer(_i64_tensor(len(resident.blocks), ctx)))
    names.append(String("__meta__.qkv_group"))
    tensors.append(ArcPointer(_i64_tensor(MINIMAX_H3_INT8_QKV_GROUP_SIZE, ctx)))
    names.append(String("__meta__.out_group"))
    tensors.append(ArcPointer(_i64_tensor(MINIMAX_H3_INT8_OUT_GROUP_SIZE, ctx)))
    names.append(String("__meta__.fc1_group"))
    tensors.append(ArcPointer(_i64_tensor(MINIMAX_H3_INT8_FC1_GROUP_SIZE, ctx)))
    names.append(String("__meta__.fc2_group"))
    tensors.append(ArcPointer(_i64_tensor(MINIMAX_H3_INT8_FC2_GROUP_SIZE, ctx)))

    for i in range(len(resident.blocks)):
        ref block = resident.blocks[i]
        var prefix = _block_prefix(resident.start_layer + i)
        for k in range(len(block.fp8)):
            names.append(prefix + String("weight.") + String(k))
            tensors.append(block.fp8[k].copy())
            names.append(prefix + String("scale.") + String(k))
            tensors.append(block.scale[k].copy())
        for k in range(len(block.bf16)):
            names.append(prefix + String("keep.") + String(k))
            tensors.append(block.bf16[k].copy())
    save_safetensors(names, tensors, cache_path, ctx)


def load_minimax_h3_resident_cache(
    cache_path: String,
    source_index: String,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
    nblocks: Int,
    start_layer: Int = 0,
    scheme: Int = MINIMAX_H3_RESIDENT_INT8,
    direct_groupwise: Bool = False,
) raises -> MiniMaxH3ResidentFp8:
    var st = SafeTensors.open(cache_path)
    var kind = String("resident-int8-groupwise")
    if scheme == MINIMAX_H3_RESIDENT_INT8_W8A8:
        kind = String("resident-int8-w8a8-row")
    _check_common_metadata(st, kind, source_index)
    # A cache covering a larger contiguous range is also a valid source for
    # any contained subrange.  This lets the product keep a resident prefix
    # and stream already-quantized W8A8 tail blocks from the same full cache,
    # without ever staging their much larger BF16 checkpoint weights.
    var cached_start = _meta_i64(st, String("__meta__.start_layer"), -1)
    var cached_blocks = _meta_i64(st, String("__meta__.nblocks"), -1)
    if (
        cached_start < 0
        or cached_blocks < 0
        or cached_start > start_layer
        or cached_start + cached_blocks < start_layer + nblocks
    ):
        raise Error("MiniMax-H3 resident cache layer range mismatch")
    if scheme == MINIMAX_H3_RESIDENT_INT8:
        if _meta_i64(st, String("__meta__.qkv_group"), -1) \
            != MINIMAX_H3_INT8_QKV_GROUP_SIZE \
            or _meta_i64(st, String("__meta__.out_group"), -1) \
            != MINIMAX_H3_INT8_OUT_GROUP_SIZE \
            or _meta_i64(st, String("__meta__.fc1_group"), -1) \
            != MINIMAX_H3_INT8_FC1_GROUP_SIZE \
            or _meta_i64(st, String("__meta__.fc2_group"), -1) \
            != MINIMAX_H3_INT8_FC2_GROUP_SIZE:
            raise Error("MiniMax-H3 resident cache group profile mismatch")

    # Scratch must be allocated before the 16 GiB store, exactly like the
    # fresh-build path, to preserve the measured fragmentation headroom.
    var scratch = List[ArcPointer[Tensor]]()
    if (
        nblocks > 0
        and scheme != MINIMAX_H3_RESIDENT_INT8_W8A8
        and not direct_groupwise
    ):
        scratch = minimax_h3_allocate_resident_scratch(
            config, start_layer, ctx
        )
    var blocks = List[MiniMaxH3BlockResidentFp8]()
    for i in range(nblocks):
        var layer = start_layer + i
        var prefix = _block_prefix(layer)
        var expected = minimax_h3_block_tensor_names(layer)
        var quant_names = List[String]()
        var quant = List[ArcPointer[Tensor]]()
        var scales = List[ArcPointer[Tensor]]()
        var keep_names = List[String]()
        var keeps = List[ArcPointer[Tensor]]()
        var qslot = 0
        var kslot = 0
        for k in range(len(expected)):
            ref name = expected[k]
            var shape = minimax_h3_expected_shape(name, config)
            var rows = shape[0]
            var cols = shape[1] if len(shape) == 2 else 0
            var cls = minimax_h3_fp8_class(name, rows, cols)
            if cls == H3_FP8_ROW:
                var weight_name = prefix + String("weight.") + String(qslot)
                var scale_name = prefix + String("scale.") + String(qslot)
                var wi = st.tensor_info(weight_name)
                var si = st.tensor_info(scale_name)
                if wi.dtype != STDtype.I8 or wi.shape != shape:
                    raise Error(String("MiniMax-H3 resident cache weight mismatch: ") + name)
                if scheme == MINIMAX_H3_RESIDENT_INT8_W8A8:
                    if (
                        si.dtype != STDtype.F32
                        or len(si.shape) != 1
                        or si.shape[0] != rows
                    ):
                        raise Error(String("MiniMax-H3 resident W8A8 scale mismatch: ") + name)
                else:
                    var group = minimax_h3_int8_group_size(name)
                    if (
                        si.dtype != STDtype.F16
                        or len(si.shape) != 2
                        or si.shape[0] != rows
                        or si.shape[1] != cols // group
                    ):
                        raise Error(String("MiniMax-H3 resident cache scale mismatch: ") + name)
                quant_names.append(name.copy())
                quant.append(ArcPointer(_load_raw_h2d(st, weight_name, ctx)))
                scales.append(ArcPointer(_load_raw_h2d(st, scale_name, ctx)))
                qslot += 1
            elif cls == H3_FP8_BF16_KEEP:
                var keep_name = prefix + String("keep.") + String(kslot)
                var ki = st.tensor_info(keep_name)
                if ki.dtype != STDtype.BF16 or ki.shape != shape:
                    raise Error(String("MiniMax-H3 resident cache keep mismatch: ") + name)
                keep_names.append(name.copy())
                keeps.append(ArcPointer(_load_raw_h2d(st, keep_name, ctx)))
                kslot += 1
            else:
                raise Error(String("MiniMax-H3 resident cache class mismatch: ") + name)
        blocks.append(
            MiniMaxH3BlockResidentFp8(
                quant_names^, quant^, scales^, keep_names^, keeps^
            )
        )
        st.release_to_os()
        # One-block loads are the streamed tail, not resident-cache setup.
        # Logging every tail visit emits 50 identical "1 / 1" lines per
        # evaluation and makes the product progress parser look stuck in
        # cache setup. Multi-block prefix loads retain their useful progress.
        if nblocks > 1 and ((i + 1) % 5 == 0 or i + 1 == nblocks):
            print("  resident cache: loaded block", i + 1, "/", nblocks)
    return MiniMaxH3ResidentFp8(
        blocks^, start_layer, scratch^, scheme
    )


def reload_minimax_h3_resident_w8a8_block(
    mut resident: MiniMaxH3ResidentFp8,
    cache_path: String,
    source_index: String,
    config: MiniMaxH3DiTConfig,
    layer: Int,
    ctx: DeviceContext,
) raises:
    """Refill one W8A8 block store without allocating any device tensors.

    A fresh one-block store for every streamed tail layer eventually
    fragments MAX's CUDA pool even when nvidia-smi reports several GiB free.
    The four quantized weights, scales, and small BF16 keeps have identical
    shapes in every H3 block, so one store can be filled in place and reused.
    """
    if resident.scheme != MINIMAX_H3_RESIDENT_INT8_W8A8:
        raise Error("MiniMax-H3 reusable tail requires the W8A8 scheme")
    if len(resident.blocks) != 1:
        raise Error("MiniMax-H3 reusable tail requires exactly one block")
    var st = SafeTensors.open(cache_path)
    _check_common_metadata(st, String("resident-int8-w8a8-row"), source_index)
    var cached_start = _meta_i64(st, String("__meta__.start_layer"), -1)
    var cached_blocks = _meta_i64(st, String("__meta__.nblocks"), -1)
    if layer < cached_start or layer >= cached_start + cached_blocks:
        raise Error("MiniMax-H3 reusable tail layer outside cache range")

    ref block = resident.blocks[0]
    var expected = minimax_h3_block_tensor_names(layer)
    var qslot = 0
    var kslot = 0
    var prefix = _block_prefix(layer)
    for k in range(len(expected)):
        ref name = expected[k]
        var shape = minimax_h3_expected_shape(name, config)
        var rows = shape[0]
        var cols = shape[1] if len(shape) == 2 else 0
        var cls = minimax_h3_fp8_class(name, rows, cols)
        if cls == H3_FP8_ROW:
            if qslot >= len(block.fp8):
                raise Error("MiniMax-H3 reusable tail quantized-slot mismatch")
            _load_raw_h2d_into(
                st, prefix + String("weight.") + String(qslot),
                block.fp8[qslot][], ctx,
            )
            _load_raw_h2d_into(
                st, prefix + String("scale.") + String(qslot),
                block.scale[qslot][], ctx,
            )
            block.fp8_names[qslot] = name.copy()
            qslot += 1
        elif cls == H3_FP8_BF16_KEEP:
            if kslot >= len(block.bf16):
                raise Error("MiniMax-H3 reusable tail keep-slot mismatch")
            _load_raw_h2d_into(
                st, prefix + String("keep.") + String(kslot),
                block.bf16[kslot][], ctx,
            )
            block.bf16_names[kslot] = name.copy()
            kslot += 1
        else:
            raise Error("MiniMax-H3 reusable tail cache class mismatch")
    if qslot != len(block.fp8) or kslot != len(block.bf16):
        raise Error("MiniMax-H3 reusable tail slot-count mismatch")
    resident.start_layer = layer
    st.release_to_os()


def reload_minimax_h3_resident_groupwise_block(
    mut resident: MiniMaxH3ResidentFp8,
    cache_path: String,
    source_index: String,
    config: MiniMaxH3DiTConfig,
    layer: Int,
    ctx: DeviceContext,
) raises:
    """Refill one direct-groupwise block store without device allocation."""
    if resident.scheme != MINIMAX_H3_RESIDENT_INT8:
        raise Error("MiniMax-H3 reusable groupwise tail requires INT8 groupwise")
    if len(resident.blocks) != 1:
        raise Error("MiniMax-H3 reusable groupwise tail requires one block")
    var st = SafeTensors.open(cache_path)
    _check_common_metadata(st, String("resident-int8-groupwise"), source_index)
    var cached_start = _meta_i64(st, String("__meta__.start_layer"), -1)
    var cached_blocks = _meta_i64(st, String("__meta__.nblocks"), -1)
    if layer < cached_start or layer >= cached_start + cached_blocks:
        raise Error("MiniMax-H3 reusable groupwise layer outside cache range")

    ref block = resident.blocks[0]
    var expected = minimax_h3_block_tensor_names(layer)
    var qslot = 0
    var kslot = 0
    var prefix = _block_prefix(layer)
    for k in range(len(expected)):
        ref name = expected[k]
        var shape = minimax_h3_expected_shape(name, config)
        var rows = shape[0]
        var cols = shape[1] if len(shape) == 2 else 0
        var cls = minimax_h3_fp8_class(name, rows, cols)
        if cls == H3_FP8_ROW:
            if qslot >= len(block.fp8):
                raise Error("MiniMax-H3 reusable groupwise quantized-slot mismatch")
            _load_raw_h2d_into(
                st, prefix + String("weight.") + String(qslot),
                block.fp8[qslot][], ctx,
            )
            _load_raw_h2d_into(
                st, prefix + String("scale.") + String(qslot),
                block.scale[qslot][], ctx,
            )
            block.fp8_names[qslot] = name.copy()
            qslot += 1
        elif cls == H3_FP8_BF16_KEEP:
            if kslot >= len(block.bf16):
                raise Error("MiniMax-H3 reusable groupwise keep-slot mismatch")
            _load_raw_h2d_into(
                st, prefix + String("keep.") + String(kslot),
                block.bf16[kslot][], ctx,
            )
            block.bf16_names[kslot] = name.copy()
            kslot += 1
        else:
            raise Error("MiniMax-H3 reusable groupwise cache class mismatch")
    if qslot != len(block.fp8) or kslot != len(block.bf16):
        raise Error("MiniMax-H3 reusable groupwise slot-count mismatch")
    resident.start_layer = layer
    st.release_to_os()


def save_minimax_h3_modcache(
    cache: MiniMaxH3ModCache,
    source_index: String,
    cache_path: String,
    steps: Int,
    ctx: DeviceContext,
) raises:
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    _append_common_metadata(
        names, tensors, String("adaln-modulation"), source_index, ctx
    )
    names.append(String("__meta__.steps"))
    tensors.append(ArcPointer(_i64_tensor(steps, ctx)))
    names.append(String("__meta__.distinct_timesteps"))
    tensors.append(ArcPointer(_i64_tensor(cache.distinct_timesteps, ctx)))
    names.append(String("__meta__.nblocks"))
    tensors.append(ArcPointer(_i64_tensor(len(cache.block_mod), ctx)))
    for i in range(len(cache.block_mod)):
        names.append(String("block.") + String(i))
        tensors.append(cache.block_mod[i].copy())
    names.append(String("final"))
    tensors.append(cache.final_mod.copy())
    save_safetensors(names, tensors, cache_path, ctx)


def load_minimax_h3_modcache(
    cache_path: String,
    source_index: String,
    config: MiniMaxH3DiTConfig,
    steps: Int,
    distinct_timesteps: Int,
    ctx: DeviceContext,
) raises -> MiniMaxH3ModCache:
    var st = SafeTensors.open(cache_path)
    _check_common_metadata(st, String("adaln-modulation"), source_index)
    if _meta_i64(st, String("__meta__.steps"), -1) != steps \
        or _meta_i64(st, String("__meta__.distinct_timesteps"), -1) \
        != distinct_timesteps \
        or _meta_i64(st, String("__meta__.nblocks"), -1) != config.num_layers:
        raise Error("MiniMax-H3 modulation cache schedule mismatch")
    var mods = List[ArcPointer[Tensor]]()
    var expected_shape: List[Int] = [
        distinct_timesteps * config.adaln_rows_per_timestep(),
        6 * config.hidden_size,
    ]
    for i in range(config.num_layers):
        var name = String("block.") + String(i)
        var info = st.tensor_info(name)
        if info.dtype != STDtype.BF16 or info.shape != expected_shape:
            raise Error(String("MiniMax-H3 modulation cache block mismatch: ") + String(i))
        mods.append(ArcPointer(_load_raw_h2d(st, name, ctx)))
    var final_info = st.tensor_info(String("final"))
    var final_shape: List[Int] = [
        distinct_timesteps, config.final_adaln_out_features
    ]
    if final_info.dtype != STDtype.BF16 or final_info.shape != final_shape:
        raise Error("MiniMax-H3 modulation cache final tensor mismatch")
    var final = ArcPointer(_load_raw_h2d(st, String("final"), ctx))
    return MiniMaxH3ModCache(mods^, final^, distinct_timesteps)
