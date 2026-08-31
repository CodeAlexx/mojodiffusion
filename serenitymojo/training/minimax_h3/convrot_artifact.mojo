# Strict read-only intake and bounded block loader for the released MiniMax-H3
# FL2VA pruned INT8 ConvRot artifact.
#
# This is an artifact/layout contract, not an executable training path.  It
# deliberately stops after loading one block's four quantized projections.
# The ConvRot projection backward itself remains a separate parity milestone.
#
# Selected artifact identity:
#   minimax_h3_fl2va_pruned_int8_convrot.safetensors
#   modelspec.hash_sha256 =
#     0xd8d5a93cbdaad2a60b8654980424b0f1bcacba549f2ce9c516fe8d1ed8224bb5
#
# FL2VA and Ref2VA have identical shapes/counts.  Therefore shape detection is
# insufficient: `open` pins the exact FL2VA title and modelspec identity and
# rejects Ref2VA before any device allocation.
#
# Layout boundary (exactly one runtime convention):
#   * ConvRot QKV is already [all-q; all-k; all-v].  It is uploaded verbatim;
#     the legacy raw-BF16 per-head deinterleave MUST NOT be applied here.  This
#     fact is pinned to the selected artifact identity and is not inferred from
#     its shape (a permutation cannot be inferred from a shape).
#   * ConvRot FC1 storage is raw Musubi [gate; value].  Both the I8 weight rows
#     and their F32 row scales are half-swapped once on the device to Serenity's
#     runtime [value; gate].  The raw sources are private locals and the result
#     type is explicitly named `RuntimeBlock`, so this API exposes no second
#     raw-to-runtime transform boundary.
#
# All payload access remains mmap/read-only through SafeTensors.  I8/F32 bytes
# use BatchedTensorUploader.from_view_raw verbatim; Python is not a product
# dependency.

from max.gpu.host import DeviceContext
from std.collections import List
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import BatchedTensorUploader, Tensor


comptime MINIMAX_H3_FL2VA_CONVROT_TITLE = (
    "minimax_h3_fl2va_pruned_int8_convrot"
)
comptime MINIMAX_H3_FL2VA_CONVROT_MODELSPEC_SHA256 = (
    "0xd8d5a93cbdaad2a60b8654980424b0f1bcacba549f2ce9c516fe8d1ed8224bb5"
)
comptime MINIMAX_H3_FL2VA_CONVROT_BLOCKS = 50
comptime MINIMAX_H3_FL2VA_CONVROT_TRIPLES = 200
comptime MINIMAX_H3_FL2VA_CONVROT_GROUP_SIZE = 256
comptime MINIMAX_H3_FL2VA_HIDDEN = 5376
comptime MINIMAX_H3_FL2VA_HEADS = 56
comptime MINIMAX_H3_FL2VA_HEAD_DIM = 128
comptime MINIMAX_H3_FL2VA_INNER = 7168
comptime MINIMAX_H3_FL2VA_FFN = 14336
comptime MINIMAX_H3_FL2VA_ADALN_GRID = 1025
comptime MINIMAX_H3_FL2VA_ADALN_BASIS = 8

comptime _EXPECTED_TENSOR_COUNT = 932
comptime _EXPECTED_BLOCK_TENSOR_COUNT = 900
comptime _EXPECTED_I8_COUNT = 200
comptime _EXPECTED_U8_COUNT = 200
comptime _EXPECTED_F32_COUNT = 210
comptime _EXPECTED_F16_COUNT = 102
comptime _EXPECTED_BF16_COUNT = 220
comptime _CONVROT_SPEC = (
    "{\"format\": \"int8_tensorwise\", \"convrot\": true, "
    "\"convrot_groupsize\": 256}"
)
comptime _DYN1 = Layout.row_major(-1)
comptime _REORDER_BLOCK = 256


@fieldwise_init
struct MiniMaxH3FL2VAConvRotReceipt(Copyable, Movable):
    """Host-only proof that one exact FL2VA ConvRot artifact passed intake."""

    var tensor_count: Int
    var block_count: Int
    var convrot_triples: Int
    var group_size: Int
    var qkv_layout: String
    var storage_fc1_layout: String
    var runtime_fc1_layout: String


@fieldwise_init
struct MiniMaxH3ConvRotRuntimeProjection(Movable):
    """One resident I8 weight plus its F32 per-output-row scale."""

    var weight: Tensor
    var weight_scale: Tensor


@fieldwise_init
struct MiniMaxH3FL2VAConvRotRuntimeBlock(Movable):
    """Four ConvRot projection pairs in Serenity's single runtime layout.

    `qkv` is an upload-only pass-through in [all-q;all-k;all-v].  `fc1` has
    already undergone the one permitted [gate;value] -> [value;gate] swap.
    """

    var layer_index: Int
    var qkv: MiniMaxH3ConvRotRuntimeProjection
    var out_proj: MiniMaxH3ConvRotRuntimeProjection
    var fc1: MiniMaxH3ConvRotRuntimeProjection
    var fc2: MiniMaxH3ConvRotRuntimeProjection


def _projection_base(layer: Int, projection: Int) -> String:
    var prefix = String("blocks.") + String(layer)
    if projection == 0:
        return prefix + String(".attn.qkv_proj")
    if projection == 1:
        return prefix + String(".attn.out_proj")
    if projection == 2:
        return prefix + String(".mlp.fc1")
    return prefix + String(".mlp.fc2")


def _projection_shape(projection: Int) -> List[Int]:
    if projection == 0:
        return [3 * MINIMAX_H3_FL2VA_INNER, MINIMAX_H3_FL2VA_HIDDEN]
    if projection == 1:
        return [MINIMAX_H3_FL2VA_HIDDEN, MINIMAX_H3_FL2VA_INNER]
    if projection == 2:
        return [2 * MINIMAX_H3_FL2VA_FFN, MINIMAX_H3_FL2VA_HIDDEN]
    return [MINIMAX_H3_FL2VA_HIDDEN, MINIMAX_H3_FL2VA_FFN]


def _require_tensor(
    st: SafeTensors,
    name: String,
    dtype: STDtype,
    shape: List[Int],
) raises:
    if not st.has_tensor(name):
        raise Error(String("MiniMax-H3 FL2VA ConvRot missing tensor: ") + name)
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(
            String("MiniMax-H3 FL2VA ConvRot dtype mismatch for ") + name
            + String(": expected ") + dtype.name() + String(", got ")
            + info.dtype.name()
        )
    if info.shape != shape:
        raise Error(
            String("MiniMax-H3 FL2VA ConvRot shape mismatch for ") + name
        )


def _is_expected_projection_name(name: String, suffix: String) -> Bool:
    for layer in range(MINIMAX_H3_FL2VA_CONVROT_BLOCKS):
        for projection in range(4):
            if name == _projection_base(layer, projection) + suffix:
                return True
    return False


def _require_convrot_spec(st: SafeTensors, name: String) raises:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.U8 or info.shape != [72] or info.size != 72:
        raise Error(
            String("MiniMax-H3 FL2VA ConvRot malformed quant spec: ") + name
        )
    var got = st.tensor_bytes(name)
    var expected_text = String(_CONVROT_SPEC)
    var expected = expected_text.as_bytes()
    if len(got) != len(expected):
        raise Error(
            String("MiniMax-H3 FL2VA ConvRot quant spec length mismatch: ")
            + name
        )
    for i in range(len(expected)):
        if got[i] != expected[i]:
            raise Error(
                String("MiniMax-H3 FL2VA ConvRot quant spec mismatch: ")
                + name
            )


def _require_metadata(st: SafeTensors, key: String, expected: String) raises:
    if not st.has_metadata(key):
        raise Error(
            String("MiniMax-H3 FL2VA ConvRot metadata missing: ") + key
        )
    var got = st.metadata_value(key)
    if got != expected:
        raise Error(
            String("MiniMax-H3 FL2VA ConvRot metadata mismatch for ") + key
            + String(": got '") + got + String("'")
        )


def validate_minimax_h3_fl2va_convrot_artifact(
    st: SafeTensors
) raises -> MiniMaxH3FL2VAConvRotReceipt:
    """Validate identity, complete ConvRot inventory, and released geometry.

    This reads only the safetensors header plus the 200 tiny 72-byte quant
    specs.  No device context and no tensor payload allocation are involved.
    """
    _require_metadata(
        st, String("modelspec.sai_model_spec"), String("1.0.0")
    )
    _require_metadata(
        st, String("modelspec.architecture"), String("minimax-h3")
    )
    _require_metadata(
        st, String("modelspec.title"), String(MINIMAX_H3_FL2VA_CONVROT_TITLE)
    )
    _require_metadata(
        st,
        String("modelspec.hash_sha256"),
        String(MINIMAX_H3_FL2VA_CONVROT_MODELSPEC_SHA256),
    )
    if st.count() != _EXPECTED_TENSOR_COUNT:
        raise Error(
            String("MiniMax-H3 FL2VA ConvRot tensor count mismatch: expected ")
            + String(_EXPECTED_TENSOR_COUNT) + String(", got ")
            + String(st.count())
        )

    var convrot_specs = 0
    var row_scales = 0
    var i8_weights = 0
    var block_tensors = 0
    var i8_count = 0
    var u8_count = 0
    var f32_count = 0
    var f16_count = 0
    var bf16_count = 0
    var names = st.names()
    for name in names:
        var info = st.tensor_info(name)
        if info.dtype == STDtype.I8:
            i8_count += 1
        elif info.dtype == STDtype.U8:
            u8_count += 1
        elif info.dtype == STDtype.F32:
            f32_count += 1
        elif info.dtype == STDtype.F16:
            f16_count += 1
        elif info.dtype == STDtype.BF16:
            bf16_count += 1
        else:
            raise Error(
                String("MiniMax-H3 FL2VA ConvRot unsupported storage dtype in ")
                + name + String(": ") + info.dtype.name()
            )
        if name.startswith(String("blocks.")):
            block_tensors += 1
        if name.endswith(String(".comfy_quant")):
            convrot_specs += 1
            if not _is_expected_projection_name(name, String(".comfy_quant")):
                raise Error(
                    String("MiniMax-H3 FL2VA ConvRot orphan quant spec: ") + name
                )
        if name.endswith(String(".weight_scale")):
            row_scales += 1
            if not _is_expected_projection_name(name, String(".weight_scale")):
                raise Error(
                    String("MiniMax-H3 FL2VA ConvRot orphan row scale: ") + name
                )
        if info.dtype == STDtype.I8:
            i8_weights += 1
            if not _is_expected_projection_name(name, String(".weight")):
                raise Error(
                    String("MiniMax-H3 FL2VA ConvRot orphan I8 weight: ") + name
                )

    if block_tensors != _EXPECTED_BLOCK_TENSOR_COUNT:
        raise Error("MiniMax-H3 FL2VA ConvRot block tensor count mismatch")
    if convrot_specs != MINIMAX_H3_FL2VA_CONVROT_TRIPLES \
            or row_scales != MINIMAX_H3_FL2VA_CONVROT_TRIPLES \
            or i8_weights != MINIMAX_H3_FL2VA_CONVROT_TRIPLES:
        raise Error(
            "MiniMax-H3 FL2VA ConvRot requires exactly 200 complete triples"
        )
    if i8_count != _EXPECTED_I8_COUNT or u8_count != _EXPECTED_U8_COUNT \
            or f32_count != _EXPECTED_F32_COUNT \
            or f16_count != _EXPECTED_F16_COUNT \
            or bf16_count != _EXPECTED_BF16_COUNT:
        raise Error("MiniMax-H3 FL2VA ConvRot global dtype inventory mismatch")

    for layer in range(MINIMAX_H3_FL2VA_CONVROT_BLOCKS):
        var block = String("blocks.") + String(layer)
        for projection in range(4):
            var base = _projection_base(layer, projection)
            var weight_shape = _projection_shape(projection)
            _require_tensor(
                st, base + String(".weight"), STDtype.I8, weight_shape
            )
            _require_tensor(
                st,
                base + String(".weight_scale"),
                STDtype.F32,
                [weight_shape[0], 1],
            )
            var spec_name = base + String(".comfy_quant")
            _require_tensor(st, spec_name, STDtype.U8, [72])
            _require_convrot_spec(st, spec_name)

        _require_tensor(
            st, block + String(".attn.q_norm.weight"), STDtype.BF16, [128]
        )
        _require_tensor(
            st, block + String(".attn.k_norm.weight"), STDtype.BF16, [128]
        )
        _require_tensor(
            st, block + String(".norm1.weight"), STDtype.BF16,
            [MINIMAX_H3_FL2VA_HIDDEN],
        )
        _require_tensor(
            st, block + String(".norm2.weight"), STDtype.BF16,
            [MINIMAX_H3_FL2VA_HIDDEN],
        )
        _require_tensor(
            st,
            block + String(".adaln_proj.linear.weight"),
            STDtype.F16,
            [96768, MINIMAX_H3_FL2VA_ADALN_BASIS],
        )
        _require_tensor(
            st,
            block + String(".adaln_proj.linear.bias"),
            STDtype.F16,
            [96768],
        )

    # Pruned AdaLN topology: one shared [1025,8] F32 curve table, narrow
    # per-block [96768,8] projections, and narrow final [10752,8] projection.
    _require_tensor(
        st,
        String("adaln_t_table"),
        STDtype.F32,
        [MINIMAX_H3_FL2VA_ADALN_GRID, MINIMAX_H3_FL2VA_ADALN_BASIS],
    )
    _require_tensor(
        st,
        String("final_layer.adaln_proj.linear.weight"),
        STDtype.F16,
        [10752, MINIMAX_H3_FL2VA_ADALN_BASIS],
    )
    _require_tensor(
        st,
        String("final_layer.adaln_proj.linear.bias"),
        STDtype.F16,
        [10752],
    )

    return MiniMaxH3FL2VAConvRotReceipt(
        st.count(),
        MINIMAX_H3_FL2VA_CONVROT_BLOCKS,
        convrot_specs,
        MINIMAX_H3_FL2VA_CONVROT_GROUP_SIZE,
        String("all_q_all_k_all_v"),
        String("gate_value"),
        String("value_gate"),
    )


def _fc1_swap_i8_kernel(
    src: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.int8, _DYN1, MutAnyOrigin],
    total_w: Int64,
    cols_w: Int32,
):
    var i = Int(global_idx.x)
    var total = Int(total_w)
    if i < total:
        var cols = Int(cols_w)
        var row = i // cols
        var col = i % cols
        var source_row = (
            row + MINIMAX_H3_FL2VA_FFN
            if row < MINIMAX_H3_FL2VA_FFN
            else row - MINIMAX_H3_FL2VA_FFN
        )
        dst[i] = rebind[dst.element_type](
            rebind[Scalar[DType.int8]](src[source_row * cols + col])
        )


def _fc1_swap_scale_kernel(
    src: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    rows_w: Int32,
):
    var row = Int(global_idx.x)
    var rows = Int(rows_w)
    if row < rows:
        var source_row = (
            row + MINIMAX_H3_FL2VA_FFN
            if row < MINIMAX_H3_FL2VA_FFN
            else row - MINIMAX_H3_FL2VA_FFN
        )
        dst[row] = rebind[dst.element_type](
            rebind[Scalar[DType.float32]](src[source_row])
        )


def _fc1_to_runtime(
    raw_weight: Tensor,
    raw_scale: Tensor,
    ctx: DeviceContext,
) raises -> MiniMaxH3ConvRotRuntimeProjection:
    if raw_weight.dtype() != STDtype.I8 \
            or raw_weight.shape() != [
                2 * MINIMAX_H3_FL2VA_FFN, MINIMAX_H3_FL2VA_HIDDEN
            ]:
        raise Error("MiniMax-H3 FL2VA ConvRot raw FC1 I8 geometry mismatch")
    if raw_scale.dtype() != STDtype.F32 \
            or raw_scale.shape() != [2 * MINIMAX_H3_FL2VA_FFN, 1]:
        raise Error("MiniMax-H3 FL2VA ConvRot raw FC1 scale geometry mismatch")

    var total = raw_weight.numel()
    var weight_buf = ctx.enqueue_create_buffer[DType.uint8](total)
    var scale_buf = ctx.enqueue_create_buffer[DType.uint8](
        raw_scale.numel() * STDtype.F32.byte_size()
    )
    var weight_layout = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var scale_layout = RuntimeLayout[_DYN1].row_major(
        IndexList[1](raw_scale.numel())
    )
    var RawW = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(raw_weight.buf.unsafe_ptr().bitcast[Int8]())
        ),
        runtime_layout=weight_layout,
    )
    var RuntimeW = LayoutTensor[DType.int8, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.int8], MutAnyOrigin](
            unsafe_from_address=Int(weight_buf.unsafe_ptr().bitcast[Int8]())
        ),
        runtime_layout=weight_layout,
    )
    var RawS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(raw_scale.buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=scale_layout,
    )
    var RuntimeS = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        unsafe_ptr=Pointer[Scalar[DType.float32], MutAnyOrigin](
            unsafe_from_address=Int(scale_buf.unsafe_ptr().bitcast[Float32]())
        ),
        runtime_layout=scale_layout,
    )
    var weight_grid = (total + _REORDER_BLOCK - 1) // _REORDER_BLOCK
    ctx.enqueue_function[_fc1_swap_i8_kernel](
        RawW,
        RuntimeW,
        Int64(total),
        Int32(MINIMAX_H3_FL2VA_HIDDEN),
        grid_dim=weight_grid,
        block_dim=_REORDER_BLOCK,
    )
    var scale_rows = 2 * MINIMAX_H3_FL2VA_FFN
    var scale_grid = (scale_rows + _REORDER_BLOCK - 1) // _REORDER_BLOCK
    ctx.enqueue_function[_fc1_swap_scale_kernel](
        RawS,
        RuntimeS,
        Int32(scale_rows),
        grid_dim=scale_grid,
        block_dim=_REORDER_BLOCK,
    )
    return MiniMaxH3ConvRotRuntimeProjection(
        Tensor(weight_buf^, raw_weight.shape(), STDtype.I8),
        Tensor(scale_buf^, raw_scale.shape(), STDtype.F32),
    )


def _upload_projection(
    st: SafeTensors,
    base: String,
    mut uploader: BatchedTensorUploader,
    ctx: DeviceContext,
) raises -> MiniMaxH3ConvRotRuntimeProjection:
    var weight_info = st.tensor_info(base + String(".weight"))
    var scale_info = st.tensor_info(base + String(".weight_scale"))
    var weight = uploader.from_view_raw(
        from_parts(
            weight_info.dtype,
            weight_info.shape.copy(),
            st.tensor_bytes(base + String(".weight")),
        ),
        ctx,
    )
    var scale = uploader.from_view_raw(
        from_parts(
            scale_info.dtype,
            scale_info.shape.copy(),
            st.tensor_bytes(base + String(".weight_scale")),
        ),
        ctx,
    )
    return MiniMaxH3ConvRotRuntimeProjection(weight^, scale^)


struct MiniMaxH3FL2VAConvRotArtifact(Movable):
    """Validated, read-only mmap handle for the one selected FL2VA artifact."""

    var _storage: SafeTensors
    var _receipt: MiniMaxH3FL2VAConvRotReceipt

    def __init__(
        out self,
        var storage: SafeTensors,
        receipt: MiniMaxH3FL2VAConvRotReceipt,
    ):
        self._storage = storage^
        self._receipt = receipt.copy()

    @staticmethod
    def open(path: String) raises -> MiniMaxH3FL2VAConvRotArtifact:
        """Open read-only and complete all host validation before device use."""
        var storage = SafeTensors.open(path)
        var receipt = validate_minimax_h3_fl2va_convrot_artifact(storage)
        return MiniMaxH3FL2VAConvRotArtifact(storage^, receipt)

    def receipt(self) -> MiniMaxH3FL2VAConvRotReceipt:
        return self._receipt.copy()

    def load_runtime_block(
        self,
        layer: Int,
        mut uploader: BatchedTensorUploader,
        ctx: DeviceContext,
    ) raises -> MiniMaxH3FL2VAConvRotRuntimeBlock:
        """Load one validated block; QKV pass-through, FC1 one-time swap.

        The final fence is required because the raw FC1 sources are private
        temporaries.  It completes both their uploads and the queued row
        permutations before those source buffers are destroyed.
        """
        if layer < 0 or layer >= MINIMAX_H3_FL2VA_CONVROT_BLOCKS:
            raise Error("MiniMax-H3 FL2VA ConvRot layer index out of range")
        var qkv = _upload_projection(
            self._storage, _projection_base(layer, 0), uploader, ctx
        )
        var out_proj = _upload_projection(
            self._storage, _projection_base(layer, 1), uploader, ctx
        )
        var raw_fc1 = _upload_projection(
            self._storage, _projection_base(layer, 2), uploader, ctx
        )
        var fc2 = _upload_projection(
            self._storage, _projection_base(layer, 3), uploader, ctx
        )
        var fc1 = _fc1_to_runtime(
            raw_fc1.weight, raw_fc1.weight_scale, ctx
        )
        uploader.finish(ctx)
        return MiniMaxH3FL2VAConvRotRuntimeBlock(
            layer, qkv^, out_proj^, fc1^, fc2^
        )
