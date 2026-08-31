# Real-artifact intake and bounded device-layout gate for the selected
# MiniMax-H3 FL2VA pruned INT8 ConvRot checkpoint.
#
# This gate validates the complete 932-tensor header/200 ConvRot triples, then
# loads only block 0.  Exact payload markers prove QKV was uploaded verbatim in
# [all-q;all-k;all-v] and that FC1 I8 rows plus F32 row scales were both
# half-swapped once to runtime [value;gate].  It is not projection/block
# forward-backward parity and does not make the artifact trainable.

from json.parser import loads
from max.gpu.host import DeviceContext
from std.pathlib import Path

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import BatchedTensorUploader, Tensor
from serenitymojo.training.minimax_h3.convrot_artifact import (
    MINIMAX_H3_FL2VA_CONVROT_BLOCKS,
    MINIMAX_H3_FL2VA_CONVROT_GROUP_SIZE,
    MINIMAX_H3_FL2VA_CONVROT_TRIPLES,
    MINIMAX_H3_FL2VA_FFN,
    MINIMAX_H3_FL2VA_HIDDEN,
    MINIMAX_H3_FL2VA_INNER,
    MiniMaxH3FL2VAConvRotArtifact,
)


comptime ARTIFACT = (
    "/home/alex/SwarmUI/Models/diffusion_models/"
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
)
comptime REF2VA_ARTIFACT = (
    "/home/alex/SwarmUI/Models/diffusion_models/"
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
)
comptime FIXTURE = (
    "serenitymojo/training/parity/fixtures/"
    "minimax_h3_fl2va_convrot_artifact_v1.json"
)
comptime FIXTURE_SHA256 = (
    "c0e7b8a87843c970078919e034bfd611432f89574b5fb912e22f20145a1b7111"
)
comptime SCHEMA = "serenity.minimax_h3.fl2va_convrot_artifact.v1"
comptime HEADER_SHA256 = (
    "33f7e56818346cbca53fc5a10515bf11fbc15e4c8b0141c37bdd582b900c87c0"
)
comptime _UPLOAD_SLAB_BYTES = 192 * 1024 * 1024


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(
            String("MiniMax-H3 FL2VA ConvRot artifact gate failed: ") + message
        )


def _i8_at(
    tensor: Tensor, row: Int, col: Int, cols: Int, ctx: DeviceContext
) raises -> Int:
    _check(tensor.dtype() == STDtype.I8, "marker tensor must be I8")
    var offset = row * cols + col
    _check(offset >= 0 and offset < tensor.numel(), "I8 marker in range")
    var host = ctx.enqueue_create_host_buffer[DType.uint8](1)
    var source = tensor.buf.create_sub_buffer[DType.uint8](offset, 1)
    ctx.enqueue_copy(dst_buf=host, src_buf=source)
    ctx.synchronize()
    return Int(host.unsafe_ptr().bitcast[Int8]()[0])


def _f32_bits_at(
    tensor: Tensor, row: Int, ctx: DeviceContext
) raises -> UInt32:
    _check(tensor.dtype() == STDtype.F32, "scale marker tensor must be F32")
    _check(row >= 0 and row < tensor.numel(), "F32 marker in range")
    var host = ctx.enqueue_create_host_buffer[DType.uint8](4)
    var source = tensor.buf.create_sub_buffer[DType.uint8](row * 4, 4)
    ctx.enqueue_copy(dst_buf=host, src_buf=source)
    ctx.synchronize()
    var ptr = host.unsafe_ptr()
    return UInt32(ptr[0]) | (UInt32(ptr[1]) << 8) \
        | (UInt32(ptr[2]) << 16) | (UInt32(ptr[3]) << 24)


def main() raises:
    var fixture = loads(Path(String(FIXTURE)).read_text())
    _check(fixture[String("schema")].as_string() == String(SCHEMA), "schema")
    var artifact_meta = fixture[String("artifact")]
    _check(
        artifact_meta[String("header_sha256")].as_string()
            == String(HEADER_SHA256),
        "fixture header SHA256",
    )
    var contract = fixture[String("contract")]
    _check(
        contract[String("blocks")].as_int() == MINIMAX_H3_FL2VA_CONVROT_BLOCKS,
        "fixture block count",
    )
    _check(
        contract[String("convrot_triples")].as_int()
            == MINIMAX_H3_FL2VA_CONVROT_TRIPLES,
        "fixture triple count",
    )
    _check(
        contract[String("qkv_transform")].as_string() == String("none"),
        "fixture QKV pass-through",
    )
    _check(
        contract[String("fc1_transform_count")].as_int() == 1,
        "fixture FC1 transform count",
    )

    # Identity is not shape-detectable: prove the sibling Ref2VA file is
    # rejected by the exact title/modelspec pin before any DeviceContext exists.
    var rejected_ref2va = False
    try:
        _ = MiniMaxH3FL2VAConvRotArtifact.open(String(REF2VA_ARTIFACT))
    except:
        rejected_ref2va = True
    _check(rejected_ref2va, "Ref2VA identity rejection")

    var artifact = MiniMaxH3FL2VAConvRotArtifact.open(String(ARTIFACT))
    var receipt = artifact.receipt()
    _check(receipt.tensor_count == 932, "real tensor count")
    _check(receipt.block_count == 50, "real block count")
    _check(receipt.convrot_triples == 200, "real ConvRot triples")
    _check(
        receipt.group_size == MINIMAX_H3_FL2VA_CONVROT_GROUP_SIZE,
        "real group size",
    )
    _check(
        receipt.qkv_layout == String("all_q_all_k_all_v"),
        "real QKV layout receipt",
    )
    _check(receipt.storage_fc1_layout == String("gate_value"), "raw FC1 layout")
    _check(receipt.runtime_fc1_layout == String("value_gate"), "runtime FC1 layout")

    var ctx = DeviceContext()
    var uploader = BatchedTensorUploader(_UPLOAD_SLAB_BYTES, ctx)
    var block = artifact.load_runtime_block(0, uploader, ctx)
    _check(block.layer_index == 0, "loaded layer index")
    _check(
        block.qkv.weight.dtype() == STDtype.I8
            and block.qkv.weight.shape()
                == [3 * MINIMAX_H3_FL2VA_INNER, MINIMAX_H3_FL2VA_HIDDEN],
        "QKV resident dtype/shape",
    )
    _check(
        block.qkv.weight_scale.dtype() == STDtype.F32
            and block.qkv.weight_scale.shape()
                == [3 * MINIMAX_H3_FL2VA_INNER, 1],
        "QKV scale resident dtype/shape",
    )
    _check(
        block.out_proj.weight.dtype() == STDtype.I8
            and block.out_proj.weight.shape()
                == [MINIMAX_H3_FL2VA_HIDDEN, MINIMAX_H3_FL2VA_INNER],
        "out projection resident dtype/shape",
    )
    _check(
        block.fc1.weight.dtype() == STDtype.I8
            and block.fc1.weight.shape()
                == [2 * MINIMAX_H3_FL2VA_FFN, MINIMAX_H3_FL2VA_HIDDEN],
        "FC1 runtime dtype/shape",
    )
    _check(
        block.fc1.weight_scale.dtype() == STDtype.F32
            and block.fc1.weight_scale.shape()
                == [2 * MINIMAX_H3_FL2VA_FFN, 1],
        "FC1 runtime scale dtype/shape",
    )
    _check(
        block.fc2.weight.dtype() == STDtype.I8
            and block.fc2.weight.shape()
                == [MINIMAX_H3_FL2VA_HIDDEN, MINIMAX_H3_FL2VA_FFN],
        "FC2 resident dtype/shape",
    )

    var markers = fixture[String("block0_markers")]
    var qkv_markers = markers[String("qkv_uploaded_verbatim")]
    for index in range(qkv_markers.length()):
        var marker = qkv_markers[index]
        var got = _i8_at(
            block.qkv.weight,
            marker[String("row")].as_int(),
            marker[String("col")].as_int(),
            MINIMAX_H3_FL2VA_HIDDEN,
            ctx,
        )
        _check(got == marker[String("value")].as_int(), "QKV exact marker")

    var fc1_markers = markers[String("fc1_runtime_weight")]
    for index in range(fc1_markers.length()):
        var marker = fc1_markers[index]
        var runtime_row = marker[String("runtime_row")].as_int()
        var source_row = marker[String("source_row")].as_int()
        _check(
            source_row
                == (
                    runtime_row + MINIMAX_H3_FL2VA_FFN
                    if runtime_row < MINIMAX_H3_FL2VA_FFN
                    else runtime_row - MINIMAX_H3_FL2VA_FFN
                ),
            "fixture FC1 directional mapping",
        )
        var got = _i8_at(
            block.fc1.weight,
            runtime_row,
            marker[String("col")].as_int(),
            MINIMAX_H3_FL2VA_HIDDEN,
            ctx,
        )
        _check(got == marker[String("value")].as_int(), "FC1 I8 exact marker")

    var scale_markers = markers[String("fc1_runtime_scale")]
    for index in range(scale_markers.length()):
        var marker = scale_markers[index]
        var runtime_row = marker[String("runtime_row")].as_int()
        var source_row = marker[String("source_row")].as_int()
        _check(
            source_row
                == (
                    runtime_row + MINIMAX_H3_FL2VA_FFN
                    if runtime_row < MINIMAX_H3_FL2VA_FFN
                    else runtime_row - MINIMAX_H3_FL2VA_FFN
                ),
            "fixture FC1 scale directional mapping",
        )
        var got = _f32_bits_at(block.fc1.weight_scale, runtime_row, ctx)
        _check(
            got == UInt32(marker[String("f32_bits")].as_int()),
            "FC1 F32 scale exact bits",
        )

    print("MiniMax-H3 FL2VA ConvRot artifact intake PASS")
    print("  header_sha256:", String(HEADER_SHA256))
    print("  tensors/blocks/triples:", receipt.tensor_count, receipt.block_count, receipt.convrot_triples)
    print("  QKV: upload-only [all-q;all-k;all-v] exact markers:", qkv_markers.length())
    print("  FC1: [gate;value] -> [value;gate] I8 markers:", fc1_markers.length())
    print("  FC1: paired F32 scale exact-bit markers:", scale_markers.length())
    print("  scope: intake/layout loader only; no projection or training claim")
