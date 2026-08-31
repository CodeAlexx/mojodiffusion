# Bounded MiniMax-H3 ConvRot component parity against pinned Musubi/Torch.
#
# Claim: regular H4-Kronecker H256 BF16 rotation/involution, ConvRot BF16
# activation row quantization, and BF16 backward weight dequant only.  This is
# intentionally not projection parity, block parity, or training readiness.

from json.parser import loads
from json.value import JSONValue
from max.gpu.host import DeviceContext
from std.math import abs, sqrt
from std.pathlib import Path

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.minimax_h3.convrot_component import (
    minimax_h3_convrot_dequant_weight_bf16,
    minimax_h3_convrot_quantize_activation_bf16,
    minimax_h3_convrot_regular_h256,
    minimax_h3_convrot_rotate_h256,
)


comptime FIXTURE = (
    "serenitymojo/training/parity/fixtures/"
    "minimax_h3_convrot_component_v1.json"
)
comptime FIXTURE_SHA256 = (
    "2e2efff9bbd2515deda897715410f9d59a7f322166dbaf1d8cc0365145980df3"
)
comptime SCHEMA = "serenity.minimax_h3.convrot_component_oracle.v1"
comptime ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
comptime KERNEL_SOURCE = "src/musubi_tuner/modules/convrot_int8_kernels.py"
comptime UTIL_SOURCE = "src/musubi_tuner/modules/convrot_int8_utils.py"
comptime KERNEL_SHA256 = (
    "47af7db50bd4017df8fedbf2bf3de726064076fc4fb662dd7bb43a9c9c268978"
)
comptime UTIL_SHA256 = (
    "8f864adfb204c6115ae7237ef0a0debe86f56d3b6c129014e59bb4f055b5c22b"
)


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("MiniMax-H3 ConvRot component parity failed: ") + message)


def _json_f32_list(value: JSONValue) raises -> List[Float32]:
    var out = List[Float32](capacity=value.length())
    for i in range(value.length()):
        out.append(Float32(value[i].as_float()))
    return out^


def _json_i32_list(value: JSONValue) raises -> List[Int32]:
    var out = List[Int32](capacity=value.length())
    for i in range(value.length()):
        out.append(Int32(value[i].as_int()))
    return out^


def _json_shape(value: JSONValue) raises -> List[Int]:
    var out = List[Int](capacity=value.length())
    for i in range(value.length()):
        out.append(Int(value[i].as_int()))
    return out^


def _i8_tensor(
    values: List[Int32], shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    _check(n == len(values), "I8 fixture shape")
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
    for i in range(n):
        host.unsafe_ptr()[i] = UInt8(Int(values[i]) & 255)
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, shape.copy(), STDtype.I8)


def _i8_host(tensor: Tensor, ctx: DeviceContext) raises -> List[Int32]:
    _check(tensor.dtype() == STDtype.I8, "I8 readback dtype")
    var host = ctx.enqueue_create_host_buffer[DType.uint8](tensor.numel())
    ctx.enqueue_copy(dst_buf=host, src_buf=tensor.buf)
    ctx.synchronize()
    var out = List[Int32](capacity=tensor.numel())
    var ptr = host.unsafe_ptr().bitcast[Int8]()
    for i in range(tensor.numel()):
        out.append(Int32(ptr[i]))
    return out^


def _check_i8_exact(got: List[Int32], want: List[Int32], label: String) raises:
    _check(len(got) == len(want), label + String(" length"))
    var mismatches = 0
    for i in range(len(got)):
        if got[i] != want[i]:
            mismatches += 1
    print(label, " code_mismatches=", mismatches)
    _check(mismatches == 0, label + String(" exact codes"))


def _metrics(
    got: List[Float32], want: List[Float32]
) raises -> Tuple[Float32, Float32]:
    _check(len(got) == len(want), "metric length")
    var max_abs = Float32(0.0)
    var diff_sq = Float64(0.0)
    var want_sq = Float64(0.0)
    for i in range(len(got)):
        var delta = got[i] - want[i]
        var magnitude = abs(delta)
        if magnitude > max_abs:
            max_abs = magnitude
        diff_sq += Float64(delta) * Float64(delta)
        want_sq += Float64(want[i]) * Float64(want[i])
    var rel_l2 = Float32(sqrt(diff_sq) / sqrt(want_sq)) \
        if want_sq > Float64(0.0) else Float32(sqrt(diff_sq))
    return (max_abs, rel_l2)


def _check_close(
    got: List[Float32],
    want: List[Float32],
    max_abs_limit: Float32,
    rel_l2_limit: Float32,
    label: String,
) raises:
    var metrics = _metrics(got, want)
    print(label, " max_abs=", metrics[0], " rel_l2=", metrics[1])
    _check(metrics[0] <= max_abs_limit, label + String(" max_abs"))
    _check(metrics[1] <= rel_l2_limit, label + String(" rel_l2"))


def _check_exact(got: List[Float32], want: List[Float32], label: String) raises:
    _check_close(got, want, Float32(0.0), Float32(0.0), label)


def _regular_h_sign(row: Int, col: Int) -> Float32:
    var r = row
    var c = col
    var sign = Float32(1.0)
    for _ in range(4):
        if r % 4 + c % 4 == 3:
            sign = -sign
        r //= 4
        c //= 4
    return sign * Float32(0.0625)


def main() raises:
    var doc = loads(Path(String(FIXTURE)).read_text())
    _check(doc[String("schema")].as_string() == String(SCHEMA), "fixture schema")
    _check(
        doc[String("oracle_commit")].as_string() == String(ORACLE_COMMIT),
        "oracle commit",
    )
    var sources = doc[String("source_files")]
    _check(
        sources[String(KERNEL_SOURCE)].as_string() == String(KERNEL_SHA256),
        "kernel source SHA256",
    )
    _check(
        sources[String(UTIL_SOURCE)].as_string() == String(UTIL_SHA256),
        "backward source SHA256",
    )
    var geometry = doc[String("geometry")]
    _check(geometry[String("group_size")].as_int() == 256, "H256 geometry")
    _check(geometry[String("kronecker_base")].as_int() == 4, "H4 base")
    _check(geometry[String("kronecker_factors")].as_int() == 4, "H4 factors")
    var sensitivity = doc[String("sensitivity")]
    _check(
        sensitivity[String("regular_vs_sylvester_rel_l2")].as_float() > 0.25,
        "wrong Sylvester fixture sensitivity",
    )
    _check(
        sensitivity[String("wrong_f32_scale_code_mismatches")].as_int() > 0,
        "BF16 denominator fixture sensitivity",
    )
    _check(
        sensitivity[String("wrong_no_quotient_bf16_code_mismatches")].as_int() > 0,
        "BF16 quotient fixture sensitivity",
    )
    _check(
        sensitivity[String("wrong_f32_weight_dequant_mismatches")].as_int() > 0,
        "BF16 weight dequant fixture sensitivity",
    )

    var inputs = doc[String("inputs")]
    var outputs = doc[String("outputs")]
    var ctx = DeviceContext()

    # Entire resident H256 is checked, not only a few marker entries.
    var h = minimax_h3_convrot_regular_h256(ctx)
    _check(h.dtype() == STDtype.BF16, "H256 dtype")
    _check(h.shape() == [256, 256], "H256 shape")
    var h_host = h.to_host(ctx)
    for row in range(256):
        for col in range(256):
            _check(
                h_host[row * 256 + col] == _regular_h_sign(row, col),
                String("regular H256 entry ") + String(row) + String(",") + String(col),
            )

    var rotation_shape = _json_shape(geometry[String("rotation_shape")])
    var x = Tensor.from_host(
        _json_f32_list(inputs[String("rotation_x_bf16_as_f32")]),
        rotation_shape.copy(),
        STDtype.BF16,
        ctx,
    )
    var rotated = minimax_h3_convrot_rotate_h256(x, h, ctx)
    _check(rotated.dtype() == STDtype.BF16, "rotation output dtype")
    _check(rotated.shape() == rotation_shape, "rotation output shape")
    var rotated_host = rotated.to_host(ctx)
    var rotated_want = _json_f32_list(outputs[String("rotation_y_bf16_as_f32")])
    _check_close(
        rotated_host,
        rotated_want,
        Float32(0.0625),
        Float32(0.003),
        "regular H256 BF16 rotation",
    )
    var wrong = _json_f32_list(outputs[String("wrong_sylvester_y_bf16_as_f32")])
    var wrong_metrics = _metrics(rotated_host, wrong)
    print(
        "wrong Sylvester distance max_abs=", wrong_metrics[0],
        " rel_l2=", wrong_metrics[1],
    )
    _check(wrong_metrics[1] > Float32(0.25), "wrong Sylvester rejection")

    var involuted = minimax_h3_convrot_rotate_h256(rotated, h, ctx)
    _check_close(
        involuted.to_host(ctx),
        _json_f32_list(outputs[String("involution_y_bf16_as_f32")]),
        Float32(0.0625),
        Float32(0.003),
        "regular H256 BF16 involution",
    )

    var quant_shape = _json_shape(geometry[String("quant_shape")])
    var quant_x = Tensor.from_host(
        _json_f32_list(inputs[String("activation_x_bf16_as_f32")]),
        quant_shape.copy(),
        STDtype.BF16,
        ctx,
    )
    var quantized = minimax_h3_convrot_quantize_activation_bf16(quant_x, ctx)
    _check(quantized.values.dtype() == STDtype.I8, "activation code dtype")
    _check(quantized.values.shape() == quant_shape, "activation code shape")
    _check(
        quantized.scale.dtype() == STDtype.F32,
        "activation scale dtype",
    )
    _check(
        quantized.scale.shape() == [quant_shape[0], 1],
        "activation scale shape",
    )
    _check_i8_exact(
        _i8_host(quantized.values, ctx),
        _json_i32_list(outputs[String("activation_q_int8")]),
        "ConvRot BF16 activation quant",
    )
    _check_exact(
        quantized.scale.to_host(ctx),
        _json_f32_list(outputs[String("activation_scale_f32")]),
        "ConvRot activation F32 scale",
    )

    var weight_shape = _json_shape(geometry[String("weight_shape")])
    var weight_q = _i8_tensor(
        _json_i32_list(inputs[String("weight_q_int8")]),
        weight_shape.copy(),
        ctx,
    )
    var weight_scale = Tensor.from_host(
        _json_f32_list(inputs[String("weight_scale_f32")]),
        [weight_shape[0], 1],
        STDtype.F32,
        ctx,
    )
    var dequant = minimax_h3_convrot_dequant_weight_bf16(
        weight_q, weight_scale, ctx
    )
    _check(dequant.dtype() == STDtype.BF16, "weight dequant dtype")
    _check(dequant.shape() == weight_shape, "weight dequant shape")
    _check_exact(
        dequant.to_host(ctx),
        _json_f32_list(outputs[String("weight_dequant_bf16_as_f32")]),
        "BF16 backward weight dequant",
    )

    print("MiniMax-H3 ConvRot component Musubi/Torch parity PASS")
    print("bounded claim: H256/activation-quant/weight-dequant only")
