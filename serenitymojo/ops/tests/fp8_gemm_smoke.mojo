# ops/tests/fp8_gemm_smoke.mojo - focused smoke for fused FP8 GEMM.
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/ops/tests/fp8_gemm_smoke.mojo -o /tmp/fp8_gemm_smoke

from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.fp8_gemm import linear_fp8


def _require(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(msg)


def _u8_to_tensor(
    bytes: List[UInt8], var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    _require(n == len(bytes), "_u8_to_tensor: shape/byte count mismatch")
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
    var hp = host.unsafe_ptr()
    for i in range(n):
        hp[i] = bytes[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, shape^, STDtype.F8_E4M3)


def _check_values(name: String, got: List[Float32], want: List[Float32]) raises:
    _require(len(got) == len(want), name + ": length mismatch")
    for i in range(len(want)):
        if got[i] != want[i]:
            raise Error(
                name + ": mismatch at "
                + String(i) + " got " + String(got[i])
                + " want " + String(want[i])
            )


def main() raises:
    var ctx = DeviceContext()

    # x = [[1, 2, 3], [-1, 0.5, 4]]
    var x = Tensor.from_host(
        [1.0, 2.0, 3.0, -1.0, 0.5, 4.0], [2, 3], STDtype.BF16, ctx
    )
    # w rows are [[1, -1, 0.5], [2, 0, -0.5]] in E4M3 bytes.
    var w = _u8_to_tensor(
        [UInt8(56), UInt8(184), UInt8(48), UInt8(64), UInt8(0), UInt8(176)],
        [2, 3],
        ctx,
    )
    var scale = Tensor.from_host([1.0, 0.25], [2], STDtype.F32, ctx)

    var y = linear_fp8(x, w, scale, None, ctx)
    _require(y.dtype() == STDtype.BF16, "no-bias output dtype is not BF16")
    var y_shape = y.shape()
    _require(len(y_shape) == 2 and y_shape[0] == 2 and y_shape[1] == 2,
             "no-bias output shape mismatch")
    _check_values("no-bias", y.to_host(ctx), [0.5, 0.125, 0.5, -1.0])

    var bias = Tensor.from_host([0.5, -1.0], [2], STDtype.BF16, ctx)
    var yb = linear_fp8(x, w, scale, Optional[Tensor](bias^), ctx)
    _require(yb.dtype() == STDtype.BF16, "bias output dtype is not BF16")
    _check_values("bias", yb.to_host(ctx), [1.0, -0.875, 1.0, -2.0])

    print("PASS: fp8_gemm smoke")
