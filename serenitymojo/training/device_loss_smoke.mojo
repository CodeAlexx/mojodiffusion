# device_loss_smoke.mojo — shared device MSE loss/grad smoke.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/device_loss_smoke.mojo

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.device_loss import (
    device_mse_loss_grad,
    device_mse_loss_grad_into,
    device_mse_loss_grad_into_scratch,
)


def _fabs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _close(a: Float32, b: Float32, tol: Float32) -> Bool:
    return _fabs(a - b) <= tol


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("device_loss_smoke FAILED: ") + msg)


def _check_vec(
    got: List[Float32], want: List[Float32], tol: Float32, label: String
) raises:
    _check(len(got) == len(want), label + String(" length mismatch"))
    for i in range(len(got)):
        if not _close(got[i], want[i], tol):
            raise Error(
                String("device_loss_smoke FAILED: ")
                + label
                + String("[")
                + String(i)
                + String("] got=")
                + String(got[i])
                + String(" want=")
                + String(want[i])
            )


def main() raises:
    var ctx = DeviceContext()

    var shape: List[Int] = [4]
    var pred = Tensor.from_host(
        [Float32(1.0), Float32(2.0), Float32(-1.0), Float32(0.0)],
        shape.copy(),
        STDtype.F32,
        ctx,
    )
    var target = Tensor.from_host(
        [Float32(0.0), Float32(1.0), Float32(-3.0), Float32(0.5)],
        shape.copy(),
        STDtype.F32,
        ctx,
    )
    var res = device_mse_loss_grad(pred, target, STDtype.F32, ctx)
    res.validate()
    _check(_close(res.loss, Float32(1.5625), Float32(1.0e-6)), "F32 loss")
    _check(res.scalar_readback_count == 1, "F32 scalar readback count")
    _check(res.full_tensor_readback_count == 0, "F32 full tensor readback count")
    _check(res.sync_count == 1, "F32 sync count")
    _check(res.d_pred.dtype() == STDtype.F32, "F32 grad dtype")
    _check_vec(
        res.d_pred.to_host(ctx),
        [Float32(0.5), Float32(0.5), Float32(1.0), Float32(-0.25)],
        Float32(1.0e-6),
        String("F32 grad"),
    )
    var grad_out = Tensor.from_host(
        [Float32(9.0), Float32(9.0), Float32(9.0), Float32(9.0)],
        shape.copy(),
        STDtype.F32,
        ctx,
    )
    var into = device_mse_loss_grad_into(pred, target, grad_out, ctx)
    into.validate()
    _check(into.backend == String("device-mse-block-reduce-into"), "into backend")
    _check(_close(into.loss, Float32(1.5625), Float32(1.0e-6)), "F32 into loss")
    _check(into.scalar_readback_count == 1, "into scalar readback count")
    _check(into.full_tensor_readback_count == 0, "into full tensor readback count")
    _check(into.sync_count == 1, "into sync count")
    _check_vec(
        grad_out.to_host(ctx),
        [Float32(0.5), Float32(0.5), Float32(1.0), Float32(-0.25)],
        Float32(1.0e-6),
        String("F32 into grad_out"),
    )
    var grad_out2 = Tensor.from_host(
        [Float32(7.0), Float32(7.0), Float32(7.0), Float32(7.0)],
        shape.copy(),
        STDtype.F32,
        ctx,
    )
    var loss_scratch = Tensor.from_host([Float32(-9.0)], [1], STDtype.F32, ctx)
    var into_scratch = device_mse_loss_grad_into_scratch(
        pred, target, grad_out2, loss_scratch, ctx
    )
    into_scratch.validate()
    _check(
        into_scratch.backend == String("device-mse-block-reduce-into-scratch"),
        "into scratch backend",
    )
    _check(
        _close(into_scratch.loss, Float32(1.5625), Float32(1.0e-6)),
        "F32 into scratch loss",
    )
    _check(into_scratch.scalar_readback_count == 1, "into scratch scalar readback count")
    _check(into_scratch.full_tensor_readback_count == 0, "into scratch full tensor readback count")
    _check(into_scratch.sync_count == 1, "into scratch sync count")
    _check_vec(
        grad_out2.to_host(ctx),
        [Float32(0.5), Float32(0.5), Float32(1.0), Float32(-0.25)],
        Float32(1.0e-6),
        String("F32 into scratch grad_out"),
    )

    var pred_bf = Tensor.from_host(
        [Float32(1.0), Float32(0.0), Float32(-1.0), Float32(2.0)],
        shape.copy(),
        STDtype.BF16,
        ctx,
    )
    var target_bf = Tensor.from_host(
        [Float32(0.0), Float32(1.0), Float32(-1.0), Float32(0.0)],
        shape.copy(),
        STDtype.BF16,
        ctx,
    )
    var res_bf = device_mse_loss_grad(pred_bf, target_bf, STDtype.BF16, ctx)
    res_bf.validate()
    _check(_close(res_bf.loss, Float32(1.5), Float32(1.0e-6)), "BF16 loss")
    _check(res_bf.d_pred.dtype() == STDtype.BF16, "BF16 grad dtype")
    _check(res_bf.full_tensor_readback_count == 0, "BF16 full tensor readback count")
    _check_vec(
        res_bf.d_pred.to_host(ctx),
        [Float32(0.5), Float32(-0.5), Float32(0.0), Float32(1.0)],
        Float32(1.0e-3),
        String("BF16 grad"),
    )

    var pred_half = Tensor.from_host(
        [Float32(1.0), Float32(-1.0), Float32(3.0), Float32(-3.0)],
        shape.copy(),
        STDtype.F16,
        ctx,
    )
    var target_half = Tensor.from_host(
        [Float32(0.0), Float32(0.0), Float32(1.0), Float32(-1.0)],
        shape.copy(),
        STDtype.F16,
        ctx,
    )
    var res_half_f32 = device_mse_loss_grad(pred_half, target_half, STDtype.F32, ctx)
    res_half_f32.validate()
    _check(_close(res_half_f32.loss, Float32(2.5), Float32(1.0e-6)), "F16->F32 loss")
    _check(res_half_f32.d_pred.dtype() == STDtype.F32, "F16 input F32 grad dtype")
    _check_vec(
        res_half_f32.d_pred.to_host(ctx),
        [Float32(0.5), Float32(-0.5), Float32(1.0), Float32(-1.0)],
        Float32(1.0e-6),
        String("F16 input F32 grad"),
    )

    print("PASS: device MSE loss+grad uses one scalar readback and no full tensor readback")
