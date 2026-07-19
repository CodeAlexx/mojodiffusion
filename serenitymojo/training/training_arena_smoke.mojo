# training_arena_smoke.mojo — shared training arena smoke.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/training_arena_smoke.mojo

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.device_loss import device_mse_loss_grad_with_arena
from serenitymojo.training.training_arena import (
    TRAINING_ARENA_PHASE_LOSS,
    TrainingArena,
)


def _fabs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _close(a: Float32, b: Float32, tol: Float32) -> Bool:
    return _fabs(a - b) <= tol


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("training_arena_smoke FAILED: ") + msg)


def _check_vec(
    got: List[Float32], want: List[Float32], tol: Float32, label: String
) raises:
    _check(len(got) == len(want), label + String(" length mismatch"))
    for i in range(len(got)):
        if not _close(got[i], want[i], tol):
            raise Error(
                String("training_arena_smoke FAILED: ")
                + label
                + String("[")
                + String(i)
                + String("] got=")
                + String(got[i])
                + String(" want=")
                + String(want[i])
            )


def _expect_alloc_raises_without_counting(ctx: DeviceContext) raises:
    var tiny = TrainingArena(ctx, 16, 1, 1)
    try:
        var _buf = tiny.alloc_bytes(17)
    except e:
        print("  raised as expected [arena oversized alloc]:", String(e))
        var s = tiny.stats()
        _check(s.allocation_count == 0, "failed alloc should not increment count")
        _check(s.allocated_bytes == 0, "failed alloc should not increment bytes")
        return
    raise Error("training_arena_smoke FAILED: expected oversized arena allocation raise")


def main() raises:
    var ctx = DeviceContext()
    _expect_alloc_raises_without_counting(ctx)
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

    var arena = TrainingArena(ctx, 512, 2)
    var loss_mark = arena.mark(TRAINING_ARENA_PHASE_LOSS)
    var res = device_mse_loss_grad_with_arena(
        pred, target, STDtype.F32, arena, ctx
    )
    res.validate()
    _check(res.backend == String("device-mse-block-reduce-arena"), "backend label")
    _check(_close(res.loss, Float32(1.5625), Float32(1.0e-6)), "arena loss")
    _check(res.scalar_readback_count == 1, "arena loss scalar readback count")
    _check(res.full_tensor_readback_count == 0, "arena loss full readback count")
    _check_vec(
        res.d_pred.to_host(ctx),
        [Float32(0.5), Float32(0.5), Float32(1.0), Float32(-0.25)],
        Float32(1.0e-6),
        String("arena grad"),
    )

    var mid = arena.stats()
    _check(mid.allocation_count == 2, "loss should allocate grad and scalar scratch")
    _check(mid.allocated_bytes == 20, "raw requested bytes should be tracked")
    _check(mid.current_used_bytes > 0, "arena should report live loss scratch")
    _check(mid.peak_bytes > 0, "arena should report peak usage")
    _check(mid.host_device_transfer_count == 1, "loss scalar transfer count")
    _check(mid.sync_count == 1, "loss scalar sync count")
    _check(mid.scalar_sync_count == 1, "loss scalar sync reason")

    arena.rewind(loss_mark)

    var done = arena.stats()
    _check(done.rewind_count == 1, "rewind count")
    _check(done.current_used_bytes == 0, "rewind should restore loss mark")
    _check(done.host_device_transfer_count == 1, "transfer count")
    _check(done.sync_count == 1, "sync count")
    _check(done.scalar_sync_count == 1, "scalar sync count")
    print("PASS: training arena backs device loss scratch and tracks rewinds/syncs")
