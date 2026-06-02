# Microsoft Lens FlowMatch tensor scheduler smoke.
#
# Exercises the GPU Euler update and its BF16 delta behavior. The scalar schedule
# is covered by `lens_flowmatch_smoke.mojo`.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.sampling.lens_flowmatch import (
    LensFlowMatchScheduler,
    lens_euler_step,
)
from serenitymojo.tensor import Tensor


def _shape2() -> List[Int]:
    var sh = List[Int]()
    sh.append(2)
    sh.append(2)
    return sh^


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var diff = got - expected
    if diff < 0.0:
        diff = -diff
    if diff > tol:
        raise Error(
            name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def main() raises:
    var ctx = DeviceContext()
    print("=== Microsoft Lens FlowMatch tensor smoke ===")
    var lat_vals = List[Float32]()
    lat_vals.append(0.5)
    lat_vals.append(-0.25)
    lat_vals.append(1.0)
    lat_vals.append(-1.0)
    var pred_vals = List[Float32]()
    pred_vals.append(2.0)
    pred_vals.append(-4.0)
    pred_vals.append(0.5)
    pred_vals.append(1.0)

    var latents = Tensor.from_host(lat_vals, _shape2(), STDtype.BF16, ctx)
    var pred = Tensor.from_host(pred_vals, _shape2(), STDtype.BF16, ctx)
    var stepped = lens_euler_step(latents, pred, Float32(1.0), Float32(0.5), ctx)
    var out = stepped.to_host(ctx)
    _check_close(String("out[0]"), out[0], -0.5, 0.0001)
    _check_close(String("out[1]"), out[1], 1.75, 0.0001)
    _check_close(String("out[2]"), out[2], 0.75, 0.0001)
    _check_close(String("out[3]"), out[3], -1.5, 0.0001)

    var sched = LensFlowMatchScheduler.for_resolution(1024, 1024, 20)
    var sched_step = sched.step(latents, pred, 19, ctx)
    var sched_out = sched_step.to_host(ctx)
    if stepped.dtype() != STDtype.BF16 or sched_step.dtype() != STDtype.BF16:
        raise Error("Lens tensor step must preserve BF16 latent dtype")

    print("  manual step:", out[0], out[1], out[2], out[3])
    print("  terminal sched step first:", sched_out[0])
    print("Microsoft Lens FlowMatch tensor smoke PASS")
