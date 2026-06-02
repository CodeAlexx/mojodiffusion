# lance_t2v_smoke.mojo - compile/run gate for shared Lance sampling helpers.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.sampling.lance_t2v import (
    build_lance_timestep_schedule,
    lance_cfg,
    lance_cfg_renorm,
    lance_denoise_step,
    lance_timestep_tensor,
)


def _abs(x: Float32) -> Float32:
    return x if x >= Float32(0.0) else -x


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var diff = _abs(got - expected)
    print("[lance-sampling]", name, "got=", got, "expected=", expected, "diff=", diff)
    if diff > tol:
        raise Error(String("lance sampling mismatch: ") + name)


def main() raises:
    var ctx = DeviceContext()
    var sched = build_lance_timestep_schedule(2, Float32(3.5))
    _check_close(String("schedule[0]"), sched[0], Float32(1.0), Float32(1.0e-6))
    _check_close(String("schedule[1]"), sched[1], Float32(0.7777778), Float32(1.0e-5))
    _check_close(String("schedule[2]"), sched[2], Float32(0.0), Float32(1.0e-6))

    var sh = List[Int]()
    sh.append(2)
    sh.append(2)
    var uncond = Tensor.from_host([1.0, 2.0, 3.0, 4.0], sh.copy(), STDtype.F32, ctx)
    var cond = Tensor.from_host([2.0, 4.0, 6.0, 8.0], sh.copy(), STDtype.F32, ctx)
    var guided = lance_cfg(uncond, cond, Float32(4.0), ctx)
    var vals = guided.to_host(ctx)
    _check_close(String("cfg[0]"), vals[0], Float32(5.0), Float32(1.0e-6))
    _check_close(String("cfg[3]"), vals[3], Float32(20.0), Float32(1.0e-6))

    var renormed = lance_cfg_renorm(guided, cond, Float32(0.0), Float32(1.0), ctx)
    var rv = renormed.to_host(ctx)
    var cond_norm = sqrt(Float32(2.0*2.0 + 4.0*4.0 + 6.0*6.0 + 8.0*8.0))
    var guided_norm = sqrt(Float32(5.0*5.0 + 10.0*10.0 + 15.0*15.0 + 20.0*20.0))
    var scale = cond_norm / (guided_norm + Float32(1.0e-8))
    _check_close(String("renorm[3]"), rv[3], Float32(20.0) * scale, Float32(2.0e-5))

    var step = lance_denoise_step(cond, uncond, Float32(0.25), ctx)
    var sv = step.to_host(ctx)
    _check_close(String("step[0]"), sv[0], Float32(1.75), Float32(1.0e-6))
    _check_close(String("step[3]"), sv[3], Float32(7.0), Float32(1.0e-6))

    var tt = lance_timestep_tensor(3, sched[1], ctx)
    var tv = tt.to_host(ctx)
    _check_close(String("timestep_tensor[2]"), tv[2], sched[1], Float32(1.0e-6))
