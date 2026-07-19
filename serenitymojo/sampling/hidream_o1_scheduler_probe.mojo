# Compile-only probe for the HiDream-O1 scheduler. DO NOT RUN.
# Build: pixi run mojo build -I . -Xlinker -lm \
#   serenitymojo/sampling/hidream_o1_scheduler_probe.mojo -o /tmp/hdsched

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.sampling.hidream_o1_scheduler import (
    HiDreamO1Scheduler,
    default_timesteps_dev,
)


def main() raises:
    var dev = HiDreamO1Scheduler.dev_28step()
    if dev.num_inference_steps() != 28:
        raise Error("dev must have 28 steps")
    if not dev.needs_step_noise():
        raise Error("dev (flash) must need step noise")
    var full = HiDreamO1Scheduler.full_n_step(50, Float32(3.0))
    if full.num_inference_steps() != 50:
        raise Error("full 50-step")
    if full.needs_step_noise():
        raise Error("full (default) must NOT need step noise")
    _ = dev.timestep(0)
    _ = full.timestep(0)
    var ts = default_timesteps_dev()
    _ = ts

    # Typecheck step() behind a False guard (no GPU execution).
    if False:
        var ctx = DeviceContext()
        var vals = List[Float32]()
        for _ in range(3072):
            vals.append(Float32(0.0))
        var sh = List[Int]()
        sh.append(1); sh.append(1); sh.append(3072)
        var mo = Tensor.from_host(vals, sh.copy(), STDtype.BF16, ctx)
        var z = Tensor.from_host(vals, sh.copy(), STDtype.BF16, ctx)
        var noise = Tensor.from_host(vals, sh.copy(), STDtype.F32, ctx)
        var out = dev.step(mo, 0, z, noise, Float32(7.5), Float32(2.5), ctx)
        _ = out

    print("hidream_o1 scheduler probe compiled")
