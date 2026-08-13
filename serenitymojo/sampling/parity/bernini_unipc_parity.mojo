# Bernini-R schedule + UniPC trajectory parity against the pinned creator path.
# Run scripts/bernini_r_scheduler_oracle.py first.

from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import alloc

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import O_RDONLY, file_size, sys_close, sys_open, sys_pread
from serenitymojo.sampling.bernini_unipc import (
    build_bernini_unipc_sigma_schedule,
    build_bernini_unipc_timesteps,
)
from serenitymojo.sampling.unipc import UniPcMultistepScheduler
from serenitymojo.tensor import Tensor


comptime REF = "/home/alex/mojodiffusion-sync/output/checks/bernini_r/scheduler_oracle/"
comptime N = 6
comptime DIM = 8


def _read_f32(name: String) raises -> List[Float32]:
    var path = String(REF) + name + String(".bin")
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open Bernini scheduler fixture: ") + path)
    var nbytes = file_size(fd)
    if nbytes <= 0 or nbytes % 4 != 0:
        _ = sys_close(fd)
        raise Error(String("invalid Bernini scheduler fixture: ") + path)
    var buf = alloc[UInt8](nbytes)
    var done = 0
    while done < nbytes:
        var got = sys_pread(fd, buf + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nbytes // 4):
        out.append(fp[i])
    buf.free()
    return out^


def _abs(v: Float64) -> Float64:
    return v if v >= 0.0 else -v


def main() raises:
    var ref_sigmas_40 = _read_f32("sigmas_40")
    var ref_timesteps_40 = _read_f32("timesteps_40")
    var sigmas_40 = build_bernini_unipc_sigma_schedule(40, 5.0, 1000)
    var timesteps_40 = build_bernini_unipc_timesteps(40, 5.0, 1000)
    if len(ref_sigmas_40) != 41 or len(ref_timesteps_40) != 40:
        raise Error("Bernini production schedule oracle shape mismatch")
    var schedule_max = 0.0
    var timestep_max = 0.0
    for i in range(41):
        var d = _abs(sigmas_40[i] - Float64(ref_sigmas_40[i]))
        if d > schedule_max:
            schedule_max = d
    for i in range(40):
        var d = _abs(Float64(timesteps_40[i] - ref_timesteps_40[i]))
        if d > timestep_max:
            timestep_max = d

    var sigmas_6 = build_bernini_unipc_sigma_schedule(N, 5.0, 1000)
    var scheduler = UniPcMultistepScheduler.from_sigmas(sigmas_6^, 2)
    var ctx = DeviceContext()
    var x = Tensor.from_host(_read_f32("x_initial"), [1, 1, DIM], STDtype.F32, ctx)
    var velocities = _read_f32("velocities")
    var reference = _read_f32("trajectory")
    var trajectory_max = 0.0
    for step in range(N):
        var vhost = List[Float32]()
        for lane in range(DIM):
            vhost.append(velocities[step * DIM + lane])
        var velocity = Tensor.from_host(vhost, [1, 1, DIM], STDtype.F32, ctx)
        x = scheduler.step(velocity, x, ctx)
        var actual = x.to_host(ctx)
        for lane in range(DIM):
            var d = _abs(Float64(actual[lane] - reference[step * DIM + lane]))
            if d > trajectory_max:
                trajectory_max = d

    print("Bernini UniPC parity:")
    print("  production sigma max_abs=", schedule_max)
    print("  production timestep max_abs=", timestep_max)
    print("  six-step trajectory max_abs=", trajectory_max)
    if schedule_max > 1.0e-8 or timestep_max != 0.0 or trajectory_max > 5.0e-5:
        raise Error("Bernini UniPC creator parity FAIL")
    print("GATE PASS Bernini UniPC schedule and trajectory match creator")
