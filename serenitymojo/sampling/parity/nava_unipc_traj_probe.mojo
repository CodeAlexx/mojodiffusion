# NAVA UniPC scheduler trajectory parity: Mojo UniPcMultistepScheduler(shift=5)
# driven 25 steps with the torch oracle's seeded eps sequence, gated EVERY step
# (accumulation test) + final vs the torch latent trajectory.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.tensor_algebra import slice, reshape
from serenitymojo.sampling.unipc import UniPcMultistepScheduler

comptime FX = "/home/alex/mojodiffusion/serenitymojo/sampling/parity/nava_unipc_fx.safetensors"
comptime NSTEP = 25


def main() raises:
    var ctx = DeviceContext()
    print("=== NAVA UniPC scheduler trajectory parity (shift=5, 25 steps) ===")
    var fx = ShardedSafeTensors.open(FX)
    var eps_all = Tensor.from_view_as_f32(fx.tensor_view("eps"), ctx)    # [25,1280,48]
    var traj_all = Tensor.from_view(fx.tensor_view("traj"), ctx)        # [26,1280,48]

    var sch = UniPcMultistepScheduler(1000, NSTEP, 5.0, 2)

    # sample0 = traj[0]
    var sample = reshape(slice(traj_all, 0, 0, 1, ctx), [1280, 48], ctx)
    var min_cos = Float64(1.0)
    for i in range(NSTEP):
        var eps_i = reshape(slice(eps_all, 0, i, 1, ctx), [1280, 48], ctx)
        sample = sch.step(eps_i, sample, ctx)
        var ref_i = reshape(slice(traj_all, 0, i + 1, 1, ctx), [1280, 48], ctx).to_host(ctx)
        var r = ParityHarness(0.999).compare(sample, ref_i, ctx)
        if r.cos < min_cos:
            min_cos = r.cos
        if i == NSTEP - 1:
            print("  final step", i, ":", r)
    print("  min cos across all 25 steps:", min_cos)
    if min_cos >= 0.999:
        print("GATE PASS: every step cos >= 0.999")
    else:
        print("GATE FAIL: a step fell below 0.999")
