# parity_denoise.mojo — STAGE 4: per-step denoise-loop parity vs diffusers.
# Uses diffusers' EXACT sigmas (from oracle meta) to isolate loop mechanics
# (timestep, CFG, final sign, Euler step) from the sigma-schedule. Compares my
# latent after each step to the oracle's lat_step_NN.bin → first divergence
# localizes the bug.
#
# Run: cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#        serenitymojo/pipeline/parity/parity_denoise.mojo
from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.models.dit.zimage_dit import NextDiT
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar
from serenitymojo.parity import ParityHarness

comptime TRANSFORMER = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
)
comptime PD = "/home/alex/mojodiffusion/serenitymojo/pipeline/parity"
comptime HL = 32
comptime WL = 32
comptime CAPLEN = 173
comptime CAPLEN_NEG = 8
comptime HIDDEN = 2560
comptime CFG = Float32(4.0)
comptime STEPS = 30


def _read_f32_bin(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var n = file_size(fd)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(n // 4):
        out.append(fp[i])
    buf.free()
    return out^


# Host round-trip dtype cast (to_host upcasts to F32; from_host casts to target).
def _cast(t: Tensor, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var h = t.to_host(ctx)
    return Tensor.from_host(h, t.shape(), dt, ctx)


def _step_path(i: Int) raises -> String:
    var s = String(i)
    if i < 10:
        s = String("0") + s
    return String(PD) + "/lat_step_" + s + ".bin"


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness(0.99)

    # diffusers' exact sigmas (oracle meta, shift=6.0, 30 steps) — 31 values.
    var sig: List[Float32] = [
        1.000000000, 0.994082808, 0.987804890, 0.981132150,
        0.974026024, 0.966442943, 0.958333373, 0.949640274,
        0.940298557, 0.930232465, 0.919354916, 0.907563090,
        0.894736826, 0.880733907, 0.865384579, 0.848484814,
        0.829787314, 0.808988750, 0.785714269, 0.759493589,
        0.729729772, 0.695652127, 0.656250000, 0.610169470,
        0.555555522, 0.489795893, 0.409090906, 0.307692289,
        0.176470578, 0.000000000, 0.000000000,
    ]

    var cap_c = Tensor.from_host(_read_f32_bin(String(PD) + "/cond.bin"), [CAPLEN, HIDDEN], STDtype.BF16, ctx)
    var cap_u = Tensor.from_host(_read_f32_bin(String(PD) + "/uncond.bin"), [CAPLEN_NEG, HIDDEN], STDtype.BF16, ctx)

    print("=== TEACHER-FORCED CFG=4 per-step vs oracle ===")
    var dit_c = NextDiT[HL, WL, CAPLEN].load(TRANSFORMER, ctx)
    var dit_u = NextDiT[HL, WL, CAPLEN_NEG](dit_c.weights.copy(), dit_c.name_to_idx.copy(), dit_c.config)

    for i in range(STEPS):
        var t = 1.0 - sig[i]
        var x_in: List[Float32]
        if i == 0:
            x_in = _read_f32_bin(String(PD) + "/noise.bin")
        else:
            x_in = _read_f32_bin(_step_path(i - 1))
        var x = Tensor.from_host(x_in, [1, 16, HL, WL], STDtype.F32, ctx)
        var x_bf = _cast(x, STDtype.BF16, ctx)
        var vc = _cast(dit_c.forward(x_bf, t, cap_c, ctx), STDtype.F32, ctx)
        var vu = _cast(dit_u.forward(x_bf, t, cap_u, ctx), STDtype.F32, ctx)
        var pred = add(vc, mul_scalar(sub(vc, vu, ctx), CFG, ctx), ctx)  # raw vc + cfg*(vc-vu)
        pred = mul_scalar(pred, -1.0, ctx)  # diffusers pipeline negates before scheduler.step()
        var dt = sig[i + 1] - sig[i]
        var x_out = add(x, mul_scalar(pred, dt, ctx), ctx)
        var r = h.compare(x_out, _read_f32_bin(_step_path(i)), ctx)
        print("  step", i, "t=", t, "dt=", dt, "->", r)
