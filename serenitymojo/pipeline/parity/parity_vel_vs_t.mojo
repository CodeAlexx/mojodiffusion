# parity_vel_vs_t.mojo — velocity cos vs t (no dt amplification). For sampled
# steps, feed my DiT the oracle's input latent at t=1-sigma[s], compare velocity
# directly to diffusers' velc_NN.bin. Isolates the velocity error's t-dependence.
from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.models.dit.zimage_dit import NextDiT
from serenitymojo.parity import ParityHarness

comptime TRANSFORMER = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
)
comptime PD = "/home/alex/mojodiffusion/serenitymojo/pipeline/parity"
comptime HL = 32
comptime WL = 32
comptime CAPLEN = 173
comptime HIDDEN = 2560


def _read(path: String) raises -> List[Float32]:
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


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness(0.99)
    var cap_c = Tensor.from_host(_read(String(PD) + "/cond.bin"), [CAPLEN, HIDDEN], STDtype.BF16, ctx)
    var dit_c = NextDiT[HL, WL, CAPLEN].load(TRANSFORMER, ctx)

    var steps: List[Int] = [0, 7, 14, 21, 28]
    var ts: List[Float32] = [0.0, 0.05036, 0.13462, 0.30435, 0.82353]
    # input latent before step s: noise (s=0) or lat_step_{s-1}
    var inbin: List[String] = ["noise", "lat_step_06", "lat_step_13", "lat_step_20", "lat_step_27"]

    print("=== velocity cos vs t (cond) ===")
    for k in range(len(steps)):
        var s = steps[k]
        var x = Tensor.from_host(_read(String(PD) + "/" + inbin[k] + ".bin"), [1, 16, HL, WL], STDtype.BF16, ctx)
        var v = dit_c.forward(x, ts[k], cap_c, ctx)
        var sref = String(s)
        if s < 10:
            sref = String("0") + sref
        var r = h.compare(v, _read(String(PD) + "/velc_" + sref + ".bin"), ctx)
        print("  step", s, "t=", ts[k], "->", r)
