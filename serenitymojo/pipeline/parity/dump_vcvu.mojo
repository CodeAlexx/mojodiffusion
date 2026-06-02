# dump my vc, vu for lat_step_27 @ t=0.82353 → guidance-term parity check
from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, sys_pwrite, file_size, O_RDONLY, O_WRONLY, O_CREAT, O_TRUNC, BytePtr
from serenitymojo.models.dit.zimage_dit import NextDiT

comptime TRANSFORMER = "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
comptime PD = "/home/alex/mojodiffusion/serenitymojo/pipeline/parity"
comptime HL = 32
comptime WL = 32
comptime HIDDEN = 2560


def _read(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
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


def _dump(path: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var nbytes = len(h) * 4
    var buf = alloc[UInt8](nbytes)
    var fp = buf.bitcast[Float32]()
    for i in range(len(h)):
        fp[i] = h[i]
    var fd = sys_open(path, O_WRONLY | O_CREAT | O_TRUNC, Int32(0o644))
    var bp = BytePtr(unsafe_from_address=Int(buf))
    var done = 0
    while done < nbytes:
        var got = sys_pwrite(fd, bp + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    buf.free()


def main() raises:
    var ctx = DeviceContext()
    var x = Tensor.from_host(_read(String(PD) + "/lat_step_27.bin"), [1, 16, HL, WL], STDtype.BF16, ctx)
    var cap_c = Tensor.from_host(_read(String(PD) + "/cond.bin"), [173, HIDDEN], STDtype.BF16, ctx)
    var cap_u = Tensor.from_host(_read(String(PD) + "/uncond.bin"), [8, HIDDEN], STDtype.BF16, ctx)
    var dit_c = NextDiT[HL, WL, 173].load(TRANSFORMER, ctx)
    var dit_u = NextDiT[HL, WL, 8](dit_c.weights.copy(), dit_c.name_to_idx.copy(), dit_c.config)
    var vc = dit_c.forward(x, 0.82353, cap_c, ctx)
    var vu = dit_u.forward(x, 0.82353, cap_u, ctx)
    _dump(String(PD) + "/myvc4.bin", vc, ctx)
    _dump(String(PD) + "/myvu4.bin", vu, ctx)
    print("dumped myvc4, myvu4")
