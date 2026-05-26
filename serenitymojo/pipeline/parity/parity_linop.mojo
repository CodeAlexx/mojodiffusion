# parity_linop.mojo — ISOLATE the final Linear. Feed the ORACLE's own
# fl_scaled (bf16) through my linear op + final-layer weights, compare to the
# oracle's after_final_layer (IMAGE tokens). If near-perfect, my linear op is
# fine and the velocity gap is purely projecting the upstream 0.99995 fl_scaled
# difference. If still ~0.9997, my linear op itself diverges from torch Linear.
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.models.dit.zimage_dit import NextDiT

comptime TRANSFORMER = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
)
comptime BD = "/home/alex/mojodiffusion/serenitymojo/pipeline/parity/blkdbg"
comptime HL = 32
comptime WL = 32
comptime CAPLEN = 173


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


def _cos_rows(a: List[Float32], b: List[Float32], d: Int, r0: Int, r1: Int) -> Float64:
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for r in range(r0, r1):
        for c in range(d):
            var av = Float64(a[r * d + c])
            var bv = Float64(b[r * d + c])
            dot += av * bv
            na += av * av
            nb += bv * bv
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def main() raises:
    var ctx = DeviceContext()
    var dit = NextDiT[HL, WL, CAPLEN].load(TRANSFORMER, ctx)

    # oracle fl_scaled [1,448,3840] -> bf16 device tensor (matches what the
    # diffusers Linear actually saw: bf16 input).
    var scaled = Tensor.from_host(_read(String(BD) + "/fl_scaled.bin"), [1, 448, 3840], STDtype.BF16, ctx)
    var out = dit.debug_final_linear_only(scaled, ctx)  # [1,448,64]
    var mine = out.to_host(ctx)
    var oref = _read(String(BD) + "/after_final_layer.bin")

    print("=== final Linear ISOLATION (oracle fl_scaled -> my linear vs oracle after_final) ===")
    print("  IMAGE tokens (0:256) cos:", _cos_rows(mine, oref, 64, 0, 256))
    print("  CAP   tokens (256:448) cos:", _cos_rows(mine, oref, 64, 256, 448))
    print("  ALL   tokens (0:448) cos:", _cos_rows(mine, oref, 64, 0, 448))
