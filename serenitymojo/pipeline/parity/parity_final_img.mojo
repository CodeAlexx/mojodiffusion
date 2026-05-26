# parity_final_img.mojo — confirm the final-Linear localization:
#  (a) compare my full forward() out vs the established vf_4.bin AND vs the
#      oracle out.bin (should agree — sanity that debug path == forward path).
#  (b) compare after_final_layer restricted to IMAGE tokens (first 256 of 448)
#      vs cap tokens, to show the velocity-relevant error is the image slice.
from std.gpu.host import DeviceContext
from std.math import sqrt
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
comptime BD = "/home/alex/mojodiffusion/serenitymojo/pipeline/parity/blkdbg"
comptime HL = 32
comptime WL = 32
comptime CAPLEN = 173
comptime HIDDEN = 2560
comptime T0 = Float32(0.82353)


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


# cos of two host arrays (F64).
def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


# slice rows [r0,r1) of a [S, D] row-major host array.
def _rows(v: List[Float32], d: Int, r0: Int, r1: Int) -> List[Float32]:
    var out = List[Float32]()
    for r in range(r0, r1):
        for c in range(d):
            out.append(v[r * d + c])
    return out^


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness(0.9998)
    var x = Tensor.from_host(_read(String(PD) + "/lat_step_27.bin"), [1, 16, HL, WL], STDtype.BF16, ctx)
    var cap = Tensor.from_host(_read(String(PD) + "/cond.bin"), [CAPLEN, HIDDEN], STDtype.BF16, ctx)
    var dit = NextDiT[HL, WL, CAPLEN].load(TRANSFORMER, ctx)

    print("=== full forward() reconciliation + after_final image/cap split ===")
    # (a) forward path
    var v = dit.forward(x, T0, cap, ctx)
    print("  forward() out vs vf_4.bin (established ref):", h.compare(v, _read(String(PD) + "/vf_4.bin"), ctx))
    print("  forward() out vs oracle out.bin            :", h.compare(v, _read(String(BD) + "/out.bin"), ctx))

    # (b) after_final_layer image vs cap token split.
    var af = dit.debug_stage(x, T0, cap, 8, ctx)  # [1,448,64]
    var afh = af.to_host(ctx)
    var oref = _read(String(BD) + "/after_final_layer.bin")
    var af_img = _rows(afh, 64, 0, 256)
    var rf_img = _rows(oref, 64, 0, 256)
    var af_cap = _rows(afh, 64, 256, 448)
    var rf_cap = _rows(oref, 64, 256, 448)
    print("  after_final_layer IMAGE tokens (0:256) cos:", _cos(af_img, rf_img))
    print("  after_final_layer CAP   tokens (256:448) cos:", _cos(af_cap, rf_cap))
