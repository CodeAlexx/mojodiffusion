# parity_imgtok.mojo — image-token-only cos through the back half of the main
# stack + final layer, to show the velocity-relevant (image-token) trajectory.
# The full-seq cos was dragged down by cap tokens; this isolates the 256 image
# rows that actually become the velocity.
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
    var x = Tensor.from_host(_read(String(PD) + "/lat_step_27.bin"), [1, 16, HL, WL], STDtype.BF16, ctx)
    var cap = Tensor.from_host(_read(String(PD) + "/cond.bin"), [CAPLEN, HIDDEN], STDtype.BF16, ctx)
    var dit = NextDiT[HL, WL, CAPLEN].load(TRANSFORMER, ctx)

    print("=== IMAGE-token-only cos (rows 0:256) — the velocity carrier ===")
    var checkpoints: List[Int] = [0, 9, 19, 24, 27, 29]
    for k in range(len(checkpoints)):
        var li = checkpoints[k]
        var m = dit.debug_main_layer(x, T0, cap, li, ctx)
        var mh = m.to_host(ctx)
        var rf = _read(String(BD) + "/unified_after_layer_" + String(li) + ".bin")
        print("  unified_after_layer_", li, " IMG cos:", _cos_rows(mh, rf, 3840, 0, 256))

    var nm = dit.debug_final_sub(x, T0, cap, 1, ctx)  # fl_norm
    var nmh = nm.to_host(ctx)
    print("  fl_norm    IMG cos:", _cos_rows(nmh, _read(String(BD) + "/fl_norm.bin"), 3840, 0, 256))
    var sd = dit.debug_final_sub(x, T0, cap, 2, ctx)  # fl_scaled
    var sdh = sd.to_host(ctx)
    print("  fl_scaled  IMG cos:", _cos_rows(sdh, _read(String(BD) + "/fl_scaled.bin"), 3840, 0, 256))
    var af = dit.debug_stage(x, T0, cap, 8, ctx)      # after_final_layer
    var afh = af.to_host(ctx)
    print("  after_final IMG cos:", _cos_rows(afh, _read(String(BD) + "/after_final_layer.bin"), 64, 0, 256))
