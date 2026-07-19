# parity_final.mojo — drill into the FINAL LAYER sub-steps to pin which op
# drops cos from 0.99985 (unified_after_main) to 0.99783 (after_final_layer).
# Sub-steps: fl_scale (1+Linear(SiLU(adaln))), fl_norm (LayerNorm no-affine),
#   fl_scaled (fl_norm * fl_scale), then after_final_layer (Linear), out.
#
# Run: cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#        serenitymojo/pipeline/parity/parity_final.mojo
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


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness(0.9998)
    var x = Tensor.from_host(_read(String(PD) + "/lat_step_27.bin"), [1, 16, HL, WL], STDtype.BF16, ctx)
    var cap = Tensor.from_host(_read(String(PD) + "/cond.bin"), [CAPLEN, HIDDEN], STDtype.BF16, ctx)
    var dit = NextDiT[HL, WL, CAPLEN].load(TRANSFORMER, ctx)

    print("=== FINAL LAYER sub-step parity (lat_step_27, t=", T0, ") ===")
    var m29 = dit.debug_main_layer(x, T0, cap, 29, ctx)
    print("  unified_after_main (L29)   ", h.compare(m29, _read(String(BD) + "/unified_after_layer_29.bin"), ctx))

    var sc = dit.debug_final_sub(x, T0, cap, 0, ctx)
    print("  fl_scale (1+Lin(SiLU))     ", h.compare(sc, _read(String(BD) + "/fl_scale.bin"), ctx))

    var nm = dit.debug_final_sub(x, T0, cap, 1, ctx)
    print("  fl_norm (LayerNorm)        ", h.compare(nm, _read(String(BD) + "/fl_norm.bin"), ctx))

    var sd = dit.debug_final_sub(x, T0, cap, 2, ctx)
    print("  fl_scaled (norm*scale)     ", h.compare(sd, _read(String(BD) + "/fl_scaled.bin"), ctx))

    var af = dit.debug_stage(x, T0, cap, 8, ctx)
    print("  after_final_layer (Linear) ", h.compare(af, _read(String(BD) + "/after_final_layer.bin"), ctx))

    var out = dit.debug_stage(x, T0, cap, 9, ctx)
    print("  out (velocity)             ", h.compare(out, _read(String(BD) + "/out.bin"), ctx))
