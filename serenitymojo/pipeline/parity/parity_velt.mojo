# parity_velt.mojo — my DiT velocity vs FRESH single-sample diffusers velocity
# for the exact (latent,t) pairs parity_vel_vs_t flagged. If all ~0.9999, the
# DiT is correct and the earlier 0.38 was a batched-hook reference artifact.
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
    var cap_u = Tensor.from_host(_read(String(PD) + "/uncond.bin"), [8, HIDDEN], STDtype.BF16, ctx)
    var dit_c = NextDiT[HL, WL, CAPLEN].load(TRANSFORMER, ctx)
    var dit_u = NextDiT[HL, WL, 8](dit_c.weights.copy(), dit_c.name_to_idx.copy(), dit_c.config)

    var names: List[String] = ["noise", "lat_step_06", "lat_step_13", "lat_step_20", "lat_step_27"]
    var ts: List[Float32] = [0.0, 0.05036, 0.13462, 0.30435, 0.82353]
    print("=== COND DiT velocity vs fresh diffusers (+ magnitude ratio) ===")
    for k in range(len(names)):
        var x = Tensor.from_host(_read(String(PD) + "/" + names[k] + ".bin"), [1, 16, HL, WL], STDtype.BF16, ctx)
        var v = dit_c.forward(x, ts[k], cap_c, ctx)
        var refv = _read(String(PD) + "/vf_" + String(k) + ".bin")
        var mine = v.to_host(ctx)
        var xl = _read(String(PD) + "/" + names[k] + ".bin")  # the latent
        var nm = 0.0
        var nr = 0.0
        var dxm = 0.0  # <v_mine, x>
        var dxr = 0.0  # <v_ref, x>
        var nx = 0.0
        for j in range(len(refv)):
            nm += Float64(mine[j]) * Float64(mine[j])
            nr += Float64(refv[j]) * Float64(refv[j])
            dxm += Float64(mine[j]) * Float64(xl[j])
            dxr += Float64(refv[j]) * Float64(xl[j])
            nx += Float64(xl[j]) * Float64(xl[j])
        # normalized projection of velocity onto x (inward/outward); err = mine-ref
        print(
            "  ", names[k], "t=", ts[k], "magratio=", Float32((nm / nr) ** 0.5),
            "  <vmine,x>/|x|^2=", Float32(dxm / nx), "  <vref,x>/|x|^2=", Float32(dxr / nx),
            "  err_proj=", Float32((dxm - dxr) / nx),
        )
    print("=== UNCOND DiT velocity (CAPLEN=8) vs fresh diffusers ===")
    for k in range(len(names)):
        var x = Tensor.from_host(_read(String(PD) + "/" + names[k] + ".bin"), [1, 16, HL, WL], STDtype.BF16, ctx)
        var v = dit_u.forward(x, ts[k], cap_u, ctx)
        print("  ", names[k], "t=", ts[k], "->", h.compare(v, _read(String(PD) + "/vfu_" + String(k) + ".bin"), ctx))
