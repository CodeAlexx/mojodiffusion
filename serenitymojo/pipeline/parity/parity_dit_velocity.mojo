# parity_dit_velocity.mojo — STAGE 3 (THE SIGN TEST): my NextDiT forward at the
# step-0 inputs vs diffusers' raw transformer velocity.
#   inputs: noise.bin (latent@step0), cond.bin (cap_feats), t = 1 - sigma[0] = 0.0
#   ref:    vel_cond0.bin / vel_uncond0.bin  (16,1,32,32) == [1,16,32,32] flat
# cos≈+1 → DiT matches diffusers (no negate); cos≈-1 → sign-flipped; cos≪1 → diverges.
#
# Run: cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#        serenitymojo/pipeline/parity/parity_dit_velocity.mojo
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
comptime CAPLEN_NEG = 8
comptime HIDDEN = 2560
comptime T0 = Float32(0.0)  # 1 - sigma[0] = 1 - 1.0


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
    var count = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(count):
        out.append(fp[i])
    buf.free()
    return out^


def main() raises:
    var ctx = DeviceContext()
    var harness = ParityHarness(0.99)

    # inputs (f32 bins → bf16, matching what diffusers' transformer saw)
    var noise_v = _read_f32_bin(String(PD) + "/noise.bin")
    var x = Tensor.from_host(noise_v, [1, 16, HL, WL], STDtype.BF16, ctx)
    var cond_v = _read_f32_bin(String(PD) + "/cond.bin")
    var cap_c = Tensor.from_host(cond_v, [CAPLEN, HIDDEN], STDtype.BF16, ctx)
    var unc_v = _read_f32_bin(String(PD) + "/uncond.bin")
    var cap_u = Tensor.from_host(unc_v, [CAPLEN_NEG, HIDDEN], STDtype.BF16, ctx)

    print("=== STAGE 3: DiT velocity sign test (t =", T0, ") ===")
    var dit_c = NextDiT[HL, WL, CAPLEN].load(TRANSFORMER, ctx)
    var vc = dit_c.forward(x, T0, cap_c, ctx)  # [1,16,32,32]
    var vc_ref = _read_f32_bin(String(PD) + "/vel_cond0.bin")
    print("  v_cond  vs diffusers:", harness.compare(vc, vc_ref, ctx))

    var dit_u = NextDiT[HL, WL, CAPLEN_NEG](
        dit_c.weights.copy(), dit_c.name_to_idx.copy(), dit_c.config
    )
    var vu = dit_u.forward(x, T0, cap_u, ctx)
    var vu_ref = _read_f32_bin(String(PD) + "/vel_uncond0.bin")
    print("  v_uncond vs diffusers:", harness.compare(vu, vu_ref, ctx))
