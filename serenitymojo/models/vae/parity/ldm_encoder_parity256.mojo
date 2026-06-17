# ldm_encoder_parity.mojo — SDXL/LDM VAE encoder BF16-vs-BF16 parity gate.
#
# Loads the SDXL LDM encoder (BF16 weights, faithful to Rust to_dtype(BF16)),
# feeds the EXACT fixed image the oracle dumped (parity/ldmenc_img_256x256.bin,
# [1,3,64,64] in [-1,1], fed as BF16), and compares:
#   * MOMENTS  (NHWC [1,8,8,8], mean|logvar) vs parity/ldmenc_moments_256x256.bin
#   * MODE     (NCHW [1,4,8,8], the mean)    vs parity/ldmenc_mode_256x256.bin
# against the diffusers AutoencoderKL bf16-on-GPU oracle (ldm_encoder_oracle.py).
#
# Reports cos similarity (magnitude-blind) AND magnitude ratio |mine|/|ref| for
# BOTH. GATE: cos >= 0.999 on the moments.
#
# Run: pixi run mojo run -I . serenitymojo/models/vae/parity/ldm_encoder_parity.mojo
# DEV-ONLY: Python never runs here; the .bin files are static host references.

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.models.vae.ldm_encoder import load_sdxl_ldm_encoder


comptime SDXL_VAE = "/home/alex/.serenity/models/vaes/OfficialStableDiffusion/sdxl_vae.safetensors"
comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"


def _read_f32_bin(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var n = file_size(fd)
    if n <= 0 or n % 4 != 0:
        _ = sys_close(fd)
        raise Error(String("bad bin size for ") + path)
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


def _l2(v: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(v)):
        s += Float64(v[i]) * Float64(v[i])
    return sqrt(s)


def _mag_ratio(mine: List[Float32], reference: List[Float32]) -> Float64:
    var nr = _l2(reference)
    if nr == 0.0:
        return Float64(0.0)
    return _l2(mine) / nr


def main() raises:
    var ctx = DeviceContext()
    print("[parity] loading SDXL LDM encoder (BF16) from", SDXL_VAE)
    var enc = load_sdxl_ldm_encoder[32, 32](String(SDXL_VAE), ctx)
    print("[parity] encoder loaded")

    # Oracle's fixed image [1,3,64,64] in [-1,1], fed as BF16 (Rust path).
    var imgvals = _read_f32_bin(String(PARITY_DIR) + "/ldmenc_img_256x256.bin")
    var ish = List[Int]()
    ish.append(1); ish.append(3); ish.append(256); ish.append(256)
    var img = Tensor.from_host(imgvals, ish^, STDtype.BF16, ctx)

    print("[parity] encoding (moments + mode) ...")
    var moments = enc.encode_moments(img, ctx)   # NHWC [1,8,8,8]
    var mode = enc.encode_mean(img, ctx)          # NCHW [1,4,8,8]

    var msh = moments.shape()
    print("[parity] moments shape:", msh[0], msh[1], msh[2], msh[3], "(256x256 realistic input)")
    var dsh = mode.shape()
    print("[parity] mode shape:", dsh[0], dsh[1], dsh[2], dsh[3])

    var harness = ParityHarness(0.999)

    # MOMENTS (the gate).
    var ref_moments = _read_f32_bin(String(PARITY_DIR) + "/ldmenc_moments_256x256.bin")
    var mine_moments = moments.to_host(ctx)
    var res_m = harness.compare(moments, ref_moments, ctx)
    var mag_m = _mag_ratio(mine_moments, ref_moments)
    print("[parity] MOMENTS cos=", res_m.cos, " max_abs=", res_m.max_abs,
          " mag_ratio=", mag_m)

    # MODE (the mean latent).
    var ref_mode = _read_f32_bin(String(PARITY_DIR) + "/ldmenc_mode_256x256.bin")
    var mine_mode = mode.to_host(ctx)
    var res_d = harness.compare(mode, ref_mode, ctx)
    var mag_d = _mag_ratio(mine_mode, ref_mode)
    print("[parity] MODE    cos=", res_d.cos, " max_abs=", res_d.max_abs,
          " mag_ratio=", mag_d)

    # CEILING cross-check: Mojo-BF16 moments vs the diffusers *F32* oracle. The
    # diffusers F32-vs-BF16 self-delta on this VAE/input is moments cos~0.99882
    # (BF16 rounding through ~30 convs on a low-magnitude ramp). If our BF16 port
    # sits at the same ceiling, it is faithful — the 0.999 gate is unreachable in
    # BF16 even for diffusers-vs-itself.
    var ref_moments_f32 = _read_f32_bin(
        String(PARITY_DIR) + "/ldmenc_moments_f32_256x256.bin"
    )
    var res_mf = harness.compare(moments, ref_moments_f32, ctx)
    print("[parity] MOMENTS-vs-F32oracle cos=", res_mf.cos,
          " (diffusers F32-vs-BF16 self-ceiling ~0.99882)")

    if res_m.cos >= 0.999:
        print("[parity] GATE PASS (moments cos >= 0.999)")
    else:
        print("[parity] GATE FAIL (moments cos < 0.999)")
