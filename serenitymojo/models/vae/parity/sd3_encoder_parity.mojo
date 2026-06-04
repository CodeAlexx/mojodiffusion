# sd3_encoder_parity.mojo — SD3.5 embedded VAE encoder BF16 parity gate (16ch).
#
# Loads the SD3.5 embedded LDM encoder (first_stage_model.encoder.*, BF16
# weights on disk, no quant_conv), feeds the EXACT fixed image the oracle dumped
# (parity/sd3enc_img_256x256.bin, [1,3,256,256] in [-1,1], fed as BF16), and
# compares:
#   * MOMENTS  (NHWC [1,32,32,32], mean|logvar) vs sd3enc_moments_256x256.bin
#   * MODE     (NCHW [1,16,32,32], the mean)    vs sd3enc_mode_256x256.bin
# against the diffusers AutoencoderKL bf16-on-GPU oracle (sd3_encoder_oracle.py),
# the SAME embedded VAE weights via from_single_file.
#
# Reports cos similarity AND magnitude ratio |mine|/|ref| for BOTH.
# GATE: cos >= 0.999 on the moments.
#
# Run: pixi run mojo run -I . serenitymojo/models/vae/parity/sd3_encoder_parity.mojo
# DEV-ONLY: Python never runs here; the .bin files are static host references.

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.models.vae.ldm_encoder import load_sd3_embedded_ldm_encoder


comptime SD3_CKPT = "/home/alex/.serenity/models/checkpoints/stablediffusion35_medium.safetensors"
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
    print("[parity] loading SD3.5 embedded LDM encoder (BF16, 16ch) from", SD3_CKPT)
    # 256x256 image -> latent 32x32 (LH=LW=32).
    var enc = load_sd3_embedded_ldm_encoder[32, 32](String(SD3_CKPT), ctx)
    print("[parity] encoder loaded")

    # Oracle's fixed image [1,3,256,256] in [-1,1], fed as BF16 (faithful path).
    var imgvals = _read_f32_bin(String(PARITY_DIR) + "/sd3enc_img_256x256.bin")
    var ish = List[Int]()
    ish.append(1); ish.append(3); ish.append(256); ish.append(256)
    var img = Tensor.from_host(imgvals, ish^, STDtype.BF16, ctx)

    print("[parity] encoding (moments + mode) ...")
    var moments = enc.encode_moments(img, ctx)   # NHWC [1,32,32,32]
    var mode = enc.encode_mean(img, ctx)          # NCHW [1,16,32,32]

    var msh = moments.shape()
    print("[parity] moments shape:", msh[0], msh[1], msh[2], msh[3])
    var dsh = mode.shape()
    print("[parity] mode shape:", dsh[0], dsh[1], dsh[2], dsh[3])

    var harness = ParityHarness(0.999)

    # MOMENTS (the gate).
    var ref_moments = _read_f32_bin(String(PARITY_DIR) + "/sd3enc_moments_256x256.bin")
    var mine_moments = moments.to_host(ctx)
    var res_m = harness.compare(moments, ref_moments, ctx)
    var mag_m = _mag_ratio(mine_moments, ref_moments)
    print("[parity] MOMENTS cos=", res_m.cos, " max_abs=", res_m.max_abs,
          " mag_ratio=", mag_m)

    # MODE (the mean latent, first 16 channels).
    var ref_mode = _read_f32_bin(String(PARITY_DIR) + "/sd3enc_mode_256x256.bin")
    var mine_mode = mode.to_host(ctx)
    var res_d = harness.compare(mode, ref_mode, ctx)
    var mag_d = _mag_ratio(mine_mode, ref_mode)
    print("[parity] MODE    cos=", res_d.cos, " max_abs=", res_d.max_abs,
          " mag_ratio=", mag_d)

    # CEILING cross-check: Mojo-BF16 moments vs the diffusers *F32* oracle.
    var ref_moments_f32 = _read_f32_bin(
        String(PARITY_DIR) + "/sd3enc_moments_f32_256x256.bin"
    )
    var res_mf = harness.compare(moments, ref_moments_f32, ctx)
    print("[parity] MOMENTS-vs-F32oracle cos=", res_mf.cos,
          " (diffusers BF16-vs-F32 self-ceiling ~0.999998)")

    if res_m.cos >= 0.999:
        print("[parity] GATE PASS (moments cos >= 0.999)")
    else:
        print("[parity] GATE FAIL (moments cos < 0.999)")
