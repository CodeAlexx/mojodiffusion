# qwenimage_encoder_roundtrip.mojo — encode->decode round-trip cross-check.
#
# Encodes the fixed oracle image with the new QwenImageVaeEncoder, then decodes
# the mean latent with the EXISTING QwenImageVaeDecoder (Wan-key path) and
# reports mean-abs-error vs the input image. This cross-checks that the encoder
# and decoder agree on the NDHWC / zero-left-pad / unnormalize conventions.
#
# Run: pixi run mojo run -I . serenitymojo/models/vae/parity/qwenimage_encoder_roundtrip.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder


comptime VAE_FILE = (
    "/home/alex/.serenity/models/anima/split_files/vae/qwen_image_vae.safetensors"
)
comptime PARITY_DIR = (
    "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"
)


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


def main() raises:
    var ctx = DeviceContext()
    comptime IH = 128
    comptime IW = 128
    comptime LH = IH // 8
    comptime LW = IW // 8

    var enc = QwenImageVaeEncoder[IH, IW].load(String(VAE_FILE), ctx)
    var dec = QwenImageVaeDecoder[LH, LW].load_wan21_keys(String(VAE_FILE), ctx)
    print("[rt] encoder + decoder loaded")

    var imgvals = _read_f32_bin(String(PARITY_DIR) + "/qie_img_128x128.bin")
    var ish = List[Int]()
    ish.append(1); ish.append(3); ish.append(IH); ish.append(IW)
    var img = Tensor.from_host(imgvals, ish^, STDtype.BF16, ctx)

    var lat = enc.encode_mean(img, ctx)        # NCHW [1,16,LH,LW]
    var recon = dec.decode_wan21_keys(lat, ctx)  # NCHW [1,3,IH,IW]
    var rsh = recon.shape()
    print("[rt] recon shape:", rsh[0], rsh[1], rsh[2], rsh[3])

    var recon_h = recon.to_host(ctx)
    # mean abs err vs input (both [-1,1], same NCHW order)
    var n = len(imgvals)
    var finite = True
    var sae = Float64(0.0)
    for i in range(n):
        var r = recon_h[i]
        if not (r == r) or r > Float32(1e30) or r < Float32(-1e30):
            finite = False
        sae += Float64(abs(r - imgvals[i]))
    var mae = sae / Float64(n)
    print("[rt] decoded finite:", finite)
    print("[rt] mean-abs-err(recon, input):", mae)
