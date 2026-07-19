# krea2_vae_parity_probe.mojo — krea2 VAE DECODE parity vs the torch oracle.
#
# krea2 (ai-toolkit krea2.py) decodes its 16-ch latents with the Qwen-Image VAE
# (AutoencoderKLQwenImage, "Qwen/Qwen-Image"), denorm = z * std + mean. The Mojo
# QwenImageVaeDecoder already implements exactly that: its decode() does
#   z = z / inv_std + mean   with inv_std = 1/std   ==   z * std + mean
# and the hardcoded _vae_mean()/_vae_std() are byte-identical to the
# "Qwen/Qwen-Image" config (verified in krea2vae_meta_32x32.json). So the SAME
# decoder is reused — no krea2-specific decoder is needed.
#
# This loads the FIXED latent the oracle dumped (krea2vae_latent_32x32.bin,
# [1,16,32,32]), runs the Mojo decoder loaded from the real Qwen/Qwen-Image VAE
# dir, and compares the decoded RGB to the oracle (krea2vae_rgb_32x32.bin,
# [1,3,256,256]) with cos (ParityHarness 0.999) + PSNR. FAIL-LOUD on mismatch.
#
# Resident/display-safe: 32x32 latent -> 256x256 image, VAE only (no DiT).
#
# Run: pixi run mojo run -I . \
#   serenitymojo/models/vae/parity/krea2_vae_parity_probe.mojo
# DEV-ONLY: Python never runs here; the .bin files are static host references.

from std.gpu.host import DeviceContext
from std.math import log10
from std.memory import alloc

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder


# The VAE krea2.py loads (AutoencoderKLQwenImage "Qwen/Qwen-Image" subfolder=vae),
# resolved to the local HF snapshot. Same dir the oracle decoded with.
comptime VAE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen-Image/"
    "snapshots/75e0b4be04f60ec59a75f475837eced720f823b6/vae"
)
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


def _psnr(actual: List[Float32], reference: List[Float32]) raises -> Float64:
    """PSNR over the [-1,1] RGB signal (peak-to-peak = 2.0)."""
    var n = len(actual)
    if n != len(reference) or n == 0:
        raise Error("psnr: bad lengths")
    var mse: Float64 = 0.0
    for i in range(n):
        var d = Float64(actual[i]) - Float64(reference[i])
        mse += d * d
    mse /= Float64(n)
    if mse <= 0.0:
        return Float64(99.0)
    # peak = 2.0 (signal range [-1,1]); PSNR = 10*log10(peak^2 / mse)
    return Float64(10.0) * log10(Float64(4.0) / mse)


def main() raises:
    var ctx = DeviceContext()
    comptime LH = 32
    comptime LW = 32
    comptime OH = 8 * LH
    comptime OW = 8 * LW

    print("[parity] krea2 VAE decode — loading Qwen-Image VAE from", VAE_DIR)
    var vae = QwenImageVaeDecoder[LH, LW].load(String(VAE_DIR), ctx)
    print("[parity] decoder loaded")

    # Load the oracle's fixed latent [1,16,LH,LW] (f32), feed to the decoder in
    # BF16 — the dtype krea2's VAE (and this decoder) runs in.
    var latvals = _read_f32_bin(String(PARITY_DIR) + "/krea2vae_latent_32x32.bin")
    var lsh = List[Int]()
    lsh.append(1)
    lsh.append(16)
    lsh.append(LH)
    lsh.append(LW)
    var latent = Tensor.from_host(latvals, lsh^, STDtype.BF16, ctx)

    print("[parity] decoding ...")
    var rgb = vae.decode(latent, ctx)  # NCHW [1,3,OH,OW], [-1,1]
    var osh = rgb.shape()
    print("[parity] rgb out shape:", osh[0], osh[1], osh[2], osh[3])
    if osh[0] != 1 or osh[1] != 3 or osh[2] != OH or osh[3] != OW:
        raise Error("krea2 VAE decode: unexpected RGB shape")

    var refv = _read_f32_bin(String(PARITY_DIR) + "/krea2vae_rgb_32x32.bin")
    var rgb_host = rgb.to_host(ctx)
    if len(rgb_host) != len(refv):
        raise Error(
            String("rgb length mismatch: mojo=")
            + String(len(rgb_host))
            + " ref="
            + String(len(refv))
        )

    var harness = ParityHarness(0.999)
    var res = harness.compare_host(rgb_host, refv)
    var psnr = _psnr(rgb_host, refv)
    print("[parity] krea2 VAE DECODE (RGB):", res)
    print("[parity] PSNR (dB):", psnr)

    if not res.passed:
        raise Error(
            String("krea2 VAE decode parity FAILED: cos=")
            + String(res.cos)
            + " < 0.999 (PSNR="
            + String(psnr)
            + ")"
        )
    print("[parity] krea2 VAE decode parity PASS (cos>=0.999)")
