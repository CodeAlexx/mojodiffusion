# flux_vae_decode_parity.mojo — GATE A: REAL-weight Flux.1 AE DECODER parity vs
# the torch oracle (flux_vae_decode_oracle.py).
#
# Loads the deterministic RAW latent dumped by the oracle, runs the Mojo
# LdmVaeDecoder (REAL ae.safetensors weights, FLUX config: 16ch, scale 0.3611,
# shift 0.1159, no post_quant_conv — the rescale z/scale+shift is folded inside
# decode(), so we feed the SAME raw latent the oracle fed pre-rescale), and
# compares the decoded image against the torch reference.
#
# Metric: PSNR (dB) over pixels + mean-abs-diff + cosine. Bar: PSNR > 40 dB
# (bf16/VAE noise floor ~50-55). This is the right metric for SAME-input
# decoded pixels per the parity playbook.
#
# Run (oracle FIRST, separate command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   python3 serenitymojo/vae/parity/flux_vae_decode_oracle.py 64 64
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/vae/parity/flux_vae_decode_parity.mojo

from std.math import sqrt, log10
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.vae.ldm_decoder import load_flux1_ldm_decoder


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/vae/parity/"
comptime AE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"
comptime LH = 64
comptime LW = 64
comptime IH = 8 * LH
comptime IW = 8 * LW
comptime ZC = 16


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing oracle (run the oracle first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def main() raises:
    var ctx = DeviceContext()
    print("=== GATE A: Flux.1 AE DECODER REAL-weight parity vs torch ===")
    print("  ae:", AE_PATH)
    print("  latent [1,16,", LH, ",", LW, "] -> image [1,3,", IH, ",", IW, "]")

    # Raw latent dumped by the oracle (pre-rescale; decode() folds z/scale+shift).
    var z_h = _read_bin_f32(REF_DIR + "flux_vae_dec_z.bin")
    if len(z_h) != ZC * LH * LW:
        raise Error("latent bin size wrong: " + String(len(z_h)))
    var z = Tensor.from_host(z_h, [1, ZC, LH, LW], STDtype.F32, ctx)

    # Load REAL decoder weights and run.
    var dec = load_flux1_ldm_decoder[LH, LW](String(AE_PATH), ctx)
    var img = dec.decode(z, ctx)                   # [1,3,IH,IW]
    var ish = img.shape()
    if len(ish) != 4 or ish[0] != 1 or ish[1] != 3 or ish[2] != IH or ish[3] != IW:
        raise Error("image shape wrong: expected [1,3,IH,IW]")
    print("  image shape OK: [1,", ish[1], ",", ish[2], ",", ish[3], "]")

    var img_h = (img if img.dtype() == STDtype.F32 else img).to_host(ctx)
    for i in range(len(img_h)):
        var v = img_h[i]
        if not (v == v) or _abs(v) > 1.0e30:
            raise Error("image non-finite at " + String(i))
    print("  image all finite OK")

    # Compare vs torch reference image.
    var oracle = _read_bin_f32(REF_DIR + "flux_vae_dec_img.bin")
    if len(oracle) != len(img_h):
        raise Error("oracle/mojo size mismatch " + String(len(oracle)) + " vs " + String(len(img_h)))

    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    var sse: Float32 = 0.0
    var sad: Float32 = 0.0
    var rmax: Float32 = -1.0e30
    var rmin: Float32 = 1.0e30
    for i in range(len(oracle)):
        var a = img_h[i]
        var b = oracle[i]
        dot += a * b
        na += a * a
        nb += b * b
        var d = a - b
        sse += d * d
        sad += _abs(d)
        if b > rmax: rmax = b
        if b < rmin: rmin = b
    var n = Float32(len(oracle))
    var cos = dot / (sqrt(na) * sqrt(nb)) if (na > 0.0 and nb > 0.0) else Float32(1.0)
    var mse = sse / n
    var mad = sad / n
    var rng = rmax - rmin
    # PSNR in dB relative to the reference's actual dynamic range (peak = range).
    var psnr = Float32(99.0)
    if mse > 0.0:
        psnr = 20.0 * log10(rng) - 10.0 * log10(mse)

    print("  oracle range [", rmin, ",", rmax, "]  (peak", rng, ")")
    print("  cos          =", cos)
    print("  mean-abs-diff=", mad)
    print("  MSE          =", mse)
    print("  PSNR (dB)    =", psnr)
    if psnr < 40.0:
        raise Error("Flux VAE decode parity FAIL: PSNR " + String(psnr) + " < 40 dB")

    print("VERDICT: PASS — Flux.1 AE decoder REAL ae.safetensors, image",
          "[1,3,IH,IW] finite, PSNR vs torch =", psnr, "dB (>= 40)")
