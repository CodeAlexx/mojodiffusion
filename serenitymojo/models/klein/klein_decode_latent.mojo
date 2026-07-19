# models/klein/klein_decode_latent.mojo — PROCESS-SEPARATED latent → PNG decode.
#
# WHY (FT_INLINE_SAMPLING_PLAN_2026-07-08, model #3): the klein FULL-FT arm's
# inline sampler completes its 512-class denoise within 16GB and persists each
# sample's final latent TOKENS as <out.png>.lat.bin (io/cap_cache
# save_tensor_bin, F32 [N_IMG, 128]) BEFORE attempting the in-process VAE
# decode. When the in-process decode cannot fit beside the trainer's residency
# (pinned-host-store trainer + pool blocks), this CLI decodes the persisted
# latent in a FRESH process (virgin GPU pool). Mirrors
# models/zimage/zimage_decode_latent.mojo / models/krea2/krea2_decode_latent.mojo.
#
# The decode path is BYTE-IDENTICAL to the trainer's in-process attempt: both
# call sampling/klein_sample_resident.klein_decode_latent_to_png (packed-NCHW
# pack + klein_tiled_decode + save_image SIGNED) — so a fallback PNG matches an
# in-process PNG bit-for-bit (the krea2 gate precedent).
#
# usage: klein_decode_latent <latent.bin> <vae.safetensors> <out.png>
#   latent.bin: [N_IMG, 128] denoised klein latent tokens (F32 or BF16).
#   N_IMG dispatches the comptime grid: 1024 -> 512px (LH=LW=32),
#   4096 -> 1024px (LH=LW=64).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.sampling.klein_sample_resident import klein_decode_latent_to_png


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error(
            "usage: klein_decode_latent <latent.bin> <vae.safetensors> <out.png>"
        )
    var lat_path = String(args[1])
    var vae_path = String(args[2])
    var out_png = String(args[3])

    var ctx = DeviceContext()
    var latent = load_tensor_bin(lat_path, ctx)
    var sh = latent.shape()
    if len(sh) != 2 or sh[1] != 128:
        raise Error("klein_decode_latent: expected latent tokens [N_IMG, 128]")
    print("[klein-decode] latent", lat_path, " shape [", sh[0], ",", sh[1],
          "] -> ", out_png)
    if sh[0] == 1024:
        klein_decode_latent_to_png[32, 32](latent, vae_path, out_png, ctx)
        print("[klein-decode] wrote", out_png, " ( 512 x 512 )")
    elif sh[0] == 4096:
        klein_decode_latent_to_png[64, 64](latent, vae_path, out_png, ctx)
        print("[klein-decode] wrote", out_png, " ( 1024 x 1024 )")
    else:
        raise Error(
            "klein_decode_latent: only the 1024-token (512px) and 4096-token "
            "(1024px) grids are wired"
        )
