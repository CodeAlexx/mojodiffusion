# models/zimage/zimage_decode_latent.mojo — PROCESS-SEPARATED latent → PNG decode.
#
# WHY (FT_INLINE_SAMPLING_PLAN_2026-07-08, model #2): the zimage FULL-FT arm's
# inline sampler completes its training-bucket denoise within 16GB, but the
# trainer holds ~14.7GB resident at cadence (12.3GB live resident weights +
# pool blocks fragmented around them — cuMemPoolTrimTo reclaims ~nothing, the
# measured serve-worker physics), so the in-process VAE decode OOMs. The
# sampler persists each sample's latent as <out.png>.lat.bin
# (io/cap_cache save_tensor_bin, dtype-preserving — BF16 [1,16,LH,LW]); this
# CLI decodes it in a FRESH process (virgin GPU pool — decode transients
# always fit). Mirrors models/krea2/krea2_decode_latent.mojo.
#
# usage: zimage_decode_latent <latent.bin> <vae_dir> <out.png>
#   latent.bin: BF16 (or F32) [1,16,LH,LW]; LH/LW dispatch the comptime
#   decoder shape — the trainer's 512-class buckets 72x56 and 64x64 are wired.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.models.vae.zimage_decoder import ZImageDecoder
from serenitymojo.image.png import save_png, ValueRange


def _cast(t: Tensor, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var hh = t.to_host(ctx)
    return Tensor.from_host(hh, t.shape(), dt, ctx)


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error("usage: zimage_decode_latent <latent.bin> <vae_dir> <out.png>")
    var lat_path = String(args[1])
    var vae_dir = String(args[2])
    var out_png = String(args[3])

    var ctx = DeviceContext()
    var latent = load_tensor_bin(lat_path, ctx)
    var sh = latent.shape()
    if len(sh) != 4 or sh[0] != 1 or sh[1] != 16:
        raise Error("zimage_decode_latent: expected latent [1,16,LH,LW]")
    print("[zimage-decode] latent", lat_path, " shape [", sh[0], ",", sh[1], ",",
          sh[2], ",", sh[3], "] -> ", out_png)
    # BF16 boundary for the decoder; BF16→F32→BF16 host round-trip is lossless
    # (the same _render_cast boundary the trainer render uses).
    var latent_bf16 = _cast(latent, STDtype.BF16, ctx)
    if sh[2] == 72 and sh[3] == 56:
        var dec = ZImageDecoder[72, 56].load(vae_dir, ctx)
        var img = dec.decode(latent_bf16, ctx)
        save_png(img, out_png, ctx, ValueRange.SIGNED)
    elif sh[2] == 64 and sh[3] == 64:
        var dec = ZImageDecoder[64, 64].load(vae_dir, ctx)
        var img = dec.decode(latent_bf16, ctx)
        save_png(img, out_png, ctx, ValueRange.SIGNED)
    else:
        raise Error(
            "zimage_decode_latent: only the trainer's 72x56 and 64x64 latent "
            "buckets are wired"
        )
    print("[zimage-decode] wrote", out_png)
