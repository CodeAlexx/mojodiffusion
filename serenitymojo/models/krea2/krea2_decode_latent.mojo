# models/krea2/krea2_decode_latent.mojo — PROCESS-SEPARATED latent → PNG decode.
#
# WHY: the trainer's inline sampler completes 1024² denoising within 16GB but
# the in-process VAE decode can OOM in resume-sampling processes (open issue,
# 2026-07-07). The sampler now persists each sample's latent as
# <out.png>.lat.bin (io/cap_cache save_tensor_bin); this CLI decodes it in a
# FRESH process (virgin GPU pool — decode transients always fit).
#
# usage: krea2_decode_latent <latent.bin> <vae_dir> <out.png> [<size=1024>]
#   latent.bin: F32 [1,16,SIZE/8,SIZE/8] (the sampler's accumulator dtype)
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import argv
from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.image.png import save_png, ValueRange


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error("usage: krea2_decode_latent <latent.bin> <vae_dir> <out.png> [<size=1024>]")
    var lat_path = String(args[1])
    var vae_dir = String(args[2])
    var out_png = String(args[3])
    var size = 1024
    if len(args) >= 5:
        size = Int(String(args[4]))

    var ctx = DeviceContext()
    var latent = load_tensor_bin(lat_path, ctx)
    var sh = latent.shape()
    print("[krea2-decode] latent", lat_path, " shape [", sh[0], ",", sh[1], ",",
          sh[2], ",", sh[3], "] -> ", out_png)
    var latent_bf16 = torch_f32_to_bf16_rne(latent, ctx)
    if size == 1024:
        var dec = QwenImageVaeDecoder[128, 128].load(vae_dir, ctx)
        var img = dec.decode(latent_bf16, ctx)
        save_png(img, out_png, ctx, ValueRange.SIGNED)
    elif size == 512:
        var dec = QwenImageVaeDecoder[64, 64].load(vae_dir, ctx)
        var img = dec.decode(latent_bf16, ctx)
        save_png(img, out_png, ctx, ValueRange.SIGNED)
    else:
        raise Error("krea2_decode_latent: size must be 1024 or 512")
    print("[krea2-decode] wrote", out_png)
