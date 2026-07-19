# models/chroma/chroma_decode_latent.mojo — PROCESS-SEPARATED latent → PNG decode.
#
# WHY (FT_INLINE_SAMPLING_PLAN_2026-07-08, model #4): the chroma FULL-FT arm's
# inline sampler completes its 512px denoise within 16GB and persists each
# sample's final PACKED latent as <out.png>.lat.bin (io/cap_cache
# save_tensor_bin, F32 [N_IMG, 64]) BEFORE attempting the in-process FLUX-VAE
# decode. When the in-process decode cannot fit beside the trainer's residency
# (pinned-host-store trainer + pool blocks), this CLI decodes the persisted
# latent in a FRESH process (virgin GPU pool). Mirrors
# models/klein/klein_decode_latent.mojo / models/zimage/zimage_decode_latent.mojo.
#
# The decode path is BYTE-IDENTICAL to the trainer's in-process attempt: both
# call training/chroma_sample_resident.chroma_decode_latent_to_png (unpack
# [N_IMG,64] -> [1,16,LH,LW] + load_flux1_ldm_decoder decode + save_png SIGNED)
# — so a fallback PNG matches an in-process PNG bit-for-bit (the krea2 gate
# precedent).
#
# usage: chroma_decode_latent <latent.bin> <ae.safetensors> <out.png>
#   latent.bin: [N_IMG, 64] denoised packed chroma latent (F32 or BF16),
#               trainer-scaled space (the FLUX VAE decode inverts the scale).
#   N_IMG dispatches the comptime grid: 1024 -> 512px (LAT_H=LAT_W=64).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.training.chroma_sample_resident import chroma_decode_latent_to_png


def _host_f32(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    return t.to_host(ctx)


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error(
            "usage: chroma_decode_latent <latent.bin> <ae.safetensors> <out.png>"
        )
    var lat_path = String(args[1])
    var vae_path = String(args[2])
    var out_png = String(args[3])

    var ctx = DeviceContext()
    var latent = load_tensor_bin(lat_path, ctx)
    var sh = latent.shape()
    if len(sh) != 2 or sh[1] != 64:
        raise Error("chroma_decode_latent: expected packed latent [N_IMG, 64]")
    print("[chroma-decode] latent", lat_path, " shape [", sh[0], ",", sh[1],
          "] -> ", out_png)
    var packed = _host_f32(latent, ctx)
    if sh[0] == 1024:
        # 512px grid: LAT_C=16, LAT_H=LAT_W=64, patch grid 32x32, PACK=2.
        chroma_decode_latent_to_png[16, 64, 64, 32, 32, 2, 1024, 64](
            packed, vae_path, out_png, ctx,
        )
        print("[chroma-decode] wrote", out_png, " ( 512 x 512 )")
    else:
        raise Error(
            "chroma_decode_latent: only the 1024-token (512px) grid is wired"
        )
