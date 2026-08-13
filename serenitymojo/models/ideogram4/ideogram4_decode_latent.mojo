# models/ideogram4/ideogram4_decode_latent.mojo — PROCESS-SEPARATED latent →
# PNG decode.
#
# WHY (FT_INLINE_SAMPLING_PLAN_2026-07-08, model #6): the ideogram4 FULL-FT
# arm's inline sampler completes its 512px denoise within 16GB and persists
# each sample's final PATCH-SPACE latent as <out.png>.lat.bin (io/cap_cache
# save_tensor_bin, F32 [1, 128, GH, GW]) BEFORE attempting the in-process VAE
# decode. When the in-process decode cannot fit beside the trainer's residency
# (frozen-globals residents + pinned-host-store slots + pool blocks), this CLI
# decodes the persisted latent in a FRESH process (virgin GPU pool). Mirrors
# models/chroma/chroma_decode_latent.mojo / models/klein/klein_decode_latent.mojo.
#
# The decode tail below is a 1:1 copy of the trainer's in-process path
# (Serenity Ideogram4SampleResident.ideogram4_decode_latent_to_png, itself 1:1
# with the parity-gated inference decode — ideogram4_pipeline.mojo:93-112):
#   denorm : zd = z*latent_scale + latent_shift            (128-dim broadcast)
#   unpatch: [1,GH,GW,2,2,32] -> permute(0,5,1,3,2,4) -> [1,32,2GH,2GW]
#   decode : LdmVaeDecoder[2GH,2GW].decode(bf16 latent) -> [1,3,16GH,16GW]
#   write  : save_png (SIGNED [-1,1] range, the VAE output convention)
# Same latentnorm fixture, same VAE default, same op sequence — ideogram4 is
# the byte-deterministic class, so a fallback PNG matches an in-process PNG
# bit-for-bit (verified in the FT inline-sampling smoke).
#
# usage: ideogram4_decode_latent <latent.bin> <vae.safetensors|-> <out.png>
#   latent.bin: [1, 128, GH, GW] denoised patch-space latent (F32),
#               GH/GW dispatch the comptime grid: 32x32 -> 512px.
#   vae:        the ideogram-4 LDM VAE checkpoint ("-" = the model dir default).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import argv
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.cap_cache import load_tensor_bin
from serenitymojo.ops.tensor_algebra import add, mul, reshape, permute
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.image.png import save_png
from serenitymojo.models.vae.ldm_decoder import load_ideogram4_vae_decoder


# Canonical Ideogram-4 latent-norm + VAE paths — identical to the parity-gated
# inference path (ideogram4_pipeline.mojo:27) and the trainer's in-process
# decode (Serenity Ideogram4SampleResident.mojo:77-78).
comptime I4_LATENTNORM = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_latentnorm.safetensors"
comptime I4_VAE = "/home/alex/.serenity/models/ideogram-4-fp8/vae/diffusion_pytorch_model.safetensors"


# 1:1 copy of Ideogram4SampleResident.ideogram4_decode_latent_to_png with the
# VAE path as an argument (comptime default preserved by the caller below).
def ideogram4_decode_patch_latent_to_png[GH: Int, GW: Int](
    z: Tensor,               # [1, 128, GH, GW] F32 (denoised patch-space latent)
    vae_path: String,
    out_path: String,
    ctx: DeviceContext,
) raises:
    var ln = ShardedSafeTensors.open(I4_LATENTNORM)
    var scale = reshape(Tensor.from_view(ln.tensor_view("latent_scale"), ctx), [1, 1, 128], ctx)
    var shift = reshape(Tensor.from_view(ln.tensor_view("latent_shift"), ctx), [1, 1, 128], ctx)

    # z is [1,128,GH,GW]; the inference denorm broadcasts over the 128 channel
    # dim with z laid out [1,NIMG,128]. Match the pipeline by permuting to
    # [1,GH,GW,128] -> reshape [1,NIMG,128] before the channel-broadcast multiply,
    # then it folds straight into the [1,GH,GW,2,2,32] unpatch.
    var z_hwc = permute(z, [0, 2, 3, 1], ctx)                 # [1,GH,GW,128]
    var z_tok = reshape(z_hwc, [1, GH * GW, 128], ctx)        # [1,NIMG,128]
    var zd = add(mul(z_tok, scale, ctx), shift, ctx)          # [1,NIMG,128] F32

    var z6 = reshape(zd, [1, GH, GW, 2, 2, 32], ctx)
    var zp = permute(z6, [0, 5, 1, 3, 2, 4], ctx)             # [1,32,GH,2,GW,2]
    var latent = reshape(zp, [1, 32, 2 * GH, 2 * GW], ctx)    # [1,32,2GH,2GW]

    var dec = load_ideogram4_vae_decoder[2 * GH, 2 * GW](vae_path, ctx)
    var img = dec.decode(cast_tensor(latent, STDtype.BF16, ctx), ctx)  # [1,3,16GH,16GW]
    save_png(img, out_path, ctx)


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error(
            "usage: ideogram4_decode_latent <latent.bin> <vae.safetensors|-> <out.png>"
        )
    var lat_path = String(args[1])
    var vae_path = String(args[2])
    if vae_path == String("-") or vae_path.byte_length() == 0:
        vae_path = String(I4_VAE)
    var out_png = String(args[3])

    var ctx = DeviceContext()
    var latent = load_tensor_bin(lat_path, ctx)
    var sh = latent.shape()
    if len(sh) != 4 or sh[0] != 1 or sh[1] != 128:
        raise Error(
            "ideogram4_decode_latent: expected patch-space latent [1, 128, GH, GW]"
        )
    print("[ideogram4-decode] latent", lat_path, " shape [", sh[0], ",", sh[1],
          ",", sh[2], ",", sh[3], "] -> ", out_png)
    if sh[2] == 32 and sh[3] == 32:
        # 512px grid: GH=GW=32 (the giger 512-class training cache geometry).
        ideogram4_decode_patch_latent_to_png[32, 32](latent, vae_path, out_png, ctx)
        print("[ideogram4-decode] wrote", out_png, " ( 512 x 512 )")
    else:
        raise Error(
            "ideogram4_decode_latent: only the 32x32-grid (512px) latent is wired"
        )
