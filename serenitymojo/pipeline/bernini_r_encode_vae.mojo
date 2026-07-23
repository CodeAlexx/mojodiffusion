# serenitymojo/pipeline/bernini_r_encode_vae.mojo
#
# BERNINI-R data-cache VAE encode stage (pure Mojo + MAX, standalone process).
# REUSES the certified LingBot Wan2.1 VAE encoder struct (models/lingbotvideo/
# vae_encoder.mojo). This driver reimplements NO encoder math — it only stages a
# single square-resolution RGB frame through `LingBotWanVaeEncoder.encode` at the
# Bernini renderer geometry (256->32 target, 128->16 conditioning) and writes the
# normalized 16-ch latent for scripts/bernini_r_build_cache.py to assemble.
#
# One model per process (16 GB): the encoder is loaded, used, and freed when the
# process exits, so its activation lifetime never overlaps umt5 or the denoiser.
#
# argv:
#   bernini_r_encode_vae <staged.safetensors> <vae.safetensors> <out.safetensors>
#
# `staged.safetensors` carries key "pixel" = NCTHW [1,3,F,H,W] in [0,1], with
# H==W in {256, 128}. When F==1 the single-frame image encoder runs (unchanged);
# when F>1 the multi-frame clip is encoded through `LingBotWanVaeEncoder.
# encode_video` (Tlat = 1+(F-1)//4). Output key "latent" = NCTHW [1,16,f,H/8,W/8]
# f32 (Wan2.1 per-channel normalized — the SAME convention as the wan22/scail2
# caches; f==1 for images, f>1 for video conditioning / multi-frame targets).
#
# BUILD (binary):
#   cd /home/alex/mojodiffusion; rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 \
#     -I . -I /home/alex/MOJO-libs -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/pipeline/bernini_r_encode_vae.mojo -o output/bin/bernini_r_encode_vae
#
# Mojo 1.0.0b1, NVIDIA GPU (sm_120 / 16GB).

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.lingbotvideo.vae_encoder import LingBotWanVaeEncoder
from serenitymojo.tensor import Tensor


def _shape_string(shape: List[Int]) -> String:
    var out = String("[")
    for i in range(len(shape)):
        if i > 0:
            out += String(",")
        out += String(shape[i])
    return out + String("]")


def _save(name: String, var value: Tensor, path: String, ctx: DeviceContext) raises:
    var names = List[String]()
    names.append(name)
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(value^))
    save_safetensors(names, tensors, path, ctx)


def _encode_at[RES: Int](
    pixel: Tensor, vae_path: String, ctx: DeviceContext,
) raises -> Tensor:
    var encoder = LingBotWanVaeEncoder[RES, RES].load(vae_path, ctx)
    var latent = encoder.encode(pixel, ctx)
    var latent_shape = latent.shape()
    var lr = RES // 8
    if latent_shape != [1, 16, 1, lr, lr]:
        raise Error(
            String("bernini_r VAE latent shape mismatch: ")
            + _shape_string(latent_shape)
        )
    var host = latent.to_host(ctx)
    var s = Float64(0.0)
    var ss = Float64(0.0)
    for i in range(len(host)):
        s += Float64(host[i])
        ss += Float64(host[i]) * Float64(host[i])
    var mean = s / Float64(len(host))
    var varp = ss / Float64(len(host)) - mean * mean
    if varp < 0.0:
        varp = 0.0
    from std.math import sqrt as _sqrt
    var std = _sqrt(varp)
    var flag = String("")
    # A single normalized Wan2.1 latent std is legitimately ~0.30-0.65 (the
    # per-channel population std includes inter-image variance one sample lacks);
    # a near-zero or >1.3 std would signal an HWC<->CHW / scramble bug.
    if not (0.15 <= std <= 1.30):
        flag = String("  <-- STD OUT OF RANGE (possible scramble bug!)")
    print(
        "bernini_r VAE latent", _shape_string(latent_shape),
        " std=", std, " mean=", mean, flag,
    )
    return latent^


# Multi-frame clip encode: pixel NCTHW [1,3,F,H,W] (F>1) -> normalized latent
# NCTHW [1,16,Tlat,H/8,W/8], Tlat = 1+(F-1)//4. REUSES LingBotWanVaeEncoder.
# encode_video verbatim (the scail2 `pose` mode precedent) — no encoder math here.
def _encode_video_at[RES: Int](
    pixel: Tensor, vae_path: String, ctx: DeviceContext,
) raises -> Tensor:
    var encoder = LingBotWanVaeEncoder[RES, RES].load(vae_path, ctx)
    var latent = encoder.encode_video(pixel, ctx)
    var latent_shape = latent.shape()
    var lr = RES // 8
    if (
        len(latent_shape) != 5 or latent_shape[0] != 1 or latent_shape[1] != 16
        or latent_shape[3] != lr or latent_shape[4] != lr
    ):
        raise Error(
            String("bernini_r VAE video latent shape mismatch: ")
            + _shape_string(latent_shape)
        )
    var host = latent.to_host(ctx)
    var s = Float64(0.0)
    var ss = Float64(0.0)
    for i in range(len(host)):
        s += Float64(host[i])
        ss += Float64(host[i]) * Float64(host[i])
    var mean = s / Float64(len(host))
    var varp = ss / Float64(len(host)) - mean * mean
    if varp < 0.0:
        varp = 0.0
    from std.math import sqrt as _sqrt
    var std = _sqrt(varp)
    var flag = String("")
    if not (0.15 <= std <= 1.30):
        flag = String("  <-- STD OUT OF RANGE (possible scramble bug!)")
    print(
        "bernini_r VAE VIDEO latent", _shape_string(latent_shape),
        " std=", std, " mean=", mean, " Tlat=", latent_shape[2], flag,
    )
    return latent^


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error(
            "usage: bernini_r_encode_vae <staged.safetensors> <vae.safetensors> "
            "<out.safetensors>"
        )
    var staged_path = String(args[1])
    var vae_path = String(args[2])
    var out_path = String(args[3])

    var ctx = DeviceContext()
    var staged = ShardedSafeTensors.open(staged_path)
    if String("pixel") not in staged.names():
        raise Error("bernini_r staged input missing 'pixel'")
    var pixel = Tensor.from_view_as_f32(staged.tensor_view(String("pixel")), ctx)
    var shape = pixel.shape()
    if (
        len(shape) != 5 or shape[0] != 1 or shape[1] != 3 or shape[2] < 1
        or shape[3] != shape[4]
    ):
        raise Error(
            String("bernini_r pixel must be NCTHW [1,3,F,H,W] with H==W, got ")
            + _shape_string(shape)
        )
    var nframe = shape[2]
    var res = shape[3]
    var latent: Tensor
    if nframe == 1:
        # single-frame image (unchanged path + std gate).
        if res == 256:
            latent = _encode_at[256](pixel, vae_path, ctx)
        elif res == 128:
            latent = _encode_at[128](pixel, vae_path, ctx)
        else:
            raise Error(
                String("bernini_r pixel resolution must be 256 or 128, got ")
                + String(res)
            )
    else:
        # multi-frame clip -> encode_video.
        if res == 256:
            latent = _encode_video_at[256](pixel, vae_path, ctx)
        elif res == 128:
            latent = _encode_video_at[128](pixel, vae_path, ctx)
        else:
            raise Error(
                String("bernini_r pixel resolution must be 256 or 128, got ")
                + String(res)
            )
    _save(String("latent"), latent^, out_path, ctx)
    print("bernini_r VAE encode PASS output=", out_path)
