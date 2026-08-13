# SCAIL-2 creator-input VAE staging, pure Mojo and process-separated.
#
# argv:
#   scail2_encode_vae ref  <staged.safetensors> <vae.safetensors> <out.safetensors>
#   scail2_encode_vae pose <staged.safetensors> <vae.safetensors> <out.safetensors>
#   scail2_encode_vae additional <staged.safetensors> <vae.safetensors> <out.safetensors>
#
# `ref` consumes reference_pixel [1,3,1,512,896] and emits
# reference_latent [1,16,1,64,112]. `pose` consumes
# pose_pixel [1,3,65,256,448] and emits pose_latent [1,16,17,32,56].
# Run the two modes as separate processes so their activation lifetimes cannot
# overlap the visual encoder, UMT5, or the 14B denoiser.

from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.lingbotvideo.vae_encoder import LingBotWanVaeEncoder
from serenitymojo.models.scail2.scail2_manifest import (
    write_scail2_condition_manifest,
)
from serenitymojo.ops.tensor_algebra import concat, slice
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


def _encode_reference(
    staged_path: String, vae_path: String, output_path: String,
) raises:
    var ctx = DeviceContext()
    var staged = ShardedSafeTensors.open(staged_path)
    var key = String("reference_pixel")
    if key not in staged.names():
        raise Error("SCAIL-2 staged input missing reference_pixel")
    var pixel = Tensor.from_view_as_f32(staged.tensor_view(key), ctx)
    var shape = pixel.shape()
    if shape != [1, 3, 1, 512, 896]:
        raise Error(
            String("reference_pixel shape must be [1,3,1,512,896], got ")
            + _shape_string(shape)
        )
    var encoder = LingBotWanVaeEncoder[512, 896].load(vae_path, ctx)
    var latent = encoder.encode(pixel, ctx)
    var latent_shape = latent.shape()
    if latent_shape != [1, 16, 1, 64, 112]:
        raise Error(
            String("reference_latent shape mismatch: ") + _shape_string(latent_shape)
        )
    print("SCAIL-2 reference latent", _shape_string(latent_shape))
    _save(String("reference_latent"), latent^, output_path, ctx)


def _encode_pose(
    staged_path: String, vae_path: String, output_path: String,
) raises:
    var ctx = DeviceContext()
    var staged = ShardedSafeTensors.open(staged_path)
    var key = String("pose_pixel")
    if key not in staged.names():
        raise Error("SCAIL-2 staged input missing pose_pixel")
    var pixel = Tensor.from_view_as_f32(staged.tensor_view(key), ctx)
    var shape = pixel.shape()
    if shape != [1, 3, 65, 256, 448]:
        raise Error(
            String("pose_pixel shape must be [1,3,65,256,448], got ")
            + _shape_string(shape)
        )
    var encoder = LingBotWanVaeEncoder[256, 448].load(vae_path, ctx)
    var latent = encoder.encode_video(pixel, ctx)
    var latent_shape = latent.shape()
    if latent_shape != [1, 16, 17, 32, 56]:
        raise Error(
            String("pose_latent shape mismatch: ") + _shape_string(latent_shape)
        )
    print("SCAIL-2 pose latent", _shape_string(latent_shape))
    _save(String("pose_latent"), latent^, output_path, ctx)


def _encode_additional(
    staged_path: String, vae_path: String, output_path: String,
) raises:
    var ctx = DeviceContext()
    var staged = ShardedSafeTensors.open(staged_path)
    var key = String("additional_reference_pixel")
    if key not in staged.names():
        raise Error("SCAIL-2 staged input missing additional_reference_pixel")
    var pixel = Tensor.from_view_as_f32(staged.tensor_view(key), ctx)
    var shape = pixel.shape()
    if (
        len(shape) != 5 or shape[0] != 1 or shape[1] != 3
        or shape[2] < 1 or shape[2] > 3 or shape[3] != 512 or shape[4] != 896
    ):
        raise Error(
            String("additional_reference_pixel shape must be [1,3,N,512,896], got ")
            + _shape_string(shape)
        )
    var encoder = LingBotWanVaeEncoder[512, 896].load(vae_path, ctx)
    var latent = encoder.encode(slice(pixel, 2, 0, 1, ctx), ctx)
    for i in range(1, shape[2]):
        var next_latent = encoder.encode(slice(pixel, 2, i, 1, ctx), ctx)
        latent = concat(2, ctx, latent, next_latent)
    var latent_shape = latent.shape()
    if latent_shape != [1, 16, shape[2], 64, 112]:
        raise Error(
            String("additional_reference_latent shape mismatch: ")
            + _shape_string(latent_shape)
        )
    print("SCAIL-2 additional reference latent", _shape_string(latent_shape))
    _save(String("additional_reference_latent"), latent^, output_path, ctx)


def main() raises:
    var args = argv()
    if len(args) != 5:
        raise Error(
            "usage: scail2_encode_vae <ref|pose|additional> <staged.safetensors> "
            "<vae.safetensors> <out.safetensors>"
        )
    var mode = String(args[1])
    var staged = String(args[2])
    var vae = String(args[3])
    var output = String(args[4])
    if mode == String("ref"):
        _encode_reference(staged, vae, output)
    elif mode == String("pose"):
        _encode_pose(staged, vae, output)
    elif mode == String("additional"):
        _encode_additional(staged, vae, output)
    else:
        raise Error("SCAIL-2 VAE mode must be ref, pose, or additional")
    _ = write_scail2_condition_manifest(
        String(args[0]), mode, staged, vae, output
    )
