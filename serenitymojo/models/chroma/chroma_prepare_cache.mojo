# Product-owned Chroma training-cache producer.
#
# Shared staged image/caption records become complete per-sample cache records:
#   latent   [1,16,64,64] raw Flux VAE posterior mean
#   t5_embed [1,512,4096] frozen T5-XXL context
#
# The two encoders run in separate scopes so T5-XXL and the VAE are never
# resident together on a 16 GB card. All model paths are explicit arguments.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.os import makedirs
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.ffi import sys_remove
from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer
from serenitymojo.models.text_encoder.t5_encoder import T5Config, T5Encoder
from serenitymojo.vae.flux_vae_encoder import FluxVaeEncoder

comptime SEQ = 512
comptime IH = 512
comptime IW = 512
comptime LH = 64
comptime LW = 64
comptime TArc = ArcPointer[Tensor]


def _stem(i: Int) -> String:
    var value = String(i)
    var padding = String("")
    for _ in range(5 - value.byte_length() if value.byte_length() < 5 else 0):
        padding += String("0")
    return String("sample_") + padding + value


def _read_text(path: String) raises -> String:
    var file = open(path, "r")
    var value = file.read()
    file.close()
    return value^


def _temp_path(out_dir: String, i: Int) -> String:
    return out_dir + String("/.text_") + _stem(i) + String(".safetensors")


def _encode_text_pass(
    stage_dir: String, out_dir: String, count: Int,
    t5_weights: String, tokenizer_path: String, ctx: DeviceContext,
) raises:
    var tokenizer = T5Tokenizer.load(tokenizer_path)
    var encoder = T5Encoder[SEQ].load(t5_weights, T5Config.t5_xxl(), ctx)
    for i in range(count):
        var caption = _read_text(stage_dir + String("/") + _stem(i) + String(".txt"))
        var raw = tokenizer.encode(caption)
        var ids = List[Int]()
        var keep = len(raw) if len(raw) < SEQ else SEQ
        for j in range(keep):
            ids.append(raw[j])
        while len(ids) < SEQ:
            ids.append(0)
        var context = encoder.encode(ids^, ctx)
        var names = List[String]()
        names.append(String("t5_embed"))
        var tensors = List[TArc]()
        tensors.append(TArc(context^))
        save_safetensors(names, tensors, _temp_path(out_dir, i), ctx)
        print("[chroma-prepare] text", i + 1, "/", count)


def _encode_image_pass(
    stage_dir: String, out_dir: String, count: Int,
    vae_path: String, ctx: DeviceContext,
) raises:
    var encoder = FluxVaeEncoder[LH, LW].load(vae_path, ctx)
    for i in range(count):
        var stem = _stem(i)
        var staged = ShardedSafeTensors.open(
            stage_dir + String("/") + stem + String(".safetensors")
        )
        var info = staged.tensor_info(String("image"))
        if (
            len(info.shape) != 4 or info.shape[0] != 1 or info.shape[1] != 3
            or info.shape[2] != IH or info.shape[3] != IW
        ):
            raise Error("chroma_prepare_cache: staged image must be [1,3,512,512]")
        var image = Tensor.from_view(staged.tensor_view(String("image")), ctx)
        var latent = encoder.encode_mean(image, ctx)
        var text = ShardedSafeTensors.open(_temp_path(out_dir, i))
        var context = Tensor.from_view(text.tensor_view(String("t5_embed")), ctx)

        var names = List[String]()
        names.append(String("latent"))
        names.append(String("t5_embed"))
        var tensors = List[TArc]()
        tensors.append(TArc(latent^))
        tensors.append(TArc(context^))
        var output = out_dir + String("/") + stem + String(".safetensors")
        save_safetensors(names, tensors, output, ctx)
        if sys_remove(_temp_path(out_dir, i)) != 0:
            raise Error("chroma_prepare_cache: failed to remove completed text temporary")
        print("[chroma-prepare] wrote", output)


def main() raises:
    var args = argv()
    if len(args) != 7:
        raise Error(
            "usage: chroma_prepare_cache <stage_dir> <out_dir> <n>"
            " <flux_vae> <t5_xxl_weights> <t5_tokenizer.json>"
        )
    var stage_dir = String(args[1])
    var out_dir = String(args[2])
    var count = Int(String(args[3]))
    if count <= 0:
        raise Error("chroma_prepare_cache: n must be positive")
    makedirs(out_dir, exist_ok=True)
    var ctx = DeviceContext()
    _encode_text_pass(
        stage_dir, out_dir, count, String(args[5]), String(args[6]), ctx
    )
    _encode_image_pass(stage_dir, out_dir, count, String(args[4]), ctx)
    print("[chroma-prepare] complete samples=", count, " cache=", out_dir)
