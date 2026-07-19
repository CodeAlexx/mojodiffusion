# Product-owned Anima training-cache producer.
#
# Input is the shared image/caption stage. Output is one atomic safetensors
# record per sample containing every frozen input consumed by the trainer:
#   latent       [1,16,64,64]  BF16
#   context_cond [1,256,1024]  F32
#
# Every model asset is an explicit argument. The caller owns path resolution;
# this executable contains no workstation-specific fallback.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.os import makedirs
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Config, Qwen3Encoder
from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder
from serenitymojo.models.anima.anima_text_context import (
    AnimaAdapterWeights, tokenize_anima_text, encode_anima_context,
)

comptime IH = 512
comptime IW = 512
comptime TRAIN_CONTEXT_TOKENS = 256
comptime CONTEXT_DIM = 1024
comptime TArc = ArcPointer[Tensor]


def _sample_name(i: Int) -> String:
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


def main() raises:
    var args = argv()
    if len(args) != 9:
        raise Error(
            "usage: anima_prepare_cache <stage_dir> <out_dir> <n>"
            " <anima_checkpoint> <qwen3_encoder> <qwen3_tokenizer.json>"
            " <t5_tokenizer.json> <qwen_image_vae_encoder>"
        )

    var stage_dir = String(args[1])
    var out_dir = String(args[2])
    var count = Int(String(args[3]))
    var checkpoint = String(args[4])
    var qwen_encoder = String(args[5])
    var qwen_tokenizer = String(args[6])
    var t5_tokenizer = String(args[7])
    var vae_encoder = String(args[8])
    if count <= 0:
        raise Error("anima_prepare_cache: n must be positive")

    makedirs(out_dir, exist_ok=True)
    var ctx = DeviceContext()
    var qtok = Qwen3Tokenizer(qwen_tokenizer)
    var t5tok = T5Tokenizer.load(t5_tokenizer)
    var text_encoder = Qwen3Encoder.load(
        qwen_encoder, Qwen3Config.qwen3_06b(), ctx
    )
    var adapter = AnimaAdapterWeights.load_checkpoint(checkpoint, ctx)
    var vae = QwenImageVaeEncoder[IH, IW].load(vae_encoder, ctx)

    for i in range(count):
        var stem = _sample_name(i)
        var staged_path = stage_dir + String("/") + stem + String(".safetensors")
        var caption_path = stage_dir + String("/") + stem + String(".txt")
        var staged = ShardedSafeTensors.open(staged_path)
        var image_info = staged.tensor_info(String("image"))
        if (
            len(image_info.shape) != 4 or image_info.shape[0] != 1
            or image_info.shape[1] != 3 or image_info.shape[2] != IH
            or image_info.shape[3] != IW
        ):
            raise Error(
                String("anima_prepare_cache: ") + staged_path
                + String(" image must be [1,3,512,512]")
            )

        var image_f32 = Tensor.from_view(staged.tensor_view(String("image")), ctx)
        var image = cast_tensor(image_f32, STDtype.BF16, ctx)
        var latent = vae.encode_mean(image, ctx)

        var caption = _read_text(caption_path)
        var tokens = tokenize_anima_text(caption, qtok, t5tok)
        var context_512 = encode_anima_context(tokens, text_encoder, adapter, ctx)
        var context = slice(context_512, 1, 0, TRAIN_CONTEXT_TOKENS, ctx)
        var context_shape = context.shape()
        if (
            len(context_shape) != 3 or context_shape[0] != 1
            or context_shape[1] != TRAIN_CONTEXT_TOKENS
            or context_shape[2] != CONTEXT_DIM
        ):
            raise Error("anima_prepare_cache: context shape contract failed")

        var names = List[String]()
        names.append(String("latent"))
        names.append(String("context_cond"))
        var tensors = List[TArc]()
        tensors.append(TArc(latent^))
        tensors.append(TArc(context^))
        var out_path = out_dir + String("/") + stem + String(".safetensors")
        save_safetensors(names, tensors, out_path, ctx)
        print("[anima-prepare] wrote", out_path)

    print("[anima-prepare] complete samples=", count, " cache=", out_dir)
