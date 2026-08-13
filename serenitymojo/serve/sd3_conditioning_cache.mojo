# Short-lived Mojo SD3.5 conditioning producer.
#
# The SD3 worker cannot retain CLIP-L + CLIP-G + T5-XXL allocations beside the
# streamed MMDiT on a 16 GiB card.  This executable reuses the worker's exact,
# parity-gated `_assemble_one` path, writes the four BF16 conditioning tensors,
# and exits so the OS reclaims every encoder allocation before inference.

from std.collections import List
from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.text_encoder.clip_encoder import ClipEncoder, ClipConfig
from serenitymojo.models.text_encoder.t5_encoder import T5Encoder, T5Config
from serenitymojo.serve.model_scan import _read_text_file
from serenitymojo.serve.sd3_backend import _assemble_one
from serenitymojo.tensor import Tensor
from serenitymojo.tokenizer.clip_tokenizer import ClipTokenizer
from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer


comptime CLIP_L_PATH = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors"
comptime CLIP_G_PATH = "/home/alex/.serenity/models/text_encoders/clip_g.safetensors"
comptime T5_PATH = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors"
comptime CLIP_L_TOK = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json"
comptime CLIP_G_TOK = "/home/alex/.serenity/models/text_encoders/clip_g.tokenizer.json"
comptime T5_TOK = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json"
comptime CLIP_G_TEXT_PROJ = "text_projection.weight"
comptime T5_LEN = 256


def main() raises:
    var args = argv()
    if len(args) != 4:
        raise Error(
            "usage: serenity_sd3_conditioning <prompt.txt> <negative.txt> <out.safetensors>"
        )
    var prompt = _read_text_file(String(args[1]))
    var negative = _read_text_file(String(args[2]))
    var out_path = String(args[3])

    var ctx = DeviceContext()
    var clip_l = ClipEncoder.load(String(CLIP_L_PATH), ClipConfig.clip_l(), ctx)
    var clip_g = ClipEncoder.load(String(CLIP_G_PATH), ClipConfig.clip_g(), ctx)
    var g_st = ShardedSafeTensors.open(String(CLIP_G_PATH))
    var text_proj_g = Tensor.from_view(
        g_st.tensor_view(String(CLIP_G_TEXT_PROJ)), ctx
    )
    var t5 = T5Encoder[T5_LEN].load(String(T5_PATH), T5Config.t5_xxl(), ctx)
    var clip_l_tok = ClipTokenizer(String(CLIP_L_TOK))
    var clip_g_tok = ClipTokenizer(String(CLIP_G_TOK))
    var t5_tok = T5Tokenizer(String(T5_TOK))

    var pos = _assemble_one(
        prompt, clip_l, clip_g, text_proj_g, t5,
        clip_l_tok, clip_g_tok, t5_tok, ctx,
    )
    var neg = _assemble_one(
        negative, clip_l, clip_g, text_proj_g, t5,
        clip_l_tok, clip_g_tok, t5_tok, ctx,
    )

    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    names.append(String("context"))
    tensors.append(ArcPointer(pos[0].clone(ctx)))
    names.append(String("context_uncond"))
    tensors.append(ArcPointer(neg[0].clone(ctx)))
    names.append(String("pooled"))
    tensors.append(ArcPointer(pos[1].clone(ctx)))
    names.append(String("pooled_uncond"))
    tensors.append(ArcPointer(neg[1].clone(ctx)))
    save_safetensors(names, tensors, out_path, ctx)
    print("[sd3-conditioning] wrote", out_path)
