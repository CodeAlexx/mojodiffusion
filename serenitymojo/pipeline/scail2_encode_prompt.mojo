# SCAIL-2 UMT5-XXL prompt conditioning (standalone production process).
#
# argv:
#   scail2_encode_prompt <umt5_model_dir> <tokenizer.json>
#                        <positive_prompt> <negative_prompt> <output.safetensors>
#
# Output contract consumed by scail2_animation:
#   pos_embed [1,512,4096] BF16, neg_embed [1,512,4096] BF16
#   pos_len [1] F32, neg_len [1] F32
# The process exits after writing, releasing the resident UMT5 encoder before
# CLIP, VAE, or the 14B denoiser starts.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.models.text_encoder.umt5_encoder import Umt5Encoder, Umt5Config
from serenitymojo.models.scail2.scail2_manifest import write_scail2_prompt_manifest
from serenitymojo.tensor import Tensor
from serenitymojo.tokenizer.t5_tokenizer import T5Tokenizer


comptime SEQ = 512
comptime DIM = 4096
comptime PAD_ID = 0
comptime EOS_ID = 1


def _pad_to_seq(ids_full: List[Int]) raises -> List[Int]:
    if len(ids_full) < 1 or ids_full[len(ids_full) - 1] != EOS_ID:
        raise Error("SCAIL-2 UMT5 tokenizer must append EOS id 1")
    var ids = List[Int](capacity=SEQ)
    if len(ids_full) > SEQ:
        for i in range(SEQ - 1):
            ids.append(ids_full[i])
        ids.append(EOS_ID)
    else:
        for id in ids_full:
            ids.append(id)
        for _ in range(SEQ - len(ids_full)):
            ids.append(PAD_ID)
    if len(ids) != SEQ:
        raise Error("SCAIL-2 UMT5 padded token count must be 512")
    return ids^


def _valid_count(ids_full: List[Int]) -> Int:
    return SEQ if len(ids_full) > SEQ else len(ids_full)


def _length_tensor(value: Int, ctx: DeviceContext) raises -> Tensor:
    if value < 1 or value > SEQ:
        raise Error("SCAIL-2 UMT5 valid length must be in [1,512]")
    return Tensor.from_host([Float32(value)], [1], STDtype.F32, ctx)


def _validate_embedding(
    label: String, value: Tensor, valid: Int, ctx: DeviceContext,
) raises:
    var shape = value.shape()
    if len(shape) != 3 or shape[0] != 1 or shape[1] != SEQ or shape[2] != DIM:
        raise Error(label + String(" must be [1,512,4096]"))
    if value.dtype() != STDtype.BF16:
        raise Error(label + String(" must be BF16"))
    var host = value.to_host(ctx)
    var valid_nonzero = False
    var valid_values = valid * DIM
    for i in range(len(host)):
        var scalar = host[i]
        if not (scalar == scalar) or scalar > Float32(3.0e38) or scalar < Float32(-3.0e38):
            raise Error(label + String(" contains a non-finite value"))
        if i < valid_values:
            if scalar != Float32(0.0):
                valid_nonzero = True
        elif scalar != Float32(0.0):
            raise Error(label + String(" has nonzero padded output rows"))
    if not valid_nonzero:
        raise Error(label + String(" valid output rows are all zero"))


def main() raises:
    var args = argv()
    if len(args) != 6:
        raise Error(
            "usage: scail2_encode_prompt <umt5_model_dir> <tokenizer.json> "
            "<positive_prompt> <negative_prompt> <output.safetensors>"
        )
    var model_dir = String(args[1])
    var tokenizer_path = String(args[2])
    var positive_text = String(args[3])
    var negative_text = String(args[4])
    var output_path = String(args[5])
    if output_path == model_dir or output_path == tokenizer_path:
        raise Error("SCAIL-2 prompt output must not overwrite an input artifact")

    var tokenizer = T5Tokenizer.load(tokenizer_path)
    var positive_full = tokenizer.encode(positive_text)
    var negative_full = tokenizer.encode(negative_text)
    var positive_valid = _valid_count(positive_full)
    var negative_valid = _valid_count(negative_full)
    var positive_ids = _pad_to_seq(positive_full)
    var negative_ids = _pad_to_seq(negative_full)
    print(
        "SCAIL-2 UMT5 tokens positive=", positive_valid,
        "negative=", negative_valid, "padded=512",
    )

    var ctx = DeviceContext()
    var encoder = Umt5Encoder[SEQ].load(
        model_dir, Umt5Config.umt5_xxl(), ctx
    )
    var positive = encoder.encode_masked(positive_ids, positive_valid, ctx)
    var negative = encoder.encode_masked(negative_ids, negative_valid, ctx)
    _validate_embedding(String("pos_embed"), positive, positive_valid, ctx)
    _validate_embedding(String("neg_embed"), negative, negative_valid, ctx)
    var positive_len = _length_tensor(positive_valid, ctx)
    var negative_len = _length_tensor(negative_valid, ctx)

    var names = List[String]()
    names.append(String("pos_embed"))
    names.append(String("neg_embed"))
    names.append(String("pos_len"))
    names.append(String("neg_len"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer(positive^))
    tensors.append(ArcPointer(negative^))
    tensors.append(ArcPointer(positive_len^))
    tensors.append(ArcPointer(negative_len^))
    save_safetensors(names, tensors, output_path, ctx)
    var prompt_manifest = write_scail2_prompt_manifest(
        String(args[0]), model_dir, tokenizer_path, positive_text, negative_text,
        positive_valid, negative_valid, output_path,
    )
    print("SCAIL-2 UMT5 conditioning PASS output=", output_path)
    print("SCAIL-2 prompt manifest=", prompt_manifest)
