# Phase 4a temp smoke (DO NOT COMMIT): dump the lens worker's GPT-OSS encode.
# Replicates serve/lens_backend.mojo::_encode_one_prompt EXACTLY (lens chat template
# render -> Qwen3Tokenizer o200k -> 512 pad @ 199999 -> GptOssEncoder.encode([5,11,17,23])
# -> offset-97 trim -> FIRST 64 post-offset rows) and dumps the 4 layers' [64,2880]
# post-trim features + the token ids, so an HF GptOssModel reference can compare.
from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.gpt_oss_encoder import (
    GptOssEncoder, GptOssConfig, lens_extract_layers,
)
from serenitymojo.ops.tensor_algebra import reshape, slice as t_slice
from serenitymojo.io.safetensors_writer import save_safetensors

comptime LENS_TEXT_ENCODER_DIR = "/home/alex/.serenity/models/microsoft_lens/text_encoder"
comptime LENS_TOKENIZER_JSON = "/home/alex/.serenity/models/microsoft_lens/tokenizer/tokenizer.json"
comptime MAX_TEXT_LEN = 512
comptime DEFAULT_TXT_OFFSET = 97
comptime GPT_OSS_PAD_ID = 199999
comptime N_TXT = 64
comptime ENC_HIDDEN = 2880
comptime CHAT_SYSTEM = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background."
comptime CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
comptime PROMPT = "a photorealistic red fox sitting in autumn leaves"
comptime OUT = "/home/alex/mojodiffusion/output/checks/phase4a/lens_worker_cond.safetensors"


def _render_lens_chat_template(prompt: String) -> String:
    var date = String("2026-06-15")
    var s = String("")
    s += "<|start|>system<|message|>"
    s += "You are ChatGPT, a large language model trained by OpenAI.\n"
    s += "Knowledge cutoff: 2024-06\n"
    s += "Current date: "
    s += date
    s += "\n\n"
    s += "Reasoning: medium\n\n"
    s += "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    s += "<|end|>"
    s += "<|start|>developer<|message|>"
    s += "# Instructions\n\n"
    s += String(CHAT_SYSTEM)
    s += "\n\n"
    s += "<|end|>"
    s += "<|start|>user<|message|>"
    s += prompt
    s += "<|end|>"
    s += "<|start|>assistant<|channel|>analysis<|message|>"
    s += String(CHAT_ASSISTANT_THINKING)
    s += "<|end|>"
    s += "<|start|>assistant<|channel|>final<|message|>"
    return s


def _ids_tensor(ids: List[Int], n: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    for i in range(len(ids)):
        h.append(Float32(ids[i]))
    return Tensor.from_host(h, [n], STDtype.F32, ctx)


def main() raises:
    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(String(LENS_TOKENIZER_JSON), True)  # o200k=True
    var text = _render_lens_chat_template(String(PROMPT))
    var raw = tok.encode(text)
    var real_len = len(raw)
    if real_len > MAX_TEXT_LEN:
        real_len = MAX_TEXT_LEN
    var ids = List[Int]()
    for i in range(real_len):
        ids.append(raw[i])
    while len(ids) < MAX_TEXT_LEN:
        ids.append(GPT_OSS_PAD_ID)
    print("[lens-dump] raw tokens =", len(raw), "real_len =", real_len, "padded_to", MAX_TEXT_LEN)

    var enc = GptOssEncoder.load(String(LENS_TEXT_ENCODER_DIR), GptOssConfig.lens_default(), ctx)
    var extract = lens_extract_layers()  # [5,11,17,23]
    var caps = enc.encode(ids, extract, ctx)  # 4 x [1,512,2880] ascending
    if len(caps) != 4:
        raise Error("lens-dump: expected 4 capture layers")

    # offset-trim 97 then FIRST 64 -> [1,64,2880] = t_slice(1, 97, 64), reshape [64,2880].
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    var tags = List[String]()
    tags.append(String("l05"))
    tags.append(String("l11"))
    tags.append(String("l17"))
    tags.append(String("l23"))
    for i in range(4):
        var sl = t_slice(caps[i][], 1, DEFAULT_TXT_OFFSET, N_TXT, ctx)  # [1,64,2880]
        var r2 = reshape(sl, [N_TXT, ENC_HIDDEN], ctx)                   # [64,2880]
        names.append(tags[i].copy())
        tensors.append(ArcPointer(r2^))
    names.append(String("ids"))
    tensors.append(ArcPointer(_ids_tensor(ids, MAX_TEXT_LEN, ctx)))
    save_safetensors(names, tensors, String(OUT), ctx)
    print("[lens-dump] wrote", OUT)
