# ERNIE validation prompt precache.
#
# Reads a shared serenity.sample_prompts.v1 JSON, encodes every prompt and
# negative prompt with the pure-Mojo ERNIE Mistral-3B text encoder, and writes
# cap-cache safetensors with key `text_embedding`. The trainer's built-in
# sampler reads those cap paths during cadence sampling.
#
# Run:
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/pipeline/ernie_precache_sample_prompts.mojo -o /tmp/ernie_precache_sample_prompts
#   /tmp/ernie_precache_sample_prompts serenitymojo/configs/ernie_image_samples.json

from std.sys import argv
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.training.sample_prompt_config import (
    read_sample_prompt_config, SamplePrompt,
)
from serenitymojo.models.text_encoder.mistral3b_encoder import (
    Mistral3bEncoder, mistral3_tokenize,
)


comptime DEFAULT_PROMPTS = "/home/alex/mojodiffusion/serenitymojo/configs/ernie_image_samples.json"
comptime TOK_JSON = "/home/alex/models/ERNIE-Image/tokenizer/tokenizer.json"
comptime TEXT_ENCODER_DIR = "/home/alex/models/ERNIE-Image/text_encoder"
comptime SEQ = 256
comptime TEXT_IN = 3072


def _substr(s: String, start: Int, end: Int) -> String:
    var out = String("")
    var i = 0
    for ch in s.codepoint_slices():
        if i >= start and i < end:
            out += String(ch)
        i += 1
    return out^


def _dirname(path: String) -> String:
    var last = -1
    var i = 0
    for ch in path.codepoint_slices():
        if String(ch) == String("/"):
            last = i
        i += 1
    if last <= 0:
        return String(".")
    return _substr(path, 0, last)


def _mkdir_parent(path: String):
    _ = sys_system(String("mkdir -p ") + _dirname(path))


def _save_text_sidecar(
    emb: Tensor, real_len: Int, out_path: String, ctx: DeviceContext
) raises:
    var es = emb.shape()
    if len(es) != 3 or es[0] != 1 or es[1] != SEQ or es[2] != TEXT_IN:
        raise Error("ERNIE prompt embedding shape must be [1,256,3072]")
    var names = List[String]()
    names.append(String("text_embedding"))
    names.append(String("text_real_len"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer[Tensor](emb.clone(ctx)))
    var rl = List[Float32]()
    rl.append(Float32(real_len))
    tensors.append(ArcPointer[Tensor](Tensor.from_host(rl, [1], STDtype.F32, ctx)))
    _mkdir_parent(out_path)
    save_safetensors(names, tensors, out_path, ctx)


def _encode_text(
    enc: Mistral3bEncoder,
    tok: Qwen3Tokenizer,
    label: String,
    text: String,
    out_path: String,
    ctx: DeviceContext,
) raises:
    if out_path == String(""):
        raise Error(String("ERNIE prompt has no cap path: ") + label)
    var ids = mistral3_tokenize(tok, text)
    if len(ids) > SEQ:
        raise Error(
            String("ERNIE prompt too long for ") + String(SEQ)
            + String(" tokens: ") + label + String(" tokens=") + String(len(ids))
        )
    print("[ernie-precache] ", label, " tokens ", len(ids), " -> ", SEQ)
    var out = enc.encode(ids, SEQ, ctx)
    _save_text_sidecar(out[0], out[1], out_path, ctx)
    print("[ernie-precache] wrote ", out_path)


def _encode_one(
    enc: Mistral3bEncoder,
    tok: Qwen3Tokenizer,
    p: SamplePrompt,
    ctx: DeviceContext,
) raises:
    _encode_text(enc, tok, p.label + String(":pos"), p.prompt, p.caps_pos, ctx)
    _encode_text(enc, tok, p.label + String(":neg"), p.negative, p.caps_neg, ctx)


def main() raises:
    var args = argv()
    var prompt_path = String(DEFAULT_PROMPTS)
    if len(args) >= 2:
        prompt_path = String(args[1])

    print("=== ERNIE validation prompt precache ===")
    print("[ernie-precache] prompt config: ", prompt_path)
    var cfg = read_sample_prompt_config(prompt_path)
    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var enc = Mistral3bEncoder.load(String(TEXT_ENCODER_DIR), ctx)

    for i in range(len(cfg.prompts)):
        _encode_one(enc, tok, cfg.prompts[i], ctx)

    print("[ernie-precache] done; encoder memory frees on process exit")
