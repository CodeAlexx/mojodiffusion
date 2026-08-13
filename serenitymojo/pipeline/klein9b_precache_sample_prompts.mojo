# Klein 9B validation prompt precache.
#
# Reads a shared serenity.sample_prompts.v1 JSON, encodes every prompt and
# negative prompt with Qwen3-8B, writes cap-cache files, then exits so encoder
# GPU memory is released before the trainer/sampler process starts.

from std.sys import argv
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config, _clone
from serenitymojo.io.cap_cache import save_tensor_bin
from serenitymojo.ops.tensor_algebra import concat
from serenitymojo.io.ffi import sys_system
from serenitymojo.training.sample_prompt_config import read_sample_prompt_config, SamplePrompt


comptime DEFAULT_PROMPTS = "serenitymojo/configs/klein9b_alina_samples.json"
comptime QWEN4_DIR = "models/qwen3-4b"
comptime QWEN8_DIR = "models/qwen3-8b"
comptime PAD_ID = 151643
comptime SEQ = 512


def _klein_template(prompt: String) -> String:
    return (
        String("<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _tokenize_512(tok: Qwen3Tokenizer, label: String, prompt: String) raises -> List[Int]:
    var ids_full = tok.encode(_klein_template(prompt))
    if len(ids_full) > SEQ:
        raise Error(String("Klein prompt too long for 512 tokens: ") + label)
    var ids = List[Int](capacity=SEQ)
    for i in range(len(ids_full)):
        ids.append(ids_full[i])
    for _ in range(SEQ - len(ids_full)):
        ids.append(PAD_ID)
    print("[precache] ", label, " tokens ", len(ids_full), " -> ", SEQ)
    return ids^


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


def _variant_name(raw: String) raises -> String:
    var v = String(raw.lower())
    if v == String("") or v == String("9b") or v == String("klein9b"):
        return String("9b")
    if v == String("4b") or v == String("klein4b"):
        return String("4b")
    raise Error(String("klein precache: expected variant 9b or 4b, got ") + raw)


# encode_klein with an explicit layer-26 stop (the klein_prepare.mojo:215
# precedent): byte-identical to encode_klein (which concats layer states
# [8,17,26] — running layers past 26 cannot change an earlier state), and it
# pairs with Qwen3Encoder.load(..., max_layer=26) below, which skips lm_head +
# layers 27..35 (~4.8GB) — the difference between the 16GB 5080 and OOM.
def _encode_klein26(enc: Qwen3Encoder, ids: List[Int], ctx: DeviceContext) raises -> Tensor:
    var states = enc.encode_layer_states(ids, ctx, 26)
    var h8 = _clone(states[8][], ctx)
    var h17 = _clone(states[17][], ctx)
    var h26 = _clone(states[26][], ctx)
    return concat(2, ctx, h8, h17, h26)


def _encode_one(
    enc: Qwen3Encoder,
    tok: Qwen3Tokenizer,
    p: SamplePrompt,
    ctx: DeviceContext,
) raises:
    if p.caps_pos == String("") or p.caps_neg == String(""):
        raise Error(String("sample prompt has no cap paths: ") + p.label)
    _mkdir_parent(p.caps_pos)
    _mkdir_parent(p.caps_neg)
    var pos_ids = _tokenize_512(tok, p.label + String(":pos"), p.prompt)
    var neg_ids = _tokenize_512(tok, p.label + String(":neg"), p.negative)
    var pos = _encode_klein26(enc, pos_ids, ctx)
    var neg = _encode_klein26(enc, neg_ids, ctx)
    save_tensor_bin(pos, p.caps_pos, ctx)
    save_tensor_bin(neg, p.caps_neg, ctx)
    print("[precache] wrote ", p.caps_pos)
    print("[precache] wrote ", p.caps_neg)


def main() raises:
    var args = argv()
    var prompt_path = String(DEFAULT_PROMPTS)
    if len(args) >= 2:
        prompt_path = String(args[1])
    var variant = String("9b")
    if len(args) >= 3:
        variant = _variant_name(String(args[2]))

    var qwen_dir = String(QWEN8_DIR)
    var enc_cfg = Qwen3Config.klein_9b()
    if variant == String("4b"):
        qwen_dir = String(QWEN4_DIR)
        enc_cfg = Qwen3Config.klein_4b()

    print("=== Klein validation prompt precache ===")
    print("[precache] variant: ", variant)
    print("[precache] prompt config: ", prompt_path)
    var cfg = read_sample_prompt_config(prompt_path)
    var ctx = DeviceContext()
    var tok = Qwen3Tokenizer(qwen_dir + String("/tokenizer.json"))
    # max_layer=26: encode extracts layers [8,17,26] only — skipping lm_head +
    # layers 27..35 fits the 16GB 5080 (klein_prepare.mojo:184 precedent,
    # byte-identical encodes).
    var enc = Qwen3Encoder.load(qwen_dir, enc_cfg, ctx, 26)

    for i in range(len(cfg.prompts)):
        _encode_one(enc, tok, cfg.prompts[i], ctx)

    print("[precache] done; encoder memory frees on process exit")
