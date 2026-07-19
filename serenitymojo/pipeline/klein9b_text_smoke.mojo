# klein9b_text_smoke.mojo — Qwen3-8B text conditioning for FLUX.2 Klein 9B.
#
# This is the first runnable Klein 9B slice. It reuses the Z-Image Qwen3
# encoder with the Klein chat template, the Qwen3-8B BF16 Hugging Face snapshot,
# and layer stacking [8,17,26] -> [1,512,12288].
#
# The .serenity `qwen_3_8b.safetensors` file is Comfy-quantized; this smoke uses
# the dense BF16 HF cache because the current Mojo Qwen3 path has no dequantizer.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.tensor import Tensor


comptime QWEN8_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
comptime TOK_JSON = QWEN8_DIR + "/tokenizer.json"
comptime PAD_ID = 151643
comptime SEQ = 512

comptime PROMPT = (
    "Beautiful young woman sitting on a park bench in golden hour sunlight,"
    " professional model photoshoot, soft bokeh background, editorial fashion"
    " photography, natural skin texture, warm color grading"
)
comptime NEGATIVE = (
    "lowres, bad quality, worst quality, bad anatomy, blurry, watermark,"
    " jpeg artifacts, ugly"
)


def _klein_template(prompt: String) -> String:
    return (
        String("<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _tokenize_512(tok: Qwen3Tokenizer, prompt: String) raises -> List[Int]:
    var ids_full = tok.encode(_klein_template(prompt))
    if len(ids_full) > SEQ:
        raise Error(
            String("klein9b_text_smoke: prompt too long: ")
            + String(len(ids_full))
            + " > "
            + String(SEQ)
        )
    var ids = List[Int](capacity=SEQ)
    for i in range(len(ids_full)):
        ids.append(ids_full[i])
    for _ in range(SEQ - len(ids_full)):
        ids.append(PAD_ID)
    print("  tokens:", len(ids_full), "->", SEQ)
    return ids^


def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    for i in range(n):
        var v = Float64(h[i])
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print(
        "  [stat]", name, "mean=", Float32(mean), "std=", Float32(sqrt(var_)),
        "absmax=", Float32(amax), "n=", n,
    )


def _print_shape(label: String, t: Tensor):
    var s = t.shape()
    print(label, s[0], s[1], s[2])


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein 9B text smoke — Qwen3-8B -> [1,512,12288] ===")
    var tok = Qwen3Tokenizer(TOK_JSON)
    var pos_ids = _tokenize_512(tok, PROMPT)
    var neg_ids = _tokenize_512(tok, NEGATIVE)

    print("[load] Qwen3-8B BF16 shards")
    var enc = Qwen3Encoder.load(QWEN8_DIR, Qwen3Config.klein_9b(), ctx)

    print("[encode] positive")
    var pos = enc.encode_klein(pos_ids, ctx)
    _print_shape("  pos shape:", pos)
    _stats("pos", pos, ctx)

    print("[encode] negative")
    var neg = enc.encode_klein(neg_ids, ctx)
    _print_shape("  neg shape:", neg)
    _stats("neg", neg, ctx)

    print("Klein 9B text smoke PASS")
