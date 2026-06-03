# Mistral-3B ERNIE text encoder smoke.
#
# Loads /home/alex/models/ERNIE-Image/text_encoder/model.safetensors and runs a
# short fixed prompt through the pure-Mojo Mistral3bEncoder. This is a smoke
# gate, not a full numeric oracle; it proves the local ERNIE text path can load,
# run, and emit the expected [1,256,3072] training embedding.
#
# Run:
#   cd /home/alex/mojodiffusion
#   pixi run mojo run -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/text_encoder/parity/mistral3b_smoke.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.mistral3b_encoder import (
    Mistral3bEncoder,
    mistral3_tokenize,
)


comptime TOK_JSON: StaticString = "/home/alex/models/ERNIE-Image/tokenizer/tokenizer.json"
comptime TEXT_ENCODER_DIR: StaticString = "/home/alex/models/ERNIE-Image/text_encoder"
comptime PROMPT: StaticString = "box1jana, a high-resolution studio portrait, ornate chair, cocktail, studio lighting"


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
    print("=== ERNIE Mistral-3B text smoke -> [1,256,3072] ===")
    var tok = Qwen3Tokenizer(String(TOK_JSON))
    var ids = mistral3_tokenize(tok, String(PROMPT))
    print("tokens:", len(ids))

    print("[load] ERNIE text_encoder BF16")
    var enc = Mistral3bEncoder.load(String(TEXT_ENCODER_DIR), ctx)

    print("[encode] prompt")
    var out = enc.encode(ids, 256, ctx)
    var real_len = out[1]
    print("real_len:", real_len)
    _print_shape("embedding shape:", out[0])
    _stats("mistral3b", out[0], ctx)
    print("Mistral3B text smoke PASS")
