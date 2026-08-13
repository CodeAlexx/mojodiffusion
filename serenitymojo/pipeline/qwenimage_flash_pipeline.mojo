# qwenimage_pipeline_1024_multistep.mojo — Qwen-Image 1024 quality runner.
#
# This is separate from qwenimage_pipeline_smoke.mojo on purpose. The smoke is a
# wiring proof; this runner follows the diffusers/Rust text-conditioning path:
# encode the template-padded prompt at a supported text-encoder length, drop
# the 34-token system/template prefix, and run a real multistep CFG denoise
# before VAE decode.
#
# Unlike the original (which hardcoded `N_TXT_POS = 27` / `N_TXT_NEG = 14` for
# the kept-token counts of two specific prompts), this version pads both
# branches to a generous comptime maximum `N_TXT_KEPT = 512` and tells the DiT
# the true number of non-padded text tokens at runtime. The DiT then builds an
# additive attention mask that zeroes out padded text positions (-1e4 bias on
# the masked key columns). Any prompt up to N_TXT_KEPT tokens works without
# recompiling — matching the Rust reference's `narrow(1, DROP_IDX, kept_len)`
# behavior, just done via mask instead of physical narrowing (Mojo can't
# narrow at runtime because tensor shapes are comptime-fixed).

from max.gpu.host import DeviceContext
from std.sys import argv
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen25vl_encoder import (
    Qwen25VLEncoder,
    Qwen25VLConfig,
)
from serenitymojo.models.dit.qwenimage_dit import QwenImageDitOffloaded
from serenitymojo.models.vae.qwenimage_tiled_decode import qwenimage_tiled_decode
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.layout import patchify, unpatchify
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.sampling.flow_match import (
    Scheduler,
    cfg_qwen,
)
from serenitymojo.image.png import save_png, ValueRange


comptime QWENIMAGE_DIR = "/home/alex/.serenity/models/checkpoints/qwen-image-2512"
comptime TEXT_ENCODER_DIR = QWENIMAGE_DIR + "/text_encoder"
comptime TOK_JSON = QWENIMAGE_DIR + "/tokenizer/tokenizer.json"
comptime DIT_DIR = "/mnt/disk2/models/Qwen-Image-Flash/transformer"  # Flash distilled
comptime VAE_DIR = QWENIMAGE_DIR + "/vae"
comptime OUT = "/home/alex/mojodiffusion/output/qwenimage_flash_4step.png"

comptime PAD_ID = 151643
comptime DROP_IDX = 34
comptime N_TXT_KEPT = 512  # comptime cap on kept text tokens; pad mask handles slack
comptime N_ENC = N_TXT_KEPT + DROP_IDX  # 546 — full encoder seq length we feed in
comptime LH = 128
comptime LW = 128
comptime PATCH = 2
comptime N_IMG = (LH // PATCH) * (LW // PATCH)
comptime S_POS = N_IMG + N_TXT_KEPT
comptime S_NEG = N_IMG + N_TXT_KEPT
comptime FRAME = 1
comptime FH = LH // PATCH
comptime FW = LW // PATCH
comptime STEPS = 4
comptime CFG = Float32(1.0)  # Flash: CFG internalized
comptime SEED = UInt64(42)
comptime EXTRACT_LAYER = 27

comptime PROMPT = "Set in medieval times. A woman is riding a horse down a village street. She is riding away from the viewer. there is a large foreboding castle in the distance. Lightning can be seen streaking across the sky. She has long messy auburn hair. She is wearing dark leather armor. The horse has a saddle and saddle bags. It is raining and there are puddles forming in the dirt. sharp scenery, lush background, ultra-detailed environment, natural textures, vibrant lighting, crisp clouds, realistic water surface, vivid skies, photo-real terrain, fantstyle, MythP0rt, raz'sscenesmith-mk.1"
comptime NEGATIVE = "low quality, blurry, watermark, jpeg artifacts"


@fieldwise_init
struct QwenCaps(Movable):
    var pos: Tensor
    var neg: Tensor
    var real_pos: Int
    var real_neg: Int


@fieldwise_init
struct EncodedCaption(Movable):
    var hidden: Tensor
    var real_len: Int

    # Consume self + neg, build the final QwenCaps in one expression so we
    # never have a partial-move of either struct (Mojo's borrow checker
    # rejects reading one field after `^`-moving another).
    def into_caps(deinit self, deinit neg: EncodedCaption) -> QwenCaps:
        return QwenCaps(self.hidden^, neg.hidden^, self.real_len, neg.real_len)


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


def _qwen_template(prompt: String) -> String:
    return (
        String("<|im_start|>system\nDescribe the image by detailing the color,"
        " shape, size, texture, quantity, text, spatial relationships of the"
        " objects and background:<|im_end|>\n<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


# Tokenize and pad to N_ENC. Returns (ids, real_kept_len) where real_kept_len
# is the number of non-padded text tokens *after* the DROP_IDX template prefix
# (matches what the Rust reference passes as kept_len to narrow). Raises if the
# prompt would overflow the comptime max or has no tokens past the template.
def _tokenize_for_encoder(
    tok: Qwen3Tokenizer, prompt: String
) raises -> Tuple[List[Int], Int]:
    var ids_full = tok.encode(_qwen_template(prompt))
    var real_len = len(ids_full)
    if real_len <= DROP_IDX:
        raise Error(
            String("Qwen-Image prompt tokenized to ")
            + String(real_len)
            + " tokens, not enough past template DROP_IDX="
            + String(DROP_IDX)
        )
    if real_len > N_ENC:
        raise Error(
            String("Qwen-Image prompt tokenized to ")
            + String(real_len)
            + " tokens, exceeding N_ENC="
            + String(N_ENC)
            + " (N_TXT_KEPT="
            + String(N_TXT_KEPT)
            + " + DROP_IDX="
            + String(DROP_IDX)
            + "); raise N_TXT_KEPT or shorten the prompt"
        )
    var real_kept_len = real_len - DROP_IDX
    var ids = List[Int](capacity=N_ENC)
    for i in range(real_len):
        ids.append(ids_full[i])
    for _ in range(N_ENC - real_len):
        ids.append(PAD_ID)
    print("  tokens:", real_len, "-> drop", DROP_IDX, "-> kept", real_kept_len, "(cap", N_TXT_KEPT, ")")
    return (ids^, real_kept_len)


# Encode the padded ids, then slice out the kept text region as [1, N_TXT_KEPT, 3584].
# The kept region still contains pad-token hidden states past `real_kept_len`;
# the DiT mask blocks attention to them, so they don't affect the output.
def _encode_trimmed(
    enc: Qwen25VLEncoder, tok: Qwen3Tokenizer, prompt: String, ctx: DeviceContext
) raises -> EncodedCaption:
    var tup = _tokenize_for_encoder(tok, prompt)
    var ids = tup[0].copy()
    var real_kept_len = tup[1]
    var pre = enc.encode(ids, EXTRACT_LAYER, ctx)
    var full = enc.final_norm(pre, ctx)
    var hidden = slice(full, 1, DROP_IDX, N_TXT_KEPT, ctx)
    return EncodedCaption(hidden^, real_kept_len)


def encode_prompt(prompt: String, ctx: DeviceContext) raises -> EncodedCaption:
    print("[text] Qwen2.5-VL text encoder (CFG-free single prompt), N_TXT_KEPT=", N_TXT_KEPT)
    var tok = Qwen3Tokenizer(TOK_JSON)
    var enc = Qwen25VLEncoder.load(
        TEXT_ENCODER_DIR, Qwen25VLConfig.qwen_image(), ctx
    )
    var pos = _encode_trimmed(enc, tok, prompt, ctx)
    _stats("text_trimmed", pos.hidden, ctx)
    return pos^


def initial_latent_packed(ctx: DeviceContext) raises -> Tensor:
    var nchw_shape = List[Int]()
    nchw_shape.append(1)
    nchw_shape.append(16)
    nchw_shape.append(LH)
    nchw_shape.append(LW)
    var noise = randn(nchw_shape^, SEED, STDtype.BF16, ctx)
    return patchify(noise, PATCH, ctx)


def denoise(cap: EncodedCaption, ctx: DeviceContext) raises -> Tensor:
    print("[denoise] loading Qwen-Image-Flash MMDiT (block-streamed)")
    var model = QwenImageDitOffloaded.load(DIT_DIR, ctx)
    var sched = Scheduler(STEPS, Float32(3.0))  # Flash: static shift-3 -> sigmas [1,.9,.75,.5,0]
    var sigmas = sched.sigmas()
    print("[denoise]", STEPS, "steps, CFG-FREE single branch, seed", SEED)
    print("  real text tokens=", cap.real_len, " (cap", N_TXT_KEPT, ")")
    var x = initial_latent_packed(ctx)
    _stats("init_latent", x, ctx)
    for i in range(STEPS):
        print("  step", i + 1, "/", STEPS, "sigma", sigmas[i], "->", sigmas[i + 1])
        var pred = model.forward_masked[N_IMG, N_TXT_KEPT, S_POS](
            x, cap.hidden, sigmas[i], cap.real_len, FRAME, FH, FW, ctx
        )
        x = sched.step(x, pred, i, ctx)
        if (i + 1) % 5 == 0 or i == 0 or i + 1 == STEPS:
            _stats("latent", x, ctx)
    return x^


def main() raises:
    var a = argv()
    var prompt = String(a[1]) if len(a) > 1 else PROMPT
    var out = String(a[2]) if len(a) > 2 else OUT
    var ctx = DeviceContext()
    print("=== Qwen-Image-Flash 4-step CFG-free (pure Mojo) ===")
    print("[prompt]", prompt)
    var cap = encode_prompt(prompt, ctx)
    var tokens = denoise(cap, ctx)
    print("[vae] unpack + tiled decode")
    var latent = unpatchify(tokens, 16, LH, LW, PATCH, ctx)
    latent = cast_tensor(latent, STDtype.BF16, ctx)
    var img = qwenimage_tiled_decode[LH, LW](latent, VAE_DIR, ctx)
    var sh = img.shape()
    print("  image shape:", sh[0], sh[1], sh[2], sh[3])
    _stats("image", img, ctx)
    save_png(img, out, ctx, ValueRange.SIGNED)
    print("[done] saved", out)
