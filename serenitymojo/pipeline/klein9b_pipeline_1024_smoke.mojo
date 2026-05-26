# klein9b_pipeline_1024_smoke.mojo - first end-to-end Klein 9B image smoke.
#
# This is intentionally one denoise step: it proves the real 1024 token shape
# and image path before optimizing 50-step quality/performance.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.dit.klein_dit import Klein9BOffloaded, build_klein_rope_tables
from serenitymojo.models.vae.klein_decoder import KleinVaeDecoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar, reshape, permute
from serenitymojo.sampling.flux2_klein import build_flux2_sigma_schedule, flux2_cfg
from serenitymojo.image.png import save_png, ValueRange


comptime QWEN8_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
comptime TOK_JSON = QWEN8_DIR + "/tokenizer.json"
comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
comptime VAE_PATH = "/home/alex/.serenity/models/vaes/flux2-vae.safetensors"
comptime OUT = "/home/alex/mojodiffusion/output/klein9b_first_1024.png"
comptime PAD_ID = 151643
comptime SEQ = 512
comptime N_IMG = 4096
comptime N_TXT = 512
comptime S = N_IMG + N_TXT
comptime LH = 64
comptime LW = 64
comptime STEPS = 1
comptime CFG = Float32(4.0)
comptime SEED = UInt64(42)

comptime PROMPT = (
    "a cinematic portrait of a woman in dramatic neon lighting, sharp focus,"
    " rich color, detailed face, fashion editorial, 1024x1024"
)
comptime NEGATIVE = "low quality, blurry, watermark, jpeg artifacts"


@fieldwise_init
struct KleinCaps(Movable):
    var pos: Tensor
    var neg: Tensor


def _klein_template(prompt: String) -> String:
    return (
        String("<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _tokenize_512(tok: Qwen3Tokenizer, prompt: String) raises -> List[Int]:
    var ids_full = tok.encode(_klein_template(prompt))
    if len(ids_full) > SEQ:
        raise Error("Klein prompt too long for 512 tokens")
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


def encode_captions(ctx: DeviceContext) raises -> KleinCaps:
    print("[text] Qwen3-8B Klein conditioning")
    var tok = Qwen3Tokenizer(TOK_JSON)
    var pos_ids = _tokenize_512(tok, PROMPT)
    var neg_ids = _tokenize_512(tok, NEGATIVE)
    var enc = Qwen3Encoder.load(QWEN8_DIR, Qwen3Config.klein_9b(), ctx)
    var pos = enc.encode_klein(pos_ids, ctx)
    var neg = enc.encode_klein(neg_ids, ctx)
    _stats("text_pos", pos, ctx)
    return KleinCaps(pos^, neg^)


def initial_tokens(ctx: DeviceContext) raises -> Tensor:
    # Match the reference layout: draw NCHW [1,128,LH,LW], then pack to
    # token NHWC [1,N_IMG,128]. The draw itself stays GPU-resident.
    var nchw_shape = List[Int]()
    nchw_shape.append(1)
    nchw_shape.append(128)
    nchw_shape.append(LH)
    nchw_shape.append(LW)
    var noise_nchw = randn(nchw_shape^, SEED, STDtype.F32, ctx)
    var p = List[Int]()
    p.append(0)
    p.append(2)
    p.append(3)
    p.append(1)
    var nhwc = permute(noise_nchw, p^, ctx)
    var sh = List[Int]()
    sh.append(1)
    sh.append(N_IMG)
    sh.append(128)
    return reshape(nhwc, sh^, ctx)


def tokens_to_packed_nchw(tokens: Tensor, ctx: DeviceContext) raises -> Tensor:
    var nhwc_shape = List[Int]()
    nhwc_shape.append(1)
    nhwc_shape.append(LH)
    nhwc_shape.append(LW)
    nhwc_shape.append(128)
    var nhwc = reshape(tokens, nhwc_shape^, ctx)
    var p = List[Int]()
    p.append(0)
    p.append(3)
    p.append(1)
    p.append(2)
    return permute(nhwc, p^, ctx)


def denoise(caps: KleinCaps, ctx: DeviceContext) raises -> Tensor:
    print("[denoise] loading Klein 9B offloaded DiT")
    var model = Klein9BOffloaded.load(KLEIN9B_PATH, ctx)
    var rope = build_klein_rope_tables[N_IMG, N_TXT, 32, 128](ctx, STDtype.BF16)
    var sigmas = build_flux2_sigma_schedule(STEPS, N_IMG)
    print("[denoise]", STEPS, "steps, CFG", CFG, "seed", SEED)
    var x = initial_tokens(ctx)
    _stats("init_tokens", x, ctx)
    for i in range(STEPS):
        print("  step", i + 1, "/", STEPS, "sigma", sigmas[i], "->", sigmas[i + 1])
        var tvals = List[Float32]()
        tvals.append(sigmas[i])
        var tsh = List[Int]()
        tsh.append(1)
        var timestep = Tensor.from_host(tvals, tsh^, STDtype.F32, ctx)
        var xb = cast_tensor(x, STDtype.BF16, ctx)
        var pred_pos = cast_tensor(
            model.forward_full[N_IMG, N_TXT, S](
                xb, caps.pos, timestep, rope[0], rope[1], ctx
            ),
            STDtype.F32,
            ctx,
        )
        var pred_neg = cast_tensor(
            model.forward_full[N_IMG, N_TXT, S](
                xb, caps.neg, timestep, rope[0], rope[1], ctx
            ),
            STDtype.F32,
            ctx,
        )
        var pred = flux2_cfg(pred_pos, pred_neg, CFG, ctx)
        var dt = sigmas[i + 1] - sigmas[i]
        x = add(x, mul_scalar(pred, dt, ctx), ctx)
        _stats("tokens", x, ctx)
    return x^


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein 9B first image smoke - 1024x1024, one denoise step ===")
    var caps = encode_captions(ctx)
    var tokens = denoise(caps, ctx)
    print("[vae] decode")
    var packed = tokens_to_packed_nchw(tokens, ctx)
    var vae = KleinVaeDecoder[LH, LW].load(VAE_PATH, ctx)
    var img = vae.decode(packed, ctx)
    var sh = img.shape()
    print("  image shape:", sh[0], sh[1], sh[2], sh[3])
    _stats("image", img, ctx)
    save_png(img, OUT, ctx, ValueRange.SIGNED)
    print("[done] saved", OUT)
