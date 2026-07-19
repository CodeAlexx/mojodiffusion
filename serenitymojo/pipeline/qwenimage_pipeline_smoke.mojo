# qwenimage_pipeline_smoke.mojo — first end-to-end Qwen-Image T2I smoke.
#
# Wires the full pure-Mojo Qwen-Image path:
#   tokenizer -> Qwen2.5-VL text encoder (last_hidden_state) -> MMDiT denoise
#   (rectified-flow Euler + true CFG) -> Qwen-Image 3D causal VAE decode -> PNG.
#
# *** CODE-ONLY: this file is COMPILE-VERIFIED, NOT executed. The GPU is wedged
# (illegal-address fault awaiting reboot). RUN later post-reboot. ***
#
# Resolution: 512x512 (latent 64x64). Keeps the comptime instantiations modest:
#   * DiT seq:  N_TXT (text) + N_IMG (image patches). N_IMG = (64/2)^2 = 1024.
#   * VAE mid-attn seq: LH*LW = 64*64 = 4096.
# Scale up to 1024 by changing LH/LW (and N_IMG) — the comptime sdpa cases grow.
#
# References (read line-by-line):
#   * inference-flame/src/bin/qwenimage_gen.rs (pack/unpack, Euler loop)
#   * inference-flame/src/models/qwenimage_dit.rs (DiT forward)
#   * inference-flame/src/vae/qwenimage_decoder.rs + wan21_vae.rs (VAE)
#   * sampling/flow_match.mojo Scheduler.qwen / cfg_qwen (Qwen schedule + CFG)
#
# NOTE: the diffusers Qwen-Image pipeline drops the chat-template prefix tokens
# from the text hidden states before the DiT (drop_idx). This smoke keeps the
# full padded sequence as text conditioning (wiring/memory proof). Token
# dropping + exact template are a parity-time refinement — flagged in report.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen25vl_encoder import (
    Qwen25VLEncoder,
    Qwen25VLConfig,
)
from serenitymojo.models.dit.qwenimage_dit import QwenImageDitOffloaded
from serenitymojo.models.vae.qwenimage_decoder import QwenImageVaeDecoder
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn
from serenitymojo.ops.layout import patchify, unpatchify
from serenitymojo.ops.tensor_algebra import add, mul_scalar, reshape, permute, slice
from serenitymojo.sampling.flow_match import (
    Scheduler,
    cfg_qwen,
    build_qwen_sigma_schedule,
)
from serenitymojo.image.png import save_png, ValueRange


# ── paths / config ────────────────────────────────────────────────────────────
# Qwen-Image text encoder (Qwen2.5-VL-7B language tower) + DiT + VAE dirs. These
# are placeholders pointing at the standard diffusers Qwen-Image layout; adjust
# at run time. The text encoder is the `text_encoder/` subdir of the snapshot.
comptime QWENIMAGE_DIR = "/home/alex/.serenity/models/checkpoints/qwen-image-2512"
comptime TEXT_ENCODER_DIR = QWENIMAGE_DIR + "/text_encoder"
comptime TOK_JSON = QWENIMAGE_DIR + "/tokenizer/tokenizer.json"
comptime DIT_DIR = QWENIMAGE_DIR + "/transformer"
comptime VAE_DIR = QWENIMAGE_DIR + "/vae"
comptime OUT = "/home/alex/mojodiffusion/output/qwenimage_first_512.png"

comptime PAD_ID = 151643
comptime N_TXT = 256          # padded text length
comptime LH = 64              # latent height (512 / 8)
comptime LW = 64              # latent width
comptime PATCH = 2
comptime N_IMG = (LH // PATCH) * (LW // PATCH)  # (32)*(32) = 1024
comptime S = N_TXT + N_IMG
comptime FRAME = 1
comptime FH = LH // PATCH      # rope patch-grid height = 32
comptime FW = LW // PATCH      # rope patch-grid width  = 32
comptime STEPS = 1
comptime CFG = Float32(4.0)
comptime SEED = UInt64(42)
comptime EXTRACT_LAYER = 27    # Qwen2.5-VL 28-layer -> last index (then final_norm)

comptime PROMPT = (
    "a cinematic portrait of a woman in dramatic neon lighting, sharp focus,"
    " rich color, detailed face, fashion editorial"
)
comptime NEGATIVE = "low quality, blurry, watermark, jpeg artifacts"


@fieldwise_init
struct QwenCaps(Movable):
    var pos: Tensor   # [1, N_TXT, 3584]
    var neg: Tensor


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


# Qwen-Image chat template (diffusers pipeline_qwenimage.py prompt template).
def _qwen_template(prompt: String) -> String:
    return (
        String("<|im_start|>system\nDescribe the image by detailing the color,"
        " shape, size, texture, quantity, text, spatial relationships of the"
        " objects and background:<|im_end|>\n<|im_start|>user\n")
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def _tokenize_padded(tok: Qwen3Tokenizer, prompt: String) raises -> List[Int]:
    var ids_full = tok.encode(_qwen_template(prompt))
    var ids = List[Int](capacity=N_TXT)
    var keep = len(ids_full)
    if keep > N_TXT:
        keep = N_TXT
    for i in range(keep):
        ids.append(ids_full[i])
    for _ in range(N_TXT - keep):
        ids.append(PAD_ID)
    print("  tokens:", len(ids_full), "-> padded", N_TXT)
    return ids^


def encode_captions(ctx: DeviceContext) raises -> QwenCaps:
    print("[text] Qwen2.5-VL text encoder")
    var tok = Qwen3Tokenizer(TOK_JSON)
    var pos_ids = _tokenize_padded(tok, PROMPT)
    var neg_ids = _tokenize_padded(tok, NEGATIVE)
    var enc = Qwen25VLEncoder.load(
        TEXT_ENCODER_DIR, Qwen25VLConfig.qwen_image(), ctx
    )
    # last_hidden_state = encode(extract=last) then final_norm (matches Rust
    # qwen25vl_encoder.rs:609-642 encode()).
    var pos_pre = enc.encode(pos_ids, EXTRACT_LAYER, ctx)
    var pos = enc.final_norm(pos_pre, ctx)
    var neg_pre = enc.encode(neg_ids, EXTRACT_LAYER, ctx)
    var neg = enc.final_norm(neg_pre, ctx)
    _stats("text_pos", pos, ctx)
    return QwenCaps(pos^, neg^)


# Draw NCHW [1,16,LH,LW] latent noise, pack to DiT tokens [1,N_IMG,64].
def initial_latent_packed(ctx: DeviceContext) raises -> Tensor:
    var nchw_shape = List[Int]()
    nchw_shape.append(1)
    nchw_shape.append(16)
    nchw_shape.append(LH)
    nchw_shape.append(LW)
    var noise = randn(nchw_shape^, SEED, STDtype.F32, ctx)
    # patchify: [1,16,LH,LW] -> [1, N_IMG, 64] (within-patch order (c,ph,pw),
    # matching diffusers _pack_latents — ops/layout.patchify header + Rust pack).
    return patchify(noise, PATCH, ctx)


def denoise(caps: QwenCaps, ctx: DeviceContext) raises -> Tensor:
    print("[denoise] loading Qwen-Image MMDiT (block-streamed)")
    var model = QwenImageDitOffloaded.load(DIT_DIR, ctx)
    var sched = Scheduler.qwen(STEPS, Float32(N_IMG))
    var sigmas = sched.sigmas()
    print("[denoise]", STEPS, "steps, CFG", CFG, "seed", SEED)
    var x = initial_latent_packed(ctx)  # [1, N_IMG, 64] F32
    _stats("init_latent", x, ctx)
    for i in range(STEPS):
        print("  step", i + 1, "/", STEPS, "sigma", sigmas[i], "->", sigmas[i + 1])
        var t_sigma = sigmas[i]
        var xb = cast_tensor(x, STDtype.BF16, ctx)
        # cond + uncond forwards (true CFG). DiT returns velocity [1, N_IMG, 64].
        # Smoke pipeline uses a fixed N_TXT with no padded tokens, so pass
        # real_txt_len = N_TXT to make the padding mask a no-op (matches the
        # historical _zeros_mask behavior).
        var preds = model.forward_cfg[N_IMG, N_TXT, S](
            xb, caps.pos, caps.neg, t_sigma, N_TXT, FRAME, FH, FW, ctx
        )
        var pred_pos = cast_tensor(
            preds.pos,
            STDtype.F32,
            ctx,
        )
        var pred_neg = cast_tensor(
            preds.neg,
            STDtype.F32,
            ctx,
        )
        var pred = cfg_qwen(pred_pos, pred_neg, CFG, ctx)
        x = sched.step(x, pred, i, ctx)
        _stats("latent", x, ctx)
    return x^


def main() raises:
    var ctx = DeviceContext()
    print("=== Qwen-Image first image smoke — 512x512, one denoise step ===")
    var caps = encode_captions(ctx)
    var tokens = denoise(caps, ctx)  # [1, N_IMG, 64]
    print("[vae] unpack + decode")
    # unpatchify [1,N_IMG,64] -> latent NCHW [1,16,LH,LW]
    var latent = unpatchify(tokens, 16, LH, LW, PATCH, ctx)
    latent = cast_tensor(latent, STDtype.BF16, ctx)
    var vae = QwenImageVaeDecoder[LH, LW].load(VAE_DIR, ctx)
    var img = vae.decode(latent, ctx)  # [1,3,8*LH,8*LW]
    var sh = img.shape()
    print("  image shape:", sh[0], sh[1], sh[2], sh[3])
    _stats("image", img, ctx)
    save_png(img, OUT, ctx, ValueRange.SIGNED)
    print("[done] saved", OUT)
