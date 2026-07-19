# training/ernie_validation_sampler.mojo — ERNIE LoRA validation sample path.
#
# This is the training-path sampler for ERNIE's current 512px real-cache LoRA
# smoke: cached Mistral text embedding -> resident BF16 ERNIE LoRA DiT denoise
# -> Klein/ERNIE VAE decode -> PNG. It also supports a baseline-vs-live LoRA
# pair so the trainer can report a concrete sample-shift metric.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, sin as fsin, cos as fcos

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import silu
from serenitymojo.ops.embeddings import timestep_embedding_sin_first
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.dit.ernie_contract import ERNIE_VAE_FILE
from serenitymojo.models.dit.ernie_image import build_ernie_rope_tables
from serenitymojo.models.vae.klein_decoder import KleinVaeDecoder
from serenitymojo.models.ernie.block import ErnieModVecs
from serenitymojo.models.ernie.weights import ErnieBlockWeights, ErnieStackBase
from serenitymojo.models.ernie.ernie_stack_lora import (
    ErnieLoraSet,
    ernie_lora_set_to_device,
    ernie_stack_lora_predict_resident_device,
)
from serenitymojo.sampling.ernie_sampling import (
    build_ernie_sigma_schedule,
    ernie_model_timestep_from_sigma,
)


comptime H = 32
comptime Dh = 128
comptime D = H * Dh
comptime F = 12288
comptime IN_CH = 128
comptime TEXT_IN = 3072
comptime OUT_CH = 128
comptime NUM_LAYERS = 36
comptime EPS = Float32(1e-06)

comptime IMG_H = 32
comptime IMG_W = 32
comptime N_IMG = IMG_H * IMG_W
comptime N_TXT = 256
comptime S = N_IMG + N_TXT
comptime SAMPLE_SHIFT = Float32(3.0)


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _ones(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(1.0)
    return o^


def _chunk(src: List[Float32], idx: Int, width: Int) -> List[Float32]:
    var o = List[Float32]()
    var off = idx * width
    for i in range(width):
        o.append(src[off + i])
    return o^


def _shared_adaln_source(
    base: ErnieStackBase, timestep_value: Float32, ctx: DeviceContext
) raises -> Tuple[ErnieModVecs, List[Float32], List[Float32]]:
    var ts = List[Float32]()
    ts.append(timestep_value)
    var ts_t = Tensor.from_host(ts, [1], STDtype.F32, ctx)
    var emb_in = timestep_embedding_sin_first(
        ts_t, D, ctx, 10000.0, base.te_w1[].dtype()
    )
    var h1 = linear(emb_in, base.te_w1[], Optional[Tensor](base.te_b1[].clone(ctx)), ctx)
    h1 = silu(h1, ctx)
    var c = linear(h1, base.te_w2[], Optional[Tensor](base.te_b2[].clone(ctx)), ctx)

    var sc = silu(c, ctx)
    var adaln = linear(sc, base.adaln_w[], Optional[Tensor](base.adaln_b[].clone(ctx)), ctx)
    var adaln_h = adaln.to_host(ctx)

    var fmod = linear(c, base.final_norm_w[], Optional[Tensor](base.final_norm_b[].clone(ctx)), ctx)
    var fmod_h = fmod.to_host(ctx)

    var mv = ErnieModVecs(
        _chunk(adaln_h, 0, D), _chunk(adaln_h, 1, D), _chunk(adaln_h, 2, D),
        _chunk(adaln_h, 3, D), _chunk(adaln_h, 4, D), _chunk(adaln_h, 5, D),
    )
    var f_scale = _chunk(fmod_h, 0, D)
    var f_shift = _chunk(fmod_h, 1, D)
    return (mv^, f_scale^, f_shift^)


def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var have = False
    var spare = Float32(0.0)
    for _ in range(n):
        if have:
            out.append(spare)
            have = False
            continue
        state = state * 6364136223846793005 + 1442695040888963407
        var u1 = (Float32(Int(state >> 40)) + 1.0) * Float32(1.0 / 16777217.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2 = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        var r = sqrt(Float32(-2.0) * flog(u1))
        var theta = Float32(6.283185307179586) * u2
        out.append(r * fcos(theta))
        spare = r * fsin(theta)
        have = True
    return out^


def _tokens_to_latent_nchw(tokens: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    var hw = IMG_H * IMG_W
    for ch in range(IN_CH):
        for t in range(hw):
            out.append(tokens[t * IN_CH + ch])
    return out^


def _text_to_txt_tokens(text: List[Float32], t_cache: Int) -> List[Float32]:
    var out = List[Float32]()
    for r in range(N_TXT):
        if r < t_cache:
            for c in range(TEXT_IN):
                out.append(text[r * TEXT_IN + c])
        else:
            for _c in range(TEXT_IN):
                out.append(Float32(0.0))
    return out^


def load_ernie_sample_text_tokens(path: String, ctx: DeviceContext) raises -> List[Float32]:
    """Load ERNIE cached text_embedding from a sample-prompt caps path.

    Current ERNIE training uses prepared sample `.safetensors` files with a
    `text_embedding` tensor [1,T,3072]. The shared prompt JSON points at one of
    those files until the Mojo Mistral encoder/cache writer exists.
    """
    var st = SafeTensors.open(path)
    var info = st.tensor_info(String("text_embedding"))
    if len(info.shape) != 3 or Int(info.shape[2]) != TEXT_IN:
        raise Error(String("ERNIE sample caps text_embedding shape mismatch: ") + path)
    var bytes = st.tensor_bytes(String("text_embedding"))
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    var f = cast_tensor(t, STDtype.F32, ctx)
    var h = f.to_host(ctx)
    return _text_to_txt_tokens(h^, Int(info.shape[1]))


def denoise_ernie_training_latent(
    base: ErnieStackBase,
    blocks: List[ErnieBlockWeights],
    lora: ErnieLoraSet,
    txt_tokens: List[Float32],
    sample_steps: Int,
    seed: UInt64,
    ctx: DeviceContext,
) raises -> Tensor:
    if sample_steps < 1:
        raise Error("denoise_ernie_training_latent: sample_steps must be >= 1")
    if len(blocks) != NUM_LAYERS:
        raise Error("denoise_ernie_training_latent: expected 36 ERNIE blocks")
    if len(txt_tokens) != N_TXT * TEXT_IN:
        raise Error("denoise_ernie_training_latent: text token shape mismatch")

    var lora_dev = ernie_lora_set_to_device(lora, STDtype.BF16, ctx)
    var rope = build_ernie_rope_tables[N_IMG, N_TXT, H, Dh](
        IMG_H, IMG_W, N_TXT, ctx, STDtype.BF16
    )
    var sigmas = build_ernie_sigma_schedule(sample_steps, SAMPLE_SHIFT)
    var latent_tokens = _host_noise(N_IMG * IN_CH, seed)

    for step in range(sample_steps):
        var sigma = sigmas[step]
        var sigma_next = sigmas[step + 1]
        var dt = sigma_next - sigma
        var src = _shared_adaln_source(
            base, ernie_model_timestep_from_sigma(sigma), ctx
        )
        var pred = ernie_stack_lora_predict_resident_device[H, Dh, N_IMG, N_TXT, S](
            latent_tokens.copy(), txt_tokens.copy(), base, blocks, lora_dev, src[0],
            src[1].copy(), src[2].copy(), rope[0], rope[1],
            D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
        )
        for i in range(len(latent_tokens)):
            latent_tokens[i] = latent_tokens[i] + pred[i] * dt

    var nchw = _tokens_to_latent_nchw(latent_tokens)
    return Tensor.from_host(nchw, [1, IN_CH, IMG_H, IMG_W], STDtype.F32, ctx)


def pixel_l1(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float32:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        raise Error("pixel_l1: shape mismatch")
    if len(ah) == 0:
        return Float32(0.0)
    var s = Float64(0.0)
    for i in range(len(ah)):
        var d = Float64(ah[i] - bh[i])
        s += d if d >= 0.0 else -d
    return Float32(s / Float64(len(ah)))


def _resize_rgb_nchw_nearest(
    img: Tensor, target_width: Int, target_height: Int, ctx: DeviceContext
) raises -> Tensor:
    var shape = img.shape()
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
        raise Error("_resize_rgb_nchw_nearest: expected RGB [1,3,H,W]")
    var src_h = shape[2]
    var src_w = shape[3]
    if target_width < 1024 or target_height < 1024:
        raise Error("ERNIE sample output must be 1024x1024 or larger")
    if target_width < src_w or target_height < src_h:
        raise Error("_resize_rgb_nchw_nearest: refusing to downscale validation sample")
    if target_width == src_w and target_height == src_h:
        return img.clone(ctx)

    var src = img.to_host(ctx)
    var out = List[Float32]()
    var src_plane = src_h * src_w
    for c in range(3):
        for y in range(target_height):
            var sy = (y * src_h) // target_height
            for x in range(target_width):
                var sx = (x * src_w) // target_width
                out.append(src[c * src_plane + sy * src_w + sx])
    return Tensor.from_host(
        out, [1, 3, target_height, target_width], STDtype.F32, ctx
    )


def generate_ernie_lora_shift_pair(
    base: ErnieStackBase,
    blocks: List[ErnieBlockWeights],
    baseline_lora: ErnieLoraSet,
    live_lora: ErnieLoraSet,
    txt_tokens: List[Float32],
    out_png: String,
    target_width: Int,
    target_height: Int,
    sample_steps: Int,
    seed: UInt64,
    ctx: DeviceContext,
) raises -> Float32:
    print("[ernie-sample] baseline denoise steps=", sample_steps)
    var base_latent = denoise_ernie_training_latent(
        base, blocks, baseline_lora, txt_tokens.copy(), sample_steps, seed, ctx
    )
    print("[ernie-sample] live-LoRA denoise steps=", sample_steps)
    var live_latent = denoise_ernie_training_latent(
        base, blocks, live_lora, txt_tokens, sample_steps, seed, ctx
    )

    print("[ernie-sample] VAE decode")
    var vae = KleinVaeDecoder[IMG_H, IMG_W].load(String(ERNIE_VAE_FILE), ctx)
    var base_img = vae.decode(base_latent, ctx)
    var live_img = vae.decode(live_latent, ctx)
    base_img = _resize_rgb_nchw_nearest(base_img, target_width, target_height, ctx)
    live_img = _resize_rgb_nchw_nearest(live_img, target_width, target_height, ctx)
    var diff = pixel_l1(base_img, live_img, ctx)

    if out_png != String(""):
        save_png(base_img, out_png + String(".baseline.png"), ctx, ValueRange.SIGNED)
        save_png(live_img, out_png, ctx, ValueRange.SIGNED)
        print("[ernie-sample] saved baseline:", out_png + String(".baseline.png"))
        print("[ernie-sample] saved live:", out_png)
    print("[ernie-sample] pixel_l1 baseline_vs_live=", diff)
    return diff
