# pipeline/ernie_lora_sample_1024.mojo — full 1024 ERNIE LoRA sampler.
#
# This is a quality sampler for the completed Ernie LoRA, separate from the
# trainer's low-step validation sampler. It uses cached Mistral sidecars, real
# text lengths, CFG, streamed block loading, and the LoRA overlay.

from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.sys import argv

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.ffi import sys_system
from serenitymojo.ops.activations import silu
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.embeddings import timestep_embedding_sin_first
from serenitymojo.ops.linear import linear
from serenitymojo.ops.random import randn
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.dit.ernie_contract import (
    ERNIE_TRANSFORMER_DIR,
    ERNIE_VAE_FILE,
)
from serenitymojo.models.dit.ernie_image import build_ernie_rope_tables
from serenitymojo.models.vae.klein_decoder import KleinVaeDecoder
from serenitymojo.models.ernie.block import ErnieModVecs
from serenitymojo.models.ernie.weights import (
    ErnieStackBase,
    load_ernie_stack_base,
)
from serenitymojo.models.ernie.ernie_stack_lora import (
    ErnieLoraSet,
    load_ernie_lora_resume,
    ernie_lora_set_to_device,
    ernie_stack_lora_predict_streamed_device,
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
comptime RANK = 16
comptime ALPHA = Float32(1.0)
comptime EPS = Float32(1.0e-06)

comptime LH = 64
comptime LW = 64
comptime N_IMG = LH * LW
comptime NEG_TXT = 1
comptime SHIFT = Float32(3.0)
comptime CFG = Float32(4.0)
comptime DEFAULT_STEPS = 30
comptime DEFAULT_LORA = "/home/alex/mojodiffusion/serenitymojo/output/ernie_boxjana_2500/ernie_lora.safetensors"
comptime OUT_DIR = "/home/alex/mojodiffusion/serenitymojo/output"
comptime CAPS_DIR = "/home/alex/mojodiffusion/serenitymojo/output/ernie_prompt_caps"


@fieldwise_init
struct PromptCaps(Movable):
    var tokens: List[Float32]
    var real_len: Int


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


def _load_prompt_caps(path: String, expected_len: Int, ctx: DeviceContext) raises -> PromptCaps:
    var st = SafeTensors.open(path)

    var rinfo = st.tensor_info(String("text_real_len"))
    var rbytes = st.tensor_bytes(String("text_real_len"))
    var rtv = from_parts(rinfo.dtype, rinfo.shape.copy(), rbytes)
    var rt = cast_tensor(Tensor.from_view(rtv, ctx), STDtype.F32, ctx)
    var rh = rt.to_host(ctx)
    var real_len = Int(rh[0])
    if real_len != expected_len:
        raise Error(
            String("text_real_len mismatch for ") + path
            + String(": got ") + String(real_len)
            + String(" expected ") + String(expected_len)
        )

    var info = st.tensor_info(String("text_embedding"))
    if len(info.shape) != 3 or Int(info.shape[2]) != TEXT_IN:
        raise Error(String("text_embedding shape mismatch: ") + path)
    if Int(info.shape[1]) < expected_len:
        raise Error(String("text_embedding shorter than text_real_len: ") + path)
    var bytes = st.tensor_bytes(String("text_embedding"))
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    var t = Tensor.from_view(tv, ctx)
    var f = cast_tensor(t, STDtype.F32, ctx)
    var h = f.to_host(ctx)

    var out = List[Float32]()
    for r in range(expected_len):
        for c in range(TEXT_IN):
            out.append(h[r * TEXT_IN + c])
    return PromptCaps(out^, real_len)


def _latent_nchw_to_tokens(latent: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    var hw = LH * LW
    for t in range(hw):
        for ch in range(IN_CH):
            out.append(latent[ch * hw + t])
    return out^


def _tokens_to_latent_nchw(tokens: List[Float32], ctx: DeviceContext) raises -> Tensor:
    var out = List[Float32]()
    var hw = LH * LW
    for ch in range(IN_CH):
        for t in range(hw):
            out.append(tokens[t * IN_CH + ch])
    return Tensor.from_host(out, [1, IN_CH, LH, LW], STDtype.F32, ctx)


def _cfg(cond: List[Float32], neg: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(cond)):
        out.append(neg[i] + CFG * (cond[i] - neg[i]))
    return out^


def _stats(name: String, vals: List[Float32]):
    var n = len(vals)
    var s = Float64(0.0)
    var s2 = Float64(0.0)
    var amax = Float64(0.0)
    for i in range(n):
        var v = Float64(vals[i])
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(n)
    var var_ = s2 / Float64(n) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print("  [stat]", name, "mean=", Float32(mean), "std=", Float32(sqrt(var_)), "absmax=", Float32(amax))


def _parse_int(s: String) -> Int:
    var out = 0
    for ch in s.codepoint_slices():
        var c = String(ch)
        if c >= String("0") and c <= String("9"):
            out = out * 10 + (ord(c) - ord(String("0")))
    return out


def _sample_prompt[COND_TXT: Int, S_COND: Int](
    label: String,
    pos_path: String,
    neg_path: String,
    out_path: String,
    seed: UInt64,
    steps: Int,
    base: ErnieStackBase,
    st: ShardedSafeTensors,
    lora: ErnieLoraSet,
    ctx: DeviceContext,
) raises:
    print("\n[sample]", label, "steps=", steps, "seed=", seed)
    var cond = _load_prompt_caps(pos_path, COND_TXT, ctx)
    var neg = _load_prompt_caps(neg_path, NEG_TXT, ctx)

    var lora_dev = ernie_lora_set_to_device(lora, STDtype.BF16, ctx)
    var cond_rope = build_ernie_rope_tables[N_IMG, COND_TXT, H, Dh](
        LH, LW, cond.real_len, ctx, STDtype.BF16
    )
    var neg_rope = build_ernie_rope_tables[N_IMG, NEG_TXT, H, Dh](
        LH, LW, neg.real_len, ctx, STDtype.BF16
    )

    var sh = List[Int]()
    sh.append(1)
    sh.append(IN_CH)
    sh.append(LH)
    sh.append(LW)
    var noise = randn(sh^, seed, STDtype.BF16, ctx)
    var latent_tokens = _latent_nchw_to_tokens(noise.to_host(ctx))
    var sigmas = build_ernie_sigma_schedule(steps, SHIFT)

    for step in range(steps):
        var sigma = sigmas[step]
        var dt = sigmas[step + 1] - sigma
        print("  step", step + 1, "/", steps, "sigma=", sigma)
        var src = _shared_adaln_source(
            base, ernie_model_timestep_from_sigma(sigma), ctx
        )
        var cond_pred = ernie_stack_lora_predict_streamed_device[
            H, Dh, N_IMG, COND_TXT, S_COND
        ](
            latent_tokens.copy(), cond.tokens.copy(), base, st, lora_dev, src[0],
            src[1].copy(), src[2].copy(), cond_rope[0], cond_rope[1],
            D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
        )
        var neg_pred = ernie_stack_lora_predict_streamed_device[
            H, Dh, N_IMG, NEG_TXT, N_IMG + NEG_TXT
        ](
            latent_tokens.copy(), neg.tokens.copy(), base, st, lora_dev, src[0],
            src[1].copy(), src[2].copy(), neg_rope[0], neg_rope[1],
            D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
        )
        var pred = _cfg(cond_pred, neg_pred)
        for i in range(len(latent_tokens)):
            latent_tokens[i] = latent_tokens[i] + pred[i] * dt

    _stats(String("final_latent_tokens_") + label, latent_tokens)
    var latent = _tokens_to_latent_nchw(latent_tokens, ctx)
    print("  VAE decode")
    var vae = KleinVaeDecoder[LH, LW].load(String(ERNIE_VAE_FILE), ctx)
    var img = vae.decode(latent, ctx)
    _stats(String("decoded_image_") + label, img.to_host(ctx))
    save_png(img, out_path, ctx, ValueRange.SIGNED)
    print("  saved", out_path)


def main() raises:
    var args = argv()
    var lora_path = String(DEFAULT_LORA)
    var steps = DEFAULT_STEPS
    var mode = String("default")
    if len(args) >= 2:
        lora_path = String(args[1])
    if len(args) >= 3:
        steps = _parse_int(String(args[2]))
    if len(args) >= 4:
        mode = String(args[3])
    if steps < 1:
        raise Error("steps must be >= 1")

    _ = sys_system(String("mkdir -p ") + String(OUT_DIR))

    var ctx = DeviceContext()
    print("=== ERNIE 1024 LoRA sampler ===")
    print("  lora=", lora_path)
    print("  steps=", steps, "cfg=", CFG, "shift=", SHIFT)

    var st = ShardedSafeTensors.open(String(ERNIE_TRANSFORMER_DIR))
    var base = load_ernie_stack_base(st, D, IN_CH, ctx)
    var lora = load_ernie_lora_resume(NUM_LAYERS, RANK, ALPHA, lora_path, ctx)

    if mode == String("caption3"):
        _sample_prompt[202, N_IMG + 202](
            String("boxjana_caption3_beach_photo"),
            String(CAPS_DIR) + String("/boxjana_caption3_beach_photo_pos.safetensors"),
            String(CAPS_DIR) + String("/boxjana_caption3_beach_photo_neg.safetensors"),
            String(OUT_DIR) + String("/ernie_lora1024_step2500_boxjana_caption3_beach_photo.png"),
            UInt64(44),
            steps,
            base,
            st,
            lora,
            ctx,
        )
        return

    _sample_prompt[19, N_IMG + 19](
        String("boxjana_studio_cocktail"),
        String(CAPS_DIR) + String("/boxjana_studio_cocktail_pos.safetensors"),
        String(CAPS_DIR) + String("/boxjana_studio_cocktail_neg.safetensors"),
        String(OUT_DIR) + String("/ernie_lora1024_step2500_boxjana_studio_cocktail.png"),
        UInt64(42),
        steps,
        base,
        st,
        lora,
        ctx,
    )
    _sample_prompt[18, N_IMG + 18](
        String("boxjana_beach_sunset"),
        String(CAPS_DIR) + String("/boxjana_beach_sunset_pos.safetensors"),
        String(CAPS_DIR) + String("/boxjana_beach_sunset_neg.safetensors"),
        String(OUT_DIR) + String("/ernie_lora1024_step2500_boxjana_beach_sunset.png"),
        UInt64(43),
        steps,
        base,
        st,
        lora,
        ctx,
    )
