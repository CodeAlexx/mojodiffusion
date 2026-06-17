# zimage_pipeline.mojo — Z-Image text→image capstone (pure Mojo + MAX, GPU).
#
#   tokenizer → Qwen3-4B encoder (layer-34 penultimate) → CFG denoise loop
#   (dual NextDiT forward + rectified-flow Euler) → Z-Image VAE decode → PNG.
#
# Conventions taken VERBATIM from the verified references:
#   • text  : prepare_l2p.rs — template + extract layer 34, NO final norm, feed
#             exact valid_len tokens (DiT pads to mult-32 via learned cap_pad_token).
#   • denoise: diffusers ZImagePipeline.__call__ (AUTHORITATIVE for this ckpt;
#             model_index="ZImagePipeline") — timestep = (1000-t)/1000 = (1-sigma)
#             (DiT *1000 internally); CFG CODE FORM pred = v_cond + cfg*(v_cond -
#             v_uncond); then diffusers does `noise_pred = -noise_pred` before
#             scheduler.step, so x += (sigma_next - sigma) * (-pred).
#   • base model (Tongyi-MAI/Z-Image README): undistilled → needs CFG 3–5 + 28–50
#             steps (turbo's 8/cfg=0 diverges/produces noise).
#   • VAE   : scale=0.3611, shift=0.1159 (baked in decoder); PNG SIGNED [-1,1].
#
# CAPLEN / CAPLEN_NEG are COMPILE-TIME (probe with count_tokens.mojo per prompt).
# CFG uses TWO NextDiT instantiations (cond @CAPLEN, uncond @CAPLEN_NEG) sharing
# ONE set of GPU weights via ArcPointer.copy() — no VRAM duplication.
#
# Run: cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#        serenitymojo/pipeline/zimage_pipeline.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt, log, cos

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.dit.zimage_dit import NextDiT, NextDiTConfig
from serenitymojo.models.vae.zimage_decoder import ZImageDecoder
from serenitymojo.sampling.flow_match import Scheduler
from serenitymojo.ops.tensor_algebra import reshape, mul_scalar, add, sub, slice
from serenitymojo.image.png import save_png, ValueRange


# Host round-trip dtype cast (to_host→F32, from_host→target).
def _cast(t: Tensor, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var hh = t.to_host(ctx)
    return Tensor.from_host(hh, t.shape(), dt, ctx)

# ── shared verified checkpoint (DiT + VAE parity used this exact snapshot) ──
comptime ZROOT = "/home/alex/.serenity/models/zimage_base"
comptime TRANSFORMER = ZROOT + "/transformer"
comptime TEXT_ENCODER = ZROOT + "/text_encoder"
comptime VAE_DIR = ZROOT + "/vae"
comptime TOK_JSON = ZROOT + "/tokenizer/tokenizer.json"
comptime OUT = "/home/alex/mojodiffusion/output/zimage_hollywood_punk_1024.png"

# ── run config (HL=WL=128 → 1024²) ──
comptime PROMPT = (
    "a glamorous platinum blonde woman with vintage old Hollywood finger-wave"
    " hair screaming with mouth wide open, head tilted back, face bathed in a"
    " harsh dual-tone color gel treatment of deep cobalt blue shadow and acid"
    " green neon highlight, bold neon yellow-green spray paint XX marks slashed"
    " aggressively across her face with dripping paint trails running down over"
    " her nose and cheeks, her ringed hand pressed against her chin adorned with"
    " oversized diamond and jeweled rings, a thick chunky gold chain necklace"
    " draped at her chest, the entire image set against a saturated blood red"
    " background, heavy analog film grain and scratched surface texture over the"
    " entire composition, the photograph treated with a high-contrast duotone"
    " darkroom effect that renders skin in cold blue tones against the violent red"
    " field, the aesthetic of a subversive underground concert poster meets"
    " transgressive fashion editorial, punk energy, neo-noir, Andy Warhol meets"
    " Gaspar Noe, hyper-saturated analog photography, gritty film scan texture,"
    " cinematic portrait, ultra detailed, 8k"
)
comptime HL = 128            # latent height = image_h / 8 (1024²)
comptime WL = 128            # latent width  = image_w / 8
comptime CAPLEN = 210        # positive prompt templated token count (probe!)
comptime CAPLEN_NEG = 8      # empty/negative prompt templated token count
comptime HIDDEN = 2560
comptime ENC_SEQ = 512       # encoder runs at this (supported sdpa seq); output sliced to real len
comptime PAD_ID = 151643     # Qwen pad token (prepare_l2p PAD_TOKEN_ID); right-pad, causal-masked
comptime EXTRACT_LAYER = 34  # Qwen3-4B penultimate (Z-Image canonical)
comptime STEPS = 30          # base model: README 28–50; t1b reference uses 30
comptime SHIFT = Float32(6.0)  # diffusers uses scheduler_config.json shift=6.0
comptime SEED = 42
comptime CFG = Float32(4.0)  # README recommended
comptime PI = 6.283185307179586  # 2*pi


# ── caption pair (cond + uncond) so the encoder is freed before DiT load ──
@fieldwise_init
struct CapFeats(Movable):
    var cond: Tensor    # [CAPLEN, HIDDEN]
    var uncond: Tensor  # [CAPLEN_NEG, HIDDEN]


# splitmix64 → uniform [0,1). Self-contained, deterministic.
def _u01(mut state: UInt64) -> Float64:
    state = state + 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


# ── seeded Gaussian noise on host (Box-Muller); uploaded as BF16 ──
def gaussian_noise(n: Int) raises -> List[Float32]:
    var state = UInt64(SEED)
    var out = List[Float32]()
    var i = 0
    while i < n:
        var u1 = _u01(state)
        var u2 = _u01(state)
        if u1 < 1e-12:
            u1 = 1e-12
        var r = sqrt(-2.0 * log(u1))
        out.append(Float32(r * cos(PI * u2)))
        if i + 1 < n:
            out.append(Float32(r * cos(PI * u2 + 1.5707963267948966)))  # +pi/2 → sin
        i += 2
    return out^


# ── DEBUG: tensor stats (mean/std/absmax) ──
def _stats(name: String, t: Tensor, ctx: DeviceContext) raises:
    var h = t.to_host(ctx)
    var n = len(h)
    if n == 0:
        return
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


# ── one prompt → templated tokens → layer-34 cap_feats [caplen, HIDDEN] ──
def _encode_text(
    tok: Qwen3Tokenizer, enc: Qwen3Encoder, prompt: String, caplen: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    var templated = (
        String("<|im_start|>user\n") + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    )
    var ids_full = tok.encode(templated)
    if len(ids_full) != caplen:
        raise Error(
            String("caplen mismatch: tokens=") + String(len(ids_full)) + " caplen="
            + String(caplen) + " — re-probe count_tokens.mojo."
        )
    # Pad to ENC_SEQ (a supported sdpa seq) with PAD_ID; the SDPA dispatch is
    # comptime-specialized on seq, so we always encode at 512 and slice to the
    # real length. Qwen3 is causal → trailing pad tokens don't affect the first
    # `caplen` outputs (matches prepare_l2p: pad to TXT_PAD_LEN, narrow to valid).
    var ids = List[Int](capacity=ENC_SEQ)
    for i in range(caplen):
        ids.append(ids_full[i])
    for _ in range(ENC_SEQ - caplen):
        ids.append(PAD_ID)
    var cf = enc.encode(ids, EXTRACT_LAYER, ctx)  # [1, ENC_SEQ, HIDDEN]
    var cf_real = slice(cf, 1, 0, caplen, ctx)    # [1, caplen, HIDDEN] (drop pad rows)
    # DiT wants rank-2 [caplen, HIDDEN] (parity in_cap.shape="32,2560"); squeeze.
    return reshape(cf_real, [caplen, HIDDEN], ctx)


# ── encode cond + uncond in ONE encoder session; encoder freed on return ──
def encode_captions(ctx: DeviceContext) raises -> CapFeats:
    var tok = Qwen3Tokenizer(TOK_JSON)
    var enc = Qwen3Encoder.load(TEXT_ENCODER, Qwen3Config.zimage(), ctx)
    print("[text] encoding cond(", CAPLEN, ") + uncond(", CAPLEN_NEG, ") layer", EXTRACT_LAYER)
    var cond = _encode_text(tok, enc, PROMPT, CAPLEN, ctx)
    var uncond = _encode_text(tok, enc, String(""), CAPLEN_NEG, ctx)
    return CapFeats(cond^, uncond^)  # enc + tok destroyed → ~8GB freed


# ── CFG denoise: dual DiT forward + diffusers CFG + scheduler sign flip ──
def denoise(cap_c: Tensor, cap_u: Tensor, ctx: DeviceContext) raises -> Tensor:
    print("[denoise] loading NextDiT", HL, "x", WL, "(cond CAPLEN", CAPLEN, "+ uncond", CAPLEN_NEG, ")")
    var dit_c = NextDiT[HL, WL, CAPLEN].load(TRANSFORMER, ctx)
    # uncond instance shares the SAME GPU weights (ArcPointer copy = refcount++,
    # no VRAM duplication); only the comptime CAPLEN_NEG differs.
    var dit_u = NextDiT[HL, WL, CAPLEN_NEG](
        dit_c.weights.copy(), dit_c.name_to_idx.copy(), dit_c.config
    )

    var noise = gaussian_noise(16 * HL * WL)
    var nshape = [1, 16, HL, WL]
    # Keep the latent boundary BF16; scheduler arithmetic below casts locally.
    var x = Tensor.from_host(noise, nshape^, STDtype.BF16, ctx)

    # diffusers FlowMatchEulerDiscreteScheduler sigmas (steps=30, shift=6.0) —
    # exact values from the oracle. (build_sigma_schedule spaces the pre-shift
    # grid by 1/N over N+1 points; diffusers spaces by 1/(N-1) over N points then
    # appends 0 → my sigmas were offset, drifting the trajectory. TODO: fix
    # build_sigma_schedule to match; hardcoded here to unblock the first image.)
    var sigmas: List[Float32] = [
        1.000000000, 0.994082808, 0.987804890, 0.981132150,
        0.974026024, 0.966442943, 0.958333373, 0.949640274,
        0.940298557, 0.930232465, 0.919354916, 0.907563090,
        0.894736826, 0.880733907, 0.865384579, 0.848484814,
        0.829787314, 0.808988750, 0.785714269, 0.759493589,
        0.729729772, 0.695652127, 0.656250000, 0.610169470,
        0.555555522, 0.489795893, 0.409090906, 0.307692289,
        0.176470578, 0.000000000, 0.000000000,
    ]
    print("[denoise]", STEPS, "steps, shift", SHIFT, "CFG", CFG)
    _stats("init_noise", x, ctx)
    for i in range(STEPS):
        var t = 1.0 - sigmas[i]  # DiT timestep convention
        var vc = _cast(dit_c.forward(x, t, cap_c, ctx), STDtype.F32, ctx)
        var vu = _cast(dit_u.forward(x, t, cap_u, ctx), STDtype.F32, ctx)
        # Diffusers CFG (code form, F32), then pipeline_z_image.py negates before
        # FlowMatchEulerDiscreteScheduler.step().
        var pred = add(vc, mul_scalar(sub(vc, vu, ctx), CFG, ctx), ctx)
        pred = mul_scalar(pred, -1.0, ctx)
        if i == 0:
            _stats("v_cond", vc, ctx)
            _stats("v_uncond", vu, ctx)
        var dt = sigmas[i + 1] - sigmas[i]
        var x_compute = _cast(x, STDtype.F32, ctx)
        x = _cast(add(x_compute, mul_scalar(pred, dt, ctx), ctx), STDtype.BF16, ctx)
        if (i + 1) % 10 == 0 or i == STEPS - 1:
            print("  step", i + 1, "/", STEPS, "sigma", sigmas[i], "→", sigmas[i + 1])
    _stats("final_latent", x, ctx)
    return x^  # dit_c + dit_u destroyed → weights freed before VAE load


def main() raises:
    var ctx = DeviceContext()
    print("=== Z-Image pipeline (pure Mojo, CFG) — 1024x1024 ===")

    var caps = encode_captions(ctx)
    var latent = denoise(caps.cond, caps.uncond, ctx)

    print("[vae] decoding latent → RGB")
    var dec = ZImageDecoder[HL, WL].load(VAE_DIR, ctx)
    var rgb = dec.decode(_cast(latent, STDtype.BF16, ctx), ctx)  # [1,3,8*HL,8*WL]
    var rs = rgb.shape()
    print("[vae] image:", rs[0], rs[1], rs[2], rs[3])

    save_png(rgb, OUT, ctx, ValueRange.SIGNED)
    print("[done] saved:", OUT)
