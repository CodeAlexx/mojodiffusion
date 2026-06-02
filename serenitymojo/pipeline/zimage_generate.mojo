# zimage_generate.mojo — reusable Z-Image text→image entry (RUNTIME prompt).
#
# Extracted from zimage_pipeline.mojo's hardcoded main(). Same stages:
#   tokenizer → Qwen3-4B encoder (layer-34 penultimate) → CFG denoise loop
#   (dual NextDiT forward + rectified-flow Euler) → Z-Image VAE decode.
#
# THE ONLY BEHAVIORAL CHANGE vs the verified pipeline is the caption length
# handling: the base pipeline COMPTIME-specializes the NextDiT on the exact
# token count (CAPLEN/CAPLEN_NEG). That cannot accept a runtime-typed prompt.
# Here we adopt FIXED PADDING (mirrors zimage_l2p_pipeline_512_multistep.mojo's
# CAP_PADDED pattern and qwenimage_pipeline_1024_multistep.mojo's N_TXT_KEPT +
# runtime real_len): the NextDiT is specialized ONCE at a comptime CAPLEN_MAX
# (multiple of 32). We encode at ENC_SEQ=512 (Qwen3 is causal → trailing PAD
# rows are inert per-position), slice to CAPLEN_MAX, and tell the DiT the TRUE
# token count (real_caplen) so its learned cap_pad_token overwrites the trailing
# rows — exactly what the model already does for its own mult-of-32 slack.
#
# EVERY OTHER NUMERIC CONVENTION IS BYTE-FOR-BYTE PRESERVED (see header of
# zimage_pipeline.mojo + docs/ZIMAGE_DENOISE_SIGN_CONVENTION.md):
#   • timestep = (1 - sigma) (DiT ×1000 internally)
#   • CFG code-form pred_raw = vc + cfg*(vc - vu), then negate before scheduler:
#       noise_pred = -(vc + cfg*(vc - vu))
#   • Euler: x += (sigma_next - sigma) * noise_pred  (latent kept F32)
#   • sigmas: diffusers FlowMatchEuler, shift=6.0 (hardcoded full-f32 table)
#   • encoder: layer-34, cap fed rank-2 [caplen, 2560]
#   • VAE: scale=0.3611, shift=0.1159 (baked in decoder); PNG SIGNED [-1,1]
#
# ⚠️  GPU RE-VERIFICATION REQUIRED for the padded-caption change:
#   The fixed-padding shifts the cap_padded count (and thus the image RoPE base
#   position cap_padded+1) and attends over more cap_pad_token rows than the
#   comptime-exact run. The L2P sibling accepts this; for base Z-Image it must
#   be re-checked against the diffusers oracle once the GPU is free.
#
# GPU is busy → compile-only gate:
#   cd /home/alex/mojodiffusion && pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/zimage_generate.mojo -o /tmp/zimage_generate_check
# Do NOT run a generation here.
#
# Runtime:
#   /tmp/zimage_generate_check [lora_path|base] [out_png]

from std.sys import argv
from std.gpu.host import DeviceContext
from std.math import sqrt, log, cos

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.dit.zimage_dit import NextDiT, NextDiTConfig
from serenitymojo.models.vae.zimage_decoder import ZImageDecoder
from serenitymojo.ops.tensor_algebra import reshape, mul_scalar, add, sub, slice
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.lora import LoraSet


# ── shared verified checkpoint snapshot (DiT + VAE parity used this exact one) ──
comptime ZROOT = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"
)
comptime TRANSFORMER = ZROOT + "/transformer"
comptime TEXT_ENCODER = ZROOT + "/text_encoder"
comptime VAE_DIR = ZROOT + "/vae"
comptime TOK_JSON = ZROOT + "/tokenizer/tokenizer.json"

# ── fixed-caption + encoder constants ──
comptime HIDDEN = 2560
comptime ENC_SEQ = 512        # encoder runs at this supported sdpa seq, sliced to CAPLEN_MAX
comptime CAPLEN_MAX = 256     # comptime cap buffer (multiple of 32 → cap_padded == CAPLEN_MAX)
comptime PAD_ID = 151643      # Qwen pad token (prepare_l2p PAD_TOKEN_ID); right-pad, causal-masked
comptime EXTRACT_LAYER = 34   # Qwen3-4B penultimate (Z-Image canonical)
comptime SHIFT = Float32(6.0) # diffusers scheduler_config.json shift=6.0
comptime PI = 6.283185307179586  # 2*pi

# ── default demo config (preserves the original standalone main() image) ──
comptime DEFAULT_HL = 128     # latent height = image_h / 8 (1024²)
comptime DEFAULT_WL = 128
comptime DEFAULT_STEPS = 30
comptime DEFAULT_CFG = Float32(4.0)
comptime DEFAULT_SEED = UInt64(42)
comptime OUT = "/home/alex/mojodiffusion/output/zimage_generate_1024.png"
comptime DEFAULT_PROMPT = (
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


# ── progress event (drained by the caller; mirrors the m8 mock tick model) ──
comptime ZEVENT_STARTED = 0
comptime ZEVENT_STEP = 1
comptime ZEVENT_DONE = 2
comptime ZEVENT_FAILED = 3


@fieldwise_init
struct ZImageEvent(Copyable, Movable):
    """Progress event emitted by zimage_generate. The caller drains `events`
    after each call (or, when driven step-by-step, between steps)."""

    var kind: Int      # ZEVENT_STARTED | ZEVENT_STEP | ZEVENT_DONE | ZEVENT_FAILED
    var step: Int      # current step (1-based) for ZEVENT_STEP, else 0
    var total: Int     # total steps
    var message: String


# ── fixed-padded caption pair (real_caplen tracked) ──
@fieldwise_init
struct CapFeatsFixed(Movable):
    var cond: Tensor      # [CAPLEN_MAX, HIDDEN]
    var uncond: Tensor    # [CAPLEN_MAX, HIDDEN]
    var real_cond: Int    # true cond token count (≤ CAPLEN_MAX)
    var real_uncond: Int  # true uncond token count (≤ CAPLEN_MAX)


# ── one encoded caption (fixed buffer + true token count) ──
@fieldwise_init
struct EncodedCap(Movable):
    var feats: Tensor   # [CAPLEN_MAX, HIDDEN]
    var real_len: Int   # true token count (≤ CAPLEN_MAX)

    # Consume self + uncond into the final pair in ONE expression so neither
    # struct is read after a partial `^`-move (mirrors qwenimage EncodedCaption
    # .into_caps; `deinit` is valid here because the args are Self type).
    def into_pair(deinit self, deinit uncond: EncodedCap) -> CapFeatsFixed:
        return CapFeatsFixed(self.feats^, uncond.feats^, self.real_len, uncond.real_len)


# splitmix64 → uniform [0,1). Self-contained, deterministic. (verbatim)
def _u01(mut state: UInt64) -> Float64:
    state = state + 0x9E3779B97F4A7C15
    var z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0)


# seeded Gaussian noise on host (Box-Muller); seed is now a runtime param. (verbatim math)
def gaussian_noise(n: Int, seed: UInt64) raises -> List[Float32]:
    var state = seed
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


# Host round-trip dtype cast (to_host→F32, from_host→target). (verbatim)
def _cast(t: Tensor, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var hh = t.to_host(ctx)
    return Tensor.from_host(hh, t.shape(), dt, ctx)


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


# ── one prompt → templated tokens → layer-34 cap_feats [CAPLEN_MAX, HIDDEN] ──
# Returns the fixed-size [CAPLEN_MAX, HIDDEN] feature buffer plus the TRUE token
# count. Trailing rows [real_caplen, CAPLEN_MAX) hold encoder outputs of PAD
# tokens; they are inert in the causal encoder for the first real_caplen rows,
# and the NextDiT overwrites them with the learned cap_pad_token (real_len=
# real_caplen), so they do not influence the velocity for the real tokens.
def _encode_text_fixed(
    tok: Qwen3Tokenizer, enc: Qwen3Encoder, prompt: String, ctx: DeviceContext
) raises -> EncodedCap:
    var templated = (
        String("<|im_start|>user\n") + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    )
    var ids_full = tok.encode(templated)
    var real_caplen = len(ids_full)
    if real_caplen > CAPLEN_MAX:
        # Truncate + log instead of crashing (spec requirement).
        print(
            "  [warn] prompt tokenized to", real_caplen, "tokens > CAPLEN_MAX=",
            CAPLEN_MAX, "→ truncating. Raise CAPLEN_MAX or shorten the prompt.",
        )
        real_caplen = CAPLEN_MAX
    # Pad to ENC_SEQ with PAD_ID; encode at the supported sdpa seq, slice the
    # leading CAPLEN_MAX rows (causal → first real_caplen are exact). (verbatim
    # encode-at-512-and-slice approach; only the slice length differs: CAPLEN_MAX
    # vs the old exact caplen.)
    var ids = List[Int](capacity=ENC_SEQ)
    for i in range(real_caplen):
        ids.append(ids_full[i])
    for _ in range(ENC_SEQ - real_caplen):
        ids.append(PAD_ID)
    var cf = enc.encode(ids, EXTRACT_LAYER, ctx)        # [1, ENC_SEQ, HIDDEN]
    var cf_fixed = slice(cf, 1, 0, CAPLEN_MAX, ctx)     # [1, CAPLEN_MAX, HIDDEN]
    var rank2 = reshape(cf_fixed, [CAPLEN_MAX, HIDDEN], ctx)  # rank-2 for DiT
    return EncodedCap(rank2^, real_caplen)


# ── encode cond + uncond in ONE encoder session; encoder freed on return ──
def encode_captions_fixed(
    prompt: String, negative: String, ctx: DeviceContext
) raises -> CapFeatsFixed:
    var tok = Qwen3Tokenizer(TOK_JSON)
    var enc = Qwen3Encoder.load(TEXT_ENCODER, Qwen3Config.zimage(), ctx)
    print("[text] encoding cond + uncond at fixed CAPLEN_MAX=", CAPLEN_MAX, "layer", EXTRACT_LAYER)
    var cond = _encode_text_fixed(tok, enc, prompt, ctx)
    var uncond = _encode_text_fixed(tok, enc, negative, ctx)
    print("  real cond tokens=", cond.real_len, " real uncond tokens=", uncond.real_len)
    return cond^.into_pair(uncond^)  # enc + tok freed


# ── diffusers FlowMatchEuler sigmas (shift=6.0). Returns a `steps+1`-length list.
# For steps != 30 we build the analytic schedule (linspace 1→0 over `steps`,
# then shift = SHIFT*s / (1 + (SHIFT-1)*s)) and append a terminal 0. For the
# default 30-step base run we return the exact diffusers oracle table VERBATIM
# (the same hardcoded constants the verified pipeline uses) to preserve numerics.
def _build_sigmas(steps: Int) raises -> List[Float32]:
    if steps == 30:
        var s30: List[Float32] = [
            1.000000000, 0.994082808, 0.987804890, 0.981132150,
            0.974026024, 0.966442943, 0.958333373, 0.949640274,
            0.940298557, 0.930232465, 0.919354916, 0.907563090,
            0.894736826, 0.880733907, 0.865384579, 0.848484814,
            0.829787314, 0.808988750, 0.785714269, 0.759493589,
            0.729729772, 0.695652127, 0.656250000, 0.610169470,
            0.555555522, 0.489795893, 0.409090906, 0.307692289,
            0.176470578, 0.000000000, 0.000000000,
        ]
        return s30^
    # Analytic FlowMatchEuler with shift, for non-default step counts. diffusers
    # spaces the pre-shift grid by 1/(N-1) over N points then appends 0.
    var out = List[Float32](capacity=steps + 1)
    var n = steps
    for i in range(n):
        var lin = Float32(1.0) - Float32(i) / Float32(n - 1) if n > 1 else Float32(1.0)
        var sh = (SHIFT * lin) / (Float32(1.0) + (SHIFT - Float32(1.0)) * lin)
        out.append(sh)
    out.append(Float32(0.0))
    return out^


# ── CFG denoise: dual DiT forward + diffusers CFG + scheduler sign flip ──
# CAPLEN comptime param is CAPLEN_MAX for BOTH cond and uncond (fixed padding);
# the runtime real_cond/real_uncond drive the cap_pad_token substitution.
def _denoise[HL: Int, WL: Int](
    caps: CapFeatsFixed, steps: Int, cfg: Float32, seed: UInt64,
    lora_path: String, lora_multiplier: Float32,
    mut events: List[ZImageEvent], ctx: DeviceContext,
) raises -> Tensor:
    print("[denoise] loading NextDiT", HL, "x", WL, "(fixed CAPLEN_MAX", CAPLEN_MAX, ")")
    var dit_c = NextDiT[HL, WL, CAPLEN_MAX].load(TRANSFORMER, ctx)
    if lora_path.byte_length() > 0:
        print("[lora] loading", lora_path)
        var lora = LoraSet.load(lora_path)
        var merged = lora.merge_into_indexed(
            dit_c.weights, dit_c.name_to_idx, lora_multiplier, ctx
        )
        print("[lora] merged", merged, "modules into NextDiT")
    # uncond instance shares the SAME GPU weights (ArcPointer copy = refcount++,
    # no VRAM duplication). With fixed padding BOTH share comptime CAPLEN_MAX, so
    # we can reuse dit_c directly for the uncond forward — no second instance and
    # no second weight refcount needed.

    var noise = gaussian_noise(16 * HL * WL, seed)
    var nshape = [1, 16, HL, WL]
    # latent kept in F32 (diffusers keeps latents fp32; bf16 only for model input).
    var x = Tensor.from_host(noise, nshape^, STDtype.F32, ctx)

    var sigmas = _build_sigmas(steps)
    print("[denoise]", steps, "steps, shift", SHIFT, "CFG", cfg, "seed", seed)
    _stats("init_noise", x, ctx)
    events.append(ZImageEvent(ZEVENT_STARTED, 0, steps, String("started")))
    for i in range(steps):
        var t = 1.0 - sigmas[i]  # DiT timestep convention (= 1 - sigma)
        var x_bf = _cast(x, STDtype.BF16, ctx)  # bf16 only to feed the DiT
        # cond + uncond velocities at fixed CAPLEN_MAX, runtime real lengths.
        var vc = _cast(
            dit_c.forward_runtime_cap(x_bf, t, caps.cond, caps.real_cond, ctx),
            STDtype.F32, ctx,
        )
        var vu = _cast(
            dit_c.forward_runtime_cap(x_bf, t, caps.uncond, caps.real_uncond, ctx),
            STDtype.F32, ctx,
        )
        # Diffusers CFG (code form, F32): pred_raw = vc + cfg*(vc - vu);
        # pipeline_z_image.py negates before FlowMatchEulerDiscreteScheduler.step.
        var pred = add(vc, mul_scalar(sub(vc, vu, ctx), cfg, ctx), ctx)
        pred = mul_scalar(pred, -1.0, ctx)
        if i == 0:
            _stats("v_cond", vc, ctx)
            _stats("v_uncond", vu, ctx)
        var dt = sigmas[i + 1] - sigmas[i]
        x = add(x, mul_scalar(pred, dt, ctx), ctx)  # F32: x += (σ_next-σ)*(-pred_raw)
        events.append(ZImageEvent(ZEVENT_STEP, i + 1, steps, String("denoise")))
        if (i + 1) % 10 == 0 or i == steps - 1:
            print("  step", i + 1, "/", steps, "sigma", sigmas[i], "→", sigmas[i + 1])
    _stats("final_latent", x, ctx)
    return x^  # dit_c destroyed → weights freed before VAE load


# ── reusable generation entry (RUNTIME prompt) ──────────────────────────────
# steps/cfg/seed honored at runtime; width/height drive the comptime latent
# grid via the wrapper below. width/height must be one of the supported comptime
# specializations (default 1024² → HL=WL=128). Returns decoded RGB [1,3,8HL,8WL].
def zimage_generate(
    prompt: String, negative: String,
    steps: Int, cfg: Float32, seed: UInt64,
    width: Int, height: Int,
    lora_path: String, lora_multiplier: Float32,
    mut events: List[ZImageEvent], ctx: DeviceContext,
) raises -> Tensor:
    var caps = encode_captions_fixed(prompt, negative, ctx)
    # Dispatch the comptime latent grid from runtime width/height. Only the
    # verified 1024² grid is wired today; add cases as other sizes are verified.
    var hl = height // 8
    var wl = width // 8
    if hl == DEFAULT_HL and wl == DEFAULT_WL:
        var latent = _denoise[DEFAULT_HL, DEFAULT_WL](
            caps, steps, cfg, seed, lora_path, lora_multiplier, events, ctx
        )
        print("[vae] decoding latent → RGB")
        var dec = ZImageDecoder[DEFAULT_HL, DEFAULT_WL].load(VAE_DIR, ctx)
        var rgb = dec.decode(_cast(latent, STDtype.BF16, ctx), ctx)
        events.append(ZImageEvent(ZEVENT_DONE, steps, steps, String("done")))
        return rgb^
    events.append(
        ZImageEvent(
            ZEVENT_FAILED, 0, steps,
            String("unsupported size ") + String(width) + "x" + String(height)
            + " (only 1024x1024 is wired; latent grid is comptime)",
        )
    )
    raise Error(
        String("zimage_generate: unsupported width/height ")
        + String(width) + "x" + String(height)
        + " — only 1024x1024 (HL=WL=128) is comptime-specialized today."
    )


# ── standalone demo: drives zimage_generate with the original constants so the
# existing single-image path still works (compile + behavior preserved). ──
def main() raises:
    var ctx = DeviceContext()
    print("=== Z-Image generate (runtime prompt, fixed-padded caption) — 1024x1024 ===")
    var a = argv()
    var lora_path = String("")
    var out_path = String(OUT)
    if len(a) >= 2:
        var arg_lora = String(a[1])
        if arg_lora != String("base") and arg_lora != String("none") and arg_lora != String(""):
            lora_path = arg_lora
    if len(a) >= 3:
        out_path = String(a[2])
    var events = List[ZImageEvent]()
    var rgb = zimage_generate(
        DEFAULT_PROMPT, String(""),
        DEFAULT_STEPS, DEFAULT_CFG, DEFAULT_SEED,
        1024, 1024, lora_path, Float32(1.0), events, ctx,
    )
    var rs = rgb.shape()
    print("[vae] image:", rs[0], rs[1], rs[2], rs[3], " events:", len(events))
    save_png(rgb, out_path, ctx, ValueRange.SIGNED)
    print("[done] saved:", out_path)
