# zimage_generate.mojo — reusable Z-Image text→image entry (RUNTIME prompt).
#
# Same public stages as zimage_pipeline.mojo:
#   tokenizer → Qwen3-4B encoder (layer-34 penultimate) → CFG denoise loop
#   (dual Mojo Z-Image stack forward + rectified-flow Euler) → Z-Image VAE decode.
#
# LoRA behavior is AI Toolkit-style FORWARD OVERLAY, not a weight merge:
#   base projection forward + lora_up(lora_down(x)) * multiplier * alpha/rank.
# Do not swap this back to `LoraSet.merge_into_indexed`; the production trainer
# saves main-layer PEFT/PERT adapters and sampling must exercise that same path.
#
# Runtime prompts use fixed CAPLEN_MAX padding. We encode at ENC_SEQ=512 (Qwen3
# is causal), slice to CAPLEN_MAX, and replace rows [real_caplen, CAPLEN_MAX)
# with the learned cap_pad_token. RoPE pad rows follow the trainer convention:
# cap pad positions are (0,0,0).
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
# Verified on 2026-06-02: three 1024 caption-based samples from
# output/alina_zimage/zimage_lora_step2000.safetensors completed with the LoRA
# overlay loaded (210 main adapters, alpha/rank=0.0625).
#
# Build:
#   cd /home/alex/mojodiffusion && pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/zimage_generate.mojo -o /tmp/zimage_generate_check
#
# Runtime:
#   /tmp/zimage_generate_check [lora_path|base] [out_png] [seed] [prompt]
#   /tmp/zimage_generate_check [lora_path|base] [out_png] [sample_prompts.json] [prompt_id]

from std.sys import argv
from std.gpu.host import DeviceContext
from std.math import sqrt, log, cos
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer
from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Encoder, Qwen3Config
from serenitymojo.models.vae.zimage_decoder import ZImageDecoder
from serenitymojo.models.zimage.weights import (
    ZImageBlockWeights, load_zimage_block_weights_prefixed_mixed,
)
from serenitymojo.models.zimage.block import ZImageModVecs
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraDeviceSet, build_zimage_lora_set, load_zimage_lora_main_only_resume,
    zimage_lora_set_to_device, zimage_stack_lora_predict_main_device,
)
from serenitymojo.models.zimage.real_weights import (
    ZImageRealAux, load_zimage_real_aux, build_adaln, build_block_modvecs,
    build_f_scale, build_cap_seq, build_x_seq, build_rope, build_positions,
)
from serenitymojo.ops.tensor_algebra import reshape, permute, mul_scalar, add, sub, slice
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)
from serenitymojo.training.progress_display import print_sample_step, print_sample_saved


# ── shared verified checkpoint snapshot (DiT + VAE parity used this exact one) ──
comptime ZROOT = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"
)
comptime TRANSFORMER = "/home/alex/.serenity/models/zimage_base/transformer"
comptime TEXT_ENCODER = ZROOT + "/text_encoder"
comptime VAE_DIR = ZROOT + "/vae"
comptime TOK_JSON = ZROOT + "/tokenizer/tokenizer.json"

# ── Z-Image transformer constants shared with train_zimage_real.mojo ─────────
comptime H = 30
comptime Dh = 128
comptime D = H * Dh
comptime F = 10240
comptime CAP_DIM = 2560
comptime ADALN_DIM = 256
comptime T_SCALE = Float32(1000.0)
comptime ROPE_THETA = Float32(256.0)
comptime AXIS0 = 32
comptime AXIS1 = 48
comptime AXIS2 = 48
comptime EPS = Float32(1e-5)
comptime FINAL_EPS = Float32(1e-6)
comptime LAT_C = 16
comptime OUT_CH = 64
comptime PATCH = 2
comptime NUM_NR = 2
comptime NUM_CR = 2
comptime MAIN_DEPTH = 30
comptime RANK = 16
comptime ALPHA = Float32(1.0)

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
    "alverone, a high-resolution photograph featuring a young caucasian woman"
    " with long, straight, platinum blonde hair and fair skin, standing in a"
    " cobblestone courtyard in a pink sleeveless dress with a fantasy castle in"
    " the background, overcast sky, casual relaxed atmosphere"
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


def _unified_positions(x_pos: List[List[Int]], cap_pos: List[List[Int]]) -> List[List[Int]]:
    var uni_pos = List[List[Int]]()
    for i in range(len(x_pos)):
        uni_pos.append(x_pos[i].copy())
    for i in range(len(cap_pos)):
        uni_pos.append(cap_pos[i].copy())
    return uni_pos^


def _cap_seq_with_pad(
    aux: ZImageRealAux, cap_feats: Tensor, real_caplen: Int,
    cap_pad_h: List[Float32], ctx: DeviceContext,
) raises -> List[Float32]:
    var cap_f32 = _cast(cap_feats, STDtype.F32, ctx)
    var cap_seq = build_cap_seq(aux, cap_f32, EPS, ctx)
    var real_len = real_caplen
    if real_len < 0:
        real_len = 0
    if real_len > CAPLEN_MAX:
        real_len = CAPLEN_MAX
    for r in range(real_len, CAPLEN_MAX):
        for c in range(D):
            cap_seq[r * D + c] = cap_pad_h[c]
    return cap_seq^


def _unpatchify_patch_list[HL: Int, WL: Int](
    patches: List[Float32], ctx: DeviceContext
) raises -> Tensor:
    comptime HT = HL // PATCH
    comptime WT = WL // PATCH
    comptime N_IMG_REAL = HT * WT
    comptime REAL_PATCH_VALUES = N_IMG_REAL * OUT_CH
    if len(patches) < REAL_PATCH_VALUES:
        raise Error("zimage_generate: predicted patch list is shorter than the real image grid")
    var real_vals = List[Float32]()
    for i in range(REAL_PATCH_VALUES):
        real_vals.append(patches[i])
    var seq = Tensor.from_host(real_vals^, [N_IMG_REAL, OUT_CH], STDtype.F32, ctx)
    var v = reshape(seq, [HT, WT, PATCH, PATCH, LAT_C], ctx)
    var perm = List[Int]()
    perm.append(4); perm.append(0); perm.append(2); perm.append(1); perm.append(3)
    var p = permute(v, perm^, ctx)
    return reshape(p, [1, LAT_C, HL, WL], ctx)


def _latent_velocity_overlay[
    HL: Int, WL: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    latent: Tensor, cap_seq: List[Float32],
    nr_blocks: List[ZImageBlockWeights], nr_mod: List[ZImageModVecs],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights], main_mod: List[ZImageModVecs],
    lora: ZImageLoraDeviceSet,
    f_scale: List[Float32],
    aux: ZImageRealAux,
    final_lin_w: Tensor, final_lin_b: Tensor,
    x_pad_h: List[Float32],
    x_cos: Tensor, x_sin: Tensor,
    cap_cos: Tensor, cap_sin: Tensor,
    uni_cos: Tensor, uni_sin: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    comptime HT = HL // PATCH
    comptime WT = WL // PATCH
    comptime N_IMG_REAL = HT * WT
    comptime IMG_PAD = N_IMG - N_IMG_REAL

    var x_seq = build_x_seq(aux, latent, LAT_C, HL, WL, PATCH, ctx)
    for _pad in range(IMG_PAD):
        for c in range(D):
            x_seq.append(x_pad_h[c])

    var patches = zimage_stack_lora_predict_main_device[H, Dh, N_IMG, N_TXT, S](
        x_seq.copy(), cap_seq.copy(),
        nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora,
        f_scale.copy(), final_lin_w, final_lin_b,
        x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )
    return _unpatchify_patch_list[HL, WL](patches, ctx)


def _cfg_pred_overlay[
    HL: Int, WL: Int, N_IMG: Int, N_TXT: Int, S: Int
](
    latent: Tensor, t: Float32, cfg: Float32,
    cap_seq_cond: List[Float32], cap_seq_uncond: List[Float32],
    nr_blocks: List[ZImageBlockWeights],
    cr_blocks: List[ZImageBlockWeights],
    main_blocks: List[ZImageBlockWeights],
    lora: ZImageLoraDeviceSet,
    aux: ZImageRealAux,
    final_lin_w: Tensor, final_lin_b: Tensor,
    x_pad_h: List[Float32],
    x_cos_cond: Tensor, x_sin_cond: Tensor,
    cap_cos_cond: Tensor, cap_sin_cond: Tensor,
    uni_cos_cond: Tensor, uni_sin_cond: Tensor,
    x_cos_uncond: Tensor, x_sin_uncond: Tensor,
    cap_cos_uncond: Tensor, cap_sin_uncond: Tensor,
    uni_cos_uncond: Tensor, uni_sin_uncond: Tensor,
    trace: Bool,
    ctx: DeviceContext,
) raises -> Tensor:
    var adaln = build_adaln(aux, t, ADALN_DIM, T_SCALE, ctx)
    var nr_mod = List[ZImageModVecs]()
    for j in range(NUM_NR):
        nr_mod.append(build_block_modvecs(aux.nr_mod_w[j][], aux.nr_mod_b[j][], adaln, D, ctx))
    var main_mod = List[ZImageModVecs]()
    for j in range(MAIN_DEPTH):
        main_mod.append(build_block_modvecs(aux.main_mod_w[j][], aux.main_mod_b[j][], adaln, D, ctx))
    var f_scale = build_f_scale(aux, adaln, D, ctx)

    var vc = _latent_velocity_overlay[HL, WL, N_IMG, N_TXT, S](
        latent, cap_seq_cond,
        nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora,
        f_scale.copy(), aux, final_lin_w, final_lin_b, x_pad_h,
        x_cos_cond, x_sin_cond, cap_cos_cond, cap_sin_cond,
        uni_cos_cond, uni_sin_cond, ctx,
    )
    var vu = _latent_velocity_overlay[HL, WL, N_IMG, N_TXT, S](
        latent, cap_seq_uncond,
        nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora,
        f_scale.copy(), aux, final_lin_w, final_lin_b, x_pad_h,
        x_cos_uncond, x_sin_uncond, cap_cos_uncond, cap_sin_uncond,
        uni_cos_uncond, uni_sin_uncond, ctx,
    )
    if trace:
        _stats("v_cond", vc, ctx)
        _stats("v_uncond", vu, ctx)

    var pred = add(vc, mul_scalar(sub(vc, vu, ctx), cfg, ctx), ctx)
    return mul_scalar(pred, -1.0, ctx)


def _parse_nonnegative_int(s: String) raises -> Int:
    var out = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        var ch = bs[i]
        if ch < 0x30 or ch > 0x39:
            raise Error(String("expected integer, got ") + s)
        out = out * 10 + Int(ch - 0x30)
    return out


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
    comptime HT = HL // PATCH
    comptime WT = WL // PATCH
    comptime N_IMG_REAL = HT * WT
    comptime IMG_PAD = (32 - (N_IMG_REAL % 32)) % 32
    comptime N_IMG = N_IMG_REAL + IMG_PAD
    comptime N_TXT = CAPLEN_MAX
    comptime S = N_IMG + N_TXT

    print("[denoise] loading Mojo Z-Image stack", HL, "x", WL, "(CAPLEN_MAX", CAPLEN_MAX, ")")
    var st = ShardedSafeTensors.open(String(TRANSFORMER))
    var aux = load_zimage_real_aux(st, NUM_NR, MAIN_DEPTH, ctx)
    var nr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_NR):
        nr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("noise_refiner.") + String(i), ctx))
    var cr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_CR):
        cr_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("context_refiner.") + String(i), ctx))
    var main_blocks = List[ZImageBlockWeights]()
    for i in range(MAIN_DEPTH):
        main_blocks.append(load_zimage_block_weights_prefixed_mixed(st, String("layers.") + String(i), ctx))
    var final_lin_w = aux.final_lin_w[].clone(ctx)
    var final_lin_b = aux.final_lin_b[].clone(ctx)
    var x_pad_h = aux.x_pad_token[].to_host(ctx)
    var cap_pad_h = aux.cap_pad_token[].to_host(ctx)

    var lora_alpha = ALPHA * lora_multiplier
    var lora = build_zimage_lora_set(NUM_NR, NUM_CR, MAIN_DEPTH, D, F, RANK, lora_alpha)
    if lora_path.byte_length() > 0:
        print("[lora] loading", lora_path)
        lora = load_zimage_lora_main_only_resume(
            NUM_NR, NUM_CR, MAIN_DEPTH, RANK, lora_alpha, D, F, lora_path, ctx,
        )
        print("[lora] overlay loaded", MAIN_DEPTH * 7, "main-layer adapters; scale alpha/rank =", lora_alpha / Float32(RANK))
    else:
        print("[lora] base mode: zero LoRA overlay")
    var lora_dev = zimage_lora_set_to_device(lora, ctx)

    var cap_seq_cond = _cap_seq_with_pad(aux, caps.cond, caps.real_cond, cap_pad_h, ctx)
    var cap_seq_uncond = _cap_seq_with_pad(aux, caps.uncond, caps.real_uncond, cap_pad_h, ctx)

    var pos_cond = build_positions(N_IMG, HT, WT, CAPLEN_MAX, caps.real_cond)
    var x_pos_cond = pos_cond[0].copy()
    var cap_pos_cond = pos_cond[1].copy()
    var uni_pos_cond = _unified_positions(x_pos_cond, cap_pos_cond)
    var xr_cond = build_rope(x_pos_cond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos_cond = xr_cond[0].copy(); var x_sin_cond = xr_cond[1].copy()
    var cr_cond = build_rope(cap_pos_cond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cap_cos_cond = cr_cond[0].copy(); var cap_sin_cond = cr_cond[1].copy()
    var ur_cond = build_rope(uni_pos_cond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos_cond = ur_cond[0].copy(); var uni_sin_cond = ur_cond[1].copy()

    var pos_uncond = build_positions(N_IMG, HT, WT, CAPLEN_MAX, caps.real_uncond)
    var x_pos_uncond = pos_uncond[0].copy()
    var cap_pos_uncond = pos_uncond[1].copy()
    var uni_pos_uncond = _unified_positions(x_pos_uncond, cap_pos_uncond)
    var xr_uncond = build_rope(x_pos_uncond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var x_cos_uncond = xr_uncond[0].copy(); var x_sin_uncond = xr_uncond[1].copy()
    var cr_uncond = build_rope(cap_pos_uncond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var cap_cos_uncond = cr_uncond[0].copy(); var cap_sin_uncond = cr_uncond[1].copy()
    var ur_uncond = build_rope(uni_pos_uncond, H, Dh, ROPE_THETA, AXIS0, AXIS1, AXIS2, ctx)
    var uni_cos_uncond = ur_uncond[0].copy(); var uni_sin_uncond = ur_uncond[1].copy()

    var noise = gaussian_noise(16 * HL * WL, seed)
    var nshape = [1, 16, HL, WL]
    # Keep the latent boundary BF16; scheduler arithmetic below casts locally.
    var x = Tensor.from_host(noise, nshape^, STDtype.BF16, ctx)

    var sigmas = _build_sigmas(steps)
    print("[denoise]", steps, "steps, shift", SHIFT, "CFG", cfg, "seed", seed)
    _stats("init_noise", x, ctx)
    events.append(ZImageEvent(ZEVENT_STARTED, 0, steps, String("started")))
    for i in range(steps):
        var step_t0 = perf_counter_ns()
        var t = 1.0 - sigmas[i]  # DiT timestep convention (= 1 - sigma)
        # Diffusers CFG (code form, F32): pred_raw = vc + cfg*(vc - vu);
        # pipeline_z_image.py negates before FlowMatchEulerDiscreteScheduler.step.
        var pred = _cfg_pred_overlay[HL, WL, N_IMG, N_TXT, S](
            x, t, cfg, cap_seq_cond, cap_seq_uncond,
            nr_blocks, cr_blocks, main_blocks, lora_dev,
            aux, final_lin_w, final_lin_b, x_pad_h,
            x_cos_cond[], x_sin_cond[], cap_cos_cond[], cap_sin_cond[],
            uni_cos_cond[], uni_sin_cond[],
            x_cos_uncond[], x_sin_uncond[], cap_cos_uncond[], cap_sin_uncond[],
            uni_cos_uncond[], uni_sin_uncond[],
            i == 0, ctx,
        )
        var dt = sigmas[i + 1] - sigmas[i]
        var x_compute = _cast(x, STDtype.F32, ctx)
        x = _cast(add(x_compute, mul_scalar(pred, dt, ctx), ctx), STDtype.BF16, ctx)
        events.append(ZImageEvent(ZEVENT_STEP, i + 1, steps, String("denoise")))
        if (i + 1) % 10 == 0 or i == steps - 1:
            var secs = Float64(perf_counter_ns() - step_t0) / 1.0e9
            var rate = Float64(0.0)
            if secs > 0.0:
                rate = Float64(1.0) / secs
            print_sample_step(String("ZImage-sample"), i + 1, steps, sigmas[i], secs, rate)
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


def _select_prompt(sample_cfg: SamplePromptConfig, wanted: String) raises -> SamplePrompt:
    if len(sample_cfg.prompts) == 0:
        raise Error("zimage_generate: sample prompt JSON has no prompts")
    if wanted == String(""):
        return sample_cfg.prompts[0].copy()
    for i in range(len(sample_cfg.prompts)):
        if sample_cfg.prompts[i].label == wanted:
            return sample_cfg.prompts[i].copy()
    raise Error(String("zimage_generate: prompt id not found: ") + wanted)


def _load_prompt_json(
    path: String, wanted: String,
    mut prompt: String, mut negative: String,
    mut steps: Int, mut cfg: Float32, mut seed: UInt64,
    mut width: Int, mut height: Int,
) raises:
    var sample_cfg = read_sample_prompt_config(path)
    var p = _select_prompt(sample_cfg, wanted)
    if p.frames != 1:
        raise Error("zimage_generate: only image prompts are supported")
    prompt = p.prompt.copy()
    negative = p.negative.copy()
    steps = p.steps
    cfg = p.cfg
    seed = p.seed
    width = p.width
    height = p.height


# ── standalone demo: drives zimage_generate with the original constants so the
# existing single-image path still works (compile + behavior preserved). ──
def main() raises:
    var ctx = DeviceContext()
    print("=== Z-Image generate (runtime prompt, fixed-padded caption) — 1024x1024 ===")
    var a = argv()
    var lora_path = String("")
    var out_path = String(OUT)
    var seed = DEFAULT_SEED
    var prompt = String(DEFAULT_PROMPT)
    var negative = String("")
    var steps = DEFAULT_STEPS
    var cfg = DEFAULT_CFG
    var width = 1024
    var height = 1024
    if len(a) >= 2:
        var arg_lora = String(a[1])
        if arg_lora != String("base") and arg_lora != String("none") and arg_lora != String(""):
            lora_path = arg_lora
    if len(a) >= 3:
        out_path = String(a[2])
    if len(a) >= 4 and String(a[3]).endswith(".json"):
        var wanted = String("")
        if len(a) >= 5:
            wanted = String(a[4])
        _load_prompt_json(String(a[3]), wanted, prompt, negative, steps, cfg, seed, width, height)
    elif len(a) >= 4:
        seed = UInt64(_parse_nonnegative_int(String(a[3])))
        if len(a) >= 5 and String(a[4]).endswith(".json"):
            var wanted2 = String("")
            if len(a) >= 6:
                wanted2 = String(a[5])
            _load_prompt_json(String(a[4]), wanted2, prompt, negative, steps, cfg, seed, width, height)
        elif len(a) >= 5:
            prompt = String(a[4])
    print("[prompt]", prompt)
    var events = List[ZImageEvent]()
    var rgb = zimage_generate(
        prompt, negative,
        steps, cfg, seed,
        width, height, lora_path, Float32(1.0), events, ctx,
    )
    var rs = rgb.shape()
    print("[vae] image:", rs[0], rs[1], rs[2], rs[3], " events:", len(events))
    save_png(rgb, out_path, ctx, ValueRange.SIGNED)
    print_sample_saved(String("ZImage-sample"), out_path)
    print("[done] saved:", out_path)
