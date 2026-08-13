# serenitymojo/pipeline/wan22_t2v.mojo — Wan2.2-TI2V-5B text-to-video, pure Mojo.
#
# First-video generation pipeline for the Wan2.2-TI2V-5B checkpoint. BF16
# blocks are copied once into a complete pinned-host store before step 0;
# denoise stages from RAM only. FP8 uses the Mojo-generated resident sidecar.
# Wraps the parity-gated DiT spine with the creator's flow-UniPC scheduler + CFG
# denoise loop, then the temporal VAE decoder
# (models/vae/wan22_decoder.Wan22VaeImageDecoder) → PNG frames.
#
# ── RECIPE (sourced from reference implementations, file:line) ───────────────
#   * num_inference_steps = 50   (native wan_ti2v_5B.py:34 sample_steps=50;
#       diffusers pipeline_wan.py:390). Runtime-overridable via argv.
#   * guidance_scale (CFG) = 5.0 native (wan_ti2v_5B.py:35). The earlier 3.0
#       workaround applied to the non-creator Euler trajectory and is not
#       carried into this creator-UniPC route. Runtime-overridable via argv.
#   * flow shift          = 5.0 (native wan_ti2v_5B.py:33 sample_shift=5.0).
#   * sampler: creator FlowUniPCMultistepScheduler, bh2, order=2, predict_x0,
#       lower_order_final, final sigma zero. The shared Mojo scheduler is gated
#       step-by-step against that canonical implementation.
#   * CFG: v = v_uncond + guidance*(v_cond - v_uncond)  (diffusers :629;
#       native textimage2video.py:385).
#   * geometry: latent channels z_dim=48 (config.json in_dim=48); VAE stride
#       (temporal 4, spatial 16) (native wan_ti2v_5B.py:17 vae_stride=(4,16,16));
#       DiT patch (t,h,w)=(1,2,2) (wan_ti2v_5B.py:20). num_latent_frames =
#       (F-1)//4 + 1 (diffusers pipeline_wan.py:340). H,W must be multiples of
#       vae_spatial*patch = 32.
#   * latent init: randn([48, T_lat, H_lat, W_lat]) F32, init_noise_sigma=1
#       (diffusers :341-347). Timestep in [0,1000] fed to the DiT.
#
# ── CONDS CONTRACT (input safetensors; produced by the umt5 encoder CLI) ─────
#   The DiT applies text_embedding (Linear-GELU-Linear) INTERNALLY, so the conds
#   file carries the RAW umt5-xxl encoder hidden states (dim text_dim=4096),
#   zero-padded/truncated to EXACTLY text_len=512 tokens (Wan pads context to
#   512 and cross-attends over all 512, padding unmasked — wan22_dit.mojo:311):
#     key "pos_embed": [512, 4096]   positive-prompt umt5 last_hidden_state
#     key "neg_embed": [512, 4096]   negative-prompt umt5 last_hidden_state
#   dtype F32 or BF16 (this pipeline casts to BF16). The 512 length is COMPTIME
#   here (CTXL) — the encoder MUST emit exactly 512 rows.
#
# ── GEOMETRY IS COMPTIME (Mojo) ──────────────────────────────────────────────
#   The DiT forward is specialized on FG/HG/WG/S and the VAE decoder on [LH,LW],
#   so resolution + frame count are COMPILE-TIME constants below. To change
#   them, edit HEIGHT/WIDTH/FRAMES and rebuild. argv `frames` is validated
#   against the compiled FRAMES (errors with a rebuild hint on mismatch); `steps`
#   and `seed` are true runtime args.
#   RTX 5080 product: 832x480, 121 frames (5.04s@24fps), S=12090. The prior
#   832x576/S=14508 route was measured at 14,711 MiB before an OOM on UniPC step
#   3. A full 832x480 run completed but peaked at 15,568/16,303 MiB, leaving only
#   734 MiB; scoped release proved byte-identical but measured the same full-run
#   peak, locating it in late denoise. The inference path now ends the training
#   SDPA wrapper's padded Q/K/V/statistics lifetime immediately after each
#   cuDNN forward; creator full-forward parity remains unchanged at 0.997239.
#
# ── DTYPE / VRAM ─────────────────────────────────────────────────────────────
#   300 large block matrices load from the pinned 4.8GB E4M3 sidecar with F32
#   per-row scales; 525 small tensors remain BF16. Each active block is restored
#   to BF16 for the existing math. Full-forward creator parity is 0.997239.
#
# argv: <conds.safetensors> <out_dir> [frames=121] [steps=50] [seed=0]
#       [guidance=5.0] [decode=1] [first_image=""] [lora="-"]
#       [lora_mult=1.0] [precision="fp8"|"bf16"]
#
# Mojo 1.0.0b1, NVIDIA GPU. NO in-pipeline text encode (ephemeral-encode split).

from std.sys import argv
from std.sys.defines import get_defined_int
from max.gpu.host import DeviceContext
from std.math import round
from image.studio_ops import resize_lanczos
from image.transform import crop

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.lora import LoraSet
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    add, mul, mul_scalar, permute, reshape, slice, full_device, concat,
)
from serenitymojo.models.dit.wan22_dit import (
    Wan22Config, Wan22DiT, Wan22DiTOffloaded,
)
from serenitymojo.models.vae.wan22_decoder import Wan22VaeImageDecoder
from serenitymojo.models.vae.wan22_vae_encoder import Wan22VaeImageEncoder
from serenitymojo.sampling.unipc import UniPcMultistepScheduler
from serenitymojo.components.artifacts import (
    video_frame_path, build_ffmpeg_mux_command, save_video_frame_png,
)
from serenitymojo.image.png import ValueRange
from serenitymojo.serve.image_io import (
    decode_image_any, image_to_signed_nchw,
)


# ── Paths ────────────────────────────────────────────────────────────────────
comptime CKPT_DIR = (
    "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-Mojo"
)
comptime FP8_CACHE = (
    "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-Mojo/"
    "wan22_dit_fp8_e4m3_b8fff7315c768468.safetensors"
)
comptime VAE_PATH = "/home/alex/.serenity/models/vaes/wan2.2_vae.safetensors"

# ── First-video geometry (COMPTIME — rebuild to change) ──────────────────────
comptime HEIGHT = get_defined_int["WAN22_HEIGHT", 480]()
comptime WIDTH = get_defined_int["WAN22_WIDTH", 832]()
comptime FRAMES = get_defined_int["WAN22_FRAMES", 121]()
comptime H_LAT = HEIGHT // 16  # 30 (VAE spatial stride 16)
comptime W_LAT = WIDTH // 16   # 52
comptime T_LAT = (FRAMES - 1) // 4 + 1  # 31 (VAE temporal stride 4)
comptime FG = T_LAT            # 31 (DiT temporal patch 1)
comptime HG = H_LAT // 2       # 15 (DiT spatial patch 2)
comptime WG = W_LAT // 2       # 26
comptime S = FG * HG * WG      # 12090 tokens
comptime TXT = 512             # text_len (cross-attn kv length)
comptime CTXL = 512            # raw context length fed in (== TXT; conds padded)
comptime TEXT_DIM = 4096       # umt5-xxl hidden
comptime NH = 24               # heads
comptime HD = 128              # head_dim

# ── Recipe scalars ───────────────────────────────────────────────────────────
# guidance_scale (CFG). Native recipe = 5.0, but that OVER-DRIVES the low-sigma
# denoise steps for multi-frame renders (F>=4): guidance*5 amplifies the sharp
# late velocity F-dependently, so the F=4 latent std climbs 0.77->1.31 over the
# last ~8 low-sigma steps and decodes to a fractal/over-sharpened frame. 3.0
# keeps std ~1.10 and decodes coherent (HANDOFF 2026-07-11; MEASURED F=4 std
# 1.31->1.10, fractal->coherent face). Runtime-overridable via argv (GUIDANCE is
# used in the CFG loop, not comptime-specialized, so it can be a runtime scalar).
comptime DEFAULT_GUIDANCE = Float32(5.0)
comptime CREATOR_NATIVE_SHIFT = Float32(5.0)
comptime DEFAULT_T2V_STEPS = 50
comptime DEFAULT_I2V_STEPS = 50
comptime DEFAULT_SEED = 0
comptime NUM_TRAIN_TIMESTEPS = Float32(1000.0)


# ── Load a [512,4096] context embed as BF16, shape-checked ───────────────────
def _load_embed(
    st: ShardedSafeTensors, key: String, ctx: DeviceContext
) raises -> Tensor:
    if key not in st.names():
        raise Error(String("conds file missing key '") + key + "'")
    var tv = st.tensor_view(key)
    var t = Tensor.from_view_as_bf16(tv, ctx)
    if t.numel() != CTXL * TEXT_DIM:
        raise Error(
            String("conds '") + key + "' has numel " + String(t.numel())
            + " != " + String(CTXL * TEXT_DIM)
            + " (expected [512,4096] raw umt5-xxl hidden, zero-padded to 512)"
        )
    # Normalize to [CTXL, TEXT_DIM] for the DiT forward's _pad_context.
    return reshape(t, [CTXL, TEXT_DIM], ctx)


# ── Valid (non-pad) token count from a [1] F32 conds len key ─────────────────
# Missing key → treat all CTXL rows as valid (the old, degenerate behavior).
def _load_len(
    st: ShardedSafeTensors, key: String, ctx: DeviceContext
) raises -> Int:
    if key not in st.names():
        return CTXL
    var tv = st.tensor_view(key)
    var h = Tensor.from_view(tv, ctx).to_host(ctx)
    if len(h) < 1:
        return CTXL
    var v = Int(h[0])
    if v < 0:
        v = 0
    if v > CTXL:
        v = CTXL
    return v


# ── Zero context rows >= valid (matches diffusers pipeline_wan.py:188-190) ────
# Wan zero-pads the umt5 output to 512 AND the DiT cross-attends UNMASKED over
# all 512 (wan22_dit.mojo:310 context_lens=None). So the pad rows MUST be exactly
# zero — otherwise the umt5 port's non-zero pad rows (measured row-norm ~5-8, ≥
# the valid rows, ~500 of them) dominate the text cross-attention and collapse
# the denoise to a near-constant latent (the dark-green first-light frames).
# Zeroing here makes the pad rows' cross-attn k/v identical to diffusers (bias-
# only); the valid rows remain perturbed by the encoder's no-mask self-attention
# (the recorded ~cos-0.99 exactness lever, fixed encoder-side with a pad mask).
def _zero_pad_rows(
    embed: Tensor, valid: Int, ctx: DeviceContext
) raises -> Tensor:
    var mvals = List[Float32](capacity=CTXL)
    for i in range(CTXL):
        if i < valid:
            mvals.append(Float32(1.0))
        else:
            mvals.append(Float32(0.0))
    var mask = Tensor.from_host(mvals, [CTXL, 1], STDtype.BF16, ctx)
    return mul(embed, mask, ctx)


def _zero_condition(ctx: DeviceContext) raises -> Tensor:
    return full_device(
        [48, T_LAT, H_LAT, W_LAT], Float32(0.0), STDtype.F32, ctx
    )


def _encode_i2v_first_frame(
    image_path: String, ctx: DeviceContext
) raises -> Tensor:
    """Creator TI2V image preprocessing and VAE first-frame conditioning.

    Official Wan cover-resizes with preserved aspect ratio and center-crops to
    the 32-aligned output grid, normalizes RGB to [-1,1], VAE-encodes one
    image, and leaves the remaining latent frames empty.
    """
    var image = decode_image_any(image_path)
    var scale = max(
        Float64(WIDTH) / Float64(image.width),
        Float64(HEIGHT) / Float64(image.height),
    )
    var resized_width = Int(round(Float64(image.width) * scale))
    var resized_height = Int(round(Float64(image.height) * scale))
    var resized = resize_lanczos(image, resized_width, resized_height)
    var crop_left = (resized.width - WIDTH) // 2
    var crop_top = (resized.height - HEIGHT) // 2
    var cropped = crop(resized, crop_left, crop_top, WIDTH, HEIGHT)
    var values = image_to_signed_nchw(cropped)
    var pixels = Tensor.from_host(
        values, [1, 3, HEIGHT, WIDTH], STDtype.BF16, ctx
    )
    var vae = Wan22VaeImageEncoder[HEIGHT, WIDTH].load(
        String(VAE_PATH), ctx
    )
    var encoded = vae.encode_image(pixels, ctx)
    var first = cast_tensor(
        reshape(encoded, [48, 1, H_LAT, W_LAT], ctx),
        STDtype.F32,
        ctx,
    )
    var tail = full_device(
        [48, T_LAT - 1, H_LAT, W_LAT],
        Float32(0.0),
        STDtype.F32,
        ctx,
    )
    return concat(1, ctx, first, tail)


def _load_i2v_first_frame(
    latent_path: String, ctx: DeviceContext
) raises -> Tensor:
    """Load the process-isolated creator VAE result and append zero tail."""
    var st = ShardedSafeTensors.open(latent_path)
    var key = String("first_latent")
    if key not in st.names():
        raise Error("Wan first-frame latent cache missing 'first_latent'")
    var first = Tensor.from_view_as_f32(st.tensor_view(key), ctx)
    if first.numel() != 48 * H_LAT * W_LAT:
        raise Error(
            String("Wan first-frame latent numel ")
            + String(first.numel()) + String(" != ")
            + String(48 * H_LAT * W_LAT)
        )
    first = reshape(first, [48, 1, H_LAT, W_LAT], ctx)
    var tail = full_device(
        [48, T_LAT - 1, H_LAT, W_LAT],
        Float32(0.0),
        STDtype.F32,
        ctx,
    )
    return concat(1, ctx, first, tail)


def _clamp_creator_first_frame(
    x: Tensor, condition: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var first = slice(condition, 1, 0, 1, ctx)
    var generated = slice(x, 1, 1, T_LAT - 1, ctx)
    return concat(1, ctx, first, generated)


def _creator_i2v_token_timesteps(
    timestep: Float32, ctx: DeviceContext
) raises -> Tensor:
    """mask2[0][0][:,::2,::2] * t, flattened in patch-token order."""
    comptime FIRST_FRAME_TOKENS = HG * WG
    var conditioned = full_device(
        [FIRST_FRAME_TOKENS], Float32(0.0), STDtype.F32, ctx
    )
    var generated = full_device(
        [S - FIRST_FRAME_TOKENS], timestep, STDtype.F32, ctx
    )
    return concat(0, ctx, conditioned, generated)


def _denoise_scoped(
    pos: Tensor,
    neg: Tensor,
    condition: Tensor,
    i2v: Bool,
    steps: Int,
    seed: UInt64,
    guidance: Float32,
    shift: Float32,
    lora_path: String,
    lora_mult: Float32,
    precision: String,
    ctx: DeviceContext,
) raises -> Tensor:
    """Run the DiT in its own lifetime so resident weights die before VAE load."""
    var cfg = Wan22Config.ti2v_5b()
    if precision == String("bf16"):
        print(
            "  loading Wan2.2-TI2V-5B exact-BF16 pinned-host store from",
            CKPT_DIR,
        )
        var offloaded: Wan22DiTOffloaded
        if lora_path.byte_length() > 0 and lora_path != String("-"):
            print("  loading Wan TI2V-5B BF16 host-resident LoRA:", lora_path)
            offloaded = Wan22DiTOffloaded.load_with_lora(
                String(CKPT_DIR), cfg, lora_path, lora_mult, ctx
            )
        else:
            offloaded = Wan22DiTOffloaded.load(
                String(CKPT_DIR), cfg, ctx
            )
        print("  weights ready.")
        var x = randn(
            [48, T_LAT, H_LAT, W_LAT], seed, STDtype.F32, ctx
        )
        if i2v:
            x = _clamp_creator_first_frame(x, condition, ctx)
        var scheduler = UniPcMultistepScheduler(
            1000, steps, Float64(shift), 2
        )
        var sigmas = scheduler.sigmas()
        ctx.synchronize()
        print("  phase=sampling step=0 total=", steps)
        for i in range(steps):
            var t = Float32(sigmas[i]) * NUM_TRAIN_TIMESTEPS
            var token_timesteps: Tensor
            if i2v:
                token_timesteps = _creator_i2v_token_timesteps(t, ctx)
            else:
                token_timesteps = full_device(
                    [S], t, STDtype.F32, ctx
                )
            var x_bf = cast_tensor(x, STDtype.BF16, ctx)
            var preds = offloaded.forward_cfg_timesteps[
                FG, HG, WG, S, TXT, CTXL, NH, HD
            ](x_bf, token_timesteps, pos, neg, ctx)
            var vc = cast_tensor(preds.cond, STDtype.F32, ctx)
            var vu = cast_tensor(preds.uncond, STDtype.F32, ctx)
            var v = add(
                mul_scalar(vu, Float32(1.0) - guidance, ctx),
                mul_scalar(vc, guidance, ctx),
                ctx,
            )
            x = scheduler.step(v, x, ctx)
            if i2v:
                x = _clamp_creator_first_frame(x, condition, ctx)
            # Fence one complete creator step. This makes the lifetime of all
            # 30 released block buffers explicit and bounds the async allocator
            # high-water mark before the next CFG pair begins.
            ctx.synchronize()
            print(
                "  phase=sampling step=", i + 1, " total=", steps,
                " sigma=", sigmas[i], " t=", t,
            )
        return x^

    print("  loading Wan2.2-TI2V-5B weights from FP8 cache", FP8_CACHE)
    var model = Wan22DiT.load_fp8_cache(String(FP8_CACHE), cfg, ctx)
    print("  weights loaded.")
    if lora_path.byte_length() > 0 and lora_path != String("-"):
        print("  loading Wan TI2V-5B LoRA:", lora_path)
        _ = model.merge_lora_fp8_resident(
            LoraSet.load(lora_path), lora_mult, ctx
        )
    var x = randn([48, T_LAT, H_LAT, W_LAT], seed, STDtype.F32, ctx)
    if i2v:
        # Official Wan applies this replacement both before the first denoise
        # call and after every scheduler step.
        x = _clamp_creator_first_frame(x, condition, ctx)
    var scheduler = UniPcMultistepScheduler(
        1000, steps, Float64(shift), 2
    )
    var sigmas = scheduler.sigmas()
    ctx.synchronize()
    print("  phase=sampling step=0 total=", steps)
    for i in range(steps):
        var t = Float32(sigmas[i]) * NUM_TRAIN_TIMESTEPS
        var x_bf = cast_tensor(x, STDtype.BF16, ctx)
        var v_cond: Tensor
        var v_unc: Tensor
        if i2v:
            var token_timesteps = _creator_i2v_token_timesteps(t, ctx)
            v_cond = model.forward_timesteps[
                FG, HG, WG, S, TXT, CTXL, NH, HD
            ](x_bf, token_timesteps, pos, ctx)
            v_unc = model.forward_timesteps[
                FG, HG, WG, S, TXT, CTXL, NH, HD
            ](x_bf, token_timesteps, neg, ctx)
        else:
            v_cond = model.forward[FG, HG, WG, S, TXT, CTXL, NH, HD](
                x_bf, t, pos, ctx
            )
            v_unc = model.forward[FG, HG, WG, S, TXT, CTXL, NH, HD](
                x_bf, t, neg, ctx
            )
        var vc = cast_tensor(v_cond, STDtype.F32, ctx)
        var vu = cast_tensor(v_unc, STDtype.F32, ctx)
        var v = add(
            mul_scalar(vu, Float32(1.0) - guidance, ctx),
            mul_scalar(vc, guidance, ctx),
            ctx,
        )
        x = scheduler.step(v, x, ctx)
        if i2v:
            x = _clamp_creator_first_frame(x, condition, ctx)
        ctx.synchronize()
        print(
            "  phase=sampling step=", i + 1, " total=", steps,
            " sigma=", sigmas[i], " t=", t,
        )
    return x^


def main() raises:
    var args = argv()
    if len(args) < 3:
        print("usage: wan22_t2v <conds.safetensors> <out_dir> [frames] [steps] [seed] [guidance] [decode=1] [first_image=\"\"] [lora=\"-\"] [lora_mult=1.0] [precision=fp8|bf16]")
        print("  conds.safetensors: keys pos_embed[512,4096], neg_embed[512,4096]")
        print("  first_image: PNG/JPEG/WebP enables creator TI2V first-frame conditioning")
        print("  compiled geometry:", WIDTH, "x", HEIGHT, ",", FRAMES, "frames, S=", S)
        return

    var conds_path = String(args[1])
    var out_dir = String(args[2])
    var req_frames = FRAMES
    if len(args) >= 4:
        req_frames = atol(String(args[3]))
    var image_path = String("")
    if len(args) >= 9:
        image_path = String(args[8])
    var i2v = image_path.byte_length() > 0
    var lora_path = String("-")
    if len(args) >= 10:
        lora_path = String(args[9])
    var lora_mult = Float32(1.0)
    if len(args) >= 11:
        lora_mult = Float32(Float64(String(args[10])))
    var precision = String("fp8")
    if len(args) >= 12:
        precision = String(args[11])
    var steps = DEFAULT_I2V_STEPS if i2v else DEFAULT_T2V_STEPS
    if len(args) >= 5:
        steps = atol(String(args[4]))
    var seed = UInt64(DEFAULT_SEED)
    if len(args) >= 6:
        seed = UInt64(atol(String(args[5])))
    var guidance = DEFAULT_GUIDANCE
    if len(args) >= 7:
        guidance = Float32(Float64(String(args[6])))
    var decode = True
    if len(args) >= 8:
        decode = atol(String(args[7])) != 0

    if req_frames != FRAMES:
        raise Error(
            String("this binary is compiled for FRAMES=") + String(FRAMES)
            + " (geometry is comptime); to render " + String(req_frames)
            + " frames, edit FRAMES in wan22_t2v.mojo and rebuild"
        )
    if steps < 1:
        raise Error("steps must be >= 1")
    if precision != String("fp8") and precision != String("bf16"):
        raise Error("precision must be 'fp8' or 'bf16'")

    # The official TI2V-5B 720p CLI inherits config shift=5 for both T2V and
    # I2V. Shift=3 is only recommended by the creator for a 480p I2V render.
    var shift = CREATOR_NATIVE_SHIFT
    print("=== Wan2.2-TI2V-5B", "I2V" if i2v else "T2V", "===")
    print("  geometry:", WIDTH, "x", HEIGHT, ",", FRAMES, "frames ->",
          "latent [48,", T_LAT, ",", H_LAT, ",", W_LAT, "], S=", S)
    print(
        "  steps=", steps, " guidance=", guidance, " shift=", shift,
        " seed=", seed, " precision=", precision,
    )
    if lora_path.byte_length() > 0 and lora_path != String("-"):
        print("  LoRA:", lora_path, " multiplier=", lora_mult)

    var ctx = DeviceContext()
    # 1. Conds (pre-encoded umt5-xxl hidden states, raw, padded to 512).
    print("  loading conds:", conds_path)
    var cst = ShardedSafeTensors.open(conds_path)
    var pos = _load_embed(cst, String("pos_embed"), ctx)
    var neg = _load_embed(cst, String("neg_embed"), ctx)
    # Zero the pad rows (>= valid token count) so the unmasked DiT cross-attn
    # sees exactly what diffusers feeds — else non-zero umt5 pad rows dominate.
    var pos_valid = _load_len(cst, String("pos_len"), ctx)
    var neg_valid = _load_len(cst, String("neg_len"), ctx)
    print("  conds valid rows: pos=", pos_valid, " neg=", neg_valid,
          "(rows >= valid zeroed to match diffusers zero-pad)")
    pos = _zero_pad_rows(pos, pos_valid, ctx)
    neg = _zero_pad_rows(neg, neg_valid, ctx)

    # 2. Product I2V passes a process-isolated VAE latent cache. Direct image
    # input remains a developer fallback, but native BF16 product runs must use
    # the cache so the VAE allocator cannot fragment the DiT process.
    var condition = _zero_condition(ctx)
    if i2v:
        if image_path.endswith(".safetensors"):
            print("  loading process-isolated creator first frame:", image_path)
            condition = _load_i2v_first_frame(image_path, ctx)
        else:
            print("  creator first-frame encode (developer fallback):", image_path)
            condition = _encode_i2v_first_frame(image_path, ctx)
        ctx.synchronize()
        print("  first-frame condition ready")

    # 3-6. Cached DiT + creator UniPC in a nested lifetime. On return, resident
    # weights and scheduler history are gone before the VAE is allocated.
    var x = _denoise_scoped(
        pos, neg, condition, i2v, steps, seed, guidance, shift,
        lora_path, lora_mult, precision, ctx
    )
    print("  denoise scope released; loading VAE next")

    if not decode:
        print("GATE denoise-only final latent numel=", x.numel())
        return

    # 6. Denoised latent [48,T,H,W] → tokens [T*H*W, 48] (t,h,w-major, ch last).
    var thwc = permute(x, [1, 2, 3, 0], ctx)          # [T,H,W,C]
    var tokens = reshape(thwc, [T_LAT * H_LAT * W_LAT, 48], ctx)

    # 7. VAE temporal decode → [1,3,(T-1)*4+1, 16*H_LAT, 16*W_LAT]. The decoder
    #    denormalizes (z*std+mean) internally.
    print("  VAE decode (T_lat=", T_LAT, ")...")
    var vae = Wan22VaeImageDecoder[H_LAT, W_LAT].load(String(VAE_PATH), ctx)
    var video = vae.decode_video_tokens(tokens, T_LAT, ctx)
    var vs = video.shape()
    print("  video:", vs[0], vs[1], vs[2], vs[3], vs[4])

    # 8. Write PNG frames.
    _ = sys_system(String("mkdir -p '") + out_dir + "'")
    var prefix = out_dir + String("/frame_")
    var nframes = vs[2]
    for f in range(nframes):
        var p = video_frame_path(prefix, f)
        save_video_frame_png(video, f, p, H_LAT, W_LAT, ctx, ValueRange.SIGNED)
    print("  wrote", nframes, "frames ->", out_dir)

    # ffmpeg mux command (printed, NOT run).
    var mux = build_ffmpeg_mux_command(
        prefix, String(".png"), out_dir + String("/wan22_t2v.mp4"), 24
    )
    print("  mux (run manually):", mux)
    print("GATE done frames=", nframes)
