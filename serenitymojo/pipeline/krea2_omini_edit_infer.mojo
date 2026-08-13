# pipeline/krea2_omini_edit_infer.mojo — Krea-2 OminiControl EDIT inference (C8).
#
# THE PROBLEM THIS FIXES. The C6 trainer trains a COND-ROW-ONLY LoRA: the adapter
# delta is computed on rows [cond_off, cond_off+cond_len) of the sequence
# [TXT_real | IMG | COND | TXT_pad] and scatter-added back there, and nowhere
# else (C3). The existing krea2 sampler builds [TXT | IMG] with NO condition
# rows, so that LoRA contributes EXACTLY NOTHING there — a render through it is
# the frozen base wearing a trained-sample label. train_krea2 refuses to do that
# and prints a deferral message instead. This binary is what the message defers
# to: the SAME sequence, the SAME switches, the SAME streamed stack.
#
# WHAT IT DOES, step by step:
#   1. VAE-encodes the user's condition image with QwenImageVaeEncoder and
#      normalizes it (z - latents_mean)/latents_std — byte-for-byte the contract
#      krea2_prepare_cache._encode_one_latent used to write `ref.<i>`, so the
#      condition the model sees at render time is produced the same way as the
#      condition it saw at train time.
#   2. Builds [TXT_real(lt) | IMG | COND | TXT_pad] through
#      krea2_infer.build_conditioning's CONDL > 0 arm, which reuses the READER's
#      krea2_build_pos_cond / krea2_reorder_combined_edit and the LAYOUT module's
#      Krea2OminiLayout — no duplicated offset or position math anywhere.
#   3. Runs krea2_stack_lora_forward_streamed with vec_cond / cond_off / cond_len
#      — the identical three switches the C6 trainer step passes — so the LoRA
#      lands on the condition rows and the per-segment modulation (mods(t) for
#      txt+img, mods(t=0) for cond) matches training.
#   4. Denoises the IMAGE ROWS ONLY. The velocity is final[:, lt : lt+imglen, :];
#      the euler update touches only `latent`; the condition tokens are rebuilt
#      from the CLEAN source every step and are never noised.
#   5. VAE-decodes to PNG.
#
# THE TWO OMINICONTROL STRENGTH KNOBS (flux_omini.py):
#   --condition-scale S   additive log(S) attention-logit bias between COND and
#                         non-COND tokens, both directions (:280-341).
#                         S = 1.0 is THE IDENTITY (log 1 = 0) and is the default:
#                         no bias tensor is built and the attention takes the
#                         cuDNN flash-padmask arm the LoRA was trained under.
#                         S != 1.0 swaps in a masked math SDPA — see the loud
#                         UNVERIFIED banner it prints.
#   --image-guidance G    true image-CFG (:657-658, :758-781): a second forward
#                         whose ONLY change is a condition encoded from a BLACK
#                         image, then unc + G*(cond - unc). G = 1.0 = OFF.
#
# ⚠ HONEST CAVEAT ON IMAGE-CFG, STATED UP FRONT. OminiControl's image-CFG works
# because they train with drop_text 0.1 AND drop_image 0.1 — the model has SEEN
# an empty condition and learned what "no condition" means. OUR run
# (krea2_omini_edit_512.json) has caption_dropout_prob 0.0 and NO condition
# dropout at all. So the black-image branch is off-distribution for this adapter
# and image-CFG may do nothing useful, or may degrade the render. It is wired
# because the vertical needs it, not because it is expected to help this
# checkpoint. Do not read a bad --image-guidance render as a bug in the wiring.
#
# ── DEVICE-VERIFIED 2026-07-30 (was: "nothing here has been executed") ───────
# V1/V2/V3 all PASS on the RTX 5080, and the --cond-image arm — the one the C8
# author could never run — is now measured, and it WAS BROKEN. See
# _load_signed_image for the defect (aspect squash vs the stager's centre-crop)
# and the numbers. After the fix:
#   * --cond-image fed the SAME source the cache holds reproduces the --from-cache
#     render to RMSE 0.98/255 (0.00, i.e. BIT-IDENTICAL, when the resample is a
#     no-op) — so the VAE-encode, the inline_cond_from_context_edit context/pos
#     build, and the cond-row LoRA routing on this arm are all correct.
#   * HELD-OUT (never-trained images, instruction "restore and colorize this
#     photo", 20 steps, seed 1234, c6_3000): a modern B&W portrait comes back
#     colorised with structure intact (saturation 0.0 -> 34.4; the --no-lora
#     partner stays at 3.3 and is the usual checkerboard mush). A synthetically
#     damaged colour photo (scratches/spots/fade) comes back clean at RMSE 41.9
#     to its ground truth vs 50.8 for the damaged input and 58.1 for --no-lora.
#     The vertical GENERALISES; it is not a memoriser of the 96 training pairs.
# Still unrun: V5 condition_scale != 1, V6 image-CFG, V7 --trainer-text A/B.
#
# ── RUN ───────────────────────────────────────────────────────────────────────
# Two conditioning sources. Pick ONE.
#
# (A) FROM THE TRAINING CACHE — no text encoder, no VAE encode. This is the
#     round-trip check: condition = the exact `ref.<i>` the trainer saw, text =
#     the exact `context.<i>`, negative = the cached `context_uncond`.
#       pixi run mojo run -I . -I vendor/mojo-libs [BUILD FLAGS] \
#         serenitymojo/pipeline/krea2_omini_edit_infer.mojo \
#         --from-cache /home/alex/trainings/krea2_omini_edit_cache/cache.safetensors \
#         --index 0 --lora <c6_3000.safetensors> --out out.png
#
# (B) FROM A USER IMAGE + PROMPT. Encode the prompt FIRST in a child process (the
#     ~8 GB Qwen3-VL TE must not co-reside with the DiT):
#       pixi run mojo run -I . -I vendor/mojo-libs \
#         serenitymojo/pipeline/krea2_encode_cli.mojo \
#         "<instruction>" "" out/ctx_pos.bin out/ctx_neg.bin
#     then
#       pixi run mojo run -I . -I vendor/mojo-libs [BUILD FLAGS] \
#         serenitymojo/pipeline/krea2_omini_edit_infer.mojo \
#         --cond-image <source.png> --ctx-pos out/ctx_pos.bin \
#         --ctx-neg out/ctx_neg.bin --lora <c6_3000.safetensors> --out out.png
#
# Common flags: --steps N --seed N --cfg F --condition-scale F --image-guidance F
#               --no-lora  (frozen-base A/B: identical everything, LoRA removed)
#               --lora-mult F --stream (bf16 disk base instead of squareq_w4)
#               --trainer-text (drop the txtfusion refiner mask, matching the
#                               trainer's text conditioning exactly)
#
# ── VERIFICATION PLAN — ALL SEVEN RUN AND PASSED 2026-07-30 (RTX 5080) ───────
# Results, index 0, 20 steps, seed 1234 unless stated, c6_3000, distances RMSE/255
# in 512x512 pixel space against the STAGED tensors (sample_00000.safetensors),
# which are the trainer's own view of the pair:
#   V1 PASS  CONDLEN=1024 LFULL_EDIT=2432, "LoRA overlay loaded: 224 adapters".
#   V2 PASS  LoRA 13.4 from target vs --no-lora 58.7 — the cond-row LoRA reaches
#            the render. (The --no-lora panel is checkerboard mush, as documented.)
#   V3 PASS  render 13.4 from target vs the damaged condition's own 42.8: it moves
#            two thirds of the way to the target. Reproduces the handoff's 13.1
#            exactly — this binary is bit-identical to the recorded sample.
#   V4 PASS  seed 999 differs from seed 1234 (10.5) while both land the same
#            distance from the target (11.4 vs 13.4): the output is being
#            integrated, the condition is not.
#   V5 PASS  --condition-scale 1.0001 completes on the masked-SDPA arm, no OOM
#            (13.4 GB peak headroom held), 39.3 s vs 23.8 s, and the image is 0.52
#            from the identity render — the arm is sane, not just non-crashing.
#   V6 PASS(mechanically)  --image-guidance 1.5 completes, 47.6 s ~= 2x (the second
#            forward), no OOM. QUALITY: pushes 18.1 off the identity render and
#            AWAY from the target (27.6 vs 13.4) — exactly the off-distribution
#            degradation the dropout caveat above predicts. Not a wiring bug.
#   V7 PASS  --trainer-text is 1.24 from the default and marginally CLOSER to the
#            target (13.05 vs 13.31): the refiner-mask train/infer divergence is
#            real but tiny. Not worth chasing.
# Plus the arm C8 could not reach at all — see the DEVICE-VERIFIED block above.
#
# Original plan text follows. BIN = output/bin/krea2_omini_edit_infer,
# CACHE = /home/alex/trainings/krea2_omini_edit_cache/cache.safetensors,
# CKPT  = /home/alex/trainings/krea2_omini_edit_run1/checkpoints/c6_3000.safetensors
#
#  V1 ARGUMENT ECHO. $BIN --from-cache $CACHE --index 0 --lora $CKPT --steps 1
#     --out /tmp/v1.png
#     PASS: prints CONDLEN=1024 LFULL_EDIT=2432, "LoRA overlay loaded: 224
#     adapters", and completes. FAIL: any shape/count mismatch.
#  V2 LORA-ON vs LORA-OFF (THE headline check — proves the cond-row LoRA reaches
#     the render at all). Same index/seed/steps, twice:
#       $BIN --from-cache $CACHE --index 0 --lora $CKPT --seed 1234 --out /tmp/on.png
#       $BIN --from-cache $CACHE --index 0 --no-lora   --seed 1234 --out /tmp/off.png
#     PASS: the two PNGs DIFFER (compare bytes, then look). FAIL: identical ->
#     the LoRA is contributing nothing and C8 has not fixed C6's problem.
#  V3 ROUND TRIP ON A TRAINING PAIR. Index i's condition is a known source photo
#     and clean.<i> is its known target. Render index i (V2's "on" image is
#     already this for i=0) and compare against the decoded clean.<i>.
#     PASS: the render moves TOWARD the target (visually, and by any distance you
#     trust) relative to the --no-lora render of the same index.
#  V4 COND ROWS NEVER DENOISED / NEVER LOSSED. Two independent observations:
#     (a) code: the velocity is stack.velocity = final[:, lt : lt+imglen, :] and
#         the euler update writes only `latent`; the condition tokens are rebuilt
#         from the CLEAN source each step. (b) run: render index i twice with
#         different --seed. PASS: the CONDITION is bit-identical in both (it is
#         not part of the state being integrated) while the output differs.
#         A stronger check is the existing gate:
#         parity/krea2_omini_c6_loss_mask_gate (already PASS).
#  V5 CONDITION_SCALE IDENTITY. --condition-scale 1.0 vs the default: must be the
#     SAME code path (no bias tensor). Then --condition-scale 1.0001 to smoke the
#     masked-SDPA arm. PASS for the smoke: it completes and the image is CLOSE to
#     the identity render. FAIL: OOM, crash, or a wildly different image -> the
#     masked-SDPA arm is wrong; fall back to condition_scale = 1.0.
#  V6 IMAGE-CFG. --image-guidance 1.5. PASS: completes, ~2x the step time. Its
#     QUALITY is NOT a pass criterion — see the dropout caveat above.
#  V7 TRAINER-TEXT A/B. --trainer-text vs default, same seed. Tells you how much
#     of any residual gap is the refiner-mask train/infer divergence.
#
# Mojo 1.0.0b1, NVIDIA GPU. Inference-only.

from std.sys import argv
from max.gpu.host import DeviceContext
from std.collections import Optional
from std.memory import ArcPointer
from std.time import perf_counter_ns
from std.sys.defines import get_defined_int

from image.buffer import Image
from image.transform import crop, resize_bicubic

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.cap_cache import load_tensor_bin, save_tensor_bin
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.serve.image_io import decode_image_any, image_to_signed_nchw

from serenitymojo.models.vae.qwenimage_encoder import QwenImageVaeEncoder
from serenitymojo.models.krea2.krea2_prepare_cache import (
    _mean_ch, _std_ch, _normalize_latent, KREA2_VAE_ENC_FILE,
)
from serenitymojo.models.krea2.krea2_cache_reader import (
    KreaTrainCache, krea2_patchify,
)
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StreamFinal,
    Krea2ResidentFp8, Krea2ResidentSquareq, build_krea2_resident_squareq,
    build_krea2_resident_int8,
)
from serenitymojo.models.dit.krea2_dit import (
    Krea2ResidentInt8, Krea2HostInt8Inf, build_krea2_host_int8_inf,
)
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_infer import (
    Krea2InlineCond, Krea2ResidentCond, load_krea2_resident_cond,
    inline_cond_from_context_edit, inline_cond_from_context_edit_grid,
    load_krea2_stack_lora, load_krea2_omini_first_lora,
    empty_krea2_stack_lora, krea2_sample_latent, krea2_decode_latent_to_png,
)
from serenitymojo.pipeline.krea2_paths import (
    KREA2_RAW, KREA2_VAE_DIR, KREA2_RAW_KEY_PREFIX,
)

comptime TArc = ArcPointer[Tensor]

# ── COMPILE-TIME SAMPLE SHAPE ─────────────────────────────────────────────────
# Defaults preserve the device-verified 512 build byte-for-byte. Validation may
# deliberately render above the training resolution: build with
#   -DKREA2_EDIT_HEIGHT=1024 -DKREA2_EDIT_WIDTH=1024
# and the image/condition token counts both derive from that canvas. The edit
# condition overlaps the output spatially, so CONDLEN must always equal IMGLEN.
# LTMAX remains 384 by default to preserve the trainer's text layout; it can be
# raised independently for a cache with longer captions.
comptime HEIGHT = get_defined_int["KREA2_EDIT_HEIGHT", 512]()
comptime WIDTH = get_defined_int["KREA2_EDIT_WIDTH", 512]()
comptime LH = HEIGHT // 8
comptime LW = WIDTH // 8
comptime IMGLEN = (LH // 2) * (LW // 2)
comptime COND_HEIGHT = get_defined_int["KREA2_EDIT_COND_HEIGHT", HEIGHT]()
comptime COND_WIDTH = get_defined_int["KREA2_EDIT_COND_WIDTH", WIDTH]()
comptime COND_LH = COND_HEIGHT // 8
comptime COND_LW = COND_WIDTH // 8
comptime LTMAX = get_defined_int["KREA2_EDIT_LTMAX", 384]()
comptime CONDLEN = (COND_LH // 2) * (COND_LW // 2)
comptime LFULL_EDIT = LTMAX + IMGLEN + CONDLEN
comptime NBLOCKS = 28

# OminiControl condition type "edit": delta [0,0], scale 1.0 -> the condition
# grid EQUALS the image grid (spatial overlap). These are the values the cache
# was staged with (krea2_omini_edit_512.json omini_position_delta/scale) and the
# trainer fail-louds if a sample disagrees.
comptime COND_DELTA_H = 0
comptime COND_DELTA_W = 0
comptime COND_POS_SCALE = Float32(HEIGHT // COND_HEIGHT)

comptime SQUAREQ_SIDECAR = String("models/krea2/squareq_w4_r32g32")
comptime STEPS_DEFAULT = 28
comptime CFG_DEFAULT = Float32(0.0)          # krea2 creator profile: CFG off
comptime SEED_DEFAULT = UInt64(1234)


def _arg_str(args: List[String], flag: String, dflt: String) -> String:
    for i in range(len(args)):
        if args[i] == flag and i + 1 < len(args):
            return args[i + 1]
    return dflt


def _arg_has(args: List[String], flag: String) -> Bool:
    for i in range(len(args)):
        if args[i] == flag:
            return True
    return False


def _arg_int(args: List[String], flag: String, dflt: Int) raises -> Int:
    var s = _arg_str(args, flag, String(""))
    if s == String(""):
        return dflt
    return Int(s)


def _arg_f32(args: List[String], flag: String, dflt: Float32) raises -> Float32:
    var s = _arg_str(args, flag, String(""))
    if s == String(""):
        return dflt
    return Float32(Float64(s))


# ── condition image -> patchified CLEAN condition tokens ─────────────────────
# EXACTLY the edit cache's ref.<i> production chain
# (krea2_prepare_cache._encode_one_latent): [-1,1] NCHW -> BF16 -> encode_mean ->
# F32 -> (z - latents_mean)/latents_std -> BF16 -> krea2_patchify. Anything else
# here would hand the model a condition in a different space than training.
def _encode_condition(
    enc: QwenImageVaeEncoder[COND_HEIGHT, COND_WIDTH],
    img_signed: Tensor,          # [1,3,COND_HEIGHT,COND_WIDTH] F32 in [-1,1]
    ctx: DeviceContext,
) raises -> Tensor:
    var img_bf = cast_tensor(img_signed, STDtype.BF16, ctx)
    var lat_mean = enc.encode_mean(img_bf, ctx)                # [1,16,LH,LW] BF16
    var lat_f32 = cast_tensor(lat_mean, STDtype.F32, ctx)
    var mean_ch = _mean_ch(ctx)
    var std_ch = _std_ch(ctx)
    var clean_f32 = _normalize_latent(lat_f32, mean_ch, std_ch, ctx)
    var clean_bf16 = cast_tensor(clean_f32, STDtype.BF16, ctx)
    return krea2_patchify[COND_LH, COND_LW](clean_bf16, ctx)  # [1,CONDLEN,64]


def _center_crop_to_aspect(img: Image, target_w: Int, target_h: Int) raises -> Image:
    """Byte-for-byte the stager's `_center_crop_to_aspect`
    (scripts/krea2_omini_stage_edit.py:153-165): crop the LONG axis so the aspect
    matches the bucket, keeping the centre. Same integer rounding (round-half-up
    on positive values) so the crop box agrees with the Python one."""
    var w = img.width
    var h = img.height
    var target_ar = Float64(target_w) / Float64(target_h)
    var ar = Float64(w) / Float64(h)
    if ar > target_ar:
        var new_w = Int(Float64(h) * target_ar + 0.5)
        var left = (w - new_w) // 2
        return crop(img, left, 0, new_w, h)
    var new_h = Int(Float64(w) / target_ar + 0.5)
    var top = (h - new_h) // 2
    return crop(img, 0, top, w, new_h)


# MEASURED, NOT ASSUMED (2026-07-30, index 0 = dataset stem 1-01, 1696x2624):
# the original code here was `resize_bilinear(img, 512, 512)` — an ASPECT SQUASH.
# The trainer's condition came from the stager, which CENTER-CROPS to the bucket
# aspect first (LANCZOS). On a 1696x2624 portrait the two disagree by RMSE 54.5
# in pixel space, and the renders they produce disagree by RMSE 57.9: the
# squashed-condition render sits 53.5 from the ground truth while the
# correctly-cropped one sits 13.3. So this was not cosmetic — it fed the adapter
# a geometry it never saw in training and the whole --cond-image path was broken
# for any non-square source.
# Filter choice: the vendor resampler has no LANCZOS, and its `resize_bilinear`
# is PIL-equivalent ANTIALIASED triangle (transform.mojo:90-99 scales the support
# by the downscale factor), so aliasing was never the issue — the crop was. Of
# what is available, cubic is closest to the stager's LANCZOS: measured against a
# PIL LANCZOS reference on this same crop, BICUBIC is 0.61 RMSE away and BILINEAR
# 1.33 — both negligible next to the 54.5 the crop was worth, cubic simply wins.
def _load_signed_image(path: String, ctx: DeviceContext) raises -> Tensor:
    var img = decode_image_any(path)
    var cropped = _center_crop_to_aspect(img, COND_WIDTH, COND_HEIGHT)
    var resized = resize_bicubic(cropped, COND_WIDTH, COND_HEIGHT)
    var host = image_to_signed_nchw(resized)
    print("[krea2-omini-edit] condition", path, "(", img.width, "x", img.height,
          ") -> centre-crop", cropped.width, "x", cropped.height,
          "-> resize", COND_WIDTH, "x", COND_HEIGHT,
          "(the stager's chain: crop-to-aspect then resample)")
    return Tensor.from_host(
        host, [1, 3, COND_HEIGHT, COND_WIDTH], STDtype.F32, ctx
    )


def _black_signed_image(ctx: DeviceContext) raises -> Tensor:
    """OminiControl's empty condition is `Image.new("RGB", size, (0,0,0))`
    (flux_omini.py:125). In the [-1,1] convention image_to_signed_nchw uses
    (v = px/127.5 - 1), pixel 0 maps to -1.0 — NOT to a zero tensor, and NOT to
    a zero latent: it still goes through the VAE."""
    var host = List[Float32](capacity=3 * COND_HEIGHT * COND_WIDTH)
    for _ in range(3 * COND_HEIGHT * COND_WIDTH):
        host.append(Float32(-1.0))
    return Tensor.from_host(
        host^, [1, 3, COND_HEIGHT, COND_WIDTH], STDtype.F32, ctx
    )


def main() raises:
    comptime assert HEIGHT * COND_WIDTH == WIDTH * COND_HEIGHT
    comptime assert HEIGHT % COND_HEIGHT == 0
    comptime assert WIDTH % COND_WIDTH == 0
    comptime assert HEIGHT // COND_HEIGHT == WIDTH // COND_WIDTH
    var ctx = DeviceContext()
    var raw = argv()
    var args = List[String]()
    for i in range(len(raw)):
        args.append(String(raw[i]))

    var cache_path = _arg_str(args, String("--from-cache"), String(""))
    var index = _arg_int(args, String("--index"), 0)
    var cond_image = _arg_str(args, String("--cond-image"), String(""))
    var ctx_pos_bin = _arg_str(args, String("--ctx-pos"), String(""))
    var ctx_neg_bin = _arg_str(args, String("--ctx-neg"), String(""))
    var out_png = _arg_str(args, String("--out"), String("output/krea2_omini_edit.png"))
    var lora_path = _arg_str(args, String("--lora"), String(""))
    var lora_mult = _arg_f32(args, String("--lora-mult"), Float32(1.0))
    var no_lora = _arg_has(args, String("--no-lora"))
    var steps = _arg_int(args, String("--steps"), STEPS_DEFAULT)
    var seed = UInt64(_arg_int(args, String("--seed"), Int(SEED_DEFAULT)))
    var cfg_scale = _arg_f32(args, String("--cfg"), CFG_DEFAULT)
    var condition_scale = _arg_f32(args, String("--condition-scale"), Float32(1.0))
    var image_guidance = _arg_f32(args, String("--image-guidance"), Float32(1.0))
    var use_stream = _arg_has(args, String("--stream"))
    var use_int8 = _arg_has(args, String("--int8"))
    var i8_resident_blocks = _arg_int(
        args, String("--i8-resident-blocks"), Int(0)
    )
    var trainer_text = _arg_has(args, String("--trainer-text"))
    var latent_only = _arg_has(args, String("--latent-only"))
    var ckpt = _arg_str(args, String("--checkpoint"), String(KREA2_RAW))
    var sidecar = _arg_str(args, String("--squareq-sidecar"), String(SQUAREQ_SIDECAR))

    var from_cache = cache_path != String("")
    if not from_cache and cond_image == String(""):
        raise Error(
            "usage: krea2_omini_edit_infer"
            " (--from-cache <cache.safetensors> --index N"
            "  | --cond-image <img> --ctx-pos <pos.bin> --ctx-neg <neg.bin>)"
            " --out <png> [--lora <p>] [--no-lora] [--steps N] [--seed N]"
            " [--cfg F] [--condition-scale F] [--image-guidance F]"
            " [--lora-mult F] [--stream|--int8]"
            " [--i8-resident-blocks N] [--trainer-text] [--latent-only]"
        )
    if use_stream and use_int8:
        raise Error(
            "krea2_omini_edit_infer: --stream and --int8 are mutually exclusive"
        )
    if not from_cache and ctx_pos_bin == String(""):
        raise Error(
            "krea2_omini_edit_infer: --cond-image needs --ctx-pos (run"
            " serenitymojo/pipeline/krea2_encode_cli.mojo first — the Qwen3-VL"
            " text encoder must not co-reside with the DiT)"
        )
    if lora_path == String("") and not no_lora:
        raise Error(
            "krea2_omini_edit_infer: pass --lora <checkpoint.safetensors>, or"
            " --no-lora to deliberately render the FROZEN BASE. Refusing to"
            " guess: a cond-row-only edit LoRA is exactly the thing that goes"
            " silently missing, so the base render must be an explicit request."
        )

    print("[krea2-omini-edit] shape", HEIGHT, "x", WIDTH,
          " LTMAX=", LTMAX, " IMGLEN=", IMGLEN,
          " condition=", COND_HEIGHT, "x", COND_WIDTH,
          " CONDLEN=", CONDLEN, " pos_scale=", COND_POS_SCALE,
          " LFULL_EDIT=", LFULL_EDIT)
    print("[krea2-omini-edit] steps=", steps, " seed=", seed, " cfg=", cfg_scale,
          " condition_scale=", condition_scale,
          " image_guidance_scale=", image_guidance,
          " lora=", String("NONE (frozen base)") if no_lora else lora_path)
    if image_guidance != Float32(1.0):
        print(
            "[krea2-omini-edit] NOTE: this adapter was trained with"
            " caption_dropout 0.0 and NO condition dropout, while OminiControl"
            " drops text and image 0.1 each. The black-image branch is therefore"
            " OFF-DISTRIBUTION here and image-CFG may underperform or hurt.",
        )

    # ── 1) conditioning: cached (round-trip) or user image + encoded prompt ───
    var cond_tokens: Tensor
    var cond_black = Optional[Tensor](None)
    var pos_cond: Krea2InlineCond
    var neg_cond: Krea2InlineCond
    var _t = Int(perf_counter_ns())

    if from_cache:
        print("[krea2-omini-edit] FROM CACHE", cache_path, " index=", index)
        var cache = KreaTrainCache.open(cache_path)
        if index < 0 or index >= cache.len():
            raise Error(
                String("krea2_omini_edit_infer: --index ") + String(index)
                + " out of range [0, " + String(cache.len()) + ")"
            )
        var smp = cache.sample_padded_edit[LH, LW, LTMAX](index, ctx)
        if smp.cond_len != CONDLEN:
            raise Error(
                String("krea2_omini_edit_infer: cache condition length ")
                + String(smp.cond_len) + " != build CONDLEN " + String(CONDLEN)
            )
        if (
            smp.pos_delta_h != COND_DELTA_H or smp.pos_delta_w != COND_DELTA_W
            or smp.pos_scale != COND_POS_SCALE
        ):
            raise Error(
                "krea2_omini_edit_infer: cached condition position"
                " delta/scale disagree with this binary's EDIT constants"
            )
        cond_tokens = smp.cond_img[].clone(ctx)
        # `base` already carries the LTMAX-padded context AND the SOURCE-order
        # extended pos table the trainer used — reuse them verbatim.
        pos_cond = Krea2InlineCond(
            TArc(smp.base.context[].clone(ctx)),
            TArc(smp.base.pos[].clone(ctx)),
            smp.base.text_len,
        )
        var unc = cache.uncond[LH, LW](ctx)
        neg_cond = inline_cond_from_context_edit[LH, LW, LTMAX, CONDLEN](
            unc.context[].clone(ctx),
            COND_DELTA_H, COND_DELTA_W, COND_POS_SCALE, ctx,
        )
        print("[krea2-omini-edit] cache sample", index, " LT=", pos_cond.text_len,
              " LT(uncond)=", neg_cond.text_len)
        if image_guidance != Float32(1.0):
            # Even the cache path needs the VAE for the BLACK condition — there
            # is no cached "empty condition" latent.
            var enc = QwenImageVaeEncoder[COND_HEIGHT, COND_WIDTH].load(
                KREA2_VAE_ENC_FILE, ctx
            )
            cond_black = Optional[Tensor](
                _encode_condition(enc, _black_signed_image(ctx), ctx)
            )
    else:
        var enc = QwenImageVaeEncoder[COND_HEIGHT, COND_WIDTH].load(
            KREA2_VAE_ENC_FILE, ctx
        )
        cond_tokens = _encode_condition(
            enc, _load_signed_image(cond_image, ctx), ctx
        )
        if image_guidance != Float32(1.0):
            cond_black = Optional[Tensor](
                _encode_condition(enc, _black_signed_image(ctx), ctx)
            )
        var cpos = load_tensor_bin(ctx_pos_bin, ctx)
        pos_cond = inline_cond_from_context_edit_grid[
            LH, LW, COND_LH, COND_LW, LTMAX, CONDLEN
        ](
            cpos^, COND_DELTA_H, COND_DELTA_W, COND_POS_SCALE, ctx,
        )
        var neg_path = ctx_neg_bin if ctx_neg_bin != String("") else ctx_pos_bin
        var cneg = load_tensor_bin(neg_path, ctx)
        neg_cond = inline_cond_from_context_edit_grid[
            LH, LW, COND_LH, COND_LW, LTMAX, CONDLEN
        ](
            cneg^, COND_DELTA_H, COND_DELTA_W, COND_POS_SCALE, ctx,
        )
        print("[krea2-omini-edit] LT(pos)=", pos_cond.text_len,
              " LT(neg)=", neg_cond.text_len)
    ctx.synchronize()
    print("[krea2-omini-edit] PHASE conditioning =",
          Float64(Int(perf_counter_ns()) - _t) / 1e9, "s")
    _t = Int(perf_counter_ns())

    var cts = cond_tokens.shape()
    if len(cts) != 3 or cts[0] != 1 or cts[1] != CONDLEN or cts[2] != 64:
        raise Error(
            "krea2_omini_edit_infer: condition tokens must be [1, CONDLEN, 64]"
        )

    # ── 2) frozen weights: shared conditioning set + final layer + base ───────
    var st = ShardedSafeTensors.open(ckpt)
    var cond_w = load_krea2_resident_cond(st, String(KREA2_RAW_KEY_PREFIX), ctx)
    var fin = Krea2StreamFinal.load(st, String(KREA2_RAW_KEY_PREFIX), ctx)

    # DEFAULT = squareq_w4 resident, because that is the base numerics the
    # original C6 edit run trained against. MagicBrush uses the promoted
    # tensorwise INT8 W8A8 path, selected explicitly with --int8. Rendering a
    # LoRA against different base numerics is a silent A/B change, so every
    # non-default base is opt-in and loud.
    var resident_sq = Optional[Krea2ResidentSquareq](None)
    var resident_i8 = Optional[Krea2ResidentInt8](None)
    var host_i8 = Optional[Krea2HostInt8Inf](None)
    if use_stream:
        print("[krea2-omini-edit] BASE = per-block bf16 DISK STREAM (opt-in;"
              " use only for a LoRA trained against bf16 base numerics).")
    elif use_int8:
        if i8_resident_blocks < 0 or i8_resident_blocks > NBLOCKS:
            raise Error(
                "krea2_omini_edit_infer: --i8-resident-blocks must be in [0, 28]"
            )
        print("[krea2-omini-edit] BASE = tensorwise INT8 W8A8: resident",
              i8_resident_blocks, "/", NBLOCKS,
              "blocks + pinned-host remainder (training numerics) ...")
        if i8_resident_blocks > 0:
            resident_i8 = Optional[Krea2ResidentInt8](
                build_krea2_resident_int8(
                    st, String(KREA2_RAW_KEY_PREFIX), NBLOCKS,
                    i8_resident_blocks, ctx,
                )
            )
        if i8_resident_blocks < NBLOCKS:
            host_i8 = Optional[Krea2HostInt8Inf](
                build_krea2_host_int8_inf(
                    st, String(KREA2_RAW_KEY_PREFIX), NBLOCKS,
                    i8_resident_blocks, ctx,
                )
            )
        print("[krea2-omini-edit] INT8 W8A8 base: DONE.")
    else:
        print("[krea2-omini-edit] BASE = squareq_w4 resident from", sidecar,
              "(original C6 training numerics) ...")
        var sq_sc = ShardedSafeTensors.open(sidecar)
        resident_sq = Optional[Krea2ResidentSquareq](
            build_krea2_resident_squareq(
                sq_sc, String(KREA2_RAW_KEY_PREFIX), NBLOCKS, NBLOCKS, ctx
            )
        )
        print("[krea2-omini-edit] squareq_w4 resident: DONE (", NBLOCKS, "blocks).")
    ctx.synchronize()
    print("[krea2-omini-edit] PHASE base =",
          Float64(Int(perf_counter_ns()) - _t) / 1e9, "s")
    _t = Int(perf_counter_ns())

    # ── 3) the cond-row LoRA overlay (or an explicit identity overlay) ────────
    var lora = empty_krea2_stack_lora()
    var first_lora = Optional[LoraAdapterDevice](None)
    if not no_lora:
        lora = load_krea2_stack_lora(lora_path, lora_mult, ctx)
        first_lora = Optional[LoraAdapterDevice](
            load_krea2_omini_first_lora(lora_path, lora_mult, ctx)
        )
    else:
        print("[krea2-omini-edit] --no-lora: identity overlay, FROZEN BASE"
              " render (the A/B partner for a LoRA render).")

    # ── 4) denoise. CONDL=CONDLEN selects the EDIT arm of the shared sampler. ─
    var latent = krea2_sample_latent[LH, LW, LTMAX, LFULL_EDIT, CONDLEN](
        st, String(KREA2_RAW_KEY_PREFIX), cond_w, fin, lora,
        pos_cond, neg_cond, steps, cfg_scale, seed,
        Optional[Krea2ResidentFp8](None), ctx,
        use_fixed_mu_1_15=False,
        progress_fd=Int32(-1),
        cond_tokens=Optional[Tensor](cond_tokens^),
        first_lora=first_lora^,
        cond_tokens_black=cond_black^,
        image_guidance_scale=image_guidance,
        condition_scale=condition_scale,
        resident_sq=resident_sq^,
        resident_i8=resident_i8^,
        host_i8=host_i8^,
        use_refiner_mask=not trainer_text,
    )
    ctx.synchronize()
    print("[krea2-omini-edit] PHASE denoise =",
          Float64(Int(perf_counter_ns()) - _t) / 1e9, "s")

    # ── 5) decode, or persist for a fresh-process decode ─────────────────────
    # At 1024 EDIT length (IMG 4096 + COND 4096), denoising fits the 16GB card
    # but the in-process Qwen VAE decoder does not fit beside the DiT allocator.
    # --latent-only is the production-safe process boundary: save raw F32 latent
    # bytes, exit (releasing every DiT allocation), then krea2_decode_latent loads
    # and decodes them in a virgin GPU process. The default inline path remains
    # unchanged for the device-verified 512 build.
    if latent_only:
        var lat_bin = out_png + String(".lat.bin")
        save_tensor_bin(latent, lat_bin, ctx)
        print("[krea2-omini-edit] LATENT-ONLY wrote", lat_bin)
    else:
        krea2_decode_latent_to_png[LH, LW](
            latent, String(KREA2_VAE_DIR), out_png, ctx
        )
        print("[krea2-omini-edit] wrote", out_png)
