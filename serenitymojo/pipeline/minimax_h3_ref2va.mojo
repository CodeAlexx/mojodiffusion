# serenitymojo/pipeline/minimax_h3_ref2va.mojo — MiniMax-H3 ref2va
# (omni-reference video+audio), pure Mojo. See PARTIAL MODE below for the
# full-plumbing device chain this file now also runs.
#
# Modeled on `serenitymojo/pipeline/minimax_h3_t2va.mojo`, which is the house
# pattern for an H3 device pipeline and the file that produced valid video:
# comptime geometry via `get_defined_int`, preflight BEFORE `DeviceContext()`,
# named loud failures, request/result printing.
#
# ── SCOPE OF THIS FILE ───────────────────────────────────────────────────────
# This is the entry point, the PLAN, and — WITH `--partial` (see below) — the
# full device denoise chain, run against STUBBED conditioning and STUBBED (or
# injected) condition rows. WITHOUT `--partial` it now runs the REAL reference
# encode chain — ffmpeg decode -> 24 fps resample -> canvas LANCZOS resize ->
# pixel norm -> the GATED video-VAE seam — and raises at the NEXT unbuilt
# stage (condition-row packing + the Qwen3-VL conditioning), so a plain
# invocation now touches the GPU up to that seam and can still never be
# mistaken for a full generation.
#
# ── PIPELINE STAGES, AND WHAT EXISTS ─────────────────────────────────────────
#   1. media-in           BUILT  `pipeline/minimax_h3_media_in.mojo` (unit A)
#                                probe: pipeline/parity/minimax_h3_media_in_probe
#   2. ref-encode         BUILT  references -> 24 fps resample -> canvas
#                                LANCZOS resize (`pipeline/minimax_h3_
#                                ref_frames.mojo`, gated bit-exact) -> pixel
#                                norm -> video VAE (tiled) -> sample posterior
#                                (seed 42) — runs FOR REAL without `--partial`
#                                through the gated seam, then raises at the
#                                condition-row packing / conditioning seam.
#                                The fp16 round -> normalize -> patchify tail
#                                and the audio branch are built and gated but
#                                not yet consumed there. WITH `--partial`,
#                                this stage is STUBBED (Stage A below).
#   3. ref-pack           BUILT  `models/dit/minimax_h3_ref_geometry.mojo`
#                                (unit B) over the gated
#                                `models/minimax_h3/packing_ref2va.mojo`
#                                probe: models/dit/parity/minimax_h3_ref_geometry_probe
#   4. conditioning       SEAM   ref2va presentation (labels + vision blocks)
#                                needs the Qwen3-VL vision tower's per-reference
#                                token counts; `models/text_encoder/
#                                minimax_h3_qwen3vl_vision.mojo::
#                                minimax_h3_vision_forward_seam` is geometry-
#                                complete but has no weights (text_encoder
#                                shard 14 of 14 is absent). STUBBED under
#                                `--partial` (Stage B below) — the LAYOUT
#                                (which rows are vision-block rows) is real,
#                                built from the declared references; only the
#                                VALUES are fixed-seed random.
#   5. ref-denoise        BUILT  four-timestep rows, real streamed blocks,
#                                real Euler steps on target rows only — see
#                                PARTIAL MODE / Stage C below. The row-
#                                timestep half was already BUILT and gated
#                                (unit B, `minimax_h3_ref2va_row_timesteps`);
#                                this pass wires the GLOBAL per-step 4-row
#                                modcache addressing around it.
#   6. prompt             BUILT  `pipeline/minimax_h3_ref_prompt.mojo` (unit C)
#                                probe: pipeline/parity/minimax_h3_ref_prompt_probe
#   7. decode             REUSE  identical to t2va's decode tail — out of this
#                                unit's scope; latents are saved so a decode
#                                pass can be run separately (see Stage C's
#                                final raise).
#
# ── CHECKPOINT: Ref2VA, NOT FL2VA ────────────────────────────────────────────
# A SEPARATE 134 GiB checkpoint whose weight VALUES differ. Measured, on disk:
#   * `transformer/config.json`  BYTE-IDENTICAL to FL2VA — same DiT code, so
#     every block/adaLN/final-layer module is reused unchanged.
#   * `video_vae/config.json`    BYTE-IDENTICAL to FL2VA, `latents_mean` and
#     `latents_std` included. So the t2va pipeline's hardcoded video latent
#     normalization constants are correct for Ref2VA too and must NOT be
#     re-derived. Same for `audio_vae/config.json` and
#     `processor/preprocessor_config.json`, both byte-identical.
#   * `model_index.json`         differs ONLY in the `partition`/`tasks` labels
#     (`ref2va` vs `t2va`/`fl2va`).
# The upshot: no config value changes for ref2va. Only the weight files and the
# layout do.
#
# ── PARTIAL MODE (`--partial`) ───────────────────────────────────────────────
# Mirrors t2va's own PARTIAL MODE in spirit (fixed-seed stand-ins where the
# real inputs are not buildable yet, an unmissable banner, a real 50-block
# streamed denoise underneath), adapted to ref2va's two genuinely unreachable
# stages — the reference video-VAE encode (Stage A) and the Qwen3-VL vision
# tower (Stage B) — rather than to a partial transformer-shard count (ref2va
# always streams the full `config.num_layers` blocks; there is no `max_blocks`
# knob here).
#
#   Stage A — CONDITION ROWS. `--condition-rows=PATH` injects a precomputed
#     safetensors file (`video_condition_rows`/`audio_condition_rows`,
#     STRICTLY shape/dtype validated against this request's plan — the
#     contract the real GPU ref-encode will eventually feed); otherwise
#     fixed-seed random. Either way the VIDEO half is then noise-mixed at the
#     constant 0.999 through the GATED `minimax_h3_mix_condition_rows`
#     (models/vae/minimax_h3_ref_encode.mojo, encoders.py:612-630); the AUDIO
#     half is NEVER mixed (that asymmetry is the vendor's own — audio
#     condition rows are the VAE posterior MODE, never sampled, never
#     noise-mixed, yet pinned at row timestep 1.0 by unit B's own math).
#
#   Stage B — STUB CONDITIONING. The LAYOUT is real: which text-region rows
#     are a reference's vision-block rows (tag 0, video) versus label/prompt
#     rows (tag 1, text) is built from the declared references
#     (`_minimax_h3_ref2va_stub_conditioning_tags`, called from `main()`
#     before the plan, host-only). Only the VALUES are fixed-seed random
#     `[1, H3_TEXT_TOKENS, text_dim]` BF16, drawn in the device stage.
#
#   Stage C — DENOISE. Real: packed layout, RoPE tables, dual schedule, a
#     4-ROW-PER-STEP modulation cache (video, audio, condition-video,
#     condition-audio — t2va's own layout has 2), `config.num_layers` real
#     streamed blocks, Euler steps restricted to TARGET rows (condition rows
#     are PINNED — never stepped, at any step), latents saved to
#     `out_dir/latents.safetensors`.
#
#   Stage D — BANNER. Printed before the loop and written into
#     `out_dir/result.json`: PARTIAL MODE; conditioner STUBBED; condition
#     rows INJECTED or SYNTHETIC; THIS IS NOT A VALID MINIMAX-H3 REF2VA
#     GENERATION.
#
# `H3_REF_SEQ_LEN` (comptime, like `H3_TEXT_TOKENS`): a ref2va request's
# packed sequence length depends on the references, and
# `minimax_h3_block_forward`'s self-attention instantiates its row count `S`
# at COMPTIME (models/dit/minimax_h3_dit.mojo) — so unlike t2va's fixed
# `SEQ_LEN`, this cannot be read off the request at runtime; it must be
# rebuilt with `-D H3_REF_SEQ_LEN=<plan.sequence_length()>` per request shape,
# checked at runtime with a rebuild-hint error otherwise.
#
# EXPECTED TO FAIL AT A SEAM once actually run on GPU (incomplete checkpoint,
# an approximation in Stage B's token count, etc.) — this file's job is that
# it COMPILES and the WIRING is right, not that a GPU run succeeds today.
#
# argv: <prompt_file> <out_dir> [steps=30] [seed=0] [--partial]
#       [--condition-rows=PATH] [reference specs...]
#   A reference spec is `kind:path`, e.g. `video:/clips/ref.mp4`,
#   `audio:/clips/voice.wav`, `image:/stills/subject.png`. ORDER MATTERS — it
#   labels the references in the prompt and advances the shared rotary clock
#   (packing_ref2va.py:25-29), so the specs are consumed in the order given and
#   are never sorted. `--partial` and `--condition-rows=PATH` may appear
#   anywhere in argv and are stripped before the positional arguments are
#   parsed (reference specs already occupy every position from argv[5] on).

from std.collections import Dict, List
from std.sys import argv
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import reshape, slice, concat
from serenitymojo.serve.product_manifest import json_escape, json_bool, write_text_file

from serenitymojo.models.minimax_h3.packing import MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG
from serenitymojo.models.minimax_h3.packing_ref2va import (
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_MAX_REFERENCES,
    MINIMAX_H3_REF_AUDIO,
    MINIMAX_H3_REF_IMAGE,
    MINIMAX_H3_REF_VIDEO,
    MiniMaxH3PreparedReference,
    minimax_h3_trim_reference_num_frames,
)
from serenitymojo.models.dit.minimax_h3_ref_geometry import (
    MiniMaxH3ReferenceMedia,
    MiniMaxH3Ref2VAPlan,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    minimax_h3_build_ref2va_plan,
    minimax_h3_ref2va_row_timesteps,
    minimax_h3_reference_audio_latents,
    minimax_h3_resolve_references,
    minimax_h3_target_audio_latents,
)
from serenitymojo.pipeline.minimax_h3_media_in import (
    MINIMAX_H3_MEDIA_SAMPLE_RATE,
    minimax_h3_ffmpeg_extract_rgb,
    minimax_h3_ffprobe_video_geometry,
    minimax_h3_has_audio_stream,
    minimax_h3_read_wav,
)
from serenitymojo.models.vae.minimax_h3_ref_encode import (
    MINIMAX_H3_VIDEO_LATENT_CHANNELS,
    minimax_h3_encode_reference_visual_seam,
    minimax_h3_pixel_normalize_frames,
    minimax_h3_mix_condition_rows,
)
from serenitymojo.models.vae.minimax_h3_video_encoder_device import (
    MiniMaxH3VideoEncoderDevice,
    minimax_h3_video_released_encoder_config,
)
from serenitymojo.pipeline.minimax_h3_ref_frames import (
    minimax_h3_prepare_reference_frames,
    minimax_h3_resample_reference_frames,
)
from serenitymojo.pipeline.minimax_h3_video_vae_spatial_tiling import (
    minimax_h3_video_released_tiling_config,
)
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    MINIMAX_H3_HEADS,
    MINIMAX_H3_HEAD_DIM,
    minimax_h3_released_config,
    minimax_h3_adaln_rows,
    minimax_h3_block_tensor_names,
    minimax_h3_expected_shape,
    minimax_h3_block_forward,
)
from serenitymojo.models.dit.minimax_h3_loader_device import (
    minimax_h3_load_block_device,
    minimax_h3_load_qkv_device,
    minimax_h3_load_fc1_device,
)
from serenitymojo.models.dit.minimax_h3_modcache import (
    minimax_h3_check_modcache_weights,
    minimax_h3_build_modulation_cache,
)
from serenitymojo.models.dit.minimax_h3_rope import build_minimax_h3_rope_tables
from serenitymojo.models.dit.minimax_h3_frontend import (
    minimax_h3_frontend_embed,
    minimax_h3_final_layer,
    minimax_h3_timestep_embedding,
)
from serenitymojo.models.dit.minimax_h3_sampling import MiniMaxH3DualSchedule
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import MiniMaxH3VisionGrid


# ── Checkpoint layout (Ref2VA, not FL2VA) ────────────────────────────────────
comptime H3_ROOT = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/Ref2VA"
comptime TRANSFORMER_DIR = H3_ROOT + "/transformer"
comptime TEXT_ENCODER_DIR = H3_ROOT + "/text_encoder"
comptime PROCESSOR_DIR = H3_ROOT + "/processor"
comptime TOKENIZER_DIR = H3_ROOT + "/tokenizer"
comptime AUDIO_VAE_PATH = H3_ROOT + "/audio_vae/model.safetensors"
comptime VIDEO_VAE_DIR = H3_ROOT + "/video_vae/source"
comptime AUDIO_SAMPLE_RATE = MINIMAX_H3_MEDIA_SAMPLE_RATE


# ── Geometry (COMPTIME — rebuild to change, exactly as t2va does; see that
# file's "FIXED PROMPT LENGTH" note for why the text budget is comptime too:
# H3 runs ONE packed self-attention document with no mask, so a padded text
# region would pollute every row's attention, not just its own). ─────────────
comptime HEIGHT = get_defined_int["H3_HEIGHT", 480]()
comptime WIDTH = get_defined_int["H3_WIDTH", 832]()
comptime FRAMES = get_defined_int["H3_FRAMES", 22]()
comptime TEXT_TOKENS = get_defined_int["H3_TEXT_TOKENS", 32]()

comptime LATENT_H = HEIGHT // 16
comptime LATENT_W = WIDTH // 16
comptime PATCH_H = 2
comptime PATCH_W = 2
comptime ROWS_PER_FRAME = (LATENT_H // PATCH_H) * (LATENT_W // PATCH_W)
comptime NUM_LATENT_FRAMES = (FRAMES - 5) // 17 * 5 + 2
comptime NUM_VIDEO_ROWS = NUM_LATENT_FRAMES * ROWS_PER_FRAME

comptime DEFAULT_STEPS = 30
comptime DEFAULT_SEED = 0

# A ref2va request's FULL packed sequence length (text + reference blocks +
# target audio + target video) depends on the references, so — unlike
# t2va's fixed `SEQ_LEN` — it cannot be a fixed formula of HEIGHT/WIDTH/FRAMES
# alone. `minimax_h3_block_forward`'s self-attention instantiates its row
# count `S` at COMPTIME regardless (models/dit/minimax_h3_dit.mojo), so this
# is the same "rebuild to change" discipline as H3_TEXT_TOKENS, just resolved
# per-request instead of once: run with `--partial`, read the printed
# `plan.sequence_length()`, rebuild with `-D H3_REF_SEQ_LEN=<that number>`.
# Default 1 (not 0) is deliberately a valid, instantiable comptime value —
# `_minimax_h3_ref2va_generate` checks it against `plan.sequence_length()` at
# runtime and raises a rebuild-hint error on any mismatch, including the
# never-set default.
comptime H3_REF_SEQ_LEN = get_defined_int["H3_REF_SEQ_LEN", 1]()

# The real 50-block streamed stack (`minimax_h3_block_forward`) calls
# `ops/attention_flash.sdpa_flash_infer_fwd`, which needs the cuDNN v9 SDPA
# shim linked in (`flame_cudnn_sdpa_bf16` — see minimax_h3_t2va.mojo's own
# "LINKER" header note; confirmed here too: a bare build against this file
# with the block loop unconditional fails at LINK, not compile, with
# `undefined reference to flame_cudnn_sdpa_bf16`). This file's own build
# command (this repo's acceptance gate for this file) does not pass the
# extra `-Xlinker` flags t2va's own production builds always do
# (output/logs/h3_shots_orchestrator.sh, pixi.toml's build-ltx2-request and
# siblings), so the real block loop is gated behind this comptime flag,
# DEFAULT OFF, exactly like t2va's own `H3_VAE_TEMPORAL`/`H3_VAE_STREAM_
# DECODE` optional-linkage `@parameter if` features. With it off, `--partial`
# still runs Stage A (condition rows), Stage B (stub embeds), the packed
# layout, RoPE tables, the dual schedule and the 4-row modulation cache for
# REAL, then raises a named, rebuild-hint error at the first block-forward
# call — an honest seam, not a silently-skipped stage. Rebuild with
# `-D H3_REF2VA_REAL_BLOCKS=1` PLUS the linker flags below (identical to
# t2va's) to run the real denoise:
#   -Xlinker -lm -Xlinker -lcuda -Xlinker -Lserenitymojo/ops/cshim/lib
#   -Xlinker -lserenity_cudnn_sdpa -Xlinker
#   -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn
comptime H3_REF2VA_REAL_BLOCKS = get_defined_int["H3_REF2VA_REAL_BLOCKS", 0]()


@fieldwise_init
struct MiniMaxH3ReferenceSpec(Copyable, Movable):
    """One `kind:path` reference from the command line, in request order."""

    var kind: Int
    var path: String

    def kind_name(self) -> String:
        if self.kind == MINIMAX_H3_REF_IMAGE:
            return String("image")
        if self.kind == MINIMAX_H3_REF_VIDEO:
            return String("video")
        return String("audio")


def _parse_reference_spec(spec: String) raises -> MiniMaxH3ReferenceSpec:
    var at = spec.find(String(":"))
    if at <= 0:
        raise Error(
            String("minimax_h3_ref2va: bad reference spec '") + spec
            + "' — expected kind:path, e.g. video:/clips/ref.mp4"
        )
    var kind_text = String(spec[byte=0 : at])
    var path = String(spec[byte = at + 1 :])
    if path == String(""):
        raise Error(
            String("minimax_h3_ref2va: reference spec '") + spec + "' has no path"
        )
    if kind_text == String("image"):
        return MiniMaxH3ReferenceSpec(MINIMAX_H3_REF_IMAGE, path^)
    if kind_text == String("video"):
        return MiniMaxH3ReferenceSpec(MINIMAX_H3_REF_VIDEO, path^)
    if kind_text == String("audio"):
        return MiniMaxH3ReferenceSpec(MINIMAX_H3_REF_AUDIO, path^)
    raise Error(
        String("minimax_h3_ref2va: unknown reference kind '") + kind_text
        + "' — expected image, video or audio"
    )


def _validate_reference_counts(
    specs: List[MiniMaxH3ReferenceSpec]
) raises:
    """The documented per-request limits (packing_ref2va.py:83-86), plus the
    rule that an audio reference never stands on its own (:257-262)."""
    if len(specs) == 0:
        raise Error(
            "minimax_h3_ref2va: ref2va needs at least one reference — pass"
            " reference specs like video:/clips/ref.mp4 (use the t2va pipeline"
            " for a reference-free request)"
        )
    if len(specs) > MINIMAX_H3_MAX_REFERENCES:
        raise Error(
            String("minimax_h3_ref2va: ") + String(len(specs))
            + " references exceeds the documented limit of "
            + String(MINIMAX_H3_MAX_REFERENCES)
        )
    var images = 0
    var videos = 0
    var audios = 0
    for i in range(len(specs)):
        if specs[i].kind == MINIMAX_H3_REF_IMAGE:
            images += 1
        elif specs[i].kind == MINIMAX_H3_REF_VIDEO:
            videos += 1
        else:
            audios += 1
    if images > MINIMAX_H3_MAX_REFERENCE_IMAGES:
        raise Error(
            String("minimax_h3_ref2va: ") + String(images)
            + " image references exceeds the limit of "
            + String(MINIMAX_H3_MAX_REFERENCE_IMAGES)
        )
    if videos > MINIMAX_H3_MAX_REFERENCE_VIDEOS:
        raise Error(
            String("minimax_h3_ref2va: ") + String(videos)
            + " video references exceeds the limit of "
            + String(MINIMAX_H3_MAX_REFERENCE_VIDEOS)
        )
    if audios > MINIMAX_H3_MAX_REFERENCE_AUDIOS:
        raise Error(
            String("minimax_h3_ref2va: ") + String(audios)
            + " audio references exceeds the limit of "
            + String(MINIMAX_H3_MAX_REFERENCE_AUDIOS)
        )
    if images == 0 and videos == 0:
        raise Error(
            "minimax_h3_ref2va: an audio reference has to be paired with at"
            " least one image or video reference (packing_ref2va.py:257-262)"
        )


def _preflight_geometry() raises:
    if (FRAMES - 5) % 17 != 0:
        raise Error(
            String("minimax_h3_ref2va: FRAMES=") + String(FRAMES)
            + " is not of the form 17n+5 (the video VAE's chunk size); edit"
            " H3_FRAMES and rebuild — e.g. 5, 22, 39, 56, ..."
        )
    if HEIGHT <= 0 or WIDTH <= 0 or HEIGHT % 32 != 0 or WIDTH % 32 != 0:
        raise Error(
            String("minimax_h3_ref2va: H3_HEIGHT/H3_WIDTH must be positive")
            + " multiples of 32; got " + String(WIDTH) + "x" + String(HEIGHT)
        )
    if TEXT_TOKENS <= 0:
        raise Error("minimax_h3_ref2va: H3_TEXT_TOKENS must be positive")
    if NUM_LATENT_FRAMES <= 0:
        raise Error(
            "minimax_h3_ref2va: derived NUM_LATENT_FRAMES is non-positive —"
            " rebuild with a larger H3_FRAMES"
        )


def _preflight_checkpoint() raises:
    """Report what is on disk. Header-only opens: no tensor bytes are read and
    no DeviceContext is needed.

    NON-FATAL by design at this stage. The Ref2VA checkpoint was still
    downloading when this file was written, and the point of the skeleton is to
    show the PLAN even against an incomplete checkpoint. The seam below raises
    regardless, so an incomplete checkpoint can never be mistaken for a run."""
    print("  preflight: transformer", String(TRANSFORMER_DIR))
    try:
        var shards = ShardedSafeTensors.open(String(TRANSFORMER_DIR))
        print(
            "    ", shards.num_shards(), "shard(s),", shards.num_tensors(),
            "tensors",
        )
    except e:
        print("     INCOMPLETE:", e)

    print("  preflight: text_encoder", String(TEXT_ENCODER_DIR))
    try:
        var text_shards = ShardedSafeTensors.open(String(TEXT_ENCODER_DIR))
        print(
            "    ", text_shards.num_shards(), "shard(s),",
            text_shards.num_tensors(), "tensors",
        )
    except e:
        print("     INCOMPLETE:", e)

    print("  preflight: audio_vae", String(AUDIO_VAE_PATH))
    try:
        var audio = SafeTensors.open(String(AUDIO_VAE_PATH))
        print("     ", len(audio.names()), "tensors")
    except e:
        print("     INCOMPLETE:", e)

    print("  preflight: video_vae", String(VIDEO_VAE_DIR))
    print("  preflight: processor", String(PROCESSOR_DIR))
    print("  preflight: tokenizer", String(TOKENIZER_DIR))


def _probe_reference_media(
    spec: MiniMaxH3ReferenceSpec
) raises -> MiniMaxH3ReferenceMedia:
    """Read one reference's MEDIA geometry off disk (unit A). CPU only.

    An image reference's pixel size is NOT read here: no pure-Mojo PNG/JPEG
    header reader is wired into this path yet, and guessing it would corrupt the
    reference's own spatial grid. That is called out as a seam rather than
    defaulted."""
    if spec.kind == MINIMAX_H3_REF_AUDIO:
        var wave = minimax_h3_read_wav(spec.path)
        if wave.sample_rate != AUDIO_SAMPLE_RATE:
            raise Error(
                String("minimax_h3_ref2va: audio reference ") + spec.path
                + " is at " + String(wave.sample_rate) + " Hz; the audio VAE"
                " needs " + String(AUDIO_SAMPLE_RATE)
                + " Hz. Convert it first (minimax_h3_ffmpeg_decode_audio), and"
                " see that function's note on resampler parity."
            )
        return MiniMaxH3ReferenceMedia(
            MINIMAX_H3_REF_AUDIO, 0, 0, 0, wave.num_samples
        )

    if spec.kind == MINIMAX_H3_REF_IMAGE:
        raise Error(
            String("minimax_h3_ref2va: SEAM — image reference ") + spec.path
            + ": no image-header reader is wired into this pipeline, so the"
            " reference's own resolution cannot be read. Use a video or audio"
            " reference, or wire an image decoder into"
            " pipeline/minimax_h3_media_in.mojo first."
        )

    var geometry = minimax_h3_ffprobe_video_geometry(
        spec.path, spec.path + String(".h3probe")
    )
    var width = Int(geometry[0])
    var height = Int(geometry[1])
    var fps = geometry[2]
    # The frame count is not read here: the exact count only follows from the
    # rawvideo dump (see minimax_h3_ffmpeg_extract_rgb), and this stage is a
    # plan, not a decode. The 24 fps grid bound is what the layout needs.
    var approx_frames = FRAMES
    var num_audio_samples = 0
    if minimax_h3_has_audio_stream(spec.path):
        # A video reference conditions on its own soundtrack, whose sample count
        # is resolved at decode. Planned at the target duration, which is the
        # cap the vendor applies anyway.
        num_audio_samples = FRAMES * AUDIO_SAMPLE_RATE // 24
    print(
        "    ", spec.path, ":", width, "x", height, "@", fps, "fps, audio:",
        "yes" if num_audio_samples > 0 else "no",
    )
    return MiniMaxH3ReferenceMedia(
        MINIMAX_H3_REF_VIDEO, height, width, approx_frames, num_audio_samples
    )


def _minimax_h3_encode_references(
    specs: List[MiniMaxH3ReferenceSpec],
    references: List[MiniMaxH3PreparedReference],
) raises:
    """Reference encode, stage 2 — the CPU prep chain, then THE GATED SEAM.

    Runs for real, per video reference, in packed order:
      * decode the frames (unit A, ffmpeg -> rgb24 + sidecar)
      * 24 fps resample (vendor order: before_encoder.py:371 runs it BEFORE
        the resize; index math gated in minimax_h3_ref2va_parity)
      * truncate + LANCZOS resize onto the reference's OWN canvas
        (`prepare_reference_frames`, packing_ref2va.py:654-681 — 768 short
        edge; gated BIT-EXACT against the vendor's own function by
        pipeline/parity/minimax_h3_ref_frames_probe.mojo)
      * 17n+5 trim (`trim_reference_num_frames`, encoders.py:574)
      * pixel-normalize onto the video VAE's ImageNet convention (gated)
      * `minimax_h3_encode_reference_visual_seam` — the REAL device encode
        (tiled, temporal-chunked) + posterior sample at seed 42, gated by
        models/vae/parity/minimax_h3_ref_encode_gate.mojo

    then raises at the NEXT unbuilt stage: packing the sampled latents into
    this request's condition rows, and the Qwen3-VL conditioning (stage 4's
    seam — text_encoder shard 14 of 14 carries no vision-tower weights). The
    steps between the sample and the rows — fp16 round, latent normalize,
    channel-slowest patchify, the 0.999 noise mix — are BUILT and gated
    host-side (models/vae/parity/minimax_h3_ref_encode_probe.mojo).

    Only reached WITHOUT `--partial` — see `_minimax_h3_ref2va_generate`.
    Constructs the `DeviceContext`: this stage IS the real encode now, so a
    plain invocation touches the GPU up to the conditioning seam."""
    var ctx = DeviceContext()
    var enc_cfg = minimax_h3_video_released_encoder_config()
    var encoder = MiniMaxH3VideoEncoderDevice.load(
        String(VIDEO_VAE_DIR), enc_cfg, ctx
    )
    var tiling = minimax_h3_video_released_tiling_config()

    var encoded = 0
    for i in range(len(specs)):
        if specs[i].kind != MINIMAX_H3_REF_VIDEO:
            continue
        var rgb_path = specs[i].path + String(".rgb")
        var sidecar_path = specs[i].path + String(".rgb.json")
        var scratch_path = specs[i].path + String(".h3probe")
        print("    decoding", specs[i].path, "-> rgb24")
        var frames = minimax_h3_ffmpeg_extract_rgb(
            specs[i].path, rgb_path, sidecar_path, scratch_path
        )
        print(
            "      ", frames.num_frames, "frames", frames.width, "x",
            frames.height, "@", frames.fps, "fps",
        )

        # Vendor order (before_encoder.py:371-372): 24 fps resample FIRST,
        # then truncate + canvas resize. The resample output is capped at the
        # target's frame count because `prepare_reference_frames` truncates
        # there anyway (packing_ref2va.py:675) — a pure output truncation,
        # gated as such in the ref_frames probe.
        var on_grid = minimax_h3_resample_reference_frames(frames, FRAMES)
        if frames.fps != Float64(24.0):
            print(
                "      24 fps resample ->", on_grid.num_frames, "frames kept",
            )
        var prepared = minimax_h3_prepare_reference_frames(on_grid, FRAMES)
        print(
            "      canvas resize ->", prepared.num_frames, "frames",
            prepared.width, "x", prepared.height,
        )

        # `frames[: trim_reference_num_frames(...)]` (encoders.py:574): snap
        # DOWN to a 17n+5 the VAE encodes without padding. The plan
        # (`minimax_h3_resolve_reference`) already refused references shorter
        # than 22 frames, so tripping this here means the decode and the plan
        # disagree — fail loud, the layout reserved rows on the plan's count.
        var encode_frames = minimax_h3_trim_reference_num_frames(
            prepared.num_frames
        )
        if encode_frames > prepared.num_frames:
            raise Error(
                String("minimax_h3_ref2va: reference ") + specs[i].path
                + " decodes to " + String(prepared.num_frames)
                + " frames on the 24 fps grid but the VAE's 17n+5 chunking"
                " needs " + String(encode_frames)
                + " — shorter than the plan assumed. The plan probes geometry"
                " only (no frame count); supply a reference of at least 22"
                " frames at 24 fps."
            )
        var frame_bytes = prepared.height * prepared.width * 3
        var kept = List[UInt8]()
        kept.resize(encode_frames * frame_bytes, 0)
        for b in range(encode_frames * frame_bytes):
            kept[b] = prepared.pixels[b]

        var pixels = minimax_h3_pixel_normalize_frames(
            kept, encode_frames, prepared.height, prepared.width
        )
        print(
            "      pixel-normalized ->", len(pixels),
            "f32 values [3,", encode_frames, ",", prepared.height, ",",
            prepared.width, "]",
        )

        # THE SEAM, now wired: real device encode + posterior sample.
        var sampled = minimax_h3_encode_reference_visual_seam(
            encoder, pixels, 3, encode_frames, prepared.height,
            prepared.width, tiling, ctx,
        )

        # The plan predicted this reference's latent grid from the SAME
        # canvas law (`minimax_h3_resolve_reference`); the encode must land
        # exactly on it — the packed layout reserved rows for it.
        var expected = (
            MINIMAX_H3_VIDEO_LATENT_CHANNELS * references[i].num_latent_frames
            * references[i].latent_height * references[i].latent_width
        )
        if references[i].kind != MINIMAX_H3_REF_VIDEO or len(sampled) != expected:
            raise Error(
                String("minimax_h3_ref2va: reference ") + specs[i].path
                + " encoded to " + String(len(sampled))
                + " latent values but the plan reserved rows for "
                + String(expected)
                + " — the encode and the plan disagree on the latent grid"
            )
        print(
            "      encoded + sampled ->", len(sampled), "latent f32 values ["
            , MINIMAX_H3_VIDEO_LATENT_CHANNELS, ",",
            references[i].num_latent_frames, ",",
            references[i].latent_height, ",", references[i].latent_width, "]",
        )
        encoded += 1

    if encoded == 0:
        raise Error(
            "minimax_h3_ref2va: SEAM — no video reference reached the VAE"
            " encode. Pass at least one video:PATH reference, or pass"
            " --partial to run the full-plumbing chain with this stage"
            " STUBBED instead."
        )
    raise Error(
        "minimax_h3_ref2va: SEAM — the reference video-VAE encode ran for"
        " real (canvas resize -> pixel norm ->"
        " minimax_h3_encode_reference_visual_seam, all gated), but packing"
        " the sampled latents into this request's condition rows and the"
        " Qwen3-VL conditioning (stage 4 — text_encoder shard 14 of 14 has"
        " no vision-tower weights) are not wired in this path. Use --partial"
        " for the full-plumbing chain."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PARTIAL MODE — Stage B: stub-conditioning TAGS (host-only, no DeviceContext).
#
# Called from `main()`, BEFORE the plan is built, so the plan's own
# `token_tags` reflect the REAL presentation layout even though this file
# cannot run the real Qwen3-VL vision tower yet. Only the VALUES (the actual
# BF16 embeds) are noise, drawn later in the device stage
# (`_minimax_h3_ref2va_generate`) — building them here would need a
# DeviceContext this file's header promises not to construct before the
# reference-encode seam (or, under `--partial`, before Stage A/C proper).
# ═════════════════════════════════════════════════════════════════════════════
def _minimax_h3_ref2va_stub_conditioning_tags(
    references: List[MiniMaxH3PreparedReference], num_text_tokens: Int
) raises -> List[Int]:
    """Approximates `build_ref2va_presentation`'s per-reference layout
    (packing_ref2va.py:802-818):
      audio reference -> "<Audio j>: " label      (approximated: 6 tag-1 rows)
      image reference -> label (6 tag-1) + ONE vision block
                          (tag-0, `MiniMaxH3VisionGrid.num_tokens()` rows)
      video reference -> label (6 tag-1) + PER MERGED FRAME PAIR: a timestamp
                          label (approximated: 6 tag-1) + one vision block
                          (tag-0, `tokens_per_block()` rows)
    then the prompt itself fills whatever budget remains as tag 1.

    APPROXIMATION, stated plainly rather than smoothed over: a reference's
    vision GRID here is `(t=num_latent_frames or 1, h=latent_height,
    w=latent_width)` — the DiT's OWN latent-space geometry (unit B's
    `MiniMaxH3PreparedReference`), NOT the processor's `smart_resize` PIXEL
    grid the real Qwen3-VL tower would compute. It exists only to make the
    stub's TOKEN COUNT plausible; the real grid is the tower's job once its
    weights exist (text_encoder shard 14 of 14).

    Raises with the exact required `H3_TEXT_TOKENS` if the references alone
    (labels + vision blocks) exceed the compiled text-token budget — there is
    no room left for the prompt otherwise."""
    var tags = List[Int]()
    for i in range(len(references)):
        ref r = references[i]
        if r.kind == MINIMAX_H3_REF_AUDIO:
            for _ in range(6):
                tags.append(MINIMAX_H3_TEXT_TAG)
        elif r.kind == MINIMAX_H3_REF_IMAGE:
            for _ in range(6):
                tags.append(MINIMAX_H3_TEXT_TAG)
            var grid = MiniMaxH3VisionGrid(1, r.latent_height, r.latent_width)
            for _ in range(grid.num_tokens()):
                tags.append(MINIMAX_H3_VIDEO_TAG)
        elif r.kind == MINIMAX_H3_REF_VIDEO:
            for _ in range(6):
                tags.append(MINIMAX_H3_TEXT_TAG)
            var grid = MiniMaxH3VisionGrid(
                r.num_latent_frames, r.latent_height, r.latent_width
            )
            var per_block = grid.tokens_per_block()
            for _ in range(grid.t):
                for _ in range(6):
                    tags.append(MINIMAX_H3_TEXT_TAG)
                for _ in range(per_block):
                    tags.append(MINIMAX_H3_VIDEO_TAG)
        else:
            raise Error(
                String("minimax_h3_ref2va: reference kind ") + String(r.kind)
                + " is not image, video or audio"
            )

    var used = len(tags)
    if used > num_text_tokens:
        raise Error(
            String("minimax_h3_ref2va: the stub presentation needs ")
            + String(used) + " text-region tokens (reference labels +"
            " approximated vision blocks) but this binary is compiled for"
            " H3_TEXT_TOKENS=" + String(num_text_tokens) + " — rebuild with"
            " -D H3_TEXT_TOKENS=" + String(used)
            + " (or larger, to leave room for the prompt itself)"
        )
    for _ in range(num_text_tokens - used):
        tags.append(MINIMAX_H3_TEXT_TAG)
    return tags^


# ═════════════════════════════════════════════════════════════════════════════
# PARTIAL MODE — Stage A: condition rows (injected or synthetic + the real
# 0.999 noise mix). Device stage — called from `_minimax_h3_ref2va_generate`,
# AFTER `DeviceContext()`.
# ═════════════════════════════════════════════════════════════════════════════
def _h3_load_condition_tensor(
    st: SafeTensors, name: String, want_shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """Load one `--condition-rows=PATH` tensor and validate its shape/dtype
    EXACTLY against the layout this plan resolved. This is the entry point
    the real GPU ref-encode will eventually feed, so the contract is strict,
    not best-effort — a shape/dtype mismatch is a loud, named error, never a
    silent reinterpretation."""
    if not st.has_tensor(name):
        raise Error(
            String("minimax_h3_ref2va: --condition-rows file is missing tensor '")
            + name + "'"
        )
    var info = st.tensor_info(name)
    if info.dtype != STDtype.F32:
        raise Error(
            String("minimax_h3_ref2va: --condition-rows tensor '") + name
            + "' is not F32"
        )
    if len(info.shape) != len(want_shape):
        raise Error(
            String("minimax_h3_ref2va: --condition-rows tensor '") + name
            + "' has rank " + String(len(info.shape)) + ", want "
            + String(len(want_shape))
        )
    for d in range(len(want_shape)):
        if info.shape[d] != want_shape[d]:
            raise Error(
                String("minimax_h3_ref2va: --condition-rows tensor '") + name
                + "' shape mismatch at dim " + String(d) + ": got "
                + String(info.shape[d]) + ", want " + String(want_shape[d])
            )
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _h3_ref2va_load_video_condition(
    num_rows: Int, patch_dim: Int, condition_rows_path: String,
    seed: UInt64, ctx: DeviceContext,
) raises -> Tensor:
    """One function, one `if/return` — NOT a `var` declared then conditionally
    assigned in two branches (Mojo does not support that; see t2va's own
    `_minimax_h3_open_transformer_shards` for the same reasoning)."""
    var want_shape: List[Int] = [num_rows, patch_dim]
    if condition_rows_path != String(""):
        var st = SafeTensors.open(condition_rows_path)
        return _h3_load_condition_tensor(
            st, String("video_condition_rows"), want_shape, ctx
        )
    return randn(want_shape^, seed + 100, STDtype.F32, ctx)


def _h3_ref2va_load_audio_condition(
    num_rows: Int, latent_dim: Int, condition_rows_path: String,
    seed: UInt64, ctx: DeviceContext,
) raises -> Tensor:
    """Same one-function `if/return` discipline as the video sibling above.

    `num_rows == 0` (no audio reference and no video reference carries a
    soundtrack) returns a 1-row PLACEHOLDER: `randn` (ops/random.mojo)
    rejects a zero-sized shape outright, and `_h3_ref2va_scatter_rows` never
    reads this tensor's contents when the caller's condition-row count for
    this modality is zero — only its existence as a valid `Tensor` matters."""
    if num_rows == 0:
        var placeholder_shape: List[Int] = [1, latent_dim]
        return randn(placeholder_shape^, seed + 101, STDtype.F32, ctx)
    var want_shape: List[Int] = [num_rows, latent_dim]
    if condition_rows_path != String(""):
        var st = SafeTensors.open(condition_rows_path)
        return _h3_load_condition_tensor(
            st, String("audio_condition_rows"), want_shape, ctx
        )
    return randn(want_shape^, seed + 101, STDtype.F32, ctx)


def _minimax_h3_ref2va_condition_rows(
    plan: MiniMaxH3Ref2VAPlan,
    config: MiniMaxH3DiTConfig,
    condition_rows_path: String,
    seed: UInt64,
    ctx: DeviceContext,
) raises -> Tuple[Tensor, Tensor]:
    """Stage A: this run's (video, audio) CONDITION rows.

    VIDEO is noise-mixed at the CONSTANT 0.999 through the GATED
    `minimax_h3_mix_condition_rows` (models/vae/minimax_h3_ref_encode.mojo) —
    via a host round trip (`.to_host`/`Tensor.from_host`) so the GATED
    function is what actually runs, per this file's own instruction to prefer
    that over restating `0.999*x + 0.001*noise` on-device. AUDIO is NEVER
    mixed — the vendor's own asymmetry (encoders.py:612-630): audio condition
    rows are the VAE posterior MODE, never sampled, never noise-augmented,
    yet pinned at row timestep 1.0 by unit B's own row-timestep math."""
    if plan.num_condition_video_rows == 0:
        raise Error(
            "minimax_h3_ref2va: this request resolved zero condition-video"
            " rows — ref2va requires at least one image or video reference"
            " (_validate_reference_counts already enforces this), so this"
            " would indicate a bug in plan construction, not a normal"
            " user-facing case"
        )
    var video_raw = _h3_ref2va_load_video_condition(
        plan.num_condition_video_rows, config.video_patch_dim(),
        condition_rows_path, seed, ctx,
    )
    var video_shape: List[Int] = [plan.num_condition_video_rows, config.video_patch_dim()]
    var noise = randn(video_shape.copy(), seed + 102, STDtype.F32, ctx)
    var video_host = video_raw.to_host(ctx)
    var noise_host = noise.to_host(ctx)
    var mixed_host = minimax_h3_mix_condition_rows(video_host, noise_host)
    var video_mixed = Tensor.from_host(mixed_host, video_shape^, STDtype.F32, ctx)

    var audio_raw = _h3_ref2va_load_audio_condition(
        plan.num_condition_audio_rows, config.audio_latents_dim,
        condition_rows_path, seed, ctx,
    )

    return (video_mixed^, audio_raw^)


# ═════════════════════════════════════════════════════════════════════════════
# PARTIAL MODE — Stage C helpers: the GLOBAL 4-row-per-step AdaLN addressing
# and the per-modality (condition, target) row-buffer combine.
# ═════════════════════════════════════════════════════════════════════════════
def _minimax_h3_ref2va_global_timestep_row(
    plan: MiniMaxH3Ref2VAPlan, step: Int
) raises -> List[Int]:
    """Per-row index into the GLOBAL `4*num_steps`-row modulation cache for
    denoising step `step` — ref2va's difference from t2va's 2-row-per-step
    layout (this file's own header, MODCACHE / Stage C):
      `4*step+0`  video/text rows (default — text inherits the video
                  timestep, packing.mojo's own convention, extended here to
                  a reference's vision-tagged text-region rows too)
      `4*step+1`  TARGET audio rows
      `4*step+2`  CONDITION video rows (max(video_t, 0.999))
      `4*step+3`  CONDITION audio rows (constant 1.0)
    Built by starting all-`4*step+0` then overwriting from the condition/
    target index lists, in that order — generalizes t2va's own
    `_minimax_h3_global_timestep_row` from 2 rows to 4 (that function is
    module-private in minimax_h3_t2va.mojo; this is an independent
    reimplementation of the SAME bookkeeping idea for ref2va's 4-timestep
    layout, not a copy of its 2-timestep body)."""
    var out = List[Int](capacity=plan.sequence_length())
    for _ in range(plan.sequence_length()):
        out.append(4 * step + 0)
    for i in range(plan.num_condition_video_rows):
        out[plan.layout.video_indices[i]] = 4 * step + 2
    for i in range(plan.num_condition_audio_rows):
        out[plan.layout.audio_indices[i]] = 4 * step + 3
    for i in range(plan.num_condition_audio_rows, len(plan.layout.audio_indices)):
        out[plan.layout.audio_indices[i]] = 4 * step + 1
    return out^


def _h3_ref2va_concat_rows(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    """dim=0 concat of two rank-2, same-dtype, same-trailing-dim tensors.
    WORKAROUND for a SIGSEGV localized (GDB, 2026-08-04) in
    ops.tensor_algebra.concat's variadic `*tensors` pack at THIS call site
    only: indexing the pack inside the heavily-inlined ref2va denoise loop
    dereferences a garbage pointer while materializing `tensors[0].shape()`.
    Isolated repros at identical shapes run clean — the trigger is the
    instantiation context, not the shapes. Row-major dim=0 concat needs no
    interleaving — A's bytes then B's bytes — so two D2D enqueue_copys into
    sub-buffer views reproduce concat's own output without the variadic
    pack."""
    var ash = a.shape()
    var bsh = b.shape()
    if len(ash) != 2 or len(bsh) != 2:
        raise Error("_h3_ref2va_concat_rows: both inputs must be rank-2")
    if ash[1] != bsh[1]:
        raise Error("_h3_ref2va_concat_rows: trailing dim mismatch")
    if a.dtype() != b.dtype():
        raise Error("_h3_ref2va_concat_rows: dtype mismatch")
    var a_bytes = a.nbytes()
    var b_bytes = b.nbytes()
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](a_bytes + b_bytes)
    var a_dst = out_buf.create_sub_buffer[DType.uint8](0, a_bytes)
    ctx.enqueue_copy(dst_buf=a_dst, src_buf=a.buf)
    var b_dst = out_buf.create_sub_buffer[DType.uint8](a_bytes, b_bytes)
    ctx.enqueue_copy(dst_buf=b_dst, src_buf=b.buf)
    var out_shape: List[Int] = [ash[0] + bsh[0], ash[1]]
    return Tensor(out_buf^, out_shape^, a.dtype())


def _h3_ref2va_scatter_rows(
    condition: Tensor, target: Tensor, num_condition: Int, ctx: DeviceContext
) raises -> Tensor:
    """Combine one modality's PINNED condition rows with its Euler-evolving
    target rows into the single buffer `minimax_h3_frontend_embed` addresses
    by `video_indices`/`audio_indices` — condition rows first, target rows
    after (this file's own `MiniMaxH3Ref2VAPlan` header). `slice`-duplicates
    `target` when there are no condition rows of this modality (Tensor is
    Movable-not-Copyable, so returning the borrowed `target` directly would
    need a copy that does not exist for this type)."""
    if num_condition == 0:
        return slice(target, 0, 0, target.shape()[0], ctx)
    return _h3_ref2va_concat_rows(condition, target, ctx)


# ═════════════════════════════════════════════════════════════════════════════
# Frontend weight loading — REPLICATED from t2va's private
# `_h3_fp32_frontend_keys` / `_minimax_h3_load_frontend_weights`
# (minimax_h3_t2va.mojo, module-private, not importable/importable-by-
# convention). Ref2VA's `transformer/config.json` is byte-identical to
# FL2VA's (this file's own header), so the same 12 fp32 keys and the same
# load recipe apply verbatim.
# ═════════════════════════════════════════════════════════════════════════════
def _h3_ref2va_fp32_frontend_keys() -> List[String]:
    var out = List[String]()
    out.append(String("video_patch_proj.weight"))
    out.append(String("video_patch_proj.bias"))
    out.append(String("audio_patch_proj.weight"))
    out.append(String("audio_patch_proj.bias"))
    out.append(String("time_embedder.proj_in.weight"))
    out.append(String("time_embedder.proj_in.bias"))
    out.append(String("time_embedder.proj_out.weight"))
    out.append(String("time_embedder.proj_out.bias"))
    out.append(String("final_layer.video_out.weight"))
    out.append(String("final_layer.video_out.bias"))
    out.append(String("final_layer.audio_out.weight"))
    out.append(String("final_layer.audio_out.bias"))
    return out^


def _minimax_h3_ref2va_preflight_block_tensors(
    shards: ShardedSafeTensors, config: MiniMaxH3DiTConfig
) raises:
    """REPLICATED from t2va's private `_preflight_block_tensors`."""
    for layer in range(config.num_layers):
        var names = minimax_h3_block_tensor_names(layer)
        for i in range(len(names)):
            ref name = names[i]
            if not shards.has_tensor(name):
                raise Error(
                    String("minimax_h3_ref2va preflight: missing block tensor ")
                    + name
                )
            var info = shards.tensor_info(name)
            var want = minimax_h3_expected_shape(name, config)
            if len(info.shape) != len(want):
                raise Error(
                    String("minimax_h3_ref2va preflight: rank mismatch for ")
                    + name
                )
            for d in range(len(want)):
                if info.shape[d] != want[d]:
                    raise Error(
                        String("minimax_h3_ref2va preflight: shape mismatch for ")
                        + name
                    )
            if info.dtype != STDtype.BF16:
                raise Error(
                    String("minimax_h3_ref2va preflight: ") + name
                    + " is not BF16"
                )


def _minimax_h3_ref2va_preflight_frontend_tensors(
    shards: ShardedSafeTensors, config: MiniMaxH3DiTConfig
) raises:
    """REPLICATED from t2va's private `_preflight_frontend_tensors`."""
    var fp32_keys = _h3_ref2va_fp32_frontend_keys()
    for i in range(len(fp32_keys)):
        var name = fp32_keys[i]
        if not shards.has_tensor(name):
            raise Error(
                String("minimax_h3_ref2va preflight: missing frontend fp32 tensor ")
                + name
            )
        if shards.tensor_info(name).dtype != STDtype.F32:
            raise Error(
                String("minimax_h3_ref2va preflight: ") + name + " is not F32"
                " (one of the checkpoint's 12 dtype-trap tensors)"
            )
    var bf16_names = List[String]()
    bf16_names.append(String("condition_proj.weight"))
    bf16_names.append(String("condition_proj.bias"))
    bf16_names.append(String("token_refiner.final_norm.weight"))
    bf16_names.append(String("final_layer.norm.weight"))
    for layer in range(config.token_refiner_num_layers):
        var p = String("token_refiner.blocks.") + String(layer)
        bf16_names.append(p + ".norm1.weight")
        bf16_names.append(p + ".attn.qkv_proj.weight")
        bf16_names.append(p + ".attn.q_norm.weight")
        bf16_names.append(p + ".attn.k_norm.weight")
        bf16_names.append(p + ".attn.out_proj.weight")
        bf16_names.append(p + ".norm2.weight")
        bf16_names.append(p + ".mlp.fc1.weight")
        bf16_names.append(p + ".mlp.fc2.weight")
    for i in range(len(bf16_names)):
        ref name = bf16_names[i]
        if not shards.has_tensor(name):
            raise Error(
                String("minimax_h3_ref2va preflight: missing frontend tensor ")
                + name
            )
        if shards.tensor_info(name).dtype != STDtype.BF16:
            raise Error(
                String("minimax_h3_ref2va preflight: ") + name + " is not BF16"
            )


def _minimax_h3_ref2va_load_frontend_weights(
    shards: ShardedSafeTensors, config: MiniMaxH3DiTConfig, ctx: DeviceContext
) raises -> Dict[String, ArcPointer[Tensor]]:
    """REPLICATED from t2va's private `_minimax_h3_load_frontend_weights`."""
    var w = Dict[String, ArcPointer[Tensor]]()
    var fp32_keys = _h3_ref2va_fp32_frontend_keys()
    for i in range(len(fp32_keys)):
        var name = fp32_keys[i]
        w[name] = ArcPointer(Tensor.from_view(shards.tensor_view(name), ctx))

    w["condition_proj.weight"] = ArcPointer(
        Tensor.from_view(shards.tensor_view("condition_proj.weight"), ctx)
    )
    w["condition_proj.bias"] = ArcPointer(
        Tensor.from_view(shards.tensor_view("condition_proj.bias"), ctx)
    )

    var heads = config.num_attention_heads
    var head_dim = config.attention_head_dim
    var hidden = config.hidden_size
    var ffn = config.ffn_hidden_size
    for layer in range(config.token_refiner_num_layers):
        var p = String("token_refiner.blocks.") + String(layer)
        w[p + ".norm1.weight"] = ArcPointer(
            Tensor.from_view(shards.tensor_view(p + ".norm1.weight"), ctx)
        )
        var qkv_name = p + ".attn.qkv_proj.weight"
        w[qkv_name] = ArcPointer(
            minimax_h3_load_qkv_device(shards, qkv_name, heads, head_dim, hidden, ctx)
        )
        w[p + ".attn.q_norm.weight"] = ArcPointer(
            Tensor.from_view(shards.tensor_view(p + ".attn.q_norm.weight"), ctx)
        )
        w[p + ".attn.k_norm.weight"] = ArcPointer(
            Tensor.from_view(shards.tensor_view(p + ".attn.k_norm.weight"), ctx)
        )
        w[p + ".attn.out_proj.weight"] = ArcPointer(
            Tensor.from_view(shards.tensor_view(p + ".attn.out_proj.weight"), ctx)
        )
        w[p + ".norm2.weight"] = ArcPointer(
            Tensor.from_view(shards.tensor_view(p + ".norm2.weight"), ctx)
        )
        var fc1_name = p + ".mlp.fc1.weight"
        w[fc1_name] = ArcPointer(
            minimax_h3_load_fc1_device(shards, fc1_name, ffn, hidden, ctx)
        )
        w[p + ".mlp.fc2.weight"] = ArcPointer(
            Tensor.from_view(shards.tensor_view(p + ".mlp.fc2.weight"), ctx)
        )
    w["token_refiner.final_norm.weight"] = ArcPointer(
        Tensor.from_view(shards.tensor_view("token_refiner.final_norm.weight"), ctx)
    )
    w["final_layer.norm.weight"] = ArcPointer(
        Tensor.from_view(shards.tensor_view("final_layer.norm.weight"), ctx)
    )
    return w^


def _minimax_h3_ref2va_generate(
    plan: MiniMaxH3Ref2VAPlan,
    specs: List[MiniMaxH3ReferenceSpec],
    references: List[MiniMaxH3PreparedReference],
    partial_mode: Bool,
    condition_rows_path: String,
    steps: Int,
    seed: UInt64,
    out_dir: String,
) raises:
    """The generation tail. Two shapes:

      * WITHOUT `--partial`: runs the REAL reference encode — ffmpeg decode,
        24 fps resample, the gated canvas LANCZOS resize, pixel norm, and
        the GATED device seam (`minimax_h3_encode_reference_visual_seam`) —
        then raises at the condition-row packing / Qwen3-VL conditioning
        seam. Touches the GPU exactly as far as the encode.

      * WITH `--partial` (this pass): the FULL plumbing chain, with the two
        stages that cannot run today — Stage A's video-VAE forward and Stage
        B's Qwen3-VL vision tower — STUBBED (see file header PARTIAL MODE).
        Everything else — the packed layout, RoPE tables, the 4-row-per-step
        modulation cache, the REAL `config.num_layers`-block streamed
        denoise, Euler steps restricted to target rows, the save — is real.

    `DeviceContext()` is constructed inside the branch taken — the
    `--partial` chain here, the encode chain in
    `_minimax_h3_encode_references` — never before the preflight."""
    if not partial_mode:
        _minimax_h3_encode_references(specs, references)
        return

    if H3_REF_SEQ_LEN != plan.sequence_length():
        raise Error(
            String("minimax_h3_ref2va: this request's plan.sequence_length()")
            + " is " + String(plan.sequence_length()) + " but this binary is"
            " compiled for H3_REF_SEQ_LEN=" + String(H3_REF_SEQ_LEN)
            + " (default 1, i.e. never set) — minimax_h3_block_forward's"
            " self-attention instantiates its row count S at COMPTIME"
            " (models/dit/minimax_h3_dit.mojo — the same reason"
            " H3_TEXT_TOKENS is comptime in this file already), and a ref2va"
            " request's sequence_length depends on the references, so it"
            " cannot be read off the request at runtime the way t2va's fixed"
            " SEQ_LEN can. Rebuild with -D H3_REF_SEQ_LEN="
            + String(plan.sequence_length())
        )
    if plan.num_text_tokens != TEXT_TOKENS:
        raise Error(
            "minimax_h3_ref2va: plan.num_text_tokens != H3_TEXT_TOKENS"
            " (internal — the stub presentation's tag list should always be"
            " padded to exactly TEXT_TOKENS)"
        )

    var config = minimax_h3_released_config()
    config.validate()

    var injected = condition_rows_path != String("")
    print("")
    print("  ################################################################")
    print("  # PARTIAL MODE (ref2va): conditioner STUBBED — fixed-seed random")
    print("  # [1,", TEXT_TOKENS, ",", config.text_dim, "] BF16, NOT the real")
    print("  # prompt: the Qwen3-VL vision tower's 351 tensors live in")
    print("  # text_encoder shard 14 of 14, which is absent.")
    if injected:
        print("  # Condition rows: INJECTED from", condition_rows_path)
    else:
        print("  # Condition rows: SYNTHETIC (fixed-seed random + the real")
        print("  #   0.999 noise mix)")
    print("  # THIS IS NOT A VALID MINIMAX-H3 REF2VA GENERATION. Plumbing test")
    print("  # only: condition rows -> stub conditioning -> frontend embed ->")
    print(
        "  # packed sequence -> modcache (4 rows/step) ->", config.num_layers,
        "real streamed",
    )
    print("  # blocks -> Euler (target rows only) -> final layer -> save latents.")
    print("  ################################################################")
    print("")

    print("  preflight: opening transformer shards:", String(TRANSFORMER_DIR))
    var transformer_shards = ShardedSafeTensors.open(String(TRANSFORMER_DIR))
    print(
        "  preflight: ", transformer_shards.num_shards(), "shard(s), ",
        transformer_shards.num_tensors(), "tensors",
    )
    minimax_h3_check_modcache_weights(transformer_shards, config)
    _minimax_h3_ref2va_preflight_block_tensors(transformer_shards, config)
    _minimax_h3_ref2va_preflight_frontend_tensors(transformer_shards, config)
    print(
        "  preflight: transformer OK (adaLN + ", config.num_layers,
        "blocks + frontend)",
    )

    _ = sys_system(String("mkdir -p '") + out_dir + "'")

    var ctx = DeviceContext()

    # ── Stage A: condition rows ──────────────────────────────────────────
    var t_cond_rows0 = perf_counter_ns()
    var cond = _minimax_h3_ref2va_condition_rows(
        plan, config, condition_rows_path, seed, ctx
    )
    var t_cond_rows1 = perf_counter_ns()
    print(
        "  condition rows", "[injected]" if injected else "[synthetic]", ": ",
        plan.num_condition_video_rows, "video +", plan.num_condition_audio_rows,
        "audio (", Float64(t_cond_rows1 - t_cond_rows0) / 1.0e9, "s)",
    )

    # ── Stage B: stub-conditioning VALUES. The LAYOUT (which text-region
    # rows are vision-block rows) is already baked into `plan.layout.
    # token_tags` — `main()` built it for real, from the declared
    # references, before calling this function (see
    # `_minimax_h3_ref2va_stub_conditioning_tags`). Only the embeds
    # themselves are fixed-seed random here. ─────────────────────────────
    var text_shape: List[Int] = [1, TEXT_TOKENS, config.text_dim]
    var text_embeds = randn(text_shape^, seed + 200, STDtype.BF16, ctx)
    var text_rows = reshape(text_embeds, [TEXT_TOKENS, config.text_dim], ctx)

    # ── RoPE tables ─────────────────────────────────────────────────────
    var positions_f32 = List[Float32](capacity=len(plan.layout.position_ids))
    for i in range(len(plan.layout.position_ids)):
        positions_f32.append(Float32(plan.layout.position_ids[i]))
    var positions_shape: List[Int] = [plan.sequence_length() * 3]
    var positions_tensor = Tensor.from_host(positions_f32, positions_shape^, STDtype.F32, ctx)
    var rope = build_minimax_h3_rope_tables(positions_tensor, ctx, config.rope_inv_freq_len)
    var rotary_dim = rope[0].shape()[1]
    print(
        "  geometry + rope ready: S=", plan.sequence_length(),
        " rotary_dim=", rotary_dim,
    )

    # ── Dual schedule ────────────────────────────────────────────────────
    var schedule = MiniMaxH3DualSchedule()
    schedule.set_timesteps(steps)
    var num_steps = schedule.num_inference_steps()
    print("  schedule: ", num_steps, " model evaluations")

    # ── Modcache: 4 rows/step — video, audio, condition-video, condition-
    # audio (t2va's own equivalent is 2 rows/step; ref2va needs two more for
    # the pinned condition timesteps — see file header MODCACHE). ────────
    var t_mod0 = perf_counter_ns()
    var distinct_timesteps = 4 * num_steps
    var temb_timesteps = List[Float32](capacity=distinct_timesteps)
    for i in range(num_steps):
        var video_t = schedule.video_timestep(i)
        var audio_t = schedule.audio_timestep(i)
        temb_timesteps.append(video_t)
        temb_timesteps.append(audio_t)
        var cond_video_t = video_t
        if cond_video_t < MINIMAX_H3_KEYFRAME_NOISE_AUG:
            cond_video_t = MINIMAX_H3_KEYFRAME_NOISE_AUG
        temb_timesteps.append(cond_video_t)
        temb_timesteps.append(Float32(1.0))
        # Per-step distinct-value COUNT can legitimately collapse below 4
        # (video_t >= 0.999 merges the condition-video row into it; a layout
        # with zero condition-audio rows never actually reads the 1.0 row) —
        # verified here, not assumed. A collapsed step just computes the
        # same temb row twice: wasteful, never wrong — the same argument
        # t2va's own header makes for its 2-row layout.
        var ts = minimax_h3_ref2va_row_timesteps(plan.layout, video_t, audio_t)
        if len(ts.values) > 4:
            raise Error(
                "minimax_h3_ref2va: row-timesteps produced more than 4"
                " distinct values (internal — this modcache sizing assumes"
                " at most 4)"
            )
    var temb_shape: List[Int] = [distinct_timesteps]
    var temb_timesteps_tensor = Tensor.from_host(temb_timesteps, temb_shape^, STDtype.F32, ctx)

    var frontend_w = _minimax_h3_ref2va_load_frontend_weights(transformer_shards, config, ctx)
    var temb = minimax_h3_timestep_embedding(temb_timesteps_tensor, frontend_w, config, ctx)
    var modcache = minimax_h3_build_modulation_cache(transformer_shards, temb, config, ctx)
    ctx.synchronize()
    var t_mod1 = perf_counter_ns()
    print(
        "  modcache: ", distinct_timesteps, " rows, ",
        Float64(modcache.total_bytes()) / (1024.0 * 1024.0), " MiB (",
        Float64(t_mod1 - t_mod0) / 1.0e9, "s)",
    )

    # ── Denoise loop. Condition rows (`cond[0]`/`cond[1]`) are PINNED —
    # Stage A's output, never re-mixed, never stepped. Target rows are the
    # F32 accumulating Euler state, exactly like t2va's video_state/
    # audio_state. ───────────────────────────────────────────────────────
    var t_denoise0 = perf_counter_ns()
    var video_target_shape: List[Int] = [plan.num_target_video_rows, config.video_patch_dim()]
    var video_target = randn(video_target_shape^, seed, STDtype.F32, ctx)
    var audio_target_shape: List[Int] = [plan.num_target_audio_rows, config.audio_latents_dim]
    var audio_target = randn(audio_target_shape^, seed + 1, STDtype.F32, ctx)

    for i in range(num_steps):
        var t_step0 = perf_counter_ns()
        var video_ts = schedule.video_timestep(i)
        var audio_ts = schedule.audio_timestep(i)
        var global_row = _minimax_h3_ref2va_global_timestep_row(plan, i)
        var block_adaln_indices = minimax_h3_adaln_rows(global_row, plan.layout.token_tags)

        var video_rows_combined = _h3_ref2va_scatter_rows(
            cond[0], video_target, plan.num_condition_video_rows, ctx
        )
        var audio_rows_combined = _h3_ref2va_scatter_rows(
            cond[1], audio_target, plan.num_condition_audio_rows, ctx
        )

        var placeholder_ts_shape: List[Int] = [1]
        var placeholder_ts = Tensor.from_host([video_ts], placeholder_ts_shape^, STDtype.F32, ctx)
        var embed = minimax_h3_frontend_embed[TEXT_TOKENS, MINIMAX_H3_HEADS, MINIMAX_H3_HEAD_DIM](
            video_rows_combined, audio_rows_combined, text_rows, placeholder_ts,
            plan.layout.video_indices, plan.layout.audio_indices, plan.layout.text_indices,
            plan.sequence_length(), frontend_w, config, ctx,
        )

        # RANK ADAPTER — the SAME seam t2va documents at its own header /
        # ~lines 1448-1470: `minimax_h3_frontend_embed` hands back RANK 2
        # ([S,hidden]) but `minimax_h3_block_forward` needs RANK 3
        # ([1,S,hidden]). Reshape in immediately before the block loop,
        # back out immediately after.
        var hidden3 = reshape(embed.hidden, [1, H3_REF_SEQ_LEN, config.hidden_size], ctx)

        # The real streamed-block stack is gated behind `H3_REF2VA_REAL_
        # BLOCKS` (see that comptime's own header note): `minimax_h3_block_
        # forward` calls `sdpa_flash_infer_fwd`, which needs the cuDNN SDPA
        # shim linked in, and THIS FILE'S build command does not pass the
        # extra `-Xlinker` flags t2va's own production builds always do.
        # `@parameter if` is a COMPILE-TIME branch (like C++ `if constexpr`):
        # with the flag at its default of 0, the block-forward call below is
        # never instantiated and the shim is never referenced, so the
        # DEFAULT build links clean. Everything ABOVE this point in the
        # step — the scatter, the frontend embed (token refiner uses
        # `sdpa_nomask`, which needs no shim) — already ran for real.
        comptime if H3_REF2VA_REAL_BLOCKS != 0:
            for layer in range(config.num_layers):
                var block_w = minimax_h3_load_block_device(transformer_shards, layer, config, ctx)
                hidden3 = minimax_h3_block_forward[H3_REF_SEQ_LEN, MINIMAX_H3_HEADS, MINIMAX_H3_HEAD_DIM](
                    hidden3, block_w, layer, config, modcache.block_mod[layer][],
                    block_adaln_indices, rope[0], rope[1], rotary_dim, ctx,
                )
                # `block_w` drops here: one block's ~0.77 GiB bf16 is freed
                # before the next layer's load, exactly like t2va.

            var hidden2 = reshape(hidden3, [H3_REF_SEQ_LEN, config.hidden_size], ctx)
            var frontend_out = minimax_h3_final_layer(
                hidden2, modcache.final_mod[], global_row,
                plan.layout.video_indices, plan.layout.audio_indices,
                frontend_w, config, ctx,
            )

            # EULER ON TARGET ROWS ONLY. `frontend_out.video_out`/`.audio_out`
            # cover ALL video/audio rows (condition + target, condition rows
            # FIRST — this file's own MiniMaxH3Ref2VAPlan header). Slice off
            # just the target suffix and step it; `cond[0]`/`cond[1]` are
            # never touched again this step or any other — PINNED, matching
            # the vendor's own recipe (noise-mixed ONCE at encode time, never
            # re-mixed per step).
            var target_video_out = slice(
                frontend_out.video_out, 0, plan.num_condition_video_rows,
                plan.num_target_video_rows, ctx,
            )
            video_target = schedule.step_video_device(target_video_out, video_ts, video_target, ctx)
            var target_audio_out = slice(
                frontend_out.audio_out, 0, plan.num_condition_audio_rows,
                plan.num_target_audio_rows, ctx,
            )
            audio_target = schedule.step_audio_device(target_audio_out, audio_ts, audio_target, ctx)
        else:
            raise Error(
                "minimax_h3_ref2va: SEAM — the real streamed-block stack"
                " (minimax_h3_block_forward -> sdpa_flash_infer_fwd ->"
                " flame_cudnn_sdpa_bf16) needs the cuDNN SDPA shim, which"
                " this binary's link line does not include. Rebuild with"
                " -D H3_REF2VA_REAL_BLOCKS=1 -Xlinker -lm -Xlinker -lcuda"
                " -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker"
                " -lserenity_cudnn_sdpa -Xlinker"
                " -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn"
                " (the same flags t2va's own production builds use — see"
                " output/logs/h3_shots_orchestrator.sh) to run the real"
                " denoise. Condition rows, stub conditioning, packed layout,"
                " rope, dual schedule and the modulation cache all ran for"
                " real above this seam."
            )
        ctx.synchronize()
        var t_step1 = perf_counter_ns()
        print(
            "  phase=denoise step=", i + 1, " total=", num_steps,
            " video_t=", video_ts, " audio_t=", audio_ts,
            " (", Float64(t_step1 - t_step0) / 1.0e9, "s)",
        )

    # ── Save final latents. TARGET slices under the t2va-convention names,
    # plus the PINNED condition rows for reproducibility. ─────────────────
    var lat_names = List[String]()
    lat_names.append(String("video_state_rows"))
    lat_names.append(String("audio_state_rows"))
    lat_names.append(String("condition_video_rows"))
    lat_names.append(String("condition_audio_rows"))
    var lat_tensors = List[ArcPointer[Tensor]]()
    lat_tensors.append(ArcPointer[Tensor](video_target^))
    lat_tensors.append(ArcPointer[Tensor](audio_target^))
    lat_tensors.append(ArcPointer[Tensor](slice(cond[0], 0, 0, cond[0].shape()[0], ctx)))
    lat_tensors.append(ArcPointer[Tensor](slice(cond[1], 0, 0, cond[1].shape()[0], ctx)))
    save_safetensors(lat_names, lat_tensors, out_dir + "/latents.safetensors", ctx)
    var t_denoise1 = perf_counter_ns()
    print("  saved final latents ->", out_dir + "/latents.safetensors")
    print("  denoise done (", Float64(t_denoise1 - t_denoise0) / 1.0e9, "s)")

    # ── Result JSON — Stage D banner, persisted. ───────────────────────────
    var result_body = String("{\n")
    result_body += String("  \"partial_mode\":") + json_bool(partial_mode) + String(",\n")
    result_body += String(
        "  \"WARNING\":\"THIS IS NOT A VALID MINIMAX-H3 REF2VA GENERATION."
        " Conditioner STUBBED (fixed-seed random, not the real prompt —"
        " the Qwen3-VL vision tower is not loadable, text_encoder shard 14"
        " of 14 is absent); condition rows "
    ) + (String("INJECTED") if injected else String("SYNTHETIC")) + String(
        ". Plumbing test only.\",\n"
    )
    result_body += String("  \"condition_rows\":\"") + (
        String("injected") if injected else String("synthetic")
    ) + String("\",\n")
    if injected:
        result_body += String("  \"condition_rows_path\":\"") + json_escape(condition_rows_path) + String("\",\n")
    result_body += String("  \"conditioning\":\"stubbed\",\n")
    result_body += String("  \"steps\":") + String(num_steps) + String(",\n")
    result_body += String("  \"seed\":") + String(seed) + String(",\n")
    result_body += String("  \"sequence_length\":") + String(plan.sequence_length()) + String(",\n")
    result_body += String("  \"num_condition_video_rows\":") + String(plan.num_condition_video_rows) + String(",\n")
    result_body += String("  \"num_condition_audio_rows\":") + String(plan.num_condition_audio_rows) + String(",\n")
    result_body += String("  \"num_target_video_rows\":") + String(plan.num_target_video_rows) + String(",\n")
    result_body += String("  \"num_target_audio_rows\":") + String(plan.num_target_audio_rows) + String(",\n")
    result_body += String("  \"latents\":\"") + json_escape(out_dir + String("/latents.safetensors")) + String("\",\n")
    result_body += String(
        "  \"decode\":\"not wired here — identical to t2va's own decode tail;"
        " reuse minimax_h3_t2va's decode_only mode against these latents\"\n"
    )
    result_body += String("}\n")
    write_text_file(out_dir + String("/result.json"), result_body)
    print("  wrote", out_dir + String("/result.json"))

    print("")
    print("  ################################################################")
    print("  # PARTIAL MODE (ref2va): denoise complete, latents saved. THIS IS")
    print("  # NOT A VALID MINIMAX-H3 REF2VA GENERATION — plumbing test only.")
    print("  ################################################################")

    raise Error(
        String("minimax_h3_ref2va: SEAM — decode is out of this unit's scope")
        + " (identical to t2va's own decode tail, not this unit's job). Final"
        " latents were saved to " + out_dir + "/latents.safetensors under"
        " video_state_rows/audio_state_rows (TARGET rows only, patch-token"
        " space) plus condition_video_rows/condition_audio_rows; decode them"
        " with minimax_h3_t2va's decode_only mode."
    )


def main() raises:
    var raw_args = argv()
    var partial_mode = False
    var condition_rows_path = String("")
    var condition_rows_prefix = String("--condition-rows=")
    var args = List[String]()
    for i in range(len(raw_args)):
        var a = String(raw_args[i])
        if a == String("--partial"):
            partial_mode = True
            continue
        if a.startswith(condition_rows_prefix):
            condition_rows_path = String(a[byte = condition_rows_prefix.byte_length() :])
            continue
        args.append(a)

    if len(args) < 3:
        print(
            "usage: minimax_h3_ref2va <prompt_file> <out_dir> [steps=30]"
            " [seed=0] [--partial] [--condition-rows=PATH] [kind:path ...]"
        )
        print(
            "  compiled geometry:", WIDTH, "x", HEIGHT, ",", FRAMES,
            "frames, text_tokens=", TEXT_TOKENS, ", ref_seq_len=", H3_REF_SEQ_LEN,
        )
        print("  reference kinds: image, video, audio (ORDER IS SEMANTIC)")
        print(
            "  WITHOUT --partial: raises at the reference-encode seam; never"
            " touches the GPU (the original SKELETON behavior)."
        )
        print(
            "  WITH --partial: FULL-PLUMBING run — conditioner + condition-row"
            " encode STUBBED, everything else (packed layout, rope, modcache,"
            " real streamed blocks, Euler, save) is real. NOT a valid"
            " generation — see file header PARTIAL MODE."
        )
        return

    var prompt_file = String(args[1])
    var out_dir = String(args[2])
    var steps = DEFAULT_STEPS
    if len(args) >= 4:
        steps = atol(String(args[3]))
    var seed = UInt64(DEFAULT_SEED)
    if len(args) >= 5:
        seed = UInt64(atol(String(args[4])))

    var specs = List[MiniMaxH3ReferenceSpec]()
    for i in range(5, len(args)):
        specs.append(_parse_reference_spec(String(args[i])))

    print(
        "=== MiniMax-H3 ref2va", "(PARTIAL)" if partial_mode else "(SKELETON)",
        "===",
    )
    print("  prompt file:", prompt_file)
    print("  out_dir:", out_dir)
    print("  steps=", steps, " seed=", seed)

    var t0 = perf_counter_ns()
    _preflight_geometry()

    var num_audio_latents = minimax_h3_target_audio_latents(FRAMES)
    print("")
    print("  ── target geometry ──")
    print(
        "  ", WIDTH, "x", HEIGHT, ",", FRAMES, "frames -> latent [",
        NUM_LATENT_FRAMES, ",", LATENT_H, ",", LATENT_W, "],",
        NUM_VIDEO_ROWS, "video rows",
    )
    print(
        "   audio latents:", num_audio_latents, "->",
        num_audio_latents * 2, "audio rows (stereo, channel-major)",
    )
    print("   text tokens (comptime budget):", TEXT_TOKENS)

    print("")
    print("  ── checkpoint plan (Ref2VA, NOT FL2VA) ──")
    _preflight_checkpoint()

    print("")
    print("  ── prompt ──")
    var prompt: String
    with open(prompt_file, "r") as f:
        prompt = f.read()
    print("   ", prompt.byte_length(), "bytes from", prompt_file)
    print(
        "    NOTE: section structure is NOT validated here — build the prompt"
        " with pipeline/minimax_h3_ref_prompt.mojo, which is byte-gated against"
        " the vendor's own request."
    )

    print("")
    print("  ── references (request order) ──")
    _validate_reference_counts(specs)
    var medias = List[MiniMaxH3ReferenceMedia]()
    for i in range(len(specs)):
        print("   [", i, "]", specs[i].kind_name(), specs[i].path)
        medias.append(_probe_reference_media(specs[i]))

    var references = minimax_h3_resolve_references(medias, FRAMES)
    print("")
    print("  ── resolved reference latent geometry ──")
    for i in range(len(references)):
        print(
            "   [", i, "] latent_frames=", references[i].num_latent_frames,
            " latent=", references[i].latent_height, "x",
            references[i].latent_width,
            " audio_latents=", references[i].num_audio_latents,
            " -> video rows", references[i].num_video_rows(),
            ", audio rows", references[i].num_audio_rows(),
        )

    # Text tags: the STUB presentation's REAL layout (Stage B) — a
    # reference's vision-block rows tagged 0 (video), everything else 1
    # (text), built from the declared references. Only the VALUES (the
    # actual embeds) are noise, drawn later in the device stage.
    var text_tags = _minimax_h3_ref2va_stub_conditioning_tags(references, TEXT_TOKENS)

    var plan = minimax_h3_build_ref2va_plan(
        text_tags,
        references,
        NUM_LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        num_audio_latents,
        PATCH_H,
        PATCH_W,
    )
    print("")
    print("  ── packed layout [text | references | target audio | target video] ──")
    print("    sequence_length      ", plan.sequence_length())
    print("    text rows            [0,", plan.num_text_tokens, ")")
    print(
        "    reference rows       [", plan.reference_start(), ",",
        plan.reference_end(), ") —", plan.num_condition_video_rows,
        "video +", plan.num_condition_audio_rows, "audio",
    )
    print(
        "    target audio rows    [", plan.target_audio_start, ",",
        plan.target_video_start, ")",
    )
    print(
        "    target video rows    [", plan.target_video_start, ",",
        plan.sequence_length(), ")",
    )
    print(
        "    NOTE: text-region tags come from the STUBBED presentation's"
        " layout (Stage B) — vision-block rows tagged 0 (video), labels/"
        "prompt tagged 1 (text). Grid sizes are an APPROXIMATION (see"
        " _minimax_h3_ref2va_stub_conditioning_tags); only the VALUES are"
        " noise, not the layout."
    )

    var ts = minimax_h3_ref2va_row_timesteps(plan.layout, Float32(0.75), Float32(0.60))
    print(
        "    row timesteps @ (video 0.75, audio 0.60):", len(ts.values),
        "distinct — condition video max(t, 0.999), condition audio 1.0",
    )

    print("")
    print("  plan resolved in", Float64(perf_counter_ns() - t0) / 1.0e6, "ms")
    _ = sys_system(String("mkdir -p '") + out_dir + "'")

    print("")
    print("  ── stage 2: reference encode", "(--partial: STUBBED)" if partial_mode else "", "──")
    _minimax_h3_ref2va_generate(
        plan, specs, references, partial_mode, condition_rows_path, steps,
        seed, out_dir,
    )
