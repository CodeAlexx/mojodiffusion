# serenitymojo/pipeline/minimax_h3_ref2va.mojo — MiniMax-H3 ref2va
# (omni-reference video+audio), pure Mojo. SKELETON — see SCOPE below.
#
# Modeled on `serenitymojo/pipeline/minimax_h3_t2va.mojo`, which is the house
# pattern for an H3 device pipeline and the file that produced valid video:
# comptime geometry via `get_defined_int`, preflight BEFORE `DeviceContext()`,
# named loud failures, request/result printing.
#
# ── SCOPE OF THIS FILE, RIGHT NOW ────────────────────────────────────────────
# This is the entry point and the PLAN, not the generation. It runs every
# CPU-side stage that exists today, prints exactly what it would load and the
# packed layout it resolved, and then RAISES at the first stage that is not
# built yet. It is expected to raise. It is not a generation.
#
# CRITICALLY: it never constructs a `DeviceContext`. The seam raises before any
# device work would begin, so running this file cannot touch the GPU. When the
# remaining units land, the `DeviceContext()` goes in `_minimax_h3_ref2va_
# generate` below, after the seam that currently raises.
#
# ── PIPELINE STAGES, AND WHAT EXISTS ─────────────────────────────────────────
#   1. media-in           BUILT  `pipeline/minimax_h3_media_in.mojo` (unit A)
#                                probe: pipeline/parity/minimax_h3_media_in_probe
#   2. ref-encode         SEAM   references -> pixel norm -> video VAE (tiled)
#                                -> sample posterior (seed 42) -> fp16 round ->
#                                normalize -> patchify; waveform -> audio VAE
#                                posterior MODE -> normalize. THIS IS WHERE THIS
#                                FILE RAISES.
#   3. ref-pack           BUILT  `models/dit/minimax_h3_ref_geometry.mojo`
#                                (unit B) over the gated
#                                `models/minimax_h3/packing_ref2va.mojo`
#                                probe: models/dit/parity/minimax_h3_ref_geometry_probe
#   4. conditioning       SEAM   ref2va presentation (labels + vision blocks)
#                                needs the Qwen3-VL vision tower's per-reference
#                                token counts; `models/minimax_h3/presentation.
#                                mojo::minimax_h3_ref2va_presentation` is ported
#                                but has nothing to feed it yet.
#   5. ref-denoise        SEAM   four-timestep rows. The row-timestep half is
#                                BUILT and gated (unit B,
#                                `minimax_h3_ref2va_row_timesteps`); the
#                                condition-row noise mix is not.
#   6. prompt             BUILT  `pipeline/minimax_h3_ref_prompt.mojo` (unit C)
#                                probe: pipeline/parity/minimax_h3_ref_prompt_probe
#   7. decode             REUSE  identical to t2va's decode tail.
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
# argv: <prompt_file> <out_dir> [steps=30] [seed=0] [reference specs...]
#   A reference spec is `kind:path`, e.g. `video:/clips/ref.mp4`,
#   `audio:/clips/voice.wav`, `image:/stills/subject.png`. ORDER MATTERS — it
#   labels the references in the prompt and advances the shared rotary clock
#   (packing_ref2va.py:25-29), so the specs are consumed in the order given and
#   are never sorted.

from std.collections import List
from std.sys import argv
from std.sys.defines import get_defined_int
from std.time import perf_counter_ns

from serenitymojo.io.ffi import sys_system
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors

from serenitymojo.models.minimax_h3.packing import MINIMAX_H3_TEXT_TAG
from serenitymojo.models.minimax_h3.packing_ref2va import (
    MINIMAX_H3_MAX_REFERENCE_AUDIOS,
    MINIMAX_H3_MAX_REFERENCE_IMAGES,
    MINIMAX_H3_MAX_REFERENCE_VIDEOS,
    MINIMAX_H3_MAX_REFERENCES,
    MINIMAX_H3_REF_AUDIO,
    MINIMAX_H3_REF_IMAGE,
    MINIMAX_H3_REF_VIDEO,
    MiniMaxH3PreparedReference,
)
from serenitymojo.models.dit.minimax_h3_ref_geometry import (
    MiniMaxH3ReferenceMedia,
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
    minimax_h3_encode_reference_visual_seam,
    minimax_h3_pixel_normalize_frames,
)


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
    """Reference encode, stage 2 — the CPU half, then THE SEAM.

    Runs for real, per reference, in packed order:
      * decode the frames (unit A, ffmpeg -> rgb24 + sidecar)
      * pixel-normalize onto the video VAE's ImageNet convention, channels-first

    Then raises at the VAE forward, which needs a `DeviceContext`. The steps
    AFTER the forward — fp16 round, latent normalize, channel-slowest patchify,
    the 0.999 noise mix, the audio branch's channel-major packing — are BUILT
    and gated host-side (models/vae/parity/minimax_h3_ref_encode_probe.mojo,
    12 bit-exact checks); they are simply unreachable until the forward exists."""
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

        # TRUNCATE TO THE TARGET FRAME COUNT FIRST. `prepare_reference_frames`
        # does `frames[:num_frames]` before anything else (packing_ref2va.py:675),
        # and the packed layout above was resolved on that truncated count — a
        # reference longer than the generated video must not contribute more
        # latent frames than the layout reserved rows for.
        var used_frames = frames.num_frames
        if used_frames > FRAMES:
            used_frames = FRAMES
            print("      truncated to the target's", FRAMES, "frames")
        var frame_bytes = frames.height * frames.width * 3
        var kept = List[UInt8]()
        kept.resize(used_frames * frame_bytes, 0)
        for i in range(used_frames * frame_bytes):
            kept[i] = frames.pixels[i]

        # NOT DONE HERE, and it must be before the encode: the LANCZOS resize
        # onto the reference's own canvas (`prepare_reference_frames` resolves
        # `resolve_canvas_size(width, height)` and resizes frame by frame). The
        # geometry printed above already assumes the canvas; these pixels are
        # still at source resolution. Wiring the resize belongs with the encode.
        var pixels = minimax_h3_pixel_normalize_frames(
            kept, used_frames, frames.height, frames.width
        )
        print(
            "      pixel-normalized ->", len(pixels),
            "f32 values [3,", used_frames, ",", frames.height, ",",
            frames.width, "]",
        )
        # THE SEAM. Raises.
        var _latents = minimax_h3_encode_reference_visual_seam(
            pixels, 3, used_frames, frames.height, frames.width
        )
        _ = len(_latents)
    _ = references
    raise Error(
        "minimax_h3_ref2va: SEAM — no video reference reached the VAE encode."
        " Pass at least one video:PATH reference."
    )


def _minimax_h3_ref2va_generate(
    specs: List[MiniMaxH3ReferenceSpec],
    references: List[MiniMaxH3PreparedReference],
    num_audio_latents: Int,
    steps: Int,
    seed: UInt64,
    out_dir: String,
) raises:
    """The generation tail. The `DeviceContext()` belongs HERE, once the encode
    lands — deliberately not at the call site, so that today this whole file
    runs without ever touching the GPU."""
    _ = num_audio_latents
    _ = steps
    _ = seed
    _ = out_dir
    _minimax_h3_encode_references(specs, references)


def main() raises:
    var args = argv()
    if len(args) < 3:
        print(
            "usage: minimax_h3_ref2va <prompt_file> <out_dir> [steps=30]"
            " [seed=0] [kind:path ...]"
        )
        print(
            "  compiled geometry:", WIDTH, "x", HEIGHT, ",", FRAMES,
            "frames, text_tokens=", TEXT_TOKENS,
        )
        print("  reference kinds: image, video, audio (ORDER IS SEMANTIC)")
        print("  SKELETON: raises at the reference-encode seam; never touches the GPU")
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

    print("=== MiniMax-H3 ref2va (SKELETON) ===")
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

    # Text tags: a placeholder of the compiled budget. The REAL tags come from
    # the ref2va presentation, which tags a reference's vision-block rows 0
    # (video) rather than 1 (text) — stage 4, not built.
    var text_tags = List[Int]()
    for _ in range(TEXT_TOKENS):
        text_tags.append(MINIMAX_H3_TEXT_TAG)

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
        "    NOTE: the text-tag placeholder above is NOT the real presentation"
        " — reference vision blocks are tagged 0 (video), not 1 (text)."
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
    print("  ── stage 2: reference encode ──")
    _minimax_h3_ref2va_generate(
        specs, references, num_audio_latents, steps, seed, out_dir
    )
