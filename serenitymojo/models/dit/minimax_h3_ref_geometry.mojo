# serenitymojo/models/dit/minimax_h3_ref_geometry.mojo — MiniMax-H3 ref2va
# UNIT B: the device path's ref-pack geometry.
#
# ── WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT ────────────────────────────
# The ref2va rotary/row math is ALREADY ported and ALREADY gated bit-exact
# against the vendor, in `models/minimax_h3/packing_ref2va.mojo` (its gate:
# `models/minimax_h3/parity/minimax_h3_ref2va_parity.mojo`, driven by
# `scripts/minimax_h3_ref2va_oracle.py`). This module does NOT re-derive one
# coordinate of it — it CALLS it. Re-deriving gated float math would be strictly
# worse: two implementations to keep in step, one of them ungated.
#
# What is genuinely missing, and is what this file adds, is the layer between a
# REQUEST and that builder:
#
#   1. media geometry -> LATENT geometry, per reference
#        (`minimax_h3_resolve_reference`) — an image reference is sized by
#        `resolve_reference_image_size`, a video reference by the target canvas
#        and the 17n+5 chunking, and neither of those resolutions lives in the
#        packing module.
#   2. sample count -> audio latent count (`minimax_h3_reference_audio_latents`).
#   3. the packed layout's ROW RANGES (`MiniMaxH3Ref2VAPlan`) — which slice of
#      the sequence is text, reference, target audio, target video. The builder
#      returns index LISTS; the pipeline needs contiguous ranges to slice
#      tensors with.
#   4. the four-timestep assignment (`minimax_h3_ref2va_row_timesteps`), whose
#      two CONDITION values are not in the packing module at all — they come
#      from the call site, `before_denoise.py`.
#
# ── THE TWO CONDITION TIMESTEPS (measured, not assumed) ──────────────────────
# `MiniMaxH3SetTimestepsStep.__call__` (before_denoise.py:413-417) — a step
# SHARED by the ref2va block list (modular_blocks_minimax_h3.py:310-318, which
# lists `MiniMaxH3SetTimestepsStep` unchanged between `MiniMaxH3Ref2VAPrepare
# LayoutStep` and `MiniMaxH3Ref2VADenoiseStep`) — calls:
#
#     build_row_timesteps(layout, float(timestep), float(audio_timestep),
#                         max(float(timestep), MINIMAX_H3_KEYFRAME_NOISE_AUG),
#                         1.0)
#
# so, per denoising step:
#   * condition VIDEO rows -> `max(video_timestep, 0.999)`. NOT a constant: it
#     tracks the video schedule and only clamps at the bottom, so at the very
#     first steps (video t near 1.0) the condition rows and the target rows can
#     share ONE timestep value and the distinct-value table collapses from four
#     entries to three. Any code sizing a modulation cache off "ref2va always
#     has four timesteps" is wrong; size it off the returned table.
#   * condition AUDIO rows -> `1.0`, a genuine constant, independent of both
#     schedules.
#
# Note the asymmetry this creates, which is the vendor's own and is reproduced,
# not smoothed over: the audio reference rows carry CLEAN latents (the audio VAE
# posterior MODE, never sampled, never noise-mixed — encoders.py:596-601) yet
# are pinned at timestep 1.0, while the video reference rows carry latents
# genuinely mixed with noise at their augmentation level.
#
# ── AUDIO LATENT COUNT ───────────────────────────────────────────────────────
# The vendor does not compute it — it reads it off the encode
# (`reference.num_audio_latents = latents.shape[1]`, encoders.py:599). So the
# predictor here has to agree with our own audio encoder, and it is derived from
# that encoder's own documented arithmetic rather than from the 40-latents-per-
# second rule of thumb: `audio_encoder.mojo::minimax_h3_audio_encode_trunk`
# right-pads the waveform to a multiple of `hop_length()` (the product of the
# encoder rates, 800) and the trunk then lands on exactly `padded / hop`. Hence
# `ceil(num_samples / 800)`. At 32 kHz that is the 40 latents/s the rotary clock
# assumes, but the CEILING is the part that matters: a waveform that is not a
# whole number of hops still contributes a full trailing latent, and rounding
# instead of ceiling would silently drop a row and shift every later reference
# on the shared clock.
#
# `minimax_h3_resolve_reference` therefore PREDICTS the geometry so the layout
# can be built before the VAEs run; `minimax_h3_build_ref2va_plan` also accepts
# already-encoded references, which is the path to take once the encoder has
# run and its shapes are known. When both are available the encoder's shapes
# WIN — this predictor is for planning, not for truth.

from std.collections import List

from serenitymojo.models.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    MiniMaxH3PackedSequence,
    MiniMaxH3RowTimesteps,
    minimax_h3_build_row_timesteps,
    minimax_h3_resolve_canvas_size,
    minimax_h3_video_latent_num_frames,
)
from serenitymojo.models.minimax_h3.packing_ref2va import (
    MINIMAX_H3_REF_AUDIO,
    MINIMAX_H3_REF_IMAGE,
    MINIMAX_H3_REF_VIDEO,
    MiniMaxH3PreparedReference,
    minimax_h3_build_ref2va_motion_context_packed_sequence,
    minimax_h3_build_ref2va_packed_sequence,
    minimax_h3_resolve_reference_image_size,
    minimax_h3_trim_reference_num_frames,
)
from serenitymojo.models.minimax_h3.vae_geometry import (
    MINIMAX_H3_VAE_SPATIAL_RATIO,
)


# Conditioning rows are not clean: the released model noises video conditioning
# latents to t = 0.999 and runs them there for every denoising step
# (packing.py:84, MINIMAX_H3_KEYFRAME_NOISE_AUG).
comptime MINIMAX_H3_KEYFRAME_NOISE_AUG = Float32(0.999)

# Audio reference rows are pinned at a hard 1.0 (before_denoise.py:417).
comptime MINIMAX_H3_CONDITION_AUDIO_TIMESTEP = Float32(1.0)

# The audio VAE's hop: the product of its encoder rates. Cross-checked against
# `audio_encoder.mojo::MiniMaxH3AudioEncoderConfig.hop_length()` by this
# module's probe rather than trusted as a magic number.
comptime MINIMAX_H3_AUDIO_HOP = 800

# The transformer's video patch. `MiniMaxH3PreparedReference.num_video_rows`
# hardcodes the same 2x2 (packing_ref2va.py:379 does too).
comptime MINIMAX_H3_PATCH_H = 2
comptime MINIMAX_H3_PATCH_W = 2


@fieldwise_init
struct MiniMaxH3ReferenceMedia(Copyable, Movable):
    """One reference as the REQUEST describes it — media geometry, pre-encode.

    This is the input to `minimax_h3_resolve_reference`, which turns it into the
    `MiniMaxH3PreparedReference` the gated packing builder consumes.

    Attributes:
        kind: `MINIMAX_H3_REF_IMAGE` / `_VIDEO` / `_AUDIO`.
        pixel_height, pixel_width: source pixel size of an image or video
            reference, BEFORE any resize. Ignored for an audio reference.
        num_frames: source frame count of a video reference, already on H3's
            24 fps grid (i.e. after
            `minimax_h3_resample_reference_repeats` has been applied).
            Ignored otherwise.
        num_audio_samples: soundtrack length in samples AT THE AUDIO VAE'S OWN
            RATE. Zero means this reference contributes no audio rows — which
            is the normal case for an image, and the case for a video whose
            container carried no audio stream. Required non-zero for an audio
            reference."""

    var kind: Int
    var pixel_height: Int
    var pixel_width: Int
    var num_frames: Int
    var num_audio_samples: Int


def minimax_h3_reference_audio_latents(
    num_samples: Int, hop: Int = MINIMAX_H3_AUDIO_HOP
) raises -> Int:
    """Audio latents the VAE produces for `num_samples` samples: ceil(n / hop).

    Mirrors `audio_encoder.mojo::minimax_h3_audio_encode_trunk`'s own
    `padded_n = ((n + hop - 1) // hop) * hop` followed by a trunk that outputs
    `padded_n / hop`. Zero samples give zero latents."""
    if hop < 1:
        raise Error("minimax_h3_ref_geometry: audio hop must be positive")
    if num_samples < 0:
        raise Error("minimax_h3_ref_geometry: sample count must not be negative")
    if num_samples == 0:
        return 0
    return (num_samples + hop - 1) // hop


def minimax_h3_resolve_reference(
    media: MiniMaxH3ReferenceMedia, target_num_frames: Int
) raises -> MiniMaxH3PreparedReference:
    """Media geometry -> latent geometry, for one reference.

    `target_num_frames` is the GENERATED video's frame count (a 17n+5 value): a
    video reference is truncated to it before encoding
    (`prepare_reference_frames`, packing_ref2va.py:675) and only then snapped
    down to a 17n+5 the VAE encodes without padding
    (`trim_reference_num_frames`, :822).

    An IMAGE gets its own 2048-short-edge resolution and encodes to exactly one
    latent frame — the vendor runs the spatial encoder alone on it
    (`_encode_clip`, encoders.py:578-582).

    A VIDEO gets MiniMax-H3's own 768 canvas, resolved from its OWN aspect
    ratio: `prepare_reference_frames` calls `resolve_canvas_size(width, height)`
    on the reference's shape, NOT the target's — "a reference never binds the
    target geometry" (packing_ref2va.py:31-33), so a reference at a different
    aspect ratio than the request keeps its own spatial grid."""
    if media.kind == MINIMAX_H3_REF_AUDIO:
        if media.num_audio_samples < 1:
            raise Error(
                "minimax_h3_ref_geometry: an audio reference must carry samples"
            )
        return MiniMaxH3PreparedReference(
            MINIMAX_H3_REF_AUDIO,
            1,
            0,
            0,
            minimax_h3_reference_audio_latents(media.num_audio_samples),
        )

    if media.pixel_height < 1 or media.pixel_width < 1:
        raise Error(
            String("minimax_h3_ref_geometry: a visual reference needs a positive")
            + " size, got " + String(media.pixel_width) + "x"
            + String(media.pixel_height)
        )

    var num_audio_latents = minimax_h3_reference_audio_latents(
        media.num_audio_samples
    )

    if media.kind == MINIMAX_H3_REF_IMAGE:
        var size = minimax_h3_resolve_reference_image_size(
            media.pixel_width, media.pixel_height
        )
        return MiniMaxH3PreparedReference(
            MINIMAX_H3_REF_IMAGE,
            1,
            size[0] // MINIMAX_H3_VAE_SPATIAL_RATIO,
            size[1] // MINIMAX_H3_VAE_SPATIAL_RATIO,
            num_audio_latents,
        )

    if media.kind != MINIMAX_H3_REF_VIDEO:
        raise Error(
            String("minimax_h3_ref_geometry: unknown reference kind ")
            + String(media.kind)
        )

    if media.num_frames < 1:
        raise Error(
            "minimax_h3_ref_geometry: a video reference must have at least one"
            " frame"
        )
    if target_num_frames < 1:
        raise Error(
            "minimax_h3_ref_geometry: target_num_frames must be positive"
        )
    var capped = media.num_frames
    if capped > target_num_frames:
        capped = target_num_frames
    var encoded_frames = minimax_h3_trim_reference_num_frames(capped)
    # SHORT-REFERENCE TRAP (the vendor's own, reproduced as a loud failure
    # rather than a silent mis-prediction). `trim_reference_num_frames` floors
    # its chunk count at 1 (`max(1, (n - 5) // 17)`, packing_ref2va.py:838), so
    # for any reference under 22 frames it returns 22 — MORE frames than the
    # reference has. The vendor then slices `frames[:22]` on a shorter array
    # (encoders.py:583) and hands the VAE a frame count that is not 17n+5 at
    # all, so the latent geometry this function would predict is not the
    # geometry that encode produces. There is no correct prediction to make
    # here; the request itself is unrepresentable.
    if encoded_frames > capped:
        raise Error(
            String("minimax_h3_ref_geometry: video reference carries ")
            + String(capped) + " frames but the VAE's 17n+5 chunking needs "
            + String(encoded_frames)
            + " — MiniMax-H3 cannot encode a reference shorter than 22 frames"
            " (the vendor's own trim floors at one 17-frame chunk). Supply a"
            " longer reference, or pad it to 22 frames before this call."
        )
    var canvas = minimax_h3_resolve_canvas_size(
        Float64(media.pixel_width), Float64(media.pixel_height)
    )
    return MiniMaxH3PreparedReference(
        MINIMAX_H3_REF_VIDEO,
        minimax_h3_video_latent_num_frames(encoded_frames),
        canvas.height // MINIMAX_H3_VAE_SPATIAL_RATIO,
        canvas.width // MINIMAX_H3_VAE_SPATIAL_RATIO,
        num_audio_latents,
    )


def minimax_h3_resolve_references(
    medias: List[MiniMaxH3ReferenceMedia], target_num_frames: Int
) raises -> List[MiniMaxH3PreparedReference]:
    """`minimax_h3_resolve_reference` over a request's references, IN ORDER.

    Order is preserved and is load-bearing twice over: it labels the references
    in the prompt presentation and it advances the shared rotary clock
    (packing_ref2va.py:25-29). Never sort this."""
    var out = List[MiniMaxH3PreparedReference]()
    for i in range(len(medias)):
        out.append(minimax_h3_resolve_reference(medias[i], target_num_frames))
    return out^


@fieldwise_init
struct MiniMaxH3Ref2VAPlan(Copyable, Movable):
    """The packed ref2va layout plus the contiguous ROW RANGES over it.

    The four regions are contiguous and in this order — the whole point of the
    ref2va layout (packing_ref2va.py:22):

        [ text | reference blocks | target audio | target video ]

    `[text_start, text_end)` is `[0, num_text_tokens)`, the reference region is
    `[text_end, target_audio_start)`, and the two target regions follow. The
    reference region is NOT internally contiguous by modality — a video
    reference interleaves its soundtrack rows ahead of its own video rows — so
    per-modality addressing inside it must go through
    `layout.audio_indices` / `layout.video_indices`, whose leading
    `num_condition_*_rows` entries are exactly the reference rows."""

    var layout: MiniMaxH3PackedSequence
    var num_text_tokens: Int
    var num_condition_video_rows: Int
    var num_condition_audio_rows: Int
    var num_target_audio_rows: Int
    var num_target_video_rows: Int
    var target_audio_start: Int
    var target_video_start: Int

    def sequence_length(self) -> Int:
        return self.layout.sequence_length

    def reference_start(self) -> Int:
        return self.num_text_tokens

    def reference_end(self) -> Int:
        return self.target_audio_start


def minimax_h3_build_ref2va_plan(
    text_token_tags: List[Int],
    references: List[MiniMaxH3PreparedReference],
    num_latent_frames: Int,
    latent_height: Int,
    latent_width: Int,
    num_audio_latents: Int,
    patch_h: Int = MINIMAX_H3_PATCH_H,
    patch_w: Int = MINIMAX_H3_PATCH_W,
) raises -> MiniMaxH3Ref2VAPlan:
    """Build the ref2va layout and its row ranges.

    The layout itself comes from the GATED builder
    (`packing_ref2va.mojo::minimax_h3_build_ref2va_packed_sequence`); this adds
    only the range bookkeeping, which is integer arithmetic over row counts —
    no float coordinate is computed here."""
    var layout = minimax_h3_build_ref2va_packed_sequence(
        text_token_tags,
        references,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_h,
        patch_w,
    )
    var num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    var num_target_video_rows = (
        num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    )
    var target_video_start = layout.sequence_length - num_target_video_rows
    var target_audio_start = target_video_start - num_target_audio_rows
    if target_audio_start < len(text_token_tags):
        raise Error(
            "minimax_h3_ref_geometry: derived target row ranges overlap the"
            " text region — reference geometry and target geometry disagree"
        )
    return MiniMaxH3Ref2VAPlan(
        layout^,
        len(text_token_tags),
        layout.num_condition_video_rows,
        layout.num_condition_audio_rows,
        num_target_audio_rows,
        num_target_video_rows,
        target_audio_start,
        target_video_start,
    )


def minimax_h3_build_ref2va_motion_context_plan(
    text_token_tags: List[Int],
    references: List[MiniMaxH3PreparedReference],
    num_latent_frames: Int,
    latent_height: Int,
    latent_width: Int,
    num_audio_latents: Int,
    context_frames: Int,
    source_audio_overhang: Float64,
    patch_h: Int = MINIMAX_H3_PATCH_H,
    patch_w: Int = MINIMAX_H3_PATCH_W,
) raises -> MiniMaxH3Ref2VAPlan:
    """Build ordered Ref2VA blocks plus a pinned previous-clip A/V tail."""
    var layout = minimax_h3_build_ref2va_motion_context_packed_sequence(
        text_token_tags,
        references,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_h,
        patch_w,
        context_frames,
        source_audio_overhang,
    )
    var num_target_audio_rows = num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
    var num_target_video_rows = (
        num_latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    )
    var target_video_start = layout.sequence_length - num_target_video_rows
    var target_audio_start = target_video_start - num_target_audio_rows
    if target_audio_start < len(text_token_tags):
        raise Error(
            "minimax_h3_ref_geometry: combined target rows overlap text"
        )
    return MiniMaxH3Ref2VAPlan(
        layout^,
        len(text_token_tags),
        layout.num_condition_video_rows,
        layout.num_condition_audio_rows,
        num_target_audio_rows,
        num_target_video_rows,
        target_audio_start,
        target_video_start,
    )


def minimax_h3_ref2va_row_timesteps(
    layout: MiniMaxH3PackedSequence,
    video_timestep: Float32,
    audio_timestep: Float32,
) raises -> MiniMaxH3RowTimesteps:
    """The four ref2va row timesteps for one denoising step.

    Wraps the gated `minimax_h3_build_row_timesteps` with the two CONDITION
    values the ref2va call site supplies (before_denoise.py:413-417):
    `max(video_timestep, 0.999)` for the video reference rows and a constant
    `1.0` for the audio reference rows.

    The returned table has AT MOST four entries and can legitimately have
    three (when `video_timestep >= 0.999` the condition-video value collapses
    onto it) or fewer. Size anything downstream off `len(values)`."""
    var condition_video = video_timestep
    if condition_video < MINIMAX_H3_KEYFRAME_NOISE_AUG:
        condition_video = MINIMAX_H3_KEYFRAME_NOISE_AUG
    return minimax_h3_build_row_timesteps(
        layout,
        video_timestep,
        audio_timestep,
        condition_video,
        MINIMAX_H3_CONDITION_AUDIO_TIMESTEP,
    )


def minimax_h3_target_audio_latents(num_frames: Int) -> Int:
    """Audio latents covering the GENERATED video, on the 40 Hz latent grid.

    Distinct from `minimax_h3_reference_audio_latents`, which counts a
    reference's latents from its SAMPLES. The target has no waveform yet, so
    its count comes from the frame count and the shared clock: 24 fps video and
    40 latents/s audio."""
    var seconds = Float64(num_frames) / Float64(MINIMAX_H3_FPS)
    var scaled = seconds * 40.0
    var lower = Float64(Int(scaled))
    var frac = scaled - lower
    if frac > 0.5:
        return Int(lower) + 1
    if frac < 0.5:
        return Int(lower)
    var lower_int = Int(lower)
    return lower_int if lower_int % 2 == 0 else lower_int + 1
