# serenitymojo/pipeline/minimax_h3_ref_frames.mojo — put a MiniMax-H3 ref2va
# VIDEO REFERENCE onto its own canvas. Host only: no Tensor, no DeviceContext,
# no GPU. This is the unit `pipeline/minimax_h3_media_in.mojo`'s header
# promises: "the resampler is applied once, later, by the unit that also owns
# the pixel resize" — this is that unit.
#
# ── WHAT THE VENDOR DOES (packing_ref2va.py + before_encoder.py) ─────────────
# A video reference goes through two passes, in this order
# (before_encoder.py:371-372):
#   1. `resample_reference_frames(frames, fps)` (packing_ref2va.py:622-651):
#      constant-frame-rate resample onto H3's 24 fps by DROPPING and
#      DUPLICATING whole frames — ffmpeg's `fps` filter semantics, source frame
#      `i` lands on slot `floor(i * 24/fps + 0.5)`. Identity (the same array)
#      at 24 fps. The INDEX MATH is already ported and gated as
#      `models/minimax_h3/packing_ref2va.mojo::
#      minimax_h3_resample_reference_repeats`; this file only APPLIES it to
#      pixels.
#   2. `prepare_reference_frames(frames, num_frames)`
#      (packing_ref2va.py:654-681): truncate to the TARGET's frame count
#      (`frames[:num_frames]`, :675), resolve the canvas from the reference's
#      OWN aspect ratio (`resolve_canvas_size(width, height)`, :676 — 768
#      short edge, 768*1344 area cap, both axes rounded to a multiple of 32,
#      RAISES outside 1:4..4:1), pass frames already at the canvas through
#      UNTOUCHED (:677-678), and otherwise resize FRAME BY FRAME with
#      `Image.fromarray(frame).resize((width, height),
#      Image.Resampling.LANCZOS)` (:679-681) — PIL LANCZOS on UINT8
#      channels-last RGB, BEFORE any float normalization. The /255 +
#      mean/std normalization happens later, on the resized uint8 pixels
#      (encoders.py:576).
#
# So the resize KERNEL is exactly the one the keyframe lane already ported and
# gates BIT-EXACT against Pillow — `pipeline/minimax_h3_keyframe_image.mojo::
# minimax_h3_lanczos_resize` (Pillow's own ImagingResample 8bpc fixed-point) —
# and the canvas LAW is exactly the one `models/minimax_h3/packing.mojo::
# minimax_h3_resolve_canvas_size` already gates. Nothing numeric is re-derived
# here; this file is composition plus the frame loop.
#
# NOT ffmpeg's scaler: `-vf scale=...:flags=lanczos` is swscale, a DIFFERENT
# filter (own coefficient table, own intermediate precision) — see the
# keyframe_image.mojo header. The vendor resizes with PIL, so this does too.
#
# Gate: pipeline/parity/minimax_h3_ref_frames_probe.mojo, BIT-EXACT against
# the vendor's own `prepare_reference_frames` / `resample_reference_frames`
# (exec'd out of the reference tree by scripts/minimax_h3_ref_frames_oracle.py)
# on real 1920x1080 frames. Bit-exact is the right bar: every pass here is
# uint8-in/uint8-out integer arithmetic — there is no float tolerance to
# justify because the vendor's own dtype at this point in the chain is uint8.

from std.collections import List

from serenitymojo.models.minimax_h3.packing import (
    MINIMAX_H3_FPS,
    minimax_h3_resolve_canvas_size,
)
from serenitymojo.models.minimax_h3.packing_ref2va import (
    minimax_h3_resample_reference_repeats,
)
from serenitymojo.pipeline.minimax_h3_keyframe_image import (
    MiniMaxH3RgbImage,
    minimax_h3_lanczos_resize,
)
from serenitymojo.pipeline.minimax_h3_media_in import MiniMaxH3RgbFrames


def _validate_frames(frames: MiniMaxH3RgbFrames) raises:
    if frames.num_frames < 1 or frames.height < 1 or frames.width < 1:
        raise Error("minimax_h3_ref_frames: frame geometry must be positive")
    var expected = frames.num_frames * frames.height * frames.width * 3
    if len(frames.pixels) != expected:
        raise Error(
            String("minimax_h3_ref_frames: buffer holds ")
            + String(len(frames.pixels)) + " bytes, geometry implies "
            + String(expected)
        )


def minimax_h3_resample_reference_frames(
    frames: MiniMaxH3RgbFrames, max_frames: Int = -1
) raises -> MiniMaxH3RgbFrames:
    """`resample_reference_frames(frames, fps)` (packing_ref2va.py:622-651):
    the reference onto H3's 24 fps grid by whole-frame drop/duplicate.

    Applies the GATED `minimax_h3_resample_reference_repeats` index math to
    the pixels — frame `i` appears `repeats[i]` times (possibly 0). Identity
    for frames already at 24 fps, exactly as the vendor returns the input
    array itself.

    `max_frames` caps how many OUTPUT slots are materialized (`-1` = all).
    That is a pure output truncation — identical to the vendor's full
    resample followed by `[:max_frames]` — so the caller that will truncate
    to the target frame count anyway (`prepare_reference_frames` does) can
    skip materializing the tail of a long reference. The parity probe checks
    the capped output against the vendor's full output sliced."""
    _validate_frames(frames)

    var frame_bytes = frames.height * frames.width * 3
    if frames.fps == Float64(MINIMAX_H3_FPS):
        # Already on the grid — the vendor's identity route (no resample pass).
        if max_frames < 0 or max_frames >= frames.num_frames:
            return frames.copy()
        var kept = List[UInt8](capacity=max_frames * frame_bytes)
        for i in range(max_frames * frame_bytes):
            kept.append(frames.pixels[i])
        return MiniMaxH3RgbFrames(
            kept^, max_frames, frames.height, frames.width, frames.fps
        )

    var repeats = minimax_h3_resample_reference_repeats(
        frames.num_frames, frames.fps
    )
    var total = 0
    for i in range(len(repeats)):
        total += repeats[i]
    if total < 1:
        raise Error(
            "minimax_h3_ref_frames: the 24 fps resample leaves no frames —"
            " the reference is shorter than one output slot"
        )
    var out_frames = total
    if max_frames >= 0 and max_frames < out_frames:
        out_frames = max_frames

    var out = List[UInt8](capacity=out_frames * frame_bytes)
    var emitted = 0
    for i in range(frames.num_frames):
        for _ in range(repeats[i]):
            if emitted == out_frames:
                break
            var src = i * frame_bytes
            for b in range(frame_bytes):
                out.append(frames.pixels[src + b])
            emitted += 1
        if emitted == out_frames:
            break
    return MiniMaxH3RgbFrames(
        out^, out_frames, frames.height, frames.width, Float64(MINIMAX_H3_FPS)
    )


def minimax_h3_prepare_reference_frames(
    frames: MiniMaxH3RgbFrames, num_frames: Int
) raises -> MiniMaxH3RgbFrames:
    """`prepare_reference_frames(frames, num_frames)`
    (packing_ref2va.py:654-681), op for op: truncate to the target's frame
    count, then LANCZOS-resize every frame onto the canvas the reference's
    OWN aspect ratio resolves to (768 short edge — a reference never binds
    the target geometry).

    The input must already be on the 24 fps grid
    (`minimax_h3_resample_reference_frames` first — the vendor calls the two
    in exactly that order, before_encoder.py:371-372); a clip still carrying
    another rate is refused rather than silently resized at the wrong frame
    selection.

    Raises through `minimax_h3_resolve_canvas_size` for everything the canvas
    law does not cover (aspect ratio outside 1:4..4:1), exactly as the
    keyframe lane does. Frames already at the canvas pass through untouched —
    the vendor's parity-exact route for pre-sized references."""
    _validate_frames(frames)
    if num_frames < 1:
        raise Error(
            "minimax_h3_ref_frames: the target frame count must be positive"
        )
    if frames.fps != Float64(MINIMAX_H3_FPS):
        raise Error(
            String("minimax_h3_ref_frames: frames carry ") + String(frames.fps)
            + " fps — resample onto the 24 fps grid first"
            " (minimax_h3_resample_reference_frames); the vendor resamples"
            " BEFORE it resizes (before_encoder.py:371-372)"
        )

    # `frames = frames[:num_frames]` (packing_ref2va.py:675).
    var kept_frames = frames.num_frames
    if kept_frames > num_frames:
        kept_frames = num_frames
    var frame_bytes = frames.height * frames.width * 3

    # `resolve_canvas_size(frames.shape[2], frames.shape[1])` — (width, height)
    # of the reference itself (:676). Raises outside 1:4..4:1.
    var canvas = minimax_h3_resolve_canvas_size(
        Float64(frames.width), Float64(frames.height)
    )

    if canvas.height == frames.height and canvas.width == frames.width:
        # Already the canvas: flow through untouched (:677-678).
        var kept = List[UInt8](capacity=kept_frames * frame_bytes)
        for i in range(kept_frames * frame_bytes):
            kept.append(frames.pixels[i])
        return MiniMaxH3RgbFrames(
            kept^, kept_frames, frames.height, frames.width, frames.fps
        )

    # Frame by frame through the PIL-exact LANCZOS (:679-681) — the very
    # kernel `prepare_reference_image` uses, already gated bit-exact.
    var out_bytes = canvas.height * canvas.width * 3
    var out = List[UInt8](capacity=kept_frames * out_bytes)
    for f in range(kept_frames):
        var src = f * frame_bytes
        var one = List[UInt8](capacity=frame_bytes)
        for b in range(frame_bytes):
            one.append(frames.pixels[src + b])
        var image = MiniMaxH3RgbImage(one^, frames.height, frames.width)
        var resized = minimax_h3_lanczos_resize(
            image, canvas.height, canvas.width
        )
        for b in range(out_bytes):
            out.append(resized.pixels[b])
    return MiniMaxH3RgbFrames(
        out^, kept_frames, canvas.height, canvas.width, frames.fps
    )
