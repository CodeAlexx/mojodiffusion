# serenitymojo/pipeline/parity/minimax_h3_ref2va_presentation_probe.mojo
#
# GATE: the REAL ref2va conditioner presentation, id-exact and bit-exact
# against the vendor's own chain on ONE concrete request — the trimmed
# reference clip + the vendor 768p prompt, i.e. exactly the request
# pipeline/minimax_h3_ref2va.mojo runs.
#
# Oracle: scripts/minimax_h3_ref2va_conditioning_oracle.py — the vendor's own
# `resample_reference_frames` -> `prepare_reference_frames` ->
# `sample_reference_video_frames` -> the REAL Qwen3-VL video processor ->
# `build_ref2va_presentation` with the REAL tokenizer, on frames decoded with
# the EXACT ffmpeg command this side uses (byte-identical input pixels).
#
# What is gated, stage by stage (a failure lands on a stage, not on "the
# presentation"):
#   [1] conditioner frame sampling — indices + block timestamps (f64 exact)
#   [2] video processor grid `[t, h, w]` (int exact)
#   [3] token ids + modality tags (id-exact, full stream)
#   [4] `<|video_pad|>` positions == the oracle's video-tagged pad rows
#   [5] pixel patch rows vs `pixel_values_videos` (BIT-exact f32)
#
# Host only: no Tensor, no DeviceContext, no GPU.
#
# Run:
#   /home/alex/torchref/venv/bin/python scripts/minimax_h3_ref2va_conditioning_oracle.py
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_ref2va_presentation_probe.mojo \
#     -o <scratch>/h3_ref2va_pres_probe && <scratch>/h3_ref2va_pres_probe

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.packing_ref2va import (
    minimax_h3_sample_reference_video_frames,
)
from serenitymojo.models.minimax_h3.presentation import (
    MINIMAX_H3_REF_AUDIO,
    MINIMAX_H3_REF_VIDEO,
    MINIMAX_H3_VIDEO_PAD,
    MiniMaxH3PresentationReference,
    minimax_h3_ref2va_presentation,
    minimax_h3_special_id,
)
from serenitymojo.pipeline.minimax_h3_media_in import (
    minimax_h3_ffmpeg_extract_rgb,
)
from serenitymojo.pipeline.minimax_h3_ref_frames import (
    minimax_h3_prepare_reference_frames,
    minimax_h3_resample_reference_frames,
)
from serenitymojo.pipeline.minimax_h3_vision_preprocess import (
    minimax_h3_video_patch_grid,
    minimax_h3_vision_video_patch_rows,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer

comptime ORACLE = "/home/alex/mojodiffusion/output/minimax_h3_ref2va/conditioning_oracle.safetensors"
comptime CLIP = "/home/alex/mojodiffusion/output/h3_ref2va_media/ref_video_trim.mp4"
comptime PROMPT_FILE = "/home/alex/mojodiffusion/output/minimax_h3_prompts/ref2va_vendor_768p.txt"
comptime PROCESSOR_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/Ref2VA/processor"
comptime SCRATCH = "/home/alex/mojodiffusion/output/minimax_h3_ref2va/probe_trim"
comptime TARGET_FRAMES = 22
comptime MERGE_UNIT = 4


def _read_i64(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.I64:
        raise Error(String("_read_i64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int64]()
    var out = List[Int](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def _read_f64(ref st: SafeTensors, name: String) raises -> List[Float64]:
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.F64:
        raise Error(String("_read_f64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float64]()
    var out = List[Float64](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _read_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.F32:
        raise Error(String("_read_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def main() raises:
    print("MiniMax-H3 ref2va REAL-presentation gate — vs the vendor's own chain")
    var st = SafeTensors.open(String(ORACLE))
    var pass_all = True
    var checks = 0

    # ── decode + prepare, through the GATED Mojo chain ──────────────────────
    var frames = minimax_h3_ffmpeg_extract_rgb(
        String(CLIP), String(SCRATCH) + ".rgb", String(SCRATCH) + ".rgb.json",
        String(SCRATCH) + ".probe",
    )
    print(
        "  decoded", frames.num_frames, "frames", frames.width, "x",
        frames.height, "@", frames.fps,
    )
    var on_grid = minimax_h3_resample_reference_frames(frames, TARGET_FRAMES)
    var prepared = minimax_h3_prepare_reference_frames(on_grid, TARGET_FRAMES)
    var want_shape = _read_i64(st, String("prepared_shape"))
    checks += 1
    if (
        prepared.num_frames != want_shape[0]
        or prepared.height != want_shape[1]
        or prepared.width != want_shape[2]
    ):
        pass_all = False
        print(
            "  FAIL prepared geometry:", prepared.num_frames, "x",
            prepared.height, "x", prepared.width, "vs oracle", want_shape[0],
            "x", want_shape[1], "x", want_shape[2],
        )
    else:
        print(
            "  ok   prepared:", prepared.num_frames, "frames",
            prepared.width, "x", prepared.height,
        )

    # ── [1] conditioner frame sampling ──────────────────────────────────────
    var sampled = minimax_h3_sample_reference_video_frames(prepared.num_frames)
    var want_ts = _read_f64(st, String("block_timestamps"))
    checks += 1
    var ts_ok = len(sampled.block_timestamps) == len(want_ts)
    if ts_ok:
        for i in range(len(want_ts)):
            if sampled.block_timestamps[i] != want_ts[i]:
                ts_ok = False
    if ts_ok:
        print(
            "  ok   [1] sampling:", len(sampled.indices), "frames,",
            len(sampled.block_timestamps), "block timestamp(s), f64-exact",
        )
    else:
        pass_all = False
        print("  FAIL [1] block timestamps differ from the oracle")

    # ── [2] video processor grid ────────────────────────────────────────────
    var grid = minimax_h3_video_patch_grid(
        prepared.height, prepared.width, len(sampled.indices)
    )
    var want_grid = _read_i64(st, String("video_grid_thw"))
    checks += 1
    if (
        grid[0] != want_grid[0] or grid[1] != want_grid[1]
        or grid[2] != want_grid[2]
    ):
        pass_all = False
        print(
            "  FAIL [2] grid [", grid[0], ",", grid[1], ",", grid[2],
            "] vs oracle [", want_grid[0], ",", want_grid[1], ",",
            want_grid[2], "]",
        )
    else:
        print(
            "  ok   [2] video grid [", grid[0], ",", grid[1], ",", grid[2],
            "] == the processor's video_grid_thw",
        )
    var tokens_per_block = grid[1] * grid[2] // MERGE_UNIT

    # ── [3] the presentation, id-exact ──────────────────────────────────────
    # `merge_additional_special_tokens` is LOAD-BEARING here: the vendor
    # prompt carries a `<d>...</d>` dialogue line, and H3's `<d>`/`</d>` live
    # only in tokenizer_config.json (ids 151669/151670). Without the merge
    # they BPE-split and the stream is 2 tokens long — measured, and exactly
    # what this gate exists to catch.
    var tokenizer = Qwen3Tokenizer(String(PROCESSOR_DIR) + "/tokenizer.json")
    _ = tokenizer.merge_additional_special_tokens(
        String(PROCESSOR_DIR) + "/tokenizer_config.json"
    )
    var prompt: String
    with open(String(PROMPT_FILE), "r") as f:
        prompt = f.read()

    var refs = List[MiniMaxH3PresentationReference]()
    refs.append(
        MiniMaxH3PresentationReference(
            MINIMAX_H3_REF_VIDEO, True, sampled.block_timestamps.copy(),
            tokens_per_block,
        )
    )
    refs.append(
        MiniMaxH3PresentationReference(
            MINIMAX_H3_REF_AUDIO, True, List[Float64](), 0
        )
    )
    var presentation = minimax_h3_ref2va_presentation(tokenizer, prompt, refs)

    var want_ids = _read_i64(st, String("token_ids"))
    var want_tags = _read_i64(st, String("token_tags"))
    checks += 1
    if len(presentation.ids) != len(want_ids):
        pass_all = False
        print(
            "  FAIL [3] presentation length", len(presentation.ids),
            "vs oracle", len(want_ids),
        )
    else:
        var id_bad = 0
        var tag_bad = 0
        var first_bad = -1
        for i in range(len(want_ids)):
            if presentation.ids[i] != want_ids[i]:
                id_bad += 1
                if first_bad < 0:
                    first_bad = i
            if presentation.tags[i] != want_tags[i]:
                tag_bad += 1
                if first_bad < 0:
                    first_bad = i
        if id_bad == 0 and tag_bad == 0:
            print(
                "  ok   [3] presentation ID-EXACT:", len(want_ids),
                "token ids + tags match the vendor's build_ref2va_presentation",
            )
        else:
            pass_all = False
            print(
                "  FAIL [3]", id_bad, "id mismatches,", tag_bad,
                "tag mismatches, first at", first_bad,
            )

    # ── [4] the video_pad positions (what the tower splices into) ───────────
    var video_pad_id = minimax_h3_special_id(
        tokenizer, String(MINIMAX_H3_VIDEO_PAD)
    )
    var pads = List[Int]()
    for i in range(len(presentation.ids)):
        if presentation.ids[i] == video_pad_id:
            pads.append(i)
    var want_pads = 0
    for i in range(len(want_ids)):
        if want_ids[i] == video_pad_id:
            want_pads += 1
    checks += 1
    var expected_pads = grid[0] * tokens_per_block
    if len(pads) != want_pads or len(pads) != expected_pads:
        pass_all = False
        print(
            "  FAIL [4] video_pad rows:", len(pads), "here,", want_pads,
            "in the oracle,", expected_pads, "implied by the grid",
        )
    else:
        print(
            "  ok   [4]", len(pads), "video_pad rows == grid t*h*w/4 ==",
            "the oracle's own count",
        )

    # ── [5] pixel patch rows, BIT-exact vs pixel_values_videos ──────────────
    var frame_bytes = prepared.height * prepared.width * 3
    var sampled_pixels = List[UInt8](
        capacity=len(sampled.indices) * frame_bytes
    )
    for i in range(len(sampled.indices)):
        var src = sampled.indices[i] * frame_bytes
        for b in range(frame_bytes):
            sampled_pixels.append(prepared.pixels[src + b])
    var rows = minimax_h3_vision_video_patch_rows(
        sampled_pixels, len(sampled.indices), prepared.height, prepared.width
    )
    var want_rows = _read_f32(st, String("pixel_values_videos"))
    checks += 1
    if len(rows) != len(want_rows):
        pass_all = False
        print(
            "  FAIL [5] patch rows hold", len(rows), "floats, oracle",
            len(want_rows),
        )
    else:
        var bad = 0
        for i in range(len(rows)):
            if rows[i] != want_rows[i]:
                bad += 1
        if bad == 0:
            print(
                "  ok   [5] patch rows BIT-EXACT over", len(want_rows),
                "f32 values vs the processor's pixel_values_videos",
            )
        else:
            pass_all = False
            print("  FAIL [5]", bad, "of", len(want_rows), "values differ")

    print("")
    if pass_all:
        print("PASS —", checks, "checks, all green")
    else:
        raise Error("minimax_h3_ref2va_presentation_probe: FAIL — see above")
