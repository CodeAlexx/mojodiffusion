# serenitymojo/pipeline/parity/minimax_h3_ref_frames_probe.mojo
#
# Gates `pipeline/minimax_h3_ref_frames.mojo` BIT-EXACT against the vendor's
# own `prepare_reference_frames` / `resample_reference_frames`, on REAL
# 1920x1080 frames from output/h3_ref2va_media/ref_video.mp4. Host only — no
# DeviceContext, no GPU, no model weights.
#
# The oracle (scripts/minimax_h3_ref_frames_oracle.py) does not transcribe the
# vendor functions: it `exec`s them straight out of
# /home/alex/minimax_h3_ref/diffusers-src/.../minimax_h3/packing_ref2va.py
# (and `resolve_canvas_size` out of packing.py), so the LANCZOS filter
# (Pillow's own C), the canvas law and the fps drop/duplicate selection are
# the real thing, not a restatement.
#
# BIT-EXACT IS THE RIGHT BAR: the vendor resizes UINT8 frames with PIL before
# any float normalization (packing_ref2va.py:679-681), the fps resample moves
# whole frames, and the truncation is a slice — every pass is uint8-in/
# uint8-out integer arithmetic, so there is no float tolerance to justify.
#
# Run:
#   CUDA_VISIBLE_DEVICES="" /home/alex/OneTrainer/venv/bin/python \
#     scripts/minimax_h3_ref_frames_oracle.py
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_ref_frames_probe.mojo \
#     -o /tmp/h3_ref_frames_probe -Xlinker -lm \
#   && /tmp/h3_ref_frames_probe output/minimax_h3_ref_frames/ref_frames_ref.safetensors

from std.sys import argv
from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.pipeline.minimax_h3_media_in import MiniMaxH3RgbFrames
from serenitymojo.pipeline.minimax_h3_ref_frames import (
    minimax_h3_prepare_reference_frames,
    minimax_h3_resample_reference_frames,
)


struct Report(Copyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def ok(mut self, label: String, detail: String):
        self.checks += 1
        print("  ok  ", label, "—", detail)

    def fail(mut self, label: String, detail: String):
        self.checks += 1
        self.failures += 1
        print("  FAIL", label, "—", detail)


def _read_frames(
    ref st: SafeTensors, name: String, fps: Float64
) raises -> MiniMaxH3RgbFrames:
    if not st.has_tensor(name):
        raise Error(String("probe: reference file has no tensor ") + name)
    var info = st.tensor_info(name)
    if info.dtype != STDtype.U8:
        raise Error(String("probe: ") + name + " is not U8")
    if len(info.shape) != 4 or info.shape[3] != 3:
        raise Error(String("probe: ") + name + " is not [T, H, W, 3]")
    var t = info.shape[0]
    var h = info.shape[1]
    var w = info.shape[2]
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    var p = tv.data.unsafe_ptr()
    var pixels = List[UInt8](capacity=t * h * w * 3)
    for i in range(t * h * w * 3):
        pixels.append(p[i])
    return MiniMaxH3RgbFrames(pixels^, t, h, w, fps)


def _compare(
    mut report: Report,
    label: String,
    got: MiniMaxH3RgbFrames,
    want: MiniMaxH3RgbFrames,
) raises:
    if (
        got.num_frames != want.num_frames or got.height != want.height
        or got.width != want.width
    ):
        report.fail(
            label,
            String("shape [") + String(got.num_frames) + ", "
            + String(got.height) + ", " + String(got.width) + "], want ["
            + String(want.num_frames) + ", " + String(want.height) + ", "
            + String(want.width) + "]",
        )
        return
    var total = want.num_frames * want.height * want.width * 3
    var bad = 0
    var first = -1
    var max_gap = 0
    for i in range(total):
        if got.pixels[i] != want.pixels[i]:
            bad += 1
            if first < 0:
                first = i
            var d = Int(got.pixels[i]) - Int(want.pixels[i])
            if d < 0:
                d = -d
            if d > max_gap:
                max_gap = d
    if bad == 0:
        report.ok(
            label,
            String("bit-exact over ") + String(total) + " bytes ("
            + String(want.num_frames) + " x " + String(want.width) + "x"
            + String(want.height) + ")",
        )
    else:
        var pct = Float64(bad) * 100.0 / Float64(total)
        report.fail(
            label,
            String(bad) + " of " + String(total) + " bytes differ ("
            + String(pct) + "%), max gap " + String(max_gap)
            + ", first at byte " + String(first),
        )


def main() raises:
    var args = argv()
    var ref_path = String(
        "output/minimax_h3_ref_frames/ref_frames_ref.safetensors"
    )
    if len(args) >= 2:
        ref_path = String(args[1])
    print("MiniMax-H3 ref2va — reference-frame canvas resize probe")
    print("  reference:", ref_path)
    print("")
    var report = Report()
    var st = SafeTensors.open(ref_path)

    # ── [1] the real thing: 5 REAL 1920x1080 frames onto their canvas ────────
    print("[1] prepare_reference_frames on real 1920x1080 frames")
    var real_in = _read_frames(st, String("real_in"), Float64(24.0))
    var real_out = _read_frames(st, String("real_out"), Float64(24.0))
    if real_out.height == 768 and real_out.width == 1344:
        report.ok(
            "canvas law",
            "vendor canvas for 1920x1080 is 1344x768 (area-capped 16:9)",
        )
    else:
        report.fail(
            "canvas law",
            String("oracle canvas is ") + String(real_out.width) + "x"
            + String(real_out.height) + ", expected 1344x768",
        )
    var got_real = minimax_h3_prepare_reference_frames(real_in, 5)
    _compare(report, "real 5-frame resize", got_real, real_out)

    # Truncation inside prepare: cap at 3 of the 5 frames.
    var realcap_out = _read_frames(st, String("realcap_out"), Float64(24.0))
    var got_cap = minimax_h3_prepare_reference_frames(real_in, 3)
    _compare(report, "real capped at 3", got_cap, realcap_out)

    # Frames already at the canvas pass through untouched.
    var got_ident = minimax_h3_prepare_reference_frames(real_out, 5)
    _compare(report, "canvas-sized passthrough", got_ident, real_out)

    # ── [2] the portrait branch of the canvas law ───────────────────────────
    print("")
    print("[2] portrait reference (480x854)")
    var portrait_in = _read_frames(st, String("portrait_in"), Float64(24.0))
    var portrait_out = _read_frames(st, String("portrait_out"), Float64(24.0))
    var got_portrait = minimax_h3_prepare_reference_frames(portrait_in, 2)
    _compare(report, "portrait resize", got_portrait, portrait_out)

    # ── [3] what the law does not cover must RAISE, not resize ──────────────
    print("")
    print("[3] refusals")
    var wide_pixels = List[UInt8]()
    wide_pixels.resize(64 * 320 * 3, UInt8(0))
    var wide = MiniMaxH3RgbFrames(wide_pixels^, 1, 64, 320, Float64(24.0))
    var wide_raised = False
    try:
        var bad = minimax_h3_prepare_reference_frames(wide, 1)
        _ = bad.num_frames
    except:
        wide_raised = True
    if wide_raised:
        report.ok("aspect 5:1", "refused, as the vendor refuses it")
    else:
        report.fail("aspect 5:1", "accepted a canvas the law refuses")

    # A clip not yet on the 24 fps grid must be refused by prepare — the
    # vendor resamples BEFORE it resizes (before_encoder.py:371-372).
    var offgrid_pixels = List[UInt8]()
    offgrid_pixels.resize(48 * 64 * 3, UInt8(0))
    var offgrid = MiniMaxH3RgbFrames(offgrid_pixels^, 1, 48, 64, Float64(30.0))
    var offgrid_raised = False
    try:
        var bad2 = minimax_h3_prepare_reference_frames(offgrid, 1)
        _ = bad2.num_frames
    except:
        offgrid_raised = True
    if offgrid_raised:
        report.ok("30 fps into prepare", "refused — resample-first enforced")
    else:
        report.fail("30 fps into prepare", "accepted un-resampled frames")

    # ── [4] the 24 fps resample applied to pixels ───────────────────────────
    print("")
    print("[4] resample_reference_frames (drop and duplicate)")
    var fps30_in = _read_frames(st, String("fps30_in"), Float64(30.0))
    var fps30_out = _read_frames(st, String("fps30_out"), Float64(24.0))
    var got30 = minimax_h3_resample_reference_frames(fps30_in)
    _compare(report, "8 @ 30 fps -> 24 grid", got30, fps30_out)

    var fps12_in = _read_frames(st, String("fps12_in"), Float64(12.0))
    var fps12_out = _read_frames(st, String("fps12_out"), Float64(24.0))
    var got12 = minimax_h3_resample_reference_frames(fps12_in)
    _compare(report, "5 @ 12 fps -> 24 grid", got12, fps12_out)

    # The output cap is a pure truncation of the full resample.
    var got30_capped = minimax_h3_resample_reference_frames(fps30_in, 3)
    var want30_sliced_pixels = List[UInt8]()
    var slice_bytes = 3 * fps30_out.height * fps30_out.width * 3
    for i in range(slice_bytes):
        want30_sliced_pixels.append(fps30_out.pixels[i])
    var want30_sliced = MiniMaxH3RgbFrames(
        want30_sliced_pixels^, 3, fps30_out.height, fps30_out.width,
        Float64(24.0),
    )
    _compare(report, "capped resample == full[:3]", got30_capped, want30_sliced)

    # 24 fps input is the identity route.
    var ident24_in = _read_frames(st, String("fps30_in"), Float64(24.0))
    var got_ident24 = minimax_h3_resample_reference_frames(ident24_in)
    _compare(report, "24 fps identity", got_ident24, ident24_in)

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_ref_frames probe FAILED")
