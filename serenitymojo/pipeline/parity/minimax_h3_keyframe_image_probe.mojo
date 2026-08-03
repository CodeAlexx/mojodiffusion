# serenitymojo/pipeline/parity/minimax_h3_keyframe_image_probe.mojo
#
# Gates `pipeline/minimax_h3_keyframe_image.mojo` BIT-EXACT against Pillow, on
# real images. Host only — no DeviceContext, no GPU, no model weights.
#
# The oracle (scratchpad/h3_kf_image_oracle.py) does not transcribe the vendor's
# `prepare_keyframe_image`: it `exec`s the function straight out of
# /home/alex/minimax_h3_ref/diffusers/packing.py, so both the LANCZOS filter
# (Pillow's own C) and the stretch/cover-crop policy (the vendor's own Python)
# are the real thing, not a restatement of them.
#
# BIT-EXACT IS THE RIGHT BAR HERE and it is reachable: after the coefficient
# table is built in double, every pixel is fixed-point integer arithmetic, so
# the only way to be "nearly" right is to have the wrong table, the wrong pass
# order, or a float intermediate where Pillow keeps 8 bits. All three of those
# are bugs that would quietly change the conditioning anchor the model sees.
#
# Run:
#   CUDA_VISIBLE_DEVICES="" <venv>/bin/python \
#     <scratch>/h3_kf_image_oracle.py <scratch>/h3_keyframe_image_ref.safetensors
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_keyframe_image_probe.mojo \
#     -o <scratch>/h3_keyframe_image_probe -Xlinker -lm \
#   && <scratch>/h3_keyframe_image_probe <scratch>/h3_keyframe_image_ref.safetensors

from std.sys import argv
from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.pipeline.minimax_h3_keyframe_image import (
    MiniMaxH3RgbImage,
    minimax_h3_exif_transpose,
    minimax_h3_keyframe_anchors,
    minimax_h3_lanczos_resize,
    minimax_h3_prepare_keyframe_image,
    minimax_h3_prepare_keyframes,
)
from serenitymojo.pipeline.minimax_h3_media_in import (
    minimax_h3_ffmpeg_read_image,
    minimax_h3_jpeg_exif_orientation,
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


def _read_rgb(ref st: SafeTensors, name: String) raises -> MiniMaxH3RgbImage:
    if not st.has_tensor(name):
        raise Error(String("probe: reference file has no tensor ") + name)
    var info = st.tensor_info(name)
    if info.dtype != STDtype.U8:
        raise Error(String("probe: ") + name + " is not U8")
    if len(info.shape) != 3 or info.shape[2] != 3:
        raise Error(String("probe: ") + name + " is not [H, W, 3]")
    var h = info.shape[0]
    var w = info.shape[1]
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    var p = tv.data.unsafe_ptr()
    var pixels = List[UInt8](capacity=h * w * 3)
    for i in range(h * w * 3):
        pixels.append(p[i])
    return MiniMaxH3RgbImage(pixels^, h, w)


def _compare(
    mut report: Report,
    label: String,
    got: MiniMaxH3RgbImage,
    want: MiniMaxH3RgbImage,
) raises:
    if got.height != want.height or got.width != want.width:
        report.fail(
            label,
            String("shape ") + String(got.width) + "x" + String(got.height)
            + ", want " + String(want.width) + "x" + String(want.height),
        )
        return
    var bad = 0
    var first = -1
    var max_gap = 0
    for i in range(want.numel()):
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
            String("bit-exact over ") + String(want.numel()) + " bytes ("
            + String(want.width) + "x" + String(want.height) + ")",
        )
    else:
        var pct = Float64(bad) * 100.0 / Float64(want.numel())
        report.fail(
            label,
            String(bad) + " of " + String(want.numel()) + " bytes differ ("
            + String(pct) + "%), max gap " + String(max_gap)
            + ", first at byte " + String(first),
        )


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: minimax_h3_keyframe_image_probe <keyframe_image_ref.safetensors>")
        return
    var ref_path = String(args[1])
    print("MiniMax-H3 keyframe unit — PIL LANCZOS / prepare_keyframe_image probe")
    print("  reference:", ref_path)
    print("")
    var report = Report()
    var st = SafeTensors.open(ref_path)

    # ── [1] the filter alone: PIL.Image.resize(..., LANCZOS) ─────────────────
    print("[1] LANCZOS resize vs PIL (upscale, downscale, mixed)")
    var rcases = [0, 1, 2]
    var rdims = [
        64, 96,      # case 0 out h, w
        480, 640,    # case 1
        768, 1344,   # case 2
    ]
    for i in range(len(rcases)):
        var src = _read_rgb(st, String("rin_") + String(i))
        var want = _read_rgb(st, String("rout_") + String(i))
        var got = minimax_h3_lanczos_resize(src, rdims[2 * i], rdims[2 * i + 1])
        _compare(report, String("resize case ") + String(i), got, want)

    # ── [2] the vendor's own keyframe policy ─────────────────────────────────
    print("")
    print("[2] prepare_keyframe_image vs the vendor's own function")
    var names = [
        String("small_stretch"), String("up_stretch"), String("mixed_stretch"),
        String("cover_crop"), String("cover_crop_tall"), String("identity"),
    ]
    var dims = [
        64, 96,
        768, 1344,
        768, 1344,
        768, 1344,
        768, 1344,
        768, 1344,
    ]
    var stretches = [True, True, True, False, False, True]
    for i in range(len(names)):
        var src = _read_rgb(st, String("in_") + names[i])
        var want = _read_rgb(st, String("out_") + names[i])
        var got = minimax_h3_prepare_keyframe_image(
            src, dims[2 * i], dims[2 * i + 1], stretches[i]
        )
        _compare(report, names[i], got, want)

    # ── [3] the anchor/stretch policy across the three keyframe tasks ────────
    print("")
    print("[3] I2VA / FL2VA / L2VA anchor + stretch policy")
    var i2va = minimax_h3_keyframe_anchors(True, False)
    if len(i2va) == 1 and i2va[0] == 0:
        report.ok("I2VA anchors", "[first]")
    else:
        report.fail("I2VA anchors", String(len(i2va)) + " entries")
    var l2va = minimax_h3_keyframe_anchors(False, True)
    if len(l2va) == 1 and l2va[0] == 1:
        report.ok("L2VA anchors", "[last] — one keyframe, anchored at the END")
    else:
        report.fail("L2VA anchors", String(len(l2va)) + " entries")
    var fl2va = minimax_h3_keyframe_anchors(True, True)
    if len(fl2va) == 2 and fl2va[0] == 0 and fl2va[1] == 1:
        report.ok("FL2VA anchors", "[first, last]")
    else:
        report.fail("FL2VA anchors", String(len(fl2va)) + " entries")

    # L2VA's single keyframe must be STRETCHED (filtered index 0), not
    # cover-cropped — the trap this file's header calls out.
    var lone = _read_rgb(st, String("in_cover_crop"))
    var lone_stretched = _read_rgb(st, String("in_cover_crop"))
    var l2va_list = List[MiniMaxH3RgbImage]()
    l2va_list.append(lone^)
    var prepared = minimax_h3_prepare_keyframes(l2va_list, False, True, 768, 1344)
    var direct = minimax_h3_prepare_keyframe_image(lone_stretched, 768, 1344, True)
    if len(prepared) == 1:
        _compare(report, "L2VA lone keyframe is stretched", prepared[0], direct)
    else:
        report.fail("L2VA lone keyframe is stretched", "wrong keyframe count")

    # A mismatched (images, have_first/have_last) pair must fail loud.
    var mismatched = False
    try:
        var empty = List[MiniMaxH3RgbImage]()
        var bad = minimax_h3_prepare_keyframes(empty, True, True, 768, 1344)
        _ = len(bad)
    except:
        mismatched = True
    if mismatched:
        report.ok("keyframe count cross-check", "mismatch rejected")
    else:
        report.fail("keyframe count cross-check", "accepted a mismatched request")

    # ── [4] EXIF orientation: the transpose, and the byte-level tag parse ────
    print("")
    print("[4] EXIF orientation vs PIL.ImageOps.exif_transpose")
    var orient_src = _read_rgb(st, String("orient_in"))
    for o in range(1, 9):
        var want = _read_rgb(st, String("orient_out_") + String(o))
        var src = _read_rgb(st, String("orient_in"))
        var got = minimax_h3_exif_transpose(src, o)
        _compare(report, String("orientation ") + String(o), got, want)
    # Orientation 0 is how "no tag" reaches this function; it must be identity.
    var zero_src = _read_rgb(st, String("orient_in"))
    var zero_out = minimax_h3_exif_transpose(zero_src, 0)
    _compare(report, "orientation 0 (absent) is identity", zero_out, orient_src)

    if len(args) >= 3:
        var fixture_dir = String(args[2])
        print("")
        print("[5] EXIF tag parsed off the JPEG's own bytes:", fixture_dir)
        for o in range(1, 9):
            var path = fixture_dir + "/orient_" + String(o) + ".jpg"
            var got_tag = minimax_h3_jpeg_exif_orientation(path)
            if got_tag == o:
                report.ok(
                    String("orient_") + String(o) + ".jpg", String("tag ") + String(got_tag)
                )
            else:
                report.fail(
                    String("orient_") + String(o) + ".jpg",
                    String("read tag ") + String(got_tag) + ", want " + String(o),
                )
        var no_exif = minimax_h3_jpeg_exif_orientation(fixture_dir + "/no_exif.jpg")
        if no_exif == 1:
            report.ok("no_exif.jpg", "reads as 1 (identity), like PIL")
        else:
            report.fail("no_exif.jpg", String("read tag ") + String(no_exif))
        var png_tag = minimax_h3_jpeg_exif_orientation(fixture_dir + "/plain.png")
        if png_tag == 1:
            report.ok("plain.png", "reads as 1 (not a JPEG)")
        else:
            report.fail("plain.png", String("read tag ") + String(png_tag))

        # ── [6] the ffmpeg still-image ingest route, byte for byte ──────────
        print("")
        print("[6] ffmpeg still-image ingest vs PIL's decode of the same PNG")
        var want_png = _read_rgb(st, String("png_in"))
        var frames = minimax_h3_ffmpeg_read_image(
            fixture_dir + "/plain.png",
            fixture_dir + "/plain.rgb",
            fixture_dir + "/probe_scratch.txt",
        )
        if frames.num_frames != 1:
            report.fail("ingest frame count", String(frames.num_frames))
        else:
            var got_png = MiniMaxH3RgbImage(
                frames.pixels.copy(), frames.height, frames.width
            )
            _compare(report, "PNG ingest", got_png, want_png)
    else:
        print("")
        print("[5-6] SKIPPED — pass the EXIF fixture directory as argv[2] to run")
        print("      the JPEG tag parser and the ffmpeg still-image ingest checks")

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_keyframe_image probe FAILED")
