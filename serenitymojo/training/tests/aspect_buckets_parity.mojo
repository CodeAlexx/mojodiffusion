# aspect_buckets_parity.mojo — T2.D gate (a): bucket-set generation + bucket
# ASSIGNMENT parity vs SimpleTuner's own python code.
#
# Prints one canonical line per case (integer fields only — float formatting
# never hits stdout); the python oracle
# (training/tests/aspect_buckets_parity_ref.py, run with SimpleTuner's venv so
# it executes SimpleTuner's REAL MultiaspectImage code) prints the identical
# format. The gate is `diff` = empty (exact match, every field).
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/training/tests/aspect_buckets_parity.mojo \
#       -o /tmp/aspect_buckets_parity && /tmp/aspect_buckets_parity > /tmp/ab_mojo.txt
#   /home/alex/SimpleTuner/.venv/bin/python \
#     /home/alex/mojodiffusion/serenitymojo/training/tests/aspect_buckets_parity_ref.py \
#       > /tmp/ab_py.txt
#   diff /tmp/ab_mojo.txt /tmp/ab_py.txt && echo "PARITY PASS"

from std.collections import List

from serenitymojo.training.aspect_buckets import (
    AspectBucket, AspectBucketAssignment, default_aspect_ladder,
    generate_aspect_buckets, assign_aspect_bucket, pixel_resolution,
)


def _a100(a: Float64) -> Int:
    return Int(a * 100.0 + 0.5)


def main() raises:
    var ladder = default_aspect_ladder()

    # two budgets (512px and 1024px ladders) x two alignments (SimpleTuner
    # default 64, and the VAE-stride x patch divisibility 16).
    # mp encoded as integer pixel area (mp * 1e6) so no float ever prints.
    var areas = List[Int]()
    areas.append(262144)    # 512x512 budget  (0.262144 MP)
    areas.append(1048576)   # 1024x1024 budget (1.048576 MP)
    var aligns = List[Int]()
    aligns.append(64)
    aligns.append(16)

    # ── bucket-set generation parity ─────────────────────────────────────────
    for mi in range(len(areas)):
        var mp = Float64(areas[mi]) / 1.0e6
        for ai in range(len(aligns)):
            var bs = generate_aspect_buckets(mp, aligns[ai], ladder)
            print(
                String("SET area=") + String(areas[mi])
                + String(" align=") + String(aligns[ai])
                + String(" pixres=") + String(pixel_resolution(mp, aligns[ai]))
                + String(" n=") + String(len(bs))
            )
            for i in range(len(bs)):
                print(
                    String("  bucket[") + String(i)
                    + String("] aspect_x100=") + String(_a100(bs[i].aspect))
                    + String(" target=") + String(bs[i].width)
                    + String("x") + String(bs[i].height)
                    + String(" key_x100=") + String(bs[i].aspect_key_x100)
                )

    # ── assignment parity: 20 synthetic WxH cases ────────────────────────────
    # mixed portrait/landscape/square, extreme aspects, near-tie aspects,
    # small upscale sources, and dims that exercise int() truncation.
    var ws = List[Int]()
    var hs = List[Int]()
    ws.append(1080); hs.append(1350)   # 0.8  portrait (alina-class)
    ws.append(1080); hs.append(1920)   # 0.56 tall video frame
    ws.append(1920); hs.append(1080)   # 1.78 landscape
    ws.append(1024); hs.append(1024)   # 1.0  square
    ws.append(512);  hs.append(512)    # 1.0  square at budget
    ws.append(4000); hs.append(3000)   # 1.33 large landscape
    ws.append(3000); hs.append(4000)   # 0.75 large portrait
    ws.append(1500); hs.append(1000)   # 1.5
    ws.append(1000); hs.append(1500)   # 0.67
    ws.append(1040); hs.append(1000)   # 1.04 NEAR-TIE between 0.75 and 1.33
    ws.append(1000); hs.append(1040)   # 0.96
    ws.append(2560); hs.append(1080)   # 2.37 ultrawide (clamps to 1.78)
    ws.append(1080); hs.append(2560)   # 0.42 ultratall (clamps to 0.56)
    ws.append(640);  hs.append(480)    # 1.33 small (upscale)
    ws.append(300);  hs.append(400)    # 0.75 tiny (upscale)
    ws.append(777);  hs.append(555)    # 1.4  odd dims
    ws.append(555);  hs.append(777)    # 0.71 odd dims
    ws.append(1349); hs.append(1351)   # ~1.0 near-square
    ws.append(897);  hs.append(1123)   # 0.8  odd portrait
    ws.append(1123); hs.append(897)    # 1.25 odd landscape

    for mi in range(len(areas)):
        var mp = Float64(areas[mi]) / 1.0e6
        for ai in range(len(aligns)):
            for ci in range(len(ws)):
                var asn = assign_aspect_bucket(
                    ws[ci], hs[ci], ladder, mp, aligns[ai],
                )
                print(
                    String("ASSIGN ") + String(ws[ci]) + String("x") + String(hs[ci])
                    + String(" area=") + String(areas[mi])
                    + String(" align=") + String(aligns[ai])
                    + String(" -> bucket=") + String(asn.bucket_index)
                    + String(" aspect_x100=") + String(_a100(asn.aspect))
                    + String(" target=") + String(asn.target_w)
                    + String("x") + String(asn.target_h)
                    + String(" inter=") + String(asn.inter_w)
                    + String("x") + String(asn.inter_h)
                    + String(" crop=(") + String(asn.crop_x)
                    + String(",") + String(asn.crop_y) + String(")")
                    + String(" key_x100=") + String(asn.aspect_key_x100)
                )

    print(String("DONE cases=") + String(len(ws) * len(areas) * len(aligns)))
