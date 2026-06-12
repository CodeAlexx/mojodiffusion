#!/usr/bin/env python
# aspect_buckets_parity_ref.py — T2.D gate (a) python oracle.
#
# Runs SimpleTuner's OWN bucketing code (MultiaspectImage from
# /home/alex/SimpleTuner/simpletuner/helpers/multiaspect/image.py) plus the
# crop_aspect="closest" selection and the calculate_target_size tail from
# /home/alex/SimpleTuner/simpletuner/helpers/image_manipulation/training_sample.py
# (lines 561-624), and prints the EXACT line format of
# training/tests/aspect_buckets_parity.mojo. Gate: diff == empty.
#
# Run with SimpleTuner's venv (so the import is the real code):
#   /home/alex/SimpleTuner/.venv/bin/python aspect_buckets_parity_ref.py
#
# StateTracker is patched to (a) carry the per-case alignment/rounding args
# and (b) keep the aspect->resolution memo empty, so the PURE formula runs
# (same contract as the Mojo module, which is stateless).

import sys
from math import sqrt
from types import SimpleNamespace

sys.path.insert(0, "/home/alex/SimpleTuner")

from simpletuner.helpers.training.state_tracker import StateTracker  # noqa: E402

_ARGS = SimpleNamespace(aspect_bucket_alignment=64, aspect_bucket_rounding=None)
StateTracker.get_args = staticmethod(lambda: _ARGS)
StateTracker.get_resolution_by_aspect = staticmethod(
    lambda dataloader_resolution, aspect: None
)
StateTracker.set_resolution_by_aspect = staticmethod(
    lambda dataloader_resolution, aspect, resolution: None
)

from simpletuner.helpers.multiaspect.image import MultiaspectImage  # noqa: E402


def ladder():
    # default_aspect_ladder(): 1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3 as the
    # 2-decimal floats a crop_aspect_buckets list would carry.
    return [
        1.0,
        round(4.0 / 3.0, 2),
        round(3.0 / 4.0, 2),
        round(16.0 / 9.0, 2),
        round(9.0 / 16.0, 2),
        round(3.0 / 2.0, 2),
        round(2.0 / 3.0, 2),
    ]


def a100(a):
    return int(a * 100.0 + 0.5)


def pixel_resolution(megapixels):
    # TrainingSample._set_resolution, resolution_type="area" (line 222)
    return int(MultiaspectImage._round_to_nearest_multiple(sqrt(megapixels * 1e6)))


def assign(w0, h0, lad, megapixels):
    # calculate_image_aspect_ratio(original_size) (training_sample.py:571)
    ar_img = MultiaspectImage.calculate_image_aspect_ratio((w0, h0))
    # _select_random_aspect crop_aspect="closest" (training_sample.py:266-273)
    closest = min(lad, key=lambda bucket: abs(bucket - ar_img))
    bucket_index = lad.index(closest)
    # calculate_new_size_by_pixel_area (image.py:178-276) — SimpleTuner's code
    target, inter, _adj = MultiaspectImage.calculate_new_size_by_pixel_area(
        closest, megapixels, (w0, h0)
    )
    # calculate_target_size tail (training_sample.py:610-618)
    key = MultiaspectImage.calculate_image_aspect_ratio(target)
    pixres = pixel_resolution(megapixels)
    w_t, h_t = target
    w_i, h_i = inter
    if key == 1.0:
        # correct_intermediary_square_size (training_sample.py:545-559)
        if w_i < pixres:
            w_i, h_i = pixres, pixres
        w_t, h_t = pixres, pixres
    # crop_style="center" offsets
    crop_x = (w_i - w_t) // 2
    crop_y = (h_i - h_t) // 2
    return bucket_index, closest, w_t, h_t, w_i, h_i, crop_x, crop_y, a100(key)


def generate(megapixels, lad):
    pixres = pixel_resolution(megapixels)
    out = []
    for a in lad:
        target, _inter, _adj = MultiaspectImage.calculate_new_size_by_pixel_area(
            a, megapixels, (1000, 1000)
        )
        w_t, h_t = target
        key = MultiaspectImage.calculate_image_aspect_ratio((w_t, h_t))
        if key == 1.0:
            w_t, h_t = pixres, pixres
        if any(b[1] == w_t and b[2] == h_t for b in out):
            continue
        out.append((a, w_t, h_t, a100(key)))
    return out, pixres


def main():
    lad = ladder()
    areas = [262144, 1048576]
    aligns = [64, 16]

    for area in areas:
        mp = area / 1e6
        for align in aligns:
            _ARGS.aspect_bucket_alignment = align
            buckets, pixres = generate(mp, lad)
            print(f"SET area={area} align={align} pixres={pixres} n={len(buckets)}")
            for i, (a, w, h, key) in enumerate(buckets):
                print(
                    f"  bucket[{i}] aspect_x100={a100(a)} target={w}x{h} key_x100={key}"
                )

    cases = [
        (1080, 1350), (1080, 1920), (1920, 1080), (1024, 1024), (512, 512),
        (4000, 3000), (3000, 4000), (1500, 1000), (1000, 1500), (1040, 1000),
        (1000, 1040), (2560, 1080), (1080, 2560), (640, 480), (300, 400),
        (777, 555), (555, 777), (1349, 1351), (897, 1123), (1123, 897),
    ]

    for area in areas:
        mp = area / 1e6
        for align in aligns:
            _ARGS.aspect_bucket_alignment = align
            for (w0, h0) in cases:
                bi, asp, w_t, h_t, w_i, h_i, cx, cy, key = assign(w0, h0, lad, mp)
                print(
                    f"ASSIGN {w0}x{h0} area={area} align={align}"
                    f" -> bucket={bi} aspect_x100={a100(asp)}"
                    f" target={w_t}x{h_t} inter={w_i}x{h_i}"
                    f" crop=({cx},{cy}) key_x100={key}"
                )

    print(f"DONE cases={len(cases) * len(areas) * len(aligns)}")


if __name__ == "__main__":
    main()
