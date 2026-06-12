# zimage_comptime_ladder_gate.mojo — T2.D follow-up gate (a): the COMPTIME
# integer ladder (aspect_buckets.mojo ZIMAGE_T2D_* + zimage_t2d_lat_h/w) must
# emit EXACTLY the same bucket set as the runtime generator
# generate_aspect_buckets(0.262144, 64, default_aspect_ladder()) — same
# length, same order, same (lat_h, lat_w) per index, same x100 aspect keys.
#
# The comptime side is materialized through `comptime for` + comptime params
# (the same instantiation mechanism train_zimage_real.mojo's dispatch uses),
# so a PASS here also proves the ladder functions are comptime-evaluable.
#
# Build/run (host-only, no GPU):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/training/tests/zimage_comptime_ladder_gate.mojo \
#       -o /tmp/zimage_comptime_ladder_gate && /tmp/zimage_comptime_ladder_gate

from std.collections import List

from serenitymojo.training.aspect_buckets import (
    default_aspect_ladder, generate_aspect_buckets,
    ZIMAGE_T2D_LADDER_LEN, ZIMAGE_T2D_LADDER_X100,
    zimage_t2d_lat_h, zimage_t2d_lat_w,
)

comptime T2D_MEGAPIXELS = Float64(0.262144)  # 512x512 budget
comptime T2D_ALIGN = 64


def main() raises:
    print("=== T2.D comptime-vs-runtime ladder gate (512px / align-64) ===")

    # comptime set, materialized through comptime params (proves evaluability)
    var c_h = List[Int]()
    var c_w = List[Int]()
    var c_x100 = List[Int]()
    comptime for i in range(ZIMAGE_T2D_LADDER_LEN):
        comptime X100_I = ZIMAGE_T2D_LADDER_X100[i]
        comptime LH_I = zimage_t2d_lat_h(X100_I)
        comptime LW_I = zimage_t2d_lat_w(X100_I)
        c_h.append(LH_I)
        c_w.append(LW_I)
        c_x100.append(X100_I)

    # runtime set (the SimpleTuner-parity Float64 generator)
    var ladder = default_aspect_ladder()
    var buckets = generate_aspect_buckets(T2D_MEGAPIXELS, T2D_ALIGN, ladder)

    if len(ladder) != ZIMAGE_T2D_LADDER_LEN:
        raise Error("GATE FAIL: runtime ladder length != comptime LEN")
    if len(buckets) != ZIMAGE_T2D_LADDER_LEN:
        raise Error(
            String("GATE FAIL: runtime bucket count ") + String(len(buckets))
            + String(" != comptime LEN ") + String(ZIMAGE_T2D_LADDER_LEN)
            + String(" (dedup/snap divergence)")
        )

    for i in range(ZIMAGE_T2D_LADDER_LEN):
        var rt_x100 = Int(ladder[i] * 100.0 + 0.5)
        var rt_h = buckets[i].height // 8
        var rt_w = buckets[i].width // 8
        print(
            "  [", i, "] aspect_x100 comptime=", c_x100[i], " runtime=", rt_x100,
            " lat comptime=", c_h[i], "x", c_w[i],
            " runtime=", rt_h, "x", rt_w,
        )
        if buckets[i].height % 8 != 0 or buckets[i].width % 8 != 0:
            raise Error("GATE FAIL: runtime bucket canvas not /8-divisible")
        if rt_x100 != c_x100[i]:
            raise Error(String("GATE FAIL: aspect key mismatch at index ") + String(i))
        if rt_h != c_h[i] or rt_w != c_w[i]:
            raise Error(
                String("GATE FAIL: latent dims mismatch at index ") + String(i)
                + String(": comptime ") + String(c_h[i]) + String("x") + String(c_w[i])
                + String(" runtime ") + String(rt_h) + String("x") + String(rt_w)
            )

    # set equality both directions follows from len equality + per-index
    # equality, but assert no duplicate (lat_h, lat_w) pairs in the comptime
    # set so every dispatch arm is a distinct bucket.
    for i in range(ZIMAGE_T2D_LADDER_LEN):
        for j in range(i + 1, ZIMAGE_T2D_LADDER_LEN):
            if c_h[i] == c_h[j] and c_w[i] == c_w[j]:
                raise Error("GATE FAIL: duplicate latent bucket in comptime ladder")

    print("PASS: comptime integer ladder == generate_aspect_buckets output (",
          ZIMAGE_T2D_LADDER_LEN, "buckets, exact )")
