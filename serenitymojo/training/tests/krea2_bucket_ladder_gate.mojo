# krea2_bucket_ladder_gate.mojo — T2.D krea2 aspect-bucketing gate (host-only).
#
# Proves the krea2 COMPTIME integer ladder (models/krea2/krea2_buckets.mojo:
# KREA2_LADDER_X100 + krea2_lat_h/w via aspect_buckets.aspect_lat_{h,w}_units)
# emits EXACTLY the bucket set the runtime SimpleTuner-parity Float64 generator
# generate_aspect_buckets produces — for BOTH krea2 area budgets:
#   512px  → megapixels 0.262144, align 64, edge units e=8   (square latent 64)
#   1024px → megapixels 1.048576, align 64, edge units e=16  (square latent 128)
# Same length, same order, same (lat_h, lat_w) per index, same x100 keys, and no
# duplicate latent bucket (every comptime dispatch arm is a distinct canvas).
#
# This is the krea2 twin of zimage_comptime_ladder_gate: the comptime side is
# materialized through `comptime for` + comptime params (the SAME instantiation
# mechanism the trainer's bucket dispatch uses), so a PASS also proves the ladder
# functions are comptime-evaluable at the krea2 edge.
#
# Build/run (host-only, no GPU):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build --optimization-level 2 -I . -Xlinker -lm \
#       serenitymojo/training/tests/krea2_bucket_ladder_gate.mojo \
#       -o /tmp/krea2_bucket_ladder_gate && /tmp/krea2_bucket_ladder_gate

from std.collections import List

from serenitymojo.training.aspect_buckets import (
    default_aspect_ladder, generate_aspect_buckets,
)
from serenitymojo.models.krea2.krea2_buckets import (
    KREA2_LADDER_LEN, KREA2_LADDER_X100, krea2_lat_h, krea2_lat_w,
)


def _check_edge[E_UNITS: Int](megapixels: Float64) raises:
    print("--- krea2 ladder gate: edge units e=", E_UNITS,
          " (megapixels ", megapixels, ", align 64) ---")

    # comptime set, materialized through comptime params (proves evaluability)
    var c_h = List[Int]()
    var c_w = List[Int]()
    var c_x100 = List[Int]()
    comptime for i in range(KREA2_LADDER_LEN):
        comptime X100_I = KREA2_LADDER_X100[i]
        comptime LH_I = krea2_lat_h(X100_I, E_UNITS)
        comptime LW_I = krea2_lat_w(X100_I, E_UNITS)
        c_h.append(LH_I)
        c_w.append(LW_I)
        c_x100.append(X100_I)

    # runtime set (the SimpleTuner-parity Float64 generator)
    var ladder = default_aspect_ladder()
    var buckets = generate_aspect_buckets(megapixels, 64, ladder)

    if len(ladder) != KREA2_LADDER_LEN:
        raise Error("GATE FAIL: runtime ladder length != KREA2_LADDER_LEN")
    if len(buckets) != KREA2_LADDER_LEN:
        raise Error(
            String("GATE FAIL: runtime bucket count ") + String(len(buckets))
            + String(" != KREA2_LADDER_LEN ") + String(KREA2_LADDER_LEN)
            + String(" (dedup/snap divergence at e=") + String(E_UNITS) + String(")")
        )

    for i in range(KREA2_LADDER_LEN):
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

    # distinct-arm assertion: no duplicate (lat_h, lat_w) in the comptime set
    for i in range(KREA2_LADDER_LEN):
        for j in range(i + 1, KREA2_LADDER_LEN):
            if c_h[i] == c_h[j] and c_w[i] == c_w[j]:
                raise Error(
                    String("GATE FAIL: duplicate latent bucket at e=")
                    + String(E_UNITS)
                )
    print("  PASS e=", E_UNITS, ": comptime ladder == generate_aspect_buckets (",
          KREA2_LADDER_LEN, "buckets, exact )")


def main() raises:
    print("=== krea2 aspect-bucket comptime-vs-runtime ladder gate ===")
    _check_edge[8](Float64(0.262144))    # 512px  area, square latent 64
    _check_edge[16](Float64(1.048576))   # 1024px area, square latent 128
    print("PASS: krea2 comptime ladder matches SimpleTuner generator at 512px + 1024px")
