# serenitymojo/models/ltx2/parity/ltx2_conditioning_mask_gate.mojo
#
# LTX2 intrinsic-conditioning MASK builder gate (P5.5 unit 1). HOST-ONLY, no GPU,
# no torch oracle: the mask math is pure index arithmetic, so it is EXACT-compared
# against the official LTX-2 flexible.py formulas (temporal :521-539, spatial-crop
# :541-570, first-frame :491-492, union). Grid = the trainer video geometry
# 4x9x16 (seq=576, tokens_per_frame=144).
#
# Run (no GPU):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . \
#       serenitymojo/models/ltx2/parity/ltx2_conditioning_mask_gate.mojo \
#       -o /tmp/ltx2_conditioning_mask_gate && /tmp/ltx2_conditioning_mask_gate

from serenitymojo.training.ltx2.conditioning_mask import (
    temporal_mask, first_frame_mask, spatial_crop_mask, union_into,
)

comptime NF = 4
comptime NH = 9
comptime NW = 16
comptime SEQ = NF * NH * NW          # 576
comptime TPF = NH * NW               # 144 tokens per frame


def _count(m: List[Bool]) -> Int:
    var n = 0
    for i in range(len(m)):
        if m[i]:
            n += 1
    return n


def main() raises:
    print("=== LTX2 intrinsic-conditioning MASK gate (host-only, 4x9x16) ===")
    print("  seq", SEQ, " tokens_per_frame", TPF)
    var fails = 0

    def check(name: String, cond: Bool) raises:
        if cond:
            print("  [PASS]", name)
        else:
            print("  [FAIL]", name)

    # ── FIRST-FRAME = temporal(num_frames=1, from_end=False): first TPF tokens ──
    var ff = first_frame_mask(NF, NH, NW)
    check("first_frame len == seq", len(ff) == SEQ)
    check("first_frame count == TPF (144)", _count(ff) == TPF)
    check("first_frame m[0]=T m[143]=T m[144]=F m[575]=F",
          ff[0] and ff[TPF - 1] and (not ff[TPF]) and (not ff[SEQ - 1]))

    # ── PREFIX tb=2: first 2*TPF tokens ─────────────────────────────────────────
    var pre = temporal_mask(NF, NH, NW, 2, False)
    check("prefix(tb=2) count == 288", _count(pre) == 2 * TPF)
    check("prefix m[287]=T m[288]=F", pre[2 * TPF - 1] and (not pre[2 * TPF]))

    # ── SUFFIX tb=2: last 2*TPF tokens (from_end) ───────────────────────────────
    var suf = temporal_mask(NF, NH, NW, 2, True)
    check("suffix(tb=2) count == 288", _count(suf) == 2 * TPF)
    check("suffix m[288]=T m[287]=F m[575]=T",
          suf[2 * TPF] and (not suf[2 * TPF - 1]) and suf[SEQ - 1])

    # ── SPATIAL_CROP pixel (y1,x1,y2,x2)=(64,64,192,256) -> //32 latent ──────────
    #   ly1=2 ly2=6 lx1=2 lx2=8; rect h in [2,6) w in [2,8) = 4*6=24/frame *4 = 96.
    var sc = spatial_crop_mask(NF, NH, NW, 64, 64, 192, 256)
    check("spatial_crop count == 96 (24/frame * 4)", _count(sc) == 24 * NF)
    # frame 0, h=3 w=4 -> token 3*16+4 = 52 (inside)
    check("spatial_crop inside (f0,h3,w4)=T", sc[3 * NW + 4])
    # frame 0, h=1 (above ly1=2) -> token 1*16+4 = 20 (outside)
    check("spatial_crop above-rect (f0,h1)=F", not sc[1 * NW + 4])
    # frame 0, h=3 w=1 (left of lx1=2) -> token 3*16+1 = 49 (outside)
    check("spatial_crop left-of-rect (f0,h3,w1)=F", not sc[3 * NW + 1])
    # SAME rectangle tiled: frame 3, h=3 w=4 -> token 3*144 + 52 = 484 (inside)
    check("spatial_crop tiled to frame 3 (f3,h3,w4)=T", sc[3 * TPF + 3 * NW + 4])

    # clamp: y2=100000 -> ly2 clamped to NH; region covers rows [2, NH)
    var scc = spatial_crop_mask(NF, NH, NW, 64, 0, 100000, 100000)
    check("spatial_crop y2 clamp (f0,h8,w0)=T", scc[8 * NW + 0])

    # ── UNION per-token OR: first_frame ∪ suffix(tb=2) = 0..143 + 288..575 ───────
    var acc = first_frame_mask(NF, NH, NW)
    union_into(acc, temporal_mask(NF, NH, NW, 2, True))
    check("union count == 144+288 == 432 (disjoint)", _count(acc) == TPF + 2 * TPF)
    check("union m[143]=T m[144]=F m[287]=F m[288]=T",
          acc[TPF - 1] and (not acc[TPF]) and (not acc[2 * TPF - 1]) and acc[2 * TPF])

    # ── UNION with OVERLAP: prefix(tb=2) ∪ first_frame -> still 0..287 (288) ─────
    var acc2 = temporal_mask(NF, NH, NW, 2, False)
    union_into(acc2, first_frame_mask(NF, NH, NW))
    check("union overlap count == 288 (first_frame subset of prefix)", _count(acc2) == 2 * TPF)

    # recount fails from the closure via a sentinel scan (closure can't mutate outer)
    var all_pass = (len(ff) == SEQ and _count(ff) == TPF and _count(pre) == 2 * TPF
        and _count(suf) == 2 * TPF and _count(sc) == 24 * NF
        and _count(acc) == TPF + 2 * TPF and _count(acc2) == 2 * TPF
        and ff[0] and (not ff[TPF]) and suf[SEQ - 1] and sc[3 * NW + 4]
        and (not sc[1 * NW + 4]) and sc[3 * TPF + 3 * NW + 4] and scc[8 * NW])
    if not all_pass:
        raise Error("LTX2 CONDITIONING MASK GATE FAIL")
    print("LTX2 CONDITIONING MASK GATE PASS (temporal prefix/suffix/first-frame + spatial-crop + union)")
