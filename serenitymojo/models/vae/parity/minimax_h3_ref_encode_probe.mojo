# serenitymojo/models/vae/parity/minimax_h3_ref_encode_probe.mojo
#
# HOST probe for the CPU half of MiniMax-H3 ref2va UNIT 2 (ref-encode).
#
# Gates `serenitymojo/models/vae/minimax_h3_ref_encode.mojo`: pixel
# normalization, the fp16 round trip, latent normalization, the channel-slowest
# condition-row composition, the channel-major audio row packing, and the
# 0.999 noise mix. No DeviceContext, no GPU, no weights, no oracle file.
#
# The VAE forward itself is NOT gated here — it is the remaining seam and needs
# the GPU. Its gate is `minimax_h3_ref_encode_gate.mojo`, written and marked
# PENDING-GPU.
#
# ── WHERE THE EXPECTED NUMBERS COME FROM ─────────────────────────────────────
# Derived INDEPENDENTLY in numpy float32, transcribing the vendor's ops by hand
# from encoders.py:566-604 and encoders.py:612-628 — not read back from this
# port. Every comparison is EXACT float32 equality: these are all elementwise
# add/sub/mul/div in one precision, so there is no legitimate reason for a bit
# to differ, and a tolerance would only hide an ordering bug.
#
# ── WHAT EACH CHECK IS ACTUALLY FOR ──────────────────────────────────────────
# [1] Pixel normalize doubles as a LAYOUT check: input is channels-LAST uint8
#     (what media-in produces) and output channels-FIRST f32 (what the VAE
#     wants), so a transpose bug shows up as a value mismatch, not just a shape.
# [2] The fp16 round trip is step 5 of the visual recipe and is load-bearing for
#     bit-exactness ("~11 bits of every conditioning latent"). 1e-8 flushing to
#     zero and 1/3 landing on 0.333251953125 are what prove it is a REAL fp16
#     narrowing rather than a no-op cast.
# [3] The composition (fp16 -> normalize -> patchify) is checked as ONE chain,
#     because the ORDER is the thing that goes wrong: rounding after
#     normalizing quantizes a differently-scaled value and lands on different
#     bits. The 24-channel single-row case also pins the channel-SLOWEST column
#     order — column = c*4 + ih*2 + iw for a (1,2,2) patch.
# [4] Audio rows are channel-MAJOR: all T rows of the left channel, then all T
#     of the right, from a [2, C, T] posterior mode. The two stereo channels
#     carry different values, so a transpose bug cannot pass.
# [5] The mix is at the CONSTANT 0.999, and 1 - 0.999 in float32 is
#     0.0009999871253967285, NOT 0.001 — the probe pins that too, since a port
#     that hardcoded 0.001 would drift.
#
# Run (no GPU, no weights):
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/vae/parity/minimax_h3_ref_encode_probe.mojo \
#     -o <scratch>/minimax_h3_ref_encode_probe \
#   && <scratch>/minimax_h3_ref_encode_probe

from std.collections import List

from serenitymojo.models.vae.minimax_h3_ref_encode import (
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    minimax_h3_audio_condition_rows,
    minimax_h3_fp16_round,
    minimax_h3_mix_condition_rows,
    minimax_h3_pixel_normalize_frames,
    minimax_h3_video_condition_rows,
    minimax_h3_video_latents_mean,
    minimax_h3_video_latents_std,
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

    def eq_int(mut self, label: String, got: Int, want: Int):
        if got == want:
            self.ok(label, String(got))
        else:
            self.fail(label, String("got ") + String(got) + ", want " + String(want))

    def eq_f32(
        mut self, label: String, got: List[Float32], want: List[Float32]
    ):
        if len(got) != len(want):
            self.fail(
                label,
                String("length ") + String(len(got)) + " != " + String(len(want)),
            )
            return
        var bad = 0
        var first = -1
        for i in range(len(want)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            self.ok(
                label, String("exact over ") + String(len(want)) + " float32 values"
            )
        else:
            self.fail(
                label,
                String(bad) + " of " + String(len(want)) + " differ; first at "
                + String(first) + ": got " + String(got[first]) + ", want "
                + String(want[first]),
            )


def main() raises:
    print("MiniMax-H3 ref2va UNIT 2 probe — ref-encode, CPU half")
    print("")
    var report = Report()

    # ── [1] pixel normalize, channels-last uint8 -> channels-first f32 ───────
    print("[1] pixel normalize (ImageNet over [0,1]), 1 frame 2x2")
    var rgb: List[UInt8] = [
        UInt8(0), UInt8(64), UInt8(128),
        UInt8(255), UInt8(32), UInt8(16),
        UInt8(7), UInt8(200), UInt8(99),
        UInt8(250), UInt8(5), UInt8(180),
    ]
    var want_pix: List[Float32] = [
        Float32(-2.1179039478302), Float32(2.248908281326294),
        Float32(-1.998030662536621), Float32(2.1632845401763916),
        Float32(-0.9152660369873047), Float32(-1.4754900932312012),
        Float32(1.465686321258545), Float32(-1.9481792449951172),
        Float32(0.4264925718307495), Float32(-1.5255773067474365),
        Float32(-0.07895416766405106), Float32(1.332810640335083),
    ]
    var got_pix = minimax_h3_pixel_normalize_frames(rgb, 1, 2, 2)
    report.eq_f32("pixel normalize + transpose", got_pix, want_pix)

    # ── [2] fp16 round trip ──────────────────────────────────────────────────
    print("")
    print("[2] fp16 round trip (step 5 — ~11 bits, load-bearing)")
    var fp16_in: List[Float32] = [
        Float32(0.10000000149011612), Float32(0.3333333432674408),
        Float32(65504.0), Float32(9.99999993922529e-09),
        Float32(-2.7182817459106445), Float32(0.0),
    ]
    var fp16_want: List[Float32] = [
        Float32(0.0999755859375), Float32(0.333251953125),
        Float32(65504.0), Float32(0.0), Float32(-2.71875), Float32(0.0),
    ]
    var fp16_got = List[Float32]()
    for i in range(len(fp16_in)):
        fp16_got.append(minimax_h3_fp16_round(fp16_in[i]))
    report.eq_f32("fp16 narrowing", fp16_got, fp16_want)

    # ── [3] latent constants ─────────────────────────────────────────────────
    print("")
    print("[3] video latent normalization constants")
    var lm = minimax_h3_video_latents_mean()
    var ls = minimax_h3_video_latents_std()
    report.eq_int("latents_mean length", len(lm), 24)
    report.eq_int("latents_std length", len(ls), 24)
    if lm[0] == Float32(0.858090341091156) and ls[23] == Float32(2.6127843856811523):
        report.ok("latent constants match config.json", "first mean / last std")
    else:
        report.fail("latent constants match config.json", "endpoint mismatch")

    # ── [4] composition: fp16 -> normalize -> patchify (channel-slowest) ─────
    print("")
    print("[4] condition rows: fp16 -> normalize -> patchify, C=24 T=1 2x2, patch (1,2,2)")
    var latents = List[Float32]()
    for i in range(96):
        latents.append(Float32(((i * 37) % 101) - 50) / Float32(11.0))
    var want_rows: List[Float32] = [
        Float32(-4.421682834625244), Float32(-1.668658971786499), Float32(1.0827672481536865), Float32(-3.6771042346954346),
        Float32(0.5388422012329102), Float32(3.1740989685058594), Float32(-1.3831493854522705), Float32(1.2507688999176025),
        Float32(1.7430342435836792), Float32(-1.713736891746521), Float32(0.28501713275909424), Float32(-3.1723341941833496),
        Float32(-0.2280111461877823), Float32(1.6890043020248413), Float32(-1.6264030933380127), Float32(0.290056049823761),
        Float32(2.3253960609436035), Float32(-1.3956828117370605), Float32(0.7558976411819458), Float32(2.907478094100952),
        Float32(-0.08128775656223297), Float32(1.452001929283142), Float32(-1.199765920639038), Float32(0.3333013355731964),
        Float32(3.0890350341796875), Float32(-2.9383931159973145), Float32(0.5470040440559387), Float32(4.031895637512207),
        Float32(-1.8094472885131836), Float32(1.3724993467330933), Float32(4.556293487548828), Float32(-0.9492868781089783),
        Float32(3.1230883598327637), Float32(-3.787496328353882), Float32(0.20685185492038727), Float32(4.201779842376709),
        Float32(-3.585984945297241), Float32(0.7656189799308777), Float32(5.115960121154785), Float32(-2.4098076820373535),
        Float32(0.41589319705963135), Float32(-2.6530044078826904), Float32(-0.8787415623664856), Float32(0.8955211043357849),
        Float32(-3.857011079788208), Float32(-0.30393946170806885), Float32(3.249260902404785), Float32(-2.895756244659424),
        Float32(1.7062057256698608), Float32(5.914426803588867), Float32(-1.3638663291931152), Float32(2.8431336879730225),
        Float32(-8.193819999694824), Float32(-0.7191030979156494), Float32(6.757784366607666), Float32(-6.175093173980713),
        Float32(0.5815891027450562), Float32(5.254331111907959), Float32(-2.8296430110931396), Float32(1.8446252346038818),
        Float32(6.04622745513916), Float32(-2.339240312576294), Float32(2.5095760822296143), Float32(-5.878707408905029),
        Float32(-0.05175463482737541), Float32(1.0839039087295532), Float32(-0.8803715109825134), Float32(0.2552870810031891),
        Float32(1.3024450540542603), Float32(-0.7984856367111206), Float32(0.41594967246055603), Float32(-1.68427574634552),
        Float32(0.009792880155146122), Float32(1.1129661798477173), Float32(-0.7949312329292297), Float32(0.30788183212280273),
        Float32(1.338884711265564), Float32(-1.4201936721801758), Float32(0.1749129742383957), Float32(1.7695565223693848),
        Float32(-0.514398992061615), Float32(0.5124707818031311), Float32(-1.2640585899353027), Float32(-0.23689080774784088),
        Float32(0.7222983837127686), Float32(-1.1173619031906128), Float32(-0.05362498387694359), Float32(1.0100734233856201),
        Float32(-1.1359703540802002), Float32(0.33806610107421875), Float32(1.8121024370193481), Float32(-0.7375010848045349),
        Float32(0.4606490135192871), Float32(-1.7658580541610718), Float32(-0.47880464792251587), Float32(0.8086224794387817),
    ]
    var got_rows = minimax_h3_video_condition_rows(latents, 24, 1, 2, 2, 1, 2, 2)
    report.eq_int("row buffer size", len(got_rows), 96)
    report.eq_f32("fp16 -> normalize -> patchify", got_rows, want_rows)

    # ── [5] audio condition rows, channel-major ──────────────────────────────
    print("")
    print("[5] audio condition rows: [2,C,T] mode -> transpose -> normalize -> [2T,C]")
    var mode = List[Float32]()
    for i in range(24):
        mode.append(Float32(((i * 13) % 29) - 14) / Float32(7.0))
    var amean: List[Float32] = [
        Float32(0.25), Float32(-0.5), Float32(1.5), Float32(0.125)
    ]
    var astd: List[Float32] = [
        Float32(2.0), Float32(0.5), Float32(4.0), Float32(1.0)
    ]
    var want_arows: List[Float32] = [
        Float32(-1.125), Float32(-0.14285719394683838), Float32(-0.1607142835855484), Float32(-1.9821428060531616),
        Float32(-0.1964285671710968), Float32(3.5714285373687744), Float32(-0.7321428656578064), Float32(-0.125),
        Float32(0.7321428656578064), Float32(-1.0), Float32(-0.2678571343421936), Float32(1.7321428060531616),
        Float32(-0.3392857313156128), Float32(3.0), Float32(-0.8035714626312256), Float32(-0.4107142984867096),
        Float32(0.5892857313156128), Float32(-1.5714285373687744), Float32(-0.3392857015132904), Float32(1.4464285373687744),
        Float32(-0.5535714626312256), Float32(2.142857074737549), Float32(0.125), Float32(-0.8392857313156128),
    ]
    var got_arows = minimax_h3_audio_condition_rows(mode, 4, 3, amean, astd)
    report.eq_f32("audio rows (channel-major)", got_arows, want_arows)

    # ── [6] the 0.999 noise mix ──────────────────────────────────────────────
    print("")
    print("[6] condition noise mix at the CONSTANT t = 0.999")
    var one_minus = Float32(1.0) - Float32(0.999)
    if one_minus == Float32(0.0009999871253967285):
        report.ok("1 - 0.999 in float32", "0.0009999871253967285, not 0.001")
    else:
        report.fail("1 - 0.999 in float32", String(one_minus))

    var mix_x: List[Float32] = [
        Float32(1.0), Float32(-2.5), Float32(0.0),
        Float32(0.0010000000474974513), Float32(7.25),
    ]
    var mix_n: List[Float32] = [
        Float32(-1.0), Float32(3.5), Float32(2.0), Float32(-4.0), Float32(0.5)
    ]
    var mix_want: List[Float32] = [
        Float32(0.9980000257492065), Float32(-2.49399995803833),
        Float32(0.001999974250793457), Float32(-0.003000948578119278),
        Float32(7.243250370025635),
    ]
    var mix_got = minimax_h3_mix_condition_rows(mix_x, mix_n)
    report.eq_f32("scale_noise(x, 0.999, noise)", mix_got, mix_want)

    # Length mismatch must be refused, not silently zipped short.
    var short_noise: List[Float32] = [Float32(1.0), Float32(2.0)]
    var refused = False
    try:
        var bad = minimax_h3_mix_condition_rows(mix_x, short_noise)
        _ = len(bad)
    except:
        refused = True
    if refused:
        report.ok("mismatched noise length", "rejected")
    else:
        report.fail("mismatched noise length", "accepted a short noise buffer")

    # ── [7] the encode seed is the vendor's fixed 42 ─────────────────────────
    print("")
    print("[7] constants")
    report.eq_int("keyframe encode seed", MINIMAX_H3_KEYFRAME_ENCODE_SEED, 42)

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_ref_encode probe FAILED")
