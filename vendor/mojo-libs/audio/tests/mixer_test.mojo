# audio/tests/mixer_test.mojo — self-test for the integer audio mixer.
#
# Two layers of checks:
#   (A) hand-computed worked vectors from the parity SPEC, asserted bit-exact
#       (seeds the test, documents the model, exercises _wrap_i16 directly);
#   (B) the documented ORACLE parity cross-check: load /tmp/mixer_oracle.json,
#       run mix_tracks_i16 on each case's (tracks, vols), and assert the Mojo
#       `out` equals the oracle `out` element-for-element (MIXER.md lines 128-137).
#       This covers the broader saturation/mixed-vol/extremes cases the hand
#       vectors don't reach (single_vol50, pair_pos/neg, neg_trunc, three_tracks,
#       extremes, stereo_buf, zero_vol, ...).
#
# Prints a single grep-able 'MIXER_SELFTEST: pass' / '...: fail' line (plus
# per-case lines), AND raises on failure so the process exit code is non-zero
# for harnesses that check status rather than grep stdout.
#
# Run the oracle first to (re)generate the JSON, then this test:
#   python3 /home/alex/MOJO-libs/audio/tests/mixer_oracle.py
#   mojo audio/tests/mixer_test.mojo

from audio.mixer import (
    gain_i16,
    soft_add_i16,
    mix2_i16,
    mix_tracks_i16,
    _wrap_i16,
)
from json.parser import loads
from json.value import JSONValue


comptime ORACLE_PATH = "/tmp/mixer_oracle.json"


def _check_i16(name: String, got: Int16, want: Int16, mut fails: Int) -> None:
    var ok = Int(got) == Int(want)
    if not ok:
        fails += 1
    var tag = "ok" if ok else "FAIL"
    print(
        "  [", tag, "] ", name,
        "  got=", Int(got), " want=", Int(want),
    )


def _check_int(name: String, got: Int, want: Int, mut fails: Int) -> None:
    var ok = got == want
    if not ok:
        fails += 1
    var tag = "ok" if ok else "FAIL"
    print(
        "  [", tag, "] ", name,
        "  got=", got, " want=", want,
    )


def main() raises:
    var fails = 0

    print("== gain_i16 (step 1: volume gain, trunc-toward-zero) ==")
    # gain(20000, 50) = 20000*50/100 = 10000
    _check_i16("gain(20000,50)", gain_i16(Int16(20000), 50), Int16(10000), fails)
    # gain(-7, 50) = trunc(-350/100) = -3  (NOT -4 — floor would give -4)
    _check_i16("gain(-7,50)", gain_i16(Int16(-7), 50), Int16(-3), fails)
    # vol=0 -> 0 ; vol=100 -> identity ; clamp edges
    _check_i16("gain(12345,0)", gain_i16(Int16(12345), 0), Int16(0), fails)
    _check_i16("gain(12345,100)", gain_i16(Int16(12345), 100), Int16(12345), fails)
    _check_i16("gain(-32768,100)", gain_i16(Int16(-32768), 100), Int16(-32768), fails)
    # another negative trunc check: -200*50/100 = -100 exact
    _check_i16("gain(-200,50)", gain_i16(Int16(-200), 50), Int16(-100), fails)
    # negative trunc: -3*50/100 = trunc(-1.5) = -1 (floor would give -2)
    _check_i16("gain(-3,50)", gain_i16(Int16(-3), 50), Int16(-1), fails)

    print("== soft_add_i16 (step 2: saturating soft-add, verbatim C++) ==")
    # soft_add(0, 12345) = 12345  (accumulator seed folds clean)
    _check_i16("soft_add(0,12345)", soft_add_i16(Int16(0), Int16(12345)), Int16(12345), fails)
    _check_i16("soft_add(12345,0)", soft_add_i16(Int16(12345), Int16(0)), Int16(12345), fails)
    # both>0: 40000 - (20000*20000/32767) = 40000 - 12207 = 27793
    _check_i16("soft_add(20000,20000)", soft_add_i16(Int16(20000), Int16(20000)), Int16(27793), fails)
    # both<0: -40000 + (400000000/32767) = -40000 + 12207 = -27793
    _check_i16("soft_add(-20000,-20000)", soft_add_i16(Int16(-20000), Int16(-20000)), Int16(-27793), fails)
    # mixed signs (else branch, negative product, trunc-toward-zero):
    #   d=10000 s=-10000 -> 0 - (-100000000/32767) = 0 - (-3051) = 3051
    _check_i16("soft_add(10000,-10000)", soft_add_i16(Int16(10000), Int16(-10000)), Int16(3051), fails)

    print("== _wrap_i16 (defensive (int16) narrowing — branch not hit by mix path) ==")
    # In-range values pass through unchanged.
    _check_i16("wrap(0)", _wrap_i16(0), Int16(0), fails)
    _check_i16("wrap(32767)", _wrap_i16(32767), Int16(32767), fails)
    _check_i16("wrap(-32768)", _wrap_i16(-32768), Int16(-32768), fails)
    # Out-of-range values exercise the two's-complement wrap branch the soft-add
    # never reaches: 40000 -> 40000-65536 = -25536 ; 70000 -> 70000-65536 = 4464.
    _check_i16("wrap(40000)", _wrap_i16(40000), Int16(-25536), fails)
    _check_i16("wrap(70000)", _wrap_i16(70000), Int16(4464), fails)

    print("== mix2_i16 (step 3a: two-track fold) ==")
    # A=[20000,-7]  B=[20000,-20000]  va=vb=50  n=2
    # lane0: g=10000,10000 -> acc=soft_add(0,10000)=10000 ->
    #        soft_add(10000,10000)=20000-(100000000/32767=3051)=16949
    # lane1: g(-7,50)=-3, g(-20000,50)=-10000 -> soft_add(0,-3)=-3 ->
    #        soft_add(-3,-10000) both<0 = -10003 + (30000/32767=0) = -10003
    var a: List[Int16] = [Int16(20000), Int16(-7)]
    var b: List[Int16] = [Int16(20000), Int16(-20000)]
    var out2 = List[Int16]()
    mix2_i16(a, b, 50, 50, 2, out2)
    _check_i16("mix2[0]", out2[0], Int16(16949), fails)
    _check_i16("mix2[1]", out2[1], Int16(-10003), fails)

    print("== mix_tracks_i16 (step 3b: general K-track fold) ==")
    # Same two tracks via the K-track path must equal mix2 exactly (fold order).
    var tracks = List[List[Int16]]()
    tracks.append(a.copy())
    tracks.append(b.copy())
    var vols: List[Int] = [50, 50]
    var outk = List[Int16]()
    mix_tracks_i16(tracks, vols, outk, 2)
    _check_i16("mixK[0]==mix2[0]", outk[0], out2[0], fails)
    _check_i16("mixK[1]==mix2[1]", outk[1], out2[1], fails)

    # 3-track fold: verify accumulator chains across tracks in order.
    # t0=[10000], t1=[10000], t2=[10000], vols=[100,100,100], n=1
    #   acc=soft_add(0,10000)=10000
    #   acc=soft_add(10000,10000)=20000-(100000000/32767=3051)=16949
    #   acc=soft_add(16949,10000) both>0 = 26949 - (169490000/32767=5172)=21777
    var t3 = List[List[Int16]]()
    var row3: List[Int16] = [Int16(10000)]
    t3.append(row3.copy())
    t3.append(row3.copy())
    t3.append(row3.copy())
    var v3: List[Int] = [100, 100, 100]
    var out3 = List[Int16]()
    mix_tracks_i16(t3, v3, out3, 1)
    _check_i16("mix3[0]", out3[0], Int16(21777), fails)

    # single first-track fold cleanliness: soft_add(0, gain) == gain
    var t1 = List[List[Int16]]()
    var row1: List[Int16] = [Int16(20000), Int16(-7)]
    t1.append(row1.copy())
    var v1: List[Int] = [50]
    var out1 = List[Int16]()
    mix_tracks_i16(t1, v1, out1, 2)
    _check_i16("mix1[0]==gain", out1[0], gain_i16(Int16(20000), 50), fails)
    _check_i16("mix1[1]==gain", out1[1], gain_i16(Int16(-7), 50), fails)

    # ---- (B) ORACLE PARITY CROSS-CHECK (MIXER.md lines 128-137) ----
    # Load /tmp/mixer_oracle.json, run mix_tracks_i16 on each kase, assert the
    # Mojo `out` equals the oracle `out` element-for-element.
    print("== oracle parity (mix_tracks_i16 vs /tmp/mixer_oracle.json) ==")
    var oracle_text: String
    try:
        oracle_text = open(ORACLE_PATH, "r").read()
    except:
        raise Error(
            "could not read oracle file "
            + ORACLE_PATH
            + " — first run: python3 "
            + "/home/alex/MOJO-libs/audio/tests/mixer_oracle.py"
        )
    var doc = loads(oracle_text)
    if not doc.is_object():
        raise Error("oracle JSON root is not an object: " + ORACLE_PATH)

    # Sanity-pin the division contract the oracle was generated under so we know
    # we are matching trunc-toward-zero (not floor).
    var div_sem = doc["div_semantics"].as_string()
    _check_int(
        "div_semantics is trunc-toward-zero",
        1 if (div_sem == "trunc-toward-zero (C '/'); NOT floor") else 0,
        1,
        fails,
    )

    var cases = doc["cases"]
    if not cases.is_array():
        raise Error("oracle JSON has no 'cases' array: " + ORACLE_PATH)
    var ncases = cases.length()
    _check_int("oracle kase count > 0", 1 if ncases > 0 else 0, 1, fails)

    for ci in range(ncases):
        var kase = cases[ci]
        var case_name = kase["name"].as_string()

        # Build Mojo tracks/vols from the JSON.
        var jt = kase["tracks"]
        var jv = kase["vols"]
        var jout = kase["out"]
        var ktracks = jt.length()
        var n = jt[0].length() if ktracks > 0 else 0

        var mtracks = List[List[Int16]]()
        for k in range(ktracks):
            var jrow = jt[k]
            var row = List[Int16]()
            for i in range(jrow.length()):
                row.append(Int16(jrow[i].as_int()))
            mtracks.append(row^)

        var mvols = List[Int]()
        for k in range(jv.length()):
            mvols.append(jv[k].as_int())

        var got = List[Int16]()
        mix_tracks_i16(mtracks, mvols, got, n)

        # Element-for-element comparison against the oracle `out`.
        var case_fail = 0
        var olen = jout.length()
        if len(got) != olen:
            case_fail += 1
        else:
            for i in range(olen):
                if Int(got[i]) != jout[i].as_int():
                    case_fail += 1
        if case_fail != 0:
            fails += 1
            print("  [ FAIL ]  oracle kase '", case_name, "' : ", case_fail, " lane(s) differ")
            # dump the first divergence for debugging
            for i in range(olen):
                if i < len(got) and Int(got[i]) != jout[i].as_int():
                    print(
                        "      lane", i, " got=", Int(got[i]),
                        " want=", jout[i].as_int(),
                    )
                    break
        else:
            print("  [ ok ]  oracle kase '", case_name, "' (", olen, " lanes)")

    print("")
    if fails == 0:
        print("MIXER_SELFTEST: pass")
    else:
        print("MIXER_SELFTEST: fail (", fails, " kase(s))")
        # Also fail the process exit code for harnesses that check status (not just
        # grep stdout) — mirrors wav_test.mojo's 'raise Error' on failure.
        raise Error("MIXER_SELFTEST FAILED (" + String(fails) + " kase(s))")
