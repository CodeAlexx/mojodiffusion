#!/usr/bin/env python3
# mixer_oracle.py — pure-python (numpy optional) reference for the yangfan/mixer
# audio model. This is the PARITY ORACLE: the Mojo mixer must equal this oracle
# bit-for-bit on every (tracks, vols) case.
#
# MODEL (all integer math, int16 PCM, interleaved stereo, N = sampleFrames*2):
#
#   (1) per-track volume gain  (template.json audio.trackers.{name}.volume, 0..100):
#         g = trunc_toward_zero(int32(sample) * volume, 100)
#         g = clamp(g, -32768, 32767)
#       C '/' truncates toward zero. Mojo Int '//' is FLOOR (toward -inf) and
#       numpy '//' is ALSO floor — both differ for negative samples. We implement
#       trunc-toward-zero explicitly so Mojo == oracle.
#
#   (2) pairwise saturating soft-add — VERBATIM from
#       mixerEngine/engine_tracker.cpp, per lane i with int32 d, s:
#         if (d < 0 && s < 0)  r = d + s + (d * s / 32767);
#         else                 r = d + s - (d * s / 32767);
#         dest[i] = (int16) r;
#       '/32767' is integer division TRUNCATED TOWARD ZERO (d*s can be negative in
#       the else branch when signs differ). Result stored as int16 (wrap).
#
#   (3) mix K tracks each int16[N], vols[K] -> out int16[N]:
#         out[i] = 0
#         for k in 0..K:            # fold in TRACK ORDER (soft-add is non-assoc.)
#           for i in 0..N:
#             g = gain(tracks[k][i], vols[k])
#             out[i] = soft_add(out[i], g)
#       soft_add(0, g) = 0 + g - 0 = g, so the first track folds in cleanly.
#
# IMPORTANT: every value below is a python int (arbitrary precision). We never let
# the intermediate products overflow; only the FINAL stored sample is wrapped to
# int16 to match the C (int16) cast exactly.

import json

# numpy is allowed but NOT required and NOT used for any mix math (its '//' is
# floor and would corrupt the trunc-toward-zero contract). Imported only to prove
# availability / for callers who hand us np arrays.
try:
    import numpy as _np  # noqa: F401
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False

INT16_MIN = -32768
INT16_MAX = 32767


# ---------------------------------------------------------------------------
# trunc-toward-zero integer division (the load-bearing primitive)
# ---------------------------------------------------------------------------
def trunc_div(a: int, b: int) -> int:
    """Integer division truncated toward zero, matching C '/' for ints.

    NOT floor division. For a=-350, b=100 -> -3 (python // gives -4).
    """
    a = int(a)
    b = int(b)
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q


def clamp16(x: int) -> int:
    if x < INT16_MIN:
        return INT16_MIN
    if x > INT16_MAX:
        return INT16_MAX
    return x


def wrap16(r: int) -> int:
    """Wrap an int into int16 range, matching the C (int16) cast.

    The soft-add formula keeps r within int16 range, but we wrap anyway to match
    C exactly (two's-complement truncation to 16 bits).
    """
    r = int(r) & 0xFFFF
    if r >= 0x8000:
        r -= 0x10000
    return r


# ---------------------------------------------------------------------------
# (1) per-track volume gain
# ---------------------------------------------------------------------------
def gain(sample: int, volume: int) -> int:
    """g = clamp(trunc(sample*volume / 100), -32768, 32767). volume in 0..100."""
    g = trunc_div(int(sample) * int(volume), 100)
    return clamp16(g)


# ---------------------------------------------------------------------------
# (2) pairwise saturating soft-add (verbatim engine_tracker.cpp)
# ---------------------------------------------------------------------------
def soft_add(d: int, s: int) -> int:
    d = int(d)
    s = int(s)
    if d < 0 and s < 0:
        r = d + s + trunc_div(d * s, 32767)
    else:
        r = d + s - trunc_div(d * s, 32767)
    return wrap16(r)


# ---------------------------------------------------------------------------
# (3) K-track accumulator fold
# ---------------------------------------------------------------------------
def mix_tracks(tracks, vols):
    """tracks: list of int16 lists (each length N). vols: list of int 0..100.

    Returns: out int16 list of length N. Folds in track order.
    """
    if not tracks:
        return []
    n = len(tracks[0])
    for t in tracks:
        if len(t) != n:
            raise ValueError("all tracks must have the same length N")
    if len(vols) != len(tracks):
        raise ValueError("vols length must equal number of tracks")

    out = [0] * n
    for k in range(len(tracks)):
        tk = tracks[k]
        vk = vols[k]
        for i in range(n):
            g = gain(tk[i], vk)
            out[i] = soft_add(out[i], g)
    return out


# ---------------------------------------------------------------------------
# worked vectors from the SPEC — print PASS/FAIL for each
# ---------------------------------------------------------------------------
def _run_worked_vectors():
    cases = [
        ("gain(20000, 50)",        gain(20000, 50),         10000),
        ("gain(-7, 50)",           gain(-7, 50),            -3),     # trunc, NOT -4
        ("soft_add(0, 12345)",     soft_add(0, 12345),       12345),
        ("soft_add(20000, 20000)", soft_add(20000, 20000),   27793),
        ("soft_add(-20000,-20000)",soft_add(-20000, -20000), -27793),
    ]
    all_ok = True
    for name, got, want in cases:
        ok = (got == want)
        all_ok = all_ok and ok
        print("  [{}] {} = {} (want {})".format(
            "PASS" if ok else "FAIL", name, got, want))
    return all_ok


# ---------------------------------------------------------------------------
# JSON dump of cross-check cases for the Mojo test
# ---------------------------------------------------------------------------
def _build_json_cases():
    cases = []

    def add(name, tracks, vols):
        out = mix_tracks(tracks, vols)
        cases.append({
            "name": name,
            "tracks": [list(map(int, t)) for t in tracks],
            "vols": list(map(int, vols)),
            "out": list(map(int, out)),
        })

    # single track, gain only (vol 100 = identity, vol 50 = halve w/ trunc)
    add("single_vol100", [[0, 12345, -32768, 32767, -7, 7]], [100])
    add("single_vol50",  [[0, 12345, -32768, 32767, -7, 7]], [50])

    # the worked soft-add pair (two tracks, vol 100, fold order matters)
    add("pair_pos", [[20000, 0], [20000, 0]], [100, 100])
    add("pair_neg", [[-20000, 0], [-20000, 0]], [100, 100])

    # negative-sample trunc parity (vol 50): -7*50/100 = -3, NOT -4
    add("neg_trunc", [[-7, -3, -1, -350, -351]], [50])

    # K=3 fold, mixed signs, mixed vols — exercises accumulator order
    add("three_tracks", [
        [10000, -10000, 32767, -32768, 5, -5],
        [10000,  10000, 30000,  -100, 5, -5],
        [-5000,  20000, -5000,  10000, 3,  3],
    ], [100, 50, 75])

    # extremes: saturating add of two near-max positives, near-min negatives
    add("extremes", [
        [32767, -32768, 32767, -32768],
        [32767, -32768, -32768, 32767],
    ], [100, 100])

    # interleaved-stereo flavored buffer (N=8 = 4 frames * 2 ch), two tracks
    add("stereo_buf", [
        [ 1000, -1000,  2000, -2000,  3000, -3000,  4000, -4000],
        [  500,   500,  -500,  -500,  1500, -1500, -1500,  1500],
    ], [80, 60])

    # zero volume drops a track entirely (gain -> 0, soft_add(acc,0)=acc)
    add("zero_vol", [
        [10000, -10000, 32767, -32768],
        [20000,  20000, -5000,   5000],
    ], [100, 0])

    return cases


def main():
    print("MIXER_ORACLE worked vectors:")
    worked_ok = _run_worked_vectors()

    cases = _build_json_cases()
    dump = {
        "model": "yangfan/mixer int16 soft-add",
        "int16_min": INT16_MIN,
        "int16_max": INT16_MAX,
        "div_semantics": "trunc-toward-zero (C '/'); NOT floor",
        "have_numpy": HAVE_NUMPY,
        "cases": cases,
    }
    out_path = "/tmp/mixer_oracle.json"
    with open(out_path, "w") as f:
        json.dump(dump, f, indent=2)
    print("MIXER_ORACLE wrote {} ({} cases)".format(out_path, len(cases)))

    if worked_ok:
        print("MIXER_ORACLE: pass")
    else:
        print("MIXER_ORACLE: fail")


if __name__ == "__main__":
    main()
