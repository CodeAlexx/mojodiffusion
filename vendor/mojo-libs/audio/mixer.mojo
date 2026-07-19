# audio/mixer.mojo — pure-Mojo integer audio mixer. No FFI, no GPU, no float.
#
# Reusable, parity-faithful reproduction of the yangfan/mixer audio model. All
# math is INTEGER on int16 PCM, interleaved stereo (2 channels): a buffer holds
# N int16 "lanes" where N = sampleFrames * 2. The mix path never touches float.
#
# Two-step model:
#   (1) PER-TRACK VOLUME GAIN — template.json audio.trackers.{name}.volume is an
#       integer 0..100, applied as integer gain BEFORE mixing:
#           g = (int32(sample) * volume) / 100   # C '/' TRUNCATES TOWARD ZERO
#           g = clamp(g, -32768, 32767)
#       C '/' truncates toward zero; Mojo Int '//' is FLOOR (toward -inf), which
#       differs for negative samples (-7*50/100 -> C=-3, Mojo//=-4). We implement
#       trunc-toward-zero explicitly. numpy '//' is also floor; the oracle must
#       use the SAME trunc-toward-zero to stay bit-exact with Mojo.
#
#   (2) PAIRWISE SATURATING SOFT-ADD — VERBATIM from
#       mixerEngine/engine_tracker.cpp, per lane i with int32 d=dest[i], s=sour[i]:
#           if (d < 0 && s < 0)  r = d + s + (d * s / 32767);
#           else                 r = d + s - (d * s / 32767);
#           dest[i] = (int16) r;
#       d*s is an int32 product; '/32767' is integer division TRUNCATED TOWARD
#       ZERO (the else branch can be negative when signs differ). The result is
#       cast to int16 (wrap) to match C exactly; r is already within int16 range.
#
#   (3) MIX K TRACKS each int16[N], vols[K] (each 0..100) -> out int16[N]:
#           out[i] = 0
#           for k in 0..K: for i in 0..N: out[i] = soft_add(out[i], gain(t[k][i], vols[k]))
#       Folding order = track order (load-bearing: soft-add is NON-associative).
#       With out starting at 0: soft_add(0, g) = 0+g-0 = g, first track folds clean.


# ---- trunc-toward-zero integer division (C '/' semantics) ----
def _trunc_div(a: Int, b: Int) -> Int:
    """C integer division: truncate toward zero (Mojo '//' is floor — differs for
    negative operands). Used for the volume '/100' and the soft-add '/32767'."""
    # |a|//|b| is exact magnitude; reapply the sign of the true quotient.
    var aa = a if a >= 0 else -a
    var bb = b if b >= 0 else -b
    var q = aa // bb
    var neg = (a < 0) != (b < 0)
    return -q if neg else q


def _clamp_i16(v: Int) -> Int:
    """Clamp an Int into the signed 16-bit range [-32768, 32767]."""
    return 32767 if v > 32767 else (-32768 if v < -32768 else v)


def _wrap_i16(v: Int) -> Int16:
    """Build an Int16 by wrapping v into [-32768, 32767] (C '(int16)r' cast).
    The soft-add result is provably already in range (worst case both-min gives
    r = -32767), so the wrap (w >= 32768) branch is DEFENSIVE parity with C's
    narrowing cast and is not normally reached by the mix path. It is exercised
    directly by the self-test (e.g. _wrap_i16(40000) == -25536) to lock the
    two's-complement narrowing against regression."""
    var w = v & 0xFFFF                       # low 16 bits, two's complement
    return Int16(w - 65536 if w >= 32768 else w)


# ---- step (1): per-track volume gain ----
def gain_i16(sample: Int16, volume: Int) -> Int16:
    """Integer per-track volume gain. Mirrors template.json upstream gain
    audio.trackers.{name}.volume (int 0..100), applied BEFORE the soft-add:
        g = (int32(sample) * volume) / 100   # TRUNCATE TOWARD ZERO (C '/')
        g = clamp(g, -32768, 32767)
    volume>=0 and 100>0, so the product sign = sample sign; trunc-toward-zero
    keeps Mojo == numpy/C oracle for negative samples (e.g. gain(-7,50) = -3)."""
    var prod = Int(sample) * volume          # int32-range product, done in Int
    var g = _trunc_div(prod, 100)
    return Int16(_clamp_i16(g))


# ---- step (2): pairwise saturating soft-add ----
def soft_add_i16(dest: Int16, sour: Int16) -> Int16:
    """Saturating soft-add, VERBATIM from mixerEngine/engine_tracker.cpp:
        int32 d = dest, s = sour;
        if (d < 0 && s < 0)  r = d + s + (d * s / 32767);
        else                 r = d + s - (d * s / 32767);
        dest = (int16) r;
    '/32767' truncates toward zero (negative in the else branch when signs
    differ); the int16 cast wraps. NON-associative — fold order matters."""
    var d = Int(dest)
    var s = Int(sour)
    # d*s is computed in Mojo's 64-bit Int by design. This is BIT-IDENTICAL to the
    # C int32 product ONLY because int16*int16 cannot overflow int32 (max |d*s| =
    # 32768*32768 = 1073741824 < INT32_MAX 2147483647), so a true int32 path never
    # wraps here. If the lane width ever changes (or out-of-range "int16" inputs are
    # fed in), the 64-bit path would silently diverge from a wrapping int32 path.
    var ds = _trunc_div(d * s, 32767)        # '/32767' trunc-toward-zero
    var r = (d + s + ds) if (d < 0 and s < 0) else (d + s - ds)
    return _wrap_i16(r)


# ---- step (3a): two-track convenience ----
def mix2_i16(
    a: List[Int16],
    b: List[Int16],
    va: Int,
    vb: Int,
    n: Int,
    mut out: List[Int16],
) -> None:
    """Mix two int16 tracks into `out` over `n` lanes (n = sampleFrames*2):
        out[i] = soft_add(soft_add(0, gain(a[i],va)), gain(b[i],vb))
    Fold order is a then b — soft-add is non-associative, so order is load-bearing.
    `out` is resized to n and overwritten."""
    out.clear()
    for i in range(n):
        var acc = Int16(0)
        acc = soft_add_i16(acc, gain_i16(a[i], va))
        acc = soft_add_i16(acc, gain_i16(b[i], vb))
        out.append(acc)


# ---- step (3b): general K-track fold ----
def mix_tracks_i16(
    tracks: List[List[Int16]],
    vols: List[Int],
    mut out: List[Int16],
    n: Int,
) -> None:
    """General K-track accumulator fold into `out` over `n` lanes:
        out[i] = 0
        for k in 0..K: for i in 0..n: out[i] = soft_add(out[i], gain(tracks[k][i], vols[k]))
    Folding order = track order (k outer). soft_add(0, g) = g, so the first track
    folds in cleanly. `out` is resized to n (zeroed) and accumulated in place."""
    out.clear()
    for _ in range(n):
        out.append(Int16(0))
    var k = len(tracks)
    for ti in range(k):
        var vt = vols[ti]
        for i in range(n):
            out[i] = soft_add_i16(out[i], gain_i16(tracks[ti][i], vt))
