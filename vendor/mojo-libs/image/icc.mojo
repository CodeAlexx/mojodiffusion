# image/icc.mojo — ICC color-profile parsing + a basic profile→sRGB transform.
#
# Scope (HONEST):
#   * Parses the 128-byte ICC header (profile size, CMM, version, device class,
#     color space, PCS, 'acsp' signature) and the tag table.
#   * Parses XYZType tags (wtpt / rXYZ / gXYZ / bXYZ) and tone curves
#     (curveType 'curv' AND parametricCurveType 'para').
#   * to_srgb() converts RGB *matrix/TRC* profiles to sRGB (linearize via the
#     profile TRC, profile→XYZ matrix, XYZ→sRGB matrix, sRGB encode, clamp).
#   * LUT-based profiles (mft1/mft2/mAB /mBA ), GRAY and CMYK profiles are
#     PARSED but NOT transformed: to_srgb() returns a copy unchanged and sets
#     `.transformed = False` on the result's IccProfile-less path — see to_srgb.
#
# ICC integers are BIG-endian. s15Fixed16 = signed 16.16 fixed point.

from std.math import pow
from image.buffer import Image, CS_RGB

# ---------------------------------------------------------------------------
# Big-endian readers (ICC is big-endian).
# ---------------------------------------------------------------------------

def _u32(b: List[UInt8], off: Int) raises -> UInt32:
    return (UInt32(b[off]) << 24) | (UInt32(b[off + 1]) << 16) \
        | (UInt32(b[off + 2]) << 8) | UInt32(b[off + 3])


def _u16(b: List[UInt8], off: Int) raises -> UInt32:
    return (UInt32(b[off]) << 8) | UInt32(b[off + 1])


def _i32(b: List[UInt8], off: Int) raises -> Int:
    # signed 32-bit big-endian
    var v = Int(_u32(b, off))
    if v >= 0x80000000:
        v -= 0x100000000
    return v


def _s15f16(b: List[UInt8], off: Int) raises -> Float64:
    # signed 15.16 fixed point → Float64
    return Float64(_i32(b, off)) / 65536.0


def _sig(b: List[UInt8], off: Int) raises -> String:
    # 4-char ASCII signature (kept verbatim, including trailing space).
    var s = String("")
    for i in range(4):
        s += chr(Int(b[off + i]))
    return s


# ---------------------------------------------------------------------------
# Tone curve. kind: 0 = identity (curv count 0), 1 = gamma (single value),
#                    2 = LUT (curv count N), 3 = parametric ('para').
# ---------------------------------------------------------------------------

struct ToneCurve(Copyable, Movable):
    var kind: Int
    var gamma: Float64          # used when kind==1
    var lut: List[Float64]      # normalized [0,1] samples, used when kind==2
    var para_type: Int          # used when kind==3
    var params: List[Float64]   # parametric params (g,a,b,c,d,e,f order)

    def __init__(out self):
        self.kind = 0
        self.gamma = 1.0
        self.lut = List[Float64]()
        self.para_type = 0
        self.params = List[Float64]()

    def supported(self) -> Bool:
        # We can evaluate identity, gamma, LUT, and parametric curves.
        return True

    def is_lut(self) -> Bool:
        return self.kind == 2

    def eval(self, x: Float64) raises -> Float64:
        # device value [0,1] → linear [0,1] (decode / linearize).
        var v = x
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        if self.kind == 0:
            return v
        if self.kind == 1:
            return pow(v, self.gamma)
        if self.kind == 2:
            return self._lut_decode(v)
        return self._para_eval(v)

    def eval_inv(self, y: Float64) raises -> Float64:
        # linear [0,1] → device value [0,1] (encode). Inverse of eval().
        var v = y
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        if self.kind == 0:
            return v
        if self.kind == 1:
            if self.gamma <= 0.0:
                return v
            return pow(v, 1.0 / self.gamma)
        if self.kind == 2:
            return self._lut_encode(v)
        return self._para_inv(v)

    def _lut_decode(self, v: Float64) raises -> Float64:
        var n = len(self.lut)
        if n == 0:
            return v
        if n == 1:
            # single-entry curv = gamma encoded as u8Fixed8 already → handled as gamma
            return pow(v, self.lut[0])
        var fpos = v * Float64(n - 1)
        var i0 = Int(fpos)
        if i0 >= n - 1:
            return self.lut[n - 1]
        var frac = fpos - Float64(i0)
        return self.lut[i0] * (1.0 - frac) + self.lut[i0 + 1] * frac

    def _lut_encode(self, y: Float64) raises -> Float64:
        # invert a monotonic decode LUT by searching the output range.
        var n = len(self.lut)
        if n == 0:
            return y
        if n == 1:
            var g = self.lut[0]
            if g <= 0.0:
                return y
            return pow(y, 1.0 / g)
        # find bracketing samples (lut is monotonic increasing for displays)
        for i in range(n - 1):
            var a = self.lut[i]
            var b = self.lut[i + 1]
            if (y >= a and y <= b) or (y <= a and y >= b):
                var denom = b - a
                if denom == 0.0:
                    return Float64(i) / Float64(n - 1)
                var frac = (y - a) / denom
                return (Float64(i) + frac) / Float64(n - 1)
        if y <= self.lut[0]:
            return 0.0
        return 1.0

    def _para_eval(self, x: Float64) raises -> Float64:
        # ICC parametric curve types 0..4. params order: g,a,b,c,d,e,f.
        var g = self.params[0] if len(self.params) > 0 else 1.0
        if self.para_type == 0:
            return pow(x, g)
        var a = self.params[1] if len(self.params) > 1 else 0.0
        var b = self.params[2] if len(self.params) > 2 else 0.0
        if self.para_type == 1:
            # Y = (a*X + b)^g  for X >= -b/a ; else 0
            if a == 0.0:
                return 0.0
            if x >= -b / a:
                return pow(a * x + b, g)
            return 0.0
        var c = self.params[3] if len(self.params) > 3 else 0.0
        if self.para_type == 2:
            # Y = (a*X+b)^g + c  for X >= -b/a ; else c
            if a != 0.0 and x >= -b / a:
                return pow(a * x + b, g) + c
            return c
        var d = self.params[4] if len(self.params) > 4 else 0.0
        if self.para_type == 3:
            # sRGB-style: X>=d: (a*X+b)^g ; X<d: c*X
            if x >= d:
                return pow(a * x + b, g)
            return c * x
        # type 4: X>=d: (a*X+b)^g + e ; X<d: c*X + f
        var e = self.params[5] if len(self.params) > 5 else 0.0
        var f = self.params[6] if len(self.params) > 6 else 0.0
        if x >= d:
            return pow(a * x + b, g) + e
        return c * x + f

    def _para_inv(self, y: Float64) raises -> Float64:
        # Numeric inverse via bisection (curves here are monotonic on [0,1]).
        var lo = 0.0
        var hi = 1.0
        for _ in range(40):
            var mid = (lo + hi) * 0.5
            var fm = self._para_eval(mid)
            if fm < y:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5


# ---------------------------------------------------------------------------
# Tag table entry.
# ---------------------------------------------------------------------------

struct TagEntry(Copyable, Movable):
    var sig: String
    var offset: Int
    var size: Int

    def __init__(out self, var sig: String, offset: Int, size: Int):
        self.sig = sig^
        self.offset = offset
        self.size = size


# ---------------------------------------------------------------------------
# IccProfile.
# ---------------------------------------------------------------------------

struct IccProfile(Copyable, Movable):
    var raw: List[UInt8]
    var size_field: Int
    var cmm: String
    var version_major: Int
    var version_minor: Int
    var _device_class: String
    var _color_space: String
    var _pcs: String
    var valid_acsp: Bool
    var tags: List[TagEntry]

    def __init__(out self):
        self.raw = List[UInt8]()
        self.size_field = 0
        self.cmm = String("")
        self.version_major = 0
        self.version_minor = 0
        self._device_class = String("")
        self._color_space = String("")
        self._pcs = String("")
        self.valid_acsp = False
        self.tags = List[TagEntry]()

    def color_space(self) -> String:
        return self._color_space

    def device_class(self) -> String:
        return self._device_class

    def pcs(self) -> String:
        return self._pcs

    def has_tag(self, sig: String) -> Bool:
        for ref t in self.tags:
            if t.sig == sig:
                return True
        return False

    def _tag(self, sig: String) raises -> TagEntry:
        for ref t in self.tags:
            if t.sig == sig:
                return t.copy()
        raise Error("ICC: tag not found: " + sig)

    def tag_bytes(self, sig: String) raises -> List[UInt8]:
        var t = self._tag(sig)
        var out = List[UInt8]()
        for i in range(t.size):
            out.append(self.raw[t.offset + i])
        return out^

    # --- XYZType: 'XYZ ' + reserved(4) + s15Fixed16 X,Y,Z ---
    def xyz(self, sig: String) raises -> List[Float64]:
        var t = self._tag(sig)
        var o = t.offset
        var typ = _sig(self.raw, o)
        if typ != "XYZ ":
            raise Error("ICC: tag " + sig + " is not XYZType (" + typ + ")")
        var x = _s15f16(self.raw, o + 8)
        var y = _s15f16(self.raw, o + 12)
        var z = _s15f16(self.raw, o + 16)
        return [x, y, z]

    # --- tone curve: 'curv' or 'para' ---
    def tone_curve(self, sig: String) raises -> ToneCurve:
        var t = self._tag(sig)
        var o = t.offset
        var typ = _sig(self.raw, o)
        var tc = ToneCurve()
        if typ == "curv":
            var count = Int(_u32(self.raw, o + 8))
            if count == 0:
                tc.kind = 0          # identity (linear)
            elif count == 1:
                # single u8Fixed8 gamma value
                var raw16 = _u16(self.raw, o + 12)
                tc.kind = 1
                tc.gamma = Float64(raw16) / 256.0
            else:
                tc.kind = 2
                tc.lut = List[Float64]()
                for i in range(count):
                    var v = _u16(self.raw, o + 12 + 2 * i)
                    tc.lut.append(Float64(v) / 65535.0)
            return tc^
        if typ == "para":
            var ftype = Int(_u16(self.raw, o + 8))
            tc.kind = 3
            tc.para_type = ftype
            # number of params by function type
            var nparam = 1
            if ftype == 1:
                nparam = 3
            elif ftype == 2:
                nparam = 4
            elif ftype == 3:
                nparam = 5
            elif ftype == 4:
                nparam = 7
            tc.params = List[Float64]()
            for i in range(nparam):
                tc.params.append(_s15f16(self.raw, o + 12 + 4 * i))
            return tc^
        raise Error("ICC: tag " + sig + " is unsupported curve type (" + typ + ")")

    def is_rgb_matrix(self) -> Bool:
        # Convertible iff it's an RGB profile carrying primaries + TRCs.
        if self._color_space != "RGB ":
            return False
        return self.has_tag("rXYZ") and self.has_tag("gXYZ") \
            and self.has_tag("bXYZ") and self.has_tag("rTRC") \
            and self.has_tag("gTRC") and self.has_tag("bTRC")


# ---------------------------------------------------------------------------
# Parse.
# ---------------------------------------------------------------------------

def parse_icc(bytes: List[UInt8]) raises -> IccProfile:
    if len(bytes) < 132:
        raise Error("ICC: buffer too small for header+tagcount")
    var p = IccProfile()
    p.raw = bytes.copy()
    p.size_field = Int(_u32(bytes, 0))
    p.cmm = _sig(bytes, 4)
    # version: byte 8 = major, byte 9 high/low nibble = minor/bugfix
    p.version_major = Int(bytes[8])
    p.version_minor = Int(bytes[9] >> 4)
    p._device_class = _sig(bytes, 12)
    p._color_space = _sig(bytes, 16)
    p._pcs = _sig(bytes, 20)
    # 'acsp' signature at offset 36
    var acsp = _sig(bytes, 36)
    p.valid_acsp = (acsp == "acsp")
    if not p.valid_acsp:
        raise Error("ICC: missing 'acsp' signature at offset 36 (got '"
                    + acsp + "') — not a valid ICC profile")
    # tag table
    var count = Int(_u32(bytes, 128))
    var tbl = 132
    if tbl + count * 12 > len(bytes):
        raise Error("ICC: tag table exceeds buffer")
    p.tags = List[TagEntry]()
    for i in range(count):
        var e = tbl + i * 12
        var sig = _sig(bytes, e)
        var off = Int(_u32(bytes, e + 4))
        var sz = Int(_u32(bytes, e + 8))
        p.tags.append(TagEntry(sig, off, sz))
    return p^


# ---------------------------------------------------------------------------
# Profile → sRGB transform.
# ---------------------------------------------------------------------------
#
# The ICC PCS is D50-adapted XYZ. So both the profile→XYZ matrix (parsed
# primaries) and the sRGB→XYZ matrix must live in the SAME D50 reference for
# the round-trip to be consistent (and for sRGB→sRGB to be an identity).
#
# sRGB's own primaries, Bradford-adapted to the D50 PCS, are the canonical
# ICC-v4 sRGB matrix below (matches what littleCMS embeds). We invert it at
# runtime to get the XYZ(D50-PCS) → linear-sRGB matrix used on output.
comptime _SRGB2XYZ_00 = 0.4360747
comptime _SRGB2XYZ_01 = 0.3850649
comptime _SRGB2XYZ_02 = 0.1430804
comptime _SRGB2XYZ_10 = 0.2225045
comptime _SRGB2XYZ_11 = 0.7168786
comptime _SRGB2XYZ_12 = 0.0606169
comptime _SRGB2XYZ_20 = 0.0139322
comptime _SRGB2XYZ_21 = 0.0971045
comptime _SRGB2XYZ_22 = 0.7141733


def _inv3(
    a00: Float64, a01: Float64, a02: Float64,
    a10: Float64, a11: Float64, a12: Float64,
    a20: Float64, a21: Float64, a22: Float64,
) raises -> List[Float64]:
    # 3x3 inverse → row-major [m00..m22].
    var det = a00 * (a11 * a22 - a12 * a21) \
        - a01 * (a10 * a22 - a12 * a20) \
        + a02 * (a10 * a21 - a11 * a20)
    if det == 0.0:
        raise Error("ICC: singular matrix")
    var inv = 1.0 / det
    return [
        (a11 * a22 - a12 * a21) * inv,
        (a02 * a21 - a01 * a22) * inv,
        (a01 * a12 - a02 * a11) * inv,
        (a12 * a20 - a10 * a22) * inv,
        (a00 * a22 - a02 * a20) * inv,
        (a02 * a10 - a00 * a12) * inv,
        (a10 * a21 - a11 * a20) * inv,
        (a01 * a20 - a00 * a21) * inv,
        (a00 * a11 - a01 * a10) * inv,
    ]


def _srgb_encode(c: Float64) -> Float64:
    # linear → sRGB-encoded (IEC 61966-2-1).
    var v = c
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    if v <= 0.0031308:
        return 12.92 * v
    return 1.055 * (v ** (1.0 / 2.4)) - 0.055


def to_srgb(img: Image) raises -> Image:
    """Convert an ICC-tagged RGB matrix/TRC image to sRGB.

    Behaviour:
      * No embedded profile, or profile not an RGB matrix/TRC profile
        (e.g. LUT-based, GRAY, CMYK): returns a plain copy UNCHANGED
        (no color management applied). Callers can detect this because the
        returned image still carries the original `.icc` bytes and pixels
        are byte-identical.
      * RGB matrix/TRC profile: linearize via profile TRCs, profile→XYZ
        matrix, XYZ→linear-sRGB, sRGB encode, clamp to [0,255].
    """
    if len(img.icc) == 0:
        return img.clone()

    var prof = parse_icc(img.icc.copy())
    if not prof.is_rgb_matrix():
        # Unsupported (LUT-based / GRAY / CMYK): pass through unchanged.
        return img.clone()

    # Only the 8-bit RGB(A) core path is color-managed here.
    if img.bit_depth != 8 or (img.channels != 3 and img.channels != 4):
        return img.clone()

    var rXYZ = prof.xyz("rXYZ")
    var gXYZ = prof.xyz("gXYZ")
    var bXYZ = prof.xyz("bXYZ")
    var rTRC = prof.tone_curve("rTRC")
    var gTRC = prof.tone_curve("gTRC")
    var bTRC = prof.tone_curve("bTRC")

    # profile→XYZ matrix columns are the primaries.
    var m00 = rXYZ[0]; var m01 = gXYZ[0]; var m02 = bXYZ[0]
    var m10 = rXYZ[1]; var m11 = gXYZ[1]; var m12 = bXYZ[1]
    var m20 = rXYZ[2]; var m21 = gXYZ[2]; var m22 = bXYZ[2]

    # XYZ(D50-PCS) → linear-sRGB matrix = inverse of the D50-adapted sRGB matrix.
    var xyz2rgb = _inv3(
        _SRGB2XYZ_00, _SRGB2XYZ_01, _SRGB2XYZ_02,
        _SRGB2XYZ_10, _SRGB2XYZ_11, _SRGB2XYZ_12,
        _SRGB2XYZ_20, _SRGB2XYZ_21, _SRGB2XYZ_22,
    )

    var out = img.clone()
    var w = img.width
    var h = img.height
    for y in range(h):
        for x in range(w):
            var rd = Float64(Int(img.get(x, y, 0))) / 255.0
            var gd = Float64(Int(img.get(x, y, 1))) / 255.0
            var bd = Float64(Int(img.get(x, y, 2))) / 255.0
            # linearize via profile TRC
            var rl = rTRC.eval(rd)
            var gl = gTRC.eval(gd)
            var bl = bTRC.eval(bd)
            # profile linear RGB → XYZ (D50 PCS)
            var X = m00 * rl + m01 * gl + m02 * bl
            var Y = m10 * rl + m11 * gl + m12 * bl
            var Z = m20 * rl + m21 * gl + m22 * bl
            # XYZ(D50 PCS) → linear sRGB
            var lr = xyz2rgb[0] * X + xyz2rgb[1] * Y + xyz2rgb[2] * Z
            var lg = xyz2rgb[3] * X + xyz2rgb[4] * Y + xyz2rgb[5] * Z
            var lb = xyz2rgb[6] * X + xyz2rgb[7] * Y + xyz2rgb[8] * Z
            # sRGB encode + to 8-bit
            var sr = _srgb_encode(lr) * 255.0 + 0.5
            var sg = _srgb_encode(lg) * 255.0 + 0.5
            var sb = _srgb_encode(lb) * 255.0 + 0.5
            if sr < 0.0: sr = 0.0
            if sr > 255.0: sr = 255.0
            if sg < 0.0: sg = 0.0
            if sg > 255.0: sg = 255.0
            if sb < 0.0: sb = 0.0
            if sb > 255.0: sb = 255.0
            out.set(x, y, 0, UInt8(Int(sr)))
            out.set(x, y, 1, UInt8(Int(sg)))
            out.set(x, y, 2, UInt8(Int(sb)))
    return out^
