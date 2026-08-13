# svg/pathdata.mojo — parse the SVG path `d` mini-language into absolute segments.
#
# Output is a flat List[SvgSeg] in ABSOLUTE coordinates, using only M/L/C/Q/Z
# (smooth S/T are resolved to C/Q via control-point reflection; elliptical arcs
# A/a are converted to a sequence of cubic C segments). This intermediate form
# is exactly testable; `build_path` then emits it into a graphics.Path for
# rasterization by the existing fill/stroke engine.
#
# Grammar handled: M m L l H h V v C c S s Q q T t A a Z z, implicit command
# repetition (and the M->L / m->l implicit-lineto rule), comma/whitespace number
# separators, signs/decimals/exponents, and flag digits for arcs.

from std.math import sin, cos, sqrt, acos, tan, pi
from graphics.path import Path

comptime SEG_M: Int = 0
comptime SEG_L: Int = 1
comptime SEG_C: Int = 2
comptime SEG_Q: Int = 3
comptime SEG_Z: Int = 4


struct SvgSeg(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var x: Float64
    var y: Float64
    var x1: Float64
    var y1: Float64
    var x2: Float64
    var y2: Float64

    def __init__(
        out self, kind: Int,
        x: Float64 = 0.0, y: Float64 = 0.0,
        x1: Float64 = 0.0, y1: Float64 = 0.0,
        x2: Float64 = 0.0, y2: Float64 = 0.0,
    ):
        self.kind = kind
        self.x = x
        self.y = y
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


# ---- byte helpers ----
def _is_digit(c: Int) -> Bool:
    return c >= 48 and c <= 57


def _is_alpha(c: Int) -> Bool:
    return (c >= 65 and c <= 90) or (c >= 97 and c <= 122)


def _is_sep(c: Int) -> Bool:
    return c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D or c == 0x2C


def _skip_seps(b: List[UInt8], n: Int, mut pos: Int):
    while pos < n and _is_sep(Int(b[pos])):
        pos += 1


def _read_num(b: List[UInt8], n: Int, mut pos: Int) raises -> Float64:
    _skip_seps(b, n, pos)
    var start = pos
    if pos < n and (Int(b[pos]) == 43 or Int(b[pos]) == 45):  # + -
        pos += 1
    while pos < n and _is_digit(Int(b[pos])):
        pos += 1
    if pos < n and Int(b[pos]) == 46:  # .
        pos += 1
        while pos < n and _is_digit(Int(b[pos])):
            pos += 1
    if pos < n and (Int(b[pos]) == 101 or Int(b[pos]) == 69):  # e E
        pos += 1
        if pos < n and (Int(b[pos]) == 43 or Int(b[pos]) == 45):
            pos += 1
        while pos < n and _is_digit(Int(b[pos])):
            pos += 1
    if pos == start:
        raise Error("svg path: expected number at index " + String(pos))
    return atof(String(StringSlice(unsafe_from_utf8=Span(unsafe_ptr=b.unsafe_ptr() + start, length=pos - start))))


def _read_flag(b: List[UInt8], n: Int, mut pos: Int) raises -> Int:
    # Arc flags are a single '0' or '1' (may be packed against the next number).
    _skip_seps(b, n, pos)
    if pos < n and (Int(b[pos]) == 48 or Int(b[pos]) == 49):
        var f = Int(b[pos]) - 48
        pos += 1
        return f
    raise Error("svg path: expected 0/1 flag at index " + String(pos))


# ---- elliptical arc (endpoint param) -> cubic segments ----
def _arc_to_cubics(
    mut segs: List[SvgSeg],
    x1: Float64, y1: Float64,
    rx_in: Float64, ry_in: Float64, phi_deg: Float64,
    fa: Int, fs: Int,
    x2: Float64, y2: Float64,
) raises:
    var rx = rx_in if rx_in >= 0.0 else -rx_in
    var ry = ry_in if ry_in >= 0.0 else -ry_in
    if rx == 0.0 or ry == 0.0 or (x1 == x2 and y1 == y2):
        segs.append(SvgSeg(SEG_L, x=x2, y=y2))   # degenerate -> line
        return

    var phi = phi_deg * pi / 180.0
    var cphi = cos(phi)
    var sphi = sin(phi)
    var dx2 = (x1 - x2) / 2.0
    var dy2 = (y1 - y2) / 2.0
    var x1p = cphi * dx2 + sphi * dy2
    var y1p = -sphi * dx2 + cphi * dy2

    # correct out-of-range radii
    var lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        var s = sqrt(lam)
        rx *= s
        ry *= s

    var rx2 = rx * rx
    var ry2 = ry * ry
    var num = rx2 * ry2 - rx2 * (y1p * y1p) - ry2 * (x1p * x1p)
    var den = rx2 * (y1p * y1p) + ry2 * (x1p * x1p)
    var rad = num / den
    if rad < 0.0:
        rad = 0.0
    var co = sqrt(rad)
    if fa == fs:
        co = -co
    var cxp = co * (rx * y1p / ry)
    var cyp = co * (-ry * x1p / rx)
    var cx = cphi * cxp - sphi * cyp + (x1 + x2) / 2.0
    var cy = sphi * cxp + cphi * cyp + (y1 + y2) / 2.0

    var ux = (x1p - cxp) / rx
    var uy = (y1p - cyp) / ry
    var vx = (-x1p - cxp) / rx
    var vy = (-y1p - cyp) / ry

    var theta1 = _angle(1.0, 0.0, ux, uy)
    var dtheta = _angle(ux, uy, vx, vy)
    if fs == 0 and dtheta > 0.0:
        dtheta -= 2.0 * pi
    if fs == 1 and dtheta < 0.0:
        dtheta += 2.0 * pi

    var n_segs = Int(_ceil_abs(dtheta / (pi / 2.0)))
    if n_segs == 0:
        n_segs = 1
    var delta = dtheta / Float64(n_segs)
    var t = (4.0 / 3.0) * tan(delta / 4.0)

    var th = theta1
    for _ in range(n_segs):
        var cos1 = cos(th)
        var sin1 = sin(th)
        var th2 = th + delta
        var cos2 = cos(th2)
        var sin2 = sin(th2)

        # endpoint of this sub-arc
        var ex = cx + rx * cos2 * cphi - ry * sin2 * sphi
        var ey = cy + rx * cos2 * sphi + ry * sin2 * cphi
        # start point (on-ellipse) for the tangents
        var spx = cx + rx * cos1 * cphi - ry * sin1 * sphi
        var spy = cy + rx * cos1 * sphi + ry * sin1 * cphi
        # tangents (derivative of ellipse param)
        var d1x = -rx * sin1 * cphi - ry * cos1 * sphi
        var d1y = -rx * sin1 * sphi + ry * cos1 * cphi
        var d2x = -rx * sin2 * cphi - ry * cos2 * sphi
        var d2y = -rx * sin2 * sphi + ry * cos2 * cphi

        var c1x = spx + t * d1x
        var c1y = spy + t * d1y
        var c2x = ex - t * d2x
        var c2y = ey - t * d2y
        segs.append(SvgSeg(SEG_C, x=ex, y=ey, x1=c1x, y1=c1y, x2=c2x, y2=c2y))
        th = th2


def _angle(ux: Float64, uy: Float64, vx: Float64, vy: Float64) raises -> Float64:
    var dot = ux * vx + uy * vy
    var lu = sqrt(ux * ux + uy * uy)
    var lv = sqrt(vx * vx + vy * vy)
    var c = dot / (lu * lv)
    if c < -1.0:
        c = -1.0
    if c > 1.0:
        c = 1.0
    var a = acos(c)
    if (ux * vy - uy * vx) < 0.0:
        a = -a
    return a


def _ceil_abs(x: Float64) -> Float64:
    var v = x if x >= 0.0 else -x
    var i = Float64(Int(v))
    return i if i == v else i + 1.0


# ---- main parser ----
def parse_path_d(d: String) raises -> List[SvgSeg]:
    var src = d.as_bytes()
    var b = List[UInt8]()
    for i in range(len(src)):
        b.append(src[i])
    var n = len(b)

    var segs = List[SvgSeg]()
    var pos = 0
    var cmd = 0          # current command byte (0 = none yet)
    var cx = 0.0
    var cy = 0.0         # current point
    var sx = 0.0
    var sy = 0.0         # subpath start
    var pcx = 0.0
    var pcy = 0.0        # previous cubic 2nd control (for S)
    var pqx = 0.0
    var pqy = 0.0        # previous quad control (for T)
    var prev_was_cubic = False
    var prev_was_quad = False

    while True:
        _skip_seps(b, n, pos)
        if pos >= n:
            break
        var c = Int(b[pos])
        if _is_alpha(c):
            cmd = c
            pos += 1
        else:
            if cmd == 0:
                raise Error("svg path: number before any command")
            if cmd == 77:      # 'M' implicit -> 'L'
                cmd = 76
            elif cmd == 109:   # 'm' implicit -> 'l'
                cmd = 108

        var rel = cmd >= 97   # lowercase => relative
        var this_cubic = False
        var this_quad = False

        if cmd == 77 or cmd == 109:                       # M / m
            var x = _read_num(b, n, pos)
            var y = _read_num(b, n, pos)
            if rel:
                x += cx
                y += cy
            cx = x
            cy = y
            sx = x
            sy = y
            segs.append(SvgSeg(SEG_M, x=x, y=y))
        elif cmd == 76 or cmd == 108:                     # L / l
            var x = _read_num(b, n, pos)
            var y = _read_num(b, n, pos)
            if rel:
                x += cx
                y += cy
            cx = x
            cy = y
            segs.append(SvgSeg(SEG_L, x=x, y=y))
        elif cmd == 72 or cmd == 104:                     # H / h
            var x = _read_num(b, n, pos)
            if rel:
                x += cx
            cx = x
            segs.append(SvgSeg(SEG_L, x=x, y=cy))
        elif cmd == 86 or cmd == 118:                     # V / v
            var y = _read_num(b, n, pos)
            if rel:
                y += cy
            cy = y
            segs.append(SvgSeg(SEG_L, x=cx, y=y))
        elif cmd == 67 or cmd == 99:                      # C / c
            var x1 = _read_num(b, n, pos)
            var y1 = _read_num(b, n, pos)
            var x2 = _read_num(b, n, pos)
            var y2 = _read_num(b, n, pos)
            var x = _read_num(b, n, pos)
            var y = _read_num(b, n, pos)
            if rel:
                x1 += cx; y1 += cy; x2 += cx; y2 += cy; x += cx; y += cy
            segs.append(SvgSeg(SEG_C, x=x, y=y, x1=x1, y1=y1, x2=x2, y2=y2))
            pcx = x2; pcy = y2; cx = x; cy = y
            this_cubic = True
        elif cmd == 83 or cmd == 115:                     # S / s (smooth cubic)
            var x2 = _read_num(b, n, pos)
            var y2 = _read_num(b, n, pos)
            var x = _read_num(b, n, pos)
            var y = _read_num(b, n, pos)
            if rel:
                x2 += cx; y2 += cy; x += cx; y += cy
            var x1 = cx
            var y1 = cy
            if prev_was_cubic:
                x1 = 2.0 * cx - pcx
                y1 = 2.0 * cy - pcy
            segs.append(SvgSeg(SEG_C, x=x, y=y, x1=x1, y1=y1, x2=x2, y2=y2))
            pcx = x2; pcy = y2; cx = x; cy = y
            this_cubic = True
        elif cmd == 81 or cmd == 113:                     # Q / q
            var qx = _read_num(b, n, pos)
            var qy = _read_num(b, n, pos)
            var x = _read_num(b, n, pos)
            var y = _read_num(b, n, pos)
            if rel:
                qx += cx; qy += cy; x += cx; y += cy
            segs.append(SvgSeg(SEG_Q, x=x, y=y, x1=qx, y1=qy))
            pqx = qx; pqy = qy; cx = x; cy = y
            this_quad = True
        elif cmd == 84 or cmd == 116:                     # T / t (smooth quad)
            var x = _read_num(b, n, pos)
            var y = _read_num(b, n, pos)
            if rel:
                x += cx; y += cy
            var qx = cx
            var qy = cy
            if prev_was_quad:
                qx = 2.0 * cx - pqx
                qy = 2.0 * cy - pqy
            segs.append(SvgSeg(SEG_Q, x=x, y=y, x1=qx, y1=qy))
            pqx = qx; pqy = qy; cx = x; cy = y
            this_quad = True
        elif cmd == 65 or cmd == 97:                      # A / a
            var rx = _read_num(b, n, pos)
            var ry = _read_num(b, n, pos)
            var rot = _read_num(b, n, pos)
            var fa = _read_flag(b, n, pos)
            var fs = _read_flag(b, n, pos)
            var x = _read_num(b, n, pos)
            var y = _read_num(b, n, pos)
            if rel:
                x += cx; y += cy
            _arc_to_cubics(segs, cx, cy, rx, ry, rot, fa, fs, x, y)
            cx = x; cy = y
        elif cmd == 90 or cmd == 122:                     # Z / z
            segs.append(SvgSeg(SEG_Z))
            cx = sx; cy = sy
        else:
            raise Error("svg path: unknown command byte " + String(cmd))

        prev_was_cubic = this_cubic
        prev_was_quad = this_quad

    return segs^


def build_path(segs: List[SvgSeg]) raises -> Path:
    """Emit parsed absolute segments into a graphics.Path (flattens curves)."""
    var p = Path()
    for i in range(len(segs)):
        var s = segs[i]
        if s.kind == SEG_M:
            p.move_to(s.x, s.y)
        elif s.kind == SEG_L:
            p.line_to(s.x, s.y)
        elif s.kind == SEG_C:
            p.cubic_to(s.x1, s.y1, s.x2, s.y2, s.x, s.y)
        elif s.kind == SEG_Q:
            p.quad_to(s.x1, s.y1, s.x, s.y)
        elif s.kind == SEG_Z:
            p.close()
    return p^
