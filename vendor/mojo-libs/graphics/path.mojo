# graphics.path — vector paths: move/line/quad/cubic/close, fill + stroke.
#
# A Path is one or more subpaths of flattened points (Bézier curves are flattened
# to line segments when added). fill_path scanline-fills with the even-odd rule
# across ALL subpaths together, so an inner subpath cuts a hole. stroke_path draws
# each segment as a width-thick quad with round joins/caps. Transform a path with
# an Affine (translate/scale/rotate) before drawing.
#
# Draw at Nx and Canvas.downsampled() for smooth (anti-aliased) fills/strokes.

from std.math import sqrt
from graphics.canvas import Canvas
from graphics.color import Color
from graphics.draw import fill_polygon, fill_circle, fill_triangle, fill_polygon_aa
from graphics.transform import Affine

comptime CURVE_STEPS = 32  # Bézier flattening resolution

# stroke cap styles (open ends) and join styles (interior vertices)
comptime CAP_BUTT = 0
comptime CAP_ROUND = 1
comptime CAP_SQUARE = 2
comptime JOIN_ROUND = 0
comptime JOIN_BEVEL = 1
comptime JOIN_MITER = 2


def _floor_i(x: Float64) -> Int:
    var i = Int(x)
    return i if Float64(i) <= x else i - 1


def _ceil_i(x: Float64) -> Int:
    var i = Int(x)
    return i if Float64(i) >= x else i + 1


struct Path(Copyable, Movable):
    var xs: List[Float64]
    var ys: List[Float64]
    var starts: List[Int]   # index into xs where each subpath begins
    var cx: Float64
    var cy: Float64

    def __init__(out self):
        self.xs = List[Float64]()
        self.ys = List[Float64]()
        self.starts = List[Int]()
        self.cx = 0.0
        self.cy = 0.0

    def move_to(mut self, x: Float64, y: Float64):
        self.starts.append(len(self.xs))
        self.xs.append(x)
        self.ys.append(y)
        self.cx = x
        self.cy = y

    def line_to(mut self, x: Float64, y: Float64):
        self.xs.append(x)
        self.ys.append(y)
        self.cx = x
        self.cy = y

    def quad_to(mut self, qx: Float64, qy: Float64, x: Float64, y: Float64):
        var x0 = self.cx
        var y0 = self.cy
        for i in range(1, CURVE_STEPS + 1):
            var t = Float64(i) / Float64(CURVE_STEPS)
            var u = 1.0 - t
            self.xs.append(u * u * x0 + 2.0 * u * t * qx + t * t * x)
            self.ys.append(u * u * y0 + 2.0 * u * t * qy + t * t * y)
        self.cx = x
        self.cy = y

    def cubic_to(mut self, c1x: Float64, c1y: Float64, c2x: Float64, c2y: Float64, x: Float64, y: Float64):
        var x0 = self.cx
        var y0 = self.cy
        for i in range(1, CURVE_STEPS + 1):
            var t = Float64(i) / Float64(CURVE_STEPS)
            var u = 1.0 - t
            var uu = u * u
            var tt = t * t
            self.xs.append(uu * u * x0 + 3.0 * uu * t * c1x + 3.0 * u * tt * c2x + tt * t * x)
            self.ys.append(uu * u * y0 + 3.0 * uu * t * c1y + 3.0 * u * tt * c2y + tt * t * y)
        self.cx = x
        self.cy = y

    def close(mut self):
        # fill closes each subpath implicitly; close() is a no-op marker kept for
        # API familiarity (stroke also treats subpaths as closed loops via joins).
        pass

    def transformed(self, m: Affine) -> Path:
        var p = Path()
        for i in range(len(self.starts)):
            p.starts.append(self.starts[i])
        for i in range(len(self.xs)):
            p.xs.append(m.tx(self.xs[i], self.ys[i]))
            p.ys.append(m.ty(self.xs[i], self.ys[i]))
        return p^


def fill_path(mut c: Canvas, p: Path, col: Color) raises:
    """Even-odd scanline fill across all subpaths (inner subpaths cut holes)."""
    var n = len(p.xs)
    if n < 3:
        return
    var ns = len(p.starts)
    var fminy = p.ys[0]
    var fmaxy = p.ys[0]
    for i in range(1, n):
        if p.ys[i] < fminy:
            fminy = p.ys[i]
        if p.ys[i] > fmaxy:
            fmaxy = p.ys[i]
    var ymin = _floor_i(fminy)
    var ymax = _ceil_i(fmaxy)
    for y in range(ymin, ymax + 1):
        var yc = Float64(y) + 0.5
        var xints = List[Float64]()
        for s in range(ns):
            var a0 = p.starts[s]
            var a1 = p.starts[s + 1] if (s + 1 < ns) else n
            if a1 - a0 < 2:
                continue
            var j = a1 - 1  # closing edge: last -> first
            for i in range(a0, a1):
                var yi = p.ys[i]
                var yj = p.ys[j]
                if (yi <= yc and yj > yc) or (yj <= yc and yi > yc):
                    xints.append(p.xs[i] + ((yc - yi) / (yj - yi)) * (p.xs[j] - p.xs[i]))
                j = i
        # insertion sort
        for a in range(1, len(xints)):
            var key = xints[a]
            var b = a - 1
            while b >= 0 and xints[b] > key:
                xints[b + 1] = xints[b]
                b -= 1
            xints[b + 1] = key
        var k = 0
        while k + 1 < len(xints):
            var xa = Int(xints[k] + 0.5)
            var xb = Int(xints[k + 1] + 0.5)
            for x in range(xa, xb):
                c.set_pixel(x, y, col)
            k += 2


def fill_path_aa(mut c: Canvas, p: Path, col: Color, samples: Int = 4) raises:
    """Analytic anti-aliased path fill (even-odd across subpaths). Sub-scanline
    sampling + fractional horizontal coverage -> smooth edges at 1x. Float points
    (curves already flattened), so Bézier shapes antialias too."""
    var n = len(p.xs)
    if n < 3:
        return
    var W = c.w
    if W <= 0:
        return
    var ns = len(p.starts)
    var fminy = p.ys[0]
    var fmaxy = p.ys[0]
    for i in range(1, n):
        if p.ys[i] < fminy:
            fminy = p.ys[i]
        if p.ys[i] > fmaxy:
            fmaxy = p.ys[i]
    var ymin = _floor_i(fminy)
    var ymax = _ceil_i(fmaxy)
    if ymin < 0:
        ymin = 0
    if ymax > c.h - 1:
        ymax = c.h - 1
    var ss = samples if samples >= 1 else 1
    var inc = 1.0 / Float64(ss)
    var cov = List[Float64]()
    for _ in range(W):
        cov.append(0.0)
    for y in range(ymin, ymax + 1):
        for x in range(W):
            cov[x] = 0.0
        for s in range(ss):
            var yc = Float64(y) + (Float64(s) + 0.5) * inc
            var xints = List[Float64]()
            for sp in range(ns):
                var a0 = p.starts[sp]
                var a1 = p.starts[sp + 1] if (sp + 1 < ns) else n
                if a1 - a0 < 2:
                    continue
                var j = a1 - 1
                for i in range(a0, a1):
                    var yi = p.ys[i]
                    var yj = p.ys[j]
                    if (yi <= yc and yj > yc) or (yj <= yc and yi > yc):
                        xints.append(p.xs[i] + ((yc - yi) / (yj - yi)) * (p.xs[j] - p.xs[i]))
                    j = i
            for a in range(1, len(xints)):
                var key = xints[a]
                var b = a - 1
                while b >= 0 and xints[b] > key:
                    xints[b + 1] = xints[b]
                    b -= 1
                xints[b + 1] = key
            var k = 0
            while k + 1 < len(xints):
                var xa = xints[k]
                var xb = xints[k + 1]
                k += 2
                if xa < 0.0:
                    xa = 0.0
                if xb > Float64(W):
                    xb = Float64(W)
                if xb <= xa:
                    continue
                var ix0 = Int(xa)
                var ix1 = Int(xb)
                if ix0 == ix1:
                    cov[ix0] += (xb - xa) * inc
                else:
                    cov[ix0] += (Float64(ix0 + 1) - xa) * inc
                    for x in range(ix0 + 1, ix1):
                        cov[x] += inc
                    if ix1 < W:
                        cov[ix1] += (xb - Float64(ix1)) * inc
        for x in range(W):
            var cv = cov[x]
            if cv > 0.0:
                if cv > 1.0:
                    cv = 1.0
                var alpha = Int(Float64(Int(col.a)) * cv + 0.5)
                c.set_pixel(x, y, Color(col.r, col.g, col.b, UInt8(alpha)))


def _thick_segment(mut c: Canvas, x0: Float64, y0: Float64, x1: Float64, y1: Float64, w: Float64, col: Color) raises:
    var dx = x1 - x0
    var dy = y1 - y0
    var ln = sqrt(dx * dx + dy * dy)
    if ln < 0.0001:
        return
    var nx = -dy / ln * (w * 0.5)
    var ny = dx / ln * (w * 0.5)
    var xs = [Int(x0 + nx), Int(x1 + nx), Int(x1 - nx), Int(x0 - nx)]
    var ys = [Int(y0 + ny), Int(y1 + ny), Int(y1 - ny), Int(y0 - ny)]
    fill_polygon(c, xs, ys, col)


def _cap(mut c: Canvas, px: Float64, py: Float64, qx: Float64, qy: Float64, r: Float64, col: Color, cap: Int) raises:
    """Cap the open end at (px,py); (qx,qy) is its neighbor (stroke comes from q)."""
    if cap == CAP_ROUND:
        var ri = Int(r)
        if ri > 0:
            fill_circle(c, Int(px), Int(py), ri, col)
    elif cap == CAP_SQUARE:
        var dx = px - qx
        var dy = py - qy
        var ln = sqrt(dx * dx + dy * dy)
        if ln < 0.0001:
            return
        var ox = dx / ln
        var oy = dy / ln
        _thick_segment(c, px, py, px + ox * r, py + oy * r, r * 2.0, col)
    # CAP_BUTT: nothing


def _join(mut c: Canvas, ux: Float64, uy: Float64, vx: Float64, vy: Float64, wx: Float64, wy: Float64, r: Float64, col: Color, join: Int, miter_limit: Float64) raises:
    """Join the corner at V between segment U->V and V->W."""
    if join == JOIN_ROUND:
        var ri = Int(r)
        if ri > 0:
            fill_circle(c, Int(vx), Int(vy), ri, col)
        return
    var din_x = vx - ux
    var din_y = vy - uy
    var dout_x = wx - vx
    var dout_y = wy - vy
    var lin = sqrt(din_x * din_x + din_y * din_y)
    var lout = sqrt(dout_x * dout_x + dout_y * dout_y)
    if lin < 0.0001 or lout < 0.0001:
        return
    din_x /= lin; din_y /= lin
    dout_x /= lout; dout_y /= lout
    var pin_x = -din_y * r   # perpendicular offsets (radius)
    var pin_y = din_x * r
    var pout_x = -dout_y * r
    var pout_y = dout_x * r
    # bevel base (both sides; the inner one is overdraw inside the quads)
    fill_triangle(c, Int(vx), Int(vy), Int(vx + pin_x), Int(vy + pin_y), Int(vx + pout_x), Int(vy + pout_y), col)
    fill_triangle(c, Int(vx), Int(vy), Int(vx - pin_x), Int(vy - pin_y), Int(vx - pout_x), Int(vy - pout_y), col)
    if join == JOIN_MITER:
        var dot = din_x * dout_x + din_y * dout_y
        if dot > -0.999:
            var denom = 1.0 + dot
            var mx = (pin_x + pout_x) / denom
            var my = (pin_y + pout_y) / denom
            var ratio = sqrt(mx * mx + my * my) / r
            if ratio <= miter_limit:
                # miter tip wedges on both sides
                fill_triangle(c, Int(vx + pin_x), Int(vy + pin_y), Int(vx + mx), Int(vy + my), Int(vx + pout_x), Int(vy + pout_y), col)
                fill_triangle(c, Int(vx - pin_x), Int(vy - pin_y), Int(vx - mx), Int(vy - my), Int(vx - pout_x), Int(vy - pout_y), col)


def stroke_path(mut c: Canvas, p: Path, col: Color, width: Float64, closed: Bool = False, cap: Int = CAP_ROUND, join: Int = JOIN_ROUND, miter_limit: Float64 = 4.0) raises:
    """Stroke each subpath at `width` with selectable line caps (CAP_BUTT/ROUND/
    SQUARE at open ends) and joins (JOIN_ROUND/BEVEL/MITER at interior vertices).
    `closed=True` strokes the closing edge and joins every vertex (no caps)."""
    var ns = len(p.starts)
    var n = len(p.xs)
    var r = width * 0.5
    for s in range(ns):
        var a0 = p.starts[s]
        var a1 = p.starts[s + 1] if (s + 1 < ns) else n
        var cnt = a1 - a0
        if cnt < 2:
            if cnt == 1 and cap == CAP_ROUND and Int(r) > 0:
                fill_circle(c, Int(p.xs[a0]), Int(p.ys[a0]), Int(r), col)
            continue
        var last = a1 - 1
        for i in range(a0, last):
            _thick_segment(c, p.xs[i], p.ys[i], p.xs[i + 1], p.ys[i + 1], width, col)
        if closed:
            _thick_segment(c, p.xs[last], p.ys[last], p.xs[a0], p.ys[a0], width, col)
            for v in range(a0, a1):
                var pu = v - 1 if v > a0 else last
                var pw = v + 1 if v < last else a0
                _join(c, p.xs[pu], p.ys[pu], p.xs[v], p.ys[v], p.xs[pw], p.ys[pw], r, col, join, miter_limit)
        else:
            for v in range(a0 + 1, last):
                _join(c, p.xs[v - 1], p.ys[v - 1], p.xs[v], p.ys[v], p.xs[v + 1], p.ys[v + 1], r, col, join, miter_limit)
            _cap(c, p.xs[a0], p.ys[a0], p.xs[a0 + 1], p.ys[a0 + 1], r, col, cap)
            _cap(c, p.xs[last], p.ys[last], p.xs[last - 1], p.ys[last - 1], r, col, cap)


def stroke_path_dashed(mut c: Canvas, p: Path, col: Color, width: Float64, dash: Float64, gap: Float64, closed: Bool = False) raises:
    """Dashed stroke: `dash` on / `gap` off (butt-capped), phase carried across
    segments so the pattern is continuous along the path."""
    var period = dash + gap
    if period <= 0.0:
        return
    var ns = len(p.starts)
    var n = len(p.xs)
    for s in range(ns):
        var a0 = p.starts[s]
        var a1 = p.starts[s + 1] if (s + 1 < ns) else n
        var npts = a1 - a0
        if npts < 2:
            continue
        var nedges = npts - 1 + (1 if closed else 0)
        var phase = 0.0
        for e in range(nedges):
            var i0 = a0 + e
            var i1 = a0 + (e + 1) if (e + 1 < npts) else a0
            var x0 = p.xs[i0]
            var y0 = p.ys[i0]
            var x1 = p.xs[i1]
            var y1 = p.ys[i1]
            var dx = x1 - x0
            var dy = y1 - y0
            var seglen = sqrt(dx * dx + dy * dy)
            if seglen < 0.000001:
                continue
            var ux = dx / seglen
            var uy = dy / seglen
            var pos = 0.0
            while pos < seglen:
                var on = phase < dash
                var rem = (dash - phase) if on else (period - phase)
                var step = rem
                if pos + step > seglen:
                    step = seglen - pos
                if on and step > 0.0:
                    _thick_segment(c, x0 + ux * pos, y0 + uy * pos, x0 + ux * (pos + step), y0 + uy * (pos + step), width, col)
                pos += step
                phase += step
                if phase >= period:
                    phase -= period


def stroke_outline_aa(mut c: Canvas, p: Path, col: Color, width: Float64, miter_limit: Float64 = 4.0) raises:
    """Anti-aliased stroke with NO join seams: builds a single offset-outline
    polygon per (open) subpath and AA-fills it once (so overlapping segment edges
    can't double-blend). Miter-style joins (clamped to miter_limit -> bevel-ish),
    butt caps. Best for gentle paths; very sharp inner turns may pinch."""
    var ns = len(p.starts)
    var n = len(p.xs)
    var r = width * 0.5
    for s in range(ns):
        var a0 = p.starts[s]
        var a1 = p.starts[s + 1] if (s + 1 < ns) else n
        if a1 - a0 < 2:
            continue
        var lx = List[Int](); var ly = List[Int]()
        var rx = List[Int](); var ry = List[Int]()
        for i in range(a0, a1):
            var dix = 0.0; var diy = 0.0; var dox = 0.0; var doy = 0.0
            if i > a0:
                dix = p.xs[i] - p.xs[i - 1]; diy = p.ys[i] - p.ys[i - 1]
                var li = sqrt(dix * dix + diy * diy)
                if li > 0.000001:
                    dix /= li; diy /= li
            if i < a1 - 1:
                dox = p.xs[i + 1] - p.xs[i]; doy = p.ys[i + 1] - p.ys[i]
                var lo = sqrt(dox * dox + doy * doy)
                if lo > 0.000001:
                    dox /= lo; doy /= lo
            var nx = 0.0; var ny = 0.0
            if i == a0:
                nx = -doy; ny = dox          # start cap normal (out-segment)
            elif i == a1 - 1:
                nx = -diy; ny = dix          # end cap normal (in-segment)
            else:
                var pinx = -diy; var piny = dix
                var ponx = -doy; var pony = dox
                var dot = dix * dox + diy * doy
                if dot <= -0.99:
                    nx = pinx; ny = piny     # near U-turn: use in-perp
                else:
                    var denom = 1.0 + dot
                    nx = (pinx + ponx) / denom; ny = (piny + pony) / denom
                    var ml = sqrt(nx * nx + ny * ny)
                    if ml > miter_limit and ml > 0.000001:
                        nx = nx / ml * miter_limit; ny = ny / ml * miter_limit
            lx.append(Int(p.xs[i] + nx * r)); ly.append(Int(p.ys[i] + ny * r))
            rx.append(Int(p.xs[i] - nx * r)); ry.append(Int(p.ys[i] - ny * r))
        var xs = List[Int](); var ys = List[Int]()
        for i in range(len(lx)):
            xs.append(lx[i]); ys.append(ly[i])
        for i in range(len(rx) - 1, -1, -1):
            xs.append(rx[i]); ys.append(ry[i])
        fill_polygon_aa(c, xs, ys, col)
