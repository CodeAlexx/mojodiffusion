# graphics.draw — 2D primitives on a Canvas (free functions, all via set_pixel).
#
# Integer rasterizers, no floats in the hot loops: Bresenham lines, midpoint
# circles, scanline circle fill (integer sqrt). Everything clips for free because
# Canvas.set_pixel ignores out-of-bounds writes.

from graphics.canvas import Canvas
from graphics.color import Color


def _iabs(x: Int) -> Int:
    return -x if x < 0 else x


def _isqrt(n: Int) -> Int:
    """Floor of the square root of n (Newton), for the circle scanline fill."""
    if n <= 0:
        return 0
    var x = n
    var y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def hline(mut c: Canvas, x0: Int, x1: Int, y: Int, col: Color):
    var a = x0 if x0 <= x1 else x1
    var b = x1 if x0 <= x1 else x0
    for x in range(a, b + 1):
        c.set_pixel(x, y, col)


def vline(mut c: Canvas, x: Int, y0: Int, y1: Int, col: Color):
    var a = y0 if y0 <= y1 else y1
    var b = y1 if y0 <= y1 else y0
    for y in range(a, b + 1):
        c.set_pixel(x, y, col)


def line(mut c: Canvas, x0: Int, y0: Int, x1: Int, y1: Int, col: Color):
    """Bresenham line, any slope (handles horizontal/vertical/diagonal)."""
    var x = x0
    var y = y0
    var dx = _iabs(x1 - x0)
    var dy = -_iabs(y1 - y0)
    var sx = 1 if x0 < x1 else -1
    var sy = 1 if y0 < y1 else -1
    var err = dx + dy
    while True:
        c.set_pixel(x, y, col)
        if x == x1 and y == y1:
            break
        var e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


# ── anti-aliased line (Xiaolin Wu) ────────────────────────────────────────────
def _ffloor(x: Float64) -> Int:
    var i = Int(x)
    return i if Float64(i) <= x else i - 1


def _fpart(x: Float64) -> Float64:
    return x - Float64(_ffloor(x))


def _rfpart(x: Float64) -> Float64:
    return 1.0 - _fpart(x)


def _fabs(x: Float64) -> Float64:
    return -x if x < 0.0 else x


def _iround(x: Float64) -> Int:
    return _ffloor(x + 0.5)


def _plot_aa(mut c: Canvas, x: Int, y: Int, col: Color, cov: Float64):
    if cov <= 0.0:
        return
    var cc = cov if cov < 1.0 else 1.0
    var a = Int(Float64(Int(col.a)) * cc + 0.5)
    c.set_pixel(x, y, Color(col.r, col.g, col.b, UInt8(a)))


def line_aa(mut c: Canvas, x0f: Float64, y0f: Float64, x1f: Float64, y1f: Float64, col: Color):
    """Anti-aliased line (Wu): pixels are blended by edge coverage, so diagonals
    look smooth instead of stairstepped. Float endpoints."""
    var x0 = x0f; var y0 = y0f; var x1 = x1f; var y1 = y1f
    var steep = _fabs(y1 - y0) > _fabs(x1 - x0)
    if steep:
        var t = x0; x0 = y0; y0 = t
        t = x1; x1 = y1; y1 = t
    if x0 > x1:
        var t = x0; x0 = x1; x1 = t
        t = y0; y0 = y1; y1 = t
    var dx = x1 - x0
    var dy = y1 - y0
    var grad = 1.0 if dx == 0.0 else dy / dx

    # endpoint 1
    var xend1 = Float64(_iround(x0))
    var yend1 = y0 + grad * (xend1 - x0)
    var xgap1 = _rfpart(x0 + 0.5)
    var xpxl1 = Int(xend1)
    var ypxl1 = _ffloor(yend1)
    if steep:
        _plot_aa(c, ypxl1, xpxl1, col, _rfpart(yend1) * xgap1)
        _plot_aa(c, ypxl1 + 1, xpxl1, col, _fpart(yend1) * xgap1)
    else:
        _plot_aa(c, xpxl1, ypxl1, col, _rfpart(yend1) * xgap1)
        _plot_aa(c, xpxl1, ypxl1 + 1, col, _fpart(yend1) * xgap1)
    var intery = yend1 + grad

    # endpoint 2
    var xend2 = Float64(_iround(x1))
    var yend2 = y1 + grad * (xend2 - x1)
    var xgap2 = _fpart(x1 + 0.5)
    var xpxl2 = Int(xend2)
    var ypxl2 = _ffloor(yend2)
    if steep:
        _plot_aa(c, ypxl2, xpxl2, col, _rfpart(yend2) * xgap2)
        _plot_aa(c, ypxl2 + 1, xpxl2, col, _fpart(yend2) * xgap2)
    else:
        _plot_aa(c, xpxl2, ypxl2, col, _rfpart(yend2) * xgap2)
        _plot_aa(c, xpxl2, ypxl2 + 1, col, _fpart(yend2) * xgap2)

    # main span
    for x in range(xpxl1 + 1, xpxl2):
        var iy = _ffloor(intery)
        if steep:
            _plot_aa(c, iy, x, col, _rfpart(intery))
            _plot_aa(c, iy + 1, x, col, _fpart(intery))
        else:
            _plot_aa(c, x, iy, col, _rfpart(intery))
            _plot_aa(c, x, iy + 1, col, _fpart(intery))
        intery += grad


def rect(mut c: Canvas, x: Int, y: Int, w: Int, h: Int, col: Color):
    """Rectangle outline (1px). Top-left (x,y), size w x h."""
    if w <= 0 or h <= 0:
        return
    hline(c, x, x + w - 1, y, col)
    hline(c, x, x + w - 1, y + h - 1, col)
    vline(c, x, y, y + h - 1, col)
    vline(c, x + w - 1, y, y + h - 1, col)


def circle(mut c: Canvas, cx: Int, cy: Int, r: Int, col: Color):
    """Circle outline (midpoint algorithm, 8-way symmetry)."""
    if r < 0:
        return
    var x = r
    var y = 0
    var err = 1 - r
    while x >= y:
        c.set_pixel(cx + x, cy + y, col)
        c.set_pixel(cx - x, cy + y, col)
        c.set_pixel(cx + x, cy - y, col)
        c.set_pixel(cx - x, cy - y, col)
        c.set_pixel(cx + y, cy + x, col)
        c.set_pixel(cx - y, cy + x, col)
        c.set_pixel(cx + y, cy - x, col)
        c.set_pixel(cx - y, cy - x, col)
        y += 1
        if err < 0:
            err += 2 * y + 1
        else:
            x -= 1
            err += 2 * (y - x) + 1


def polygon(mut c: Canvas, xs: List[Int], ys: List[Int], col: Color) raises:
    """Closed polygon outline through the points."""
    var n = len(xs)
    if n < 2:
        return
    for i in range(n):
        var j = (i + 1) % n
        line(c, xs[i], ys[i], xs[j], ys[j], col)


def fill_polygon(mut c: Canvas, xs: List[Int], ys: List[Int], col: Color) raises:
    """Scanline polygon fill (even-odd rule). Handles convex and concave shapes;
    self-intersections fill by parity. For smooth edges, fill at Nx and
    downsample (SSAA)."""
    var n = len(xs)
    if n < 3:
        return
    var ymin = ys[0]
    var ymax = ys[0]
    for i in range(1, n):
        if ys[i] < ymin:
            ymin = ys[i]
        if ys[i] > ymax:
            ymax = ys[i]
    for y in range(ymin, ymax + 1):
        var yc = Float64(y) + 0.5
        var xints = List[Float64]()
        var j = n - 1
        for i in range(n):
            var yi = Float64(ys[i])
            var yj = Float64(ys[j])
            if (yi <= yc and yj > yc) or (yj <= yc and yi > yc):
                var tnum = yc - yi
                var x = Float64(xs[i]) + (tnum / (yj - yi)) * (Float64(xs[j]) - Float64(xs[i]))
                xints.append(x)
            j = i
        # insertion sort the crossings
        for a in range(1, len(xints)):
            var key = xints[a]
            var b = a - 1
            while b >= 0 and xints[b] > key:
                xints[b + 1] = xints[b]
                b -= 1
            xints[b + 1] = key
        # fill between consecutive pairs
        var k = 0
        while k + 1 < len(xints):
            var xa = Int(xints[k] + 0.5)
            var xb = Int(xints[k + 1] + 0.5)
            if xb > xa:
                hline(c, xa, xb - 1, y, col)
            k += 2


def fill_triangle(mut c: Canvas, x0: Int, y0: Int, x1: Int, y1: Int, x2: Int, y2: Int, col: Color) raises:
    var xs = [x0, x1, x2]
    var ys = [y0, y1, y2]
    fill_polygon(c, xs, ys, col)


def fill_triangle_aa(mut c: Canvas, x0: Int, y0: Int, x1: Int, y1: Int, x2: Int, y2: Int, col: Color, samples: Int = 4) raises:
    """Anti-aliased filled triangle (coverage rasterizer)."""
    var xs = [x0, x1, x2]
    var ys = [y0, y1, y2]
    fill_polygon_aa(c, xs, ys, col, samples)


def fill_polygon_aa(mut c: Canvas, xs: List[Int], ys: List[Int], col: Color, samples: Int = 4) raises:
    """Analytic anti-aliased polygon fill (even-odd). Each output row is sampled
    on `samples` sub-scanlines; horizontal span ends contribute fractional
    coverage. Edge pixels get partial alpha — smooth edges at 1x resolution, no
    supersampled canvas. (Coverage accumulates in a single per-row buffer.)"""
    var n = len(xs)
    if n < 3:
        return
    var W = c.w
    if W <= 0:
        return
    var ymin = ys[0]
    var ymax = ys[0]
    for i in range(1, n):
        if ys[i] < ymin:
            ymin = ys[i]
        if ys[i] > ymax:
            ymax = ys[i]
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
            var j = n - 1
            for i in range(n):
                var yi = Float64(ys[i])
                var yj = Float64(ys[j])
                if (yi <= yc and yj > yc) or (yj <= yc and yi > yc):
                    xints.append(Float64(xs[i]) + ((yc - yi) / (yj - yi)) * (Float64(xs[j]) - Float64(xs[i])))
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


def fill_circle(mut c: Canvas, cx: Int, cy: Int, r: Int, col: Color):
    """Filled disc: one horizontal span per row, width from integer sqrt."""
    if r < 0:
        return
    var r2 = r * r
    for dy in range(-r, r + 1):
        var dx = _isqrt(r2 - dy * dy)
        hline(c, cx - dx, cx + dx, cy + dy, col)


def rounded_rect_fill(mut c: Canvas, x: Int, y: Int, w: Int, h: Int, radius: Int, col: Color):
    """Filled rectangle with rounded corners (radius). Two overlapping bands plus
    four corner discs — the UI card/button shape. Draw at Nx + downsample for AA."""
    if w <= 0 or h <= 0:
        return
    var r = radius
    var maxr = (w if w < h else h) // 2
    if r > maxr:
        r = maxr
    if r <= 0:
        c.fill_rect(x, y, w, h, col)
        return
    c.fill_rect(x + r, y, w - 2 * r, h, col)        # vertical band
    c.fill_rect(x, y + r, w, h - 2 * r, col)        # horizontal band
    fill_circle(c, x + r, y + r, r, col)            # corners
    fill_circle(c, x + w - 1 - r, y + r, r, col)
    fill_circle(c, x + r, y + h - 1 - r, r, col)
    fill_circle(c, x + w - 1 - r, y + h - 1 - r, r, col)
