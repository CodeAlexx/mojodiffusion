# graphics.gradient — linear and radial gradient fills + color interpolation.

from std.math import sqrt
from graphics.canvas import Canvas
from graphics.color import Color


def lerp_color(a: Color, b: Color, t: Float64) -> Color:
    """Linear blend a->b at t in [0,1] (per channel, incl. alpha)."""
    var tt = t
    if tt < 0.0:
        tt = 0.0
    elif tt > 1.0:
        tt = 1.0
    var u = 1.0 - tt
    return Color(
        UInt8(Int(Float64(Int(a.r)) * u + Float64(Int(b.r)) * tt + 0.5)),
        UInt8(Int(Float64(Int(a.g)) * u + Float64(Int(b.g)) * tt + 0.5)),
        UInt8(Int(Float64(Int(a.b)) * u + Float64(Int(b.b)) * tt + 0.5)),
        UInt8(Int(Float64(Int(a.a)) * u + Float64(Int(b.a)) * tt + 0.5)),
    )


def linear_gradient(mut c: Canvas, x: Int, y: Int, w: Int, h: Int, c0: Color, c1: Color, horizontal: Bool = True):
    """Fill the rect (x,y,w,h) with a c0->c1 linear gradient (horizontal or vertical)."""
    if w <= 0 or h <= 0:
        return
    var denom_h = Float64(w - 1) if w > 1 else 1.0
    var denom_v = Float64(h - 1) if h > 1 else 1.0
    for j in range(h):
        for i in range(w):
            var t = Float64(i) / denom_h if horizontal else Float64(j) / denom_v
            c.set_pixel(x + i, y + j, lerp_color(c0, c1, t))


def fill_triangle_gradient(
    mut c: Canvas,
    x0: Int, y0: Int, c0: Color,
    x1: Int, y1: Int, c1: Color,
    x2: Int, y2: Int, c2: Color,
) raises:
    """Gouraud-shaded triangle: color interpolated per-vertex via barycentric
    weights (smooth color across the face)."""
    var denom = Float64((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
    if denom == 0.0:
        return
    var minx = x0
    if x1 < minx: minx = x1
    if x2 < minx: minx = x2
    var maxx = x0
    if x1 > maxx: maxx = x1
    if x2 > maxx: maxx = x2
    var miny = y0
    if y1 < miny: miny = y1
    if y2 < miny: miny = y2
    var maxy = y0
    if y1 > maxy: maxy = y1
    if y2 > maxy: maxy = y2
    for py in range(miny, maxy + 1):
        for px in range(minx, maxx + 1):
            var w0 = Float64((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
            var w1 = Float64((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
            var w2 = 1.0 - w0 - w1
            if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                var r = Int(w0 * Float64(Int(c0.r)) + w1 * Float64(Int(c1.r)) + w2 * Float64(Int(c2.r)) + 0.5)
                var g = Int(w0 * Float64(Int(c0.g)) + w1 * Float64(Int(c1.g)) + w2 * Float64(Int(c2.g)) + 0.5)
                var b = Int(w0 * Float64(Int(c0.b)) + w1 * Float64(Int(c1.b)) + w2 * Float64(Int(c2.b)) + 0.5)
                c.set_pixel(px, py, Color(UInt8(r), UInt8(g), UInt8(b), UInt8(255)))


def _stop_color(positions: List[Float64], colors: List[Color], t: Float64) -> Color:
    """Color at parameter t in [0,1] across multi-stop `positions` (ascending)."""
    var ns = len(positions)
    if t <= positions[0]:
        return colors[0]
    if t >= positions[ns - 1]:
        return colors[ns - 1]
    var k = 0
    while k + 1 < ns and positions[k + 1] < t:
        k += 1
    var p0 = positions[k]
    var p1 = positions[k + 1]
    var lt = 0.0 if p1 <= p0 else (t - p0) / (p1 - p0)
    return lerp_color(colors[k], colors[k + 1], lt)


def linear_gradient_multi(
    mut c: Canvas, x: Int, y: Int, w: Int, h: Int,
    positions: List[Float64], colors: List[Color], horizontal: Bool = True,
) raises:
    """Multi-stop linear gradient. `positions` (ascending, 0..1) pair with `colors`."""
    if w <= 0 or h <= 0 or len(positions) < 2 or len(positions) != len(colors):
        return
    var dh = Float64(w - 1) if w > 1 else 1.0
    var dv = Float64(h - 1) if h > 1 else 1.0
    for j in range(h):
        for i in range(w):
            var t = Float64(i) / dh if horizontal else Float64(j) / dv
            c.set_pixel(x + i, y + j, _stop_color(positions, colors, t))


def radial_gradient_multi(
    mut c: Canvas, cx: Int, cy: Int, r: Int,
    positions: List[Float64], colors: List[Color],
) raises:
    """Multi-stop radial gradient (center t=0 -> edge t=1)."""
    if r <= 0 or len(positions) < 2 or len(positions) != len(colors):
        return
    var rf = Float64(r)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            var d = sqrt(Float64(dx * dx + dy * dy))
            if d <= rf:
                c.set_pixel(cx + dx, cy + dy, _stop_color(positions, colors, d / rf))


def radial_gradient(mut c: Canvas, cx: Int, cy: Int, r: Int, inner: Color, outer: Color):
    """Fill a disc of radius r with inner (center) -> outer (edge) gradient."""
    if r <= 0:
        return
    var rf = Float64(r)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            var d = sqrt(Float64(dx * dx + dy * dy))
            if d <= rf:
                c.set_pixel(cx + dx, cy + dy, lerp_color(inner, outer, d / rf))
