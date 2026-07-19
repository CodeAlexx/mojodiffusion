# graphics.chart — bar / line / scatter plots, built on draw + text.
#
# Each chart draws into a plot rectangle (px, py, pw, ph) you provide: x-axis
# (baseline) at the bottom, y-axis at the left, values scaled to the data range.
# Data is List[Float64]. Labels use the 5x7 font. No floats leak into pixel math
# beyond the value->pixel mapping.

from std.math import sin, cos
from graphics.canvas import Canvas
from graphics.color import Color, rgb, white, gray
from graphics.draw import line, hline, vline, rect, fill_circle, fill_polygon
from graphics.text import draw_text, text_width

comptime _TAU = 6.283185307179586
comptime _HALF_PI = 1.5707963267948966


def _imax(a: Int, b: Int) -> Int:
    return a if a > b else b


def _round_int(v: Float64) -> Int:
    if v >= 0.0:
        return Int(v + 0.5)
    return -Int(-v + 0.5)


def _maxf(v: List[Float64]) -> Float64:
    var m = v[0]
    for i in range(1, len(v)):
        if v[i] > m:
            m = v[i]
    return m


def _minf(v: List[Float64]) -> Float64:
    var m = v[0]
    for i in range(1, len(v)):
        if v[i] < m:
            m = v[i]
    return m


def _axes(mut c: Canvas, px: Int, py: Int, pw: Int, ph: Int, axis: Color):
    vline(c, px, py, py + ph, axis)             # y-axis
    hline(c, px, px + pw, py + ph, axis)        # x-axis / baseline


def _gridlines(mut c: Canvas, px: Int, py: Int, pw: Int, ph: Int, n: Int, col: Color):
    for i in range(1, n + 1):
        var gy = py + ph - (ph * i) // (n + 1)
        hline(c, px, px + pw, gy, col)


def bar_chart(
    mut c: Canvas, px: Int, py: Int, pw: Int, ph: Int,
    values: List[Float64], bar: Color, axis: Color, label: Bool = True,
) raises:
    """Vertical bars scaled to the max value, baseline at the bottom."""
    var n = len(values)
    if n == 0:
        return
    var vmax = _maxf(values)
    if vmax <= 0.0:
        vmax = 1.0
    _gridlines(c, px, py, pw, ph, 3, rgb(45, 48, 60))
    _axes(c, px, py, pw, ph, axis)
    var slot = pw // n
    var bw = _imax(1, slot * 7 // 10)
    for i in range(n):
        var bh = _round_int((values[i] / vmax) * Float64(ph))
        var bx = px + i * slot + (slot - bw) // 2
        var by = py + ph - bh
        c.fill_rect(bx, by, bw, bh, bar)
        if label:
            var s = String(_round_int(values[i]))
            var lx = bx + (bw - text_width(s, 1)) // 2
            draw_text(c, lx, by - 9, s, axis, 1)


def line_chart(
    mut c: Canvas, px: Int, py: Int, pw: Int, ph: Int,
    values: List[Float64], lc: Color, axis: Color,
) raises:
    """Polyline of the values across the plot, scaled to [min,max], with point dots."""
    var n = len(values)
    if n == 0:
        return
    _gridlines(c, px, py, pw, ph, 3, rgb(45, 48, 60))
    _axes(c, px, py, pw, ph, axis)
    if n == 1:
        fill_circle(c, px, py + ph, 2, lc)
        return
    var vmax = _maxf(values)
    var vmin = _minf(values)
    var span = vmax - vmin
    if span <= 0.0:
        span = 1.0
    var prevx = 0
    var prevy = 0
    for i in range(n):
        var x = px + (pw * i) // (n - 1)
        var y = py + ph - _round_int(((values[i] - vmin) / span) * Float64(ph))
        if i > 0:
            line(c, prevx, prevy, x, y, lc)
        prevx = x
        prevy = y
    # dots on top so they sit above the line
    for i in range(n):
        var x = px + (pw * i) // (n - 1)
        var y = py + ph - _round_int(((values[i] - vmin) / span) * Float64(ph))
        fill_circle(c, x, y, 2, lc)


def scatter(
    mut c: Canvas, px: Int, py: Int, pw: Int, ph: Int,
    xs: List[Float64], ys: List[Float64], dot: Color, axis: Color, radius: Int = 2,
) raises:
    """Scatter xs vs ys, each axis auto-scaled to its own [min,max]."""
    var n = len(xs)
    if n == 0 or len(ys) != n:
        return
    _axes(c, px, py, pw, ph, axis)
    var xmax = _maxf(xs)
    var xmin = _minf(xs)
    var ymax = _maxf(ys)
    var ymin = _minf(ys)
    var xspan = xmax - xmin
    var yspan = ymax - ymin
    if xspan <= 0.0:
        xspan = 1.0
    if yspan <= 0.0:
        yspan = 1.0
    for i in range(n):
        var x = px + _round_int(((xs[i] - xmin) / xspan) * Float64(pw))
        var y = py + ph - _round_int(((ys[i] - ymin) / yspan) * Float64(ph))
        fill_circle(c, x, y, radius, dot)


def _sumf(v: List[Float64]) -> Float64:
    var s = 0.0
    for i in range(len(v)):
        s += v[i]
    return s


def pie_chart(
    mut c: Canvas, cx: Int, cy: Int, r: Int,
    values: List[Float64], colors: List[Color],
) raises:
    """Pie slices proportional to values, starting at the top, clockwise."""
    var total = _sumf(values)
    if total <= 0.0 or len(values) == 0:
        return
    var ang = -_HALF_PI  # start at 12 o'clock
    var nc = len(colors)
    for i in range(len(values)):
        var sweep = (values[i] / total) * _TAU
        var steps = Int(sweep / 0.08)
        if steps < 2:
            steps = 2
        var xs = List[Int]()
        var ys = List[Int]()
        xs.append(cx); ys.append(cy)
        for s in range(steps + 1):
            var a = ang + sweep * Float64(s) / Float64(steps)
            xs.append(cx + Int(Float64(r) * cos(a)))
            ys.append(cy + Int(Float64(r) * sin(a)))
        fill_polygon(c, xs, ys, colors[i % nc])
        ang += sweep


def donut_chart(
    mut c: Canvas, cx: Int, cy: Int, r: Int, inner_r: Int,
    values: List[Float64], colors: List[Color], hole: Color,
) raises:
    """Pie with the center punched out to `inner_r` (filled with `hole`)."""
    pie_chart(c, cx, cy, r, values, colors)
    fill_circle(c, cx, cy, inner_r, hole)


def area_chart(
    mut c: Canvas, px: Int, py: Int, pw: Int, ph: Int,
    values: List[Float64], fill: Color, lc: Color, axis: Color,
) raises:
    """Filled area under the series (0-baseline), with the series line on top."""
    var n = len(values)
    if n < 2:
        return
    var vmax = _maxf(values)
    if vmax <= 0.0:
        vmax = 1.0
    _gridlines(c, px, py, pw, ph, 3, rgb(45, 48, 60))
    _axes(c, px, py, pw, ph, axis)
    var xs = List[Int]()
    var ys = List[Int]()
    for i in range(n):
        xs.append(px + (pw * i) // (n - 1))
        ys.append(py + ph - _round_int((values[i] / vmax) * Float64(ph)))
    # close the polygon down the baseline
    var fxs = List[Int]()
    var fys = List[Int]()
    for i in range(n):
        fxs.append(xs[i]); fys.append(ys[i])
    fxs.append(px + pw); fys.append(py + ph)
    fxs.append(px); fys.append(py + ph)
    fill_polygon(c, fxs, fys, fill)
    for i in range(n - 1):
        line(c, xs[i], ys[i], xs[i + 1], ys[i + 1], lc)
