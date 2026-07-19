# svg/shapes.mojo — convert SVG basic shapes to absolute SvgSeg lists.
#
# <rect> (incl. rx/ry rounding), <circle>, <ellipse>, <line>, <polyline>,
# <polygon>. Rounded corners and circles/ellipses are emitted as cubic
# segments via the same arc converter the path parser uses, so downstream
# rasterization is identical to <path>.

from svg.pathdata import SvgSeg, SEG_M, SEG_L, SEG_C, SEG_Q, SEG_Z, _arc_to_cubics


def rect_segs(x: Float64, y: Float64, w: Float64, h: Float64,
              rx_in: Float64 = 0.0, ry_in: Float64 = 0.0) raises -> List[SvgSeg]:
    var segs = List[SvgSeg]()
    if w <= 0.0 or h <= 0.0:
        return segs^
    # SVG rounding rules: a missing rx/ry mirrors the other; clamp to half-extent.
    var rx = rx_in
    var ry = ry_in
    if rx <= 0.0 and ry > 0.0:
        rx = ry
    if ry <= 0.0 and rx > 0.0:
        ry = rx
    if rx > w / 2.0:
        rx = w / 2.0
    if ry > h / 2.0:
        ry = h / 2.0

    if rx <= 0.0 or ry <= 0.0:
        segs.append(SvgSeg(SEG_M, x=x, y=y))
        segs.append(SvgSeg(SEG_L, x=x + w, y=y))
        segs.append(SvgSeg(SEG_L, x=x + w, y=y + h))
        segs.append(SvgSeg(SEG_L, x=x, y=y + h))
        segs.append(SvgSeg(SEG_Z))
        return segs^

    # rounded: start after the top-left corner, go clockwise, arc each corner.
    segs.append(SvgSeg(SEG_M, x=x + rx, y=y))
    segs.append(SvgSeg(SEG_L, x=x + w - rx, y=y))
    _arc_to_cubics(segs, x + w - rx, y, rx, ry, 0.0, 0, 1, x + w, y + ry)
    segs.append(SvgSeg(SEG_L, x=x + w, y=y + h - ry))
    _arc_to_cubics(segs, x + w, y + h - ry, rx, ry, 0.0, 0, 1, x + w - rx, y + h)
    segs.append(SvgSeg(SEG_L, x=x + rx, y=y + h))
    _arc_to_cubics(segs, x + rx, y + h, rx, ry, 0.0, 0, 1, x, y + h - ry)
    segs.append(SvgSeg(SEG_L, x=x, y=y + ry))
    _arc_to_cubics(segs, x, y + ry, rx, ry, 0.0, 0, 1, x + rx, y)
    segs.append(SvgSeg(SEG_Z))
    return segs^


def ellipse_segs(cx: Float64, cy: Float64, rx: Float64, ry: Float64) raises -> List[SvgSeg]:
    var segs = List[SvgSeg]()
    if rx <= 0.0 or ry <= 0.0:
        return segs^
    # full ellipse as two 180-degree arcs (sweep=1), start at the right vertex.
    segs.append(SvgSeg(SEG_M, x=cx + rx, y=cy))
    _arc_to_cubics(segs, cx + rx, cy, rx, ry, 0.0, 0, 1, cx - rx, cy)
    _arc_to_cubics(segs, cx - rx, cy, rx, ry, 0.0, 0, 1, cx + rx, cy)
    segs.append(SvgSeg(SEG_Z))
    return segs^


def circle_segs(cx: Float64, cy: Float64, r: Float64) raises -> List[SvgSeg]:
    return ellipse_segs(cx, cy, r, r)


def line_segs(x1: Float64, y1: Float64, x2: Float64, y2: Float64) -> List[SvgSeg]:
    var segs = List[SvgSeg]()
    segs.append(SvgSeg(SEG_M, x=x1, y=y1))
    segs.append(SvgSeg(SEG_L, x=x2, y=y2))
    return segs^


def poly_segs(pts: List[Float64], closed: Bool) raises -> List[SvgSeg]:
    """pts is a flat [x0,y0,x1,y1,...]; `closed` appends Z (polygon vs polyline)."""
    var segs = List[SvgSeg]()
    var n = len(pts) // 2
    if n == 0:
        return segs^
    segs.append(SvgSeg(SEG_M, x=pts[0], y=pts[1]))
    for i in range(1, n):
        segs.append(SvgSeg(SEG_L, x=pts[2 * i], y=pts[2 * i + 1]))
    if closed:
        segs.append(SvgSeg(SEG_Z))
    return segs^
