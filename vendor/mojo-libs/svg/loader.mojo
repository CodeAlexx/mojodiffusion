# svg/loader.mojo — load an SVG (icon subset) into a graphics.Canvas (RGBA).
#
# Pipeline: parse_xml -> walk elements -> resolve inherited fill/stroke/width +
# transform (incl. viewBox fit) -> path/shape -> graphics.Path (device space) ->
# even-odd fill_path_aa and/or stroke_path on a transparent Canvas.
#
# Scope: <path>, <rect>/<circle>/<ellipse>/<line>/<polyline>/<polygon>, <g>,
# nested transforms, viewBox (xMidYMid meet), fill/stroke/stroke-width/opacity
# from presentation attributes and inline style="". Out of scope (ignored, not
# errored): gradients/patterns (paints fall back to currentColor), <use>/<defs>
# references, clip-paths, filters, masks, text.

from std.collections import Dict
from std.math import sqrt
from graphics.canvas import Canvas
from graphics.color import Color, rgb, rgba, transparent
from graphics.path import Path, fill_path_aa, stroke_path
from graphics.transform import Affine, identity
from svg.xml import XmlNode, parse_xml
from svg.pathdata import parse_path_d, build_path, SvgSeg, SEG_Z
from svg.shapes import rect_segs, ellipse_segs, circle_segs, line_segs, poly_segs
from svg.style import Paint, parse_color, parse_opacity, parse_viewbox, parse_transform


def _attrf(node: XmlNode, key: String, default: Float64) raises -> Float64:
    if node.has(key):
        var v = node.attr(key, String(""))
        if len(v) > 0:
            return atof(v)
    return default


def _parse_style(s: String) raises -> Dict[String, String]:
    var d = Dict[String, String]()
    var b = s.as_bytes()
    var n = len(b)
    var i = 0
    while i < n:
        # read up to ';'
        var start = i
        while i < n and Int(b[i]) != 59:   # ';'
            i += 1
        var decl = String(StringSlice(ptr=b.unsafe_ptr() + start, length=i - start))
        i += 1
        var colon = decl.find(String(":"))
        if colon > 0:
            var k = _strip(decl, 0, colon)
            var v = _strip(decl, colon + 1, decl.byte_length())
            if len(k) > 0:
                d[k] = v
    return d^


def _strip(s: String, a: Int, b_: Int) -> String:
    var b = s.as_bytes()
    var i = a
    var j = b_
    while i < j and (Int(b[i]) == 0x20 or Int(b[i]) == 0x09):
        i += 1
    while j > i and (Int(b[j - 1]) == 0x20 or Int(b[j - 1]) == 0x09):
        j -= 1
    if j <= i:
        return String("")
    return String(StringSlice(ptr=b.unsafe_ptr() + i, length=j - i))


def _prop(node: XmlNode, style: Dict[String, String], key: String) raises -> String:
    # inline style="" wins over a presentation attribute
    if key in style:
        return style[key]
    if node.has(key):
        return node.attr(key, String(""))
    return String("")


def _avg_scale(m: Affine) -> Float64:
    var det = m.a * m.d - m.b * m.c
    var ad = det if det >= 0.0 else -det
    return sqrt(ad)


def _poly_points(raw: String) raises -> List[Float64]:
    var out = List[Float64]()
    var b = raw.as_bytes()
    var n = len(b)
    var i = 0
    while i < n:
        var c = Int(b[i])
        if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D or c == 0x2C:
            i += 1
            continue
        var start = i
        if Int(b[i]) == 43 or Int(b[i]) == 45:
            i += 1
        while i < n and ((Int(b[i]) >= 48 and Int(b[i]) <= 57) or Int(b[i]) == 46 or Int(b[i]) == 101 or Int(b[i]) == 69):
            i += 1
        if i > start:
            out.append(atof(String(StringSlice(ptr=b.unsafe_ptr() + start, length=i - start))))
        else:
            i += 1
    return out^


def _segs_for(node: XmlNode) raises -> List[SvgSeg]:
    var nm = node.name
    if nm == "path":
        return parse_path_d(node.attr(String("d"), String("")))
    if nm == "rect":
        return rect_segs(
            _attrf(node, String("x"), 0.0), _attrf(node, String("y"), 0.0),
            _attrf(node, String("width"), 0.0), _attrf(node, String("height"), 0.0),
            _attrf(node, String("rx"), 0.0), _attrf(node, String("ry"), 0.0),
        )
    if nm == "circle":
        return circle_segs(_attrf(node, String("cx"), 0.0), _attrf(node, String("cy"), 0.0), _attrf(node, String("r"), 0.0))
    if nm == "ellipse":
        return ellipse_segs(_attrf(node, String("cx"), 0.0), _attrf(node, String("cy"), 0.0), _attrf(node, String("rx"), 0.0), _attrf(node, String("ry"), 0.0))
    if nm == "line":
        return line_segs(_attrf(node, String("x1"), 0.0), _attrf(node, String("y1"), 0.0), _attrf(node, String("x2"), 0.0), _attrf(node, String("y2"), 0.0))
    if nm == "polygon":
        return poly_segs(_poly_points(node.attr(String("points"), String(""))), True)
    if nm == "polyline":
        return poly_segs(_poly_points(node.attr(String("points"), String(""))), False)
    return List[SvgSeg]()


def _is_closed_shape(node: XmlNode, segs: List[SvgSeg]) -> Bool:
    var nm = node.name
    if nm == "rect" or nm == "circle" or nm == "ellipse" or nm == "polygon":
        return True
    if nm == "path":
        for i in range(len(segs)):
            if segs[i].kind == SEG_Z:
                return True
    return False


def _with_alpha(c: Color, op: Float64) -> Color:
    var a = Int(255.0 * op)
    if a < 0:
        a = 0
    if a > 255:
        a = 255
    return rgba(Int(c.r), Int(c.g), Int(c.b), a)


def _render(
    mut canvas: Canvas, node: XmlNode, ptf: Affine,
    pfill: Paint, pstroke: Paint, pwidth: Float64, tint: Color,
) raises:
    var nm = node.name
    if nm == "defs" or nm == "title" or nm == "desc" or nm == "style" or nm == "metadata" or nm == "clipPath" or nm == "mask":
        return

    var style = _parse_style(node.attr(String("style"), String("")))

    # transform (element-local applied first, then parent's)
    var eff = ptf
    if node.has(String("transform")):
        eff = parse_transform(node.attr(String("transform"), String(""))).then(ptf)

    # inherited fill/stroke/width
    var fill = pfill
    var fs = _prop(node, style, String("fill"))
    if len(fs) > 0:
        fill = parse_color(fs, tint)
    var stroke = pstroke
    var ss = _prop(node, style, String("stroke"))
    if len(ss) > 0:
        stroke = parse_color(ss, tint)
    var width = pwidth
    var ws = _prop(node, style, String("stroke-width"))
    if len(ws) > 0:
        width = atof(ws)

    # opacity (combined onto paint alpha)
    var op = 1.0
    var os = _prop(node, style, String("opacity"))
    if len(os) > 0:
        op = parse_opacity(os)
    var fop = op
    var fos = _prop(node, style, String("fill-opacity"))
    if len(fos) > 0:
        fop = op * parse_opacity(fos)
    var sop = op
    var sos = _prop(node, style, String("stroke-opacity"))
    if len(sos) > 0:
        sop = op * parse_opacity(sos)

    # geometry
    var segs = _segs_for(node)
    if len(segs) > 0:
        var p = build_path(segs).transformed(eff)
        if fill.has:
            fill_path_aa(canvas, p, _with_alpha(fill.color, fop))
        if stroke.has and width > 0.0:
            var dw = width * _avg_scale(eff)
            if dw < 1.0:
                dw = 1.0
            stroke_path(canvas, p, _with_alpha(stroke.color, sop), dw, closed=_is_closed_shape(node, segs))

    # recurse children with resolved state
    for i in range(len(node.children)):
        var ch = node.children[i].copy()
        _render(canvas, ch, eff, fill, stroke, width, tint)


def load_svg_text(text: String, out_w: Int, out_h: Int, current_color: Color) raises -> Canvas:
    """Render an SVG string to an out_w x out_h transparent RGBA Canvas.

    `current_color` resolves `currentColor` and is the fallback for unsupported
    paints (gradients) — pass your theme's icon tint.
    """
    var root = parse_xml(text)

    # viewBox / size -> fit transform (xMidYMid meet)
    var minx = 0.0
    var miny = 0.0
    var vbw = Float64(out_w)
    var vbh = Float64(out_h)
    var vb = parse_viewbox(root.attr(String("viewBox"), String("")))
    if len(vb) == 4 and vb[2] > 0.0 and vb[3] > 0.0:
        minx = vb[0]; miny = vb[1]; vbw = vb[2]; vbh = vb[3]
    else:
        var w = _attrf(root, String("width"), 0.0)
        var h = _attrf(root, String("height"), 0.0)
        if w > 0.0 and h > 0.0:
            vbw = w; vbh = h

    var sx = Float64(out_w) / vbw
    var sy = Float64(out_h) / vbh
    var s = sx if sx < sy else sy
    var offx = (Float64(out_w) - vbw * s) / 2.0
    var offy = (Float64(out_h) - vbh * s) / 2.0
    var base = Affine(s, 0.0, 0.0, s, offx - s * minx, offy - s * miny)

    var canvas = Canvas(out_w, out_h)
    canvas.clear_rgba(transparent())

    var def_fill = Paint(True, rgb(0, 0, 0))     # SVG default fill is black
    var def_stroke = Paint(False, rgb(0, 0, 0))  # default stroke is none
    _render(canvas, root, base, def_fill, def_stroke, 1.0, current_color)
    return canvas^


def load_svg_file(path: String, out_w: Int, out_h: Int, current_color: Color) raises -> Canvas:
    var data = String("")
    with open(path, "r") as f:
        data = f.read()
    return load_svg_text(data, out_w, out_h, current_color)


def to_rgba_list(c: Canvas) -> List[UInt8]:
    """Copy the canvas pixels to a List[UInt8] (RGBA8) for texture upload,
    e.g. MojoUI `Backend.make_texture_rgba(w, h, list)`."""
    var out = List[UInt8]()
    var n = c.w * c.h * 4
    for i in range(n):
        out.append(c.px[i])
    return out^
