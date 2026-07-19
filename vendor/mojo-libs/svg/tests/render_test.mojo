# svg/tests/render_test.mojo — end-to-end: SVG text -> Canvas -> PNG.
# Renders a few representative icons and writes PNGs + a coverage report the
# oracle script cross-checks. Coverage = count of non-transparent pixels and the
# painted bounding box, which a PIL reference reproduces for the same geometry.

from graphics.canvas import Canvas
from graphics.color import rgb
from graphics.png import write_png
from svg.loader import load_svg_text


def _report(name: String, c: Canvas):
    var painted = 0
    var minx = c.w
    var miny = c.h
    var maxx = -1
    var maxy = -1
    for y in range(c.h):
        for x in range(c.w):
            var i = (y * c.w + x) * 4
            if Int(c.px[i + 3]) > 0:
                painted += 1
                if x < minx: minx = x
                if y < miny: miny = y
                if x > maxx: maxx = x
                if y > maxy: maxy = y
    print("REPORT", name, "painted", painted, "bbox", minx, miny, maxx, maxy)


def main() raises:
    var tint = rgb(20, 20, 20)

    # 1) filled rect via viewBox fit: a 24x24 viewBox, 10x10 rect at (7,7)
    var rect_svg = String("<svg viewBox='0 0 24 24'><rect x='7' y='7' width='10' height='10' fill='#000'/></svg>")
    var c1 = load_svg_text(rect_svg, 96, 96, tint)
    write_png(String("/tmp/svg_rect.png"), c1)
    _report(String("rect"), c1)

    # 2) circle r=10 centered in 24x24
    var circ_svg = String("<svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='10' fill='black'/></svg>")
    var c2 = load_svg_text(circ_svg, 96, 96, tint)
    write_png(String("/tmp/svg_circle.png"), c2)
    _report(String("circle"), c2)

    # 3) a real-world style stroked path (a check mark), stroke only
    var check_svg = String("<svg viewBox='0 0 24 24'><path d='M4 12 L10 18 L20 6' fill='none' stroke='black' stroke-width='2'/></svg>")
    var c3 = load_svg_text(check_svg, 96, 96, tint)
    write_png(String("/tmp/svg_check.png"), c3)
    _report(String("check"), c3)

    # 4) currentColor uses the tint; group transform translate
    var cc_svg = String("<svg viewBox='0 0 24 24'><g transform='translate(2 2)'><rect x='0' y='0' width='8' height='8' fill='currentColor'/></g></svg>")
    var c4 = load_svg_text(cc_svg, 96, 96, rgb(200, 30, 30))
    write_png(String("/tmp/svg_currentcolor.png"), c4)
    _report(String("currentcolor"), c4)
    # verify the tint actually landed (sample a pixel inside the translated rect)
    # rect (2..10, 2..10) in viewBox -> *4 scale -> ~ (8..40) device; sample (20,20)
    var i = (20 * c4.w + 20) * 4
    print("SAMPLE currentcolor rgba", Int(c4.px[i]), Int(c4.px[i + 1]), Int(c4.px[i + 2]), Int(c4.px[i + 3]))

    print("render: done")
