# svg — load SVG icons in pure Mojo (subset → raster)

Parse an **SVG icon** and rasterize it onto a `graphics.Canvas` (RGBA8), ready to
save as PNG or upload as a GPU texture. Pure Mojo, no FFI — it sits on top of the
`graphics` vector engine (`Path` fill/stroke, `Affine` transforms). This is an
**icon-subset** loader, not a full SVG renderer (see *Scope* below).

## Modules

| Module | What it is |
|---|---|
| `pathdata.mojo` | `parse_path_d(d)` → `List[SvgSeg]` (absolute M/L/C/Q/Z). Handles `M m L l H h V v C c S s Q q T t A a Z z`, relative/absolute, implicit command repetition, smooth-control reflection (S/T), and elliptical-arc→cubic conversion. `build_path(segs)` emits into a `graphics.Path`. |
| `shapes.mojo` | `rect_segs` (incl. `rx`/`ry` rounding), `circle_segs`, `ellipse_segs`, `line_segs`, `poly_segs` → `List[SvgSeg]`. |
| `xml.mojo` | `parse_xml(text)` → `XmlNode` tree (name + attrs + children). Skips prolog/comments/DOCTYPE/PIs and text; `find_all(node, name, out)` collects descendants. |
| `style.mojo` | `parse_color` (`#rgb`/`#rrggbb`/`rgb()`/named/`currentColor`/`none`), `parse_transform` (translate/scale/rotate/matrix list), `parse_viewbox`, `parse_opacity`. |
| `loader.mojo` | `load_svg_text` / `load_svg_file` → `Canvas`. Walks the tree, resolves inherited fill/stroke/stroke-width/opacity (presentation attrs **and** inline `style=""`) and transforms (incl. viewBox `xMidYMid meet` fit), then even-odd `fill_path_aa` + `stroke_path` on a transparent canvas. `to_rgba_list(canvas)` copies pixels for texture upload. |

## Example

```mojo
from graphics.color import rgb
from graphics.png import write_png
from svg.loader import load_svg_file, load_svg_text, to_rgba_list

# from a file, tinted (currentColor + unsupported-paint fallback)
var canvas = load_svg_file(String("icons/check.svg"), 64, 64, rgb(230, 230, 235))
write_png(String("/tmp/check.png"), canvas)

# ...or upload to a MojoUI GPU texture:
#   var px  = to_rgba_list(canvas)
#   var tex = Backend.make_texture_rgba(Int32(canvas.w), Int32(canvas.h), px)
```

## Verified (measured — not asserted)

```bash
# from a dir with the Mojo toolchain (e.g. MojoUI's pixi env), -I this repo:
pixi run mojo run -I . svg/tests/pathdata_test.mojo   # 26/26 exact (commands, rel/abs, S/T reflect, arc endpoints)
pixi run mojo run -I . svg/tests/shapes_test.mojo     # 15/15 exact (rect/rounded/circle/poly geometry)
pixi run mojo run -I . svg/tests/xml_test.mojo        # 7/7 (tree, attrs, find_all)
pixi run mojo run -I . svg/tests/render_test.mojo     # renders 4 icons -> /tmp/*.png + coverage report
python3 svg/tests/oracle.py                           # PIL cross-check on the PNGs
```

Independent **PIL oracle** on the rendered PNGs: filled rect is pixel-exact
(1600 px, exact bbox); a filled circle is within **0.3%** of a true filled
ellipse of the same radius; `currentColor` lands the tint exactly and a
group `transform` + `viewBox` fit place it at the exact bbox; a stroked check
mark renders with round caps/joins.

## Scope / limitations (honest)

**Icon subset, not full SVG.** Supported: `<path>`, `<rect>`/`<circle>`/
`<ellipse>`/`<line>`/`<polyline>`/`<polygon>`, `<g>`, nested `transform`,
`viewBox` (`xMidYMid meet`), fill/stroke/stroke-width/opacity from presentation
attributes and inline `style=""`, `currentColor`, even-odd fills (holes work).

**Not** supported (ignored, not errored): gradients & patterns (paints fall back
to `currentColor`), `<use>`/`<defs>` references, `clip-path`/`mask`/`filter`,
`<text>`, `preserveAspectRatio` other than the `xMidYMid meet` default,
`stroke-dasharray` (the underlying `graphics` engine supports dashes, but it is
not wired from SVG), and percent/ICC colors. Stroke width is uniform. Partial-
coverage edge alpha from `fill_path_aa` looks correct but is not separately
validated.
```
