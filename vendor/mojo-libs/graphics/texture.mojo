# graphics.texture — bridge a Canvas to a GPU texture upload.
#
# A `Canvas` is already row-major RGBA8, which is exactly what a texture upload
# wants. `to_rgba_list` copies it into a `List[UInt8]` — the format MojoUI's
# `Backend.make_texture_rgba(w, h, rgba_pixels)` (and most GPU upload paths)
# consume. So the workflow is:
#
#     var cv = Canvas(w, h)
#     bar_chart(cv, ...)                       # draw with graphics primitives
#     var px = to_rgba_list(cv)
#     var tex = Backend.make_texture_rgba(Int32(w), Int32(h), px)   # MojoUI
#     Backend.draw_image_rect(Rect(x, y, w, h), tex, white)         # MojoUI
#
# i.e. render charts / custom 2D into a Canvas on the CPU, then show it as a
# textured quad inside a MojoUI (sokol/GPU) frame.

from graphics.canvas import Canvas


def to_rgba_list(c: Canvas) -> List[UInt8]:
    """Row-major RGBA8 copy of the canvas (length = w*h*4). Ready to hand to a
    GPU texture upload such as MojoUI Backend.make_texture_rgba."""
    var n = c.w * c.h * 4
    var out = List[UInt8]()
    out.reserve(n if n > 0 else 1)
    for i in range(n):
        out.append(c.px[i])
    return out^
