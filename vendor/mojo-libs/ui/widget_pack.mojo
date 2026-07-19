# widget_pack.mojo — a Tier-2 IMMEDIATE-MODE widget pack for Mojo 1.0.0b1, drawn on the
# MojoGUI C rendering backend (librender.so) via external_call ONLY. No backend change.
#
# WORKSTREAM B. Each widget COMPOSES the MojoGUI primitives (set_color/draw_filled_rectangle/
# draw_rectangle/draw_line/draw_filled_circle/draw_text/get_text_size). Each takes EXPLICIT
# geometry (x,y,w,h) like core_widgets.mojo — the layout cursor is a SEPARATE workstream and is
# NOT a dependency here.
#
# FFI idioms mirrored EXACTLY from the working, framebuffer-verified exemplar
#   /home/alex/framepfx-mojo/src/keyframe_editor.mojo (to_cstr lines 37-43; col/fill/line_t/
#   fcircle/txt wrappers lines 46-63) and the sibling toolkit /tmp/mojoui/core_widgets.mojo
#   (UiContext with hot_id/active_id, set_input/begin_frame, press+release click semantics,
#   point_in_rect/clampf). This module uses the SAME UiContext concept and is designed to
#   interoperate with it: it carries the same mouse fields and adds the extra per-frame state
#   the Tier-2 widgets need (drag press-anchor, menu/tree open-state, disabled flag).
#
# INJECTABLE INPUT is mandatory: no widget calls get_mouse_* internally. The main loop drives
# everything through Ctx.set_input(mx,my,down) with synthetic input for headless verification.
#
# NO `from math` import: the windowed FFI binary cannot link transcendentals (sincos@GLIBC).
# Only builtin min/max/abs/Int()/Float32() arithmetic is used (same constraint as the exemplars).
# list_clipper floor/ceil are done with integer arithmetic, NOT math.floor (no libm pull-in).

from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from builtin.type_aliases import MutExternalOrigin


# ============================================================================
# C backend helpers  (mirror keyframe_editor.mojo:37-63 / core_widgets.mojo:32-82 exactly)
# ============================================================================
def to_cstr(s: String) -> UnsafePointer[Int8, MutExternalOrigin]:
    var b = s.as_bytes()
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(b[i])
    buf[len(b)] = 0
    return buf


def set_color(r: Float32, g: Float32, b: Float32, a: Float32):
    _ = external_call["set_color", Int32](r, g, b, a)


def col(r: Int, g: Int, b: Int):
    # 0..255 convenience wrapper (keyframe_editor.mojo:46-47)
    _ = external_call["set_color", Int32](
        Float32(r) / 255.0, Float32(g) / 255.0, Float32(b) / 255.0, Float32(1.0)
    )


def fill(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_filled_rectangle", Int32](x, y, w, h)


def rect(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_rectangle", Int32](x, y, w, h)


def line_t(x1: Float32, y1: Float32, x2: Float32, y2: Float32, thick: Float32):
    _ = external_call["draw_line", Int32](x1, y1, x2, y2, thick)


def fcircle(x: Float32, y: Float32, radius: Float32, segs: Int):
    _ = external_call["draw_filled_circle", Int32](x, y, radius, Int32(segs))


def txt(s: String, x: Float32, y: Float32, size: Float32):
    _ = external_call["draw_text", Int32](to_cstr(s), x, y, size)


# get_text_size(const char*, float size, float* outW, float* outH) -> (w, h) in pixels.
def measure_text(s: String, size: Float32) -> Tuple[Float32, Float32]:
    var wbuf = alloc[Float32](1)
    var hbuf = alloc[Float32](1)
    wbuf[0] = Float32(0.0)
    hbuf[0] = Float32(0.0)
    _ = external_call["get_text_size", Int32](to_cstr(s), size, wbuf, hbuf)
    var w = wbuf[0]
    var h = hbuf[0]
    return Tuple[Float32, Float32](w, h)


# ============================================================================
# geometry helpers (mirror core_widgets.mojo:88-97)
# ============================================================================
def point_in_rect(px: Float32, py: Float32, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
    return px >= x and px < x + w and py >= y and py < y + h


def clampf(v: Float32, lo: Float32, hi: Float32) -> Float32:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def clampi(v: Int, lo: Int, hi: Int) -> Int:
    return max(min(v, hi), lo)


# Format a Float32 to a String using ONLY String(Int) (the constructor the verified exemplars
# exercise — keyframe_editor/core_widgets only ever build String from Int or literals). String(Float32)
# is NOT exercised in either verified exemplar, so we avoid it here and format by hand: sign + integer
# part + '.' + 2 fixed decimal digits. No libm; pure integer arithmetic (Int() truncation only).
def f2s(value: Float32) raises -> String:
    var v = value
    var sign = String("")
    if v < Float32(0.0):
        sign = String("-")
        v = -v
    var ipart = Int(v)                       # truncated integer part (v >= 0 here)
    var frac = v - Float32(ipart)
    var hundredths = Int(frac * Float32(100.0) + Float32(0.5))   # round to 2 dp
    if hundredths >= 100:                    # rounding carried into the integer part
        ipart += 1
        hundredths -= 100
    var ds = String(hundredths)
    if hundredths < 10:                      # zero-pad to 2 digits
        ds = String("0") + ds
    return sign + String(ipart) + String(".") + ds


# ============================================================================
# Ctx — the immediate-mode core for the Tier-2 pack.
#
# It is the SAME UiContext concept as /tmp/mojoui/core_widgets.mojo (carries the per-frame mouse
# state + hot_id/active_id with press+release click semantics; injectable via set_input). It
# ADDS the small amount of extra per-frame state the Tier-2 widgets require:
#
#   press_x, press_y : Int   cursor position at the moment a drag widget latched active_id.
#                            RELATIVE drag deltas are measured from here (drag_float/drag_int).
#   open_menu        : Int   id of the currently-open vertical menu (or -1).
#   disabled         : Bool  set by begin_disabled(); while True every widget draws grayed and
#                            ignores clicks/drags. Restored by end_disabled().
#
# Widget ids are EXPLICIT (passed per call-site) — never hashed from labels (matches core_widgets).
# ============================================================================
struct Ctx(Copyable, Movable):
    var mouse_x: Int
    var mouse_y: Int
    var mouse_down: Bool
    var prev_mouse_down: Bool
    var mouse_clicked: Bool
    var mouse_released: Bool
    var hot_id: Int
    var active_id: Int
    var press_x: Int            # cursor x when active_id was latched (drag anchor)
    var press_y: Int            # cursor y when active_id was latched
    var open_menu: Int          # id of open vertical menu, or -1
    var disabled: Bool          # begin_disabled/end_disabled gate

    def __init__(out self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_down = False
        self.prev_mouse_down = False
        self.mouse_clicked = False
        self.mouse_released = False
        self.hot_id = -1
        self.active_id = -1
        self.press_x = 0
        self.press_y = 0
        self.open_menu = -1
        self.disabled = False

    # ---- INJECTABLE input (mandatory for headless verification) ----
    # Derives click/release edges from the previous frame's down-state, then advances
    # prev_mouse_down. Call once per simulated frame (identical to core_widgets.set_input).
    def set_input(mut self, mx: Int, my: Int, down: Bool):
        self.mouse_clicked = down and not self.prev_mouse_down
        self.mouse_released = self.prev_mouse_down and not down
        self.mouse_x = mx
        self.mouse_y = my
        self.mouse_down = down
        self.prev_mouse_down = down

    # ---- LIVE input from the backend (production path) ----
    def pump_live_input(mut self):
        var mx = Int(external_call["get_mouse_x", Int32]())
        var my = Int(external_call["get_mouse_y", Int32]())
        var bs = Int(external_call["get_mouse_button_state", Int32](Int32(0)))
        var down = bs != 0
        self.mouse_clicked = down and not self.prev_mouse_down
        self.mouse_released = self.prev_mouse_down and not down
        self.mouse_x = mx
        self.mouse_y = my
        self.mouse_down = down
        self.prev_mouse_down = down

    # ---- per-frame reset of hot tracking (active/open/disabled persist as appropriate) ----
    def begin_frame(mut self):
        self.hot_id = -1

    def _mouse_in(self, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
        return point_in_rect(Float32(self.mouse_x), Float32(self.mouse_y), x, y, w, h)


# ============================================================================
# 7) begin_disabled / end_disabled — gate clicks AND gray out drawing.
#    Implemented as free functions over Ctx (so they read like immediate-mode GUI's BeginDisabled()).
#    While ctx.disabled is True every widget below short-circuits its interaction and draws
#    with a desaturated palette.
# ============================================================================
def begin_disabled(mut ctx: Ctx):
    ctx.disabled = True


def end_disabled(mut ctx: Ctx):
    ctx.disabled = False


# ============================================================================
# 1a) drag_float — RELATIVE drag. value += (mouse_dx) * speed, clamped to [vmin, vmax].
#     The drag anchor (press_x) is recorded in ctx when active_id latches; on every held frame
#     the delta is measured from the CURRENT mouse_x to press_x, applied, and press_x advanced
#     so the next frame measures only the incremental motion (so cumulative == total dx*speed).
#     Returns True if value changed this frame. Honors ctx.disabled.
# ============================================================================
def drag_float(
    mut ctx: Ctx,
    id: Int,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    mut value: Float32,
    speed: Float32,
    vmin: Float32,
    vmax: Float32,
) raises -> Bool:
    var inside = ctx._mouse_in(x, y, w, h)
    var changed = False

    if not ctx.disabled:
        if inside:
            ctx.hot_id = id
        # press latches active + records the drag anchor.
        if inside and ctx.mouse_clicked:
            ctx.active_id = id
            ctx.press_x = ctx.mouse_x
            ctx.press_y = ctx.mouse_y
        # held: apply the incremental delta since the anchor. Advance the anchor ONLY when the
        # value actually changed (mirrors drag_int's no-loss policy): if the move was clamped at
        # vmin/vmax (nv == value) the anchor is left in place so accumulated travel is NOT dropped
        # and a reversing drag takes effect immediately. (Finding #4/#5 fix.)
        if ctx.active_id == id and ctx.mouse_down:
            var dx = Float32(ctx.mouse_x - ctx.press_x)
            if dx != Float32(0.0):
                var nv = clampf(value + dx * speed, vmin, vmax)
                if nv != value:
                    value = nv
                    changed = True
                    ctx.press_x = ctx.mouse_x      # consumed -> advance anchor (no double-count)
        if ctx.mouse_released and ctx.active_id == id:
            ctx.active_id = -1

    # ---- draw: field box + centered "value" text ----
    if ctx.disabled:
        col(40, 40, 44)
    elif ctx.active_id == id:
        col(70, 100, 150)
    elif ctx.hot_id == id:
        col(52, 52, 60)
    else:
        col(38, 38, 44)
    fill(x, y, w, h)
    col(0, 0, 0)
    rect(x, y, w, h)
    if ctx.disabled:
        col(110, 110, 116)
    else:
        col(225, 225, 230)
    var s = f2s(value)                         # hand-rolled Float32->String (no String(Float32))
    var m = measure_text(s, Float32(12.0))
    var tw = m[0]
    var th = m[1]
    txt(s, x + (w - tw) / Float32(2.0), y + (h - th) / Float32(2.0), Float32(12.0))
    return changed


# ============================================================================
# 1b) drag_int — same RELATIVE drag for an Int value. value += Int(mouse_dx * speed), clamped.
#     The Int truncation of (dx*speed) is applied to the anchor delta; the anchor only advances
#     when at least one whole unit was consumed, so sub-unit motion accumulates (no lost dx).
# ============================================================================
def drag_int(
    mut ctx: Ctx,
    id: Int,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    mut value: Int,
    speed: Float32,
    vmin: Int,
    vmax: Int,
) raises -> Bool:
    var inside = ctx._mouse_in(x, y, w, h)
    var changed = False

    if not ctx.disabled:
        if inside:
            ctx.hot_id = id
        if inside and ctx.mouse_clicked:
            ctx.active_id = id
            ctx.press_x = ctx.mouse_x
            ctx.press_y = ctx.mouse_y
        if ctx.active_id == id and ctx.mouse_down:
            var dxpx = ctx.mouse_x - ctx.press_x
            var delta = Int(Float32(dxpx) * speed)     # whole-unit step
            if delta != 0:
                var nv = clampi(value + delta, vmin, vmax)
                if nv != value:
                    value = nv
                    changed = True
                ctx.press_x = ctx.mouse_x               # consumed -> advance anchor
        if ctx.mouse_released and ctx.active_id == id:
            ctx.active_id = -1

    # ---- draw ----
    if ctx.disabled:
        col(40, 40, 44)
    elif ctx.active_id == id:
        col(70, 100, 150)
    elif ctx.hot_id == id:
        col(52, 52, 60)
    else:
        col(38, 38, 44)
    fill(x, y, w, h)
    col(0, 0, 0)
    rect(x, y, w, h)
    if ctx.disabled:
        col(110, 110, 116)
    else:
        col(225, 225, 230)
    var s = String(value)
    var m = measure_text(s, Float32(12.0))
    var tw = m[0]
    var th = m[1]
    txt(s, x + (w - tw) / Float32(2.0), y + (h - th) / Float32(2.0), Float32(12.0))
    return changed


# ============================================================================
# 1c) slider_int — ABSOLUTE mapping of mouse-x across [x, x+w] to [vmin, vmax] (Int).
#     Mirrors core_widgets.slider_float exactly but rounds to the nearest integer.
#     Returns True if value changed this frame. Honors ctx.disabled.
# ============================================================================
def slider_int(
    mut ctx: Ctx,
    id: Int,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    mut value: Int,
    vmin: Int,
    vmax: Int,
) raises -> Bool:
    var inside = ctx._mouse_in(x, y, w, h)
    var changed = False

    if not ctx.disabled:
        if inside:
            ctx.hot_id = id
        if inside and ctx.mouse_clicked:
            ctx.active_id = id
        if ctx.active_id == id and ctx.mouse_down:
            var t = (Float32(ctx.mouse_x) - x) / w      # 0..1 across the track
            t = clampf(t, Float32(0.0), Float32(1.0))
            var span = Float32(vmax - vmin)
            var nv = vmin + Int(t * span + Float32(0.5))    # round to nearest int
            nv = clampi(nv, vmin, vmax)
            if nv != value:
                value = nv
                changed = True
        if ctx.mouse_released and ctx.active_id == id:
            ctx.active_id = -1

    # ---- draw track + filled portion + handle ----
    if ctx.disabled:
        col(26, 26, 30)
    else:
        col(28, 28, 32)
    fill(x, y, w, h)
    col(0, 0, 0)
    rect(x, y, w, h)
    var denom = Float32(vmax - vmin)
    var frac = Float32(0.0)
    if denom != Float32(0.0):
        frac = Float32(value - vmin) / denom
    frac = clampf(frac, Float32(0.0), Float32(1.0))
    if ctx.disabled:
        col(50, 50, 56)
    else:
        col(60, 100, 150)
    fill(x, y, w * frac, h)
    var hw = Float32(8.0)
    var hx = x + frac * w - hw / Float32(2.0)
    hx = clampf(hx, x, x + w - hw)
    if ctx.disabled:
        col(90, 90, 96)
    elif ctx.active_id == id:
        col(160, 200, 255)
    else:
        col(200, 200, 210)
    fill(hx, y, hw, h)
    return changed


# ============================================================================
# 2a) tree_node — a clickable header with a caret triangle (made from draw_line) that toggles
#     `open` on a press+release click. Returns the (possibly toggled) open state. Indented body
#     is the CALLER's responsibility (draw children when this returns True).
#     `w` is the header width; height is fixed at 20px. Honors ctx.disabled.
# ============================================================================
def tree_node(
    mut ctx: Ctx,
    id: Int,
    label: String,
    x: Float32,
    y: Float32,
    w: Float32,
    mut open: Bool,
) raises -> Bool:
    var h = Float32(20.0)
    var inside = ctx._mouse_in(x, y, w, h)

    if not ctx.disabled:
        if inside:
            ctx.hot_id = id
        if inside and ctx.mouse_clicked:
            ctx.active_id = id
        if ctx.mouse_released and ctx.active_id == id:
            if inside:
                open = not open
            ctx.active_id = -1

    # ---- draw header background ----
    if ctx.disabled:
        col(30, 30, 34)
    elif ctx.hot_id == id:
        col(50, 50, 58)
    else:
        col(40, 40, 46)
    fill(x, y, w, h)

    # ---- caret triangle from draw_line (right-pointing when closed, down when open) ----
    var cx = x + Float32(8.0)
    var cy = y + h / Float32(2.0)
    if ctx.disabled:
        col(110, 110, 116)
    else:
        col(210, 210, 216)
    if open:
        # down-pointing caret: apex below, two arms up-left / up-right
        line_t(cx - Float32(4.0), cy - Float32(3.0), cx, cy + Float32(4.0), Float32(2.0))
        line_t(cx + Float32(4.0), cy - Float32(3.0), cx, cy + Float32(4.0), Float32(2.0))
    else:
        # right-pointing caret: apex to the right, two arms back-up / back-down
        line_t(cx - Float32(3.0), cy - Float32(4.0), cx + Float32(4.0), cy, Float32(2.0))
        line_t(cx - Float32(3.0), cy + Float32(4.0), cx + Float32(4.0), cy, Float32(2.0))

    # ---- label after the caret ----
    if ctx.disabled:
        col(120, 120, 126)
    else:
        col(225, 225, 230)
    txt(label, x + Float32(18.0), y + Float32(4.0), Float32(12.0))
    return open


# ============================================================================
# 2b) collapsing_header — identical interaction/visual to tree_node but conventionally drawn
#     WITHOUT child indentation (the caller draws children at the SAME x). Same signature.
# ============================================================================
def collapsing_header(
    mut ctx: Ctx,
    id: Int,
    label: String,
    x: Float32,
    y: Float32,
    w: Float32,
    mut open: Bool,
) raises -> Bool:
    # Reuse tree_node's behaviour verbatim; the only difference is the documented caller contract
    # (no indent for children). Slightly bolder bg to read as a section header.
    var h = Float32(22.0)
    var inside = ctx._mouse_in(x, y, w, h)

    if not ctx.disabled:
        if inside:
            ctx.hot_id = id
        if inside and ctx.mouse_clicked:
            ctx.active_id = id
        if ctx.mouse_released and ctx.active_id == id:
            if inside:
                open = not open
            ctx.active_id = -1

    if ctx.disabled:
        col(34, 34, 38)
    elif ctx.hot_id == id:
        col(58, 58, 66)
    else:
        col(48, 48, 54)
    fill(x, y, w, h)
    col(0, 0, 0)
    rect(x, y, w, h)

    var cx = x + Float32(8.0)
    var cy = y + h / Float32(2.0)
    if ctx.disabled:
        col(110, 110, 116)
    else:
        col(220, 220, 226)
    if open:
        line_t(cx - Float32(4.0), cy - Float32(3.0), cx, cy + Float32(4.0), Float32(2.0))
        line_t(cx + Float32(4.0), cy - Float32(3.0), cx, cy + Float32(4.0), Float32(2.0))
    else:
        line_t(cx - Float32(3.0), cy - Float32(4.0), cx + Float32(4.0), cy, Float32(2.0))
        line_t(cx - Float32(3.0), cy + Float32(4.0), cx + Float32(4.0), cy, Float32(2.0))

    if ctx.disabled:
        col(120, 120, 126)
    else:
        col(230, 230, 235)
    txt(label, x + Float32(18.0), y + Float32(5.0), Float32(12.0))
    return open


# ============================================================================
# 3a) begin_menu_bar — draws the horizontal bar background. Pure visual; returns nothing.
# ============================================================================
def begin_menu_bar(x: Float32, y: Float32, w: Float32, h: Float32):
    col(36, 36, 42)
    fill(x, y, w, h)
    col(0, 0, 0)
    rect(x, y, w, h)


# ============================================================================
# 3b) menu_item — a clickable item (in a bar OR inside an open menu). Returns True the frame it
#     is clicked (press+release inside). Honors ctx.disabled.
# ============================================================================
def menu_item(
    mut ctx: Ctx,
    id: Int,
    label: String,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
) raises -> Bool:
    var inside = ctx._mouse_in(x, y, w, h)
    var clicked = False

    if not ctx.disabled:
        if inside:
            ctx.hot_id = id
        if inside and ctx.mouse_clicked:
            ctx.active_id = id
        if ctx.mouse_released and ctx.active_id == id:
            if inside:
                clicked = True
            ctx.active_id = -1

    # ---- draw ----
    if ctx.disabled:
        col(34, 34, 38)
    elif ctx.hot_id == id:
        col(70, 110, 165)
    else:
        col(44, 44, 50)
    fill(x, y, w, h)
    if ctx.disabled:
        col(120, 120, 126)
    else:
        col(225, 225, 230)
    txt(label, x + Float32(8.0), y + (h - Float32(12.0)) / Float32(2.0), Float32(12.0))
    return clicked


# ============================================================================
# 3c) menu — a bar title that toggles a vertical list of items open/closed (open-state in
#     ctx.open_menu). Clicking the title toggles; with the menu open, clicking an item returns
#     its index and closes the menu. Returns the clicked item index, or -1 if none this frame.
#     `items` are drawn as a vertical stack of `row_h` rows below the title.
# ============================================================================
def menu(
    mut ctx: Ctx,
    id: Int,
    title: String,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    items: List[String],
) raises -> Int:
    var inside_title = ctx._mouse_in(x, y, w, h)
    var result = -1

    if not ctx.disabled:
        if inside_title:
            ctx.hot_id = id
        # toggle open on a click of the title
        if inside_title and ctx.mouse_clicked:
            if ctx.open_menu == id:
                ctx.open_menu = -1
            else:
                ctx.open_menu = id

    # ---- draw the title ----
    if ctx.disabled:
        col(34, 34, 38)
    elif ctx.open_menu == id:
        col(70, 110, 165)
    elif ctx.hot_id == id:
        col(56, 56, 64)
    else:
        col(36, 36, 42)
    fill(x, y, w, h)
    if ctx.disabled:
        col(120, 120, 126)
    else:
        col(225, 225, 230)
    txt(title, x + Float32(8.0), y + (h - Float32(12.0)) / Float32(2.0), Float32(12.0))

    # ---- if open, draw the vertical item list and hit-test ----
    if ctx.open_menu == id and not ctx.disabled:
        var row_h = h
        var n = len(items)
        for i in range(n):
            var iy = y + h + Float32(i) * row_h
            var item_id = id * 1000 + i + 1            # stable per-item id
            var inside_item = ctx._mouse_in(x, iy, w, row_h)
            if inside_item:
                ctx.hot_id = item_id
            if inside_item and ctx.mouse_clicked:
                ctx.active_id = item_id
            if ctx.mouse_released and ctx.active_id == item_id:
                if inside_item:
                    result = i
                ctx.active_id = -1
                ctx.open_menu = -1                     # selection closes the menu
            # draw item row
            if inside_item:
                col(70, 110, 165)
            else:
                col(44, 44, 50)
            fill(x, iy, w, row_h)
            col(0, 0, 0)
            rect(x, iy, w, row_h)
            col(225, 225, 230)
            txt(items[i].copy(), x + Float32(8.0), iy + (row_h - Float32(12.0)) / Float32(2.0), Float32(12.0))
    return result


# ============================================================================
# 4) tab — draws one tab; the active tab is brighter. Returns True if clicked this frame
#    (press+release inside). Caller supplies an explicit width `w` (>0) like every other widget
#    here; there is no w<=0 auto-fit path. Honors ctx.disabled.
# ============================================================================
def tab(
    mut ctx: Ctx,
    id: Int,
    label: String,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    active: Bool,
) raises -> Bool:
    var inside = ctx._mouse_in(x, y, w, h)
    var clicked = False

    if not ctx.disabled:
        if inside:
            ctx.hot_id = id
        if inside and ctx.mouse_clicked:
            ctx.active_id = id
        if ctx.mouse_released and ctx.active_id == id:
            if inside:
                clicked = True
            ctx.active_id = -1

    # ---- draw: active tab brighter; hovered lifts a touch ----
    if ctx.disabled:
        col(34, 34, 38)
    elif active:
        col(64, 96, 150)
    elif ctx.hot_id == id:
        col(56, 56, 64)
    else:
        col(40, 40, 46)
    fill(x, y, w, h)
    col(0, 0, 0)
    rect(x, y, w, h)
    if ctx.disabled:
        col(120, 120, 126)
    elif active:
        col(240, 240, 245)
    else:
        col(200, 200, 208)
    var m = measure_text(label, Float32(12.0))
    var tw = m[0]
    var th = m[1]
    txt(label, x + (w - tw) / Float32(2.0), y + (h - th) / Float32(2.0), Float32(12.0))
    return clicked


# ============================================================================
# 5) progress_bar — filled portion proportional to `fraction` (clamped 0..1) + centered % text.
#    Pure visual (no interaction). fraction expressed 0..1.
# ============================================================================
def progress_bar(fraction: Float32, x: Float32, y: Float32, w: Float32, h: Float32) raises:
    var f = clampf(fraction, Float32(0.0), Float32(1.0))
    # track
    col(28, 28, 32)
    fill(x, y, w, h)
    # filled portion
    col(60, 140, 90)
    fill(x, y, w * f, h)
    # border
    col(0, 0, 0)
    rect(x, y, w, h)
    # centered "NN%" text (integer percent; no libm — Int() truncation of f*100 + 0.5 round)
    var pct = Int(f * Float32(100.0) + Float32(0.5))
    var s = String(pct) + String("%")
    var m = measure_text(s, Float32(12.0))
    var tw = m[0]
    var th = m[1]
    col(235, 235, 240)
    txt(s, x + (w - tw) / Float32(2.0), y + (h - th) / Float32(2.0), Float32(12.0))


# ============================================================================
# 6) color_button — a filled swatch (r,g,b in 0..1) + border. Returns True when clicked
#    (press+release inside). Honors ctx.disabled (grays the swatch + ignores clicks).
# ============================================================================
def color_button(
    mut ctx: Ctx,
    id: Int,
    r: Float32,
    g: Float32,
    b: Float32,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
) raises -> Bool:
    var inside = ctx._mouse_in(x, y, w, h)
    var clicked = False

    if not ctx.disabled:
        if inside:
            ctx.hot_id = id
        if inside and ctx.mouse_clicked:
            ctx.active_id = id
        if ctx.mouse_released and ctx.active_id == id:
            if inside:
                clicked = True
            ctx.active_id = -1

    # ---- draw swatch ----
    if ctx.disabled:
        # desaturate toward gray: average the channel and lerp half-way
        var avg = (r + g + b) / Float32(3.0)
        set_color(
            (r + avg) / Float32(2.0),
            (g + avg) / Float32(2.0),
            (b + avg) / Float32(2.0),
            Float32(1.0),
        )
    else:
        set_color(r, g, b, Float32(1.0))
    fill(x, y, w, h)
    # border (brighter when hot for affordance)
    if ctx.hot_id == id and not ctx.disabled:
        col(240, 240, 245)
    else:
        col(0, 0, 0)
    rect(x, y, w, h)
    return clicked


# ============================================================================
# 8a) bullet_text — a small filled dot + a label to its right (no interaction).
# ============================================================================
def bullet_text(label: String, x: Float32, y: Float32) raises:
    col(180, 180, 190)
    fcircle(x + Float32(4.0), y + Float32(8.0), Float32(3.0), 12)
    col(220, 220, 226)
    txt(label, x + Float32(14.0), y + Float32(2.0), Float32(12.0))


# ============================================================================
# 8b) separator — a 1px horizontal divider line spanning width `w`.
# ============================================================================
def separator(x: Float32, y: Float32, w: Float32):
    col(70, 70, 80)
    line_t(x, y, x + w, y, Float32(1.0))


# ============================================================================
# 9) list_clipper — virtualization helper for long lists. Given the total item count, the row
#    height, the viewport's top y, its height, and the current scroll offset (px from the very
#    top of the virtual content), returns the INCLUSIVE [first, last] row index range that
#    intersects the viewport so the caller draws ONLY those rows.
#
#    Derivation (integer floor, NO libm):
#      The viewport covers virtual y in [scroll, scroll + view_h].
#      first = floor(scroll / row_height)
#      last  = floor((scroll + view_h) / row_height)   (the row whose top is at/just below the
#              viewport bottom is included so partial bottom rows are drawn).
#      Both clamped to [0, total_items-1]. If total_items==0 -> (0, -1) meaning "draw nothing".
#    Returns Tuple[Int, Int] = (first, last). view_y is accepted for API completeness (the caller
#    positions row i at view_y + i*row_height - scroll); it does not affect the index math.
# ============================================================================
def list_clipper(
    total_items: Int,
    row_height: Int,
    view_y: Int,
    view_h: Int,
    scroll: Int,
) -> Tuple[Int, Int]:
    if total_items <= 0 or row_height <= 0:
        return Tuple[Int, Int](0, -1)            # nothing to draw
    var s = scroll
    if s < 0:
        s = 0
    var first = s // row_height                  # integer floor (s >= 0)
    var last = (s + view_h) // row_height         # floor of bottom edge
    if first < 0:
        first = 0
    if first > total_items - 1:
        first = total_items - 1
    if last > total_items - 1:
        last = total_items - 1
    if last < first:
        last = first
    return Tuple[Int, Int](first, last)


# ============================================================================
# SELF-TEST  (compiled + run headlessly by the MAIN LOOP — do NOT run it yourself)
#
# Press+release widgets need two frames: frame 1 = down (latch active), frame 2 = up (fire).
# Drag widgets need: frame 1 = press (latch active + record anchor), frame 2..N = move (apply
# delta from anchor). Every widget issues draw_* calls so the framebuffer readback sees pixels.
#
# Each expected number below is derived from first principles in the comment beside it.
# ============================================================================
def main() raises:
    var ctx = Ctx()
    var pass_all = True

    # ----- 1a) DRAG_FLOAT: value 10.0, speed 0.5; press @x=100, drag to x=130 (dx=30). -----
    #   delta = dx * speed = 30 * 0.5 = 15.0 ; value = 10.0 + 15.0 = 25.0
    #   field rect = (100, 100, 120, 24).
    var fval = Float32(10.0)
    # frame 1: press inside @ x=100 (anchor press_x=100), no motion yet.
    ctx.begin_frame()
    ctx.set_input(100, 112, True)
    _ = drag_float(ctx, 1, Float32(100.0), Float32(100.0), Float32(120.0), Float32(24.0),
                   fval, Float32(0.5), Float32(0.0), Float32(1000.0))
    # frame 2: still held, move to x=130 -> dx=30 -> +15.0.
    ctx.begin_frame()
    ctx.set_input(130, 112, True)
    _ = drag_float(ctx, 1, Float32(100.0), Float32(100.0), Float32(120.0), Float32(24.0),
                   fval, Float32(0.5), Float32(0.0), Float32(1000.0))
    print("WIDGETPACK_SELFTEST: drag_float_value=", fval)        # EXPECT 25.0
    if fval != Float32(25.0):
        pass_all = False
    # release to clear capture
    ctx.begin_frame()
    ctx.set_input(130, 112, False)
    _ = drag_float(ctx, 1, Float32(100.0), Float32(100.0), Float32(120.0), Float32(24.0),
                   fval, Float32(0.5), Float32(0.0), Float32(1000.0))

    # ----- 1b) DRAG_INT: value 0, speed 1.0; press @x=200, drag to x=250 (dx=50). -----
    #   delta = Int(dx * speed) = Int(50 * 1.0) = 50 ; value = 0 + 50 = 50
    var ival = 0
    ctx.begin_frame()
    ctx.set_input(200, 112, True)
    _ = drag_int(ctx, 2, Float32(180.0), Float32(100.0), Float32(120.0), Float32(24.0),
                 ival, Float32(1.0), 0, 1000)
    ctx.begin_frame()
    ctx.set_input(250, 112, True)
    _ = drag_int(ctx, 2, Float32(180.0), Float32(100.0), Float32(120.0), Float32(24.0),
                 ival, Float32(1.0), 0, 1000)
    print("WIDGETPACK_SELFTEST: drag_int_value=", ival)          # EXPECT 50
    if ival != 50:
        pass_all = False
    ctx.begin_frame()
    ctx.set_input(250, 112, False)
    _ = drag_int(ctx, 2, Float32(180.0), Float32(100.0), Float32(120.0), Float32(24.0),
                 ival, Float32(1.0), 0, 1000)

    # ----- 1c) SLIDER_INT: track (100,200,200,24), range [0,100]; mouse_x=200 -> t=0.5 -> 50 -----
    #   t = (200-100)/200 = 0.5 ; value = 0 + round(0.5 * 100) = 50
    var slival = 0
    ctx.begin_frame()
    ctx.set_input(200, 212, True)        # press inside @ exactly mid-track
    _ = slider_int(ctx, 3, Float32(100.0), Float32(200.0), Float32(200.0), Float32(24.0),
                   slival, 0, 100)
    print("WIDGETPACK_SELFTEST: slider_int_value=", slival)      # EXPECT 50
    if slival != 50:
        pass_all = False
    ctx.begin_frame()
    ctx.set_input(200, 212, False)
    _ = slider_int(ctx, 3, Float32(100.0), Float32(200.0), Float32(200.0), Float32(24.0),
                   slival, 0, 100)

    # ----- 2) TREE_NODE: toggle twice. open starts False. -----
    #   header (100, 260, 200, 20). center inside @ (140, 270).
    #   click1 (down+up) -> True ; click2 (down+up) -> False.
    var topen = False
    # click 1
    ctx.begin_frame()
    ctx.set_input(140, 270, True)
    _ = tree_node(ctx, 4, String("Transform"), Float32(100.0), Float32(260.0), Float32(200.0), topen)
    ctx.begin_frame()
    ctx.set_input(140, 270, False)
    _ = tree_node(ctx, 4, String("Transform"), Float32(100.0), Float32(260.0), Float32(200.0), topen)
    print("WIDGETPACK_SELFTEST: tree_open_after_1=", topen)      # EXPECT True
    if topen != True:
        pass_all = False
    # click 2
    ctx.begin_frame()
    ctx.set_input(140, 270, True)
    _ = tree_node(ctx, 4, String("Transform"), Float32(100.0), Float32(260.0), Float32(200.0), topen)
    ctx.begin_frame()
    ctx.set_input(140, 270, False)
    _ = tree_node(ctx, 4, String("Transform"), Float32(100.0), Float32(260.0), Float32(200.0), topen)
    print("WIDGETPACK_SELFTEST: tree_open_after_2=", topen)      # EXPECT False
    if topen != False:
        pass_all = False

    # ----- 3) MENU_ITEM: press+release inside -> clicked True. -----
    #   item rect (100, 300, 120, 22). center inside @ (160, 311).
    ctx.begin_frame()
    ctx.set_input(160, 311, True)
    var mi1 = menu_item(ctx, 5, String("Open..."), Float32(100.0), Float32(300.0), Float32(120.0), Float32(22.0))
    ctx.begin_frame()
    ctx.set_input(160, 311, False)
    var mi2 = menu_item(ctx, 5, String("Open..."), Float32(100.0), Float32(300.0), Float32(120.0), Float32(22.0))
    print("WIDGETPACK_SELFTEST: menu_item_clicked=", mi2)        # EXPECT True
    if mi1 != False:
        pass_all = False
    if mi2 != True:
        pass_all = False

    # ----- 5) PROGRESS_BAR(0.5): draws a half-filled bar + "50%". (visual; no assert value) -----
    progress_bar(Float32(0.5), Float32(100.0), Float32(340.0), Float32(200.0), Float32(20.0))
    print("WIDGETPACK_SELFTEST: progress_bar_drawn=", True)      # EXPECT True (draw issued)

    # ----- RENDER-GATE COVERAGE: every remaining VISUAL widget must issue real draw_* calls so the
    # glReadPixels framebuffer readback exercises their paint paths. Mouse is parked far away
    # (5000,5000) so NONE of these interact (no hit, no active capture) — pure draw coverage.
    # (Finding #9: collapsing_header / tab / color_button / bullet_text / separator + menu were
    # implemented but never drawn in main(), so the render gate could not see them.)
    ctx.begin_frame()
    ctx.set_input(5000, 5000, False)
    # collapsing_header (drawn open=True so the down-caret arms are exercised too)
    var chopen = True
    _ = collapsing_header(ctx, 90, String("Section"), Float32(360.0), Float32(100.0), Float32(180.0), chopen)
    # tab (active + inactive both painted)
    _ = tab(ctx, 91, String("Scene"), Float32(360.0), Float32(130.0), Float32(70.0), Float32(24.0), True)
    _ = tab(ctx, 92, String("Render"), Float32(432.0), Float32(130.0), Float32(70.0), Float32(24.0), False)
    # color_button (red swatch + border)
    var cb_clicked = color_button(ctx, 93, Float32(0.85), Float32(0.20), Float32(0.20),
                                  Float32(360.0), Float32(160.0), Float32(28.0), Float32(28.0))
    if cb_clicked != False:                  # parked far away -> never clicked
        pass_all = False
    # bullet_text + separator
    bullet_text(String("First bullet"), Float32(360.0), Float32(196.0))
    separator(Float32(360.0), Float32(214.0), Float32(180.0))
    # menu drawn OPEN (forces the vertical item list + items[i].copy() loop to paint).
    ctx.open_menu = 94
    var mitems = List[String]()
    mitems.append(String("New"))
    mitems.append(String("Open"))
    mitems.append(String("Save"))
    _ = menu(ctx, 94, String("File"), Float32(360.0), Float32(224.0), Float32(120.0), Float32(22.0), mitems)
    ctx.open_menu = -1                       # restore (no leakage into later blocks)
    print("WIDGETPACK_SELFTEST: render_coverage_drawn=", True)   # EXPECT True (draws issued)

    # ----- 7) DISABLED BUTTON: a menu_item under begin_disabled must NOT click. -----
    #   Same press+release sequence as the menu_item above, but disabled -> clicked stays False.
    begin_disabled(ctx)
    ctx.begin_frame()
    ctx.set_input(160, 411, True)
    var db1 = menu_item(ctx, 6, String("Disabled"), Float32(100.0), Float32(400.0), Float32(120.0), Float32(22.0))
    ctx.begin_frame()
    ctx.set_input(160, 411, False)
    var db2 = menu_item(ctx, 6, String("Disabled"), Float32(100.0), Float32(400.0), Float32(120.0), Float32(22.0))
    end_disabled(ctx)
    print("WIDGETPACK_SELFTEST: disabled_button_clicked=", db2)  # EXPECT False
    if db2 != False:
        pass_all = False

    # ----- 9) LIST_CLIPPER: 1000 items, row=20, view_y=0, view_h=200, scroll=400. -----
    #   first = floor(400 / 20) = 20
    #   last  = floor((400 + 200) / 20) = floor(600 / 20) = 30
    var clip = list_clipper(1000, 20, 0, 200, 400)
    var first = clip[0]
    var last = clip[1]
    print("WIDGETPACK_SELFTEST: clip_first=", first)             # EXPECT 20
    print("WIDGETPACK_SELFTEST: clip_last=", last)               # EXPECT 30
    if first != 20:
        pass_all = False
    if last != 30:
        pass_all = False

    # ----- summary line the main loop gates on -----
    if pass_all:
        print("WIDGETPACK_SELFTEST: pass (drag_float=25.0, drag_int=50, slider_int=50, "
              + "tree True->False, menu_item True, disabled False, clip 20..30)")
    else:
        print("WIDGETPACK_SELFTEST: fail")
