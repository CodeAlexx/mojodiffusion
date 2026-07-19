# widget_variants.mojo — WORKSTREAM: WIDGET VARIANTS, an IMMEDIATE-MODE pack for Mojo 1.0.0b1,
# drawn on the MojoGUI C rendering backend (librender.so) via external_call ONLY. No backend change.
#
# Each widget COMPOSES the MojoGUI primitives (set_color/draw_filled_rectangle/draw_rectangle/
# draw_line/draw_filled_circle/draw_filled_triangle/draw_text/get_text_size/set_clip_rect/
# clear_clip_rect). Each takes EXPLICIT geometry (x,y,w[,h]) like /tmp/mojoui/core_widgets.mojo and
# /tmp/mojoui2/widget_pack.mojo — the layout cursor is a SEPARATE workstream, not a dependency here.
#
# FFI idioms mirrored EXACTLY from the working, framebuffer-verified exemplars:
#   /home/alex/framepfx-mojo/src/keyframe_editor.mojo (to_cstr 37-43; col/fill/line_t/fcircle/txt
#   46-63), /tmp/mojoui/core_widgets.mojo (UiContext input model: mouse_x/mouse_y:Int,
#   mouse_down/prev_mouse_down/mouse_clicked/mouse_released:Bool, hot_id/active_id:Int, set_input
#   edge derivation, point_in_rect/clampf, slider_float ABSOLUTE mouse-x->value mapping), and
#   /tmp/mojoui2/widget_pack.mojo (Ctx shape with press_x/press_y drag anchor + open state,
#   drag_float RELATIVE math, f2s hand-rolled Float32->String, list rows + clip pattern).
#
# This module uses the SAME UiContext concept (struct VCtx) carrying the standard mouse fields +
# hot_id/active_id + a drag anchor + an editing buffer for the +/- text fields. Widget ids are
# EXPLICIT (passed per call-site) — never hashed from labels (matches the exemplars).
#
# INJECTABLE INPUT is mandatory: the GATED logic path of every widget reads mouse + typed chars
# from VCtx fields (mouse_*, char_queue), NEVER from get_mouse_*/get_char internally. A separate
# pump_live_input()/pump_live_chars() does the live binding for production. The main loop drives
# everything with synthetic input for headless verification.
#
# NO `from math` import: the windowed FFI binary cannot link transcendentals (sincos@GLIBC). Only
# builtin min/max/abs/Int()/Float32() arithmetic is used. The arrow_button triangles use
# draw_filled_triangle (no trig). slider_angle uses a comptime DEG_PER_RAD constant (a plain ratio,
# NOT a trig call). No libm pull-in anywhere.

from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from builtin.type_aliases import MutExternalOrigin


# ============================================================================
# C backend helpers  (mirror keyframe_editor.mojo:37-63 / widget_pack.mojo:32-96 EXACTLY)
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


def ftri(x1: Float32, y1: Float32, x2: Float32, y2: Float32, x3: Float32, y3: Float32):
    _ = external_call["draw_filled_triangle", Int32](x1, y1, x2, y2, x3, y3)


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


def push_clip(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["set_clip_rect", Int32](Int32(x), Int32(y), Int32(w), Int32(h))


def pop_clip():
    _ = external_call["clear_clip_rect", Int32]()


# ============================================================================
# geometry helpers (mirror core_widgets.mojo:88-97 / widget_pack.mojo:87-100)
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


# Hand-rolled Float32 -> String, sign + integer part + '.' + 2 fixed decimals (NO String(Float32),
# which neither verified exemplar exercises). Pure integer arithmetic (Int() truncation). Mirrors
# widget_pack.mojo:107-122 f2s EXACTLY.
def f2s(value: Float32) raises -> String:
    var v = value
    var sign = String("")
    if v < Float32(0.0):
        sign = String("-")
        v = -v
    var ipart = Int(v)
    var frac = v - Float32(ipart)
    var hundredths = Int(frac * Float32(100.0) + Float32(0.5))
    if hundredths >= 100:
        ipart += 1
        hundredths -= 100
    var ds = String(hundredths)
    if hundredths < 10:
        ds = String("0") + ds
    return sign + String(ipart) + String(".") + ds


# ============================================================================
# VCtx — the immediate-mode core for the widget-variants pack.
#
# Same UiContext concept as core_widgets.mojo / widget_pack.mojo: per-frame mouse state +
# hot_id/active_id with press+release click semantics, injectable via set_input. It ADDS:
#
#   press_x, press_y : Int        cursor at the moment a drag widget latched active_id (drag anchor).
#   edit_id          : Int        id of the text field CURRENTLY being edited via typed chars, or -1.
#   edit_buf         : List[UInt8] the in-progress digit/char buffer for that field.
#   char_queue       : List[Int]  INJECTED Unicode codepoints to consume this frame (set_chars()).
#
# char input is INJECTABLE: gated logic reads ctx.char_queue (filled by set_chars()), never
# get_char(). pump_live_chars() drains the backend's get_char() into char_queue for production.
#
# Widget ids are EXPLICIT (passed per call-site) — never hashed from labels.
# ============================================================================
struct VCtx(Copyable, Movable):
    var mouse_x: Int
    var mouse_y: Int
    var mouse_down: Bool
    var prev_mouse_down: Bool
    var mouse_clicked: Bool
    var mouse_released: Bool
    var hot_id: Int
    var active_id: Int
    var press_x: Int
    var press_y: Int
    var edit_id: Int
    var edit_buf: List[UInt8]
    var char_queue: List[Int]

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
        self.edit_id = -1
        self.edit_buf = List[UInt8]()
        self.char_queue = List[Int]()

    # ---- INJECTABLE mouse input (mandatory for headless verification) ----
    # Derives click/release edges from the previous frame's down-state, then advances
    # prev_mouse_down. Mirrors core_widgets.set_input / widget_pack.set_input EXACTLY.
    def set_input(mut self, mx: Int, my: Int, down: Bool):
        self.mouse_clicked = down and not self.prev_mouse_down
        self.mouse_released = self.prev_mouse_down and not down
        self.mouse_x = mx
        self.mouse_y = my
        self.mouse_down = down
        self.prev_mouse_down = down

    # ---- INJECTABLE char input (mandatory for headless verification) ----
    # Replace the per-frame typed-codepoint queue with a synthetic list. Gated widget logic
    # consumes from this queue; never calls get_char() in the gated path.
    def set_chars(mut self, cps: List[Int]):
        self.char_queue = cps.copy()

    def clear_chars(mut self):
        self.char_queue = List[Int]()

    # ---- LIVE input from the backend (production path; NOT used by gated logic) ----
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

    def pump_live_chars(mut self):
        # Drain the backend's typed-codepoint queue (get_char() returns 0 when empty).
        self.char_queue = List[Int]()
        var guard = 0
        while guard < 256:
            var c = Int(external_call["get_char", Int32]())
            if c == 0:
                break
            self.char_queue.append(c)
            guard += 1

    # ---- per-frame reset of hot tracking (active/edit persist as appropriate) ----
    def begin_frame(mut self):
        self.hot_id = -1

    def _mouse_in(self, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
        return point_in_rect(Float32(self.mouse_x), Float32(self.mouse_y), x, y, w, h)


# ============================================================================
# slider/drag core math (single-component), reused by the multi-component variants below.
#
# slider_core: ABSOLUTE mouse-x mapping across [x, x+w] -> [vmin, vmax] (core_widgets.slider_float).
# drag_core:   RELATIVE drag, value += mouse_dx * speed clamped (widget_pack.drag_float).
# Both honour press+release capture via active_id, and the drag anchor press_x in ctx. Each draws
# a track/field box + handle/value so the render gate sees pixels. Returns True if value changed.
# ============================================================================
def slider_core(
    mut ctx: VCtx,
    id: Int,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    mut value: Float32,
    vmin: Float32,
    vmax: Float32,
) raises -> Bool:
    var inside = ctx._mouse_in(x, y, w, h)
    if inside:
        ctx.hot_id = id
    if inside and ctx.mouse_clicked:
        ctx.active_id = id
    var changed = False
    if ctx.active_id == id and ctx.mouse_down:
        var t = (Float32(ctx.mouse_x) - x) / w
        t = clampf(t, Float32(0.0), Float32(1.0))
        var nv = vmin + t * (vmax - vmin)
        if nv != value:
            value = nv
            changed = True
    if ctx.mouse_released and ctx.active_id == id:
        ctx.active_id = -1

    # ---- draw track + filled portion + handle ----
    col(28, 28, 32)
    fill(x, y, w, h)
    col(0, 0, 0)
    rect(x, y, w, h)
    var denom = vmax - vmin
    var frac = Float32(0.0)
    if denom != Float32(0.0):
        frac = (value - vmin) / denom
    frac = clampf(frac, Float32(0.0), Float32(1.0))
    col(60, 100, 150)
    fill(x, y, w * frac, h)
    var hw = Float32(8.0)
    var hx = x + frac * w - hw / Float32(2.0)
    hx = clampf(hx, x, x + w - hw)
    if ctx.active_id == id:
        col(160, 200, 255)
    else:
        col(200, 200, 210)
    fill(hx, y, hw, h)
    # value text on top of the track
    var s = f2s(value)
    col(225, 225, 230)
    txt(s, x + Float32(4.0), y + Float32(3.0), Float32(11.0))
    return changed


def drag_core(
    mut ctx: VCtx,
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
    if inside:
        ctx.hot_id = id
    if inside and ctx.mouse_clicked:
        ctx.active_id = id
        ctx.press_x = ctx.mouse_x
        ctx.press_y = ctx.mouse_y
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

    # ---- draw: field box + centered value text ----
    if ctx.active_id == id:
        col(70, 100, 150)
    elif ctx.hot_id == id:
        col(52, 52, 60)
    else:
        col(38, 38, 44)
    fill(x, y, w, h)
    col(0, 0, 0)
    rect(x, y, w, h)
    var s = f2s(value)
    var m = measure_text(s, Float32(12.0))
    var tw = m[0]
    var th = m[1]
    col(225, 225, 230)
    txt(s, x + (w - tw) / Float32(2.0), y + (h - th) / Float32(2.0), Float32(12.0))
    return changed


# Split a [x,x+w] row carrying a shared label into N equal component sub-rects. The label occupies
# the LEFT `label_w` px (drawn once); the remaining width is split into N cells with `gap` between.
#   cell_w = (w - label_w - (N-1)*gap) / N ;  cell i x = x + label_w + i*(cell_w + gap)
# Returns the cell x of component `i` and the shared cell width.
def comp_cell_x(x: Float32, w: Float32, label_w: Float32, gap: Float32, n: Int, i: Int) -> Float32:
    var cell_w = (w - label_w - Float32(n - 1) * gap) / Float32(n)
    return x + label_w + Float32(i) * (cell_w + gap)


def comp_cell_w(w: Float32, label_w: Float32, gap: Float32, n: Int) -> Float32:
    return (w - label_w - Float32(n - 1) * gap) / Float32(n)


# ============================================================================
# 1) slider_float2/3/4 — N sliders in a row sharing one label. Each component its own sub-rect.
#    Returns True if ANY component changed. Each sub-id = id*16 + component_index (stable, unique).
#    label drawn once on the LEFT; LABEL_W fixed so headless layout is deterministic (no font dep).
# ============================================================================
comptime VLABEL_W = Float32(40.0)
comptime VGAP = Float32(4.0)


def slider_floatN(
    mut ctx: VCtx,
    id: Int,
    label: String,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    mut v0: Float32,
    mut v1: Float32,
    mut v2: Float32,
    mut v3: Float32,
    n: Int,
    vmin: Float32,
    vmax: Float32,
) raises -> Bool:
    var changed = False
    var cw = comp_cell_w(w, VLABEL_W, VGAP, n)
    # component 0
    var cx0 = comp_cell_x(x, w, VLABEL_W, VGAP, n, 0)
    if slider_core(ctx, id * 16 + 0, cx0, y, cw, h, v0, vmin, vmax):
        changed = True
    # component 1
    if n >= 2:
        var cx1 = comp_cell_x(x, w, VLABEL_W, VGAP, n, 1)
        if slider_core(ctx, id * 16 + 1, cx1, y, cw, h, v1, vmin, vmax):
            changed = True
    # component 2
    if n >= 3:
        var cx2 = comp_cell_x(x, w, VLABEL_W, VGAP, n, 2)
        if slider_core(ctx, id * 16 + 2, cx2, y, cw, h, v2, vmin, vmax):
            changed = True
    # component 3
    if n >= 4:
        var cx3 = comp_cell_x(x, w, VLABEL_W, VGAP, n, 3)
        if slider_core(ctx, id * 16 + 3, cx3, y, cw, h, v3, vmin, vmax):
            changed = True
    # shared label on the left
    col(210, 210, 216)
    txt(label, x, y + Float32(3.0), Float32(12.0))
    return changed


# Convenience wrappers (overloaded names; NO default args — untested idiom). v3 placeholder is
# ignored for n<4; pass any Float32 (the gated component loop never reads it when n<4).
def slider_float2(
    mut ctx: VCtx, id: Int, label: String, x: Float32, y: Float32, w: Float32, h: Float32,
    mut v0: Float32, mut v1: Float32, vmin: Float32, vmax: Float32,
) raises -> Bool:
    var dummy_a = Float32(0.0)
    var dummy_b = Float32(0.0)
    return slider_floatN(ctx, id, label, x, y, w, h, v0, v1, dummy_a, dummy_b, 2, vmin, vmax)


def slider_float3(
    mut ctx: VCtx, id: Int, label: String, x: Float32, y: Float32, w: Float32, h: Float32,
    mut v0: Float32, mut v1: Float32, mut v2: Float32, vmin: Float32, vmax: Float32,
) raises -> Bool:
    var dummy_b = Float32(0.0)
    return slider_floatN(ctx, id, label, x, y, w, h, v0, v1, v2, dummy_b, 3, vmin, vmax)


def slider_float4(
    mut ctx: VCtx, id: Int, label: String, x: Float32, y: Float32, w: Float32, h: Float32,
    mut v0: Float32, mut v1: Float32, mut v2: Float32, mut v3: Float32, vmin: Float32, vmax: Float32,
) raises -> Bool:
    return slider_floatN(ctx, id, label, x, y, w, h, v0, v1, v2, v3, 4, vmin, vmax)


# ============================================================================
# 1b) drag_float2/3/4 — N RELATIVE drag fields in a row sharing one label.
#     Each sub-id = id*16 + 8 + component_index (distinct from slider sub-ids above).
# ============================================================================
def drag_floatN(
    mut ctx: VCtx,
    id: Int,
    label: String,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    mut v0: Float32,
    mut v1: Float32,
    mut v2: Float32,
    mut v3: Float32,
    n: Int,
    speed: Float32,
    vmin: Float32,
    vmax: Float32,
) raises -> Bool:
    var changed = False
    var cw = comp_cell_w(w, VLABEL_W, VGAP, n)
    var cx0 = comp_cell_x(x, w, VLABEL_W, VGAP, n, 0)
    if drag_core(ctx, id * 16 + 8 + 0, cx0, y, cw, h, v0, speed, vmin, vmax):
        changed = True
    if n >= 2:
        var cx1 = comp_cell_x(x, w, VLABEL_W, VGAP, n, 1)
        if drag_core(ctx, id * 16 + 8 + 1, cx1, y, cw, h, v1, speed, vmin, vmax):
            changed = True
    if n >= 3:
        var cx2 = comp_cell_x(x, w, VLABEL_W, VGAP, n, 2)
        if drag_core(ctx, id * 16 + 8 + 2, cx2, y, cw, h, v2, speed, vmin, vmax):
            changed = True
    if n >= 4:
        var cx3 = comp_cell_x(x, w, VLABEL_W, VGAP, n, 3)
        if drag_core(ctx, id * 16 + 8 + 3, cx3, y, cw, h, v3, speed, vmin, vmax):
            changed = True
    col(210, 210, 216)
    txt(label, x, y + Float32(3.0), Float32(12.0))
    return changed


def drag_float2(
    mut ctx: VCtx, id: Int, label: String, x: Float32, y: Float32, w: Float32, h: Float32,
    mut v0: Float32, mut v1: Float32, speed: Float32, vmin: Float32, vmax: Float32,
) raises -> Bool:
    var dummy_a = Float32(0.0)
    var dummy_b = Float32(0.0)
    return drag_floatN(ctx, id, label, x, y, w, h, v0, v1, dummy_a, dummy_b, 2, speed, vmin, vmax)


def drag_float3(
    mut ctx: VCtx, id: Int, label: String, x: Float32, y: Float32, w: Float32, h: Float32,
    mut v0: Float32, mut v1: Float32, mut v2: Float32, speed: Float32, vmin: Float32, vmax: Float32,
) raises -> Bool:
    var dummy_b = Float32(0.0)
    return drag_floatN(ctx, id, label, x, y, w, h, v0, v1, v2, dummy_b, 3, speed, vmin, vmax)


def drag_float4(
    mut ctx: VCtx, id: Int, label: String, x: Float32, y: Float32, w: Float32, h: Float32,
    mut v0: Float32, mut v1: Float32, mut v2: Float32, mut v3: Float32,
    speed: Float32, vmin: Float32, vmax: Float32,
) raises -> Bool:
    return drag_floatN(ctx, id, label, x, y, w, h, v0, v1, v2, v3, 4, speed, vmin, vmax)


# ============================================================================
# 2) small_button / invisible_button / button helpers used by input_float steppers.
#
# small_button: a compact framed labelled button; returns True on press+release inside.
# invisible_button: hit-test ONLY (no draw); returns True on press+release inside.
# ============================================================================
def button_core(
    mut ctx: VCtx, id: Int, x: Float32, y: Float32, w: Float32, h: Float32, draw: Bool, label: String
) raises -> Bool:
    var inside = ctx._mouse_in(x, y, w, h)
    var clicked = False
    if inside:
        ctx.hot_id = id
    if inside and ctx.mouse_clicked:
        ctx.active_id = id
    if ctx.mouse_released and ctx.active_id == id:
        if inside:
            clicked = True
        ctx.active_id = -1
    if draw:
        if ctx.active_id == id and inside:
            col(70, 110, 165)
        elif ctx.hot_id == id:
            col(80, 80, 92)
        else:
            col(58, 58, 66)
        fill(x, y, w, h)
        col(0, 0, 0)
        rect(x, y, w, h)
        if len(label) > 0:
            var m = measure_text(label, Float32(12.0))
            var tw = m[0]
            var th = m[1]
            col(228, 228, 233)
            txt(label, x + (w - tw) / Float32(2.0), y + (h - th) / Float32(2.0), Float32(12.0))
    return clicked


def small_button(mut ctx: VCtx, id: Int, label: String, x: Float32, y: Float32) raises -> Bool:
    # compact button auto-sized from the label width estimate (deterministic CHAR_W).
    var bw = Float32(len(label)) * Float32(7.0) + Float32(10.0)
    var bh = Float32(18.0)
    return button_core(ctx, id, x, y, bw, bh, True, label)


def invisible_button(mut ctx: VCtx, id: Int, x: Float32, y: Float32, w: Float32, h: Float32) raises -> Bool:
    # hit-test ONLY (no draw call). Useful for overlay click regions.
    return button_core(ctx, id, x, y, w, h, False, String(""))


# ============================================================================
# 3) arrow_button — a square button with a triangle glyph via draw_filled_triangle.
#    dir: 0=up, 1=down, 2=left, 3=right. Returns True on press+release inside. No trig.
# ============================================================================
def arrow_button(mut ctx: VCtx, id: Int, x: Float32, y: Float32, size: Float32, dir: Int) raises -> Bool:
    var clicked = button_core(ctx, id, x, y, size, size, True, String(""))
    # draw the triangle glyph on TOP of the button-core fill (button_core drew bg+border).
    var pad = size * Float32(0.28)
    var x0 = x + pad
    var y0 = y + pad
    var x1 = x + size - pad
    var y1 = y + size - pad
    var cx = (x0 + x1) / Float32(2.0)
    var cy = (y0 + y1) / Float32(2.0)
    col(225, 225, 230)
    if dir == 0:
        # up: apex top-center, base bottom edge
        ftri(cx, y0, x0, y1, x1, y1)
    elif dir == 1:
        # down: apex bottom-center, base top edge
        ftri(x0, y0, x1, y0, cx, y1)
    elif dir == 2:
        # left: apex left-center, base right edge
        ftri(x0, cy, x1, y0, x1, y1)
    else:
        # right: apex right-center, base left edge
        ftri(x0, y0, x0, y1, x1, cy)
    return clicked


# ============================================================================
# 4) input_float / input_int — a text field showing the number, with [-] and [+] step buttons.
#    Typed digits (INJECTED codepoints in ctx.char_queue) edit a buffer while the field is focused
#    (ctx.edit_id == id); the buffer is parsed on COMMIT. Commit happens when:
#       - Enter (codepoint 10 or 13) is typed, OR
#       - a +/- step button is pressed (parses current buffer first, then applies the step).
#    Focus is acquired by clicking the field (press+release inside the text box).
#
#    Layout:  [ - ][   text field   ][ + ]   step buttons are `bh`-square on each side.
#
#    parse rules (no libm): accept ASCII digits '0'..'9', a single leading '-', and (input_float
#    only) a single '.'; everything else ignored. Empty/invalid buffer parses to 0.
# ============================================================================
def parse_int_buf(buf: List[UInt8]) raises -> Int:
    var sign = 1
    var acc = 0
    var started = False
    var n = len(buf)
    for i in range(n):
        var c = Int(buf[i])
        if c == 45 and not started:        # '-' only as a leading sign
            sign = -1
            started = True
        elif c >= 48 and c <= 57:          # '0'..'9'
            acc = acc * 10 + (c - 48)
            started = True
    return sign * acc


def parse_float_buf(buf: List[UInt8]) raises -> Float32:
    var sign = Float32(1.0)
    var ipart = 0
    var frac = Float32(0.0)
    var scale = Float32(0.1)
    var seen_dot = False
    var n = len(buf)
    for i in range(n):
        var c = Int(buf[i])
        if c == 45 and i == 0:
            sign = Float32(-1.0)
        elif c == 46:                      # '.'
            seen_dot = True
        elif c >= 48 and c <= 57:
            if seen_dot:
                frac = frac + Float32(c - 48) * scale
                scale = scale * Float32(0.1)
            else:
                ipart = ipart * 10 + (c - 48)
    return sign * (Float32(ipart) + frac)


# Consume the injected char queue into ctx.edit_buf while id is focused. Returns True if an Enter
# (commit) codepoint was seen. Handles backspace (8) by trimming the buffer tail.
def consume_chars_into_buf(mut ctx: VCtx, id: Int, accept_dot: Bool) raises -> Bool:
    var commit = False
    if ctx.edit_id != id:
        return False
    var n = len(ctx.char_queue)
    for i in range(n):
        var c = ctx.char_queue[i]
        if c == 10 or c == 13:             # Enter -> commit
            commit = True
        elif c == 8:                       # backspace
            var bl = len(ctx.edit_buf)
            if bl > 0:
                _ = ctx.edit_buf.pop()
        elif c == 45:                      # '-' allowed only as the first char
            if len(ctx.edit_buf) == 0:
                ctx.edit_buf.append(UInt8(45))
        elif c == 46 and accept_dot:       # '.' (input_float only)
            ctx.edit_buf.append(UInt8(46))
        elif c >= 48 and c <= 57:          # digit
            ctx.edit_buf.append(UInt8(c))
    return commit


def buf_to_string(buf: List[UInt8]) raises -> String:
    # chr(Int) -> String, accumulated with += (verified idiom: property_editor.mojo:380).
    var s = String("")
    var n = len(buf)
    for i in range(n):
        s += chr(Int(buf[i]))
    return s


def string_to_buf(s: String) raises -> List[UInt8]:
    var buf = List[UInt8]()
    var b = s.as_bytes()
    for i in range(len(b)):
        buf.append(UInt8(Int(b[i])))
    return buf^


# input_int(ctx,id,x,y,w, mut value, step). Returns True if value changed this frame.
def input_int(
    mut ctx: VCtx, id: Int, x: Float32, y: Float32, w: Float32, mut value: Int, step: Int
) raises -> Bool:
    var bh = Float32(22.0)
    var btn_w = bh
    var field_x = x + btn_w
    var field_w = w - Float32(2.0) * btn_w
    var changed = False

    # [-] step button (id*8+1)
    if arrow_button(ctx, id * 8 + 1, x, y, btn_w, 2):     # dir 2 used as a generic "minus" glyph? -> use small minus
        # parse current buffer if editing, then step down
        if ctx.edit_id == id:
            value = parse_int_buf(ctx.edit_buf)
            ctx.edit_id = -1
        value = value - step
        changed = True

    # focus acquisition: clicking the text field focuses it for typed editing.
    var inside_field = ctx._mouse_in(field_x, y, field_w, bh)
    if inside_field:
        ctx.hot_id = id
    if inside_field and ctx.mouse_clicked:
        ctx.active_id = id
    if ctx.mouse_released and ctx.active_id == id:
        if inside_field:
            if ctx.edit_id != id:
                ctx.edit_id = id
                ctx.edit_buf = string_to_buf(String(value))   # seed buffer with current value
        ctx.active_id = -1

    # consume injected chars while focused; Enter commits.
    var did_commit = consume_chars_into_buf(ctx, id, False)
    if did_commit:
        var nv = parse_int_buf(ctx.edit_buf)
        if nv != value:
            value = nv
            changed = True
        ctx.edit_id = -1

    # [+] step button (id*8+2)
    if arrow_button(ctx, id * 8 + 2, x + w - btn_w, y, btn_w, 3):
        if ctx.edit_id == id:
            value = parse_int_buf(ctx.edit_buf)
            ctx.edit_id = -1
        value = value + step
        changed = True

    # ---- draw the field box + text (current buffer if editing, else the value) ----
    if ctx.edit_id == id:
        col(46, 46, 54)
    elif ctx.hot_id == id:
        col(40, 40, 48)
    else:
        col(30, 30, 36)
    fill(field_x, y, field_w, bh)
    col(0, 0, 0)
    rect(field_x, y, field_w, bh)
    var shown = String(value)
    if ctx.edit_id == id:
        shown = buf_to_string(ctx.edit_buf)
    col(225, 225, 230)
    txt(shown, field_x + Float32(5.0), y + Float32(4.0), Float32(12.0))
    return changed


# input_float(ctx,id,x,y,w, mut value, step). Returns True if value changed this frame.
def input_float(
    mut ctx: VCtx, id: Int, x: Float32, y: Float32, w: Float32, mut value: Float32, step: Float32
) raises -> Bool:
    var bh = Float32(22.0)
    var btn_w = bh
    var field_x = x + btn_w
    var field_w = w - Float32(2.0) * btn_w
    var changed = False

    if arrow_button(ctx, id * 8 + 1, x, y, btn_w, 2):
        if ctx.edit_id == id:
            value = parse_float_buf(ctx.edit_buf)
            ctx.edit_id = -1
        value = value - step
        changed = True

    var inside_field = ctx._mouse_in(field_x, y, field_w, bh)
    if inside_field:
        ctx.hot_id = id
    if inside_field and ctx.mouse_clicked:
        ctx.active_id = id
    if ctx.mouse_released and ctx.active_id == id:
        if inside_field:
            if ctx.edit_id != id:
                ctx.edit_id = id
                ctx.edit_buf = string_to_buf(f2s(value))
        ctx.active_id = -1

    var did_commit = consume_chars_into_buf(ctx, id, True)
    if did_commit:
        var nv = parse_float_buf(ctx.edit_buf)
        if nv != value:
            value = nv
            changed = True
        ctx.edit_id = -1

    if arrow_button(ctx, id * 8 + 2, x + w - btn_w, y, btn_w, 3):
        if ctx.edit_id == id:
            value = parse_float_buf(ctx.edit_buf)
            ctx.edit_id = -1
        value = value + step
        changed = True

    if ctx.edit_id == id:
        col(46, 46, 54)
    elif ctx.hot_id == id:
        col(40, 40, 48)
    else:
        col(30, 30, 36)
    fill(field_x, y, field_w, bh)
    col(0, 0, 0)
    rect(field_x, y, field_w, bh)
    var shown = f2s(value)
    if ctx.edit_id == id:
        shown = buf_to_string(ctx.edit_buf)
    col(225, 225, 230)
    txt(shown, field_x + Float32(5.0), y + Float32(4.0), Float32(12.0))
    return changed


# ============================================================================
# 5) input_text_multiline — a List[List[UInt8]] of lines edited by injected codepoints.
#    Enter (10/13) splits the current line at the cursor (col) into a new line below; Backspace (8)
#    at col 0 MERGES the current line into the previous one (cursor moves to the join point);
#    other printable codepoints are inserted at the cursor. Cursor is (row, col) carried in ctx-less
#    out params (the caller owns them). Returns True if the buffer changed.
#
#    The buffer is owned by the CALLER (a List[List[UInt8]]); cursor row/col are mut params.
#    Draws each line as text inside the field box; clips to the box via set_clip_rect.
# ============================================================================
def input_text_multiline(
    mut ctx: VCtx,
    id: Int,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    mut lines: List[List[UInt8]],
    mut cur_row: Int,
    mut cur_col: Int,
) raises -> Bool:
    var changed = False
    var line_h = Float32(16.0)

    # focus acquisition by clicking the box.
    var inside = ctx._mouse_in(x, y, w, h)
    if inside:
        ctx.hot_id = id
    if inside and ctx.mouse_clicked:
        ctx.active_id = id
        ctx.edit_id = id
    if ctx.mouse_released and ctx.active_id == id:
        ctx.active_id = -1

    # ensure at least one line exists.
    if len(lines) == 0:
        lines.append(List[UInt8]())
        cur_row = 0
        cur_col = 0

    # consume injected chars only while focused.
    #
    # IMPLEMENTATION NOTE: List.insert(idx,val) and List.pop(idx) are NOT proven in the verified
    # exemplars (only no-arg List.pop() and List.append() are). So every structural mutation
    # (Enter-split, line-merge) REBUILDS the `lines` list with append-only into a fresh List and
    # reassigns it with ^ — using ONLY the proven primitives.
    if ctx.edit_id == id:
        var n = len(ctx.char_queue)
        for i in range(n):
            var c = ctx.char_queue[i]
            if cur_row < 0:
                cur_row = 0
            if cur_row > len(lines) - 1:
                cur_row = len(lines) - 1
            var cur_line = lines[cur_row].copy()
            if cur_col < 0:
                cur_col = 0
            if cur_col > len(cur_line):
                cur_col = len(cur_line)

            if c == 10 or c == 13:                 # Enter -> split the current line at the cursor
                var head = List[UInt8]()
                var tail = List[UInt8]()
                for k in range(len(cur_line)):
                    if k < cur_col:
                        head.append(cur_line[k])
                    else:
                        tail.append(cur_line[k])
                # rebuild lines: rows < cur_row unchanged, cur_row -> head, then tail, then rows > cur_row.
                var rebuilt = List[List[UInt8]]()
                var total = len(lines)
                for r in range(total):
                    if r == cur_row:
                        rebuilt.append(head.copy())
                        rebuilt.append(tail.copy())
                    else:
                        rebuilt.append(lines[r].copy())
                lines = rebuilt^
                cur_row = cur_row + 1
                cur_col = 0
                changed = True
            elif c == 8:                            # Backspace
                if cur_col > 0:
                    var nl = List[UInt8]()
                    for k in range(len(cur_line)):
                        if k != cur_col - 1:
                            nl.append(cur_line[k])
                    lines[cur_row] = nl^
                    cur_col = cur_col - 1
                    changed = True
                elif cur_row > 0:                   # col 0 -> merge into the previous line
                    var prev = lines[cur_row - 1].copy()
                    var join_col = len(prev)
                    var merged = List[UInt8]()
                    for k in range(len(prev)):
                        merged.append(prev[k])
                    for k in range(len(cur_line)):
                        merged.append(cur_line[k])
                    # rebuild lines: row(cur_row-1) -> merged, DROP row cur_row, others unchanged.
                    var rebuilt2 = List[List[UInt8]]()
                    var total2 = len(lines)
                    for r in range(total2):
                        if r == cur_row - 1:
                            rebuilt2.append(merged.copy())
                        elif r == cur_row:
                            pass                     # dropped (merged away)
                        else:
                            rebuilt2.append(lines[r].copy())
                    lines = rebuilt2^
                    cur_row = cur_row - 1
                    cur_col = join_col
                    changed = True
            elif c >= 32:                           # printable -> insert at the cursor
                var nl2 = List[UInt8]()
                for k in range(len(cur_line)):
                    if k == cur_col:
                        nl2.append(UInt8(c))
                    nl2.append(cur_line[k])
                if cur_col >= len(cur_line):
                    nl2.append(UInt8(c))
                lines[cur_row] = nl2^
                cur_col = cur_col + 1
                changed = True

    # ---- draw the box + clipped lines ----
    if ctx.edit_id == id:
        col(40, 40, 48)
    else:
        col(28, 28, 34)
    fill(x, y, w, h)
    col(0, 0, 0)
    rect(x, y, w, h)
    push_clip(x, y, w, h)
    col(225, 225, 230)
    var nlines = len(lines)
    for i in range(nlines):
        var ly = y + Float32(4.0) + Float32(i) * line_h
        var s = buf_to_string(lines[i].copy())
        txt(s, x + Float32(5.0), ly, Float32(12.0))
    pop_clip()
    return changed


# ============================================================================
# 6) selectable — a full-width row, highlighted when `selected`. Returns True if CLICKED this frame
#    (press+release inside). Caller owns the selection model (toggles on the returned click).
# ============================================================================
def selectable(
    mut ctx: VCtx, id: Int, label: String, x: Float32, y: Float32, w: Float32, h: Float32, selected: Bool
) raises -> Bool:
    var inside = ctx._mouse_in(x, y, w, h)
    var clicked = False
    if inside:
        ctx.hot_id = id
    if inside and ctx.mouse_clicked:
        ctx.active_id = id
    if ctx.mouse_released and ctx.active_id == id:
        if inside:
            clicked = True
        ctx.active_id = -1

    # ---- draw row background ----
    if selected:
        col(70, 110, 165)
    elif ctx.hot_id == id:
        col(52, 52, 60)
    else:
        col(34, 34, 40)
    fill(x, y, w, h)
    if selected:
        col(240, 240, 245)
    else:
        col(220, 220, 226)
    txt(label, x + Float32(6.0), y + h - Float32(6.0), Float32(12.0))
    return clicked


# ============================================================================
# 7) list_box — a framed, clipped list of selectables. Clicking a row sets `sel` to that index.
#    Returns True if the selection changed this frame. Rows are `row_h` tall; the list is clipped
#    to (x,y,w,h) via set_clip_rect so overflow rows are scissored.
# ============================================================================
def list_box(
    mut ctx: VCtx,
    id: Int,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    items: List[String],
    mut sel: Int,
) raises -> Bool:
    var row_h = Float32(20.0)
    var changed = False

    # frame
    col(24, 24, 28)
    fill(x, y, w, h)
    col(0, 0, 0)
    rect(x, y, w, h)

    push_clip(x, y, w, h)
    var n = len(items)
    for i in range(n):
        var iy = y + Float32(i) * row_h
        var row_id = id * 4096 + i + 1                # stable per-row id
        var is_sel = i == sel
        if selectable(ctx, row_id, items[i].copy(), x, iy, w, row_h, is_sel):
            if sel != i:
                sel = i
                changed = True
    pop_clip()
    return changed


# A scrollable list box: like list_box but with a draggable scrollbar when the content
# overflows. `scroll` is caller-owned (pixels from top). Returns True if selection changed.
def list_box_scroll(mut ctx: VCtx, id: Int, x: Float32, y: Float32, w: Float32, h: Float32, items: List[String], mut sel: Int, mut scroll: Float32) raises -> Bool:
    var row_h = Float32(20.0)
    var n = len(items)
    var content_h = Float32(n) * row_h
    var max_scroll = content_h - h
    if max_scroll < Float32(0.0): max_scroll = Float32(0.0)
    var has_sb = max_scroll > Float32(0.0)
    var sb_w = Float32(12.0)
    var inner_w = w
    if has_sb: inner_w = w - sb_w
    # frame
    col(24, 24, 28)
    fill(x, y, w, h)
    col(0, 0, 0)
    rect(x, y, w, h)
    # scrollbar drag (right edge); distinct id from the row ids
    var sb_id = id * 7 + 3
    var thumb_h = Float32(16.0)
    var track = Float32(0.0)
    if has_sb:
        var sb_x = x + w - sb_w
        var inside_sb = ctx._mouse_in(sb_x, y, sb_w, h)
        if inside_sb: ctx.hot_id = sb_id
        if inside_sb and ctx.mouse_clicked: ctx.active_id = sb_id
        thumb_h = h * (h / content_h)
        if thumb_h < Float32(16.0): thumb_h = Float32(16.0)
        track = h - thumb_h
        if ctx.active_id == sb_id and ctx.mouse_down:
            var t = (Float32(ctx.mouse_y) - y - thumb_h / Float32(2.0)) / track
            if t < Float32(0.0): t = Float32(0.0)
            if t > Float32(1.0): t = Float32(1.0)
            scroll = t * max_scroll
        if ctx.mouse_released and ctx.active_id == sb_id: ctx.active_id = -1
    if scroll < Float32(0.0): scroll = Float32(0.0)
    if scroll > max_scroll: scroll = max_scroll
    # rows (clipped + offset)
    push_clip(x, y, inner_w, h)
    var changed = False
    for i in range(n):
        var iy = y - scroll + Float32(i) * row_h
        if iy + row_h < y or iy > y + h:
            continue
        var row_id = id * 4096 + i + 1
        if selectable(ctx, row_id, items[i].copy(), x, iy, inner_w, row_h, i == sel):
            if sel != i:
                sel = i
                changed = True
    pop_clip()
    # scrollbar visual
    if has_sb:
        var sb_x = x + w - sb_w
        col(34, 34, 40)
        fill(sb_x, y, sb_w, h)
        var thumb_y = y + (scroll / max_scroll) * track
        if ctx.active_id == sb_id:
            col(140, 140, 155)
        else:
            col(90, 90, 102)
        fill(sb_x + Float32(2.0), thumb_y, sb_w - Float32(4.0), thumb_h)
    return changed


# ============================================================================
# 8) slider_angle — maps a [-180, 180] degree slider to radians. The stored value is RADIANS;
#    the display + the slider track operate in DEGREES. No trig: a plain constant ratio converts.
#       deg = radians * DEG_PER_RAD ; radians = deg / DEG_PER_RAD
#    DEG_PER_RAD = 180 / pi ; pi baked as a comptime literal (NOT a trig call).
#    Returns True if the radian value changed this frame.
# ============================================================================
comptime PI = Float32(3.14159265358979323846)
# DEG_PER_RAD = 180/pi baked as a LITERAL (not a computed comptime — no exemplar proves computed
# comptime folding, and this is a constant ratio, NOT a trig call -> no libm pull-in).
comptime DEG_PER_RAD = Float32(57.29577951308232)


def slider_angle(
    mut ctx: VCtx, id: Int, x: Float32, y: Float32, w: Float32, mut radians: Float32
) raises -> Bool:
    var h = Float32(22.0)
    var deg = radians * DEG_PER_RAD
    var inside = ctx._mouse_in(x, y, w, h)
    if inside:
        ctx.hot_id = id
    if inside and ctx.mouse_clicked:
        ctx.active_id = id
    var changed = False
    if ctx.active_id == id and ctx.mouse_down:
        var t = (Float32(ctx.mouse_x) - x) / w
        t = clampf(t, Float32(0.0), Float32(1.0))
        var nd = Float32(-180.0) + t * Float32(360.0)     # map track to [-180,180] deg
        var nr = nd / DEG_PER_RAD
        if nr != radians:
            radians = nr
            deg = nd
            changed = True
    if ctx.mouse_released and ctx.active_id == id:
        ctx.active_id = -1

    # ---- draw track + handle + "NNN deg" label ----
    col(28, 28, 32)
    fill(x, y, w, h)
    col(0, 0, 0)
    rect(x, y, w, h)
    var frac = clampf((deg + Float32(180.0)) / Float32(360.0), Float32(0.0), Float32(1.0))
    col(60, 100, 150)
    fill(x, y, w * frac, h)
    var hw = Float32(8.0)
    var hx = clampf(x + frac * w - hw / Float32(2.0), x, x + w - hw)
    if ctx.active_id == id:
        col(160, 200, 255)
    else:
        col(200, 200, 210)
    fill(hx, y, hw, h)
    var s = f2s(deg) + String(" deg")
    col(225, 225, 230)
    txt(s, x + Float32(5.0), y + Float32(4.0), Float32(11.0))
    return changed


# ============================================================================
# SELF-TEST  (compiled + run headlessly by the MAIN LOOP — do NOT run it yourself)
#
# Press+release widgets need two frames (down -> up). Slider widgets capture active on the press
# frame then drag-track on the held frame(s). Every widget issues real draw_* calls so the
# glReadPixels framebuffer readback exercises their paint paths.
#
# Each expected number below is derived from FIRST PRINCIPLES in the comment beside it.
# ============================================================================
def main() raises:
    var ctx = VCtx()
    var pass_all = True

    # ===================================================================
    # 1) SLIDER_FLOAT3 — three sliders sharing a label, range [0,100].
    #
    # Row rect chosen so cell_w is a CLEAN 100.0 (no Float32 rounding noise):
    #   x=100, y=100, w=348, h=22.  VLABEL_W=40, VGAP=4, n=3.
    #   cell_w = (348 - 40 - (3-1)*4) / 3 = (348 - 40 - 8) / 3 = 300 / 3 = 100.0  px  (EXACT)
    #   cell0_x = 100 + 40 + 0*(100 + 4) = 140.0   -> cell0 = [140, 240)
    #   cell1_x = 100 + 40 + 1*(100 + 4) = 244.0   -> cell1 = [244, 344)
    #   cell2_x = 100 + 40 + 2*(100 + 4) = 348.0   -> cell2 = [348, 448)
    #
    # ABSOLUTE mouse-x map (core_widgets.slider_float):  value = vmin + t*(vmax-vmin),
    #   t = clamp((mouse_x - cell_x)/cell_w, 0, 1).  Each component driven INSIDE ITS OWN cell so
    #   only that component captures (cells are disjoint -> no cross-capture):
    #     comp0 @ mouse_x = 140 + 0.25*100 = 165  -> t = (165-140)/100 = 0.25 -> 0 + 0.25*100 = 25.0
    #     comp1 @ mouse_x = 244 + 0.50*100 = 294  -> t = (294-244)/100 = 0.50 -> 0 + 0.50*100 = 50.0
    #     comp2 @ mouse_x = 348 + 0.75*100 = 423  -> t = (423-348)/100 = 0.75 -> 0 + 0.75*100 = 75.0
    #   165∈[140,240) only ; 294∈[244,344) only ; 423∈[348,448) only -> each captures exactly one comp.
    # ===================================================================
    var v0 = Float32(0.0)
    var v1 = Float32(0.0)
    var v2 = Float32(0.0)
    var rx = Float32(100.0)
    var ry = Float32(100.0)
    var rw = Float32(348.0)
    var rh = Float32(22.0)
    var rmy = Int(ry + Float32(11.0))     # 111, inside the row vertically

    # comp0 @ mouse_x=165 (t=0.25) -> v0 = 25.0. Press latches active(comp0), drag-tracks same frame.
    ctx.begin_frame()
    ctx.set_input(165, rmy, True)
    _ = slider_float3(ctx, 1, String("Pos"), rx, ry, rw, rh, v0, v1, v2, Float32(0.0), Float32(100.0))
    ctx.begin_frame()
    ctx.set_input(165, rmy, False)    # release -> clear capture before next component
    _ = slider_float3(ctx, 1, String("Pos"), rx, ry, rw, rh, v0, v1, v2, Float32(0.0), Float32(100.0))

    # comp1 @ mouse_x=294 (t=0.50) -> v1 = 50.0.
    ctx.begin_frame()
    ctx.set_input(294, rmy, True)
    _ = slider_float3(ctx, 1, String("Pos"), rx, ry, rw, rh, v0, v1, v2, Float32(0.0), Float32(100.0))
    ctx.begin_frame()
    ctx.set_input(294, rmy, False)
    _ = slider_float3(ctx, 1, String("Pos"), rx, ry, rw, rh, v0, v1, v2, Float32(0.0), Float32(100.0))

    # comp2 @ mouse_x=423 (t=0.75) -> v2 = 75.0.
    ctx.begin_frame()
    ctx.set_input(423, rmy, True)
    _ = slider_float3(ctx, 1, String("Pos"), rx, ry, rw, rh, v0, v1, v2, Float32(0.0), Float32(100.0))
    ctx.begin_frame()
    ctx.set_input(423, rmy, False)
    _ = slider_float3(ctx, 1, String("Pos"), rx, ry, rw, rh, v0, v1, v2, Float32(0.0), Float32(100.0))

    print("WIDGETVARIANTS_SELFTEST: sf3_v0=", v0, " sf3_v1=", v1, " sf3_v2=", v2)
    # EXPECT: sf3_v0= 25.0   sf3_v1= 50.0   sf3_v2= 75.0
    #   comp0 t=0.25 -> 25.0 ; comp1 t=0.50 -> 50.0 ; comp2 t=0.75 -> 75.0 (cell_w exactly 100.0).
    if v0 != Float32(25.0):
        pass_all = False
    if v1 != Float32(50.0):
        pass_all = False
    if v2 != Float32(75.0):
        pass_all = False

    # ===================================================================
    # 2) INPUT_INT — parse codepoints '4','2' -> 42 ; step +1 -> 43.
    #
    # Field layout: x=100, y=160, w=120. bh=22, btn_w=22. field_x = 100+22 = 122, field_w = 76.
    #   Focus: click the text field (press+release inside) -> ctx.edit_id = id, buffer seeded with
    #   String(value)="0" -> wait, seeding with current value (0) would PREPEND. To verify a clean
    #   "42" parse we focus on an EMPTY-seed: value starts at 0, seed="0". Typing '4','2' appends ->
    #   buffer becomes "0","4","2" = "042" -> parse_int_buf = 42 (leading zero is harmless: 0*10... )
    #   Actually 0,4,2 -> ((0*10+4)*10+2)=42. Good: parse yields 42 regardless of the leading 0.
    #   Then Enter (codepoint 10) commits -> value = 42.
    #   Then press the [+] step button -> value = 42 + 1 = 43.
    # ===================================================================
    var ival = 0
    var ix = Float32(100.0)
    var iy = Float32(160.0)
    var iw = Float32(120.0)
    var ibtn = Float32(22.0)
    var ifield_x = ix + ibtn
    var ifield_w = iw - Float32(2.0) * ibtn
    var ifield_cx = Int(ifield_x + ifield_w / Float32(2.0))     # center x of the text field
    var ifield_cy = Int(iy + Float32(11.0))

    # frame A: click the text field (down) to start focus capture.
    ctx.begin_frame()
    ctx.clear_chars()
    ctx.set_input(ifield_cx, ifield_cy, True)
    _ = input_int(ctx, 2, ix, iy, iw, ival, 1)
    # frame B: release inside the field -> focus acquired (edit_id=2, buffer seeded "0").
    ctx.begin_frame()
    ctx.clear_chars()
    ctx.set_input(ifield_cx, ifield_cy, False)
    _ = input_int(ctx, 2, ix, iy, iw, ival, 1)

    # frame C: type '4' (52), '2' (50), Enter (10) -> commit -> value 42. Mouse parked away.
    ctx.begin_frame()
    var typed = List[Int]()
    typed.append(52)    # '4'
    typed.append(50)    # '2'
    typed.append(10)    # Enter -> commit
    ctx.set_chars(typed)
    ctx.set_input(5000, 5000, False)
    _ = input_int(ctx, 2, ix, iy, iw, ival, 1)
    print("WIDGETVARIANTS_SELFTEST: input_int_parsed=", ival)
    # EXPECT: input_int_parsed= 42   (buffer "0"+"4"+"2"="042" -> parse_int_buf = 42; Enter commits)
    if ival != 42:
        pass_all = False

    # frame D: press the [+] step button (sub-id 2*8+2=18) at its center -> value 42+1 = 43.
    #   [+] button rect = (ix + iw - btn_w, iy, btn_w, btn_w) = (100+120-22, 160, 22, 22) = (198,160,22,22).
    var plus_cx = Int(ix + iw - ibtn + ibtn / Float32(2.0))     # 198 + 11 = 209
    var plus_cy = Int(iy + ibtn / Float32(2.0))                 # 160 + 11 = 171
    ctx.begin_frame()
    ctx.clear_chars()
    ctx.set_input(plus_cx, plus_cy, True)
    _ = input_int(ctx, 2, ix, iy, iw, ival, 1)
    ctx.begin_frame()
    ctx.clear_chars()
    ctx.set_input(plus_cx, plus_cy, False)
    _ = input_int(ctx, 2, ix, iy, iw, ival, 1)
    print("WIDGETVARIANTS_SELFTEST: input_int_stepped=", ival)
    # EXPECT: input_int_stepped= 43   (42 + step 1 on [+] release)
    if ival != 43:
        pass_all = False

    # ===================================================================
    # 3) SELECTABLE — click a full-width row -> returns True.
    #   row rect = (100, 220, 200, 24). center inside @ (200, 232).
    #   press (down) frame latches active; release (up) inside fires clicked True.
    # ===================================================================
    var selx = Float32(100.0)
    var sely = Float32(220.0)
    var selw = Float32(200.0)
    var selh = Float32(24.0)
    var sel_cx = Int(selx + selw / Float32(2.0))     # 200
    var sel_cy = Int(sely + selh / Float32(2.0))     # 232
    ctx.begin_frame()
    ctx.clear_chars()
    ctx.set_input(sel_cx, sel_cy, True)
    var s1 = selectable(ctx, 3, String("Layer 1"), selx, sely, selw, selh, False)
    ctx.begin_frame()
    ctx.set_input(sel_cx, sel_cy, False)
    var s2 = selectable(ctx, 3, String("Layer 1"), selx, sely, selw, selh, False)
    print("WIDGETVARIANTS_SELFTEST: selectable_clicked=", s2)
    # EXPECT: selectable_clicked= True   (press inside + release inside == click)
    if s1 != False:
        pass_all = False
    if s2 != True:
        pass_all = False

    # ===================================================================
    # 4) LIST_BOX — pick row index 2 -> sel becomes 2.
    #   box rect = (100, 260, 200, 100). row_h = 20. Row i rect = (100, 260 + i*20, 200, 20).
    #   Row 2 rect = (100, 260+40, 200, 20) = (100, 300, 200, 20). center @ (200, 310).
    #   Selecting row 2 (press+release inside) sets sel=2 and returns changed=True (sel started -1).
    # ===================================================================
    var items = List[String]()
    items.append(String("Alpha"))
    items.append(String("Beta"))
    items.append(String("Gamma"))
    items.append(String("Delta"))
    var lsel = -1
    var lbx = Float32(100.0)
    var lby = Float32(260.0)
    var lbw = Float32(200.0)
    var lbh = Float32(100.0)
    var row2_cx = Int(lbx + lbw / Float32(2.0))      # 200
    var row2_cy = Int(lby + Float32(2.0) * Float32(20.0) + Float32(10.0))  # 260 + 40 + 10 = 310
    ctx.begin_frame()
    ctx.set_input(row2_cx, row2_cy, True)
    var lc1 = list_box(ctx, 4, lbx, lby, lbw, lbh, items, lsel)
    ctx.begin_frame()
    ctx.set_input(row2_cx, row2_cy, False)
    var lc2 = list_box(ctx, 4, lbx, lby, lbw, lbh, items, lsel)
    print("WIDGETVARIANTS_SELFTEST: list_box_sel=", lsel, " changed=", lc2)
    # EXPECT: list_box_sel= 2   changed= True
    #   row 2 center (200,310) is inside row-2 rect; press+release -> selectable returns clicked ->
    #   sel was -1 != 2 -> sel=2, changed=True.
    if lsel != 2:
        pass_all = False
    if lc2 != True:
        pass_all = False

    # ===================================================================
    # 5) ARROW_BUTTON — click -> True.
    #   button rect = (100, 380, 22, 22), dir=1 (down). center inside @ (111, 391).
    #   press (down) latches active; release (up) inside fires clicked True.
    # ===================================================================
    var abx = Float32(100.0)
    var aby = Float32(380.0)
    var absz = Float32(22.0)
    var ab_cx = Int(abx + absz / Float32(2.0))       # 111
    var ab_cy = Int(aby + absz / Float32(2.0))       # 391
    ctx.begin_frame()
    ctx.set_input(ab_cx, ab_cy, True)
    var a1 = arrow_button(ctx, 5, abx, aby, absz, 1)
    ctx.begin_frame()
    ctx.set_input(ab_cx, ab_cy, False)
    var a2 = arrow_button(ctx, 5, abx, aby, absz, 1)
    print("WIDGETVARIANTS_SELFTEST: arrow_button_clicked=", a2)
    # EXPECT: arrow_button_clicked= True   (press inside + release inside == click)
    if a1 != False:
        pass_all = False
    if a2 != True:
        pass_all = False

    # ===================================================================
    # RENDER-GATE COVERAGE — exercise the remaining VISUAL widgets so glReadPixels sees their paint
    # paths. Mouse parked far away (5000,5000): no interaction, pure draw coverage.
    #   slider_float2, slider_float4, drag_float2/3/4, small_button, invisible_button,
    #   input_float, input_text_multiline, slider_angle, arrow_button dirs 0/2/3.
    # ===================================================================
    ctx.begin_frame()
    ctx.clear_chars()
    ctx.set_input(5000, 5000, False)

    var sf2a = Float32(10.0)
    var sf2b = Float32(20.0)
    _ = slider_float2(ctx, 60, String("XY"), Float32(460.0), Float32(100.0), Float32(300.0), Float32(22.0),
                      sf2a, sf2b, Float32(0.0), Float32(100.0))
    var sf4a = Float32(1.0)
    var sf4b = Float32(2.0)
    var sf4c = Float32(3.0)
    var sf4d = Float32(4.0)
    _ = slider_float4(ctx, 61, String("RGBA"), Float32(460.0), Float32(130.0), Float32(300.0), Float32(22.0),
                      sf4a, sf4b, sf4c, sf4d, Float32(0.0), Float32(10.0))
    var df2a = Float32(5.0)
    var df2b = Float32(6.0)
    _ = drag_float2(ctx, 62, String("UV"), Float32(460.0), Float32(160.0), Float32(300.0), Float32(22.0),
                    df2a, df2b, Float32(0.1), Float32(-100.0), Float32(100.0))
    var df3a = Float32(7.0)
    var df3b = Float32(8.0)
    var df3c = Float32(9.0)
    _ = drag_float3(ctx, 63, String("Rot"), Float32(460.0), Float32(190.0), Float32(300.0), Float32(22.0),
                    df3a, df3b, df3c, Float32(0.1), Float32(-100.0), Float32(100.0))
    var df4a = Float32(1.0)
    var df4b = Float32(2.0)
    var df4c = Float32(3.0)
    var df4d = Float32(4.0)
    _ = drag_float4(ctx, 64, String("Quat"), Float32(460.0), Float32(220.0), Float32(300.0), Float32(22.0),
                    df4a, df4b, df4c, df4d, Float32(0.1), Float32(-100.0), Float32(100.0))
    _ = small_button(ctx, 65, String("Reset"), Float32(460.0), Float32(250.0))
    _ = invisible_button(ctx, 66, Float32(560.0), Float32(250.0), Float32(60.0), Float32(18.0))
    var ifv = Float32(3.5)
    _ = input_float(ctx, 67, Float32(460.0), Float32(280.0), Float32(120.0), ifv, Float32(0.5))
    # input_text_multiline: seed two lines, draw (no typing this frame).
    var mlines = List[List[UInt8]]()
    mlines.append(string_to_buf(String("hello")))
    mlines.append(string_to_buf(String("world")))
    var mrow = 0
    var mcol = 0
    _ = input_text_multiline(ctx, 68, Float32(460.0), Float32(310.0), Float32(200.0), Float32(60.0),
                             mlines, mrow, mcol)
    var angv = PI / Float32(2.0)        # 90 deg
    _ = slider_angle(ctx, 69, Float32(460.0), Float32(380.0), Float32(200.0), angv)
    # arrow_button dirs 0/2/3 for triangle-glyph coverage
    _ = arrow_button(ctx, 70, Float32(670.0), Float32(380.0), Float32(22.0), 0)
    _ = arrow_button(ctx, 71, Float32(694.0), Float32(380.0), Float32(22.0), 2)
    _ = arrow_button(ctx, 72, Float32(718.0), Float32(380.0), Float32(22.0), 3)
    print("WIDGETVARIANTS_SELFTEST: render_coverage_drawn=", True)   # EXPECT True (draws issued)

    # ===================================================================
    # 6) INPUT_TEXT_MULTILINE behaviour check (logic-only, no draw assertion):
    #   Start one empty line. Type 'a','b' -> line0 = "ab", cursor col 2. Enter -> split: line0="ab",
    #   line1="" , cursor (row1,col0). Type 'c' -> line1="c". Then Backspace at col 1 deletes 'c' ->
    #   line1="". Then Backspace at col 0 -> merge line1 into line0 -> line0="ab", 1 line total,
    #   cursor (row0,col2). We assert: 1 line, line0 == "ab".
    # ===================================================================
    var tlines = List[List[UInt8]]()
    var trow = 0
    var tcol = 0
    # frame: focus the box by clicking it, then type. Box (100,440,200,60).
    ctx.begin_frame()
    ctx.clear_chars()
    ctx.set_input(150, 460, True)
    _ = input_text_multiline(ctx, 80, Float32(100.0), Float32(440.0), Float32(200.0), Float32(60.0),
                             tlines, trow, tcol)
    ctx.begin_frame()
    ctx.set_input(150, 460, False)
    _ = input_text_multiline(ctx, 80, Float32(100.0), Float32(440.0), Float32(200.0), Float32(60.0),
                             tlines, trow, tcol)
    # type 'a','b'
    ctx.begin_frame()
    var t1 = List[Int]()
    t1.append(97)   # 'a'
    t1.append(98)   # 'b'
    ctx.set_chars(t1)
    ctx.set_input(5000, 5000, False)
    _ = input_text_multiline(ctx, 80, Float32(100.0), Float32(440.0), Float32(200.0), Float32(60.0),
                             tlines, trow, tcol)
    # Enter -> split
    ctx.begin_frame()
    var t2 = List[Int]()
    t2.append(10)   # Enter
    ctx.set_chars(t2)
    ctx.set_input(5000, 5000, False)
    _ = input_text_multiline(ctx, 80, Float32(100.0), Float32(440.0), Float32(200.0), Float32(60.0),
                             tlines, trow, tcol)
    # type 'c' on the new line, then backspace twice (delete 'c', then merge)
    ctx.begin_frame()
    var t3 = List[Int]()
    t3.append(99)   # 'c'
    t3.append(8)    # backspace -> delete 'c'
    t3.append(8)    # backspace at col 0 -> merge into line0
    ctx.set_chars(t3)
    ctx.set_input(5000, 5000, False)
    _ = input_text_multiline(ctx, 80, Float32(100.0), Float32(440.0), Float32(200.0), Float32(60.0),
                             tlines, trow, tcol)
    var line_count = len(tlines)
    var line0s = buf_to_string(tlines[0].copy())
    print("WIDGETVARIANTS_SELFTEST: multiline_lines=", line_count, " line0=", line0s)
    # EXPECT: multiline_lines= 1   line0= ab
    #   'a','b' -> "ab"(col2); Enter splits at col2 -> ["ab",""](row1,col0); 'c' -> line1="c"(col1);
    #   backspace col1 -> line1=""(col0); backspace col0 -> merge -> line0="ab", 1 line.
    if line_count != 1:
        pass_all = False
    if line0s != String("ab"):
        pass_all = False

    # ----- summary line the main loop gates on -----
    if pass_all:
        print("WIDGETVARIANTS_SELFTEST: pass (sf3=25/50/75, input_int 42->43, selectable True, "
              + "list_box sel=2, arrow True, multiline merge -> 'ab')")
    else:
        print("WIDGETVARIANTS_SELFTEST: fail")
