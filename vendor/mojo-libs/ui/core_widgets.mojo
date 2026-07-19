# core_widgets.mojo — a immediate-mode IMMEDIATE-MODE widget toolkit for Mojo 1.0.0b1,
# drawn on the MojoGUI C rendering backend (librender.so) via external_call ONLY.
#
# This is the Mojo analogue of what MediaEditor builds with immediate-mode GUI: a UiContext that
# carries per-frame input, tracks hot (hovered) / active (mouse-captured) widget ids, and a
# set of stateless-ish widget calls (text/button/checkbox/radio/slider_float/child/tooltip/combo).
#
# FFI idioms mirrored EXACTLY from a working, framebuffer-verified exemplar
#   /home/alex/framepfx-mojo/src/keyframe_editor.mojo :
#     - to_cstr (lines 37-43): as_bytes() -> alloc[Int8](len+1) -> copy -> NUL terminate -> return.
#     - col/fill/line_t/fcircle/txt thin wrappers over external_call[...] (lines 46-63):
#         set_color / draw_filled_rectangle / draw_line / draw_filled_circle / draw_text.
#     - the frame_begin -> draw -> fpx_dump_fb -> frame_end verification harness (lines 252-257).
# Slider / drag mouse->value math mirrors property_editor.mojo NumberDragger handling
#   (property_editor.mojo:218-239 move_dragger: clamp(value, lo, hi)); here slider_float maps the
#   absolute mouse-x position across [x, x+w] to [vmin, vmax] (immediate-mode GUI SliderFloat behaviour) rather
#   than the relative-delta NumberDragger, because the task spec asks for "mouse x -> value".
# get_text_size two-out-pointer usage mirrors the documented backend signature
#   get_text_size(text, size, float* outW, float* outH) — used by text_width() below.
#
# NO `from math` import: the windowed FFI binary cannot link transcendentals (sincos@GLIBC); only
# builtin min/max/abs/Int()/Float32() arithmetic is used (same constraint noted in the exemplars).

from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from builtin.type_aliases import MutExternalOrigin


# ============================================================================
# C backend helpers  (mirrors keyframe_editor.mojo:37-63 exactly)
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
    # convenience 0..255 wrapper (mirrors keyframe_editor.mojo:46-47)
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


# get_text_size(const char*, float size, float* outW, float* outH) -> width in pixels.
# Two out-pointers, exactly as the documented backend signature. Returns (w, h).
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
# geometry helpers
# ============================================================================
def point_in_rect(px: Float32, py: Float32, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
    return px >= x and px < x + w and py >= y and py < y + h


def clampf(v: Float32, lo: Float32, hi: Float32) -> Float32:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


# ============================================================================
# UiContext — the immediate-mode core.
#
# Input model (one frame of state):
#   mouse_x, mouse_y : Int   current cursor position
#   mouse_down       : Bool  is the left button held THIS frame
#   prev_mouse_down  : Bool  was it held LAST frame (for edge detection)
#   mouse_clicked    : Bool  pressed-this-frame edge = mouse_down and not prev_mouse_down
#   mouse_released   : Bool  released-this-frame edge = prev_mouse_down and not mouse_down
#
# Interaction model (immediate-mode GUI style):
#   hot_id    : the widget currently hovered (recomputed each frame in begin_frame -> -1)
#   active_id : the widget that captured the mouse (e.g. a slider being dragged). Set on press
#               inside a widget, cleared on release. Persists across frames while held.
#
#   "clicked" for a button = press inside AND release inside the SAME widget (standard immediate-mode GUI
#   press+release semantics): on mouse_clicked over the widget we latch active_id; on
#   mouse_released, if active_id is still us AND the cursor is still inside, we report a click.
#
# Widget ids are EXPLICIT (passed by the caller per call-site) — never hashed from labels.
# ============================================================================
struct UiContext(Copyable, Movable):
    var mouse_x: Int
    var mouse_y: Int
    var mouse_down: Bool
    var prev_mouse_down: Bool
    var mouse_clicked: Bool
    var mouse_released: Bool
    var hot_id: Int
    var active_id: Int
    var open_combo: Int        # id of the currently open combo, or -1

    def __init__(out self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_down = False
        self.prev_mouse_down = False
        self.mouse_clicked = False
        self.mouse_released = False
        self.hot_id = -1
        self.active_id = -1
        self.open_combo = -1

    # ---- INJECTABLE input (mandatory for headless verification) ----
    # Set the input state DIRECTLY (synthetic). Derives the click/release edges from the
    # previous frame's down-state, then advances prev_mouse_down so the NEXT set_input
    # computes edges correctly. Call once per simulated frame.
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

    # ---- per-frame reset of hot tracking (active persists across frames) ----
    def begin_frame(mut self):
        self.hot_id = -1

    # ---- helpers ----
    def _mouse_in(self, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
        return point_in_rect(Float32(self.mouse_x), Float32(self.mouse_y), x, y, w, h)

    # ========================================================================
    # 1) text — a simple label (no interaction).
    # ========================================================================
    def text(self, s: String, x: Float32, y: Float32, size: Float32) raises:
        col(225, 225, 230)
        txt(s, x, y, size)

    # ========================================================================
    # 2) button — returns True the frame it is clicked (press inside + release inside).
    #    Hot when hovered; active while the press is latched on it.
    # ========================================================================
    def button(mut self, id: Int, label: String, x: Float32, y: Float32, w: Float32, h: Float32) raises -> Bool:
        var inside = self._mouse_in(x, y, w, h)
        if inside:
            self.hot_id = id
        # press latches active
        if inside and self.mouse_clicked:
            self.active_id = id
        var clicked = False
        # release: if we own the press and the cursor is still inside -> click
        if self.mouse_released:
            if self.active_id == id:
                if inside:
                    clicked = True
                self.active_id = -1

        # ---- draw: fill changes by state (idle / hot / active-pressed) ----
        if self.active_id == id and inside:
            col(70, 110, 165)        # pressed
        elif self.hot_id == id:
            col(80, 80, 92)          # hovered
        else:
            col(58, 58, 66)          # idle
        fill(x, y, w, h)
        col(0, 0, 0)
        rect(x, y, w, h)
        # centered label
        var m = measure_text(label, Float32(13.0))
        var tw = m[0]
        var th = m[1]
        var tx = x + (w - tw) / Float32(2.0)
        var ty = y + (h - th) / Float32(2.0)
        col(228, 228, 233)
        txt(label, tx, ty, Float32(13.0))
        return clicked

    # ========================================================================
    # 3) checkbox — toggles `value` on click, returns True if it changed.
    #    Click = press+release inside the box rect (square of side `box`).
    # ========================================================================
    def checkbox(mut self, id: Int, label: String, x: Float32, y: Float32, mut value: Bool) raises -> Bool:
        var box = Float32(18.0)
        var inside = self._mouse_in(x, y, box, box)
        if inside:
            self.hot_id = id
        if inside and self.mouse_clicked:
            self.active_id = id
        var changed = False
        if self.mouse_released:
            if self.active_id == id:
                if inside:
                    value = not value
                    changed = True
                self.active_id = -1

        # ---- draw box ----
        if self.hot_id == id:
            col(46, 46, 52)
        else:
            col(30, 30, 34)
        fill(x, y, box, box)
        col(0, 0, 0)
        rect(x, y, box, box)
        if value:
            col(90, 170, 250)
            fill(x + Float32(4.0), y + Float32(4.0), box - Float32(8.0), box - Float32(8.0))
        # label to the right of the box
        col(210, 210, 216)
        txt(label, x + box + Float32(6.0), y + Float32(3.0), Float32(12.0))
        return changed

    # ========================================================================
    # 4) radio — returns True if clicked (caller owns the group exclusivity).
    #    `active` = is this the selected option (filled inner dot).
    # ========================================================================
    def radio(mut self, id: Int, label: String, x: Float32, y: Float32, active: Bool) raises -> Bool:
        var d = Float32(18.0)
        var r = d / Float32(2.0)
        var cx = x + r
        var cy = y + r
        var inside = self._mouse_in(x, y, d, d)
        if inside:
            self.hot_id = id
        if inside and self.mouse_clicked:
            self.active_id = id
        var clicked = False
        if self.mouse_released:
            if self.active_id == id:
                if inside:
                    clicked = True
                self.active_id = -1

        # ---- draw outer ring (filled circle + darker inner punch-out) ----
        if self.hot_id == id:
            col(120, 120, 132)
        else:
            col(90, 90, 100)
        fcircle(cx, cy, r, 20)
        col(30, 30, 34)
        fcircle(cx, cy, r - Float32(2.0), 20)
        if active:
            col(90, 170, 250)
            fcircle(cx, cy, r - Float32(6.0), 20)
        col(210, 210, 216)
        txt(label, x + d + Float32(6.0), y + Float32(3.0), Float32(12.0))
        return clicked

    # ========================================================================
    # 5) slider_float — absolute mouse-x -> value mapping across [x, x+w].
    #    Drag (press inside, then hold) maps cursor x to [vmin, vmax]; returns True if changed.
    #    Mirrors property_editor.mojo clamp(value,lo,hi) (line 234/236); mapping is immediate-mode.
    # ========================================================================
    def slider_float(
        mut self,
        id: Int,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        mut value: Float32,
        vmin: Float32,
        vmax: Float32,
    ) -> Bool:
        var inside = self._mouse_in(x, y, w, h)
        if inside:
            self.hot_id = id
        if inside and self.mouse_clicked:
            self.active_id = id
        # while we own the slider AND the button is down, drag-track the value.
        var changed = False
        if self.active_id == id and self.mouse_down:
            var t = (Float32(self.mouse_x) - x) / w        # 0..1 across the track
            t = clampf(t, Float32(0.0), Float32(1.0))
            var nv = vmin + t * (vmax - vmin)
            if nv != value:
                value = nv
                changed = True
        # release relinquishes capture
        if self.mouse_released and self.active_id == id:
            self.active_id = -1

        # ---- draw track + handle ----
        col(28, 28, 32)
        fill(x, y, w, h)
        col(0, 0, 0)
        rect(x, y, w, h)
        var denom = vmax - vmin
        var frac = Float32(0.0)
        if denom != Float32(0.0):
            frac = (value - vmin) / denom
        frac = clampf(frac, Float32(0.0), Float32(1.0))
        # filled portion
        col(60, 100, 150)
        fill(x, y, w * frac, h)
        # handle
        var hw = Float32(8.0)
        var hx = x + frac * w - hw / Float32(2.0)
        hx = clampf(hx, x, x + w - hw)
        if self.active_id == id:
            col(160, 200, 255)
        else:
            col(200, 200, 210)
        fill(hx, y, hw, h)
        return changed

    # ========================================================================
    # 6) begin_child / end_child — vertical-scroll region.
    #    Returns the scroll OFFSET (>=0) that the caller should SUBTRACT from content y.
    #    The backend has no scissor primitive, so clipping is LOGICAL: this draws the panel
    #    background + a proportional scrollbar and returns the offset; CALLERS MUST CULL/offset
    #    their own content by the returned value (documented contract).
    #    Scroll is driven by dragging the scrollbar thumb (mouse_y within the track).
    #    Scroll position is stored in `scroll` (caller-owned Float32), updated in place.
    # ========================================================================
    def begin_child(
        mut self,
        id: Int,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        content_height: Float32,
        mut scroll: Float32,
    ) -> Float32:
        # panel background
        col(24, 24, 28)
        fill(x, y, w, h)
        col(0, 0, 0)
        rect(x, y, w, h)

        var max_scroll = content_height - h
        if max_scroll < Float32(0.0):
            max_scroll = Float32(0.0)

        # scrollbar geometry (right edge)
        var sb_w = Float32(12.0)
        var sb_x = x + w - sb_w
        var inside_sb = self._mouse_in(sb_x, y, sb_w, h)
        if inside_sb:
            self.hot_id = id
        if inside_sb and self.mouse_clicked:
            self.active_id = id

        if max_scroll > Float32(0.0):
            # thumb height proportional to visible fraction
            var visible_frac = h / content_height
            var thumb_h = h * visible_frac
            if thumb_h < Float32(16.0):
                thumb_h = Float32(16.0)
            var track = h - thumb_h
            # drag: map mouse_y on the track -> scroll
            if self.active_id == id and self.mouse_down:
                var ty = Float32(self.mouse_y) - y - thumb_h / Float32(2.0)
                var t = ty / track
                t = clampf(t, Float32(0.0), Float32(1.0))
                scroll = t * max_scroll
            scroll = clampf(scroll, Float32(0.0), max_scroll)
            # draw track + thumb
            col(34, 34, 40)
            fill(sb_x, y, sb_w, h)
            var thumb_y = y + (scroll / max_scroll) * track
            if self.active_id == id:
                col(140, 140, 155)
            else:
                col(90, 90, 102)
            fill(sb_x + Float32(2.0), thumb_y, sb_w - Float32(4.0), thumb_h)
        else:
            scroll = Float32(0.0)

        if self.mouse_released and self.active_id == id:
            self.active_id = -1
        return scroll

    def end_child(self):
        # No backend scissor to pop; present for API symmetry / future framing.
        pass

    # ========================================================================
    # 7) tooltip — small filled box + text near (x, y).
    # ========================================================================
    def tooltip(self, s: String, x: Float32, y: Float32) raises:
        var m = measure_text(s, Float32(12.0))
        var tw = m[0]
        var th = m[1]
        var pad = Float32(6.0)
        var bw = tw + pad * Float32(2.0)
        var bh = th + pad * Float32(2.0)
        var bx = x + Float32(14.0)
        var by = y + Float32(14.0)
        col(245, 240, 200)
        fill(bx, by, bw, bh)
        col(0, 0, 0)
        rect(bx, by, bw, bh)
        col(20, 20, 24)
        txt(s, bx + pad, by + pad, Float32(12.0))

    # ========================================================================
    # 8) combo — dropdown. Click the closed box to open; with the list open, clicking an
    #    item rect sets `selected` and closes. Open-state lives in ctx.open_combo (id or -1).
    #    Returns True if the selection changed this frame.
    # ========================================================================
    def combo(
        mut self,
        id: Int,
        x: Float32,
        y: Float32,
        w: Float32,
        items: List[String],
        mut selected: Int,
    ) raises -> Bool:
        var row_h = Float32(22.0)
        var inside_box = self._mouse_in(x, y, w, row_h)
        if inside_box:
            self.hot_id = id
        var changed = False

        # ---- toggle open on click of the closed box ----
        if inside_box and self.mouse_clicked:
            if self.open_combo == id:
                self.open_combo = -1
            else:
                self.open_combo = id

        # ---- draw the closed box ----
        if self.hot_id == id:
            col(42, 42, 48)
        else:
            col(30, 30, 34)
        fill(x, y, w, row_h)
        col(0, 0, 0)
        rect(x, y, w, row_h)
        var label = String("")
        if selected >= 0 and selected < len(items):
            label = items[selected].copy()
        col(225, 225, 230)
        txt(label, x + Float32(6.0), y + Float32(4.0), Float32(12.0))
        # down-arrow glyph
        var axx = x + w - Float32(14.0)
        var ayy = y + Float32(8.0)
        col(180, 180, 188)
        fill(axx, ayy, Float32(7.0), Float32(2.0))
        fill(axx + Float32(1.0), ayy + Float32(2.0), Float32(5.0), Float32(2.0))
        fill(axx + Float32(2.0), ayy + Float32(4.0), Float32(3.0), Float32(2.0))

        # ---- if open, draw the item list below and hit-test clicks ----
        if self.open_combo == id:
            var n = len(items)
            for i in range(n):
                var iy = y + row_h + Float32(i) * row_h
                var inside_item = self._mouse_in(x, iy, w, row_h)
                if inside_item:
                    self.hot_id = id
                # background (highlight hovered / selected)
                if inside_item:
                    col(70, 110, 165)
                elif i == selected:
                    col(48, 60, 80)
                else:
                    col(34, 34, 40)
                fill(x, iy, w, row_h)
                col(0, 0, 0)
                rect(x, iy, w, row_h)
                col(225, 225, 230)
                txt(items[i].copy(), x + Float32(6.0), iy + Float32(4.0), Float32(12.0))
                # click selects + closes
                if inside_item and self.mouse_clicked:
                    if selected != i:
                        selected = i
                        changed = True
                    self.open_combo = -1
        return changed


# ============================================================================
# SELF-TEST  (compiled + run headlessly by the MAIN LOOP — do NOT run it yourself)
#
# Each `set_input` call simulates one input frame; widgets are then invoked. For press+release
# widgets we need two frames: frame 1 = down inside (latch active), frame 2 = up inside (fire).
# Every widget issues draw_* calls so the framebuffer readback sees pixels.
# ============================================================================
def main() raises:
    var ctx = UiContext()
    var pass_all = True

    # ----- 1) BUTTON: press inside (down) then release inside (up) -> click on the UP frame -----
    # button rect = (100, 100, 120, 40); center = (160, 120). Mouse placed at (160, 120).
    # frame 1: down  -> latches active_id, returns False.
    ctx.begin_frame()
    ctx.set_input(160, 120, True)
    var c1 = ctx.button(1, String("OK"), Float32(100.0), Float32(100.0), Float32(120.0), Float32(40.0))
    # frame 2: up (still inside) -> release fires click = True.
    ctx.begin_frame()
    ctx.set_input(160, 120, False)
    var c2 = ctx.button(1, String("OK"), Float32(100.0), Float32(100.0), Float32(120.0), Float32(40.0))
    print("BTN_CLICK=", c2)              # EXPECT: BTN_CLICK= True
    if c1 != False:
        pass_all = False
    if c2 != True:
        pass_all = False

    # ----- 2) SLIDER: inject mouse at 25% / 50% / 75% of width; map [0,100] -----
    # slider rect = (100, 200, 200, 24). x in [100, 300], range [0,100].
    #   25% -> mouse_x = 100 + 0.25*200 = 150 -> value 25.0
    #   50% -> mouse_x = 100 + 0.50*200 = 200 -> value 50.0
    #   75% -> mouse_x = 100 + 0.75*200 = 250 -> value 75.0
    var sval = Float32(0.0)
    var sx = Float32(100.0)
    var sy = Float32(200.0)
    var sw = Float32(200.0)
    var sh = Float32(24.0)

    # 25%: need active_id captured -> press inside on this frame (mouse_clicked True) then drag.
    ctx.begin_frame()
    ctx.set_input(150, 212, True)        # click+down inside @ 25%
    _ = ctx.slider_float(2, sx, sy, sw, sh, sval, Float32(0.0), Float32(100.0))
    print("SLIDER_25=", sval)            # EXPECT: SLIDER_25= 25.0
    if sval != Float32(25.0):
        pass_all = False

    # 50%: still held (active), move to mouse_x=200.
    ctx.begin_frame()
    ctx.set_input(200, 212, True)
    _ = ctx.slider_float(2, sx, sy, sw, sh, sval, Float32(0.0), Float32(100.0))
    print("SLIDER_50=", sval)            # EXPECT: SLIDER_50= 50.0
    if sval != Float32(50.0):
        pass_all = False

    # 75%: still held, move to mouse_x=250.
    ctx.begin_frame()
    ctx.set_input(250, 212, True)
    _ = ctx.slider_float(2, sx, sy, sw, sh, sval, Float32(0.0), Float32(100.0))
    print("SLIDER_75=", sval)            # EXPECT: SLIDER_75= 75.0
    if sval != Float32(75.0):
        pass_all = False
    # release to clear capture
    ctx.begin_frame()
    ctx.set_input(250, 212, False)
    _ = ctx.slider_float(2, sx, sy, sw, sh, sval, Float32(0.0), Float32(100.0))

    # ----- 3) CHECKBOX: toggle twice (each toggle = a down frame + an up frame) -----
    # checkbox box = (100, 300, 18, 18). center inside @ (105, 305).
    var cbval = False

    # toggle #1: down then up -> True
    ctx.begin_frame()
    ctx.set_input(105, 305, True)
    _ = ctx.checkbox(3, String("Enabled"), Float32(100.0), Float32(300.0), cbval)
    ctx.begin_frame()
    ctx.set_input(105, 305, False)
    _ = ctx.checkbox(3, String("Enabled"), Float32(100.0), Float32(300.0), cbval)
    print("CHECK_1=", cbval)             # EXPECT: CHECK_1= True
    if cbval != True:
        pass_all = False

    # toggle #2: down then up -> back to False
    ctx.begin_frame()
    ctx.set_input(105, 305, True)
    _ = ctx.checkbox(3, String("Enabled"), Float32(100.0), Float32(300.0), cbval)
    ctx.begin_frame()
    ctx.set_input(105, 305, False)
    _ = ctx.checkbox(3, String("Enabled"), Float32(100.0), Float32(300.0), cbval)
    print("CHECK_2=", cbval)             # EXPECT: CHECK_2= False
    if cbval != False:
        pass_all = False

    # ----- summary line the main loop gates on -----
    if pass_all:
        print("CORE_WIDGETS_SELFTEST: pass (btn=True, slider 25/50/75 = 25.0/50.0/75.0, checkbox True->False)")
    else:
        print("CORE_WIDGETS_SELFTEST: fail")
