# color_picker.mojo — WORKSTREAM: COLOR PICKER (NO runtime trig) for Mojo 1.0.0b1, drawn on the
# MojoGUI C rendering backend (librender.so) via external_call ONLY. No backend change.
#
# Public widgets (immediate-mode, injected mouse — NEVER calls get_mouse_* in the gated path):
#   color_picker(ctx,id,x,y, mut h, mut s, mut v):
#       SV SQUARE (x-axis=saturation 0..1, y-axis=value 1..0 top->bottom, filled by sampling a grid
#       of draw_filled_rectangle cells) + a vertical HUE BAR beside it (gradient of cells) + a
#       preview swatch. Dragging inside the square sets (s,v); dragging the hue bar sets h.
#   color_edit(ctx,id,x,y, mut r, mut g, mut b):
#       3 RGB drag rows (drag a row horizontally -> 0..1 channel) + a hex label "#RRGGBB".
#
# Pure color math (NO trig — piecewise-linear only, see the windowed-FFI link constraint below):
#   hsv_to_rgb(h_deg,s,v) -> Tuple[Float32,Float32,Float32]   (piecewise-linear sectors)
#   rgb_to_hsv(r,g,b)     -> Tuple[Float32,Float32,Float32]   (max/min, no trig)
#
# FFI idioms mirrored EXACTLY from the framebuffer-verified exemplar
#   /home/alex/framepfx-mojo/src/keyframe_editor.mojo (to_cstr 37-43; col/fill/line_t/fcircle/txt
#   46-63) and the sibling toolkits /tmp/mojoui/core_widgets.mojo (UiContext input model:
#   mouse_x/mouse_y:Int, mouse_down/prev_mouse_down/mouse_clicked/mouse_released:Bool; set_input
#   edge derivation; point_in_rect/clampf; press+drag active_id capture) and
#   /tmp/mojoui2/widget_pack.mojo (drag rows). This module defines its OWN context (PickerCtx)
#   carrying the SAME field shape so it interoperates with those UiContext concepts.
#
# NO `from math` import: the windowed FFI binary cannot link transcendentals (sincos@GLIBC).
# HSV is computed with the PIECEWISE-LINEAR formula (abs/min/max/Float32 arithmetic only) — NO
# sinf/cosf/pow. All geometry is Float32. (Same constraint proven by every exemplar.)

from std.ffi import external_call
from std.memory import UnsafePointer, alloc


# ============================================================================
# C backend helpers  (mirror keyframe_editor.mojo:37-63 / core_widgets.mojo:32-82 EXACTLY)
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


def absf(v: Float32) -> Float32:
    if v < Float32(0.0):
        return -v
    return v


def maxf(a: Float32, b: Float32) -> Float32:
    if a > b:
        return a
    return b


def minf(a: Float32, b: Float32) -> Float32:
    if a < b:
        return a
    return b


# ============================================================================
# COLOR MATH — piecewise-linear, NO trig.
# ============================================================================

# hsv_to_rgb(h_deg, s, v): standard piecewise-linear HSV->RGB (Smith 1978).
#   chroma  c  = v * s
#   hp = h/60 wrapped into [0,6)        (which of 6 linear sectors)
#   x  = c * (1 - |hp mod 2 - 1|)       (the linearly-rising/falling secondary)
#   m  = v - c                          (the achromatic floor added to all channels)
# Sector (by floor(hp)):
#   0:(c,x,0) 1:(x,c,0) 2:(0,c,x) 3:(0,x,c) 4:(x,0,c) 5:(c,0,x) ; then add m to each.
# NO trig, NO pow — only abs/min/max/Int(floor for non-negatives)/Float32 arithmetic.
def hsv_to_rgb(h_deg: Float32, s: Float32, v: Float32) -> Tuple[Float32, Float32, Float32]:
    var hh = h_deg
    # wrap hue into [0,360): cheap loops (hue is small, no fmod/trig).
    while hh >= Float32(360.0):
        hh = hh - Float32(360.0)
    while hh < Float32(0.0):
        hh = hh + Float32(360.0)
    var ss = clampf(s, Float32(0.0), Float32(1.0))
    var vv = clampf(v, Float32(0.0), Float32(1.0))

    var c = vv * ss
    var hp = hh / Float32(60.0)                       # in [0,6)
    var sector = Int(hp)                              # floor for hp>=0
    # hp mod 2 in [0,2): subtract the even part (sector & ~1 == sector for even, sector-1 for odd)
    var hmod2 = hp - Float32((sector // 2) * 2)
    var x = c * (Float32(1.0) - absf(hmod2 - Float32(1.0)))
    var m = vv - c

    var r1 = Float32(0.0)
    var g1 = Float32(0.0)
    var b1 = Float32(0.0)
    if sector == 0:
        r1 = c
        g1 = x
        b1 = Float32(0.0)
    elif sector == 1:
        r1 = x
        g1 = c
        b1 = Float32(0.0)
    elif sector == 2:
        r1 = Float32(0.0)
        g1 = c
        b1 = x
    elif sector == 3:
        r1 = Float32(0.0)
        g1 = x
        b1 = c
    elif sector == 4:
        r1 = x
        g1 = Float32(0.0)
        b1 = c
    else:
        # sector == 5 (and the hp==6.0 edge cannot occur after wrap into [0,360))
        r1 = c
        g1 = Float32(0.0)
        b1 = x

    return Tuple[Float32, Float32, Float32](r1 + m, g1 + m, b1 + m)


# rgb_to_hsv(r,g,b): inverse (max/min based). NO trig.
#   mx=max(r,g,b), mn=min(r,g,b), d=mx-mn ; v=mx ; s=d/mx (0 if mx==0).
#   hue (degrees, wrapped into [0,360)):
#     d==0 -> 0
#     mx==r -> 60*((g-b)/d) , then if negative add 360 (the r-sector wraps through 0/360)
#     mx==g -> 60*((b-r)/d + 2)
#     mx==b -> 60*((r-g)/d + 4)
def rgb_to_hsv(r: Float32, g: Float32, b: Float32) -> Tuple[Float32, Float32, Float32]:
    var mx = maxf(r, maxf(g, b))
    var mn = minf(r, minf(g, b))
    var d = mx - mn

    var v = mx
    var s = Float32(0.0)
    if mx > Float32(0.0):
        s = d / mx

    var h = Float32(0.0)
    if d > Float32(0.0):
        if mx == r:
            var hp = (g - b) / d                      # may be negative (sector 5 region)
            if hp < Float32(0.0):
                hp = hp + Float32(6.0)
            h = Float32(60.0) * hp
        elif mx == g:
            h = Float32(60.0) * ((b - r) / d + Float32(2.0))
        else:
            h = Float32(60.0) * ((r - g) / d + Float32(4.0))
    # wrap into [0,360)
    while h >= Float32(360.0):
        h = h - Float32(360.0)
    while h < Float32(0.0):
        h = h + Float32(360.0)
    return Tuple[Float32, Float32, Float32](h, s, v)


# hex helper: "#RRGGBB" from 0..1 channels (no trig; integer rounding via +0.5 floor).
def to_hex_byte(c: Float32) -> Int:
    var cc = clampf(c, Float32(0.0), Float32(1.0))
    var n = Int(cc * Float32(255.0) + Float32(0.5))     # round-to-nearest
    if n < 0:
        n = 0
    if n > 255:
        n = 255
    return n


def hex_digit(n: Int) -> String:
    if n == 0:
        return String("0")
    elif n == 1:
        return String("1")
    elif n == 2:
        return String("2")
    elif n == 3:
        return String("3")
    elif n == 4:
        return String("4")
    elif n == 5:
        return String("5")
    elif n == 6:
        return String("6")
    elif n == 7:
        return String("7")
    elif n == 8:
        return String("8")
    elif n == 9:
        return String("9")
    elif n == 10:
        return String("A")
    elif n == 11:
        return String("B")
    elif n == 12:
        return String("C")
    elif n == 13:
        return String("D")
    elif n == 14:
        return String("E")
    return String("F")


def byte_to_hex2(n: Int) -> String:
    var hi = n // 16
    var lo = n % 16
    return hex_digit(hi) + hex_digit(lo)


def rgb_to_hex(r: Float32, g: Float32, b: Float32) -> String:
    var rh = byte_to_hex2(to_hex_byte(r))
    var gh = byte_to_hex2(to_hex_byte(g))
    var bh = byte_to_hex2(to_hex_byte(b))
    return String("#") + rh + gh + bh


# ============================================================================
# PickerCtx — the immediate-mode context (SAME shape as core_widgets UiContext).
#
# Input model (one frame of state), mirrors core_widgets.mojo:121-141:
#   mouse_x/mouse_y:Int  mouse_down/prev_mouse_down/mouse_clicked/mouse_released:Bool
#   hot_id/active_id:Int
# active_id captures a widget on press (mouse_clicked inside) and is held while dragging until
# release, exactly like the slider/drag in the exemplars. INJECTABLE: set_input drives it; the
# gated logic NEVER calls get_mouse_* (pump_live_input is the separate live-binding path).
# ============================================================================
struct PickerCtx(Copyable, Movable):
    var mouse_x: Int
    var mouse_y: Int
    var mouse_down: Bool
    var prev_mouse_down: Bool
    var mouse_clicked: Bool
    var mouse_released: Bool
    var hot_id: Int
    var active_id: Int

    def __init__(out self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_down = False
        self.prev_mouse_down = False
        self.mouse_clicked = False
        self.mouse_released = False
        self.hot_id = -1
        self.active_id = -1

    # ---- INJECTABLE input (mandatory for headless verification) ----
    # mirrors core_widgets.mojo:147-153 — derive click/release edges from the prev frame, then
    # advance prev_mouse_down so the NEXT set_input computes edges correctly.
    def set_input(mut self, mx: Int, my: Int, down: Bool):
        self.mouse_clicked = down and not self.prev_mouse_down
        self.mouse_released = self.prev_mouse_down and not down
        self.mouse_x = mx
        self.mouse_y = my
        self.mouse_down = down
        self.prev_mouse_down = down

    # ---- LIVE input from the backend (production path; NOT used in the gated logic) ----
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

    def begin_frame(mut self):
        self.hot_id = -1

    def _mouse_in(self, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
        return point_in_rect(Float32(self.mouse_x), Float32(self.mouse_y), x, y, w, h)

    # ========================================================================
    # color_picker(id, x, y, mut h, mut s, mut v)
    #
    # Geometry (constants; documented mapping):
    #   SV SQUARE:  origin (x, y), side SQ=160. x-axis = saturation, y-axis = value INVERTED.
    #       s = (mouse_x - x) / SQ                 (left edge s=0 ; right edge s=1)
    #       v = 1 - (mouse_y - y) / SQ             (top edge  v=1 ; bottom edge v=0)
    #       => CENTER mouse (x+SQ/2, y+SQ/2) gives s=0.5, v=0.5.   [documented self-test mapping]
    #   HUE BAR:   origin (x + SQ + GAP, y), width HBW=22, height SQ. Vertical gradient:
    #       hue = ((mouse_y - y) / SQ) * 360       (top hue=0 ; bottom hue~360)
    #   PREVIEW SWATCH: below the square, current (h,s,v)->rgb fill.
    #
    # Each region is filled by a GRID of draw_filled_rectangle CELLS sampling the color
    # function (no per-pixel shader on this backend). Two ids are used:
    #   id      -> SV square drag capture
    #   id+1    -> hue bar drag capture
    # Returns True if any of (h,s,v) changed this frame.
    # ========================================================================
    def color_picker(
        mut self,
        id: Int,
        x: Float32,
        y: Float32,
        mut h: Float32,
        mut s: Float32,
        mut v: Float32,
    ) raises -> Bool:
        var SQ = Float32(160.0)
        var GAP = Float32(10.0)
        var HBW = Float32(22.0)
        var hue_id = id + 1
        var changed = False

        # ----- SV SQUARE interaction -----
        var in_sv = self._mouse_in(x, y, SQ, SQ)
        if in_sv:
            self.hot_id = id
        if in_sv and self.mouse_clicked:
            self.active_id = id
        if self.active_id == id and self.mouse_down:
            var ns = (Float32(self.mouse_x) - x) / SQ
            var nv = Float32(1.0) - (Float32(self.mouse_y) - y) / SQ
            ns = clampf(ns, Float32(0.0), Float32(1.0))
            nv = clampf(nv, Float32(0.0), Float32(1.0))
            if ns != s:
                s = ns
                changed = True
            if nv != v:
                v = nv
                changed = True
        if self.mouse_released and self.active_id == id:
            self.active_id = -1

        # ----- HUE BAR interaction -----
        var hbx = x + SQ + GAP
        var in_hue = self._mouse_in(hbx, y, HBW, SQ)
        if in_hue:
            self.hot_id = hue_id
        if in_hue and self.mouse_clicked:
            self.active_id = hue_id
        if self.active_id == hue_id and self.mouse_down:
            var t = (Float32(self.mouse_y) - y) / SQ
            t = clampf(t, Float32(0.0), Float32(1.0))
            var nh = t * Float32(360.0)
            if nh != h:
                h = nh
                changed = True
        if self.mouse_released and self.active_id == hue_id:
            self.active_id = -1

        # ===== DRAW: SV square as a grid of cells (sample hsv at each cell center) =====
        var cells = 16
        var cw = SQ / Float32(cells)
        for cy in range(cells):
            for cx in range(cells):
                # cell center -> (s,v); value inverted on the y axis (top = v 1).
                var cs = (Float32(cx) + Float32(0.5)) / Float32(cells)
                var cv = Float32(1.0) - (Float32(cy) + Float32(0.5)) / Float32(cells)
                var rgb = hsv_to_rgb(h, cs, cv)
                set_color(rgb[0], rgb[1], rgb[2], Float32(1.0))
                fill(x + Float32(cx) * cw, y + Float32(cy) * cw, cw + Float32(1.0), cw + Float32(1.0))
        # border + selection crosshair on the SV square
        col(0, 0, 0)
        rect(x, y, SQ, SQ)
        var sel_sx = x + clampf(s, Float32(0.0), Float32(1.0)) * SQ
        var sel_sy = y + (Float32(1.0) - clampf(v, Float32(0.0), Float32(1.0))) * SQ
        col(255, 255, 255)
        line_t(sel_sx - Float32(5.0), sel_sy, sel_sx + Float32(5.0), sel_sy, Float32(1.0))
        line_t(sel_sx, sel_sy - Float32(5.0), sel_sx, sel_sy + Float32(5.0), Float32(1.0))

        # ===== DRAW: HUE BAR as a vertical gradient of cells =====
        var hcells = 24
        var hch = SQ / Float32(hcells)
        for hy in range(hcells):
            var cell_hue = (Float32(hy) + Float32(0.5)) / Float32(hcells) * Float32(360.0)
            var hrgb = hsv_to_rgb(cell_hue, Float32(1.0), Float32(1.0))
            set_color(hrgb[0], hrgb[1], hrgb[2], Float32(1.0))
            fill(hbx, y + Float32(hy) * hch, HBW, hch + Float32(1.0))
        col(0, 0, 0)
        rect(hbx, y, HBW, SQ)
        # hue selection marker (horizontal white tick)
        var hy_sel = y + clampf(h / Float32(360.0), Float32(0.0), Float32(1.0)) * SQ
        col(255, 255, 255)
        line_t(hbx, hy_sel, hbx + HBW, hy_sel, Float32(2.0))

        # ===== DRAW: PREVIEW SWATCH (current full color) below the square =====
        var prgb = hsv_to_rgb(h, s, v)
        var swy = y + SQ + Float32(8.0)
        set_color(prgb[0], prgb[1], prgb[2], Float32(1.0))
        fill(x, swy, SQ, Float32(28.0))
        col(0, 0, 0)
        rect(x, swy, SQ, Float32(28.0))
        # hex label of the preview color
        col(235, 235, 240)
        txt(rgb_to_hex(prgb[0], prgb[1], prgb[2]), x + Float32(4.0), swy + Float32(8.0), Float32(12.0))

        return changed

    # ========================================================================
    # color_edit(id, x, y, mut r, mut g, mut b)
    #
    # 3 RGB DRAG ROWS (R/G/B). Each row is a horizontal track of width RW=180, height RH=20,
    # stacked GAPY=4 apart. Dragging a row maps mouse-x across the track to 0..1:
    #   channel = (mouse_x - x) / RW            (clamped 0..1)
    # ids: id (R), id+1 (G), id+2 (B). Below the rows: a swatch + a hex label "#RRGGBB".
    # Returns True if any channel changed.
    # ========================================================================
    def color_edit(
        mut self,
        id: Int,
        x: Float32,
        y: Float32,
        mut r: Float32,
        mut g: Float32,
        mut b: Float32,
    ) raises -> Bool:
        var RW = Float32(180.0)
        var RH = Float32(20.0)
        var GAPY = Float32(4.0)
        var changed = False

        # one row helper inlined 3x (R,G,B) so each owns its own id/value.
        # --- R row ---
        var ry = y
        var rid = id
        var in_r = self._mouse_in(x, ry, RW, RH)
        if in_r:
            self.hot_id = rid
        if in_r and self.mouse_clicked:
            self.active_id = rid
        if self.active_id == rid and self.mouse_down:
            var nv = clampf((Float32(self.mouse_x) - x) / RW, Float32(0.0), Float32(1.0))
            if nv != r:
                r = nv
                changed = True
        if self.mouse_released and self.active_id == rid:
            self.active_id = -1

        # --- G row ---
        var gy = y + (RH + GAPY)
        var gid = id + 1
        var in_g = self._mouse_in(x, gy, RW, RH)
        if in_g:
            self.hot_id = gid
        if in_g and self.mouse_clicked:
            self.active_id = gid
        if self.active_id == gid and self.mouse_down:
            var nv2 = clampf((Float32(self.mouse_x) - x) / RW, Float32(0.0), Float32(1.0))
            if nv2 != g:
                g = nv2
                changed = True
        if self.mouse_released and self.active_id == gid:
            self.active_id = -1

        # --- B row ---
        var by = y + Float32(2.0) * (RH + GAPY)
        var bid = id + 2
        var in_b = self._mouse_in(x, by, RW, RH)
        if in_b:
            self.hot_id = bid
        if in_b and self.mouse_clicked:
            self.active_id = bid
        if self.active_id == bid and self.mouse_down:
            var nv3 = clampf((Float32(self.mouse_x) - x) / RW, Float32(0.0), Float32(1.0))
            if nv3 != b:
                b = nv3
                changed = True
        if self.mouse_released and self.active_id == bid:
            self.active_id = -1

        # ===== DRAW the 3 rows (track bg + filled portion + handle + channel letter) =====
        self._draw_chan_row(String("R"), x, ry, RW, RH, r, Float32(0.78), Float32(0.20), Float32(0.20))
        self._draw_chan_row(String("G"), x, gy, RW, RH, g, Float32(0.20), Float32(0.70), Float32(0.20))
        self._draw_chan_row(String("B"), x, by, RW, RH, b, Float32(0.30), Float32(0.45), Float32(0.85))

        # ===== swatch + hex label =====
        var swy = by + RH + Float32(8.0)
        set_color(clampf(r, Float32(0.0), Float32(1.0)), clampf(g, Float32(0.0), Float32(1.0)),
                  clampf(b, Float32(0.0), Float32(1.0)), Float32(1.0))
        fill(x, swy, Float32(40.0), Float32(24.0))
        col(0, 0, 0)
        rect(x, swy, Float32(40.0), Float32(24.0))
        col(235, 235, 240)
        txt(rgb_to_hex(r, g, b), x + Float32(48.0), swy + Float32(6.0), Float32(13.0))

        return changed

    # draw a single channel drag row: dark track, colored filled portion, white handle, letter.
    def _draw_chan_row(
        self,
        letter: String,
        x: Float32,
        y: Float32,
        w: Float32,
        h: Float32,
        value: Float32,
        fr: Float32,
        fg: Float32,
        fb: Float32,
    ) raises:
        col(28, 28, 32)
        fill(x, y, w, h)
        var frac = clampf(value, Float32(0.0), Float32(1.0))
        set_color(fr, fg, fb, Float32(1.0))
        fill(x, y, w * frac, h)
        col(0, 0, 0)
        rect(x, y, w, h)
        # handle
        var hw = Float32(6.0)
        var hx = clampf(x + frac * w - hw / Float32(2.0), x, x + w - hw)
        col(235, 235, 240)
        fill(hx, y, hw, h)
        # channel letter on the left
        col(20, 20, 24)
        txt(letter, x + Float32(3.0), y + Float32(3.0), Float32(12.0))


# ============================================================================
# SELF-TEST  (compiled + run headlessly by the MAIN LOOP — do NOT run it yourself)
#
# Every print line is gated by the main loop; the EXACT expected number is in a comment,
# derived from first principles. Every visual element issues real draw_* calls (render gate).
# ============================================================================
def approx(a: Float32, b: Float32, tol: Float32) -> Bool:
    return absf(a - b) <= tol


def main() raises:
    var pass_all = True

    # ===================================================================
    # COLORPICKER_SELFTEST — pure hsv_to_rgb checks (piecewise-linear).
    #
    # Derivation (c=v*s, x=c*(1-|hp mod2 -1|), m=v-c ; sector=floor(h/60)):
    #   (0,1,1):   c=1, hp=0, sector0, x=1*(1-|0-1|)=0, m=0 -> (c,x,0)=(1,0,0)  RED
    #   (120,1,1): c=1, hp=2, sector2, x=1*(1-|0-1|)=0, m=0 -> (0,c,x)=(0,1,0)  GREEN
    #   (240,1,1): c=1, hp=4, sector4, x=1*(1-|0-1|)=0, m=0 -> (x,0,c)=(0,0,1)  BLUE
    #   (0,0,1):   c=0, hp=0, sector0, x=0,            m=1 -> (0,0,0)+1=(1,1,1) WHITE
    # ===================================================================
    var red = hsv_to_rgb(Float32(0.0), Float32(1.0), Float32(1.0))
    var grn = hsv_to_rgb(Float32(120.0), Float32(1.0), Float32(1.0))
    var blu = hsv_to_rgb(Float32(240.0), Float32(1.0), Float32(1.0))
    var wht = hsv_to_rgb(Float32(0.0), Float32(0.0), Float32(1.0))

    print("COLORPICKER_SELFTEST: red=(", red[0], ",", red[1], ",", red[2], ")")     # EXPECT red=( 1.0 , 0.0 , 0.0 )
    print("COLORPICKER_SELFTEST: green=(", grn[0], ",", grn[1], ",", grn[2], ")")   # EXPECT green=( 0.0 , 1.0 , 0.0 )
    print("COLORPICKER_SELFTEST: blue=(", blu[0], ",", blu[1], ",", blu[2], ")")    # EXPECT blue=( 0.0 , 0.0 , 1.0 )
    print("COLORPICKER_SELFTEST: white=(", wht[0], ",", wht[1], ",", wht[2], ")")   # EXPECT white=( 1.0 , 1.0 , 1.0 )

    if not (red[0] == Float32(1.0) and red[1] == Float32(0.0) and red[2] == Float32(0.0)):
        pass_all = False
    if not (grn[0] == Float32(0.0) and grn[1] == Float32(1.0) and grn[2] == Float32(0.0)):
        pass_all = False
    if not (blu[0] == Float32(0.0) and blu[1] == Float32(0.0) and blu[2] == Float32(1.0)):
        pass_all = False
    if not (wht[0] == Float32(1.0) and wht[1] == Float32(1.0) and wht[2] == Float32(1.0)):
        pass_all = False

    # ===================================================================
    # ROUND-TRIP hue: rgb_to_hsv(hsv_to_rgb(h,1,1)) hue within 1 deg for h in {0,60,120,180,240,300}.
    #
    # With s=v=1 -> c=1,m=0. Each fully-saturated primary/secondary maps back to its exact hue:
    #   h=0   -> rgb(1,0,0)  -> mx=r,d=1,hp=(0-0)/1=0   -> h=0
    #   h=60  -> rgb(1,1,0)  -> mx=r,d=1,hp=(1-0)/1=1   -> h=60
    #   h=120 -> rgb(0,1,0)  -> mx=g     -> 60*((0-0)/1+2)=120
    #   h=180 -> rgb(0,1,1)  -> mx=g     -> 60*((1-0)/1+2)=180
    #   h=240 -> rgb(0,0,1)  -> mx=b     -> 60*((0-0)/1+4)=240
    #   h=300 -> rgb(1,0,1)  -> mx=r,hp=(0-1)/1=-1 ->+6=5 -> h=300
    # All EXACT (tolerance 1 deg trivially satisfied).
    # ===================================================================
    var hues = List[Float32]()
    hues.append(Float32(0.0))
    hues.append(Float32(60.0))
    hues.append(Float32(120.0))
    hues.append(Float32(180.0))
    hues.append(Float32(240.0))
    hues.append(Float32(300.0))
    var rt_ok = True
    var hmax_err = Float32(0.0)
    for i in range(len(hues)):
        var hin = hues[i]
        var rgb = hsv_to_rgb(hin, Float32(1.0), Float32(1.0))
        var hsv = rgb_to_hsv(rgb[0], rgb[1], rgb[2])
        var err = absf(hsv[0] - hin)
        if err > Float32(180.0):
            err = Float32(360.0) - err          # circular distance (not needed here, safety)
        if err > hmax_err:
            hmax_err = err
        if err > Float32(1.0):
            rt_ok = False
    print("COLORPICKER_SELFTEST: roundtrip_hue_max_err=", hmax_err, " ok=", rt_ok)
    # EXPECT (print joins args with one space; the literal " ok=" adds its own leading space -> two spaces):
    # COLORPICKER_SELFTEST: roundtrip_hue_max_err= 0.0  ok= True
    if not rt_ok:
        pass_all = False

    # ===================================================================
    # SV-square DRAG to center -> s=0.5, v=0.5  (documented mapping).
    #
    # color_picker at (x=40,y=40). SQ=160 -> square spans x[40,200], y[40,200].
    # Center mouse = (40+80, 40+80) = (120,120).
    #   s = (120-40)/160 = 80/160 = 0.5
    #   v = 1 - (120-40)/160 = 1 - 0.5 = 0.5
    # Need active capture: frame 1 press at center latches active_id; the SAME frame the drag
    # branch runs (active_id==id AND mouse_down) -> sets s,v immediately.
    # ===================================================================
    var ctx = PickerCtx()
    var ph = Float32(200.0)      # arbitrary starting hue (drag must NOT change it here)
    var ps = Float32(0.0)
    var pv = Float32(0.0)
    ctx.begin_frame()
    ctx.set_input(120, 120, True)        # press+down at SV-square center
    _ = ctx.color_picker(10, Float32(40.0), Float32(40.0), ph, ps, pv)
    print("COLORPICKER_SELFTEST: sv_center s=", ps, " v=", pv)   # EXPECT s= 0.5  v= 0.5
    if ps != Float32(0.5):
        pass_all = False
    if pv != Float32(0.5):
        pass_all = False
    # hue must be untouched by an SV-square drag
    print("COLORPICKER_SELFTEST: sv_center hue_unchanged=", ph == Float32(200.0))  # EXPECT True
    if ph != Float32(200.0):
        pass_all = False
    # release to clear capture
    ctx.begin_frame()
    ctx.set_input(120, 120, False)
    _ = ctx.color_picker(10, Float32(40.0), Float32(40.0), ph, ps, pv)

    # ===================================================================
    # HUE-BAR DRAG: hue bar at x = 40 + SQ(160) + GAP(10) = 210, width 22, y[40,200].
    # Press at the VERTICAL MIDPOINT (mouse_y = 40 + 80 = 120), mouse_x = 210+11 = 221 (inside bar).
    #   t = (120-40)/160 = 0.5  -> h = 0.5*360 = 180.0
    # ===================================================================
    var ph2 = Float32(0.0)
    var ps2 = Float32(1.0)
    var pv2 = Float32(1.0)
    ctx.begin_frame()
    ctx.set_input(221, 120, True)        # press in hue bar at vertical midpoint
    _ = ctx.color_picker(10, Float32(40.0), Float32(40.0), ph2, ps2, pv2)
    print("COLORPICKER_SELFTEST: hue_mid h=", ph2)   # EXPECT h= 180.0
    if ph2 != Float32(180.0):
        pass_all = False
    ctx.begin_frame()
    ctx.set_input(221, 120, False)
    _ = ctx.color_picker(10, Float32(40.0), Float32(40.0), ph2, ps2, pv2)

    # ===================================================================
    # RENDER CHECK: hue bar TOP cell vs MID cell are DIFFERENT colors.
    #   hcells=24 ; cell hue = (i+0.5)/24*360.
    #   top  cell i=0  -> hue = 0.5/24*360  = 7.5   -> hsv_to_rgb(7.5,1,1) ~ red-orange  (r=1, g small, b=0)
    #   mid  cell i=12 -> hue = 12.5/24*360 = 187.5 -> hsv_to_rgb(187.5,1,1) ~ cyan-blue (r=0, g~0.875, b=1)
    # These differ markedly (different sectors entirely): assert NOT equal on at least one channel.
    # ===================================================================
    var top_hue = (Float32(0.0) + Float32(0.5)) / Float32(24.0) * Float32(360.0)
    var mid_hue = (Float32(12.0) + Float32(0.5)) / Float32(24.0) * Float32(360.0)
    var ctop = hsv_to_rgb(top_hue, Float32(1.0), Float32(1.0))
    var cmid = hsv_to_rgb(mid_hue, Float32(1.0), Float32(1.0))
    var cells_differ = (ctop[0] != cmid[0]) or (ctop[1] != cmid[1]) or (ctop[2] != cmid[2])
    print("COLORPICKER_SELFTEST: huebar top=(", ctop[0], ",", ctop[1], ",", ctop[2],
          ") mid=(", cmid[0], ",", cmid[1], ",", cmid[2], ") differ=", cells_differ)
    # EXPECT (verbatim, print joins every arg with one space; each "," is its own arg):
    # COLORPICKER_SELFTEST: huebar top=( 1.0 , 0.125 , 0.0 ) mid=( 0.0 , 0.875 , 1.0 ) differ= True
    #   top: hp=7.5/60=0.125 sector0 c=1 x=1*(1-|0.125-1|)=1*(1-0.875)=0.125 m=0 -> (1,0.125,0)
    #   mid: hp=187.5/60=3.125 sector3 c=1 x=1*(1-|(3.125-2)-1|)=1*(1-|1.125-1|)=1*(1-0.125)=0.875
    #        sector3=(0,x,c)=(0,0.875,1)
    if not cells_differ:
        pass_all = False

    # ===================================================================
    # color_edit DRAG: 3 RGB rows at (x=40,y=300), RW=180, RH=20, GAPY=4.
    #   R row y=300 ; G row y=324 ; B row y=348. Track x[40,220].
    # Drag R row to 25%: mouse_x = 40 + 0.25*180 = 85 ; mouse_y = 310 (inside R row).
    #   r = (85-40)/180 = 45/180 = 0.25
    # ===================================================================
    var er = Float32(0.0)
    var eg = Float32(0.0)
    var eb = Float32(0.0)
    ctx.begin_frame()
    ctx.set_input(85, 310, True)         # press+drag in R row at 25%
    _ = ctx.color_edit(30, Float32(40.0), Float32(300.0), er, eg, eb)
    print("COLORPICKER_SELFTEST: edit_R=", er)   # EXPECT edit_R= 0.25
    if er != Float32(0.25):
        pass_all = False
    ctx.begin_frame()
    ctx.set_input(85, 310, False)
    _ = ctx.color_edit(30, Float32(40.0), Float32(300.0), er, eg, eb)

    # hex of (1,0,0) == "#FF0000" ; of (0,1,0)=="#00FF00" — verify the hex builder.
    var hx_red = rgb_to_hex(Float32(1.0), Float32(0.0), Float32(0.0))
    var hx_grn = rgb_to_hex(Float32(0.0), Float32(1.0), Float32(0.0))
    print("COLORPICKER_SELFTEST: hex_red=", hx_red, " hex_green=", hx_grn)
    # EXPECT (two spaces before hex_green: inter-arg space + literal " hex_green="):
    # COLORPICKER_SELFTEST: hex_red= #FF0000  hex_green= #00FF00
    if hx_red != String("#FF0000"):
        pass_all = False
    if hx_grn != String("#00FF00"):
        pass_all = False

    # ----- summary line the main loop gates on -----
    if pass_all:
        print("COLOR_PICKER_SELFTEST: pass (red/green/blue/white exact, roundtrip hue err 0.0, "
              + "sv-center s=v=0.5, hue-mid=180.0, edit_R=0.25, hex #FF0000/#00FF00, huebar cells differ)")
    else:
        print("COLOR_PICKER_SELFTEST: fail")
