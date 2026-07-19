# fade.mojo — per-clip FADE in/out for the MojoMedia NLE, Mojo 1.0.0b1.
#
# Two pieces:
#   1) fade_opacity(local_frame, clip_len, fade_in, fade_out) -> Float32 in [0,1]
#      Pure math (NO GL): a linear ramp 0->1 over the first `fade_in` frames, flat 1.0 in
#      the middle, a linear ramp 1->0 over the last `fade_out` frames. Clamped to [0,1].
#      fade_in==0 / fade_out==0 disable that side (no ramp); both 0 => 1.0 everywhere.
#   2) fade_handles(ui, id0, clip_x, clip_y, clip_w, clip_h, mut fade_in_px, mut fade_out_px, draw)
#      Two draggable corner handles on the top edge of the clip rect. The LEFT handle sets the
#      fade-in width (px from the left edge); the RIGHT handle sets the fade-out width (px from
#      the right edge). Returns True if either handle moved this frame.
#
# This file is SELF-CONTAINED: its own FFI helpers + a local UiContext (the SAME input model as
# ui/widgets/core_widgets.mojo: mouse_x/mouse_y:Int, mouse_down/prev_mouse_down/mouse_clicked/
# mouse_released:Bool, hot_id/active_id:Int, set_input edge derivation, begin_frame hot reset).
# It adds press_x as a RELATIVE-drag anchor (mirrors widget_variants.VCtx.press_x / drag_core).
#
# The `draw: Bool` field/param follows media_widgets.Toolbar EXACTLY: draw=False skips ALL
# external_call/GL so the self-test runs HEADLESS. All numeric conversions are explicit; no
# `from math` (the windowed FFI binary cannot link transcendentals). The self-test in main()
# drives the handle with set_input and prints 'FADE_SELFTEST: pass' or '...: fail'.

from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from builtin.type_aliases import MutExternalOrigin


# ============================================================================
# C backend helpers (mirror core_widgets.mojo:32-69 — used only when draw=True).
# ============================================================================
def to_cstr(s: String) -> UnsafePointer[Int8, MutExternalOrigin]:
    var b = s.as_bytes()
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(b[i])
    buf[len(b)] = 0
    return buf


def col(r: Int, g: Int, b: Int):
    _ = external_call["set_color", Int32](
        Float32(r) / 255.0, Float32(g) / 255.0, Float32(b) / 255.0, Float32(1.0)
    )


def fill(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_filled_rectangle", Int32](x, y, w, h)


def rect(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_rectangle", Int32](x, y, w, h)


def line_t(x1: Float32, y1: Float32, x2: Float32, y2: Float32, thick: Float32):
    _ = external_call["draw_line", Int32](x1, y1, x2, y2, thick)


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


# ============================================================================
# UiContext — local copy of the core_widgets immediate-mode core, plus a press_x drag anchor
# (widget_variants.VCtx style) so the handles can do RELATIVE dragging without snapping.
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
    var press_x: Int        # cursor x captured when a handle latched active_id (drag anchor)

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

    # Inject one frame of synthetic input; derive click/release edges from the previous frame.
    def set_input(mut self, mx: Int, my: Int, down: Bool):
        self.mouse_clicked = down and not self.prev_mouse_down
        self.mouse_released = self.prev_mouse_down and not down
        self.mouse_x = mx
        self.mouse_y = my
        self.mouse_down = down
        self.prev_mouse_down = down

    # Live input from the backend (production path; not used by gated logic).
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


# ============================================================================
# fade_opacity — the per-frame opacity envelope. PURE math, no GL, no allocation.
#
#   local_frame : 0-based frame index within the clip (0 .. clip_len-1).
#   clip_len    : total clip length in frames (>= 1 for a sane clip).
#   fade_in     : ramp-up length in frames (0 disables the fade-in).
#   fade_out    : ramp-down length in frames (0 disables the fade-out).
#
# Envelope:
#   in-ramp : at local_frame f in [0, fade_in), alpha = f / fade_in    (0 at f=0, ->1)
#   out-ramp: let d = (clip_len-1) - f  (frames remaining to the last frame). For d in
#             [0, fade_out), alpha = d / fade_out  (0 at the LAST frame, ->1 moving inward).
#   middle  : 1.0.
# The final alpha is the MIN of the in and out factors (so overlapping fades both apply), then
# clamped to [0,1]. With fade_in==0 the in-factor is forced 1.0; same for fade_out.
# ============================================================================
def fade_opacity(local_frame: Int, clip_len: Int, fade_in: Int, fade_out: Int) -> Float32:
    if clip_len <= 0:
        return Float32(0.0)
    # clamp the query frame into the clip
    var f = local_frame
    if f < 0:
        f = 0
    if f > clip_len - 1:
        f = clip_len - 1

    # ---- fade-in factor ----
    var a_in = Float32(1.0)
    if fade_in > 0:
        if f < fade_in:
            a_in = Float32(f) / Float32(fade_in)
        else:
            a_in = Float32(1.0)

    # ---- fade-out factor (distance to the last frame) ----
    var a_out = Float32(1.0)
    if fade_out > 0:
        var d = (clip_len - 1) - f
        if d < fade_out:
            a_out = Float32(d) / Float32(fade_out)
        else:
            a_out = Float32(1.0)

    var a = a_in
    if a_out < a:
        a = a_out
    return clampf(a, Float32(0.0), Float32(1.0))


# ============================================================================
# fade_handles — two draggable corner handles on the clip's TOP edge.
#
#   id0          : base widget id; the left handle uses id0, the right uses id0+1.
#   clip_x/y/w/h : the clip rect on the timeline.
#   fade_in_px   : (mut) current fade-in width in px from the LEFT edge.  Dragging the left
#                  handle right grows it; clamped to [0, clip_w].
#   fade_out_px  : (mut) current fade-out width in px from the RIGHT edge. Dragging the right
#                  handle left grows it; clamped to [0, clip_w].
#   draw         : False => layout/hit-test only, NO GL (headless self-test). True => paint.
#
# RELATIVE drag (widget_variants.drag_core): on press inside a handle we latch active_id and
# record press_x; while held, the px width changes by (mouse_x - press_x), then press_x advances
# so motion is not double-counted. Returns True if either width changed this frame.
# ============================================================================
def fade_handles(
    mut ui: UiContext,
    id0: Int,
    clip_x: Float32,
    clip_y: Float32,
    clip_w: Float32,
    clip_h: Float32,
    mut fade_in_px: Float32,
    mut fade_out_px: Float32,
    draw: Bool,
) raises -> Bool:
    var hs = Float32(10.0)          # handle hit/draw size (square)
    var moved = False

    # current handle anchor x positions along the top edge
    var in_x = clip_x + clampf(fade_in_px, Float32(0.0), clip_w)
    var out_x = clip_x + clip_w - clampf(fade_out_px, Float32(0.0), clip_w)

    var left_id = id0
    var right_id = id0 + 1

    # ---- LEFT handle (fade-in): centered on (in_x, clip_y) along the top edge ----
    var lx = in_x - hs / Float32(2.0)
    var ly = clip_y - hs / Float32(2.0)
    var l_inside = ui._mouse_in(lx, ly, hs, hs)
    if l_inside:
        ui.hot_id = left_id
    if l_inside and ui.mouse_clicked:
        ui.active_id = left_id
        ui.press_x = ui.mouse_x
    if ui.active_id == left_id and ui.mouse_down:
        var dx = Float32(ui.mouse_x - ui.press_x)
        if dx != Float32(0.0):
            var nv = clampf(fade_in_px + dx, Float32(0.0), clip_w)
            if nv != fade_in_px:
                fade_in_px = nv
                moved = True
            ui.press_x = ui.mouse_x        # consumed -> advance anchor
    if ui.mouse_released and ui.active_id == left_id:
        ui.active_id = -1

    # ---- RIGHT handle (fade-out): centered on (out_x, clip_y); dragging LEFT grows it ----
    var rx = out_x - hs / Float32(2.0)
    var ry = clip_y - hs / Float32(2.0)
    var r_inside = ui._mouse_in(rx, ry, hs, hs)
    if r_inside:
        ui.hot_id = right_id
    if r_inside and ui.mouse_clicked:
        ui.active_id = right_id
        ui.press_x = ui.mouse_x
    if ui.active_id == right_id and ui.mouse_down:
        var dxr = Float32(ui.mouse_x - ui.press_x)
        if dxr != Float32(0.0):
            # moving the cursor left (dxr < 0) increases the fade-out width
            var nvr = clampf(fade_out_px - dxr, Float32(0.0), clip_w)
            if nvr != fade_out_px:
                fade_out_px = nvr
                moved = True
            ui.press_x = ui.mouse_x
    if ui.mouse_released and ui.active_id == right_id:
        ui.active_id = -1

    # ---- draw (skipped entirely when headless) ----
    if draw:
        # recompute anchors after any drag this frame so the visuals track the values
        var fin = clampf(fade_in_px, Float32(0.0), clip_w)
        var fout = clampf(fade_out_px, Float32(0.0), clip_w)
        var ax = clip_x + fin
        var bx = clip_x + clip_w - fout
        var top = clip_y
        var bot = clip_y + clip_h
        # fade-in wedge: hypotenuse from bottom-left up to (ax, top)
        col(120, 170, 230)
        line_t(clip_x, bot, ax, top, Float32(1.5))
        # fade-out wedge: hypotenuse from (bx, top) down to bottom-right
        line_t(bx, top, clip_x + clip_w, bot, Float32(1.5))

        # left handle box
        if ui.active_id == left_id:
            col(160, 200, 255)
        elif ui.hot_id == left_id:
            col(140, 180, 235)
        else:
            col(100, 150, 215)
        fill(ax - Float32(5.0), top - Float32(5.0), Float32(10.0), Float32(10.0))
        col(0, 0, 0)
        rect(ax - Float32(5.0), top - Float32(5.0), Float32(10.0), Float32(10.0))

        # right handle box
        if ui.active_id == right_id:
            col(160, 200, 255)
        elif ui.hot_id == right_id:
            col(140, 180, 235)
        else:
            col(100, 150, 215)
        fill(bx - Float32(5.0), top - Float32(5.0), Float32(10.0), Float32(10.0))
        col(0, 0, 0)
        rect(bx - Float32(5.0), top - Float32(5.0), Float32(10.0), Float32(10.0))

    return moved


# ============================================================================
# SELF-TEST  (HEADLESS — no GL; draw=False + pure-logic asserts). Run by the MAIN LOOP.
# Each expected number is derived from first principles in the comment beside it.
# ============================================================================
def main() raises:
    var pass_all = True

    # ----- 1) fade_opacity envelope -----
    # clip_len=100, fade_in=10, fade_out=10. Frame indices 0..99.
    var cl = 100
    var fin = 10
    var fout = 10

    # at local_frame 0 (with fade_in>0): a_in = 0/10 = 0.0 -> alpha 0.0
    var a0 = fade_opacity(0, cl, fin, fout)
    print("FADE op@0=", a0)               # EXPECT 0.0
    if a0 != Float32(0.0):
        pass_all = False

    # middle (frame 50): both factors saturate to 1.0 -> alpha ~1.0
    var amid = fade_opacity(50, cl, fin, fout)
    print("FADE op@mid=", amid)           # EXPECT 1.0
    if amid != Float32(1.0):
        pass_all = False

    # last frame (frame 99, with fade_out>0): d = 99-99 = 0 -> a_out = 0/10 = 0.0 -> alpha 0.0
    var alast = fade_opacity(cl - 1, cl, fin, fout)
    print("FADE op@last=", alast)         # EXPECT 0.0
    if alast != Float32(0.0):
        pass_all = False

    # halfway up the in-ramp (frame 5): a_in = 5/10 = 0.5
    var ahalf = fade_opacity(5, cl, fin, fout)
    print("FADE op@5=", ahalf)            # EXPECT 0.5
    if ahalf != Float32(0.5):
        pass_all = False

    # both fades disabled -> exactly 1.0 EVERYWHERE (first, mid, last frame).
    var z0 = fade_opacity(0, cl, 0, 0)
    var zmid = fade_opacity(50, cl, 0, 0)
    var zlast = fade_opacity(cl - 1, cl, 0, 0)
    print("FADE op_nofade 0/mid/last=", z0, " ", zmid, " ", zlast)   # EXPECT 1.0 1.0 1.0
    if z0 != Float32(1.0):
        pass_all = False
    if zmid != Float32(1.0):
        pass_all = False
    if zlast != Float32(1.0):
        pass_all = False

    # ----- 2) fade_handles drag (HEADLESS: draw=False) -----
    # clip rect = (100, 200, 300, 40). Initial fade_in_px = 0 -> left handle anchored at
    # (clip_x + 0, clip_y) = (100, 200), hit box [95,105) x [195,205). fade_out_px = 0 ->
    # right handle at (clip_x + clip_w, clip_y) = (400, 200), hit box [395,405) x [195,205).
    var ui = UiContext()
    var clip_x = Float32(100.0)
    var clip_y = Float32(200.0)
    var clip_w = Float32(300.0)
    var clip_h = Float32(40.0)
    var fin_px = Float32(0.0)
    var fout_px = Float32(0.0)

    # frame A: press the LEFT handle at its center (100, 200) -> latches active, press_x=100.
    ui.begin_frame()
    ui.set_input(100, 200, True)
    var m0 = fade_handles(ui, 700, clip_x, clip_y, clip_w, clip_h, fin_px, fout_px, False)
    # frame B: still held, drag the cursor RIGHT to x=130 -> dx=+30 -> fade_in_px 0 -> 30.
    ui.begin_frame()
    ui.set_input(130, 200, True)
    var m1 = fade_handles(ui, 700, clip_x, clip_y, clip_w, clip_h, fin_px, fout_px, False)
    print("FADE drag fin_px=", fin_px, " moved=", m1)   # EXPECT fin_px= 30.0  moved= True
    if fin_px != Float32(30.0):
        pass_all = False
    if m1 != True:
        pass_all = False
    if m0 != False:                                     # the press frame alone moves nothing
        pass_all = False
    # release to clear capture
    ui.begin_frame()
    ui.set_input(130, 200, False)
    _ = fade_handles(ui, 700, clip_x, clip_y, clip_w, clip_h, fin_px, fout_px, False)

    # ----- summary line the main loop gates on -----
    if pass_all:
        print("FADE_SELFTEST: pass (op 0/0.5/1/0, no-fade=1.0, drag fin_px 0->30)")
    else:
        print("FADE_SELFTEST: fail")
