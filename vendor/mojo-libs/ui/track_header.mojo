# track_header.mojo — a track-header WIDGET row (V1 / A1 headers, like Resolve's), in pure
# Mojo 1.0.0b1 on the MojoGUI C rendering backend (librender.so) via external_call ONLY.
#
# A track header is the left-edge block of a timeline track: the track NAME plus a strip of
# small toggle buttons (Lock / Mute / Solo / Enable). Clicking a toggle flips its bool; the
# row returns True if ANY toggle changed this frame.
#
# Follows the media_widgets.Toolbar `draw: Bool` pattern EXACTLY: every external_call/GL call is
# gated behind `draw`, so passing draw=False runs the FULL hit-test + flip logic with ZERO GL —
# that is the headless self-test path. Hit-testing mirrors core_widgets.UiContext.button and
# media_widgets._tflat: press-IN latches active_id, release-IN over the SAME id fires the click.
#
# FFI idioms mirrored EXACTLY from the framebuffer-verified exemplars
#   core_widgets.mojo (to_cstr 32-38; col/fill/rect/txt 45-69; UiContext input model + button
#   press+release semantics 187-220) and media_widgets.mojo (Toolbar draw-gate 359-410).
#
# Self-contained: it carries its OWN to_cstr/col/fill/rect/txt helpers and its OWN UiContext
# (same public shape as core_widgets.UiContext: mouse_x/mouse_y, mouse_clicked/mouse_released,
# hot_id/active_id, set_input/begin_frame) so it compiles + self-tests standalone (the main loop
# cannot build a mojopkg). NO `from math` import (the windowed FFI binary cannot link sincos).

from std.ffi import external_call
from std.memory import UnsafePointer, alloc


# ============================================================================
# C backend helpers  (mirror core_widgets.mojo:32-69 EXACTLY) — only ever called when draw=True.
# ============================================================================
def to_cstr(s: String) -> UnsafePointer[Int8, MutExternalOrigin]:
    var b = s.as_bytes()
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(b[i])
    buf[len(b)] = 0
    return buf


def col(r: Int, g: Int, b: Int):
    # 0..255 convenience wrapper (core_widgets.mojo:45-49)
    _ = external_call["set_color", Int32](
        Float32(r) / 255.0, Float32(g) / 255.0, Float32(b) / 255.0, Float32(1.0)
    )


def fill(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_filled_rectangle", Int32](x, y, w, h)


def rect(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_rectangle", Int32](x, y, w, h)


def txt(s: String, x: Float32, y: Float32, size: Float32):
    _ = external_call["draw_text", Int32](to_cstr(s), x, y, size)


# ============================================================================
# geometry helpers (mirror core_widgets.mojo:88-89)
# ============================================================================
def point_in_rect(px: Float32, py: Float32, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
    return px >= x and px < x + w and py >= y and py < y + h


# ============================================================================
# UiContext — the immediate-mode core (same public shape as core_widgets.UiContext).
# Per-frame mouse state + hot_id/active_id with press+release click semantics, injectable via
# set_input for headless verification. Widget ids are EXPLICIT (passed per call-site).
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
    # Derives click/release edges from the previous frame's down-state, then advances
    # prev_mouse_down. Mirrors core_widgets.set_input EXACTLY. Call once per simulated frame.
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

    def _mouse_in(self, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
        return point_in_rect(Float32(self.mouse_x), Float32(self.mouse_y), x, y, w, h)


# ============================================================================
# toggle hit-test + draw — ONE small square toggle button.
#
# Hit-test mirrors UiContext.button (core_widgets.mojo:187-220) / media_widgets._tflat: press-IN
# latches active_id; on release, if we still own active_id AND cursor is still inside -> flip the
# bool and report changed. ALL drawing is gated behind `draw` (Toolbar pattern) so draw=False is a
# pure-logic path with no GL. `glyph` is a single letter drawn in the square; `on_col` tints the
# square when `value` is True so the toggle's state is visible.
# ============================================================================
def _toggle(
    mut ui: UiContext,
    id: Int,
    x: Float32,
    y: Float32,
    s: Float32,
    mut value: Bool,
    glyph: String,
    on_r: Int,
    on_g: Int,
    on_b: Int,
    draw: Bool,
) raises -> Bool:
    var inside = ui._mouse_in(x, y, s, s)
    if inside:
        ui.hot_id = id
    if inside and ui.mouse_clicked:
        ui.active_id = id
    var changed = False
    if ui.mouse_released and ui.active_id == id:
        if inside:
            value = not value
            changed = True
        ui.active_id = -1

    # ---- draw the square + glyph (idle / hot / pressed / ON-tint) ----
    if draw:
        var pressed = (ui.active_id == id) and inside
        if value:
            col(on_r, on_g, on_b)            # ON: state tint
        elif pressed:
            col(70, 110, 165)                # pressed
        elif ui.hot_id == id:
            col(60, 60, 70)                  # hovered
        else:
            col(40, 40, 46)                  # idle
        fill(x, y, s, s)
        col(0, 0, 0)
        rect(x, y, s, s)
        # single-letter glyph, roughly centered in the square
        if value:
            col(20, 20, 24)
        else:
            col(210, 210, 216)
        txt(glyph, x + s * Float32(0.28), y + s * Float32(0.18), Float32(11.0))
    return changed


# ============================================================================
# track_header — the public widget row.
#
#   [ name .................... ][L][M][S][E]
#
# Draws (when draw): a row background + the track name on the left, then 4 small toggle buttons
# (Lock, Mute, Solo, Enable) right-aligned. Clicking a toggle flips its bool. Uses 4 ids from id0
# (id0+0=Lock, +1=Mute, +2=Solo, +3=Enable). Hit-test mirrors UiContext.button (press-in +
# release-in) and works with draw=False (headless). Returns True if ANY toggle changed this frame.
# ============================================================================
def track_header(
    mut ui: UiContext,
    id0: Int,
    x: Float32,
    y: Float32,
    w: Float32,
    h: Float32,
    name: String,
    mut muted: Bool,
    mut locked: Bool,
    mut soloed: Bool,
    mut enabled: Bool,
    draw: Bool,
) raises -> Bool:
    # toggle geometry: 4 squares right-aligned, `bs` square with `pad` gap, vertically centered.
    var bs = h - Float32(8.0)
    if bs > Float32(18.0):
        bs = Float32(18.0)
    var pad = Float32(3.0)
    var by = y + (h - bs) / Float32(2.0)
    # rightmost square's x, then step LEFT for each: Lock, Mute, Solo, Enable left->right.
    var x_lock = x + w - pad - bs * Float32(4.0) - pad * Float32(3.0)
    var x_mute = x_lock + bs + pad
    var x_solo = x_mute + bs + pad
    var x_enab = x_solo + bs + pad

    # ---- row background + name (draw-gated) ----
    if draw:
        col(34, 34, 40)
        fill(x, y, w, h)
        col(0, 0, 0)
        rect(x, y, w, h)
        col(220, 220, 226)
        txt(name, x + Float32(6.0), y + (h - Float32(12.0)) / Float32(2.0), Float32(12.0))

    # ---- the 4 toggles (logic always runs; draw gated inside _toggle) ----
    var changed = False
    # Lock: amber when on
    if _toggle(ui, id0 + 0, x_lock, by, bs, locked, String("L"), 200, 160, 60, draw):
        changed = True
    # Mute: red when on
    if _toggle(ui, id0 + 1, x_mute, by, bs, muted, String("M"), 200, 70, 70, draw):
        changed = True
    # Solo: yellow when on
    if _toggle(ui, id0 + 2, x_solo, by, bs, soloed, String("S"), 220, 200, 70, draw):
        changed = True
    # Enable: green when on
    if _toggle(ui, id0 + 3, x_enab, by, bs, enabled, String("E"), 80, 180, 90, draw):
        changed = True
    return changed


# ============================================================================
# SELF-TEST  (compiled + run headlessly by the MAIN LOOP — do NOT run it yourself).
#
# draw=False on every call -> ZERO GL. Press+release semantics need two frames (down then up).
# We drive set_input over the MUTE toggle's rect to flip `muted`, and over EMPTY space to prove
# clicking nothing changes nothing.
#
# Geometry (derived from track_header's layout, FIRST PRINCIPLES):
#   header rect: x=100, y=50, w=200, h=24  -> bs = min(24-8, 18) = 16 ; by = 50 + (24-16)/2 = 54.
#   pad=3. x_lock = 100 + 200 - 3 - 16*4 - 3*3 = 300 - 3 - 64 - 9 = 224.0
#   x_mute = 224 + 16 + 3 = 243.0  -> Mute rect = [243, 259) x [54, 70).  center = (251, 62).
#   empty space: (110, 62) is INSIDE the header row but LEFT of all toggles -> hits no toggle.
# ============================================================================
def main() raises:
    var ui = UiContext()
    var pass_all = True

    var hx = Float32(100.0)
    var hy = Float32(50.0)
    var hw = Float32(200.0)
    var hh = Float32(24.0)

    var muted = False
    var locked = False
    var soloed = False
    var enabled = True

    # ---- 1) MUTE toggle: press inside (down) then release inside (up) -> muted flips on UP ----
    # frame 1: down @ Mute center (251, 62) -> latches active_id, returns False (no flip yet).
    ui.begin_frame()
    ui.set_input(251, 62, True)
    var c1 = track_header(ui, 10, hx, hy, hw, hh, String("V1"),
                          muted, locked, soloed, enabled, False)
    # frame 2: up (still inside) -> release fires -> muted flips True, returns True.
    ui.begin_frame()
    ui.set_input(251, 62, False)
    var c2 = track_header(ui, 10, hx, hy, hw, hh, String("V1"),
                          muted, locked, soloed, enabled, False)
    print("MUTE press(no-flip)=", c1, " release(flip)=", c2, " muted=", muted)
    if c1 != False:
        pass_all = False        # the DOWN frame must not flip anything
    if c2 != True:
        pass_all = False        # the UP frame fires the change
    if muted != True:
        pass_all = False        # mute bool actually flipped
    # untouched toggles stayed put
    if locked != False or soloed != False or enabled != True:
        pass_all = False

    # ---- 2) EMPTY space: press+release where NO toggle lives -> nothing changes ----
    # (110, 62): inside the header row but to the LEFT of x_lock(=224) -> no toggle hit.
    var pm = muted
    var pl = locked
    var ps = soloed
    var pe = enabled
    ui.begin_frame()
    ui.set_input(110, 62, True)
    var e1 = track_header(ui, 10, hx, hy, hw, hh, String("V1"),
                          muted, locked, soloed, enabled, False)
    ui.begin_frame()
    ui.set_input(110, 62, False)
    var e2 = track_header(ui, 10, hx, hy, hw, hh, String("V1"),
                          muted, locked, soloed, enabled, False)
    print("EMPTY press=", e1, " release=", e2)
    if e1 != False or e2 != False:
        pass_all = False        # clicking empty space reports no change
    if muted != pm or locked != pl or soloed != ps or enabled != pe:
        pass_all = False        # ...and flips nothing

    if pass_all:
        print("TRACK_HEADER_SELFTEST: pass (mute flip on release; empty click no-op)")
    else:
        print("TRACK_HEADER_SELFTEST: fail")
