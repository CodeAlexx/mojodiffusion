# audio_widgets.mojo — reusable audio-mixer controls for media apps.
#
# Knob (rotary), vertical dB fader, segmented level meter, and a 2D pan pad. Drawn via the
# MojoGUI C backend; interactive widgets use core_widgets.UiContext (hot/active drag, the
# same pattern as slider_float). Pair with the fpx_audio filter chains (vol/pan/EQ/comp/
# limiter/gate) to build a mixer like MediaEditor's.
#
# Public API:
#   knob(ui, id, cx, cy, r, mut value, vmin, vmax) -> Bool      rotary; drag up/down
#   fader_db(ui, id, x, y, w, h, mut db, db_min, db_max) -> Bool vertical fader (dB)
#   level_meter(x, y, w, h, level01)                            segmented VU meter (green->red)
#   pan_pad(ui, id, x, y, size, mut lr, mut fb) -> Bool         2D pan (L/R x F/B), drag dot
from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from builtin.type_aliases import MutExternalOrigin
from std.math import sin, cos
from core_widgets import UiContext


fn _cstr(s: String) raises -> UnsafePointer[Int8, MutExternalOrigin]:
    var b = s.as_bytes()
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(b[i])
    buf[len(b)] = 0
    return buf

fn _col(r: Int, g: Int, b: Int):
    _ = external_call["set_color", Int32](Float32(r) / 255.0, Float32(g) / 255.0, Float32(b) / 255.0, Float32(1.0))
fn _fill(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_filled_rectangle", Int32](x, y, w, h)
fn _rect(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_rectangle", Int32](x, y, w, h)
fn _fcircle(x: Float32, y: Float32, radius: Float32, segs: Int):
    _ = external_call["draw_filled_circle", Int32](x, y, radius, Int32(segs))
fn _txt(s: String, x: Float32, y: Float32, size: Float32) raises:
    _ = external_call["draw_text", Int32](_cstr(s), x, y, size)
fn _clampf(v: Float32, lo: Float32, hi: Float32) -> Float32:
    if v < lo: return lo
    if v > hi: return hi
    return v
fn _pir(px: Float32, py: Float32, x: Float32, y: Float32, w: Float32, h: Float32) -> Bool:
    return px >= x and px <= x + w and py >= y and py <= y + h


# Rotary knob. Drag vertically (up = increase) over a ~90px span. Returns True if changed.
def knob(mut ui: UiContext, id: Int, cx: Float32, cy: Float32, r: Float32, mut value: Float32, vmin: Float32, vmax: Float32) raises -> Bool:
    var inside = _pir(Float32(ui.mouse_x), Float32(ui.mouse_y), cx - r, cy - r, r * 2.0, r * 2.0)
    if inside: ui.hot_id = id
    if inside and ui.mouse_clicked: ui.active_id = id
    var changed = False
    var span = Float32(90.0)
    if ui.active_id == id and ui.mouse_down:
        var t = (cy + span * 0.5 - Float32(ui.mouse_y)) / span
        t = _clampf(t, 0.0, 1.0)
        var nv = vmin + t * (vmax - vmin)
        if nv != value:
            value = nv; changed = True
    if ui.mouse_released and ui.active_id == id: ui.active_id = -1
    var denom = vmax - vmin
    var frac = Float32(0.0)
    if denom != 0.0: frac = _clampf((value - vmin) / denom, 0.0, 1.0)
    _col(70, 72, 82); _fcircle(cx, cy, r, 26)
    _col(34, 35, 42); _fcircle(cx, cy, r - 3.0, 26)
    # indicator dot at -135deg..+135deg
    var a = Float32(-2.3561945) + frac * Float32(4.712389)
    var ix = cx + Float32(sin(a)) * (r - 5.0)
    var iy = cy - Float32(cos(a)) * (r - 5.0)
    if ui.active_id == id:
        _col(160, 200, 255)
    else:
        _col(220, 225, 235)
    _fcircle(ix, iy, 3.0, 10)
    return changed


# Vertical dB fader: value in dB, top = db_max. Returns True if changed.
def fader_db(mut ui: UiContext, id: Int, x: Float32, y: Float32, w: Float32, h: Float32, mut db: Float32, db_min: Float32, db_max: Float32) raises -> Bool:
    var inside = _pir(Float32(ui.mouse_x), Float32(ui.mouse_y), x, y, w, h)
    if inside: ui.hot_id = id
    if inside and ui.mouse_clicked: ui.active_id = id
    var changed = False
    if ui.active_id == id and ui.mouse_down:
        var t = (y + h - Float32(ui.mouse_y)) / h     # bottom=0, top=1
        t = _clampf(t, 0.0, 1.0)
        var nv = db_min + t * (db_max - db_min)
        if nv != db:
            db = nv; changed = True
    if ui.mouse_released and ui.active_id == id: ui.active_id = -1
    _col(26, 27, 32); _fill(x, y, w, h)
    _col(0, 0, 0); _rect(x, y, w, h)
    var denom = db_max - db_min
    var frac = Float32(0.0)
    if denom != 0.0: frac = _clampf((db - db_min) / denom, 0.0, 1.0)
    var hy = y + (1.0 - frac) * h          # handle y
    _col(50, 95, 150); _fill(x + 2.0, hy, w - 4.0, y + h - hy)   # filled below handle
    if ui.active_id == id:
        _col(160, 200, 255)
    else:
        _col(205, 210, 225)
    _fill(x - 2.0, hy - 4.0, w + 4.0, 8.0)  # handle bar
    return changed


# Segmented VU meter: lights bottom-up to level01 (0..1); green -> yellow -> red. Pure draw.
def level_meter(x: Float32, y: Float32, w: Float32, h: Float32, level01: Float32) raises:
    var n = 24
    var lit = Int(_clampf(level01, 0.0, 1.0) * Float32(n) + 0.5)
    var seg_h = h / Float32(n)
    for i in range(n):
        var frac = Float32(i) / Float32(n)
        var sy = y + h - Float32(i + 1) * seg_h
        var on = i < lit
        if frac > 0.85:
            if on: _col(220, 60, 50) else: _col(60, 24, 22)
        elif frac > 0.6:
            if on: _col(220, 200, 60) else: _col(60, 56, 24)
        else:
            if on: _col(70, 210, 90) else: _col(24, 56, 30)
        _fill(x, sy + 1.0, w, seg_h - 1.5)


# 2D pan pad: lr / fb in [-1,1]; drag the dot. Returns True if changed.
def pan_pad(mut ui: UiContext, id: Int, x: Float32, y: Float32, size: Float32, mut lr: Float32, mut fb: Float32) raises -> Bool:
    var inside = _pir(Float32(ui.mouse_x), Float32(ui.mouse_y), x, y, size, size)
    if inside: ui.hot_id = id
    if inside and ui.mouse_clicked: ui.active_id = id
    var changed = False
    var cx = x + size * 0.5
    var cy = y + size * 0.5
    var half = size * 0.5
    if ui.active_id == id and ui.mouse_down:
        var nlr = _clampf((Float32(ui.mouse_x) - cx) / half, -1.0, 1.0)
        var nfb = _clampf((Float32(ui.mouse_y) - cy) / half, -1.0, 1.0)
        if nlr != lr or nfb != fb:
            lr = nlr; fb = nfb; changed = True
    if ui.mouse_released and ui.active_id == id: ui.active_id = -1
    _col(20, 24, 34); _fill(x, y, size, size)
    _col(70, 90, 120); _rect(x, y, size, size)
    # crosshair
    _col(120, 95, 60); _fill(x, cy - 1.0, size, 2.0); _fill(cx - 1.0, y, 2.0, size)
    # dot
    var dx = cx + lr * half
    var dy = cy + fb * half
    _col(110, 150, 230); _fcircle(dx, dy, 6.0, 14)
    return changed


# Logic self-test (no GL): knob/fader/pan drag math via a synthetic UiContext.
def main() raises:
    var ui = UiContext()
    var ok = True
    # knob: click center, drag up 40px -> value should rise above start (0.5)
    var kv = Float32(0.5)
    ui.begin_frame(); ui.set_input(100, 100, True); _ = knob(ui, 1, 100.0, 100.0, 20.0, kv, Float32(0.0), Float32(1.0))
    ui.begin_frame(); ui.set_input(100, 60, True);  _ = knob(ui, 1, 100.0, 100.0, 20.0, kv, Float32(0.0), Float32(1.0))
    print("AUDIO_WIDGETS knob after drag-up:", kv, " (expect > 0.5)")
    if not (kv > Float32(0.5)): ok = False
    ui.begin_frame(); ui.set_input(100, 60, False)
    # fader: click mid, drag to near top -> dB near db_max
    var fv = Float32(-30.0)
    ui.begin_frame(); ui.set_input(50, 150, True); _ = fader_db(ui, 2, 40.0, 100.0, 20.0, 200.0, fv, Float32(-60.0), Float32(6.0))
    ui.begin_frame(); ui.set_input(50, 104, True); _ = fader_db(ui, 2, 40.0, 100.0, 20.0, 200.0, fv, Float32(-60.0), Float32(6.0))
    print("AUDIO_WIDGETS fader after drag-top:", fv, " (expect near +6)")
    if not (fv > Float32(0.0)): ok = False
    ui.begin_frame(); ui.set_input(50, 104, False)   # release the fader first
    # pan: click bottom-right of a pad at (100,100,size=200), center (200,200) -> lr>0, fb>0
    var lr = Float32(0.0); var fb = Float32(0.0)
    ui.begin_frame(); ui.set_input(250, 250, True); _ = pan_pad(ui, 3, 100.0, 100.0, 200.0, lr, fb)
    print("AUDIO_WIDGETS pan after drag-BR: lr=", lr, " fb=", fb, " (expect both > 0)")
    if not (lr > Float32(0.0) and fb > Float32(0.0)): ok = False
    if ok: print("AUDIO_WIDGETS_SELFTEST: pass")
    else: print("AUDIO_WIDGETS_SELFTEST: fail")
