# media_widgets.mojo — reusable controls for media / video apps.
#
# Extracted from MojoMedia (NLE) + MojoClip (dataset clipper) so any future app can reuse
# them. All drawing goes through the MojoGUI C backend (draw_text / set_color / draw_image)
# and the DrawList path; the transport bar uses core_widgets.UiContext for its buttons.
#
# Public API:
#   fmt_timecode(frames, fps) -> "M:SS.ss"
#   transport_bar(ui, id0, x, y, btn_w, playing) -> TransportResult   (|< <| [] >/|| |> >|)
#   filmstrip(x, y, w, h, thumbs, n, tw, th)                          row of RGBA8 thumbnails
#   seconds_ruler(x, y, w, total_frames, fps)                         adaptive ticks + labels
#   content_strip(d, x, y, w, h, env, n)                              green=content / dark=black
#   waveform_strip(d, x, center_y, half_h, w, env, n)                 centered audio waveform
#   apply_grade_u8(src, dst, n, bright, contrast)                     export grade (CPU)
#   apply_grade_gain_u8(src, dst, n, bright, contrast, gain, do_grade) preview grade + view gain
#   draw_icon(d, kind, cx, cy, r, col) / icon_button(ui, id, x,y,w,h, kind)   vector glyphs
#   begin_toolbar(x,y,w,h[,btn_w,pad,draw]) -> Toolbar                 auto-layout toolbar
#     .button/.toggle/.sep/.gap/.label/.right/.at                      (icon items + cursor)
from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from core_widgets import UiContext, measure_text
from draw_list_paths import DrawList, Col4


fn _cstr(s: String) raises -> UnsafePointer[Int8, MutExternalOrigin]:
    var b = s.as_bytes()
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(b[i])
    buf[len(b)] = 0
    return buf


fn _txt(s: String, x: Float32, y: Float32, size: Float32) raises:
    _ = external_call["draw_text", Int32](_cstr(s), x, y, size)


fn _col(r: Float32, g: Float32, b: Float32):
    _ = external_call["set_color", Int32](r, g, b, Float32(1.0))


# ---- timecode -------------------------------------------------------------------------
# frame index -> "M:SS.ss" at the given fps (e.g. 50 @ 30fps -> "0:01.66").
fn fmt_timecode(frames: Int, fps: Float64) raises -> String:
    var f = frames
    if f < 0: f = 0
    var total = Float64(f) / fps
    var whole = Int(total)
    var mins = whole // 60
    var secs = whole % 60
    var centi = Int((total - Float64(whole)) * 100.0)
    var ss = String(secs) if secs >= 10 else String("0") + String(secs)
    var cc = String(centi) if centi >= 10 else String("0") + String(centi)
    return String(mins) + String(":") + ss + String(".") + cc


# ---- transport bar --------------------------------------------------------------------
@fieldwise_init
struct TransportResult(Copyable, Movable):
    var to_start: Bool      # |<  jump to first frame
    var step_back: Bool     # <|  step -1 frame
    var stop: Bool          # []  stop (pause, keep position)
    var play_toggle: Bool   # >/|| play-pause toggle
    var step_fwd: Bool      # |>  step +1 frame
    var to_end: Bool        # >|  jump to last frame


# --- transport icon shapes (crisp triangles/bars drawn on top of the buttons) ---
fn _tri_right(mut d: DrawList, cx: Float32, cy: Float32, c: Col4):
    d.add_triangle_filled(cx - 5.0, cy - 6.0, cx - 5.0, cy + 6.0, cx + 6.0, cy, c)
fn _tri_left(mut d: DrawList, cx: Float32, cy: Float32, c: Col4):
    d.add_triangle_filled(cx + 5.0, cy - 6.0, cx + 5.0, cy + 6.0, cx - 6.0, cy, c)
fn _pause(mut d: DrawList, cx: Float32, cy: Float32, c: Col4):
    d.add_rect_filled(cx - 5.0, cy - 6.0, 3.0, 12.0, c)
    d.add_rect_filled(cx + 2.0, cy - 6.0, 3.0, 12.0, c)
fn _stop(mut d: DrawList, cx: Float32, cy: Float32, c: Col4):
    d.add_rect_filled(cx - 5.0, cy - 5.0, 10.0, 10.0, c)
fn _to_start(mut d: DrawList, cx: Float32, cy: Float32, c: Col4):
    d.add_rect_filled(cx - 7.0, cy - 6.0, 2.5, 12.0, c)
    d.add_triangle_filled(cx + 6.0, cy - 6.0, cx + 6.0, cy + 6.0, cx - 3.0, cy, c)
fn _to_end(mut d: DrawList, cx: Float32, cy: Float32, c: Col4):
    d.add_triangle_filled(cx - 6.0, cy - 6.0, cx - 6.0, cy + 6.0, cx + 3.0, cy, c)
    d.add_rect_filled(cx + 5.0, cy - 6.0, 2.5, 12.0, c)


# Borderless transport button: hit-test (press-in + release-in) + add a state bg to `d`
# (idle=none, hover=grey, pressed=blue, active=accent + underline). Matches Toolbar style.
fn _tflat(mut ui: UiContext, mut d: DrawList, id: Int, x: Float32, y: Float32, w: Float32, h: Float32, active: Bool) -> Bool:
    var mxf = Float32(ui.mouse_x)
    var myf = Float32(ui.mouse_y)
    var inside = (mxf >= x) and (mxf < x + w) and (myf >= y) and (myf < y + h)
    var clicked = False
    if inside:
        ui.hot_id = id
    if inside and ui.mouse_clicked:
        ui.active_id = id
    if ui.mouse_released and ui.active_id == id:
        if inside:
            clicked = True
        ui.active_id = -1
    var pressed = (ui.active_id == id) and inside
    var hot = ui.hot_id == id
    if active:
        d.add_rect_filled(x, y, w, h, Col4(0.20, 0.33, 0.52, 1.0))
        d.add_rect_filled(x, y + h - 2.0, w, 2.0, Col4(0.353, 0.667, 0.980, 1.0))
    elif pressed:
        d.add_rect_filled(x, y, w, h, Col4(0.275, 0.431, 0.647, 1.0))
    elif hot:
        d.add_rect_filled(x, y, w, h, Col4(0.314, 0.314, 0.361, 1.0))
    return clicked


# Draw the 6-button transport bar — flat/borderless (matches Toolbar): icons with hover
# highlight, the play button shows a toggle accent while `playing`. Uses 6 ids from id0.
def transport_bar(mut ui: UiContext, id0: Int, x: Float32, y: Float32, btn_w: Float32, playing: Bool) raises -> TransportResult:
    var r = TransportResult(False, False, False, False, False, False)
    var h = Float32(22.0)
    var gap = btn_w + 4.0
    var pw = btn_w + 12.0
    var x0 = x
    var x1 = x + gap
    var x2 = x + gap * 2.0
    var x3 = x + gap * 3.0
    var x4 = x + gap * 4.0 + 12.0
    var x5 = x + gap * 5.0 + 12.0
    var d = DrawList()
    # state backgrounds first (so icons draw on top in the same list)
    r.to_start = _tflat(ui, d, id0 + 0, x0, y, btn_w, h, False)
    r.step_back = _tflat(ui, d, id0 + 1, x1, y, btn_w, h, False)
    r.stop = _tflat(ui, d, id0 + 2, x2, y, btn_w, h, False)
    r.play_toggle = _tflat(ui, d, id0 + 3, x3, y, pw, h, playing)
    r.step_fwd = _tflat(ui, d, id0 + 4, x4, y, btn_w, h, False)
    r.to_end = _tflat(ui, d, id0 + 5, x5, y, btn_w, h, False)
    var c = Col4(0.9, 0.92, 0.97, 1.0)
    var cy = y + 11.0
    _to_start(d, x0 + btn_w / 2.0, cy, c)
    _tri_left(d, x1 + btn_w / 2.0, cy, c)
    _stop(d, x2 + btn_w / 2.0, cy, c)
    if playing: _pause(d, x3 + pw / 2.0, cy, c)
    else: _tri_right(d, x3 + pw / 2.0, cy, c)
    _tri_right(d, x4 + btn_w / 2.0, cy, c)
    _to_end(d, x5 + btn_w / 2.0, cy, c)
    var _f = d.flush()
    return r^


# ---- filmstrip ------------------------------------------------------------------------
# Draw n RGBA8 thumbnails (each tw*th*4 bytes, packed contiguously in `thumbs`) evenly
# across [x, x+w] at height h. Reusable timeline filmstrip / contact sheet.
def filmstrip(x: Float32, y: Float32, w: Float32, h: Float32, thumbs: UnsafePointer[UInt8, MutExternalOrigin], n: Int, tw: Int, th: Int) raises:
    if n <= 0: return
    var cw = w / Float32(n)
    for i in range(n):
        var tx = x + Float32(i) * cw
        var slot = thumbs + (i * tw * th * 4)
        _ = external_call["draw_image", Int32](tx, y, cw + 1.0, h, slot, Int32(tw), Int32(th))


# ---- seconds ruler --------------------------------------------------------------------
# Self-contained: adaptive tick spacing + labels (s for short clips, m for long) across
# [x, x+w] mapping [0, total_frames] at fps. Draws its own DrawList + text.
def seconds_ruler(x: Float32, y: Float32, w: Float32, total_frames: Int, fps: Float64) raises:
    var tsec = Int(Float64(total_frames) / fps)
    if tsec <= 0: tsec = 1
    var step = 1
    if tsec > 1200: step = 120
    elif tsec > 600: step = 60
    elif tsec > 240: step = 30
    elif tsec > 60: step = 10
    elif tsec > 24: step = 5
    elif tsec > 12: step = 2
    var d = DrawList()
    var sec = 0
    while sec <= tsec:
        var rx = x + Float32(Float64(sec) * fps) / Float32(total_frames) * w
        d.add_rect_filled(rx, y, 1.0, 9.0, Col4(0.5, 0.55, 0.62, 1.0))
        sec += step
    var _f = d.flush()
    _col(0.55, 0.6, 0.68)
    var s2 = 0
    while s2 <= tsec:
        var rx2 = x + Float32(Float64(s2) * fps) / Float32(total_frames) * w
        var lbl = (String(s2 // 60) + String("m")) if step >= 60 else (String(s2) + String("s"))
        _txt(lbl, rx2 + 2.0, y + 12.0, 11.0)
        s2 += step


# ---- timeline content / waveform strips ----------------------------------------------
# Add a content band to DrawList `d`: green where env[i] is high (picture), dark where low
# (black). `env` is n floats in [0,1] (e.g. fraction of non-black pixels per time-bucket).
def content_strip(mut d: DrawList, x: Float32, y: Float32, w: Float32, h: Float32, env: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    if n <= 0: return
    var cw = w / Float32(n) + 1.0
    for i in range(n):
        var cx = x + Float32(i) / Float32(n) * w
        var lv = env[i]
        d.add_rect_filled(cx, y, cw, h, Col4(lv * 0.4, 0.16 + lv * 0.72, lv * 0.3, 1.0))


# Add a centered audio waveform to DrawList `d`: env[i] is peak amplitude in [0,1] per
# bucket; bars extend +/- (env*half_h) around center_y across [x, x+w].
def waveform_strip(mut d: DrawList, x: Float32, center_y: Float32, half_h: Float32, w: Float32, env: UnsafePointer[Float32, MutExternalOrigin], n: Int):
    if n <= 0: return
    var cw = w / Float32(n) + 1.0
    for i in range(n):
        var wx = x + Float32(i) / Float32(n) * w
        var amp = env[i]
        if amp > 1.0: amp = 1.0
        var hh = amp * half_h
        d.add_rect_filled(wx, center_y - hh, cw, hh * 2.0 + 1.0, Col4(0.35, 0.62, 0.85, 0.8))


# ---- CPU grade helpers ----------------------------------------------------------------
# Brightness/contrast on RGBA8 src -> dst (A passthrough). For baked export grade.
fn apply_grade_u8(src: UnsafePointer[UInt8, MutExternalOrigin], dst: UnsafePointer[UInt8, MutExternalOrigin], n: Int, bright: Float32, contrast: Float32):
    for i in range(n):
        if i % 4 == 3:
            dst[i] = src[i]
        else:
            var v = Float32(Int(src[i])) / 255.0
            var o = (v - 0.5) * contrast + 0.5 + bright
            if o < 0.0: o = 0.0
            if o > 1.0: o = 1.0
            dst[i] = UInt8(Int(o * 255.0 + 0.5))


# Preview grade with an extra view-only gain (lights up dark footage without affecting
# export). do_grade gates the brightness/contrast; gain always applies.
fn apply_grade_gain_u8(src: UnsafePointer[UInt8, MutExternalOrigin], dst: UnsafePointer[UInt8, MutExternalOrigin], n: Int, bright: Float32, contrast: Float32, gain: Float32, do_grade: Bool):
    for i in range(n):
        if i % 4 == 3:
            dst[i] = src[i]
        else:
            var v = Float32(Int(src[i])) / 255.0
            if do_grade:
                v = (v - 0.5) * contrast + 0.5 + bright
            v = v * gain
            if v < 0.0: v = 0.0
            if v > 1.0: v = 1.0
            dst[i] = UInt8(Int(v * 255.0 + 0.5))


# ---- vector icon set (drawn shapes, resolution-independent — no image assets) ----
comptime ICON_PLAY = 0
comptime ICON_PAUSE = 1
comptime ICON_STOP = 2
comptime ICON_PREV = 3      # |<
comptime ICON_NEXT = 4      # >|
comptime ICON_PLUS = 5
comptime ICON_MINUS = 6
comptime ICON_TRASH = 7
comptime ICON_SPLIT = 8     # razor: cut line + blade
comptime ICON_UNDO = 9      # left arrow
comptime ICON_REDO = 10     # right arrow
comptime ICON_SAVE = 11     # floppy disk
comptime ICON_OPEN = 12     # folder
comptime ICON_MARK_IN = 13  # [ bracket
comptime ICON_MARK_OUT = 14 # ] bracket
comptime ICON_EXPORT = 15   # download/export (down arrow into a tray)
comptime ICON_MIXER = 16    # mixer sliders
comptime ICON_GRAPH = 17    # node graph (two boxes + link)
comptime ICON_LAYERS = 18   # compositing (two stacked plates)
comptime ICON_COLOR = 19    # color grade (sun / brightness)
comptime ICON_AUDIO = 20    # speaker + waves
comptime ICON_FILM = 21     # film strip
comptime ICON_GEAR = 22     # settings

# Draw an icon `kind` centered at (cx,cy), ~r px radius, colour c, into DrawList d.
fn draw_icon(mut d: DrawList, kind: Int, cx: Float32, cy: Float32, r: Float32, c: Col4):
    if kind == ICON_PLAY:
        d.add_triangle_filled(cx - r * 0.5, cy - r * 0.7, cx - r * 0.5, cy + r * 0.7, cx + r * 0.7, cy, c)
    elif kind == ICON_PAUSE:
        d.add_rect_filled(cx - r * 0.55, cy - r * 0.7, r * 0.35, r * 1.4, c)
        d.add_rect_filled(cx + r * 0.2, cy - r * 0.7, r * 0.35, r * 1.4, c)
    elif kind == ICON_STOP:
        d.add_rect_filled(cx - r * 0.6, cy - r * 0.6, r * 1.2, r * 1.2, c)
    elif kind == ICON_PREV:
        d.add_rect_filled(cx - r * 0.7, cy - r * 0.7, r * 0.28, r * 1.4, c)
        d.add_triangle_filled(cx + r * 0.6, cy - r * 0.7, cx + r * 0.6, cy + r * 0.7, cx - r * 0.3, cy, c)
    elif kind == ICON_NEXT:
        d.add_triangle_filled(cx - r * 0.6, cy - r * 0.7, cx - r * 0.6, cy + r * 0.7, cx + r * 0.3, cy, c)
        d.add_rect_filled(cx + r * 0.42, cy - r * 0.7, r * 0.28, r * 1.4, c)
    elif kind == ICON_PLUS:
        d.add_rect_filled(cx - r * 0.7, cy - r * 0.18, r * 1.4, r * 0.36, c)
        d.add_rect_filled(cx - r * 0.18, cy - r * 0.7, r * 0.36, r * 1.4, c)
    elif kind == ICON_MINUS:
        d.add_rect_filled(cx - r * 0.7, cy - r * 0.18, r * 1.4, r * 0.36, c)
    elif kind == ICON_TRASH:
        d.add_rect_filled(cx - r * 0.25, cy - r * 0.95, r * 0.5, r * 0.18, c)   # handle
        d.add_rect_filled(cx - r * 0.7, cy - r * 0.75, r * 1.4, r * 0.2, c)      # lid
        d.add_rect_filled(cx - r * 0.55, cy - r * 0.5, r * 1.1, r * 1.2, c)      # body
    elif kind == ICON_SPLIT:
        d.add_rect_filled(cx - r * 0.08, cy - r * 0.85, r * 0.16, r * 1.7, c)    # cut line
        d.add_triangle_filled(cx - r * 0.45, cy - r * 0.95, cx + r * 0.45, cy - r * 0.95, cx, cy - r * 0.35, c)  # blade
    elif kind == ICON_UNDO:
        d.add_triangle_filled(cx - r * 0.6, cy, cx + r * 0.05, cy - r * 0.6, cx + r * 0.05, cy + r * 0.6, c)
        d.add_rect_filled(cx + r * 0.05, cy - r * 0.18, r * 0.6, r * 0.36, c)
    elif kind == ICON_REDO:
        d.add_triangle_filled(cx + r * 0.6, cy, cx - r * 0.05, cy - r * 0.6, cx - r * 0.05, cy + r * 0.6, c)
        d.add_rect_filled(cx - r * 0.65, cy - r * 0.18, r * 0.6, r * 0.36, c)
    elif kind == ICON_SAVE:
        d.add_rect_filled(cx - r * 0.7, cy - r * 0.7, r * 1.4, r * 1.4, c)              # disk body
        d.add_rect_filled(cx - r * 0.4, cy + r * 0.05, r * 0.8, r * 0.6, Col4(0.10, 0.11, 0.14, 1.0))  # label cutout
        d.add_rect_filled(cx + r * 0.15, cy - r * 0.6, r * 0.35, r * 0.45, Col4(0.10, 0.11, 0.14, 1.0))  # slider notch
    elif kind == ICON_OPEN:
        d.add_rect_filled(cx - r * 0.7, cy - r * 0.55, r * 0.6, r * 0.22, c)            # folder tab
        d.add_rect_filled(cx - r * 0.7, cy - r * 0.4, r * 1.4, r * 0.9, c)              # folder body
    elif kind == ICON_MARK_IN:
        d.add_rect_filled(cx - r * 0.5, cy - r * 0.7, r * 0.26, r * 1.4, c)             # vbar
        d.add_rect_filled(cx - r * 0.5, cy - r * 0.7, r * 0.8, r * 0.24, c)             # top stub
        d.add_rect_filled(cx - r * 0.5, cy + r * 0.46, r * 0.8, r * 0.24, c)            # bottom stub
    elif kind == ICON_MARK_OUT:
        d.add_rect_filled(cx + r * 0.24, cy - r * 0.7, r * 0.26, r * 1.4, c)            # vbar
        d.add_rect_filled(cx - r * 0.3, cy - r * 0.7, r * 0.8, r * 0.24, c)             # top stub
        d.add_rect_filled(cx - r * 0.3, cy + r * 0.46, r * 0.8, r * 0.24, c)            # bottom stub
    elif kind == ICON_EXPORT:
        d.add_rect_filled(cx - r * 0.16, cy - r * 0.75, r * 0.32, r * 0.75, c)          # arrow stem
        d.add_triangle_filled(cx - r * 0.45, cy - r * 0.05, cx + r * 0.45, cy - r * 0.05, cx, cy + r * 0.5, c)  # arrowhead
        d.add_rect_filled(cx - r * 0.7, cy + r * 0.6, r * 1.4, r * 0.22, c)             # tray
    elif kind == ICON_MIXER:
        d.add_rect_filled(cx - r * 0.62, cy - r * 0.7, r * 0.14, r * 1.4, c)            # 3 slider tracks
        d.add_rect_filled(cx - r * 0.07, cy - r * 0.7, r * 0.14, r * 1.4, c)
        d.add_rect_filled(cx + r * 0.48, cy - r * 0.7, r * 0.14, r * 1.4, c)
        d.add_rect_filled(cx - r * 0.74, cy - r * 0.35, r * 0.38, r * 0.22, c)          # knobs at varied heights
        d.add_rect_filled(cx - r * 0.19, cy + r * 0.2, r * 0.38, r * 0.22, c)
        d.add_rect_filled(cx + r * 0.36, cy - r * 0.1, r * 0.38, r * 0.22, c)
    elif kind == ICON_GRAPH:
        d.add_rect_filled(cx - r * 0.75, cy - r * 0.3, r * 0.5, r * 0.6, c)             # node A
        d.add_rect_filled(cx + r * 0.25, cy - r * 0.3, r * 0.5, r * 0.6, c)             # node B
        d.add_rect_filled(cx - r * 0.25, cy - r * 0.08, r * 0.5, r * 0.16, c)           # link
    elif kind == ICON_LAYERS:
        d.add_triangle_filled(cx, cy - r * 0.8, cx - r * 0.85, cy - r * 0.35, cx + r * 0.85, cy - r * 0.35, c)   # top plate ^
        d.add_triangle_filled(cx, cy + r * 0.1, cx - r * 0.85, cy - r * 0.35, cx + r * 0.85, cy - r * 0.35, c)   # top plate v
        d.add_triangle_filled(cx, cy + r * 0.15, cx - r * 0.85, cy + r * 0.6, cx + r * 0.85, cy + r * 0.6, c)    # bottom plate ^
        d.add_triangle_filled(cx, cy + r * 1.05, cx - r * 0.85, cy + r * 0.6, cx + r * 0.85, cy + r * 0.6, c)    # bottom plate v
    elif kind == ICON_COLOR:
        d.add_rect_filled(cx - r * 0.35, cy - r * 0.35, r * 0.7, r * 0.7, c)            # core (sun)
        d.add_rect_filled(cx - r * 0.08, cy - r * 0.9, r * 0.16, r * 0.32, c)           # ray N
        d.add_rect_filled(cx - r * 0.08, cy + r * 0.58, r * 0.16, r * 0.32, c)          # ray S
        d.add_rect_filled(cx - r * 0.9, cy - r * 0.08, r * 0.32, r * 0.16, c)           # ray W
        d.add_rect_filled(cx + r * 0.58, cy - r * 0.08, r * 0.32, r * 0.16, c)          # ray E
    elif kind == ICON_AUDIO:
        d.add_rect_filled(cx - r * 0.78, cy - r * 0.32, r * 0.4, r * 0.64, c)           # speaker box
        d.add_triangle_filled(cx - r * 0.4, cy - r * 0.78, cx - r * 0.4, cy + r * 0.78, cx + r * 0.22, cy, c)  # cone
        d.add_rect_filled(cx + r * 0.42, cy - r * 0.45, r * 0.12, r * 0.9, c)           # wave bar 1
        d.add_rect_filled(cx + r * 0.68, cy - r * 0.25, r * 0.12, r * 0.5, c)           # wave bar 2
    elif kind == ICON_FILM:
        d.add_rect_filled(cx - r * 0.82, cy - r * 0.72, r * 1.64, r * 1.44, c)          # strip body
        d.add_rect_filled(cx - r * 0.6, cy - r * 0.58, r * 0.26, r * 0.26, Col4(0.10, 0.11, 0.14, 1.0))   # top holes
        d.add_rect_filled(cx - r * 0.13, cy - r * 0.58, r * 0.26, r * 0.26, Col4(0.10, 0.11, 0.14, 1.0))
        d.add_rect_filled(cx + r * 0.34, cy - r * 0.58, r * 0.26, r * 0.26, Col4(0.10, 0.11, 0.14, 1.0))
        d.add_rect_filled(cx - r * 0.6, cy + r * 0.32, r * 0.26, r * 0.26, Col4(0.10, 0.11, 0.14, 1.0))   # bottom holes
        d.add_rect_filled(cx - r * 0.13, cy + r * 0.32, r * 0.26, r * 0.26, Col4(0.10, 0.11, 0.14, 1.0))
        d.add_rect_filled(cx + r * 0.34, cy + r * 0.32, r * 0.26, r * 0.26, Col4(0.10, 0.11, 0.14, 1.0))
        d.add_rect_filled(cx - r * 0.32, cy - r * 0.18, r * 0.64, r * 0.36, Col4(0.10, 0.11, 0.14, 1.0))  # center frame
    elif kind == ICON_GEAR:
        d.add_rect_filled(cx - r * 0.5, cy - r * 0.5, r * 1.0, r * 1.0, c)              # body
        d.add_rect_filled(cx - r * 0.18, cy - r * 0.88, r * 0.36, r * 0.32, c)          # tooth N
        d.add_rect_filled(cx - r * 0.18, cy + r * 0.56, r * 0.36, r * 0.32, c)          # tooth S
        d.add_rect_filled(cx - r * 0.88, cy - r * 0.18, r * 0.32, r * 0.36, c)          # tooth W
        d.add_rect_filled(cx + r * 0.56, cy - r * 0.18, r * 0.32, r * 0.36, c)          # tooth E
        d.add_rect_filled(cx - r * 0.2, cy - r * 0.2, r * 0.4, r * 0.4, Col4(0.10, 0.11, 0.14, 1.0))  # center hole


# An icon-only button: empty button rect (hover/click) + a drawn icon centered on top.
def icon_button(mut ui: UiContext, id: Int, x: Float32, y: Float32, w: Float32, h: Float32, kind: Int) raises -> Bool:
    var clicked = ui.button(id, String(""), x, y, w, h)
    var d = DrawList()
    var sz = h * 0.32
    if sz > 9.0: sz = 9.0
    draw_icon(d, kind, x + w * 0.5, y + h * 0.5, sz, Col4(0.9, 0.92, 0.97, 1.0))
    var _f = d.flush()
    return clicked


# =======================================================================================
# Toolbar — a proper image/glyph toolbar widget (auto-layout, borderless hover-highlight
# icon buttons, toggle/active accent, separators, labels, tooltips, right-alignment).
#
# Immediate-mode + cursor-based, modelled on a real GUI lib's toolbar: you don't position
# buttons by hand, you advance a cursor.  Items are borderless (no idle box) on a single
# panel bar; they highlight on hover, darken on press, and show a blue underline+tint when
# toggled "on".  Hit-testing replicates UiContext.button exactly (press-in + release-in).
#
#   var tb = begin_toolbar(x, y, w, h)            # draws the panel bar, returns the cursor
#   if tb.button(ui, 1, ICON_PLAY, "Play"): ...   # icon button (+ optional hover tooltip)
#   tb.sep()                                       # vertical divider
#   _ = tb.toggle(ui, 2, ICON_MIXER, mix_on, "Mixer")   # toggle: accent when `active`
#   _ = tb.label("Render", 12.0)                   # inline text label
#   tb.right(1)                                     # right-align the remaining group
#   if tb.button(ui, 3, ICON_EXPORT, "Export"): ...
#
# Pass draw=False to begin_toolbar for headless layout/hit-test unit tests (no GL).
# =======================================================================================
@fieldwise_init
struct Toolbar(Copyable, Movable):
    var cx: Float32       # cursor: left edge of the NEXT item
    var y: Float32        # toolbar top
    var h: Float32        # toolbar height
    var x_end: Float32    # right edge (x + w) — for right()
    var btn_w: Float32    # default icon-button width
    var pad: Float32      # gap between adjacent items
    var draw: Bool        # False => layout/hit-test only, no GL (unit tests)

    # full-control item: active=toggled-on accent, enabled=interactive, tip=hover label.
    def item(mut self, mut ui: UiContext, id: Int, kind: Int, active: Bool, enabled: Bool, tip: String) raises -> Bool:
        var bx = self.cx
        var bw = self.btn_w
        var by = self.y + 3.0
        var bh = self.h - 6.0
        var mxf = Float32(ui.mouse_x)
        var myf = Float32(ui.mouse_y)
        var inside = (mxf >= bx) and (mxf < bx + bw) and (myf >= by) and (myf < by + bh)
        var clicked = False
        if enabled:
            if inside:
                ui.hot_id = id
            if inside and ui.mouse_clicked:
                ui.active_id = id
            if ui.mouse_released and ui.active_id == id:
                if inside:
                    clicked = True
                ui.active_id = -1
        if self.draw:
            var d = DrawList()
            var pressed = (ui.active_id == id) and inside
            var hot = ui.hot_id == id
            if active:
                d.add_rect_filled(bx, by, bw, bh, Col4(0.20, 0.33, 0.52, 1.0))
            elif pressed:
                d.add_rect_filled(bx, by, bw, bh, Col4(0.275, 0.431, 0.647, 1.0))
            elif hot and enabled:
                d.add_rect_filled(bx, by, bw, bh, Col4(0.314, 0.314, 0.361, 1.0))
            if active:
                d.add_rect_filled(bx, by + bh - 2.0, bw, 2.0, Col4(0.353, 0.667, 0.980, 1.0))
            var icol = Col4(0.9, 0.92, 0.97, 1.0)
            if not enabled:
                icol = Col4(0.45, 0.46, 0.5, 1.0)
            var sz = bh * 0.34
            if sz > 10.0: sz = 10.0
            draw_icon(d, kind, bx + bw * 0.5, by + bh * 0.5, sz, icol)
            var _f = d.flush()
            if enabled and (ui.hot_id == id) and len(tip) > 0:
                ui.tooltip(tip, mxf, myf)
        self.cx = bx + bw + self.pad
        return clicked

    # plain icon button (+ optional tooltip; pass "" for none).
    def button(mut self, mut ui: UiContext, id: Int, kind: Int, tip: String) raises -> Bool:
        return self.item(ui, id, kind, False, True, tip)

    # toggle button: renders the accent fill + underline while `active` is True.
    def toggle(mut self, mut ui: UiContext, id: Int, kind: Int, active: Bool, tip: String) raises -> Bool:
        return self.item(ui, id, kind, active, True, tip)

    # icon + text button (auto-width) — for a top menu/File toolbar (Open, Export, ...).
    def button_label(mut self, mut ui: UiContext, id: Int, kind: Int, text: String, tip: String) raises -> Bool:
        var by = self.y + 3.0
        var bh = self.h - 6.0
        var tw = Float32(0.0)
        var th = Float32(12.0)
        if self.draw:
            var m = measure_text(text, 11.0)
            tw = m[0]; th = m[1]
        else:
            tw = Float32(len(text)) * 6.6                # headless estimate (no GL)
        var icon_cell = bh                                # square icon zone at the left
        var bw = icon_cell + tw + 12.0
        var bx = self.cx
        var mxf = Float32(ui.mouse_x)
        var myf = Float32(ui.mouse_y)
        var inside = (mxf >= bx) and (mxf < bx + bw) and (myf >= by) and (myf < by + bh)
        var clicked = False
        if inside:
            ui.hot_id = id
        if inside and ui.mouse_clicked:
            ui.active_id = id
        if ui.mouse_released and ui.active_id == id:
            if inside:
                clicked = True
            ui.active_id = -1
        if self.draw:
            var d = DrawList()
            var pressed = (ui.active_id == id) and inside
            var hot = ui.hot_id == id
            if pressed:
                d.add_rect_filled(bx, by, bw, bh, Col4(0.275, 0.431, 0.647, 1.0))
            elif hot:
                d.add_rect_filled(bx, by, bw, bh, Col4(0.314, 0.314, 0.361, 1.0))
            var sz = bh * 0.30
            if sz > 8.0: sz = 8.0
            draw_icon(d, kind, bx + icon_cell * 0.5, by + bh * 0.5, sz, Col4(0.9, 0.92, 0.97, 1.0))
            var _f = d.flush()
            _col(0.86, 0.89, 0.95)
            _txt(text, bx + icon_cell, by + (bh - th) * 0.5, 11.0)
            if hot and len(tip) > 0:
                ui.tooltip(tip, mxf, myf)
        self.cx = bx + bw + self.pad
        return clicked

    # text-only button (auto-width) — for non-iconic actions (Trim+, Bndry>, …).
    def text_button(mut self, mut ui: UiContext, id: Int, text: String, tip: String) raises -> Bool:
        var by = self.y + 3.0
        var bh = self.h - 6.0
        var tw = Float32(0.0)
        var th = Float32(12.0)
        if self.draw:
            var m = measure_text(text, 12.0)
            tw = m[0]; th = m[1]
        else:
            tw = Float32(len(text)) * 7.0
        var bw = tw + 16.0
        var bx = self.cx
        var mxf = Float32(ui.mouse_x)
        var myf = Float32(ui.mouse_y)
        var inside = (mxf >= bx) and (mxf < bx + bw) and (myf >= by) and (myf < by + bh)
        var clicked = False
        if inside:
            ui.hot_id = id
        if inside and ui.mouse_clicked:
            ui.active_id = id
        if ui.mouse_released and ui.active_id == id:
            if inside:
                clicked = True
            ui.active_id = -1
        if self.draw:
            var d = DrawList()
            var pressed = (ui.active_id == id) and inside
            var hot = ui.hot_id == id
            if pressed:
                d.add_rect_filled(bx, by, bw, bh, Col4(0.275, 0.431, 0.647, 1.0))
            elif hot:
                d.add_rect_filled(bx, by, bw, bh, Col4(0.314, 0.314, 0.361, 1.0))
            var _f = d.flush()
            _col(0.86, 0.89, 0.95)
            _txt(text, bx + (bw - tw) * 0.5, by + (bh - th) * 0.5, 12.0)
            if hot and len(tip) > 0:
                ui.tooltip(tip, mxf, myf)
        self.cx = bx + bw + self.pad
        return clicked

    # ribbon TAB: icon + label, with a selected-tab accent (fill + thick bottom underline)
    # when `active`. Use a begin_toolbar row of these to switch tool groups (TabToolbar-style).
    def tab(mut self, mut ui: UiContext, id: Int, kind: Int, text: String, active: Bool, tip: String) raises -> Bool:
        var by = self.y + 2.0
        var bh = self.h - 4.0
        var tw = Float32(0.0)
        var th = Float32(12.0)
        if self.draw:
            var m = measure_text(text, 12.0)
            tw = m[0]; th = m[1]
        else:
            tw = Float32(len(text)) * 7.0                # headless estimate (no GL)
        var icon_cell = bh * 0.9                          # square icon zone at the left
        var bw = icon_cell + tw + 16.0
        var bx = self.cx
        var mxf = Float32(ui.mouse_x)
        var myf = Float32(ui.mouse_y)
        var inside = (mxf >= bx) and (mxf < bx + bw) and (myf >= by) and (myf < by + bh)
        var clicked = False
        if inside:
            ui.hot_id = id
        if inside and ui.mouse_clicked:
            ui.active_id = id
        if ui.mouse_released and ui.active_id == id:
            if inside:
                clicked = True
            ui.active_id = -1
        if self.draw:
            var d = DrawList()
            var pressed = (ui.active_id == id) and inside
            var hot = ui.hot_id == id
            if active:
                d.add_rect_filled(bx, by, bw, bh, Col4(0.20, 0.33, 0.52, 1.0))
            elif pressed:
                d.add_rect_filled(bx, by, bw, bh, Col4(0.275, 0.431, 0.647, 1.0))
            elif hot:
                d.add_rect_filled(bx, by, bw, bh, Col4(0.314, 0.314, 0.361, 1.0))
            if active:
                d.add_rect_filled(bx, by + bh - 2.5, bw, 2.5, Col4(0.353, 0.667, 0.980, 1.0))
            var icol = Col4(0.9, 0.92, 0.97, 1.0)
            if not active:
                icol = Col4(0.78, 0.80, 0.86, 1.0)
            var sz = bh * 0.30
            if sz > 9.0: sz = 9.0
            draw_icon(d, kind, bx + icon_cell * 0.5 + 2.0, by + bh * 0.5, sz, icol)
            var _f = d.flush()
            _col(0.88, 0.91, 0.96)
            _txt(text, bx + icon_cell, by + (bh - th) * 0.5, 12.0)
            if hot and len(tip) > 0:
                ui.tooltip(tip, mxf, myf)
        self.cx = bx + bw + self.pad
        return clicked

    # vertical divider between groups.
    def sep(mut self) raises:
        if self.draw:
            var d = DrawList()
            var sh = self.h * 0.5
            var sy = self.y + (self.h - sh) * 0.5
            d.add_rect_filled(self.cx + 2.0, sy, 1.0, sh, Col4(0.32, 0.33, 0.38, 1.0))
            var _f = d.flush()
        self.cx = self.cx + 6.0

    # explicit horizontal gap.
    def gap(mut self, w: Float32):
        self.cx = self.cx + w

    # inline text label (vertically centered); advances + returns its width.
    def label(mut self, s: String, size: Float32) raises -> Float32:
        var m = measure_text(s, size)
        var tw = m[0]
        var th = m[1]
        if self.draw:
            _col(0.70, 0.76, 0.86)
            _txt(s, self.cx, self.y + (self.h - th) * 0.5, size)
        self.cx = self.cx + tw + self.pad
        return tw

    # right-align: jump the cursor so the next `n` default-width buttons end flush at the
    # toolbar's right edge.
    def right(mut self, n: Int):
        var needed = Float32(n) * self.btn_w + Float32(n - 1) * self.pad
        self.cx = self.x_end - self.pad - needed

    # current cursor x (left edge of the next item).
    def at(self) -> Float32:
        return self.cx


# Begin a toolbar: draw the panel bar over [x,x+w]x[y,y+h] and return the layout cursor.
# btn_w/pad style the items; draw=False skips all GL (headless layout/hit-test tests).
def begin_toolbar(x: Float32, y: Float32, w: Float32, h: Float32, btn_w: Float32 = 30.0, pad: Float32 = 4.0, draw: Bool = True) raises -> Toolbar:
    if draw:
        var d = DrawList()
        d.add_rect_filled(x, y, w, h, Col4(0.157, 0.161, 0.188, 1.0))           # panel
        d.add_rect_filled(x, y + h - 1.0, w, 1.0, Col4(0.07, 0.07, 0.09, 1.0))  # bottom border
        var _f = d.flush()
    return Toolbar(x + pad, y, h, x + w, btn_w, pad, draw)


# Logic self-test (no GL): fmt_timecode + grade math.
def main() raises:
    var ok = True
    if fmt_timecode(50, 30.0) != String("0:01.66"): ok = False
    if fmt_timecode(390, 30.0) != String("0:13.00"): ok = False
    if fmt_timecode(0, 30.0) != String("0:00.00"): ok = False
    # 2:20.33 = 140.33s @ 30fps = frame 4210 (whole=140 -> 2:20 ; centi from .33)
    print("MEDIA_WIDGETS tc(50,30)=", fmt_timecode(50, 30.0), " tc(390,30)=", fmt_timecode(390, 30.0))
    # grade: identity (bright 0, contrast 1, gain 1) preserves a mid value
    var s = alloc[UInt8](4)
    var dpix = alloc[UInt8](4)
    s[0] = 100; s[1] = 150; s[2] = 200; s[3] = 255
    apply_grade_gain_u8(s, dpix, 4, Float32(0.0), Float32(1.0), Float32(1.0), True)
    var ident = (Int(dpix[0]) == 100) and (Int(dpix[1]) == 150) and (Int(dpix[2]) == 200) and (Int(dpix[3]) == 255)
    if not ident: ok = False
    # gain x2 on 100 -> 200; alpha passthrough
    apply_grade_gain_u8(s, dpix, 4, Float32(0.0), Float32(1.0), Float32(2.0), False)
    var g2 = (Int(dpix[0]) == 200) and (Int(dpix[3]) == 255)
    if not g2: ok = False
    print("MEDIA_WIDGETS grade ident=", ident, " gain2=", g2)

    # ---- Toolbar layout + hit-test (headless: draw=False, no GL) ----------------------
    # bar at x=100,y=10,w=300,h=30; btn_w=30, pad=4 => cursor starts 104, then 138, 172, 206.
    var ui = UiContext()
    # frame 1: press inside button #501 (its rect is [138,168) x [13,37)); click fires on release.
    ui.begin_frame(); ui.set_input(150, 22, True)
    var tb = begin_toolbar(100.0, 10.0, 300.0, 30.0, 30.0, 4.0, False)
    var p0 = tb.button(ui, 500, ICON_PLAY, String(""))
    var p1 = tb.button(ui, 501, ICON_MARK_IN, String(""))
    var p2 = tb.button(ui, 502, ICON_MARK_OUT, String(""))
    # frame 2: release at the same point -> #501 clicks, others don't.
    ui.begin_frame(); ui.set_input(150, 22, False)
    var tb2 = begin_toolbar(100.0, 10.0, 300.0, 30.0, 30.0, 4.0, False)
    var c0 = tb2.button(ui, 500, ICON_PLAY, String(""))
    var c1 = tb2.button(ui, 501, ICON_MARK_IN, String(""))
    var c2 = tb2.button(ui, 502, ICON_MARK_OUT, String(""))
    var press_no_click = (not p0) and (not p1) and (not p2)        # click only on release
    var only_501 = c1 and (not c0) and (not c2)                    # hit-test picked the right one
    var cursor_ok = (tb2.at() >= 205.9) and (tb2.at() <= 206.1)    # 104 + 3*(30+4) = 206
    # sep advances the cursor by 6
    var before = tb2.at(); tb2.sep(); var sep_ok = (tb2.at() - before) >= 5.9 and (tb2.at() - before) <= 6.1
    # right(2): two trailing buttons end flush at the right edge (x_end=400, pad=4, btn_w=30)
    tb2.right(2); var right_ok = (tb2.at() >= 331.9) and (tb2.at() <= 332.1)   # 400 - 4 - (2*30+1*4)=332
    if not (press_no_click and only_501 and cursor_ok and sep_ok and right_ok): ok = False
    print("MEDIA_WIDGETS toolbar press_no_click=", press_no_click, " only_501=", only_501,
          " cursor206=", cursor_ok, " sep6=", sep_ok, " right=", right_ok)

    # ---- ribbon Tab layout + hit-test (headless) -------------------------------------
    # h=30 -> bh=26, icon_cell=23.4 ; tw("Edit")=4*7=28 ; bw=23.4+28+16=67.4 ; rect [104,171.4) x [12,38)
    var ui2 = UiContext()
    ui2.begin_frame(); ui2.set_input(130, 25, True)                 # press inside "Edit"
    var rb = begin_toolbar(100.0, 10.0, 400.0, 30.0, 30.0, 4.0, False)
    var tp = rb.tab(ui2, 600, ICON_SPLIT, String("Edit"), True, String(""))
    ui2.begin_frame(); ui2.set_input(130, 25, False)               # release -> click
    var rb2 = begin_toolbar(100.0, 10.0, 400.0, 30.0, 30.0, 4.0, False)
    var tc = rb2.tab(ui2, 600, ICON_SPLIT, String("Edit"), True, String(""))
    var tab_click = tc and (not tp)                                # click only on release
    var tab_adv = (rb2.at() > 170.0) and (rb2.at() < 176.0)        # 104 + 67.4 + 4(pad) = 175.4
    if not (tab_click and tab_adv): ok = False
    print("MEDIA_WIDGETS tab click=", tab_click, " adv=", tab_adv, " at=", rb2.at())

    if ok: print("MEDIA_WIDGETS_SELFTEST: pass")
    else: print("MEDIA_WIDGETS_SELFTEST: fail")
