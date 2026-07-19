# marker_ruler.mojo — timeline MARKERS on a ruler, an IMMEDIATE-MODE-adjacent widget for
# Mojo 1.0.0b1, drawn on the MojoGUI C rendering backend (librender.so) via external_call ONLY.
#
# Self-contained (own FFI helpers, no module imports beyond std.ffi / std.memory), mirroring the
# idioms of the sibling exemplars:
#   core_widgets.mojo  — to_cstr/col/fill/rect/txt thin wrappers over external_call; Float32() casts.
#   media_widgets.mojo — the Toolbar 'draw: Bool' HEADLESS pattern (draw=False skips ALL GL so the
#                        self-test runs with no backend); seconds_ruler's frame->x mapping
#                        rx = x + Float32(Float64(sec)*fps)/Float32(total)*w.
#   widget_variants.mojo — clampi/clampf geometry helpers; List .copy() read semantics.
#
# A Marker is a (frame, color) pip. add_marker appends; nearest_marker finds the closest marker
# within max_dist frames (pure logic, GL-free); marker_ruler draws colored pips along [x, x+w]
# mapped over total_frames (the ONLY GL path — fully gated by `draw`).
#
# NO `from math` import: the windowed FFI binary cannot link transcendentals; only builtin
# min/max/abs/Int()/Float32() arithmetic is used (same constraint as the exemplars).

from std.ffi import external_call
from std.memory import UnsafePointer, alloc
from builtin.type_aliases import MutExternalOrigin


# ============================================================================
# C backend helpers  (mirror core_widgets.mojo:32-69 EXACTLY)
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
    # 0..255 convenience wrapper (core_widgets.mojo:45-49)
    _ = external_call["set_color", Int32](
        Float32(r) / 255.0, Float32(g) / 255.0, Float32(b) / 255.0, Float32(1.0)
    )


def fill(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_filled_rectangle", Int32](x, y, w, h)


def rect(x: Float32, y: Float32, w: Float32, h: Float32):
    _ = external_call["draw_rectangle", Int32](x, y, w, h)


# ============================================================================
# geometry helpers (mirror widget_variants.mojo:106-115)
# ============================================================================
def clampf(v: Float32, lo: Float32, hi: Float32) -> Float32:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def clampi(v: Int, lo: Int, hi: Int) -> Int:
    return max(min(v, hi), lo)


# ============================================================================
# Marker — a single timeline marker: a frame index + a packed 0xRRGGBB color.
# `color` is an Int holding 24-bit RGB (e.g. 0xFF4040). Pure data, no GL.
# ============================================================================
@fieldwise_init
struct Marker(Copyable, Movable):
    var frame: Int
    var color: Int


# Append a marker (frame, color) to `markers`. Pure logic — no GL, no ordering imposed
# (callers may keep insertion order; nearest_marker scans linearly so order is irrelevant).
def add_marker(mut markers: List[Marker], frame: Int, color: Int):
    markers.append(Marker(frame, color))


# Index of the marker whose frame is closest to `frame`, provided the absolute frame distance
# is <= max_dist. Returns -1 when the list is empty or no marker lies within max_dist. On ties
# the FIRST (lowest-index) qualifying marker wins (linear scan keeps the earliest best).
def nearest_marker(markers: List[Marker], frame: Int, max_dist: Int) -> Int:
    var best = -1
    var best_dist = max_dist + 1            # sentinel strictly worse than any in-range distance
    var n = len(markers)
    for i in range(n):
        var d = markers[i].frame - frame
        if d < 0:
            d = -d                          # abs without libm
        if d <= max_dist and d < best_dist:
            best = i
            best_dist = d
    return best


# ============================================================================
# marker_ruler — draw colored marker pips along [x, x+w] mapped over total_frames.
#
# Each marker's frame maps to px = x + (frame/total_frames)*w (same normalized mapping as
# media_widgets.seconds_ruler). A pip is a thin vertical bar (3px wide) with a small triangle-ish
# cap rendered as a wider top block, in the marker's packed 0xRRGGBB color. Pips whose frame falls
# outside [0, total_frames] are clamped to the ruler edges so they stay visible.
#
# draw=False => HEADLESS: compute the layout (and exercise the same arithmetic) but issue NO
# external_call/GL, so the self-test can run with no backend. This is the media_widgets Toolbar
# 'draw' gate pattern: every GL call sits behind `if draw:`.
# ============================================================================
def marker_ruler(
    x: Float32, y: Float32, w: Float32, total_frames: Int, markers: List[Marker], draw: Bool
) raises:
    var h = Float32(14.0)
    # ruler track background (gated)
    if draw:
        col(24, 24, 28)
        fill(x, y, w, h)
        col(0, 0, 0)
        rect(x, y, w, h)

    var tf = total_frames
    if tf <= 0:
        tf = 1                              # guard div-by-zero; degenerate ruler maps all to x
    var n = len(markers)
    for i in range(n):
        var f = clampi(markers[i].frame, 0, tf)
        var frac = Float32(f) / Float32(tf)
        frac = clampf(frac, Float32(0.0), Float32(1.0))
        var px = x + frac * w
        # keep the 3px pip fully on the track
        var pw = Float32(3.0)
        var pxc = clampf(px - pw / Float32(2.0), x, x + w - pw)
        # unpack 0xRRGGBB into 0..255 components (pure integer arithmetic, no libm)
        var c = markers[i].color
        var r = (c // 65536) % 256
        var g = (c // 256) % 256
        var b = c % 256
        if draw:
            col(r, g, b)
            fill(pxc, y, pw, h)                                  # stem
            fill(pxc - Float32(2.0), y, pw + Float32(4.0), Float32(4.0))  # cap (wider top block)


# ============================================================================
# SELF-TEST  (compiled + run HEADLESSLY by the MAIN LOOP — do NOT run it yourself)
#
# Pure-logic asserts for add_marker/nearest_marker; marker_ruler invoked with draw=False to prove
# it issues no GL and does not crash. Every expected value is derived from first principles below.
# ============================================================================
def main() raises:
    var pass_all = True

    # ----- build a marker list: frames 100, 200, 350 with distinct colors -----
    var markers = List[Marker]()
    add_marker(markers, 100, 0xFF4040)      # red-ish
    add_marker(markers, 200, 0x40FF40)      # green-ish
    add_marker(markers, 350, 0x4060FF)      # blue-ish
    if len(markers) != 3:
        pass_all = False
    print("MARKER_RULER count=", len(markers))   # EXPECT: 3

    # ----- nearest_marker: closest within range -----
    # frame 210, max_dist 30: dists = |100-210|=110, |200-210|=10, |350-210|=140.
    #   only 200 is within 30 -> index 1.
    var n0 = nearest_marker(markers, 210, 30)
    print("MARKER_RULER near(210,30)=", n0)       # EXPECT: 1
    if n0 != 1:
        pass_all = False

    # frame 120, max_dist 50: dists = 20, 80, 230 -> only 100 within 50 -> index 0.
    var n1 = nearest_marker(markers, 120, 50)
    print("MARKER_RULER near(120,50)=", n1)       # EXPECT: 0
    if n1 != 0:
        pass_all = False

    # frame 340, max_dist 25: dists = 240, 140, 10 -> only 350 within 25 -> index 2.
    var n2 = nearest_marker(markers, 340, 25)
    print("MARKER_RULER near(340,25)=", n2)       # EXPECT: 2
    if n2 != 2:
        pass_all = False

    # ----- nearest_marker: beyond range -> -1 -----
    # frame 1000, max_dist 30: dists = 900, 800, 650 -> none within 30 -> -1.
    var n3 = nearest_marker(markers, 1000, 30)
    print("MARKER_RULER near(1000,30)=", n3)      # EXPECT: -1
    if n3 != -1:
        pass_all = False

    # frame 270, max_dist 5: dists = 170, 70, 80 -> none within 5 -> -1.
    var n4 = nearest_marker(markers, 270, 5)
    print("MARKER_RULER near(270,5)=", n4)        # EXPECT: -1
    if n4 != -1:
        pass_all = False

    # tie-break: equidistant 150 between 100 (dist 50) and 200 (dist 50), max_dist 60.
    #   linear scan keeps the FIRST best -> index 0.
    var n5 = nearest_marker(markers, 150, 60)
    print("MARKER_RULER near(150,60)=", n5)       # EXPECT: 0
    if n5 != 0:
        pass_all = False

    # empty list -> -1 (no markers to match).
    var empty = List[Marker]()
    var n6 = nearest_marker(empty, 100, 100)
    if n6 != -1:
        pass_all = False

    # ----- marker_ruler with draw=False must not crash (HEADLESS, no GL) -----
    marker_ruler(Float32(100.0), Float32(50.0), Float32(400.0), 500, markers, False)
    # also exercise the degenerate total_frames<=0 guard and out-of-range clamp paths headlessly.
    marker_ruler(Float32(0.0), Float32(0.0), Float32(200.0), 0, markers, False)
    var oob = List[Marker]()
    add_marker(oob, -50, 0xFFFFFF)          # below 0 -> clamped to left edge
    add_marker(oob, 9999, 0x000000)         # above total -> clamped to right edge
    marker_ruler(Float32(10.0), Float32(10.0), Float32(300.0), 500, oob, False)
    print("MARKER_RULER draw_false_ok= True")

    # ----- summary line the main loop gates on -----
    if pass_all:
        print("MARKER_RULER_SELFTEST: pass")
    else:
        print("MARKER_RULER_SELFTEST: fail")
