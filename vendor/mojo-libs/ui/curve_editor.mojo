# =============================================================================
# curve_editor.mojo  --  KEYFRAME CURVE EDITOR
# Mojo 1.0.0b1 port of MediaEditor's immediate-mode GUI keyframe curve editor addon, drawn DIRECTLY on
# the MojoGUI C backend (librender.so) via external_call ONLY. No other Mojo dep.
#
# BEHAVIOR REFERENCE (cited lines are in the MediaEditor tree under /tmp/Med):
#   /tmp/Med/immediate-mode/addon/ImCurve/immediate-mode_curve.h
#     :89-114   struct Curve  -> name, points(vector<KeyPoint>), color, type, min/max
#     :67-87    struct KeyPoint -> ImVec4 val (x,y,z,t) + CurveType type; here we
#               use the 2D form (t = time on X in [0,1], value on Y in [0,1]).
#     :26-57    enum CurveType { ... Linear=2, Smooth=3, ... } -- we port the two
#               core point-edit types Linear & Smooth.
#   /tmp/Med/immediate-mode/addon/ImCurve/immediate-mode_curve.cpp
#     :38-43    smoothstep(): Linear -> return t ; Smooth -> t*t*(3-2*t)   (we mirror)
#     :139-165  DrawPoint(): handle at pos*size+offset; ±5px hit anchor box;
#               WHITE(0xFFFFFFFF) when edited/selected, light-blue when hovered,
#               blue(0xFF0080FF) otherwise. We draw it as a small filled circle.
#     :367-389  curve polyline: between consecutive points sample subStepCount-1
#               segments; rt = smoothstep(p1.x,p2.x,sp.x,ctype); AddLine each.
#     :392-411  per-point DrawPoint + hover/selection bookkeeping.
#     :418-481  drag selected point: p = original + mouseDelta*sizeOfPixel, then
#               clamped to value range (CURVE_EDIT_FLAG_VALUE_LIMITED).
#     :494-508  add point with double-click on the over-curve: np = (mouse-offset)/
#               viewSize -> range; AddPoint(curve, np, ctype).
#     :522-539  delete point with (shift+)click on an over point.
#   /tmp/Med/immediate-mode/addon/ImCurve/immediate-mode_curve.cpp:976-1050 KeyPointEditor::AddPoint/
#               DeletePoint/SortValues -> points kept SORTED by t after edits.
#
# Y-AXIS NOTE: immediate-mode GUI keyframe curve editor uses viewSize.y = -edit_size.y (curve.cpp:197) so
# value increases UPWARD on screen. We mirror: larger value => smaller screen-y.
#
# MATH NOTE (mirrors the proven keyframe_editor.mojo exemplar): hit-tests compare
# SQUARED distances (no `**0.5`) to avoid pulling libm exp/log into a windowed FFI
# binary. smoothstep Linear/Smooth use only +,-,* (no transcendentals). This keeps
# us inside the same safe-symbol set as the verified exemplar.
# =============================================================================
from std.ffi import external_call
from std.memory import UnsafePointer, alloc


# ---------- backend FFI helpers (idioms copied from keyframe_editor.mojo) ----------
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


def col_a(r: Int, g: Int, b: Int, a: Int):
    _ = external_call["set_color", Int32](
        Float32(r) / 255.0, Float32(g) / 255.0, Float32(b) / 255.0, Float32(a) / 255.0
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


# ---------- curve math (immediate-mode_curve.cpp:38-43 smoothstep) ----------
# CurveType: we port the two core point-edit types.
comptime CT_LINEAR: Int = 2   # immediate-mode_curve.h:31  Linear
comptime CT_SMOOTH: Int = 3   # immediate-mode_curve.h:32  Smooth


# smoothstep(edge0, edge1, x, ctype): returns the eased blend factor in [0,1].
# Linear -> t (straight lerp). Smooth -> t*t*(3-2*t) (cubic). (curve.cpp:38-43)
def ease(t_in: Float32, ctype: Int) -> Float32:
    var t = t_in
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    if ctype == CT_SMOOTH:
        return t * t * (Float32(3.0) - Float32(2.0) * t)
    return t  # CT_LINEAR (and default)


# ---------- model ----------
# KeyPoint: (t in [0,1] time on X, value in [0,1] on Y, point edit type).
# Mirrors immediate-mode_curve.h:67-87 KeyPoint(val.w=t, val.<dim>=value, type=CurveType).
# POD -> ImplicitlyCopyable/Movable so List ops are cheap.
@fieldwise_init
struct CurvePoint(ImplicitlyCopyable, Movable):
    var t: Float32       # time   (X), normalized [0,1]
    var value: Float32   # value  (Y), normalized [0,1]
    var ptype: Int       # CT_LINEAR | CT_SMOOTH (per-point, immediate-mode_curve.h:174 EditPoint ctype)


# Result of a handle hit-test: (curve index, point index); both -1 when nothing hit.
@fieldwise_init
struct Hit2(ImplicitlyCopyable, Movable):
    var curve: Int
    var point: Int


# A named curve = list of points + color + default per-point type.
# Mirrors immediate-mode_curve.h:89-114 struct Curve.
struct Curve(Copyable, Movable):
    var name: String
    var points: List[CurvePoint]   # kept SORTED by t (KeyPointEditor::SortValues)
    var cr: Int
    var cg: Int
    var cb: Int
    var default_type: Int          # type assigned to newly added points

    def __init__(out self, var name: String, cr: Int, cg: Int, cb: Int, default_type: Int):
        self.name = name^
        self.points = List[CurvePoint]()
        self.cr = cr
        self.cg = cg
        self.cb = cb
        self.default_type = default_type

    # insert a point keeping the list sorted by t (curve.cpp:976-1024 AddPoint+SortValues).
    # Implemented as append + a sorted shift loop (no List.insert) using .copy() on the
    # read AND write-back per the owned-type rules.
    def add_point(mut self, t: Float32, value: Float32, ptype: Int):
        self.points.append(CurvePoint(t, value, ptype))
        var i = len(self.points) - 1
        while i > 0 and self.points[i - 1].t > t:
            var a = self.points[i - 1].copy()
            var b = self.points[i].copy()
            self.points[i - 1] = b^
            self.points[i] = a^
            i -= 1

    # remove point at index (curve.cpp:1026-1050 DeletePoint)
    def delete_point(mut self, idx: Int):
        if idx < 0 or idx >= len(self.points):
            return
        _ = self.points.pop(idx)


# ---------- editor ----------
# Holds multiple named curves and the interactive edit state. Drawing area is the
# rectangle (ax,ay,aw,ah); the plotting plane sits below a HEADER_H label strip.
struct CurveEditor(Copyable, Movable):
    var curves: List[Curve]
    var ax: Float32
    var ay: Float32
    var aw: Float32
    var ah: Float32
    # selection / drag state (curve.cpp Delegate fields: selectedPoints/overSelectedPoint)
    var sel_curve: Int       # selected curve index, -1 none
    var sel_point: Int       # selected point index within sel_curve, -1 none
    var dragging: Bool       # mouse currently capturing a point
    var prev_down: Bool      # edge detection for click vs hold

    comptime HEADER_H: Float32 = 24.0   # top label strip
    comptime HANDLE_R: Float32 = 4.0    # handle circle radius (DrawPoint ~5px anchor)
    comptime HIT_R2: Float32 = 49.0     # hit radius^2 (7px) for grabbing a handle
    comptime LINE_TH: Float32 = 2.0     # curve polyline thickness
    comptime SEG_SAMPLES: Int = 16      # samples per segment (curve.cpp subStepCount)
    comptime ADD_GAP: Float32 = 0.012   # min t-gap from existing point to allow an ADD click

    def __init__(out self, ax: Float32, ay: Float32, aw: Float32, ah: Float32):
        self.curves = List[Curve]()
        self.ax = ax
        self.ay = ay
        self.aw = aw
        self.ah = ah
        self.sel_curve = -1
        self.sel_point = -1
        self.dragging = False
        self.prev_down = False

    # ---- plane geometry (below header strip) ----
    def _plane_x(self) -> Float32:
        return self.ax

    def _plane_y(self) -> Float32:
        return self.ay + Self.HEADER_H

    def _plane_w(self) -> Float32:
        return self.aw

    def _plane_h(self) -> Float32:
        return self.ah - Self.HEADER_H

    # normalized (t,value) -> screen pixel. Y inverted (value up). (curve.cpp:197,380)
    def _to_px(self, t: Float32) -> Float32:
        return self._plane_x() + t * self._plane_w()

    def _to_py(self, value: Float32) -> Float32:
        return self._plane_y() + (Float32(1.0) - value) * self._plane_h()

    # screen pixel -> normalized (inverse of the above) for click/drag mapping.
    def _from_px(self, px: Float32) -> Float32:
        var w = self._plane_w()
        if w <= 0.0:
            return 0.0
        var t = (px - self._plane_x()) / w
        if t < 0.0:
            t = 0.0
        if t > 1.0:
            t = 1.0
        return t

    def _from_py(self, py: Float32) -> Float32:
        var h = self._plane_h()
        if h <= 0.0:
            return 0.0
        var v = Float32(1.0) - (py - self._plane_y()) / h
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return v

    # ---- hit-test a point handle (curve.cpp DrawPoint ±5px anchor) ----
    # Returns a Hit2(curve_index, point_index); scans every curve with a SQUARED
    # distance compare, no sqrt (exemplar rule). (-1,-1) when nothing is hit.
    def hit_point(self, mx: Float32, my: Float32) -> Hit2:
        var best = Float32(Self.HIT_R2)
        var rc = -1
        var rp = -1
        for ci in range(len(self.curves)):
            var n = len(self.curves[ci].points)
            for pi in range(n):
                var p = self.curves[ci].points[pi]
                var hx = self._to_px(p.t)
                var hy = self._to_py(p.value)
                var dx = mx - hx
                var dy = my - hy
                var d2 = dx * dx + dy * dy
                if d2 <= best:
                    best = d2
                    rc = ci
                    rp = pi
        return Hit2(rc, rp)

    # is (mx,my) inside the plotting plane?
    def _in_plane(self, mx: Float32, my: Float32) -> Bool:
        return (
            mx >= self._plane_x()
            and mx <= self._plane_x() + self._plane_w()
            and my >= self._plane_y()
            and my <= self._plane_y() + self._plane_h()
        )

    # is there already a point near time t in curve ci? (avoid stacking on ADD)
    def _has_point_near_t(self, ci: Int, t: Float32) -> Bool:
        for pi in range(len(self.curves[ci].points)):
            var dt = self.curves[ci].points[pi].t - t
            if dt < 0.0:
                dt = -dt
            if dt < Self.ADD_GAP:
                return True
        return False

    # ==========================================================================
    # update(mx,my,down): SINGLE interaction step driven by injected mouse state.
    # NEVER reads get_mouse_* directly (headless verification requirement).
    #   - press on a handle      -> select + begin drag        (curve.cpp:392-411,418)
    #   - hold while dragging     -> move selected point         (curve.cpp:418-481)
    #   - press on empty plane    -> ADD a point to curve 0      (curve.cpp:494-508)
    #   - release                 -> end drag                    (curve.cpp:483-491)
    # ==========================================================================
    def update(mut self, mx: Int, my: Int, down: Bool):
        var fmx = Float32(mx)
        var fmy = Float32(my)
        var pressed = down and (not self.prev_down)   # rising edge = a click
        var released = (not down) and self.prev_down

        if pressed:
            # try to grab a handle first
            var h = self.hit_point(fmx, fmy)
            if h.curve >= 0:
                self.sel_curve = h.curve
                self.sel_point = h.point
                self.dragging = True
            else:
                # empty-plane click -> ADD a point to curve 0 at clicked (t,value)
                if self._in_plane(fmx, fmy) and len(self.curves) > 0:
                    var t = self._from_px(fmx)
                    var v = self._from_py(fmy)
                    if not self._has_point_near_t(0, t):
                        var c0 = self.curves[0].copy()
                        c0.add_point(t, v, c0.default_type)
                        self.curves[0] = c0^
                        # select the freshly added point (find by t)
                        self.sel_curve = 0
                        self.sel_point = -1
                        for pi in range(len(self.curves[0].points)):
                            if self.curves[0].points[pi].t == t:
                                self.sel_point = pi
                        self.dragging = True
        elif down and self.dragging and self.sel_curve >= 0 and self.sel_point >= 0:
            # move the selected point. X clamped between neighbors so order is kept
            # (curve.cpp clamps via VALUE_LIMITED + neighbor ordering on sort).
            var ci = self.sel_curve
            var pi = self.sel_point
            var new_t = self._from_px(fmx)
            var new_v = self._from_py(fmy)
            var c = self.curves[ci].copy()
            var lo = Float32(0.0)
            var hi = Float32(1.0)
            if pi > 0:
                lo = c.points[pi - 1].t
            if pi < len(c.points) - 1:
                hi = c.points[pi + 1].t
            if new_t < lo:
                new_t = lo
            if new_t > hi:
                new_t = hi
            var moved = c.points[pi].copy()
            moved.t = new_t
            moved.value = new_v
            c.points[pi] = moved^
            self.curves[ci] = c^
        elif released:
            self.dragging = False

        self.prev_down = down

    # delete the currently selected point (curve.cpp:522-539 shift+click delete).
    # Exposed as an explicit call so the caller can bind it to a modifier+click.
    def delete_selected(mut self):
        if self.sel_curve < 0 or self.sel_point < 0:
            return
        var c = self.curves[self.sel_curve].copy()
        c.delete_point(self.sel_point)
        self.curves[self.sel_curve] = c^
        self.sel_point = -1
        self.dragging = False

    # ==========================================================================
    # draw(self): issue real MojoGUI draw_* for EVERY element so the framebuffer
    # verifier (fpx_dump_fb) sees grid + axes + each curve polyline + handles.
    # ==========================================================================
    def draw(self):
        var px = self._plane_x()
        var py = self._plane_y()
        var pw = self._plane_w()
        var ph = self._plane_h()

        # panel background + header strip + first-curve label
        col(34, 34, 39)
        fill(self.ax, self.ay, self.aw, self.ah)
        col(28, 28, 32)
        fill(self.ax, self.ay, self.aw, Self.HEADER_H)
        col(200, 200, 208)
        if len(self.curves) > 0:
            txt(self.curves[0].name, self.ax + 8.0, self.ay + 6.0, 12.0)
        else:
            txt(String("Curve Editor"), self.ax + 8.0, self.ay + 6.0, 12.0)

        # plotting-plane background
        col(22, 22, 26)
        fill(px, py, pw, ph)

        # ---- grid (graticule, immediate-mode_curve.h:214 GetGraticuleColor ~0x202020) ----
        col_a(60, 60, 68, 160)
        var gv = 0
        while gv <= 4:
            var gy = py + Float32(gv) * (ph / Float32(4.0))
            line_t(px, gy, px + pw, gy, 1.0)
            gv += 1
        var gh = 0
        while gh <= 8:
            var gx = px + Float32(gh) * (pw / Float32(8.0))
            line_t(gx, py, gx, py + ph, 1.0)
            gh += 1

        # ---- axes (left + bottom emphasized) ----
        col(90, 90, 100)
        line_t(px, py, px, py + ph, 2.0)            # value (Y) axis
        line_t(px, py + ph, px + pw, py + ph, 2.0)  # time (X) axis

        # ---- each curve: polyline (sampled smoothstep) then handles on top ----
        for ci in range(len(self.curves)):
            var cr = self.curves[ci].cr
            var cg = self.curves[ci].cg
            var cb = self.curves[ci].cb
            var n = len(self.curves[ci].points)
            if n == 0:
                continue

            # leading flat run from left edge to first point (visual continuity)
            var p0 = self.curves[ci].points[0]
            var fy = self._to_py(p0.value)
            col(cr, cg, cb)
            line_t(px, fy, self._to_px(p0.t), fy, Self.LINE_TH)

            # between consecutive points, sample the eased blend (curve.cpp:367-389)
            var si = 0
            while si < n - 1:
                var a = self.curves[ci].points[si]
                var b = self.curves[ci].points[si + 1]
                var ax0 = self._to_px(a.t)
                var ay0 = self._to_py(a.value)
                var ax1 = self._to_px(b.t)
                var ay1 = self._to_py(b.value)
                var prevx = ax0
                var prevy = ay0
                var s = 1
                while s <= Self.SEG_SAMPLES:
                    var tt = Float32(s) / Float32(Self.SEG_SAMPLES)
                    # rt = smoothstep(p1.x,p2.x,sp.x,ctype) ; here sp.x is linear in tt
                    var rt = ease(tt, a.ptype)
                    var sx = ax0 + (ax1 - ax0) * tt          # X advances linearly
                    var sy = ay0 + (ay1 - ay0) * rt          # Y eased by point type
                    col(cr, cg, cb)
                    line_t(prevx, prevy, sx, sy, Self.LINE_TH)
                    prevx = sx
                    prevy = sy
                    s += 1
                si += 1

            # trailing flat run from last point to right edge
            var pl = self.curves[ci].points[n - 1]
            var ly = self._to_py(pl.value)
            col(cr, cg, cb)
            line_t(self._to_px(pl.t), ly, px + pw, ly, Self.LINE_TH)

            # handles (DrawPoint): white when selected, point color otherwise
            var hi = 0
            while hi < n:
                var p = self.curves[ci].points[hi]
                var hx = self._to_px(p.t)
                var hy = self._to_py(p.value)
                if ci == self.sel_curve and hi == self.sel_point:
                    col(255, 255, 255)
                    fcircle(hx, hy, Self.HANDLE_R + Float32(2.0), 16)
                    col(255, 255, 255)
                    fcircle(hx, hy, Self.HANDLE_R, 16)
                else:
                    col(cr, cg, cb)
                    fcircle(hx, hy, Self.HANDLE_R, 16)
                # outline ring for visibility (DrawPoint AddPolyline)
                col(10, 10, 12)
                rect(hx - Self.HANDLE_R, hy - Self.HANDLE_R, Self.HANDLE_R * 2.0, Self.HANDLE_R * 2.0)
                hi += 1


# =============================================================================
# SELF-TEST (do NOT run; the single builder compiles & framebuffer-verifies).
# Builds a curve with 3 points, drags the MIDDLE point UP by +20px in value, then
# clicks empty grid to ADD a point. Asserts value delta + point-count rise.
# =============================================================================
def main() raises:
    # plane: ax=0 ay=0 aw=400 ah=300 -> plane is (0,24)..(400,300): pw=400 ph=276
    var ed = CurveEditor(Float32(0.0), Float32(0.0), Float32(400.0), Float32(300.0))
    var c = Curve(String("Opacity"), 255, 140, 0, CT_LINEAR)
    c.add_point(Float32(0.0), Float32(0.2), CT_LINEAR)   # t=0.0  v=0.20
    c.add_point(Float32(0.5), Float32(0.5), CT_LINEAR)   # t=0.5  v=0.50  <- MIDDLE
    c.add_point(Float32(1.0), Float32(0.8), CT_LINEAR)   # t=1.0  v=0.80
    ed.curves.append(c^)

    # --- middle point pixel position ---
    # t=0.5 -> px = 0 + 0.5*400 = 200
    # v=0.5 -> py = 24 + (1-0.5)*276 = 24 + 138 = 162
    var mid_px = ed._to_px(Float32(0.5))   # expect 200.0
    var mid_py = ed._to_py(Float32(0.5))   # expect 162.0

    var before_v = ed.curves[0].points[1].value   # expect 0.5

    # press ON the middle handle (selects + starts drag)
    ed.update(Int(mid_px), Int(mid_py), True)
    # drag UP by 20px (screen-y DECREASES => value INCREASES).
    # new value = 1 - (162-20 - 24)/276 = 1 - (118)/276 = 1 - 0.42753 = 0.57246...
    # delta = +0.072463...  (==> +20/276)
    ed.update(Int(mid_px), Int(mid_py) - 20, True)
    ed.update(Int(mid_px), Int(mid_py) - 20, False)   # release

    var after_v = ed.curves[0].points[1].value
    var delta = after_v - before_v
    # EXPECTED delta = 20/276 = 0.0724637...   (assert within tol)
    var ok_drag = (delta > Float32(0.066)) and (delta < Float32(0.079))

    # --- ADD a point on empty grid ---
    var count_before = len(ed.curves[0].points)   # expect 3
    # click at t=0.25 (px=100), value=0.30 -> py = 24 + 0.7*276 = 24 + 193.2 = 217.2
    var add_px = ed._to_px(Float32(0.25))   # 100.0
    var add_py = ed._to_py(Float32(0.30))   # 217.2
    ed.update(Int(add_px), Int(add_py), True)
    ed.update(Int(add_px), Int(add_py), False)
    var count_after = len(ed.curves[0].points)   # expect 4
    var ok_add = (count_after == count_before + 1)

    # issue the draw calls once (framebuffer-verifiable)
    ed.draw()

    print("CURVE_SELFTEST:",
        "before_v=", before_v, "after_v=", after_v, "delta=", delta,
        "drag_ok=", ok_drag,
        "count_before=", count_before, "count_after=", count_after, "add_ok=", ok_add)
    # EXPECTED self-test numbers (for the main-loop gate):
    #   before_v       == 0.5
    #   after_v        ~= 0.5724638   (0.5 + 20/276)
    #   delta          ~= 0.0724638   (== 20/276)  -> drag_ok == True
    #   count_before   == 3
    #   count_after    == 4           -> add_ok == True
