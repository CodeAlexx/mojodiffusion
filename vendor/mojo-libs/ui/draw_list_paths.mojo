# draw_list_paths.mojo — WORKSTREAM: the full DrawList PATH API on the batch, in pure
# Mojo 1.0.0b1 logic on the MojoGUI C backend (librender.so) via external_call ONLY.
#
# A DrawList accumulates EVERY primitive as triangles into two flat List[Float32]s
#   xy   : interleaved x,y per vertex
#   rgba : interleaved r,g,b,a per vertex
# and flush()s the WHOLE batch in ONE FFI crossing: draw_triangles_batch(xy, rgba, vert_count).
# This is the Mojo analogue of immediate-mode GUI's DrawList: PrimReserve + the path helpers all push
# triangles into one vertex/index stream so the GPU sees a single draw.
#
# FFI idioms mirrored EXACTLY from framebuffer-verified exemplars:
#   - to_cstr wrapper : /home/alex/framepfx-mojo/src/keyframe_editor.mojo (37-43),
#     /tmp/mojoui/draw_list.mojo (11-17).  as_bytes()/alloc do NOT raise in this build, so the
#     to_cstr / unit_cos / unit_sin / rgb / fsqrt module helpers are NON-raising `def` (the
#     exemplars prove this — keyframe_editor.mojo:37 declares a non-raising `def to_cstr`).
#   - DrawList batch shape (xy/rgba/nverts + _vert + add_rect 6-vert + flush() ONE external_call)
#     mirrors /tmp/mojoui/draw_list.mojo (20-57) EXACTLY: flush calls
#       external_call["draw_triangles_batch", Int32](xy.unsafe_ptr(), rgba.unsafe_ptr(), Int32(nverts)).
#   - render harness (initialize_gl_context -> frame_begin -> draw -> fpx_dump_fb -> frame_end ->
#     cleanup_gl) mirrors /tmp/mojoui/draw_list.mojo:60-76 and /tmp/mojoui2/render_gate.mojo.
#
# NO RUNTIME TRIG (HARD RULE): the windowed FFI binary cannot link sincos@GLIBC. Circles/arcs use a
# BAKED comptime unit-circle table UC_COS[i]/UC_SIN[i] of 33 (cos,sin) samples (i = 0..32, angle =
# i/32 * 2pi), computed at SOURCE-WRITE TIME (Python, build-time) and hard-coded as Float32 literals
# — there is ZERO call to sinf/cosf/libm at runtime. Bezier sampling uses only the cubic polynomial
# (mul/add) — also no trig. All geometry is Float32.
#
# INJECTABLE/PURE: the path API is pure geometry — it reads only its arguments and appends to the
# batch; it never touches get_mouse_*/get_char. The self-test injects fixed numeric params and the
# flush returns the exact triangle count, gated against a first-principles-derived constant.

from std.ffi import external_call
from std.memory import UnsafePointer, alloc


# ============================================================================
# C backend helpers  (mirror keyframe_editor.mojo:37-63 / draw_list.mojo:11-17 EXACTLY)
# ============================================================================
def to_cstr(s: String) -> UnsafePointer[Int8, MutExternalOrigin]:
    var b = s.as_bytes()
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(b[i])
    buf[len(b)] = 0
    return buf


# ============================================================================
# BAKED comptime unit-circle table — NO runtime trig.
# 33 entries (i = 0..32); angle_i = (i / 32) * 2*pi; UC_COS[i] = cos(angle_i), UC_SIN[i] = sin.
# Computed at build-time (python -c 'math.cos/sin') and pasted as Float32 literals. The runtime
# code only INDEXES this table — it never evaluates a transcendental. 32 segments is the max; a
# circle of `segs` segments samples indices round(k*32/segs) for k=0..segs (proportional stride).
# ============================================================================
comptime UC_N = 32   # table resolution (entries 0..32 inclusive == one full turn)


# unit_cos(i): baked cosine for table index i in [0, 32].
def unit_cos(i: Int) -> Float32:
    var t = i
    if t < 0:
        t = 0
    if t > UC_N:
        t = UC_N
    if t == 0: return Float32(1.0)
    if t == 1: return Float32(0.98078528)
    if t == 2: return Float32(0.92387953)
    if t == 3: return Float32(0.83146961)
    if t == 4: return Float32(0.70710678)
    if t == 5: return Float32(0.55557023)
    if t == 6: return Float32(0.38268343)
    if t == 7: return Float32(0.19509032)
    if t == 8: return Float32(0.0)
    if t == 9: return Float32(-0.19509032)
    if t == 10: return Float32(-0.38268343)
    if t == 11: return Float32(-0.55557023)
    if t == 12: return Float32(-0.70710678)
    if t == 13: return Float32(-0.83146961)
    if t == 14: return Float32(-0.92387953)
    if t == 15: return Float32(-0.98078528)
    if t == 16: return Float32(-1.0)
    if t == 17: return Float32(-0.98078528)
    if t == 18: return Float32(-0.92387953)
    if t == 19: return Float32(-0.83146961)
    if t == 20: return Float32(-0.70710678)
    if t == 21: return Float32(-0.55557023)
    if t == 22: return Float32(-0.38268343)
    if t == 23: return Float32(-0.19509032)
    if t == 24: return Float32(0.0)
    if t == 25: return Float32(0.19509032)
    if t == 26: return Float32(0.38268343)
    if t == 27: return Float32(0.55557023)
    if t == 28: return Float32(0.70710678)
    if t == 29: return Float32(0.83146961)
    if t == 30: return Float32(0.92387953)
    if t == 31: return Float32(0.98078528)
    return Float32(1.0)   # t == 32 -> cos(2pi) = 1


# unit_sin(i): baked sine for table index i in [0, 32].
def unit_sin(i: Int) -> Float32:
    var t = i
    if t < 0:
        t = 0
    if t > UC_N:
        t = UC_N
    if t == 0: return Float32(0.0)
    if t == 1: return Float32(0.19509032)
    if t == 2: return Float32(0.38268343)
    if t == 3: return Float32(0.55557023)
    if t == 4: return Float32(0.70710678)
    if t == 5: return Float32(0.83146961)
    if t == 6: return Float32(0.92387953)
    if t == 7: return Float32(0.98078528)
    if t == 8: return Float32(1.0)
    if t == 9: return Float32(0.98078528)
    if t == 10: return Float32(0.92387953)
    if t == 11: return Float32(0.83146961)
    if t == 12: return Float32(0.70710678)
    if t == 13: return Float32(0.55557023)
    if t == 14: return Float32(0.38268343)
    if t == 15: return Float32(0.19509032)
    if t == 16: return Float32(0.0)
    if t == 17: return Float32(-0.19509032)
    if t == 18: return Float32(-0.38268343)
    if t == 19: return Float32(-0.55557023)
    if t == 20: return Float32(-0.70710678)
    if t == 21: return Float32(-0.83146961)
    if t == 22: return Float32(-0.92387953)
    if t == 23: return Float32(-0.98078528)
    if t == 24: return Float32(-1.0)
    if t == 25: return Float32(-0.98078528)
    if t == 26: return Float32(-0.92387953)
    if t == 27: return Float32(-0.83146961)
    if t == 28: return Float32(-0.70710678)
    if t == 29: return Float32(-0.55557023)
    if t == 30: return Float32(-0.38268343)
    if t == 31: return Float32(-0.19509032)
    return Float32(0.0)   # t == 32 -> sin(2pi) = 0


# ============================================================================
# Col4 — an RGBA color in 0..1. ImplicitlyCopyable POD so it passes by value cheaply.
# ============================================================================
@fieldwise_init
struct Col4(ImplicitlyCopyable, Movable):
    var r: Float32
    var g: Float32
    var b: Float32
    var a: Float32


# rgb(r,g,b): opaque Col4 helper.
def rgb(r: Float32, g: Float32, b: Float32) -> Col4:
    return Col4(r, g, b, Float32(1.0))


# rgba(r,g,b,a): explicit-alpha Col4 helper.
def rgba(r: Float32, g: Float32, b: Float32, a: Float32) -> Col4:
    return Col4(r, g, b, a)


# A 2D point used by add_polyline (ImplicitlyCopyable POD).
@fieldwise_init
struct Pt(ImplicitlyCopyable, Movable):
    var x: Float32
    var y: Float32


# ============================================================================
# small math helpers (no trig, no libm)
# ============================================================================
# fsqrt(v): a libm-free square root. The ONLY place we need a length is the thick-line normal
# (to scale the perpendicular by thick/2 / len). We do NOT use `v ** 0.5` (that lowers through
# Float64.__pow__ -> _powf_scalar -> std.math.exp/log, pulling in libm — the exact transcendental
# link-failure class the HARD RULE forbids in a windowed FFI binary, per keyframe_editor.mojo:24-30).
# Instead we run a pure-arithmetic Newton-Raphson iteration  x_{n+1} = 0.5*(x_n + v/x_n)  which uses
# only + - * / — no imports, no libm, fully linkable. 8 iterations converge to Float32 accuracy for
# the pixel-scale lengths we feed it; the result only sets stroke-width offsets so sub-pixel error
# is invisible. Zero/negative input returns 0 (degenerate zero-length lines are skipped by add_line).
def fsqrt(v: Float32) -> Float32:
    if v <= Float32(0.0):
        return Float32(0.0)
    # Newton-Raphson for sqrt: x_{n+1} = 0.5*(x_n + v/x_n). Seed with v (good enough; converges).
    var x = v
    if x < Float32(1.0):
        x = Float32(1.0)
    for _ in range(8):
        x = Float32(0.5) * (x + v / x)
    return x


# ============================================================================
# DrawList — the batch accumulator + the full PATH API.
#
# Every add_* helper appends triangles (3 verts each) to xy/rgba; flush() issues ONE
# draw_triangles_batch for the whole frame and returns the TRIANGLE count flushed.
# ============================================================================
struct DrawList(Copyable, Movable):
    var xy: List[Float32]       # interleaved x,y per vertex
    var rgba: List[Float32]     # interleaved r,g,b,a per vertex
    var nverts: Int             # vertices accumulated (== 3 * triangles)

    def __init__(out self):
        self.xy = List[Float32]()
        self.rgba = List[Float32]()
        self.nverts = 0

    # ---- low-level: push one vertex (x,y) with color c ----
    def _vert(mut self, x: Float32, y: Float32, c: Col4):
        self.xy.append(x)
        self.xy.append(y)
        self.rgba.append(c.r)
        self.rgba.append(c.g)
        self.rgba.append(c.b)
        self.rgba.append(c.a)
        self.nverts += 1

    # ---- low-level: push one triangle (3 verts, same color) ----
    def _tri(mut self, x0: Float32, y0: Float32, x1: Float32, y1: Float32,
             x2: Float32, y2: Float32, c: Col4):
        self._vert(x0, y0, c)
        self._vert(x1, y1, c)
        self._vert(x2, y2, c)

    # ---- low-level: push one axis-arbitrary quad (4 corners) as 2 triangles ----
    # Corners given in winding order p0->p1->p2->p3 (e.g. a thick-line quad).
    def _quad(mut self, ax: Float32, ay: Float32, bx: Float32, by: Float32,
              cx: Float32, cy: Float32, dx: Float32, dy: Float32, c: Col4):
        self._tri(ax, ay, bx, by, cx, cy, c)   # a,b,c
        self._tri(ax, ay, cx, cy, dx, dy, c)   # a,c,d

    # ========================================================================
    # add_rect_filled(x,y,w,h,col): a solid rectangle = 2 triangles (6 verts).
    # ========================================================================
    def add_rect_filled(mut self, x: Float32, y: Float32, w: Float32, h: Float32, col: Col4):
        self._quad(x, y, x + w, y, x + w, y + h, x, y + h, col)

    # ========================================================================
    # add_line(x1,y1,x2,y2,thick,col): a thick line = a quad (2 triangles).
    # The quad is the segment swept perpendicular by thick/2 on each side. The perpendicular of
    # the direction (dx,dy) is (-dy,dx); normalized by the segment length (libm-free fsqrt) and
    # scaled by thick/2 gives the offset. Zero-length lines are skipped (no degenerate tris).
    # ========================================================================
    def add_line(mut self, x1: Float32, y1: Float32, x2: Float32, y2: Float32,
                 thick: Float32, col: Col4):
        var dx = x2 - x1
        var dy = y2 - y1
        var len2 = dx * dx + dy * dy
        if len2 <= Float32(0.0):
            return
        var ln = fsqrt(len2)
        var half = thick * Float32(0.5)
        # unit perpendicular * half
        var nx = (-dy / ln) * half
        var ny = (dx / ln) * half
        # quad corners: p1+n, p2+n, p2-n, p1-n  (CCW-ish; winding handled by _quad's two tris)
        self._quad(
            x1 + nx, y1 + ny,
            x2 + nx, y2 + ny,
            x2 - nx, y2 - ny,
            x1 - nx, y1 - ny,
            col,
        )

    # ========================================================================
    # add_rect(x,y,w,h,thick,col): an OUTLINE rect = 4 thick lines (each a quad).
    # Drawn as 4 add_line calls -> 4 quads -> 8 triangles. Corners overlap slightly (acceptable,
    # matches immediate-mode GUI's non-mitered AddRect for thin strokes).
    # ========================================================================
    def add_rect(mut self, x: Float32, y: Float32, w: Float32, h: Float32, thick: Float32, col: Col4):
        var x2 = x + w
        var y2 = y + h
        self.add_line(x, y, x2, y, thick, col)        # top
        self.add_line(x2, y, x2, y2, thick, col)      # right
        self.add_line(x2, y2, x, y2, thick, col)      # bottom
        self.add_line(x, y2, x, y, thick, col)        # left

    # ========================================================================
    # add_triangle_filled(x1,y1,x2,y2,x3,y3,col): one solid triangle (3 verts).
    # ========================================================================
    def add_triangle_filled(mut self, x1: Float32, y1: Float32, x2: Float32, y2: Float32,
                            x3: Float32, y3: Float32, col: Col4):
        self._tri(x1, y1, x2, y2, x3, y3, col)

    # ========================================================================
    # add_circle_filled(cx,cy,r,col,segs): a filled disc = a triangle FAN of `segs` triangles.
    # Each triangle is (center, P_k, P_{k+1}) where P_k uses the BAKED unit-circle table. The
    # stride into the 32-entry table is k*32/segs so any segs in [3,32] maps onto the table with
    # NO runtime trig. -> `segs` triangles == segs*3 vertices.
    # ========================================================================
    def add_circle_filled(mut self, cx: Float32, cy: Float32, r: Float32, col: Col4, segs: Int):
        var n = segs
        if n < 3:
            n = 3
        if n > UC_N:
            n = UC_N
        for k in range(n):
            var i0 = (k * UC_N) // n
            var i1 = ((k + 1) * UC_N) // n
            var ax = cx + r * unit_cos(i0)
            var ay = cy + r * unit_sin(i0)
            var bx = cx + r * unit_cos(i1)
            var by = cy + r * unit_sin(i1)
            self._tri(cx, cy, ax, ay, bx, by, col)

    # ========================================================================
    # add_circle(cx,cy,r,thick,col,segs): an OUTLINE circle = `segs` thick line segments around
    # the perimeter (each a quad = 2 tris). Uses the same baked table. -> segs*2 triangles.
    # ========================================================================
    def add_circle(mut self, cx: Float32, cy: Float32, r: Float32, thick: Float32, col: Col4, segs: Int):
        var n = segs
        if n < 3:
            n = 3
        if n > UC_N:
            n = UC_N
        for k in range(n):
            var i0 = (k * UC_N) // n
            var i1 = ((k + 1) * UC_N) // n
            var ax = cx + r * unit_cos(i0)
            var ay = cy + r * unit_sin(i0)
            var bx = cx + r * unit_cos(i1)
            var by = cy + r * unit_sin(i1)
            self.add_line(ax, ay, bx, by, thick, col)

    # ========================================================================
    # add_bezier_cubic(p0..p3, thick, col, steps): sample the cubic Bezier with the standard
    # polynomial B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3 at t = k/steps for
    # k = 0..steps (NO trig), then emit a POLYLINE of `steps` thick segments. -> steps quads ->
    # steps*2 triangles. (Coincident/zero-length samples are skipped by add_line.)
    # ========================================================================
    def add_bezier_cubic(mut self, p0x: Float32, p0y: Float32, p1x: Float32, p1y: Float32,
                         p2x: Float32, p2y: Float32, p3x: Float32, p3y: Float32,
                         thick: Float32, col: Col4, steps: Int):
        var n = steps
        if n < 1:
            n = 1
        var prevx = p0x
        var prevy = p0y
        for k in range(1, n + 1):
            var t = Float32(k) / Float32(n)
            var u = Float32(1.0) - t
            var w0 = u * u * u
            var w1 = Float32(3.0) * u * u * t
            var w2 = Float32(3.0) * u * t * t
            var w3 = t * t * t
            var px = w0 * p0x + w1 * p1x + w2 * p2x + w3 * p3x
            var py = w0 * p0y + w1 * p1y + w2 * p2y + w3 * p3y
            self.add_line(prevx, prevy, px, py, thick, col)
            prevx = px
            prevy = py

    # ========================================================================
    # add_polyline(points, thick, col, closed): a thick polyline through `points`. With `closed`
    # the last point connects back to the first. Each segment is a thick line (quad = 2 tris).
    # -> (len-1) segments open, (len) segments closed.
    # ========================================================================
    def add_polyline(mut self, points: List[Pt], thick: Float32, col: Col4, closed: Bool):
        var n = len(points)
        if n < 2:
            return
        for i in range(n - 1):
            var a = points[i].copy()
            var b = points[i + 1].copy()
            self.add_line(a.x, a.y, b.x, b.y, thick, col)
        if closed:
            var a = points[n - 1].copy()
            var b = points[0].copy()
            self.add_line(a.x, a.y, b.x, b.y, thick, col)

    # ========================================================================
    # tri_count(): triangles currently accumulated (== nverts / 3). Read without flushing.
    # ========================================================================
    def tri_count(self) -> Int:
        return self.nverts // 3

    # ========================================================================
    # flush(): issue ONE draw_triangles_batch for the whole accumulated stream, then clear and
    # return the number of TRIANGLES that were flushed (== verts/3).
    # ========================================================================
    def flush(mut self) -> Int:
        var tris = self.nverts // 3
        if self.nverts == 0:
            return 0
        _ = external_call["draw_triangles_batch", Int32](
            self.xy.unsafe_ptr(), self.rgba.unsafe_ptr(), Int32(self.nverts)
        )
        self.xy = List[Float32]()
        self.rgba = List[Float32]()
        self.nverts = 0
        return tris


# ============================================================================
# RENDER (separate def): draw the filled rect + circle to the framebuffer so the main loop's
# glReadPixels gate sees real pixels. Initializes GL, batches geometry, flushes in ONE call,
# dumps the framebuffer, tears down. NOT run by the builder — the main loop runs it.
# ============================================================================
comptime WIN_W = 200
comptime WIN_H = 160


def render_paths() raises:
    var ir = external_call["initialize_gl_context", Int32](
        Int32(WIN_W), Int32(WIN_H), to_cstr(String("draw_list paths"))
    )
    if Int(ir) != 0:
        print("FAIL gl init", ir)
        return
    _ = external_call["frame_begin", Int32]()

    var dl = DrawList()
    # red filled rect (top-left quadrant)
    dl.add_rect_filled(Float32(20.0), Float32(20.0), Float32(60.0), Float32(40.0), rgb(Float32(1.0), Float32(0.0), Float32(0.0)))
    # green filled circle (center)
    dl.add_circle_filled(Float32(140.0), Float32(50.0), Float32(28.0), rgb(Float32(0.0), Float32(1.0), Float32(0.0)), 16)
    # blue cubic bezier sweep (lower band)
    dl.add_bezier_cubic(
        Float32(20.0), Float32(120.0),
        Float32(60.0), Float32(80.0),
        Float32(140.0), Float32(150.0),
        Float32(180.0), Float32(110.0),
        Float32(3.0), rgb(Float32(0.3), Float32(0.5), Float32(1.0)), 8,
    )
    var tris = dl.flush()   # ONE external_call for the whole frame
    print("RENDER_PATHS: flushed", tris, "triangles in 1 FFI call")

    _ = external_call["fpx_dump_fb", Int32](
        Int32(WIN_W), Int32(WIN_H), to_cstr(String("/tmp/drawlist_paths_fb.ppm"))
    )
    _ = external_call["frame_end", Int32]()
    _ = external_call["cleanup_gl", Int32]()
    print("draw_list paths -> /tmp/drawlist_paths_fb.ppm")


# ============================================================================
# SELF-TEST  (compiled + run headlessly by the MAIN LOOP — do NOT run it yourself).
#
# First-principles vertex/triangle derivation (each triangle = 3 verts):
#
#   add_rect_filled            -> 1 quad   = 2 triangles =  6 verts
#   add_circle_filled(segs=16) -> 16-tri fan = 16 triangles = 48 verts   (segs triangles, segs*3 verts)
#   add_bezier_cubic(steps=8)  -> 8 thick segments, each a quad = 2 tris
#                                 = 8*2 = 16 triangles = 48 verts        (steps*2 tris, steps*6 verts)
#
#   TOTAL triangles = 2 + 16 + 16 = 34
#   TOTAL vertices  = 6 + 48 + 48 = 102      (== 34 * 3, the batch invariant)
#
# flush() returns the triangle count (== verts/3). We assert:
#   tri_count() before flush == 34 ; nverts == 102 ; flush() returns 34 ; after flush nverts == 0.
# Every add_* issued real geometry, so render_paths() (separate def) gives the render gate pixels.
# ============================================================================
def main() raises:
    var pass_all = True

    var dl = DrawList()

    # --- 1) red filled rect: +2 tris (+6 verts) ---
    dl.add_rect_filled(Float32(10.0), Float32(10.0), Float32(40.0), Float32(30.0),
                       rgb(Float32(1.0), Float32(0.0), Float32(0.0)))
    var after_rect_verts = dl.nverts
    print("DRAWLIST_PATHS_SELFTEST: after_rect_verts=", after_rect_verts, " (expect 6)")
    # EXPECT: after_rect_verts= 6
    if after_rect_verts != 6:
        pass_all = False

    # --- 2) green filled circle, segs=16: +16 tris (+48 verts) -> running 54 verts ---
    dl.add_circle_filled(Float32(100.0), Float32(50.0), Float32(25.0),
                         rgb(Float32(0.0), Float32(1.0), Float32(0.0)), 16)
    var after_circle_verts = dl.nverts
    print("DRAWLIST_PATHS_SELFTEST: after_circle_verts=", after_circle_verts, " (expect 54)")
    # EXPECT: after_circle_verts= 54   (6 rect + 48 circle)
    if after_circle_verts != 54:
        pass_all = False

    # --- 3) blue cubic bezier, steps=8: +16 tris (+48 verts) -> running 102 verts ---
    dl.add_bezier_cubic(
        Float32(20.0), Float32(120.0),
        Float32(60.0), Float32(80.0),
        Float32(140.0), Float32(150.0),
        Float32(180.0), Float32(110.0),
        Float32(2.0), rgb(Float32(0.2), Float32(0.4), Float32(1.0)), 8,
    )
    var after_bezier_verts = dl.nverts
    print("DRAWLIST_PATHS_SELFTEST: after_bezier_verts=", after_bezier_verts, " (expect 102)")
    # EXPECT: after_bezier_verts= 102   (6 + 48 + 48)
    if after_bezier_verts != 102:
        pass_all = False

    # --- triangle count BEFORE flush ---
    var tris_before = dl.tri_count()
    print("DRAWLIST_PATHS_SELFTEST: tris_before_flush=", tris_before, " (expect 34)")
    # EXPECT: tris_before_flush= 34   (2 + 16 + 16)
    if tris_before != 34:
        pass_all = False

    # --- flush returns the triangle count and clears the batch ---
    var flushed = dl.flush()
    var after_flush_verts = dl.nverts
    print("DRAWLIST_PATHS_SELFTEST: flushed_tris=", flushed, " after_flush_verts=", after_flush_verts,
          " (expect 34 then 0)")
    # EXPECT: flushed_tris= 34   after_flush_verts= 0
    if flushed != 34:
        pass_all = False
    if after_flush_verts != 0:
        pass_all = False

    # --- baked-table spot check (no runtime trig): cos(0)=1, sin(8/32 turn = 90deg)=1, cos(16)= -1 ---
    var c0 = unit_cos(0)
    var s8 = unit_sin(8)
    var c16 = unit_cos(16)
    print("DRAWLIST_PATHS_SELFTEST: unit_cos0=", c0, " unit_sin8=", s8, " unit_cos16=", c16,
          " (expect 1.0 1.0 -1.0)")
    # EXPECT: unit_cos0= 1.0   unit_sin8= 1.0   unit_cos16= -1.0
    if c0 != Float32(1.0):
        pass_all = False
    if s8 != Float32(1.0):
        pass_all = False
    if c16 != Float32(-1.0):
        pass_all = False

    # --- secondary primitives exercise (also issue real geometry for completeness):
    #     add_line (1 quad=2 tris=6v), add_rect outline (4 quads=8 tris=24v),
    #     add_triangle_filled (1 tri=3v), add_circle outline segs=12 (12 quads=24 tris=72v),
    #     add_polyline 4 pts closed (4 segs=4 quads=8 tris=24v).
    #     running verts = 6 + 24 + 3 + 72 + 24 = 129 -> tris = 43.
    var dl2 = DrawList()
    dl2.add_line(Float32(0.0), Float32(0.0), Float32(10.0), Float32(0.0), Float32(2.0),
                 rgb(Float32(1.0), Float32(1.0), Float32(1.0)))                      # +6v
    dl2.add_rect(Float32(0.0), Float32(0.0), Float32(20.0), Float32(20.0), Float32(2.0),
                 rgb(Float32(1.0), Float32(1.0), Float32(0.0)))                      # +24v
    dl2.add_triangle_filled(Float32(0.0), Float32(0.0), Float32(5.0), Float32(0.0),
                            Float32(0.0), Float32(5.0), rgb(Float32(0.0), Float32(1.0), Float32(1.0)))  # +3v
    dl2.add_circle(Float32(50.0), Float32(50.0), Float32(20.0), Float32(2.0),
                   rgb(Float32(1.0), Float32(0.0), Float32(1.0)), 12)               # +72v
    var pts = List[Pt]()
    pts.append(Pt(Float32(0.0), Float32(0.0)))
    pts.append(Pt(Float32(10.0), Float32(0.0)))
    pts.append(Pt(Float32(10.0), Float32(10.0)))
    pts.append(Pt(Float32(0.0), Float32(10.0)))
    dl2.add_polyline(pts, Float32(2.0), rgb(Float32(0.5), Float32(0.5), Float32(0.5)), True)  # +24v
    var dl2_verts = dl2.nverts
    var dl2_tris = dl2.tri_count()
    print("DRAWLIST_PATHS_SELFTEST: dl2_verts=", dl2_verts, " dl2_tris=", dl2_tris,
          " (expect 129 then 43)")
    # EXPECT: dl2_verts= 129   dl2_tris= 43   (6 + 24 + 3 + 72 + 24 = 129 ; 129/3 = 43)
    if dl2_verts != 129:
        pass_all = False
    if dl2_tris != 43:
        pass_all = False

    # ----- summary line the main loop gates on -----
    if pass_all:
        print("DRAWLIST_PATHS_SELFTEST: pass (primary 102 verts/34 tris flushed; baked-trig ok; secondary 129v/43t)")
    else:
        print("DRAWLIST_PATHS_SELFTEST: fail")
