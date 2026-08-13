# draw_list.mojo — Tier-4 DrawList-style batch. Accumulate all geometry, flush
# in ONE FFI call (draw_triangles_batch) instead of one external_call per primitive.
from std.ffi import external_call
from std.memory import UnsafePointer, alloc

comptime WIN_W = 170
comptime WIN_H = 80


fn to_cstr(s: String) raises -> UnsafePointer[Int8, MutExternalOrigin]:
    var b = s.as_bytes()
    var buf = alloc[Int8](len(b) + 1)
    for i in range(len(b)):
        buf[i] = Int8(b[i])
    buf[len(b)] = 0
    return buf


struct DrawList(Copyable, Movable):
    var xy: List[Float32]       # interleaved x,y per vertex
    var rgba: List[Float32]     # interleaved r,g,b,a per vertex
    var nverts: Int
    var prims: Int              # primitive count batched (for the FFI-reduction metric)

    def __init__(out self):
        self.xy = List[Float32]()
        self.rgba = List[Float32]()
        self.nverts = 0
        self.prims = 0

    def _vert(mut self, x: Float32, y: Float32, r: Float32, g: Float32, b: Float32, a: Float32):
        self.xy.append(x); self.xy.append(y)
        self.rgba.append(r); self.rgba.append(g); self.rgba.append(b); self.rgba.append(a)
        self.nverts += 1

    # one rect = 2 triangles (6 verts), all the same color.
    def add_rect(mut self, x: Float32, y: Float32, w: Float32, h: Float32,
                 r: Float32, g: Float32, b: Float32, a: Float32):
        self._vert(x, y, r, g, b, a)
        self._vert(x + w, y, r, g, b, a)
        self._vert(x + w, y + h, r, g, b, a)
        self._vert(x, y, r, g, b, a)
        self._vert(x + w, y + h, r, g, b, a)
        self._vert(x, y + h, r, g, b, a)
        self.prims += 1

    # flush ALL accumulated geometry in ONE FFI crossing.
    def flush(mut self) -> Int:
        if self.nverts == 0:
            return 0
        _ = external_call["draw_triangles_batch", Int32](
            self.xy.unsafe_ptr(), self.rgba.unsafe_ptr(), Int32(self.nverts))
        var n = self.prims
        self.xy = List[Float32](); self.rgba = List[Float32]()
        self.nverts = 0; self.prims = 0
        return n   # number of primitives flushed in this single call


def main() raises:
    var ir = external_call["initialize_gl_context", Int32](
        Int32(WIN_W), Int32(WIN_H), to_cstr(String("draw_list batch")))
    if Int(ir) != 0: print("FAIL gl init"); return
    _ = external_call["frame_begin", Int32]()

    var dl = DrawList()
    dl.add_rect(Float32(20), Float32(20), Float32(30), Float32(30), Float32(1), Float32(0), Float32(0), Float32(1))   # red
    dl.add_rect(Float32(70), Float32(20), Float32(30), Float32(30), Float32(0), Float32(1), Float32(0), Float32(1))   # green
    dl.add_rect(Float32(120), Float32(20), Float32(30), Float32(30), Float32(0), Float32(0), Float32(1), Float32(1))  # blue
    var n = dl.flush()   # ONE external_call for all 3 rects
    print("flushed", n, "primitives in 1 FFI call (expect 3)")

    _ = external_call["fpx_dump_fb", Int32](Int32(WIN_W), Int32(WIN_H), to_cstr(String("/tmp/drawlist_fb.ppm")))
    _ = external_call["frame_end", Int32]()
    _ = external_call["cleanup_gl", Int32]()
    print("draw_list batch -> /tmp/drawlist_fb.ppm")
