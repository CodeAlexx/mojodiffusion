# graphics.canvas — an RGBA8 framebuffer you draw into.
#
# Owns a heap pixel buffer (width*height*4 bytes, row-major, R,G,B,A). Movable
# (the move transfers the pointer) but not Copyable, so the buffer is freed
# exactly once. set_pixel does source-over alpha blending; opaque writes are a
# straight store. Everything else (primitives, text, charts) is built on
# set_pixel / fill_rect.

from std.memory import alloc, UnsafePointer
from graphics.color import Color, rgb

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]


struct Canvas(Movable):
    var w: Int
    var h: Int
    var px: BytePtr  # RGBA8, row-major, w*h*4 bytes
    var clip_x0: Int  # active clip rect [x0,x1) x [y0,y1); defaults to full canvas
    var clip_y0: Int
    var clip_x1: Int
    var clip_y1: Int

    def __init__(out self, width: Int, height: Int):
        self.w = width
        self.h = height
        var n = width * height * 4
        self.px = alloc[UInt8](n if n > 0 else 4)
        for i in range(n):  # default: opaque white
            self.px[i] = 255
        self.clip_x0 = 0
        self.clip_y0 = 0
        self.clip_x1 = width
        self.clip_y1 = height

    def __del__(deinit self):
        self.px.free()

    def in_bounds(self, x: Int, y: Int) -> Bool:
        return x >= 0 and x < self.w and y >= 0 and y < self.h

    def _in_clip(self, x: Int, y: Int) -> Bool:
        return (x >= self.clip_x0 and x < self.clip_x1
                and y >= self.clip_y0 and y < self.clip_y1)

    def set_clip(mut self, x: Int, y: Int, w: Int, h: Int):
        """Restrict drawing to the rect (x,y,w,h), intersected with the canvas.
        set_pixel/put_pixel (and everything built on them) drop pixels outside it."""
        var x0 = x if x > 0 else 0
        var y0 = y if y > 0 else 0
        var x1 = x + w
        var y1 = y + h
        if x1 > self.w:
            x1 = self.w
        if y1 > self.h:
            y1 = self.h
        self.clip_x0 = x0
        self.clip_y0 = y0
        self.clip_x1 = x1 if x1 > x0 else x0
        self.clip_y1 = y1 if y1 > y0 else y0

    def reset_clip(mut self):
        self.clip_x0 = 0
        self.clip_y0 = 0
        self.clip_x1 = self.w
        self.clip_y1 = self.h

    def set_pixel(mut self, x: Int, y: Int, c: Color):
        """Source-over blend `c` onto pixel (x,y). Outside the clip rect (default:
        the whole canvas) is a no-op — so primitives clip for free."""
        if not self._in_clip(x, y):
            return
        var i = (y * self.w + x) * 4
        var sa = Int(c.a)
        if sa == 255:
            self.px[i] = c.r
            self.px[i + 1] = c.g
            self.px[i + 2] = c.b
            self.px[i + 3] = 255
            return
        if sa == 0:
            return
        var ia = 255 - sa
        self.px[i] = UInt8((Int(c.r) * sa + Int(self.px[i]) * ia) // 255)
        self.px[i + 1] = UInt8((Int(c.g) * sa + Int(self.px[i + 1]) * ia) // 255)
        self.px[i + 2] = UInt8((Int(c.b) * sa + Int(self.px[i + 2]) * ia) // 255)
        self.px[i + 3] = 255  # canvas stays opaque

    def put_pixel(mut self, x: Int, y: Int, c: Color):
        """Direct RGBA store (no blending) — for compositing / downsampling where
        the caller supplies the final pixel, including its alpha."""
        if not self._in_clip(x, y):
            return
        var i = (y * self.w + x) * 4
        self.px[i] = c.r
        self.px[i + 1] = c.g
        self.px[i + 2] = c.b
        self.px[i + 3] = c.a

    def get_pixel(self, x: Int, y: Int) -> Color:
        if not self.in_bounds(x, y):
            return Color(UInt8(0), UInt8(0), UInt8(0), UInt8(0))
        var i = (y * self.w + x) * 4
        return Color(self.px[i], self.px[i + 1], self.px[i + 2], self.px[i + 3])

    def clear(mut self, c: Color):
        for y in range(self.h):
            var rb = y * self.w * 4
            for x in range(self.w):
                var i = rb + x * 4
                self.px[i] = c.r
                self.px[i + 1] = c.g
                self.px[i + 2] = c.b
                self.px[i + 3] = 255

    def clear_rgba(mut self, c: Color):
        # Like `clear`, but honors the alpha channel (clear() forces a=255).
        # Needed for transparent backgrounds, e.g. loading SVG icons.
        for y in range(self.h):
            var rb = y * self.w * 4
            for x in range(self.w):
                var i = rb + x * 4
                self.px[i] = c.r
                self.px[i + 1] = c.g
                self.px[i + 2] = c.b
                self.px[i + 3] = c.a

    def fill_rect(mut self, x: Int, y: Int, w: Int, h: Int, c: Color):
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                self.set_pixel(xx, yy, c)

    def blit(mut self, src: Canvas, dx: Int, dy: Int):
        """Composite `src` onto this canvas at (dx,dy), alpha-blended per pixel."""
        for y in range(src.h):
            var rb = y * src.w * 4
            for x in range(src.w):
                var i = rb + x * 4
                self.set_pixel(dx + x, dy + y,
                    Color(src.px[i], src.px[i + 1], src.px[i + 2], src.px[i + 3]))

    def blit_scaled(mut self, src: Canvas, dx: Int, dy: Int, dw: Int, dh: Int, smooth: Bool = False):
        """Composite `src` scaled to dw x dh at (dx,dy), alpha-blended, clip-aware.
        `smooth=False` is nearest-neighbor (sharp/fast); `smooth=True` is bilinear
        (smooth thumbnails). Use for sprites / drawing a chart-canvas at any size."""
        if dw <= 0 or dh <= 0 or src.w <= 0 or src.h <= 0:
            return
        if not smooth:
            for j in range(dh):
                var sy = (j * src.h) // dh
                var rb = sy * src.w * 4
                for i in range(dw):
                    var sx = (i * src.w) // dw
                    var si = rb + sx * 4
                    self.set_pixel(dx + i, dy + j,
                        Color(src.px[si], src.px[si + 1], src.px[si + 2], src.px[si + 3]))
            return
        # bilinear
        var sxr = Float64(src.w) / Float64(dw)
        var syr = Float64(src.h) / Float64(dh)
        for j in range(dh):
            var fy = (Float64(j) + 0.5) * syr - 0.5
            var y0 = Int(fy) if fy >= 0.0 else 0
            if fy < 0.0:
                fy = 0.0
            var y1 = y0 + 1
            if y1 > src.h - 1:
                y1 = src.h - 1
            var ty = fy - Float64(y0)
            for i in range(dw):
                var fx = (Float64(i) + 0.5) * sxr - 0.5
                var x0 = Int(fx) if fx >= 0.0 else 0
                var lfx = fx if fx >= 0.0 else 0.0
                var x1 = x0 + 1
                if x1 > src.w - 1:
                    x1 = src.w - 1
                var tx = lfx - Float64(x0)
                var i00 = (y0 * src.w + x0) * 4
                var i01 = (y0 * src.w + x1) * 4
                var i10 = (y1 * src.w + x0) * 4
                var i11 = (y1 * src.w + x1) * 4
                var r = self._bilerp(Int(src.px[i00]), Int(src.px[i01]), Int(src.px[i10]), Int(src.px[i11]), tx, ty)
                var g = self._bilerp(Int(src.px[i00 + 1]), Int(src.px[i01 + 1]), Int(src.px[i10 + 1]), Int(src.px[i11 + 1]), tx, ty)
                var b = self._bilerp(Int(src.px[i00 + 2]), Int(src.px[i01 + 2]), Int(src.px[i10 + 2]), Int(src.px[i11 + 2]), tx, ty)
                var a = self._bilerp(Int(src.px[i00 + 3]), Int(src.px[i01 + 3]), Int(src.px[i10 + 3]), Int(src.px[i11 + 3]), tx, ty)
                self.set_pixel(dx + i, dy + j, Color(UInt8(r), UInt8(g), UInt8(b), UInt8(a)))

    @staticmethod
    def _bilerp(c00: Int, c01: Int, c10: Int, c11: Int, tx: Float64, ty: Float64) -> Int:
        var top = Float64(c00) * (1.0 - tx) + Float64(c01) * tx
        var bot = Float64(c10) * (1.0 - tx) + Float64(c11) * tx
        return Int(top * (1.0 - ty) + bot * ty + 0.5)

    def downsampled(self, factor: Int) -> Canvas:
        """Box-filter down by `factor` — supersampled anti-aliasing: draw at
        N x size, then downsample to get smooth edges across every primitive,
        text, and chart at once. Returns a new w/factor x h/factor canvas."""
        var f = factor if factor >= 1 else 1
        var ow = self.w // f
        var oh = self.h // f
        var out = Canvas(ow if ow > 0 else 1, oh if oh > 0 else 1)
        var n = f * f
        for y in range(oh):
            for x in range(ow):
                var rs = 0
                var gs = 0
                var bs = 0
                var as_ = 0
                for dy in range(f):
                    var sy = y * f + dy
                    var rb = sy * self.w * 4
                    for dx in range(f):
                        var i = rb + (x * f + dx) * 4
                        rs += Int(self.px[i])
                        gs += Int(self.px[i + 1])
                        bs += Int(self.px[i + 2])
                        as_ += Int(self.px[i + 3])
                out.put_pixel(x, y, Color(UInt8(rs // n), UInt8(gs // n), UInt8(bs // n), UInt8(as_ // n)))
        return out^
