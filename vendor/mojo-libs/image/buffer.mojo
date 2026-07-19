# image/buffer.mojo — the Image pixel container shared by every codec and op.
#
# Row-major, top-left origin. The CORE path is 8-bit/channel (Gray/RGB/RGBA) —
# what PNG(8)/JPEG/WebP/manipulation use. The struct also carries forward-compat
# metadata (bit_depth, colorspace, icc, exif) so the later "studio" phase can add
# 16-bit / F32-HDR / CMYK / ICC / EXIF WITHOUT changing this type's shape.
#
# Bridges to/from graphics.Canvas (RGBA8) so drawing and photo ops compose.

from std.memory import alloc, UnsafePointer
from graphics.canvas import Canvas
from graphics.color import Color

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]

# Colorspace tags (disambiguate 4-channel RGBA vs CMYK, etc.)
comptime CS_GRAY: Int = 0
comptime CS_RGB: Int = 1
comptime CS_RGBA: Int = 2
comptime CS_CMYK: Int = 3

# Sample formats (studio phase lights up U16 / F32).
comptime FMT_U8: Int = 0
comptime FMT_U16: Int = 1
comptime FMT_F32: Int = 2


def _default_colorspace(channels: Int) raises -> Int:
    if channels == 1:
        return CS_GRAY
    if channels == 3:
        return CS_RGB
    if channels == 4:
        return CS_RGBA
    raise Error("Image: unsupported channel count " + String(channels))


struct Image(Movable):
    var width: Int
    var height: Int
    var channels: Int        # 1=Gray, 3=RGB, 4=RGBA (or CMYK w/ colorspace=CS_CMYK)
    var bit_depth: Int       # 8 (core) | 16 (studio)
    var sample_format: Int   # FMT_U8 (core) | FMT_U16 | FMT_F32 (studio)
    var colorspace: Int      # CS_*
    var data: BytePtr        # width*height*channels*bytes_per_sample bytes
    var icc: List[UInt8]     # embedded ICC profile (empty = none)
    var exif: List[UInt8]    # raw EXIF blob (empty = none)

    @staticmethod
    def new(width: Int, height: Int, channels: Int) raises -> Image:
        """Allocate a zeroed 8-bit image."""
        return Image.new_ex(width, height, channels, 8, FMT_U8)

    @staticmethod
    def new_ex(
        width: Int, height: Int, channels: Int, bit_depth: Int, sample_format: Int
    ) raises -> Image:
        if width <= 0 or height <= 0:
            raise Error("Image: width/height must be > 0")
        var cs = _default_colorspace(channels)
        var bps = bit_depth // 8
        if sample_format == FMT_F32:
            bps = 4
        var n = width * height * channels * bps
        var data = alloc[UInt8](n)
        for i in range(n):
            data[i] = 0
        return Image(width, height, channels, bit_depth, sample_format, cs, data,
                     List[UInt8](), List[UInt8]())

    def __init__(
        out self,
        width: Int,
        height: Int,
        channels: Int,
        bit_depth: Int,
        sample_format: Int,
        colorspace: Int,
        var data: BytePtr,
        var icc: List[UInt8],
        var exif: List[UInt8],
    ):
        self.width = width
        self.height = height
        self.channels = channels
        self.bit_depth = bit_depth
        self.sample_format = sample_format
        self.colorspace = colorspace
        self.data = data
        self.icc = icc^
        self.exif = exif^

    @always_inline
    def byte_len(self) -> Int:
        var bps = self.bit_depth // 8
        if self.sample_format == FMT_F32:
            bps = 4
        return self.width * self.height * self.channels * bps

    @always_inline
    def _idx8(self, x: Int, y: Int, c: Int) -> Int:
        # Byte offset of channel c at (x,y) for the 8-bit path.
        return (y * self.width + x) * self.channels + c

    # ---- 8-bit channel access (core path; raises if not 8-bit) ----
    def get(self, x: Int, y: Int, c: Int) raises -> UInt8:
        if self.bit_depth != 8 or self.sample_format != FMT_U8:
            raise Error("Image.get: 8-bit accessor on a non-8-bit image")
        return self.data[self._idx8(x, y, c)]

    def set(mut self, x: Int, y: Int, c: Int, v: UInt8) raises:
        if self.bit_depth != 8 or self.sample_format != FMT_U8:
            raise Error("Image.set: 8-bit accessor on a non-8-bit image")
        self.data[self._idx8(x, y, c)] = v

    def get_pixel(self, x: Int, y: Int) raises -> Color:
        """Read (x,y) as RGBA8, expanding Gray/RGB. 8-bit path."""
        var base = (y * self.width + x) * self.channels
        if self.channels == 1:
            var g = self.data[base]
            return Color(g, g, g, UInt8(255))
        elif self.channels == 3:
            return Color(self.data[base], self.data[base + 1], self.data[base + 2], UInt8(255))
        else:
            return Color(self.data[base], self.data[base + 1], self.data[base + 2], self.data[base + 3])

    def set_pixel(mut self, x: Int, y: Int, color: Color) raises:
        """Write (x,y) from an RGBA8 Color, collapsing to the image's channels."""
        var base = (y * self.width + x) * self.channels
        if self.channels == 1:
            # luma (BT.601)
            var luma = (UInt32(color.r) * 77 + UInt32(color.g) * 150 + UInt32(color.b) * 29) >> 8
            self.data[base] = UInt8(luma & 0xFF)
        elif self.channels == 3:
            self.data[base] = color.r
            self.data[base + 1] = color.g
            self.data[base + 2] = color.b
        else:
            self.data[base] = color.r
            self.data[base + 1] = color.g
            self.data[base + 2] = color.b
            self.data[base + 3] = color.a

    def clone(self) raises -> Image:
        var n = self.byte_len()
        var d = alloc[UInt8](n)
        for i in range(n):
            d[i] = self.data[i]
        return Image(self.width, self.height, self.channels, self.bit_depth,
                     self.sample_format, self.colorspace, d,
                     self.icc.copy(), self.exif.copy())

    # ---- graphics.Canvas bridge (8-bit) ----
    def to_canvas(self) raises -> Canvas:
        var cv = Canvas(self.width, self.height)
        for y in range(self.height):
            for x in range(self.width):
                cv.put_pixel(x, y, self.get_pixel(x, y))
        return cv^

    @staticmethod
    def from_canvas(cv: Canvas) raises -> Image:
        var img = Image.new(cv.w, cv.h, 4)
        for y in range(cv.h):
            for x in range(cv.w):
                img.set_pixel(x, y, cv.get_pixel(x, y))
        return img^

    def __del__(deinit self):
        self.data.free()
