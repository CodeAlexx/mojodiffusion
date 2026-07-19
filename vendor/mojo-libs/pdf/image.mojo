# pdf/image.mojo
# Embed an RGB raster as an Image XObject (FlateDecode compressed).

from graphics.deflate import deflate


def _put_str(mut b: List[UInt8], s: String) raises:
    var src = s.as_bytes()
    for i in range(len(src)):
        b.append(src[i])


def _put_int(mut b: List[UInt8], v: Int) raises:
    if v == 0:
        b.append(UInt8(ord("0")))
        return
    var n = v
    var neg = False
    if n < 0:
        neg = True
        n = -n
    var digits = List[UInt8]()
    while n > 0:
        var d = n % 10
        digits.append(UInt8(ord("0") + d))
        n = n // 10
    if neg:
        b.append(UInt8(ord("-")))
    var k = len(digits)
    while k > 0:
        k = k - 1
        b.append(digits[k])


def _put_real(mut b: List[UInt8], v: Float64) raises:
    var x = v
    var neg = False
    if x < 0.0:
        neg = True
        x = -x

    comptime SCALE = 10000
    var scaled = Int(x * Float64(SCALE) + 0.5)
    var int_part = scaled // SCALE
    var frac_part = scaled % SCALE

    if neg and (int_part != 0 or frac_part != 0):
        b.append(UInt8(ord("-")))

    _put_int(b, int_part)

    if frac_part == 0:
        return

    var frac_digits = List[UInt8]()
    var f = frac_part
    var place = SCALE // 10
    var count = 0
    while count < 4:
        var d = (f // place) % 10
        frac_digits.append(UInt8(ord("0") + d))
        place = place // 10
        count = count + 1

    var last = len(frac_digits)
    while last > 0 and frac_digits[last - 1] == UInt8(ord("0")):
        last = last - 1

    if last == 0:
        return

    b.append(UInt8(ord(".")))
    var j = 0
    while j < last:
        b.append(frac_digits[j])
        j = j + 1


def _zlib_wrap(raw: List[UInt8]) raises -> List[UInt8]:
    # Wrap a raw DEFLATE stream in a zlib (RFC 1950) container so that PDF's
    # /FlateDecode filter accepts it: 2-byte header (CMF=0x78, FLG) + raw
    # deflate body + 4-byte big-endian Adler-32 checksum of the UNcompressed
    # data. graphics.deflate produces RAW deflate, hence this wrapper.
    var body = deflate(raw)
    var z = List[UInt8]()
    # CMF: CM=8 (deflate), CINFO=7 (32K window) -> 0x78.
    z.append(UInt8(0x78))
    # FLG: chosen so (CMF*256 + FLG) % 31 == 0; no preset dict, default level.
    z.append(UInt8(0x9C))
    for i in range(len(body)):
        z.append(body[i])
    # Adler-32 of the uncompressed input.
    var a: Int = 1
    var b2: Int = 0
    var MOD: Int = 65521
    for i in range(len(raw)):
        a = (a + Int(raw[i])) % MOD
        b2 = (b2 + a) % MOD
    var adler = (b2 << 16) | a
    z.append(UInt8((adler >> 24) & 0xFF))
    z.append(UInt8((adler >> 16) & 0xFF))
    z.append(UInt8((adler >> 8) & 0xFF))
    z.append(UInt8(adler & 0xFF))
    return z.copy()


def image_xobject_stream(width: Int, height: Int, rgb: List[UInt8]) raises -> List[UInt8]:
    # rgb is width*height*3 bytes (R,G,B per pixel, row-major).
    # Returns the FULL object body bytes: dict + stream + zlib(deflate(rgb)) + endstream.
    var compressed = _zlib_wrap(rgb)
    var length = len(compressed)

    var b = List[UInt8]()
    _put_str(b, "<< /Type /XObject /Subtype /Image /Width ")
    _put_int(b, width)
    _put_str(b, " /Height ")
    _put_int(b, height)
    _put_str(b, " /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length ")
    _put_int(b, length)
    _put_str(b, " >>\nstream\n")
    for i in range(length):
        b.append(compressed[i])
    _put_str(b, "\nendstream")
    return b.copy()


def draw_image(mut b: List[UInt8], res_name: String, x: Float64, y: Float64, w: Float64, h: Float64) raises:
    # "q\nw 0 0 h x y cm\n/Im1 Do\nQ\n"
    # The cm matrix scales the unit image to w x h at (x, y).
    _put_str(b, "q\n")
    _put_real(b, w)
    _put_str(b, " 0 0 ")
    _put_real(b, h)
    b.append(UInt8(ord(" ")))
    _put_real(b, x)
    b.append(UInt8(ord(" ")))
    _put_real(b, y)
    _put_str(b, " cm\n/")
    _put_str(b, res_name)
    _put_str(b, " Do\nQ\n")
