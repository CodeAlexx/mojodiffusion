# svg/style.mojo — color / transform / viewBox / opacity parsing for SVG.

from std.math import pi
from graphics.color import Color, rgb, rgba
from graphics.transform import Affine, identity, translate, scale, rotate


struct Paint(Copyable, ImplicitlyCopyable, Movable):
    var has: Bool          # False == "none" (no paint)
    var color: Color
    def __init__(out self, has: Bool, color: Color):
        self.has = has
        self.color = color


def _trim(s: String) -> String:
    var b = s.as_bytes()
    var n = len(b)
    var i = 0
    while i < n and (Int(b[i]) == 0x20 or Int(b[i]) == 0x09 or Int(b[i]) == 0x0A or Int(b[i]) == 0x0D):
        i += 1
    var j = n
    while j > i and (Int(b[j - 1]) == 0x20 or Int(b[j - 1]) == 0x09 or Int(b[j - 1]) == 0x0A or Int(b[j - 1]) == 0x0D):
        j -= 1
    if j <= i:
        return String("")
    return String(StringSlice(ptr=b.unsafe_ptr() + i, length=j - i))


def _lower(s: String) -> String:
    var out = String("")
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = Int(b[i])
        if c >= 65 and c <= 90:
            c += 32
        out += chr(c)
    return out^


def _hexval(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 97 + 10
    if c >= 65 and c <= 70:
        return c - 65 + 10
    return 0


def parse_color(raw: String, current: Color) raises -> Paint:
    var s = _trim(raw)
    if len(s) == 0:
        return Paint(False, current)
    var low = _lower(s)
    if low == "none" or low == "transparent":
        return Paint(False, current)
    if low == "currentcolor":
        return Paint(True, current)

    var b = s.as_bytes()
    if Int(b[0]) == 35:   # '#'
        var hexs = String("")
        for i in range(1, len(b)):
            hexs += chr(Int(b[i]))
        var hb = hexs.as_bytes()
        var hn = len(hb)
        if hn == 3:
            var r = _hexval(Int(hb[0])); r = r * 16 + r
            var g = _hexval(Int(hb[1])); g = g * 16 + g
            var bl = _hexval(Int(hb[2])); bl = bl * 16 + bl
            return Paint(True, rgb(r, g, bl))
        if hn >= 6:
            var r = _hexval(Int(hb[0])) * 16 + _hexval(Int(hb[1]))
            var g = _hexval(Int(hb[2])) * 16 + _hexval(Int(hb[3]))
            var bl = _hexval(Int(hb[4])) * 16 + _hexval(Int(hb[5]))
            return Paint(True, rgb(r, g, bl))
        return Paint(True, current)

    if low.startswith(String("rgb")):
        # rgb(r,g,b) or rgba(r,g,b,a) — integers only (percent not supported)
        var nums = _nums_in_parens(s)
        if len(nums) >= 3:
            var a = Int(nums[3]) if len(nums) >= 4 else 255
            return Paint(True, rgba(Int(nums[0]), Int(nums[1]), Int(nums[2]), a))
        return Paint(True, current)

    # named subset
    if low == "black": return Paint(True, rgb(0, 0, 0))
    if low == "white": return Paint(True, rgb(255, 255, 255))
    if low == "red": return Paint(True, rgb(255, 0, 0))
    if low == "green": return Paint(True, rgb(0, 128, 0))
    if low == "lime": return Paint(True, rgb(0, 255, 0))
    if low == "blue": return Paint(True, rgb(0, 0, 255))
    if low == "yellow": return Paint(True, rgb(255, 255, 0))
    if low == "orange": return Paint(True, rgb(255, 165, 0))
    if low == "purple": return Paint(True, rgb(128, 0, 128))
    if low == "gray" or low == "grey": return Paint(True, rgb(128, 128, 128))
    if low == "silver": return Paint(True, rgb(192, 192, 192))
    if low == "cyan" or low == "aqua": return Paint(True, rgb(0, 255, 255))
    if low == "magenta" or low == "fuchsia": return Paint(True, rgb(255, 0, 255))
    # unknown name -> treat as current (so icons still render)
    return Paint(True, current)


def _nums_in_parens(s: String) raises -> List[Float64]:
    var out = List[Float64]()
    var b = s.as_bytes()
    var n = len(b)
    var i = 0
    # skip to '('
    while i < n and Int(b[i]) != 40:
        i += 1
    i += 1
    while i < n and Int(b[i]) != 41:
        # skip separators
        var c = Int(b[i])
        if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D or c == 0x2C:
            i += 1
            continue
        var start = i
        if Int(b[i]) == 43 or Int(b[i]) == 45:
            i += 1
        while i < n and ((Int(b[i]) >= 48 and Int(b[i]) <= 57) or Int(b[i]) == 46 or Int(b[i]) == 101 or Int(b[i]) == 69 or Int(b[i]) == 43 or Int(b[i]) == 45):
            # crude: stop at separators/paren handled by outer; allow digit . e + -
            if (Int(b[i]) == 43 or Int(b[i]) == 45) and i != start and not (Int(b[i - 1]) == 101 or Int(b[i - 1]) == 69):
                break
            i += 1
        if i > start:
            out.append(atof(String(StringSlice(ptr=b.unsafe_ptr() + start, length=i - start))))
        else:
            i += 1
    return out^


def parse_opacity(raw: String) raises -> Float64:
    var s = _trim(raw)
    if len(s) == 0:
        return 1.0
    var v = atof(s)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def parse_viewbox(raw: String) raises -> List[Float64]:
    """Returns [minx, miny, w, h]; empty list if unparseable."""
    var out = List[Float64]()
    var s = _trim(raw)
    var b = s.as_bytes()
    var n = len(b)
    var i = 0
    while i < n and len(out) < 4:
        var c = Int(b[i])
        if c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D or c == 0x2C:
            i += 1
            continue
        var start = i
        if Int(b[i]) == 43 or Int(b[i]) == 45:
            i += 1
        while i < n and ((Int(b[i]) >= 48 and Int(b[i]) <= 57) or Int(b[i]) == 46 or Int(b[i]) == 101 or Int(b[i]) == 69):
            i += 1
        if i > start:
            out.append(atof(String(StringSlice(ptr=b.unsafe_ptr() + start, length=i - start))))
        else:
            i += 1
    return out^


# ---- transform list ----
def parse_transform(raw: String) raises -> Affine:
    var acc = identity()
    var b = raw.as_bytes()
    var n = len(b)
    var i = 0
    while i < n:
        # read a function name
        var c = Int(b[i])
        if not ((c >= 65 and c <= 90) or (c >= 97 and c <= 122)):
            i += 1
            continue
        var nstart = i
        while i < n and ((Int(b[i]) >= 65 and Int(b[i]) <= 90) or (Int(b[i]) >= 97 and Int(b[i]) <= 122)):
            i += 1
        var fname = _lower(String(StringSlice(ptr=b.unsafe_ptr() + nstart, length=i - nstart)))
        # gather args within parens
        var sub = String("")
        var j = i
        while j < n and Int(b[j]) != 40:
            j += 1
        var pstart = j
        while j < n and Int(b[j]) != 41:
            j += 1
        # build "(...)" string for _nums_in_parens
        if j <= n:
            sub = String(StringSlice(ptr=b.unsafe_ptr() + pstart, length=(j - pstart) + 1)) if j < n else String("()")
        var args = _nums_in_parens(sub)
        i = j + 1

        var tok = identity()
        if fname == "translate":
            var tx = args[0] if len(args) >= 1 else 0.0
            var ty = args[1] if len(args) >= 2 else 0.0
            tok = translate(tx, ty)
        elif fname == "scale":
            var sx = args[0] if len(args) >= 1 else 1.0
            var sy = args[1] if len(args) >= 2 else sx
            tok = scale(sx, sy)
        elif fname == "rotate":
            var ang = (args[0] if len(args) >= 1 else 0.0) * pi / 180.0
            if len(args) >= 3:
                tok = translate(args[1], args[2]).then(rotate(ang)).then(translate(-args[1], -args[2]))
            else:
                tok = rotate(ang)
        elif fname == "matrix":
            if len(args) >= 6:
                tok = Affine(args[0], args[1], args[2], args[3], args[4], args[5])
        # left-to-right list T1 T2 .. => T1∘T2∘.. : acc = tok.then(acc)
        acc = tok.then(acc)
    return acc
