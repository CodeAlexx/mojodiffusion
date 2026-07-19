# json.parser — a recursive-descent JSON parser (RFC 8259). 100% Mojo, no FFI.
#
# Parses bytes into a JSONValue tree. Handles objects, arrays, strings (with all
# escapes incl. \uXXXX and surrogate pairs → UTF-8), numbers (int vs float),
# true/false/null, and insignificant whitespace. Malformed input raises an Error
# carrying the byte offset.

from std.memory import UnsafePointer, alloc
from std.builtin.type_aliases import MutExternalOrigin
from json.value import JSONValue

comptime BytePtr = UnsafePointer[UInt8, MutExternalOrigin]

# byte constants
comptime SP = 0x20
comptime TAB = 0x09
comptime LF = 0x0A
comptime CR = 0x0D
comptime QUOTE = 0x22  # "
comptime BACKSLASH = 0x5C  # \
comptime LBRACE = 0x7B  # {
comptime RBRACE = 0x7D  # }
comptime LBRACK = 0x5B  # [
comptime RBRACK = 0x5D  # ]
comptime COLON = 0x3A  # :
comptime COMMA = 0x2C  # ,


def _hexval(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87  # a-f
    if c >= 65 and c <= 70:
        return c - 55  # A-F
    return -1


comptime MAX_DEPTH = 512  # nesting limit — guards against stack-overflow DoS


struct Parser:
    var src: String  # keeps the input buffer alive
    var p: BytePtr
    var n: Int
    var pos: Int
    var depth: Int

    def __init__(out self, var src: String):
        self.src = src^
        self.n = self.src.byte_length()
        self.p = BytePtr(unsafe_from_address=Int(self.src.unsafe_ptr()))
        self.pos = 0
        self.depth = 0

    def _peek(self) -> Int:
        if self.pos >= self.n:
            return -1
        return Int(self.p[self.pos])

    def _skip_ws(mut self):
        while self.pos < self.n:
            var c = Int(self.p[self.pos])
            if c == SP or c == TAB or c == LF or c == CR:
                self.pos += 1
            else:
                break

    def parse(mut self) raises -> JSONValue:
        """Parse a whole document; trailing non-whitespace is an error."""
        self._skip_ws()
        var v = self._value()
        self._skip_ws()
        if self.pos != self.n:
            raise Error("trailing data at byte " + String(self.pos))
        return v^

    def _value(mut self) raises -> JSONValue:
        self._skip_ws()
        var c = self._peek()
        if c == LBRACE:
            return self._object()
        if c == LBRACK:
            return self._array()
        if c == QUOTE:
            return JSONValue.from_string(self._string())
        if c == 0x74:  # 't' true
            return self._literal("true", JSONValue.from_bool(True))
        if c == 0x66:  # 'f' false
            return self._literal("false", JSONValue.from_bool(False))
        if c == 0x6E:  # 'n' null
            return self._literal("null", JSONValue.null())
        if c == 0x2D or (c >= 48 and c <= 57):  # '-' or digit
            return self._number()
        raise Error("unexpected byte at " + String(self.pos))

    def _literal(mut self, word: String, var v: JSONValue) raises -> JSONValue:
        var wb = word.as_bytes()
        var wn = word.byte_length()
        if self.pos + wn > self.n:
            raise Error("unexpected end parsing literal at " + String(self.pos))
        for i in range(wn):
            if Int(self.p[self.pos + i]) != Int(wb[i]):
                raise Error("invalid literal at " + String(self.pos))
        self.pos += wn
        return v^

    def _string(mut self) raises -> String:
        # assumes current byte is the opening quote
        self.pos += 1
        var buf = List[UInt8]()
        while True:
            if self.pos >= self.n:
                raise Error("unterminated string")
            var c = Int(self.p[self.pos])
            if c == QUOTE:
                self.pos += 1
                break
            if c == BACKSLASH:
                self.pos += 1
                if self.pos >= self.n:
                    raise Error("unterminated escape")
                var e = Int(self.p[self.pos])
                if e == QUOTE:
                    buf.append(0x22)
                elif e == BACKSLASH:
                    buf.append(0x5C)
                elif e == 0x2F:  # /
                    buf.append(0x2F)
                elif e == 0x62:  # b
                    buf.append(0x08)
                elif e == 0x66:  # f
                    buf.append(0x0C)
                elif e == 0x6E:  # n
                    buf.append(0x0A)
                elif e == 0x72:  # r
                    buf.append(0x0D)
                elif e == 0x74:  # t
                    buf.append(0x09)
                elif e == 0x75:  # u XXXX
                    self.pos += 1
                    var cp = self._hex4()
                    if cp >= 0xD800 and cp <= 0xDBFF:
                        # high surrogate; expect \uXXXX low surrogate
                        if self.pos + 1 < self.n and Int(self.p[self.pos]) == BACKSLASH and Int(self.p[self.pos + 1]) == 0x75:
                            self.pos += 2
                            var lo = self._hex4()
                            if lo < 0xDC00 or lo > 0xDFFF:
                                raise Error("invalid low surrogate")
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                        else:
                            raise Error("missing low surrogate")
                    self._encode_utf8(buf, cp)
                    continue  # _hex4 already advanced pos
                else:
                    raise Error("invalid escape at " + String(self.pos))
                self.pos += 1
            else:
                buf.append(UInt8(c))
                self.pos += 1
        return _bytes_to_string(buf)

    def _hex4(mut self) raises -> Int:
        if self.pos + 4 > self.n:
            raise Error("truncated \\u escape")
        var v = 0
        for _ in range(4):
            var h = _hexval(Int(self.p[self.pos]))
            if h < 0:
                raise Error("bad hex in \\u escape at " + String(self.pos))
            v = (v << 4) | h
            self.pos += 1
        return v

    def _encode_utf8(self, mut buf: List[UInt8], cp: Int):
        if cp < 0x80:
            buf.append(UInt8(cp))
        elif cp < 0x800:
            buf.append(UInt8(0xC0 | (cp >> 6)))
            buf.append(UInt8(0x80 | (cp & 0x3F)))
        elif cp < 0x10000:
            buf.append(UInt8(0xE0 | (cp >> 12)))
            buf.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            buf.append(UInt8(0x80 | (cp & 0x3F)))
        else:
            buf.append(UInt8(0xF0 | (cp >> 18)))
            buf.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
            buf.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
            buf.append(UInt8(0x80 | (cp & 0x3F)))

    def _number(mut self) raises -> JSONValue:
        var start = self.pos
        var is_float = False
        if self._peek() == 0x2D:  # -
            self.pos += 1
        while self.pos < self.n:
            var c = Int(self.p[self.pos])
            if c >= 48 and c <= 57:
                self.pos += 1
            elif c == 0x2E or c == 0x65 or c == 0x45:  # . e E
                is_float = True
                self.pos += 1
            elif c == 0x2B or c == 0x2D:  # + - (in exponent)
                self.pos += 1
            else:
                break
        var text = String(StringSlice(ptr=self.p + start, length=self.pos - start))
        if is_float:
            return JSONValue.from_float(Float64(text))
        # integer
        var neg = False
        var i = 0
        var tb = text.as_bytes()
        var tn = text.byte_length()
        if tn > 0 and Int(tb[0]) == 0x2D:
            neg = True
            i = 1
        var v = 0
        for j in range(i, tn):
            v = v * 10 + (Int(tb[j]) - 48)
        return JSONValue.from_int(-v if neg else v)

    def _array(mut self) raises -> JSONValue:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise Error("max nesting depth exceeded")
        self.pos += 1  # [
        var arr = JSONValue.new_array()
        self._skip_ws()
        if self._peek() == RBRACK:
            self.pos += 1
            self.depth -= 1
            return arr^
        while True:
            arr.append(self._value())
            self._skip_ws()
            var c = self._peek()
            if c == COMMA:
                self.pos += 1
            elif c == RBRACK:
                self.pos += 1
                break
            else:
                raise Error("expected ',' or ']' at " + String(self.pos))
        self.depth -= 1
        return arr^

    def _object(mut self) raises -> JSONValue:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise Error("max nesting depth exceeded")
        self.pos += 1  # {
        var obj = JSONValue.new_object()
        self._skip_ws()
        if self._peek() == RBRACE:
            self.pos += 1
            self.depth -= 1
            return obj^
        while True:
            self._skip_ws()
            if self._peek() != QUOTE:
                raise Error("expected string key at " + String(self.pos))
            var key = self._string()
            self._skip_ws()
            if self._peek() != COLON:
                raise Error("expected ':' at " + String(self.pos))
            self.pos += 1
            obj.set(key, self._value())
            self._skip_ws()
            var c = self._peek()
            if c == COMMA:
                self.pos += 1
            elif c == RBRACE:
                self.pos += 1
                break
            else:
                raise Error("expected ',' or '}' at " + String(self.pos))
        self.depth -= 1
        return obj^


def _bytes_to_string(b: List[UInt8]) -> String:
    var n = len(b)
    if n == 0:
        return String("")
    var buf = alloc[UInt8](n)
    for i in range(n):
        buf[i] = b[i]
    var s = String(StringSlice(ptr=BytePtr(unsafe_from_address=Int(buf)), length=n))
    buf.free()
    return s


def loads(text: String) raises -> JSONValue:
    """Parse a JSON document from a String."""
    var p = Parser(text)
    return p.parse()
