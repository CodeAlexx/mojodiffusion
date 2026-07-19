# pdf/content.mojo
# Pure-Mojo PDF content-stream operator builder.
#
# PDF content coordinate system: origin BOTTOM-LEFT, y increases UP,
# default unit = 1/72 inch (a "point"). Colors are 0.0..1.0.
#
# All operands are written as compact PDF real numbers (no exponent
# notation — PDF forbids forms like 1e5). Operands are space-separated
# and every operator line ends with a newline.


struct Content(Movable):
    var b: List[UInt8]

    def __init__(out self):
        self.b = List[UInt8]()

    # ---- low-level byte helpers ----------------------------------------

    def _push_str(mut self, s: String):
        var bs = s.as_bytes()
        for i in range(len(bs)):
            self.b.append(bs[i])

    def _push_char(mut self, c: UInt8):
        self.b.append(c)

    # Compact PDF real formatter. Writes values like:
    #   3.14   100   -0.5   0
    # No exponent notation, no trailing-zero/dot cruft.
    def _push_real(mut self, value: Float64):
        var v = value

        # Handle non-finite defensively: emit 0.
        if v != v:  # NaN
            self._push_char(UInt8(ord("0")))
            return

        var neg = v < 0.0
        if neg:
            v = -v

        # Round to 6 decimal places (PDF reals are typically <= 5-6 digits).
        comptime SCALE = 1000000.0
        var scaled = v * SCALE + 0.5
        var total = Int(scaled)  # truncates toward zero on non-negative

        var int_part = total // 1000000
        var frac_part = total % 1000000  # 0 .. 999999

        # Sign (suppress "-0").
        if neg and (int_part != 0 or frac_part != 0):
            self._push_char(UInt8(ord("-")))

        # Integer part digits.
        if int_part == 0:
            self._push_char(UInt8(ord("0")))
        else:
            var digits = List[UInt8]()
            var n = int_part
            while n > 0:
                var d = n % 10
                digits.append(UInt8(ord("0") + d))
                n = n // 10
            # digits are reversed; emit back-to-front.
            var k = len(digits)
            while k > 0:
                k -= 1
                self._push_char(digits[k])

        # Fractional part: emit only if non-zero, stripping trailing zeros.
        if frac_part != 0:
            # Build the 6-digit fractional string with leading zeros.
            var fdigits = List[UInt8]()
            var f = frac_part
            # Produce exactly 6 digits (reversed first, then fix order).
            var tmp = List[UInt8]()
            var count = 0
            while count < 6:
                var d = f % 10
                tmp.append(UInt8(ord("0") + d))
                f = f // 10
                count += 1
            # tmp is least-significant-first; reverse into fdigits.
            var j = len(tmp)
            while j > 0:
                j -= 1
                fdigits.append(tmp[j])
            # Strip trailing zeros from fdigits (most-significant-first list).
            var end = len(fdigits)
            while end > 0 and fdigits[end - 1] == UInt8(ord("0")):
                end -= 1
            if end > 0:
                self._push_char(UInt8(ord(".")))
                for i in range(end):
                    self._push_char(fdigits[i])

    def _space(mut self):
        self._push_char(UInt8(ord(" ")))

    def _newline(mut self):
        self._push_char(UInt8(ord("\n")))

    # ---- path construction ---------------------------------------------

    def move_to(mut self, x: Float64, y: Float64):
        self._push_real(x)
        self._space()
        self._push_real(y)
        self._space()
        self._push_char(UInt8(ord("m")))
        self._newline()

    def line_to(mut self, x: Float64, y: Float64):
        self._push_real(x)
        self._space()
        self._push_real(y)
        self._space()
        self._push_char(UInt8(ord("l")))
        self._newline()

    def rect(mut self, x: Float64, y: Float64, w: Float64, h: Float64):
        self._push_real(x)
        self._space()
        self._push_real(y)
        self._space()
        self._push_real(w)
        self._space()
        self._push_real(h)
        self._space()
        self._push_str("re")
        self._newline()

    def close_path(mut self):
        self._push_char(UInt8(ord("h")))
        self._newline()

    # ---- painting -------------------------------------------------------

    def fill(mut self):
        self._push_char(UInt8(ord("f")))
        self._newline()

    def stroke(mut self):
        self._push_char(UInt8(ord("S")))
        self._newline()

    def fill_stroke(mut self):
        self._push_char(UInt8(ord("B")))
        self._newline()

    # ---- color / graphics state ----------------------------------------

    def set_fill_rgb(mut self, r: Float64, g: Float64, b: Float64):
        self._push_real(r)
        self._space()
        self._push_real(g)
        self._space()
        self._push_real(b)
        self._space()
        self._push_str("rg")
        self._newline()

    def set_stroke_rgb(mut self, r: Float64, g: Float64, b: Float64):
        self._push_real(r)
        self._space()
        self._push_real(g)
        self._space()
        self._push_real(b)
        self._space()
        self._push_str("RG")
        self._newline()

    def set_line_width(mut self, w: Float64):
        self._push_real(w)
        self._space()
        self._push_char(UInt8(ord("w")))
        self._newline()

    def save_state(mut self):
        self._push_char(UInt8(ord("q")))
        self._newline()

    def restore_state(mut self):
        self._push_char(UInt8(ord("Q")))
        self._newline()

    # ---- escape hatch ---------------------------------------------------

    # Append arbitrary operator text verbatim (for text/image teammates).
    # The caller is responsible for whitespace/newlines.
    def raw(mut self, s: String):
        self._push_str(s)

    # ---- output ---------------------------------------------------------

    def bytes(self) -> List[UInt8]:
        return self.b.copy()
