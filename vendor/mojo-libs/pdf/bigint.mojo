# pdf/bigint.mojo
# Pure-Mojo arbitrary-precision UNSIGNED integer, enough for RSA-2048.
#
# Representation: List[UInt32] limbs, LITTLE-ENDIAN (limb[0] = least significant).
# A "normalized" BigInt has no leading zero limbs except the value 0 which is
# represented by a single zero limb (or an empty list, treated as zero).
#
# All limb arithmetic carries through UInt64 intermediates and masks with
# 0xFFFFFFFF, exactly like the project's md5.mojo style.
#
# Provided: from/to big-endian bytes, from decimal/hex string, compare, add,
# sub, mul, divmod (schoolbook long division), mod, and modexp (square &
# multiply) which is the RSA hot path.


comptime LIMB_BITS = 32
comptime LIMB_MASK: UInt64 = 0xFFFFFFFF


struct DivMod(Copyable, Movable):
    # Result of long division: dividend = q * divisor + r, 0 <= r < divisor.
    var q: BigInt
    var r: BigInt

    def __init__(out self, var q: BigInt, var r: BigInt):
        self.q = q^
        self.r = r^


struct BigInt(Copyable, Movable):
    var limbs: List[UInt32]  # little-endian, normalized (no leading zeros)

    # ---- constructors ----

    def __init__(out self):
        # Zero.
        self.limbs = List[UInt32]()
        self.limbs.append(UInt32(0))

    def __init__(out self, var limbs: List[UInt32]):
        self.limbs = limbs^
        self._normalize()

    def __init__(out self, value: UInt64):
        self.limbs = List[UInt32]()
        self.limbs.append(UInt32(value & LIMB_MASK))
        self.limbs.append(UInt32((value >> 32) & LIMB_MASK))
        self._normalize()

    def __init__(out self, *, copy: Self):
        self.limbs = copy.limbs.copy()

    # ---- helpers ----

    def _normalize(mut self):
        # Drop leading zero limbs but keep at least one limb (zero).
        while len(self.limbs) > 1 and self.limbs[len(self.limbs) - 1] == UInt32(0):
            _ = self.limbs.pop()

    def is_zero(self) -> Bool:
        return len(self.limbs) == 1 and self.limbs[0] == UInt32(0)

    def num_limbs(self) -> Int:
        return len(self.limbs)

    def limb(self, i: Int) -> UInt32:
        # Out-of-range limbs are zero.
        if i < 0 or i >= len(self.limbs):
            return UInt32(0)
        return self.limbs[i]

    def bit_length(self) -> Int:
        # Number of significant bits (0 for the value zero).
        var n = len(self.limbs)
        var top = self.limbs[n - 1]
        if n == 1 and top == UInt32(0):
            return 0
        var bits = (n - 1) * LIMB_BITS
        var t = top
        while t != UInt32(0):
            bits += 1
            t = t >> 1
        return bits

    def test_bit(self, i: Int) -> Bool:
        var limb_idx = i // LIMB_BITS
        var bit_idx = i % LIMB_BITS
        if limb_idx >= len(self.limbs):
            return False
        return ((self.limbs[limb_idx] >> bit_idx) & UInt32(1)) == UInt32(1)

    # ---- comparison ----  returns -1, 0, +1
    def compare(self, other: BigInt) -> Int:
        var na = len(self.limbs)
        var nb = len(other.limbs)
        if na != nb:
            if na < nb:
                return -1
            return 1
        var i = na - 1
        while i >= 0:
            var a = self.limbs[i]
            var b = other.limbs[i]
            if a != b:
                if a < b:
                    return -1
                return 1
            i -= 1
        return 0

    # ---- add ----
    def add(self, other: BigInt) raises -> BigInt:
        var out_limbs = List[UInt32]()
        var n = len(self.limbs)
        var m = len(other.limbs)
        var maxn = n
        if m > maxn:
            maxn = m
        var carry = UInt64(0)
        for i in range(maxn):
            var s = UInt64(self.limb(i)) + UInt64(other.limb(i)) + carry
            out_limbs.append(UInt32(s & LIMB_MASK))
            carry = s >> 32
        if carry != UInt64(0):
            out_limbs.append(UInt32(carry & LIMB_MASK))
        return BigInt(out_limbs^)

    # ---- sub ----  (requires self >= other; unsigned)
    def sub(self, other: BigInt) raises -> BigInt:
        # Assumes self >= other.
        var out_limbs = List[UInt32]()
        var n = len(self.limbs)
        var borrow = Int64(0)
        for i in range(n):
            var diff = Int64(Int(self.limb(i))) - Int64(Int(other.limb(i))) - borrow
            if diff < Int64(0):
                diff += Int64(1) << Int64(32)
                borrow = Int64(1)
            else:
                borrow = Int64(0)
            out_limbs.append(UInt32(UInt64(diff) & LIMB_MASK))
        return BigInt(out_limbs^)

    # ---- mul ----  schoolbook O(n*m)
    def mul(self, other: BigInt) raises -> BigInt:
        var n = len(self.limbs)
        var m = len(other.limbs)
        var out_limbs = List[UInt32]()
        for _ in range(n + m):
            out_limbs.append(UInt32(0))
        for i in range(n):
            var carry = UInt64(0)
            var ai = UInt64(self.limbs[i])
            for j in range(m):
                var idx = i + j
                var cur = UInt64(out_limbs[idx])
                var prod = ai * UInt64(other.limbs[j]) + cur + carry
                out_limbs[idx] = UInt32(prod & LIMB_MASK)
                carry = prod >> 32
            # propagate remaining carry
            var k = i + m
            while carry != UInt64(0):
                var cur2 = UInt64(out_limbs[k]) + carry
                out_limbs[k] = UInt32(cur2 & LIMB_MASK)
                carry = cur2 >> 32
                k += 1
        return BigInt(out_limbs^)

    # ---- shift left by whole bits ----
    def shl_bits(self, shift: Int) raises -> BigInt:
        if self.is_zero() or shift == 0:
            return BigInt(copy=self)
        var limb_shift = shift // LIMB_BITS
        var bit_shift = shift % LIMB_BITS
        var out_limbs = List[UInt32]()
        for _ in range(limb_shift):
            out_limbs.append(UInt32(0))
        if bit_shift == 0:
            for i in range(len(self.limbs)):
                out_limbs.append(self.limbs[i])
        else:
            var carry = UInt32(0)
            for i in range(len(self.limbs)):
                var v = self.limbs[i]
                var shifted = (UInt64(v) << bit_shift) | UInt64(carry)
                out_limbs.append(UInt32(shifted & LIMB_MASK))
                carry = UInt32((UInt64(v) >> (LIMB_BITS - bit_shift)) & LIMB_MASK)
            if carry != UInt32(0):
                out_limbs.append(carry)
        return BigInt(out_limbs^)

    # ---- divmod ----  schoolbook bit-by-bit long division.
    # Slow but correct; modexp keeps operand sizes near the modulus so this is
    # acceptable for RSA-2048 (each step divides a <=4096-bit value by n).
    def divmod(self, divisor: BigInt) raises -> DivMod:
        if divisor.is_zero():
            raise Error("BigInt.divmod: division by zero")
        if self.compare(divisor) < 0:
            return DivMod(BigInt(), BigInt(copy=self))

        var n_bits = self.bit_length()
        var q = BigInt()
        var r = BigInt()
        # quotient limbs (initialized to zero, sized to dividend)
        var q_limbs = List[UInt32]()
        for _ in range(len(self.limbs)):
            q_limbs.append(UInt32(0))

        var i = n_bits - 1
        while i >= 0:
            # r = (r << 1) | bit_i(self)
            r = r.shl_bits(1)
            if self.test_bit(i):
                # set r's bit 0
                if len(r.limbs) == 0:
                    r.limbs.append(UInt32(0))
                r.limbs[0] = r.limbs[0] | UInt32(1)
            if r.compare(divisor) >= 0:
                r = r.sub(divisor)
                # set quotient bit i
                var lidx = i // LIMB_BITS
                var bidx = i % LIMB_BITS
                q_limbs[lidx] = q_limbs[lidx] | (UInt32(1) << bidx)
            i -= 1

        q = BigInt(q_limbs^)
        return DivMod(q^, r^)

    def mod(self, modulus: BigInt) raises -> BigInt:
        var dm = self.divmod(modulus)
        return BigInt(copy=dm.r)

    # ---- modular exponentiation: self^exp mod modulus ----
    # Square-and-multiply, left-to-right over exponent bits.
    def modexp(self, exp: BigInt, modulus: BigInt) raises -> BigInt:
        if modulus.is_zero():
            raise Error("BigInt.modexp: modulus is zero")
        # result = 1 mod modulus
        var result = BigInt(UInt64(1)).mod(modulus)
        var base = self.mod(modulus)
        var nbits = exp.bit_length()
        var i = nbits - 1
        while i >= 0:
            # square
            result = result.mul(result).mod(modulus)
            if exp.test_bit(i):
                result = result.mul(base).mod(modulus)
            i -= 1
        return result^

    # ---- big-endian byte I/O ----
    @staticmethod
    def from_bytes_be(data: List[UInt8]) raises -> BigInt:
        # Interpret data as a big-endian unsigned integer.
        var out_limbs = List[UInt32]()
        var n = len(data)
        # process from least significant (end) toward most significant
        var idx = n - 1
        var cur = UInt32(0)
        var shift = 0
        while idx >= 0:
            cur = cur | (UInt32(data[idx]) << shift)
            shift += 8
            if shift == 32:
                out_limbs.append(cur)
                cur = UInt32(0)
                shift = 0
            idx -= 1
        if shift != 0:
            out_limbs.append(cur)
        if len(out_limbs) == 0:
            out_limbs.append(UInt32(0))
        return BigInt(out_limbs^)

    def to_bytes_be(self) raises -> List[UInt8]:
        # Minimal big-endian byte representation (no leading zero byte except
        # for the value 0 which yields a single 0x00).
        var tmp = List[UInt8]()
        var n = len(self.limbs)
        for i in range(n):
            var v = self.limbs[i]
            tmp.append(UInt8(v & 0xFF))
            tmp.append(UInt8((v >> 8) & 0xFF))
            tmp.append(UInt8((v >> 16) & 0xFF))
            tmp.append(UInt8((v >> 24) & 0xFF))
        # tmp is little-endian byte order; reverse and strip leading zeros.
        var rev = List[UInt8]()
        var j = len(tmp) - 1
        while j >= 0:
            rev.append(tmp[j])
            j -= 1
        # strip leading zeros
        var start = 0
        while start < len(rev) - 1 and rev[start] == UInt8(0):
            start += 1
        var result = List[UInt8]()
        for k in range(start, len(rev)):
            result.append(rev[k])
        return result^

    def to_bytes_be_padded(self, length: Int) raises -> List[UInt8]:
        # Fixed-length big-endian (left-pad with zeros). Raises if too big.
        var minimal = self.to_bytes_be()
        if len(minimal) > length:
            raise Error("BigInt.to_bytes_be_padded: value too large for length")
        var result = List[UInt8]()
        var pad = length - len(minimal)
        for _ in range(pad):
            result.append(UInt8(0))
        for i in range(len(minimal)):
            result.append(minimal[i])
        return result^

    # ---- string parsing ----
    @staticmethod
    def from_hex(s: String) raises -> BigInt:
        # Parse a hex string (optional 0x prefix, case-insensitive).
        var bytes = s.as_bytes()
        var start = 0
        var n = s.byte_length()
        # skip optional 0x / 0X
        if n >= 2 and bytes[0] == UInt8(0x30):
            if bytes[1] == UInt8(0x78) or bytes[1] == UInt8(0x58):
                start = 2
        # collect nibble values
        var nibbles = List[Int]()
        for i in range(start, n):
            var c = Int(bytes[i])
            var v: Int
            if c >= 0x30 and c <= 0x39:
                v = c - 0x30
            elif c >= 0x61 and c <= 0x66:
                v = c - 0x61 + 10
            elif c >= 0x41 and c <= 0x46:
                v = c - 0x41 + 10
            elif c == 0x20 or c == 0x0A or c == 0x0D or c == 0x09 or c == 0x3A:
                # skip whitespace and ':' separators
                continue
            else:
                raise Error("BigInt.from_hex: invalid hex digit")
            nibbles.append(v)
        if len(nibbles) == 0:
            return BigInt()
        # Build big-endian byte list from nibbles, then convert.
        var data = List[UInt8]()
        var nn = len(nibbles)
        # if odd number of nibbles, the first is the high nibble of byte 0
        var first = 0
        if (nn % 2) == 1:
            data.append(UInt8(nibbles[0]))
            first = 1
        var k = first
        while k < nn:
            var hi = nibbles[k]
            var lo = nibbles[k + 1]
            data.append(UInt8((hi << 4) | lo))
            k += 2
        return BigInt.from_bytes_be(data)

    @staticmethod
    def from_decimal(s: String) raises -> BigInt:
        var bytes = s.as_bytes()
        var n = s.byte_length()
        var acc = BigInt()
        var ten = BigInt(UInt64(10))
        for i in range(n):
            var c = Int(bytes[i])
            if c == 0x20 or c == 0x0A or c == 0x0D or c == 0x09:
                continue
            if c < 0x30 or c > 0x39:
                raise Error("BigInt.from_decimal: invalid digit")
            var d = BigInt(UInt64(c - 0x30))
            acc = acc.mul(ten).add(d)
        return acc^

    def to_hex(self) raises -> String:
        comptime HEX = "0123456789abcdef"
        var hexb = HEX.as_bytes()
        var data = self.to_bytes_be()
        var s = String("")
        for i in range(len(data)):
            var b = Int(data[i])
            s += chr(Int(hexb[(b >> 4) & 0xF]))
            s += chr(Int(hexb[b & 0xF]))
        return s^


# ---- signed value (magnitude + sign), module-level, for extended Euclid ----
struct SignedBI(Copyable, Movable):
    var neg: Bool
    var mag: BigInt

    def __init__(out self, neg: Bool, var mag: BigInt):
        # Normalize: zero is never negative.
        if mag.is_zero():
            self.neg = False
        else:
            self.neg = neg
        self.mag = mag^

    def __init__(out self, *, copy: Self):
        self.neg = copy.neg
        self.mag = BigInt(copy=copy.mag)


def _signed_sub(a: SignedBI, b: SignedBI) raises -> SignedBI:
    # Return a - b as a SignedBI.
    if a.neg == b.neg:
        var cmp = a.mag.compare(b.mag)
        if cmp >= 0:
            return SignedBI(a.neg, a.mag.sub(b.mag))
        else:
            return SignedBI(not a.neg, b.mag.sub(a.mag))
    else:
        # a - b with opposite signs => magnitudes add, sign follows a.
        return SignedBI(a.neg, a.mag.add(b.mag))


def bi_modinv(a: BigInt, m: BigInt) raises -> BigInt:
    # Modular inverse a^-1 mod m via the extended Euclidean algorithm.
    # Works for any modulus with gcd(a, m) == 1 (e.g. RSA d = e^-1 mod lambda).
    # t, newt tracked as signed; r, newr as non-negative BigInt.
    var t = SignedBI(False, BigInt())            # 0
    var newt = SignedBI(False, BigInt(UInt64(1)))  # 1
    var r = BigInt(copy=m)
    var newr = a.mod(m)
    while not newr.is_zero():
        var dm = r.divmod(newr)
        var q = dm.q.copy()
        # (t, newt) = (newt, t - q*newt)
        var qn = SignedBI(newt.neg, q.mul(newt.mag))
        var next_t = _signed_sub(t, qn)
        t = newt.copy()
        newt = next_t.copy()
        # (r, newr) = (newr, r - q*newr) = (newr, dm.r)
        r = newr.copy()
        newr = dm.r.copy()
    if r.compare(BigInt(UInt64(1))) != 0:
        raise Error("bi_modinv: not invertible (gcd != 1)")
    # t is the inverse; reduce into [0, m).
    if t.neg:
        # t + m  (t is -|t|, so m - (|t| mod m))
        var tmod = t.mag.mod(m)
        if tmod.is_zero():
            return BigInt()
        return m.sub(tmod)
    return t.mag.mod(m)
