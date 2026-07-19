# pdf/ec.mojo
# Pure-Mojo NIST P-256 (prime256v1 / secp256r1) elliptic curve arithmetic.
#
# Builds on pdf/bigint.mojo (unsigned big integers). Field is GF(p) with the
# P-256 prime p; the group order is n. Both p and n are PRIME, so modular
# inverses use Fermat's little theorem: a^-1 = a^(m-2) mod m (via BigInt.modexp).
#
# Points are stored in AFFINE form (x, y, inf). Internally, scalar
# multiplication and point add/double run in JACOBIAN coordinates
# (X, Y, Z) with x = X/Z^2, y = Y/Z^3, doing one final modular inverse to
# convert back to affine. This avoids a modular inverse per group operation.
#
# Curve: y^2 = x^3 + a*x + b  (mod p),  a = p - 3.

from pdf.bigint import BigInt


# ----------------------------------------------------------------------
# P-256 domain parameters (hex constants from the prompt / SEC2 / FIPS 186-4)
# ----------------------------------------------------------------------
comptime P256_P_HEX = "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff"
comptime P256_B_HEX = "5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b"
comptime P256_GX_HEX = "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"
comptime P256_GY_HEX = "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5"
comptime P256_N_HEX = "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551"


def p256_p() raises -> BigInt:
    return BigInt.from_hex(P256_P_HEX)


def p256_n() raises -> BigInt:
    return BigInt.from_hex(P256_N_HEX)


def p256_b() raises -> BigInt:
    return BigInt.from_hex(P256_B_HEX)


def p256_gx() raises -> BigInt:
    return BigInt.from_hex(P256_GX_HEX)


def p256_gy() raises -> BigInt:
    return BigInt.from_hex(P256_GY_HEX)


# ----------------------------------------------------------------------
# Field arithmetic mod p
# ----------------------------------------------------------------------
def _fadd(a: BigInt, b: BigInt, p: BigInt) raises -> BigInt:
    var s = a.add(b)
    if s.compare(p) >= 0:
        return s.sub(p)
    return s^


def _fsub(a: BigInt, b: BigInt, p: BigInt) raises -> BigInt:
    # (a - b) mod p, both already reduced in [0, p).
    if a.compare(b) >= 0:
        return a.sub(b)
    # a < b : a - b + p = p - (b - a)
    return p.sub(b.sub(a))


def _fmul(a: BigInt, b: BigInt, p: BigInt) raises -> BigInt:
    return a.mul(b).mod(p)


def _fsqr(a: BigInt, p: BigInt) raises -> BigInt:
    return a.mul(a).mod(p)


def _finv(a: BigInt, p: BigInt) raises -> BigInt:
    # a^-1 mod p via Fermat (p prime): a^(p-2) mod p.
    var two = BigInt(UInt64(2))
    var pm2 = p.sub(two)
    return a.modexp(pm2, p)


def field_inverse(a: BigInt, p: BigInt) raises -> BigInt:
    # Public wrapper used by sign_ec for k^-1 mod n (n is also prime).
    return _finv(a, p)


# ----------------------------------------------------------------------
# Affine point
# ----------------------------------------------------------------------
struct Point(Copyable, Movable):
    var x: BigInt
    var y: BigInt
    var inf: Bool

    def __init__(out self, var x: BigInt, var y: BigInt, inf: Bool):
        self.x = x^
        self.y = y^
        self.inf = inf

    def __init__(out self, *, copy: Self):
        self.x = BigInt(copy=copy.x)
        self.y = BigInt(copy=copy.y)
        self.inf = copy.inf

    @staticmethod
    def infinity() raises -> Point:
        return Point(BigInt(), BigInt(), True)

    def is_inf(self) -> Bool:
        return self.inf

    def equals(self, other: Point) raises -> Bool:
        if self.inf or other.inf:
            return self.inf == other.inf
        return self.x.compare(other.x) == 0 and self.y.compare(other.y) == 0


# ----------------------------------------------------------------------
# Jacobian point (X, Y, Z); affine (X/Z^2, Y/Z^3). Z==0 => point at infinity.
# ----------------------------------------------------------------------
struct Jacobian(Copyable, Movable):
    var X: BigInt
    var Y: BigInt
    var Z: BigInt

    def __init__(out self, var X: BigInt, var Y: BigInt, var Z: BigInt):
        self.X = X^
        self.Y = Y^
        self.Z = Z^

    def __init__(out self, *, copy: Self):
        self.X = BigInt(copy=copy.X)
        self.Y = BigInt(copy=copy.Y)
        self.Z = BigInt(copy=copy.Z)

    def is_inf(self) -> Bool:
        return self.Z.is_zero()


def _jac_infinity() raises -> Jacobian:
    # Canonical infinity: Z = 0.
    return Jacobian(BigInt(UInt64(1)), BigInt(UInt64(1)), BigInt())


def _to_jacobian(P: Point) raises -> Jacobian:
    if P.inf:
        return _jac_infinity()
    return Jacobian(BigInt(copy=P.x), BigInt(copy=P.y), BigInt(UInt64(1)))


def _jac_double(J: Jacobian, p: BigInt) raises -> Jacobian:
    # Point doubling in Jacobian coords for a = -3 curves.
    #   delta = Z^2
    #   gamma = Y^2
    #   beta  = X * gamma
    #   alpha = 3 * (X - delta) * (X + delta)        (uses a = -3 shortcut)
    #   X3 = alpha^2 - 8*beta
    #   Z3 = (Y + Z)^2 - gamma - delta
    #   Y3 = alpha * (4*beta - X3) - 8*gamma^2
    if J.Z.is_zero():
        return _jac_infinity()
    if J.Y.is_zero():
        return _jac_infinity()

    var delta = _fsqr(J.Z, p)
    var gamma = _fsqr(J.Y, p)
    var beta = _fmul(J.X, gamma, p)

    var xmd = _fsub(J.X, delta, p)
    var xpd = _fadd(J.X, delta, p)
    var t = _fmul(xmd, xpd, p)               # (X-delta)*(X+delta)
    var alpha = _fadd(_fadd(t, t, p), t, p)  # 3 * t

    var alpha2 = _fsqr(alpha, p)
    var beta4 = _fadd(_fadd(beta, beta, p), _fadd(beta, beta, p), p)  # 4*beta
    var beta8 = _fadd(beta4, beta4, p)                               # 8*beta
    var X3 = _fsub(alpha2, beta8, p)

    var ypz = _fadd(J.Y, J.Z, p)
    var Z3 = _fsub(_fsub(_fsqr(ypz, p), gamma, p), delta, p)

    var gamma2 = _fsqr(gamma, p)
    var gamma8 = _fadd(_fadd(_fadd(gamma2, gamma2, p), _fadd(gamma2, gamma2, p), p),
                       _fadd(_fadd(gamma2, gamma2, p), _fadd(gamma2, gamma2, p), p), p)  # 8*gamma^2
    var b4mx = _fsub(beta4, X3, p)
    var Y3 = _fsub(_fmul(alpha, b4mx, p), gamma8, p)

    return Jacobian(X3^, Y3^, Z3^)


def _jac_add(J1: Jacobian, J2: Jacobian, p: BigInt) raises -> Jacobian:
    # Add two Jacobian points (general add; handles equal/inverse via doubling).
    if J1.Z.is_zero():
        return Jacobian(copy=J2)
    if J2.Z.is_zero():
        return Jacobian(copy=J1)

    var Z1Z1 = _fsqr(J1.Z, p)
    var Z2Z2 = _fsqr(J2.Z, p)
    var U1 = _fmul(J1.X, Z2Z2, p)             # X1 * Z2^2
    var U2 = _fmul(J2.X, Z1Z1, p)             # X2 * Z1^2
    var S1 = _fmul(_fmul(J1.Y, J2.Z, p), Z2Z2, p)  # Y1 * Z2^3
    var S2 = _fmul(_fmul(J2.Y, J1.Z, p), Z1Z1, p)  # Y2 * Z1^3

    if U1.compare(U2) == 0:
        if S1.compare(S2) == 0:
            # Same point -> double.
            return _jac_double(J1, p)
        else:
            # P + (-P) = infinity.
            return _jac_infinity()

    var H = _fsub(U2, U1, p)
    var R = _fsub(S2, S1, p)
    var HH = _fsqr(H, p)
    var HHH = _fmul(H, HH, p)
    var V = _fmul(U1, HH, p)

    # X3 = R^2 - HHH - 2*V
    var R2 = _fsqr(R, p)
    var X3 = _fsub(_fsub(R2, HHH, p), _fadd(V, V, p), p)
    # Y3 = R*(V - X3) - S1*HHH
    var Y3 = _fsub(_fmul(R, _fsub(V, X3, p), p), _fmul(S1, HHH, p), p)
    # Z3 = Z1*Z2*H
    var Z3 = _fmul(_fmul(J1.Z, J2.Z, p), H, p)

    return Jacobian(X3^, Y3^, Z3^)


def _to_affine(J: Jacobian, p: BigInt) raises -> Point:
    if J.Z.is_zero():
        return Point.infinity()
    var zinv = _finv(J.Z, p)
    var zinv2 = _fsqr(zinv, p)
    var zinv3 = _fmul(zinv2, zinv, p)
    var x = _fmul(J.X, zinv2, p)
    var y = _fmul(J.Y, zinv3, p)
    return Point(x^, y^, False)


# ----------------------------------------------------------------------
# Public group operations
# ----------------------------------------------------------------------
def point_add(P: Point, Q: Point) raises -> Point:
    var p = p256_p()
    var j = _jac_add(_to_jacobian(P), _to_jacobian(Q), p)
    return _to_affine(j, p)


def point_double(P: Point) raises -> Point:
    var p = p256_p()
    var j = _jac_double(_to_jacobian(P), p)
    return _to_affine(j, p)


def scalar_mul(k: BigInt, P: Point) raises -> Point:
    # Double-and-add over the bits of k (MSB first), Jacobian internally.
    var p = p256_p()
    if P.inf or k.is_zero():
        return Point.infinity()
    var acc = _jac_infinity()
    var base = _to_jacobian(P)
    var nbits = k.bit_length()
    var i = nbits - 1
    while i >= 0:
        acc = _jac_double(acc, p)
        if k.test_bit(i):
            acc = _jac_add(acc, base, p)
        i -= 1
    return _to_affine(acc, p)


def generator() raises -> Point:
    return Point(p256_gx(), p256_gy(), False)


def point_negate(P: Point) raises -> Point:
    # -(x, y) = (x, p - y).
    if P.inf:
        return Point.infinity()
    var p = p256_p()
    var ny = p.sub(P.y)
    return Point(BigInt(copy=P.x), ny^, False)
