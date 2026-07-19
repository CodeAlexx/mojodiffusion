# graphics.transform — 2x3 affine matrix for path transforms.
#
# A point (x, y) maps to (a*x + c*y + e, b*x + d*y + f). Compose with `then`:
# `A.then(B)` applies A first, then B. Builders: identity / translate / scale /
# rotate (radians).

from std.math import sin, cos


@fieldwise_init
struct Affine(Copyable, ImplicitlyCopyable, Movable):
    var a: Float64
    var b: Float64
    var c: Float64
    var d: Float64
    var e: Float64
    var f: Float64

    def tx(self, x: Float64, y: Float64) -> Float64:
        return self.a * x + self.c * y + self.e

    def ty(self, x: Float64, y: Float64) -> Float64:
        return self.b * x + self.d * y + self.f

    def then(self, m: Affine) -> Affine:
        """Result maps p -> m(self(p)) — apply self first, then m."""
        return Affine(
            m.a * self.a + m.c * self.b,
            m.b * self.a + m.d * self.b,
            m.a * self.c + m.c * self.d,
            m.b * self.c + m.d * self.d,
            m.a * self.e + m.c * self.f + m.e,
            m.b * self.e + m.d * self.f + m.f,
        )


def identity() -> Affine:
    return Affine(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def translate(tx: Float64, ty: Float64) -> Affine:
    return Affine(1.0, 0.0, 0.0, 1.0, tx, ty)


def scale(sx: Float64, sy: Float64) -> Affine:
    return Affine(sx, 0.0, 0.0, sy, 0.0, 0.0)


def rotate(theta: Float64) -> Affine:
    var co = cos(theta)
    var si = sin(theta)
    return Affine(co, si, -si, co, 0.0, 0.0)
