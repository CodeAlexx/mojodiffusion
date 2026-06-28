# tests/oft_carrier_parity.mojo — OneTrainer-OFT full-delta (a,b)-carrier gate.
# The carrier (a=I, b=W_eff-W) reproduces oft_ot_forward (base x@Wᵀ + carrier),
# and the chained carrier grad reproduces oft_ot_backward's d_vec. Small dims
# (r_eff=in → VRAM-bound at scale; this proves the mechanism).
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/training/tests/oft_carrier_parity.mojo -o /tmp/oft_carrier_parity \
#   && /tmp/oft_carrier_parity
from std.collections import List
from std.math import sqrt
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.oft_onetrainer import oft_ot_forward, oft_ot_backward
from serenitymojo.training.oft_stack import oft_ot_carrier_adapter, oft_ot_chain_carrier_grads

comptime COS_BAR = 0.999
comptime NREL_BAR = 3.0e-2


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _matmul(a: List[Float32], ra: Int, ca: Int, b: List[Float32], rb: Int, cb: Int) raises -> List[Float32]:
    if ca != rb:
        raise Error("gate _matmul: dim mismatch")
    var out = _zeros(ra * cb)
    for i in range(ra):
        for k in range(ca):
            var aik = a[i * ca + k]
            if aik == Float32(0.0):
                continue
            for j in range(cb):
                out[i * cb + j] = out[i * cb + j] + aik * b[k * cb + j]
    return out^


def _transpose(a: List[Float32], r: Int, c: Int) -> List[Float32]:
    var out = _zeros(r * c)
    for i in range(r):
        for j in range(c):
            out[j * r + i] = a[i * c + j]
    return out^


def _bf16_to_f32(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: len mismatch")
    var dot = 0.0; var na = 0.0; var nb = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("cos: zero vector")
    return dot / (sqrt(na) * sqrt(nb))


def _nrel(a: List[Float32], b: List[Float32]) raises -> Float64:
    var d = 0.0; var n = 0.0
    for i in range(len(a)):
        var dd = Float64(a[i]) - Float64(b[i])
        d += dd * dd
        n += Float64(b[i]) * Float64(b[i])
    if n == 0.0:
        return sqrt(d)
    return sqrt(d / n)


def _check(name: String, ok: Bool) raises:
    if ok:
        print("  PASS:", name)
    else:
        raise Error(String("GATE FAIL: ") + name)


struct StackSim(Movable):
    var y: List[Float32]
    var d_a: List[Float32]
    var d_b: List[Float32]

    def __init__(out self, var y: List[Float32], var d_a: List[Float32], var d_b: List[Float32]):
        self.y = y^
        self.d_a = d_a^
        self.d_b = d_b^


def _stack_sim(car: LoraAdapter, x: List[Float32], d_y: List[Float32], M: Int) raises -> StackSim:
    var R = car.rank
    var IN = car.in_f
    var OUT = car.out_f
    var a = _bf16_to_f32(car.a)
    var b = _bf16_to_f32(car.b)
    var a_t = _transpose(a, R, IN)
    var xa = _matmul(x, M, IN, a_t, IN, R)
    var b_t = _transpose(b, OUT, R)
    var y = _matmul(xa, M, R, b_t, R, OUT)
    var dy_t = _transpose(d_y, M, OUT)
    var d_b = _matmul(dy_t, OUT, M, xa, M, R)
    var dyb = _matmul(d_y, M, OUT, b, OUT, R)
    var dyb_t = _transpose(dyb, M, R)
    var d_a = _matmul(dyb_t, R, M, x, M, IN)
    return StackSim(y^, d_a^, d_b^)


def main() raises:
    var IN = 12; var OUT = 16; var M = 5; var b = 4; var r = IN // b
    var ne = b * (b - 1) // 2
    print("[oft carrier] IN=", IN, " OUT=", OUT, " b=", b, " r=", r, " r_eff(carrier)=", IN, " (full delta)")

    var vec = _randn(r * ne, 11, 0.5)
    var W = _randn(OUT * IN, 22, 0.5)
    var x = _randn(M * IN, 100, 1.0)
    var d_y = _randn(M * OUT, 200, 0.5)

    # ── 1. base x@Wᵀ + carrier reproduces oft_ot_forward ──
    var car = oft_ot_carrier_adapter(vec.copy(), W.copy(), IN, OUT, b, r)
    var sim = _stack_sim(car, x.copy(), d_y.copy(), M)
    var w_t = _transpose(W, OUT, IN)
    var y_base = _matmul(x, M, IN, w_t, IN, OUT)
    var y_total = y_base.copy()
    for i in range(len(y_total)):
        y_total[i] = y_total[i] + sim.y[i]
    var y_ref = oft_ot_forward(x.copy(), vec.copy(), W.copy(), M, IN, OUT, b, r)
    var cy = _cos(y_total, y_ref); var ny = _nrel(y_total, y_ref)
    print("  forward(base+carrier) cos=", cy, " nrel=", ny)
    _check("oft carrier forward", cy >= COS_BAR and ny <= NREL_BAR)

    # ── 2. chained carrier grad matches oft_ot_backward.d_vec ──
    var d_vec_c = oft_ot_chain_carrier_grads(vec.copy(), W.copy(), sim.d_a, sim.d_b, IN, OUT, b, r)
    var g_ref = oft_ot_backward(d_y.copy(), x.copy(), vec.copy(), W.copy(), M, IN, OUT, b, r)
    var cv = _cos(d_vec_c, g_ref.d_vec); var nv = _nrel(d_vec_c, g_ref.d_vec)
    print("  d_vec cos=", cv, " nrel=", nv)
    _check("oft d_vec", cv >= COS_BAR and nv <= NREL_BAR)

    print("ALL GATES PASS — oft_carrier_parity (full-delta carrier reproduces OFT fwd + d_vec)")
