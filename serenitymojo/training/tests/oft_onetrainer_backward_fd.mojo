# tests/oft_onetrainer_backward_fd.mojo — finite-difference gate for the
# OneTrainer-OFT analytic backward (oft_ot_backward). The forward is already
# verified vs OneTrainer (oft_onetrainer_parity.mojo cos~1.0), so FD against the
# forward proves the analytic d_vec and d_x. MJ-1024.
#
# Loss L = Σ d_y·y(vec,x).  Analytic ∂L/∂vec = d_vec, ∂L/∂x = d_x.  Central FD
# (h=1e-3, F32) must match: cos>=0.999 and max|rel| small.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/training/tests/oft_onetrainer_backward_fd.mojo -o /tmp/oft_ot_bwd_fd \
#   && /tmp/oft_ot_bwd_fd
from std.collections import List
from std.math import sqrt
from serenitymojo.training.oft_onetrainer import oft_ot_forward, oft_ot_backward

comptime COS_BAR = 0.999
comptime REL_BAR = 5.0e-3


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _loss(d_y: List[Float32], y: List[Float32]) raises -> Float64:
    var s = Float64(0.0)
    for i in range(len(y)):
        s += Float64(d_y[i]) * Float64(y[i])
    return s


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    var dot = 0.0; var na = 0.0; var nb = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("cos: zero vector")
    return dot / (sqrt(na) * sqrt(nb))


def _maxrel(a: List[Float32], b: List[Float32]) -> Float64:
    var worst = Float64(0.0)
    var scale = Float64(0.0)
    for i in range(len(b)):
        var ab = Float64(b[i]); ab = ab if ab >= 0.0 else -ab
        if ab > scale: scale = ab
    if scale < 1e-12: scale = 1e-12
    for i in range(len(a)):
        var d = Float64(a[i]) - Float64(b[i])
        d = d if d >= 0.0 else -d
        var rel = d / scale
        if rel > worst: worst = rel
    return worst


def main() raises:
    var IN = 12; var OUT = 16; var M = 5; var b = 4; var r = IN // b
    var ne = b * (b - 1) // 2
    print("[oft OneTrainer backward FD] IN=", IN, " OUT=", OUT, " M=", M, " b=", b, " r=", r)

    var vec = _fill(r * ne, 11, 0.5)
    var W = _fill(OUT * IN, 22, 0.5)
    var x = _fill(M * IN, 33, 1.0)
    var d_y = _fill(M * OUT, 44, 0.5)

    var g = oft_ot_backward(d_y.copy(), x.copy(), vec.copy(), W.copy(), M, IN, OUT, b, r)

    var h = Float32(1.0e-3)
    # ── FD ∂L/∂vec ──
    var d_vec_fd = List[Float32]()
    for t in range(r * ne):
        var vp = vec.copy(); vp[t] = vp[t] + h
        var lm = vec.copy(); lm[t] = lm[t] - h
        var yp = oft_ot_forward(x.copy(), vp, W.copy(), M, IN, OUT, b, r)
        var ym = oft_ot_forward(x.copy(), lm, W.copy(), M, IN, OUT, b, r)
        d_vec_fd.append(Float32((_loss(d_y, yp) - _loss(d_y, ym)) / Float64(2.0 * h)))
    var cv = _cos(g.d_vec, d_vec_fd); var rv = _maxrel(g.d_vec, d_vec_fd)
    print("  d_vec  cos=", cv, " maxrel=", rv)

    # ── FD ∂L/∂x ──
    var d_x_fd = List[Float32]()
    for t in range(M * IN):
        var xp = x.copy(); xp[t] = xp[t] + h
        var xm = x.copy(); xm[t] = xm[t] - h
        var yp = oft_ot_forward(xp, vec.copy(), W.copy(), M, IN, OUT, b, r)
        var ym = oft_ot_forward(xm, vec.copy(), W.copy(), M, IN, OUT, b, r)
        d_x_fd.append(Float32((_loss(d_y, yp) - _loss(d_y, ym)) / Float64(2.0 * h)))
    var cx = _cos(g.d_x, d_x_fd); var rx = _maxrel(g.d_x, d_x_fd)
    print("  d_x    cos=", cx, " maxrel=", rx)

    if cv >= COS_BAR and rv <= REL_BAR and cx >= COS_BAR and rx <= REL_BAR:
        print("PASS — OneTrainer-OFT analytic backward matches FD (d_vec + d_x)")
    else:
        raise Error("FAIL: OFT backward FD mismatch (see cos/maxrel above)")
