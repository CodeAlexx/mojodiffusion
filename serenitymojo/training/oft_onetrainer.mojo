# training/oft_onetrainer.mojo — OneTrainer-parity OFT (Diag-OFT, Linear).
#
# Distinct from oft_adapter.mojo (which is the lycoris exact-Cayley / output-side
# variant). This mirrors OneTrainer's OFTModule + OFTRotationModule EXACTLY (MJ-1024):
#   - param = a per-block UPPER-TRIANGULAR vector weight[r, ne], ne=b(b-1)/2,
#     where r = in_features // block_size blocks over the INPUT dimension.
#   - skew Q = matrix(vec) - matrix(vec)ᵀ   (NO 0.5 factor; oft_utils.py:69-79).
#   - R = 5-term NEUMANN polynomial (NOT exact Cayley; oft_utils.py:97-110):
#         R = I + 2Q + 2Q² + 2Q³ + Q⁴      (a non-orthogonal truncation by design).
#   - applied INPUT-side, per block: x_rot[...,g,c] = Σ_k x[...,g,k]·R_g[k,c]
#     (einsum "...rk,rkc->...rc"), then the FROZEN linear: y = x_rot @ Wᵀ.
# At weight=0 → Q=0 → R=I → y = x@Wᵀ (identity at init).

from std.collections import List


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _matmul(a: List[Float32], ra: Int, ca: Int, b: List[Float32], rb: Int, cb: Int) raises -> List[Float32]:
    if ca != rb:
        raise Error("oft_ot _matmul: dim mismatch")
    var out = _zeros(ra * cb)
    for i in range(ra):
        for k in range(ca):
            var aik = a[i * ca + k]
            if aik == Float32(0.0):
                continue
            var brow = k * cb
            var orow = i * cb
            for j in range(cb):
                out[orow + j] = out[orow + j] + aik * b[brow + j]
    return out^


def _transpose(a: List[Float32], r: Int, c: Int) -> List[Float32]:
    var out = _zeros(r * c)
    for i in range(r):
        for j in range(c):
            out[j * r + i] = a[i * c + j]
    return out^


# Skew Q[b,b] from a row-major upper-triangular vector (i<j order, matches
# torch.triu_indices(b,b,1)): Q[i,j]=vec, Q[j,i]=-vec, diag=0.
def oft_ot_skew(vec_block: List[Float32], b: Int) -> List[Float32]:
    var q = _zeros(b * b)
    var k = 0
    for i in range(b):
        for j in range(i + 1, b):
            var v = vec_block[k]
            q[i * b + j] = v
            q[j * b + i] = -v
            k += 1
    return q^


# 5-term Neumann: R = I + 2Q + 2Q² + 2Q³ + Q⁴  (OneTrainer use_cayley_neumann default).
def oft_ot_neumann_r(q: List[Float32], b: Int) raises -> List[Float32]:
    var r = _zeros(b * b)
    for i in range(b):
        r[i * b + i] = Float32(1.0)                 # I
    for i in range(b * b):
        r[i] = r[i] + Float32(2.0) * q[i]           # + 2Q
    var q2 = _matmul(q, b, b, q, b, b)
    for i in range(b * b):
        r[i] = r[i] + Float32(2.0) * q2[i]          # + 2Q²
    var q3 = _matmul(q2, b, b, q, b, b)
    for i in range(b * b):
        r[i] = r[i] + Float32(2.0) * q3[i]          # + 2Q³
    var q4 = _matmul(q3, b, b, q, b, b)
    for i in range(b * b):
        r[i] = r[i] + q4[i]                          # + Q⁴
    return r^


# Forward: rotate x in INPUT-blocks then the frozen linear. x:[M,IN] → y:[M,OUT].
#   vec : [r * ne]  (r = IN//b blocks, ne = b(b-1)/2)
#   W   : [OUT,IN]  (frozen base)
def oft_ot_forward(
    x_h: List[Float32], vec: List[Float32], w: List[Float32],
    M: Int, IN: Int, OUT: Int, b: Int, r: Int,
) raises -> List[Float32]:
    var ne = b * (b - 1) // 2
    # x_rot[m, g*b + c] = Σ_k x[m, g*b + k] · R_g[k,c]
    var x_rot = _zeros(M * IN)
    for g in range(r):
        var vblk = List[Float32]()
        for t in range(ne):
            vblk.append(vec[g * ne + t])
        var q = oft_ot_skew(vblk, b)
        var rg = oft_ot_neumann_r(q, b)             # [b,b]
        var base = g * b
        for m in range(M):
            for c in range(b):
                var acc = Float32(0.0)
                for k in range(b):
                    acc += x_h[m * IN + base + k] * rg[k * b + c]
                x_rot[m * IN + base + c] = acc
    # y = x_rot @ Wᵀ
    var w_t = _transpose(w, OUT, IN)                 # [in,out]
    return _matmul(x_rot, M, IN, w_t, IN, OUT)       # [M,out]


# ── backward ───────────────────────────────────────────────────────────────
# d_Q for R = I + 2Q + 2Q² + 2Q³ + Q⁴ given Ḡ = d_R. For a term c·Qⁿ,
#   d_Q += c · Σ_{p=0}^{n-1} (Qᵀ)ᵖ · Ḡ · (Qᵀ)^{n-1-p}.
def _neumann_backward(q: List[Float32], gbar: List[Float32], b: Int) raises -> List[Float32]:
    var qt = _transpose(q, b, b)                     # Qᵀ
    var qt2 = _matmul(qt, b, b, qt, b, b)            # (Q²)ᵀ
    var qt3 = _matmul(qt2, b, b, qt, b, b)           # (Q³)ᵀ
    var dq = _zeros(b * b)
    var n2a = _matmul(gbar, b, b, qt, b, b)                                # Ḡ Qᵀ
    var n2b = _matmul(qt, b, b, gbar, b, b)                                # Qᵀ Ḡ
    var n3a = _matmul(gbar, b, b, qt2, b, b)                               # Ḡ (Q²)ᵀ
    var n3b = _matmul(_matmul(qt, b, b, gbar, b, b), b, b, qt, b, b)       # Qᵀ Ḡ Qᵀ
    var n3c = _matmul(qt2, b, b, gbar, b, b)                               # (Q²)ᵀ Ḡ
    var n4a = _matmul(gbar, b, b, qt3, b, b)                               # Ḡ (Q³)ᵀ
    var n4b = _matmul(_matmul(qt, b, b, gbar, b, b), b, b, qt2, b, b)      # Qᵀ Ḡ (Q²)ᵀ
    var n4c = _matmul(_matmul(qt2, b, b, gbar, b, b), b, b, qt, b, b)      # (Q²)ᵀ Ḡ Qᵀ
    var n4d = _matmul(qt3, b, b, gbar, b, b)                               # (Q³)ᵀ Ḡ
    for i in range(b * b):
        dq[i] = (
            Float32(2.0) * gbar[i]
            + Float32(2.0) * (n2a[i] + n2b[i])
            + Float32(2.0) * (n3a[i] + n3b[i] + n3c[i])
            + (n4a[i] + n4b[i] + n4c[i] + n4d[i])
        )
    return dq^


struct OFTOTGrads(Movable):
    var d_vec: List[Float32]   # [r*ne]  (trainable param grad)
    var d_x: List[Float32]     # [M,in]  (residual stream)

    def __init__(out self, var d_vec: List[Float32], var d_x: List[Float32]):
        self.d_vec = d_vec^
        self.d_x = d_x^


# Backward through y = oft_ot_forward(x, vec, W). W frozen → only d_vec (+ d_x).
def oft_ot_backward(
    d_y_h: List[Float32], x_h: List[Float32], vec: List[Float32], w: List[Float32],
    M: Int, IN: Int, OUT: Int, b: Int, r: Int,
) raises -> OFTOTGrads:
    var ne = b * (b - 1) // 2
    # d_x_rot = d_y @ W   [M,IN]
    var d_x_rot = _matmul(d_y_h, M, OUT, w, OUT, IN)
    var d_vec = _zeros(r * ne)
    var d_x = _zeros(M * IN)
    for g in range(r):
        var base = g * b
        var vblk = List[Float32]()
        for t in range(ne):
            vblk.append(vec[g * ne + t])
        var q = oft_ot_skew(vblk, b)
        var rg = oft_ot_neumann_r(q, b)               # [b,b]
        # d_R_g[k,c] = Σ_m d_x_rot[m,base+c]·x[m,base+k]
        var d_rg = _zeros(b * b)
        for k in range(b):
            for c in range(b):
                var acc = Float32(0.0)
                for m in range(M):
                    acc += d_x_rot[m * IN + base + c] * x_h[m * IN + base + k]
                d_rg[k * b + c] = acc
        # d_x[m,base+k] = Σ_c d_x_rot[m,base+c]·R_g[k,c]
        for m in range(M):
            for k in range(b):
                var acc = Float32(0.0)
                for c in range(b):
                    acc += d_x_rot[m * IN + base + c] * rg[k * b + c]
                d_x[m * IN + base + k] = acc
        # d_Q_g through the Neumann polynomial, then d_vec through the skew.
        var dq = _neumann_backward(q, d_rg, b)        # [b,b]
        var t = 0
        for i in range(b):
            for j in range(i + 1, b):
                d_vec[g * ne + t] = dq[i * b + j] - dq[j * b + i]
                t += 1
    return OFTOTGrads(d_vec^, d_x^)
