# training/oft_adapter.mojo — LyCORIS Diag-OFT (Orthogonal Fine-Tuning) adapter.
#
# Ports eri-lycoris/lycoris-rs/src/algorithms/oft.rs (OFTModule) + EDv2
# crates/eridiffusion-core/src/lycoris.rs OFT save convention
# (`.oft_blocks.weight` [num_blocks, b, b] + `.alpha`, lycoris.rs:905-906).
#
# ── The math (EXACT Cayley; this port deviates from the ref's Neumann approx) ──
# The reference oft.rs approximates the Cayley rotation with a 5-term Neumann
# series ("no matrix inverse needed"). THIS PORT uses the EXACT Cayley transform
# the task specifies, because small blocks make the inverse cheap AND exact, and
# an exact R gives EXACT orthogonality (RᵀR == I to F32 floor) — which is what
# lets the FD-parity gate prove the analytic backward is correct without an
# approximation-error term confounding the comparison.
#
# Per output block r (edge length b), the trainable parameter is a full [b,b]
# matrix S_r. Skew-symmetrize:
#       Q = 0.5 * (S - Sᵀ)                              (antisymmetric)
# Cayley rotation (exact):
#       M = I + Q ,  A = I - Q ,  R = M⁻¹ @ A           (orthogonal: RᵀR = I)
# At init S = 0 → Q = 0 → R = I → W_eff = W (identity-at-init, the LoRA-B==0
# convention). oft.rs uses Q = blocks - blocksᵀ (no 0.5); we fold the 0.5 in so
# S itself is the unconstrained trainable square (a constant rescale of S; the
# 0.5 is an AGENT-DEFAULT, flagged — it only rescales the learning-rate-effective
# magnitude of S, R is identical for the equivalent S).
#
# ── How R is applied (W_eff = R @ W, block-diagonal over OUTPUT groups) ────────
# The task specifies W_eff = R @ W with R block-diagonal over output groups.
# Base weight W : [out, in]; out is split into num_blocks groups of size b.
# R is block-diagonal [out,out]; W_eff[g*b + i, :] = Σ_j R_g[i,j] * W[g*b + j, :].
# Forward (frozen base W + the OFT rotation): y = x @ W_effᵀ , x:[M,in]→[M,out].
# (The ref rotates the INPUT axis instead — output-side vs input-side is the
# documented OFT save-format divergence; we take the task's output-side W_eff = R@W.
# AGENT-DEFAULT, flagged.)
#
# ── Trainable param = the per-block S matrices (num_blocks * b * b floats) ─────
# Block-diag identity disambiguation: the SAVE tensor is [num_blocks, b, b]
# holding the S blocks (the ref stores `blocks`, the pre-skew square — we store
# S likewise so a reopened S reconstructs Q, R, W_eff bit-identically).
#
# ── Backward (exact Cayley derivative) ─────────────────────────────────────────
# Forward chain per block: S → Q=0.5(S-Sᵀ) → R=M⁻¹A → W_eff=R@W → y=x@W_effᵀ.
#   d_y:[M,out].  d_W_eff = d_yᵀ @ x  [out,in].  (W frozen → no d_W.)
#   d_x = d_y @ W_eff  [M,in].
# Per block g, R_g is [b,b]; W_eff rows g*b..g*b+b come from R_g @ W_g where
# W_g = W[g*b:(g+1)*b, :]  [b,in].  d_W_eff_g : [b,in].
#   d_R_g = d_W_eff_g @ W_gᵀ              [b,b]
# Cayley derivative (dR = -M⁻¹ dQ (R + I), see header math):  given Ḡ = d_R_g,
#   d_Q_g = -M⁻ᵀ @ Ḡ @ (R + I)ᵀ          [b,b]
# Skew chain (Q = 0.5(S - Sᵀ)):
#   d_S_g = 0.5 * (d_Q_g - d_Q_gᵀ)        [b,b]
#
# BF16 trainable S storage, F32 internal compute/moments.
# Mojo 0.26.x: `def` not `fn`; multi-return via Movable struct. MIRRORS
# loha/dora/lokr_adapter.mojo structure.

from std.collections import List
from std.math import sqrt


# ── small host helpers (row-major flat) ──────────────────────────────────────
def _matmul(a: List[Float32], ra: Int, ca: Int, b: List[Float32], rb: Int, cb: Int) raises -> List[Float32]:
    if ca != rb:
        raise Error(String("oft _matmul: inner dim mismatch ") + String(ca) + " != " + String(rb))
    var out = List[Float32]()
    for _ in range(ra * cb):
        out.append(Float32(0.0))
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
    var out = List[Float32]()
    for _ in range(r * c):
        out.append(Float32(0.0))
    for i in range(r):
        for j in range(c):
            out[j * r + i] = a[i * c + j]
    return out^


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


def _f32_to_bf16_list(v: List[Float32]) -> List[BFloat16]:
    var out = List[BFloat16]()
    for i in range(len(v)):
        out.append(BFloat16(v[i]))
    return out^


# ── exact b×b inverse via Gauss-Jordan with partial pivoting (host F32) ───────
# Small b (gate uses b=2..4) → exact + cheap. Raises if singular (M=I+Q for a
# skew Q is always invertible — its eigenvalues are 1±i·imag, never 0 — so this
# never fires in practice, but we fail loud rather than return garbage).
def _inverse(a: List[Float32], n: Int) raises -> List[Float32]:
    # augmented [A | I]
    var aug = List[Float32]()
    for i in range(n):
        for j in range(n):
            aug.append(a[i * n + j])
        for j in range(n):
            aug.append(Float32(1.0) if i == j else Float32(0.0))
    var w = 2 * n
    for col in range(n):
        # partial pivot: find max-abs in this column at/below the diagonal
        var piv = col
        var best = aug[col * w + col]
        var babs = best if best >= Float32(0.0) else -best
        for r in range(col + 1, n):
            var v = aug[r * w + col]
            var va = v if v >= Float32(0.0) else -v
            if va > babs:
                babs = va
                piv = r
        if babs == Float32(0.0):
            raise Error("oft _inverse: singular matrix (no pivot)")
        if piv != col:
            for k in range(w):
                var tmp = aug[col * w + k]
                aug[col * w + k] = aug[piv * w + k]
                aug[piv * w + k] = tmp
        var diag = aug[col * w + col]
        var inv_diag = Float32(1.0) / diag
        for k in range(w):
            aug[col * w + k] = aug[col * w + k] * inv_diag
        for r in range(n):
            if r == col:
                continue
            var factor = aug[r * w + col]
            if factor == Float32(0.0):
                continue
            for k in range(w):
                aug[r * w + k] = aug[r * w + k] - factor * aug[col * w + k]
    var out = _zeros(n * n)
    for i in range(n):
        for j in range(n):
            out[i * n + j] = aug[i * w + n + j]
    return out^


# ── the OFT adapter — per-block S squares + frozen base W + AdamW ─────────────
# S       : [num_blocks, b, b] flat (trainable; the pre-skew square per block)
# w_base  : [out, in]               (frozen base weight; out = num_blocks * b)
struct OFTAdapter(Copyable, Movable):
    var s: List[BFloat16]           # [num_blocks*b*b]
    var w_base: List[Float32]       # [out, in]  (frozen)
    var num_blocks: Int
    var block_size: Int
    var in_f: Int
    var out_f: Int
    var alpha: Float32
    # AdamW moments over S only (W is frozen).
    var m_s: List[Float32]
    var v_s: List[Float32]

    def __init__(
        out self,
        var s: List[Float32], var w_base: List[Float32],
        num_blocks: Int, block_size: Int, in_f: Int, out_f: Int, alpha: Float32,
        var m_s: List[Float32], var v_s: List[Float32],
    ):
        self.s = _f32_to_bf16_list(s)
        self.w_base = w_base^
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.in_f = in_f
        self.out_f = out_f
        self.alpha = alpha
        self.m_s = m_s^
        self.v_s = v_s^


# Construct a fresh OFT adapter for Linear(in,out) with block_size b.
# out MUST be divisible by b (block-diagonal over output groups). S starts at
# zero → Q=0 → R=I → W_eff=W (identity-at-init). w_base is supplied (the frozen
# base weight the adapter rotates). Mirrors oft.rs new_linear (blocks=zeros).
def new_oft_adapter(in_f: Int, out_f: Int, block_size: Int, alpha: Float32,
                    var w_base: List[Float32]) raises -> OFTAdapter:
    if in_f == 0 or out_f == 0:
        raise Error("new_oft_adapter: in/out must be > 0")
    if block_size == 0:
        raise Error("new_oft_adapter: block_size must be > 0")
    if out_f % block_size != 0:
        raise Error(String("new_oft_adapter: out (") + String(out_f) + ") not divisible by block_size (" + String(block_size) + ")")
    if len(w_base) != out_f * in_f:
        raise Error(String("new_oft_adapter: w_base numel ") + String(len(w_base)) + " != out*in")
    var nb = out_f // block_size
    var s = _zeros(nb * block_size * block_size)   # ZERO → R=I at init
    return OFTAdapter(
        s^, w_base^, nb, block_size, in_f, out_f, alpha,
        _zeros(nb * block_size * block_size), _zeros(nb * block_size * block_size),
    )


# ── extract block g's S square [b,b] ─────────────────────────────────────────
def _block_s(lo: OFTAdapter, g: Int) -> List[Float32]:
    var b = lo.block_size
    var out = List[Float32]()
    var base = g * b * b
    for i in range(b * b):
        out.append(lo.s[base + i].cast[DType.float32]())
    return out^


# ── skew Q = 0.5*(S - Sᵀ) for a [b,b] block ──────────────────────────────────
def _skew(sblk: List[Float32], b: Int) -> List[Float32]:
    var q = _zeros(b * b)
    for i in range(b):
        for j in range(b):
            q[i * b + j] = Float32(0.5) * (sblk[i * b + j] - sblk[j * b + i])
    return q^


# ── exact Cayley R = (I+Q)⁻¹ (I-Q) for a [b,b] skew Q ─────────────────────────
def _cayley(q: List[Float32], b: Int) raises -> List[Float32]:
    var mplus = _zeros(b * b)    # I + Q
    var aminus = _zeros(b * b)   # I - Q
    for i in range(b):
        for j in range(b):
            var qij = q[i * b + j]
            var id = Float32(1.0) if i == j else Float32(0.0)
            mplus[i * b + j] = id + qij
            aminus[i * b + j] = id - qij
    var minv = _inverse(mplus, b)
    return _matmul(minv, b, b, aminus, b, b)


# ── full block-diagonal R [out,out] (for diagnostics / RtR≈I check) ───────────
def oft_rotation_full(lo: OFTAdapter) raises -> List[Float32]:
    var out = lo.out_f
    var b = lo.block_size
    var R = _zeros(out * out)
    for g in range(lo.num_blocks):
        var sblk = _block_s(lo, g)
        var q = _skew(sblk, b)
        var rg = _cayley(q, b)
        var base = g * b
        for i in range(b):
            for j in range(b):
                R[(base + i) * out + (base + j)] = rg[i * b + j]
    return R^


# ── W_eff = R @ W  [out,in]  (block-diagonal over output groups) ──────────────
def oft_effective_weight(lo: OFTAdapter) raises -> List[Float32]:
    var IN = lo.in_f
    var b = lo.block_size
    var w_eff = lo.w_base.copy()    # rows untouched if R_g were I; we overwrite per block
    for g in range(lo.num_blocks):
        var sblk = _block_s(lo, g)
        var q = _skew(sblk, b)
        var rg = _cayley(q, b)      # [b,b]
        var base = g * b
        # W_eff[base+i, :] = Σ_j rg[i,j] * W[base+j, :]
        for i in range(b):
            for c in range(IN):
                var acc = Float32(0.0)
                for j in range(b):
                    acc += rg[i * b + j] * lo.w_base[(base + j) * IN + c]
                w_eff[(base + i) * IN + c] = acc
    return w_eff^


# ── forward delta: y = x @ W_effᵀ   x:[M,in] → [M,out] ───────────────────────
def oft_forward(x_h: List[Float32], lo: OFTAdapter, M: Int) raises -> List[Float32]:
    var w_eff = oft_effective_weight(lo)              # [out,in]
    var w_eff_t = _transpose(w_eff, lo.out_f, lo.in_f)  # [in,out]
    return _matmul(x_h, M, lo.in_f, w_eff_t, lo.in_f, lo.out_f)  # [M,out]


# ── grads of S + d_x ──────────────────────────────────────────────────────────
struct OFTGrads(Copyable, Movable):
    var d_s: List[Float32]     # [num_blocks*b*b]
    var d_x: List[Float32]     # [M,in]

    def __init__(out self, var d_s: List[Float32], var d_x: List[Float32]):
        self.d_s = d_s^
        self.d_x = d_x^


# Backward through y = x @ W_effᵀ with d_y:[M,out].
def oft_backward(d_y_h: List[Float32], x_h: List[Float32], lo: OFTAdapter, M: Int) raises -> OFTGrads:
    var IN = lo.in_f
    var OUT = lo.out_f
    var b = lo.block_size

    # d_W_eff = d_yᵀ @ x  [out,in] ; d_x = d_y @ W_eff  [M,in].
    var d_y_t = _transpose(d_y_h, M, OUT)              # [out,M]
    var d_w_eff = _matmul(d_y_t, OUT, M, x_h, M, IN)   # [out,in]
    var w_eff = oft_effective_weight(lo)               # [out,in]
    var d_x = _matmul(d_y_h, M, OUT, w_eff, OUT, IN)   # [M,in]

    var d_s = _zeros(len(lo.s))

    for g in range(lo.num_blocks):
        var sblk = _block_s(lo, g)
        var q = _skew(sblk, b)
        var minv = _inverse_mplus(q, b)                # M⁻¹ where M=I+Q
        var rg = _cayley_from_minv(minv, q, b)         # R_g = M⁻¹(I-Q)

        # d_R_g = d_W_eff_g @ W_gᵀ   [b,b]
        var base = g * b
        var d_rg = _zeros(b * b)
        for i in range(b):
            for j in range(b):
                var acc = Float32(0.0)
                for c in range(IN):
                    acc += d_w_eff[(base + i) * IN + c] * lo.w_base[(base + j) * IN + c]
                d_rg[i * b + j] = acc

        # d_Q_g = -M⁻ᵀ @ d_R_g @ (R + I)ᵀ      [b,b]
        var minv_t = _transpose(minv, b, b)            # M⁻ᵀ
        var rpi = rg.copy()                            # R + I
        for i in range(b):
            rpi[i * b + i] = rpi[i * b + i] + Float32(1.0)
        var rpi_t = _transpose(rpi, b, b)              # (R+I)ᵀ
        var tmp = _matmul(minv_t, b, b, d_rg, b, b)    # M⁻ᵀ @ d_R_g
        var dq = _matmul(tmp, b, b, rpi_t, b, b)       # M⁻ᵀ @ d_R_g @ (R+I)ᵀ
        for i in range(b * b):
            dq[i] = -dq[i]

        # d_S_g = 0.5 * (d_Q_g - d_Q_gᵀ)
        var dq_t = _transpose(dq, b, b)
        var dbase = g * b * b
        for i in range(b):
            for j in range(b):
                d_s[dbase + i * b + j] = Float32(0.5) * (dq[i * b + j] - dq_t[i * b + j])

    return OFTGrads(d_s^, d_x^)


# helpers shared by backward: M⁻¹ from Q (M=I+Q), and R from that M⁻¹.
def _inverse_mplus(q: List[Float32], b: Int) raises -> List[Float32]:
    var mplus = _zeros(b * b)
    for i in range(b):
        for j in range(b):
            var id = Float32(1.0) if i == j else Float32(0.0)
            mplus[i * b + j] = id + q[i * b + j]
    return _inverse(mplus, b)


def _cayley_from_minv(minv: List[Float32], q: List[Float32], b: Int) raises -> List[Float32]:
    var aminus = _zeros(b * b)
    for i in range(b):
        for j in range(b):
            var id = Float32(1.0) if i == j else Float32(0.0)
            aminus[i * b + j] = id - q[i * b + j]
    return _matmul(minv, b, b, aminus, b, b)


# ── AdamW one step over S (W frozen) ──────────────────────────────────────────
def _adamw_host_list(
    mut p: List[BFloat16], g: List[Float32],
    mut mom: List[Float32], mut vmo: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
) raises:
    var n = len(p)
    if len(g) != n or len(mom) != n or len(vmo) != n:
        raise Error("oft _adamw_host_list: param/grad/m/v len mismatch")
    if t < 1:
        raise Error("oft _adamw_host_list: t must be >= 1")
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    for i in range(n):
        var gv = g[i]
        var mi = beta1 * mom[i] + (Float32(1.0) - beta1) * gv
        var vi = beta2 * vmo[i] + (Float32(1.0) - beta2) * gv * gv
        mom[i] = mi
        vmo[i] = vi
        var m_hat = mi / bc1
        var v_hat = vi / bc2
        var pv = p[i].cast[DType.float32]() - lr * m_hat / (sqrt(v_hat) + eps)
        if weight_decay > 0.0:
            pv = pv - lr * weight_decay * pv
        p[i] = BFloat16(pv)


def oft_adamw(
    mut lo: OFTAdapter, g: OFTGrads, t: Int, lr: Float32,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    _adamw_host_list(lo.s, g.d_s, lo.m_s, lo.v_s, t, lr, beta1, beta2, eps, weight_decay)
