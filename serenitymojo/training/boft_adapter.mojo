# training/boft_adapter.mojo — LyCORIS BOFT (Butterfly Orthogonal Fine-Tuning).
#
# Ports eri-lycoris/lycoris-rs/src/algorithms/boft.rs (BOFTModule) + EDv2
# crates/eridiffusion-core/src/lycoris.rs BOFT save convention
# (`.oft_blocks.weight` 4D [boft_m, num_blocks, b, b] + `.alpha`,
# lycoris.rs:908-913 — same suffix as OFT, the 4D rank disambiguates).
#
# ── The math (m-stage butterfly product of exact-Cayley rotations) ────────────
# Diag-OFT applies ONE block-diagonal orthogonal rotation. BOFT applies `m`
# consecutive rotations, with a fixed butterfly PERMUTATION between stages so the
# blocks of one stage interleave the blocks of the next — the product's group
# structure is exponentially richer than a single block-diagonal map (boft.rs
# header). We express each stage as an explicit orthogonal matrix so the overall
# transform is verifiably orthogonal AND the per-stage backward is a clean
# reverse-order matrix accumulation (FD-checkable for EVERY factor param).
#
# Per output block (edge b), per stage i, the trainable param is a full [b,b]
# square S_i[g]. Skew + exact Cayley exactly as OFT:
#       Q_i[g] = 0.5*(S_i[g] - S_i[g]ᵀ)
#       R_i[g] = (I + Q_i[g])⁻¹ (I - Q_i[g])            (orthogonal block)
# R_i = block-diagonal([R_i[0], …, R_i[nb-1]])  [out,out]  (orthogonal).
# Fixed butterfly permutation P_i [out,out] (a permutation matrix → orthogonal).
# Per-stage orthogonal factor (conjugation keeps orthogonality + realizes the
# interleave):
#       B_i = P_iᵀ R_i P_i                               (orthogonal)
# Total transform (reverse stage order, butterfly product):
#       T = B_{m-1} @ … @ B_0                            (orthogonal: TᵀT = I)
# Effective weight + forward (W_eff = T @ W, block/whole-output rotation):
#       W_eff = T @ W   [out,in] ;  y = x @ W_effᵀ   x:[M,in]→[M,out].
# At init every S_i = 0 → Q_i = 0 → R_i = I → B_i = I → T = I → W_eff = W
# (identity-at-init, the LoRA-B==0 convention).
#
# ── boft_m (stage count) ──────────────────────────────────────────────────────
# boft.rs: boft_m = popcount(num_blocks - 1) + 1 (= log2(num_blocks)+1 for a
# power-of-two num_blocks). We compute the literal popcount form to stay faithful.
#
# ── Butterfly permutation P_i ──────────────────────────────────────────────────
# boft.rs stage i uses g=2, k=2^i*(b/2): reshape (.,c,g,k)→transpose(-2,-1)→
# flatten, i.e. a deterministic index permutation of the feature axis. We
# materialize the SAME permutation as an explicit [out,out] matrix P_i from that
# index map (AGENT-DEFAULT, flagged: explicit-matrix form of boft.rs's reshape/
# transpose permute, applied output-side per the task's W_eff=T@W spec instead
# of the ref's input-side rotation). P_i is an involution-class permutation; its
# inverse is its transpose. At b=2 stages collapse to plain block rotations with
# the trivial permutation, which is still a valid (degenerate) butterfly.
#
# ── Backward (reverse-order accumulation through the butterfly product) ────────
# d_y:[M,out] → W_eff = T@W (W frozen): d_W_eff = d_yᵀ @ x [out,in];
#   d_x = d_y @ W_eff [M,in];  d_T = d_W_eff @ Wᵀ  [out,out].
# For T = B_{m-1}…B_0:  d_B_i = (left_i)ᵀ @ d_T @ (right_i)ᵀ
#   where left_i = B_{m-1}…B_{i+1} (I if i=m-1), right_i = B_{i-1}…B_0 (I if i=0).
# B_i = P_iᵀ R_i P_i (P orthogonal): d_R_i = P_i @ d_B_i @ P_iᵀ.
# Per block g of R_i (exact-Cayley backward, identical to OFT):
#   d_R_i[g] = the [b,b] diagonal block of d_R_i at rows/cols g*b..g*b+b
#   d_Q_i[g] = -M⁻ᵀ @ d_R_i[g] @ (R_i[g] + I)ᵀ ,  M = I+Q_i[g]
#   d_S_i[g] = 0.5*(d_Q_i[g] - d_Q_i[g]ᵀ)
#
# BF16 trainable S storage, F32 internal compute/moments. Mojo 0.26.x. MIRRORS oft_adapter.mojo (reuses its
# inverse/skew/Cayley helpers re-implemented here to keep the file self-contained).

from std.collections import List
from std.math import sqrt


# ── small host helpers (row-major flat) ──────────────────────────────────────
def _matmul(a: List[Float32], ra: Int, ca: Int, b: List[Float32], rb: Int, cb: Int) raises -> List[Float32]:
    if ca != rb:
        raise Error(String("boft _matmul: inner dim mismatch ") + String(ca) + " != " + String(rb))
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


def _identity(n: Int) -> List[Float32]:
    var out = _zeros(n * n)
    for i in range(n):
        out[i * n + i] = Float32(1.0)
    return out^


def _f32_to_bf16_list(v: List[Float32]) -> List[BFloat16]:
    var out = List[BFloat16]()
    for i in range(len(v)):
        out.append(BFloat16(v[i]))
    return out^


# ── exact b×b inverse via Gauss-Jordan with partial pivoting (host F32) ───────
def _inverse(a: List[Float32], n: Int) raises -> List[Float32]:
    var aug = List[Float32]()
    for i in range(n):
        for j in range(n):
            aug.append(a[i * n + j])
        for j in range(n):
            aug.append(Float32(1.0) if i == j else Float32(0.0))
    var w = 2 * n
    for col in range(n):
        var piv = col
        var best = aug[col * w + col]
        var babs = best if best >= Float32(0.0) else -best
        for r in range(col + 1, n):
            var v = aug[r * w + col]
            var va = v if v >= Float32(0.0) else -v
            if va > babs:
                babs = va; piv = r
        if babs == Float32(0.0):
            raise Error("boft _inverse: singular matrix (no pivot)")
        if piv != col:
            for k in range(w):
                var tmp = aug[col * w + k]
                aug[col * w + k] = aug[piv * w + k]
                aug[piv * w + k] = tmp
        var inv_diag = Float32(1.0) / aug[col * w + col]
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


# popcount(x)
def _popcount(x: Int) -> Int:
    var v = x
    var c = 0
    while v > 0:
        c += v & 1
        v = v >> 1
    return c


# boft_m = popcount(num_blocks - 1) + 1  (boft.rs:260-264).
def boft_stage_count(num_blocks: Int) -> Int:
    if num_blocks <= 1:
        return 1
    return _popcount(num_blocks - 1) + 1


# ── butterfly permutation P_i as an explicit [out,out] permutation matrix ─────
# Realizes boft.rs stage-i feature-axis permute (g=2, k=2^i*(b/2)):
#   index map: out → reshape (c,g,k) → transpose(g,k) → flatten.
# perm[new] = old. P_i[new, old] = 1. If g*k does not divide out for this stage
# (degenerate small config), P_i = I (the butterfly collapses to a plain block
# rotation that stage — still a valid orthogonal factor).
def _butterfly_perm(out_f: Int, b: Int, stage: Int) raises -> List[Float32]:
    var g = 2
    var k = (1 << stage) * (b // 2)
    var gk = g * k
    if gk == 0 or out_f % gk != 0:
        return _identity(out_f)
    var c = out_f // gk
    # forward index: element at (cc, gg, kk) in [c,g,k] moves to (cc, kk, gg) in
    # [c,k,g] after transpose, then flatten. new_idx walks the [c,k,g] layout.
    var P = _zeros(out_f * out_f)
    for cc in range(c):
        for gg in range(g):
            for kk in range(k):
                var old_idx = (cc * g + gg) * k + kk
                var new_idx = (cc * k + kk) * g + gg
                P[new_idx * out_f + old_idx] = Float32(1.0)
    return P^


# ── the BOFT adapter — per-stage per-block S squares + frozen base W + AdamW ──
# s       : [boft_m, num_blocks, b, b] flat (trainable; pre-skew squares)
# w_base  : [out, in]                       (frozen; out = num_blocks * b)
struct BOFTAdapter(Copyable, Movable):
    var s: List[BFloat16]           # [boft_m*num_blocks*b*b]
    var w_base: List[Float32]       # [out, in]  (frozen)
    var boft_m: Int
    var num_blocks: Int
    var block_size: Int
    var in_f: Int
    var out_f: Int
    var alpha: Float32
    var m_s: List[Float32]
    var v_s: List[Float32]

    def __init__(
        out self,
        var s: List[Float32], var w_base: List[Float32],
        boft_m: Int, num_blocks: Int, block_size: Int, in_f: Int, out_f: Int, alpha: Float32,
        var m_s: List[Float32], var v_s: List[Float32],
    ):
        self.s = _f32_to_bf16_list(s)
        self.w_base = w_base^
        self.boft_m = boft_m
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.in_f = in_f
        self.out_f = out_f
        self.alpha = alpha
        self.m_s = m_s^
        self.v_s = v_s^


# Construct a fresh BOFT adapter for Linear(in,out). out split into num_blocks
# blocks of block_size b (out = num_blocks*b). S starts zero → T=I → W_eff=W
# (identity-at-init). w_base is the supplied frozen base weight. b MUST be even
# (the butterfly permute splits into groups of 2, boft.rs:269). Mirrors boft.rs
# new_linear (blocks=zeros, boft_m=popcount).
def new_boft_adapter(in_f: Int, out_f: Int, block_size: Int, alpha: Float32,
                     var w_base: List[Float32]) raises -> BOFTAdapter:
    if in_f == 0 or out_f == 0:
        raise Error("new_boft_adapter: in/out must be > 0")
    if block_size == 0 or block_size % 2 != 0:
        raise Error(String("new_boft_adapter: block_size (") + String(block_size) + ") must be > 0 and even")
    if out_f % block_size != 0:
        raise Error(String("new_boft_adapter: out (") + String(out_f) + ") not divisible by block_size (" + String(block_size) + ")")
    if len(w_base) != out_f * in_f:
        raise Error(String("new_boft_adapter: w_base numel ") + String(len(w_base)) + " != out*in")
    var nb = out_f // block_size
    var m = boft_stage_count(nb)
    var s = _zeros(m * nb * block_size * block_size)   # ZERO → T=I at init
    return BOFTAdapter(
        s^, w_base^, m, nb, block_size, in_f, out_f, alpha,
        _zeros(m * nb * block_size * block_size), _zeros(m * nb * block_size * block_size),
    )


# ── skew + exact Cayley for one [b,b] block ──────────────────────────────────
def _skew(sblk: List[Float32], b: Int) -> List[Float32]:
    var q = _zeros(b * b)
    for i in range(b):
        for j in range(b):
            q[i * b + j] = Float32(0.5) * (sblk[i * b + j] - sblk[j * b + i])
    return q^


def _cayley(q: List[Float32], b: Int) raises -> List[Float32]:
    var mplus = _zeros(b * b)
    var aminus = _zeros(b * b)
    for i in range(b):
        for j in range(b):
            var qij = q[i * b + j]
            var id = Float32(1.0) if i == j else Float32(0.0)
            mplus[i * b + j] = id + qij
            aminus[i * b + j] = id - qij
    var minv = _inverse(mplus, b)
    return _matmul(minv, b, b, aminus, b, b)


# ── block-diagonal R_i [out,out] from stage i's S squares ─────────────────────
def _stage_R(lo: BOFTAdapter, stage: Int) raises -> List[Float32]:
    var out = lo.out_f
    var b = lo.block_size
    var R = _zeros(out * out)
    var stage_base = stage * lo.num_blocks * b * b
    for g in range(lo.num_blocks):
        var sblk = List[Float32]()
        var sb = stage_base + g * b * b
        for i in range(b * b):
            sblk.append(lo.s[sb + i].cast[DType.float32]())
        var q = _skew(sblk, b)
        var rg = _cayley(q, b)
        var base = g * b
        for i in range(b):
            for j in range(b):
                R[(base + i) * out + (base + j)] = rg[i * b + j]
    return R^


# ── per-stage orthogonal factor B_i = P_iᵀ R_i P_i [out,out] ──────────────────
def _stage_B(lo: BOFTAdapter, stage: Int) raises -> List[Float32]:
    var out = lo.out_f
    var R = _stage_R(lo, stage)
    var P = _butterfly_perm(out, lo.block_size, stage)
    var Pt = _transpose(P, out, out)
    var tmp = _matmul(Pt, out, out, R, out, out)     # Pᵀ R
    return _matmul(tmp, out, out, P, out, out)       # Pᵀ R P


# ── total transform T = B_{m-1} @ … @ B_0 [out,out] ──────────────────────────
def boft_transform(lo: BOFTAdapter) raises -> List[Float32]:
    var out = lo.out_f
    var T = _identity(out)
    for stage in range(lo.boft_m):           # apply B_0 first, then B_1, …
        var B = _stage_B(lo, stage)
        T = _matmul(B, out, out, T, out, out)  # T ← B_stage @ T
    return T^


# ── W_eff = T @ W  [out,in] ───────────────────────────────────────────────────
def boft_effective_weight(lo: BOFTAdapter) raises -> List[Float32]:
    var T = boft_transform(lo)
    return _matmul(T, lo.out_f, lo.out_f, lo.w_base, lo.out_f, lo.in_f)


# ── forward: y = x @ W_effᵀ   x:[M,in] → [M,out] ─────────────────────────────
def boft_forward(x_h: List[Float32], lo: BOFTAdapter, M: Int) raises -> List[Float32]:
    var w_eff = boft_effective_weight(lo)
    var w_eff_t = _transpose(w_eff, lo.out_f, lo.in_f)
    return _matmul(x_h, M, lo.in_f, w_eff_t, lo.in_f, lo.out_f)


# ── grads of all stage S squares + d_x ───────────────────────────────────────
struct BOFTGrads(Copyable, Movable):
    var d_s: List[Float32]     # [boft_m*num_blocks*b*b]
    var d_x: List[Float32]     # [M,in]

    def __init__(out self, var d_s: List[Float32], var d_x: List[Float32]):
        self.d_s = d_s^
        self.d_x = d_x^


# Backward through the butterfly product (reverse-order accumulation).
def boft_backward(d_y_h: List[Float32], x_h: List[Float32], lo: BOFTAdapter, M: Int) raises -> BOFTGrads:
    var IN = lo.in_f
    var OUT = lo.out_f
    var b = lo.block_size
    var m = lo.boft_m

    # d_W_eff = d_yᵀ @ x [out,in] ; d_x = d_y @ W_eff [M,in] ; d_T = d_W_eff @ Wᵀ.
    var d_y_t = _transpose(d_y_h, M, OUT)                 # [out,M]
    var d_w_eff = _matmul(d_y_t, OUT, M, x_h, M, IN)      # [out,in]
    var w_eff = boft_effective_weight(lo)                 # [out,in]
    var d_x = _matmul(d_y_h, M, OUT, w_eff, OUT, IN)      # [M,in]
    var w_base_t = _transpose(lo.w_base, OUT, IN)         # [in,out]
    var d_T = _matmul(d_w_eff, OUT, IN, w_base_t, IN, OUT)  # [out,out]

    # Precompute the per-stage B factors and prefix products for left/right.
    var Bs = List[List[Float32]]()
    for stage in range(m):
        Bs.append(_stage_B(lo, stage))

    var d_s = _zeros(len(lo.s))

    # left_i = B_{m-1}…B_{i+1} ; right_i = B_{i-1}…B_0.
    # d_B_i = left_iᵀ @ d_T @ right_iᵀ.
    for i in range(m):
        # right_i = B_{i-1} … B_0
        var right = _identity(OUT)
        for s2 in range(i):
            right = _matmul(Bs[s2], OUT, OUT, right, OUT, OUT)  # right ← B_s2 @ right
        # left_i = B_{m-1} … B_{i+1}
        var left = _identity(OUT)
        for s2 in range(i + 1, m):
            left = _matmul(Bs[s2], OUT, OUT, left, OUT, OUT)    # left ← B_s2 @ left

        var left_t = _transpose(left, OUT, OUT)
        var right_t = _transpose(right, OUT, OUT)
        var tmp = _matmul(left_t, OUT, OUT, d_T, OUT, OUT)      # leftᵀ d_T
        var d_B = _matmul(tmp, OUT, OUT, right_t, OUT, OUT)     # leftᵀ d_T rightᵀ

        # d_R_i = P_i @ d_B_i @ P_iᵀ
        var P = _butterfly_perm(OUT, b, i)
        var Pt = _transpose(P, OUT, OUT)
        var t2 = _matmul(P, OUT, OUT, d_B, OUT, OUT)
        var d_R = _matmul(t2, OUT, OUT, Pt, OUT, OUT)           # [out,out]

        # Per block g: exact-Cayley backward to d_S_i[g].
        var stage_base = i * lo.num_blocks * b * b
        for grp in range(lo.num_blocks):
            var base = grp * b
            # extract S block + recompute Q, M⁻¹, R_g
            var sblk = List[Float32]()
            var sb = stage_base + grp * b * b
            for q in range(b * b):
                sblk.append(lo.s[sb + q].cast[DType.float32]())
            var qmat = _skew(sblk, b)
            var minv = _inverse_mplus(qmat, b)
            var rg = _cayley_from_minv(minv, qmat, b)

            # d_R_g = the [b,b] diagonal block of d_R at rows/cols base..base+b
            var d_rg = _zeros(b * b)
            for ii in range(b):
                for jj in range(b):
                    d_rg[ii * b + jj] = d_R[(base + ii) * OUT + (base + jj)]

            # d_Q_g = -M⁻ᵀ @ d_R_g @ (R_g + I)ᵀ
            var minv_t = _transpose(minv, b, b)
            var rpi = rg.copy()
            for ii in range(b):
                rpi[ii * b + ii] = rpi[ii * b + ii] + Float32(1.0)
            var rpi_t = _transpose(rpi, b, b)
            var tt = _matmul(minv_t, b, b, d_rg, b, b)
            var dq = _matmul(tt, b, b, rpi_t, b, b)
            for q in range(b * b):
                dq[q] = -dq[q]

            # d_S_g = 0.5*(d_Q_g - d_Q_gᵀ)
            var dq_t = _transpose(dq, b, b)
            for ii in range(b):
                for jj in range(b):
                    d_s[sb + ii * b + jj] = Float32(0.5) * (dq[ii * b + jj] - dq_t[ii * b + jj])

    return BOFTGrads(d_s^, d_x^)


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
        raise Error("boft _adamw_host_list: param/grad/m/v len mismatch")
    if t < 1:
        raise Error("boft _adamw_host_list: t must be >= 1")
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
        var pv = p[i].cast[DType.float32]()
        if weight_decay > 0.0:
            pv = pv * (Float32(1.0) - lr * weight_decay)
        pv = pv - lr * m_hat / (sqrt(v_hat) + eps)
        p[i] = BFloat16(pv)


def boft_adamw(
    mut lo: BOFTAdapter, g: BOFTGrads, t: Int, lr: Float32,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    _adamw_host_list(lo.s, g.d_s, lo.m_s, lo.v_s, t, lr, beta1, beta2, eps, weight_decay)
