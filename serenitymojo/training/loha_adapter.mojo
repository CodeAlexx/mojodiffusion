# training/loha_adapter.mojo — LyCORIS LoHa (LoRA with Hadamard product) adapter.
#
# Ports eri-lycoris/lycoris-rs/src/algorithms/loha.rs (LoHaModule, Linear path)
# + upstream lycoris/functional/loha.py HadaWeight (the canonical 4-factor
# forward/backward) + EDv2 crates/eridiffusion-core/src/lycoris.rs LoHa save
# convention (hada_w1_a/b, hada_w2_a/b).
#
# ── The math (Linear, the only path this wave ships) ──────────────────────────
# Weight layout follows the Flame Linear contract used by loha.rs: factors are
#   w1a, w2a : [IN, RANK]      w1b, w2b : [RANK, OUT]
# and the diff weight is
#   ΔW = (w1a @ w1b) ⊙ (w2a @ w2b) * scale          (loha.rs get_diff_weight)
#       = HadaWeight.forward in functional/loha.py (w1d=w?b, w1u=w?a; the call
#         site passes hada_w1_b, hada_w1_a so `w1u @ w1d == w1a @ w1b`).
# A layer applies it as a delta on the linear output:
#   y_delta = x @ ΔW        x:[M,IN]  ΔW:[IN,OUT]  →  y_delta:[M,OUT]
# This matches the plain-LoRA convention in train_step.mojo where the LoRA
# contribution is ADDED to the projection output (B@A delta), only here the
# low-rank delta is the Hadamard of two rank-r products.
#
# ── Backward — ALL FOUR factors (the #1 historical LoHa bug) ──────────────────
# Given d_y [M,OUT] flowing back into y_delta = x @ ΔW:
#   d_ΔW = xᵀ @ d_y                       [IN,OUT]   (standard linear-weight grad)
#   d_x  = d_y @ ΔWᵀ                       [M,IN]     (for the residual stream)
# Then HadaWeight.backward (functional/loha.py:18-30), with grad_out := d_ΔW:
#   g  = d_ΔW * scale
#   t1 = g ⊙ (w2a @ w2b)                   [IN,OUT]
#   d_w1a = t1 @ w1bᵀ                       [IN,RANK]
#   d_w1b = w1aᵀ @ t1                       [RANK,OUT]
#   t2 = g ⊙ (w1a @ w1b)                   [IN,OUT]
#   d_w2a = t2 @ w2bᵀ                       [IN,RANK]
#   d_w2b = w2aᵀ @ t2                       [RANK,OUT]
# Each factor's grad depends on the OTHER pair via the Hadamard term — dropping
# any of the four (the known bug) leaves a factor with an identically-zero grad
# that GradCoverage flags as dead. We compute all four explicitly.
#
# ── Init (matches loha.rs new_linear_for_training == upstream use_scalar=False)─
#   Upstream intent: w1a ~ N(0, 0.1)  w1b ~ N(0, 1)  w2a = 0  w2b ~ N(0, 1).
#   This port uses the project _randn convention (centered-uniform * scale, see
#   _randn below): w1a ~ U(-0.05, 0.05), w1b/w2b ~ U(-0.5, 0.5), w2a = 0. The
#   distribution shape differs from upstream Gaussian — the same approximation the
#   whole Mojo port makes for every weight init — but the zero-leaf (w2a=0) and the
#   ΔW==0 identity at init are exact, which is all the correctness gates depend on.
# Identity at init: (w1a@w1b) ⊙ (0@w2b) = (·) ⊙ 0 = 0, so ΔW == 0 (like LoRA B=0).
# At step 0 only w2a (the zero leaf) gets a nonzero grad — exactly upstream
# behavior. (GradCoverage is asserted at step >= 1, after the optimizer drives
# w2a off zero, when ALL FOUR are nonzero — see grad_coverage.mojo step gate.)
#
# BF16 trainable storage, F32 internal compute/moments. The factors are tiny
# ([IN,RANK]/[RANK,OUT]/RANK²), so host matmul is exact and auditable after
# explicit BF16->F32 materialization.
#
# Mojo 1.0.0b1: `def` not `fn`; move-only Tensor → ArcPointer in collections;
# multi-return via Movable struct; STDtype.F32 is a value.

from std.collections import List
from std.math import sqrt


# ── small host matmul / hadamard helpers (row-major flat List[Float32]) ──────
# A[ra,ca] @ B[rb,cb] with ca == rb → [ra,cb].
def _matmul(a: List[Float32], ra: Int, ca: Int, b: List[Float32], rb: Int, cb: Int) raises -> List[Float32]:
    if ca != rb:
        raise Error(
            String("_matmul: inner dim mismatch ") + String(ca) + " != " + String(rb)
        )
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


# Aᵀ : [r,c] → [c,r]
def _transpose(a: List[Float32], r: Int, c: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(r * c):
        out.append(Float32(0.0))
    for i in range(r):
        for j in range(c):
            out[j * r + i] = a[i * c + j]
    return out^


# elementwise product of two equal-length flats.
def _hadamard(a: List[Float32], b: List[Float32]) raises -> List[Float32]:
    if len(a) != len(b):
        raise Error(
            String("_hadamard: length mismatch ") + String(len(a)) + " != " + String(len(b))
        )
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(a[i] * b[i])
    return out^


def _scale_inplace(mut a: List[Float32], s: Float32):
    for i in range(len(a)):
        a[i] = a[i] * s


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    # Same deterministic host randn as train_step.mojo (centered uniform * scale).
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _f32_to_bf16_list(v: List[Float32]) -> List[BFloat16]:
    var out = List[BFloat16]()
    for i in range(len(v)):
        out.append(BFloat16(v[i]))
    return out^


def _bf16_to_f32_list(v: List[BFloat16]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(v)):
        out.append(v[i].cast[DType.float32]())
    return out^


# ── the LoHa adapter (Linear) — 4 factors + AdamW moments, host F32 ──────────
struct LoHaAdapter(Copyable, Movable):
    var w1a: List[BFloat16]  # [in, rank]
    var w1b: List[BFloat16]  # [rank, out]
    var w2a: List[BFloat16]  # [in, rank]
    var w2b: List[BFloat16]  # [rank, out]
    var rank: Int
    var in_f: Int
    var out_f: Int
    var alpha: Float32
    var scale: Float32       # alpha / rank
    # AdamW first/second moments, one pair per factor (same flat layout).
    var m1a: List[Float32]
    var v1a: List[Float32]
    var m1b: List[Float32]
    var v1b: List[Float32]
    var m2a: List[Float32]
    var v2a: List[Float32]
    var m2b: List[Float32]
    var v2b: List[Float32]

    def __init__(
        out self,
        var w1a: List[Float32], var w1b: List[Float32],
        var w2a: List[Float32], var w2b: List[Float32],
        rank: Int, in_f: Int, out_f: Int, alpha: Float32,
        var m1a: List[Float32], var v1a: List[Float32],
        var m1b: List[Float32], var v1b: List[Float32],
        var m2a: List[Float32], var v2a: List[Float32],
        var m2b: List[Float32], var v2b: List[Float32],
    ):
        self.w1a = _f32_to_bf16_list(w1a)
        self.w1b = _f32_to_bf16_list(w1b)
        self.w2a = _f32_to_bf16_list(w2a)
        self.w2b = _f32_to_bf16_list(w2b)
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f
        self.alpha = alpha
        self.scale = (alpha / Float32(rank)) if rank > 0 else Float32(0.0)
        self.m1a = m1a^
        self.v1a = v1a^
        self.m1b = m1b^
        self.v1b = v1b^
        self.m2a = m2a^
        self.v2a = v2a^
        self.m2b = m2b^
        self.v2b = v2b^


# Construct a fresh LoHa adapter with upstream init (use_scalar=False):
#   Upstream intent: w1a~N(0,0.1) w1b~N(0,1) w2a=0 w2b~N(0,1). Identity at init (ΔW=0).
# This port uses centered-uniform * scale (project _randn convention), not true
# Gaussian — w2a=0 zero-leaf and ΔW==0 at init are exact regardless.
# Mirrors loha.rs new_linear_for_training + loha.py:146-154.
def new_loha_adapter(in_f: Int, out_f: Int, rank: Int, alpha: Float32, seed: UInt64) -> LoHaAdapter:
    var w1a = _randn(in_f * rank, seed + 1, 0.1)     # U(-0.05, 0.05)  (intent N(0,0.1))
    var w1b = _randn(rank * out_f, seed + 2, 1.0)    # U(-0.5, 0.5)    (intent N(0,1))
    var w2a = _zeros(in_f * rank)                    # 0  (zero leaf)
    var w2b = _randn(rank * out_f, seed + 3, 1.0)    # U(-0.5, 0.5)    (intent N(0,1))
    return LoHaAdapter(
        w1a^, w1b^, w2a^, w2b^,
        rank, in_f, out_f, alpha,
        _zeros(in_f * rank), _zeros(in_f * rank),    # m1a, v1a
        _zeros(rank * out_f), _zeros(rank * out_f),  # m1b, v1b
        _zeros(in_f * rank), _zeros(in_f * rank),    # m2a, v2a
        _zeros(rank * out_f), _zeros(rank * out_f),  # m2b, v2b
    )


# ── the diff weight ΔW = (w1a@w1b) ⊙ (w2a@w2b) * scale  →  [in,out] flat ──────
# Mirrors loha.rs get_diff_weight (Linear) + loha.py HadaWeight.forward.
def loha_diff_weight(lo: LoHaAdapter) raises -> List[Float32]:
    if lo.scale == Float32(0.0):
        return _zeros(lo.in_f * lo.out_f)
    var w1a = _bf16_to_f32_list(lo.w1a)
    var w1b = _bf16_to_f32_list(lo.w1b)
    var w2a = _bf16_to_f32_list(lo.w2a)
    var w2b = _bf16_to_f32_list(lo.w2b)
    var w1 = _matmul(w1a, lo.in_f, lo.rank, w1b, lo.rank, lo.out_f)  # [in,out]
    var w2 = _matmul(w2a, lo.in_f, lo.rank, w2b, lo.rank, lo.out_f)  # [in,out]
    var diff = _hadamard(w1, w2)
    _scale_inplace(diff, lo.scale)
    return diff^


# ── forward delta: y_delta = x @ ΔW   x:[M,in] → [M,out] ─────────────────────
# Returns the adapter contribution to ADD to the base linear output.
def loha_forward(x_h: List[Float32], lo: LoHaAdapter, M: Int) raises -> List[Float32]:
    var diff = loha_diff_weight(lo)                       # [in,out]
    return _matmul(x_h, M, lo.in_f, diff, lo.in_f, lo.out_f)  # [M,out]


# ── the 4 factor grads + d_x — ALL FOUR factors (the critical correctness arm) ─
struct LoHaGrads(Copyable, Movable):
    var d_w1a: List[Float32]   # [in,rank]
    var d_w1b: List[Float32]   # [rank,out]
    var d_w2a: List[Float32]   # [in,rank]
    var d_w2b: List[Float32]   # [rank,out]
    var d_x: List[Float32]     # [M,in]

    def __init__(
        out self, var d_w1a: List[Float32], var d_w1b: List[Float32],
        var d_w2a: List[Float32], var d_w2b: List[Float32], var d_x: List[Float32],
    ):
        self.d_w1a = d_w1a^
        self.d_w1b = d_w1b^
        self.d_w2a = d_w2a^
        self.d_w2b = d_w2b^
        self.d_x = d_x^


# Backward through y_delta = x @ ΔW with grad d_y [M,out].
# d_ΔW = xᵀ @ d_y ; d_x = d_y @ ΔWᵀ ; then HadaWeight.backward over d_ΔW.
def loha_backward(d_y_h: List[Float32], x_h: List[Float32], lo: LoHaAdapter, M: Int) raises -> LoHaGrads:
    var IN = lo.in_f
    var OUT = lo.out_f
    var R = lo.rank

    # d_ΔW = xᵀ @ d_y    [in,out]
    var x_t = _transpose(x_h, M, IN)                          # [in,M]
    var d_diff = _matmul(x_t, IN, M, d_y_h, M, OUT)           # [in,out]

    # d_x = d_y @ ΔWᵀ    [M,in]
    var diff = loha_diff_weight(lo)                           # [in,out]
    var diff_t = _transpose(diff, IN, OUT)                    # [out,in]
    var d_x = _matmul(d_y_h, M, OUT, diff_t, OUT, IN)         # [M,in]

    # HadaWeight.backward (loha.py:18-30), grad_out := d_diff, scaled by `scale`.
    var g = d_diff.copy()
    _scale_inplace(g, lo.scale)                               # g = d_diff * scale

    var w1a = _bf16_to_f32_list(lo.w1a)
    var w1b = _bf16_to_f32_list(lo.w1b)
    var w2a = _bf16_to_f32_list(lo.w2a)
    var w2b = _bf16_to_f32_list(lo.w2b)
    var w1 = _matmul(w1a, IN, R, w1b, R, OUT)                 # [in,out]
    var w2 = _matmul(w2a, IN, R, w2b, R, OUT)                 # [in,out]

    # --- factor pair 1: temp = g ⊙ (w2a@w2b) ---
    var t1 = _hadamard(g, w2)                                 # [in,out]
    var w1b_t = _transpose(w1b, R, OUT)                       # [out,rank]
    var d_w1a = _matmul(t1, IN, OUT, w1b_t, OUT, R)           # [in,rank]
    var w1a_t = _transpose(w1a, IN, R)                        # [rank,in]
    var d_w1b = _matmul(w1a_t, R, IN, t1, IN, OUT)            # [rank,out]

    # --- factor pair 2: temp = g ⊙ (w1a@w1b) ---
    var t2 = _hadamard(g, w1)                                 # [in,out]
    var w2b_t = _transpose(w2b, R, OUT)                       # [out,rank]
    var d_w2a = _matmul(t2, IN, OUT, w2b_t, OUT, R)           # [in,rank]
    var w2a_t = _transpose(w2a, IN, R)                        # [rank,in]
    var d_w2b = _matmul(w2a_t, R, IN, t2, IN, OUT)            # [rank,out]

    return LoHaGrads(d_w1a^, d_w1b^, d_w2a^, d_w2b^, d_x^)


# ── AdamW one step over all 4 factors (mirrors train_step._adamw_host_list) ──
def _adamw_host_list(
    mut p: List[BFloat16], g: List[Float32],
    mut m: List[Float32], mut v: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
) raises:
    var n = len(p)
    if len(g) != n or len(m) != n or len(v) != n:
        raise Error("loha _adamw_host_list: param/grad/m/v len mismatch")
    if t < 1:
        raise Error("loha _adamw_host_list: t must be >= 1")
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    for i in range(n):
        var gv = g[i]
        var mi = beta1 * m[i] + (Float32(1.0) - beta1) * gv
        var vi = beta2 * v[i] + (Float32(1.0) - beta2) * gv * gv
        m[i] = mi
        v[i] = vi
        var m_hat = mi / bc1
        var v_hat = vi / bc2
        var pv = p[i].cast[DType.float32]()
        if weight_decay > 0.0:
            pv = pv * (Float32(1.0) - lr * weight_decay)
        pv = pv - lr * m_hat / (sqrt(v_hat) + eps)
        p[i] = BFloat16(pv)


def loha_adamw(
    mut lo: LoHaAdapter, g: LoHaGrads, t: Int, lr: Float32,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    _adamw_host_list(lo.w1a, g.d_w1a, lo.m1a, lo.v1a, t, lr, beta1, beta2, eps, weight_decay)
    _adamw_host_list(lo.w1b, g.d_w1b, lo.m1b, lo.v1b, t, lr, beta1, beta2, eps, weight_decay)
    _adamw_host_list(lo.w2a, g.d_w2a, lo.m2a, lo.v2a, t, lr, beta1, beta2, eps, weight_decay)
    _adamw_host_list(lo.w2b, g.d_w2b, lo.m2b, lo.v2b, t, lr, beta1, beta2, eps, weight_decay)
