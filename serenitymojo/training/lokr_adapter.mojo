# training/lokr_adapter.mojo — LyCORIS LoKr (Kronecker LoRA) adapter.
#
# Ports eri-lycoris/lycoris-rs/src/algorithms/lokr.rs (LoKrModule, Linear path)
# + eri-lycoris/lycoris-rs/src/ops/kronecker.rs (make_kronecker) + upstream
# lycoris/functional/lokr.py diff_weight + EDv2
# crates/eridiffusion-core/src/lycoris.rs LoKr save convention
# (lokr_w1[.weight] | lokr_w1_a/b, lokr_w2[.weight] | lokr_w2_a/b, .alpha).
#
# ── The math (Linear, the only path this wave ships) ──────────────────────────
# The delta weight is a Kronecker product of two factor matrices, scaled:
#       ΔW = kron(W1, W2) * scale                              (lokr.rs:2)
# Dimension split (lokr.rs new_linear): with the upstream factorization()
#       (out_l, out_k) = factorization(out, factor)
#       (in_m,  in_n)  = factorization(in,  factor)
# W1 is [out_l, in_m] and W2 is [out_k, in_n], so
#       kron(W1, W2) : [out_l*out_k, in_m*in_n] = [out, in].
# Layout follows the PEFT/Klein base-weight convention W:[out,in] (a layer
# applies y = x @ ΔWᵀ over the frozen base).  make_kronecker (kronecker.rs:30-82)
# defines, for W1:[m,n], W2:[p,q]:
#       kron[(l*p + k), (a*q + b)] = W1[l,a] * W2[k,b]
# i.e. reshape W1→[m,1,n,1], W2→[1,p,1,q], broadcast-multiply→[m,p,n,q],
# reshape→[m*p, n*q]. We open-code exactly this index map.
#
# ── FACTORED vs FULL W2 (the lokr.rs dead-leaf rule) ──────────────────────────
# lokr.rs:243-256 / functional/lokr.py: W2 is FACTORED (w2a:[out_k,r] @
# w2b:[r,in_n]) iff  rank < max(out_k, in_n) / 2  ; otherwise W2 is FULL
# ([out_k,in_n]) — upstream prints a "force full" warning, we silently take the
# full path (matching the Rust port's "silently take the full path" comment).
# W1 is factored (w1a:[out_l,r] @ w1b:[r,in_m]) only when decompose_both AND
# rank < max(out_l,in_m)/2 (lokr.rs:201). BOTH paths now ship:
#   - W1 FULL   (decompose_both=false, the EDv2 default): single full `w1`.
#   - W1 FACTORED (decompose_both=true && rank<max(out_l,in_m)/2): w1a@w1b.
# Per lokr.rs:205-219, when W1 is factored BOTH legs are kaiming-init (NO zero
# leg on the W1 side) — ΔW==0 at init still relies solely on the W2 zero leg
# (w2 full → zero, or factored → w2b zero). So a W1-factored + W2-factored config
# has FOUR live factors {w1a, w1b, w2a, w2b}, all of which must receive grad.
#
# ── The "factor too large → w2b dead leaf" memory note ────────────────────────
# Project memory feedback_lokr_full_rank_vs_factored: `--rank R --lokr-factor F`
# with F large enough that rank >= max(out_k,in_n)/2 makes W2 FULL (no w2b at
# all). When W2 IS factored but rank is so large that w2b is zero-init AND the
# scale collapses, w2b can be a genuine dead leaf. We RESPECT lokr.rs exactly:
# the factorize decision is `rank < max(out_k,in_n)/2`. When factored, BOTH w2a
# and w2b are live and MUST receive nonzero grad (the gate asserts this after
# w2b is driven off its zero init); when full, only the single `w2` is live.
# The smoke exercises BOTH a factored-W2 config and a full-W2 config.
#
# ── Init (lokr.rs new_linear, use_scalar=false) ───────────────────────────────
# Upstream kaiming-uniform(a=√5) every leaf EXCEPT the canonical zero leg:
#   full W2  → zero ; factored W2 → w2b zero (so initial ΔW=0).
# This port uses the project centered-uniform _randn for the kaiming legs (the
# same approximation the whole Mojo port makes); the zero leg + ΔW==0-at-init
# are exact regardless, which is all the gates depend on.
#
# ── Backward (live factors only) ──────────────────────────────────────────────
# d_y:[M,out] → y = x @ ΔWᵀ.  d_ΔW = d_yᵀ @ x  [out,in] ; d_x = d_y @ ΔW  [M,in].
# g = d_ΔW * scale.  From kron[(l*p+k),(a*q+b)] = W1[l,a]*W2[k,b]:
#   FULL W2:   d_W1[l,a] = Σ_{k,b} g[(l*p+k),(a*q+b)] * W2[k,b]
#              d_W2[k,b] = Σ_{l,a} g[(l*p+k),(a*q+b)] * W1[l,a]
#   FACTORED W2 (W2 = w2a@w2b): d_W2 as above, then d_w2a = d_W2 @ w2bᵀ,
#              d_w2b = w2aᵀ @ d_W2.
# (W1 full only this wave → d_w1 is the single full-W1 grad.)
#
# BF16 trainable storage, F32 internal compute/moments.
# Mojo 0.26.x: `def` not `fn`; multi-return via Movable struct. MIRRORS
# loha_adapter.mojo structure.

from std.collections import List
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from serenitymojo.util.bf16_stochastic_rounding import copy_stochastic_value


# ── factorization(dimension, factor) → (m, n) with m <= n, m*n == dimension ──
# Faithful port of lycoris.functional.general.factorization (lokr.rs:96-141).
def lokr_factorization(dimension: Int, factor: Int) -> Tuple[Int, Int]:
    var factor_pos = factor if factor > 0 else 0
    if factor > 0 and (dimension % factor_pos) == 0:
        var m = factor_pos
        var n = dimension // factor_pos
        if m > n:
            var tmp = m; m = n; n = tmp
        return (m, n)
    var cap = dimension if factor < 0 else factor_pos
    var m = 1
    var n = dimension
    var length = m + n
    while True:
        var new_m = m + 1
        while new_m <= n and dimension % new_m != 0:
            new_m += 1
        if new_m > n:
            break
        var new_n = dimension // new_m
        if new_m + new_n > length or new_m > cap:
            break
        m = new_m
        n = new_n
        length = m + n
        if m >= n:
            break
    if m > n:
        var tmp = m; m = n; n = tmp
    return (m, n)


# ── small host helpers (row-major flat) ──────────────────────────────────────
def _matmul(a: List[Float32], ra: Int, ca: Int, b: List[Float32], rb: Int, cb: Int) raises -> List[Float32]:
    if ca != rb:
        raise Error(String("lokr _matmul: inner dim mismatch ") + String(ca) + " != " + String(rb))
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


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
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


# ── the LoKr adapter (Linear) — W1 full + W2 (full or factored) + AdamW ──────
# W1 : [out_l, in_m]  (full this wave)
# W2 : full [out_k, in_n]  OR  factored w2a:[out_k,r] @ w2b:[r,in_n]
# w2_factored flags which W2 storage is live.
struct LoKrAdapter(Copyable, Movable):
    var w1: List[BFloat16]        # [out_l, in_m]  (empty if W1 factored)
    var w1a: List[BFloat16]       # [out_l, rank]  (empty if W1 full)
    var w1b: List[BFloat16]       # [rank, in_m]   (empty if W1 full)
    var w1_factored: Bool
    var w2: List[BFloat16]        # [out_k, in_n]  (empty if factored)
    var w2a: List[BFloat16]       # [out_k, rank]  (empty if full)
    var w2b: List[BFloat16]       # [rank, in_n]   (empty if full)
    var w2_factored: Bool
    var rank: Int
    var in_f: Int
    var out_f: Int
    var out_l: Int
    var out_k: Int
    var in_m: Int
    var in_n: Int
    var alpha: Float32
    var scale: Float32            # alpha / rank  (1.0 if rank==0)
    # AdamW moments (only the live factors are used; empties carried for full).
    var m_w1: List[Float32]
    var v_w1: List[Float32]
    var m_w1a: List[Float32]
    var v_w1a: List[Float32]
    var m_w1b: List[Float32]
    var v_w1b: List[Float32]
    var m_w2: List[Float32]
    var v_w2: List[Float32]
    var m_w2a: List[Float32]
    var v_w2a: List[Float32]
    var m_w2b: List[Float32]
    var v_w2b: List[Float32]

    def __init__(
        out self,
        var w1: List[Float32],
        var w1a: List[Float32], var w1b: List[Float32], w1_factored: Bool,
        var w2: List[Float32],
        var w2a: List[Float32], var w2b: List[Float32], w2_factored: Bool,
        rank: Int, in_f: Int, out_f: Int,
        out_l: Int, out_k: Int, in_m: Int, in_n: Int, alpha: Float32,
        var m_w1: List[Float32], var v_w1: List[Float32],
        var m_w1a: List[Float32], var v_w1a: List[Float32],
        var m_w1b: List[Float32], var v_w1b: List[Float32],
        var m_w2: List[Float32], var v_w2: List[Float32],
        var m_w2a: List[Float32], var v_w2a: List[Float32],
        var m_w2b: List[Float32], var v_w2b: List[Float32],
    ):
        self.w1 = _f32_to_bf16_list(w1)
        self.w1a = _f32_to_bf16_list(w1a)
        self.w1b = _f32_to_bf16_list(w1b)
        self.w1_factored = w1_factored
        self.w2 = _f32_to_bf16_list(w2)
        self.w2a = _f32_to_bf16_list(w2a)
        self.w2b = _f32_to_bf16_list(w2b)
        self.w2_factored = w2_factored
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f
        self.out_l = out_l
        self.out_k = out_k
        self.in_m = in_m
        self.in_n = in_n
        self.alpha = alpha
        self.scale = (alpha / Float32(rank)) if rank > 0 else Float32(1.0)
        # MEASURED upstream quirk (lycoris/modules/lokr.py:209-211, T2.F-2
        # 2026-06-11): when BOTH W1 and W2 are full (use_w1 and use_w2),
        # upstream forces alpha = lora_dim, i.e. scale = 1 — regardless of the
        # alpha the user passed (and regardless of the alpha in a loaded file).
        # Mirror it exactly so the both-full forward matches upstream.
        if (not w1_factored) and (not w2_factored):
            self.scale = Float32(1.0)
        self.m_w1 = m_w1^
        self.v_w1 = v_w1^
        self.m_w1a = m_w1a^
        self.v_w1a = v_w1a^
        self.m_w1b = m_w1b^
        self.v_w1b = v_w1b^
        self.m_w2 = m_w2^
        self.v_w2 = v_w2^
        self.m_w2a = m_w2a^
        self.v_w2a = v_w2a^
        self.m_w2b = m_w2b^
        self.v_w2b = v_w2b^


# Construct a fresh LoKr adapter for Linear(in,out) with split factor.
# decompose_both (default false → W1 full); when true AND rank<max(out_l,in_m)/2
# AND NOT full_matrix, W1 is factored w1a@w1b (upstream lokr.py:153-162, both
# legs kaiming, NO zero leg on W1).
# W2 is factored iff rank < max(out_k,in_n)/2 AND NOT full_matrix
# (upstream lokr.py:164-172; full_matrix=True is the SimpleTuner shipped
# default and forces BOTH factors full → the forced scale=1 quirk applies).
# Zero leg: full W2 → zero, factored → w2b zero, so ΔW==0 at init (regardless
# of the W1 path). Mirrors upstream LokrModule.__init__ (use_scalar=false).
# The /2 comparisons are the upstream FLOAT semantics `lora_dim < max/2`,
# spelled `2*rank < max` so odd max behaves exactly like upstream (the old
# `rank < max//2` spelling diverged for odd max — T2.G fix 2026-06-11).
# AGENT-DEFAULT (flagged): kaiming legs use the project centered-uniform _randn
# (not true kaiming-uniform(a=√5)); the W2 zero leg + ΔW==0-at-init are exact.
def new_lokr_adapter(
    in_f: Int, out_f: Int, rank: Int, alpha: Float32, factor: Int, seed: UInt64,
    decompose_both: Bool = False,
    full_matrix: Bool = False,
) raises -> LoKrAdapter:
    if in_f == 0 or out_f == 0:
        raise Error("new_lokr_adapter: in/out must be > 0")
    if rank == 0:
        raise Error("new_lokr_adapter: rank must be > 0 for fresh construction")
    var outsplit = lokr_factorization(out_f, factor)
    var insplit = lokr_factorization(in_f, factor)
    var out_l = outsplit[0]
    var out_k = outsplit[1]
    var in_m = insplit[0]
    var in_n = insplit[1]

    # ── W1 path: factored iff decompose_both AND 2*rank < max(out_l,in_m)
    #            AND not full_matrix (upstream lokr.py:153-156) ──
    var max_w1 = out_l if out_l > in_m else in_m
    var factorize_w1 = decompose_both and (2 * rank < max_w1) and (not full_matrix)
    var w1 = List[Float32]()
    var w1a = List[Float32]()
    var w1b = List[Float32]()
    if factorize_w1:
        # BOTH legs kaiming (no zero leg on W1 — lokr.rs:205-219).
        w1a = _randn(out_l * rank, seed + 3, 0.1)
        w1b = _randn(rank * in_m, seed + 4, 0.1)
    else:
        # W1 full [out_l, in_m], kaiming-ish.
        w1 = _randn(out_l * in_m, seed + 1, 0.1)

    # ── W2 path: factored iff 2*rank < max(out_k,in_n) AND not full_matrix ──
    var max_w2 = out_k if out_k > in_n else in_n
    var factorize_w2 = (2 * rank < max_w2) and (not full_matrix)
    var w2 = List[Float32]()
    var w2a = List[Float32]()
    var w2b = List[Float32]()
    if factorize_w2:
        w2a = _randn(out_k * rank, seed + 2, 0.1)   # kaiming leg
        w2b = _zeros(rank * in_n)                   # ZERO leg → ΔW=0 at init
    else:
        w2 = _zeros(out_k * in_n)                   # full W2 zero → ΔW=0 at init

    return LoKrAdapter(
        w1^, w1a^, w1b^, factorize_w1,
        w2^, w2a^, w2b^, factorize_w2,
        rank, in_f, out_f, out_l, out_k, in_m, in_n, alpha,
        _zeros(out_l * in_m), _zeros(out_l * in_m),                       # m/v w1 (full)
        _zeros(out_l * rank), _zeros(out_l * rank),                       # m/v w1a
        _zeros(rank * in_m), _zeros(rank * in_m),                         # m/v w1b
        _zeros(out_k * in_n), _zeros(out_k * in_n),                       # m/v w2 (full)
        _zeros(out_k * rank), _zeros(out_k * rank),                       # m/v w2a
        _zeros(rank * in_n), _zeros(rank * in_n),                         # m/v w2b
    )


# Resolve W2 as a dense [out_k, in_n] matrix (materialize w2a@w2b if factored).
def lokr_resolve_w2(lo: LoKrAdapter) raises -> List[Float32]:
    if lo.w2_factored:
        var w2a = _bf16_to_f32_list(lo.w2a)
        var w2b = _bf16_to_f32_list(lo.w2b)
        return _matmul(w2a, lo.out_k, lo.rank, w2b, lo.rank, lo.in_n)  # [out_k,in_n]
    return _bf16_to_f32_list(lo.w2)


# Resolve W1 as a dense [out_l, in_m] matrix (materialize w1a@w1b if factored).
def lokr_resolve_w1(lo: LoKrAdapter) raises -> List[Float32]:
    if lo.w1_factored:
        var w1a = _bf16_to_f32_list(lo.w1a)
        var w1b = _bf16_to_f32_list(lo.w1b)
        return _matmul(w1a, lo.out_l, lo.rank, w1b, lo.rank, lo.in_m)  # [out_l,in_m]
    return _bf16_to_f32_list(lo.w1)


# ── ΔW = kron(W1, W2) * scale  →  [out,in] flat ──────────────────────────────
# kron[(l*out_k+k),(a*in_n+b)] = W1[l,a] * W2[k,b]   (make_kronecker index map).
def lokr_delta_weight(lo: LoKrAdapter) raises -> List[Float32]:
    if lo.scale == Float32(0.0):
        return _zeros(lo.out_f * lo.in_f)
    var w2d = lokr_resolve_w2(lo)                       # [out_k, in_n]
    var w1d = lokr_resolve_w1(lo)                       # [out_l, in_m]
    var OL = lo.out_l; var OK = lo.out_k; var IM = lo.in_m; var INn = lo.in_n
    var OUT = lo.out_f; var IN = lo.in_f
    var out = List[Float32]()
    for _ in range(OUT * IN):
        out.append(Float32(0.0))
    for l in range(OL):
        for a in range(IM):
            var w1la = w1d[l * IM + a]
            if w1la == Float32(0.0):
                continue
            for k in range(OK):
                var row = (l * OK + k)
                for b in range(INn):
                    var col = (a * INn + b)
                    out[row * IN + col] = w1la * w2d[k * INn + b] * lo.scale
    return out^


# ── forward delta: y = x @ ΔWᵀ   x:[M,in] → [M,out] ──────────────────────────
def lokr_forward(x_h: List[Float32], lo: LoKrAdapter, M: Int) raises -> List[Float32]:
    var dw = lokr_delta_weight(lo)                     # [out,in]
    var dwt = _transpose(dw, lo.out_f, lo.in_f)        # [in,out]
    return _matmul(x_h, M, lo.in_f, dwt, lo.in_f, lo.out_f)  # [M,out]


# ── grads of the live factors + d_x ──────────────────────────────────────────
struct LoKrGrads(Copyable, Movable):
    var d_w1: List[Float32]    # [out_l,in_m]  (W1-full path; empty if factored)
    var d_w1a: List[Float32]   # [out_l,rank]  (W1-factored path; empty if full)
    var d_w1b: List[Float32]   # [rank,in_m]   (W1-factored path; empty if full)
    var d_w2: List[Float32]    # [out_k,in_n]  (W2-full path; empty if factored)
    var d_w2a: List[Float32]   # [out_k,rank]  (W2-factored path; empty if full)
    var d_w2b: List[Float32]   # [rank,in_n]   (W2-factored path; empty if full)
    var d_x: List[Float32]     # [M,in]

    def __init__(
        out self, var d_w1: List[Float32],
        var d_w1a: List[Float32], var d_w1b: List[Float32],
        var d_w2: List[Float32],
        var d_w2a: List[Float32], var d_w2b: List[Float32], var d_x: List[Float32],
    ):
        self.d_w1 = d_w1^
        self.d_w1a = d_w1a^
        self.d_w1b = d_w1b^
        self.d_w2 = d_w2^
        self.d_w2a = d_w2a^
        self.d_w2b = d_w2b^
        self.d_x = d_x^


# Backward through y = x @ ΔWᵀ with d_y:[M,out].
def lokr_backward(d_y_h: List[Float32], x_h: List[Float32], lo: LoKrAdapter, M: Int) raises -> LoKrGrads:
    var IN = lo.in_f
    var OUT = lo.out_f
    var OL = lo.out_l; var OK = lo.out_k; var IM = lo.in_m; var INn = lo.in_n
    var R = lo.rank

    # d_ΔW = d_yᵀ @ x  [out,in] ; d_x = d_y @ ΔW  [M,in].
    var d_y_t = _transpose(d_y_h, M, OUT)                 # [out,M]
    var d_dw = _matmul(d_y_t, OUT, M, x_h, M, IN)         # [out,in]
    var dw = lokr_delta_weight(lo)                        # [out,in]
    var d_x = _matmul(d_y_h, M, OUT, dw, OUT, IN)         # [M,in]

    # g = d_ΔW * scale
    var g = d_dw.copy()
    for i in range(len(g)):
        g[i] = g[i] * lo.scale                            # [out,in]

    var w2d = lokr_resolve_w2(lo)                         # [out_k,in_n]
    var w1d = lokr_resolve_w1(lo)                         # [out_l,in_m]

    # d_W1[l,a] = Σ_{k,b} g[(l*OK+k),(a*INn+b)] * W2[k,b]  (dense W1 grad)
    var d_w1full = _zeros(OL * IM)
    # d_W2[k,b] = Σ_{l,a} g[(l*OK+k),(a*INn+b)] * W1[l,a]  (dense W2 grad)
    var d_w2full = _zeros(OK * INn)
    for l in range(OL):
        for a in range(IM):
            var w1la = w1d[l * IM + a]
            var acc1 = Float32(0.0)
            for k in range(OK):
                var row = l * OK + k
                for b in range(INn):
                    var gv = g[row * IN + (a * INn + b)]
                    acc1 += gv * w2d[k * INn + b]
                    d_w2full[k * INn + b] = d_w2full[k * INn + b] + gv * w1la
            d_w1full[l * IM + a] = acc1

    # ── chain the dense W1 grad to the live W1 factors ──
    var d_w1 = List[Float32]()
    var d_w1a = List[Float32]()
    var d_w1b = List[Float32]()
    if lo.w1_factored:
        # W1 = w1a @ w1b ;  d_w1a = d_W1 @ w1bᵀ ;  d_w1b = w1aᵀ @ d_W1.
        var w1b = _bf16_to_f32_list(lo.w1b)
        var w1a = _bf16_to_f32_list(lo.w1a)
        var w1b_t = _transpose(w1b, R, IM)                # [in_m,rank]
        d_w1a = _matmul(d_w1full, OL, IM, w1b_t, IM, R)   # [out_l,rank]
        var w1a_t = _transpose(w1a, OL, R)                # [rank,out_l]
        d_w1b = _matmul(w1a_t, R, OL, d_w1full, OL, IM)   # [rank,in_m]
    else:
        d_w1 = d_w1full.copy()

    # ── chain the dense W2 grad to the live W2 factors ──
    var d_w2 = List[Float32]()
    var d_w2a = List[Float32]()
    var d_w2b = List[Float32]()
    if lo.w2_factored:
        # W2 = w2a @ w2b ;  d_w2a = d_W2 @ w2bᵀ ;  d_w2b = w2aᵀ @ d_W2.
        var w2b = _bf16_to_f32_list(lo.w2b)
        var w2a = _bf16_to_f32_list(lo.w2a)
        var w2b_t = _transpose(w2b, R, INn)               # [in_n,rank]
        d_w2a = _matmul(d_w2full, OK, INn, w2b_t, INn, R) # [out_k,rank]
        var w2a_t = _transpose(w2a, OK, R)                # [rank,out_k]
        d_w2b = _matmul(w2a_t, R, OK, d_w2full, OK, INn)  # [rank,in_n]
    else:
        d_w2 = d_w2full.copy()

    return LoKrGrads(d_w1^, d_w1a^, d_w1b^, d_w2^, d_w2a^, d_w2b^, d_x^)


# ── AdamW one step over the live factors ─────────────────────────────────────
# BF16 param storage takes a STOCHASTIC-ROUNDING writeback (T2.G fix
# 2026-06-12, measured): with RNE, a parameter at magnitude ~1.0 (the
# perturbed-init lokr_w1 = ones case) has a bf16 half-ULP of ~2e-3, so every
# lr=4e-4 AdamW update rounds straight back — w1 was BIT-FROZEN across a
# 10-step klein run (max|w1-1| == 0.0). The repo's fused OT LoRA-AdamW kernel
# already writes bf16 params through an SR stream for exactly this reason;
# this reuses the canonical util/bf16_stochastic_rounding helper
# (deterministic per (t-derived seed ^ stream, element index)).
def _adamw_host_list(
    mut p: List[BFloat16], g: List[Float32],
    mut mom: List[Float32], mut vmo: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
    sr_stream: UInt64 = UInt64(0),
) raises:
    var n = len(p)
    if len(g) != n or len(mom) != n or len(vmo) != n:
        raise Error("lokr _adamw_host_list: param/grad/m/v len mismatch")
    if t < 1:
        raise Error("lokr _adamw_host_list: t must be >= 1")
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p
    var sr_seed = UInt32((UInt64(t) * 2654435761 + sr_stream * 40503 + 977) & 0xFFFFFFFF)
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
        p[i] = copy_stochastic_value(pv, sr_seed, i)


# ── SimpleTuner --init_lokr_norm: perturbed-normal LoKr init ─────────────────
# Faithful port of SimpleTuner simpletuner/helpers/training/peft_init.py:
#   init_lokr_network_with_perturbed_normal(lycoris, scale):
#       lora.lokr_w1.fill_(1.0)
#       approximate_normal_tensor(lora.org_weight, lora.lokr_w2, scale)
# where approximate_normal_tensor does, IN THIS ORDER (peft_init.py:4-18):
#       t = randn_like(w2)
#       t *= org.norm() / t.norm()
#       t *= org.std() / t.std()          # torch.std = UNBIASED (n-1)
#       t  = t - t.mean() + org.mean()
#       t *= scale
# i.e. ΔW starts as TINY NOISE statistically shaped like the ORIGINAL layer
# weight instead of exact zero. ST runs it AFTER create_lycoris (trainer.py:
# 2757-2761), so it OVERWRITES the default zero-init.
#
# APPLICABILITY (measured upstream): the helper indexes lora.lokr_w1 and
# lora.lokr_w2 directly, so it requires BOTH factors full (use_w1 and use_w2)
# — exactly what ST's shipped full_matrix=true configs produce. On a factored
# module the ST helper raises AttributeError; we mirror that as a loud Error.
#
# RNG: repo Box-Muller normal (noise_stats discipline), deterministic per
# (seed). std/mean/norm reductions in F64.
def lokr_perturbed_normal_init(
    mut lo: LoKrAdapter,
    org_norm: Float64, org_mean: Float64, org_std: Float64,
    scale: Float64, seed: UInt64,
) raises:
    if lo.w1_factored or lo.w2_factored:
        raise Error(
            "lokr_perturbed_normal_init: requires BOTH LoKr factors full "
            + "(full_matrix), exactly like SimpleTuner init_lokr_network_with_"
            + "perturbed_normal (it indexes lokr_w1/lokr_w2 directly)"
        )
    var n = lo.out_k * lo.in_n
    if len(lo.w2) != n:
        raise Error("lokr_perturbed_normal_init: w2 numel mismatch")
    # w1 <- 1.0 everywhere
    for i in range(len(lo.w1)):
        lo.w1[i] = BFloat16(Float32(1.0))
    # t = randn(n)  — Box-Muller on the repo LCG stream
    var t = List[Float64]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(u1f))
        var theta = 6.283185307179586 * u2f
        t.append(r * fcos(theta))
        if i + 1 < n:
            t.append(r * fsin(theta))
        i += 2
    # t *= org_norm / t.norm()
    var ss = Float64(0.0)
    for j in range(n):
        ss += t[j] * t[j]
    var tnorm = sqrt(ss)
    if tnorm <= 0.0:
        raise Error("lokr_perturbed_normal_init: degenerate randn draw")
    var s1 = org_norm / tnorm
    for j in range(n):
        t[j] *= s1
    # t *= org_std / t.std()   (unbiased std, torch default)
    var mean1 = Float64(0.0)
    for j in range(n):
        mean1 += t[j]
    mean1 /= Float64(n)
    var var1 = Float64(0.0)
    for j in range(n):
        var d = t[j] - mean1
        var1 += d * d
    if n > 1:
        var1 /= Float64(n - 1)
    var tstd = sqrt(var1)
    if tstd <= 0.0:
        raise Error("lokr_perturbed_normal_init: degenerate randn std")
    var s2 = org_std / tstd
    for j in range(n):
        t[j] *= s2
    # t = t - t.mean() + org_mean ; t *= scale
    var mean2 = Float64(0.0)
    for j in range(n):
        mean2 += t[j]
    mean2 /= Float64(n)
    for j in range(n):
        lo.w2[j] = BFloat16(Float32((t[j] - mean2 + org_mean) * scale))


def lokr_adamw(
    mut lo: LoKrAdapter, g: LoKrGrads, t: Int, lr: Float32,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    # sr_stream folds the adapter identity (param sizes) + leg id so every leg
    # of every adapter draws an independent deterministic SR stream.
    var ident = UInt64(lo.in_f) * 1315423911 + UInt64(lo.out_f) * 2654435761 + UInt64(lo.rank)
    if lo.w1_factored:
        _adamw_host_list(lo.w1a, g.d_w1a, lo.m_w1a, lo.v_w1a, t, lr, beta1, beta2, eps, weight_decay, ident * 8 + 1)
        _adamw_host_list(lo.w1b, g.d_w1b, lo.m_w1b, lo.v_w1b, t, lr, beta1, beta2, eps, weight_decay, ident * 8 + 2)
    else:
        _adamw_host_list(lo.w1, g.d_w1, lo.m_w1, lo.v_w1, t, lr, beta1, beta2, eps, weight_decay, ident * 8 + 3)
    if lo.w2_factored:
        _adamw_host_list(lo.w2a, g.d_w2a, lo.m_w2a, lo.v_w2a, t, lr, beta1, beta2, eps, weight_decay, ident * 8 + 4)
        _adamw_host_list(lo.w2b, g.d_w2b, lo.m_w2b, lo.v_w2b, t, lr, beta1, beta2, eps, weight_decay, ident * 8 + 5)
    else:
        _adamw_host_list(lo.w2, g.d_w2, lo.m_w2, lo.v_w2, t, lr, beta1, beta2, eps, weight_decay, ident * 8 + 6)
