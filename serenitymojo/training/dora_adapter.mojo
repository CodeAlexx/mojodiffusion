# training/dora_adapter.mojo — DoRA (Weight-Decomposed LoRA) adapter.
#
# Ports eri-lycoris/lycoris-rs/src/dora.rs (apply_weight_decompose / init_magnitude)
# + EDv2 crates/eridiffusion-core/src/lycoris.rs DoRA save convention
# (`.dora_scale`, with `.magnitude_vector` accepted as a load alias) + the
# OneTrainer DoRAModule (LoRAModule.py:473-571) and the DoRA paper §4.3 (NVIDIA
# 2024, "DoRA: Weight-Decomposed Low-Rank Adaptation").
#
# ── The decomposition (Linear; the only path this wave ships) ─────────────────
# DoRA splits the *effective* weight into a learned magnitude × a normalized
# direction:
#       WP        = W_orig + ΔW            ΔW = (B @ A) * scale   (the LoRA delta)
#       WP_dora   = m * (WP / (‖WP‖₂.detach() + eps))            (dora.rs:75-162)
# Weight layout follows the PEFT/Klein base-weight convention W:[out,in]
# (lora_B:[out,rank] @ lora_A:[rank,in] → ΔW:[out,in]; lora_save.mojo:24-25),
# i.e. a layer applies y = x @ Wᵀ.  We use wd_on_out=true (lycoris-upstream
# default, dora.rs:31-38): the L2 norm is taken along the INPUT axis (dim 1 of
# [out,in]), giving a per-output-column magnitude m of shape [out] (logical
# [out,1]).  m is initialized to ‖W_orig‖₂ along that axis (init_magnitude,
# dora.rs:164-217) so WP_dora == W_orig at init (ΔW=0 because lora_up B=0, the
# standard LoRA zero leaf) — exactly OneTrainer/PEFT init behavior.
#
# ── DETACHED norm (paper §4.3 / dora.rs:124-139) ──────────────────────────────
# The L2 norm denominator is DETACHED: the gradient does NOT flow through the
# renormalization. Backward sees den[o] := ‖WP[o,:]‖₂ + eps as a constant. This
# is the single most load-bearing DoRA detail — without the detach the magnitude
# would not train as the paper intends and the grads below would be wrong.
#
# ── Forward delta returned to the caller ──────────────────────────────────────
# A DoRA layer does NOT add the adapter on top of the frozen base output; it
# REPLACES the effective weight with WP_dora (dora.rs:56-60: "Use the returned
# WP_dora as the effective weight ... caller must NOT add base separately").
# So `dora_forward` returns the FULL forward y = x @ WP_doraᵀ, and the
# stack-integration wave (later) is responsible for routing the base linear
# through WP_dora instead of W_orig. This file is a fail-loud PRIMITIVE only.
#
# ── Backward — lora_down (A), lora_up (B), AND magnitude (m) ──────────────────
# Loss L flows back as d_y:[M,out] into y = x @ WP_doraᵀ. With WP_dora:[out,in]:
#   d_WPdora = d_yᵀ @ x                                 [out,in]   (∂(x@Wᵀ)/∂W)
#   d_x      = d_y @ WP_dora                            [M,in]     (residual stream)
# Then the decomposition (den detached, so den is a constant per output row):
#   WP_dora[o,i] = m[o] * WP[o,i] / den[o]
#   d_m[o]  = Σ_i d_WPdora[o,i] * WP[o,i] / den[o]      [out]
#   d_WP[o,i] = d_WPdora[o,i] * m[o] / den[o]           [out,in]
# d_WP is the grad of the reconstructed weight; ΔW = (B@A)*scale so:
#   d_ΔW    = d_WP                                      (W_orig is frozen)
#   g       = d_ΔW * scale
#   d_B     = g @ Aᵀ                                    [out,rank]
#   d_A     = Bᵀ @ g                                    [rank,in]
# Every one of {A, B, m} gets a nonzero grad once B is off zero — the magnitude
# m always gets grad even at init (d_m depends on WP, not on ΔW), and A/B follow
# the plain-LoRA zero-leaf rule (B=0 at init → d_A=0 at step 0, live at step≥1).
#
# BF16 trainable storage for A/B; the magnitude m is F32. 2026-06-11 T2.F-2 FIX:
# m was previously BF16 like the low-rank legs, which broke the identity-at-init
# contract by up to ~0.4% (bf16(‖W‖)/‖W‖ ≠ 1; measured: dora_adapter_smoke
# a-init max|Δ|=3.05e-3 vs the 1e-5 bar). Upstream lycoris EXPLICITLY keeps
# dora_scale in float32 even for bf16 models (locon.py/lokr.py:
# `nn.Parameter(...).float()`) — mirrored here. F32 internal compute/moments.
# All tensors are tiny so host matmul is exact and auditable.
#
# Mojo 0.26.x: `def` not `fn`; move-only Tensor → ArcPointer in collections;
# multi-return via Movable struct. MIRRORS loha_adapter.mojo structure.

from std.collections import List
from std.math import sqrt


# ── small host helpers (row-major flat List[Float32]) ────────────────────────
# A[ra,ca] @ B[rb,cb] with ca == rb → [ra,cb].
def _matmul(a: List[Float32], ra: Int, ca: Int, b: List[Float32], rb: Int, cb: Int) raises -> List[Float32]:
    if ca != rb:
        raise Error(
            String("dora _matmul: inner dim mismatch ") + String(ca) + " != " + String(rb)
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


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    # Same deterministic host randn as loha_adapter / train_step (centered uniform).
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


# Per-output-row L2 norm of W:[out,in] along the INPUT axis (wd_on_out=true).
# Returns [out] = sqrt(Σ_i W[o,i]²). Mirrors dora.rs init_magnitude /
# apply_weight_decompose reduce path (reduce dim 1 of a 2D [out,in] weight).
def _row_l2_norm(w: List[Float32], out_f: Int, in_f: Int) -> List[Float32]:
    var n = List[Float32]()
    for o in range(out_f):
        var s = Float32(0.0)
        for i in range(in_f):
            var v = w[o * in_f + i]
            s += v * v
        n.append(sqrt(s))
    return n^


# ── the DoRA adapter (Linear) — lora_down/up + magnitude + AdamW moments ─────
# Layout (PEFT/Klein): A == lora_down [rank,in], B == lora_up [out,rank],
# magnitude m [out]. The FROZEN base weight W_orig:[out,in] is supplied to the
# forward/backward (the primitive needs it for the column-norm normalization).
struct DoRAAdapter(Copyable, Movable):
    var a: List[BFloat16]         # lora_down [rank,in]
    var b: List[BFloat16]         # lora_up   [out,rank]
    var m: List[Float32]          # magnitude [out]  (F32 — upstream keeps
                                  # dora_scale float32 even in bf16 models)
    var rank: Int
    var in_f: Int
    var out_f: Int
    var alpha: Float32
    var scale: Float32            # alpha / rank
    var eps: Float32              # added to the (detached) norm before divide
    # AdamW first/second moments, one pair per trainable tensor.
    var ma: List[Float32]
    var va: List[Float32]
    var mb: List[Float32]
    var vb: List[Float32]
    var mm: List[Float32]
    var vm: List[Float32]

    def __init__(
        out self,
        var a: List[Float32], var b: List[Float32], var m: List[Float32],
        rank: Int, in_f: Int, out_f: Int, alpha: Float32, eps: Float32,
        var ma: List[Float32], var va: List[Float32],
        var mb: List[Float32], var vb: List[Float32],
        var mm: List[Float32], var vm: List[Float32],
    ):
        self.a = _f32_to_bf16_list(a)
        self.b = _f32_to_bf16_list(b)
        self.m = m^
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f
        self.alpha = alpha
        self.scale = (alpha / Float32(rank)) if rank > 0 else Float32(0.0)
        self.eps = eps
        self.ma = ma^
        self.va = va^
        self.mb = mb^
        self.vb = vb^
        self.mm = mm^
        self.vm = vm^


# Construct a fresh DoRA adapter from the FROZEN base weight W_orig:[out,in].
#   A == lora_down ~ small noise   (intent kaiming/N(0,σ); centered-uniform here)
#   B == lora_up   == 0            (zero leaf → ΔW=0 at init, like plain LoRA)
#   m == ‖W_orig‖₂ along input axis (init_magnitude, dora.rs:164-217)
# At init WP == W_orig and ‖WP[o,:]‖ == m[o], so WP_dora == W_orig exactly
# (identity at init). Mirrors OneTrainer DoRAModule init + dora.rs.
# AGENT-DEFAULT (flagged): magnitude init = bare ‖W_orig‖₂ (eps added only in the
# per-step divide, matching dora.rs init_magnitude eps=0.0 default), and the A
# init distribution is the project centered-uniform (not true kaiming) — the
# identity-at-init and the m=‖W_orig‖ value are exact regardless of A's spread.
def new_dora_adapter(
    w_orig: List[Float32], in_f: Int, out_f: Int, rank: Int, alpha: Float32,
    seed: UInt64, eps: Float32 = Float32(1.0e-7),
) raises -> DoRAAdapter:
    if len(w_orig) != out_f * in_f:
        raise Error(
            String("new_dora_adapter: w_orig numel ") + String(len(w_orig))
            + " != out*in " + String(out_f * in_f)
        )
    var a = _randn(rank * in_f, seed + 1, 0.02)   # lora_down small noise
    var b = _zeros(out_f * rank)                  # lora_up zero leaf
    var m = _row_l2_norm(w_orig, out_f, in_f)     # magnitude = ‖W_orig‖₂ (per out)
    return DoRAAdapter(
        a^, b^, m^,
        rank, in_f, out_f, alpha, eps,
        _zeros(rank * in_f), _zeros(rank * in_f),     # ma, va
        _zeros(out_f * rank), _zeros(out_f * rank),   # mb, vb
        _zeros(out_f), _zeros(out_f),                 # mm, vm
    )


# ── ΔW = (B @ A) * scale  →  [out,in] flat ───────────────────────────────────
def dora_delta_weight(d: DoRAAdapter) raises -> List[Float32]:
    if d.scale == Float32(0.0):
        return _zeros(d.out_f * d.in_f)
    var b = _bf16_to_f32_list(d.b)
    var a = _bf16_to_f32_list(d.a)
    var dw = _matmul(b, d.out_f, d.rank, a, d.rank, d.in_f)  # [out,in]
    for i in range(len(dw)):
        dw[i] = dw[i] * d.scale
    return dw^


# ── the effective DoRA weight WP_dora = m * (WP / (‖WP‖.detach()+eps)) ────────
# Returns (wp_dora:[out,in], den:[out]) — den is the (detached) per-row norm+eps,
# returned so backward can reuse it without recompute. Mirrors apply_weight_decompose.
struct DoRAEff(Copyable, Movable):
    var wp_dora: List[Float32]   # [out,in]  effective weight
    var wp: List[Float32]        # [out,in]  W_orig + ΔW (reconstructed)
    var den: List[Float32]       # [out]     ‖WP[o,:]‖₂ + eps  (detached)

    def __init__(out self, var wp_dora: List[Float32], var wp: List[Float32], var den: List[Float32]):
        self.wp_dora = wp_dora^
        self.wp = wp^
        self.den = den^


def dora_effective_weight(w_orig: List[Float32], d: DoRAAdapter) raises -> DoRAEff:
    if len(w_orig) != d.out_f * d.in_f:
        raise Error("dora_effective_weight: w_orig numel mismatch")
    var dw = dora_delta_weight(d)                          # [out,in]
    var wp = List[Float32]()
    for i in range(len(w_orig)):
        wp.append(w_orig[i] + dw[i])                       # WP = W_orig + ΔW
    var norm = _row_l2_norm(wp, d.out_f, d.in_f)           # ‖WP[o,:]‖₂  (DETACHED)
    var den = List[Float32]()
    for o in range(d.out_f):
        den.append(norm[o] + d.eps)
    var wp_dora = List[Float32]()
    for o in range(d.out_f):
        var mo = d.m[o]
        var deno = den[o]
        for i in range(d.in_f):
            wp_dora.append(mo * wp[o * d.in_f + i] / deno)
    return DoRAEff(wp_dora^, wp^, den^)


# ── forward: y = x @ WP_doraᵀ   x:[M,in] → [M,out] ───────────────────────────
# Returns the FULL forward (DoRA replaces the effective weight; caller must NOT
# add the base linear separately — dora.rs:56-60).
def dora_forward(x_h: List[Float32], w_orig: List[Float32], d: DoRAAdapter, M: Int) raises -> List[Float32]:
    var eff = dora_effective_weight(w_orig, d)
    var wpt = _transpose(eff.wp_dora, d.out_f, d.in_f)     # [in,out]
    return _matmul(x_h, M, d.in_f, wpt, d.in_f, d.out_f)   # [M,out]


# ── the 3 grads (lora_down A, lora_up B, magnitude m) + d_x ──────────────────
struct DoRAGrads(Copyable, Movable):
    var d_a: List[Float32]   # [rank,in]
    var d_b: List[Float32]   # [out,rank]
    var d_m: List[Float32]   # [out]
    var d_x: List[Float32]   # [M,in]

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32],
        var d_m: List[Float32], var d_x: List[Float32],
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_m = d_m^
        self.d_x = d_x^


# Backward through y = x @ WP_doraᵀ with grad d_y:[M,out].
# Uses the DETACHED den (paper §4.3): den is constant in backward.
def dora_backward(d_y_h: List[Float32], x_h: List[Float32], w_orig: List[Float32], d: DoRAAdapter, M: Int) raises -> DoRAGrads:
    var IN = d.in_f
    var OUT = d.out_f
    var R = d.rank

    var eff = dora_effective_weight(w_orig, d)             # wp_dora, wp, den

    # d_WPdora = d_yᵀ @ x    [out,in]   (∂(x@Wᵀ)/∂W with W=[out,in])
    var d_y_t = _transpose(d_y_h, M, OUT)                  # [out,M]
    var d_wpdora = _matmul(d_y_t, OUT, M, x_h, M, IN)      # [out,in]

    # d_x = d_y @ WP_dora    [M,in]
    var d_x = _matmul(d_y_h, M, OUT, eff.wp_dora, OUT, IN) # [M,in]

    # decomposition backward (den detached):
    #   WP_dora[o,i] = m[o] * WP[o,i] / den[o]
    #   d_m[o]   = Σ_i d_WPdora[o,i] * WP[o,i] / den[o]
    #   d_WP[o,i]= d_WPdora[o,i] * m[o] / den[o]
    var d_m = _zeros(OUT)
    var d_wp = List[Float32]()
    for _ in range(OUT * IN):
        d_wp.append(Float32(0.0))
    for o in range(OUT):
        var deno = eff.den[o]
        var mo = d.m[o]
        var acc = Float32(0.0)
        for i in range(IN):
            var idx = o * IN + i
            var g = d_wpdora[idx]
            acc += g * eff.wp[idx] / deno
            d_wp[idx] = g * mo / deno
        d_m[o] = acc

    # ΔW = (B@A)*scale ; W_orig frozen → d_ΔW = d_WP. g = d_ΔW * scale.
    var g = d_wp.copy()
    for i in range(len(g)):
        g[i] = g[i] * d.scale                              # [out,in]

    # d_B = g @ Aᵀ  [out,rank] ; d_A = Bᵀ @ g  [rank,in]
    var a = _bf16_to_f32_list(d.a)
    var b = _bf16_to_f32_list(d.b)
    var a_t = _transpose(a, R, IN)                         # [in,rank]
    var d_b = _matmul(g, OUT, IN, a_t, IN, R)              # [out,rank]
    var b_t = _transpose(b, OUT, R)                        # [rank,out]
    var d_a = _matmul(b_t, R, OUT, g, OUT, IN)             # [rank,in]

    return DoRAGrads(d_a^, d_b^, d_m^, d_x^)


# ── AdamW one step over A, B, m (mirrors loha _adamw_host_list) ──────────────
def _adamw_host_list(
    mut p: List[BFloat16], g: List[Float32],
    mut mom: List[Float32], mut vmo: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
) raises:
    var n = len(p)
    if len(g) != n or len(mom) != n or len(vmo) != n:
        raise Error("dora _adamw_host_list: param/grad/m/v len mismatch")
    if t < 1:
        raise Error("dora _adamw_host_list: t must be >= 1")
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


# F32-param variant for the magnitude (m is F32 storage, see header fix note).
def _adamw_host_list_f32(
    mut p: List[Float32], g: List[Float32],
    mut mom: List[Float32], mut vmo: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
) raises:
    var n = len(p)
    if len(g) != n or len(mom) != n or len(vmo) != n:
        raise Error("dora _adamw_host_list_f32: param/grad/m/v len mismatch")
    if t < 1:
        raise Error("dora _adamw_host_list_f32: t must be >= 1")
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
        var pv = p[i]
        if weight_decay > 0.0:
            pv = pv * (Float32(1.0) - lr * weight_decay)
        pv = pv - lr * m_hat / (sqrt(v_hat) + eps)
        p[i] = pv


def dora_adamw(
    mut d: DoRAAdapter, g: DoRAGrads, t: Int, lr: Float32,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    _adamw_host_list(d.a, g.d_a, d.ma, d.va, t, lr, beta1, beta2, eps, weight_decay)
    _adamw_host_list(d.b, g.d_b, d.mb, d.vb, t, lr, beta1, beta2, eps, weight_decay)
    # Magnitude m: no weight decay (a norm scalar; OT/PEFT do not decay it).
    _adamw_host_list_f32(d.m, g.d_m, d.mm, d.vm, t, lr, beta1, beta2, eps, Float32(0.0))
