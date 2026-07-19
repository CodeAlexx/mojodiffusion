# training/tucker_conv_adapter.mojo — LyCORIS Tucker-decomposed conv2d adapter.
#
# Ports eri-lycoris/lycoris-rs/src/algorithms/locon.rs (the USE_TUCKER conv path:
# LoConModule::new_conv2d use_tucker=true / forward-with-mid) +
# eri-lycoris/lycoris-rs/src/ops/tucker.rs (rebuild_conv_tucker). A conv2d kernel
# is Tucker-2 decomposed into three factors:
#       down : 1x1 conv  (Cin → rank)        [1,1,Cin,rank]   stride 1, pad 0
#       core : KhxKw conv (rank → rank)       [Kh,Kw,rank,rank] the base stride/pad
#       up   : 1x1 conv  (rank → Cout)        [1,1,rank,Cout]  stride 1, pad 0
#       Δy   = up(core(down(x))) * scale       scale = alpha / rank
# The 1x1 down/up are pointwise channel mixes (they do NOT change spatial size);
# the core carries the spatial kernel + the layer's stride/padding. This matches
# locon.rs forward (down 1x1 → mid KhxKw → up 1x1, all stride1/pad0 EXCEPT we
# carry the base stride/pad on the CORE, which is where the spatial conv lives —
# locon.rs hard-codes (1,1)/(0,0) on every factor because it composes against an
# already-strided base; for a STANDALONE adapter the stride/pad must live on the
# spatial core so the adapter output matches the base conv's output grid).
#
# ── SCOPE HONESTY (per task brief) ───────────────────────────────────────────
# Conv adapters are PRIMITIVE-ONLY for Klein / Z-Image (no trained conv2d layers).
# Built for parity-completeness with the LyCORIS Tucker path; NOT integrated into
# the Klein/Z-Image stack. NOTE (T2.F skeptic, 2026-06-11): there is NO
# adapter_algo id for Tucker — io/train_config_reader.mojo rejects
# "locon_tucker" (only lora|full|loha|dora|lokr|oft|boft exist); primitive +
# save module only. Torch-lycoris parity + ecosystem-load gates:
# training/tests/lycoris_family_parity.mojo + lycoris_family_load_check.py.
#
# ── Layout (host F32, row-major) — MATCHES ops/conv2d_backward.mojo NHWC/RSCF ──
#   x      NHWC  [N, Hi, Wi, Cin]
#   down   RSCF  [1, 1, Cin, rank]   (≡ pointwise [Cin,rank])
#   core   RSCF  [Kh, Kw, rank, rank]
#   up     RSCF  [1, 1, rank, Cout]  (≡ pointwise [rank,Cout])
#   a = down(x)  NHWC  [N, Hi, Wi, rank]         (1x1, same spatial size)
#   m = core(a)  NHWC  [N, Ho, Wo, rank]         (spatial conv, stride/pad)
#   y = up(m)*sc NHWC  [N, Ho, Wo, Cout]         (1x1)
#
# ── Init (locon.rs new_conv2d use_tucker) ────────────────────────────────────
#   down ~ N(0,1), core ~ N(0,1) (project centered-uniform _randn approximation —
#   AGENT-DEFAULT), up = 0 → Δy == 0 at init (zero up leg).
#
# ── Backward (down, core, up) ─────────────────────────────────────────────────
# d_y:[N,Ho,Wo,Cout]; d_y_s = d_y * scale.
#   up 1x1 (over m, per spatial pos): m:[N,Ho,Wo,rank]
#     d_up[0,0,r,co] = Σ_{n,oh,ow} m[...,r] * d_y_s[...,co]
#     d_m[...,r]     = Σ_co d_y_s[...,co] * up[0,0,r,co]
#   core spatial conv: a:[N,Hi,Wi,rank] → m:[N,Ho,Wo,rank]
#     d_core[kh,kw,ri,ro] = Σ_{n,oh,ow} d_m[n,oh,ow,ro]
#                             * a[n, oh*sh-ph+kh, ow*sw-pw+kw, ri]   (skip OOB)
#     d_a[n,ih,iw,ri]     = Σ_{ro,kh,kw} d_m[n,oh,ow,ro] * core[kh,kw,ri,ro]
#                             for (oh,ow) s.t. ih=oh*sh-ph+kh, iw=ow*sw-pw+kw.
#   down 1x1 (over x, per spatial pos): x:[N,Hi,Wi,Cin] → a:[N,Hi,Wi,rank]
#     d_down[0,0,ci,r] = Σ_{n,ih,iw} x[...,ci] * d_a[...,r]
#     d_x[...,ci]      = Σ_r d_a[...,r] * down[0,0,ci,r]
#
# Host F32 master throughout. Open-coded so the FD gate needs no GPU. MIRRORS
# locon_conv_adapter.mojo. Mojo 0.26.x: `def` not `fn`; multi-return via Movable.

from std.collections import List
from std.math import sqrt


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


def tucker_out_h(Hi: Int, Kh: Int, sh: Int, ph: Int) -> Int:
    return (Hi + 2 * ph - Kh) // sh + 1


def tucker_out_w(Wi: Int, Kw: Int, sw: Int, pw: Int) -> Int:
    return (Wi + 2 * pw - Kw) // sw + 1


# ── the Tucker conv adapter (down 1x1 + core KhxKw + up 1x1) + AdamW ──────────
struct TuckerConvAdapter(Copyable, Movable):
    var down: List[Float32]    # [1,1,Cin,rank]
    var core: List[Float32]    # [Kh,Kw,rank,rank]
    var up: List[Float32]      # [1,1,rank,Cout]
    var cin: Int
    var cout: Int
    var kh: Int
    var kw: Int
    var rank: Int
    var stride_h: Int
    var stride_w: Int
    var pad_h: Int
    var pad_w: Int
    var alpha: Float32
    var scale: Float32         # alpha / rank
    var m_down: List[Float32]
    var v_down: List[Float32]
    var m_core: List[Float32]
    var v_core: List[Float32]
    var m_up: List[Float32]
    var v_up: List[Float32]

    def __init__(
        out self,
        var down: List[Float32], var core: List[Float32], var up: List[Float32],
        cin: Int, cout: Int, kh: Int, kw: Int, rank: Int,
        stride_h: Int, stride_w: Int, pad_h: Int, pad_w: Int, alpha: Float32,
        var m_down: List[Float32], var v_down: List[Float32],
        var m_core: List[Float32], var v_core: List[Float32],
        var m_up: List[Float32], var v_up: List[Float32],
    ):
        self.down = down^
        self.core = core^
        self.up = up^
        self.cin = cin
        self.cout = cout
        self.kh = kh
        self.kw = kw
        self.rank = rank
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.alpha = alpha
        self.scale = (alpha / Float32(rank)) if rank > 0 else Float32(1.0)
        self.m_down = m_down^
        self.v_down = v_down^
        self.m_core = m_core^
        self.v_core = v_core^
        self.m_up = m_up^
        self.v_up = v_up^


# Construct a fresh Tucker conv adapter. down/core ~ centered-uniform, up = 0
# → Δy==0. AGENT-DEFAULT (flagged): the CORE carries the base stride/pad (the
# spatial kernel); down/up are 1x1 stride1/pad0. Core kernel size = caller's
# (kh,kw). down/core use _randn (not true N(0,1)); the zero up leg is exact.
def new_tucker_conv_adapter(
    cin: Int, cout: Int, kh: Int, kw: Int, rank: Int, alpha: Float32,
    stride_h: Int, stride_w: Int, pad_h: Int, pad_w: Int, seed: UInt64,
) raises -> TuckerConvAdapter:
    if cin == 0 or cout == 0:
        raise Error("new_tucker_conv_adapter: cin/cout must be > 0")
    if rank == 0:
        raise Error("new_tucker_conv_adapter: rank must be > 0 for fresh construction")
    if kh == 0 or kw == 0:
        raise Error("new_tucker_conv_adapter: kernel dims must be > 0")
    var down = _randn(cin * rank, seed + 1, 0.1)
    var core = _randn(kh * kw * rank * rank, seed + 2, 0.1)
    var up = _zeros(rank * cout)   # zero up → Δy=0 at init
    return TuckerConvAdapter(
        down^, core^, up^, cin, cout, kh, kw, rank,
        stride_h, stride_w, pad_h, pad_w, alpha,
        _zeros(cin * rank), _zeros(cin * rank),                         # m/v down
        _zeros(kh * kw * rank * rank), _zeros(kh * kw * rank * rank),   # m/v core
        _zeros(rank * cout), _zeros(rank * cout),                       # m/v up
    )


# ── down 1x1: x[N,Hi,Wi,Cin] → a[N,Hi,Wi,rank] (pointwise) ───────────────────
def tucker_down_forward(
    x: List[Float32], lo: TuckerConvAdapter, N: Int, Hi: Int, Wi: Int,
) raises -> List[Float32]:
    var Cin = lo.cin; var R = lo.rank
    var rows = N * Hi * Wi
    if len(x) != rows * Cin:
        raise Error("tucker_down_forward: x numel mismatch")
    var a = _zeros(rows * R)
    for p in range(rows):
        var xbase = p * Cin
        var abase = p * R
        for ci in range(Cin):
            var xv = x[xbase + ci]
            if xv == Float32(0.0):
                continue
            var drow = ci * R
            for r in range(R):
                a[abase + r] = a[abase + r] + xv * lo.down[drow + r]
    return a^


# ── core spatial conv: a[N,Hi,Wi,rank] → m[N,Ho,Wo,rank] ─────────────────────
def tucker_core_forward(
    a: List[Float32], lo: TuckerConvAdapter, N: Int, Hi: Int, Wi: Int,
) raises -> List[Float32]:
    var R = lo.rank; var Kh = lo.kh; var Kw = lo.kw
    var sh = lo.stride_h; var sw = lo.stride_w; var ph = lo.pad_h; var pw = lo.pad_w
    var Ho = tucker_out_h(Hi, Kh, sh, ph)
    var Wo = tucker_out_w(Wi, Kw, sw, pw)
    var m = _zeros(N * Ho * Wo * R)
    for n in range(N):
        for oh in range(Ho):
            for ow in range(Wo):
                var mbase = (((n * Ho + oh) * Wo + ow) * R)
                for kh in range(Kh):
                    var ih = oh * sh - ph + kh
                    if ih < 0 or ih >= Hi:
                        continue
                    for kw in range(Kw):
                        var iw = ow * sw - pw + kw
                        if iw < 0 or iw >= Wi:
                            continue
                        var abase = (((n * Hi + ih) * Wi + iw) * R)
                        var cbase = (((kh * Kw + kw) * R) * R)
                        for ri in range(R):
                            var av = a[abase + ri]
                            if av == Float32(0.0):
                                continue
                            var crow = cbase + ri * R
                            for ro in range(R):
                                m[mbase + ro] = m[mbase + ro] + av * lo.core[crow + ro]
    return m^


# ── up 1x1: m[N,Ho,Wo,rank] → y[N,Ho,Wo,Cout]*scale ──────────────────────────
def tucker_up_forward(
    m: List[Float32], lo: TuckerConvAdapter, N: Int, Ho: Int, Wo: Int,
) raises -> List[Float32]:
    var R = lo.rank; var Cout = lo.cout
    var rows = N * Ho * Wo
    if len(m) != rows * R:
        raise Error("tucker_up_forward: m numel mismatch")
    var y = _zeros(rows * Cout)
    for p in range(rows):
        var mbase = p * R
        var ybase = p * Cout
        for r in range(R):
            var mv = m[mbase + r]
            if mv == Float32(0.0):
                continue
            var ubase = r * Cout
            for co in range(Cout):
                y[ybase + co] = y[ybase + co] + mv * lo.up[ubase + co]
    for i in range(len(y)):
        y[i] = y[i] * lo.scale
    return y^


# ── full forward: Δy = up(core(down(x))) * scale ─────────────────────────────
def tucker_forward(
    x: List[Float32], lo: TuckerConvAdapter, N: Int, Hi: Int, Wi: Int,
) raises -> List[Float32]:
    var Ho = tucker_out_h(Hi, lo.kh, lo.stride_h, lo.pad_h)
    var Wo = tucker_out_w(Wi, lo.kw, lo.stride_w, lo.pad_w)
    var a = tucker_down_forward(x, lo, N, Hi, Wi)
    var m = tucker_core_forward(a, lo, N, Hi, Wi)
    return tucker_up_forward(m, lo, N, Ho, Wo)


# ── grads of down + core + up + d_x ──────────────────────────────────────────
struct TuckerGrads(Copyable, Movable):
    var d_down: List[Float32]   # [1,1,Cin,rank]
    var d_core: List[Float32]   # [Kh,Kw,rank,rank]
    var d_up: List[Float32]     # [1,1,rank,Cout]
    var d_x: List[Float32]      # [N,Hi,Wi,Cin]

    def __init__(
        out self, var d_down: List[Float32], var d_core: List[Float32],
        var d_up: List[Float32], var d_x: List[Float32],
    ):
        self.d_down = d_down^
        self.d_core = d_core^
        self.d_up = d_up^
        self.d_x = d_x^


# Backward through Δy = up(core(down(x)))*scale with d_y:[N,Ho,Wo,Cout].
def tucker_backward(
    d_y: List[Float32], x: List[Float32], lo: TuckerConvAdapter, N: Int, Hi: Int, Wi: Int,
) raises -> TuckerGrads:
    var Cin = lo.cin; var Cout = lo.cout; var R = lo.rank; var Kh = lo.kh; var Kw = lo.kw
    var sh = lo.stride_h; var sw = lo.stride_w; var ph = lo.pad_h; var pw = lo.pad_w
    var Ho = tucker_out_h(Hi, Kh, sh, ph)
    var Wo = tucker_out_w(Wi, Kw, sw, pw)
    if len(d_y) != N * Ho * Wo * Cout:
        raise Error("tucker_backward: d_y numel mismatch")

    # forward intermediates (no scale; scale lives at the output).
    var a = tucker_down_forward(x, lo, N, Hi, Wi)        # [N,Hi,Wi,R]
    var m = tucker_core_forward(a, lo, N, Hi, Wi)        # [N,Ho,Wo,R]

    var d_y_s = d_y.copy()
    for i in range(len(d_y_s)):
        d_y_s[i] = d_y_s[i] * lo.scale

    # ── up grads + d_m (1x1 over m) ──
    var rows_o = N * Ho * Wo
    var d_up = _zeros(R * Cout)
    var d_m = _zeros(rows_o * R)
    for p in range(rows_o):
        var mbase = p * R
        var ybase = p * Cout
        for r in range(R):
            var mv = m[mbase + r]
            var ubase = r * Cout
            var acc = Float32(0.0)
            for co in range(Cout):
                var gy = d_y_s[ybase + co]
                d_up[ubase + co] = d_up[ubase + co] + mv * gy
                acc += gy * lo.up[ubase + co]
            d_m[mbase + r] = acc

    # ── core grads + d_a (spatial conv) ──
    var d_core = _zeros(Kh * Kw * R * R)
    var d_a = _zeros(N * Hi * Wi * R)
    for n in range(N):
        for oh in range(Ho):
            for ow in range(Wo):
                var mbase = (((n * Ho + oh) * Wo + ow) * R)
                for kh in range(Kh):
                    var ih = oh * sh - ph + kh
                    if ih < 0 or ih >= Hi:
                        continue
                    for kw in range(Kw):
                        var iw = ow * sw - pw + kw
                        if iw < 0 or iw >= Wi:
                            continue
                        var abase = (((n * Hi + ih) * Wi + iw) * R)
                        var cbase = (((kh * Kw + kw) * R) * R)
                        for ri in range(R):
                            var av = a[abase + ri]
                            var crow = cbase + ri * R
                            var acc_da = Float32(0.0)
                            for ro in range(R):
                                var dmv = d_m[mbase + ro]
                                d_core[crow + ro] = d_core[crow + ro] + dmv * av
                                acc_da += dmv * lo.core[crow + ro]
                            d_a[abase + ri] = d_a[abase + ri] + acc_da

    # ── down grads + d_x (1x1 over x) ──
    var rows_i = N * Hi * Wi
    var d_down = _zeros(Cin * R)
    var d_x = _zeros(N * Hi * Wi * Cin)
    for p in range(rows_i):
        var xbase = p * Cin
        var abase = p * R
        for ci in range(Cin):
            var xv = x[xbase + ci]
            var drow = ci * R
            var acc_dx = Float32(0.0)
            for r in range(R):
                var dav = d_a[abase + r]
                d_down[drow + r] = d_down[drow + r] + dav * xv
                acc_dx += dav * lo.down[drow + r]
            d_x[xbase + ci] = d_x[xbase + ci] + acc_dx

    return TuckerGrads(d_down^, d_core^, d_up^, d_x^)


# ── AdamW one step over down + core + up ─────────────────────────────────────
def _adamw_host_list(
    mut p: List[Float32], g: List[Float32],
    mut mom: List[Float32], mut vmo: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
) raises:
    var n = len(p)
    if len(g) != n or len(mom) != n or len(vmo) != n:
        raise Error("tucker _adamw_host_list: param/grad/m/v len mismatch")
    if t < 1:
        raise Error("tucker _adamw_host_list: t must be >= 1")
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


def tucker_adamw(
    mut lo: TuckerConvAdapter, g: TuckerGrads, t: Int, lr: Float32,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    _adamw_host_list(lo.down, g.d_down, lo.m_down, lo.v_down, t, lr, beta1, beta2, eps, weight_decay)
    _adamw_host_list(lo.core, g.d_core, lo.m_core, lo.v_core, t, lr, beta1, beta2, eps, weight_decay)
    _adamw_host_list(lo.up, g.d_up, lo.m_up, lo.v_up, t, lr, beta1, beta2, eps, weight_decay)
