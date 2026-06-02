# training/locon_conv_adapter.mojo — LyCORIS LoCon (LoRA-for-conv2d) adapter.
#
# Ports eri-lycoris/lycoris-rs/src/algorithms/locon.rs (the CONV path, no Tucker:
# LoConModule::new_conv2d / forward / get_diff_weight) + EDv2 LoCon conv. A
# conv2d layer gets a low-rank decomposition:
#       down : conv2d(Cin → rank, kernel KhxKw, the base layer's stride/pad)
#       up   : conv2d(rank → Cout, 1x1, stride 1, pad 0)
#       Δy   = up(down(x)) * scale          scale = alpha / rank
# The delta is applied to the conv OUTPUT (locon.rs forward: down → [mid] → up),
# NOT materialized as a single dense kernel for the forward. (get_diff_weight in
# locon.rs DOES build a dense kernel via the batched-matmul-per-spatial-position
# path; we materialize that only for the SAVE convention / oracle, not the hot
# forward.)
#
# ── SCOPE HONESTY (per task brief) ───────────────────────────────────────────
# Conv adapters are PRIMITIVE-ONLY by nature for Klein / Z-Image: those DiTs have
# NO trained conv2d layers (all-attention + linear). This file is built for
# parity-completeness with the LyCORIS conv path and is NOT integrated into the
# Klein/Z-Image training stack. adapter_algo "locon" (7) is opt-in and the
# trainer dispatch fails loud for it (default-off rule).
#
# ── Layout (host F32, row-major) — MATCHES ops/conv2d_backward.mojo EXACTLY ────
#   x      NHWC  [N, Hi, Wi, Cin]
#   down   RSCF  [Kh, Kw, Cin, rank]
#   up     RSCF  [1, 1, rank, Cout]   (a 1x1 conv ≡ a pointwise [rank,Cout] mix)
#   h=down(x)   NHWC  [N, Ho, Wo, rank]
#   y=up(h)*sc  NHWC  [N, Ho, Wo, Cout]
# Forward cross-correlation (what F.conv2d / the SDK kernel computes):
#   h[n,oh,ow,r] = Σ_{kh,kw,ci} x[n, oh*sh-ph+kh, ow*sw-pw+kw, ci] * down[kh,kw,ci,r]
#   y[n,oh,ow,co] = scale * Σ_r h[n,oh,ow,r] * up[0,0,r,co]
# OOB input reads are 0 (implicit zero padding) — same convention as the
# conv2d_backward primitive (kernels f32 interior).
#
# ── Init (locon.rs new_conv2d, spatial path) ─────────────────────────────────
#   down ~ N(0,1) (we use the project centered-uniform _randn approximation —
#   AGENT-DEFAULT, same as every other Mojo adapter), up = 0 → Δy == 0 at init.
#
# ── Backward (down, up) ──────────────────────────────────────────────────────
# Given d_y:[N,Ho,Wo,Cout]. Let d_y_s = d_y * scale (pull the scalar out).
#   up is 1x1: y = h @ up   (over the rank axis, per spatial position)
#     d_up[0,0,r,co] = Σ_{n,oh,ow} h[n,oh,ow,r] * d_y_s[n,oh,ow,co]
#     d_h[n,oh,ow,r] = Σ_co d_y_s[n,oh,ow,co] * up[0,0,r,co]
#   down is the spatial conv: d_down / d_x derived from d_h EXACTLY as the conv
#     backward primitive does (conv2d_backward.mojo d_w / d_x index maps), with
#     "Cout"→rank and grad_y→d_h:
#     d_down[kh,kw,ci,r] = Σ_{n,oh,ow} d_h[n,oh,ow,r]
#                            * x[n, oh*sh-ph+kh, ow*sw-pw+kw, ci]   (skip OOB)
#     d_x[n,ih,iw,ci]    = Σ_{r,kh,kw} d_h[n,oh,ow,r] * down[kh,kw,ci,r]
#                            for every (oh,ow) s.t. ih=oh*sh-ph+kh, iw=ow*sw-pw+kw.
#
# Host F32 master throughout (training masters are F32 per MOJO_CONVENTIONS §3).
# Open-coded so the FD gate needs no GPU. MIRRORS lokr_adapter.mojo structure.
# Mojo 0.26.x: `def` not `fn`; multi-return via Movable struct.

from std.collections import List
from std.math import sqrt


# ── centered-uniform "randn" (same approximation the whole Mojo port uses) ────
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


# Ho/Wo with dilation=1 (matches ops/conv2d_backward.mojo).
def locon_out_h(Hi: Int, Kh: Int, sh: Int, ph: Int) -> Int:
    return (Hi + 2 * ph - Kh) // sh + 1


def locon_out_w(Wi: Int, Kw: Int, sw: Int, pw: Int) -> Int:
    return (Wi + 2 * pw - Kw) // sw + 1


# ── the LoCon conv adapter (down spatial conv + up 1x1) + AdamW ───────────────
# down : [Kh, Kw, Cin, rank]   up : [1, 1, rank, Cout]
struct LoConConvAdapter(Copyable, Movable):
    var down: List[Float32]    # [Kh,Kw,Cin,rank]
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
    # AdamW moments.
    var m_down: List[Float32]
    var v_down: List[Float32]
    var m_up: List[Float32]
    var v_up: List[Float32]

    def __init__(
        out self,
        var down: List[Float32], var up: List[Float32],
        cin: Int, cout: Int, kh: Int, kw: Int, rank: Int,
        stride_h: Int, stride_w: Int, pad_h: Int, pad_w: Int, alpha: Float32,
        var m_down: List[Float32], var v_down: List[Float32],
        var m_up: List[Float32], var v_up: List[Float32],
    ):
        self.down = down^
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
        self.m_up = m_up^
        self.v_up = v_up^


# Construct a fresh LoCon conv adapter. down ~ centered-uniform, up = 0 → Δy==0.
# AGENT-DEFAULT (flagged): stride/pad are caller-supplied (the base conv layer's
# stride/pad); the 1x1 up conv is always stride 1, pad 0 (locon.rs hard-codes
# (1,1)/(0,0) for the up/mid 1x1 convs). down uses the project _randn (not true
# N(0,1)); the zero up leg + Δy==0-at-init are exact regardless.
def new_locon_conv_adapter(
    cin: Int, cout: Int, kh: Int, kw: Int, rank: Int, alpha: Float32,
    stride_h: Int, stride_w: Int, pad_h: Int, pad_w: Int, seed: UInt64,
) raises -> LoConConvAdapter:
    if cin == 0 or cout == 0:
        raise Error("new_locon_conv_adapter: cin/cout must be > 0")
    if rank == 0:
        raise Error("new_locon_conv_adapter: rank must be > 0 for fresh construction")
    if kh == 0 or kw == 0:
        raise Error("new_locon_conv_adapter: kernel dims must be > 0")
    var down = _randn(kh * kw * cin * rank, seed + 1, 0.1)
    var up = _zeros(rank * cout)   # 1x1 up zero → Δy=0 at init
    return LoConConvAdapter(
        down^, up^, cin, cout, kh, kw, rank,
        stride_h, stride_w, pad_h, pad_w, alpha,
        _zeros(kh * kw * cin * rank), _zeros(kh * kw * cin * rank),  # m/v down
        _zeros(rank * cout), _zeros(rank * cout),                    # m/v up
    )


# ── down conv: x[N,Hi,Wi,Cin] → h[N,Ho,Wo,rank] (cross-correlation) ──────────
def locon_down_forward(
    x: List[Float32], lo: LoConConvAdapter, N: Int, Hi: Int, Wi: Int,
) raises -> List[Float32]:
    var Cin = lo.cin; var R = lo.rank; var Kh = lo.kh; var Kw = lo.kw
    var sh = lo.stride_h; var sw = lo.stride_w; var ph = lo.pad_h; var pw = lo.pad_w
    var Ho = locon_out_h(Hi, Kh, sh, ph)
    var Wo = locon_out_w(Wi, Kw, sw, pw)
    if len(x) != N * Hi * Wi * Cin:
        raise Error("locon_down_forward: x numel mismatch")
    var h = _zeros(N * Ho * Wo * R)
    for n in range(N):
        for oh in range(Ho):
            for ow in range(Wo):
                var hbase = (((n * Ho + oh) * Wo + ow) * R)
                for kh in range(Kh):
                    var ih = oh * sh - ph + kh
                    if ih < 0 or ih >= Hi:
                        continue
                    for kw in range(Kw):
                        var iw = ow * sw - pw + kw
                        if iw < 0 or iw >= Wi:
                            continue
                        var xbase = (((n * Hi + ih) * Wi + iw) * Cin)
                        var wbase = (((kh * Kw + kw) * Cin) * R)
                        for ci in range(Cin):
                            var xv = x[xbase + ci]
                            if xv == Float32(0.0):
                                continue
                            var wrow = wbase + ci * R
                            for r in range(R):
                                h[hbase + r] = h[hbase + r] + xv * lo.down[wrow + r]
    return h^


# ── up 1x1: h[N,Ho,Wo,rank] → y[N,Ho,Wo,Cout]*scale ──────────────────────────
def locon_up_forward(
    h: List[Float32], lo: LoConConvAdapter, N: Int, Ho: Int, Wo: Int,
) raises -> List[Float32]:
    var R = lo.rank; var Cout = lo.cout
    var rows = N * Ho * Wo
    if len(h) != rows * R:
        raise Error("locon_up_forward: h numel mismatch")
    var y = _zeros(rows * Cout)
    for p in range(rows):
        var hbase = p * R
        var ybase = p * Cout
        for r in range(R):
            var hv = h[hbase + r]
            if hv == Float32(0.0):
                continue
            var ubase = r * Cout
            for co in range(Cout):
                y[ybase + co] = y[ybase + co] + hv * lo.up[ubase + co]
    for i in range(len(y)):
        y[i] = y[i] * lo.scale
    return y^


# ── full forward: Δy = up(down(x)) * scale ───────────────────────────────────
def locon_forward(
    x: List[Float32], lo: LoConConvAdapter, N: Int, Hi: Int, Wi: Int,
) raises -> List[Float32]:
    var Ho = locon_out_h(Hi, lo.kh, lo.stride_h, lo.pad_h)
    var Wo = locon_out_w(Wi, lo.kw, lo.stride_w, lo.pad_w)
    var h = locon_down_forward(x, lo, N, Hi, Wi)
    return locon_up_forward(h, lo, N, Ho, Wo)


# ── dense delta kernel ΔW [Kh,Kw,Cin,Cout] (locon.rs get_diff_weight spatial) ─
# Per spatial position: down[kh,kw,:,:] @ up[0,0,:,:] → [Cin,Cout], * scale.
# Only for the SAVE convention / oracle, never the hot forward.
def locon_delta_kernel(lo: LoConConvAdapter) raises -> List[Float32]:
    var Kh = lo.kh; var Kw = lo.kw; var Cin = lo.cin; var R = lo.rank; var Cout = lo.cout
    var out = _zeros(Kh * Kw * Cin * Cout)
    for kh in range(Kh):
        for kw in range(Kw):
            var dbase = (((kh * Kw + kw) * Cin) * R)
            var obase = (((kh * Kw + kw) * Cin) * Cout)
            for ci in range(Cin):
                var drow = dbase + ci * R
                var orow = obase + ci * Cout
                for r in range(R):
                    var dv = lo.down[drow + r]
                    if dv == Float32(0.0):
                        continue
                    var ubase = r * Cout
                    for co in range(Cout):
                        out[orow + co] = out[orow + co] + dv * lo.up[ubase + co] * lo.scale
    return out^


# ── grads of down + up + d_x ─────────────────────────────────────────────────
struct LoConGrads(Copyable, Movable):
    var d_down: List[Float32]   # [Kh,Kw,Cin,rank]
    var d_up: List[Float32]     # [1,1,rank,Cout]
    var d_x: List[Float32]      # [N,Hi,Wi,Cin]

    def __init__(
        out self, var d_down: List[Float32], var d_up: List[Float32], var d_x: List[Float32],
    ):
        self.d_down = d_down^
        self.d_up = d_up^
        self.d_x = d_x^


# Backward through Δy = up(down(x))*scale with d_y:[N,Ho,Wo,Cout].
def locon_backward(
    d_y: List[Float32], x: List[Float32], lo: LoConConvAdapter, N: Int, Hi: Int, Wi: Int,
) raises -> LoConGrads:
    var Cin = lo.cin; var Cout = lo.cout; var R = lo.rank; var Kh = lo.kh; var Kw = lo.kw
    var sh = lo.stride_h; var sw = lo.stride_w; var ph = lo.pad_h; var pw = lo.pad_w
    var Ho = locon_out_h(Hi, Kh, sh, ph)
    var Wo = locon_out_w(Wi, Kw, sw, pw)
    var rows = N * Ho * Wo
    if len(d_y) != rows * Cout:
        raise Error("locon_backward: d_y numel mismatch")

    # forward intermediate h = down(x)  (no scale; scale lives at the output).
    var h = locon_down_forward(x, lo, N, Hi, Wi)

    # pull the scalar out: d_y_s = d_y * scale (since y = scale * up(h)).
    var d_y_s = d_y.copy()
    for i in range(len(d_y_s)):
        d_y_s[i] = d_y_s[i] * lo.scale

    # ── up grads + d_h (1x1, pointwise over rank) ──
    var d_up = _zeros(R * Cout)
    var d_h = _zeros(rows * R)
    for p in range(rows):
        var hbase = p * R
        var ybase = p * Cout
        for r in range(R):
            var hv = h[hbase + r]
            var ubase = r * Cout
            var acc = Float32(0.0)
            for co in range(Cout):
                var gy = d_y_s[ybase + co]
                d_up[ubase + co] = d_up[ubase + co] + hv * gy
                acc += gy * lo.up[ubase + co]
            d_h[hbase + r] = acc

    # ── down grads: d_down[kh,kw,ci,r] = Σ_{n,oh,ow} d_h[...,r] * x[...,ci] ──
    var d_down = _zeros(Kh * Kw * Cin * R)
    # ── d_x[n,ih,iw,ci] = Σ_{r,kh,kw} d_h[n,oh,ow,r] * down[kh,kw,ci,r] ──
    var d_x = _zeros(N * Hi * Wi * Cin)
    for n in range(N):
        for oh in range(Ho):
            for ow in range(Wo):
                var hbase = (((n * Ho + oh) * Wo + ow) * R)
                for kh in range(Kh):
                    var ih = oh * sh - ph + kh
                    if ih < 0 or ih >= Hi:
                        continue
                    for kw in range(Kw):
                        var iw = ow * sw - pw + kw
                        if iw < 0 or iw >= Wi:
                            continue
                        var xbase = (((n * Hi + ih) * Wi + iw) * Cin)
                        var dxbase = xbase
                        var wbase = (((kh * Kw + kw) * Cin) * R)
                        for ci in range(Cin):
                            var xv = x[xbase + ci]
                            var ddrow = wbase + ci * R
                            var acc_dx = Float32(0.0)
                            for r in range(R):
                                var dhv = d_h[hbase + r]
                                d_down[ddrow + r] = d_down[ddrow + r] + dhv * xv
                                acc_dx += dhv * lo.down[ddrow + r]
                            d_x[dxbase + ci] = d_x[dxbase + ci] + acc_dx
    return LoConGrads(d_down^, d_up^, d_x^)


# ── AdamW one step over down + up ────────────────────────────────────────────
def _adamw_host_list(
    mut p: List[Float32], g: List[Float32],
    mut mom: List[Float32], mut vmo: List[Float32],
    t: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32,
) raises:
    var n = len(p)
    if len(g) != n or len(mom) != n or len(vmo) != n:
        raise Error("locon _adamw_host_list: param/grad/m/v len mismatch")
    if t < 1:
        raise Error("locon _adamw_host_list: t must be >= 1")
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
        var pv = p[i] - lr * m_hat / (sqrt(v_hat) + eps)
        if weight_decay > 0.0:
            pv = pv - lr * weight_decay * pv
        p[i] = pv


def locon_adamw(
    mut lo: LoConConvAdapter, g: LoConGrads, t: Int, lr: Float32,
    beta1: Float32 = Float32(0.9), beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8), weight_decay: Float32 = Float32(0.01),
) raises:
    _adamw_host_list(lo.down, g.d_down, lo.m_down, lo.v_down, t, lr, beta1, beta2, eps, weight_decay)
    _adamw_host_list(lo.up, g.d_up, lo.m_up, lo.v_up, t, lr, beta1, beta2, eps, weight_decay)
