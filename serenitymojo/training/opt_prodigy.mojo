# training/opt_prodigy.mojo — Prodigy (Mishchenko & Defazio 2023), host-F32.
#
# NEW STANDALONE MODULE. Mirrors optim.mojo's struct/step idiom (read-only
# reference) and the EXACT algorithm of EriDiffusion-v2 optimizers.rs::Prodigy
# ::step (~line 853), pinned against the reference impl
# https://github.com/konstmish/prodigy (decouple=True, safeguard_warmup=False,
# slice_p=1, use_bias_correction=False).
#
# D-adaptive auto-tuning of the AdamW step size. The scalar D estimate (`d`,
# `d_max`, `d_numerator`) is GLOBAL across all params; the per-param state
# (m, v, s, p0) is per-tensor. This module implements the SINGLE-PARAM scope
# (the parity-relevant unit — the convergence test uses one param). The inner
# product <g, p0-p> and L1 sum |s|.sum() are host reductions (parity-grade).
#
# Per step (single param, decouple=True):
#   t      += 1
#   beta3   = sqrt(beta2)
#   dlr     = d * lr            (bias_correction=1)
#   d_numerator *= beta3        (F64)
#   delta_numerator = (d/d0) * dlr * <g, p0 - p>           (F64)
#   m = beta1*m + d*(1-beta1)*g
#   v = beta2*v + d^2*(1-beta2)*g*g
#   s = beta3*s + ((d/d0)*dlr) * g
#   d_denom = sum |s|                                       (F64, L1)
#   if d_denom>0 and lr>0:
#     d_numerator += delta_numerator
#     d_hat = d_coef * d_numerator / d_denom                (d_coef=1)
#     if |d - d0| < f32_eps:  d = max(d, d_hat)             (grow only on step 1)
#     d_max = max(d_max, d_hat)
#     d = min(d_max, INF)        (growth_rate = INF → unrestricted)
#   else:
#     d_numerator /= beta3       (undo pre-decay)
#   dlr = d * lr   (recompute with new d)
#   denom = sqrt(v) + d*eps
#   if wd != 0:  p *= (1 - wd*dlr)                          # decoupled, by dlr
#   p -= dlr * m / denom
#
# d0 = 1e-6 (reference default, fixed). d_coef = 1, growth_rate = INF (fixed).
# AGENT-DEFAULT: lr typically 1.0 (reference recommends leaving it at 1).
# Mojo 0.26.x. Host-F32 path (scalar D-adaptation + reductions) — no GPU kernel.

from std.math import sqrt


comptime _D0 = Float32(1.0e-6)
comptime _D_COEF = Float64(1.0)
comptime _F32_EPS = Float32(1.19209290e-07)  # f32::EPSILON


# ── Prodigy state struct (single-param scope, host F32 + F64 numerator) ──────
struct Prodigy(Movable):
    var lr: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32
    var d: Float32
    var d_max: Float32
    var d_numerator: Float64
    var t: Int
    var m: List[Float32]
    var v: List[Float32]
    var s: List[Float32]
    var p0: List[Float32]
    var initialized: Bool

    def __init__(out self, lr: Float32, beta1: Float32, beta2: Float32,
                 eps: Float32, weight_decay: Float32):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.d = _D0
        self.d_max = _D0
        self.d_numerator = Float64(0.0)
        self.t = 0
        self.m = List[Float32]()
        self.v = List[Float32]()
        self.s = List[Float32]()
        self.p0 = List[Float32]()
        self.initialized = False

    def d_value(self) -> Float32:
        return self.d

    # One Prodigy step. `param` (host F32) is updated IN PLACE; `grad` read-only.
    def step(mut self, mut param: List[Float32], grad: List[Float32]) raises:
        var n = len(param)
        if len(grad) != n:
            raise Error("prodigy.step: param/grad numel mismatch")

        if not self.initialized:
            self.m = List[Float32]()
            self.v = List[Float32]()
            self.s = List[Float32]()
            self.p0 = List[Float32]()
            for i in range(n):
                self.m.append(0.0)
                self.v.append(0.0)
                self.s.append(0.0)
                self.p0.append(param[i])      # snapshot p0
            self.initialized = True

        self.t += 1
        var beta3 = sqrt(self.beta2)
        var d = self.d
        var d0 = _D0
        var lr = self.lr
        var dlr = d * lr                       # bias_correction = 1

        # pre-decay running numerator (F64)
        self.d_numerator *= Float64(beta3)

        # delta_numerator = (d/d0) * dlr * <g, p0 - p>   (F64)
        var inner = Float32(0.0)
        for i in range(n):
            inner += grad[i] * (self.p0[i] - param[i])
        var delta_numerator = (Float64(d) / Float64(d0)) * Float64(dlr) * Float64(inner)

        # m / v / s updates + d_denom = sum|s|  (F64)
        var s_alpha = (d / d0) * dlr
        var d_denom = Float64(0.0)
        for i in range(n):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * d * grad[i]
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * d * d * grad[i] * grad[i]
            self.s[i] = beta3 * self.s[i] + s_alpha * grad[i]
            var sa = self.s[i]
            if sa < 0.0:
                sa = -sa
            d_denom += Float64(sa)

        # scalar D estimate update
        if d_denom > 0.0 and lr > 0.0:
            self.d_numerator += delta_numerator
            var d_hat = (_D_COEF * self.d_numerator) / d_denom
            var d_hat_f32 = Float32(d_hat)
            var d_minus_d0 = self.d - d0
            if d_minus_d0 < 0.0:
                d_minus_d0 = -d_minus_d0
            if d_minus_d0 < _F32_EPS:
                if d_hat_f32 > self.d:
                    self.d = d_hat_f32
            if d_hat_f32 > self.d_max:
                self.d_max = d_hat_f32
            # growth_rate = INF → grown = INF → d = d_max
            self.d = self.d_max
        else:
            if beta3 > 0.0:
                self.d_numerator /= Float64(beta3)

        # phase 2: recompute dlr with the (possibly-updated) d
        d = self.d
        dlr = d * lr
        for i in range(n):
            var denom = sqrt(self.v[i]) + d * self.eps
            if self.weight_decay != 0.0:
                param[i] = param[i] * (1.0 - self.weight_decay * dlr)
            param[i] = param[i] - dlr * self.m[i] / denom
