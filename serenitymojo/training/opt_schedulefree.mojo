# training/opt_schedulefree.mojo — RAdamScheduleFree (Defazio 2024), host-F32.
#
# NEW STANDALONE MODULE. Mirrors optim.mojo's struct/step idiom (read-only
# reference) and the EXACT algorithm of EriDiffusion-v2
# optimizers.rs::RAdamScheduleFree::step (~line 1516) + enter/exit_eval_mode,
# pinned against its inline reference (radam_schedulefree_5_steps_matches_
# reference, ~line 2549). pytorch_optimizer.ScheduleFreeRAdam.
#
# Schedule-free learning on RAdam's variance-rectified second moment. No LR
# schedule: the y/z/x triple-sequence interpolation replaces it. r=0,
# weight_lr_power=2, silent_sgd_phase=True (upstream defaults). z init = p.
#
# Per train step:
#   step += 1
#   beta2_pow = beta2^step (F64) ; bc2 = 1 - beta2_pow
#   n_sma_max = 2/(1-beta2) - 1
#   n_sma = n_sma_max - 2*step*beta2_pow / (1 - beta2_pow)
#   rt = sqrt( (1-beta2_pow)*(n_sma-4)/(n_sma_max-4)*(n_sma-2)/n_sma
#              * n_sma_max/(n_sma_max-2) )  if n_sma >= 4  else  -1
#   lr_t = lr * rt ; if lr_t < 0: lr_t = 0 (silent_sgd_phase) else 1
#   lr_max = max(lr_max, lr_t)
#   weight = step^r * lr_max^weight_lr_power   (r=0 → step^0 = 1)
#   weight_sum += weight
#   ckpt = weight / weight_sum  (0 if weight_sum==0)
#   adaptive_y_lr = lr_t * (beta1*(1-ckpt) - 1)
#   per element:
#     v = beta2*v + (1-beta2)*g*g
#     grad_eff = g / (sqrt(v)/bc2 + eps)   if n_sma > 4   else g   (STRICT >4)
#     if wd > 0:  grad_eff += wd * p        # COUPLED L2 (weight_decouple=False)
#     p = p*(1-ckpt) + z*ckpt
#     p += grad_eff * adaptive_y_lr
#     z -= grad_eff * lr_t
#
# eval mode (sampling): x = y*(1/beta1) + z*(1 - 1/beta1). enter stashes y,
# exit restores it. The math reference and the corruption-bug receipt are in
# optimizers.rs:1447-1460.
#
# AGENT-DEFAULT: r=0, weight_lr_power=2, silent_sgd_phase=True (upstream
# recommended defaults; fixed, not exposed as args — same as the Rust struct).
# Mojo 0.26.x. Host-F32 path (scalar F64 schedule + per-elem F32) — no GPU kernel.

from std.math import sqrt, exp, log


comptime _R_POW = Float64(0.0)
comptime _WEIGHT_LR_POWER = Float64(2.0)
comptime _SILENT_SGD_PHASE = True


struct RAdamScheduleFree(Movable):
    var lr: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32
    var step_t: Int
    var lr_max: Float32
    var weight_sum: Float64
    var z: List[Float32]
    var v: List[Float32]              # exp_avg_sq
    var initialized: Bool
    var eval_stash: List[Float32]
    var in_eval: Bool

    def __init__(out self, lr: Float32, beta1: Float32, beta2: Float32,
                 eps: Float32, weight_decay: Float32):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_t = 0
        self.lr_max = Float32(-1.0)
        self.weight_sum = Float64(0.0)
        self.z = List[Float32]()
        self.v = List[Float32]()
        self.initialized = False
        self.eval_stash = List[Float32]()
        self.in_eval = False

    # One train step. `param` (host F32, the y-sequence) updated IN PLACE.
    def step(mut self, mut param: List[Float32], grad: List[Float32]) raises:
        var n = len(param)
        if len(grad) != n:
            raise Error("schedulefree.step: param/grad numel mismatch")

        if not self.initialized:
            self.z = List[Float32]()
            self.v = List[Float32]()
            for i in range(n):
                self.z.append(param[i])     # z init = p
                self.v.append(0.0)
            self.initialized = True

        self.step_t += 1
        var step = self.step_t
        var beta1 = self.beta1
        var beta2 = self.beta2
        var eps = self.eps
        var wd = self.weight_decay

        # beta2^step in F64
        var beta2_pow = Float64(1.0)
        for _ in range(step):
            beta2_pow *= Float64(beta2)
        var bias_correction2 = Float32(1.0 - beta2_pow)

        var n_sma_max = 2.0 / (1.0 - Float64(beta2)) - 1.0
        var one_minus_b2t = 1.0 - beta2_pow
        var n_sma = n_sma_max
        var omt = one_minus_b2t
        if omt < 0.0:
            omt = -omt
        if omt > 1.0e-30:
            n_sma = n_sma_max - 2.0 * Float64(step) * beta2_pow / one_minus_b2t

        var rt = Float64(-1.0)
        if n_sma >= 4.0:
            rt = sqrt(
                one_minus_b2t * (n_sma - 4.0) / (n_sma_max - 4.0)
                * (n_sma - 2.0) / n_sma * n_sma_max / (n_sma_max - 2.0)
            )

        var lr_t = self.lr * Float32(rt)
        if lr_t < 0.0:
            if _SILENT_SGD_PHASE:
                lr_t = 0.0
            else:
                lr_t = 1.0

        if lr_t > self.lr_max:
            self.lr_max = lr_t

        # weight = step^r * lr_max^weight_lr_power ; r=0 → step^0 = 1
        var step_r = exp(_R_POW * log(Float64(step)))
        var lrmax_pow = Float64(0.0)
        if self.lr_max > 0.0:
            lrmax_pow = exp(_WEIGHT_LR_POWER * log(Float64(self.lr_max)))
        # lr_max could be 0 (silent sgd) → 0^2 = 0 (handled: lrmax_pow stays 0)
        var weight = step_r * lrmax_pow
        self.weight_sum += weight

        var checkpoint = Float32(0.0)
        if self.weight_sum != 0.0:
            checkpoint = Float32(weight / self.weight_sum)

        var adaptive_y_lr = lr_t * (beta1 * (1.0 - checkpoint) - 1.0)

        var use_denom = n_sma > 4.0     # STRICT > 4
        for i in range(n):
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grad[i] * grad[i]
            var grad_eff = grad[i]
            if use_denom:
                var denom = sqrt(self.v[i]) / bias_correction2 + eps
                grad_eff = grad[i] / denom
            if wd > 0.0:
                grad_eff = grad_eff + wd * param[i]
            # p ← p*(1-ckpt) + z*ckpt ; p += grad_eff*adaptive_y_lr ; z -= grad_eff*lr_t
            param[i] = param[i] * (1.0 - checkpoint) + self.z[i] * checkpoint
            param[i] = param[i] + grad_eff * adaptive_y_lr
            self.z[i] = self.z[i] - grad_eff * lr_t

    # Swap p from train weight y to eval weight x = y*(1/beta1) + z*(1-1/beta1).
    def enter_eval_mode(mut self, mut param: List[Float32]) raises:
        if self.in_eval:
            return
        if not self.initialized:
            return
        var n = len(param)
        if len(self.z) != n:
            raise Error("enter_eval_mode: z/param numel mismatch")
        var inv_beta1 = 1.0 / self.beta1
        var one_minus_inv = 1.0 - inv_beta1
        self.eval_stash = List[Float32]()
        for i in range(n):
            self.eval_stash.append(param[i])           # stash y
            param[i] = param[i] * inv_beta1 + self.z[i] * one_minus_inv
        self.in_eval = True

    # Restore p from the stashed train weight y.
    def exit_eval_mode(mut self, mut param: List[Float32]) raises:
        if not self.in_eval:
            return
        var n = len(param)
        if len(self.eval_stash) != n:
            raise Error("exit_eval_mode: stash/param numel mismatch")
        for i in range(n):
            param[i] = self.eval_stash[i]
        self.eval_stash = List[Float32]()
        self.in_eval = False
