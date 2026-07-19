# models/lingbotvideo/scheduler.mojo — FlowUniPCMultistepScheduler (pure Mojo).
#
# Direct port of lingbot_video/scheduling_flow_unipc.py:FlowUniPCMultistepScheduler
# (flow-matching UniPC multistep predictor/corrector). INFERENCE ONLY.
#
# Design: UniPC's per-step coefficients (h, lambdas, the R-matrix bh vector, and
# the small linear solve for rhos) are inherently O(order^3) scalar host math
# (order <= solver_order, here 2). The latent update is a linear combination of a
# handful of tiny latent tensors (T2I latent = 1x16x1x8x8 = 1024 floats). We keep
# the whole scheduler in host F32 (`List[Float32]`) — exact, deterministic, and
# trivially cheap; a GPU pipeline moves latents host<->device around the step.
#
# Ported EXACTLY from the reference:
#   - _sigma_to_alpha_sigma_t(sigma) -> (1 - sigma, sigma)   [NOTE: alpha = 1-sigma]
#   - predict_x0 = True, solver_type = "bh2" (B_h = expm1(hh)), solver_order = 2,
#     lower_order_final = True, final_sigmas_type = "zero", disable_corrector = [].
#   - convert_model_output (flow / predict_x0): x0 = sample - sigma * model_output
#   - multistep_uni_p_bh_update  (PREDICTOR)
#   - multistep_uni_c_bh_update  (CORRECTOR)
#   - step (orchestration: corrector-on-previous -> convert -> shift history ->
#           order ramp -> predictor).

from std.math import log, expm1


def _solve(A: List[List[Float64]], rhs: List[Float64]) raises -> List[Float64]:
    """Gaussian elimination with partial pivoting for the small (<= order x order)
    UniPC linear system R @ x = b. order <= solver_order (typically <= 3)."""
    var n = len(rhs)
    # Build an augmented copy.
    var m = List[List[Float64]]()
    for i in range(n):
        var row = List[Float64]()
        for j in range(n):
            row.append(A[i][j])
        row.append(rhs[i])
        m.append(row^)
    # Forward elimination.
    for col in range(n):
        # pivot
        var piv = col
        var best = m[col][col]
        if best < 0.0:
            best = -best
        for r in range(col + 1, n):
            var v = m[r][col]
            if v < 0.0:
                v = -v
            if v > best:
                best = v
                piv = r
        if piv != col:
            var tmp = m[col].copy()
            m[col] = m[piv].copy()
            m[piv] = tmp^
        var diag = m[col][col]
        for r in range(col + 1, n):
            var factor = m[r][col] / diag
            for c in range(col, n + 1):
                m[r][c] = m[r][c] - factor * m[col][c]
    # Back substitution.
    var x = List[Float64]()
    for _ in range(n):
        x.append(0.0)
    for ii in range(n):
        var i = n - 1 - ii
        var s = m[i][n]
        for j in range(i + 1, n):
            s = s - m[i][j] * x[j]
        x[i] = s / m[i][i]
    return x^


struct FlowUniPCMultistepScheduler(Movable):
    # config
    var num_train_timesteps: Int
    var solver_order: Int
    var predict_x0: Bool
    var solver_type_bh2: Bool  # True => bh2 (B_h = expm1(hh)); False => bh1 (B_h = hh)
    var lower_order_final: Bool
    var final_sigmas_zero: Bool

    # schedule (built by set_timesteps)
    var sigma_max: Float64
    var sigma_min: Float64
    var sigmas: List[Float64]  # length num_inference_steps + 1
    var timesteps: List[Int]  # length num_inference_steps (int64-truncated)
    var num_inference_steps: Int

    # runtime state
    var model_outputs: List[List[Float32]]  # ring length solver_order (converted x0)
    var mo_valid: List[Bool]
    var lower_order_nums: Int
    var this_order: Int
    var step_index: Int
    var last_sample: List[Float32]
    var has_last_sample: Bool

    def __init__(out self, num_train_timesteps: Int = 1000):
        # Reference defaults (constructor built with shift=1, use_dynamic_shifting=False):
        # solver_order=2, predict_x0=True, solver_type="bh2", lower_order_final=True,
        # final_sigmas_type="zero".
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = 2
        self.predict_x0 = True
        self.solver_type_bh2 = True
        self.lower_order_final = True
        self.final_sigmas_zero = True

        # Constructor schedule (shift=1): alphas = linspace(1, 1/N, N)[::-1];
        # sigmas = 1 - alphas => sigmas[0]=1-1/N .. sigmas[-1]=0. sigma_max/min from ends.
        var ntt = Float64(num_train_timesteps)
        self.sigma_max = 1.0 - 1.0 / ntt  # 0.999 for 1000
        self.sigma_min = 0.0

        self.sigmas = List[Float64]()
        self.timesteps = List[Int]()
        self.num_inference_steps = 0

        self.model_outputs = List[List[Float32]]()
        self.mo_valid = List[Bool]()
        for _ in range(self.solver_order):
            self.model_outputs.append(List[Float32]())
            self.mo_valid.append(False)
        self.lower_order_nums = 0
        self.this_order = 0
        self.step_index = 0
        self.last_sample = List[Float32]()
        self.has_last_sample = False

    def set_timesteps(mut self, num_inference_steps: Int, shift: Float64) raises:
        """Build sigmas + integer timesteps (final_sigmas_type='zero') and reset
        the multistep history. Mirrors set_timesteps(num_inference_steps, shift)."""
        var N = num_inference_steps
        # sigmas = linspace(sigma_max, sigma_min, N+1)[:-1]
        # linspace endpoints: s[k] = sigma_max + k*(sigma_min - sigma_max)/N
        var raw = List[Float64]()
        for k in range(N):  # drop last (the [:-1])
            var s = self.sigma_max + Float64(k) * (self.sigma_min - self.sigma_max) / Float64(N)
            raw.append(s)
        # shift transform: sigma = shift*s / (1 + (shift-1)*s)
        var shifted = List[Float64]()
        for k in range(N):
            var s = raw[k]
            shifted.append(shift * s / (1.0 + (shift - 1.0) * s))

        # timesteps = sigma * num_train_timesteps, cast to int64 (truncate toward 0)
        self.timesteps = List[Int]()
        for k in range(N):
            self.timesteps.append(Int(shifted[k] * Float64(self.num_train_timesteps)))

        # sigmas = concat([shifted, [sigma_last]]);  final_sigmas_type='zero' => 0
        self.sigmas = List[Float64]()
        for k in range(N):
            self.sigmas.append(shifted[k])
        self.sigmas.append(0.0)  # sigma_last

        self.num_inference_steps = N

        # reset state
        self.model_outputs = List[List[Float32]]()
        self.mo_valid = List[Bool]()
        for _ in range(self.solver_order):
            self.model_outputs.append(List[Float32]())
            self.mo_valid.append(False)
        self.lower_order_nums = 0
        self.this_order = 0
        self.step_index = 0
        self.last_sample = List[Float32]()
        self.has_last_sample = False

    def set_timesteps_refiner(
        mut self, num_inference_steps: Int, shift: Float64,
        t_thresh: Float64, tail_steps: Int,
    ) raises:
        """Refiner sigma schedule (creator utils.compute_refiner_sigmas): the shifted
        linspace truncated to sigmas <= t_thresh (prepend t_thresh if needed) + an
        optional low-sigma tail. Starts the refine denoise at sigma=t_thresh (partial
        noise) rather than sigma_max. Mirrors set_timesteps(len(sigmas), sigmas, shift=1)."""
        var N = num_inference_steps
        var shifted = List[Float64]()
        for k in range(N):
            var s = self.sigma_max + Float64(k) * (self.sigma_min - self.sigma_max) / Float64(N)
            shifted.append(shift * s / (1.0 + (shift - 1.0) * s))
        # keep sigmas <= t_thresh (+eps)
        var sig = List[Float64]()
        for k in range(N):
            if shifted[k] <= t_thresh + 1e-6:
                sig.append(shifted[k])
        if len(sig) == 0 or sig[0] != t_thresh:
            var pre = List[Float64]()
            pre.append(t_thresh)
            for k in range(len(sig)):
                pre.append(sig[k])
            sig = pre^
        # low-sigma tail: linspace(sig[-1], min(sigma_min,sig[-1]), tail+2)[1:-1]
        if tail_steps > 0:
            var lo = sig[len(sig) - 1]
            var hi = self.sigma_min if self.sigma_min < lo else lo
            var M = tail_steps + 2
            for k in range(1, M - 1):
                sig.append(lo + Float64(k) * (hi - lo) / Float64(M - 1))
        # ingest as the explicit schedule (shift already applied above -> shift=1 here)
        self.timesteps = List[Int]()
        self.sigmas = List[Float64]()
        for k in range(len(sig)):
            self.timesteps.append(Int(sig[k] * Float64(self.num_train_timesteps)))
            self.sigmas.append(sig[k])
        self.sigmas.append(0.0)
        self.num_inference_steps = len(sig)
        self.model_outputs = List[List[Float32]]()
        self.mo_valid = List[Bool]()
        for _ in range(self.solver_order):
            self.model_outputs.append(List[Float32]())
            self.mo_valid.append(False)
        self.lower_order_nums = 0
        self.this_order = 0
        self.step_index = 0
        self.last_sample = List[Float32]()
        self.has_last_sample = False

    # ── _sigma_to_alpha_sigma_t: (1 - sigma, sigma) ─────────────────────────────
    @staticmethod
    def _alpha(sigma: Float64) -> Float64:
        return 1.0 - sigma

    def _convert_model_output(
        self, model_output: List[Float32], sample: List[Float32]
    ) raises -> List[Float32]:
        # flow / predict_x0: x0 = sample - sigma_t * model_output, sigma_t = sigmas[step_index]
        var sigma_t = Float32(self.sigmas[self.step_index])
        var out = List[Float32]()
        for j in range(len(sample)):
            out.append(sample[j] - sigma_t * model_output[j])
        return out^

    def _predictor(mut self, x: List[Float32], order: Int) raises -> List[Float32]:
        var n = len(x)
        var last = self.solver_order - 1
        ref m0 = self.model_outputs[last]

        var sigma_t = self.sigmas[self.step_index + 1]
        var sigma_s0 = self.sigmas[self.step_index]
        var alpha_t = Self._alpha(sigma_t)
        var alpha_s0 = Self._alpha(sigma_s0)
        var lambda_t = log(alpha_t) - log(sigma_t)
        var lambda_s0 = log(alpha_s0) - log(sigma_s0)
        var h = lambda_t - lambda_s0

        var rks = List[Float64]()
        var D1s = List[List[Float32]]()
        for i in range(1, order):
            var si = self.step_index - i
            ref mi = self.model_outputs[last - i]  # model_output_list[-(i+1)]
            var sigma_si = self.sigmas[si]
            var lambda_si = log(Self._alpha(sigma_si)) - log(sigma_si)
            var rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            var d = List[Float32]()
            for j in range(n):
                d.append((mi[j] - m0[j]) / Float32(rk))
            D1s.append(d^)
        rks.append(1.0)

        var hh = -h if self.predict_x0 else h
        var h_phi_1 = expm1(hh)
        var h_phi_k = h_phi_1 / hh - 1.0
        var B_h = expm1(hh) if self.solver_type_bh2 else hh
        var factorial_i = 1

        var R = List[List[Float64]]()
        var b = List[Float64]()
        for i in range(1, order + 1):
            var row = List[Float64]()
            for r in range(len(rks)):
                row.append(rks[r] ** (i - 1))
            R.append(row^)
            b.append(h_phi_k * Float64(factorial_i) / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1.0 / Float64(factorial_i)

        # rhos_p
        var rhos_p = List[Float64]()
        if len(D1s) > 0:
            if order == 2:
                rhos_p.append(0.5)
            else:
                # solve R[:-1, :-1] @ rhos = b[:-1]
                var Asub = List[List[Float64]]()
                var bsub = List[Float64]()
                for i in range(order - 1):
                    var row = List[Float64]()
                    for j in range(order - 1):
                        row.append(R[i][j])
                    Asub.append(row^)
                    bsub.append(b[i])
                rhos_p = _solve(Asub, bsub)

        # x_t_ = sigma_t/sigma_s0 * x - alpha_t*h_phi_1*m0 ; x_t = x_t_ - alpha_t*B_h*pred_res
        var c1 = sigma_t / sigma_s0
        var c2 = alpha_t * h_phi_1
        var cB = alpha_t * B_h
        var out = List[Float32]()
        for j in range(n):
            var v = Float32(c1) * x[j] - Float32(c2) * m0[j]
            if len(D1s) > 0:
                var pr: Float64 = 0.0
                for k in range(len(D1s)):
                    pr += rhos_p[k] * Float64(D1s[k][j])
                v = v - Float32(cB * pr)
            out.append(v)
        return out^

    def _corrector(
        self,
        this_model_output: List[Float32],
        last_sample: List[Float32],
        this_sample: List[Float32],
        order: Int,
    ) raises -> List[Float32]:
        var n = len(this_sample)
        var last = self.solver_order - 1
        ref m0 = self.model_outputs[last]
        ref model_t = this_model_output

        var sigma_t = self.sigmas[self.step_index]
        var sigma_s0 = self.sigmas[self.step_index - 1]
        var alpha_t = Self._alpha(sigma_t)
        var alpha_s0 = Self._alpha(sigma_s0)
        var lambda_t = log(alpha_t) - log(sigma_t)
        var lambda_s0 = log(alpha_s0) - log(sigma_s0)
        var h = lambda_t - lambda_s0

        var rks = List[Float64]()
        var D1s = List[List[Float32]]()
        for i in range(1, order):
            var si = self.step_index - (i + 1)
            ref mi = self.model_outputs[last - i]  # model_output_list[-(i+1)]
            var sigma_si = self.sigmas[si]
            var lambda_si = log(Self._alpha(sigma_si)) - log(sigma_si)
            var rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            var d = List[Float32]()
            for j in range(n):
                d.append((mi[j] - m0[j]) / Float32(rk))
            D1s.append(d^)
        rks.append(1.0)

        var hh = -h if self.predict_x0 else h
        var h_phi_1 = expm1(hh)
        var h_phi_k = h_phi_1 / hh - 1.0
        var B_h = expm1(hh) if self.solver_type_bh2 else hh
        var factorial_i = 1

        var R = List[List[Float64]]()
        var b = List[Float64]()
        for i in range(1, order + 1):
            var row = List[Float64]()
            for r in range(len(rks)):
                row.append(rks[r] ** (i - 1))
            R.append(row^)
            b.append(h_phi_k * Float64(factorial_i) / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1.0 / Float64(factorial_i)

        var rhos_c = List[Float64]()
        if order == 1:
            rhos_c.append(0.5)
        else:
            rhos_c = _solve(R, b)  # full order x order

        var c1 = sigma_t / sigma_s0
        var c2 = alpha_t * h_phi_1
        var cB = alpha_t * B_h
        var rho_last = rhos_c[len(rhos_c) - 1]
        var out = List[Float32]()
        for j in range(n):
            var corr_res: Float64 = 0.0
            for k in range(len(D1s)):  # rhos_c[:-1]
                corr_res += rhos_c[k] * Float64(D1s[k][j])
            var D1_t = Float64(model_t[j] - m0[j])
            var v = (
                Float32(c1) * last_sample[j]
                - Float32(c2) * m0[j]
                - Float32(cB * (corr_res + rho_last * D1_t))
            )
            out.append(v)
        return out^

    def step(
        mut self, model_output: List[Float32], sample: List[Float32]
    ) raises -> List[Float32]:
        """One UniPC predictor/corrector step. `sample` is the current latent,
        `model_output` the model velocity prediction. Returns the previous-timestep
        latent. Advances internal step_index and history."""
        var n = len(sample)

        # disable_corrector is empty => use_corrector = step_index>0 and last_sample set
        var use_corrector = self.step_index > 0 and self.has_last_sample

        var model_output_convert = self._convert_model_output(model_output, sample)

        var work_sample = sample.copy()
        if use_corrector:
            work_sample = self._corrector(
                model_output_convert, self.last_sample, sample, self.this_order
            )

        # shift history ring, append converted output
        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1].copy()
            self.mo_valid[i] = self.mo_valid[i + 1]
        self.model_outputs[self.solver_order - 1] = model_output_convert.copy()
        self.mo_valid[self.solver_order - 1] = True

        # order ramp
        var this_order: Int
        if self.lower_order_final:
            var rem = self.num_inference_steps - self.step_index
            this_order = self.solver_order if self.solver_order < rem else rem
        else:
            this_order = self.solver_order
        var cap = self.lower_order_nums + 1
        self.this_order = this_order if this_order < cap else cap

        self.last_sample = work_sample.copy()
        self.has_last_sample = True

        var prev_sample = self._predictor(work_sample, self.this_order)

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        self.step_index += 1

        _ = n
        return prev_sample^
