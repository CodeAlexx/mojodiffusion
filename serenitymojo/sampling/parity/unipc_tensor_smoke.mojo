# sampling/parity/unipc_tensor_smoke.mojo — UniPC bh2 TENSOR-path smoke.
#
# The scalar gate (unipc_parity.mojo) only imports the f64 coefficient helpers,
# so Mojo's lazy per-entry-point compilation never compiled the stateful tensor
# scheduler (`UniPcMultistepScheduler.__init__`, `.step`, `._predictor`,
# `._corrector`, `._convert_model_output`). This smoke INSTANTIATES the
# scheduler and RUNS `.step` for several inference steps on real device
# tensors so all those paths are compiled, and asserts the GPU trajectory
# matches an INDEPENDENT scalar reimplementation of the same scheme elementwise.
#
# Running >2 steps exercises: the order-0 (no-corrector) first step, the
# order-2 corrector + predictor steps with the full 2x2 rhos solve, and the
# ring-buffer shift. Uses F32 tensors so the tensor result matches the f64
# scalar reference tightly. PARITY-BITROT GUARD: `--bitrot` zeros the
# order-2 corrector coeff in the *expected* scalar trajectory, so the
# (correct) tensor output diverges → assertion fires → exit 1.
#
# Build / run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/sampling/parity/unipc_tensor_smoke.mojo
#   (add a trailing `--bitrot` arg for the deliberate-wrong demo)

from collections import List
from sys import argv
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.sampling.unipc import (
    UniPcMultistepScheduler,
    compute_bh2_coefficients,
)


def _shape3() -> List[Int]:
    var sh = List[Int]()
    sh.append(3)
    return sh^


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var d = _abs(got - expected)
    if d > tol:
        raise Error(
            name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
            + String(" |Δ|=")
            + String(d)
        )


# ── Independent scalar reference (one lane), mirroring the tensor scheduler ──
struct _Scalar(Movable):
    var outputs: List[Float64]   # converted x0, newest last (cap 2)
    var has: List[Bool]
    var last_sample: Float64
    var have_last: Bool
    var lower_order_nums: Int
    var this_order: Int
    var step_index: Int

    def __init__(out self):
        self.outputs = List[Float64]()
        self.outputs.append(0.0)
        self.outputs.append(0.0)
        self.has = List[Bool]()
        self.has.append(False)
        self.has.append(False)
        self.last_sample = 0.0
        self.have_last = False
        self.lower_order_nums = 0
        self.this_order = 0
        self.step_index = 0


def _scalar_predictor(
    sigmas: List[Float64], step_index: Int, sample: Float64,
    m0: Float64, m1: Float64, order: Int,
) raises -> Float64:
    var c = compute_bh2_coefficients(sigmas, step_index, order, False)
    var x_t_ = (c.sigma_t / c.sigma_s0) * sample - (c.alpha_t * c.h_phi_1) * m0
    if order == 1:
        return x_t_
    var rk = c.rks[0]
    var d1 = (m1 - m0) / rk
    var pred_res = 0.5 * d1
    return x_t_ - (c.alpha_t * c.b_h) * pred_res


def _scalar_corrector(
    sigmas: List[Float64], step_index: Int, this_model_output: Float64,
    last_sample: Float64, m0: Float64, m1: Float64, order: Int, bitrot: Bool,
) raises -> Float64:
    var c = compute_bh2_coefficients(sigmas, step_index, order, True)
    var x_t_ = (c.sigma_t / c.sigma_s0) * last_sample - (c.alpha_t * c.h_phi_1) * m0
    var rho_last: Float64
    var rho0: Float64
    if order == 1:
        rho_last = 0.5
        rho0 = 0.0
    else:
        rho0 = c.rhos[0]
        rho_last = c.rhos[1]
    var d1_t = this_model_output - m0
    var total = rho_last * d1_t
    if order == 2:
        var rk = c.rks[0]
        var d1 = (m1 - m0) / rk
        if bitrot:
            rho0 = 0.0
        total = rho0 * d1 + total
    return x_t_ - (c.alpha_t * c.b_h) * total


# Run the scalar reference scheduler for one lane; returns the trajectory of
# `prev_sample` after each step.
def _scalar_run(
    sigmas: List[Float64], n: Int, x_init: Float64, model_out: List[Float64],
    bitrot: Bool,
) raises -> List[Float64]:
    var st = _Scalar()
    var x = x_init
    var traj = List[Float64]()
    for i in range(n):
        var sigma_cur = sigmas[st.step_index]
        var mo = model_out[i]                       # raw model output (velocity)
        var mo_convert = x - sigma_cur * mo         # convert_model_output (x0)

        var use_corrector = st.step_index > 0 and st.have_last
        var sample_after = x
        if use_corrector:
            var m0 = st.outputs[1]
            var m1 = st.outputs[0]
            sample_after = _scalar_corrector(
                sigmas, st.step_index, mo_convert, st.last_sample, m0, m1,
                st.this_order, bitrot,
            )

        st.outputs[0] = st.outputs[1]
        st.has[0] = st.has[1]
        st.outputs[1] = mo_convert
        st.has[1] = True

        var this_order = 2
        var remaining = n - st.step_index
        if remaining < this_order:
            this_order = remaining
        if st.lower_order_nums + 1 < this_order:
            this_order = st.lower_order_nums + 1
        st.this_order = this_order

        st.last_sample = sample_after
        st.have_last = True

        var m0p = st.outputs[1]
        var m1p = st.outputs[0]
        x = _scalar_predictor(sigmas, st.step_index, sample_after, m0p, m1p, this_order)

        if st.lower_order_nums < 2:
            st.lower_order_nums += 1
        st.step_index += 1
        traj.append(x)
    return traj^


def main() raises:
    var args = argv()
    var bitrot = False
    for i in range(len(args)):
        if args[i] == "--bitrot":
            bitrot = True

    var ctx = DeviceContext()
    print("=== UniPC bh2 tensor-path smoke ===" + (" [BITROT]" if bitrot else ""))

    var n_inf = 5
    var sched = UniPcMultistepScheduler(1000, n_inf, 5.0, 2)
    var sigmas = sched.sigmas()  # n_inf + 1 descending, last == 0

    # 3 lanes; per-step raw model outputs (velocity) chosen arbitrarily.
    var x_init = List[Float32]()
    x_init.append(0.40)
    x_init.append(-0.60)
    x_init.append(1.10)

    # model outputs per step per lane (n_inf steps × 3 lanes).
    var r0 = List[Float32]()
    r0.append(0.10)
    r0.append(-0.20)
    r0.append(0.30)
    var r1 = List[Float32]()
    r1.append(-0.05)
    r1.append(0.25)
    r1.append(0.10)
    var r2 = List[Float32]()
    r2.append(0.20)
    r2.append(-0.10)
    r2.append(-0.15)
    var r3 = List[Float32]()
    r3.append(-0.12)
    r3.append(0.08)
    r3.append(0.22)
    var r4 = List[Float32]()
    r4.append(0.05)
    r4.append(-0.30)
    r4.append(0.18)
    var mo_step = List[List[Float32]]()
    mo_step.append(r0^)
    mo_step.append(r1^)
    mo_step.append(r2^)
    mo_step.append(r3^)
    mo_step.append(r4^)

    # ── tensor run ──
    var x = Tensor.from_host(x_init, _shape3(), STDtype.F32, ctx)
    var tensor_traj = List[List[Float32]]()
    for step in range(n_inf):
        var mo = Tensor.from_host(mo_step[step], _shape3(), STDtype.F32, ctx)
        var nxt = sched.step(mo, x, ctx)
        var nxt_host = nxt.to_host(ctx)
        var rec = List[Float32]()
        for j in range(3):
            rec.append(nxt_host[j])
        tensor_traj.append(rec^)
        x = nxt^

    # ── scalar reference per lane ──
    for lane in range(3):
        var mo_lane = List[Float64]()
        for step in range(n_inf):
            mo_lane.append(Float64(mo_step[step][lane]))
        var ref_traj = _scalar_run(sigmas, n_inf, Float64(x_init[lane]), mo_lane, bitrot)
        for step in range(n_inf):
            _check_close(
                String("lane") + String(lane) + String(" step") + String(step),
                tensor_traj[step][lane], Float32(ref_traj[step]), 1.0e-4,
            )
    print("  ran", n_inf, "scheduler steps × 3 lanes: tensor == scalar reference  OK")

    print("PASS: UniPC bh2 tensor scheduler compiles and matches scalar reference")
