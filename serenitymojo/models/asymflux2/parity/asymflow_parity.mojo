# asymflow_parity.mojo — CHUNK A parity gate for the AsymFLUX.2 asymmetric
# velocity reconstruction (models/asymflux2/asymflow.mojo).
#
# ORACLE INDEPENDENCE: SAME-SOURCE. No diffusers/python reference for
# AsymFlowMixin is on disk; the only reference is the Rust transcription in
# inference-flame/src/models/asymflux2.rs (itself a port of LakonLab's
# common.py). The host-side reference below is an INDEPENDENT re-derivation
# of the AsymFlow algebra straight from the math (subspace = state·P·P^T,
# branch-asymmetric mix), NOT a copy of the Mojo kernel calls — it uses
# plain host loops for the matmuls. This still flags as same-source overall
# (like the LTX2 VAE case) since the algebra originates from one reference,
# but the host re-derivation independently exercises the asymmetric branch
# split so a transcription bug in the Mojo path would surface.
#
# Run:
#   cd /home/alex/mojodiffusion
#   pixi run mojo run -I . serenitymojo/models/asymflux2/parity/asymflow_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.models.asymflux2.asymflow import (
    compute_calibration,
    asymflow_velocity,
)


# Small but representative dims: B=1, hw=6 tokens, D=8, R=3.
comptime B = 1
comptime HW = 6
comptime D = 8
comptime R = 3
comptime SIGMA_MIN = Float32(1e-4)


# ── deterministic host fills (Lehmer LCG in [-1,1)) ──────────────────────────
def _fill(n: Int, seed: UInt32) -> List[Float32]:
    var state = seed
    var out = List[Float32]()
    for _ in range(n):
        state = (state * 48271) % 0x7FFFFFFF
        var u = Float32(Int(state)) / Float32(0x7FFFFFFF)
        out.append(u * 2.0 - 1.0)
    return out^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


# ── host reference: independent re-derivation of AsymFlow velocity ───────────
# state: (HW, D) row-major ; p: (D, R) row-major.
def _proj_subspace(state: List[Float32], p: List[Float32]) -> List[Float32]:
    """subspace = state @ P @ P^T, returned (HW, D). Plain host loops."""
    # proj = state @ P  -> (HW, R)
    var proj = List[Float32]()
    for t in range(HW):
        for j in range(R):
            var acc = Float32(0.0)
            for di in range(D):
                acc += state[t * D + di] * p[di * R + j]
            proj.append(acc)
    # subspace = proj @ P^T  -> (HW, D) ; (P^T)[j, di] = p[di*R + j]
    var sub = List[Float32]()
    for t in range(HW):
        for di in range(D):
            var acc = Float32(0.0)
            for j in range(R):
                acc += proj[t * R + j] * p[di * R + j]
            sub.append(acc)
    return sub^


def _ref_velocity(
    u_a: List[Float32],
    x_t: List[Float32],
    p: List[Float32],
    s: Float32,
    k: Float32,
    sigma: Float32,
) -> List[Float32]:
    var u_a_sub = _proj_subspace(u_a, p)
    var x_t_sub = _proj_subspace(x_t, p)
    var sk = s * k
    var sigma_c = sigma if sigma >= SIGMA_MIN else SIGMA_MIN
    var inv = 1.0 / sigma_c
    var coef2 = (1.0 - sk) * inv
    var out = List[Float32]()
    for i in range(HW * D):
        var u_a_comp = u_a[i] - u_a_sub[i]
        var x_t_comp = x_t[i] - x_t_sub[i]
        var u_sub = sk * u_a_sub[i] + coef2 * x_t_sub[i]
        var u_comp = (x_t_comp + s * u_a_comp) * inv
        out.append(u_sub + u_comp)
    return out^


def main() raises:
    var ctx = DeviceContext()

    var u_a_h = _fill(HW * D, 0xC0FFEE)
    var x_t_h = _fill(HW * D, 0xBEEF01)
    var p_h = _fill(D * R, 0x1234AB)

    # Calibration: representative timestep, scale_buffer != 1 to exercise k.
    var cal = compute_calibration(0.3, 0.8, 1.0)

    # ── Mojo GPU path (F32 tensors) ──────────────────────────────────────────
    var u_a = Tensor.from_host(u_a_h, _shape3(B, HW, D), STDtype.F32, ctx)
    var x_t = Tensor.from_host(x_t_h, _shape3(B, HW, D), STDtype.F32, ctx)
    var p = Tensor.from_host(p_h, _shape2(D, R), STDtype.F32, ctx)

    var out = asymflow_velocity(u_a, x_t, cal, p, SIGMA_MIN, ctx)
    var actual = out.to_host(ctx)

    # ── host reference ───────────────────────────────────────────────────────
    var reference = _ref_velocity(u_a_h, x_t_h, p_h, cal.s, cal.k, cal.sigma)

    var harness = ParityHarness(0.999)
    var res = harness.compare_host(actual, reference)
    print("ASYMFLOW_BLOCK_PARITY", res)
    print("cal: s=", cal.s, " k=", cal.k, " sigma=", cal.sigma, " cal_t=", cal.cal_timestep)
    if res.passed:
        print("GATE: PASS")
    else:
        print("GATE: FAIL")
