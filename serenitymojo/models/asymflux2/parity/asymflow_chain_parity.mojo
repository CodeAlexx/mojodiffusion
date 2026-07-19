# asymflow_chain_parity.mojo — CHUNK B plumbing gate for the full AsymFLUX.2
# per-step chain (asymflux2_klein9b_infer::wrapped_forward), EXCLUDING the
# Klein transformer call (a separately-gated component, cos>=0.999, see
# models/klein/parity/klein_stack_parity.mojo).
#
# What this gates: the pixel→patchify→·k→[transformer]→asymflow_velocity→
# unpatchify data path. To exercise it numerically without the 18GB Klein
# weights (and with NO end-to-end reference on disk — the Rust binary itself
# never ran E2E; its adapter key-mapping translator is unwritten), we use an
# IDENTITY stand-in transformer: u_a = hidden_states (k-scaled packed pixels).
# The whole chain is then re-derived independently on the host and compared.
#
# ORACLE INDEPENDENCE: SAME-SOURCE (algebra from one reference) AND the
# transformer is stubbed to identity here, so this is a PLUMBING gate, not a
# weighted end-to-end numeric gate. The weighted E2E CHUNK B is BLOCKED on the
# missing reference (see report).
#
# Run:
#   cd /home/alex/mojodiffusion
#   pixi run mojo run -I . serenitymojo/models/asymflux2/parity/asymflow_chain_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.models.asymflux2.asymflow import (
    compute_calibration,
    asymflow_velocity,
    pixel_to_packed,
    scale_for_embedder,
    velocity_to_pixel,
)


comptime H = 16  # pixels
comptime W = 16
comptime P = 16  # patch -> 1x1 grid, hw=1, F = 3*16*16 = 768
comptime C = 3
comptime D = 768  # packed feature dim = C*P*P
comptime HW = (H // P) * (W // P)  # = 1
comptime R = 4  # low-rank
comptime SIGMA_MIN = Float32(1e-4)


def _fill(n: Int, seed: UInt32) -> List[Float32]:
    var state = seed
    var out = List[Float32]()
    for _ in range(n):
        state = (state * 48271) % 0x7FFFFFFF
        out.append(Float32(Int(state)) / Float32(0x7FFFFFFF) * 2.0 - 1.0)
    return out^


def _shape(*d: Int) -> List[Int]:
    var s = List[Int]()
    for i in range(len(d)):
        s.append(d[i])
    return s^


# Host patchify matching ops/layout: [C,H,W] -> [hw, C*p*p], within-patch
# flatten (c, ph, pw). With H=W=P, hw=1 and the packed vector is just the
# channel-major raster of the whole image: f = (c*P + ph)*P + pw.
def _host_patchify(img: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for c in range(C):
        for ph in range(P):
            for pw in range(P):
                out.append(img[c * H * W + ph * W + pw])
    return out^


def _host_subspace(state: List[Float32], p: List[Float32]) -> List[Float32]:
    # proj = state(1,D) @ P(D,R) -> (1,R)
    var proj = List[Float32]()
    for j in range(R):
        var acc = Float32(0.0)
        for di in range(D):
            acc += state[di] * p[di * R + j]
        proj.append(acc)
    # sub = proj @ P^T -> (1,D)
    var sub = List[Float32]()
    for di in range(D):
        var acc = Float32(0.0)
        for j in range(R):
            acc += proj[j] * p[di * R + j]
        sub.append(acc)
    return sub^


def main() raises:
    var ctx = DeviceContext()

    var img_h = _fill(C * H * W, 0xABCD01)
    var p_h = _fill(D * R, 0x55AA33)
    var cal = compute_calibration(0.42, 0.8, 1.0)

    # ── Mojo chain (identity stand-in: u_a = k-scaled packed pixels) ──────────
    var x_pixel = Tensor.from_host(img_h, _shape(1, C, H, W), STDtype.F32, ctx)
    var x_packed = pixel_to_packed(x_pixel, P, ctx)  # (1, 1, 768)
    var hidden = scale_for_embedder(x_packed, cal, ctx)  # u_a := hidden (identity)
    var p = Tensor.from_host(p_h, _shape(D, R), STDtype.F32, ctx)
    var velocity = asymflow_velocity(hidden, x_packed, cal, p, SIGMA_MIN, ctx)
    var v_pixel = velocity_to_pixel(velocity, H, W, P, ctx)  # (1, 3, H, W)
    var actual = v_pixel.to_host(ctx)

    # ── independent host re-derivation of the whole chain ─────────────────────
    var x_packed_h = _host_patchify(img_h)
    var u_a_h = List[Float32]()  # identity stand-in: k-scaled packed pixels
    for i in range(D):
        u_a_h.append(x_packed_h[i] * cal.k)
    var u_a_sub = _host_subspace(u_a_h, p_h)
    var x_t_sub = _host_subspace(x_packed_h, p_h)
    var sk = cal.s * cal.k
    var sigma_c = cal.sigma if cal.sigma >= SIGMA_MIN else SIGMA_MIN
    var inv = 1.0 / sigma_c
    var coef2 = (1.0 - sk) * inv
    # velocity (packed) then unpatchify (inverse of _host_patchify, same order)
    var vel_packed = List[Float32]()
    for i in range(D):
        var u_a_comp = u_a_h[i] - u_a_sub[i]
        var x_t_comp = x_packed_h[i] - x_t_sub[i]
        var u_sub = sk * u_a_sub[i] + coef2 * x_t_sub[i]
        var u_comp = (x_t_comp + cal.s * u_a_comp) * inv
        vel_packed.append(u_sub + u_comp)
    # unpatchify: f = (c*P+ph)*P+pw  ->  img[c,ph,pw]; here hw=1 so identity map
    var reference = List[Float32]()
    for c in range(C):
        for ph in range(P):
            for pw in range(P):
                reference.append(vel_packed[(c * P + ph) * P + pw])

    var harness = ParityHarness(0.99)
    var res = harness.compare_host(actual, reference)
    print("ASYMFLOW_CHAIN_PARITY", res)
    if res.passed:
        print("GATE: PASS")
    else:
        print("GATE: FAIL")
