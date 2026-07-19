# asymflow.mojo — AsymFLUX.2 velocity reconstruction (the ASYMMETRIC layer).
#
# Pure-Mojo port of EriDiffusion `inference-flame/src/models/asymflux2.rs`
# (LakonLab `AsymFlowMixin`, Apache 2.0). This is the "asymmetric" component
# of AsymFLUX.2 Klein 9B: a STATELESS math layer that wraps a base Klein-9B
# transformer's `proj_out` to reconstruct a full-rank flow-matching velocity
# `u` from the transformer's *asymmetric* velocity output `u_a`.
#
# WHAT IS ASYMMETRIC (the actual delta vs plain Klein):
#   The Klein backbone is UNCHANGED — same double/single blocks, same RoPE,
#   same SHARED modulation. The asymmetry lives ENTIRELY in this wrapper:
#   the transformer no longer predicts the full velocity. It predicts an
#   "asymmetric" velocity `u_a` whose low-rank subspace (the column space of
#   a fixed projection P, `proj_buffer (D=768, R=128)`) is treated DIFFERENTLY
#   from its orthogonal complement:
#     subspace:   u = s·k·u_a_sub + (1-s·k)/σ · x_t_sub      (calibrated mix)
#     complement: u = (x_t_comp + s·u_a_comp) / σ            (rescaled)
#   i.e. inside P's span the velocity is a calibrated blend of the network
#   output and the data term, while OUTSIDE P's span it is a plain rescale.
#   That branch-asymmetry between the rank-R subspace and its complement is
#   the AsymFlow construction. There is NO asymmetric ATTENTION or asymmetric
#   transformer BLOCK — the Klein DiT (models/dit/klein_dit.mojo, already
#   ported + gated) is reused verbatim as the `u_a` producer.
#
# DTYPE: F32 throughout (the reference disables autocast for the whole
# velocity reconstruction). Caller passes any dtype; we cast to F32 and
# return F32. Match Rust asymflux2.rs::asymflow_velocity exactly.
#
# Mojo 1.0.0b1, GPU-only compute.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor_if_needed
from serenitymojo.ops.linear import linear
from serenitymojo.ops.tensor_algebra import add, sub, mul_scalar, transpose
from serenitymojo.ops.layout import patchify, unpatchify


# ─────────────────────────────────────────────────────────────────────────────
# Calibration (host-scalar; matches asymflux2.rs::compute_calibration)
# ─────────────────────────────────────────────────────────────────────────────
struct AsymFlowCalibration(Copyable, Movable):
    """Per-step calibration. All host f32 scalars (B=1 inference)."""

    var s: Float32  # scale_buffer
    var k: Float32  # k = 1 / (s + (1-s)·σ)
    var cal_timestep: Float32  # timestep · k
    var sigma: Float32  # timestep / num_timesteps

    def __init__(out self, s: Float32, k: Float32, cal_timestep: Float32, sigma: Float32):
        self.s = s
        self.k = k
        self.cal_timestep = cal_timestep
        self.sigma = sigma


def compute_calibration(
    timestep: Float32, scale_buffer: Float32, num_timesteps: Float32
) -> AsymFlowCalibration:
    """sigma = timestep/num_timesteps ; k = 1/(s+(1-s)·sigma) ; cal_t = t·k.

    Mirrors asymflux2.rs::compute_calibration (no input validation — bad
    calibrations propagate as non-finite values to surface upstream bugs)."""
    var sigma = timestep / num_timesteps
    var k = 1.0 / (scale_buffer + (1.0 - scale_buffer) * sigma)
    return AsymFlowCalibration(scale_buffer, k, timestep * k, sigma)


def _clamp_sigma_nan_preserving(sigma: Float32, sigma_min: Float32) -> Float32:
    """NaN-preserving lower clamp (asymflux2.rs::clamp_sigma_nan_preserving)."""
    if sigma != sigma:  # NaN check
        return sigma
    if sigma < sigma_min:
        return sigma_min
    return sigma


# ─────────────────────────────────────────────────────────────────────────────
# Orthogonal decomposition along the column space of P
# ─────────────────────────────────────────────────────────────────────────────
def _subspace(
    state: Tensor,  # (B, hw, D) F32
    p: Tensor,  # (D, R) F32  — projection columns
    p_t: Tensor,  # (R, D) F32  — materialized transpose
    ctx: DeviceContext,
) raises -> Tensor:
    """subspace = state @ P @ P^T   (B, hw, D).

    `linear(x, W)` computes x @ W^T. So:
      proj     = state @ P     = linear(state, p_t)   [W=p_t=(R,D) ⇒ W^T=(D,R)=P]
      subspace = proj  @ P^T   = linear(proj,  p)     [W=p=(D,R)   ⇒ W^T=(R,D)=P^T]
    The complement is recovered by the caller as state - subspace.
    """
    var proj = linear(state, p_t, None, ctx)  # (B, hw, R)
    return linear(proj, p, None, ctx)  # (B, hw, D)


# ─────────────────────────────────────────────────────────────────────────────
# AsymFlow velocity reconstruction
# ─────────────────────────────────────────────────────────────────────────────
def asymflow_velocity(
    u_a_packed: Tensor,  # (B, hw, D) — transformer proj_out, any dtype
    x_t_packed: Tensor,  # (B, hw, D) — packed pre-x_embedder pixels, any dtype
    cal: AsymFlowCalibration,
    p: Tensor,  # (D, R) projection buffer, any dtype
    sigma_min: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """Full flow-matching velocity (F32). Port of asymflux2.rs::asymflow_velocity.

      sk          = s·k
      σ_c         = clamp(σ, σ_min)              (NaN-preserving)
      u_sub       = sk·u_a_sub + (1-sk)/σ_c · x_t_sub
      u_comp      = (x_t_comp + s·u_a_comp) / σ_c
      u           = u_sub + u_comp
    """
    var u_a = cast_tensor_if_needed(u_a_packed.clone(ctx), STDtype.F32, ctx)
    var x_t = cast_tensor_if_needed(x_t_packed.clone(ctx), STDtype.F32, ctx)
    var p_f32 = cast_tensor_if_needed(p.clone(ctx), STDtype.F32, ctx)

    # Materialize P^T = (R, D) once (P is (D, R)). P is a fixed buffer so this
    # one-time transpose is not in any hot loop.
    var p_t = transpose(p_f32, 0, 1, ctx)

    var u_a_sub = _subspace(u_a, p_f32, p_t, ctx)
    var u_a_comp = sub(u_a, u_a_sub, ctx)
    var x_t_sub = _subspace(x_t, p_f32, p_t, ctx)
    var x_t_comp = sub(x_t, x_t_sub, ctx)

    var sk = cal.s * cal.k
    var sigma_clamped = _clamp_sigma_nan_preserving(cal.sigma, sigma_min)
    var inv_sigma = 1.0 / sigma_clamped

    # Low-rank subspace velocity.
    var term1 = mul_scalar(u_a_sub, sk, ctx)
    var coef2 = (1.0 - sk) * inv_sigma
    var term2 = mul_scalar(x_t_sub, coef2, ctx)
    var u_sub = add(term1, term2, ctx)

    # Orthogonal complement velocity.
    var s_u_a_comp = mul_scalar(u_a_comp, cal.s, ctx)
    var comp_sum = add(x_t_comp, s_u_a_comp, ctx)
    var u_comp = mul_scalar(comp_sum, inv_sigma, ctx)

    return add(u_sub, u_comp, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Full per-step chain (CHUNK B). Mirrors asymflux2_klein9b_infer::wrapped_forward
# minus the transformer call, which is a SEPARATELY-GATED component
# (models/dit/klein_dit.mojo, cos>=0.999). The asymmetric module owns only the
# pre/post packing + the velocity reconstruction; the caller supplies `u_a`
# (the Klein transformer's proj_out at the k-scaled, BF16-cast hidden states).
#
# Chain (Rust order, ops/layout.patchify already fuses patchify+pack into the
# (B, hw, C·p²) sequence the transformer consumes):
#   x_t_packed   = patchify(x_t_pixel, p)               (B, hw, 768)
#   cal          = compute_calibration(t, s, num_ts)
#   hidden       = x_t_packed · k   (then BF16 → Klein → u_a, by the caller)
#   velocity     = asymflow_velocity(u_a, x_t_packed, cal, P, σ_min)  (F32)
#   v_pixel      = unpatchify(velocity, 3, H, W, p)     (B, 3, H, W)
# ─────────────────────────────────────────────────────────────────────────────
def pixel_to_packed(
    x_t_pixel: Tensor, patch: Int, ctx: DeviceContext
) raises -> Tensor:
    """patchify+pack the (B,3,H,W) pixel tensor to (B, hw, 3·p²)."""
    return patchify(x_t_pixel, patch, ctx)


def scale_for_embedder(
    x_t_packed: Tensor, cal: AsymFlowCalibration, ctx: DeviceContext
) raises -> Tensor:
    """hidden_states = x_t_packed · k (pre-x_embedder rescale). F32 in/out;
    the caller casts to BF16 before the Klein forward."""
    return mul_scalar(x_t_packed, cal.k, ctx)


def velocity_to_pixel(
    velocity_packed: Tensor, height: Int, width: Int, patch: Int, ctx: DeviceContext
) raises -> Tensor:
    """unpack+unpatchify (B, hw, 3·p²) F32 velocity back to (B, 3, H, W)."""
    return unpatchify(velocity_packed, 3, height, width, patch, ctx)
