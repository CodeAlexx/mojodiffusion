# models/sdxl/geglu.mojo — SDXL SpatialTransformer FF GEGLU fwd+bwd.
#
# ARCHITECTURE ONLY (Tenet 1): composes already-gated ops/ primitives —
# linear / linear_backward, slice (fwd split), split_backward (bwd join),
# gelu / gelu_backward (TANH-approx, matching flame-core Tensor::gelu which uses
# the tanh kernel, sdxl_unet.rs:649 gate_part.gelu()), mul. No new primitive.
#
# ── FORWARD (sdxl_unet.rs geglu, lines 637-655) ───────────────────────────────
#   proj = Linear(x, proj_w, proj_b)      # [M, 2*Cff]   (ff.net.0.proj)
#   x_part = proj[:, :Cff]                 # first half
#   gate   = proj[:, Cff:]                 # second half
#   g      = gelu(gate)                    # tanh-approx GELU
#   out    = x_part * g                    # [M, Cff]
#
# ── BACKWARD (d_out = dL/dout [M,Cff]) ────────────────────────────────────────
#   out = x_part * g
#     d_x_part = d_out * g
#     d_g      = d_out * x_part
#   g = gelu(gate)
#     d_gate   = gelu_backward(d_g, gate)
#   proj = cat([x_part, gate], axis=1) (the forward split's adjoint)
#     d_proj   = split_backward(d_x_part, d_gate, axis=1)   # [M, 2*Cff]
#   proj = Linear(x, proj_w, proj_b)
#     (d_x, d_proj_w, d_proj_b) = linear_backward(d_proj, x, proj_w)
#
# All F32. M = flattened tokens (B*N).

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from std.collections import Optional

from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.activations import gelu
from serenitymojo.ops.activation_backward import gelu_backward
from serenitymojo.ops.tensor_algebra import mul, slice
from serenitymojo.ops.shape_backward import split_backward


struct GegluActs(Movable):
    """Saved forward activations for the GEGLU backward."""
    var x: Tensor        # input        [M, Cin]
    var x_part: Tensor   # proj first half  [M, Cff]
    var gate: Tensor     # proj second half [M, Cff]
    var g: Tensor        # gelu(gate)    [M, Cff]

    def __init__(out self, var x: Tensor, var x_part: Tensor, var gate: Tensor, var g: Tensor):
        self.x = x^; self.x_part = x_part^; self.gate = gate^; self.g = g^


struct GegluFwd(Movable):
    var out: Tensor
    var acts: GegluActs
    def __init__(out self, var out: Tensor, var acts: GegluActs):
        self.out = out^; self.acts = acts^


struct GegluGrads(Movable):
    var d_x: Tensor
    var d_proj_w: Tensor
    var d_proj_b: Tensor
    def __init__(out self, var d_x: Tensor, var d_proj_w: Tensor, var d_proj_b: Tensor):
        self.d_x = d_x^; self.d_proj_w = d_proj_w^; self.d_proj_b = d_proj_b^


# ── FORWARD ───────────────────────────────────────────────────────────────────
# x: [M, Cin]. proj_w: [2*Cff, Cin]. proj_b: [2*Cff]. returns out [M, Cff].
def geglu_forward[
    M: Int, Cin: Int, Cff: Int,
](
    x: Tensor, proj_w: Tensor, proj_b: Tensor, ctx: DeviceContext,
) raises -> GegluFwd:
    var proj = linear(x.clone(ctx), proj_w.clone(ctx), Optional[Tensor](proj_b.clone(ctx)), ctx)  # [M,2*Cff]
    var x_part = slice(proj, 1, 0, Cff, ctx)      # [M,Cff]
    var gate = slice(proj, 1, Cff, Cff, ctx)      # [M,Cff]
    var g = gelu(gate.clone(ctx), ctx)
    var out = mul(x_part.clone(ctx), g.clone(ctx), ctx)   # [M,Cff]
    var acts = GegluActs(x.clone(ctx), x_part^, gate^, g^)
    return GegluFwd(out^, acts^)


# ── BACKWARD ───────────────────────────────────────────────────────────────────
# d_out: [M, Cff]
def geglu_backward[
    M: Int, Cin: Int, Cff: Int,
](
    d_out: Tensor, acts: GegluActs, proj_w: Tensor, ctx: DeviceContext,
) raises -> GegluGrads:
    # out = x_part * g
    var d_x_part = mul(d_out.clone(ctx), acts.g, ctx)       # d_out * gelu(gate)
    var d_g = mul(d_out.clone(ctx), acts.x_part, ctx)       # d_out * x_part
    # g = gelu(gate) -> d_gate = gelu_backward(d_g, gate)
    var d_gate = gelu_backward(d_g, acts.gate, ctx)         # [M,Cff]
    # split adjoint: cat the two halves back into d_proj [M,2*Cff] (axis=1)
    var d_proj = split_backward(d_x_part, d_gate, 1, ctx)   # [M,2*Cff]
    # proj = Linear(x) -> d_x + d_proj_w + d_proj_b
    var g_lin = linear_backward(d_proj, acts.x, proj_w, M, Cin, 2 * Cff, ctx)
    return GegluGrads(g_lin.d_x.clone(ctx), g_lin.d_w.clone(ctx), g_lin.d_b.clone(ctx))
