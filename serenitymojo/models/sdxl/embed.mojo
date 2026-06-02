# models/sdxl/embed.mojo — SDXL time + label (ADM) embedding fwd+bwd.
#
# ARCHITECTURE ONLY (Tenet 1): composes already-gated ops/ primitives —
# timestep_embedding (sinusoidal fwd, parameter-free), linear / linear_backward,
# silu / silu_backward, add. No new primitive inline.
#
# ── FORWARD (sdxl_unet.rs time_embed / label_embed, lines 511-528) ────────────
#   sinusoidal(t, dim=320, COS-first LDM)  ->  ts [B,320]   (parameter-free)
#   te = Linear(ts, t0_w, t0_b)            ->  [B,1280]      time_embed.0
#   te = SiLU(te)
#   te = Linear(te, t2_w, t2_b)            ->  [B,1280]      time_embed.2
#   le = Linear(y,  l0_w, l0_b)            ->  [B,1280]      label_emb.0.0
#   le = SiLU(le)
#   le = Linear(le, l2_w, l2_b)            ->  [B,1280]      label_emb.0.2
#   emb = te + le                          ->  [B,1280]      shared ResBlock emb
#
# ── BACKWARD (d_emb = dL/demb [B,1280]) ───────────────────────────────────────
#   emb = te + le  ->  d_te = d_emb,  d_le = d_emb   (sum-junction)
#   time MLP bwd:  d_te -> (lin2 bwd) d_t0silu, dt2_w, dt2_b
#                  -> silu_backward -> d_t0lin
#                  -> (lin0 bwd) d_ts, dt0_w, dt0_b
#     (d_ts is the grad wrt the sinusoidal vector; the sinusoidal map t->ts is a
#      fixed function of the non-trained scalar t, so backward stops here — same
#      as the Rust forward, where t is a constant input.)
#   label MLP bwd: d_le -> (lin2 bwd) d_l0silu, dl2_w, dl2_b
#                  -> silu_backward -> d_l0lin
#                  -> (lin0 bwd) d_y, dl0_w, dl0_b   (d_y = grad wrt the ADM vector)
#
# All F32. The sinusoidal dim = model_channels (320); time_embed_dim = 1280.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from std.collections import Optional

from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.activations import silu
from serenitymojo.ops.activation_backward import silu_backward
from serenitymojo.ops.tensor_algebra import add


# ── weights for the two embedding MLPs ────────────────────────────────────────
struct EmbWeights(Movable):
    """time_embed.{0,2} + label_emb.0.{0,2} linear weights/biases (F32).
    Weight layout [out_features, in_features] (torch nn.Linear)."""
    var t0_w: Tensor; var t0_b: Tensor   # time_embed.0  [1280,320]
    var t2_w: Tensor; var t2_b: Tensor   # time_embed.2  [1280,1280]
    var l0_w: Tensor; var l0_b: Tensor   # label_emb.0.0 [1280,2816]
    var l2_w: Tensor; var l2_b: Tensor   # label_emb.0.2 [1280,1280]

    def __init__(
        out self,
        var t0_w: Tensor, var t0_b: Tensor, var t2_w: Tensor, var t2_b: Tensor,
        var l0_w: Tensor, var l0_b: Tensor, var l2_w: Tensor, var l2_b: Tensor,
    ):
        self.t0_w = t0_w^; self.t0_b = t0_b^; self.t2_w = t2_w^; self.t2_b = t2_b^
        self.l0_w = l0_w^; self.l0_b = l0_b^; self.l2_w = l2_w^; self.l2_b = l2_b^


struct EmbActs(Movable):
    """Saved forward activations for the embedding backward."""
    var ts: Tensor       # sinusoidal [B,Sdim]
    var t0lin: Tensor    # time Linear0 out (pre-SiLU) [B,1280]
    var t0silu: Tensor   # SiLU(t0lin) [B,1280]
    var y: Tensor        # ADM input [B,Adm]
    var l0lin: Tensor    # label Linear0 out (pre-SiLU) [B,1280]
    var l0silu: Tensor   # SiLU(l0lin) [B,1280]

    def __init__(
        out self, var ts: Tensor, var t0lin: Tensor, var t0silu: Tensor,
        var y: Tensor, var l0lin: Tensor, var l0silu: Tensor,
    ):
        self.ts = ts^; self.t0lin = t0lin^; self.t0silu = t0silu^
        self.y = y^; self.l0lin = l0lin^; self.l0silu = l0silu^


struct EmbFwd(Movable):
    var emb: Tensor
    var acts: EmbActs
    def __init__(out self, var emb: Tensor, var acts: EmbActs):
        self.emb = emb^; self.acts = acts^


struct EmbGrads(Movable):
    var dt0_w: Tensor; var dt0_b: Tensor
    var dt2_w: Tensor; var dt2_b: Tensor
    var dl0_w: Tensor; var dl0_b: Tensor
    var dl2_w: Tensor; var dl2_b: Tensor
    var d_ts: Tensor    # grad wrt sinusoidal vector (time-MLP input)
    var d_y: Tensor     # grad wrt ADM vector

    def __init__(
        out self,
        var dt0_w: Tensor, var dt0_b: Tensor, var dt2_w: Tensor, var dt2_b: Tensor,
        var dl0_w: Tensor, var dl0_b: Tensor, var dl2_w: Tensor, var dl2_b: Tensor,
        var d_ts: Tensor, var d_y: Tensor,
    ):
        self.dt0_w = dt0_w^; self.dt0_b = dt0_b^; self.dt2_w = dt2_w^; self.dt2_b = dt2_b^
        self.dl0_w = dl0_w^; self.dl0_b = dl0_b^; self.dl2_w = dl2_w^; self.dl2_b = dl2_b^
        self.d_ts = d_ts^; self.d_y = d_y^


# ── FORWARD ───────────────────────────────────────────────────────────────────
# t: [B] scalar timesteps (F32). y: [B,Adm] ADM vector (F32).
def embed_forward[
    B: Int, Sdim: Int, Tdim: Int, Adm: Int,
](
    t: Tensor, y: Tensor, w: EmbWeights, ctx: DeviceContext,
) raises -> EmbFwd:
    # time: sinusoidal -> Linear0 -> SiLU -> Linear2
    var ts = timestep_embedding(t, Sdim, ctx, Float32(10000.0))   # [B,Sdim]
    var t0lin = linear(ts.clone(ctx), w.t0_w.clone(ctx), Optional[Tensor](w.t0_b.clone(ctx)), ctx)
    var t0silu = silu(t0lin, ctx)
    var te = linear(t0silu.clone(ctx), w.t2_w.clone(ctx), Optional[Tensor](w.t2_b.clone(ctx)), ctx)

    # label: Linear0 -> SiLU -> Linear2
    var l0lin = linear(y.clone(ctx), w.l0_w.clone(ctx), Optional[Tensor](w.l0_b.clone(ctx)), ctx)
    var l0silu = silu(l0lin, ctx)
    var le = linear(l0silu.clone(ctx), w.l2_w.clone(ctx), Optional[Tensor](w.l2_b.clone(ctx)), ctx)

    var emb = add(te, le, ctx)   # [B,Tdim]
    var acts = EmbActs(ts^, t0lin^, t0silu^, y.clone(ctx), l0lin^, l0silu^)
    return EmbFwd(emb^, acts^)


# ── BACKWARD ───────────────────────────────────────────────────────────────────
# d_emb: [B,Tdim]
def embed_backward[
    B: Int, Sdim: Int, Tdim: Int, Adm: Int,
](
    d_emb: Tensor, acts: EmbActs, w: EmbWeights, ctx: DeviceContext,
) raises -> EmbGrads:
    # emb = te + le -> d_te = d_le = d_emb
    # time MLP bwd
    var g_t2 = linear_backward(d_emb, acts.t0silu, w.t2_w, B, Tdim, Tdim, ctx)
    var dt2_w = g_t2.d_w.clone(ctx)
    var dt2_b = g_t2.d_b.clone(ctx)
    var d_t0lin = silu_backward(g_t2.d_x, acts.t0lin, ctx)
    var g_t0 = linear_backward(d_t0lin, acts.ts, w.t0_w, B, Sdim, Tdim, ctx)
    var dt0_w = g_t0.d_w.clone(ctx)
    var dt0_b = g_t0.d_b.clone(ctx)
    var d_ts = g_t0.d_x.clone(ctx)

    # label MLP bwd
    var g_l2 = linear_backward(d_emb, acts.l0silu, w.l2_w, B, Tdim, Tdim, ctx)
    var dl2_w = g_l2.d_w.clone(ctx)
    var dl2_b = g_l2.d_b.clone(ctx)
    var d_l0lin = silu_backward(g_l2.d_x, acts.l0lin, ctx)
    var g_l0 = linear_backward(d_l0lin, acts.y, w.l0_w, B, Adm, Tdim, ctx)
    var dl0_w = g_l0.d_w.clone(ctx)
    var dl0_b = g_l0.d_b.clone(ctx)
    var d_y = g_l0.d_x.clone(ctx)

    return EmbGrads(
        dt0_w^, dt0_b^, dt2_w^, dt2_b^,
        dl0_w^, dl0_b^, dl2_w^, dl2_b^,
        d_ts^, d_y^,
    )
