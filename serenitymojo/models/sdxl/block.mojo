# models/sdxl/block.mojo — SDXL ResBlock forward (save acts) + hand-chained
# backward. The first SDXL training compute unit.
#
# This is the conv-UNet analogue of klein/single_block + double_block: a single
# proven fwd+bwd unit, parity-gated before assembly into the full UNet stack.
# It is ARCHITECTURE ONLY — it reuses the parity-gated ops/ backward primitives
# (conv2d_backward, group_norm_backward, silu_backward, linear_backward) and
# builds NO new primitive inline (Tenet 1). Every backward arm it calls already
# has its own ops/parity gate.
#
# ── ResBlock (NHWC, F32), verified vs inference-flame sdxl_unet.rs::resblock ──
#   FORWARD:
#     h1 = GroupNorm(x,   gn1, G, eps=1e-5)          # in_layers.0
#     s1 = SiLU(h1)                                  # in_layers (SiLU)
#     c1 = Conv3x3(s1, conv1, pad=1)                 # in_layers.2
#     e  = SiLU(emb)                                 # emb_layers (SiLU)
#     el = Linear(e, emb_w, emb_b)        -> [N,Cout]
#     h2 = c1 + broadcast(el -> [N,1,1,Cout])        # per-channel time-emb add
#     h3 = GroupNorm(h2, gn2, G, eps=1e-5)           # out_layers.0
#     s2 = SiLU(h3)                                  # out_layers (SiLU)
#     c2 = Conv3x3(s2, conv2, pad=1)                 # out_layers.3
#     r  = Conv1x1(x, skip) if Cin!=Cout else x      # skip_connection
#     out = r + c2
#
#   BACKWARD (go = dL/dout):
#     d_c2 = go ; d_r = go
#     (conv2 bwd)  d_s2, d_conv2_w, d_conv2_b   <- conv2d_backward(s2, conv2_w, d_c2)
#     d_h3         <- silu_backward(d_s2, h3)
#     (gn2 bwd)    d_h2, d_gn2_w, d_gn2_b       <- group_norm_backward(d_h3, h2, gn2_w)
#     d_c1 = d_h2
#     d_el[N,Cout] = sum over (H,W) of d_h2[n,h,w,co]      (broadcast-add bwd)
#     (emb linear bwd) d_e, d_emb_w, d_emb_b    <- linear_backward(d_el, e, emb_w)
#     d_emb_in     <- silu_backward(d_e, emb_in)
#     (conv1 bwd)  d_s1, d_conv1_w, d_conv1_b   <- conv2d_backward(s1, conv1_w, d_c1)
#     d_h1         <- silu_backward(d_s1, h1)
#     (gn1 bwd)    d_x_a, d_gn1_w, d_gn1_b      <- group_norm_backward(d_h1, x, gn1_w)
#     d_x_skip = d_r  (if no skip)  OR  conv2d_backward(x, skip_w, d_r).d_x (if skip)
#     d_x = d_x_a + d_x_skip   (+ skip conv weight/bias grads when present)

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from std.collections import Optional

from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.norm_backward import group_norm_backward
from serenitymojo.ops.activations import silu
from serenitymojo.ops.activation_backward import silu_backward
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.conv2d_backward import conv2d_backward
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.tensor_algebra import add, reshape

from serenitymojo.models.sdxl.weights import ResBlockWeights
from serenitymojo.models.sdxl.config import GN_EPS_RES


# ── saved forward activations (needed by the hand-chained backward) ───────────
struct ResBlockActs(Movable):
    """Activations saved on the forward to drive the backward. F32, NHWC."""
    var x: Tensor       # input            [N,Hi,Wi,Cin]
    var h1: Tensor      # GroupNorm1 out    [N,Hi,Wi,Cin]
    var s1: Tensor      # SiLU(h1)          [N,Hi,Wi,Cin]
    var emb_in: Tensor  # emb input         [N,Eemb]
    var e: Tensor       # SiLU(emb)         [N,Eemb]
    var h2: Tensor      # c1 + emb          [N,Hi,Wi,Cout]
    var h3: Tensor      # GroupNorm2 out    [N,Hi,Wi,Cout]
    var s2: Tensor      # SiLU(h3)          [N,Hi,Wi,Cout]

    def __init__(
        out self, var x: Tensor, var h1: Tensor, var s1: Tensor,
        var emb_in: Tensor, var e: Tensor, var h2: Tensor,
        var h3: Tensor, var s2: Tensor,
    ):
        self.x = x^; self.h1 = h1^; self.s1 = s1^
        self.emb_in = emb_in^; self.e = e^; self.h2 = h2^
        self.h3 = h3^; self.s2 = s2^


struct ResBlockFwd(Movable):
    var out: Tensor
    var acts: ResBlockActs

    def __init__(out self, var out: Tensor, var acts: ResBlockActs):
        self.out = out^
        self.acts = acts^


# ── backward grads (d_input + every per-weight grad) ──────────────────────────
struct ResBlockGrads(Movable):
    var d_x: Tensor
    var d_gn1_w: Tensor
    var d_gn1_b: Tensor
    var d_conv1_w: Tensor
    var d_conv1_b: Tensor
    var d_emb_w: Tensor
    var d_emb_b: Tensor
    var d_gn2_w: Tensor
    var d_gn2_b: Tensor
    var d_conv2_w: Tensor
    var d_conv2_b: Tensor
    var d_emb_in: Tensor   # grad to the (pre-SiLU) emb input [N,Eemb]
    var has_skip: Bool
    var d_skip_w: Tensor
    var d_skip_b: Tensor

    def __init__(
        out self, var d_x: Tensor,
        var d_gn1_w: Tensor, var d_gn1_b: Tensor,
        var d_conv1_w: Tensor, var d_conv1_b: Tensor,
        var d_emb_w: Tensor, var d_emb_b: Tensor,
        var d_gn2_w: Tensor, var d_gn2_b: Tensor,
        var d_conv2_w: Tensor, var d_conv2_b: Tensor,
        var d_emb_in: Tensor,
        has_skip: Bool, var d_skip_w: Tensor, var d_skip_b: Tensor,
    ):
        self.d_x = d_x^
        self.d_gn1_w = d_gn1_w^; self.d_gn1_b = d_gn1_b^
        self.d_conv1_w = d_conv1_w^; self.d_conv1_b = d_conv1_b^
        self.d_emb_w = d_emb_w^; self.d_emb_b = d_emb_b^
        self.d_gn2_w = d_gn2_w^; self.d_gn2_b = d_gn2_b^
        self.d_conv2_w = d_conv2_w^; self.d_conv2_b = d_conv2_b^
        self.d_emb_in = d_emb_in^
        self.has_skip = has_skip
        self.d_skip_w = d_skip_w^; self.d_skip_b = d_skip_b^


# ── reduce d_h2 [N,Ho,Wo,Cout] over spatial -> d_el [N,Cout] (broadcast-add bwd)
# This is the adjoint of `add(c1, broadcast(el->[N,1,1,Cout]))`: each el[n,co]
# fed every spatial position, so its grad sums over them.
def _spatial_sum_to_nc(d_h2: Tensor, N: Int, HW: Int, Cout: Int, ctx: DeviceContext) raises -> Tensor:
    var host = d_h2.to_host(ctx)   # F32, NHWC flat = [(n*HW+pix)*Cout + co]
    var out = List[Float32]()
    for _ in range(N * Cout):
        out.append(0.0)
    for n in range(N):
        for pix in range(HW):
            var base = (n * HW + pix) * Cout
            for co in range(Cout):
                out[n * Cout + co] += host[base + co]
    var s = List[Int]()
    s.append(N); s.append(Cout)
    return Tensor.from_host(out, s^, STDtype.F32, ctx)


# ── FORWARD ───────────────────────────────────────────────────────────────────
# x:   [N,Hi,Wi,Cin] NHWC F32
# emb: [N,Eemb]      F32 (time+label embedding, pre-SiLU)
# Stride 1, pad 1 (3x3); skip is 1x1 pad 0 when Cin != Cout.
def resblock_forward[
    N: Int, Hi: Int, Wi: Int, Cin: Int, Cout: Int, Eemb: Int, G: Int,
](
    x: Tensor, emb: Tensor, w: ResBlockWeights, ctx: DeviceContext,
) raises -> ResBlockFwd:
    comptime eps = GN_EPS_RES

    # in_layers: GN -> SiLU -> Conv3x3
    var h1 = group_norm(x, w.gn1_w, w.gn1_b, G, eps, ctx)
    var s1 = silu(h1, ctx)
    var c1 = conv2d[N, Hi, Wi, Cin, 3, 3, Cout, 1, 1, 1, 1](
        s1, w.conv1_w.clone(ctx), Optional[Tensor](w.conv1_b.clone(ctx)), ctx
    )

    # emb_layers: SiLU -> Linear -> broadcast add
    var e = silu(emb, ctx)
    var el = linear(e, w.emb_w.clone(ctx), Optional[Tensor](w.emb_b.clone(ctx)), ctx)  # [N,Cout]
    var el4 = reshape(el, _nc11(N, Cout), ctx)  # [N,1,1,Cout] for NHWC bcast
    var h2 = add(c1, el4, ctx)                  # [N,Hi,Wi,Cout]

    # out_layers: GN -> SiLU -> Conv3x3
    var h3 = group_norm(h2, w.gn2_w, w.gn2_b, G, eps, ctx)
    var s2 = silu(h3, ctx)
    var c2 = conv2d[N, Hi, Wi, Cout, 3, 3, Cout, 1, 1, 1, 1](
        s2, w.conv2_w.clone(ctx), Optional[Tensor](w.conv2_b.clone(ctx)), ctx
    )

    # skip: 1x1 conv if Cin != Cout, else identity
    var out: Tensor
    if w.has_skip:
        var r = conv2d[N, Hi, Wi, Cin, 1, 1, Cout, 1, 1, 0, 0](
            x, w.skip_w.clone(ctx), Optional[Tensor](w.skip_b.clone(ctx)), ctx
        )
        out = add(r, c2, ctx)
    else:
        out = add(x, c2, ctx)

    var acts = ResBlockActs(x.clone(ctx), h1^, s1^, emb.clone(ctx), e^, h2^, h3^, s2^)
    return ResBlockFwd(out^, acts^)


def _nc11(N: Int, C: Int) -> List[Int]:
    var s = List[Int]()
    s.append(N); s.append(1); s.append(1); s.append(C)
    return s^


# ── BACKWARD ───────────────────────────────────────────────────────────────────
# go: dL/dout [N,Hi,Wi,Cout]
def resblock_backward[
    N: Int, Hi: Int, Wi: Int, Cin: Int, Cout: Int, Eemb: Int, G: Int,
](
    go: Tensor, acts: ResBlockActs, w: ResBlockWeights, ctx: DeviceContext,
) raises -> ResBlockGrads:
    comptime eps = GN_EPS_RES
    comptime HW = Hi * Wi

    # out = r + c2  ->  d_c2 = go,  d_r = go
    # conv2 bwd (3x3 pad1): inputs s2 [N,Hi,Wi,Cout], filter conv2_w [3,3,Cout,Cout].
    # Fully decompose each grad struct into locals (move all 3 fields out) so no
    # struct is left partially-moved (Mojo destroy-whole-value rule).
    # NOTE: the grad structs (Conv2dBwd / GroupNormBackward / LinearGrads) are
    # Movable-only with an auto-destructor that drops ALL fields together; a
    # partial `^` field move breaks that. So we .clone(ctx) every field we keep
    # and let the struct auto-drop — the proven idiom (klein_stack/checkpoint).
    var g_c2 = conv2d_backward[N, Hi, Wi, Cout, 3, 3, Cout, 1, 1, 1, 1](
        acts.s2, w.conv2_w, go, ctx
    )
    var d_conv2_w = g_c2.d_w.clone(ctx)
    var d_conv2_b = g_c2.d_b.clone(ctx)
    var d_h3 = silu_backward(g_c2.d_x, acts.h3, ctx)

    # gn2 bwd: input h2 [N,Hi,Wi,Cout], weight gn2_w [Cout]
    var g_gn2 = group_norm_backward(d_h3, acts.h2, w.gn2_w, G, eps, ctx)
    var d_h2 = g_gn2.d_x.clone(ctx)   # [N,Hi,Wi,Cout]  (= d_c1)
    var d_gn2_w = g_gn2.d_g.clone(ctx)
    var d_gn2_b = g_gn2.d_b.clone(ctx)

    # d_h2 splits: -> d_c1 (= d_h2) AND -> d_el (spatial-sum of d_h2)
    var d_el = _spatial_sum_to_nc(d_h2, N, HW, Cout, ctx)   # [N,Cout]
    # emb linear bwd: y=el [N,Cout], x=e [N,Eemb], W=emb_w [Cout,Eemb]
    var g_emb = linear_backward(d_el, acts.e, w.emb_w, N, Eemb, Cout, ctx)
    var d_emb_w = g_emb.d_w.clone(ctx)
    var d_emb_b = g_emb.d_b.clone(ctx)
    var d_emb_in = silu_backward(g_emb.d_x, acts.emb_in, ctx)  # [N,Eemb]

    # conv1 bwd (3x3 pad1): input s1 [N,Hi,Wi,Cin], filter conv1_w [3,3,Cin,Cout],
    # upstream grad d_c1 = d_h2 [N,Hi,Wi,Cout]
    var g_c1 = conv2d_backward[N, Hi, Wi, Cin, 3, 3, Cout, 1, 1, 1, 1](
        acts.s1, w.conv1_w, d_h2, ctx
    )
    var d_conv1_w = g_c1.d_w.clone(ctx)
    var d_conv1_b = g_c1.d_b.clone(ctx)
    var d_h1 = silu_backward(g_c1.d_x, acts.h1, ctx)

    # gn1 bwd: input x [N,Hi,Wi,Cin], weight gn1_w [Cin]
    var g_gn1 = group_norm_backward(d_h1, acts.x, w.gn1_w, G, eps, ctx)
    var d_x_main = g_gn1.d_x.clone(ctx)   # [N,Hi,Wi,Cin]
    var d_gn1_w = g_gn1.d_g.clone(ctx)
    var d_gn1_b = g_gn1.d_b.clone(ctx)

    # skip branch: d_r = go
    var d_x: Tensor
    if w.has_skip:
        var g_skip = conv2d_backward[N, Hi, Wi, Cin, 1, 1, Cout, 1, 1, 0, 0](
            acts.x, w.skip_w, go, ctx
        )
        var d_skip_w = g_skip.d_w.clone(ctx)
        var d_skip_b = g_skip.d_b.clone(ctx)
        d_x = add(d_x_main, g_skip.d_x, ctx)
        return ResBlockGrads(
            d_x^, d_gn1_w^, d_gn1_b^, d_conv1_w^, d_conv1_b^,
            d_emb_w^, d_emb_b^, d_gn2_w^, d_gn2_b^,
            d_conv2_w^, d_conv2_b^, d_emb_in^,
            True, d_skip_w^, d_skip_b^,
        )
    # identity skip: d_x += d_r = go (out = x + c2)
    d_x = add(d_x_main, go, ctx)
    var ph_w = _zeros1(ctx)
    var ph_b = _zeros1(ctx)
    return ResBlockGrads(
        d_x^, d_gn1_w^, d_gn1_b^, d_conv1_w^, d_conv1_b^,
        d_emb_w^, d_emb_b^, d_gn2_w^, d_gn2_b^,
        d_conv2_w^, d_conv2_b^, d_emb_in^,
        False, ph_w^, ph_b^,
    )


def _zeros1(ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    h.append(0.0)
    var s = List[Int]()
    s.append(1)
    return Tensor.from_host(h, s^, STDtype.F32, ctx)
