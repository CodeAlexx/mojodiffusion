# minimax_h3_guidance_math_gate — gate the guidance-consistent objective
# affine used by the trainer (c_hat = (g + (s-1)·e)/s, loss vs target, d_g
# scaled 1/s) against torch autograd (h3_guidance_oracle.py, GPU bf16).
# Bars: c_hat cos >= 0.9999, loss rel <= 1e-2, d_g cos >= 0.9999.
from max.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import add, mul_scalar, mul_scalar_bf16out
from serenitymojo.models.minimax_h3.h3_train_sigma import (
    h3_modality_loss, h3_loss_grad,
)

comptime ORACLE = "/home/alex/mojodiffusion/output/checks/h3_guidance_fixture.safetensors"
comptime SCALE = Float32(3.0)


def _load(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _cos(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        raise Error("cos: length mismatch")
    var dot = Float64(0)
    var na = Float64(0)
    var nb = Float64(0)
    for i in range(len(ah)):
        var x = Float64(ah[i])
        var y = Float64(bh[i])
        dot += x * y
        na += x * x
        nb += y * y
    return dot / ((na**0.5) * (nb**0.5) + 1e-30)


def main() raises:
    var ctx = DeviceContext()
    var orc = SafeTensors.open(String(ORACLE))
    var g = _load(orc, String("g"), ctx)
    var e = _load(orc, String("e"), ctx)
    var t = _load(orc, String("t"), ctx)
    var ref_chat = _load(orc, String("c_hat"), ctx)
    var ref_dg = _load(orc, String("d_g"), ctx)
    var ref_loss = Float64(_load(orc, String("loss"), ctx).to_host(ctx)[0])

    # trainer path: F32 combine -> one bf16 round
    var g32 = cast_tensor(g, STDtype.F32, ctx)
    var e32 = cast_tensor(e, STDtype.F32, ctx)
    var chat = cast_tensor(
        add(
            mul_scalar(g32, Float32(1.0) / SCALE, ctx),
            mul_scalar(e32, (SCALE - Float32(1.0)) / SCALE, ctx),
            ctx,
        ),
        STDtype.BF16, ctx,
    )
    var empty_mask = List[Bool]()
    var ml = h3_modality_loss(chat, t, empty_mask, ctx)
    var loss = ml.total / Float64(ml.elements)
    var none_mask = Optional[Tensor](None)
    var d_chat = h3_loss_grad(
        chat, t, none_mask^, 1.0, Float64(ml.elements), ctx
    )
    var d_g = mul_scalar_bf16out(
        cast_tensor(d_chat, STDtype.F32, ctx), Float32(1.0) / SCALE, ctx
    )

    var c_cos = _cos(chat, ref_chat, ctx)
    var g_cos = _cos(d_g, ref_dg, ctx)
    var l_rel = abs(loss - ref_loss) / (abs(ref_loss) + 1e-30)
    print("c_hat cos:", c_cos)
    print("loss ours:", loss, " ref:", ref_loss, " rel:", l_rel)
    print("d_g cos:", g_cos)
    var ok = c_cos >= 0.9999 and g_cos >= 0.9999 and l_rel <= 1.0e-2
    if not ok:
        raise Error("h3 guidance math gate FAIL")
    print("h3 guidance math gate PASS")
