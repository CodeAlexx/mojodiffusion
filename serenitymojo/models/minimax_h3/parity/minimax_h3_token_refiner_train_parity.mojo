# MiniMax-H3 token-refiner LoRA training parity against Torch autograd.
# Gates forward, d_x, and all eight A/B gradients at cos >= 0.999.
from std.collections import List
from std.memory import ArcPointer
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.klein.lora_block import (
    KleinLoraDeviceGradTensors, LoraAdapterDevice,
)
from serenitymojo.models.minimax_h3.h3_block_train import (
    H3BlockLoraDevice, H3BlockTrainWeights,
)
from serenitymojo.models.minimax_h3.h3_token_refiner_train import (
    h3_token_refiner_block_backward_lora,
    h3_token_refiner_block_forward_lora,
)

comptime ORACLE = "/home/alex/mojodiffusion/output/checks/h3_token_refiner_oracle.safetensors"
comptime S = 31
comptime H = 2
comptime Dh = 16
comptime D = 24
comptime F = 32
comptime RANK = 4
comptime EPS = Float32(1.0e-5)


def _load(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    return Tensor.from_view(
        from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name)), ctx
    )


def _bf16(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(_load(st, name, ctx), STDtype.BF16, ctx)


def _f32(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(_load(st, name, ctx), STDtype.F32, ctx)


def _cos(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    if len(ah) != len(bh):
        raise Error("token-refiner parity length mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(ah)):
        var x = Float64(ah[i])
        var y = Float64(bh[i])
        dot += x * y
        na += x * x
        nb += y * y
    return dot / ((na**0.5) * (nb**0.5) + 1.0e-30)


def main() raises:
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(ORACLE))
    var w = H3BlockTrainWeights(
        ArcPointer(_bf16(st, String("qkv_w"), ctx)),
        ArcPointer(_bf16(st, String("out_w"), ctx)),
        ArcPointer(_bf16(st, String("fc1_w_runtime"), ctx)),
        ArcPointer(_bf16(st, String("fc2_w"), ctx)),
        ArcPointer(_bf16(st, String("q_norm"), ctx)),
        ArcPointer(_bf16(st, String("k_norm"), ctx)),
        ArcPointer(_bf16(st, String("norm1"), ctx)),
        ArcPointer(_bf16(st, String("norm2"), ctx)),
    )

    var names: List[String] = [
        String("qkv"), String("out"), String("fc1"), String("fc2")
    ]
    var ins: List[Int] = [D, H * Dh, D, F]
    var outs: List[Int] = [3 * H * Dh, D, 2 * F, D]
    var adapters = List[LoraAdapterDevice]()
    for i in range(4):
        adapters.append(LoraAdapterDevice(
            ArcPointer(_bf16(st, names[i] + String("_a"), ctx)),
            ArcPointer(_bf16(st, names[i] + String("_b"), ctx)),
            RANK, ins[i], outs[i], Float32(1.0),
        ))
    var lora = H3BlockLoraDevice(
        Optional[LoraAdapterDevice](adapters[0].copy()),
        Optional[LoraAdapterDevice](adapters[1].copy()),
        Optional[LoraAdapterDevice](adapters[2].copy()),
        Optional[LoraAdapterDevice](adapters[3].copy()),
    )

    var x = _bf16(st, String("x"), ctx)
    var fwd = h3_token_refiner_block_forward_lora[H, Dh](
        x, w, lora, D, F, EPS, EPS, ctx,
    )
    ctx.synchronize()
    var c_out = _cos(fwd.out[], _f32(st, String("out"), ctx), ctx)
    print("cos(out) =", c_out)
    var bwd = h3_token_refiner_block_backward_lora[H, Dh](
        _bf16(st, String("d_out"), ctx), w, lora, fwd.saved,
        D, F, EPS, EPS, ctx,
    )
    ctx.synchronize()
    var c_dx = _cos(bwd.d_x[], _f32(st, String("d_x"), ctx), ctx)
    print("cos(d_x) =", c_dx)
    var ok = c_out >= 0.999 and c_dx >= 0.999
    for i in range(4):
        var g: KleinLoraDeviceGradTensors
        if i == 0:
            g = bwd.lora.qkv.value().copy()
        elif i == 1:
            g = bwd.lora.out.value().copy()
        elif i == 2:
            g = bwd.lora.fc1.value().copy()
        else:
            g = bwd.lora.fc2.value().copy()
        var ca = _cos(g.d_a[], _f32(st, String("d_") + names[i] + String("_a"), ctx), ctx)
        var cb = _cos(g.d_b[], _f32(st, String("d_") + names[i] + String("_b"), ctx), ctx)
        print("cos(d_" + names[i] + "_a) =", ca)
        print("cos(d_" + names[i] + "_b) =", cb)
        if ca < 0.999 or cb < 0.999:
            ok = False
    if not ok:
        raise Error("MiniMax-H3 token-refiner training parity failed")
    print("PASS: token-refiner LoRA forward/backward matches Torch autograd (cos>=0.999)")
