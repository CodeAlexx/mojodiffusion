# gate: linear_fp8 (fused fp8 GEMM) vs dequant+linear (proven path), real qkv.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.random import randn
from serenitymojo.ops.linear import linear
from serenitymojo.ops.fp8 import load_fp8_dequant
from serenitymojo.ops.fp8_gemm import linear_fp8

comptime T = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(T)
    var name = String("layers.0.attention.qkv.weight")
    # reference path: dequant -> bf16 weight -> vendor BLAS linear
    var w_bf16 = load_fp8_dequant(st, name, ctx)              # [13824,4608] bf16
    # fused path inputs: raw fp8 weight + f32 scale
    var wi = st.tensor_info(name)
    var w_fp8 = Tensor.from_view_raw(from_parts(wi.dtype, wi.shape.copy(), st.tensor_bytes(name)), ctx)
    var si = st.tensor_info(name + "_scale")
    var scale = Tensor.from_view_as_f32(from_parts(si.dtype, si.shape.copy(), st.tensor_bytes(name + "_scale")), ctx)

    var x = randn([1, 260, 4608], UInt64(0), STDtype.BF16, ctx)
    var y_ref = linear(x, w_bf16, None, ctx)                  # [1,260,13824]
    var y_fp8 = linear_fp8(x, w_fp8, scale, None, ctx)
    print("y_fp8 shape", y_fp8.shape()[0], y_fp8.shape()[1], y_fp8.shape()[2])
    var ref_host = y_ref.to_host(ctx)
    print("fp8 GEMM vs dequant+BLAS:", ParityHarness(0.999).compare(y_fp8, ref_host, ctx))
