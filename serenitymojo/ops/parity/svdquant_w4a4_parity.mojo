# svdquant_w4a4_parity.mojo — end-to-end W4A4 runtime gate (MJ-1099 B.3b).
#
# Loads a real 4096×4096 LTX2 layer quantized to QuaRot W4A4 (scripts/
# svdquant_w4a4_make_fixture.py) and runs the FULL Mojo runtime
# (ops/svdquant_w4a4.svdquant_linear_w4a4 = FWHT+quant → int4 GEMM → rescale →
# +low-rank +bias), asserting cos(y, bf16-ideal) ~ 0.99 — the number the SquareQ
# sim predicted. Proves FWHT + int4 GEMM + rescale + low-rank compose correctly.
#
# Build (link the int4 GEMM shim):
#   rm -f serenitymojo.mojopkg && pixi run mojo build -O2 -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_int4_gemm \
#     serenitymojo/ops/parity/svdquant_w4a4_parity.mojo -o /tmp/w4a4_parity
# Run: LD_LIBRARY_PATH=serenitymojo/ops/cshim/lib:.pixi/envs/default/lib /tmp/w4a4_parity

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.svdquant_w4a4 import SvdquantW4A4, svdquant_linear_w4a4

comptime FIXTURE = "/home/alex/mojodiffusion/serenitymojo/ops/parity/svdq_w4a4_fixture.safetensors"
comptime COS_BAR = 0.985


def _bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view(st.tensor_view(name), ctx)


def _raw(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_raw(st.tensor_view(name), ctx)


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    var dot: Float64 = 0.0; var na: Float64 = 0.0; var nb: Float64 = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("_cos: zero norm")
    return dot / (sqrt(na) * sqrt(nb))


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(FIXTURE))

    var qweight = _raw(st, String("qweight"), ctx)
    var wscale = _bf16(st, String("wscale"), ctx)
    var lora_down = _bf16(st, String("lora_down"), ctx)
    var lora_up = _bf16(st, String("lora_up"), ctx)
    var bias = _bf16(st, String("bias"), ctx)
    var x = _bf16(st, String("x"), ctx)
    var y_true = _bf16(st, String("y_true"), ctx)

    var out_f = qweight.shape()[0]
    var in_f = qweight.shape()[1] * 2
    var rank = lora_down.shape()[1]
    var M = x.shape()[0]
    print("[w4a4] REAL LTX2 layer: out=", out_f, " in=", in_f, " rank=", rank, " M=", M)

    var w = SvdquantW4A4(
        qweight^, wscale^, lora_down^, lora_up^, bias^, in_f, out_f, rank)

    var y = svdquant_linear_w4a4(x, w, ctx)
    var y_cos = _cos(y.to_host(ctx), y_true.to_host(ctx))
    print("[w4a4] y vs bf16-ideal cos =", y_cos)
    if y_cos >= COS_BAR:
        print("[w4a4] PASS (full W4A4 runtime: cos", y_cos, ">=", COS_BAR, ")")
    else:
        print("[w4a4] FAIL — cos", y_cos, "<", COS_BAR)
        raise Error("W4A4 runtime parity FAILED")
