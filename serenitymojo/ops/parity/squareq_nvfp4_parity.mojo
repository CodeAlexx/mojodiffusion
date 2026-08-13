# ops/parity/squareq_nvfp4_parity.mojo — chunk-7 gate for the native NVFP4
# forward (ops/squareq_nvfp4.squareq_nvfp4_linear) vs the Python emulation +
# the bf16 ideal (fixture: scripts/squareq_nvfp4_fixture.py).
#
# Gates: vs y_fp4ref (implementation parity)  cos >= 0.999   <- THE Mojo gate
#        vs y_ideal  (format quality, informational floor 0.995) — the Python
#        emulation itself scores 0.99639 on this synthetic fixture, so any
#        faithful W4A4 lands there; real-weight quality is judged by the
#        training smoke/A-B, not this fixture (chunk-0 lesson).
#
# Build:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     -Xlinker -lcuda -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/ops/parity/squareq_nvfp4_parity.mojo \
#     -o output/checks/squareq_nvfp4_parity
# Run: LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib \
#     output/checks/squareq_nvfp4_parity

from std.math import sqrt
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.squareq_nvfp4 import squareq_nvfp4_linear, squareq_nvfp4_reconstruct_weight

comptime FIXTURE = "/home/alex/mojodiffusion/serenitymojo/ops/parity/squareq_nvfp4_fixture.safetensors"


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: length mismatch")
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    return dot / (sqrt(na) * sqrt(nb) + 1e-30)


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(FIXTURE))
    var nvq = Tensor.from_view_raw(st.tensor_view(String("nvq")), ctx)
    var nvs = Tensor.from_view_raw(st.tensor_view(String("nvs")), ctx)
    var nvg_t = Tensor.from_view(st.tensor_view(String("nvg")), ctx)
    var ld = Tensor.from_view(st.tensor_view(String("lora_down")), ctx)
    var lu = Tensor.from_view(st.tensor_view(String("lora_up")), ctx)
    var x = Tensor.from_view(st.tensor_view(String("x")), ctx)
    var y_ideal = Tensor.from_view(st.tensor_view(String("y_ideal")), ctx)
    var y_ref = Tensor.from_view(st.tensor_view(String("y_fp4ref")), ctx)

    var nvg_host = nvg_t.to_host(ctx)
    var nvg = Float32(nvg_host[0])
    print("[nvfp4-parity] out=", nvq.shape()[0], " in=", nvq.shape()[1] * 2,
          " nvg=", nvg)

    var y = squareq_nvfp4_linear(x, nvq, nvs, nvg, ld, lu, ctx)
    var yh = y.to_host(ctx)
    var c_ref = _cos(yh, y_ref.to_host(ctx))
    var c_ideal = _cos(yh, y_ideal.to_host(ctx))
    print("[nvfp4-parity] vs emulated fp4 ref cos=", c_ref)
    print("[nvfp4-parity] vs bf16 ideal      cos=", c_ideal)
    if c_ref < 0.999:
        raise Error("nvfp4 parity FAILED vs emulation")
    if c_ideal < 0.995:
        raise Error("nvfp4 quality floor FAILED vs ideal (regression)")
    # bwd-side reconstruct vs oracle
    var w_ref = Tensor.from_view(st.tensor_view(String("w_hat_nv")), ctx)
    var w_hat = squareq_nvfp4_reconstruct_weight(
        nvq, nvs, nvg, ld, lu, nvq.shape()[1] * 2, nvq.shape()[0], ctx
    )
    var c_w = _cos(w_hat.to_host(ctx), w_ref.to_host(ctx))
    print("[nvfp4-parity] reconstruct_weight vs oracle cos=", c_w)
    if c_w < 0.9999:
        raise Error("nvfp4 reconstruct parity FAILED")
    print("[nvfp4-parity] ALL PASS")
