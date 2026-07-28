# ops/parity/squareq_parity.mojo — chunk-2 gate for ops/squareq.mojo vs the
# Python byte-level oracle (scripts/squareq/core.py, fixture by
# scripts/squareq_make_fixture.py).
#
# Gates:
#   1. rht256_grouped(rht_in) vs oracle       cos >= 0.99999, max_abs <= 0.05
#   2. squareq_dequant_derotate vs oracle     cos >= 0.9999
#   3. reconstruct_weight vs oracle W_hat     cos >= 0.9999
#   4. x @ W_hat^T linear output vs oracle    cos >= 0.9999
#
# Build:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     -Xlinker -lcuda serenitymojo/ops/parity/squareq_parity.mojo \
#     -o output/checks/squareq_parity
# Run: LD_LIBRARY_PATH=.pixi/envs/default/lib output/checks/squareq_parity

from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.linear import linear
from serenitymojo.ops.squareq import (
    rht256_grouped,
    squareq_dequant_derotate,
    squareq_reconstruct_weight,
)

comptime FIXTURE = "/home/alex/mojodiffusion/serenitymojo/ops/parity/squareq_fixture.safetensors"
comptime FIXTURE_G32 = "/home/alex/mojodiffusion/serenitymojo/ops/parity/squareq_fixture_g32.safetensors"


def _bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view(st.tensor_view(name), ctx)


def _raw(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_raw(st.tensor_view(name), ctx)


def _cos_max(
    a: List[Float32], b: List[Float32]
) raises -> Tuple[Float64, Float64]:
    if len(a) != len(b):
        raise Error("_cos_max: length mismatch")
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    var mx: Float64 = 0.0
    for i in range(len(a)):
        var av = Float64(a[i])
        var bv = Float64(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
        var d = av - bv
        if d < 0:
            d = -d
        if d > mx:
            mx = d
    if na == 0.0 or nb == 0.0:
        raise Error("_cos_max: zero norm")
    return (dot / (sqrt(na) * sqrt(nb)), mx)


def _gate(name: String, cos: Float64, bar: Float64) raises:
    if cos >= bar:
        print("[squareq-parity] PASS ", name, " cos=", cos, " (bar ", bar, ")")
    else:
        print("[squareq-parity] FAIL ", name, " cos=", cos, " < ", bar)
        raise Error(String("squareq parity FAILED: ") + name)


def _run_fixture(path: String, ctx: DeviceContext) raises:
    var st = ShardedSafeTensors.open(path)

    var qweight = _raw(st, String("qweight"), ctx)
    var wscales = _bf16(st, String("wscales"), ctx)
    var lora_down = _bf16(st, String("lora_down"), ctx)
    var lora_up = _bf16(st, String("lora_up"), ctx)
    var w_hat_ref = _bf16(st, String("w_hat"), ctx)
    var wres_ref = _bf16(st, String("wres_expected"), ctx)
    var rht_in = _bf16(st, String("rht_in"), ctx)
    var rht_ref = _bf16(st, String("rht_expected"), ctx)
    var x = _bf16(st, String("x"), ctx)
    var y_ref = _bf16(st, String("y_ref"), ctx)

    var out_f = qweight.shape()[0]
    var in_f = qweight.shape()[1] * 2
    print(
        "[squareq-parity] fixture out=", out_f, " in=", in_f,
        " rank=", lora_down.shape()[1], " M=", x.shape()[0],
    )

    # 1) grouped RHT-256
    var rot = rht256_grouped(rht_in, ctx)
    var cm = _cos_max(rot.to_host(ctx), rht_ref.to_host(ctx))
    print("[squareq-parity] rht max_abs_err=", cm[1])
    _gate(String("rht256_grouped"), cm[0], 0.99999)
    if cm[1] > 0.05:
        raise Error("rht256_grouped max_abs_err > 0.05")

    # 2) fused dequant + derotate
    var wres = squareq_dequant_derotate(qweight, wscales, in_f, out_f, ctx)
    cm = _cos_max(wres.to_host(ctx), wres_ref.to_host(ctx))
    _gate(String("dequant_derotate"), cm[0], 0.9999)

    # 3) full reconstruct
    var w_hat = squareq_reconstruct_weight(
        qweight, wscales, lora_down, lora_up, in_f, out_f, ctx
    )
    cm = _cos_max(w_hat.to_host(ctx), w_hat_ref.to_host(ctx))
    _gate(String("reconstruct_weight"), cm[0], 0.9999)

    # 4) linear output on the reconstructed weight
    var y = linear(x, w_hat, None, ctx)
    cm = _cos_max(y.to_host(ctx), y_ref.to_host(ctx))
    _gate(String("linear(x, W_hat)"), cm[0], 0.9999)

    print("[squareq-parity] fixture PASS: ", path)


def main() raises:
    var ctx = DeviceContext()
    _run_fixture(String(FIXTURE), ctx)      # group 64 (inferred from shapes)
    _run_fixture(String(FIXTURE_G32), ctx)  # group 32 (inferred from shapes)
    print("[squareq-parity] ALL PASS")
