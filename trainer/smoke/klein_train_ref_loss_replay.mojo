# Replay the real Serenity Klein/Flux2 train-step dump through the Mojo loss path.
#
# This is intentionally narrow: it verifies Mojo computes the same MSE loss
# from Serenity's dumped `output.predicted` and `output.target`. It does not
# claim Klein/Flux2 transformer backward, optimizer, full-finetune, or sampler
# parity.

from std.gpu.host import DeviceContext
from std.math import abs
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.reduce import reduce_mean_f32
from serenitymojo.ops.tensor_algebra import mul, sub
from serenitymojo.tensor import Tensor


comptime PARITY = "trainer/parity/klein_train_ref_step000.safetensors"
comptime SERENITY_LOSS = Float32(0.12243738770484924)
# Klein's loss tensor has 131072 elements and larger squared terms than the
# smaller helper gates; the Mojo F32 GPU reduction currently differs from
# PyTorch's CUDA reduction order by about 2.0e-5 on this dump.
comptime LOSS_EPS = Float32(0.000025)


def _sec(ns0: UInt, ns1: UInt) -> Float64:
    return Float64(ns1 - ns0) / Float64(1000000000.0)


def main() raises:
    var ctx = DeviceContext()
    var all0 = perf_counter_ns()

    print("=== Klein train-ref loss replay ===")
    print("[parity]", PARITY)

    var st = ShardedSafeTensors.open(String(PARITY))
    var predicted_bf16 = Tensor.from_view(st.tensor_view(String("output.predicted")), ctx)
    var target = Tensor.from_view(st.tensor_view(String("output.target")), ctx)
    var loss_ref = Tensor.from_view(st.tensor_view(String("output.loss_pre_scale")), ctx)

    # PyTorch promotes BF16 predicted - F32 target to F32 for Serenity's MSE.
    var cast0 = perf_counter_ns()
    var predicted = cast_tensor(predicted_bf16, STDtype.F32, ctx)
    var cast1 = perf_counter_ns()

    var loss0 = perf_counter_ns()
    var diff = sub(predicted, target, ctx)
    var sq = mul(diff, diff, ctx)
    var dims = List[Int]()
    for i in range(len(sq.shape())):
        dims.append(i)
    var loss = reduce_mean_f32(sq, dims^, False, ctx).to_host(ctx)[0]
    var loss1 = perf_counter_ns()

    var ref_host = loss_ref.to_host(ctx)[0]
    var err = abs(loss - ref_host)
    var err_const = abs(loss - SERENITY_LOSS)

    print("Mojo loss =", loss)
    print("dump loss =", ref_host)
    print("reference trainer const  =", SERENITY_LOSS)
    print("abs_err_dump =", err, " abs_err_const =", err_const)
    print("time_s: cast =", _sec(cast0, cast1), " loss =", _sec(loss0, loss1),
          " total =", _sec(all0, perf_counter_ns()))

    if err > LOSS_EPS or err_const > LOSS_EPS:
        raise Error("Klein train-ref loss replay mismatch")

    print("KLEIN TRAIN REF LOSS REPLAY PASS")
