# Microbench: Klein LoRA helper cost at active 4B training shapes.

from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.lora_block import (
    klein_lora_fwd_device, klein_lora_bwd_device,
)
from serenitymojo.training.train_step import LoraAdapter


def zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def make_lora(in_f: Int, out_f: Int, rank: Int) -> LoraAdapter:
    return LoraAdapter(
        zeros(rank * in_f), zeros(out_f * rank),
        rank, in_f, out_f, Float32(1.0),
        zeros(rank * in_f), zeros(rank * in_f),
        zeros(out_f * rank), zeros(out_f * rank),
    )


def zeros2(rows: Int, cols: Int, ctx: DeviceContext) raises -> Tensor:
    var buf = ctx.enqueue_create_buffer[DType.uint8](rows * cols * 4)
    buf.enqueue_fill(UInt8(0))
    var shape = List[Int]()
    shape.append(rows)
    shape.append(cols)
    return Tensor(buf^, shape^, STDtype.F32)


def bench_lora(name: String, rows: Int, in_f: Int, out_f: Int, ctx: DeviceContext) raises:
    var lo = make_lora(in_f, out_f, 16)
    var x = zeros2(rows, in_f, ctx)
    var go = zeros2(rows, out_f, ctx)

    for _ in range(2):
        var _y = klein_lora_fwd_device(x, lo, rows, ctx)
        ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(10):
        var _y = klein_lora_fwd_device(x, lo, rows, ctx)
        ctx.synchronize()
    var t1 = perf_counter_ns()
    print(name, " fwd -> ", Float64(t1 - t0) / 10.0e6, " ms")

    for _ in range(2):
        var _g = klein_lora_bwd_device(go, x, lo, rows, ctx)
        ctx.synchronize()
    t0 = perf_counter_ns()
    for _ in range(10):
        var _g = klein_lora_bwd_device(go, x, lo, rows, ctx)
        ctx.synchronize()
    t1 = perf_counter_ns()
    print(name, " bwd -> ", Float64(t1 - t0) / 10.0e6, " ms")


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein LoRA helper microbench ===")
    bench_lora(String("single qkv"), 1536, 3072, 9216, ctx)
    bench_lora(String("single out"), 1536, 3072, 3072, ctx)
    bench_lora(String("double img qkv"), 1024, 3072, 9216, ctx)
    bench_lora(String("double txt qkv"), 512, 3072, 9216, ctx)
