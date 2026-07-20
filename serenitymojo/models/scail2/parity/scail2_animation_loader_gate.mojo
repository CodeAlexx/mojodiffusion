# Negative metadata gates for the production SCAIL-2 animation loaders.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.pipeline.scail2_animation import (
    validate_scail2_animation_input,
)
from serenitymojo.tensor import Tensor


def _write_fixture(
    path: String, shape: List[Int], dtype: STDtype, ctx: DeviceContext
) raises:
    var count = 1
    for dim in shape:
        count *= dim
    var values = List[Float32]()
    for i in range(count):
        values.append(Float32(i) * 0.01)
    var names = List[String]()
    names.append(String("tensor"))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(
        ArcPointer(Tensor.from_host(values^, shape.copy(), dtype, ctx))
    )
    save_safetensors(names, tensors, path, ctx)


def _expect_rejected(
    path: String, shape: List[Int], dtype: STDtype, label: String
) raises:
    var rejected = False
    try:
        validate_scail2_animation_input(path, "tensor", shape, dtype)
    except:
        rejected = True
    if not rejected:
        raise Error(String("SCAIL-2 loader accepted invalid ") + label)


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: scail2_animation_loader_gate <temporary-dir>")
    var root = String(args[1])
    var valid = root + String("/valid.safetensors")
    var bad_rank = root + String("/bad_rank.safetensors")
    var bad_shape = root + String("/bad_shape.safetensors")
    var bad_dtype = root + String("/bad_dtype.safetensors")
    var expected = List[Int]()
    expected.append(1)
    expected.append(16)
    expected.append(1)
    expected.append(2)
    expected.append(2)
    var ctx = DeviceContext()

    _write_fixture(valid, expected.copy(), STDtype.F32, ctx)
    _write_fixture(bad_rank, [16, 1, 2, 2], STDtype.F32, ctx)
    _write_fixture(bad_shape, [1, 16, 2, 1, 2], STDtype.F32, ctx)
    _write_fixture(bad_dtype, expected.copy(), STDtype.BF16, ctx)

    validate_scail2_animation_input(
        valid, "tensor", expected.copy(), STDtype.F32
    )
    _expect_rejected(
        bad_rank, expected.copy(), STDtype.F32, "rank with equal numel"
    )
    _expect_rejected(
        bad_shape, expected.copy(), STDtype.F32, "shape with equal numel"
    )
    _expect_rejected(
        bad_dtype, expected^, STDtype.F32, "dtype"
    )
    print("GATE PASS SCAIL-2 animation exact rank/shape/dtype loaders")
