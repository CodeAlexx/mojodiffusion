# Bounded Klein activation tape offload smoke.
#
# This uses tiny BF16 tensors to prove the Klein LoRA backward tape bridge
# stores raw bytes as HostOffload and restores tensors with the original dtype.
# It is not a full model replay and does not accept CPU_OFFLOADED parity.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.models.klein.double_block import DoubleBlockSaved
from serenitymojo.models.klein.single_block import SingleBlockSaved
from serenitymojo.models.klein.klein_stack import KleinStackForward
from serenitymojo.models.klein.activation_tape import (
    offload_klein_stack_lora_backward_tape,
    restore_klein_stack_lora_backward_tape,
)


comptime TArc = ArcPointer[Tensor]


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _vals(n: Int, start: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(start + Float32(i) * Float32(0.125))
    return out^


def _arc_bf16(
    var values: List[Float32], var shape: List[Int], ctx: DeviceContext
) raises -> TArc:
    var t = Tensor.from_host(values^, shape^, STDtype.BF16, ctx)
    return TArc(t^)


def _max_abs(a: List[Float32], b: List[Float32]) -> Float32:
    var m = Float32(0.0)
    var n = len(a)
    if len(b) < n:
        n = len(b)
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


def _check_int(name: String, got: Int, expected: Int) raises:
    print("[klein-tape-offload]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("Klein tape offload mismatch: ") + name)


def _check_bool(name: String, got: Bool, expected: Bool) raises:
    print("[klein-tape-offload]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("Klein tape offload bool mismatch: ") + name)


def _check_tensor_rt(
    name: String, expected: TArc, got: TArc, ctx: DeviceContext
) raises:
    _check_bool(name + String(" dtype"), got[].dtype() == expected[].dtype(), True)
    _check_int(name + String(" nbytes"), got[].nbytes(), expected[].nbytes())
    var a = expected[].to_host(ctx)
    var b = got[].to_host(ctx)
    var m = _max_abs(a, b)
    print("[klein-tape-offload]", name, "max_abs=", m)
    if m != Float32(0.0):
        raise Error(String("Klein tape offload roundtrip mismatch: ") + name)


def main() raises:
    var ctx = DeviceContext()
    print("=== Klein activation tape offload smoke ===")

    var out = List[Float32]()
    out.append(Float32(1.0))
    out.append(Float32(2.0))

    var img_in_act = _arc_bf16(_vals(8, 0.0), _shape2(2, 4), ctx)
    var txt_in_act = _arc_bf16(_vals(4, 10.0), _shape2(1, 4), ctx)

    var dbl_img_in = List[TArc]()
    var dbl_txt_in = List[TArc]()
    for i in range(2):
        dbl_img_in.append(_arc_bf16(_vals(8, Float32(20 + i)), _shape2(2, 4), ctx))
        dbl_txt_in.append(_arc_bf16(_vals(4, Float32(30 + i)), _shape2(1, 4), ctx))

    var sgl_x_in = List[TArc]()
    for i in range(3):
        sgl_x_in.append(_arc_bf16(_vals(12, Float32(40 + i)), _shape2(3, 4), ctx))

    var dbl_saved = List[DoubleBlockSaved]()
    var sgl_saved = List[SingleBlockSaved]()
    var img_out = _arc_bf16(_vals(8, 50.0), _shape2(2, 4), ctx)
    var ln_img_out = _arc_bf16(_vals(8, 60.0), _shape2(2, 4), ctx)

    var saved = KleinStackForward(
        out^,
        img_in_act^,
        txt_in_act^,
        dbl_img_in^,
        dbl_txt_in^,
        sgl_x_in^,
        dbl_saved^,
        sgl_saved^,
        img_out^,
        ln_img_out^,
    )

    var tape = offload_klein_stack_lora_backward_tape(saved, ctx)
    _check_int(String("double blocks"), tape.num_double(), 2)
    _check_int(String("single blocks"), tape.num_single(), 3)
    _check_int(String("host bytes"), tape.total_host_bytes(), 152)
    _check_bool(String("storage dtype bf16"), tape.all_storage_dtype(STDtype.BF16), True)
    _check_bool(String("storage dtype f32 rejected"), tape.all_storage_dtype(STDtype.F32), False)

    var restored = restore_klein_stack_lora_backward_tape(tape, ctx)
    _check_tensor_rt(String("dbl_img_in[0]"), saved.dbl_img_in[0], restored.dbl_img_in[0], ctx)
    _check_tensor_rt(String("dbl_txt_in[1]"), saved.dbl_txt_in[1], restored.dbl_txt_in[1], ctx)
    _check_tensor_rt(String("sgl_x_in[2]"), saved.sgl_x_in[2], restored.sgl_x_in[2], ctx)
    _check_tensor_rt(String("img_out"), saved.img_out, restored.img_out, ctx)
    _check_tensor_rt(String("ln_img_out"), saved.ln_img_out, restored.ln_img_out, ctx)

    _check_int(String("unused input projection dummy bytes"), restored.img_in_act[].nbytes(), 2)
    _check_bool(String("no saved double internals"), len(restored.dbl_saved) == 0, True)
    _check_bool(String("no saved single internals"), len(restored.sgl_saved) == 0, True)

    print("klein_activation_tape_offload_smoke PASS")
