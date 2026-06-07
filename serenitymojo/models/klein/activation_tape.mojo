# Dtype-preserving Klein LoRA activation tape offload.
#
# This is the narrow runtime bridge needed before a bounded
# CPU_OFFLOADED/checkpoint backward replay can exist. It offloads only the
# boundaries consumed by the current LoRA backward path:
#   dbl_img_in, dbl_txt_in, sgl_x_in, img_out, ln_img_out.
# The input-projection activations are intentionally not carried here because
# current LoRA backward does not consume them.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.tensor import Tensor
from serenitymojo.training.checkpoint import (
    HostOffload,
    offload_to_host,
    restore_to_device,
)

from serenitymojo.models.klein.double_block import DoubleBlockSaved
from serenitymojo.models.klein.single_block import SingleBlockSaved
from serenitymojo.models.klein.klein_stack import KleinStackForward


comptime TArc = ArcPointer[Tensor]


struct KleinStackLoraOffloadedTape(Copyable, Movable):
    var out: List[Float32]
    var dbl_img_in: List[HostOffload]
    var dbl_txt_in: List[HostOffload]
    var sgl_x_in: List[HostOffload]
    var img_out: HostOffload
    var ln_img_out: HostOffload

    def __init__(
        out self,
        var out_values: List[Float32],
        var dbl_img_in: List[HostOffload],
        var dbl_txt_in: List[HostOffload],
        var sgl_x_in: List[HostOffload],
        var img_out: HostOffload,
        var ln_img_out: HostOffload,
    ):
        self.out = out_values^
        self.dbl_img_in = dbl_img_in^
        self.dbl_txt_in = dbl_txt_in^
        self.sgl_x_in = sgl_x_in^
        self.img_out = img_out^
        self.ln_img_out = ln_img_out^

    def num_double(self) -> Int:
        return len(self.dbl_img_in)

    def num_single(self) -> Int:
        return len(self.sgl_x_in)

    def total_host_bytes(self) -> Int:
        var total = len(self.img_out.host) + len(self.ln_img_out.host)
        for i in range(len(self.dbl_img_in)):
            total += len(self.dbl_img_in[i].host)
        for i in range(len(self.dbl_txt_in)):
            total += len(self.dbl_txt_in[i].host)
        for i in range(len(self.sgl_x_in)):
            total += len(self.sgl_x_in[i].host)
        return total

    def all_storage_dtype(self, dtype: STDtype) -> Bool:
        if self.img_out.dtype != dtype or self.ln_img_out.dtype != dtype:
            return False
        for i in range(len(self.dbl_img_in)):
            if self.dbl_img_in[i].dtype != dtype:
                return False
        for i in range(len(self.dbl_txt_in)):
            if self.dbl_txt_in[i].dtype != dtype:
                return False
        for i in range(len(self.sgl_x_in)):
            if self.sgl_x_in[i].dtype != dtype:
                return False
        return True


def offload_klein_stack_lora_backward_tape(
    saved: KleinStackForward, ctx: DeviceContext
) raises -> KleinStackLoraOffloadedTape:
    var dbl_img_in = List[HostOffload]()
    var dbl_txt_in = List[HostOffload]()
    var sgl_x_in = List[HostOffload]()

    for i in range(len(saved.dbl_img_in)):
        var off_img = offload_to_host(saved.dbl_img_in[i][], ctx)
        dbl_img_in.append(off_img^)
        var off_txt = offload_to_host(saved.dbl_txt_in[i][], ctx)
        dbl_txt_in.append(off_txt^)

    for i in range(len(saved.sgl_x_in)):
        var off_x = offload_to_host(saved.sgl_x_in[i][], ctx)
        sgl_x_in.append(off_x^)

    var img_out = offload_to_host(saved.img_out[], ctx)
    var ln_img_out = offload_to_host(saved.ln_img_out[], ctx)

    return KleinStackLoraOffloadedTape(
        saved.out.copy(),
        dbl_img_in^,
        dbl_txt_in^,
        sgl_x_in^,
        img_out^,
        ln_img_out^,
    )


def _unused_activation_arc(dtype: STDtype, ctx: DeviceContext) raises -> TArc:
    var shape = List[Int]()
    shape.append(1)
    var t = zeros_device(shape^, dtype, ctx)
    return TArc(t^)


def restore_klein_stack_lora_backward_tape(
    tape: KleinStackLoraOffloadedTape, ctx: DeviceContext
) raises -> KleinStackForward:
    var dbl_img_in = List[TArc]()
    var dbl_txt_in = List[TArc]()
    var sgl_x_in = List[TArc]()

    for i in range(len(tape.dbl_img_in)):
        var img_t = restore_to_device(tape.dbl_img_in[i], ctx)
        dbl_img_in.append(TArc(img_t^))
        var txt_t = restore_to_device(tape.dbl_txt_in[i], ctx)
        dbl_txt_in.append(TArc(txt_t^))

    for i in range(len(tape.sgl_x_in)):
        var x_t = restore_to_device(tape.sgl_x_in[i], ctx)
        sgl_x_in.append(TArc(x_t^))

    var img_out_t = restore_to_device(tape.img_out, ctx)
    var ln_img_out_t = restore_to_device(tape.ln_img_out, ctx)

    var dbl_saved = List[DoubleBlockSaved]()
    var sgl_saved = List[SingleBlockSaved]()
    var dummy_img = _unused_activation_arc(tape.img_out.dtype, ctx)
    var dummy_txt = _unused_activation_arc(tape.ln_img_out.dtype, ctx)

    return KleinStackForward(
        tape.out.copy(),
        dummy_img^,
        dummy_txt^,
        dbl_img_in^,
        dbl_txt_in^,
        sgl_x_in^,
        dbl_saved^,
        sgl_saved^,
        TArc(img_out_t^),
        TArc(ln_img_out_t^),
    )
