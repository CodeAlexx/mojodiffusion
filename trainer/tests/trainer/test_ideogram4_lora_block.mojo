from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import add_scalar, mul_scalar, zeros_device

from serenity_trainer.model.Ideogram4LoRABlock import (
    I4_SLOT_QKV,
    I4_SLOTS_PER_BLOCK,
    Ideogram4BlockWeights,
    Ideogram4StackLoraGrads,
    build_ideogram4_lora_set,
    ideogram4_block_lora_backward,
    ideogram4_block_lora_forward,
)
from serenity_trainer.module.LoRAModule import LoraAdapter
from serenity_trainer.trainer.Ideogram4StackTrain import (
    apply_ideogram4_lora_grads,
    make_ideogram4_lora_adam_state,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig

comptime LArc = ArcPointer[LoraAdapter]


def _require(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("IDEOGRAM4 LORA BLOCK FAIL: ") + msg)


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _w(out_f: Int, in_f: Int, seed: UInt64, ctx: DeviceContext) raises -> Tensor:
    return mul_scalar(randn(_shape2(out_f, in_f), seed, STDtype.BF16, ctx), Float32(0.04), ctx)


def _ones(n: Int, ctx: DeviceContext) raises -> Tensor:
    return add_scalar(zeros_device(_shape1(n), STDtype.BF16, ctx), Float32(1.0), ctx)


def _l1(x: Tensor, ctx: DeviceContext) raises -> Float32:
    var h = x.to_host(ctx)
    var s = Float32(0.0)
    for i in range(len(h)):
        var v = h[i]
        if v < Float32(0.0):
            s -= v
        else:
            s += v
    return s


def main() raises:
    var ctx = DeviceContext()

    comptime S = 3
    comptime Hidden = 8
    comptime Heads = 2
    comptime Dh = 4
    comptime FF = 12
    comptime Adaln = 5

    var w = Ideogram4BlockWeights(
        _w(4 * Hidden, Adaln, UInt64(1), ctx),
        zeros_device(_shape1(4 * Hidden), STDtype.BF16, ctx),
        _ones(Hidden, ctx),
        _ones(Hidden, ctx),
        _ones(Hidden, ctx),
        _ones(Hidden, ctx),
        _w(3 * Hidden, Hidden, UInt64(2), ctx),
        _w(Hidden, Hidden, UInt64(3), ctx),
        _ones(Dh, ctx),
        _ones(Dh, ctx),
        _w(FF, Hidden, UInt64(4), ctx),
        _w(Hidden, FF, UInt64(5), ctx),
        _w(FF, Hidden, UInt64(6), ctx),
    )
    var loras = build_ideogram4_lora_set[Hidden, FF, Adaln](
        2, Float32(2.0), ctx, 1, UInt64(77)
    )
    var bl = List[LArc]()
    for slot in range(I4_SLOTS_PER_BLOCK):
        bl.append(loras.ad[slot])

    var x = randn(_shape2(S, Hidden), UInt64(100), STDtype.BF16, ctx)
    var adaln = randn(_shape2(1, Adaln), UInt64(101), STDtype.BF16, ctx)
    var cosf = add_scalar(zeros_device(_shape3(1, S, Dh), STDtype.BF16, ctx), Float32(1.0), ctx)
    var sinf = zeros_device(_shape3(1, S, Dh), STDtype.BF16, ctx)

    var fwd = ideogram4_block_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
        x, adaln, cosf, sinf, w, bl, ctx
    )
    var d_out = randn(_shape2(S, Hidden), UInt64(200), STDtype.BF16, ctx)
    var bwd = ideogram4_block_lora_backward[S, Hidden, Heads, Dh, FF, Adaln](
        d_out, fwd.acts^, cosf, sinf, w, bl, ctx
    )

    _require(len(bwd.lora_grads.d_a) == I4_SLOTS_PER_BLOCK, "d_a slot count")
    _require(len(bwd.lora_grads.d_b) == I4_SLOTS_PER_BLOCK, "d_b slot count")
    _require(bwd.d_x.shape()[0] == S and bwd.d_x.shape()[1] == Hidden, "d_x shape")

    var grad_sum = Float32(0.0)
    for slot in range(I4_SLOTS_PER_BLOCK):
        grad_sum += _l1(bwd.lora_grads.d_b[slot][], ctx)
    _require(grad_sum > Float32(0.0), "all LoRA-B gradients are zero")

    var cfg = TrainConfig.adamw_lora_defaults()
    cfg.learning_rate = Float32(2e-3)
    cfg.stochastic_rounding = False
    cfg.seed = UInt32(123)
    var opt = make_ideogram4_lora_adam_state(loras, ctx)
    var stack_grads = Ideogram4StackLoraGrads(
        bwd.lora_grads.d_a.copy(),
        bwd.lora_grads.d_b.copy(),
        bwd.d_x.clone(ctx),
        bwd.d_adaln_input.clone(ctx),
    )
    var before = _l1(loras.ad[I4_SLOT_QKV][].b, ctx)
    var result = apply_ideogram4_lora_grads(loras, opt, stack_grads^, 1, cfg, ctx)
    var after = _l1(loras.ad[I4_SLOT_QKV][].b, ctx)
    _require(result.adapters == I4_SLOTS_PER_BLOCK, "optimizer adapter count")
    _require(result.grad_b_l1 > Float32(0.0), "optimizer saw zero LoRA-B grad")
    _require(after > before, "AdamW did not update qkv LoRA-B")

    print("IDEOGRAM4 LORA BLOCK PASS")
    print("  grad_b_l1 =", grad_sum, " qkv_b_l1 =", after)
