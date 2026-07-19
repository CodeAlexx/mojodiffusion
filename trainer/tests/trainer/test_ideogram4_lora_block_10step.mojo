from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import (
    add_scalar,
    mul,
    mul_scalar,
    sub,
    zeros_device,
)
from serenitymojo.ops.loss_swiglu_backward import mse_backward
from serenitymojo.ops.reduce import reduce_mean_f32

from serenity_trainer.model.Ideogram4LoRABlock import (
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
comptime NSTEPS = 10


def _require(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("IDEOGRAM4 LORA 10STEP FAIL: ") + msg)


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


def _mse_loss(pred: Tensor, target: Tensor, ctx: DeviceContext) raises -> Float32:
    var diff = sub(pred, target, ctx)
    var sq = mul(diff, diff, ctx)
    var dims = List[Int]()
    dims.append(0)
    dims.append(1)
    return reduce_mean_f32(sq, dims^, False, ctx).to_host(ctx)[0]


def _total_b_l1(loras: List[LArc], ctx: DeviceContext) raises -> Float32:
    var total = Float32(0.0)
    for i in range(len(loras)):
        total += _l1(loras[i][].b, ctx)
    return total


def main() raises:
    var ctx = DeviceContext()

    comptime S = 4
    comptime Hidden = 16
    comptime Heads = 4
    comptime Dh = 4
    comptime FF = 24
    comptime Adaln = 8

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
        4, Float32(4.0), ctx, 1, UInt64(7700)
    )
    var bl = List[LArc]()
    for slot in range(I4_SLOTS_PER_BLOCK):
        bl.append(loras.ad[slot])

    var x = randn(_shape2(S, Hidden), UInt64(100), STDtype.BF16, ctx)
    var adaln = randn(_shape2(1, Adaln), UInt64(101), STDtype.BF16, ctx)
    var target = randn(_shape2(S, Hidden), UInt64(102), STDtype.BF16, ctx)
    var cosf = add_scalar(zeros_device(_shape3(1, S, Dh), STDtype.BF16, ctx), Float32(1.0), ctx)
    var sinf = zeros_device(_shape3(1, S, Dh), STDtype.BF16, ctx)

    var cfg = TrainConfig.adamw_lora_defaults()
    cfg.learning_rate = Float32(1e-3)
    cfg.stochastic_rounding = False
    cfg.seed = UInt32(901)
    var opt = make_ideogram4_lora_adam_state(loras, ctx)

    var before_b = _total_b_l1(loras.ad, ctx)
    var last_loss = Float32(0.0)
    var last_grad_b = Float32(0.0)

    for step in range(1, NSTEPS + 1):
        var fwd = ideogram4_block_lora_forward[S, Hidden, Heads, Dh, FF, Adaln](
            x, adaln, cosf, sinf, w, bl, ctx
        )
        last_loss = _mse_loss(fwd.out, target, ctx)
        var d_out = mse_backward(fwd.out, target, ctx)
        var bwd = ideogram4_block_lora_backward[S, Hidden, Heads, Dh, FF, Adaln](
            d_out, fwd.acts^, cosf, sinf, w, bl, ctx
        )
        var stack_grads = Ideogram4StackLoraGrads(
            bwd.lora_grads.d_a.copy(),
            bwd.lora_grads.d_b.copy(),
            bwd.d_x.clone(ctx),
            bwd.d_adaln_input.clone(ctx),
        )
        var result = apply_ideogram4_lora_grads(
            loras, opt, stack_grads^, step, cfg, ctx
        )
        last_grad_b = result.grad_b_l1
        print("step", step, "loss", last_loss, "grad_b_l1", last_grad_b, "b_l1", result.adapter_b_l1)

    var after_b = _total_b_l1(loras.ad, ctx)
    _require(after_b > before_b, "LoRA-B did not move after 10 steps")
    _require(last_loss == last_loss and last_loss > Float32(0.0), "loss invalid")
    _require(last_grad_b > Float32(0.0), "final LoRA-B gradient was zero")

    print("IDEOGRAM4 LORA 10STEP PASS")
    print("  steps =", NSTEPS, " final_loss =", last_loss, " b_l1 =", after_b)
