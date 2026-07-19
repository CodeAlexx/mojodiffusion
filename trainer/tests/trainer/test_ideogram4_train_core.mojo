from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import zeros_device
from serenity_trainer.trainer.Ideogram4TrainCore import (
    Ideogram4FinalLinearLoRA,
    Ideogram4FinalLinearCache,
    train_ideogram4_final_linear_lora_cache,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig

comptime TArc = ArcPointer[Tensor]


def _require(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("IDEOGRAM4 TRAIN CORE FAIL: ") + msg)


def _l1(x: Tensor, ctx: DeviceContext) raises -> Float32:
    var host = x.to_host(ctx)
    var s = Float32(0.0)
    for i in range(len(host)):
        var v = host[i]
        if v < Float32(0.0):
            s -= v
        else:
            s += v
    return s


def _finite(x: Float32) -> Bool:
    if x != x:
        return False
    var a = x if x >= Float32(0.0) else -x
    return a < Float32(3.0e38)


def _shape2(a: Int, b: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    return sh^


def main() raises:
    var ctx = DeviceContext()
    comptime H = 8
    comptime O = 4
    comptime N = 6

    var cfg = TrainConfig.adamw_lora_defaults()
    cfg.learning_rate = Float32(2e-3)
    cfg.lora_rank = 2
    cfg.lora_alpha = Float32(2.0)
    cfg.stochastic_rounding = False
    cfg.seed = UInt32(123)

    var base_w = randn(_shape2(O, H), UInt64(11), STDtype.BF16, ctx)
    var bsh = List[Int]()
    bsh.append(O)
    var base_b = zeros_device(bsh^, STDtype.BF16, ctx)

    var h0 = randn(_shape2(N, H), UInt64(101), STDtype.BF16, ctx)
    var t0 = randn(_shape2(N, O), UInt64(202), STDtype.BF16, ctx)
    var h1 = randn(_shape2(N, H), UInt64(303), STDtype.BF16, ctx)
    var t1 = randn(_shape2(N, O), UInt64(404), STDtype.BF16, ctx)

    var names = List[String]()
    var tensors = List[TArc]()
    names.append(String("hidden.0"))
    tensors.append(TArc(h0^))
    names.append(String("target.0"))
    tensors.append(TArc(t0^))
    names.append(String("hidden.1"))
    tensors.append(TArc(h1^))
    names.append(String("target.1"))
    tensors.append(TArc(t1^))

    var cache_path = String("/tmp/serenity_ideogram4_final_train_cache.safetensors")
    save_safetensors(names^, tensors^, cache_path, ctx)

    var cache = Ideogram4FinalLinearCache.open(cache_path)
    _require(cache.len() == 2, "cache did not discover two samples")
    var sample = cache.sample[H, O](0, ctx)
    _require(sample.hidden[].shape()[0] == N, "hidden token count mismatch")
    _require(sample.target[].shape()[1] == O, "target output dim mismatch")

    var state = Ideogram4FinalLinearLoRA[H, O].new(
        cfg.lora_rank, cfg.lora_alpha, UInt64(777), ctx
    )
    var before = _l1(state.adapter.b, ctx)
    _require(before == Float32(0.0), "LoRA-B should start at zero")

    var summary = train_ideogram4_final_linear_lora_cache[H, O](
        state, base_w^, base_b^, cache_path, 4, cfg, ctx
    )
    var after = _l1(state.adapter.b, ctx)

    _require(summary.steps == 4, "summary steps mismatch")
    _require(summary.samples == 2, "summary samples mismatch")
    _require(_finite(summary.last_loss), "loss is not finite")
    _require(summary.last_loss > Float32(0.0), "loss did not compute")
    _require(after > before, "LoRA-B did not update")

    print("IDEOGRAM4 TRAIN CORE PASS")
    print("  samples =", summary.samples, " steps =", summary.steps)
    print("  last_loss =", summary.last_loss, " lora_b_l1 =", after)
