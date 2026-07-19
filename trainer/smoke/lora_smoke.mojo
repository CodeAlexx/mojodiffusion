# lora_smoke.mojo — COMPILE smoke for the lora unit. Exercises the public
# surface of lora.mojo / lora_save.mojo / io/weights.mojo so the orchestrator's
# integrated build type-checks the whole unit. Not run here (build-only; GPU-free
# build per unit instructions). Do NOT run.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.io.dtype import STDtype
from serenitymojo.autograd import Tape, backward
from serenitymojo.ops.tensor_algebra import zeros_device
from serenity_trainer.module.LoRAModule import (
    LoraAdapter, make_lora_adapter, lora_linear_forward,
)
from serenity_trainer.modelSaver.GenericLoRAModelSaver import (
    save_lora, save_lora_one, load_lora_one,
)
from serenity_trainer.modelLoader.mixin.HFModelLoaderMixin import (
    load_base_weights, load_named_weights,
)


def main() raises:
    var ctx = DeviceContext()

    var in_f = 8
    var out_f = 16
    var rank = 4
    var alpha = Float32(8.0)

    # build an adapter (A=kaiming-ish, B=0)
    var ad = make_lora_adapter(in_f, out_f, rank, alpha, UInt64(1234), ctx)
    _ = ad.scale()

    # frozen base weight [out, in] (untracked) + an input [M, in]
    var bw_sh = List[Int](); bw_sh.append(out_f); bw_sh.append(in_f)
    var base_w = zeros_device(bw_sh^, STDtype.BF16, ctx)
    var x_sh = List[Int](); x_sh.append(2); x_sh.append(in_f)
    var x = zeros_device(x_sh^, STDtype.BF16, ctx)

    # tape forward: track adapter, run LoRA-wrapped linear
    var tape = Tape()
    ad.track(tape)
    var y = lora_linear_forward(tape, x, base_w, ad, ctx)
    # build a trivial loss to exercise backward over the LoRA path
    var tgt = zeros_device(y.shape().copy(), STDtype.BF16, ctx)
    var loss = tape.mse_loss(y, tgt, ctx)
    var grads = backward(tape, loss, ctx)
    _ = grads

    # PEFT save / load round-trip surface
    save_lora_one(String("layer0"), ad^, String("/tmp/serenity_lora_smoke.safetensors"), ctx)
    var reloaded = load_lora_one(
        String("/tmp/serenity_lora_smoke.safetensors"), String("layer0"), alpha, ctx
    )
    _ = reloaded.rank

    # multi-adapter save surface
    var ad2 = make_lora_adapter(in_f, out_f, rank, alpha, UInt64(7), ctx)
    var pxs = List[String](); pxs.append(String("layerA"))
    var boxed = List[ArcPointer[LoraAdapter]](); boxed.append(ArcPointer(ad2^))
    save_lora(pxs^, boxed^, String("/tmp/serenity_lora_multi.safetensors"), ctx)

    # base-weight loader surface
    var want = List[String](); want.append(String("model.weight"))
    _ = load_named_weights  # reference (no file to load in smoke)
    _ = load_base_weights
    _ = want

    print("lora smoke OK")
