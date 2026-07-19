# ideogram4_lora_train_step_gate.mojo — run one real Ideogram-4 LoRA train step.
#
# Uses the real-giger predict fixture:
#   noisy latent + clean latent + noise + Qwen features
# and exercises:
#   split trainer forward -> MSE flow loss -> final backward -> stack LoRA
#   backward -> AdamW update.
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor

from serenity_trainer.model.Ideogram4LoRABlock import build_ideogram4_native_lora_set
from serenity_trainer.trainer.Ideogram4StackTrain import make_ideogram4_lora_adam_state
from serenity_trainer.trainer.Ideogram4LoRATrainStep import ideogram4_lora_train_step
from serenity_trainer.util.config.TrainConfig import TrainConfig


comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_predict.safetensors"

comptime NT = 651
comptime GH = 16
comptime GW = 16
comptime TFLOW = Float32(0.7)


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(FX)
    var st = ShardedSafeTensors.open(COND)

    var clean = Tensor.from_view(fx.tensor_view("clean_latent"), ctx)
    var noise = Tensor.from_view(fx.tensor_view("noise"), ctx)
    var noisy = Tensor.from_view(fx.tensor_view("noisy"), ctx)
    var llm = cast_tensor(
        Tensor.from_view(fx.tensor_view("llm_features"), ctx), STDtype.BF16, ctx
    )

    var loras = build_ideogram4_native_lora_set(16, Float32(16.0), ctx)
    var opt = make_ideogram4_lora_adam_state(loras, ctx)
    var cfg = TrainConfig.adamw_lora_defaults()
    cfg.learning_rate = Float32(1.0e-4)
    cfg.stochastic_rounding = False
    cfg.seed = UInt32(444)

    var result = ideogram4_lora_train_step[NT, GH, GW](
        st, noisy, clean, noise, TFLOW, llm, loras, opt, 1, cfg, ctx
    )

    print("train step loss:", result.loss)
    print("train step adapter_b_l1:", result.adapter_b_l1)
    print("train step did_update:", result.did_update)
    if result.loss != result.loss or result.loss <= Float32(0.0):
        raise Error("Ideogram4 train step produced invalid loss")
    if result.adapter_b_l1 <= Float32(0.0):
        raise Error("Ideogram4 train step did not move LoRA B")
