# ideogram4_lora_b0_gate.mojo — gate the Ideogram-4 trainer split forward.
#
# The sampler/predict path calls the monolithic serenitymojo ideogram4_forward.
# The trainer has to split that into:
#   frozen embed -> LoRA block stack -> frozen final -> velocity
#
# At cold start LoRA B=0, the LoRA stack is an identity overlay on the base
# weights. Therefore this split trainer forward must match the same real-giger
# torch predict fixture used by ideogram4_predict_gate.
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.reduce import reduce_mean_f32
from serenitymojo.ops.tensor_algebra import sub, mul

from serenity_trainer.model.Ideogram4LoRABlock import build_ideogram4_native_lora_set
from serenity_trainer.trainer.Ideogram4LoRATrainStep import ideogram4_lora_train_forward


comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_predict.safetensors"

comptime NT = 651
comptime GH = 16
comptime GW = 16
comptime TFLOW = Float32(0.7)


def _mse(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float32:
    var diff = sub(a, b, ctx)
    var sq = mul(diff, diff, ctx)
    var dims = List[Int]()
    dims.append(0)
    dims.append(1)
    dims.append(2)
    dims.append(3)
    return reduce_mean_f32(sq, dims^, False, ctx).to_host(ctx)[0]


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(FX)
    var st = ShardedSafeTensors.open(COND)

    var noisy = Tensor.from_view(fx.tensor_view("noisy"), ctx)
    var target = Tensor.from_view(fx.tensor_view("target"), ctx)
    var llm = cast_tensor(
        Tensor.from_view(fx.tensor_view("llm_features"), ctx), STDtype.BF16, ctx
    )
    var velocity_exp = Tensor.from_view(fx.tensor_view("velocity"), ctx).to_host(ctx)

    var loras = build_ideogram4_native_lora_set(16, Float32(16.0), ctx)
    var fwd = ideogram4_lora_train_forward[NT, GH, GW](
        st, noisy, TFLOW, llm, loras, ctx
    )

    print("split velocity:", fwd.velocity.shape()[0], fwd.velocity.shape()[1], fwd.velocity.shape()[2], fwd.velocity.shape()[3])
    print("B=0 split velocity parity:", ParityHarness(0.999).compare(fwd.velocity, velocity_exp, ctx))
    print("B=0 split loss:", _mse(fwd.velocity, target, ctx))
