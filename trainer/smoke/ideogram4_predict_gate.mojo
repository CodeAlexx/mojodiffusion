# ideogram4_predict_gate.mojo — gate the serenity_trainer Ideogram-4 TRAINING
# predict path (model/Ideogram4Predict) vs the REAL-giger torch oracle
# (ideogram4_fx_predict.safetensors: real image->VAE latent, real .json caption->
# Qwen3-VL features, add_noise, predict_velocity, get_loss_target).
#
# Run (GPU free):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . -I trainer/src \
#       trainer/smoke/ideogram4_predict_gate.mojo
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenity_trainer.model.Ideogram4Predict import (
    ideogram4_build_packed_inputs,
    ideogram4_predict_velocity,
    ideogram4_add_noise,
    ideogram4_flow_target,
)

comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_predict.safetensors"

comptime NT = 651     # giger 10.json caption tokens (chat-templated)
comptime GH = 16
comptime GW = 16
comptime TFLOW = Float32(0.7)


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(FX)
    var clean = Tensor.from_view(fx.tensor_view("clean_latent"), ctx)     # [1,128,16,16] F32
    var noise = Tensor.from_view(fx.tensor_view("noise"), ctx)            # [1,128,16,16] F32
    var llm = cast_tensor(Tensor.from_view(fx.tensor_view("llm_features"), ctx), STDtype.BF16, ctx)  # [1,651,53248]

    # (1) add_noise: (1-t)*clean + t*noise
    var noisy = ideogram4_add_noise[GH, GW](clean, noise, TFLOW, ctx)
    var noisy_exp = Tensor.from_view(fx.tensor_view("noisy"), ctx).to_host(ctx)
    print("add_noise parity:", ParityHarness(0.999).compare(noisy, noisy_exp, ctx))

    # (2) flow target: noise - clean
    var tgt = ideogram4_flow_target(noise, clean, ctx)
    var tgt_exp = Tensor.from_view(fx.tensor_view("target"), ctx).to_host(ctx)
    print("flow_target parity:", ParityHarness(0.999).compare(tgt, tgt_exp, ctx))

    # (3) packed inputs: x / position_ids / indicator
    var packed = ideogram4_build_packed_inputs[NT, GH, GW](noisy, llm, ctx)
    var x_exp = Tensor.from_view(fx.tensor_view("x"), ctx).to_host(ctx)
    print("packed x parity:", ParityHarness(0.999).compare(packed.x, x_exp, ctx))
    var pos_exp = Tensor.from_view(fx.tensor_view("position_ids_f32"), ctx).to_host(ctx)
    print("position_ids parity:", ParityHarness(0.9999).compare(packed.position_ids, pos_exp, ctx))
    var ind_exp = Tensor.from_view(fx.tensor_view("indicator_f32"), ctx).to_host(ctx)
    print("indicator parity:", ParityHarness(0.9999).compare(packed.indicator, ind_exp, ctx))

    # (4) full predict velocity vs torch predict_velocity (the real gate)
    var st = ShardedSafeTensors.open(COND)
    var vel = ideogram4_predict_velocity[NT, GH, GW](st, noisy, TFLOW, llm, ctx)
    print("velocity:", vel.shape()[0], vel.shape()[1], vel.shape()[2], vel.shape()[3])
    var vel_exp = Tensor.from_view(fx.tensor_view("velocity"), ctx).to_host(ctx)
    print("predict velocity parity:", ParityHarness(0.999).compare(vel, vel_exp, ctx))
