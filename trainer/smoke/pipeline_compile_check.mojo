# pipeline_compile_check.mojo — Phase 2 compile gate for the model-agnostic
# pipeline (train_config, lr_schedule, model_spec, loss, grad, flow_target).
# Compile only (GPU-free):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -I trainer/src \
#       trainer/smoke/pipeline_compile_check.mojo -o /tmp/pipe_check

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.util.lr_scheduler_util import LrSchedule, resolve_schedule, LR_COSINE
from serenity_trainer.modelSetup.BaseModelSetup import StepOutput
from serenity_trainer.modelSetup.mixin.ModelSetupDiffusionLossMixin import timestep_weight, loss_scale, LW_MIN_SNR_GAMMA, LS_BOTH
from serenity_trainer.util.grad_util import clip_grad_norm, accumulate
from serenity_trainer.modelSetup.mixin.ModelSetupFlowMatchingMixin import build_flow_target

comptime TArc = ArcPointer[Tensor]


def main() raises:
    var ctx = DeviceContext()

    # lr_schedule + train_config (pure host)
    var cfg = TrainConfig.adamw_lora_defaults()
    var sched = resolve_schedule(LR_COSINE, Float64(100.0), Float64(1.0), Float64(0.0), 1000, 1, 1)
    print("lr@0 =", sched.factor(0), " lr@550 =", sched.factor(550))

    # loss weighting (host scalars)
    print("min_snr weight =", timestep_weight(LW_MIN_SNR_GAMMA, Float32(2.0), Float32(5.0), False))
    print("loss_scale =", loss_scale(LS_BOTH, cfg.batch_size, cfg.gradient_accumulation_steps))

    # tensors
    var v = List[Float32]()
    for i in range(8):
        v.append(Float32(i) * 0.1 - 0.4)
    var sh = List[Int](); sh.append(8)
    var p = Tensor.from_host(v.copy(), sh.copy(), STDtype.BF16, ctx)
    var t = Tensor.from_host(v.copy(), sh.copy(), STDtype.BF16, ctx)

    # model_spec StepOutput
    var so = StepOutput(p^, t^, Float32(0.5))
    print("StepOutput t =", so.timestep)

    # flow_target
    var latent = Tensor.from_host(v.copy(), sh.copy(), STDtype.BF16, ctx)
    var ft = build_flow_target(latent, Float32(0.3), UInt64(7), ctx)
    print("flow noisy dtype =", ft.noisy.dtype().name(), " target dtype =", ft.target.dtype().name())

    # grad utils — List[ArcPointer[Tensor]] (move-only Tensor boxed)
    var g = Tensor.from_host(v.copy(), sh.copy(), STDtype.BF16, ctx)
    var grads = List[TArc]()
    grads.append(TArc(g^))
    var pre_norm = clip_grad_norm(grads, Float32(1.0), ctx)
    print("pre-clip grad norm =", pre_norm)

    print("PIPELINE COMPILE-CHECK OK")
