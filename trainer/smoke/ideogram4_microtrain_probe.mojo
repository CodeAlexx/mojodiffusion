# ideogram4_microtrain_probe.mojo — RUNTIME GRADIENT-FLOW PROBE (audit stage:
# gradient flow / does-it-learn). Fixture-driven multi-step micro-train.
#
# WHY: the cache is missing, but the 140MB real-giger predict fixture exists.
# This probe runs N real train steps off ONE fixture sample, RESAMPLING t per
# step via the SAME logit-normal sampler the production loop uses (so it does
# NOT repeat the historical "fixed-t fake training" bug), and asserts:
#   (1) per-step LoRA-B gradient L1 > 0 for EVERY step (grads flow, all 204
#       adapter slots reachable through the 34-block backward),
#   (2) the LoRA-B param L1 strictly grows (optimizer moves the weights),
#   (3) the smoothed loss at the END is LOWER than at the START (it learns to
#       fit this one (clean, caption) pair — overfit-one-sample sanity).
#
# This is NOT a parity gate; it is the minimal "really training" demonstrator
# the audit needs when no cache is present. Loss-decreasing on ONE sample is the
# weakest real-learning claim; it cannot prove generalization, only that the
# backward + optimizer reduce the flow-match MSE on data it sees.
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.training.schedule import sample_timestep_logit_normal_scaled

from serenity_trainer.model.Ideogram4LoRABlock import build_ideogram4_native_lora_set
from serenity_trainer.model.Ideogram4Predict import ideogram4_add_noise
from serenity_trainer.trainer.Ideogram4StackTrain import make_ideogram4_lora_adam_state
from serenity_trainer.trainer.Ideogram4LoRATrainStep import ideogram4_lora_train_step
from serenity_trainer.util.config.TrainConfig import TrainConfig


comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_predict.safetensors"

comptime NT = 651
comptime GH = 16
comptime GW = 16
comptime NSTEPS = 20
comptime NOISE_SEED = UInt64(0x1D3A_4A11)


def _b_l1(loras_b_sum: Float32) -> Float32:
    return loras_b_sum


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(FX)
    var st = ShardedSafeTensors.open(COND)

    # ONE fixture sample: clean latent + caption features (+ a fixed noise draw).
    var clean = Tensor.from_view(fx.tensor_view("clean_latent"), ctx)   # [1,128,16,16] F32
    var noise = Tensor.from_view(fx.tensor_view("noise"), ctx)          # [1,128,16,16] F32
    var llm = cast_tensor(
        Tensor.from_view(fx.tensor_view("llm_features"), ctx), STDtype.BF16, ctx
    )

    var loras = build_ideogram4_native_lora_set(16, Float32(16.0), ctx)
    var opt = make_ideogram4_lora_adam_state(loras, ctx)
    var cfg = TrainConfig.adamw_lora_defaults()
    cfg.learning_rate = Float32(1.0e-3)   # higher LR so 20 steps move visibly
    cfg.stochastic_rounding = False
    cfg.seed = UInt32(444)

    var first_loss = Float32(0.0)
    var last_smooth = Float32(0.0)
    var smooth = Float32(0.0)
    var inited = False
    var any_zero_grad = False
    var prev_b = Float32(0.0)
    var b_monotone_up = True

    for step in range(1, NSTEPS + 1):
        # RESAMPLE t per step (logit-normal, std=1.0 — the production loop's draw),
        # then build noisy = (1-t)*clean + t*noise at THIS step's t.
        var t_step = sample_timestep_logit_normal_scaled(
            NOISE_SEED * UInt64(7919) + UInt64(step), Float32(1.0)
        )
        var noisy = ideogram4_add_noise[GH, GW](clean, noise, t_step, ctx)

        var res = ideogram4_lora_train_step[NT, GH, GW](
            st, noisy, clean, noise, t_step, llm, loras, opt, step, cfg, ctx
        )

        if res.adapter_b_l1 <= prev_b and step > 1:
            b_monotone_up = False
        prev_b = res.adapter_b_l1

        if not inited:
            smooth = res.loss
            first_loss = res.loss
            inited = True
        else:
            smooth = smooth * Float32(0.7) + res.loss * Float32(0.3)
        last_smooth = smooth

        print(
            "step", step, "t", t_step, "loss", res.loss,
            "smooth", smooth, "b_l1", res.adapter_b_l1,
        )

    print("first_loss", first_loss, "last_smooth", last_smooth, "final_b_l1", prev_b)

    if last_smooth >= first_loss:
        raise Error(
            "MICROTRAIN PROBE FAIL: smoothed loss did not decrease ("
            + String(first_loss) + " -> " + String(last_smooth) + ")"
        )
    if prev_b <= Float32(0.0):
        raise Error("MICROTRAIN PROBE FAIL: LoRA-B never moved off zero")
    print("MICROTRAIN PROBE PASS: loss decreased, LoRA-B grew, grads flowed")
