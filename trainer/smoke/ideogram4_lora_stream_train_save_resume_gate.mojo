# ideogram4_lora_stream_train_save_resume_gate.mojo — staged full-block
# Ideogram4 LoRA trainer gate.
#
# Proves the trainer driver can:
#   1) stage cache metadata + one transformer weight set,
#   2) stream one real Giger sample into the train step,
#   3) save ai-toolkit/PEFT LoRA plus Adam train state,
#   4) reload LoRA + train state and resume.

from std.gpu.host import DeviceContext

from serenity_trainer.modelLoader.Ideogram4LoRALoader import (
    load_ideogram4_block_stack_lora,
)
from serenity_trainer.trainer.Ideogram4LoRATrainer import (
    Ideogram4LoRATrainRunConfig,
    train_ideogram4_lora_from_cache,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig


comptime COND = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_predict.safetensors"
comptime OUT_A = "/tmp/serenity_i4_lora_stream_gate"
comptime OUT_B = "/tmp/serenity_i4_lora_stream_gate_resume"

comptime NT = 651
comptime GH = 16
comptime GW = 16


def main() raises:
    var ctx = DeviceContext()

    var cfg = TrainConfig.adamw_lora_defaults()
    cfg.lora_rank = 4
    cfg.lora_alpha = Float32(4.0)
    cfg.learning_rate = Float32(1.0e-4)
    cfg.stochastic_rounding = False
    cfg.batch_size = 1
    cfg.seed = UInt32(777)

    var first_cfg = Ideogram4LoRATrainRunConfig.defaults(
        String(COND), String(FX), String(OUT_A)
    )
    first_cfg.steps = 1
    first_cfg.save_every_steps = 1
    first_cfg.checkpoint_every_steps = 1

    var first = train_ideogram4_lora_from_cache[NT, GH, GW](
        cfg, first_cfg, ctx
    )
    print("first run loss:", first.last_loss)
    print("first run seconds_per_step:", first.seconds_per_step)
    print("first run elapsed_seconds:", first.elapsed_seconds)
    print("first run adapter_b_l1:", first.adapter_b_l1)
    print("first run optimizer_steps:", first.optimizer_steps)
    print("first run lora:", first.lora_path)
    print("first run state:", first.state_dir)

    if first.loaded_weight_sets != 1:
        raise Error("stream trainer loaded more than one transformer weight set")
    if first.optimizer_steps != 1 or first.progress.global_step != 1:
        raise Error("first stream train progress mismatch")
    if first.last_loss <= Float32(0.0) or first.last_loss != first.last_loss:
        raise Error("first stream train loss invalid")
    if first.adapter_b_l1 <= Float32(0.0):
        raise Error("first stream train did not move LoRA B")

    var loaded = load_ideogram4_block_stack_lora(first.lora_path, ctx)
    if len(loaded.ad) != 204:
        raise Error("saved Ideogram4 LoRA did not reload as 204 adapters")
    if loaded.rank != 4:
        raise Error("saved Ideogram4 LoRA rank mismatch")

    var resume_cfg = Ideogram4LoRATrainRunConfig.defaults(
        String(COND), String(FX), String(OUT_B)
    )
    resume_cfg.steps = 1
    resume_cfg.resume_lora_path = first.lora_path
    resume_cfg.resume_state_dir = first.state_dir

    var second = train_ideogram4_lora_from_cache[NT, GH, GW](
        cfg, resume_cfg, ctx
    )
    print("resume run loss:", second.last_loss)
    print("resume seconds_per_step:", second.seconds_per_step)
    print("resume elapsed_seconds:", second.elapsed_seconds)
    print("resume run adapter_b_l1:", second.adapter_b_l1)
    print("resume optimizer_steps:", second.optimizer_steps)
    print("resume lora:", second.lora_path)
    print("resume state:", second.state_dir)

    if second.optimizer_steps != 2 or second.progress.global_step != 2:
        raise Error("resume stream train progress mismatch")
    if second.adapter_b_l1 <= first.adapter_b_l1:
        raise Error("resume stream train did not continue adapter update")

    print("IDEOGRAM4 LORA STREAM TRAIN SAVE RESUME PASS")
