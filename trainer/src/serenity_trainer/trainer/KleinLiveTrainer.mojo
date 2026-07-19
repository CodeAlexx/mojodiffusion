# KleinLiveTrainer.mojo — UI-launched cached Klein/Flux2 LoRA trainer.
#
# argv:
#   1 progress_file
#   2 checkpoint_safetensors
#   3 cache_safetensors
#   4 dataset_path
#   5 output_dir
#   6 steps
#   7 rank
#   8 alpha
#   9 learning_rate
#   10 save_every_steps
#   11 vae_safetensors

from std.gpu.host import DeviceContext
from std.os import makedirs
from std.sys import argv

from serenity_trainer.trainer.KleinLoRATrainer import (
    DEFAULT_KLEIN_VAE,
    KleinLoRATrainRunConfig,
    train_klein_lora_auto,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig


comptime DEFAULT_CHECKPOINT = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"
comptime DEFAULT_DATASET = "/home/alex/1/datasets/boxjana"
comptime DEFAULT_CACHE = "/home/alex/1/datasets/boxjana/_klein_cache/boxjana_klein512.safetensors"
comptime DEFAULT_OUTPUT = "/home/alex/mojodiffusion/output"
comptime DEFAULT_PROGRESS = "target/serenity_trainer_progress.log"


def _clear_progress(path: String) raises:
    var f = open(path, "w")
    f.write("")
    f.close()


def _append_status(path: String, text: String) raises:
    if path.byte_length() == 0:
        return
    var f = open(path, "a")
    f.write(
        String("[Serenity-callback] progress epoch 1/1 | step 0/1 | global_step 0 | loss 0.0 | smooth_loss 0.0 | grad_norm 0.0 | lr 0.0 | status ")
        + text
    )
    f.write("\n")
    f.close()


def main() raises:
    var args = argv()

    var progress_file = String(DEFAULT_PROGRESS)
    if len(args) > 1:
        var v = String(args[1])
        if v.byte_length() > 0 and v != String("-"):
            progress_file = v^

    var checkpoint = String(DEFAULT_CHECKPOINT)
    if len(args) > 2:
        var v = String(args[2])
        if v.byte_length() > 0 and v != String("-"):
            checkpoint = v^

    var cache = String(DEFAULT_CACHE)
    if len(args) > 3:
        var v = String(args[3])
        if v.byte_length() > 0 and v != String("-"):
            cache = v^

    var dataset = String(DEFAULT_DATASET)
    if len(args) > 4:
        var v = String(args[4])
        if v.byte_length() > 0 and v != String("-"):
            dataset = v^

    var output = String(DEFAULT_OUTPUT)
    if len(args) > 5:
        var v = String(args[5])
        if v.byte_length() > 0 and v != String("-"):
            output = v^
    var vae_path = String(DEFAULT_KLEIN_VAE)
    if len(args) > 11:
        var v = String(args[11])
        if v.byte_length() > 0 and v != String("-"):
            vae_path = v^

    var steps = 3000
    if len(args) > 6:
        var v = String(args[6])
        if v.byte_length() > 0 and v != String("-"):
            steps = atol(v)
    var rank = 16
    if len(args) > 7:
        var v = String(args[7])
        if v.byte_length() > 0 and v != String("-"):
            rank = atol(v)
    var alpha = Float32(rank)
    if len(args) > 8:
        var v = String(args[8])
        if v.byte_length() > 0 and v != String("-"):
            alpha = Float32(atof(v))
    var lr = Float32(4.0e-4)
    if len(args) > 9:
        var v = String(args[9])
        if v.byte_length() > 0 and v != String("-"):
            lr = Float32(atof(v))
    var save_every_steps = 500
    if len(args) > 10:
        var v = String(args[10])
        if v.byte_length() > 0 and v != String("-"):
            save_every_steps = atol(v)

    if steps < 1:
        steps = 1
    if save_every_steps < 0:
        save_every_steps = 0

    makedirs(output, exist_ok=True)
    _clear_progress(progress_file)
    _append_status(progress_file, String("Staging Klein 9B cached trainer"))
    print(
        "[Klein-lora] model FLUX_2 | type LoRA | base ",
        checkpoint,
        " | cache ",
        cache,
        " | dataset ",
        dataset,
        " | output ",
        output,
        " | steps ",
        steps,
        " | rank ",
        rank,
        " | lr ",
        lr,
        " | save_every ",
        save_every_steps,
    )

    var cfg = TrainConfig.adamw_lora_defaults()
    cfg.lora_rank = rank
    cfg.lora_alpha = alpha
    cfg.learning_rate = lr
    cfg.batch_size = 1
    cfg.gradient_accumulation_steps = 1
    cfg.stochastic_rounding = False
    cfg.seed = UInt32(777)

    var run_cfg = KleinLoRATrainRunConfig.defaults(checkpoint, cache, dataset, output)
    run_cfg.steps = steps
    run_cfg.save_every_steps = save_every_steps
    run_cfg.progress_file_path = progress_file.copy()
    run_cfg.vae_path = vae_path.copy()

    var ctx = DeviceContext()
    try:
        var summary = train_klein_lora_auto(cfg, run_cfg, ctx)
        print(
            "[Klein-lora] model FLUX_2 | type LoRA | complete | step ",
            summary.optimizer_steps,
            "/",
            summary.steps_ran,
            " | loss ",
            summary.last_loss,
            " | ",
            Float32(summary.seconds_per_step),
            "s/step | saved ",
            summary.lora_path,
        )
    except e:
        var msg = String("Klein trainer failed: ") + String(e)
        _append_status(progress_file, msg.copy())
        print("[Klein-lora] model FLUX_2 | type LoRA | failed | ", msg)
        raise Error(msg)
