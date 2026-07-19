# Ideogram4LiveTrainer.mojo — UI-launched live trainer process.
#
# argv:
#   1 progress_file
#   2 transformer_safetensors
#   3 cache_safetensors
#   4 output_dir
#   5 steps
#   6 rank
#   7 alpha
#   8 learning_rate
#   9 save_every_steps
#  10 caption_dropout_prob   (T1.D; "-" or absent = 0.0 = never drop)
#  11 levers_config_json     (T1 levers: serenitymojo train-config JSON read by
#                             io/train_config_reader read_model_config — the
#                             same format trainer_ui_runner_train_config_json
#                             emits for the config-driven runners; carries
#                             loss_fn/min_snr_gamma_flow/ema_*/optimizer*/
#                             caption_dropout_prob; "-" or absent = all
#                             default-off).
#  12 sample_every_steps     (0 or "-" disables inline sampling)
#  13 sample_steps           (optional, default Ideogram4LoRATrainer setting)
#  14 sample_cfg             (optional, default Ideogram4LoRATrainer setting)
#  15 sample_seed            (optional, default Ideogram4LoRATrainer setting)
#  16 resume_lora_path       (optional; "-" = initialize a new LoRA)
#  17 sample_prompt_json     (optional JSON string or path to a JSON prompt file)
#  18 sample_resolution      (optional square px: 512, 1024, or 2048)
#
# The recipe scalars argv already carries (lr/rank/alpha/steps/save) keep winning
# over the JSON — the JSON contributes ONLY the lever keys. Sampling is explicit
# argv so the UI can enable the resident inline sampler without forcing a levers
# JSON just to carry cadence.
#
# The UI launches this as a background process. Progress is written as
# Serenity-shaped callback lines so TrainerRuntimeBridge can tail the file.

from std.gpu.host import DeviceContext
from std.os import makedirs
from std.sys import argv

from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.sample_prompt_config import read_sample_prompt_config

from serenity_trainer.trainer.Ideogram4LoRATrainer import (
    Ideogram4LoRATrainRunConfig,
    train_ideogram4_lora_from_cache,
)
from serenity_trainer.util.config.TrainConfig import TrainConfig
from serenity_trainer.util.config.TrainConfigReader import _read_file_bytes


comptime DEFAULT_TRANSFORMER = "/home/alex/.serenity/models/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
comptime DEFAULT_CACHE = "/home/alex/trainings/ideogram4_giger_cache/cache.safetensors"
comptime DEFAULT_OUTPUT = "/home/alex/mojodiffusion/output"
comptime DEFAULT_PROGRESS = "target/serenity_trainer_progress.log"

comptime NT = 256   # giger 512px cache bucket (prepare pads ids to 256)
comptime GH = 32    # 512px -> packed 32x32
comptime GW = 32


def _clear_progress(path: String) raises:
    var f = open(path, "w")
    f.write("")
    f.close()


def _append_status(path: String, text: String) raises:
    var f = open(path, "a")
    f.write(
        String("[Serenity-callback] progress epoch 0/1 | step 0/1 | global_step 0 | loss 0.0 | smooth_loss 0.0 | grad_norm 0.0 | lr 0.0 | status ")
        + text
    )
    f.write("\n")
    f.close()


def _looks_like_json_prompt(prompt: String) -> Bool:
    var bs = prompt.as_bytes()
    for i in range(prompt.byte_length()):
        var ch = bs[i]
        if ch == 0x20 or ch == 0x09 or ch == 0x0A or ch == 0x0D:
            continue
        return ch == 0x7B or ch == 0x5B
    return False


def _read_text_file(path: String) raises -> String:
    var bytes = _read_file_bytes(path)
    return String(unsafe_from_utf8=bytes)


def main() raises:
    var args = argv()

    var progress_file = String(DEFAULT_PROGRESS)
    if len(args) > 1:
        var v = String(args[1])
        if v.byte_length() > 0 and v != String("-"):
            progress_file = v^

    var transformer = String(DEFAULT_TRANSFORMER)
    if len(args) > 2:
        var v = String(args[2])
        if v.byte_length() > 0 and v != String("-"):
            transformer = v^

    var cache = String(DEFAULT_CACHE)
    if len(args) > 3:
        var v = String(args[3])
        if v.byte_length() > 0 and v != String("-"):
            cache = v^

    var output = String(DEFAULT_OUTPUT)
    if len(args) > 4:
        var v = String(args[4])
        if v.byte_length() > 0 and v != String("-"):
            output = v^

    var steps = 3000
    if len(args) > 5:
        var v = String(args[5])
        if v.byte_length() > 0 and v != String("-"):
            steps = atol(v)

    var rank = 16
    if len(args) > 6:
        var v = String(args[6])
        if v.byte_length() > 0 and v != String("-"):
            rank = atol(v)

    var alpha = Float32(rank)
    if len(args) > 7:
        var v = String(args[7])
        if v.byte_length() > 0 and v != String("-"):
            alpha = Float32(atof(v))

    var lr = Float32(4.0e-4)
    if len(args) > 8:
        var v = String(args[8])
        if v.byte_length() > 0 and v != String("-"):
            lr = Float32(atof(v))

    var save_every_steps = 500
    if len(args) > 9:
        var v = String(args[9])
        if v.byte_length() > 0 and v != String("-"):
            save_every_steps = atol(v)

    # T1.D caption dropout (argv 10; default-off 0.0)
    var caption_dropout_prob = Float32(0.0)
    if len(args) > 10:
        var v = String(args[10])
        if v.byte_length() > 0 and v != String("-"):
            caption_dropout_prob = Float32(atof(v))

    # T1 levers config JSON (argv 11; optional — fail loud on a bad file)
    var levers_config_path = String("")
    if len(args) > 11:
        var v = String(args[11])
        if v.byte_length() > 0 and v != String("-"):
            levers_config_path = v^

    # Inline sample-during-training controls (argv 12-15). Defaults keep the
    # resident sampler disabled unless the launcher explicitly provides cadence.
    var sample_every_steps = 0
    if len(args) > 12:
        var v = String(args[12])
        if v.byte_length() > 0 and v != String("-"):
            sample_every_steps = atol(v)

    var sample_steps = 0
    if len(args) > 13:
        var v = String(args[13])
        if v.byte_length() > 0 and v != String("-"):
            sample_steps = atol(v)

    var sample_cfg = Float32(0.0)
    if len(args) > 14:
        var v = String(args[14])
        if v.byte_length() > 0 and v != String("-"):
            sample_cfg = Float32(atof(v))

    var sample_seed = UInt64(0)
    var sample_seed_set = False
    if len(args) > 15:
        var v = String(args[15])
        if v.byte_length() > 0 and v != String("-"):
            sample_seed = UInt64(atol(v))
            sample_seed_set = True

    var resume_lora_path = String("")
    if len(args) > 16:
        var v = String(args[16])
        if v.byte_length() > 0 and v != String("-"):
            resume_lora_path = v^

    var sample_prompt_json = String("")
    if len(args) > 17:
        var v = String(args[17])
        if v.byte_length() > 0 and v != String("-"):
            if _looks_like_json_prompt(v):
                sample_prompt_json = v^
            else:
                sample_prompt_json = _read_text_file(v)

    var sample_resolution = 0
    if len(args) > 18:
        var v = String(args[18])
        if v.byte_length() > 0 and v != String("-"):
            sample_resolution = atol(v)

    if steps < 1:
        steps = 1
    if save_every_steps < 0:
        save_every_steps = 0
    if sample_every_steps < 0:
        sample_every_steps = 0

    makedirs(output, exist_ok=True)
    _clear_progress(progress_file)
    _append_status(progress_file, String("Staging Ideogram4 trainer"))
    print(
        "[Ideogram4-lora] model IDEOGRAM_4 | type LoRA | base ",
        transformer,
        " | cache ",
        cache,
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
    # batch_size default 1; the levers config JSON (argv 11) may raise it to 2 to
    # enable the TRUE row-stacked device-grad b2 path (honored after the read below).
    cfg.batch_size = 1
    cfg.gradient_accumulation_steps = 1
    cfg.stochastic_rounding = False
    cfg.seed = UInt32(777)

    var run_cfg = Ideogram4LoRATrainRunConfig.defaults(transformer, cache, output)
    run_cfg.steps = steps
    run_cfg.save_every_steps = save_every_steps
    run_cfg.checkpoint_every_steps = save_every_steps
    run_cfg.progress_file_path = progress_file.copy()
    run_cfg.caption_dropout_prob = caption_dropout_prob
    run_cfg.resume_lora_path = resume_lora_path.copy()
    run_cfg.sample_every_steps = sample_every_steps
    if sample_steps > 0:
        run_cfg.sample_steps = sample_steps
    if sample_cfg > Float32(0.0):
        run_cfg.sample_cfg = sample_cfg
    if sample_seed_set:
        run_cfg.sample_seed = sample_seed
    if sample_resolution > 0:
        run_cfg.sample_resolution = sample_resolution
    # Read the levers config FIRST (argv 11) so its standard
    # validation_prompts_file key (io/train_config_reader.mojo:793) is available
    # as a sample-prompt source below.
    if levers_config_path.byte_length() > 0:
        # Lever keys (loss_fn / min_snr_gamma_flow / ema_* / optimizer* /
        # caption_dropout_prob fallback) come from the JSON; the shared recipe
        # scalars stay argv-owned (the trainer syncs them from `cfg` above).
        run_cfg.levers = read_model_config(levers_config_path)
        # TRUE batch-2: the config JSON's batch_size (default 1) selects the
        # row-stacked device-grad b2 path in the trainer. Only 1 or 2 are wired.
        if run_cfg.levers.batch_size >= 2:
            cfg.batch_size = 2
        print(
            "[Ideogram4-lora] levers config ", levers_config_path,
            " | loss_fn ", run_cfg.levers.loss_fn,
            " | min_snr_gamma_flow ", run_cfg.levers.min_snr_gamma_flow,
            " | optimizer ", run_cfg.levers.optimizer,
            " | ema_enabled ", run_cfg.levers.ema_enabled,
            " | caption_dropout_prob ", run_cfg.levers.caption_dropout_prob,
            " | batch_size ", cfg.batch_size,
        )

    # Sample-prompt source. argv 17 (sample_prompt_json) is the explicit
    # override; when it is absent the levers config's standard
    # validation_prompts_file supplies the inline sampler prompts. ideogram4 is
    # AI-toolkit-oracle, so this only ADDS the standard file source — no argv or
    # key is renamed. The file is consulted only when the inline sampler is
    # enabled (argv 12 > 0); with sampling off the key is timing-inert and
    # silently ignored (it configures the sampler, not a mistraining risk).
    if sample_prompt_json.byte_length() > 0:
        run_cfg.sample_prompts.append(sample_prompt_json.copy())
    elif (
        run_cfg.sample_every_steps > 0
        and run_cfg.levers.validation_prompts_file.byte_length() > 0
    ):
        var spc = read_sample_prompt_config(run_cfg.levers.validation_prompts_file)
        for pi in range(len(spc.prompts)):
            var p = spc.prompts[pi].copy()
            if p.enabled:
                run_cfg.sample_prompts.append(p.prompt.copy())
        print(
            "[Ideogram4-lora] inline sampler prompts from levers"
            " validation_prompts_file ",
            run_cfg.levers.validation_prompts_file,
            " | count ",
            len(run_cfg.sample_prompts),
        )

    if run_cfg.resume_lora_path.byte_length() > 0:
        print("[Ideogram4-lora] resume LoRA ", run_cfg.resume_lora_path)
    if run_cfg.sample_every_steps > 0:
        print(
            "[Ideogram4-lora] inline sampler every ",
            run_cfg.sample_every_steps,
            " steps | sample_steps ",
            run_cfg.sample_steps,
            " | sample_cfg ",
            run_cfg.sample_cfg,
            " | sample_seed ",
            run_cfg.sample_seed,
            " | sample_resolution ",
            run_cfg.sample_resolution,
            " | prompt_json_count ",
            len(run_cfg.sample_prompts),
        )

    var ctx = DeviceContext()
    var summary = train_ideogram4_lora_from_cache[NT, GH, GW](cfg, run_cfg, ctx)
    _append_status(
        progress_file,
        String("Finished Ideogram4 LoRA: loss ")
        + String(summary.last_loss)
        + String(", ")
        + String(summary.seconds_per_step)
        + String(" s/step, saved ")
        + summary.lora_path.copy(),
    )
    print(
        "[Ideogram4-lora] model IDEOGRAM_4 | type LoRA | complete | step ",
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
