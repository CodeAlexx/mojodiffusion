# train_klein_cadence.mojo — process-separated Klein LoRA cadence supervisor.
#
# This is the production entry for Klein 9B LoRA training with validation:
#   1. launch train_klein_real as a worker for one global step chunk,
#   2. let that worker exit so CUDA releases the training stack/scratch slabs,
#   3. launch klein_sample_cli once per prompt in its own process,
#   4. repeat until cfg.max_steps.
#
# The separation is intentional. In-process 1024px sampling after step 500 can
# OOM because the trainer's DeviceContext may still hold resident allocations.

from sys import argv
from std.collections import List

from serenitymojo.io.ffi import sys_system, sys_open, sys_close, O_RDONLY
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.sample_prompt_config import (
    SamplePrompt, SamplePromptConfig, read_sample_prompt_config,
)
from serenitymojo.training.serenityboard import SerenityBoardWriter


comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/klein9b.json"
comptime DEFAULT_TRAINER_BIN = "/home/alex/mojodiffusion/output/bin/train_klein_real"
comptime DEFAULT_SAMPLER_BIN = "/home/alex/mojodiffusion/output/bin/klein_sample_cli"
comptime OUT_DIR = "/home/alex/mojodiffusion/output/alina_train"


def _parse_nonnegative_int(s: String) raises -> Int:
    var out = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        var ch = bs[i]
        if ch < 0x30 or ch > 0x39:
            raise Error(String("expected integer, got ") + s)
        out = out * 10 + Int(ch - 0x30)
    return out


def _path_exists(path: String) -> Bool:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return False
    _ = sys_close(fd)
    return True


def _step_lora_path(step: Int) -> String:
    return OUT_DIR + String("/alina_lora_step") + String(step) + String(".safetensors")


def _lora_path_for_sample(step: Int, max_steps: Int) -> String:
    if step >= max_steps:
        return OUT_DIR + String("/alina_lora_final.safetensors")
    return _step_lora_path(step)


def _sample_png_path(step: Int, label: String) -> String:
    return OUT_DIR + String("/sample_step") + String(step) + String("_") + label + String(".png")


def _latest_step_checkpoint(max_steps: Int, sample_every: Int) -> Int:
    if sample_every <= 0:
        return 0
    var latest = 0
    var s = sample_every
    while s < max_steps:
        if _path_exists(_step_lora_path(s)):
            latest = s
        s += sample_every
    return latest


def _run_command(label: String, cmd: String) raises:
    print("[Klein-cadence]", label, "cmd:", cmd)
    var status = sys_system(cmd)
    if status != 0:
        raise Error(
            String("command failed status=") + String(status)
            + String(" label=") + label
        )


def _sample_one(
    cfg_path: String,
    prompt_file: String,
    sampler_bin: String,
    prompt: SamplePrompt,
    step: Int,
    lora_path: String,
    board: SerenityBoardWriter,
    seq_index: Int,
) raises:
    var out_png = _sample_png_path(step, prompt.label)
    if _path_exists(out_png):
        print("[Klein-cadence] regenerating existing sample step=", step, " label=", prompt.label, " path=", out_png)
        _ = sys_system(String("rm -f ") + out_png)
    var lora_arg = lora_path.copy()
    if lora_arg == String(""):
        lora_arg = String("\"\"")
    var cmd = (
        String("MODULAR_DEVICE_CONTEXT_SYNC_MODE=true ")
        + sampler_bin + String(" ") + cfg_path + String(" ") + lora_arg
        + String(" ") + prompt_file + String(" ") + prompt.label
        + String(" ") + out_png
    )
    _run_command(String("sample step=") + String(step) + String(" label=") + prompt.label, cmd)
    if _path_exists(out_png):
        board.log_image_png(String("samples/") + prompt.label, step, seq_index, out_png)
    else:
        raise Error(String("sampler did not produce output png: ") + out_png)
    board.log_text(String("prompts/") + prompt.label, step, prompt.prompt)


def _sample_all(
    cfg_path: String,
    prompt_file: String,
    sample_cfg: SamplePromptConfig,
    sampler_bin: String,
    step: Int,
    lora_path: String,
    board: SerenityBoardWriter,
) raises:
    for i in range(len(sample_cfg.prompts)):
        var p = sample_cfg.prompts[i].copy()
        _sample_one(
            cfg_path, prompt_file, sampler_bin, p^, step, lora_path,
            board, i,
        )


def main() raises:
    _ = sys_system(String("mkdir -p ") + OUT_DIR)
    _ = sys_system(String("mkdir -p /home/alex/mojodiffusion/output/bin"))

    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    if len(a) >= 2:
        cfg_path = String(a[1])
    var trainer_bin = String(DEFAULT_TRAINER_BIN)
    if len(a) >= 3:
        trainer_bin = String(a[2])
    var sampler_bin = String(DEFAULT_SAMPLER_BIN)
    if len(a) >= 4:
        sampler_bin = String(a[3])

    var cfg = read_model_config(cfg_path)
    if cfg.validation_prompts_file == String(""):
        raise Error("train_klein_cadence: config must set validation_prompts_file")
    var sample_cfg = read_sample_prompt_config(cfg.validation_prompts_file)
    var sample_every = sample_cfg.every_steps
    if sample_every <= 0:
        sample_every = cfg.sample_every
    if sample_every <= 0:
        raise Error("train_klein_cadence: sample_every must be > 0")

    var current = _latest_step_checkpoint(cfg.max_steps, sample_every)
    if len(a) >= 5:
        current = _parse_nonnegative_int(String(a[4]))
    if current > cfg.max_steps:
        current = cfg.max_steps
    var mode = String("")
    if len(a) >= 6:
        mode = String(a[5])

    var board = SerenityBoardWriter.open(String(OUT_DIR), String("klein_lora_cadence"), current)
    board.log_text(String("config/train"), current, cfg_path)
    board.log_text(String("config/sample_prompts"), current, cfg.validation_prompts_file)

    print("=== Klein LoRA cadence supervisor ===")
    print("  config:", cfg_path)
    print("  trainer:", trainer_bin)
    print("  sampler:", sampler_bin)
    print("  prompts:", cfg.validation_prompts_file, " count=", len(sample_cfg.prompts))
    print("  resume step:", current, " max_steps:", cfg.max_steps, " sample_every:", sample_every, " mode:", mode)

    if current == 0 and sample_cfg.sample_at_start:
        print("[Klein-cadence] step 0 baseline samples")
        _sample_all(cfg_path, cfg.validation_prompts_file, sample_cfg, sampler_bin, 0, String(""), board)
    elif current > 0:
        var resume_lora = _lora_path_for_sample(current, cfg.max_steps)
        print("[Klein-cadence] validation samples for existing checkpoint step=", current)
        _sample_all(cfg_path, cfg.validation_prompts_file, sample_cfg, sampler_bin, current, resume_lora, board)

    if mode == String("sampleonly"):
        board.set_status(String("complete"))
        board.close()
        print("DONE Klein cadence sampleonly step", current)
        return

    while current < cfg.max_steps:
        var end_step = current + sample_every
        if end_step > cfg.max_steps:
            end_step = cfg.max_steps
        var resume_arg = String("-")
        if current > 0:
            resume_arg = _lora_path_for_sample(current, cfg.max_steps)
        var train_cmd = (
            trainer_bin + String(" ") + cfg_path
            + String(" ") + String(end_step)
            + String(" ") + String(current)
            + String(" ") + resume_arg
            + String(" nosample")
        )
        _run_command(
            String("train ") + String(current + 1) + String("..") + String(end_step),
            train_cmd,
        )
        current = end_step
        var lora_for_sample = _lora_path_for_sample(current, cfg.max_steps)
        print("[Klein-cadence] sample checkpoint step=", current, " lora=", lora_for_sample)
        _sample_all(cfg_path, cfg.validation_prompts_file, sample_cfg, sampler_bin, current, lora_for_sample, board)

    board.set_status(String("complete"))
    board.close()
    print("DONE Klein cadence reached step", current)
