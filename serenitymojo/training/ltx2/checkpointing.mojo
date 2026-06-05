# checkpointing.mojo -- LTX-2 checkpoint/save/resume contract stubs.

from serenitymojo.training.ltx2.config import LTX2TrainerConfig


comptime CHECKPOINTS_DIR = "checkpoints"
comptime SAMPLES_DIR = "samples"
comptime LORA_LATEST = "lora_latest.safetensors"
comptime LORA_FINAL = "lora_final.safetensors"
comptime LORA_EMERGENCY = "lora_emergency.safetensors"
comptime TRAIN_STATE_SUFFIX = ".train_state.safetensors"


def _zero_pad(step: Int, width: Int) -> String:
    var raw = String(step)
    var out = String("")
    var pads = width - raw.byte_length()
    for _ in range(pads):
        out += String("0")
    return out + raw


def lobo_step_lora_filename(step: Int) -> String:
    return String("lora_step_") + _zero_pad(step, 6) + String(".safetensors")


def musubi_step_lora_filename(job_name: String, step: Int) -> String:
    return job_name + String("-step") + _zero_pad(step, 8) + String(".safetensors")


def train_state_filename(step: Int) -> String:
    return String("train_state_step_") + _zero_pad(step, 6) + String(TRAIN_STATE_SUFFIX)


def checkpoint_dir(output_dir: String) -> String:
    return output_dir + String("/") + String(CHECKPOINTS_DIR)


def lobo_step_lora_path(output_dir: String, step: Int) -> String:
    return checkpoint_dir(output_dir) + String("/") + lobo_step_lora_filename(step)


def lora_latest_path(output_dir: String) -> String:
    return output_dir + String("/") + String(LORA_LATEST)


def lora_final_path(output_dir: String) -> String:
    return output_dir + String("/") + String(LORA_FINAL)


def lora_emergency_path(output_dir: String) -> String:
    return output_dir + String("/") + String(LORA_EMERGENCY)


def resume_token_is_latest(token: String) -> Bool:
    return token == "latest" or token == "auto"


def optimizer_state_key(param_idx: Int, kind: String) -> String:
    if kind == "master":
        return String("param.") + String(param_idx)
    if kind == "m":
        return String("adam_m.") + String(param_idx)
    if kind == "v":
        return String("adam_v.") + String(param_idx)
    return String("__meta__")


def save_checkpoint_contract(cfg: LTX2TrainerConfig, step: Int) -> String:
    return (
        String("save LoRA A/B to ") + lobo_step_lora_path(cfg.output_dir, step)
        + String("; save optimizer state to ")
        + checkpoint_dir(cfg.output_dir) + String("/") + train_state_filename(step)
        + String("; update ") + lora_latest_path(cfg.output_dir)
    )


def resume_checkpoint_contract(cfg: LTX2TrainerConfig) -> String:
    if cfg.resume_from == "":
        return String("new run")
    if resume_token_is_latest(cfg.resume_from):
        return String("resume latest numbered lora_step_*.safetensors, then sidecars")
    return String("resume explicit checkpoint: ") + cfg.resume_from


def save_ltx2_checkpoint_stub() raises:
    raise Error("LTX2 checkpoint save is a contract stub until AV LoRA parameter storage is wired")


def load_ltx2_checkpoint_stub() raises:
    raise Error("LTX2 checkpoint resume is a contract stub until AV LoRA parameter storage is wired")
