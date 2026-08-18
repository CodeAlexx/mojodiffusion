# Config seam gate for the production MiniMax-H3 Eri2 recipe.
from std.sys import argv

from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.train_config import (
    TRAIN_DTYPE_BFLOAT_16, TRAIN_OPTIMIZER_ADAMW_8BIT,
)


def _check(ok: Bool, message: String) raises:
    if not ok:
        raise Error(String("H3 config gate: ") + message)


def main() raises:
    var args = argv()
    if len(args) != 2:
        raise Error("usage: minimax_h3_config_smoke <config.json>")
    var cfg = read_model_config(String(args[1]))
    cfg.validate_serenity_trainer_policy_config()
    cfg.validate_offload_checkpoint_config()
    _check(cfg.name == String("minimax_h3"), String("model_type"))
    _check(cfg.train_dtype == TRAIN_DTYPE_BFLOAT_16, String("BF16 compute"))
    _check(cfg.optimizer == TRAIN_OPTIMIZER_ADAMW_8BIT, String("AdamW8bit"))
    _check(
        cfg.beta1 == Float32(0.9) and cfg.beta2 == Float32(0.999)
        and cfg.eps == Float32(1.0e-8)
        and cfg.weight_decay == Float32(0.01),
        String("optimizer parameters"),
    )
    _check(cfg.quantized_resident == String("int_w8a8"), String("W8A8 base"))
    _check(cfg.lora_rank == 32 and cfg.lora_alpha == Float32(32.0), String("rank/alpha"))
    _check(
        cfg.layer_filter_preset == String("h3_mlp_fc1_fc2")
        and cfg.layer_filter_regex,
        String("MLP-only target selector"),
    )
    _check(cfg.max_steps == 4000, String("max_steps"))
    _check(cfg.save_every == 750 and cfg.sample_every == 750, String("cadence"))
    _check(cfg.lr == Float32(0.00005), String("learning rate"))
    _check(cfg.lr_warmup_steps == 50, String("warmup"))
    _check(cfg.max_grad_norm == Float32(1.0), String("grad clipping"))
    _check(cfg.guidance_scale == Float32(3.5), String("guidance"))
    _check(cfg.h3_num_timestep_buckets == 8, String("timestep buckets"))
    _check(cfg.h3_spatial_density_jitter == Float32(0.2), String("density jitter"))
    _check(
        cfg.h3_base_preservation_loss_weight == Float32(0.02),
        String("base preservation"),
    )
    _check(
        cfg.h3_base_preservation_probability == Float32(0.25),
        String("sparse base preservation"),
    )
    _check(cfg.resident_blocks == 46, String("resident blocks"))
    _check(cfg.dataset_path == String("/home/alex/eri2_with_trigger"), String("dataset"))
    _check(cfg.validation_prompts_file != String(""), String("sample config"))
    print("PASS: MiniMax-H3 production config recipe and paths parsed exactly")
