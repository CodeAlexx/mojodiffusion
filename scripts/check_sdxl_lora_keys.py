#!/usr/bin/env python3
"""Static/header guard for SDXL OneTrainer LoRA key parity."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")
TRAIN_REF_PREFIX = Path("/home/alex/onetrainer-mojo/parity/sdxl_train_ref_step000")

OT_SETUP = ONETRAINER / "modules/modelSetup/StableDiffusionXLLoRASetup.py"
OT_BASE_SETUP = ONETRAINER / "modules/modelSetup/BaseStableDiffusionXLSetup.py"
OT_DATALOADER = ONETRAINER / "modules/dataLoader/StableDiffusionXLBaseDataLoader.py"
OT_SAMPLER = ONETRAINER / "modules/modelSampler/StableDiffusionXLSampler.py"
OT_TRAINER = ONETRAINER / "modules/trainer/GenericTrainer.py"
OT_SAVER_MIXIN = ONETRAINER / "modules/modelSaver/mixin/LoRASaverMixin.py"
OT_CONFIG = ONETRAINER / "configs/sdxl_100step_baseline.json"
OT_METRICS = ONETRAINER / "output/sdxl_100step_baseline/metrics.json"
OT_BASELINE = ONETRAINER / "output/sdxl_100step_baseline/lora_last.safetensors"
OT_REPLAY_DUMP = ONETRAINER / "output/sdxl_100step_baseline/step000_replay.safetensors"
OT_REPLAY_MANIFEST = ONETRAINER / "output/sdxl_100step_baseline/step000_replay_manifest.json"
TRAIN_REF_DUMP = TRAIN_REF_PREFIX.with_suffix(".safetensors")
TRAIN_REF_ADAPTER_DUMP = TRAIN_REF_PREFIX.with_name(TRAIN_REF_PREFIX.name + "_adapters.safetensors")
TRAIN_REF_MOJO_GATE = REPO / "serenitymojo/models/sdxl/parity/sdxl_train_ref_artifact_smoke.mojo"
TRAIN_REF_ADAPTER_UPDATE_GATE = REPO / "scripts/check_sdxl_adapter_update_replay.py"
STACK = REPO / "serenitymojo/models/sdxl/sdxl_unet_stack_lora.mojo"
REAL_WEIGHTS = REPO / "serenitymojo/models/sdxl/real_weights.mojo"
LORA_SAVE = REPO / "serenitymojo/training/lora_save.mojo"

SAVE_SUFFIXES = ("alpha", "lora_down.weight", "lora_up.weight")
SAMPLE_OT_KEYS = (
    "lora_unet_add_embedding_linear_1",
    "lora_unet_conv_in",
    "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q",
    "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_ff_net_2",
)

IMPLEMENTED_ST_PREFIXES = (
    "lora_unet_down_blocks_1_attentions_0",
    "lora_unet_down_blocks_1_attentions_1",
    "lora_unet_down_blocks_2_attentions_0",
    "lora_unet_down_blocks_2_attentions_1",
    "lora_unet_mid_block_attentions_0",
    "lora_unet_up_blocks_0_attentions_0",
    "lora_unet_up_blocks_0_attentions_1",
    "lora_unet_up_blocks_0_attentions_2",
    "lora_unet_up_blocks_1_attentions_0",
    "lora_unet_up_blocks_1_attentions_1",
    "lora_unet_up_blocks_1_attentions_2",
)

IMPLEMENTED_ST_SUFFIXES = (
    "attn1_to_q",
    "attn1_to_k",
    "attn1_to_v",
    "attn1_to_out_0",
    "attn2_to_q",
    "attn2_to_k",
    "attn2_to_v",
    "attn2_to_out_0",
    "ff_net_0_proj",
    "ff_net_2",
)

EXPLICIT_UNSUPPORTED = (
    "lora_unet_conv_in",
    "lora_unet_time_embedding_linear_1",
    "lora_unet_time_embedding_linear_2",
    "lora_unet_add_embedding_linear_1",
    "lora_unet_add_embedding_linear_2",
    "lora_unet_*_resnets_*_time_emb_proj",
    "lora_unet_*_resnets_*_conv1",
    "lora_unet_*_resnets_*_conv2",
    "lora_unet_*_resnets_*_conv_shortcut",
    "lora_unet_*_samplers_*_conv",
    "lora_unet_*_attentions_*_proj_in",
    "lora_unet_*_attentions_*_proj_out",
    "lora_unet_conv_norm_out",
    "lora_unet_conv_out",
    "lora_te1",
    "lora_te2",
)

EXPECTED_REPLAY_TENSORS = (
    "batch.latent_image",
    "batch.tokens_1",
    "batch.tokens_2",
    "batch.text_encoder_1_hidden_state",
    "batch.text_encoder_2_hidden_state",
    "batch.text_encoder_2_pooled_state",
    "trace.scaled_latent_image",
    "trace.timestep",
    "trace.latent_noise",
    "trace.scaled_noisy_latent_image",
    "trace.latent_input",
    "trace.add_time_ids",
    "trace.text_encoder_output",
    "trace.pooled_text_encoder_2_output",
    "output.predicted",
    "output.target",
    "output.loss_for_backward",
)

EXPECTED_TRAIN_REF_TENSORS = (
    "batch.latent_image",
    "batch.loss_weight",
    "batch.text_encoder_1_hidden_state",
    "batch.text_encoder_2_hidden_state",
    "batch.text_encoder_2_pooled_state",
    "batch.tokens_1",
    "batch.tokens_2",
    "output.loss_for_backward",
    "output.loss_pre_scale",
    "output.predicted",
    "output.target",
    "output.timestep",
    "trace.added_cond_text_embeds",
    "trace.added_cond_time_ids",
    "trace.combined_pooled_text_encoder_2_output",
    "trace.combined_text_encoder_output",
    "trace.encode_text.pooled_text_encoder_2_output",
    "trace.encode_text.text_encoder_1_output",
    "trace.encode_text.text_encoder_2_output",
    "trace.encode_text.tokens_1",
    "trace.encode_text.tokens_2",
    "trace.encoder_hidden_states",
    "trace.latent_input",
    "trace.latent_noise",
    "trace.pooled_text_encoder_2_output",
    "trace.predicted_latent_noise",
    "trace.scaled_latent_image",
    "trace.scaled_noisy_latent_image",
    "trace.text_encoder_1_output",
    "trace.text_encoder_2_output",
    "trace.unet_timestep",
)

TRAIN_REF_SHAPES: dict[str, tuple[str, list[int]]] = {
    "batch.latent_image": ("BF16", [1, 4, 168, 96]),
    "batch.loss_weight": ("F32", [1]),
    "batch.text_encoder_1_hidden_state": ("F32", [1, 77, 768]),
    "batch.text_encoder_2_hidden_state": ("F32", [1, 77, 1280]),
    "batch.text_encoder_2_pooled_state": ("BF16", [1, 1280]),
    "batch.tokens_1": ("I64", [1, 77]),
    "batch.tokens_2": ("I64", [1, 77]),
    "output.loss_for_backward": ("F32", []),
    "output.loss_pre_scale": ("F32", []),
    "output.predicted": ("BF16", [1, 4, 168, 96]),
    "output.target": ("BF16", [1, 4, 168, 96]),
    "output.timestep": ("I32", [1]),
    "trace.added_cond_text_embeds": ("F32", [1, 1280]),
    "trace.added_cond_time_ids": ("BF16", [1, 6]),
    "trace.combined_pooled_text_encoder_2_output": ("F32", [1, 1280]),
    "trace.combined_text_encoder_output": ("F32", [1, 77, 2048]),
    "trace.encode_text.pooled_text_encoder_2_output": ("BF16", [1, 1280]),
    "trace.encode_text.text_encoder_1_output": ("F32", [1, 77, 768]),
    "trace.encode_text.text_encoder_2_output": ("F32", [1, 77, 1280]),
    "trace.encode_text.tokens_1": ("I64", [1, 77]),
    "trace.encode_text.tokens_2": ("I64", [1, 77]),
    "trace.encoder_hidden_states": ("BF16", [1, 77, 2048]),
    "trace.latent_input": ("BF16", [1, 4, 168, 96]),
    "trace.latent_noise": ("BF16", [1, 4, 168, 96]),
    "trace.pooled_text_encoder_2_output": ("F32", [1, 1280]),
    "trace.predicted_latent_noise": ("BF16", [1, 4, 168, 96]),
    "trace.scaled_latent_image": ("BF16", [1, 4, 168, 96]),
    "trace.scaled_noisy_latent_image": ("BF16", [1, 4, 168, 96]),
    "trace.text_encoder_1_output": ("F32", [1, 77, 768]),
    "trace.text_encoder_2_output": ("F32", [1, 77, 1280]),
    "trace.unet_timestep": ("I32", [1]),
}

TRAIN_REF_ADAPTER_SHAPES: dict[str, tuple[str, list[int]]] = {
    "adapter_before.lora_unet.conv_in.lora_down.weight": ("F32", [16, 4, 3, 3]),
    "adapter_before.lora_unet.conv_in.lora_up.weight": ("F32", [320, 16, 1, 1]),
    "adapter_pre.lora_unet.conv_in.lora_down.weight": ("F32", [16, 4, 3, 3]),
    "adapter_post.lora_unet.conv_in.lora_down.weight": ("F32", [16, 4, 3, 3]),
    "adapter_after.lora_unet.conv_in.lora_down.weight": ("F32", [16, 4, 3, 3]),
    "adapter_after.lora_unet.add_embedding.linear_1.lora_down.weight": ("F32", [16, 2816]),
    "adapter_after.lora_unet.add_embedding.linear_1.lora_up.weight": ("F32", [1280, 16]),
    "adapter_after.lora_unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_down.weight": ("F32", [16, 2048]),
    "adapter_after.lora_unet.up_blocks.1.attentions.2.transformer_blocks.0.ff.net.2.lora_up.weight": ("F32", [640, 16]),
}

NEXT_REPLAY_GATES = (
    (
        "predict/loss replay",
        "must consume the one-step dump tensors for latent, timestep, conditioning, predicted, target, and loss",
    ),
    (
        "LoRA backward replay",
        "must compare per-adapter SDXL LoRA dA/dB against OneTrainer on the same dumped step",
    ),
    (
        "AdamW update replay",
        "must compare adapter pre/post update tensors using the baseline optimizer fields",
    ),
    (
        "sampler replay",
        "must compare SDXL prompt conditioning, denoise trajectory, VAE decode, image output, seconds/step, and VRAM",
    ),
)


def read_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[sdxl-lora-keys] missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise SystemExit(f"[sdxl-lora-keys] missing {label}: {needle}")


def require_block(
    path: Path,
    label: str,
    start_marker: str,
    end_marker: str,
    needles: tuple[str, ...] | list[str],
) -> None:
    text = read_text(path)
    start = text.find(start_marker)
    if start < 0:
        raise SystemExit(f"[sdxl-lora-keys] missing {label} block start in {path}: {start_marker}")
    end = text.find(end_marker, start)
    if end < 0:
        raise SystemExit(f"[sdxl-lora-keys] missing {label} block end in {path}: {end_marker}")
    block = text[start:end]
    missing = [needle for needle in needles if needle not in block]
    if missing:
        raise SystemExit(
            f"[sdxl-lora-keys] missing {label} source anchors in {path}: "
            + ", ".join(missing)
        )
    print(f"[sdxl-lora-keys] OneTrainer {label}: PASS")


def read_safetensors_header(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"[sdxl-lora-keys] missing safetensors: {path}")
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise SystemExit(f"safetensors too small: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_raw = f.read(header_len)
    if len(header_raw) != header_len:
        raise SystemExit(f"[sdxl-lora-keys] truncated safetensors header: {path}")
    return json.loads(header_raw.decode("utf-8"))


def nested(data: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = data
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def require_finite_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SystemExit(f"[sdxl-lora-keys] {label} must be finite number, got {value!r}")
    return float(value)


def tensor_keys(header: dict[str, Any]) -> list[str]:
    return sorted(k for k in header if k != "__metadata__")


def check_baseline_config() -> dict[str, Any]:
    config = load_json(OT_CONFIG)
    expected: dict[str, Any] = {
        "model_type": "STABLE_DIFFUSION_XL_10_BASE",
        "training_method": "LORA",
        "batch_size": 1,
        "resolution": "1024",
        "seed": 42,
        "gradient_accumulation_steps": 1,
        "clip_grad_norm": 1.0,
        "gradient_checkpointing": "CPU_OFFLOADED",
        "learning_rate": 1e-4,
        "lora_rank": 16,
        "lora_alpha": 16,
        "train_dtype": "BFLOAT_16",
        "output_dtype": "BFLOAT_16",
        "output_model_format": "SAFETENSORS",
        "optimizer.optimizer": "ADAMW",
        "optimizer.beta1": 0.9,
        "optimizer.beta2": 0.999,
        "optimizer.eps": 1e-8,
        "optimizer.weight_decay": 0.01,
        "optimizer.fused": False,
        "optimizer.fused_back_pass": False,
        "unet.train": True,
        "unet.weight_dtype": "FLOAT_16",
        "text_encoder.train": False,
        "text_encoder.weight_dtype": "FLOAT_16",
        "text_encoder_2.train": False,
        "text_encoder_2.weight_dtype": "FLOAT_16",
        "vae.weight_dtype": "FLOAT_32",
    }
    mismatches: list[str] = []
    for dotted_key, expected_value in expected.items():
        actual = nested(config, dotted_key)
        if actual != expected_value:
            mismatches.append(f"{dotted_key}: got {actual!r}, expected {expected_value!r}")
    if mismatches:
        raise SystemExit(
            f"[sdxl-lora-keys] baseline config mismatch: {OT_CONFIG}\n  "
            + "\n  ".join(mismatches)
        )

    output_path = Path(str(config.get("output_model_destination", "")))
    if output_path != OT_BASELINE:
        raise SystemExit(
            f"[sdxl-lora-keys] output_model_destination {output_path} != {OT_BASELINE}"
        )
    base_model = Path(str(config.get("base_model_name", "")))
    if not base_model.is_absolute():
        raise SystemExit(f"[sdxl-lora-keys] base_model_name is not absolute: {base_model}")
    if not base_model.exists():
        raise SystemExit(f"[sdxl-lora-keys] local SDXL base checkpoint missing: {base_model}")
    if config.get("samples") not in ([], None):
        raise SystemExit("[sdxl-lora-keys] baseline config unexpectedly includes samples")
    if config.get("sample_after") != 9999 or config.get("sample_after_unit") != "EPOCH":
        raise SystemExit("[sdxl-lora-keys] baseline sampling cadence changed; sampler evidence must be re-audited")

    print(
        "[sdxl-lora-keys] OneTrainer baseline config: PASS "
        f"model={config['model_type']} method={config['training_method']} "
        f"batch={config['batch_size']} resolution={config['resolution']} "
        f"dtype={config['train_dtype']} optimizer={config['optimizer']['optimizer']} "
        f"rank={config['lora_rank']} alpha={config['lora_alpha']}"
    )
    return config


def check_baseline_metrics(config: dict[str, Any]) -> None:
    metrics = load_json(OT_METRICS)
    if metrics.get("config_path") != "configs/sdxl_100step_baseline.json":
        raise SystemExit(
            f"[sdxl-lora-keys] metrics config_path mismatch: {metrics.get('config_path')!r}"
        )
    if metrics.get("status") != "completed":
        raise SystemExit(f"[sdxl-lora-keys] SDXL baseline did not complete: {metrics.get('status')!r}")
    requested_steps = int(metrics.get("requested_steps", -1))
    global_steps = int(metrics.get("global_steps_seen", -1))
    if requested_steps != 100 or global_steps != 100:
        raise SystemExit(
            f"[sdxl-lora-keys] SDXL baseline step count mismatch: requested={requested_steps} seen={global_steps}"
        )
    if len(metrics.get("progress_events", [])) != 100:
        raise SystemExit("[sdxl-lora-keys] SDXL baseline missing 100 progress events")
    if int(metrics.get("loss_count", -1)) != 100 or int(metrics.get("grad_norm_count", -1)) != 100:
        raise SystemExit(
            "[sdxl-lora-keys] SDXL baseline metrics missing loss/grad_norm counts "
            f"loss={metrics.get('loss_count')} grad_norm={metrics.get('grad_norm_count')}"
        )
    if Path(str(metrics.get("output_model_destination", ""))) != OT_BASELINE:
        raise SystemExit("[sdxl-lora-keys] metrics output_model_destination does not match baseline LoRA")
    if str(config.get("workspace_dir")) != str(metrics.get("workspace_dir")):
        raise SystemExit("[sdxl-lora-keys] metrics workspace_dir does not match config")
    if str(config.get("cache_dir")) != str(metrics.get("cache_dir")):
        raise SystemExit("[sdxl-lora-keys] metrics cache_dir does not match config")

    mean_step = require_finite_number(metrics.get("mean_step_seconds_excluding_first"), "mean step seconds")
    train_wall = require_finite_number(metrics.get("train_wall_seconds"), "train wall seconds")
    max_alloc = require_finite_number(metrics.get("torch_cuda_max_allocated_mib"), "torch CUDA max allocated MiB")
    max_reserved = require_finite_number(metrics.get("torch_cuda_max_reserved_mib"), "torch CUDA max reserved MiB")
    last_loss = metrics.get("last_loss", {})
    last_grad = metrics.get("last_grad_norm", {})
    loss_value = require_finite_number(last_loss.get("value"), "last loss")
    grad_value = require_finite_number(last_grad.get("value"), "last grad norm")
    if mean_step <= 0.0 or train_wall <= 0.0 or max_alloc <= 0.0 or max_reserved <= 0.0:
        raise SystemExit("[sdxl-lora-keys] baseline timing/VRAM metrics must be positive")
    if last_loss.get("global_step") != 99 or last_grad.get("global_step") != 99:
        raise SystemExit("[sdxl-lora-keys] final scalar events should be recorded at global_step=99")

    print(
        "[sdxl-lora-keys] OneTrainer 100-step baseline metrics: PASS "
        f"steps={global_steps} mean_step={mean_step:.6f}s "
        f"loss99={loss_value:.9f} grad_norm99={grad_value:.9f} "
        f"cuda_alloc={int(max_alloc)}MiB reserved={int(max_reserved)}MiB"
    )


def check_onetrainer_sources() -> None:
    dataloader = read_text(OT_DATALOADER)
    for needle in (
        "SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')",
        "Tokenize(in_name='prompt_1', tokens_out_name='tokens_1'",
        "Tokenize(in_name='prompt_2', tokens_out_name='tokens_2'",
        "EncodeClipText(in_name='tokens_1'",
        "EncodeClipText(in_name='tokens_2'",
        "'text_encoder_1_hidden_state'",
        "'text_encoder_2_hidden_state', 'text_encoder_2_pooled_state'",
        "aspect_bucketing_quantization=64",
    ):
        require(dataloader, needle, "OT SDXL cache/data output contract")
    print("[sdxl-lora-keys] OneTrainer SDXL cache/data source contract: PASS")

    setup = read_text(OT_BASE_SETUP)
    for needle in (
        "batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()",
        "generator = torch.Generator(device=config.train_device)",
        "generator.manual_seed(batch_seed)",
        "vae_scaling_factor = model.vae.config['scaling_factor']",
        "scaled_latent_image = latent_image * vae_scaling_factor",
        "timestep = self._get_timestep_discrete(",
        "latent_noise = self._create_noise(",
        "scaled_noisy_latent_image = self._add_noise_discrete(",
        "latent_input = scaled_noisy_latent_image",
        'added_cond_kwargs = {"text_embeds": pooled_text_encoder_2_output, "time_ids": add_time_ids}',
        "predicted_latent_noise = model.unet(",
        "sample=latent_input.to(dtype=model.train_dtype.torch_dtype())",
        "encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype())",
        "'target': latent_noise",
        "target_velocity = model.noise_scheduler.get_velocity(scaled_latent_image, latent_noise, timestep)",
        "model_output_data['prediction_type'] = model.noise_scheduler.config.prediction_type",
        "return self._diffusion_losses(",
    ):
        require(setup, needle, "OT SDXL predict/loss contract")
    print("[sdxl-lora-keys] OneTrainer SDXL predict/loss source contract: PASS")

    sampler = read_text(OT_SAMPLER)
    for needle in (
        "class StableDiffusionXLSampler",
        "create.create_noise_scheduler(noise_scheduler, self.model.noise_scheduler, diffusion_steps)",
        "noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device)",
        "combined_prompt_embedding = torch.cat([negative_prompt_embedding, prompt_embedding])",
        "latent_image = torch.randn(",
        "* noise_scheduler.init_noise_sigma",
        "noise_scheduler.scale_model_input(latent_model_input, timestep)",
        "noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)",
        "cfg_rescale * noise_pred_rescaled + (1 - cfg_rescale) * noise_pred",
        "noise_scheduler.step(",
        "vae.decode(latent_image / vae.config.scaling_factor, return_dict=False)",
        "image_processor.postprocess(image, output_type='pil'",
    ):
        require(sampler, needle, "OT SDXL sampler contract")
    print("[sdxl-lora-keys] OneTrainer SDXL sampler source contract: PASS")

    saver = read_text(OT_SAVER_MIXIN)
    for needle in (
        "save_file(save_state_dict, destination",
        "convert_to_legacy_diffusers(save_state_dict, key_sets)",
        "convert_to_omi(save_state_dict, key_sets)",
    ):
        require(saver, needle, "OT SDXL LoRA saver contract")
    print("[sdxl-lora-keys] OneTrainer LoRA saver source contract: PASS")

    require_block(
        OT_TRAINER,
        "GenericTrainer train-step order",
        "step_seed = train_progress.global_step",
        "self.model.optimizer.step()",
        [
            "bf16_stochastic_rounding_set_seed(step_seed, train_device)",
            "model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)",
            "loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)",
            "loss = loss / self.config.gradient_accumulation_steps",
            "loss.backward()",
            "multi.reduce_grads_mean(self.parameters, self.config.gradient_reduce_precision)",
            "nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)",
        ],
    )


def validate_replay_dump(path: Path) -> None:
    header = read_safetensors_header(path)
    keys = set(tensor_keys(header))
    missing = sorted(set(EXPECTED_REPLAY_TENSORS) - keys)
    if missing:
        raise SystemExit(
            f"[sdxl-lora-keys] SDXL replay dump missing expected tensors in {path}: "
            + ", ".join(missing)
        )
    for key in EXPECTED_REPLAY_TENSORS:
        dtype = header[key].get("dtype")
        if key.startswith("batch.tokens"):
            if dtype not in {"I64", "I32"}:
                raise SystemExit(f"[sdxl-lora-keys] {key} dtype {dtype!r} is not integer")
        elif dtype not in {"BF16", "F16", "F32"}:
            raise SystemExit(f"[sdxl-lora-keys] {key} dtype {dtype!r} is not floating")
    print(f"[sdxl-lora-keys] OneTrainer replay dump header: PASS file={path}")


def validate_replay_manifest(path: Path) -> None:
    manifest = load_json(path)
    expected_config = str(OT_CONFIG)
    if str(manifest.get("onetrainer_config")) != expected_config:
        raise SystemExit(
            f"[sdxl-lora-keys] replay manifest onetrainer_config={manifest.get('onetrainer_config')!r}, "
            f"expected {expected_config!r}"
        )
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict):
        raise SystemExit(f"[sdxl-lora-keys] replay manifest tensors must be an object: {path}")
    missing = sorted(set(EXPECTED_REPLAY_TENSORS) - set(tensors))
    if missing:
        raise SystemExit(
            f"[sdxl-lora-keys] replay manifest missing tensor metadata: {', '.join(missing)}"
        )
    print(f"[sdxl-lora-keys] OneTrainer replay dump manifest: PASS file={path}")


def validate_train_ref_dump(path: Path) -> None:
    header = read_safetensors_header(path)
    keys = set(tensor_keys(header))
    missing = sorted(set(EXPECTED_TRAIN_REF_TENSORS) - keys)
    if missing:
        raise SystemExit(
            f"[sdxl-lora-keys] SDXL train-ref dump missing expected tensors in {path}: "
            + ", ".join(missing)
        )
    if len(keys) != 31:
        raise SystemExit(f"[sdxl-lora-keys] SDXL train-ref tensor count {len(keys)} != 31")
    meta = header.get("__metadata__", {})
    if meta.get("producer") != "scripts/sdxl_dump_train_ref.py" or meta.get("global_step") != "0":
        raise SystemExit(f"[sdxl-lora-keys] unexpected SDXL train-ref metadata: {meta!r}")
    for key, (expected_dtype, expected_shape) in TRAIN_REF_SHAPES.items():
        info = header[key]
        dtype = info.get("dtype")
        shape = info.get("shape")
        if dtype != expected_dtype or shape != expected_shape:
            raise SystemExit(
                f"[sdxl-lora-keys] {key} header mismatch: dtype={dtype!r} shape={shape!r}, "
                f"expected dtype={expected_dtype!r} shape={expected_shape!r}"
            )
    print(f"[sdxl-lora-keys] OneTrainer train-ref step artifact: PASS file={path}")


def validate_train_ref_adapter_dump(path: Path) -> None:
    header = read_safetensors_header(path)
    keys = set(tensor_keys(header))
    if len(keys) != 6352:
        raise SystemExit(f"[sdxl-lora-keys] SDXL adapter dump tensor count {len(keys)} != 6352")
    meta = header.get("__metadata__", {})
    if meta.get("producer") != "scripts/sdxl_dump_train_ref.py" or meta.get("adapter_dump") != "step":
        raise SystemExit(f"[sdxl-lora-keys] unexpected SDXL adapter metadata: {meta!r}")
    for key, (expected_dtype, expected_shape) in TRAIN_REF_ADAPTER_SHAPES.items():
        if key not in header:
            raise SystemExit(f"[sdxl-lora-keys] adapter dump missing key: {key}")
        info = header[key]
        dtype = info.get("dtype")
        shape = info.get("shape")
        if dtype != expected_dtype or shape != expected_shape:
            raise SystemExit(
                f"[sdxl-lora-keys] {key} header mismatch: dtype={dtype!r} shape={shape!r}, "
                f"expected dtype={expected_dtype!r} shape={expected_shape!r}"
            )
    print(f"[sdxl-lora-keys] OneTrainer train-ref adapter artifact: PASS file={path}")


def validate_train_ref_mojo_consumer(path: Path) -> None:
    text = read_text(path)
    require(text, str(TRAIN_REF_DUMP), "Mojo SDXL train-ref step dump path")
    require(text, str(TRAIN_REF_ADAPTER_DUMP), "Mojo SDXL train-ref adapter dump path")
    require(text, "output.loss_for_backward", "Mojo SDXL loss scalar consumer")
    require(text, "adapter_before.lora_unet.conv_in.lora_down.weight", "Mojo SDXL adapter payload consumer")
    require(text, "6352", "Mojo SDXL adapter count guard")
    print(f"[sdxl-lora-keys] Mojo train-ref artifact consumer: PASS file={path}")


def validate_train_ref_adapter_update_gate(path: Path) -> None:
    text = read_text(path)
    require(text, "check_adapter_update_replay", "SDXL adapter update shared checker import")
    require(text, '"sdxl"', "SDXL adapter update wrapper model argument")
    print(f"[sdxl-lora-keys] SDXL adapter update oracle gate: PASS file={path}")


def report_replay_plan(dump_path: Path, manifest_path: Path, require_artifact: bool) -> None:
    setup = read_text(OT_BASE_SETUP)
    has_dump_hook = "OT_DUMP_STEP1_INPUTS" in setup and "safetensors.torch" in setup
    print("[sdxl-lora-keys] SDXL ONE-STEP REPLAY PLAN")
    print(f"  product_hook_safetensors: {dump_path}")
    print(f"  product_hook_manifest_json: {manifest_path}")
    print(f"  train_ref_safetensors: {TRAIN_REF_DUMP}")
    print(f"  train_ref_adapter_safetensors: {TRAIN_REF_ADAPTER_DUMP}")
    print(f"  mojo_artifact_gate: {TRAIN_REF_MOJO_GATE}")
    print(f"  source_anchor: {OT_BASE_SETUP}::predict")
    print(f"  trainer_anchor: {OT_TRAINER}::train")
    print(f"  source_dump_hook_present: {str(has_dump_hook).lower()}")
    print("  expected_tensor_keys:")
    for key in EXPECTED_REPLAY_TENSORS:
        print(f"    - {key}")

    missing: list[Path] = []
    if dump_path.exists():
        validate_replay_dump(dump_path)
    else:
        print(f"[sdxl-lora-keys] MISSING replay dump artifact: {dump_path}")
        missing.append(dump_path)
    if manifest_path.exists():
        validate_replay_manifest(manifest_path)
    else:
        print(f"[sdxl-lora-keys] MISSING replay manifest artifact: {manifest_path}")
        missing.append(manifest_path)

    train_ref_missing: list[Path] = []
    if TRAIN_REF_DUMP.exists():
        validate_train_ref_dump(TRAIN_REF_DUMP)
    else:
        print(f"[sdxl-lora-keys] MISSING train-ref step artifact: {TRAIN_REF_DUMP}")
        train_ref_missing.append(TRAIN_REF_DUMP)
    if TRAIN_REF_ADAPTER_DUMP.exists():
        validate_train_ref_adapter_dump(TRAIN_REF_ADAPTER_DUMP)
    else:
        print(f"[sdxl-lora-keys] MISSING train-ref adapter artifact: {TRAIN_REF_ADAPTER_DUMP}")
        train_ref_missing.append(TRAIN_REF_ADAPTER_DUMP)
    if TRAIN_REF_MOJO_GATE.exists():
        validate_train_ref_mojo_consumer(TRAIN_REF_MOJO_GATE)
    else:
        print(f"[sdxl-lora-keys] MISSING Mojo train-ref artifact consumer: {TRAIN_REF_MOJO_GATE}")
        train_ref_missing.append(TRAIN_REF_MOJO_GATE)
    if TRAIN_REF_ADAPTER_UPDATE_GATE.exists():
        validate_train_ref_adapter_update_gate(TRAIN_REF_ADAPTER_UPDATE_GATE)
    else:
        print(f"[sdxl-lora-keys] MISSING adapter update oracle gate: {TRAIN_REF_ADAPTER_UPDATE_GATE}")
        train_ref_missing.append(TRAIN_REF_ADAPTER_UPDATE_GATE)

    if not has_dump_hook:
        print("[sdxl-lora-keys] WARN OneTrainer SDXL predict has no OT_DUMP_STEP1_INPUTS hook")
    print("[sdxl-lora-keys] NEXT MOJO REPLAY/SAMPLER GATES")
    for name, note in NEXT_REPLAY_GATES:
        print(f"  {name}: {note}")

    if require_artifact and train_ref_missing:
        parts = [str(path) for path in train_ref_missing]
        raise SystemExit("[sdxl-lora-keys] required SDXL train-ref artifact(s) missing: " + ", ".join(parts))


def check_no_false_replay_claims() -> None:
    claim_re = re.compile(
        r"SDXL[^\n]{0,80}(?:one-step|replay|train parity|sampler parity)[^\n]{0,80}PASS",
        re.I,
    )
    offenders: list[Path] = []
    for path in (REPO / "serenitymojo/models/sdxl/parity").glob("*smoke.mojo"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if (
            claim_re.search(text)
            and "sdxl_100step_baseline" not in text
            and "step000_replay" not in text
            and "sdxl_train_ref_step000" not in text
        ):
            offenders.append(path)
    if offenders:
        for path in offenders:
            print(f"[sdxl-lora-keys] WARN possible source-only SDXL replay claim: {path.relative_to(REPO)}")
    else:
        print("[sdxl-lora-keys] PASS no source-only SDXL one-step/sampler replay PASS claim found")


def suffix_for(key: str) -> str:
    for suffix in SAVE_SUFFIXES:
        if key.endswith(f".{suffix}"):
            return suffix
    return "<other>"


def check_ot_source() -> None:
    setup = read_text(OT_SETUP)
    require(setup, "LoRAModuleWrapper", "OT LoRA wrapper")
    require(setup, 'model.text_encoder_1, "lora_te1"', "OT text encoder 1 prefix")
    require(setup, 'model.text_encoder_2, "lora_te2"', "OT text encoder 2 prefix")
    require(setup, 'model.unet, "lora_unet"', "OT UNet prefix")
    require(setup, "config.layer_filter.split", "OT layer filter")
    print("[sdxl-lora-keys] OneTrainer source contract: PASS")


def check_saved_header(path: Path, expected_adapters: int | None, expected_rank: int | None) -> None:
    header = read_safetensors_header(path)
    keys = sorted(k for k in header if k != "__metadata__")
    suffix_counts = Counter(suffix_for(k) for k in keys)
    dtype_counts = Counter(header[k].get("dtype") for k in keys)

    if expected_adapters is not None:
        expected_tensors = expected_adapters * len(SAVE_SUFFIXES)
        if len(keys) != expected_tensors:
            raise SystemExit(
                f"[sdxl-lora-keys] tensor count mismatch: got {len(keys)} expected {expected_tensors}"
            )
    if dtype_counts != {"BF16": len(keys)}:
        raise SystemExit(f"[sdxl-lora-keys] dtype mismatch: {dict(dtype_counts)}")

    for prefix in SAMPLE_OT_KEYS:
        for suffix in SAVE_SUFFIXES:
            key = f"{prefix}.{suffix}"
            if key not in header:
                raise SystemExit(f"[sdxl-lora-keys] missing saved key: {key}")
            info = header[key]
            shape = info.get("shape", [])
            if suffix == "alpha":
                if shape != []:
                    raise SystemExit(f"[sdxl-lora-keys] {key} alpha shape {shape} != []")
            else:
                if len(shape) not in (2, 4):
                    raise SystemExit(f"[sdxl-lora-keys] {key} rank {len(shape)} not linear/conv")
                if expected_rank is not None:
                    rank_axis = 0 if suffix == "lora_down.weight" else 1
                    if shape[rank_axis] != expected_rank:
                        raise SystemExit(
                            f"[sdxl-lora-keys] {key} rank axis {shape[rank_axis]} != {expected_rank}"
                        )

    print(
        "[sdxl-lora-keys] OneTrainer saved inventory: PASS "
        f"tensors={len(keys)} adapters={len(keys) // 3} suffixes={dict(suffix_counts)}"
    )


def check_mojo_sources(strict_port: bool) -> None:
    stack = read_text(STACK)
    real_weights = read_text(REAL_WEIGHTS)
    lora_save = read_text(LORA_SAVE)

    require(stack, "save_sdxl_lora", "Mojo SDXL save hook")
    require(stack, "SDXL_SLOTS", "Mojo SDXL slot count")
    require(lora_save, "def save_lora_onetrainer", "shared OneTrainer saver")
    require(lora_save, ".lora_down.weight", "shared OT down suffix")
    require(lora_save, ".lora_up.weight", "shared OT up suffix")
    require(lora_save, ".alpha", "shared OT alpha suffix")
    require(stack, "def sdxl_lora_supported_unet_prefixes", "explicit supported OT UNet surface")
    require(stack, "def sdxl_lora_unsupported_onetrainer_targets", "explicit unsupported OT surface")
    require(stack, "def sdxl_lora_requires_text_encoder_surface", "TE fail-loud guard")
    require(stack, "def save_sdxl_lora_with_text_encoder_flags", "TE-aware save wrapper")

    blockers: list[str] = []
    if "save_lora_peft(named, path, ctx)" in stack:
        blockers.append("Mojo SDXL save path still writes PEFT lora_A/lora_B keys")
    if "save_lora_onetrainer" not in stack:
        blockers.append("Mojo SDXL save hook does not call save_lora_onetrainer")

    for prefix in IMPLEMENTED_ST_PREFIXES:
        require(stack, f'return "{prefix}"', f"OT ST prefix mapping {prefix}")
    for suffix in IMPLEMENTED_ST_SUFFIXES:
        require(stack, f'return "{suffix}"', f"OT ST suffix mapping {suffix}")
    for target in EXPLICIT_UNSUPPORTED:
        require(stack, f'out.append("{target}")', f"explicit unsupported target {target}")

    sample_supported = (
        "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q",
        "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_out_0",
        "lora_unet_up_blocks_1_attentions_2_transformer_blocks_0_ff_net_2",
    )
    for key in sample_supported:
        prefix, suffix = key.rsplit("_transformer_blocks_0_", 1)
        if prefix not in IMPLEMENTED_ST_PREFIXES or suffix not in IMPLEMENTED_ST_SUFFIXES:
            blockers.append(f"checker sample key not covered by explicit ST contract: {key}")

    if "lora_te1/lora_te2" not in stack:
        blockers.append("TE-enabled SDXL LoRA save is not fail-loud about missing lora_te1/lora_te2")

    if blockers:
        for blocker in blockers:
            print(f"[sdxl-lora-keys] WARN: {blocker}")
        if strict_port:
            raise SystemExit("[sdxl-lora-keys] strict port requested and SDXL save-key blockers remain")
    else:
        print(
            "[sdxl-lora-keys] Mojo source scaffold: PASS "
            "implemented=SpatialTransformer linears unsupported=explicit-fail-loud "
            f"local_weight_prefixes_present={('middle_block.1' in real_weights or 'output_blocks.0.1' in real_weights)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-port", action="store_true")
    parser.add_argument("--safetensors", type=Path, default=OT_BASELINE if OT_BASELINE.exists() else None)
    parser.add_argument("--expected-adapters", type=int, default=794)
    parser.add_argument("--expected-rank", type=int, default=16)
    parser.add_argument(
        "--replay-dump",
        type=Path,
        default=OT_REPLAY_DUMP,
        help="expected local OneTrainer SDXL one-step replay safetensors artifact",
    )
    parser.add_argument(
        "--replay-manifest",
        type=Path,
        default=OT_REPLAY_MANIFEST,
        help="expected local OneTrainer SDXL one-step replay manifest",
    )
    parser.add_argument(
        "--require-replay-dump",
        action="store_true",
        help="fail if the local SDXL train-ref dump/adapters or in-repo Mojo artifact consumer is absent",
    )
    args = parser.parse_args()

    config = check_baseline_config()
    check_baseline_metrics(config)
    check_onetrainer_sources()
    check_ot_source()
    if args.safetensors is not None:
        check_saved_header(args.safetensors, args.expected_adapters, args.expected_rank)
    check_mojo_sources(args.strict_port)
    report_replay_plan(args.replay_dump, args.replay_manifest, args.require_replay_dump)
    check_no_false_replay_claims()
    print("[sdxl-lora-keys] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
