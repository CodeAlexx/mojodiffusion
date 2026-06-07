#!/usr/bin/env python3
"""Static Flux.1-dev one-step dump contract from local OneTrainer source.

This script intentionally does not import OneTrainer, torch, diffusers, or run
training. It reads the local OneTrainer source and baseline config, proves the
one-step dump anchor points still exist, and reports the exact missing artifact
needed before Mojo Flux.1 one-step numeric parity can be claimed.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


ONETRAINER = Path("/home/alex/OneTrainer")
OT_CONFIG = ONETRAINER / "configs/flux1_100step_baseline.json"
OT_BASELINE_LORA = ONETRAINER / "output/flux1_100step_baseline/lora_last.safetensors"

OT_MODEL = ONETRAINER / "modules/model/FluxModel.py"
OT_SETUP = ONETRAINER / "modules/modelSetup/BaseFluxSetup.py"
OT_LORA_SETUP = ONETRAINER / "modules/modelSetup/FluxLoRASetup.py"
OT_DATALOADER = ONETRAINER / "modules/dataLoader/FluxBaseDataLoader.py"
OT_TRAINER = ONETRAINER / "modules/trainer/GenericTrainer.py"
OT_NOISE = ONETRAINER / "modules/modelSetup/mixin/ModelSetupNoiseMixin.py"
OT_FLOW = ONETRAINER / "modules/modelSetup/mixin/ModelSetupFlowMatchingMixin.py"
OT_ZIMAGE_SETUP = ONETRAINER / "modules/modelSetup/BaseZImageSetup.py"

DEFAULT_DUMP_PATH = Path("/tmp/ot_flux1_step1_inputs.safetensors")
DEFAULT_MANIFEST_PATH = Path("/tmp/ot_flux1_step1_inputs_manifest.json")
PARITY_DIR = Path("/home/alex/onetrainer-mojo/parity")
DEFAULT_PARITY_STEP_PATH = PARITY_DIR / "flux1_train_ref_step000.safetensors"
DEFAULT_PARITY_ADAPTERS_PATH = PARITY_DIR / "flux1_train_ref_step000_adapters.safetensors"
DEFAULT_PARITY_META_PATH = PARITY_DIR / "flux1_train_ref_meta.json"
DEFAULT_PARITY_CONTRACT_PATH = PARITY_DIR / "flux1_train_ref_contract.json"
DEFAULT_PARITY_BLOCKERS_PATH = PARITY_DIR / "flux1_train_ref_blockers.json"

EXPECTED_TENSORS = (
    "scaled_latent_image",
    "latent_noise",
    "scaled_noisy_latent_image",
    "latent_input",
    "packed_latent_input",
    "predicted_flow",
    "flow_target",
    "timestep_index",
    "timestep_model",
    "sigma",
    "text_encoder_output",
    "pooled_text_encoder_output",
    "text_ids",
    "image_ids",
    "guidance",
    "loss",
)

EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "scaled_latent_image": (1, 16, 64, 64),
    "latent_noise": (1, 16, 64, 64),
    "scaled_noisy_latent_image": (1, 16, 64, 64),
    "latent_input": (1, 16, 64, 64),
    "packed_latent_input": (1, 1024, 64),
    "predicted_flow": (1, 16, 64, 64),
    "flow_target": (1, 16, 64, 64),
    "timestep_index": (1,),
    "timestep_model": (1,),
    "sigma": (1,),
    "text_encoder_output": (1, 512, 4096),
    "pooled_text_encoder_output": (1, 768),
    "text_ids": (512, 3),
    "image_ids": (1024, 3),
    "guidance": (1,),
    "loss": (),
}

EXPECTED_MANIFEST_FIELDS = (
    "producer",
    "onetrainer_config",
    "base_model_name",
    "step",
    "seed",
    "train_dtype",
    "tensors",
)

EXPECTED_RUNTIME_CONFIG = {
    "model_type": "FLUX_DEV_1",
    "training_method": "LORA",
    "train_device": "cuda",
    "train_dtype": "BFLOAT_16",
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 0.0001,
    "learning_rate_scheduler": "CONSTANT",
    "optimizer": "ADAMW",
    "timestep_distribution": "LOGIT_NORMAL",
}

REQUIRED_STEP_META_FIELDS = (
    "step_index",
    "global_step",
    "batch_seed",
    "loss_pre_scale",
    "loss_for_backward",
    "grad_norm_pre_clip",
    "grad_norm_no_clip",
    "lr_before",
    "lr_after",
    "optimizer_before",
    "optimizer_after",
    "safetensors",
    "adapter_safetensors",
)

REQUIRED_TOP_META_FIELDS = (
    "producer",
    "onetrainer",
    "config_path",
    "prefix",
    "elapsed_seconds",
    "cuda",
    "runtime_config",
    "trainable_parameters",
    "steps",
)

REQUIRED_CONTRACT_FIELDS = (
    "name",
    "purpose",
    "producer_script",
    "reference_repo",
    "default_config",
    "default_outputs",
    "numeric_parity_policy",
)

REQUIRED_CONTRACT_OUTPUT_FIELDS = (
    "run_metadata",
    "step_tensors",
    "adapter_tensors",
)

REQUIRED_BLOCKERS_FIELDS = (
    "producer",
    "one_step_dump_produced",
    "numeric_parity_status",
    "structured_blockers",
    "blocked",
)

REQUIRED_READINESS_ARTIFACTS = (
    (
        "raw CUDA forward/loss dump",
        DEFAULT_DUMP_PATH,
        "BaseFluxSetup.py::predict",
        "OneTrainer tensors through scaled/noisy/packed latent, transformer predicted_flow, flow_target, and loss.",
    ),
    (
        "raw dump manifest",
        DEFAULT_MANIFEST_PATH,
        "BaseFluxSetup.py::predict",
        "Manifest with producer, OneTrainer config, base model snapshot, seed, train dtype, and per-tensor metadata.",
    ),
    (
        "normalized train step dump",
        DEFAULT_PARITY_STEP_PATH,
        "BaseFluxSetup.py::predict + BaseFluxSetup.py::calculate_loss",
        "Repo-local safetensors copy consumed by Mojo replay, including transformer input/output, target, timestep/sigma, and scalar loss.",
    ),
    (
        "normalized adapter/update dump",
        DEFAULT_PARITY_ADAPTERS_PATH,
        "GenericTrainer.py::train",
        "Adapter before/pre/post/after phases plus optimizer state evidence for LoRA backward and AdamW update replay.",
    ),
    (
        "normalized metadata",
        DEFAULT_PARITY_META_PATH,
        "GenericTrainer.py::train",
        "Runtime identity, loss, grad norms, LR before/after, AdamW state, CUDA device, elapsed time, and VRAM evidence.",
    ),
    (
        "normalized contract",
        DEFAULT_PARITY_CONTRACT_PATH,
        "dump producer",
        "Machine-readable statement that the dump is a real Flux.1-dev CUDA one-step train oracle.",
    ),
    (
        "normalized blockers",
        DEFAULT_PARITY_BLOCKERS_PATH,
        "dump producer",
        "Explicit blocker list when any required transformer/backward/AdamW/speed evidence is absent.",
    ),
)

REQUIRED_EVIDENCE = (
    (
        "transformer_forward_loss",
        "BaseFluxSetup.py::predict and BaseFluxSetup.py::calculate_loss",
        "scaled_latent_image, latent_noise, scaled_noisy_latent_image, packed_latent_input, text ids, image ids, guidance, predicted_flow, flow_target, loss",
    ),
    (
        "backward_gradients",
        "GenericTrainer.py::train after loss.backward() and grad reduction/clip",
        "per-trainable LoRA gradient payloads or adapter_post phase plus grad_norm_pre_clip and grad_norm_no_clip",
    ),
    (
        "adamw_update",
        "GenericTrainer.py::train around optimizer.step()",
        "adapter_before/pre/post/after phases, optimizer_before/after state, exp_avg, exp_avg_sq, step, lr_before, lr_after, nonzero update stats when lr_before > 0",
    ),
    (
        "speed_vram",
        "same CUDA run that produced the tensor dump",
        "seconds_per_step or elapsed_seconds, CUDA device name, allocated/reserved MiB, batch, resolution, dtype, cache state, optimizer, and checkpoint/offload policy",
    ),
)


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"[flux-onestep-dump] missing file: {path}")
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[flux-onestep-dump] missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_safetensors_header(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[flux-onestep-dump] missing safetensors: {path}")
    raw_len = path.read_bytes()[:8]
    if len(raw_len) != 8:
        raise SystemExit(f"[flux-onestep-dump] safetensors too small: {path}")
    header_len = struct.unpack("<Q", raw_len)[0]
    with path.open("rb") as fh:
        fh.seek(8)
        header_raw = fh.read(header_len)
    if len(header_raw) != header_len:
        raise SystemExit(f"[flux-onestep-dump] safetensors truncated header: {path}")
    return json.loads(header_raw.decode("utf-8"))


def resolve_port_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path("/home/alex/onetrainer-mojo") / path


def print_missing_requirements(label: str) -> None:
    if label == "raw CUDA forward/loss dump":
        print("      required_tensor_keys:")
        for key in EXPECTED_TENSORS:
            print(f"        - {key}: shape={list(EXPECTED_SHAPES[key])}")
    elif label == "raw dump manifest":
        print("      required_json_fields:")
        for field in EXPECTED_MANIFEST_FIELDS:
            print(f"        - {field}")
        print("      required_tensor_metadata: every expected tensor key with shape and dtype")
    elif label == "normalized train step dump":
        print("      required_tensor_keys:")
        for key in EXPECTED_TENSORS:
            print(f"        - {key}: shape={list(EXPECTED_SHAPES[key])}")
    elif label == "normalized adapter/update dump":
        print("      required_phases: adapter_before, adapter_pre, adapter_post, adapter_after")
        print("      required_invariant: every phase has the same trainable LoRA key set")
    elif label == "normalized metadata":
        print("      required_top_fields:")
        for field in REQUIRED_TOP_META_FIELDS:
            print(f"        - {field}")
        print("      required_step_fields:")
        for field in REQUIRED_STEP_META_FIELDS:
            print(f"        - {field}")
        print("      required_runtime_config:")
        for key, value in EXPECTED_RUNTIME_CONFIG.items():
            print(f"        - {key}={value!r}")
        print("      required_speed_vram_fields: seconds, CUDA device name, allocated MiB, reserved MiB")
    elif label == "normalized contract":
        print("      required_json_fields:")
        for field in REQUIRED_CONTRACT_FIELDS:
            print(f"        - {field}")
        print("      required_default_outputs:")
        for field in REQUIRED_CONTRACT_OUTPUT_FIELDS:
            print(f"        - {field}")
    elif label == "normalized blockers":
        print("      required_json_fields:")
        for field in REQUIRED_BLOCKERS_FIELDS:
            print(f"        - {field}")
        print("      required_invariant: numeric parity is not claimed by a blocker artifact")


def _system_exit_message(exc: SystemExit) -> str:
    if exc.code is None:
        return "validator exited"
    if isinstance(exc.code, int):
        return f"validator exited with code {exc.code}"
    return str(exc.code)


def run_artifact_validator(
    label: str,
    path: Path,
    validator,
) -> tuple[list[str], list[str]]:
    """Run a single artifact validator and keep readiness reporting aggregated."""
    try:
        result = validator(path)
    except SystemExit as exc:
        return [f"invalid {label}: {path}: {_system_exit_message(exc)}"], [
            f"invalid {label}: {path}"
        ]
    if isinstance(result, list):
        return result, []
    return [], []


def require(path: Path, label: str, needles: tuple[str, ...] | list[str]) -> None:
    text = read(path)
    missing = [needle for needle in needles if needle not in text]
    if missing:
        print(f"[flux-onestep-dump] FAIL {label}: {path}")
        for needle in missing:
            print(f"  missing: {needle}")
        raise SystemExit(1)
    print(f"[flux-onestep-dump] PASS {label}")


def require_block(
    path: Path,
    label: str,
    start_marker: str,
    end_marker: str,
    needles: tuple[str, ...] | list[str],
) -> None:
    text = read(path)
    start = text.find(start_marker)
    if start < 0:
        print(f"[flux-onestep-dump] FAIL {label}: {path}")
        print(f"  missing block start: {start_marker}")
        raise SystemExit(1)
    end = text.find(end_marker, start)
    if end < 0:
        print(f"[flux-onestep-dump] FAIL {label}: {path}")
        print(f"  missing block end: {end_marker}")
        raise SystemExit(1)
    block = text[start:end]
    missing = [needle for needle in needles if needle not in block]
    if missing:
        print(f"[flux-onestep-dump] FAIL {label}: {path}")
        for needle in missing:
            print(f"  missing in block: {needle}")
        raise SystemExit(1)
    print(f"[flux-onestep-dump] PASS {label}")


def require_config(config: dict[str, Any]) -> Path:
    print("[flux-onestep-dump] BASELINE IDENTITY FIELDS")
    print(f"  config_json: {OT_CONFIG}")
    expected = {
        "model_type": "FLUX_DEV_1",
        "training_method": "LORA",
        "batch_size": 1,
        "resolution": "512",
        "seed": 42,
        "train_dtype": "BFLOAT_16",
        "output_dtype": "BFLOAT_16",
        "learning_rate": 0.0001,
        "learning_rate_scheduler": "CONSTANT",
        "gradient_accumulation_steps": 1,
        "timestep_distribution": "LOGIT_NORMAL",
        "gradient_checkpointing": "CPU_OFFLOADED",
        "optimizer.optimizer": "ADAMW",
        "optimizer.beta1": 0.9,
        "optimizer.beta2": 0.999,
        "optimizer.eps": 1e-8,
        "optimizer.weight_decay": 0.01,
        "transformer.train": True,
        "transformer.weight_dtype": "BFLOAT_16",
        "text_encoder.train": False,
        "text_encoder.weight_dtype": "BFLOAT_16",
        "text_encoder_2.train": False,
        "text_encoder_2.weight_dtype": "BFLOAT_16",
        "vae.weight_dtype": "FLOAT_32",
    }
    mismatches: list[str] = []
    for dotted_key, expected_value in expected.items():
        print(f"  expected: {dotted_key}={expected_value!r}")
        actual: Any = config
        for key in dotted_key.split("."):
            if not isinstance(actual, dict) or key not in actual:
                mismatches.append(f"{dotted_key}: missing")
                break
            actual = actual[key]
        else:
            if actual != expected_value:
                mismatches.append(
                    f"{dotted_key}: got {actual!r}, expected {expected_value!r}"
                )
    if mismatches:
        print("[flux-onestep-dump] FAIL baseline identity invalid fields")
        for mismatch in mismatches:
            print(f"  invalid_field: {mismatch}")
        raise SystemExit(1)
    print("[flux-onestep-dump] PASS baseline identity fields invalid_fields=0")

    base_model = Path(str(config.get("base_model_name", "")))
    if not base_model.is_absolute():
        print("[flux-onestep-dump] FAIL baseline identity invalid fields")
        print(
            "  invalid_field: base_model_name: got "
            f"{config.get('base_model_name')!r}, expected local absolute path"
        )
        raise SystemExit(
            "[flux-onestep-dump] baseline base_model_name is not local absolute: "
            f"{config.get('base_model_name')!r}"
        )
    scheduler = base_model / "scheduler/scheduler_config.json"
    print(
        "[flux-onestep-dump] PASS OneTrainer Flux.1 baseline config "
        f"model={config['model_type']} method={config['training_method']} "
        f"batch={config['batch_size']} resolution={config['resolution']} "
        f"dtype={config['train_dtype']} optimizer={config['optimizer']['optimizer']} "
        f"lr={config['learning_rate']} lr_scheduler={config['learning_rate_scheduler']} "
        f"grad_accum={config['gradient_accumulation_steps']} "
        f"checkpoint={config['gradient_checkpointing']}"
    )
    print(f"[flux-onestep-dump] EXPECT base_model_name={base_model}")
    return scheduler


def require_scheduler(scheduler_path: Path) -> None:
    data = load_json(scheduler_path)
    expected = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "max_image_seq_len": 4096,
        "max_shift": 1.15,
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "use_dynamic_shifting": True,
    }
    mismatches = [
        f"{key}: got {data.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if data.get(key) != value
    ]
    if mismatches:
        print(f"[flux-onestep-dump] FAIL scheduler config: {scheduler_path}")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        raise SystemExit(1)
    print(f"[flux-onestep-dump] PASS scheduler config: {scheduler_path}")


def check_onetrainer_sources() -> None:
    require(
        OT_MODEL,
        "OneTrainer Flux.1 model helpers",
        [
            "FlowMatchEulerDiscreteScheduler",
            "FluxPipeline",
            "FluxTransformer2DModel",
            "def prepare_latent_image_ids(",
            "latent_image_ids[..., 1]",
            "latent_image_ids[..., 2]",
            "def pack_latents(self, latents: Tensor) -> Tensor:",
            "height // 2, 2, width // 2, 2",
            "channels * 4",
            "def unpack_latents(self, latents, height: int, width: int):",
            "def calculate_timestep_shift(self, latent_height: int, latent_width: int):",
            "base_seq_len = self.noise_scheduler.config.base_image_seq_len",
            "max_seq_len = self.noise_scheduler.config.max_image_seq_len",
            "return math.exp(mu)",
        ],
    )
    require(
        OT_DATALOADER,
        "OneTrainer Flux.1 data/cache outputs",
        [
            "RescaleImageChannels(image_in_name='image'",
            "EncodeVAE(in_name='image', out_name='latent_image_distribution'",
            "SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')",
            "Tokenize(in_name='prompt_1', tokens_out_name='tokens_1'",
            "Tokenize(in_name='prompt_2', tokens_out_name='tokens_2'",
            "EncodeClipText(in_name='tokens_1'",
            "EncodeT5Text(tokens_in_name='tokens_2'",
            "'text_encoder_1_pooled_state'",
            "'text_encoder_2_hidden_state'",
            "aspect_bucketing_quantization=64",
        ],
    )
    require(
        OT_LORA_SETUP,
        "OneTrainer Flux.1 LoRA setup",
        [
            "LoRAModuleWrapper(",
            '"lora_te1"',
            '"lora_te2"',
            '"lora_transformer"',
            "config.layer_filter.split(\",\")",
            "factory.register(BaseModelSetup, FluxLoRASetup, ModelType.FLUX_DEV_1, TrainingMethod.LORA)",
        ],
    )
    require(
        OT_SETUP,
        "OneTrainer Flux.1 predict one-step tensors",
        [
            "batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()",
            "generator = torch.Generator(device=config.train_device)",
            "generator.manual_seed(batch_seed)",
            "vae_scaling_factor = model.vae.config['scaling_factor']",
            "vae_shift_factor = model.vae.config['shift_factor']",
            "scaled_latent_image = (latent_image - vae_shift_factor) * vae_scaling_factor",
            "latent_noise = self._create_noise(scaled_latent_image, config, generator)",
            "shift = model.calculate_timestep_shift(scaled_latent_image.shape[-2], scaled_latent_image.shape[-1])",
            "timestep = self._get_timestep_discrete(",
            "scaled_noisy_latent_image, sigma = self._add_noise_discrete(",
            "latent_input = scaled_noisy_latent_image",
            "guidance = torch.tensor([config.transformer.guidance_scale], device=self.train_device)",
            "text_ids = torch.zeros(",
            "image_ids = model.prepare_latent_image_ids(",
            "packed_latent_input = model.pack_latents(latent_input)",
            "hidden_states=packed_latent_input.to(dtype=model.train_dtype.torch_dtype())",
            "timestep=timestep / 1000",
            "pooled_projections=pooled_text_encoder_output.to(dtype=model.train_dtype.torch_dtype())",
            "encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype())",
            "txt_ids=text_ids",
            "img_ids=image_ids",
            "predicted_flow = model.unpack_latents(",
            "flow = latent_noise - scaled_latent_image",
            "'loss_type': 'target'",
            "'predicted': predicted_flow",
            "'target': flow",
        ],
    )
    require(
        OT_SETUP,
        "OneTrainer Flux.1 loss",
        [
            "def calculate_loss(",
            "return self._flow_matching_losses(",
            "sigmas=model.noise_scheduler.sigmas",
        ],
    )
    require(
        OT_NOISE,
        "OneTrainer discrete timestep/noise contract",
        [
            "torch.randn(",
            "dtype=source_tensor.dtype",
            "def _get_timestep_discrete(",
            "TimestepDistribution.LOGIT_NORMAL",
            "torch.normal(bias, scale, size=(batch_size,), generator=generator",
            "timestep = num_train_timesteps * shift * timestep / ((shift - 1) * timestep + num_train_timesteps)",
            "return timestep.int()",
        ],
    )
    require(
        OT_FLOW,
        "OneTrainer flow-noise contract",
        [
            "def _add_noise_discrete(",
            "all_timesteps = torch.arange(start=1, end=num_timesteps + 1",
            "self.__sigma = all_timesteps / num_timesteps",
            "scaled_noisy_latent_image = latent_noise.to(dtype=sigmas.dtype) * sigmas",
            "return scaled_noisy_latent_image.to(dtype=orig_dtype), sigmas",
        ],
    )
    require_block(
        OT_TRAINER,
        "OneTrainer GenericTrainer first train-step order",
        "step_seed = train_progress.global_step",
        "self.model.optimizer.step()",
        [
            "bf16_stochastic_rounding_set_seed(step_seed, train_device)",
            "model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)",
            "loss = self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)",
            "loss = loss / self.config.gradient_accumulation_steps",
            "loss.backward()",
            "multi.reduce_grads_mean(self.parameters, self.config.gradient_reduce_precision)",
        ],
    )
    require(
        OT_ZIMAGE_SETUP,
        "Existing OneTrainer one-step dump pattern",
        [
            "OT_DUMP_STEP1_INPUTS",
            "OT_DUMP_STEP1_PATH",
            "/tmp/ot_zimage_step1_inputs.safetensors",
            "safetensors.torch",
            "scaled_latent_image",
            "latent_noise",
            "scaled_noisy_latent_image",
            "flow_target",
            "predicted_flow",
        ],
    )


def validate_dump_artifact(path: Path) -> None:
    header = read_safetensors_header(path)
    keys = {key for key in header if key != "__metadata__"}
    expected = set(EXPECTED_TENSORS)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing:
        print(f"[flux-onestep-dump] FAIL dump artifact missing keys: {path}")
        for key in missing:
            print(f"  missing: {key}")
        raise SystemExit(1)
    if extra:
        print(f"[flux-onestep-dump] FAIL dump artifact unexpected keys: {path}")
        for key in extra[:32]:
            print(f"  extra: {key}")
        if len(extra) > 32:
            print(f"  ... {len(extra) - 32} more")
        raise SystemExit(1)

    for key, expected_shape in EXPECTED_SHAPES.items():
        info = header[key]
        shape = tuple(info.get("shape", []))
        if shape != expected_shape:
            raise SystemExit(
                "[flux-onestep-dump] "
                f"{key} shape {list(shape)} != {list(expected_shape)} in {path}"
            )
        dtype = info.get("dtype")
        if key in {"text_ids", "image_ids", "timestep_index"}:
            if dtype not in {"I64", "I32"}:
                raise SystemExit(
                    f"[flux-onestep-dump] {key} dtype {dtype!r} is not integer in {path}"
                )
        elif dtype not in {"BF16", "F32", "F16"}:
            raise SystemExit(
                f"[flux-onestep-dump] {key} dtype {dtype!r} is not floating in {path}"
            )

    print(f"[flux-onestep-dump] PASS dump artifact header: {path}")


def validate_manifest_artifact(path: Path) -> None:
    manifest = load_json(path)
    missing = [field for field in EXPECTED_MANIFEST_FIELDS if field not in manifest]
    if missing:
        print(f"[flux-onestep-dump] FAIL manifest missing fields: {path}")
        for field in missing:
            print(f"  missing: {field}")
        raise SystemExit(1)
    if str(manifest.get("onetrainer_config")) != str(OT_CONFIG):
        raise SystemExit(
            "[flux-onestep-dump] manifest onetrainer_config mismatch: "
            f"{manifest.get('onetrainer_config')!r} != {str(OT_CONFIG)!r}"
        )
    manifest_tensors = manifest.get("tensors")
    if not isinstance(manifest_tensors, dict):
        raise SystemExit(f"[flux-onestep-dump] manifest tensors must be an object: {path}")
    missing_tensors = sorted(set(EXPECTED_TENSORS) - set(manifest_tensors))
    if missing_tensors:
        print(f"[flux-onestep-dump] FAIL manifest missing tensor metadata: {path}")
        for key in missing_tensors:
            print(f"  missing tensor metadata: {key}")
        raise SystemExit(1)
    malformed: list[str] = []
    for key in EXPECTED_TENSORS:
        info = manifest_tensors.get(key)
        if not isinstance(info, dict):
            malformed.append(f"{key}: metadata is not an object")
            continue
        for field in ("shape", "dtype"):
            if field not in info:
                malformed.append(f"{key}: missing {field}")
    if malformed:
        print(f"[flux-onestep-dump] FAIL manifest tensor metadata incomplete: {path}")
        for item in malformed[:32]:
            print(f"  {item}")
        if len(malformed) > 32:
            print(f"  ... {len(malformed) - 32} more")
        raise SystemExit(1)
    print(f"[flux-onestep-dump] PASS manifest artifact: {path}")


def validate_step_header_contains(path: Path) -> None:
    header = read_safetensors_header(path)
    keys = {key for key in header if key != "__metadata__"}
    missing = sorted(set(EXPECTED_TENSORS) - keys)
    if missing:
        print(f"[flux-onestep-dump] FAIL normalized step dump missing keys: {path}")
        for key in missing:
            print(f"  missing: {key}")
        raise SystemExit(1)
    for key, expected_shape in EXPECTED_SHAPES.items():
        info = header[key]
        shape = tuple(info.get("shape", []))
        if shape != expected_shape:
            raise SystemExit(
                "[flux-onestep-dump] "
                f"{key} shape {list(shape)} != {list(expected_shape)} in {path}"
            )
        dtype = info.get("dtype")
        if key in {"text_ids", "image_ids", "timestep_index"}:
            if dtype not in {"I64", "I32"}:
                raise SystemExit(
                    f"[flux-onestep-dump] {key} dtype {dtype!r} is not integer in {path}"
                )
        elif dtype not in {"BF16", "F32", "F16"}:
            raise SystemExit(
                f"[flux-onestep-dump] {key} dtype {dtype!r} is not floating in {path}"
            )
    print(f"[flux-onestep-dump] PASS normalized step dump header: {path}")


def validate_adapter_header(path: Path) -> None:
    header = read_safetensors_header(path)
    keys = sorted(key for key in header if key != "__metadata__")
    if not keys:
        raise SystemExit(f"[flux-onestep-dump] adapter dump has no tensors: {path}")

    phases = ("adapter_before", "adapter_pre", "adapter_post", "adapter_after")
    names_by_phase: dict[str, set[str]] = {phase: set() for phase in phases}
    unexpected: list[str] = []
    for key in keys:
        if "." not in key:
            unexpected.append(key)
            continue
        phase, name = key.split(".", 1)
        if phase not in names_by_phase:
            unexpected.append(key)
            continue
        names_by_phase[phase].add(name)

    if unexpected:
        print(f"[flux-onestep-dump] FAIL adapter dump unexpected keys: {path}")
        for key in unexpected[:32]:
            print(f"  unexpected: {key}")
        if len(unexpected) > 32:
            print(f"  ... {len(unexpected) - 32} more")
        raise SystemExit(1)

    base_names = names_by_phase["adapter_before"]
    if not base_names:
        raise SystemExit(f"[flux-onestep-dump] adapter_before phase is empty: {path}")
    for phase in phases[1:]:
        if names_by_phase[phase] != base_names:
            missing = sorted(base_names - names_by_phase[phase])[:8]
            extra = sorted(names_by_phase[phase] - base_names)[:8]
            raise SystemExit(
                f"[flux-onestep-dump] {phase} key set differs in {path}; "
                f"missing={missing} extra={extra}"
            )
    bad_dtype = [
        f"{key}: {header[key].get('dtype')!r}"
        for key in keys
        if header[key].get("dtype") not in {"F32", "BF16", "F16"}
    ]
    if bad_dtype:
        print(f"[flux-onestep-dump] FAIL adapter dump non-floating tensors: {path}")
        for item in bad_dtype[:32]:
            print(f"  {item}")
        if len(bad_dtype) > 32:
            print(f"  ... {len(bad_dtype) - 32} more")
        raise SystemExit(1)
    print(
        "[flux-onestep-dump] PASS normalized adapter dump header: "
        f"{path} tensors_per_phase={len(base_names)} total_tensors={len(keys)}"
    )


def _missing_fields(data: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if field not in data]


def _extract_speed_vram(meta: dict[str, Any], step: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    speed_seconds = (
        step.get("seconds_per_step")
        or step.get("step_seconds")
        or meta.get("seconds_per_step")
        or meta.get("elapsed_seconds")
    )
    allocated_mib = (
        step.get("cuda_allocated_mib")
        or step.get("cuda_memory_allocated_mib")
        or meta.get("cuda_allocated_mib")
        or meta.get("cuda_memory_allocated_mib")
    )
    reserved_mib = (
        step.get("cuda_reserved_mib")
        or step.get("cuda_memory_reserved_mib")
        or meta.get("cuda_reserved_mib")
        or meta.get("cuda_memory_reserved_mib")
    )
    device_name = (
        step.get("cuda_device_name")
        or step.get("cuda_name")
        or meta.get("cuda_device_name")
        or meta.get("cuda_name")
    )
    cuda = meta.get("cuda")
    if isinstance(cuda, dict):
        allocated_mib = allocated_mib or cuda.get("allocated_mib")
        reserved_mib = reserved_mib or cuda.get("reserved_mib")
        device_name = (
            device_name
            or cuda.get("device_name")
            or cuda.get("name")
            or cuda.get("device")
        )
    return speed_seconds, allocated_mib, reserved_mib, device_name


def validate_parity_meta_artifact(
    path: Path,
    step_path: Path,
    adapters_path: Path,
) -> None:
    meta = load_json(path)
    missing = _missing_fields(meta, REQUIRED_TOP_META_FIELDS)
    if missing:
        print(f"[flux-onestep-dump] FAIL normalized meta missing fields: {path}")
        for field in missing:
            print(f"  missing: {field}")
        raise SystemExit(1)

    runtime_config = meta.get("runtime_config")
    if not isinstance(runtime_config, dict):
        raise SystemExit(f"[flux-onestep-dump] runtime_config must be an object: {path}")
    mismatches = [
        f"{key}: got {runtime_config.get(key)!r}, expected {expected!r}"
        for key, expected in EXPECTED_RUNTIME_CONFIG.items()
        if runtime_config.get(key) != expected
    ]
    if mismatches:
        print(f"[flux-onestep-dump] FAIL normalized meta runtime identity: {path}")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        raise SystemExit(1)

    trainable = meta.get("trainable_parameters")
    if not isinstance(trainable, dict):
        raise SystemExit(f"[flux-onestep-dump] trainable_parameters must be an object: {path}")
    if int(trainable.get("count", 0)) <= 0:
        raise SystemExit(f"[flux-onestep-dump] trainable parameter count missing: {path}")
    names = trainable.get("names", [])
    if not isinstance(names, list) or not names:
        raise SystemExit(f"[flux-onestep-dump] trainable parameter names missing: {path}")

    steps = meta.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[0], dict):
        raise SystemExit(f"[flux-onestep-dump] meta.steps[0] missing: {path}")
    step = steps[0]
    missing_step = _missing_fields(step, REQUIRED_STEP_META_FIELDS)
    if missing_step:
        print(f"[flux-onestep-dump] FAIL normalized meta step missing fields: {path}")
        for field in missing_step:
            print(f"  missing: {field}")
        raise SystemExit(1)

    if Path(str(step.get("safetensors"))) != step_path:
        raise SystemExit(
            "[flux-onestep-dump] meta step safetensors mismatch: "
            f"{step.get('safetensors')!r} != {str(step_path)!r}"
        )
    if Path(str(step.get("adapter_safetensors"))) != adapters_path:
        raise SystemExit(
            "[flux-onestep-dump] meta adapter_safetensors mismatch: "
            f"{step.get('adapter_safetensors')!r} != {str(adapters_path)!r}"
        )

    for opt_key in ("optimizer_before", "optimizer_after"):
        optimizer = step.get(opt_key)
        if not isinstance(optimizer, dict):
            raise SystemExit(f"[flux-onestep-dump] {opt_key} must be an object: {path}")
        if optimizer.get("class") != "AdamW":
            raise SystemExit(
                f"[flux-onestep-dump] {opt_key}.class {optimizer.get('class')!r} != 'AdamW'"
            )
    optimizer_after = step["optimizer_after"]
    state = optimizer_after.get("state")
    if not isinstance(state, dict):
        raise SystemExit(f"[flux-onestep-dump] optimizer_after.state missing: {path}")
    state_keys = set(state.get("keys", []))
    missing_state = sorted({"exp_avg", "exp_avg_sq", "step"} - state_keys)
    if missing_state:
        print(f"[flux-onestep-dump] FAIL AdamW state missing keys: {path}")
        for key in missing_state:
            print(f"  missing: {key}")
        raise SystemExit(1)

    speed_seconds, allocated_mib, reserved_mib, device_name = _extract_speed_vram(meta, step)
    speed_missing = []
    if speed_seconds is None:
        speed_missing.append("seconds_per_step or elapsed_seconds")
    if device_name is None:
        speed_missing.append("CUDA device name")
    if allocated_mib is None:
        speed_missing.append("cuda allocated MiB")
    if reserved_mib is None:
        speed_missing.append("cuda reserved MiB")
    if speed_missing:
        print(f"[flux-onestep-dump] FAIL speed/VRAM evidence missing: {path}")
        for field in speed_missing:
            print(f"  missing: {field}")
        raise SystemExit(1)

    print(
        "[flux-onestep-dump] PASS normalized meta: "
        f"{path} trainable_tensors={trainable.get('count')} "
        f"loss={step.get('loss_for_backward')} grad_norm={step.get('grad_norm_pre_clip')} "
        f"seconds={speed_seconds} cuda_device={device_name} cuda_allocated_mib={allocated_mib} "
        f"cuda_reserved_mib={reserved_mib}"
    )


def validate_contract_artifact(path: Path) -> None:
    contract = load_json(path)
    missing = _missing_fields(contract, REQUIRED_CONTRACT_FIELDS)
    if missing:
        print(f"[flux-onestep-dump] FAIL normalized contract missing fields: {path}")
        for field in missing:
            print(f"  missing: {field}")
        raise SystemExit(1)
    if contract.get("reference_repo") != str(ONETRAINER):
        raise SystemExit(
            "[flux-onestep-dump] contract reference_repo mismatch: "
            f"{contract.get('reference_repo')!r} != {str(ONETRAINER)!r}"
        )
    if contract.get("default_config") != str(OT_CONFIG):
        raise SystemExit(
            "[flux-onestep-dump] contract default_config mismatch: "
            f"{contract.get('default_config')!r} != {str(OT_CONFIG)!r}"
        )

    outputs = contract.get("default_outputs")
    if not isinstance(outputs, dict):
        raise SystemExit(f"[flux-onestep-dump] contract default_outputs must be an object: {path}")
    missing_outputs = [
        field for field in REQUIRED_CONTRACT_OUTPUT_FIELDS if field not in outputs
    ]
    if missing_outputs:
        print(f"[flux-onestep-dump] FAIL normalized contract default_outputs missing: {path}")
        for field in missing_outputs:
            print(f"  missing default_outputs.{field}")
        raise SystemExit(1)

    policy = contract.get("numeric_parity_policy")
    if not isinstance(policy, dict):
        raise SystemExit(f"[flux-onestep-dump] numeric_parity_policy must be an object: {path}")
    status = str(policy.get("status", "")).lower()
    if status in {"accepted", "pass", "passed", "complete"}:
        raise SystemExit(
            f"[flux-onestep-dump] contract numeric parity status must not be accepted: {status!r}"
        )
    for claim_key in ("claimed", "accepted", "numeric_parity_claimed", "claimed_by_dry_run"):
        if policy.get(claim_key) is True:
            raise SystemExit(
                f"[flux-onestep-dump] contract {claim_key}=true would claim parity"
            )

    resolved_outputs = {
        field: resolve_port_path(outputs.get(field))
        for field in REQUIRED_CONTRACT_OUTPUT_FIELDS
    }
    print(
        "[flux-onestep-dump] PASS normalized contract: "
        f"{path} outputs="
        + ", ".join(f"{field}={resolved_outputs[field]}" for field in REQUIRED_CONTRACT_OUTPUT_FIELDS)
    )


def validate_blockers_artifact(path: Path) -> list[str]:
    artifact = load_json(path)
    missing = _missing_fields(artifact, REQUIRED_BLOCKERS_FIELDS)
    if missing:
        print(f"[flux-onestep-dump] FAIL normalized blockers missing fields: {path}")
        for field in missing:
            print(f"  missing: {field}")
        raise SystemExit(1)

    status = str(artifact.get("numeric_parity_status", "")).lower()
    if status in {"accepted", "pass", "passed", "complete"}:
        raise SystemExit(
            f"[flux-onestep-dump] blocker artifact numeric_parity_status={status!r} claims parity"
        )
    if artifact.get("numeric_parity_claimed") is True:
        raise SystemExit("[flux-onestep-dump] blocker artifact numeric_parity_claimed=true")

    structured = artifact.get("structured_blockers")
    if not isinstance(structured, list):
        raise SystemExit(f"[flux-onestep-dump] structured_blockers must be a list: {path}")
    if artifact.get("blocked") is True and not structured:
        raise SystemExit(f"[flux-onestep-dump] blocked=true but structured_blockers is empty: {path}")

    blockers: list[str] = []
    if artifact.get("one_step_dump_produced") is not True:
        blockers.append(
            f"{path}: one_step_dump_produced={artifact.get('one_step_dump_produced')!r}; "
            "no Flux.1 one-step train dump is recorded"
        )
    if artifact.get("blocked") is True:
        blockers.append(f"{path}: blocked=true")
    for item in structured:
        if not isinstance(item, dict):
            blockers.append(f"{path}: structured_blockers contains non-object item")
            continue
        blocker_id = item.get("id", "unnamed_blocker")
        message = item.get("message", "no message")
        category = item.get("category", "uncategorized")
        blockers.append(f"{path}: {category}.{blocker_id}: {message}")

    print(
        "[flux-onestep-dump] PASS normalized blockers: "
        f"{path} blocked={artifact.get('blocked')} structured_blockers={len(structured)} "
        f"numeric_parity_status={artifact.get('numeric_parity_status')!r}"
    )
    return blockers


def report_readiness(
    step_path: Path,
    adapters_path: Path,
    meta_path: Path,
    contract_path: Path,
    blockers_path: Path,
    require_ready: bool,
) -> tuple[list[str], list[str]]:
    print("[flux-onestep-dump] PARITY READINESS")
    print("  expected_model: FLUX_DEV_1 / LORA / BFLOAT_16 / batch=1 / resolution=512")
    print("  expected_optimizer: AdamW beta1=0.9 beta2=0.999 eps=1e-8 weight_decay=0.01")
    print("  expected_checkpointing: CPU_OFFLOADED")
    print("  required_evidence:")
    for name, anchor, evidence in REQUIRED_EVIDENCE:
        print(f"    - {name}: anchor={anchor}; evidence={evidence}")

    blockers: list[str] = []
    validation_errors: list[str] = []
    artifacts = (
        (
            "normalized train step dump",
            step_path,
            lambda path: validate_step_header_contains(path),
        ),
        (
            "normalized adapter/update dump",
            adapters_path,
            lambda path: validate_adapter_header(path),
        ),
        (
            "normalized metadata",
            meta_path,
            lambda path: validate_parity_meta_artifact(path, step_path, adapters_path),
        ),
        ("normalized contract", contract_path, validate_contract_artifact),
        ("normalized blockers", blockers_path, validate_blockers_artifact),
    )
    for label, path, validator in artifacts:
        if path.exists():
            print(f"[flux-onestep-dump] PRESENT {label}: {path}")
            content_blockers, errors = run_artifact_validator(label, path, validator)
            blockers.extend(content_blockers)
            validation_errors.extend(errors)
        else:
            print(f"[flux-onestep-dump] MISSING {label}: {path}")
            print_missing_requirements(label)
            blockers.append(f"missing {label}: {path}")

    print("  normalized_artifacts:")
    for label, path, anchor, evidence in REQUIRED_READINESS_ARTIFACTS:
        if not label.startswith("normalized "):
            continue
        status = "present" if path.exists() else "missing"
        print(f"    - {label}: {status} {path}")
        print(f"      source_anchor: {anchor}")
        print(f"      proves: {evidence}")

    if blockers:
        print("[flux-onestep-dump] NORMALIZED PARITY ARTIFACT BLOCKERS")
        for blocker in blockers:
            print(f"  - {blocker}")
        print("  - no accepted Flux.1 transformer/backward/AdamW/speed parity gate yet")
    else:
        print("[flux-onestep-dump] PASS normalized parity artifacts")
    if validation_errors:
        print("[flux-onestep-dump] ARTIFACT VALIDATION ERRORS")
        for error in validation_errors:
            print(f"  - {error}")
    if require_ready and blockers:
        print("[flux-onestep-dump] STRICT readiness incomplete")
    return blockers, validation_errors


def report_dump_plan(
    dump_path: Path,
    manifest_path: Path,
) -> tuple[list[str], list[str]]:
    print("[flux-onestep-dump] RAW FLUX.1 STEP DUMP FILES")
    print(f"  primary_safetensors: {dump_path}")
    print(f"  manifest_json: {manifest_path}")
    print("  source_anchor: /home/alex/OneTrainer/modules/modelSetup/BaseFluxSetup.py::predict")
    print("  trainer_anchor: /home/alex/OneTrainer/modules/trainer/GenericTrainer.py::train")
    print("  trigger_env: OT_DUMP_STEP1_INPUTS=1")
    print(f"  trigger_env: OT_DUMP_STEP1_PATH={dump_path}")
    print("  expected_tensor_keys:")
    for key in EXPECTED_TENSORS:
        print(f"    - {key}")

    blockers: list[str] = []
    validation_errors: list[str] = []
    if dump_path.exists():
        print(f"[flux-onestep-dump] PRESENT primary dump artifact: {dump_path}")
        content_blockers, errors = run_artifact_validator(
            "raw CUDA forward/loss dump", dump_path, validate_dump_artifact
        )
        blockers.extend(content_blockers)
        validation_errors.extend(errors)
    else:
        print(f"[flux-onestep-dump] MISSING primary dump artifact: {dump_path}")
        print_missing_requirements("raw CUDA forward/loss dump")
        blockers.append(f"missing raw CUDA forward/loss dump: {dump_path}")

    if manifest_path.exists():
        print(f"[flux-onestep-dump] PRESENT manifest artifact: {manifest_path}")
        content_blockers, errors = run_artifact_validator(
            "raw dump manifest", manifest_path, validate_manifest_artifact
        )
        blockers.extend(content_blockers)
        validation_errors.extend(errors)
    else:
        print(f"[flux-onestep-dump] MISSING manifest artifact: {manifest_path}")
        print_missing_requirements("raw dump manifest")
        blockers.append(f"missing raw dump manifest: {manifest_path}")

    print("  raw_artifacts:")
    for label, path, anchor, evidence in REQUIRED_READINESS_ARTIFACTS:
        if not label.startswith("raw "):
            continue
        status = "present" if path.exists() else "missing"
        print(f"    - {label}: {status} {path}")
        print(f"      source_anchor: {anchor}")
        print(f"      proves: {evidence}")

    if blockers:
        print("[flux-onestep-dump] RAW STEP DUMP FILE BLOCKERS")
        for blocker in blockers:
            print(f"  - {blocker}")
    else:
        print("[flux-onestep-dump] PASS raw Flux.1 step dump files")

    return blockers, validation_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the local OneTrainer Flux.1 one-step dump contract."
    )
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=DEFAULT_DUMP_PATH,
        help="primary safetensors artifact path to report",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="metadata manifest path to report",
    )
    parser.add_argument(
        "--require-artifact",
        action="store_true",
        help="fail if the reported dump artifacts are not present",
    )
    parser.add_argument(
        "--parity-step-path",
        type=Path,
        default=DEFAULT_PARITY_STEP_PATH,
        help="normalized repo-local train-step safetensors path to report",
    )
    parser.add_argument(
        "--parity-adapters-path",
        type=Path,
        default=DEFAULT_PARITY_ADAPTERS_PATH,
        help="normalized repo-local adapter/update safetensors path to report",
    )
    parser.add_argument(
        "--parity-meta-path",
        type=Path,
        default=DEFAULT_PARITY_META_PATH,
        help="normalized repo-local metadata JSON path to report",
    )
    parser.add_argument(
        "--parity-contract-path",
        type=Path,
        default=DEFAULT_PARITY_CONTRACT_PATH,
        help="normalized repo-local contract JSON path to report",
    )
    parser.add_argument(
        "--parity-blockers-path",
        type=Path,
        default=DEFAULT_PARITY_BLOCKERS_PATH,
        help="normalized repo-local blockers JSON path to report",
    )
    parser.add_argument(
        "--require-parity-ready",
        action="store_true",
        help=(
            "fail unless the Flux.1 raw dump and normalized transformer/backward/"
            "AdamW/speed readiness artifacts are present and internally valid"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_json(OT_CONFIG)
    scheduler_path = require_config(config)
    require_scheduler(scheduler_path)
    if not OT_BASELINE_LORA.exists():
        raise SystemExit(f"[flux-onestep-dump] missing baseline LoRA: {OT_BASELINE_LORA}")
    print(f"[flux-onestep-dump] PASS baseline LoRA artifact: {OT_BASELINE_LORA}")
    check_onetrainer_sources()
    raw_blockers, raw_validation_errors = report_dump_plan(
        args.dump_path,
        args.manifest_path,
    )
    readiness_blockers, readiness_validation_errors = report_readiness(
        args.parity_step_path,
        args.parity_adapters_path,
        args.parity_meta_path,
        args.parity_contract_path,
        args.parity_blockers_path,
        args.require_parity_ready,
    )
    validation_errors = raw_validation_errors + readiness_validation_errors
    if validation_errors:
        raise SystemExit("[flux-onestep-dump] artifact validation failed")
    if args.require_artifact and raw_blockers:
        raise SystemExit(
            "[flux-onestep-dump] raw Flux.1 step dump file readiness incomplete: "
            + "; ".join(raw_blockers)
        )
    parity_ready_blockers = raw_blockers + readiness_blockers
    if args.require_parity_ready and parity_ready_blockers:
        raise SystemExit(
            "[flux-onestep-dump] Flux.1 parity readiness incomplete: "
            + "raw_step_dump_files=["
            + "; ".join(raw_blockers)
            + "]; normalized_parity_artifacts=["
            + "; ".join(readiness_blockers)
            + "]"
        )
    if raw_blockers or readiness_blockers:
        print("[flux-onestep-dump] PASS source/static checks; Flux.1 train replay readiness is incomplete")
    else:
        print("[flux-onestep-dump] PASS source/static checks; Flux.1 train replay artifacts are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
