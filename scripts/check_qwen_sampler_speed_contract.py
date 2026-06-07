#!/usr/bin/env python3
"""Static guard for Qwen-Image sampler speed-parity claims.

Reference is intentionally limited to local OneTrainer. This does not claim
sampler speed parity. It blocks accepted Qwen-Image speed claims unless the
claim carries the exact run identity and measurement evidence needed to compare
OneTrainer and Mojo runs.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")

OT_SAMPLER = ONETRAINER / "modules/modelSampler/QwenSampler.py"
OT_SETUP = ONETRAINER / "modules/modelSetup/BaseQwenSetup.py"
OT_MODEL = ONETRAINER / "modules/model/QwenModel.py"
OT_SAMPLE_CONFIG = ONETRAINER / "modules/util/config/SampleConfig.py"
OT_TRAIN_CONFIG = ONETRAINER / "configs/qwen_100step_baseline.json"

MOJO_QWEN_SAMPLER = REPO / "serenitymojo/sampling/qwenimage_sampling.mojo"
MOJO_QWEN_SMOKE = REPO / "serenitymojo/sampling/qwenimage_sampling_smoke.mojo"
MOJO_PRODUCT_HARNESS = REPO / "serenitymojo/sampling/product_sampler_harness.mojo"
MOJO_PRODUCT_SMOKE = REPO / "serenitymojo/sampling/product_sampler_harness_smoke.mojo"
MOJO_OT_SAMPLER_CONTRACT = REPO / "serenitymojo/sampling/onetrainer_sampler_contract.mojo"
MOJO_PIPELINE_DIR = REPO / "serenitymojo/pipeline"
DOC = REPO / "serenitymojo/docs/SAMPLER_PRODUCT_HARNESS_2026-06-05.md"
PORT_DOC = REPO / "OT_MOJO_PORT_REMAINING.md"

QWEN_CLAIM_ALIASES = (
    "qwen",
    "qwenimage",
    "qwen-image",
    "qwen_image",
    "qwen image",
    "qwen-image-2512",
    "qwen image sampler",
)

SPEED_PARITY_CLAIM_MARKERS = (
    "speed_parity_accepted",
    "sampler speed parity accepted",
    "accepted sampler speed parity",
    "speed parity accepted",
    "sampler_speed_parity_accepted",
    "speed parity: accepted",
)

NEGATIVE_CLAIM_MARKERS = (
    "not accepted",
    "not a speed-parity claim",
    "no model has accepted",
    "cannot be accepted",
    "must not",
    "not image or speed parity",
    "not image or speed-parity",
    "not an image sampler",
    "scaffold",
    "blocked",
    "does not claim",
)

QWEN_SPEED_EVIDENCE_GROUPS = (
    (
        "OneTrainer seconds/step",
        ("onetrainer seconds/step", "ot_baseline_seconds_per_step", "ot_seconds_per_step"),
    ),
    ("Mojo seconds/step", ("mojo seconds/step", "mojo_seconds_per_step")),
    ("OneTrainer peak VRAM", ("onetrainer peak vram", "ot_peak_vram_mib")),
    ("Mojo peak VRAM", ("mojo peak vram", "mojo_peak_vram_mib")),
    ("prompt", ("prompt",)),
    ("seed", ("seed",)),
    ("resolution", ("resolution", "width", "height")),
    ("steps", ("steps", "diffusion_steps", "sample_steps")),
    ("cfg/guidance", ("cfg", "cfg_scale", "guidance", "guidance_scale")),
    ("dtype", ("dtype", "train_dtype")),
    (
        "denoise trajectory",
        ("denoise trajectory", "denoise_trajectory", "latent trajectory", "trajectory"),
    ),
)

CLAIM_SCAN_FILES = [
    MOJO_QWEN_SAMPLER,
    MOJO_QWEN_SMOKE,
    MOJO_PRODUCT_HARNESS,
    MOJO_PRODUCT_SMOKE,
    MOJO_OT_SAMPLER_CONTRACT,
    MOJO_PIPELINE_DIR / "qwenimage_pipeline_smoke.mojo",
    MOJO_PIPELINE_DIR / "qwenimage_pipeline_512_multistep.mojo",
    MOJO_PIPELINE_DIR / "qwenimage_pipeline_1024_multistep.mojo",
    MOJO_PIPELINE_DIR / "qwenimage_contract_smoke.mojo",
    MOJO_PIPELINE_DIR / "qwenimage_vae_smoke.mojo",
    DOC,
    PORT_DOC,
]


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"[qwen-sampler-speed] missing file: {path}")
    return path.read_text(encoding="utf-8")


def require(path: Path, label: str, needles: list[str]) -> None:
    text = read(path)
    missing = [needle for needle in needles if needle not in text]
    if missing:
        print(f"[qwen-sampler-speed] FAIL {label}: {path}")
        for needle in missing:
            print(f"  missing: {needle}")
        raise SystemExit(1)
    print(f"[qwen-sampler-speed] PASS {label}")


def require_json_values(path: Path, label: str, expected: dict[str, object]) -> None:
    data = json.loads(read(path))
    mismatches = [
        f"{key}: got {data.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if data.get(key) != value
    ]
    if mismatches:
        print(f"[qwen-sampler-speed] FAIL {label}: {path}")
        for mismatch in mismatches:
            print(f"  {mismatch}")
        raise SystemExit(1)
    print(f"[qwen-sampler-speed] PASS {label}")


def require_no_forced_f32_tensor_storage(path: Path, label: str) -> None:
    text = read(path)
    forbidden = [
        "STDtype.F32",
        "DType.float32",
        "Tensor.from_host",
        ".to_host(",
    ]
    found = [needle for needle in forbidden if needle in text]
    if found:
        print(f"[qwen-sampler-speed] FAIL dtype boundary {label}: {path}")
        for needle in found:
            print(f"  forbidden tensor-storage pattern: {needle}")
        raise SystemExit(1)
    require(
        path,
        f"dtype-preserving sampler math {label}",
        [
            "Schedule math is F32 host scalar work",
            "preserve the input tensor storage dtype",
        ],
    )
    print(f"[qwen-sampler-speed] PASS dtype boundary {label}")


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _claim_context(lines: list[str], index: int, radius: int = 24) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return "\n".join(lines[start:end]).lower()


def require_qwen_speed_claim_evidence() -> None:
    """Require hard run evidence before any bounded Qwen speed-parity claim."""
    found_claims = 0
    for path in CLAIM_SCAN_FILES:
        text = read(path)
        lines = text.splitlines()
        for index, line in enumerate(lines):
            lowered = line.lower()
            if not _has_any(lowered, SPEED_PARITY_CLAIM_MARKERS):
                continue
            context = _claim_context(lines, index)
            if not _has_any(context, QWEN_CLAIM_ALIASES):
                continue
            if _has_any(context, NEGATIVE_CLAIM_MARKERS):
                continue

            found_claims += 1
            missing = [
                label
                for label, markers in QWEN_SPEED_EVIDENCE_GROUPS
                if not _has_any(context, markers)
            ]
            if missing:
                print(
                    "[qwen-sampler-speed] FAIL Qwen speed-parity claim "
                    f"evidence: {path}:{index + 1}"
                )
                for label in missing:
                    print(f"  missing evidence marker near claim: {label}")
                raise SystemExit(1)

    if found_claims:
        print(f"[qwen-sampler-speed] PASS Qwen speed-parity evidence claims={found_claims}")
    else:
        print("[qwen-sampler-speed] PASS Qwen speed-parity evidence gate: no accepted claim")


def run_checks() -> None:
    require(
        OT_SAMPLE_CONFIG,
        "OneTrainer Qwen sample defaults",
        [
            "elif model_type.is_qwen():",
            '"width": 1024',
            '"height": 1024',
            '"diffusion_steps": 25',
            '"cfg_scale": 3.5',
        ],
    )
    require(
        OT_SAMPLER,
        "OneTrainer Qwen sampler run identity",
        [
            "def __sample_base(",
            "prompt: str",
            "negative_prompt: str",
            "height: int",
            "width: int",
            "seed: int",
            "diffusion_steps: int",
            "cfg_scale: float",
            "generator.manual_seed(seed)",
            "height=self.quantize_resolution(sample_config.height, 64)",
            "width=self.quantize_resolution(sample_config.width, 64)",
        ],
    )
    require(
        OT_SAMPLER,
        "OneTrainer Qwen denoise path",
        [
            "batch_size = 2 if cfg_scale > 1.0 else 1",
            "text=[prompt, negative_prompt] if cfg_scale > 1.0 else prompt",
            "dtype=torch.float32",
            "shift = self.model.calculate_timestep_shift",
            "latent_image = self.model.pack_latents(latent_image)",
            "noise_scheduler.set_timesteps(diffusion_steps, device=self.train_device, mu=math.log(shift))",
            'for i, timestep in enumerate(tqdm(timesteps, desc="sampling")):',
            "hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype())",
            "timestep=expanded_timestep / 1000",
            "encoder_hidden_states=combined_prompt_embedding.to(dtype=self.model.train_dtype.torch_dtype())",
            "noise_pred = noise_pred_negative + cfg_scale * (noise_pred_positive - noise_pred_negative)",
            "latent_image = noise_scheduler.step(",
            "on_update_progress(i + 1, len(timesteps))",
        ],
    )
    require(
        OT_SAMPLER,
        "OneTrainer Qwen VAE/postprocess path",
        [
            "latent_image = self.model.unpack_latents(",
            "latents = self.model.unscale_latents(latent_image)",
            "vae.decode(latents, return_dict=False)[0].squeeze(-3)",
            "image_processor.postprocess(image, output_type='pil', do_denormalize=do_denormalize)",
            "self.save_sampler_output(",
            "on_sample(sampler_output)",
        ],
    )
    require(
        OT_SETUP,
        "OneTrainer Qwen training/setup flow target",
        [
            "latent_image = batch['latent_image']",
            "scaled_latent_image = model.scale_latents(latent_image)",
            "model.calculate_timestep_shift(scaled_latent_image.shape[-2], scaled_latent_image.shape[-1])",
            "packed_latent_input = model.pack_latents(latent_input)",
            "hidden_states=packed_latent_input.to(dtype=model.train_dtype.torch_dtype())",
            "timestep=timestep / 1000",
            "flow = latent_noise - scaled_latent_image",
        ],
    )
    require(
        OT_MODEL,
        "OneTrainer Qwen model scale/shift/text contract",
        [
            "DEFAULT_PROMPT_TEMPLATE_CROP_START = 34",
            "PROMPT_MAX_LENGTH = 512",
            "def calculate_timestep_shift",
            "return math.exp(mu)",
            "def pack_latents(latents: Tensor) -> Tensor:",
            "def unpack_latents(latents, height: int, width: int) -> Tensor:",
            "return (latents - latents_mean) * latents_std",
            "return latents / latents_std + latents_mean",
        ],
    )
    require_json_values(
        OT_TRAIN_CONFIG,
        "OneTrainer Qwen baseline config",
        {
            "model_type": "QWEN",
            "training_method": "LORA",
            "train_dtype": "BFLOAT_16",
            "seed": 42,
            "sample_after_unit": "NEVER",
        },
    )
    require(
        MOJO_QWEN_SAMPLER,
        "Mojo Qwen sampler helper",
        [
            "build_qwenimage_onetrainer_sigmas",
            "qwenimage_dynamic_shift_value",
            "qwenimage_scheduler_timestep_from_sigma",
            "qwenimage_model_timestep_from_sigma",
            "qwenimage_cfg",
            "qwenimage_euler_step",
            "QwenImageFlowMatchScheduler",
        ],
    )
    require_no_forced_f32_tensor_storage(MOJO_QWEN_SAMPLER, "Mojo Qwen sampler helper")
    require(
        MOJO_QWEN_SMOKE,
        "Mojo Qwen sampler smoke",
        [
            "Qwen-Image packed seq_len mismatch",
            "Qwen-Image CFG changed tensor dtype",
            "Qwen-Image Euler step changed tensor dtype",
            "Qwen-Image FlowMatch scheduler/tensor smoke PASS",
        ],
    )
    require(
        MOJO_OT_SAMPLER_CONTRACT,
        "Mojo OneTrainer Qwen sampler plan",
        [
            "OT_SAMPLER_QWEN",
            'model_type == String("qwenimage")',
            "scheduler_mode = OT_SCHEDULER_MODEL_COPY_MU",
            "steps = 25",
            "cfg = Float32(3.5)",
            "quant = 64",
        ],
    )
    require(
        MOJO_PRODUCT_HARNESS,
        "Mojo product harness speed fields",
        [
            "ProductSamplerRunContract",
            "SamplePrompt",
            "run.prompt.prompt",
            "run.plan.width",
            "run.plan.height",
            "run.plan.diffusion_steps",
            "run.plan.cfg_scale",
            "SamplerProductMeasurements",
            "ot_baseline_seconds_per_step",
            "mojo_seconds_per_step",
            "ot_peak_vram_mib",
            "mojo_peak_vram_mib",
            "measured_steps",
            "speed_parity_accepted",
            "measurement scaffold only, speed parity not accepted",
        ],
    )
    require(
        DOC,
        "harness docs Qwen evidence gate",
        [
            "Qwen-Image Speed-Parity Evidence Gate",
            "OneTrainer `seconds/step`",
            "Mojo `seconds/step`",
            "denoise trajectory evidence",
            "scripts/check_qwen_sampler_speed_contract.py",
        ],
    )
    require_qwen_speed_claim_evidence()


def main() -> int:
    run_checks()
    print("[qwen-sampler-speed] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
