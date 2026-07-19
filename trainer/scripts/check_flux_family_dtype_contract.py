#!/usr/bin/env python3
"""Static dtype-boundary guard for Flux-family/Z-Image/Klein port files.

This is a source-level guard only. It does not prove numeric parity. Its job is
to catch the fatal class of regression where persistent checkpoint/model tensors
are silently loaded or stored as F32 instead of preserving checkpoint dtype.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/alex/serenity-trainer")
SERENITY = Path("/home/alex/Serenity")
ANIMA_SERENITY = Path("/home/alex/Serenity-anima-ref")


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def read_abs(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fail(message: str) -> None:
    raise RuntimeError(message)


def section(text: str, marker: str, end_marker: str | None = None) -> str:
    start = text.find(marker)
    if start < 0:
        fail(f"missing marker: {marker}")
    if end_marker is None:
        return text[start:]
    end = text.find(end_marker, start)
    if end < 0:
        fail(f"missing end marker after {marker}: {end_marker}")
    return text[start:end]


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"{label}: missing {needle!r}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        fail(f"{label}: forbidden {needle!r}")


def check_klein_weight_loader() -> list[str]:
    path = "src/serenity_trainer/model/klein/weights.mojo"
    text = read(path)
    notes: list[str] = []

    require(
        text,
        "Persistent model weights\n# must use `_load_tensor` below so BF16 checkpoint storage remains BF16 on device.",
        path,
    )
    require(text, "def _load_host_f32", path)
    require(text, "def _load_tensor", path)

    load_tensor = section(
        text,
        "def _load_tensor",
        "\n\n# True if a tensor with the given name exists",
    )
    forbid(load_tensor, "cast_tensor", "_load_tensor must not cast")
    forbid(load_tensor, "STDtype.F32", "_load_tensor must not request F32")
    require(load_tensor, "Tensor.from_view(tv, ctx)", "_load_tensor dtype preservation")
    notes.append("_load_tensor preserves safetensors dtype via Tensor.from_view")

    load_stream = section(
        text,
        "def _load_stream",
        "\n\n# Load double block `block_idx`",
    )
    forbid(load_stream, "_load_host_f32", "_load_stream persistent weights")
    require(load_stream, "_load_tensor", "_load_stream persistent weights")
    notes.append("_load_stream uses _load_tensor for resident block weights")

    load_single = section(
        text,
        "def load_single_block_weights",
        "\n\n# \u2500\u2500 shared base weights",
    )
    forbid(load_single, "_load_host_f32", "load_single_block_weights persistent weights")
    require(load_single, "_load_tensor", "load_single_block_weights persistent weights")
    notes.append("load_single_block_weights uses _load_tensor for resident weights")

    return notes


def check_sampler_reference_comments() -> list[str]:
    checks = [
        (
            "src/serenity_trainer/modelSampler/Flux2Sampler.mojo",
            [
                "persistent Euler latent stays F32 (Flux2Sampler.py:88 dtype=float32)",
                "latent_model_input is cast to train_dtype (BF16)",
            ],
        ),
        (
            "src/serenity_trainer/modelSampler/ZImageSampler.mojo",
            [
                "persistent latent / Euler state is F32 (ZImageSampler.py:84",
                "transformer INPUT is cast to train_dtype (BF16)",
            ],
        ),
        (
            "src/serenity_trainer/modelSampler/FluxSampler.mojo",
            [
                "Serenity creates the initial latent image as torch.float32",
                "reference reason as metadata",
            ],
        ),
        (
            "src/serenity_trainer/modelSampler/ChromaSampler.mojo",
            [
                "Serenity torch.randn(..., dtype=torch.float32)",
                "initial_noise_reference_reason",
            ],
        ),
    ]
    notes: list[str] = []
    for path, needles in checks:
        text = read(path)
        for needle in needles:
            require(text, needle, path)
        notes.append(f"{path} documents reference-correct F32 sampler boundary")
    return notes


def check_serenity_sampler_refs() -> list[str]:
    checks = [
        (
            SERENITY / "modules/modelSampler/FluxSampler.py",
            ["dtype=torch.float32", "latent_model_input.to(dtype=self.model.train_dtype.torch_dtype())"],
        ),
        (
            SERENITY / "modules/modelSampler/Flux2Sampler.py",
            ["dtype=torch.float32", "hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype())"],
        ),
        (
            SERENITY / "modules/modelSampler/ZImageSampler.py",
            ["dtype=torch.float32", "latent_model_input = latent_image.unsqueeze(2).to(dtype=self.model.train_dtype.torch_dtype())"],
        ),
        (
            SERENITY / "modules/modelSampler/ChromaSampler.py",
            ["dtype=torch.float32", "hidden_states=latent_model_input.to(dtype=self.model.train_dtype.torch_dtype())"],
        ),
    ]
    notes: list[str] = []
    for path, needles in checks:
        text = read_abs(path)
        for needle in needles:
            require(text, needle, str(path))
        notes.append(f"{path} confirms F32 sampler latent + train_dtype transformer input")
    return notes


def check_serenity_lora_dtype_refs() -> list[str]:
    setup = read_abs(SERENITY / "modules/modelSetup/StableDiffusionXLLoRASetup.py")
    require(setup, "model.unet_lora.to(dtype=config.lora_weight_dtype.torch_dtype())", "SDXL LoRA setup")

    config_checks = [
        (SERENITY / "configs/qwen_100step_baseline.json", '"weight_dtype": "FLOAT_32"'),
        (SERENITY / "configs/ernie_eri2_100step_baseline.json", '"weight_dtype": "FLOAT_32"'),
        (SERENITY / "configs/sdxl_100step_baseline.json", '"weight_dtype": "FLOAT_32"'),
        (SERENITY / "configs/chroma_100step_baseline.json", '"weight_dtype": "FLOAT_32"'),
        (ANIMA_SERENITY / "configs/anima_100step_baseline.json", '"weight_dtype": "FLOAT_32"'),
    ]
    notes: list[str] = []
    for path, needle in config_checks:
        text = read_abs(path)
        require(text, '"train_dtype": "BFLOAT_16"', str(path))
        require(text, needle, str(path))
        notes.append(f"{path} confirms BF16 train dtype with Serenity FLOAT_32 LoRA weight config")
    return notes


def check_zimage_saver() -> list[str]:
    path = "src/serenity_trainer/modelSaver/zImage/ZImageLoRASaver.mojo"
    text = read(path)
    require(text, "BF16 storage written verbatim", path)
    require(text, "no F32 cast", path)
    return [f"{path} documents BF16 LoRA save boundary"]


def main() -> int:
    notes: list[str] = []
    notes.extend(check_klein_weight_loader())
    notes.extend(check_sampler_reference_comments())
    notes.extend(check_serenity_sampler_refs())
    notes.extend(check_serenity_lora_dtype_refs())
    notes.extend(check_zimage_saver())
    print("FLUX FAMILY DTYPE CONTRACT PASS")
    for note in notes:
        print("-", note)
    print("scope: static source guard only; no numeric parity claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
