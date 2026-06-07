#!/usr/bin/env python3
"""Static guard for the OneTrainer product sampler harness contract.

This is a scaffold guard, not a speed-parity claim. It verifies that the local
Mojo harness mirrors OneTrainer's sampler lifecycle and exposes the required
seconds/step and VRAM fields while failing loud until full denoise/decode/save is
wired.
"""

from __future__ import annotations

from pathlib import Path

from check_qwen_sampler_speed_contract import run_checks as run_qwen_sampler_speed_contract


REPO = Path(__file__).resolve().parents[1]
ONETRAINER = Path("/home/alex/OneTrainer")

MOJO_HARNESS = REPO / "serenitymojo/sampling/product_sampler_harness.mojo"
MOJO_SMOKE = REPO / "serenitymojo/sampling/product_sampler_harness_smoke.mojo"
DOC = REPO / "serenitymojo/docs/SAMPLER_PRODUCT_HARNESS_2026-06-05.md"
PORT_DOC = REPO / "OT_MOJO_PORT_REMAINING.md"


def read(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"[sampler-product] missing file: {path}")
    return path.read_text(encoding="utf-8")


def require(path: Path, label: str, needles: list[str]) -> None:
    text = read(path)
    missing = [needle for needle in needles if needle not in text]
    if missing:
        print(f"[sampler-product] FAIL {label}: {path}")
        for needle in missing:
            print(f"  missing: {needle}")
        raise SystemExit(1)
    print(f"[sampler-product] PASS {label}")


def forbid(path: Path, label: str, needles: list[str]) -> None:
    text = read(path)
    found = [needle for needle in needles if needle in text]
    if found:
        print(f"[sampler-product] FAIL {label}: {path}")
        for needle in found:
            print(f"  forbidden: {needle}")
        raise SystemExit(1)
    print(f"[sampler-product] PASS {label}")


def main() -> int:
    require(
        ONETRAINER / "modules/util/config/SampleConfig.py",
        "OneTrainer SampleConfig fields",
        [
            "def _get_model_defaults",
            "\"diffusion_steps\"",
            "\"cfg_scale\"",
            "\"negative_prompt\"",
            "def from_train_config",
            "sample_inpainting",
        ],
    )
    require(
        ONETRAINER / "modules/modelSampler/BaseModelSampler.py",
        "OneTrainer BaseModelSampler lifecycle",
        [
            "class ModelSamplerOutput",
            "def sample(",
            "on_sample: Callable[[ModelSamplerOutput], None]",
            "on_update_progress: Callable[[int, int], None]",
            "def save_sampler_output",
            "image.save(destination + image_format.extension()",
        ],
    )
    require(
        ONETRAINER / "modules/trainer/GenericTrainer.py",
        "OneTrainer training sample loop",
        [
            "def __sample_loop(",
            "sample_config.from_train_config(self.config)",
            "self.model.to(self.temp_device)",
            "self.model_sampler.sample(",
            "on_update_progress=on_update_progress",
            "self.callbacks.on_sample_default",
        ],
    )

    for name in [
        "FluxSampler.py",
        "Flux2Sampler.py",
        "ChromaSampler.py",
        "QwenSampler.py",
        "ErnieSampler.py",
        "ZImageSampler.py",
    ]:
        require(
            ONETRAINER / f"modules/modelSampler/{name}",
            f"OneTrainer {name} sample surface",
            [
                "sampler_output = self.__sample_base(",
                "self.save_sampler_output(",
                "on_sample(sampler_output)",
                "on_update_progress(i + 1, len(timesteps))",
                "vae.decode(",
            ],
        )

    require(
        MOJO_HARNESS,
        "Mojo product harness fields",
        [
            "ProductSamplerRunContract",
            "SamplerProductStageStatus",
            "SamplerProductMeasurements",
            "text_conditioning_ready",
            "transformer_denoise_ready",
            "vae_decode_ready",
            "postprocess_save_ready",
            "progress_callbacks_ready",
            "output_callback_ready",
            "mojo_seconds_per_step",
            "ot_baseline_seconds_per_step",
            "mojo_peak_vram_mib",
            "ot_peak_vram_mib",
            "speed_parity_accepted",
            "measurement scaffold only, speed parity not accepted",
            "raise Error",
        ],
    )
    forbid(
        MOJO_HARNESS,
        "Mojo product harness dtype boundaries",
        [
            "DType.float32",
            "STDtype.F32",
            "Tensor.from_host",
            ".to_host(",
            "List[Float32]",
        ],
    )
    require(
        MOJO_SMOKE,
        "Mojo product harness smoke",
        [
            "FLUX_2_DEV",
            "flux2_dev",
            "blocked as expected",
            "empty measurement blocked as expected",
            "validate_product_sampler_ready",
            "product_sampler_harness_smoke PASS",
        ],
    )
    require(
        DOC,
        "harness docs",
        [
            "Reference is OneTrainer",
            "Measurement scaffold, not accepted speed parity",
            "Qwen-Image Speed-Parity Evidence Gate",
            "seconds/step",
            "peak VRAM",
            "text conditioning",
            "VAE decode",
        ],
    )
    require(
        PORT_DOC,
        "remaining-port doc sampler status",
        [
            "product sampler harness",
            "measurement scaffold",
            "accepted sampler speed parity",
        ],
    )
    run_qwen_sampler_speed_contract()

    print("[sampler-product] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
