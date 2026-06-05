#!/usr/bin/env python3
"""Dump real creator/LTX Desktop LTX2 phase tensors for Mojo parity.

This is oracle tooling only. It runs the Desktop fast-video wrapper so model
resolution, quantization policy, streaming, prompt encoder, scheduler, upsampler,
and decoders match the creator path.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from tqdm import tqdm

from api_types import ImageConditioningInput
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.types import Audio, LatentState
from ltx_core.utils import to_denoised, to_velocity
from ltx_pipelines.distilled import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings, post_process_latent
from ltx_pipelines.utils.types import ModalitySpec
from services.fast_video_pipeline.ltx_fast_video_pipeline import LTXFastVideoPipeline
from services.ltx_pipeline_common import default_tiling_config, video_chunks_number


REPO = Path(__file__).resolve().parents[1]
DEFAULT_APPDATA = REPO / "output/ltx2_creator_desktop_appdata_20260604"
DEFAULT_OUT = REPO / "output/ltx2_creator_phase_dumps/creator_960x512_121f_seed42"
DEFAULT_PROMPT = (
    "A cinematic handheld shot of a glass greenhouse at sunrise, mist rolling "
    "through rows of orange flowers, realistic motion, natural lighting, "
    "detailed leaves, shallow depth of field."
)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _safe_key(name: str) -> str:
    return name.replace("/", "__").replace(".", "_")


class Recorder:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.tensors: dict[str, torch.Tensor] = {}
        self.stats: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def event(self, name: str, **payload: Any) -> None:
        self.events.append({"name": name, **payload})

    def tensor(self, name: str, value: torch.Tensor | None, *, save: bool = True) -> None:
        if value is None:
            self.stats[name] = {"present": False}
            return

        detached = value.detach()
        self.stats[name] = self._tensor_stats(detached)
        if save:
            self.tensors[_safe_key(name)] = detached.contiguous().cpu()

    def state(self, prefix: str, state: LatentState | None, *, save_positions: bool = False) -> None:
        if state is None:
            self.stats[prefix] = {"present": False}
            return
        self.tensor(f"{prefix}/latent", state.latent)
        self.tensor(f"{prefix}/clean_latent", state.clean_latent)
        self.tensor(f"{prefix}/denoise_mask", state.denoise_mask)
        self.tensor(f"{prefix}/positions", state.positions, save=save_positions)
        if state.attention_mask is not None:
            self.tensor(f"{prefix}/attention_mask", state.attention_mask)

    def flush(self, metadata: dict[str, Any]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tensors_path = self.out_dir / "creator_phase_tensors.safetensors"
        save_file(self.tensors, tensors_path)
        manifest = {
            "metadata": metadata,
            "tensors_file": str(tensors_path),
            "tensor_count": len(self.tensors),
            "tensors": self.stats,
            "events": self.events,
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @staticmethod
    def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
        cpu = tensor.detach().float().cpu()
        finite = torch.isfinite(cpu)
        flat = cpu.reshape(-1)
        if flat.numel() == 0:
            mean = std = min_v = max_v = absmax = math.nan
        else:
            mean = flat.mean().item()
            std = flat.std(unbiased=False).item()
            min_v = flat.min().item()
            max_v = flat.max().item()
            absmax = flat.abs().max().item()
        return {
            "present": True,
            "shape": list(tensor.shape),
            "dtype": _dtype_name(tensor.dtype),
            "device": str(tensor.device),
            "numel": tensor.numel(),
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "absmax": absmax,
            "finite": bool(finite.all().item()) if finite.numel() else True,
        }


class DumpingGaussianNoiser(GaussianNoiser):
    def __init__(self, generator: torch.Generator, recorder: Recorder, labels: list[str]) -> None:
        super().__init__(generator)
        self.recorder = recorder
        self.labels = labels
        self.index = 0

    def __call__(self, latent_state: LatentState, noise_scale: float = 1.0) -> LatentState:
        if self.index >= len(self.labels):
            label = f"noise_call_{self.index}"
        else:
            label = self.labels[self.index]
        self.index += 1

        noise = torch.randn(
            *latent_state.latent.shape,
            device=latent_state.latent.device,
            dtype=latent_state.latent.dtype,
            generator=self.generator,
        )
        scaled_mask = latent_state.denoise_mask * noise_scale
        latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)
        out = replace(latent_state, latent=latent.to(latent_state.latent.dtype))

        self.recorder.tensor(f"{label}/pre_noise_clean_latent", latent_state.latent)
        self.recorder.tensor(f"{label}/torch_randn_noise", noise)
        self.recorder.tensor(f"{label}/scaled_mask", scaled_mask)
        self.recorder.tensor(f"{label}/initial_noised_latent", out.latent)
        self.recorder.event(
            "noiser",
            label=label,
            noise_scale=float(noise_scale),
            generator_device=str(self.generator.device),
        )
        return out


class RawVelocityDumpingX0Model(torch.nn.Module):
    """Record raw velocity-model outputs without changing X0Model semantics."""

    def __init__(self, model: torch.nn.Module, recorder: Recorder, stage: str) -> None:
        super().__init__()
        self.model = model
        self.recorder = recorder
        self.stage = stage
        self.step_index = -1
        self.capture = False

    def forward(
        self,
        video: Any | None,
        audio: Any | None,
        perturbations: Any,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.capture:
            self._record_preprocessed_args(video, audio)
        vx, ax = self.model.velocity_model(video, audio, perturbations)
        if self.capture:
            self._record("video", video, vx)
            self._record("audio", audio, ax)

        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio

    def _record(self, modality: str, mod: Any | None, velocity: torch.Tensor | None) -> None:
        prefix = f"{self.stage}/{modality}/step_{self.step_index:02d}/transformer"
        if mod is None:
            self.recorder.stats[prefix] = {"present": False}
            return
        self.recorder.tensor(f"{prefix}/latent", mod.latent)
        self.recorder.tensor(f"{prefix}/sigma", mod.sigma)
        self.recorder.tensor(f"{prefix}/timesteps", mod.timesteps)
        self.recorder.tensor(f"{prefix}/positions", mod.positions)
        self.recorder.tensor(f"{prefix}/context", mod.context, save=False)
        if mod.attention_mask is not None:
            self.recorder.tensor(f"{prefix}/attention_mask", mod.attention_mask)
        self.recorder.tensor(f"{prefix}/raw_velocity", velocity)

    def _record_preprocessed_args(self, video: Any | None, audio: Any | None) -> None:
        velocity_model = getattr(self.model, "velocity_model", self.model)
        self.recorder.event(
            "transformer_args_capture_probe",
            stage=self.stage,
            step=self.step_index,
            model_type=type(self.model).__name__,
            velocity_model_type=type(velocity_model).__name__,
            has_video_preprocessor=hasattr(velocity_model, "video_args_preprocessor"),
            has_audio_preprocessor=hasattr(velocity_model, "audio_args_preprocessor"),
        )
        if video is not None and hasattr(velocity_model, "video_args_preprocessor"):
            args = velocity_model.video_args_preprocessor.prepare(video, audio)
            self._record_args("video", args)
        if audio is not None and hasattr(velocity_model, "audio_args_preprocessor"):
            args = velocity_model.audio_args_preprocessor.prepare(audio, video)
            self._record_args("audio", args)

    def _record_args(self, modality: str, args: Any) -> None:
        prefix = f"{self.stage}/{modality}/step_{self.step_index:02d}/transformer_args"
        self.recorder.tensor(f"{prefix}/x", args.x)
        self.recorder.tensor(f"{prefix}/context", args.context)
        self.recorder.tensor(f"{prefix}/timesteps", args.timesteps)
        self.recorder.tensor(f"{prefix}/embedded_timestep", args.embedded_timestep)
        self.recorder.tensor(f"{prefix}/rope_cos", args.positional_embeddings[0])
        self.recorder.tensor(f"{prefix}/rope_sin", args.positional_embeddings[1])
        if args.context_mask is not None:
            self.recorder.tensor(f"{prefix}/context_mask", args.context_mask)
        if args.cross_positional_embeddings is not None:
            self.recorder.tensor(f"{prefix}/cross_rope_cos", args.cross_positional_embeddings[0])
            self.recorder.tensor(f"{prefix}/cross_rope_sin", args.cross_positional_embeddings[1])
        if args.cross_scale_shift_timestep is not None:
            self.recorder.tensor(f"{prefix}/cross_scale_shift_timestep", args.cross_scale_shift_timestep)
        if args.cross_gate_timestep is not None:
            self.recorder.tensor(f"{prefix}/cross_gate_timestep", args.cross_gate_timestep)
        if args.prompt_timestep is not None:
            self.recorder.tensor(f"{prefix}/prompt_timestep", args.prompt_timestep)
        if args.self_attention_mask is not None:
            self.recorder.tensor(f"{prefix}/self_attention_mask", args.self_attention_mask)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _install_raw_velocity_dump(
    transformer: Any,
    recorder: Recorder,
    stage: str,
) -> RawVelocityDumpingX0Model | None:
    base = getattr(transformer, "_model", None)
    if base is None or not hasattr(base, "velocity_model"):
        recorder.event("raw_velocity_dump_unavailable", stage=stage, transformer_type=type(transformer).__name__)
        return None
    if isinstance(base, RawVelocityDumpingX0Model):
        return base

    wrapped = RawVelocityDumpingX0Model(base, recorder, stage)
    transformer._model = wrapped
    recorder.event("raw_velocity_dump_installed", stage=stage, model_type=type(base).__name__)
    return wrapped


def _step_state_with_dump(
    recorder: Recorder,
    stage: str,
    modality: str,
    step_idx: int,
    state: LatentState | None,
    denoised: torch.Tensor | None,
    stepper: Any,
    sigmas: torch.Tensor,
    capture: bool,
) -> LatentState | None:
    if state is None or denoised is None:
        return state

    sigma = sigmas[step_idx]
    denoised_pp = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
    velocity = to_velocity(state.latent, sigma, denoised_pp)
    next_latent = stepper.step(state.latent, denoised_pp, sigmas, step_idx)

    if capture:
        prefix = f"{stage}/{modality}/step_{step_idx:02d}"
        recorder.tensor(f"{prefix}/input_latent", state.latent)
        recorder.tensor(f"{prefix}/denoised_raw", denoised)
        recorder.tensor(f"{prefix}/denoised_post_process", denoised_pp)
        recorder.tensor(f"{prefix}/velocity_from_x0", velocity)
        recorder.tensor(f"{prefix}/next_latent", next_latent)
        recorder.event(
            "denoise_step",
            stage=stage,
            modality=modality,
            step=step_idx,
            sigma=float(sigmas[step_idx].detach().float().cpu().item()),
            sigma_next=float(sigmas[step_idx + 1].detach().float().cpu().item()),
        )

    return replace(state, latent=next_latent)


def dumping_euler_loop(stage: str, recorder: Recorder):
    def loop(
        sigmas: torch.Tensor,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        stepper: Any,
        transformer: Any,
        denoiser: Any,
    ) -> tuple[LatentState | None, LatentState | None]:
        recorder.tensor(f"{stage}/sigmas", sigmas)
        recorder.state(f"{stage}/loop_init/video", video_state, save_positions=True)
        recorder.state(f"{stage}/loop_init/audio", audio_state, save_positions=True)

        last_step = int(sigmas.numel() - 2)
        capture_steps = {0, last_step}
        raw_velocity_dump = _install_raw_velocity_dump(transformer, recorder, stage)
        for step_idx, _ in enumerate(tqdm(sigmas[:-1], desc=f"{stage} creator Euler")):
            if raw_velocity_dump is not None:
                raw_velocity_dump.step_index = step_idx
                raw_velocity_dump.capture = step_idx in capture_steps
            denoised_video, denoised_audio = denoiser(transformer, video_state, audio_state, sigmas, step_idx)
            if raw_velocity_dump is not None:
                raw_velocity_dump.capture = False
            capture = step_idx in capture_steps
            video_state = _step_state_with_dump(
                recorder, stage, "video", step_idx, video_state, denoised_video, stepper, sigmas, capture
            )
            audio_state = _step_state_with_dump(
                recorder, stage, "audio", step_idx, audio_state, denoised_audio, stepper, sigmas, capture
            )

        recorder.state(f"{stage}/loop_final_patchified/video", video_state)
        recorder.state(f"{stage}/loop_final_patchified/audio", audio_state)
        return video_state, audio_state

    return loop


def _decoded_video_metadata(video: Iterator[torch.Tensor], recorder: Recorder) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for idx, chunk in enumerate(video):
        name = f"decoded_video/chunk_{idx:02d}"
        recorder.tensor(name, chunk, save=False)
        chunks.append(recorder.stats[name])
    return chunks


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    appdata = Path(args.appdata)
    models = appdata / "models"
    checkpoint = models / "ltx-2.3-22b-distilled.safetensors"
    upsampler = models / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    gemma = models / "gemma-3-12b-it-qat-q4_0-unquantized"

    device = torch.device(args.device)
    recorder = Recorder(Path(args.out_dir))
    t_start = time.perf_counter()

    fast = LTXFastVideoPipeline.create(
        checkpoint_path=str(checkpoint),
        gemma_root=str(gemma),
        upsampler_path=str(upsampler),
        device=device,
        streaming_prefetch_count=args.streaming_prefetch_count,
    )
    pipeline = fast.pipeline

    assert_resolution(height=args.height, width=args.width, is_two_stage=True)
    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
    noiser = DumpingGaussianNoiser(
        generator=generator,
        recorder=recorder,
        labels=["stage1/video", "stage1/audio", "stage2/video", "stage2/audio"],
    )

    (ctx_p,) = pipeline.prompt_encoder(
        [args.prompt],
        enhance_first_prompt=False,
        enhance_prompt_image=None,
        streaming_prefetch_count=args.streaming_prefetch_count,
    )
    video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding
    recorder.tensor("prompt/video_context", video_context)
    recorder.tensor("prompt/audio_context", audio_context)

    dtype = torch.bfloat16
    images: list[ImageConditioningInput] = []
    stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=pipeline.device)
    stage_1_w, stage_1_h = args.width // 2, args.height // 2
    stage_1_conditionings = pipeline.image_conditioner(
        lambda enc: combined_image_conditionings(
            images=images,
            height=stage_1_h,
            width=stage_1_w,
            video_encoder=enc,
            dtype=dtype,
            device=pipeline.device,
        )
    )

    video_state, audio_state = pipeline.stage(
        denoiser=SimpleDenoiser(video_context, audio_context),
        sigmas=stage_1_sigmas,
        noiser=noiser,
        width=stage_1_w,
        height=stage_1_h,
        frames=args.num_frames,
        fps=args.frame_rate,
        video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
        audio=ModalitySpec(context=audio_context),
        streaming_prefetch_count=args.streaming_prefetch_count,
        loop=dumping_euler_loop("stage1", recorder),
    )

    if video_state is None or audio_state is None:
        raise RuntimeError("creator phase dump expected both video and audio states")
    recorder.tensor("stage1/video_final_latent_unpatchified", video_state.latent)
    recorder.tensor("stage1/audio_final_latent_unpatchified", audio_state.latent)

    upscaled_video_latent = pipeline.upsampler(video_state.latent[:1])
    recorder.tensor("stage2/video_upscaled_latent", upscaled_video_latent)

    stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=pipeline.device)
    stage_2_conditionings = pipeline.image_conditioner(
        lambda enc: combined_image_conditionings(
            images=images,
            height=args.height,
            width=args.width,
            video_encoder=enc,
            dtype=dtype,
            device=pipeline.device,
        )
    )

    video_state, audio_state = pipeline.stage(
        denoiser=SimpleDenoiser(video_context, audio_context),
        sigmas=stage_2_sigmas,
        noiser=noiser,
        width=args.width,
        height=args.height,
        frames=args.num_frames,
        fps=args.frame_rate,
        video=ModalitySpec(
            context=video_context,
            conditionings=stage_2_conditionings,
            noise_scale=stage_2_sigmas[0].item(),
            initial_latent=upscaled_video_latent,
        ),
        audio=ModalitySpec(
            context=audio_context,
            noise_scale=stage_2_sigmas[0].item(),
            initial_latent=audio_state.latent,
        ),
        streaming_prefetch_count=args.streaming_prefetch_count,
        loop=dumping_euler_loop("stage2", recorder),
    )

    if video_state is None or audio_state is None:
        raise RuntimeError("creator phase dump expected both stage-2 video and audio states")
    recorder.tensor("stage2/video_final_latent_unpatchified", video_state.latent)
    recorder.tensor("stage2/audio_final_latent_unpatchified", audio_state.latent)

    tiling_config = default_tiling_config()
    video = pipeline.video_decoder(video_state.latent, tiling_config, generator)
    audio: Audio = pipeline.audio_decoder(audio_state.latent)
    decoded_chunks = _decoded_video_metadata(video, recorder)
    recorder.tensor("decoded_audio/waveform", audio.waveform)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device=device)
    elapsed = time.perf_counter() - t_start
    metadata = {
        "kind": "creator_desktop_fast_distilled_phase_dump",
        "prompt": args.prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "stage1_resolution": [stage_1_w, stage_1_h],
        "stage2_resolution": [args.width, args.height],
        "streaming_prefetch_count": args.streaming_prefetch_count,
        "checkpoint_path": str(checkpoint),
        "gemma_root": str(gemma),
        "upsampler_path": str(upsampler),
        "stage1_sigmas": [float(x) for x in stage_1_sigmas.detach().cpu().tolist()],
        "stage2_sigmas": [float(x) for x in stage_2_sigmas.detach().cpu().tolist()],
        "tiling_config": repr(tiling_config),
        "video_chunks_number": video_chunks_number(args.num_frames, tiling_config),
        "decoded_video_chunks": decoded_chunks,
        "decoded_audio_sampling_rate": audio.sampling_rate,
        "wall_time_seconds": elapsed,
        "torch_cuda_max_memory_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else None
        ),
        "torch_cuda_max_memory_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / (1024 * 1024) if torch.cuda.is_available() else None
        ),
    }
    recorder.flush(metadata)
    print(f"wrote creator phase dump: {Path(args.out_dir)}")
    print(f"  tensors: {len(recorder.tensors)}")
    print(f"  manifest: {Path(args.out_dir) / 'manifest.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appdata", default=str(DEFAULT_APPDATA))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--streaming-prefetch-count", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
