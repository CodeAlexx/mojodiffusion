#!/usr/bin/env python3
"""Instrumented reference run of the LTX-2 HQ two-stage pipeline (per-step dumps).

Mirrors ltx_pipelines/ti2vid_two_stages_hq.py (the LTX-2 runnable copy) but with
instrumentation so the Mojo HQ pipeline can be gated per-step on byte-identical
inputs (numeric-parity-testing discipline):
  - dumps gemma contexts (pos+neg, video+audio)
  - dumps init latents (or INJECTS them from --inject for the Mojo-parity pass)
  - dumps per-step latents (video+audio) for stage 1 and stage 2
  - records EVERY SDE noise tensor drawn (in consumption order) plus the
    stage-2 init noiser draws to noises.safetensors — the Mojo NoiseSource
    fixture contract:
        init_video / init_audio          stage-1 init latents (patchified)
        s2init_video / s2init_audio      raw stage-2 GaussianNoiser draws
        s{1,2}_sub{NN}_{video,audio}     substep SDE noises (post-normalize)
        s{1,2}_stp{NN}_{video,audio}     step-level SDE noises (post-normalize)
  - dumps upsampler input/output and final latents
  - decodes and writes frames

PER-STEP DUMP FIX: the old DumpingStepper keyed on (tag, step_index) but
Res2sDiffusionStep.step runs multiple times per step per modality (substep +
step injection, video + audio) so keys collided/overwrote. It is replaced by a
VERBATIM copy of `res2s_audio_video_denoising_loop`
(ltx_pipelines/utils/samplers.py:199-433, helpers :154-197) with explicit dump
hooks AFTER the step-level SDE injection per modality
(`s{stage}_s{step:02d}_{video,audio}`, recorded BF16-rounded like the state).

Consumer-GPU: OffloadMode.CPU (pinned RAM, layer streaming, ~5 GB VRAM).

Run (serenityflow venv):
  /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_hq_ref_run.py \
      --width 384 --height 256 --num-frames 17 --steps 15 --seed 42 \
      --contexts <ctx.safetensors> --out output/ltx2_hq_ref
"""
import argparse
import os
import sys
from dataclasses import replace
from functools import partial

sys.path.insert(0, "/home/alex/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/home/alex/LTX-2/packages/ltx-pipelines/src")

# ltx_pipelines.utils.media_io imports OpenImageIO (not installed); we never
# use media_io — stub it so the blocks/helpers import chain resolves.
import types
sys.modules.setdefault("OpenImageIO", types.ModuleType("OpenImageIO"))

import torch

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
# The DiT streamed by DiffusionStage: pre-dequantized BF16 export of the FP8
# checkpoint (scripts/ltx2_dequant_fp8_to_bf16.py — w_fp8*weight_scale->bf16,
# the exact numeric contract of the Mojo per-block dequant). The reference
# streaming loader cannot consume the scaled-mm-style FP8 export directly
# (.input_scale KeyError) and its fp8-cast policy re-downcasts to UNSCALED fp8
# (3.5%% relL2 weight error) — the bf16 export is the faithful oracle weighting.
DIT_CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
UPSAMPLER = "/home/alex/.serenity/models/ltx2_upscalers/ltx-2-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT = "/home/alex/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"
DISTILLED_LORA = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors"

# Guide-conformant (docs/LTX2_PROMPTING_GUIDE.md): one present-tense paragraph,
# shot -> scene -> action -> camera -> audio; dialogue IN QUOTES drives audio.
PROMPT = (
    "Medium shot of a woman climbing an industrial pegboard wall in a bright "
    "workshop, warm tungsten work lights overhead. She grips a rung, looks back "
    "over her shoulder at the camera, grinning, and says \"Almost there — watch "
    "this!\" in a clear, bright voice. The camera tracks upward with her as she "
    "pulls herself higher. Tools clink against the metal wall, and an upbeat "
    "workshop ambience of whirring machines hums beneath her laughter."
)
NEG_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, silence, mute"
)


def dump(d: dict, path: str) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().float().cpu().contiguous() for k, v in d.items()}, path)
    print("  [dump]", path, {k: tuple(v.shape) for k, v in d.items()})


# ---------------------------------------------------------------------------
# VERBATIM copy of res2s_audio_video_denoising_loop + _inject_sde_noise from
# ltx_pipelines/utils/samplers.py (loop :199-433, _inject_sde_noise :169-197,
# _get_new_noise :160-166) with TWO additions, clearly marked [HOOK]:
#   1. every SDE noise draw is recorded into `noise_log` under
#      f"{dump_prefix}_{kind}{step:02d}_{modality}" (kind in {sub, stp})
#   2. the per-modality state AFTER the step-level SDE injection is recorded
#      into `step_log` under f"{dump_prefix}_s{step:02d}_{modality}"
#      (BF16-rounded exactly like the state update `.to(model_dtype)`).
# ---------------------------------------------------------------------------
def instrumented_res2s_loop(  # noqa: PLR0913,PLR0915,PLR0912
    sigmas,
    video_state,
    audio_state,
    stepper,
    transformer,
    denoiser,
    noise_seed: int = -1,
    noise_seed_substep=None,
    eta: float = 0.5,
    bongmath: bool = True,
    bongmath_max_iter: int = 100,
    new_noise_fn=None,
    model_dtype=torch.bfloat16,
    legacy_mode: bool = True,
    *,
    dump_prefix: str = "s1",
    noise_log: dict | None = None,
    step_log: dict | None = None,
):
    from ltx_core.components.diffusion_steps import Res2sDiffusionStep
    from ltx_pipelines.utils.helpers import post_process_latent, timesteps_from_mask
    from ltx_pipelines.utils.res2s import get_res2s_coefficients
    from ltx_pipelines.utils.samplers import _get_new_noise

    if new_noise_fn is None:
        new_noise_fn = _get_new_noise
    if noise_log is None:
        noise_log = {}
    if step_log is None:
        step_log = {}

    # samplers.py:249-263
    present_state = video_state or audio_state
    if present_state is None:
        raise ValueError("At least one of video_state or audio_state must be provided")
    state_device = present_state.latent.device

    if noise_seed_substep is None:
        noise_seed_substep = noise_seed + 10000
    step_noise_generator = torch.Generator(device=state_device).manual_seed(noise_seed)
    substep_noise_generator = torch.Generator(device=state_device).manual_seed(noise_seed_substep)

    # samplers.py:169-197 (_inject_sde_noise) with the [HOOK] noise recording.
    def _inject_sde_noise_recorded(
        key: str,
        state,
        sample,
        denoised_sample,
        step_noise_generator,
        sigmas,
        step_idx,
        eta,
    ):
        sigmas_copy = sigmas.clone()
        new_noise = new_noise_fn(state.latent, step_noise_generator)
        noise_log[key] = new_noise.detach().float().cpu()  # [HOOK]
        if not legacy_mode:
            timesteps = timesteps_from_mask(state.denoise_mask.double(), sigmas_copy[step_idx].double())
            next_timesteps = timesteps_from_mask(state.denoise_mask.double(), sigmas_copy[step_idx + 1].double())
            sigmas = torch.stack([timesteps, next_timesteps])
            step_idx = 0
        x_next = stepper.step(
            sample=sample,
            denoised_sample=denoised_sample,
            sigmas=sigmas,
            step_index=step_idx,
            noise=new_noise,
            eta=eta,
        )
        if legacy_mode:
            x_next = post_process_latent(x_next, state.denoise_mask, state.clean_latent)
        return x_next

    substep_noise_injecting_fn = partial(
        _inject_sde_noise_recorded, step_noise_generator=substep_noise_generator, eta=0.5
    )
    step_noise_injecting_fn = partial(
        _inject_sde_noise_recorded, step_noise_generator=step_noise_generator, eta=eta
    )

    if not isinstance(stepper, Res2sDiffusionStep):
        raise ValueError("stepper must be an instance of Res2sDiffusionStep")

    # samplers.py:265-276
    n_full_steps = len(sigmas) - 1
    if sigmas[-1] == 0:
        sigmas = torch.cat([sigmas[:-1], torch.tensor([0.0011, 0.0], device=sigmas.device)], dim=0)
    hs = -torch.log(sigmas[1:].double().cpu() / (sigmas[:-1].double().cpu()))

    phi_cache = {}
    c2 = 0.5

    for step_idx in range(n_full_steps):
        sigma = sigmas[step_idx].double()
        sigma_next = sigmas[step_idx + 1].double()

        x_anchor_video = video_state.latent.clone().double() if video_state is not None else None
        x_anchor_audio = audio_state.latent.clone().double() if audio_state is not None else None

        # STAGE 1 (samplers.py:286-297)
        video_result, audio_result = denoiser(transformer, video_state, audio_state, sigmas, step_idx)
        denoised_video_1 = video_result.denoised if video_result is not None else None
        denoised_audio_1 = audio_result.denoised if audio_result is not None else None
        if video_state is not None and denoised_video_1 is not None:
            denoised_video_1 = post_process_latent(denoised_video_1, video_state.denoise_mask, video_state.clean_latent)
        if audio_state is not None and denoised_audio_1 is not None:
            denoised_audio_1 = post_process_latent(denoised_audio_1, audio_state.denoise_mask, audio_state.clean_latent)

        h = hs[step_idx].item()
        a21, b1, b2 = get_res2s_coefficients(h, phi_cache, c2)
        sub_sigma = torch.sqrt(sigma * sigma_next)

        # substep x (samplers.py:303-320)
        if x_anchor_video is not None and denoised_video_1 is not None:
            eps_1_video = denoised_video_1.double() - x_anchor_video
            x_mid_video = x_anchor_video.double() + h * a21 * eps_1_video
        else:
            eps_1_video = None
            x_mid_video = None
        if x_anchor_audio is not None and denoised_audio_1 is not None:
            eps_1_audio = denoised_audio_1.double() - x_anchor_audio
            x_mid_audio = x_anchor_audio.double() + h * a21 * eps_1_audio
        else:
            eps_1_audio = None
            x_mid_audio = None

        # SDE noise injection at substep (samplers.py:322-340)
        if x_mid_video is not None and video_state is not None:
            x_mid_video = substep_noise_injecting_fn(
                f"{dump_prefix}_sub{step_idx:02d}_video",  # [HOOK]
                state=video_state,
                sample=x_anchor_video,
                denoised_sample=x_mid_video,
                sigmas=torch.stack([sigma, sub_sigma]),
                step_idx=0,
            )
        if x_mid_audio is not None and audio_state is not None:
            x_mid_audio = substep_noise_injecting_fn(
                f"{dump_prefix}_sub{step_idx:02d}_audio",  # [HOOK]
                state=audio_state,
                sample=x_anchor_audio,
                denoised_sample=x_mid_audio,
                sigmas=torch.stack([sigma, sub_sigma]),
                step_idx=0,
            )

        # bong iteration (samplers.py:342-352)
        if bongmath and h < 0.5 and sigma > 0.03:
            for _ in range(bongmath_max_iter):
                if x_mid_video is not None and eps_1_video is not None:
                    x_anchor_video = x_mid_video - h * a21 * eps_1_video
                    eps_1_video = denoised_video_1.double() - x_anchor_video
                if x_mid_audio is not None and eps_1_audio is not None:
                    x_anchor_audio = x_mid_audio - h * a21 * eps_1_audio
                    eps_1_audio = denoised_audio_1.double() - x_anchor_audio

        # STAGE 2 (samplers.py:354-379)
        mid_video_state = (
            replace(video_state, latent=x_mid_video.to(model_dtype))
            if video_state is not None and x_mid_video is not None
            else None
        )
        mid_audio_state = (
            replace(audio_state, latent=x_mid_audio.to(model_dtype))
            if audio_state is not None and x_mid_audio is not None
            else None
        )
        video_result_2, audio_result_2 = denoiser(
            transformer,
            video_state=mid_video_state,
            audio_state=mid_audio_state,
            sigmas=torch.stack([sub_sigma]).to(sigmas.device),
            step_index=0,
        )
        denoised_video_2 = video_result_2.denoised if video_result_2 is not None else None
        denoised_audio_2 = audio_result_2.denoised if audio_result_2 is not None else None
        if video_state is not None and denoised_video_2 is not None:
            denoised_video_2 = post_process_latent(denoised_video_2, video_state.denoise_mask, video_state.clean_latent)
        if audio_state is not None and denoised_audio_2 is not None:
            denoised_audio_2 = post_process_latent(denoised_audio_2, audio_state.denoise_mask, audio_state.clean_latent)

        # FINAL COMBINATION (samplers.py:381-394)
        if x_anchor_video is not None and eps_1_video is not None and denoised_video_2 is not None:
            eps_2_video = denoised_video_2.double() - x_anchor_video
            x_next_video = x_anchor_video + h * (b1 * eps_1_video + b2 * eps_2_video)
        else:
            x_next_video = None
        if x_anchor_audio is not None and eps_1_audio is not None and denoised_audio_2 is not None:
            eps_2_audio = denoised_audio_2.double() - x_anchor_audio
            x_next_audio = x_anchor_audio + h * (b1 * eps_1_audio + b2 * eps_2_audio)
        else:
            x_next_audio = None

        # SDE NOISE INJECTION AT STEP LEVEL (samplers.py:396-413)
        if x_next_video is not None and video_state is not None:
            x_next_video = step_noise_injecting_fn(
                f"{dump_prefix}_stp{step_idx:02d}_video",  # [HOOK]
                state=video_state,
                sample=x_anchor_video,
                denoised_sample=x_next_video,
                sigmas=sigmas,
                step_idx=step_idx,
            )
        if x_next_audio is not None and audio_state is not None:
            x_next_audio = step_noise_injecting_fn(
                f"{dump_prefix}_stp{step_idx:02d}_audio",  # [HOOK]
                state=audio_state,
                sample=x_anchor_audio,
                denoised_sample=x_next_audio,
                sigmas=sigmas,
                step_idx=step_idx,
            )

        # state update (samplers.py:415-419) + [HOOK] per-step dump
        if video_state is not None and x_next_video is not None:
            video_state = replace(video_state, latent=x_next_video.to(model_dtype))
            step_log[f"{dump_prefix}_s{step_idx:02d}_video"] = (
                video_state.latent.detach().float().cpu()
            )
        if audio_state is not None and x_next_audio is not None:
            audio_state = replace(audio_state, latent=x_next_audio.to(model_dtype))
            step_log[f"{dump_prefix}_s{step_idx:02d}_audio"] = (
                audio_state.latent.detach().float().cpu()
            )
        print(f"  [{dump_prefix}] step {step_idx + 1}/{n_full_steps} "
              f"sigma={float(sigma):.6f}->{float(sigma_next):.6f} h={h:.6f}",
              flush=True)

    # final denoise pass (samplers.py:421-433)
    if sigmas[-1] == 0:
        video_result_final, audio_result_final = denoiser(transformer, video_state, audio_state, sigmas, n_full_steps)
        denoised_video_1 = video_result_final.denoised if video_result_final is not None else None
        denoised_audio_1 = audio_result_final.denoised if audio_result_final is not None else None
        if video_state is not None and denoised_video_1 is not None:
            denoised_video_1 = post_process_latent(denoised_video_1, video_state.denoise_mask, video_state.clean_latent)
            video_state = replace(video_state, latent=denoised_video_1.to(model_dtype))
        if audio_state is not None and denoised_audio_1 is not None:
            denoised_audio_1 = post_process_latent(denoised_audio_1, audio_state.denoise_mask, audio_state.clean_latent)
            audio_state = replace(audio_state, latent=denoised_audio_1.to(model_dtype))

    return video_state, audio_state


class RecordingNoiser:
    """GaussianNoiser (ltx_core/components/noisers.py) with init-draw recording
    and --inject support.

    Each call pops the next key from `key_queue` (call order is fixed:
    stage-1 video, stage-1 audio, stage-2 video, stage-2 audio). The raw randn
    draw is recorded into `noise_log[key]`; when `inject` holds the key, the
    injected tensor REPLACES the draw (the generator is still advanced so a
    partial inject keeps later draws aligned)."""

    def __init__(self, generator, key_queue, noise_log, inject=None):
        self.generator = generator
        self.key_queue = list(key_queue)
        self.noise_log = noise_log
        self.inject = inject or {}

    def __call__(self, latent_state, noise_scale: float = 1.0):
        key = self.key_queue.pop(0)
        noise = torch.randn(
            *latent_state.latent.shape,
            device=latent_state.latent.device,
            dtype=latent_state.latent.dtype,
            generator=self.generator,
        )
        if key in self.inject:
            noise = self.inject[key].to(latent_state.latent.device, latent_state.latent.dtype)
        self.noise_log[key] = noise.detach().float().cpu()
        scaled_mask = latent_state.denoise_mask * noise_scale
        latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)
        return replace(latent_state, latent=latent.to(latent_state.latent.dtype))


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--num-frames", type=int, default=33)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--neg", default=NEG_PROMPT)
    ap.add_argument("--out", default="output/ltx2_hq_ref")
    ap.add_argument("--inject", default="",
                    help="safetensors with init_video/init_audio (and optionally "
                         "s2init_*) noise tensors to INJECT in place of the "
                         "noiser draws (Mojo-parity pass)")
    ap.add_argument("--contexts", default="",
                    help="safetensors with video_context/audio_context (+ optional "
                         "neg_*) — skips the Gemma encode (parity passes use the "
                         "same canned contexts the Mojo pipeline reads)")
    ap.add_argument("--skip-decode", action="store_true",
                    help="stop after final_latents (math-gate runs)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import ltx_core
    assert "/LTX-2/" in ltx_core.__file__

    from ltx_core.components.diffusion_steps import Res2sDiffusionStep
    from ltx_core.components.guiders import MultiModalGuider
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.types import VideoLatentShape, VideoPixelShape
    from ltx_pipelines.utils.blocks import (
        AudioDecoder, DiffusionStage, PromptEncoder, VideoDecoder, VideoUpsampler,
    )
    from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS, STAGE_2_DISTILLED_SIGMAS
    from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
    from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    # DISK offload: the bf16 DiT export is 42 GB — pinned-CPU (OffloadMode.CPU)
    # would not fit in 62 GB RAM; DISK re-reads from page cache per pass.
    offload = OffloadMode.DISK
    quantization = None

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    noise_log: dict = {}
    inject = {}
    if args.inject:
        from safetensors.torch import load_file
        inject = load_file(args.inject)
        print("[inject]", sorted(inject.keys()))
    noiser = RecordingNoiser(
        gen,
        ["init_video", "init_audio", "s2init_video", "s2init_audio"],
        noise_log,
        inject,
    )
    stepper = Res2sDiffusionStep()
    scheduler = LTX2Scheduler()

    if args.contexts:
        from safetensors.torch import load_file

        print("[1/6] contexts INJECTED from", args.contexts)
        cf = load_file(args.contexts)
        v_p = cf["video_context"].to(dev, dtype)
        a_p = cf["audio_context"].to(dev, dtype)
        v_n = cf.get("neg_video_context")
        a_n = cf.get("neg_audio_context")
        v_n = v_n.to(dev, dtype) if v_n is not None else torch.zeros_like(v_p)
        a_n = a_n.to(dev, dtype) if a_n is not None else torch.zeros_like(a_p)
        print("  video_context", tuple(v_p.shape), "audio_context", tuple(a_p.shape))
        # The canned dump holds PRE-connector features (the
        # feature_extract_and_project output). The transformer consumes
        # POST-connector embeddings — PromptEncoder applies the
        # Embeddings1DConnectors (embeddings_processor.py), which the bare
        # --contexts inject skipped. Apply them here, with an ALL-VALID
        # attention mask (the Mojo connector port treats all 1024 tokens as
        # valid; for the MATH gate both sides must use the same contract —
        # with an all-ones mask the right-pad reorder is identity and the
        # output binary mask is all-ones).
        from ltx_core.loader.single_gpu_model_builder import (
            SingleGPUModelBuilder as Builder,
        )
        from ltx_core.loader.registry import DummyRegistry
        from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
            EMBEDDINGS_PROCESSOR_KEY_OPS,
            EmbeddingsProcessorConfigurator,
        )
        from ltx_core.text_encoders.gemma.embeddings_processor import (
            convert_to_additive_mask,
        )

        ep = Builder(
            model_path=CKPT,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            registry=DummyRegistry(),
        ).build(device=dev, dtype=dtype).eval()
        ones_mask = torch.ones(v_p.shape[0], v_p.shape[1], device=dev, dtype=torch.int64)
        add_mask = convert_to_additive_mask(ones_mask, v_p.dtype)
        v_p, a_p, _ = ep.create_embeddings(v_p, a_p, add_mask)
        v_n, a_n, _ = ep.create_embeddings(v_n, a_n, add_mask)
        del ep
        print("  [connector] applied -> video", tuple(v_p.shape),
              "audio", tuple(a_p.shape))
    else:
        print("[1/6] gemma prompt encode (pos+neg)")
        pe = PromptEncoder(CKPT, GEMMA_ROOT, dtype, dev, offload_mode=offload)
        ctx_p, ctx_n = pe([args.prompt, args.neg])
        v_p, a_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_n, a_n = ctx_n.video_encoding, ctx_n.audio_encoding
        del pe
    dump(
        {"video_context": v_p, "audio_context": a_p,
         "neg_video_context": v_n, "neg_audio_context": a_n},
        os.path.join(args.out, "contexts.safetensors"),
    )

    s1_shape = VideoPixelShape(
        batch=1, frames=args.num_frames,
        width=args.width // 2, height=args.height // 2, fps=25.0,
    )
    empty = torch.empty(VideoLatentShape.from_pixel_shape(s1_shape).to_torch_shape())
    sigmas1 = scheduler.execute(latent=empty, steps=args.steps).to(torch.float32, copy=True).to(dev)
    sigmas2 = STAGE_2_DISTILLED_SIGMAS.to(torch.float32).to(dev)
    print("  stage1 sigmas:", [round(float(s), 6) for s in sigmas1])
    print("  stage2 sigmas:", [round(float(s), 6) for s in sigmas2])

    step_log: dict = {}

    vparams = LTX_2_3_HQ_PARAMS.video_guider_params
    aparams = LTX_2_3_HQ_PARAMS.audio_guider_params
    print("[2/6] stage 1 (guided res_2s,", args.steps, "steps,",
          s1_shape.width, "x", s1_shape.height, ")")
    stage_1 = DiffusionStage(
        DIT_CKPT, dtype, dev,
        loras=(LoraPathStrengthAndSDOps(DISTILLED_LORA, 0.25, None),),
        quantization=quantization,
        offload_mode=offload,
    )
    video_state, audio_state = stage_1(
        denoiser=GuidedDenoiser(
            v_context=v_p, a_context=a_p,
            video_guider=MultiModalGuider(params=vparams, negative_context=v_n),
            audio_guider=MultiModalGuider(params=aparams, negative_context=a_n),
        ),
        sigmas=sigmas1, noiser=noiser, stepper=stepper,
        width=s1_shape.width, height=s1_shape.height,
        frames=args.num_frames, fps=25.0,
        video=ModalitySpec(context=v_p),
        audio=ModalitySpec(context=a_p),
        loop=partial(instrumented_res2s_loop, dump_prefix="s1",
                     noise_log=noise_log, step_log=step_log),
    )
    dump({k: v for k, v in step_log.items() if k.startswith("s1_")},
         os.path.join(args.out, "stage1_steps.safetensors"))
    dump({"video": video_state.latent, "audio": audio_state.latent},
         os.path.join(args.out, "stage1_final.safetensors"))
    del stage_1

    print("[3/6] spatial upsampler x2")
    ups = VideoUpsampler(CKPT, UPSAMPLER, dtype, dev)
    up_latent = ups(video_state.latent[:1])
    dump({"in": video_state.latent, "out": up_latent},
         os.path.join(args.out, "upsampler.safetensors"))
    del ups

    print("[4/6] stage 2 (simple res_2s, 3 steps,", args.width, "x", args.height, ")")
    stage_2 = DiffusionStage(
        DIT_CKPT, dtype, dev,
        loras=(LoraPathStrengthAndSDOps(DISTILLED_LORA, 0.5, None),),
        quantization=quantization,
        offload_mode=offload,
    )
    video_state, audio_state = stage_2(
        denoiser=SimpleDenoiser(v_context=v_p, a_context=a_p),
        sigmas=sigmas2, noiser=noiser, stepper=stepper,
        width=args.width, height=args.height,
        frames=args.num_frames, fps=25.0,
        video=ModalitySpec(context=v_p, noise_scale=sigmas2[0].item(),
                           initial_latent=up_latent),
        audio=ModalitySpec(context=a_p, noise_scale=sigmas2[0].item(),
                           initial_latent=audio_state.latent),
        loop=partial(instrumented_res2s_loop, dump_prefix="s2",
                     noise_log=noise_log, step_log=step_log),
    )
    dump({k: v for k, v in step_log.items() if k.startswith("s2_")},
         os.path.join(args.out, "stage2_steps.safetensors"))
    dump({"video": video_state.latent, "audio": audio_state.latent},
         os.path.join(args.out, "final_latents.safetensors"))
    del stage_2

    dump(noise_log, os.path.join(args.out, "noises.safetensors"))

    if args.skip_decode:
        print("DONE (skip-decode) ->", args.out)
        return

    print("[5/6] decode video")
    vd = VideoDecoder(CKPT, dtype, dev)
    chunks = [c for c in vd(video_state.latent, None, gen)]
    video = torch.cat(chunks, dim=0)
    from PIL import Image
    import numpy as np
    v8 = (video.float().clamp(0, 1) * 255).round().byte().cpu().numpy()
    for f in range(0, v8.shape[0], max(1, v8.shape[0] // 12)):
        Image.fromarray(v8[f]).save(os.path.join(args.out, f"ref_frame{f:03d}.png"))
    print("  frames:", v8.shape)

    print("[6/6] decode audio")
    ad = AudioDecoder(CKPT, dtype, dev)
    audio = ad(audio_state.latent)
    wav = audio.waveform.float().cpu()
    print("  audio:", tuple(wav.shape), "rms",
          wav.pow(2).mean().sqrt().item())
    import wave as wavemod
    w = wavemod.open(os.path.join(args.out, "ref_audio.wav"), "wb")
    w.setnchannels(wav.shape[0] if wav.dim() == 2 else 2)
    w.setsampwidth(2)
    w.setframerate(48000)
    pcm = (wav.clamp(-1, 1) * 32767).short().numpy()
    w.writeframes(pcm.T.tobytes() if wav.dim() == 2 else pcm.tobytes())
    w.close()
    print("DONE ->", args.out)


if __name__ == "__main__":
    main()
