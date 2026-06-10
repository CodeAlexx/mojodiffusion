#!/usr/bin/env python3
"""Instrumented reference run of the LTX-2 HQ two-stage pipeline (per-step dumps).

Mirrors ltx_pipelines/ti2vid_two_stages_hq.py (the LTX-2 runnable copy) but with
instrumentation so the Mojo HQ pipeline can be gated per-step on byte-identical
inputs (numeric-parity-testing discipline):
  - dumps gemma contexts (pos+neg, video+audio)
  - dumps init latents (or INJECTS them from --inject for the Mojo-parity pass)
  - dumps per-step latents (video+audio) for stage 1 and stage 2
  - dumps upsampler input/output and final latents
  - decodes and writes frames

Consumer-GPU: OffloadMode.CPU (pinned RAM, layer streaming, ~5 GB VRAM).

Run (serenityflow venv):
  /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_hq_ref_run.py \
      --width 768 --height 512 --num-frames 33 --steps 15 --seed 42 \
      --out output/ltx2_hq_ref
"""
import argparse
import os
import sys

sys.path.insert(0, "/home/alex/LTX-2/packages/ltx-core/src")
sys.path.insert(0, "/home/alex/LTX-2/packages/ltx-pipelines/src")

# ltx_pipelines.utils.media_io imports OpenImageIO (not installed); we never
# use media_io — stub it so the blocks/helpers import chain resolves.
import types
sys.modules.setdefault("OpenImageIO", types.ModuleType("OpenImageIO"))

import torch

CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
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
                    help="safetensors with init_video/init_audio latents (Mojo-parity pass)")
    ap.add_argument("--contexts", default="",
                    help="safetensors with video_context/audio_context (+ optional "
                         "neg_*) — skips the Gemma encode (parity passes use the "
                         "same canned contexts the Mojo pipeline reads)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import ltx_core
    assert "/LTX-2/" in ltx_core.__file__

    from ltx_core.components.diffusion_steps import Res2sDiffusionStep
    from ltx_core.components.guiders import MultiModalGuider
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.types import VideoLatentShape, VideoPixelShape
    from ltx_pipelines.utils.blocks import (
        AudioDecoder, DiffusionStage, PromptEncoder, VideoDecoder, VideoUpsampler,
    )
    from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS, STAGE_2_DISTILLED_SIGMAS
    from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
    from ltx_pipelines.utils.samplers import res2s_audio_video_denoising_loop
    from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    offload = OffloadMode.CPU

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=gen)
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

    # per-step dump hook: wrap the stepper
    step_log: list[tuple[str, torch.Tensor]] = []

    class DumpingStepper(Res2sDiffusionStep):
        def __init__(self, tag: str):
            self.tag = tag

        def step(self, sample, denoised_sample, sigmas, step_index, noise, eta=0.5):
            out = super().step(sample, denoised_sample, sigmas, step_index, noise, eta)
            step_log.append((f"{self.tag}_step{step_index:02d}", out))
            return out

    vparams = LTX_2_3_HQ_PARAMS.video_guider_params
    aparams = LTX_2_3_HQ_PARAMS.audio_guider_params
    print("[2/6] stage 1 (guided res_2s,", args.steps, "steps,",
          s1_shape.width, "x", s1_shape.height, ")")
    stage_1 = DiffusionStage(
        CKPT, dtype, dev,
        loras=(LoraPathStrengthAndSDOps(DISTILLED_LORA, 0.25, None),),
        offload_mode=offload,
    )
    s1_stepper = DumpingStepper("s1")
    video_state, audio_state = stage_1(
        denoiser=GuidedDenoiser(
            v_context=v_p, a_context=a_p,
            video_guider=MultiModalGuider(params=vparams, negative_context=v_n),
            audio_guider=MultiModalGuider(params=aparams, negative_context=a_n),
        ),
        sigmas=sigmas1, noiser=noiser, stepper=s1_stepper,
        width=s1_shape.width, height=s1_shape.height,
        frames=args.num_frames, fps=25.0,
        video=ModalitySpec(context=v_p),
        audio=ModalitySpec(context=a_p),
        loop=res2s_audio_video_denoising_loop,
    )
    dump(dict((k, v) for k, v in step_log), os.path.join(args.out, "stage1_steps.safetensors"))
    dump({"video": video_state.latent, "audio": audio_state.latent},
         os.path.join(args.out, "stage1_final.safetensors"))
    step_log.clear()
    del stage_1

    print("[3/6] spatial upsampler x2")
    ups = VideoUpsampler(CKPT, UPSAMPLER, dtype, dev)
    up_latent = ups(video_state.latent[:1])
    dump({"in": video_state.latent, "out": up_latent},
         os.path.join(args.out, "upsampler.safetensors"))
    del ups

    print("[4/6] stage 2 (simple res_2s, 3 steps,", args.width, "x", args.height, ")")
    stage_2 = DiffusionStage(
        CKPT, dtype, dev,
        loras=(LoraPathStrengthAndSDOps(DISTILLED_LORA, 0.5, None),),
        offload_mode=offload,
    )
    s2_stepper = DumpingStepper("s2")
    video_state, audio_state = stage_2(
        denoiser=SimpleDenoiser(v_context=v_p, a_context=a_p),
        sigmas=sigmas2, noiser=noiser, stepper=s2_stepper,
        width=args.width, height=args.height,
        frames=args.num_frames, fps=25.0,
        video=ModalitySpec(context=v_p, noise_scale=sigmas2[0].item(),
                           initial_latent=up_latent),
        audio=ModalitySpec(context=a_p, noise_scale=sigmas2[0].item(),
                           initial_latent=audio_state.latent),
        loop=res2s_audio_video_denoising_loop,
    )
    dump(dict((k, v) for k, v in step_log), os.path.join(args.out, "stage2_steps.safetensors"))
    dump({"video": video_state.latent, "audio": audio_state.latent},
         os.path.join(args.out, "final_latents.safetensors"))
    del stage_2

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
