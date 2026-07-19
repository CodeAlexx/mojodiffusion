#!/usr/bin/env python3
"""Resume an interrupted ltx2_hq_ref_run.py at STAGE 2 from its dumped
artifacts (contexts.safetensors [POST-connector], upsampler.safetensors [out],
stage1_final.safetensors [audio]) and finish: stage-2 denoise + video/audio
decode + wav.

HONEST DEVIATION vs an uninterrupted run: the stage-2 GaussianNoiser draws come
from a FRESH seeded generator (the original generator's state after the stage-1
init draws is not recoverable) — a different-but-valid trajectory, fine for the
recipe/quality verdict, NOT for byte parity.
"""
import argparse
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ltx2_hq_ref_run import (  # noqa: E402  (also sets LTX-2 sys.paths + OIIO stub)
    DIT_CKPT, CKPT, DISTILLED_LORA, RecordingNoiser, dump, instrumented_res2s_loop,
)

import torch  # noqa: E402


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1088)
    ap.add_argument("--num-frames", type=int, default=241)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--no-lora", action="store_true")
    args = ap.parse_args()
    d = args.run_dir

    from safetensors.torch import load_file
    from ltx_core.components.diffusion_steps import Res2sDiffusionStep
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.utils.blocks import AudioDecoder, DiffusionStage, VideoDecoder
    from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMAS
    from ltx_pipelines.utils.denoisers import SimpleDenoiser
    from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

    dev = torch.device("cuda")
    dtype = torch.bfloat16
    offload = OffloadMode.DISK

    ctxs = load_file(os.path.join(d, "contexts.safetensors"))
    v_p = ctxs["video_context"].to(dev, dtype)
    a_p = ctxs["audio_context"].to(dev, dtype)
    up = load_file(os.path.join(d, "upsampler.safetensors"))["out"].to(dev, dtype)
    a1 = load_file(os.path.join(d, "stage1_final.safetensors"))["audio"].to(dev, dtype)
    print("[resume] contexts", tuple(v_p.shape), "up_latent", tuple(up.shape),
          "audio_latent", tuple(a1.shape))

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    noise_log: dict = {}
    step_log: dict = {}
    noiser = RecordingNoiser(gen, ["s2init_video", "s2init_audio"], noise_log)
    stepper = Res2sDiffusionStep()
    sigmas2 = STAGE_2_DISTILLED_SIGMAS.to(torch.float32).to(dev)

    print("[resume] stage 2 (simple res_2s, 3 steps,", args.width, "x", args.height, ")")
    stage_2 = DiffusionStage(
        DIT_CKPT, dtype, dev,
        loras=() if args.no_lora else
              (LoraPathStrengthAndSDOps(DISTILLED_LORA, 0.5, LTXV_LORA_COMFY_RENAMING_MAP),),
        quantization=None,
        offload_mode=offload,
    )
    video_state, audio_state = stage_2(
        denoiser=SimpleDenoiser(v_context=v_p, a_context=a_p),
        sigmas=sigmas2, noiser=noiser, stepper=stepper,
        width=args.width, height=args.height,
        frames=args.num_frames, fps=args.fps,
        video=ModalitySpec(context=v_p, noise_scale=sigmas2[0].item(),
                           initial_latent=up),
        audio=ModalitySpec(context=a_p, noise_scale=sigmas2[0].item(),
                           initial_latent=a1),
        loop=partial(instrumented_res2s_loop, dump_prefix="s2",
                     noise_log=noise_log, step_log=step_log),
    )
    dump({"video": video_state.latent, "audio": audio_state.latent},
         os.path.join(d, "final_latents.safetensors"))
    del stage_2

    print("[resume] decode video")
    vd = VideoDecoder(CKPT, dtype, dev)
    chunks = [c for c in vd(video_state.latent, None, gen)]
    video = torch.cat(chunks, dim=0)
    from PIL import Image
    v8 = (video.float().clamp(0, 1) * 255).round().byte().cpu().numpy()
    for f in range(v8.shape[0]):
        Image.fromarray(v8[f]).save(os.path.join(d, f"ref_frame{f:03d}.png"))
    print("  frames:", v8.shape)
    del vd, video

    print("[resume] decode audio")
    ad = AudioDecoder(CKPT, dtype, dev)
    audio = ad(audio_state.latent)
    wav = audio.waveform.float().cpu()
    print("  audio:", tuple(wav.shape), "rms", wav.pow(2).mean().sqrt().item())
    import wave as wavemod
    w = wavemod.open(os.path.join(d, "ref_audio.wav"), "wb")
    w.setnchannels(wav.shape[0] if wav.dim() == 2 else 2)
    w.setsampwidth(2)
    w.setframerate(48000)
    pcm = (wav.clamp(-1, 1) * 32767).short().numpy()
    w.writeframes(pcm.T.tobytes() if wav.dim() == 2 else pcm.tobytes())
    w.close()

    fps_i = int(round(args.fps))
    os.system(
        f"ffmpeg -y -v error -framerate {fps_i} -i {d}/ref_frame%03d.png "
        f"-i {d}/ref_audio.wav -c:v libx264 -pix_fmt yuv420p -af apad -c:a aac "
        f"-shortest -movflags +faststart {d}/ltx2_oracle.mp4"
    )
    print("DONE ->", d, "/ltx2_oracle.mp4")


if __name__ == "__main__":
    main()
