#!/usr/bin/env python3
"""LTX-2 HQ reference STEP-1 EVAL-1 instrumented dump (divergence hunt).

Runs the SAME setup as scripts/ltx2_hq_ref_run.py's golden pass (dequant-bf16
DiT, distilled LoRA 0.25, injected golden init latents + post-connector golden
contexts) but stops after the FIRST guided-denoiser eval of stage 1
(sigma = 1.0) and dumps every pipeline-built input plus intermediate hidden
states:
  - per-pass (cond/uncond/mod) TransformerArgs for video+audio:
      x (post patchify_proj), timesteps (temb 9*dim), embedded_timestep,
      pe cos/sin, cross-pe cos/sin, cross_scale_shift_timestep,
      cross_gate_timestep, prompt_timestep, context
  - per-block video/audio hidden states after blocks 0,1,2,8,24,47 (per pass)
  - final velocities per pass, x0 (denoised) per pass, guider outputs

Out: safetensors at OUT.
"""
import os
import subprocess
import sys
from functools import partial

CREATOR_ROOT = os.environ.get("LTX2_CREATOR_ROOT", "/home/alex/LTX-2")
CREATOR_REVISION = "780984275fd47128b02bef9b5c085404276866ee"
sys.path.insert(0, f"{CREATOR_ROOT}/packages/ltx-core/src")
sys.path.insert(0, f"{CREATOR_ROOT}/packages/ltx-pipelines/src")
import types
sys.modules.setdefault("OpenImageIO", types.ModuleType("OpenImageIO"))

import torch
from safetensors.torch import load_file, save_file

GOLD = os.environ.get(
    "LTX2_REF_GOLD",
    "/home/alex/mojodiffusion/output/ltx2_hq_ref_golden",
)
DIT_CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
DISTILLED_LORA = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-lora-384.safetensors"
# --lora-sdops: pass the OFFICIAL LTXV_LORA_COMFY_RENAMING_MAP so the LoRA
# actually fuses (the golden run's sd_ops=None silently dropped ALL LoRA —
# the 2026-07-10 divergence-hunt finding). Default: None (golden contract).
USE_LORA_SDOPS = "--lora-sdops" in sys.argv
OUT = os.path.join(
    os.environ.get("LTX2_REF_OUT_DIR", GOLD),
    "step1_ref_dump_lora.safetensors" if USE_LORA_SDOPS else "step1_ref_dump.safetensors",
)

WIDTH = int(os.environ.get("LTX2_REF_WIDTH", "768"))
HEIGHT = int(os.environ.get("LTX2_REF_HEIGHT", "512"))
FRAMES = int(os.environ.get("LTX2_REF_FRAMES", "17"))
STEPS = int(os.environ.get("LTX2_REF_STEPS", "15"))
FPS = float(os.environ.get("LTX2_REF_FPS", "25.0"))
NO_LORA = os.environ.get("LTX2_REF_NO_LORA", "0") == "1"
DUMP_BLOCKS = (0, 1, 2, 8, 24, 47)


def assert_creator_revision() -> None:
    head = subprocess.run(
        ["git", "-C", CREATOR_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", CREATOR_ROOT, "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != CREATOR_REVISION or status:
        raise SystemExit(
            f"Creator oracle must be clean at {CREATOR_REVISION}; "
            f"found head={head}, dirty={bool(status)} in {CREATOR_ROOT}"
        )


@torch.inference_mode()
def main() -> None:
    assert_creator_revision()
    from ltx_core.components.diffusion_steps import Res2sDiffusionStep
    from ltx_core.components.guiders import MultiModalGuider
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.types import VideoLatentShape, VideoPixelShape
    from ltx_pipelines.utils.blocks import DiffusionStage
    from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS
    from ltx_pipelines.utils.denoisers import GuidedDenoiser
    from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

    dev = torch.device("cuda")
    dtype = torch.bfloat16

    cf = load_file(os.path.join(GOLD, "contexts.safetensors"))
    v_p = cf["video_context"].to(dev, dtype)
    a_p = cf["audio_context"].to(dev, dtype)
    v_n = cf["neg_video_context"].to(dev, dtype)
    a_n = cf["neg_audio_context"].to(dev, dtype)
    print("[ctx] post-connector golden contexts", tuple(v_p.shape), tuple(a_p.shape))

    noises = load_file(os.path.join(GOLD, "noises.safetensors"))
    inject = {k: noises[k] for k in ("init_video", "init_audio")}
    print("[inject]", {k: tuple(v.shape) for k, v in inject.items()})

    class InjectNoiser:
        def __init__(self):
            self.queue = ["init_video", "init_audio"]

        def __call__(self, latent_state, noise_scale: float = 1.0):
            from dataclasses import replace
            key = self.queue.pop(0)
            noise = inject[key].to(latent_state.latent.device, latent_state.latent.dtype)
            assert noise.shape == latent_state.latent.shape, (key, noise.shape, latent_state.latent.shape)
            scaled_mask = latent_state.denoise_mask * noise_scale
            latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)
            return replace(latent_state, latent=latent.to(latent_state.latent.dtype))

    s1_shape = VideoPixelShape(batch=1, frames=FRAMES, width=WIDTH // 2, height=HEIGHT // 2, fps=FPS)
    empty = torch.empty(VideoLatentShape.from_pixel_shape(s1_shape).to_torch_shape())
    sigmas1 = LTX2Scheduler().execute(latent=empty, steps=STEPS).to(torch.float32, copy=True).to(dev)
    print("[sigmas]", [round(float(s), 6) for s in sigmas1])

    dump: dict = {}

    def rec(name, t):
        if t is None:
            return
        dump[name] = t.detach().to(torch.float32).cpu().contiguous()

    def dump_loop(sigmas, video_state, audio_state, stepper, transformer, denoiser, **kw):
        # unwrap to the LTXModel that owns transformer_blocks + preprocessors
        model = transformer
        seen = set()
        stack = [transformer]
        found = None
        while stack:
            o = stack.pop()
            if id(o) in seen:
                continue
            seen.add(id(o))
            if hasattr(o, "transformer_blocks") and hasattr(o, "video_args_preprocessor"):
                found = o
                break
            for attr in ("_model", "velocity_model", "model", "_module", "module", "_wrapped", "wrapped"):
                if hasattr(o, attr):
                    stack.append(getattr(o, attr))
        assert found is not None, "could not locate LTXModel"
        model = found
        print("[hook] model:", type(model).__name__, "blocks:", len(model.transformer_blocks))

        pass_counter = {"v": 0, "a": 0}

        orig_v_prep = model.video_args_preprocessor.prepare
        orig_a_prep = model.audio_args_preprocessor.prepare

        def wrap_prepare(orig, tag):
            def inner(modality, cross_modality=None):
                args = orig(modality, cross_modality)
                i = pass_counter[tag]
                pass_counter[tag] += 1
                p = f"p{i}_{tag}"
                rec(f"{p}_x", args.x)
                rec(f"{p}_timesteps", args.timesteps)
                rec(f"{p}_embedded", args.embedded_timestep)
                rec(f"{p}_pe_cos", args.positional_embeddings[0])
                rec(f"{p}_pe_sin", args.positional_embeddings[1])
                if args.cross_positional_embeddings is not None:
                    rec(f"{p}_cpe_cos", args.cross_positional_embeddings[0])
                    rec(f"{p}_cpe_sin", args.cross_positional_embeddings[1])
                rec(f"{p}_ca_ss", args.cross_scale_shift_timestep)
                rec(f"{p}_ca_gate", args.cross_gate_timestep)
                rec(f"{p}_prompt_ts", args.prompt_timestep)
                rec(f"{p}_context", args.context)
                # raw modality inputs
                rec(f"{p}_latent", modality.latent)
                rec(f"{p}_positions", modality.positions)
                return args
            return inner

        model.video_args_preprocessor.prepare = wrap_prepare(orig_v_prep, "v")
        model.audio_args_preprocessor.prepare = wrap_prepare(orig_a_prep, "a")

        handles = []

        def block_hook(idx, mod, inputs, output):
            # output: (video TransformerArgs|None, audio TransformerArgs|None)
            i = pass_counter["v"] - 1  # current pass index
            vo, ao = output
            if vo is not None:
                rec(f"p{i}_blk{idx:02d}_v", vo.x)
            if ao is not None:
                rec(f"p{i}_blk{idx:02d}_a", ao.x)

        for bi in DUMP_BLOCKS:
            handles.append(model.transformer_blocks[bi].register_forward_hook(partial(block_hook, bi)))

        # velocity outputs: hook the LTXModel itself
        def model_hook(mod, inputs, output):
            i = pass_counter["v"] - 1
            vx, ax = output
            rec(f"p{i}_vel_v", vx)
            rec(f"p{i}_vel_a", ax)

        handles.append(model.register_forward_hook(model_hook))

        rv, ra = denoiser(transformer, video_state, audio_state, sigmas, 0)
        rec("x0_cond_v", rv.cond); rec("x0_uncond_v", rv.uncond); rec("x0_mod_v", rv.mod)
        rec("x0_cond_a", ra.cond); rec("x0_uncond_a", ra.uncond); rec("x0_mod_a", ra.mod)
        rec("guided_v", rv.denoised); rec("guided_a", ra.denoised)
        rec("state_v", video_state.latent)
        rec("state_a", audio_state.latent)
        rec("positions_v", video_state.positions)
        rec("positions_a", audio_state.positions)

        for h in handles:
            h.remove()
        print("[dump] captured", len(dump), "tensors; passes v/a:", pass_counter)
        return video_state, audio_state

    vparams = LTX_2_3_HQ_PARAMS.video_guider_params
    aparams = LTX_2_3_HQ_PARAMS.audio_guider_params
    print("[guider] video", vparams, "audio", aparams)

    lora_sd_ops = None
    if USE_LORA_SDOPS:
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP
        lora_sd_ops = LTXV_LORA_COMFY_RENAMING_MAP
        print("[lora] OFFICIAL renaming map — LoRA WILL fuse")
    else:
        print("[lora] sd_ops=None (golden contract) — LoRA silently dropped")
    stage_loras = () if NO_LORA else (
        LoraPathStrengthAndSDOps(DISTILLED_LORA, 0.25, lora_sd_ops),
    )
    print("[lora] explicitly disabled" if NO_LORA else "[lora] configured")
    stage_1 = DiffusionStage(
        DIT_CKPT, dtype, dev,
        loras=stage_loras,
        quantization=None,
        offload_mode=OffloadMode.DISK,
    )
    stage_1(
        denoiser=GuidedDenoiser(
            v_context=v_p, a_context=a_p,
            video_guider=MultiModalGuider(params=vparams, negative_context=v_n),
            audio_guider=MultiModalGuider(params=aparams, negative_context=a_n),
        ),
        sigmas=sigmas1, noiser=InjectNoiser(), stepper=Res2sDiffusionStep(),
        width=WIDTH // 2, height=HEIGHT // 2, frames=FRAMES, fps=FPS,
        video=ModalitySpec(context=v_p),
        audio=ModalitySpec(context=a_p),
        loop=dump_loop,
    )

    save_file(dump, OUT)
    print("[dump] wrote", len(dump), "tensors ->", OUT)
    for k in sorted(dump):
        print("   ", k, tuple(dump[k].shape))


if __name__ == "__main__":
    main()
