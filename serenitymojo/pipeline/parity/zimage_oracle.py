#!/usr/bin/env python3
# zimage_oracle.py — diffusers ZImagePipeline reference for STAGE-BY-STAGE parity
# of the pure-Mojo pipeline. Dumps, from ONE fixed-noise run:
#   noise.bin              [1,16,64,64]  initial latent (fed to BOTH diffusers + Mojo)
#   cond.bin / uncond.bin  [seq,2560]    encode_prompt outputs (real-length, layer-34)
#   vel_cond0.bin / vel_uncond0.bin  raw transformer velocity at step 0 (THE sign test)
#   lat_step_NN.bin        per-step latent trajectory (denoise parity)
#   final_latent.bin       final denoised latent (before VAE)
#   meta.txt               shapes + sigmas + config
#
# GPU bf16 + model_cpu_offload (NEVER fp32-host-load — that caused the OOM).
# Run: /tmp/vae_oracle_venv/bin/python serenitymojo/pipeline/parity/zimage_oracle.py
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import numpy as np
from diffusers import ZImagePipeline

ZROOT = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"
)
OUT = "/home/alex/mojodiffusion/serenitymojo/pipeline/parity"
os.makedirs(OUT, exist_ok=True)

PROMPT = (
    "Masterpiece, best quality, high resolution, detailed, very detailed,"
    " intricate detailed, (hourglass:1.2), gyroid, gyroid lattice, gyroid fill,"
    " Pearlum, filigree brass, pearls, beautiful photo of a sculpture made of"
    " fluorite mineral, flu0rite, translucent gems, geode, made out of transistors,"
    " LED, wires, eh1, ethereon, filigree brass, detailed background, complex"
    " background, dynamic composition, cinematic scene, perfect composition, matte"
    " finish, 85mm lens, f/1.8, layered textures, dreamy, nostalgic, perfect"
    " composition, intricate detail, depth of field, (bokeh:0.5), professional 4k"
    " highly detailed, Canon 5d mark 4, moody lighting,"
)
STEPS = 30
CFG = 4.0
SEED = 42
H = W = 256  # latent 32x32 — convention parity is resolution-independent; fits GPU
meta = []


def dump(name, t):
    a = t.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
    a.tofile(os.path.join(OUT, name + ".bin"))
    with open(os.path.join(OUT, name + ".shape"), "w") as f:
        f.write(",".join(str(int(x)) for x in t.shape))
    meta.append(f"{name}: shape={tuple(int(x) for x in t.shape)}")


import gc
print("[oracle] loading ZImagePipeline (bf16)...")
pipe = ZImagePipeline.from_pretrained(ZROOT, torch_dtype=torch.bfloat16)  # on CPU

# Fixed initial noise (CPU f32, seeded) — dumped and fed to BOTH stacks.
gen = torch.Generator(device="cpu").manual_seed(SEED)
noise = torch.randn(1, 16, H // 8, W // 8, generator=gen, dtype=torch.float32)
dump("noise", noise)

# Stage 2: encode on GPU, then FREE the encoder before the transformer loads
# (manual device mgmt — the 8GB encoder + 12GB transformer don't co-fit on 24GB).
pipe.text_encoder.to("cuda")
cond_list, uncond_list = pipe.encode_prompt(
    PROMPT, device="cuda", do_classifier_free_guidance=True, negative_prompt="",
    max_sequence_length=512,
)
dump("cond", cond_list[0])
dump("uncond", uncond_list[0])
print("[oracle] cond", tuple(cond_list[0].shape), "uncond", tuple(uncond_list[0].shape))
pipe.text_encoder.to("cpu")
pipe.text_encoder = None  # prevent __call__ from re-loading the 8GB encoder to GPU
gc.collect()
torch.cuda.empty_cache()

# move denoise modules to GPU; keep embeds/noise on cuda
pipe.transformer.to("cuda")
pipe.vae.to("cuda")
cond_list = [c.to("cuda") for c in cond_list]
uncond_list = [c.to("cuda") for c in uncond_list]

# Stage 3: capture raw transformer velocity at step 0 (batched cond+uncond).
captured = []


def fwd_hook(module, inp, out):
    if len(captured) == 0:
        captured.append(out)


h = pipe.transformer.register_forward_hook(fwd_hook)

# Stage 4: per-step latent trajectory.
def cb(pp, step, t, kw):
    dump(f"lat_step_{step:02d}", kw["latents"])
    return {}


print(f"[oracle] denoise: steps={STEPS} cfg={CFG} {H}x{W}")
result = pipe(
    prompt_embeds=cond_list,
    negative_prompt_embeds=uncond_list,
    latents=noise.to("cuda", pipe.dtype),
    num_inference_steps=STEPS,
    guidance_scale=CFG,
    height=H,
    width=W,
    output_type="latent",
    return_dict=True,
    callback_on_step_end=cb,
    callback_on_step_end_tensor_inputs=["latents"],
)
h.remove()

# final latent
fin = result.images if hasattr(result, "images") else result[0]
if isinstance(fin, (list, tuple)):
    fin = fin[0]
dump("final_latent", fin)

# step-0 raw velocity. transformer return is `(list_of_per_sample,)` or similar;
# unwrap to get [cond_velocity, uncond_velocity].
out0 = captured[0]
vel_list = out0[0] if isinstance(out0, (list, tuple)) else out0
# vel_list should be a list with batch 2 (cond, uncond)
try:
    dump("vel_cond0", vel_list[0])
    dump("vel_uncond0", vel_list[1])
    meta.append(f"vel0 type={type(out0)} len={len(vel_list)}")
except Exception as e:
    meta.append(f"vel0 unwrap failed: {e}; type(out0)={type(out0)}")
    # fall back: dump whatever it is, shaped
    try:
        dump("vel_raw0", vel_list if torch.is_tensor(vel_list) else vel_list[0])
    except Exception as e2:
        meta.append(f"vel_raw dump failed: {e2}")

# scheduler sigmas/timesteps actually used
try:
    meta.append("sigmas=" + ",".join(f"{float(s):.5f}" for s in pipe.scheduler.sigmas.tolist()))
    meta.append("timesteps=" + ",".join(f"{float(s):.3f}" for s in pipe.scheduler.timesteps.tolist()))
except Exception as e:
    meta.append(f"sigmas/timesteps unavailable: {e}")
meta.append(f"shift={getattr(pipe.scheduler.config, 'shift', '?')}  steps={STEPS}  cfg={CFG}  seed={SEED}")

with open(os.path.join(OUT, "meta.txt"), "w") as f:
    f.write("\n".join(meta) + "\n")
print("[oracle] done. meta:")
print("\n".join(meta))
