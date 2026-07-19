#!/usr/bin/env python3
# zimage_oracle_denoise.py — DENOISE-ONLY diffusers reference. Loads the
# already-dumped cond/uncond/noise bins (from the encode step) and runs the
# diffusers ZImagePipeline denoise loop with text_encoder=None (so the 8GB
# encoder is NEVER on GPU — the 12.3GB transformer fits alone). Dumps step-0
# raw velocities + per-step latent trajectory + final latent + schedule meta.
# Run: /home/alex/serenityflow-v2/.venv/bin/python <this>
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import numpy as np
from diffusers import ZImagePipeline

ZROOT = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"
)
PD = "/home/alex/mojodiffusion/serenitymojo/pipeline/parity"
STEPS = 30
CFG = 4.0
H = W = 256
meta = []


def load_bin(name):
    shape = tuple(int(x) for x in open(os.path.join(PD, name + ".shape")).read().split(","))
    a = np.fromfile(os.path.join(PD, name + ".bin"), dtype="<f4").reshape(shape)
    return torch.from_numpy(a)


def dump(name, t):
    a = t.detach().to(torch.float32).contiguous().cpu().numpy().ravel()
    a.tofile(os.path.join(PD, name + ".bin"))
    with open(os.path.join(PD, name + ".shape"), "w") as f:
        f.write(",".join(str(int(x)) for x in t.shape))
    meta.append(f"{name}: shape={tuple(int(x) for x in t.shape)}")


print("[denoise-oracle] loading transformer+vae (text_encoder=None)...")
pipe = ZImagePipeline.from_pretrained(
    ZROOT, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16
)
pipe.to("cuda")
print("[denoise-oracle] GPU alloc:", f"{torch.cuda.memory_allocated()/1e9:.2f}GB")

cond = load_bin("cond").to("cuda", torch.bfloat16)      # [173,2560]
uncond = load_bin("uncond").to("cuda", torch.bfloat16)  # [8,2560]
noise = load_bin("noise").to("cuda", torch.bfloat16)    # [1,16,32,32]

# raw velocity capture at several steps (to measure velocity error vs t)
captured = []
step_ctr = [0]
SAMPLE = (0, 7, 14, 21, 28)
def fwd_hook(m, i, o):
    s = step_ctr[0]
    vel = o[0] if isinstance(o, (list, tuple)) else o
    if s == 0 and len(captured) == 0:
        captured.append(o)
    if s in SAMPLE:
        try:
            dump(f"velc_{s:02d}", vel[0])
            dump(f"velu_{s:02d}", vel[1])
        except Exception as e:
            meta.append(f"velcap step {s} failed: {e}")
    step_ctr[0] += 1
h = pipe.transformer.register_forward_hook(fwd_hook)

def cb(pp, step, t, kw):
    dump(f"lat_step_{step:02d}", kw["latents"])
    return {}

print(f"[denoise-oracle] denoise steps={STEPS} cfg={CFG} {H}x{W}")
result = pipe(
    prompt_embeds=[cond],
    negative_prompt_embeds=[uncond],
    latents=noise,
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

fin = result.images if hasattr(result, "images") else result[0]
if isinstance(fin, (list, tuple)):
    fin = fin[0]
dump("final_latent", fin)

out0 = captured[0]
vel = out0[0] if isinstance(out0, (list, tuple)) else out0
try:
    dump("vel_cond0", vel[0])
    dump("vel_uncond0", vel[1])
    meta.append(f"vel0 type={type(out0)} outer_len={len(out0) if isinstance(out0,(list,tuple)) else 'NA'} vel_len={len(vel)}")
except Exception as e:
    meta.append(f"vel unwrap failed: {e} type(out0)={type(out0)}")

try:
    meta.append("sigmas=" + ",".join(f"{float(s):.5f}" for s in pipe.scheduler.sigmas.tolist()))
    meta.append("timesteps=" + ",".join(f"{float(s):.3f}" for s in pipe.scheduler.timesteps.tolist()))
except Exception as e:
    meta.append(f"sigmas n/a: {e}")
meta.append(f"shift={getattr(pipe.scheduler.config,'shift','?')} steps={STEPS} cfg={CFG}")

open(os.path.join(PD, "meta.txt"), "w").write("\n".join(meta) + "\n")
print("[denoise-oracle] done.")
print("\n".join(meta))
