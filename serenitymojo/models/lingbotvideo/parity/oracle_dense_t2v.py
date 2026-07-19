#!/usr/bin/env python
# Oracle-DENSE-T2V: LingBot-Video-Dense-1.3B TEXT-TO-VIDEO reference. Reuses the
# RESIDENT dense transformer + encoder from oracle_dense_t2i.py, but generates a
# MULTI-FRAME clip (num_frames = 4n+1) and decodes it with AutoencoderKLWan.decode
# which handles the temporal upsampling natively. This is the parity oracle for
# the pure-Mojo T2V pipeline (dense backbone + tiled attn + FlowUniPC + temporal
# Wan VAE decode).
#
#   prompt(JSON, MOTION) + DEFAULT_NEGATIVE_PROMPT -> Qwen3-VL encode -> free enc
#   randn init latent (1,16,latent_frames,lh,lw)
#   NUM_STEPS FlowUniPC CFG denoise (gs=GUIDANCE, shift=SHIFT) with resident dense
#   _dit_latent_to_vae -> AutoencoderKLWan.decode (TEMPORAL) -> clamp -> (x+1)/2
#
# Saves oracle_dense_t2v.safetensors (prompt/neg embeds, init + final latent,
# pixels [1,3,F,H,W]) + oracle_dense_t2v_reference.mp4 + reference frames.
#
# Configurable via env: T2V_FRAMES, T2V_H, T2V_W, T2V_STEPS (validate small,
# then scale to a 5s / 121-frame clip).
#
# Run:
#   cd /mnt/disk1/lingbot-src/lingbot-video && \
#     T2V_FRAMES=13 T2V_H=320 T2V_W=576 T2V_STEPS=40 \
#     /home/alex/SerenityTrainer/venv/bin/python \
#       /home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity/oracle_dense_t2v.py
import os, sys, json, time, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from safetensors.torch import save_file
from PIL import Image

OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
sys.path.insert(0, OUT)
sys.path.insert(0, "/mnt/disk1/lingbot-src/lingbot-video")

from oracle_dense_t2i import ResidentDenseTransformer, encode_one, _compute_crop_start
from lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler
from lingbot_video.pipeline_lingbot_video import DEFAULT_NEGATIVE_PROMPT, TOKEN_LENGTH
from diffusers import AutoencoderKLWan
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

DENSE_ROOT = "/mnt/disk1/models/lingbot-video-dense"
VAE_DIR = os.path.join(DENSE_ROOT, "vae")
MOE_ROOT = "/mnt/disk1/models/lingbot-video-moe"
TE_DIR = os.path.join(MOE_ROOT, "text_encoder")
PROC_DIR = os.path.join(MOE_ROOT, "processor")
DEV = "cuda"

NUM_FRAMES = int(os.environ.get("T2V_FRAMES", "13"))    # 4n+1
H = int(os.environ.get("T2V_H", "320"))
W = int(os.environ.get("T2V_W", "576"))
NUM_STEPS = int(os.environ.get("T2V_STEPS", "40"))
GUIDANCE = 5.0
SHIFT = 3.0
SEED = 12345
FPS = 24

assert NUM_FRAMES == 1 or (NUM_FRAMES - 1) % 4 == 0, "num_frames must be 4n+1"

# Structured-JSON caption WITH MOTION (robot cyborg dancing, energetic full-body).
PROMPT = json.dumps({
    "comprehensive_description": (
        "A sleek humanoid robot cyborg dances energetically at the center of a "
        "futuristic neon-lit dance studio, performing a continuous rhythmic dance "
        "routine from the first frame to the last: its metallic arms pump and swing "
        "up and down to the beat with sharply bending elbows, its chrome torso "
        "twists left and right as its hips rock side to side, and its articulated "
        "legs step and shuffle in place with quick footwork, knees bouncing on "
        "every beat while its head nods to the rhythm. Glowing cyan light strips "
        "along its limbs and a pulsing blue energy core in its chest flare brighter "
        "with each move, and its polished chrome plating catches sweeping magenta "
        "and cyan neon reflections as it moves. The camera stays static at eye "
        "level, holding the full-body dancer centered so the whole dance clearly "
        "reads - arms, torso, hips and feet all in constant, lively motion "
        "throughout the clip."
    ),
    "camera_info": {
        "color": "Cool",
        "frame_size": "Full Shot",
        "shot_type_angle": "Eye level",
        "lens_size": "Medium",
        "composition": "Center",
        "lighting": "Soft light",
        "lighting_type": "Artificial light",
        "camera_motion": "static shot, camera holds still while the subject dances",
    },
    "world_knowledge": [],
    "prominent_elements": [
        {
            "name": "robot cyborg dancer",
            "description": "A humanoid robot cyborg with polished chrome plating and glowing cyan accents, dancing energetically with continuous full-body motion.",
            "location": "center of the frame, standing on the reflective studio floor",
            "relative_size": "dominant",
            "shape_and_color": "humanoid silhouette in polished silver chrome with glowing cyan light strips and a pulsing blue chest core",
            "texture": "smooth polished metal with segmented articulated joints",
            "appearance_details": "arms swinging and pumping to the rhythm, torso twisting, hips rocking, legs stepping and shuffling side to side, head nodding to the beat, cyan joint lights pulsing brighter with each move",
            "relationship": "dances on the glossy studio floor, its glow reflected beneath its moving feet",
            "orientation": "upright, facing the camera while dancing",
        },
        {
            "name": "futuristic dance studio",
            "description": "A dark futuristic studio with a glossy reflective floor and neon light panels lining the walls.",
            "location": "background and floor, surrounding the dancer",
            "relative_size": "large",
            "shape_and_color": "dark blue-black walls with horizontal magenta and cyan neon strips, glossy dark floor",
            "texture": "smooth glossy reflective floor, matte dark walls with emissive neon strips",
            "appearance_details": "soft magenta and cyan neon reflections on the floor that shift as the robot moves",
            "relationship": "surrounds the dancing robot and mirrors its glowing accents",
            "orientation": "horizontal floor, vertical walls",
        },
    ],
})


def encode_prompts():
    print("[T2V] loading Qwen3-VL text encoder (8.9GB) + processor ...", flush=True)
    processor = AutoProcessor.from_pretrained(PROC_DIR, trust_remote_code=True)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        TE_DIR, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(DEV).eval()
    crop_start = _compute_crop_start(processor)
    p_embeds, p_mask = encode_one(text_encoder, processor, crop_start, PROMPT)
    n_embeds, n_mask = encode_one(text_encoder, processor, crop_start, DEFAULT_NEGATIVE_PROMPT)
    print(f"[T2V] prompt embeds {list(p_embeds.shape)} | neg {list(n_embeds.shape)}", flush=True)
    out = (p_embeds.float().cpu(), n_embeds.float().cpu())
    del text_encoder, processor
    gc.collect(); torch.cuda.empty_cache()
    print("[T2V] text encoder freed", flush=True)
    return out


def main():
    t_start = time.time()
    caps = {}
    p_embeds, n_embeds = encode_prompts()
    caps["prompt_embeds"] = p_embeds
    caps["neg_embeds"] = n_embeds
    L_cond = p_embeds.shape[1]
    L_uncond = n_embeds.shape[1]

    net = ResidentDenseTransformer()

    latent_frames = (NUM_FRAMES - 1) // 4 + 1
    lh, lw = H // 8, W // 8
    gt, gh, gw = latent_frames, lh // 2, lw // 2
    n_video = gt * gh * gw
    print(f"[T2V] frames={NUM_FRAMES} -> latent_frames={latent_frames} @ {H}x{W} "
          f"(lh={lh} lw={lw}) grid {gt}x{gh}x{gw} = {n_video} video tokens; "
          f"S_cond={n_video+L_cond} S_uncond={n_video+L_uncond}", flush=True)

    torch.manual_seed(SEED)
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    latent = torch.randn((1, 16, latent_frames, lh, lw), generator=gen, device=DEV, dtype=torch.float32)
    caps["init_latent"] = latent.float().cpu()

    sch = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sch.set_timesteps(NUM_STEPS, device=DEV, shift=SHIFT)

    p_bf16 = p_embeds.to(DEV, dtype=torch.bfloat16)
    n_bf16 = n_embeds.to(DEV, dtype=torch.bfloat16)

    for i, t in enumerate(sch.timesteps):
        t_int = int(t.item())
        tb = net.transformer_timestep(t_int)
        f0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            noise_cond = net.forward(latent, tb, p_bf16)
            noise_uncond = net.forward(latent, tb, n_bf16)
        noise = noise_uncond + GUIDANCE * (noise_cond - noise_uncond)
        latent = sch.step(noise, t, latent, return_dict=False)[0]
        print(f"[T2V] step {i:02d}/{NUM_STEPS} t={t_int:4d} "
              f"lat(std={latent.std().item():.4f}) [{time.time()-f0:.1f}s]", flush=True)

    caps["final_latent"] = latent.float().cpu()

    # Safety checkpoint: save embeds + latents BEFORE the VAE decode so a decode
    # OOM at high res doesn't lose the denoise (Mojo run only needs these).
    save_file(caps, os.path.join(OUT, "oracle_dense_t2v.safetensors"))
    meta = {"H": H, "W": W, "num_frames": NUM_FRAMES, "latent_frames": latent_frames,
            "num_steps": NUM_STEPS, "guidance": GUIDANCE, "shift": SHIFT, "seed": SEED,
            "fps": FPS, "L_cond": L_cond, "L_uncond": L_uncond,
            "gt": gt, "gh": gh, "gw": gw, "lh": lh, "lw": lw, "n_video": n_video,
            "prompt": PROMPT}
    json.dump(meta, open(os.path.join(OUT, "oracle_dense_t2v_meta.json"), "w"), indent=2)
    print("[T2V] checkpoint saved (embeds+latents, pre-VAE)", flush=True)

    print("[T2V] loading AutoencoderKLWan (fp32) + TEMPORAL decode ...", flush=True)
    del net
    gc.collect(); torch.cuda.empty_cache()
    vae = AutoencoderKLWan.from_pretrained(VAE_DIR, torch_dtype=torch.float32).to(DEV).eval()
    mean = torch.tensor(vae.config.latents_mean, device=DEV, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=DEV, dtype=torch.float32).view(1, -1, 1, 1, 1)
    with torch.no_grad():
        vae_latent = (latent.float() * std + mean).to(device=DEV, dtype=torch.float32)
        decoded = vae.decode(vae_latent)
        frames = decoded[0] if isinstance(decoded, tuple) else decoded.sample
        frames = frames.float().clamp_(-1, 1)
        frames = (frames + 1.0) / 2.0
    pixels = frames.float().cpu()   # [1,3,F,H,W]
    caps["pixels"] = pixels
    print(f"[T2V] pixels {list(pixels.shape)} mean {pixels.mean():.4f} std {pixels.std():.4f}", flush=True)

    save_file(caps, os.path.join(OUT, "oracle_dense_t2v.safetensors"))
    meta = {"H": H, "W": W, "num_frames": NUM_FRAMES, "latent_frames": latent_frames,
            "num_steps": NUM_STEPS, "guidance": GUIDANCE, "shift": SHIFT, "seed": SEED,
            "fps": FPS, "L_cond": L_cond, "L_uncond": L_uncond,
            "gt": gt, "gh": gh, "gw": gw, "lh": lh, "lw": lw, "n_video": n_video,
            "prompt": PROMPT}
    json.dump(meta, open(os.path.join(OUT, "oracle_dense_t2v_meta.json"), "w"), indent=2)

    # reference frames + mp4
    F = pixels.shape[2]
    imgs = []
    for f in range(F):
        img = pixels[0, :, f].permute(1, 2, 0).numpy()
        img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
        imgs.append(img)
    Image.fromarray(imgs[0]).save(os.path.join(OUT, "oracle_dense_t2v_frame0.png"))
    Image.fromarray(imgs[-1]).save(os.path.join(OUT, "oracle_dense_t2v_frameN.png"))
    try:
        import imageio
        imageio.mimsave(os.path.join(OUT, "oracle_dense_t2v_reference.mp4"), imgs, fps=FPS)
        print("[T2V] wrote oracle_dense_t2v_reference.mp4", flush=True)
    except Exception as e:
        print(f"[T2V] mp4 write skipped: {e}", flush=True)

    print(f"[T2V] SAVED oracle_dense_t2v.safetensors  ({F} pixel frames)  "
          f"total {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
