#!/usr/bin/env python
# boogu_c8_reference.py — C8 e2e TORCH REFERENCE harness (2-stage, NOT shipped).
#
# Stage split avoids encoder(16GB)+DiT(20GB) co-residence on the 24GB GPU — each
# stage frees fully on process exit. Run BOTH, in order:
#   python boogu_c8_reference.py encode    # dumps c8_init_latent + feats_{cond,uncond}
#   python boogu_c8_reference.py denoise   # loads them, DiT loop + VAE -> latent + PNG
# Produces (in boogu_dumps/): c8_init_latent.bin, c8_feats_cond.bin, c8_feats_uncond.bin,
#   c8_final_latent_torch.bin ; and output/boogu_t2i_256_torch.png
import os, sys, json
os.environ.setdefault("device", "cuda:0")
import numpy as np
import torch
from PIL import Image

ROOT = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base"
DUMP = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_dumps"
OUTIMG = "/home/alex/mojodiffusion/output/boogu_t2i_256_torch.png"
os.makedirs(DUMP, exist_ok=True); os.makedirs(os.path.dirname(OUTIMG), exist_ok=True)

INSTRUCTION = "A photorealistic portrait of an astronaut riding a horse on Mars."
SYS_T2I = ("You are a helpful assistant that generates high-quality images based on "
           "user instructions. The instructions are as follows.")
SYS_DROP = ("Describe the key features of the input image (color, shape, size, texture, "
            "objects, background), then explain how the user's text instruction should "
            "alter or modify the image. Generate a new image that meets the user's "
            "requirements while maintaining consistency with the original input where appropriate.")
H_LAT = W_LAT = 32
STEPS, CFG, SEED = 20, 4.0, 0
AXES_DIM, AXES_LENS, THETA = [40, 40, 40], [2048, 1664, 1664], 10000
dev, dt = "cuda:0", torch.bfloat16


def dump(name, t):
    np.asarray(t.detach().float().cpu().numpy()).ravel().astype("<f4").tofile(os.path.join(DUMP, name))


def stage_encode():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    g = torch.Generator(dev).manual_seed(SEED)
    latent = torch.randn(1, 16, H_LAT, W_LAT, generator=g, device=dev, dtype=torch.float32)
    dump("c8_init_latent.bin", latent)
    processor = AutoProcessor.from_pretrained(f"{ROOT}/mllm", trust_remote_code=True)
    mllm = Qwen3VLForConditionalGeneration.from_pretrained(
        f"{ROOT}/mllm", torch_dtype=dt, trust_remote_code=True).to(dev).eval()

    def enc(system, instruction):
        prompt = [{"role": "system", "content": [{"type": "text", "text": system}]},
                  {"role": "user", "content": [{"type": "text", "text": instruction}]}]
        vi = processor.apply_chat_template([prompt], padding="longest", padding_side="right",
                                           return_tensors="pt", tokenize=True, return_dict=True)
        vi = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in vi.items()}
        with torch.no_grad():
            return mllm(**vi, output_hidden_states=True, return_dict=True).hidden_states[-1].detach()

    fc, fu = enc(SYS_T2I, INSTRUCTION), enc(SYS_DROP, "")
    dump("c8_feats_cond.bin", fc); dump("c8_feats_uncond.bin", fu)
    json.dump({"cap_cond": fc.shape[1], "cap_uncond": fu.shape[1]},
              open(os.path.join(DUMP, "c8_meta.json"), "w"))
    print(f"[encode] cond L={fc.shape[1]} uncond L={fu.shape[1]} | init std={latent.std():.4f}")


def stage_denoise():
    from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
    from boogu.models.transformers.rope import BooguImageDoubleStreamRotaryPosEmbed
    from boogu.schedulers.scheduling_flow_match_euler_discrete_time_shifting import FlowMatchEulerDiscreteScheduler
    from diffusers import AutoencoderKL
    meta = json.load(open(os.path.join(DUMP, "c8_meta.json")))
    Lc, Lu = meta["cap_cond"], meta["cap_uncond"]
    latent = torch.from_numpy(np.fromfile(os.path.join(DUMP, "c8_init_latent.bin"), dtype="<f4")
                              .reshape(1, 16, H_LAT, W_LAT)).to(dev)
    fc = torch.from_numpy(np.fromfile(os.path.join(DUMP, "c8_feats_cond.bin"), dtype="<f4")
                          .reshape(1, Lc, 4096)).to(dev, dt)
    fu = torch.from_numpy(np.fromfile(os.path.join(DUMP, "c8_feats_uncond.bin"), dtype="<f4")
                          .reshape(1, Lu, 4096)).to(dev, dt)
    mc = torch.ones(1, Lc, dtype=torch.bool, device=dev); mu = torch.ones(1, Lu, dtype=torch.bool, device=dev)

    tf = BooguImageTransformer2DModel.from_pretrained(f"{ROOT}/transformer", torch_dtype=dt).to(dev).eval()
    freqs = BooguImageDoubleStreamRotaryPosEmbed.get_freqs_cis(AXES_DIM, AXES_LENS, THETA)
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{ROOT}/scheduler")
    sched.set_timesteps(num_inference_steps=STEPS, num_tokens=(H_LAT // 2) * (W_LAT // 2))
    with torch.no_grad():
        for i, t in enumerate(sched.timesteps):
            lb = latent.to(dt); tt = t.to(dev).reshape(1)
            pc = tf(lb, tt, fc, freqs, mc, ref_image_hidden_states=None, return_dict=False)
            pu = tf(lb, tt, fu, freqs, mu, ref_image_hidden_states=None, return_dict=False)
            pc = pc if torch.is_tensor(pc) else pc[0]; pu = pu if torch.is_tensor(pu) else pu[0]
            model_pred = pc + (CFG - 1.0) * (pc - pu)
            latent = sched.step(model_pred.float(), t, latent.float(), return_dict=False)[0]
            print(f"  step {i:2d} t={float(t):.4f} std={latent.std().item():.4f}")
    dump("c8_final_latent_torch.bin", latent)
    del tf; torch.cuda.empty_cache()

    vae = AutoencoderKL.from_pretrained(f"{ROOT}/vae", torch_dtype=torch.float32).to(dev).eval()
    with torch.no_grad():
        z = latent.float() / vae.config.scaling_factor + vae.config.shift_factor
        img = vae.decode(z, return_dict=False)[0]
    img = ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).float().cpu().numpy()
    Image.fromarray((img * 255).round().astype(np.uint8)).save(OUTIMG)
    print(f"[denoise] wrote {OUTIMG}  final latent std={latent.std().item():.4f}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "encode"
    (stage_encode if stage == "encode" else stage_denoise)()
