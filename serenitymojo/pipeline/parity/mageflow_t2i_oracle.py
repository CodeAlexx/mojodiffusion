#!/usr/bin/env python
# mageflow_t2i_oracle.py — E2E parity oracle for the MageFlow T2I turbo pipeline.
#
# Replicates mage_flow.pipeline.generate_images EXACTLY for the single-prompt,
# cfg=1.0 (turbo, no negative branch) case, using the SAME helper functions the
# real pipeline calls (get_noise + encode_noise + _encode_texts_packed +
# _slice_packed + _build_pack_ctx + _velocity + _get_scheduler + vae.decode), so
# the dumped tensors are byte-identical to a real `MageFlowPipeline.generate()`.
#
# Dumps (raw little-endian <f4 into mageflow_t2i_dumps/):
#   input_ids.bin     [L_full]        the exact templated token ids (int->f32)
#   init_noise.bin    [N_IMG,128]     img entering the denoise loop (post
#                                     get_noise+encode_noise+rearrange), bf16->f32
#   final_latent.bin  [N_IMG,128]     img after the 4 Euler steps (pre-VAE), f32
#   rgb.bin           [1,3,H,W]       vae.decode(unpack(final)), clamp[-1,1], f32
#   oracle_image.png  the decoded PIL image (visual reference)
#   meta.txt          shapes + hyperparameters
#
# Run:
#   cd /home/alex/mojodiffusion
#   PYTHONPATH=/home/alex/Mage pixi run python \
#     serenitymojo/pipeline/parity/mageflow_t2i_oracle.py
import os
import gc

import numpy as np
import torch
from einops import rearrange
from PIL import Image

MAGE_TURBO = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Turbo"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mageflow_t2i_dumps")
os.makedirs(OUT, exist_ok=True)

PROMPT = "A photorealistic portrait of an astronaut riding a horse on Mars."
SEED = 42
STEPS = 4       # turbo
CFG = 1.0       # turbo -> no negative branch
H = 256
W = 256


def dump_f32(name, arr):
    v = np.asarray(arr).ravel().astype("<f4")
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(v.tobytes())
    return list(np.asarray(arr).shape)


@torch.no_grad()
def main():
    from mage_flow.pipeline import (
        load_from_repo, _encode_texts_packed, _slice_packed, _build_pack_ctx,
        _velocity, _get_scheduler,
    )
    from mage_flow.models.utils import get_noise, PROMPT_TEMPLATE, unpack
    from mage_flow.models.modules.mage_latent import encode_noise, resolve_gs_key
    from mage_flow.models.modules._attn_backend import set_attn_backend

    dev = torch.device("cuda:0")
    print(f"[mf-t2i-oracle] loading MageFlow Turbo (CPU-resident) from {MAGE_TURBO}")
    # 16 GB card: load everything on CPU, shuttle one submodule to GPU per stage
    # (same offload discipline the Mojo pipeline uses).
    model = load_from_repo(MAGE_TURBO, "cpu")

    # No flash-attn wheel on this box (CUDA 13 / sm_120): use the torch-SDPA
    # varlen fallback for BOTH the packed text encoder and the DiT joint attn.
    # MUST be set AFTER load (model construction resets the backend to flash2).
    set_attn_backend("sdpa")

    info = PROMPT_TEMPLATE["mage-flow"]
    template = info["template"]
    drop_idx = int(info["start_idx"])

    # --- exact token ids (same call _encode_texts_packed makes) ---
    tokenizer = model.txt_enc.tokenizer
    max_len = model.txt_enc.tokenizer_max_length + drop_idx
    ids = tokenizer(template.format(PROMPT), max_length=max_len, truncation=True,
                    return_tensors="pt").input_ids.squeeze(0)
    L_full = int(ids.numel())
    keep = L_full - drop_idx
    print(f"[mf-t2i-oracle] L_full={L_full} drop={drop_idx} keep(N_TXT)={keep}")

    # --- STAGE 1: text conditioning (cfg=1 -> no negative), encoder on GPU ---
    model.txt_enc.to(dev)
    txt_flat, vec_all, lens_t = _encode_texts_packed(model, [PROMPT], template, drop_idx, dev)
    txt, txt_cu, txt_mask, vec = _slice_packed(txt_flat, vec_all, lens_t, 0, 1, dev)
    txt = txt.detach().clone(); vec = vec.detach().clone()  # keep on GPU (small)
    model.txt_enc.to("cpu")
    torch.cuda.empty_cache()
    print("[mf-t2i-oracle] STAGE1 text encoded; encoder offloaded")

    # --- initial noise (get_noise + Gaussian-Shading encode_noise), packed ---
    ch = model.vae.latent_channels
    gs_key = resolve_gs_key(None)
    torch.manual_seed(SEED)
    x = get_noise(num_samples=1, channel=ch, height=H, width=W,
                  device=dev, dtype=torch.bfloat16, seed=SEED)
    x = encode_noise(tuple(x.shape[1:]), key=gs_key, seed=SEED, device=dev,
                     dtype=torch.bfloat16)
    _, _, gh, gw = x.shape
    n_img = gh * gw
    img = rearrange(x, "b c h w -> b (h w) c")            # [1, n_img, 128]
    ids_pos = torch.zeros(gh, gw, 3, device=dev)
    ids_pos[..., 1] = ids_pos[..., 1] + torch.arange(gh, device=dev)[:, None]
    ids_pos[..., 2] = ids_pos[..., 2] + torch.arange(gw, device=dev)[None, :]
    img_ids = rearrange(ids_pos, "h w c -> (h w) c").unsqueeze(0)
    lens = [n_img]
    img_cu = torch.tensor([0, n_img], device=dev, dtype=torch.int32)
    img_shapes = [[(1, gh, gw)]]
    print(f"[mf-t2i-oracle] gh={gh} gw={gw} N_IMG={n_img} (SH={gh})")

    ctx = _build_pack_ctx(img_ids, img_cu, img_shapes, lens, txt, txt_cu, txt_mask, vec,
                          None, None, None, None, CFG, False, True, dev)

    init_noise = img.clone().float().cpu().numpy()        # [1, n_img, 128]

    # --- STAGE 2: 4-step Euler denoise (identical to generate_images loop) ---
    model.transformer.to(dev)
    scheduler = _get_scheduler(model, STEPS, "cuda:0", None)
    sig = [float(s) for s in scheduler.sigmas.tolist()]
    print(f"[mf-t2i-oracle] sigmas = {sig}")
    for si, t in enumerate(scheduler.timesteps):
        pred = _velocity(model.transformer, img, ctx, scheduler.sigmas[si].item())
        img = scheduler.step(pred, t, img, return_dict=False)[0]
    final_latent = img.float().cpu().numpy()              # [1, n_img, 128]
    model.transformer.to("cpu")
    torch.cuda.empty_cache()
    print("[mf-t2i-oracle] STAGE2 denoise done; DiT offloaded")

    # --- STAGE 3: VAE decode (identical to _decode_one) ---
    model.vae.to(dev)
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        out = model.vae.decode(unpack(img.float(), H, W))  # [1,3,H,W]
    out = out.clamp(-1, 1)
    rgb = out.float().cpu().numpy()                        # [1,3,H,W]
    img_u8 = (127.5 * (rearrange(out, "b c h w -> b h w c") + 1.0)).cpu().byte().numpy()
    Image.fromarray(img_u8[0]).save(os.path.join(OUT, "oracle_image.png"))

    shapes = {}
    shapes["input_ids.bin"] = dump_f32("input_ids.bin", ids.cpu().numpy())
    shapes["init_noise.bin"] = dump_f32("init_noise.bin", init_noise)
    shapes["final_latent.bin"] = dump_f32("final_latent.bin", final_latent)
    shapes["rgb.bin"] = dump_f32("rgb.bin", rgb)

    with open(os.path.join(OUT, "meta.txt"), "w") as f:
        f.write(f"prompt={PROMPT!r}\n")
        f.write(f"seed={SEED} steps={STEPS} cfg={CFG} H={H} W={W}\n")
        f.write(f"L_full={L_full} drop={drop_idx} N_TXT={keep} N_IMG={n_img} SH={gh}\n")
        f.write(f"sigmas={sig}\n")
        f.write(f"input_ids={ids.cpu().tolist()}\n")
        for k, v in shapes.items():
            f.write(f"{k} shape={v}\n")
        f.write(f"init_noise.std={float(init_noise.std()):.6f}\n")
        f.write(f"final_latent.std={float(final_latent.std()):.6f}\n")
        f.write(f"rgb.min={float(rgb.min()):.4f} rgb.max={float(rgb.max()):.4f} rgb.mean={float(rgb.mean()):.4f}\n")
    print(f"[mf-t2i-oracle] dumped: {shapes}")
    print(f"[mf-t2i-oracle] final_latent.std={float(final_latent.std()):.6f} "
          f"rgb[min={float(rgb.min()):.3f},max={float(rgb.max()):.3f}]")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[mf-t2i-oracle] done")


if __name__ == "__main__":
    main()
