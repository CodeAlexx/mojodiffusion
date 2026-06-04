#!/usr/bin/env python3
"""prepare_sd35_cache.py — Build SD3.5 training cache from raw jpg/txt pairs.

Cache format per EDv2 prepare_sd35.rs / OneTrainer #SD3-LoRA preset:
  latent         : [1, 16, H/8, W/8] BF16  (VAE shift=0.0609, scale=1.5305)
  text_embedding : [1, 154, 4096]    BF16  (77 CLIP-L/G padded + 77 T5)
  pooled         : [1, 2048]         BF16  (clip_l_pool[768] + clip_g_pool[1280])

Resolution: 1024x1024 (locked per EDv2 TRAIN_RES=1024).
T5 max_len: 77 (hard: combined seq = 77+77=154).

Usage:
  python3 serenitymojo/training/prepare_sd35_cache.py \
      --input_dir /home/alex/datasets/andrsd35ver1 \
      --output_dir /home/alex/datasets/andrsd35_sd35_cache
"""
import argparse
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from safetensors.torch import save_file

VAE_SCALE = 1.5305
VAE_SHIFT = 0.0609
TRAIN_RES = 1024
CLIP_MAX_LEN = 77
T5_MAX_LEN = 77

CLIP_L_PATH = "/home/alex/.serenity/models/text_encoders/clip_l.safetensors"
CLIP_G_PATH = "/home/alex/.serenity/models/text_encoders/clip_g.safetensors"
T5_PATH = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.safetensors"
SD35_CKPT = "/home/alex/.serenity/models/checkpoints/sd3.5_large.safetensors"

CLIP_L_TOK_PATH = "/home/alex/.serenity/models/text_encoders/clip_l.tokenizer.json"
CLIP_G_TOK_PATH = "/home/alex/.serenity/models/text_encoders/clip_g.tokenizer.json"
T5_TOK_PATH = "/home/alex/.serenity/models/text_encoders/t5xxl_fp16.tokenizer.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_samples", type=int, default=0)
    return p.parse_args()


def load_image(path: str, res: int = TRAIN_RES) -> torch.Tensor:
    """Load and center-crop to [1,3,res,res] BF16 in [-1,1]."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((res, res), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(torch.bfloat16)


def load_vae_from_ckpt(ckpt_path: str, device: torch.device):
    """Load SD3 VAE from the combined checkpoint."""
    from safetensors import safe_open
    from diffusers import AutoencoderKL
    state = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith("first_stage_model."):
                state[k[len("first_stage_model."):]] = f.get_tensor(k).to(torch.float32)
    # SD3 VAE is 16ch
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",)*4,
        up_block_types=("UpDecoderBlock2D",)*4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=16,
        norm_num_groups=32,
        sample_size=1024,
        scaling_factor=VAE_SCALE,
        shift_factor=VAE_SHIFT,
        force_upcast=False,
    )
    missing, unexpected = vae.load_state_dict(state, strict=False)
    if len(missing) > 10:
        print(f"  [vae] WARNING: {len(missing)} missing keys (first: {missing[:3]})")
    vae = vae.to(device).to(torch.bfloat16).eval()
    print(f"[vae] Loaded OK (missing={len(missing)}, unexpected={len(unexpected)})")
    return vae


def load_clip_l(device: torch.device):
    """Load CLIP-L encoder and tokenizer."""
    from transformers import CLIPTextModel, CLIPTokenizer
    from safetensors.torch import load_file
    print("[enc] Loading CLIP-L...")
    # Use the tokenizer JSON directly
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    state = load_file(CLIP_L_PATH)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  CLIP-L: missing={len(missing)}, unexpected={len(unexpected)}")
    return tok, model.to(device).to(torch.float16).eval()


def load_clip_g(device: torch.device):
    """Load CLIP-G encoder and tokenizer."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    from safetensors.torch import load_file
    print("[enc] Loading CLIP-G...")
    tok = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    model = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    state = load_file(CLIP_G_PATH)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  CLIP-G: missing={len(missing)}, unexpected={len(unexpected)}")
    return tok, model.to(device).to(torch.float16).eval()


def load_t5(device: torch.device):
    """Load T5-XXL encoder."""
    from transformers import T5EncoderModel, AutoTokenizer
    from safetensors.torch import load_file
    print("[enc] Loading T5-XXL...")
    tok = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
    model = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")
    state = load_file(T5_PATH)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  T5: missing={len(missing)}, unexpected={len(unexpected)}")
    return tok, model.to(device).to(torch.float16).eval()


def encode_text(
    caption: str,
    clip_l_tok, clip_l_model,
    clip_g_tok, clip_g_model,
    t5_tok, t5_model,
    device: torch.device,
):
    """Returns (text_embedding [1,154,4096] BF16, pooled [1,2048] BF16)."""
    with torch.no_grad():
        # CLIP-L
        cl_ids = clip_l_tok(caption, return_tensors="pt", max_length=CLIP_MAX_LEN,
                            padding="max_length", truncation=True).input_ids.to(device)
        cl_out = clip_l_model(cl_ids, output_hidden_states=True)
        cl_h = cl_out.hidden_states[-2].to(torch.float32)   # [1,77,768]
        cl_pool = cl_out.pooler_output.to(torch.float32)    # [1,768]

        # CLIP-G
        cg_ids = clip_g_tok(caption, return_tensors="pt", max_length=CLIP_MAX_LEN,
                            padding="max_length", truncation=True).input_ids.to(device)
        cg_out = clip_g_model(cg_ids, output_hidden_states=True)
        cg_h = cg_out.hidden_states[-2].to(torch.float32)   # [1,77,1280]
        cg_pool = cg_out.text_embeds.to(torch.float32)      # [1,1280]

        # T5
        t5_ids = t5_tok(caption, return_tensors="pt", max_length=T5_MAX_LEN,
                        padding="max_length", truncation=True).input_ids.to(device)
        t5_out = t5_model(t5_ids)
        t5_h = t5_out.last_hidden_state.to(torch.float32)   # [1,77,4096]

        # Combine: CLIP-L [1,77,768] + CLIP-G [1,77,1280] -> [1,77,2048] pad to [1,77,4096]
        clip_lg = torch.cat([cl_h, cg_h], dim=-1)           # [1,77,2048]
        pad = torch.zeros(1, CLIP_MAX_LEN, 4096 - 2048, device=device, dtype=torch.float32)
        clip_lg_pad = torch.cat([clip_lg, pad], dim=-1)     # [1,77,4096]

        # Combined context: [CLIP, T5] -> [1,154,4096]
        context = torch.cat([clip_lg_pad, t5_h], dim=1)    # [1,154,4096]

        # Pooled: [cl_pool, cg_pool] -> [1,2048]
        pooled = torch.cat([cl_pool, cg_pool], dim=-1)     # [1,2048]

    return context.to(torch.bfloat16), pooled.to(torch.bfloat16)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find pairs
    in_dir = Path(args.input_dir)
    pairs = []
    for f in sorted(in_dir.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            txt = f.with_suffix(".txt")
            if txt.exists():
                pairs.append((f, txt))
    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]
    print(f"[cache] Found {len(pairs)} pairs")

    # Load models
    vae = load_vae_from_ckpt(SD35_CKPT, device)
    cl_tok, cl_model = load_clip_l(device)
    cg_tok, cg_model = load_clip_g(device)
    t5_tok, t5_model = load_t5(device)

    for idx, (img_path, txt_path) in enumerate(pairs):
        out_file = out_dir / f"{img_path.stem}.safetensors"
        if out_file.exists():
            print(f"[{idx+1}/{len(pairs)}] skip {out_file.name}")
            continue

        caption = txt_path.read_text().strip()
        print(f"[{idx+1}/{len(pairs)}] {img_path.name}: {caption[:60]}")

        # Encode image
        img_t = load_image(str(img_path), TRAIN_RES).to(device)
        with torch.no_grad():
            dist = vae.encode(img_t).latent_dist
            z = dist.sample()
            z = (z - VAE_SHIFT) * VAE_SCALE
        latent = z.to(torch.bfloat16)

        # Encode text
        text_embedding, pooled = encode_text(
            caption, cl_tok, cl_model, cg_tok, cg_model, t5_tok, t5_model, device
        )

        out = {
            "latent": latent.cpu(),
            "text_embedding": text_embedding.cpu(),
            "pooled": pooled.cpu(),
        }
        save_file(out, str(out_file))
        print(f"  -> latent {latent.shape}, text {text_embedding.shape}, pooled {pooled.shape}")

    print(f"[done] Cache at {out_dir}")


if __name__ == "__main__":
    main()
