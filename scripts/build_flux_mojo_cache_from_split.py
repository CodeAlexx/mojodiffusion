#!/usr/bin/env python3
"""Build the Mojo Flux trainer cache from split EriDiffusion Flux caches.

Input:
  .flux_latents/<id>.safetensors with key `latents` shaped [1,H,W,16] BF16
  .flux_te/<id>.safetensors with key `text` shaped [seq,4096] BF16
  .flux_latents/<id>.meta.json with the source image path

Output per sample:
  <out>/<id>.safetensors with keys:
    latent    [16,H,W] BF16, CHW for train_flux_real.mojo pack_latents
    t5_embed  [1,seq,4096] BF16
    clip_pool [768] BF16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


DEFAULT_MODEL = (
    "/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/"
    "snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
)


def _caption_for(latent_meta: Path, dataset_root: Path) -> str:
    meta = json.loads(latent_meta.read_text())
    source_name = Path(meta["path"]).with_suffix(".txt").name
    caption_path = dataset_root / source_name
    if not caption_path.exists():
        raise FileNotFoundError(f"missing caption sidecar {caption_path}")
    return caption_path.read_text().strip()


def _load_clip_pool(
    prompt: str,
    tokenizer: CLIPTokenizer,
    model: CLIPTextModel,
    device: torch.device,
) -> torch.Tensor:
    inputs = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        pooled = model(**inputs).pooler_output
    return pooled.squeeze(0).to(dtype=torch.bfloat16, device="cpu").contiguous()


def build(args: argparse.Namespace) -> None:
    latents_dir = args.dataset / ".flux_latents"
    text_dir = args.dataset / ".flux_te"
    if not latents_dir.is_dir():
        raise FileNotFoundError(f"missing {latents_dir}")
    if not text_dir.is_dir():
        raise FileNotFoundError(f"missing {text_dir}")

    latent_ids = {p.stem for p in latents_dir.glob("*.safetensors")}
    text_ids = {p.stem for p in text_dir.glob("*.safetensors")}
    sample_ids = sorted(latent_ids & text_ids)
    if args.limit:
        sample_ids = sample_ids[: args.limit]
    if not sample_ids:
        raise RuntimeError("no matching latent/text cache samples")
    missing_text = latent_ids - text_ids
    missing_latents = text_ids - latent_ids
    if missing_text or missing_latents:
        raise RuntimeError(
            f"split cache mismatch: latent_only={len(missing_text)} text_only={len(missing_latents)}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.clip_model, subfolder="tokenizer", local_files_only=True
    )
    model = CLIPTextModel.from_pretrained(
        args.clip_model,
        subfolder="text_encoder",
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device)
    model.eval()

    manifest = []
    for sample_id in tqdm(sample_ids, desc="Building Flux Mojo cache"):
        out_path = args.output / f"{sample_id}.safetensors"
        if out_path.exists() and not args.overwrite:
            continue

        latent_path = latents_dir / f"{sample_id}.safetensors"
        text_path = text_dir / f"{sample_id}.safetensors"
        meta_path = latents_dir / f"{sample_id}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing {meta_path}")

        latent_nhwc = load_file(str(latent_path), device="cpu")["latents"]
        if latent_nhwc.ndim != 4 or latent_nhwc.shape[0] != 1 or latent_nhwc.shape[3] != 16:
            raise RuntimeError(f"{latent_path}: expected latents [1,H,W,16], got {tuple(latent_nhwc.shape)}")
        latent_chw = latent_nhwc.squeeze(0).permute(2, 0, 1).contiguous().to(torch.bfloat16)

        t5_text = load_file(str(text_path), device="cpu")["text"]
        if t5_text.ndim != 2 or t5_text.shape[1] != 4096:
            raise RuntimeError(f"{text_path}: expected text [seq,4096], got {tuple(t5_text.shape)}")
        t5_embed = t5_text.unsqueeze(0).contiguous().to(torch.bfloat16)

        prompt = _caption_for(meta_path, args.dataset)
        clip_pool = _load_clip_pool(prompt, tokenizer, model, device)
        if tuple(clip_pool.shape) != (768,):
            raise RuntimeError(f"{sample_id}: expected clip_pool [768], got {tuple(clip_pool.shape)}")

        save_file(
            {
                "latent": latent_chw,
                "t5_embed": t5_embed,
                "clip_pool": clip_pool,
            },
            str(out_path),
            metadata={
                "source": "EriDiffusion split .flux_latents/.flux_te plus local FLUX CLIP pooler",
                "sample_id": sample_id,
                "dtype": "bf16",
            },
        )
        manifest.append(
            {
                "sample_id": sample_id,
                "source_latent": str(latent_path),
                "source_text": str(text_path),
                "output": str(out_path),
            }
        )

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("/home/alex/EriDiffusion/datasets/40_woman"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/home/alex/EriDiffusion/EriDiffusion-v2/cache/flux_40_woman_512_mojo"),
    )
    p.add_argument("--clip-model", type=Path, default=Path(DEFAULT_MODEL))
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
