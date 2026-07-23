#!/usr/bin/env python3
# scripts/bernini_r_build_cache.py — OFFLINE dev tool (like the parity oracles).
#
# Builds a REAL per-sample data cache for the pure-Mojo Bernini-R conditioning
# renderer trainer (serenitymojo/training/train_bernini_r_cond.mojo). It runs the
# SHIPPED pure-Mojo encoders — it reimplements NO encoder math:
#   * VAE  : output/bin/bernini_r_encode_vae  (reuses LingBotWanVaeEncoder;
#            image RGB -> normalized Wan2.1 16-ch latent, 256->32 / 128->16).
#   * umt5 : output/bin/scail2_encode_prompt  (umt5-xxl -> [512,4096] bf16 + len).
# Each encoder runs as its OWN process, so ONE model is resident at a time (16GB):
# all VAE encodes happen first (each invocation loads+frees the VAE), then all
# umt5 encodes. Torch is used ONLY to stage RGB pixels + assemble safetensors.
#
# The caption is prefixed with the task's SYSTEM_PROMPT
# (/home/alex/Bernini/bernini/training/data.py:32-46) exactly as the upstream
# Bernini renderer trainer does, then umt5-encoded.
#
# INPUT: a manifest JSON:
#   {
#     "vae": "<vae.safetensors>",
#     "umt5_dir": "<umt5-xxl dir>",
#     "umt5_tokenizer": "<tokenizer.json>",
#     "cond_res": 256,                       # optional; conditioning encode res
#     "target_res": 256,                     # optional; target encode res
#     "samples": [
#       {"task": "t2v", "caption": "...", "target": "img.jpg", "conditioning": []},
#       {"task": "i2v", "caption": "...", "target": "img.jpg",
#        "conditioning": ["ref.jpg"]},
#       ...
#     ]
#   }
# A target/conditioning entry may take any of these forms:
#   "img.jpg"                              single RGB image  -> encode() [16,1,h,w]
#   {"frames": ["f0.jpg","f1.jpg",...]}    multi-frame clip  -> encode_video()
#                                          [16,Tlat,h,w], Tlat=1+(F-1)//4  (VIDEO)
#   {"synth_frames": N, "res": 128}        N synthesized frames (no assets needed)
#                                          -> encode_video() (gate smoke helper)
#   {"latent": "<sample.safetensors>", "key": "latent"}   reuse an EXISTING on-disk
#                                          latent [16,F,H,W] or [1,16,F,H,W] (no
#                                          re-encode; gate-sanctioned reuse path).
# A sample may also carry {"reuse_text": "<file>", "text_key": "text",
# "len_key": "text_len"} to copy a previously-encoded umt5 [512,4096] text tensor
# instead of re-running umt5 (lets a gate cache be built without the umt5 weights).
#
# OUTPUT (one file per sample) under <out_dir>:
#   sample_{i}.safetensors, keys:
#     task_id   : i32  [1]              task index 0..11 (see TASK_IDS)
#     latent    : f32  [16,F,H,W]       normalized target latent (Wan2.1 16-ch)
#     cond_count: i32  [1]              N conditioning segments
#     cond_{k}  : f32  [16,f,h,w]       CLEAN conditioning latent  (k=0..N-1)
#     cond_{k}_src_id : i32 [1]         source id = k+1 (target is 0)
#     text      : bf16 [512,4096]       umt5 of SYSTEM_PROMPT+caption
#     text_len  : i32  [1]              true umt5 token count (<=512)
#   manifest.json : {"samples":[{"task":..,"file":..,"cond_count":..}, ...]}
#
# USAGE:
#   /home/alex/ai-toolkit/venv/bin/python scripts/bernini_r_build_cache.py \
#       --manifest <manifest.json> --out-dir <cache_dir>

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file, load_file

# ── task -> integer id (stable order == training/bernini_tasks.mojo recipe set) ─
TASK_IDS = {
    "t2i": 0, "t2v": 1, "i2i": 2, "r2i": 3, "r2v": 4, "v2v": 5, "i2v": 6,
    "vi2v": 7, "vr2v": 8, "vrc2v": 9, "mv2v": 10, "ads2v": 11,
}

# SYSTEM_PROMPTS — verbatim from bernini/training/data.py:32-46.
SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "r2i": "You are a helpful assistant specialized in subject-to-image generation.",
    "i2v": "You are a helpful assistant specialized in image-to-video generation.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "vi2v": "You are a helpful assistant specialized in video editing on content propagation.",
    "vr2v": "You are a helpful assistant specialized in video editing with reference.",
    "ads2v": "You are a helpful assistant specialized in ads insertion.",
    "vrc2v": "You are a helpful assistant for editing. You may need to adjust the subject's action or position.",
    "mv2v": "You are a helpful assistant for editing. You might need to adjust the video's style, lighting, colors, textures, and the subject's pose or action.",
}

REPO = Path(__file__).resolve().parent.parent
VAE_BIN = REPO / "output/bin/bernini_r_encode_vae"
PROMPT_BIN = REPO / "output/bin/scail2_encode_prompt"

TEXT_LEN = 512
TEXT_DIM = 4096


def prompt_clean(text: str) -> str:
    """Light approximation of diffusers prompt_clean (whitespace normalize)."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stage_pixel(image_path: Path, res: int, tmp: Path) -> Path:
    """RGB image -> center-crop square -> resize res -> [0,1] -> NCTHW pixel."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((res, res), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0     # [H,W,C] in [0,1]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()   # [C,H,W]
    t = t.unsqueeze(0).unsqueeze(2).contiguous()              # [1,3,1,H,W]
    out = tmp / f"stage_{image_path.stem}_{res}.safetensors"
    save_file({"pixel": t}, str(out))
    return out


def stage_clip(frames: list[Path], res: int, tmp: Path, tag: str, idx: int) -> Path:
    """List of RGB image paths -> center-crop square -> resize res -> NCTHW clip
    [1,3,F,res,res] in [0,1]."""
    ts = []
    for fp in frames:
        img = Image.open(fp).convert("RGB")
        w, h = img.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        img = img.crop((left, top, left + s, top + s)).resize((res, res), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0        # [H,W,C]
        ts.append(torch.from_numpy(arr).permute(2, 0, 1))      # [C,H,W]
    clip = torch.stack(ts, dim=1).unsqueeze(0).contiguous()    # [1,3,F,res,res]
    out = tmp / f"clip_{idx}_{tag}_{res}.safetensors"
    save_file({"pixel": clip}, str(out))
    return out


def stage_synth_clip(n: int, res: int, tmp: Path, tag: str, idx: int) -> Path:
    """Synthesize N smooth-ish random frames -> NCTHW clip [1,3,N,res,res]. Used by
    gate smokes that exercise the multi-frame VAE path without needing assets."""
    g = torch.Generator().manual_seed(1234 + idx * 17 + hash(tag) % 997)
    base = torch.rand(3, 1, res, res, generator=g)
    frames = []
    for f in range(n):
        frames.append((base + 0.04 * f + 0.08 * torch.randn(3, 1, res, res, generator=g)).clamp(0, 1))
    clip = torch.cat(frames, dim=1).unsqueeze(0).contiguous()  # [1,3,N,res,res]
    out = tmp / f"synth_{idx}_{tag}_{res}.safetensors"
    save_file({"pixel": clip}, str(out))
    return out


def run(cmd: list[str]) -> None:
    print("[cache]   $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def encode_vae(entry, res: int, vae: str, tmp: Path, idx: int, tag: str) -> torch.Tensor:
    """Return a [16,F,H,W] f32 latent for an image path, a multi-frame clip, a
    synthesized clip, or a reuse-latent dict. F>1 <=> VIDEO conditioning/target."""
    # --- reuse an existing on-disk latent (no re-encode) ---
    if isinstance(entry, dict) and "latent" in entry:
        key = entry.get("key", "latent")
        t = load_file(entry["latent"])[key].float()
        if t.ndim == 5 and t.shape[0] == 1:      # [1,16,F,H,W] -> [16,F,H,W]
            t = t.squeeze(0)
        if t.ndim != 4 or t.shape[0] != 16:
            raise ValueError(f"reuse latent {entry['latent']}[{key}] must be [16,F,H,W], got {tuple(t.shape)}")
        print(f"[cache] sample {idx} {tag}: REUSE latent {tuple(t.shape)} "
              f"std={t.std().item():.4f} <- {entry['latent']}")
        return t.contiguous()

    # --- multi-frame clip (real frames or synthesized) -> encode_video ---
    if isinstance(entry, dict) and ("frames" in entry or "synth_frames" in entry):
        vres = int(entry.get("res", res))
        if "frames" in entry:
            staged = stage_clip([Path(p) for p in entry["frames"]], vres, tmp, tag, idx)
            nframe = len(entry["frames"])
            srcdesc = f"{nframe} frames"
        else:
            nframe = int(entry["synth_frames"])
            staged = stage_synth_clip(nframe, vres, tmp, tag, idx)
            srcdesc = f"synth x{nframe}"
        out = tmp / f"lat_{idx}_{tag}.safetensors"
        run([VAE_BIN, staged, vae, out])
        lat = load_file(str(out))["latent"].float()   # [1,16,Tlat,h,w]
        if lat.ndim == 5 and lat.shape[0] == 1:
            lat = lat.squeeze(0)                       # [16,Tlat,h,w]
        print(f"[cache] sample {idx} {tag}: VIDEO encoded {tuple(lat.shape)} "
              f"std={lat.std().item():.4f} <- {srcdesc} @res{vres}")
        return lat.contiguous()

    # --- single RGB image -> encode() [16,1,h,w] ---
    media = Path(entry)
    staged = stage_pixel(media, res, tmp)
    out = tmp / f"lat_{idx}_{tag}.safetensors"
    run([VAE_BIN, staged, vae, out])
    lat = load_file(str(out))["latent"].float()   # [1,16,1,h,w]
    if lat.ndim == 5 and lat.shape[0] == 1:
        lat = lat.squeeze(0)                       # [16,1,h,w]
    print(f"[cache] sample {idx} {tag}: encoded {tuple(lat.shape)} std={lat.std().item():.4f} <- {media.name}")
    return lat.contiguous()


def reuse_text(smp: dict, idx: int) -> tuple[torch.Tensor, int]:
    """Copy a previously-encoded umt5 [512,4096] text tensor + its true length from
    an existing sample file (no umt5 run). For gate caches without umt5 weights."""
    src = smp["reuse_text"]
    tkey = smp.get("text_key", "text")
    lkey = smp.get("len_key", "text_len")
    d = load_file(src)
    emb = d[tkey].float()
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    if emb.shape != (TEXT_LEN, TEXT_DIM):
        raise ValueError(f"reuse_text {src}[{tkey}] must be [512,4096], got {tuple(emb.shape)}")
    tl = int(d[lkey].flatten()[0].item()) if lkey in d else TEXT_LEN
    print(f"[cache] sample {idx}: REUSE text {tuple(emb.shape)} true_len={tl} <- {src}")
    return emb.to(torch.bfloat16).contiguous(), tl


def encode_text(task: str, caption: str, umt5_dir: str, tokenizer: str,
                tmp: Path, idx: int) -> tuple[torch.Tensor, int]:
    system = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["default"])
    prompt = system + prompt_clean(caption)
    out = tmp / f"text_{idx}.safetensors"
    run([PROMPT_BIN, umt5_dir, tokenizer, prompt, "", out])
    d = load_file(str(out))
    emb = d["pos_embed"].float()          # [1,512,4096]
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)              # [512,4096]
    if emb.shape != (TEXT_LEN, TEXT_DIM):
        raise ValueError(f"umt5 text must be [512,4096], got {tuple(emb.shape)}")
    tl = int(d["pos_len"].flatten()[0].item())
    print(f"[cache] sample {idx}: text {tuple(emb.shape)} true_len={tl} "
          f"prompt[:70]={prompt[:70]!r}")
    return emb.to(torch.bfloat16).contiguous(), tl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    spec = json.loads(args.manifest.read_text())
    vae = spec["vae"]
    umt5_dir = spec.get("umt5_dir", "")
    tokenizer = spec.get("umt5_tokenizer", "")
    target_res = int(spec.get("target_res", 256))
    cond_res = int(spec.get("cond_res", 256))
    samples = spec["samples"]

    needs_umt5 = any("reuse_text" not in smp for smp in samples)
    if not VAE_BIN.exists():
        raise SystemExit(f"missing VAE binary {VAE_BIN} (build bernini_r_encode_vae)")
    if needs_umt5 and not PROMPT_BIN.exists():
        raise SystemExit(f"missing prompt binary {PROMPT_BIN} (build scail2_encode_prompt)")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="bernini-cache-") as td:
        tmp = Path(td)

        # ── PHASE 1: VAE encode every target + conditioning latent ────────────
        per_sample_latents = []   # (target[16,F,H,W], [cond0, cond1, ...])
        for i, smp in enumerate(samples):
            tgt = encode_vae(smp["target"], target_res, vae, tmp, i, "target")
            conds = []
            for k, c in enumerate(smp.get("conditioning", []) or []):
                conds.append(encode_vae(c, cond_res, vae, tmp, i, f"cond{k}"))
            per_sample_latents.append((tgt, conds))
        print("[cache] VAE phase done (encoder freed between every invocation)")

        # ── PHASE 2: umt5 encode every caption (or reuse a cached text tensor) ──
        per_sample_text = []
        for i, smp in enumerate(samples):
            if "reuse_text" in smp:
                emb, tl = reuse_text(smp, i)
            else:
                emb, tl = encode_text(smp["task"], smp.get("caption", ""),
                                      umt5_dir, tokenizer, tmp, i)
            per_sample_text.append((emb, tl))

    # ── WRITE ─────────────────────────────────────────────────────────────────
    manifest_out = {"samples": []}
    for i, smp in enumerate(samples):
        task = smp["task"]
        if task not in TASK_IDS:
            raise SystemExit(f"unknown task '{task}' in sample {i}")
        tgt, conds = per_sample_latents[i]
        emb, tl = per_sample_text[i]
        tensors = {
            "task_id": torch.tensor([TASK_IDS[task]], dtype=torch.int32),
            "latent": tgt.to(torch.float32).contiguous(),
            "cond_count": torch.tensor([len(conds)], dtype=torch.int32),
            "text": emb.to(torch.bfloat16).contiguous(),
            "text_len": torch.tensor([tl], dtype=torch.int32),
        }
        for k, c in enumerate(conds):
            tensors[f"cond_{k}"] = c.to(torch.float32).contiguous()
            tensors[f"cond_{k}_src_id"] = torch.tensor([k + 1], dtype=torch.int32)
        path = out_dir / f"sample_{i}.safetensors"
        save_file(tensors, str(path))
        print(f"[cache] wrote {path}  task={task} cond_count={len(conds)} "
              f"target={tuple(tgt.shape)}")
        manifest_out["samples"].append(
            {"task": task, "task_id": TASK_IDS[task], "file": path.name,
             "cond_count": len(conds), "target_shape": list(tgt.shape)}
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
    print(f"[cache] DONE: {len(samples)} samples -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
