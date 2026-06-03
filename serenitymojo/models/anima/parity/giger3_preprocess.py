#!/usr/bin/env python3
# serenitymojo/models/anima/parity/giger3_preprocess.py
#
# DEV-TIME sidecar for the giger3 (gigerver3) LoRA dataset prep (Mojo path).
#
# For each NN.jpg + NN.txt pair in the dataset it produces, into <out_dir>:
#   <id>.png                512x512 RGB (resize-shorter-side then center crop)
#   <id>_tokens.safetensors qwen_input_ids[512], qwen_attention_mask[512],
#                           t5_input_ids[512]  (F32-encoded ints)
#
# These are the SAME two HF tokenizers OneTrainer/Anima uses (Qwen2Tokenizer for
# the Qwen3 encoder input, T5TokenizerFast for the adapter query ids), tokenizing
# the FULL caption text at max_length=512 — identical to
# anima_text_context_tokens.py, just batched over the dataset. The Mojo prepare
# (pipeline/giger3_prepare.mojo) then does the pure-Mojo VAE encode + Qwen3 +
# adapter compute. No latents/context are produced here.
#
# Run:
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/anima/parity/giger3_preprocess.py \
#       /home/alex/1/datasets/gigerver3 /home/alex/mojodiffusion/output/giger3_pre

import os
import sys
import glob
import torch
from PIL import Image
from transformers import AutoTokenizer, T5TokenizerFast
from safetensors.torch import save_file

PROMPT_MAX_LENGTH = 512
TARGET = 512


def resize_center_crop(im: Image.Image, size: int) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    # resize so the shorter side == size, preserving aspect, then center crop
    scale = size / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    im = im.resize((nw, nh), Image.LANCZOS)
    left = (nw - size) // 2
    top = (nh - size) // 2
    return im.crop((left, top, left + size, top + size))


def main():
    ds_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/alex/1/datasets/gigerver3"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/alex/mojodiffusion/output/giger3_pre"
    os.makedirs(out_dir, exist_ok=True)

    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    t5tok = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")

    jpgs = sorted(glob.glob(os.path.join(ds_dir, "*.jpg")),
                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not jpgs:
        print("ERROR: no .jpg found in", ds_dir)
        sys.exit(1)

    ids = []
    for jpg in jpgs:
        stem = os.path.splitext(os.path.basename(jpg))[0]
        txt = os.path.join(ds_dir, stem + ".txt")
        if not os.path.isfile(txt):
            print("WARN: no caption for", stem, "- skipping")
            continue
        with open(txt, "r") as f:
            prompt = f.read().strip()

        # 1) image -> 512x512 RGB PNG
        try:
            im = Image.open(jpg)
            im = resize_center_crop(im, TARGET)
            im.save(os.path.join(out_dir, stem + ".png"))
        except Exception as e:
            print("FAIL image", stem, ":", e)
            continue

        # 2) tokens sidecar (Qwen3 + T5 @512)
        q = qtok([prompt], max_length=PROMPT_MAX_LENGTH, padding="max_length",
                 truncation=True, return_tensors="pt")
        t = t5tok([prompt], max_length=PROMPT_MAX_LENGTH, padding="max_length",
                  truncation=True, return_tensors="pt")
        qids = q.input_ids[0].to(torch.float32)
        qmask = q.attention_mask[0].to(torch.float32)
        t5ids = t.input_ids[0].to(torch.float32)
        save_file({
            "qwen_input_ids": qids.contiguous(),
            "qwen_attention_mask": qmask.contiguous(),
            "t5_input_ids": t5ids.contiguous(),
        }, os.path.join(out_dir, stem + "_tokens.safetensors"))
        ids.append(stem)
        print("ok", stem, "| qwen nonpad", int(qmask.sum().item()),
              "| t5 nonpad", int((t.attention_mask[0] != 0).sum().item()))

    # write an id manifest the Mojo prepare iterates
    with open(os.path.join(out_dir, "ids.txt"), "w") as f:
        for s in ids:
            f.write(s + "\n")
    print("wrote", len(ids), "preprocessed pairs ->", out_dir)
    print("manifest:", os.path.join(out_dir, "ids.txt"))


if __name__ == "__main__":
    main()
