#!/usr/bin/env python
# prep_ti2v.py — image-template tokenization for VLM-grounded ti2v.
# Produces the Qwen3-VL processor outputs the Mojo vision+fuse path needs:
#   input_ids (with <|image_pad|> 151655), pixel_values [seq,1536], image_grid_thw,
#   crop_start. Also copies the video-res condition image (for the VAE frame-0 seed,
#   loaded separately in Mojo). This processor step is the one non-Mojo bit of the
#   VLM-grounded path (vision preprocess/tokenization); everything downstream is Mojo.
import os, sys, json
import numpy as np, torch
from safetensors.torch import save_file
from transformers import AutoProcessor

OUT = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
sys.path.insert(0, OUT); sys.path.insert(0, "/mnt/disk1/lingbot-src/lingbot-video")
from oracle_e_t2i import PROMPT_TEMPLATE, _compute_crop_start, TOKEN_LENGTH, PROC_DIR
from lingbot_video.pipeline_lingbot_video import IMG_PROMPT_TEMPLATE, DEFAULT_NEGATIVE_PROMPT

IMG = sys.argv[1] if len(sys.argv) > 1 else "/home/alex/.claude/uploads/6fd4828e-ac2b-4c91-a10b-c346b1b61e18/8280dc5c-1000004560.webp"
PROMPT_FILE = sys.argv[2] if len(sys.argv) > 2 else f"{OUT}/i2v_prompt.txt"
IMAGE_TOKEN_ID = 151655


def main():
    from PIL import Image
    proc = AutoProcessor.from_pretrained(PROC_DIR, trust_remote_code=True)
    crop_start = _compute_crop_start(proc)
    prompt = open(PROMPT_FILE).read()
    # image template goes in the user slot (IMG_PROMPT_TEMPLATE = vision_start+image_pad+vision_end)
    text = PROMPT_TEMPLATE.format(IMG_PROMPT_TEMPLATE + prompt)
    image = Image.open(IMG).convert("RGB")
    inputs = proc(text=[text], images=[image], videos=None, do_resize=True,
                  truncation=True, max_length=TOKEN_LENGTH, padding="longest", return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    pixel_values = inputs["pixel_values"]              # [seq,1536]
    grid = inputs["image_grid_thw"][0].tolist()        # [t,h,w]
    n_img = int((input_ids == IMAGE_TOKEN_ID).sum())
    L = int(input_ids.shape[0]); post = L - crop_start
    seq = int(pixel_values.shape[0])
    # negative is TEXT-ONLY (same template, no image)
    neg_ids = proc(text=[PROMPT_TEMPLATE.format(DEFAULT_NEGATIVE_PROMPT)], images=None, videos=None,
                   do_resize=False, truncation=True, max_length=TOKEN_LENGTH, padding="longest",
                   return_tensors="pt")["input_ids"][0]
    save_file({
        "input_ids": input_ids.to(torch.int32).contiguous(),
        "pixel_values": pixel_values.float().contiguous(),
        "neg_ids": neg_ids.to(torch.int32).contiguous(),
        "crop_start": torch.tensor([crop_start], dtype=torch.int32),
    }, f"{OUT}/ti2v_inputs.safetensors")
    json.dump({"grid_t": grid[0], "grid_h": grid[1], "grid_w": grid[2], "vision_seq": seq,
               "crop_start": crop_start, "L_cond_post": post, "n_image_tokens": n_img,
               "L_uncond_post": int(neg_ids.shape[0]) - crop_start},
              open(f"{OUT}/ti2v_meta.json", "w"), indent=2)
    print(f"[ti2v] grid_thw={grid} vision_seq={seq} n_image_tokens={n_img} "
          f"L_COND(post)={post} L_UNCOND(post)={int(neg_ids.shape[0])-crop_start} crop_start={crop_start}")


if __name__ == "__main__":
    main()
