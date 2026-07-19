#!/usr/bin/env python
# qwen3vl_fuse_oracle.py — torch reference for the Qwen3-VL VISION->TEXT FUSION
# (LingBot-Video image-grounded conditioning). Loads the full Qwen3VLModel from
# the LingBot text_encoder snapshot, runs the AutoProcessor on the fixed test
# image + a text prompt, then reimplements the fused forward inline (piece taps)
# and dumps the FUSED last_hidden_state plus every intermediate so the Mojo
# probe can bisect masked_scatter / mrope-positions / mrope-interleave / deepstack.
#
# Taps (float32 unless noted):
#   input_ids              [L]        int32   full input_ids (fed verbatim to Mojo)
#   pixel_values           [seq,1536]         processor output
#   image_grid_thw         [1,3]      int32
#   position_ids           [3,L]      int32   get_rope_index (T,H,W)
#   image_embeds           [nvis,2560]        vision pooler (cat)
#   deepstack_0/1/2        [nvis,2560]        vision deepstack features
#   inputs_embeds_scatter  [1,L,2560]         embeds after masked_scatter
#   last_hidden_state      [1,L,2560]         FUSED post-model.norm (the gate)
#
# Run: /home/alex/SerenityTrainer/venv/bin/python \
#        serenitymojo/models/text_encoder/parity/qwen3vl_fuse_oracle.py

import json
import os

import torch
from safetensors.torch import save_file

TE_DIR = "/mnt/disk1/models/lingbot-video-moe/text_encoder"
IMAGE = "/home/alex/.claude/uploads/6fd4828e-ac2b-4c91-a10b-c346b1b61e18/8280dc5c-1000004560.webp"
OUT = os.environ.get(
    "ORACLE_OUT",
    os.path.join(os.path.dirname(__file__), "qwen3vl_fuse_oracle.safetensors"),
)
IMAGE_TOKEN_ID = 151655


@torch.no_grad()
def main():
    dev = "cuda"
    dt = torch.float32 if os.environ.get("ORACLE_DTYPE", "bf16") == "f32" else torch.bfloat16
    print(f"[oracle] compute dtype = {dt}")

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from transformers import AutoProcessor
    from PIL import Image

    model = Qwen3VLModel.from_pretrained(TE_DIR, dtype=dt).to(dev).eval()

    processor = AutoProcessor.from_pretrained(TE_DIR)
    img = Image.open(IMAGE).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this image."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(dev)
    input_ids = inputs["input_ids"]                 # [1, L]
    pixel_values = inputs["pixel_values"].to(dt)    # [seq, 1536]
    grid_thw = inputs["image_grid_thw"]             # [n, 3]
    L = input_ids.shape[1]
    n_img = int((input_ids[0] == IMAGE_TOKEN_ID).sum().item())
    print(f"[oracle] L={L} image_pad_count={n_img} grid_thw={grid_thw.tolist()}")

    mm_token_type_ids = (input_ids == IMAGE_TOKEN_ID).long()  # image=1, text=0

    dumps = {}
    dumps["input_ids"] = input_ids[0].to(torch.int32).cpu().contiguous()
    dumps["input_ids_f32"] = input_ids[0].float().cpu().contiguous()  # exact; Mojo reads via to_host
    dumps["pixel_values"] = pixel_values.float().cpu().contiguous()
    dumps["image_grid_thw"] = grid_thw.to(torch.int32).cpu().contiguous()

    # ---- inline fused forward (mirrors Qwen3VLModel.forward:1281-1355) ----
    inputs_embeds = model.get_input_embeddings()(input_ids)

    img_out = model.get_image_features(pixel_values, grid_thw)
    image_embeds = torch.cat(img_out.pooler_output, dim=0)          # [nvis, 2560]
    deepstack = img_out.deepstack_features                          # list of 3
    dumps["image_embeds"] = image_embeds.float().cpu().contiguous()
    for i, f in enumerate(deepstack):
        dumps[f"deepstack_{i}"] = f.float().cpu().contiguous()

    image_mask = (input_ids == IMAGE_TOKEN_ID)                      # [1, L]
    mask_exp = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    inputs_embeds = inputs_embeds.masked_scatter(
        mask_exp, image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    )
    dumps["inputs_embeds_scatter"] = inputs_embeds.float().cpu().contiguous()

    position_ids, _ = model.get_rope_index(
        input_ids, mm_token_type_ids, image_grid_thw=grid_thw
    )                                                              # [3, 1, L]
    dumps["position_ids"] = position_ids[:, 0, :].to(torch.int32).cpu().contiguous()

    out = model.language_model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        visual_pos_masks=image_mask,
        deepstack_visual_embeds=deepstack,
    )
    last_hidden = out.last_hidden_state                            # [1, L, 2560]
    dumps["last_hidden_state"] = last_hidden.float().cpu().contiguous()

    for k, v in dumps.items():
        print(f"  {k:24s} {tuple(v.shape)} {v.dtype}")
    save_file(dumps, OUT)
    print(f"[oracle] wrote {OUT}")


if __name__ == "__main__":
    main()
