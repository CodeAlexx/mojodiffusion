#!/usr/bin/env python
# qwen3vl_vision_oracle.py — torch reference for the Qwen3-VL VISION tower
# (LingBot-Video). Builds ONLY model.visual (Qwen3VLVisionModel), loads the
# model.visual.* weights from the LingBot text_encoder safetensors shards in
# bf16 on GPU, runs the AutoProcessor on a fixed test image to get
# pixel_values + image_grid_thw, then reimplements the vision forward inline
# with per-stage taps and dumps everything to an oracle safetensors for the
# Mojo per-stage parity probes.
#
# Taps dumped (all float32):
#   pixel_values          [seq, 1536]     processor output (fed verbatim to Mojo)
#   image_grid_thw        [1, 3]  int32   (t, h, w)
#   s1_hidden_posembed    [seq, 1024]     hidden after patch_embed + pos_embed (modeling:783)
#   s2_cos / s2_sin       [seq, 64]       position_embeddings (modeling:791)
#   s3_block0_out         [seq, 1024]     output of blocks[0]
#   s4_pooler_output      [Nmerged, 2560] merger(last_hidden_state)
#   s4_deepstack_0/1/2    [Nmerged, 2560] deepstack features (blocks 5/11/17)
#
# Run: /home/alex/SerenityTrainer/venv/bin/python \
#        serenitymojo/models/text_encoder/parity/qwen3vl_vision_oracle.py

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TE_DIR = "/mnt/disk1/models/lingbot-video-moe/text_encoder"
IMAGE = "/home/alex/.claude/uploads/6fd4828e-ac2b-4c91-a10b-c346b1b61e18/8280dc5c-1000004560.webp"
OUT = os.environ.get("ORACLE_OUT", os.path.join(os.path.dirname(__file__), "qwen3vl_vision_oracle.safetensors"))

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers import AutoProcessor


def load_visual_state_dict(te_dir):
    idx_path = os.path.join(te_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)["weight_map"]
    # collect the shard file for each model.visual.* key
    shard_of = {}
    for k, shard in idx.items():
        if k.startswith("model.visual."):
            shard_of.setdefault(shard, []).append(k)
    sd = {}
    for shard, keys in shard_of.items():
        with safe_open(os.path.join(te_dir, shard), framework="pt", device="cpu") as f:
            for k in keys:
                sd[k[len("model.visual."):]] = f.get_tensor(k)
    return sd


@torch.no_grad()
def main():
    dev = "cuda"
    dt = torch.float32 if os.environ.get("ORACLE_DTYPE", "bf16") == "f32" else torch.bfloat16
    print(f"[oracle] compute dtype = {dt}")
    with open(os.path.join(TE_DIR, "config.json")) as f:
        vcfg = json.load(f)["vision_config"]
    config = Qwen3VLVisionConfig(**vcfg)
    model = Qwen3VLVisionModel(config).to(dtype=dt, device=dev).eval()

    sd = load_visual_state_dict(TE_DIR)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[oracle] loaded visual state_dict: {len(sd)} tensors; "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:8])
    if unexpected:
        print("  unexpected:", unexpected[:8])

    # ---- processor: fixed test image -> pixel_values + image_grid_thw ----
    processor = AutoProcessor.from_pretrained(TE_DIR)
    from PIL import Image
    img = Image.open(IMAGE).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this image."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(dev)            # [seq, 1536]
    grid_thw = inputs["image_grid_thw"].to(dev)             # [n, 3]
    print(f"[oracle] pixel_values {tuple(pixel_values.shape)} {pixel_values.dtype}; "
          f"grid_thw {grid_thw.tolist()}")

    dumps = {}
    dumps["pixel_values"] = pixel_values.float().cpu().contiguous()
    dumps["image_grid_thw"] = grid_thw.to(torch.int32).cpu().contiguous()

    # ---- inline forward with taps (mirrors modeling_qwen3_vl.py:767-823) ----
    import torch.nn.functional as F

    hidden_states = model.patch_embed(pixel_values.to(dt))
    pos_embeds = model.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds
    dumps["s1_hidden_posembed"] = hidden_states.float().cpu().contiguous()

    rotary_pos_emb = model.rot_pos_emb(grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    dumps["s2_cos"] = cos.float().cpu().contiguous()
    dumps["s2_sin"] = sin.float().cpu().contiguous()
    position_embeddings = (cos, sin)

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(model.blocks):
        hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens,
                            position_embeddings=position_embeddings)
        if layer_num == 0:
            dumps["s3_block0_out"] = hidden_states.float().cpu().contiguous()
        if layer_num in model.deepstack_visual_indexes:
            di = model.deepstack_visual_indexes.index(layer_num)
            feat = model.deepstack_merger_list[di](hidden_states)
            deepstack_feature_lists.append(feat)

    merged = model.merger(hidden_states)
    dumps["s4_pooler_output"] = merged.float().cpu().contiguous()
    for i, feat in enumerate(deepstack_feature_lists):
        dumps[f"s4_deepstack_{i}"] = feat.float().cpu().contiguous()

    for k, v in dumps.items():
        print(f"  {k:22s} {tuple(v.shape)} {v.dtype}")

    save_file(dumps, OUT)
    print(f"[oracle] wrote {OUT}")


if __name__ == "__main__":
    main()
