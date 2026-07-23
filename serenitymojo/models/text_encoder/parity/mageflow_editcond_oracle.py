#!/usr/bin/env python
# mageflow_editcond_oracle.py — torch reference for the Mage-Flow EDIT (image-
# conditioned) DiT context: image + instruction -> Qwen3-VL (vision tower ->
# deepstack -> text decoder) -> the [1, L-64, 2560] `txt` the Mage DiT consumes.
#
# Faithfully replicates mage_flow/pipeline.py generate_edits -> _encode_edits_packed
# (:396-417) + models/modules/text_encoder.py TextEncoder.forward:
#   1. resize the ref image long-edge to 384 (vl_cond_long_edge, pipeline.py:533)
#   2. formatted = template("mage-flow-edit").format(
#        "Image 1: <|vision_start|><|image_pad|><|vision_end|>" + instruction)
#   3. processor(text=[formatted], images=[ref]) -> input_ids, pixel_values,
#      image_grid_thw   (image_pad expanded to nvis tokens by the Qwen3-VL processor)
#   4. TextEncoder.forward: position_ids = arange(L) (PLAIN, 1D -> 3 identical
#      mrope axes; HF get_rope_index is SKIPPED because position_ids is supplied,
#      modeling_qwen3_vl.py:1335), single-seq causal == Mage's varlen-causal.
#   5. last_hidden_state POST model.norm -> drop the leading 64 system rows -> txt.
#
# Runs on CPU in float32 (NOT GPU bf16): the fused last_hidden_state's image-token
# rows are ill-conditioned in bf16 (HF's own bf16-vs-f32 cos there ~0.95), so — as
# for the LingBot Qwen3-VL fuse gate — the fusion MATH is only gateable at
# cos>=0.999 in F32, and an F32 full model (~17.8 GB) does not fit the 16 GB GPU.
# CPU f32 gives a clean reference AND leaves the shared GPU free for the Mojo probe.
#
# Dumps (all float32 unless noted):
#   input_ids_f32     [L]           exact ids via f32 carrier (Mojo reads to_host)
#   pixel_values      [seq,1536]    processor output (fed verbatim to the Mojo vision tower)
#   image_grid_thw    [1,3]  int32  (t,h,w)  -> comptime S=t*h*w for the probe
#   vision_pooler     [nvis,2560]   vision pooler_output (G-vision target)
#   deepstack_0/1/2   [nvis,2560]   vision deepstack features (fed to isolated fuse)
#   editcond_txt      [L-64,2560]   the DiT context res["txt"] (G-editcond target)
#   last_hidden_state [1,L,2560]    full fused post-norm (debug)
#
# Run: pixi run python \
#        serenitymojo/models/text_encoder/parity/mageflow_editcond_oracle.py

import os

import torch
from PIL import Image
from safetensors.torch import save_file

TE_DIR = "/home/alex/.serenity/models/checkpoints/Mage-Flow-Edit-Turbo/text_encoder"
IMAGE = "/home/alex/Mage/mage_flow/assets/dog.jpg"
OUT = os.environ.get(
    "ORACLE_OUT",
    os.path.join(os.path.dirname(__file__), "mageflow_editcond_oracle.safetensors"),
)
IMAGE_TOKEN_ID = 151655
EDIT_DROP_IDX = 64
VL_COND_LONG_EDGE = 384

# mage-flow-edit template (mage_flow/models/utils.py PROMPT_TEMPLATE, start_idx 64).
EDIT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture,"
    " objects, background), then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency with the original "
    "input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
_EDIT_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
INSTRUCTION = "Change the background to a snowy forest."


def _resize_long_edge(image: Image.Image, max_long_edge: int) -> Image.Image:
    # pipeline.py:_resize_long_edge — cap long edge, BICUBIC (matches training).
    if max_long_edge is None or max_long_edge <= 0:
        return image
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)


@torch.no_grad()
def main():
    dev = "cpu"
    dt = torch.float32
    print(f"[oracle] device={dev} dtype={dt} (F32 fusion-math reference)")

    from transformers import AutoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    model = Qwen3VLModel.from_pretrained(TE_DIR, dtype=dt, low_cpu_mem_usage=True).eval()
    processor = AutoProcessor.from_pretrained(TE_DIR)

    # ---- faithful _encode_edits_packed for a single (image, instruction) ----
    img = _resize_long_edge(Image.open(IMAGE).convert("RGB"), VL_COND_LONG_EDGE)
    body = f"Image 1: {_EDIT_IMAGE_PLACEHOLDER}" + INSTRUCTION
    formatted = EDIT_TEMPLATE.format(body)
    vl = processor(text=[formatted], images=[img], padding=True, return_tensors="pt")

    input_ids = vl["input_ids"]                       # [1, L]
    pixel_values = vl["pixel_values"].to(dt)          # [seq, 1536]
    grid_thw = vl["image_grid_thw"]                   # [1, 3]
    L = input_ids.shape[1]
    t, gh, gw = [int(x) for x in grid_thw[0].tolist()]
    seq = t * gh * gw
    nvis = int((input_ids[0] == IMAGE_TOKEN_ID).sum().item())
    print(f"[oracle] L={L} seq(patch)={seq} grid_thw=({t},{gh},{gw}) "
          f"nvis(image_pad)={nvis} merged={gh//2}x{gw//2}={(gh//2)*(gw//2)}")
    assert nvis == (gh // 2) * (gw // 2), "image_pad count must equal merged grid"
    assert L > EDIT_DROP_IDX, "L must exceed the 64-row system prefix"

    dumps = {}
    dumps["input_ids_f32"] = input_ids[0].float().cpu().contiguous()
    dumps["pixel_values"] = pixel_values.float().cpu().contiguous()
    dumps["image_grid_thw"] = grid_thw.to(torch.int32).cpu().contiguous()

    # ---- vision tower: pooler (G-vision target) + deepstack features ----
    img_out = model.get_image_features(pixel_values, grid_thw)
    vision_pooler = torch.cat(img_out.pooler_output, dim=0)       # [nvis, 2560]
    deepstack = img_out.deepstack_features                        # list of 3
    dumps["vision_pooler"] = vision_pooler.float().cpu().contiguous()
    for i, f in enumerate(deepstack):
        dumps[f"deepstack_{i}"] = f.float().cpu().contiguous()

    # ---- fuse: masked_scatter + PLAIN-arange M-RoPE + deepstack (TextEncoder.forward) ----
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask = (input_ids == IMAGE_TOKEN_ID)                    # [1, L]
    mask_exp = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(
        mask_exp, vision_pooler.to(inputs_embeds.dtype))

    # Mage TextEncoder.forward: position_ids = arange(L).unsqueeze(0) -> [1, L].
    # (ndim 2 -> the stock text model expands to 4 axes, all identical arange ->
    # M-RoPE collapses to plain sequential RoPE.)
    position_ids = torch.arange(L).view(1, L)

    out = model.language_model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        visual_pos_masks=image_mask,
        deepstack_visual_embeds=deepstack,
    )
    last_hidden = out.last_hidden_state                          # [1, L, 2560]
    dumps["last_hidden_state"] = last_hidden.float().cpu().contiguous().clone()
    # Drop the leading 64 system-prompt rows -> the DiT context res["txt"].
    dumps["editcond_txt"] = last_hidden[0, EDIT_DROP_IDX:].float().cpu().contiguous().clone()

    for k, v in dumps.items():
        print(f"  {k:20s} {tuple(v.shape)} {v.dtype}")
    save_file(dumps, OUT)
    print(f"[oracle] wrote {OUT}")
    print(f"[oracle] SET PROBE COMPTIME: T={t} H={gh} W={gw} S={seq}")


if __name__ == "__main__":
    main()
