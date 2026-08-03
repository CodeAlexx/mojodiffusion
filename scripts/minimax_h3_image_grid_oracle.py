"""MiniMax-H3 conditioner image-grid oracle.

Reference: the REAL Qwen3-VL image processor, configured from the
`preprocessor_config.json` we fetched with the conditioner —
/home/alex/minimax_h3_ref/creator-MiniMax-H3/FL2VA/processor.

  transformers/models/qwen2_vl/image_processing_qwen2_vl.py  smart_resize
  transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py
    Qwen2VLImageProcessorFast  (size.shortest_edge -> min_pixels,
                                size.longest_edge  -> max_pixels)

This fixes the number of vision tokens a keyframe or reference image
contributes, which unit 7's presentation takes as an input:

    grid = (1, h_bar // patch_size, w_bar // patch_size)
    num_vision_tokens = prod(grid) // merge_size ** 2

Get it wrong and the presentation emits the wrong number of `<|image_pad|>`
rows, which shifts every row index after it in the packed sequence.

WORTH NOTING: the released config is NOT the Qwen2-VL default. It carries
`shortest_edge = 65536` (256x256) and `longest_edge = 16777216` (4096x4096),
where the ComfyUI implementation hardcodes 3136 and 12845056. Anything derived
from those hardcoded numbers is wrong for this checkpoint.

The end-to-end section runs the processor itself on synthetic images so the
`smart_resize` sweep is anchored to what the real object produces, not only to
the function in isolation.

Run:
    python3 scripts/minimax_h3_image_grid_oracle.py
Writes: output/minimax_h3_image_grid/image_grid_ref.safetensors
"""

import json
import os

CONDITIONER = "/home/alex/minimax_h3_ref/creator-MiniMax-H3/FL2VA/processor"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_image_grid"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoImageProcessor  # noqa: E402
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize  # noqa: E402

# Source sizes: below the minimum, in range, above the maximum, extreme aspect
# ratios, odd numbers, and the H3 canvases themselves.
SIZES = [
    (64, 64), (100, 100), (128, 128), (255, 255), (256, 256), (257, 257),
    (333, 777), (768, 1344), (1344, 768), (1080, 1920), (1920, 1080),
    (2048, 2048), (4096, 4096), (5000, 3000), (8192, 2048),
    (31, 31), (32, 32), (33, 47), (1, 100), (100, 1),
    (2048, 512), (512, 2048), (999, 1001), (17, 4096),
]

# Sizes small enough to push through the real processor cheaply.
END_TO_END = [(64, 64), (128, 96), (256, 256), (333, 200), (512, 384)]


def main() -> None:
    processor = AutoImageProcessor.from_pretrained(CONDITIONER)
    patch_size = processor.patch_size
    merge_size = processor.merge_size
    factor = patch_size * merge_size
    min_pixels = processor.size["shortest_edge"]
    max_pixels = processor.size["longest_edge"]

    config = {
        "patch_size": patch_size,
        "merge_size": merge_size,
        "temporal_patch_size": processor.temporal_patch_size,
        "factor": factor,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "image_processor_type": type(processor).__name__,
    }
    print("resolved config:", json.dumps(config))

    tensors: dict[str, torch.Tensor] = {}
    tensors["config"] = torch.tensor(
        [patch_size, merge_size, processor.temporal_patch_size, factor, min_pixels, max_pixels],
        dtype=torch.int64,
    )

    # `smart_resize` REFUSES an absolute aspect ratio above 200. Recorded as
    # [-1, -1] / -1 so the port is gated on raising in the same places.
    sizes, resized, tokens = [], [], []
    for height, width in SIZES:
        sizes.append([height, width])
        try:
            h_bar, w_bar = smart_resize(
                height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
            )
        except ValueError:
            resized.append([-1, -1])
            tokens.append(-1)
            continue
        grid_h, grid_w = h_bar // patch_size, w_bar // patch_size
        resized.append([h_bar, w_bar])
        tokens.append(grid_h * grid_w // (merge_size**2))
    tensors["sizes"] = torch.tensor(sizes, dtype=torch.int64)
    tensors["resized"] = torch.tensor(resized, dtype=torch.int64)
    tensors["vision_tokens"] = torch.tensor(tokens, dtype=torch.int64)

    # End-to-end: the real processor's own grid, to anchor the sweep above.
    e2e_sizes, e2e_grid, e2e_tokens = [], [], []
    for height, width in END_TO_END:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        out = processor(images=[image], return_tensors="pt")
        grid = out["image_grid_thw"][0].tolist()
        e2e_sizes.append([height, width])
        e2e_grid.append(grid)
        e2e_tokens.append(int(np.prod(grid)) // (merge_size**2))
    tensors["e2e.sizes"] = torch.tensor(e2e_sizes, dtype=torch.int64)
    tensors["e2e.grid_thw"] = torch.tensor(e2e_grid, dtype=torch.int64)
    tensors["e2e.vision_tokens"] = torch.tensor(e2e_tokens, dtype=torch.int64)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "image_grid_ref.safetensors")
    save_file(tensors, path, metadata={"format": "pt"})
    with open(os.path.join(OUT_DIR, "image_grid_ref.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"wrote {len(tensors)} tensors -> {path}")
    print(f"{'source':>12}  {'resized':>12}  tokens")
    for (h, w), (hb, wb), t in zip(SIZES, resized, tokens):
        if hb < 0:
            print(f"{h:>5}x{w:<6}  {'REFUSED':>12}  (aspect ratio > 200)")
        else:
            print(f"{h:>5}x{w:<6}  {hb:>5}x{wb:<6}  {t}")
    print("end-to-end (real processor):")
    for (h, w), grid, t in zip(END_TO_END, e2e_grid, e2e_tokens):
        print(f"  {h}x{w} -> grid_thw {grid} -> {t} tokens")


if __name__ == "__main__":
    main()
