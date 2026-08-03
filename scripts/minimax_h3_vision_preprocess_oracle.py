"""MiniMax-H3 conditioner IMAGE PREPROCESSOR oracle — the REAL processor.

CPU only, no model, no CUDA. Reference for
serenitymojo/pipeline/parity/minimax_h3_vision_preprocess_probe.mojo.

Dumps, for each CANVAS size a real request resolves to, the processor's own
`pixel_values` ([num_patches, 1536]) and `image_grid_thw`, plus the exact input
image bytes so the Mojo side preprocesses the SAME pixels.

Canvas sizes only: `resolve_canvas_size` always emits a 768 short edge, so every
real keyframe is >= 589,824 px and `smart_resize` is the identity. A sub-65,536
px case is included DELIBERATELY as a negative control — the Mojo module must
REFUSE it (its resampler is unimplemented), and the probe checks that it does.

Run:
    CUDA_VISIBLE_DEVICES="" python3 scripts/minimax_h3_vision_preprocess_oracle.py
Writes: output/minimax_h3_keyframe/vision_preprocess_ref.safetensors
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np, torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

H3 = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA"
OUT_DIR = "/home/alex/mojodiffusion/output/minimax_h3_keyframe"
# (h, w, is_real_canvas)
CASES = [(768, 1184, True), (768, 1344, True), (1184, 768, True),
         (480, 832, True), (768, 768, True), (128, 192, False)]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ip = AutoProcessor.from_pretrained(os.path.join(H3, "processor")).image_processor
    P, M = ip.patch_size, ip.merge_size
    tensors, meta = {}, []
    for i, (h, w, real) in enumerate(CASES):
        rng = np.random.default_rng(1000 + i)
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        rh, rw = smart_resize(h, w, factor=P * M,
                              min_pixels=ip.size["shortest_edge"],
                              max_pixels=ip.size["longest_edge"])
        identity = (rh, rw) == (h, w)
        tensors[f"img_{i}"] = torch.from_numpy(arr.copy())
        if identity:
            out = ip(images=[Image.fromarray(arr, "RGB")], return_tensors="pt")
            tensors[f"px_{i}"] = out["pixel_values"].to(torch.float32)
            tensors[f"grid_{i}"] = out["image_grid_thw"].to(torch.int32)
        meta.append((i, h, w, int(identity)))
        print(f"  case {i}: {w}x{h} -> smart_resize {rw}x{rh}  identity={identity}"
              f"  real_canvas={real}"
              + (f"  pixel_values {tuple(tensors[f'px_{i}'].shape)}" if identity else "  (probe must REFUSE)"))
    tensors["meta"] = torch.tensor(meta, dtype=torch.int32)
    path = os.path.join(OUT_DIR, "vision_preprocess_ref.safetensors")
    save_file({k: v.contiguous() for k, v in tensors.items()}, path)
    print("wrote", path)

if __name__ == "__main__":
    main()
