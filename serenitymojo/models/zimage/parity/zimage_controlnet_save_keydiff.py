#!/usr/bin/env python3.12
# T2.E GATE (d): the trainer-saved Z-Image ControlNet checkpoint must match the
# diffusers ZImageControlNetModel state-dict schema EXACTLY (keys + shapes).
# The reference model is instantiated from the saved config.json on the meta
# device (no weight alloc) and its state_dict keys/shapes are diffed against
# the saved safetensors.
#
# Usage: python3.12 zimage_controlnet_save_keydiff.py <saved_controlnet_dir>
import json
import sys

import torch
from safetensors import safe_open
from diffusers.models.controlnets.controlnet_z_image import ZImageControlNetModel

out_dir = sys.argv[1]
cfg = json.load(open(out_dir + "/config.json"))
with torch.device("meta"):
    model = ZImageControlNetModel(
        control_layers_places=cfg["control_layers_places"],
        control_refiner_layers_places=cfg["control_refiner_layers_places"],
        control_in_dim=cfg["control_in_dim"],
        add_control_noise_refiner=cfg["add_control_noise_refiner"],
        all_patch_size=tuple(cfg["all_patch_size"]),
        all_f_patch_size=tuple(cfg["all_f_patch_size"]),
        dim=cfg["dim"],
        n_refiner_layers=cfg["n_refiner_layers"],
        n_heads=cfg["n_heads"],
        n_kv_heads=cfg["n_kv_heads"],
        norm_eps=cfg["norm_eps"],
        qk_norm=cfg["qk_norm"],
    )
ref = {k: tuple(v.shape) for k, v in model.state_dict().items()}
with safe_open(out_dir + "/diffusion_pytorch_model.safetensors", framework="pt") as f:
    ours = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}

missing = sorted(set(ref) - set(ours))
extra = sorted(set(ours) - set(ref))
mismatch = sorted(k for k in set(ref) & set(ours) if ref[k] != ours[k])

print(f"reference params: {len(ref)}  saved tensors: {len(ours)}")
ok = True
if missing:
    ok = False
    print(f"MISSING ({len(missing)}):")
    for k in missing:
        print("  ", k, ref[k])
if extra:
    ok = False
    print(f"EXTRA ({len(extra)}):")
    for k in extra:
        print("  ", k, ours[k])
if mismatch:
    ok = False
    print(f"SHAPE MISMATCH ({len(mismatch)}):")
    for k in mismatch:
        print("  ", k, "ref", ref[k], "ours", ours[k])
if ok:
    print("GATE save-format key/shape diff vs diffusers ZImageControlNetModel: PASS (diff empty)")
else:
    print("GATE save-format: FAIL")
    sys.exit(1)
