# Convert the PiD distilled student .pth state_dict to a single safetensors
# file, preserving bf16. Strips the `net.` prefix (and any net_ema/fake_score/
# discriminator keys, per pid_distill_model.load_state_dict). Reports the key
# set + count + a dtype/shape summary grouped by module family.
#
# Usage: python3 convert_pid_ckpt.py
import sys
import torch
from collections import OrderedDict
from safetensors.torch import save_file

SRC = "/home/alex/.serenity/models/pid/checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth"
DST = "/home/alex/.serenity/models/pid/checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.safetensors"

sd = torch.load(SRC, map_location="cpu", weights_only=False)
# The .pth may be a raw state_dict or wrapped.
if not isinstance(sd, dict):
    raise SystemExit(f"unexpected ckpt type {type(sd)}")
# Heuristic: if there's a nested 'net' / 'model' key holding a dict of tensors.
if "net" in sd and isinstance(sd["net"], dict) and all(isinstance(v, torch.Tensor) for v in sd["net"].values()):
    sd = sd["net"]

net_sd = OrderedDict()
for k, v in sd.items():
    if not isinstance(v, torch.Tensor):
        continue
    if k.startswith("net_ema.") or k.startswith("fake_score.") or k.startswith("discriminator."):
        continue
    nk = k[len("net.") :] if k.startswith("net.") else k
    net_sd[nk] = v.contiguous()

dtypes = {}
for k, v in net_sd.items():
    dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + 1

print("KEY_COUNT", len(net_sd))
print("DTYPES", dtypes)

# Module-family histogram (top-level prefix).
fam = {}
for k in net_sd:
    top = k.split(".")[0]
    fam[top] = fam.get(top, 0) + 1
print("FAMILIES", fam)

# Detailed: one representative key per family + a couple structural probes.
print("--- sample keys ---")
for probe in [
    "s_embedder", "t_embedder", "y_embedder", "y_pos_embedding",
    "patch_blocks.0", "patch_blocks.13",
    "pixel_blocks.0", "pixel_blocks.1",
    "pixel_embedder", "final_layer",
    "lq_proj",
]:
    matches = [k for k in net_sd if k.startswith(probe)]
    print(f"[{probe}] n={len(matches)}")
    for k in matches[:6]:
        print("   ", k, tuple(net_sd[k].shape), net_sd[k].dtype)

save_file(net_sd, DST)
print("SAVED", DST)
