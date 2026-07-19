#!/usr/bin/env python
"""Conditioning probe for the fwd-parity outlier: run the musubi reference on
pairs 0 (sigma=0.05, FAIL 0.99891) and 3 (sigma=0.5, PASS 0.99988) twice —
once with the fixture noise, once with the noise perturbed by +1 bf16 ulp on
a random 1% of elements — and measure how much the reference's OWN pred
moves. If pair 0's self-sensitivity per unit input-perturbation is ~the
size of its cross-stack residual, the outlier is input-conditioning noise
amplification (dtype class), not a code defect."""
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/home/alex/musubi-tuner/src")
import musubi_tuner.ltx2_generate_video as gv  # noqa: E402

OUT = "/home/alex/mojodiffusion/output/ltx2_parity_fwd"
CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8-dequant-bf16.safetensors"

sys.argv = ["x", "--ltx2_checkpoint", CKPT, "--ltx2_mode", "video",
            "--blocks_to_swap", "36", "--sdpa", "--prompt", "unused",
            "--use_precached_sample_prompts", "--output_dir", OUT]
args = gv.parse_args()
args.dit = args.ltx2_checkpoint
gv._configure_attention_flags(args)

from types import SimpleNamespace  # noqa: E402
from musubi_tuner.ltx2_train_network import LTX2NetworkTrainer  # noqa: E402

device = torch.device("cuda")
trainer = LTX2NetworkTrainer()
trainer.blocks_to_swap = 36
trainer.handle_model_specific_args(args)
transformer = trainer.load_transformer(
    accelerator=SimpleNamespace(device=device), args=args, dit_path=CKPT,
    attn_mode="torch", split_attn=False, loading_device="cpu", dit_weight_dtype=None)
transformer.eval()
transformer.enable_block_swap(36, device, supports_backward=False)
transformer.move_to_device_except_swap_blocks(device)
if hasattr(transformer, "switch_block_swap_for_inference"):
    transformer.switch_block_swap_for_inference()
if hasattr(transformer, "prepare_block_swap_before_forward"):
    transformer.prepare_block_swap_before_forward()

raw = [ln.rstrip("\n") for ln in open(os.path.join(OUT, "pairs.txt")) if ln.strip()]
fx = {}
with safe_open(os.path.join(OUT, "fixture.safetensors"), framework="pt") as f:
    for k in f.keys():
        fx[k] = f.get_tensor(k)

def fwd(lat, noise, s, text, mask):
    lat_d = lat.to(device); noise_d = noise.to(device)
    noisy = (1.0 - s) * lat_d.float() + s * noise_d.float()
    model_ts = torch.full((1, 1), s, device=device, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = transformer(noisy.to(torch.bfloat16), timestep=model_ts,
                           context=text.to(device, torch.bfloat16),
                           attention_mask=mask.to(device), frame_rate=25)
    return (pred[0] if isinstance(pred, (list, tuple)) else pred).float().cpu()

torch.manual_seed(7)
for k in (0, 3):
    ln = raw[k]
    arm, rest = ln.split(" ", 1)
    s = float(rest.rsplit(" ", 1)[1])
    mid = rest.rsplit(" ", 1)[0]
    cut = mid.index(".safetensors ") + len(".safetensors")
    lat_path, te_path = mid[:cut], mid[cut + 1:]
    with safe_open(lat_path, framework="pt") as f:
        lkey = [x for x in f.keys() if x.startswith("latents_")][0]
        lat = f.get_tensor(lkey).unsqueeze(0)
    with safe_open(te_path, framework="pt") as f:
        text = f.get_tensor("text_bfloat16").unsqueeze(0)
        mask = f.get_tensor("text_mask").unsqueeze(0)
    noise = fx[f"noise_{k}"]

    p_base = fwd(lat, noise, s, text, mask)
    # +1 ulp on 1% of elements
    nz = noise.clone()
    sel = torch.rand_like(nz.float()) < 0.01
    bits = nz.view(torch.int16)
    bits[sel] = bits[sel] + torch.where(bits[sel] >= 0,
                                        torch.ones_like(bits[sel]),
                                        -torch.ones_like(bits[sel]))
    p_pert = fwd(lat, nz, s, text, mask)

    in_rel = ((nz.float() - noise.float()).norm() / noise.float().norm()).item()
    cos = torch.nn.functional.cosine_similarity(
        p_base.flatten(), p_pert.flatten(), dim=0).item()
    rel = ((p_base - p_pert).norm() / p_base.norm()).item()
    print(f"pair {k} (s={s}): input perturb relL2={in_rel:.2e} -> "
          f"REF self cos={cos:.6f} relL2={rel:.4f}  amplification={rel/in_rel:.1f}x",
          flush=True)
