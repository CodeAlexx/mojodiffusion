#!/usr/bin/env python
"""ltx2 trainer forward+loss parity — REFERENCE side (musubi's own runtime).

Loads the transformer exactly as musubi's trainer/generator does (video-only
wrapper, dequant-bf16 ckpt, block swap), then for each fixture pair replicates
ltx2_train_network.call_dit's math verbatim:
  noisy = (1-s)*lat_f32 + s*noise_f32          (get_noisy..., :2070-2072)
  model_timesteps = (s*1000)/1000  -> [B,1]     (:3058-3065)
  pred = transformer(noisy_bf16, timestep, context=text_bf16,
                     attention_mask=text_mask, frame_rate=25)  (:3846-3853)
  target = noise_bf16 - lat_bf16                (:3874, both cast at :3039-3041)
  loss = mse(pred_bf16.float(), target.float()).mean()  (_masked_loss, mask None)
Dumps pred_i (BF16, latent shape) + loss_i (F32) to ref_out.safetensors.

Run (GPU, ~8 min load + seconds/pair):
  cd /home/alex/musubi-tuner && .venv/bin/python \
    /home/alex/mojodiffusion/scripts/ltx2_parity_ref_gen.py
"""
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/home/alex/musubi-tuner/src")
import musubi_tuner.ltx2_generate_video as gv  # noqa: E402

OUT = "/home/alex/mojodiffusion/output/ltx2_parity_fwd"
CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8-dequant-bf16.safetensors"

# Build args through the generator's OWN parser so load_transformer sees the
# exact attribute set musubi expects (precached-prompts flag skips Gemma).
sys.argv = [
    "x", "--ltx2_checkpoint", CKPT, "--ltx2_mode", "video",
    "--blocks_to_swap", "36", "--sdpa", "--prompt", "unused",
    "--use_precached_sample_prompts", "--output_dir", OUT,
]
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
    accelerator=SimpleNamespace(device=device),
    args=args,
    dit_path=CKPT,
    attn_mode="torch",
    split_attn=False,
    loading_device="cpu",
    dit_weight_dtype=None,
)
transformer.eval()
transformer.enable_block_swap(36, device, supports_backward=False)
if hasattr(transformer, "move_to_device_except_swap_blocks"):
    transformer.move_to_device_except_swap_blocks(device)
if hasattr(transformer, "switch_block_swap_for_inference"):
    transformer.switch_block_swap_for_inference()
if hasattr(transformer, "prepare_block_swap_before_forward"):
    transformer.prepare_block_swap_before_forward()

pairs = [ln.split() for ln in open(os.path.join(OUT, "pairs.txt")) if ln.strip()]
# pairs.txt fields: arm lat_path... te_path... sigma — paths may contain spaces;
# re-parse robustly: arm is first token, sigma last, the middle splits on the
# ".safetensors " boundary.
raw = [ln.rstrip("\n") for ln in open(os.path.join(OUT, "pairs.txt")) if ln.strip()]
parsed = []
for ln in raw:
    arm, rest = ln.split(" ", 1)
    sigma = float(rest.rsplit(" ", 1)[1])
    mid = rest.rsplit(" ", 1)[0]
    cut = mid.index(".safetensors ") + len(".safetensors")
    lat_path, te_path = mid[:cut], mid[cut + 1:]
    parsed.append((arm, lat_path, te_path, sigma))

fx = {}
with safe_open(os.path.join(OUT, "fixture.safetensors"), framework="pt") as f:
    for k in f.keys():
        fx[k] = f.get_tensor(k)

out = {}
for k, (arm, lat_path, te_path, sigma) in enumerate(parsed):
    with safe_open(lat_path, framework="pt") as f:
        lkey = [x for x in f.keys() if x.startswith("latents_")][0]
        lat = f.get_tensor(lkey).unsqueeze(0)  # [1,128,F,H,W] bf16
    with safe_open(te_path, framework="pt") as f:
        text = f.get_tensor("text_bfloat16").unsqueeze(0)  # [1,1024,4096] bf16
        mask = f.get_tensor("text_mask").unsqueeze(0)  # [1,1024] i64 (all ones)
    noise = fx[f"noise_{k}"]
    assert noise.shape == lat.shape, (noise.shape, lat.shape)
    s = float(fx[f"sigma_{k}"][0])
    assert abs(s - sigma) < 1e-6

    lat_d = lat.to(device)
    noise_d = noise.to(device)
    # get_noisy_model_input_and_timesteps math (f32 mix)
    noisy = (1.0 - s) * lat_d.to(torch.float32) + s * noise_d.to(torch.float32)
    timesteps = torch.full((1,), s * 1000.0, device=device, dtype=torch.float32)
    model_ts = (timesteps / 1000.0).to(torch.bfloat16).unsqueeze(1)  # [1,1]

    # call_dit dtype moves (network_dtype = bf16)
    lat_b = lat_d.to(torch.bfloat16)
    noise_b = noise_d.to(torch.bfloat16)
    noisy_b = noisy.to(torch.bfloat16)
    text_b = text.to(device=device, dtype=torch.bfloat16)
    mask_b = mask.to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = transformer(
            noisy_b, timestep=model_ts, context=text_b,
            attention_mask=mask_b, frame_rate=25,
        )
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    target = noise_b - lat_b
    loss = torch.nn.functional.mse_loss(
        pred.to(torch.bfloat16).float(), target.float(), reduction="none"
    ).mean()
    out[f"pred_{k}"] = pred.detach().to(torch.bfloat16).cpu().contiguous()
    out[f"loss_{k}"] = loss.detach().float().cpu().reshape(1)
    print(f"pair {k} ({arm} s={s}): ref loss = {float(loss):.6f}", flush=True)

save_file(out, os.path.join(OUT, "ref_out.safetensors"))
print("REF-GEN DONE:", len(parsed), "pairs")
