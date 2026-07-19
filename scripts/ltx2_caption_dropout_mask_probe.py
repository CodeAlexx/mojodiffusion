#!/usr/bin/env python
"""P2 caption-dropout decisive experiment (musubi's OWN runtime).

musubi's dropout (_apply_caption_dropout, ltx2_train_network.py:782-799) zeroes
the cached text embeds AND collapses text_mask to index-0-only. Our Mojo stack
has no per-sample mask plumbing (asserts all-ones). HYPOTHESIS: text is K/V-only
in the LTX2 blocks, so with ALL text values zeroed the attention output is zero
under ANY mask -> the mask collapse is forward-irrelevant and the Mojo lever can
be a cache-side zero with the mask left all-ones.

Measures, on one real image512 cache sample at sigma=0.5 (fixed noise seed 1234):
  A  : embeds NONZERO, mask all-ones            (baseline)
  B  : embeds NONZERO, mask idx0-only           (mask-liveness control: must DIFFER from A)
  C  : embeds ZERO,    mask all-ones            (the Mojo-lever candidate)
  C2 : repeat of C                              (determinism noise floor)
  D  : embeds ZERO,    mask idx0-only           (musubi's exact dropout form)

VERDICT: if relL2(C,D) is at the C-vs-C2 noise floor while relL2(A,B) is large,
zero+all-ones == musubi's drop (measured), no mask plumbing needed.

Run:
  cd /home/alex/musubi-tuner && .venv/bin/python \
    /home/alex/mojodiffusion/scripts/ltx2_caption_dropout_mask_probe.py
"""
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/home/alex/musubi-tuner/src")
import musubi_tuner.ltx2_generate_video as gv  # noqa: E402

OUT = "/home/alex/mojodiffusion/output/ltx2_dropout_probe"
CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8-dequant-bf16.safetensors"
CACHE = "/home/alex/datasets/ltx2_eri2_512/cache"
os.makedirs(OUT, exist_ok=True)

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
    accelerator=SimpleNamespace(device=device), args=args, dit_path=CKPT,
    attn_mode="torch", split_attn=False, loading_device="cpu",
    dit_weight_dtype=None,
)
transformer.eval()
transformer.enable_block_swap(36, device, supports_backward=False)
if hasattr(transformer, "move_to_device_except_swap_blocks"):
    transformer.move_to_device_except_swap_blocks(device)
if hasattr(transformer, "switch_block_swap_for_inference"):
    transformer.switch_block_swap_for_inference()

# one real image512 sample (first geometry-matching latent + its TE pair)
lat_path = te_path = None
for n in sorted(os.listdir(CACHE)):
    if n.endswith("_ltx2.safetensors") and not n.endswith("_te.safetensors"):
        with safe_open(os.path.join(CACHE, n), framework="pt") as f:
            lkeys = [x for x in f.keys() if x.startswith("latents_1x16x16_")]
        if lkeys:
            stem = None
            for t in sorted(os.listdir(CACHE)):
                if t.endswith("_ltx2_te.safetensors") and n.startswith(t[: -len("_ltx2_te.safetensors")] + "_"):
                    if stem is None or len(t) > len(stem):
                        stem = t
            if stem:
                lat_path, te_path, lkey = os.path.join(CACHE, n), os.path.join(CACHE, stem), lkeys[0]
                break
assert lat_path, "no image512 sample found"
print("sample:", lat_path, "|", te_path, flush=True)

with safe_open(lat_path, framework="pt") as f:
    lat = f.get_tensor(lkey).unsqueeze(0)
with safe_open(te_path, framework="pt") as f:
    text = f.get_tensor("text_bfloat16").unsqueeze(0)
    mask = f.get_tensor("text_mask").unsqueeze(0)

s = 0.5
g = torch.Generator(device="cpu").manual_seed(1234)
noise = torch.randn(lat.shape, generator=g, dtype=torch.float32)

lat_d = lat.to(device)
noisy = (1.0 - s) * lat_d.to(torch.float32) + s * noise.to(device)
noisy_b = noisy.to(torch.bfloat16)
model_ts = torch.full((1, 1), s, device=device, dtype=torch.bfloat16)

text_nz = text.to(device=device, dtype=torch.bfloat16)
text_z = torch.zeros_like(text_nz)
mask_ones = mask.to(device)
mask_idx0 = torch.zeros_like(mask_ones)
mask_idx0[:, 0] = 1  # musubi: all-False except index 0


def fwd(ctx, m):
    if hasattr(transformer, "prepare_block_swap_before_forward"):
        transformer.prepare_block_swap_before_forward()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        p = transformer(noisy_b, timestep=model_ts, context=ctx,
                        attention_mask=m, frame_rate=25)
    return (p[0] if isinstance(p, (list, tuple)) else p).detach().float()


def rel(a, b):
    d = (a - b).norm() / max(b.norm().item(), 1e-12)
    c = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    return float(d), float(c), float((a - b).abs().max())


runs = {}
for tag, ctx, m in (("A", text_nz, mask_ones), ("B", text_nz, mask_idx0),
                    ("C", text_z, mask_ones), ("C2", text_z, mask_ones),
                    ("D", text_z, mask_idx0)):
    runs[tag] = fwd(ctx, m)
    print(f"forward {tag} done", flush=True)

pairs = {"A_vs_B (mask liveness, nonzero text)": ("A", "B"),
         "C_vs_C2 (determinism noise floor)": ("C", "C2"),
         "C_vs_D  (THE VERDICT: zero+all-ones vs musubi drop)": ("C", "D"),
         "A_vs_C  (drop effect size, sanity)": ("A", "C")}
results = {}
for name, (x, y) in pairs.items():
    r, c, mx = rel(runs[x], runs[y])
    results[name] = (r, c, mx)
    print(f"{name}: relL2={r:.3e} cos={c:.9f} max|d|={mx:.3e}", flush=True)

save_file({k: v.to(torch.bfloat16).cpu().contiguous() for k, v in runs.items()},
          os.path.join(OUT, "probe_preds.safetensors"))

floor = results["C_vs_C2 (determinism noise floor)"][0]
verdict = results["C_vs_D  (THE VERDICT: zero+all-ones vs musubi drop)"][0]
live = results["A_vs_B (mask liveness, nonzero text)"][0]
print("\n=== SUMMARY ===")
print(f"mask liveness (must be >> floor): {live:.3e}")
print(f"noise floor:                      {floor:.3e}")
print(f"verdict C vs D:                   {verdict:.3e}")
if live < max(10 * floor, 1e-6):
    print("VERDICT: INCONCLUSIVE — mask argument appears DEAD in this forward; re-check mask plumbing")
elif verdict <= max(10 * floor, 1e-6):
    print("VERDICT: EQUIVALENT — cache-side zero with all-ones mask == musubi's drop; no mask plumbing needed")
else:
    print("VERDICT: NOT equivalent — Mojo lever needs real per-sample mask support")
