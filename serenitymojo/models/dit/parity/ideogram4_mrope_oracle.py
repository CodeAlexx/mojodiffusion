# DEV-ONLY oracle for the Ideogram-4 interleaved MRoPE cos/sin builder.
# Calls Ideogram4MRoPE.forward (modeling_ideogram4.py) on the predict fixture's
# position_ids and dumps cos/sin so the Rust build_mrope can be gated. The module
# is cast to bf16 so inv_freq is the bf16-rounded buffer the real model uses
# (forward upcasts to f32) — this dominates at the 65536 image positions.
import sys, torch
sys.path.insert(0, "/home/alex/ideogram4-ref/src")
from safetensors.torch import load_file, save_file
from ideogram4.modeling_ideogram4 import Ideogram4MRoPE

FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_predict.safetensors"
OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/ideogram4_fx_mrope.safetensors"
dev = torch.device("cuda")

pos = load_file(FX)["position_ids"].to(dev)          # [1, 907, 3] int
m = Ideogram4MRoPE(head_dim=256, base=5000000, mrope_section=(24, 20, 20)).to(torch.bfloat16).to(dev)
cos, sin = m(pos)                                    # [1, 907, 256] f32
save_file(
    {"mrope_cos": cos.float().cpu().contiguous(), "mrope_sin": sin.float().cpu().contiguous()},
    OUT,
)
print(f"[mrope] saved {tuple(cos.shape)} cos_std={float(cos.std()):.4f} sin_std={float(sin.std()):.4f}")
