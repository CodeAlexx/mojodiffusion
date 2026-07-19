"""Krea-2 SingleStreamBlock leaf-op parity oracle (RMSNorm + SwiGLU).

Imports ai-toolkit's krea2 ``RMSNorm`` and ``SwiGLU`` DIRECTLY from mmdit.py (its
only deps are torch+einops, bypassing the ai-toolkit package __init__ which pulls
torchao/quanto). Random-inits both in **bf16 on cuda** (the production dtype),
runs them on a random bf16 input, and dumps input + weights + outputs to
safetensors so the Mojo probe can run krea2_rmsnorm / krea2_swiglu with the EXACT
same weights and compare via cosine (bar >= 0.999).

RMSNorm is the precision-critical op (F32-internal, weight = scale + 1.0). SwiGLU
runs three bf16 matmuls. Both dumped at FEATURES=6144 (production width).

Run:  /home/alex/serenityflow-v2/.venv/bin/python \
        serenitymojo/models/dit/parity/gen_krea2_blockops.py
"""

import sys

import torch
from safetensors.torch import save_file

# Import mmdit.py directly (torch+einops only) to bypass the ai-toolkit package
# __init__ chain (torchao/quanto absent here).
sys.path.insert(0, "/home/alex/ai-toolkit/extensions_built_in/diffusion_models/krea2/src")
from mmdit import RMSNorm, SwiGLU  # noqa: E402

OUT = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_blockops_oracle.safetensors"
DEV = "cuda"
DT = torch.bfloat16

FEATURES = 6144
MULTIPLIER = 4
EPS = 1e-5
L = 17  # a few tokens (rows)

torch.manual_seed(4321)

# ── RMSNorm(6144) — random-init the scale (the real Parameter is zeros-init, but
# a nonzero scale exercises the weight = scale + 1.0 path; a zeros scale would
# only test weight==1). ──
rms = RMSNorm(FEATURES, eps=EPS).to(DEV, DT)
with torch.no_grad():
    rms.scale.copy_((torch.randn(FEATURES, device=DEV) * 0.1).to(rms.scale.dtype))

# ── SwiGLU(6144, multiplier=4) — random-init gate/up/down (no bias). ──
swi = SwiGLU(FEATURES, MULTIPLIER, bias=False).to(DEV, DT)
mlpdim = swi.gate.weight.shape[0]
print(f"FEATURES={FEATURES} mlpdim={mlpdim} (expect 16384)", flush=True)

x_rms = torch.randn(L, FEATURES, dtype=DT, device=DEV)
x_swi = torch.randn(L, FEATURES, dtype=DT, device=DEV)

with torch.no_grad():
    y_rms = rms(x_rms)          # F32-internal, weight=scale+1, -> bf16
    y_swi = swi(x_swi)          # down(silu(gate(x)) * up(x)) -> bf16

out = {
    # RMSNorm
    "rms_x": x_rms.float().cpu().contiguous(),
    "rms_scale": rms.scale.float().cpu().contiguous(),   # the RAW scale (probe adds +1)
    "rms_y": y_rms.float().cpu().contiguous(),
    # SwiGLU
    "swi_x": x_swi.float().cpu().contiguous(),
    "swi_gate_w": swi.gate.weight.float().cpu().contiguous(),   # [mlpdim, features]
    "swi_up_w": swi.up.weight.float().cpu().contiguous(),       # [mlpdim, features]
    "swi_down_w": swi.down.weight.float().cpu().contiguous(),   # [features, mlpdim]
    "swi_y": y_swi.float().cpu().contiguous(),
}
print(
    "shapes:",
    {k: tuple(v.shape) for k, v in out.items()},
    flush=True,
)
save_file(out, OUT)
print("OK dumped:", ", ".join(sorted(out.keys())), "->", OUT, flush=True)
