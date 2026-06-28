#!/usr/bin/env python3
# OneTrainer DoRA oracle (per-INPUT magnitude axis = DoRAModule default
# decompose_output_axis=False; MJ-1023). Runs OneTrainer's OWN DoRAModule.forward
# and dumps the exact A,B,m,W,x it used + the output y, for the Mojo gate
# (dora_onetrainer_parity.mojo) to reproduce with wd_on_out=False.
#
# Run (system python3 has torch; OneTrainer gguf-quant import is stubbed):
#   python3 serenitymojo/training/tests/gen_dora_onetrainer_oracle.py
import sys, types
sys.path.insert(0, "/home/alex/OneTrainer")
class _Any:
    def __getattr__(self, k): return 0
    def __call__(self, *a, **k): return 0
_g = types.ModuleType("gguf"); _g.GGMLQuantizationType = _Any()
_g.GGUFReader = object; _g.ReaderTensor = object
_g.dequantize = lambda *a, **k: None; _g.quants = _Any()
sys.modules.setdefault("gguf", _g)
_u = types.ModuleType("diffusers.quantizers.gguf.utils")
_u.GGUFLinear = object; _u.dequantize_gguf_tensor = lambda w, *a, **k: w
sys.modules["diffusers.quantizers.gguf.utils"] = _u

import torch
from safetensors.torch import save_file
from modules.module.LoRAModule import DoRAModule
torch.manual_seed(0)

IN, OUT, R, M = 12, 16, 4, 5
ALPHA = 2.0

def seed(p, k):
    with torch.no_grad():
        t = torch.arange(p.numel(), dtype=torch.float32).reshape(p.shape)
        p.copy_(torch.sin(t * 0.1 + 0.3) * k)

lin = torch.nn.Linear(IN, OUT, bias=False)
seed(lin.weight, 0.5)                                   # W [OUT,IN]
m = DoRAModule("dora", lin, rank=R, alpha=ALPHA, train_device=torch.device("cpu"),
               decompose_output_axis=False)             # per-INPUT (default)
m.hook_to_module()
with torch.no_grad():
    seed(m.lora_down.weight, 0.3)                       # A [R,IN]
    seed(m.lora_up.weight, 0.3)                         # B [OUT,R] (nonzero → ΔW≠0)
    m.dora_scale.mul_(1.1)                              # perturb magnitude (per-input [1,IN])
m.eval()
x = torch.sin(torch.arange(M * IN, dtype=torch.float32).reshape(M, IN) * 0.07)
# Grad-enabled forward + backward with a FIXED d_y so OneTrainer's autograd gives
# the detached-norm gradient the Mojo dora_backward must match (W frozen).
lin.weight.requires_grad_(False)
y = m.forward(x)                                        # OneTrainer's OWN DoRA forward [M,OUT]
d_y = torch.sin(torch.arange(M * OUT, dtype=torch.float32).reshape(M, OUT) * 0.05) * 0.5
m.lora_down.weight.grad = None; m.lora_up.weight.grad = None; m.dora_scale.grad = None
y.backward(d_y)                                         # autograd through detached norm

out = {
    "dora.A": m.lora_down.weight.detach().float().contiguous(),   # [R,IN]
    "dora.B": m.lora_up.weight.detach().float().contiguous(),     # [OUT,R]
    "dora.m_in": m.dora_scale.detach().float().reshape(IN).contiguous(),  # [IN] per-input
    "dora.W": lin.weight.detach().float().contiguous(),           # [OUT,IN]
    "dora.x": x.detach().contiguous(),                            # [M,IN]
    "dora.y": y.detach().float().contiguous(),                   # [M,OUT]
    "dora.dy": d_y.contiguous(),                                  # [M,OUT] fixed upstream grad
    "dora.dA": m.lora_down.weight.grad.detach().float().contiguous(),     # [R,IN]
    "dora.dB": m.lora_up.weight.grad.detach().float().contiguous(),       # [OUT,R]
    "dora.dm_in": m.dora_scale.grad.detach().float().reshape(IN).contiguous(),  # [IN]
    "dora.dims": torch.tensor([IN, OUT, R, M], dtype=torch.float32),
}
save_file(out, "/tmp/dora_ot_oracle.safetensors")
print("WROTE /tmp/dora_ot_oracle.safetensors  (per-INPUT axis, alpha=2 rank=4 eps=0, +autograd grads)")
print("  m_in shape:", tuple(m.dora_scale.shape), " y L2:", float(y.float().norm()),
      " dA L2:", float(m.lora_down.weight.grad.norm()))
