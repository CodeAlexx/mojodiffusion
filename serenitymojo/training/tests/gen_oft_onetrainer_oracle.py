#!/usr/bin/env python3
# OneTrainer OFT oracle: run OneTrainer's OWN OFTModule.forward (Linear → rotates
# the INPUT in blocks via 5-term Neumann Cayley) and dump the triu-vector param,
# block geometry, W, x, and y for the Mojo OFT-Neumann gate (oft_onetrainer_parity.mojo).
# MJ-1024.
#
# Run: python3 serenitymojo/training/tests/gen_oft_onetrainer_oracle.py
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
from modules.module.LoRAModule import OFTModule
torch.manual_seed(0)

IN, OUT, M = 12, 16, 5
BS = 4                      # oft_block_size → r = IN//BS = 3 blocks over INPUT

def seed(p, k):
    with torch.no_grad():
        t = torch.arange(p.numel(), dtype=torch.float32).reshape(p.shape)
        p.copy_(torch.sin(t * 0.3 + 0.1) * k)

lin = torch.nn.Linear(IN, OUT, bias=False)
seed(lin.weight, 0.5)                                   # W [OUT,IN]
m = OFTModule("oft", lin, oft_block_size=BS, coft=False, coft_eps=6e-5, block_share=False)
m.hook_to_module()
# m.oft_R.weight is [r, n_elements] (n_elements = BS*(BS-1)/2). Default zeros → R=I;
# seed it to a non-trivial rotation so the gate exercises the Neumann polynomial.
with torch.no_grad():
    seed(m.oft_R.weight, 0.25)
m.eval()
x = torch.sin(torch.arange(M * IN, dtype=torch.float32).reshape(M, IN) * 0.07)
with torch.no_grad():
    y = m.forward(x)                                    # OneTrainer's OWN OFT forward [M,OUT]

r = m.oft_R.r
ne = m.oft_R.n_elements
print(f"OFT: IN={IN} OUT={OUT} block_size={m.oft_block_size} r(blocks over IN)={r} n_elements={ne}")
out = {
    "oft.weight_vec": m.oft_R.weight.detach().float().contiguous(),   # [r, n_elements] triu params
    "oft.W": lin.weight.detach().float().contiguous(),               # [OUT,IN]
    "oft.x": x.contiguous(),                                          # [M,IN]
    "oft.y": y.detach().float().contiguous(),                        # [M,OUT]
    "oft.rows": m.oft_R.rows.float().contiguous(),                   # triu row indices
    "oft.cols": m.oft_R.cols.float().contiguous(),                   # triu col indices
    "oft.dims": torch.tensor([IN, OUT, M, m.oft_block_size, r, ne], dtype=torch.float32),
}
save_file(out, "/tmp/oft_ot_oracle.safetensors")
print("WROTE /tmp/oft_ot_oracle.safetensors  (5-term Neumann, input-side, eps=0)")
print("  y L2:", float(y.float().norm()))
