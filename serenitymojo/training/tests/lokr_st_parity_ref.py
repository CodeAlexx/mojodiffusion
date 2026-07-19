# tests/lokr_st_parity_ref.py — upstream oracle for the T2.G SimpleTuner-LoKr
# training-parity gate (lokr_st_parity.mojo). NEW file (T2.G owns it; the
# T2.F-2 family gate trio is untouched per coordination).
#
# Oracle = pip lycoris_lora 3.4.0 (the T2.F oracle) + SimpleTuner's own
# peft_init helper semantics (approximate_normal_tensor / init_lokr_norm,
# /home/alex/SimpleTuner/simpletuner/helpers/training/peft_init.py).
#
# Dumps /tmp/lokr_st_oracle.safetensors with:
#   A. fact_table        [N,4] f32  — (dim, factor, m, n) from upstream
#                                     lycoris.functional.factorization
#   B. shape_case_{i}    [12]  f32  — (in,out,rank,factor,decomp,full,
#                                      w1_fact,w2_fact,out_l,out_k,in_m,in_n)
#                                     from REAL LokrModule instances
#   C. pinit_*                      — perturbed-normal init reference (both-full
#                                     module; ST trainer.py:2757 + peft_init.py)
#   D. train{1,2}_*                 — reduced-dim e2e LoKr TRAINING repro:
#                                     bf16-quantized seeded init, fixed (x,t),
#                                     3 torch AdamW steps through the lycoris
#                                     wrapper's own forward; initial + final
#                                     factors dumped. D1 = upstream-default
#                                     factored (W1 full + W2 factored); D2 =
#                                     SimpleTuner flagship full_matrix=True +
#                                     init_lokr_norm perturbed init.
#
# Run: python3 serenitymojo/training/tests/lokr_st_parity_ref.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from lycoris import create_lycoris
from lycoris.functional import factorization

OUT = "/tmp/lokr_st_oracle.safetensors"
tensors = {}


def bf16q(t):
    return t.to(torch.bfloat16).to(torch.float32)


# ── A. factorization table ───────────────────────────────────────────────────
FACT_CASES = [
    (127, -1), (128, -1), (250, -1), (360, 8), (512, 16), (1024, 4),
    (4096, -1), (4096, 2), (4096, 4), (36864, 4), (24576, -1), (16384, 4),
    (12288, 96), (7, 3), (63, -1), (100, 7), (64, 16), (14, -1), (10, -1),
]
rows = []
for dim, fac in FACT_CASES:
    m, n = factorization(dim, fac)
    rows.append([float(dim), float(fac), float(m), float(n)])
tensors["fact_table"] = torch.tensor(rows, dtype=torch.float32)

# ── B. leg-selection / shape table from real LokrModule instances ────────────
SHAPE_CASES = [
    # (in, out, rank, factor, decompose_both, full_matrix)
    (64, 48, 2, -1, 0, 0),     # upstream default: W1 full + W2 factored
    (64, 48, 8, -1, 0, 0),     # rank too large -> forced both-full
    (64, 48, 2, -1, 1, 0),     # decompose_both -> both factored
    (64, 48, 4, -1, 0, 1),     # SimpleTuner full_matrix -> both full
    (64, 48, 2, -1, 1, 1),     # full_matrix blocks decompose_both too
    (14, 10, 3, -1, 0, 0),     # odd max(out_k,in_n): 3 < 7/2 -> FACTORED
    (4096, 4096, 16, 2, 0, 0),     # klein attn dims, ST module_algo_map factor 2
    (4096, 24576, 16, 4, 0, 0),    # klein ff_in
    (16384, 4096, 16, 4, 0, 0),    # klein single to_out
    (4096, 36864, 16, 4, 0, 0),    # klein single fused qkv_mlp
    (4096, 4096, 16, 2, 0, 1),     # klein attn, full_matrix (ST flagship)
]
from lycoris.modules.lokr import LokrModule

for i, (din, dout, rank, fac, dec, full) in enumerate(SHAPE_CASES):
    lin = nn.Linear(din, dout, bias=False)
    mod = LokrModule(
        f"case{i}", lin, multiplier=1.0, lora_dim=rank, alpha=1,
        factor=fac, decompose_both=bool(dec), full_matrix=bool(full),
    )
    w1_fact = 0 if mod.use_w1 else 1
    w2_fact = 0 if mod.use_w2 else 1
    in_m, in_n = factorization(din, fac)
    out_l, out_k = factorization(dout, fac)
    tensors[f"shape_case_{i}"] = torch.tensor(
        [din, dout, rank, fac, dec, full, w1_fact, w2_fact,
         out_l, out_k, in_m, in_n], dtype=torch.float32)
tensors["shape_case_count"] = torch.tensor([len(SHAPE_CASES)], dtype=torch.float32)


# ── C. perturbed-normal init reference (ST peft_init.py, both-full module) ───
# EXACT port target: approximate_normal_tensor(org, w2, scale) then w1<-1.
torch.manual_seed(7)
org = torch.randn(48, 64) * 0.02 + 0.01
org = bf16q(org)


def approximate_normal_tensor(inp, target, scale=1.0):
    tensor = torch.randn_like(target)
    desired_norm = inp.norm()
    desired_mean = inp.mean()
    desired_std = inp.std()
    tensor = tensor * (desired_norm / tensor.norm())
    tensor = tensor * (desired_std / tensor.std())
    tensor = tensor - tensor.mean() + desired_mean
    tensor.mul_(scale)
    target.copy_(tensor)


lin = nn.Linear(64, 48, bias=False)
with torch.no_grad():
    lin.weight.copy_(org)
pmod = LokrModule("pinit", lin, multiplier=1.0, lora_dim=4, alpha=1,
                  factor=-1, full_matrix=True)
assert pmod.use_w1 and pmod.use_w2
with torch.no_grad():
    pmod.lokr_w1.fill_(1.0)
    approximate_normal_tensor(org, pmod.lokr_w2.data, scale=1e-3)
w2 = pmod.lokr_w2.data
tensors["pinit_org"] = org.clone()
tensors["pinit_stats"] = torch.tensor([
    org.norm().item(), org.mean().item(), org.std().item(),
    w2.norm().item(), w2.mean().item(), w2.std().item(),
    1e-3,
], dtype=torch.float32)


# ── D. reduced-dim e2e training repro through the lycoris wrapper ────────────
class Tiny(nn.Module):
    def __init__(self, din, dout, w):
        super().__init__()
        self.proj = nn.Linear(din, dout, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(w)

    def forward(self, x):
        return self.proj(x)


def train_case(tag, din, dout, rank, alpha, fac, full, pinit, steps=3, lr=1e-2):
    torch.manual_seed(100 + len(tag))
    base_w = bf16q(torch.randn(dout, din) * 0.05)
    net = Tiny(din, dout, base_w)
    for p in net.parameters():
        p.requires_grad_(False)
    wrapper = create_lycoris(net, 1.0, linear_dim=rank, linear_alpha=alpha,
                             algo="lokr", factor=fac, full_matrix=bool(full))
    wrapper.apply_to()
    assert len(wrapper.loras) == 1
    mod = wrapper.loras[0]
    if pinit:
        with torch.no_grad():
            mod.lokr_w1.fill_(1.0)
            approximate_normal_tensor(base_w, mod.lokr_w2.data, scale=1e-3)
    # bf16-quantize every lokr param so the Mojo bf16 masters start bit-equal
    with torch.no_grad():
        for name in ("lokr_w1", "lokr_w1_a", "lokr_w1_b",
                     "lokr_w2", "lokr_w2_a", "lokr_w2_b"):
            if hasattr(mod, name):
                p = getattr(mod, name)
                p.copy_(bf16q(p.data))
    x = bf16q(torch.randn(8, din))
    t = bf16q(torch.randn(8, dout))
    init = {}
    for name in ("lokr_w1", "lokr_w1_a", "lokr_w1_b",
                 "lokr_w2", "lokr_w2_a", "lokr_w2_b"):
        if hasattr(mod, name):
            init[name] = getattr(mod, name).data.clone()
    opt = torch.optim.AdamW(wrapper.parameters(), lr=lr, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.01)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        y = net(x)
        loss = F.mse_loss(y, t)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    fin = {}
    for name in init:
        fin[name] = getattr(mod, name).data.clone()
    # the applied delta in the single-scale live-forward convention
    use_w1 = mod.use_w1
    use_w2 = mod.use_w2
    w1 = fin.get("lokr_w1") if use_w1 else fin["lokr_w1_a"] @ fin["lokr_w1_b"]
    w2 = fin.get("lokr_w2") if use_w2 else fin["lokr_w2_a"] @ fin["lokr_w2_b"]
    delta = torch.kron(w1, w2) * mod.scale
    tensors[f"{tag}_x"] = x
    tensors[f"{tag}_t"] = t
    tensors[f"{tag}_base_w"] = base_w
    tensors[f"{tag}_meta"] = torch.tensor(
        [din, dout, rank, alpha, fac, full, 1 if pinit else 0, steps, lr,
         mod.scale], dtype=torch.float32)
    tensors[f"{tag}_losses"] = torch.tensor(losses, dtype=torch.float32)
    tensors[f"{tag}_delta_final"] = delta
    for name, v in init.items():
        tensors[f"{tag}_init_{name}"] = v
    for name, v in fin.items():
        tensors[f"{tag}_final_{name}"] = v
    print(f"[{tag}] losses={losses} scale={mod.scale} use_w1={use_w1} use_w2={use_w2}")


# D1: upstream default factored (W1 full + W2 factored): rank 2 < max(8,8)/2
train_case("train1", 64, 48, 2, 1.0, -1, 0, False)
# D2: SimpleTuner flagship — full_matrix + init_lokr_norm perturbed init
train_case("train2", 24, 16, 4, 1.0, 4, 1, True)

save_file(tensors, OUT)
print("wrote", OUT, "tensors:", len(tensors))
