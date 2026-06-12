# gen_lycoris_family_oracle.py — torch lycoris-lora oracle for the T2.F
# Lycoris-family verification gate (LoCon / LoHa / Tucker / OFT).
#
# Oracle = pip lycoris_lora 3.4.0 (/home/alex/.local/lib/python3.12/site-packages/
# lycoris). For each family this builds a tiny module with SEEDED, bf16-QUANTIZED
# weights (so the Mojo side's bf16 factor storage is bit-exact and the only
# divergence left is F32 summation order), computes the family forward via the
# package's OWN weight construction (get_diff_weight / make_weight +
# F.conv2d / F.linear), and dumps:
#   - weights pre-converted to the Mojo internal layout (Flame RSCF / [in,rank]
#     loha factors / OFT S with the 0.5-folded skew convention),
#   - the input x (NHWC flat for conv families, [M,in] for linear families),
#   - the oracle output y (NHWC flat / [M,out]).
# The Mojo gate (lycoris_family_parity.mojo) loads these, runs the in-tree
# adapter forwards, and demands cos >= 0.99999 (F32). The layout conversions
# below are NOT trusted on their own: the save/load loop is closed by
# lycoris_family_load_check.py, which loads the MOJO-saved safetensors back
# into upstream lycoris modules via make_module_from_state_dict and re-compares
# the forward against the y dumped here.
#
# Layout conversions (documented, mirrored from the in-tree save modules):
#   LoCon  down: Kohya OIHW [R,Cin,Kh,Kw]  -> Flame RSCF [Kh,Kw,Cin,R]  permute(2,3,1,0)
#   LoCon  up:   Kohya OIHW [Cout,R,1,1]   -> Flame [R,Cout]            squeeze+T
#   Tucker down: Kohya [R,Cin,1,1]         -> Flame [Cin,R]             squeeze+T
#   Tucker mid:  Kohya [Ro,Ri,Kh,Kw]       -> Flame [Kh,Kw,Ri,Ro]       permute(2,3,1,0)
#   Tucker up:   Kohya [Cout,R,1,1]        -> Flame [R,Cout]            squeeze+T
#   LoHa: upstream DW = (w1a@w1b)*(w2a@w2b)*s  [out,in] with w1a [out,R], w1b [R,in].
#         Mojo  DW = (w1a@w1b)(.)(w2a@w2b)*s  [in,out] with w1a [in,R],  w1b [R,out].
#         DW_mojo = DW_up^T  =>  mojo.w1a = up.w1b^T, mojo.w1b = up.w1a^T (pair 2 same).
#   OFT:  upstream Q = blocks - blocks^T; Mojo Q = 0.5*(S - S^T)  =>  S = 2*blocks.
#         upstream applies einsum("k n m, k n i -> k m i", r, W) = r^T @ W_block with
#         r = (I+Q)(I-Q)^-1, and r^T = (I+Q)^-1 (I-Q) = Mojo's R  =>  same W_eff.
#
# Output: /tmp/lycoris_family_oracle.safetensors
# Run:    python3 serenitymojo/training/tests/gen_lycoris_family_oracle.py
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.modules.diag_oft import DiagOFTModule

torch.manual_seed(20260611)


def bq(t):
    """bf16-quantize, return f32 (so Mojo's bf16 factor storage is exact)."""
    return t.to(torch.bfloat16).to(torch.float32)


out = {}

# ── LoCon (conv, no Tucker): Cin=8 Cout=16 k3 s1 p1 rank=4 alpha=2 ───────────
Cin, Cout, K, S, P, R = 8, 16, 3, 1, 1, 4
base = torch.nn.Conv2d(Cin, Cout, K, S, P, bias=False)
m = LoConModule("locon_t", base, multiplier=1.0, lora_dim=R, alpha=2.0, use_tucker=False)
with torch.no_grad():
    m.lora_down.weight.copy_(bq(torch.randn(R, Cin, K, K) * 0.2))
    m.lora_up.weight.copy_(bq(torch.randn(Cout, R, 1, 1) * 0.2))
m.eval()
x = bq(torch.randn(2, Cin, 6, 6))
dw = m.get_diff_weight(multiplier=1)[0].float()          # [Cout,Cin,K,K], includes scale
y = F.conv2d(x, dw, stride=S, padding=P)                 # delta-only output
out["locon.down_rscf"] = m.lora_down.weight.detach().float().permute(2, 3, 1, 0).contiguous()
out["locon.up_rc"] = m.lora_up.weight.detach().float().reshape(Cout, R).t().contiguous()
out["locon.alpha"] = torch.tensor([2.0], dtype=torch.float32)
out["locon.x_nhwc"] = x.permute(0, 2, 3, 1).contiguous()
out["locon.y_nhwc"] = y.permute(0, 2, 3, 1).contiguous()

# ── Tucker (conv): Cin=8 Cout=16 k3 s2 p1 rank=4 alpha=2 ─────────────────────
Cin, Cout, K, S, P, R = 8, 16, 3, 2, 1, 4
base = torch.nn.Conv2d(Cin, Cout, K, S, P, bias=False)
mt = LoConModule("tucker_t", base, multiplier=1.0, lora_dim=R, alpha=2.0, use_tucker=True)
assert mt.tucker, "expected tucker mode"
with torch.no_grad():
    mt.lora_down.weight.copy_(bq(torch.randn(R, Cin, 1, 1) * 0.3))
    mt.lora_mid.weight.copy_(bq(torch.randn(R, R, K, K) * 0.3))
    mt.lora_up.weight.copy_(bq(torch.randn(Cout, R, 1, 1) * 0.3))
mt.eval()
x = bq(torch.randn(2, Cin, 7, 7))
dw = mt.get_diff_weight(multiplier=1)[0].float()         # rebuild_tucker path
y = F.conv2d(x, dw, stride=S, padding=P)
out["tucker.down_cr"] = mt.lora_down.weight.detach().float().reshape(R, Cin).t().contiguous()
out["tucker.core_rscf"] = mt.lora_mid.weight.detach().float().permute(2, 3, 1, 0).contiguous()
out["tucker.up_rc"] = mt.lora_up.weight.detach().float().reshape(Cout, R).t().contiguous()
out["tucker.alpha"] = torch.tensor([2.0], dtype=torch.float32)
out["tucker.x_nhwc"] = x.permute(0, 2, 3, 1).contiguous()
out["tucker.y_nhwc"] = y.permute(0, 2, 3, 1).contiguous()

# ── LoHa (linear): in=12 out=16 rank=4 alpha=2 (non-square catches transposes) ─
IN, OUT, R, M = 12, 16, 4, 5
base = torch.nn.Linear(IN, OUT, bias=False)
mh = LohaModule("loha_t", base, multiplier=1.0, lora_dim=R, alpha=2.0)
with torch.no_grad():
    mh.hada_w1_a.copy_(bq(torch.randn(OUT, R) * 0.3))
    mh.hada_w1_b.copy_(bq(torch.randn(R, IN) * 0.3))
    mh.hada_w2_a.copy_(bq(torch.randn(OUT, R) * 0.3))
    mh.hada_w2_b.copy_(bq(torch.randn(R, IN) * 0.3))
mh.eval()
x = bq(torch.randn(M, IN))
# NOTE (measured 2026-06-11): LohaModule.get_diff_weight applies self.scale TWICE
# (get_weight already folds gamma=scale into HadaWeight; get_diff_weight multiplies
# again). The module's own LIVE forward path (bypass_forward_diff = get_weight *
# scalar * multiplier) and the a1111/comfy ecosystem (l_network_hada.calc_updown +
# finalize_updown) both apply the scale ONCE. The oracle therefore uses the
# single-scale convention: DW = get_weight(shape) * scalar  == bypass path at
# multiplier=1.
dw = (mh.get_weight(mh.shape) * mh.scalar).float()       # [OUT,IN], single scale
y = F.linear(x, dw)                                      # [M,OUT] delta-only
out["loha.w1a_mojo"] = mh.hada_w1_b.detach().float().t().contiguous()   # [IN,R]
out["loha.w1b_mojo"] = mh.hada_w1_a.detach().float().t().contiguous()   # [R,OUT]
out["loha.w2a_mojo"] = mh.hada_w2_b.detach().float().t().contiguous()   # [IN,R]
out["loha.w2b_mojo"] = mh.hada_w2_a.detach().float().t().contiguous()   # [R,OUT]
out["loha.alpha"] = torch.tensor([2.0], dtype=torch.float32)
out["loha.x"] = x
out["loha.y"] = y

# ── Diag-OFT (linear): in=12 out=16 lora_dim=4 -> 4 blocks of 4, constraint=0 ─
IN, OUT, M = 12, 16, 5
base = torch.nn.Linear(IN, OUT, bias=False)
with torch.no_grad():
    base.weight.copy_(bq(torch.randn(OUT, IN) * 0.3))
mo = DiagOFTModule("oft_t", base, multiplier=1.0, lora_dim=4, constraint=0, rescaled=False)
NB, B = mo.block_num, mo.block_size
with torch.no_grad():
    mo.oft_blocks.copy_(bq(torch.randn(NB, B, B) * 0.1))
mo.eval()
x = bq(torch.randn(M, IN))
w_eff = mo.make_weight(scale=1).float()                  # [OUT,IN] full effective weight
y = F.linear(x, w_eff)                                   # [M,OUT] FULL output (not delta)
out["oft.s_mojo"] = (2.0 * mo.oft_blocks.detach().float()).contiguous()  # Mojo S = 2*blocks
out["oft.w_base"] = base.weight.detach().float().contiguous()            # [OUT,IN]
out["oft.alpha"] = torch.tensor([0.0], dtype=torch.float32)              # constraint=0
out["oft.x"] = x
out["oft.y"] = y
out["oft.nb_b"] = torch.tensor([float(NB), float(B)], dtype=torch.float32)

save_file(out, "/tmp/lycoris_family_oracle.safetensors")
for k, v in out.items():
    print(f"{k}: {tuple(v.shape)}")
print("WROTE /tmp/lycoris_family_oracle.safetensors")
