# gen_lycoris_family_oracle.py — torch lycoris-lora oracle for the T2.F
# Lycoris-family verification gate (LoCon / LoHa / Tucker / OFT) + the T2.F-2
# extension (LoKr / BOFT / DoRA).
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
#   LoKr: factor layouts are IDENTICAL (w1:[out_l,in_m], w2a:[out_k,r], w2b:[r,in_n],
#         torch.kron index map == Mojo's), so factors dump as-is (no transposes).
#         SAME upstream double-scale quirk as LoHa (measured 2026-06-11):
#         LokrModule.get_weight folds self.scale INSIDE make_kron, and
#         get_diff_weight multiplies by scale AGAIN. The live forward path
#         (forward: get_weight(shape) * scalar) and the bypass path apply it
#         once — the oracle uses the single-scale convention get_weight*scalar.
#         Both-full quirk (lokr.py:209-211): use_w1 AND use_w2 forces alpha=
#         lora_dim, i.e. scale=1 regardless of the user alpha (Mojo mirrors it).
#   BOFT: upstream applies r = (I+Q)(I-Q)^-1 DIRECTLY (einsum "b i j, b j ...",
#         boft.py make_weight — NOT transposed like diag_oft), with
#         Q = blocks - blocks^T. Mojo R = (I+Q_m)^-1(I-Q_m) with Q_m = 0.5*(S-S^T)
#         and skew rationals commute: R(Q_m) = r(-Q_m)  =>  S = -2*blocks
#         (boft_save.mojo writes blocks = -0.5*S; exact in bf16). The per-stage
#         butterfly permute (unflatten/transpose/flatten, g=2, k=2^i*(b/2)) is the
#         SAME index map as Mojo's _butterfly_perm; both sides compose stages
#         0..m-1 left-to-right (T = B_{m-1}..B_0).
#   DoRA: oracle = LoConModule(weight_decompose=True, wd_on_out=True) — the pip
#         lycoris DoRA (chosen over PEFT lora_A/lora_B+lora_magnitude_vector
#         because pip lycoris is the only LIVE upstream loader on this box and
#         the campaign oracle; the lycoris keys lora_down/lora_up + dora_scale
#         are also ComfyUI's kohya-DoRA path). Layouts identical (lora_down
#         [r,in], lora_up [out,r], dora_scale [out,1]); norm eps =
#         torch.finfo(float32).eps (the Mojo gate passes the same eps). The
#         dora_scale stays UNQUANTIZED F32 — upstream keeps it float32 even in
#         bf16 models and so does the Mojo adapter (T2.F-2 fix). y is the FULL
#         forward (get_merged_weight — DoRA REPLACES the effective weight),
#         unlike the delta-only families.
#
# Output: /tmp/lycoris_family_oracle.safetensors
# Run:    python3 serenitymojo/training/tests/gen_lycoris_family_oracle.py
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.modules.diag_oft import DiagOFTModule
from lycoris.modules.lokr import LokrModule
from lycoris.modules.boft import ButterflyOFTModule

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

# ═════════════════════════ T2.F-2: LoKr / BOFT / DoRA ════════════════════════

# ── LoKr v1 (linear): in=8 out=8 rank=1 factor=2 alpha=2 → W1 FULL + W2 FACTORED
# factorization(8,2) = (2,4): out_l=2 out_k=4 in_m=2 in_n=4; rank 1 < max(4,4)/2.
IN, OUT, M = 8, 8, 5
mk1 = LokrModule("lokr_v1", torch.nn.Linear(IN, OUT, bias=False),
                 multiplier=1.0, lora_dim=1, alpha=2.0, factor=2)
assert mk1.use_w1 and not mk1.use_w2, "v1 expected W1 full + W2 factored"
with torch.no_grad():
    mk1.lokr_w1.copy_(bq(torch.randn(2, 2) * 0.4))
    mk1.lokr_w2_a.copy_(bq(torch.randn(4, 1) * 0.4))
    mk1.lokr_w2_b.copy_(bq(torch.randn(1, 4) * 0.4))
mk1.eval()
x = bq(torch.randn(M, IN))
# Single-scale convention (see header): get_weight folds scale via make_kron;
# get_diff_weight would double-apply it. Live forward = get_weight * scalar.
# no_grad: the load check runs under no_grad — grad-enabled matmuls can pick a
# different GEMM kernel (±1 ulp), which would break the BIT-EXACT load bar.
with torch.no_grad():
    dw = (mk1.get_weight(mk1.shape) * mk1.scalar).float()  # [OUT,IN], scale=2 once
    y = F.linear(x, dw)                                    # delta-only
out["lokr1.w1"] = mk1.lokr_w1.detach().float().contiguous()
out["lokr1.w2a"] = mk1.lokr_w2_a.detach().float().contiguous()
out["lokr1.w2b"] = mk1.lokr_w2_b.detach().float().contiguous()
out["lokr1.alpha"] = torch.tensor([2.0], dtype=torch.float32)
out["lokr1.x"] = x
out["lokr1.y"] = y

# ── LoKr v2 (linear): in=8 out=8 rank=3 factor=2 alpha=2 → BOTH FULL ─────────
# rank 3 >= max(4,4)/2 → W2 full; upstream forces alpha=lora_dim → scale=1
# (lokr.py:209-211), regardless of the alpha=2 passed (and saved).
mk2 = LokrModule("lokr_v2", torch.nn.Linear(IN, OUT, bias=False),
                 multiplier=1.0, lora_dim=3, alpha=2.0, factor=2)
assert mk2.use_w1 and mk2.use_w2, "v2 expected both full"
assert float(mk2.scale) == 1.0, "v2 expected the forced scale=1 quirk"
with torch.no_grad():
    mk2.lokr_w1.copy_(bq(torch.randn(2, 2) * 0.4))
    mk2.lokr_w2.copy_(bq(torch.randn(4, 4) * 0.4))
mk2.eval()
x = bq(torch.randn(M, IN))
with torch.no_grad():
    dw = (mk2.get_weight(mk2.shape) * mk2.scalar).float()
    y = F.linear(x, dw)
out["lokr2.w1"] = mk2.lokr_w1.detach().float().contiguous()
out["lokr2.w2"] = mk2.lokr_w2.detach().float().contiguous()
out["lokr2.alpha"] = torch.tensor([2.0], dtype=torch.float32)
out["lokr2.x"] = x
out["lokr2.y"] = y

# ── LoKr v3 (linear): in=64 out=64 rank=1 factor=8 decompose_both → BOTH FACTORED
# factorization(64,8) = (8,8); rank 1 < max(8,8)/2 on both sides; alpha=2 → scale=2.
IN3, OUT3 = 64, 64
mk3 = LokrModule("lokr_v3", torch.nn.Linear(IN3, OUT3, bias=False),
                 multiplier=1.0, lora_dim=1, alpha=2.0, factor=8,
                 decompose_both=True)
assert not mk3.use_w1 and not mk3.use_w2, "v3 expected both factored"
with torch.no_grad():
    mk3.lokr_w1_a.copy_(bq(torch.randn(8, 1) * 0.4))
    mk3.lokr_w1_b.copy_(bq(torch.randn(1, 8) * 0.4))
    mk3.lokr_w2_a.copy_(bq(torch.randn(8, 1) * 0.4))
    mk3.lokr_w2_b.copy_(bq(torch.randn(1, 8) * 0.4))
mk3.eval()
x = bq(torch.randn(M, IN3))
with torch.no_grad():
    dw = (mk3.get_weight(mk3.shape) * mk3.scalar).float()
    y = F.linear(x, dw)
out["lokr3.w1a"] = mk3.lokr_w1_a.detach().float().contiguous()
out["lokr3.w1b"] = mk3.lokr_w1_b.detach().float().contiguous()
out["lokr3.w2a"] = mk3.lokr_w2_a.detach().float().contiguous()
out["lokr3.w2b"] = mk3.lokr_w2_b.detach().float().contiguous()
out["lokr3.alpha"] = torch.tensor([2.0], dtype=torch.float32)
out["lokr3.x"] = x
out["lokr3.y"] = y

# ── BOFT (linear): in=12 out=8 block_size=2 → nb=4, boft_m=3, constraint=0 ───
# All 3 stages have NONTRIVIAL butterfly permutations (g*k = 2,4,8 all divide 8).
IN, OUT, M = 12, 8, 5
base = torch.nn.Linear(IN, OUT, bias=False)
with torch.no_grad():
    base.weight.copy_(bq(torch.randn(OUT, IN) * 0.3))
mb = ButterflyOFTModule("boft_t", base, multiplier=1.0, lora_dim=2, constraint=0)
assert (mb.boft_m, mb.block_num, mb.block_size) == (3, 4, 2), "boft config drifted"
with torch.no_grad():
    mb.oft_blocks.copy_(bq(torch.randn(3, 4, 2, 2) * 0.1))
mb.eval()
x = bq(torch.randn(M, IN))
with torch.no_grad():
    w_eff = mb.make_weight(scale=1).float()              # [OUT,IN] FULL eff. weight
    y = F.linear(x, w_eff)                               # FULL output (not delta)
# Mojo S = -2*blocks (R_mojo(Q) = r_upstream(-Q); boft_save folds blocks=-0.5*S).
out["boft.s_mojo"] = (-2.0 * mb.oft_blocks.detach().float()).contiguous()
out["boft.w_base"] = base.weight.detach().float().contiguous()
out["boft.alpha"] = torch.tensor([0.0], dtype=torch.float32)   # constraint=0
out["boft.x"] = x
out["boft.y"] = y

# ── DoRA (linear): in=12 out=16 rank=4 alpha=2 → LoConModule(wd=True) ────────
IN, OUT, R, M = 12, 16, 4, 5
base = torch.nn.Linear(IN, OUT, bias=False)
with torch.no_grad():
    base.weight.copy_(bq(torch.randn(OUT, IN) * 0.3))
md = LoConModule("dora_t", base, multiplier=1.0, lora_dim=R, alpha=2.0,
                 use_tucker=False, weight_decompose=True, wd_on_out=True)
assert md.wd and md.wd_on_out, "expected weight-decompose (DoRA) mode"
with torch.no_grad():
    md.lora_down.weight.copy_(bq(torch.randn(R, IN) * 0.3))
    md.lora_up.weight.copy_(bq(torch.randn(OUT, R) * 0.3))   # NONZERO so ΔW live
    # dora_scale left as upstream init (F32 ‖W_orig‖ row norms, NOT bf16-
    # quantized): both upstream and the Mojo adapter store the magnitude F32.
md.eval()
x = bq(torch.randn(M, IN))
# FULL forward: DoRA REPLACES the effective weight (merged = m * WP/(‖WP‖+eps)).
with torch.no_grad():
    merged = md.get_merged_weight(multiplier=1)[0].float()  # [OUT,IN]
    y = F.linear(x, merged)
out["dora.down"] = md.lora_down.weight.detach().float().contiguous()  # [R,IN]
out["dora.up"] = md.lora_up.weight.detach().float().contiguous()      # [OUT,R]
out["dora.m"] = md.dora_scale.detach().float().reshape(OUT).contiguous()
out["dora.w_base"] = base.weight.detach().float().contiguous()
out["dora.alpha"] = torch.tensor([2.0], dtype=torch.float32)
out["dora.x"] = x
out["dora.y"] = y

save_file(out, "/tmp/lycoris_family_oracle.safetensors")
for k, v in out.items():
    print(f"{k}: {tuple(v.shape)}")
print("WROTE /tmp/lycoris_family_oracle.safetensors")
