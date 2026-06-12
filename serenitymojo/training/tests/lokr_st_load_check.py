# tests/lokr_st_load_check.py — T2.G ecosystem load gate for the TRAINER-saved
# klein LoKr checkpoint. NEW file (the T2.F-2 family load-check is untouched).
#
# Loads the file the klein trainer wrote (adapter_algo=4, lokr_save format,
# lycoris wrapper naming) into UPSTREAM pip lycoris_lora 3.4.0:
#   1. key-set scan: every module prefix carries a complete LoKr key family
#      (lokr_w1 | lokr_w1_a/_b, lokr_w2 | lokr_w2_a/_b, alpha) and the prefix
#      set matches the lycoris naming for the configured klein target set;
#   2. suffix-schema comparison against a REAL create_lycoris(...).save_weights
#      LoKr file (same suffix classes => ecosystem loaders parse it);
#   3. make_module_from_state_dict on representative modules of every target
#      class (dbl attn q, dbl ff_in, sgl fused qkv_mlp, sgl out): upstream
#      accepts the tensors, reconstructs the SAME split shapes as upstream
#      factorization at the config's per-class factors, and get_diff_weight
#      reproduces kron(w1, w2a@w2b)*scale from the file bit-exactly;
#   4. trained-ness: the w2-side zero leg must be NONZERO (the checkpoint came
#      from a real training run, not an init dump).
#
# Run AFTER the 10-step smoke:
#   python3 serenitymojo/training/tests/lokr_st_load_check.py <ckpt.safetensors>
import sys

import torch
import torch.nn as nn
from safetensors.torch import load_file

from lycoris import create_lycoris
from lycoris.functional import factorization
from lycoris.modules.lokr import LokrModule

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/alex/mojodiffusion/output/alina_train/klein9b_lokr_smoke.safetensors"

# klein9b_lokr_smoke.json settings (keep in sync)
D, F = 4096, 12288
ND, NS = 8, 24
RANK, ALPHA = 16, 16.0
FACTOR_ATTN, FACTOR_FF, FACTOR_SINGLE = 2, 4, 4

sd = load_file(CKPT)
print(f"[load] {CKPT}: {len(sd)} tensors")

# ── 1. key-set scan ───────────────────────────────────────────────────────────
prefixes = {}
for k in sd:
    p, suf = k.rsplit(".", 1)
    prefixes.setdefault(p, set()).add(suf)

expected_prefixes = set()
for b in range(ND):
    base = f"lycoris_transformer_blocks_{b}"
    for tail in ("attn_to_q", "attn_to_k", "attn_to_v", "attn_to_out_0",
                 "ff_linear_in", "ff_linear_out",
                 "attn_add_q_proj", "attn_add_k_proj", "attn_add_v_proj",
                 "attn_to_add_out", "ff_context_linear_in", "ff_context_linear_out"):
        expected_prefixes.add(f"{base}_{tail}")
for b in range(NS):
    base = f"lycoris_single_transformer_blocks_{b}"
    expected_prefixes.add(f"{base}_attn_to_qkv_mlp_proj")
    expected_prefixes.add(f"{base}_attn_to_out")

missing = expected_prefixes - set(prefixes)
extra = set(prefixes) - expected_prefixes
assert not missing, f"missing module prefixes: {sorted(missing)[:5]} (+{len(missing)-5 if len(missing)>5 else 0})"
assert not extra, f"unexpected module prefixes: {sorted(extra)[:5]}"
for p, sufs in prefixes.items():
    assert "alpha" in sufs, f"{p}: missing alpha"
    w1ok = ("lokr_w1" in sufs) or ("lokr_w1_a" in sufs and "lokr_w1_b" in sufs)
    w2ok = ("lokr_w2" in sufs) or ("lokr_w2_a" in sufs and "lokr_w2_b" in sufs)
    assert w1ok and w2ok, f"{p}: incomplete LoKr key family: {sorted(sufs)}"
print(f"PASS key-set: {len(prefixes)} modules, all complete LoKr families "
      f"(expected {len(expected_prefixes)} for targets=all)")

# ── 2. suffix schema vs a real upstream save ──────────────────────────────────
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 48, bias=False)

    def forward(self, x):
        return self.proj(x)


net = Tiny()
wrapper = create_lycoris(net, 1.0, linear_dim=RANK, linear_alpha=ALPHA,
                         algo="lokr", factor=-1)
wrapper.apply_to()
wrapper.save_weights("/tmp/lokr_upstream_ref_save.safetensors", torch.bfloat16, {})
ref_sd = load_file("/tmp/lokr_upstream_ref_save.safetensors")
ref_sufs = {k.rsplit(".", 1)[1] for k in ref_sd}
ours_sufs = {k.rsplit(".", 1)[1] for k in sd}
print(f"  upstream save suffixes: {sorted(ref_sufs)}")
print(f"  trainer  save suffixes: {sorted(ours_sufs)}")
assert ours_sufs <= {"lokr_w1", "lokr_w1_a", "lokr_w1_b",
                     "lokr_w2", "lokr_w2_a", "lokr_w2_b", "alpha"}, ours_sufs
assert "alpha" in ref_sufs and "alpha" in ours_sufs
print("PASS suffix schema: trainer keys use the upstream LoKr suffix classes")

# ── 3. upstream module load per target class ──────────────────────────────────
CASES = [
    ("lycoris_transformer_blocks_0_attn_to_q", D, D, FACTOR_ATTN),
    ("lycoris_transformer_blocks_0_ff_linear_in", D, 2 * F, FACTOR_FF),
    ("lycoris_single_transformer_blocks_0_attn_to_qkv_mlp_proj", D, 3 * D + 2 * F, FACTOR_SINGLE),
    ("lycoris_single_transformer_blocks_0_attn_to_out", D + F, D, FACTOR_SINGLE),
]
for p, din, dout, fac in CASES:
    g = lambda s: sd.get(f"{p}.{s}")
    base = nn.Linear(din, dout, bias=False)
    with torch.no_grad():
        mod = LokrModule.make_module_from_state_dict(
            p, base,
            g("lokr_w1"), g("lokr_w1_a"), g("lokr_w1_b"),
            g("lokr_w2"), g("lokr_w2_a"), g("lokr_w2_b"),
            None, None, g("alpha"), None,
        )
    in_m, in_n = factorization(din, fac)
    out_l, out_k = factorization(dout, fac)
    w1 = g("lokr_w1")
    assert w1 is not None and tuple(w1.shape) == (out_l, in_m), \
        f"{p}: w1 shape {tuple(w1.shape)} != upstream split ({out_l},{in_m})"
    w2a, w2b = g("lokr_w2_a"), g("lokr_w2_b")
    assert w2a is not None and tuple(w2a.shape) == (out_k, RANK)
    assert tuple(w2b.shape) == (RANK, in_n)
    assert abs(mod.scale - ALPHA / RANK) < 1e-6, f"{p}: scale {mod.scale}"
    # upstream weight reconstruction == kron from the file factors (bit-exact)
    dw = mod.get_diff_weight(1.0)[0].float() / mod.scale  # un-double-scale quirk
    kr = torch.kron(w1.float(), (w2a.float() @ w2b.float()))
    maxd = (dw - kr).abs().max().item()
    znorm = w2b.float().abs().sum().item()
    print(f"  {p}: split=({out_l},{out_k})x({in_m},{in_n}) "
          f"max|get_diff_weight/scale - kron|={maxd} |w2b|_1={znorm:.6f}")
    assert maxd == 0.0, f"{p}: upstream reconstruction != file factors"
    assert znorm > 0.0, f"{p}: zero leg still zero — file is an init dump, not trained"
print("PASS upstream load: LokrModule accepts + reconstructs every target class, trained legs nonzero")
print("ALL LOAD CHECKS PASS — lokr_st_load_check")
