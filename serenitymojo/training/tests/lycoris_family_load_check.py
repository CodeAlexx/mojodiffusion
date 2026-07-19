# lycoris_family_load_check.py — ecosystem half of the T2.F family gate.
#
# Run AFTER /tmp/lycoris_family_parity (the Mojo gate), which writes
#   /tmp/mojo_locon.safetensors  /tmp/mojo_tucker.safetensors
#   /tmp/mojo_loha.safetensors   /tmp/mojo_oft.safetensors
#   /tmp/mojo_lokr{1,2,3}.safetensors  /tmp/mojo_boft.safetensors
#   /tmp/mojo_dora.safetensors                           (T2.F-2 extension)
# This script proves the Mojo-saved files are loadable by the UPSTREAM lycoris
# loader (pip lycoris_lora 3.4.0) at VALUE level, not just key level:
#   1. key-set check: the Mojo file's keys must equal the upstream module's
#      state_dict keys (prefixed). alpha shape is [1] vs upstream 0-D — a
#      documented deviation; upstream loaders consume alpha via float(), which
#      accepts both.
#   2. load check: tensors are fed through each module class's OWN
#      make_module_from_state_dict, the upstream weight construction is run
#      (get_diff_weight / make_weight), and the forward on the oracle x must
#      match the oracle y (cos >= 0.99999, F64 accumulate).
#
# Run: python3 serenitymojo/training/tests/lycoris_family_load_check.py
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from lycoris.modules.locon import LoConModule
from lycoris.modules.loha import LohaModule
from lycoris.modules.diag_oft import DiagOFTModule
from lycoris.modules.lokr import LokrModule
from lycoris.modules.boft import ButterflyOFTModule

COS_BAR = 0.99999
torch.set_grad_enabled(False)  # upstream's loader wrapper runs under no_grad
oracle = load_file("/tmp/lycoris_family_oracle.safetensors")
ok = True


def cos(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()))


def check(name, got, exp):
    global ok
    c = cos(got, exp)
    mx = float((got.float() - exp.float()).abs().max())
    if c >= COS_BAR:
        print(f"PASS ({name} load): cos={c:.9f} max|d|={mx:.3e}")
    else:
        print(f"FAIL ({name} load): cos={c:.9f} max|d|={mx:.3e}")
        ok = False


def keycheck(name, sd, expected):
    global ok
    if set(sd.keys()) == set(expected):
        print(f"PASS ({name} keys): {sorted(sd.keys())}")
    else:
        print(f"FAIL ({name} keys): got {sorted(sd.keys())} want {sorted(expected)}")
        ok = False


# ── LoCon ─────────────────────────────────────────────────────────────────────
p = "lora_unet_locon_t"
sd = load_file("/tmp/mojo_locon.safetensors")
keycheck("locon", sd, [f"{p}.lora_down.weight", f"{p}.lora_up.weight", f"{p}.alpha"])
base = torch.nn.Conv2d(8, 16, 3, 1, 1, bias=False)
m = LoConModule.make_module_from_state_dict(
    p, base,
    sd[f"{p}.lora_up.weight"].float(), sd[f"{p}.lora_down.weight"].float(),
    None, float(sd[f"{p}.alpha"]), None,
)
m.eval()
x = oracle["locon.x_nhwc"].permute(0, 3, 1, 2).contiguous()
y = F.conv2d(x, m.get_diff_weight(1)[0].float(), stride=1, padding=1)
check("locon", y.permute(0, 2, 3, 1).contiguous(), oracle["locon.y_nhwc"])

# ── Tucker ────────────────────────────────────────────────────────────────────
p = "lora_unet_tucker_t"
sd = load_file("/tmp/mojo_tucker.safetensors")
keycheck("tucker", sd, [f"{p}.lora_down.weight", f"{p}.lora_mid.weight",
                        f"{p}.lora_up.weight", f"{p}.alpha"])
base = torch.nn.Conv2d(8, 16, 3, 2, 1, bias=False)
m = LoConModule.make_module_from_state_dict(
    p, base,
    sd[f"{p}.lora_up.weight"].float(), sd[f"{p}.lora_down.weight"].float(),
    sd[f"{p}.lora_mid.weight"].float(), float(sd[f"{p}.alpha"]), None,
)
assert m.tucker, "upstream module did not enter tucker mode from the Mojo file"
m.eval()
x = oracle["tucker.x_nhwc"].permute(0, 3, 1, 2).contiguous()
y = F.conv2d(x, m.get_diff_weight(1)[0].float(), stride=2, padding=1)
check("tucker", y.permute(0, 2, 3, 1).contiguous(), oracle["tucker.y_nhwc"])

# ── LoHa ──────────────────────────────────────────────────────────────────────
p = "lora_unet_loha_t"
sd = load_file("/tmp/mojo_loha.safetensors")
keycheck("loha", sd, [f"{p}.hada_w1_a", f"{p}.hada_w1_b",
                      f"{p}.hada_w2_a", f"{p}.hada_w2_b", f"{p}.alpha"])
base = torch.nn.Linear(12, 16, bias=False)
m = LohaModule.make_module_from_state_dict(
    p, base,
    sd[f"{p}.hada_w1_a"].float(), sd[f"{p}.hada_w1_b"].float(),
    sd[f"{p}.hada_w2_a"].float(), sd[f"{p}.hada_w2_b"].float(),
    None, None, float(sd[f"{p}.alpha"]), None,
)
m.eval()
# Single-scale convention (the module's live bypass path + a1111/comfy apply
# alpha/rank ONCE; get_diff_weight double-applies it — see oracle generator note).
y = F.linear(oracle["loha.x"], (m.get_weight(m.shape) * m.scalar).float())
check("loha", y, oracle["loha.y"])

# ── Diag-OFT ──────────────────────────────────────────────────────────────────
p = "lora_unet_oft_t"
sd = load_file("/tmp/mojo_oft.safetensors")
keycheck("oft", sd, [f"{p}.oft_blocks", f"{p}.alpha"])
base = torch.nn.Linear(12, 16, bias=False)
with torch.no_grad():
    base.weight.copy_(oracle["oft.w_base"])
m = DiagOFTModule.make_module_from_state_dict(
    p, base, sd[f"{p}.oft_blocks"].float(), None, float(sd[f"{p}.alpha"]),
)
m.eval()
y = F.linear(oracle["oft.x"], m.make_weight(scale=1).float())
check("oft", y, oracle["oft.y"])

# ═══════════════════════ T2.F-2: LoKr / BOFT / DoRA ══════════════════════════
# LoKr forward convention: get_weight folds self.scale inside make_kron;
# get_diff_weight double-applies it (same class as the LoHa quirk) — the live
# forward path is get_weight(shape) * scalar, which we use here.
#
# x inputs are .clone()d: safetensors mmap pointers can be less aligned than
# fresh allocations, which makes F.linear pick a different GEMM path (±1 ulp)
# and would break the BIT-EXACT bar (measured 2026-06-11: 1.5e-8 mmap vs 0.0
# clone on identical bits).

def lokr_y(m, x):
    return F.linear(x, (m.get_weight(m.shape) * m.scalar).float())


# ── LoKr v1: W1 full + W2 factored ───────────────────────────────────────────
p = "lora_unet_lokr1"
sd = load_file("/tmp/mojo_lokr1.safetensors")
keycheck("lokr1", sd, [f"{p}.lokr_w1", f"{p}.lokr_w2_a", f"{p}.lokr_w2_b", f"{p}.alpha"])
base = torch.nn.Linear(8, 8, bias=False)
m = LokrModule.make_module_from_state_dict(
    p, base,
    sd[f"{p}.lokr_w1"].float(), None, None,
    None, sd[f"{p}.lokr_w2_a"].float(), sd[f"{p}.lokr_w2_b"].float(),
    None, None, float(sd[f"{p}.alpha"]), None,
)
assert m.use_w1 and not m.use_w2, "lokr1: upstream did not reconstruct W1full+W2factored"
m.eval()
check("lokr1", lokr_y(m, oracle["lokr1.x"].clone()), oracle["lokr1.y"])

# ── LoKr v2: BOTH full (upstream forces scale=1 regardless of alpha) ─────────
p = "lora_unet_lokr2"
sd = load_file("/tmp/mojo_lokr2.safetensors")
keycheck("lokr2", sd, [f"{p}.lokr_w1", f"{p}.lokr_w2", f"{p}.alpha"])
base = torch.nn.Linear(8, 8, bias=False)
m = LokrModule.make_module_from_state_dict(
    p, base,
    sd[f"{p}.lokr_w1"].float(), None, None,
    sd[f"{p}.lokr_w2"].float(), None, None,
    None, None, float(sd[f"{p}.alpha"]), None,
)
assert m.use_w1 and m.use_w2, "lokr2: upstream did not reconstruct both-full"
assert float(m.scale) == 1.0, "lokr2: upstream forced-scale=1 quirk missing"
m.eval()
check("lokr2", lokr_y(m, oracle["lokr2.x"].clone()), oracle["lokr2.y"])

# ── LoKr v3: BOTH factored (decompose_both) ──────────────────────────────────
p = "lora_unet_lokr3"
sd = load_file("/tmp/mojo_lokr3.safetensors")
keycheck("lokr3", sd, [f"{p}.lokr_w1_a", f"{p}.lokr_w1_b",
                       f"{p}.lokr_w2_a", f"{p}.lokr_w2_b", f"{p}.alpha"])
base = torch.nn.Linear(64, 64, bias=False)
m = LokrModule.make_module_from_state_dict(
    p, base,
    None, sd[f"{p}.lokr_w1_a"].float(), sd[f"{p}.lokr_w1_b"].float(),
    None, sd[f"{p}.lokr_w2_a"].float(), sd[f"{p}.lokr_w2_b"].float(),
    None, None, float(sd[f"{p}.alpha"]), None,
)
assert not m.use_w1 and not m.use_w2, "lokr3: upstream did not reconstruct both-factored"
m.eval()
check("lokr3", lokr_y(m, oracle["lokr3.x"].clone()), oracle["lokr3.y"])

# ── BOFT ──────────────────────────────────────────────────────────────────────
p = "lora_unet_boft_t"
sd = load_file("/tmp/mojo_boft.safetensors")
keycheck("boft", sd, [f"{p}.oft_blocks", f"{p}.alpha"])
# a1111/comfy + upstream algo_check distinguish BOFT from Diag-OFT by RANK:
# butterfly oft_blocks MUST be 4D [boft_m, num_blocks, b, b] (OFT is 3D).
if sd[f"{p}.oft_blocks"].ndim == 4 and ButterflyOFTModule.algo_check(sd, p):
    print(f"PASS (boft rank): oft_blocks is 4D {tuple(sd[f'{p}.oft_blocks'].shape)}"
          " — upstream algo_check selects ButterflyOFT")
else:
    print(f"FAIL (boft rank): oft_blocks ndim={sd[f'{p}.oft_blocks'].ndim}, want 4")
    ok = False
base = torch.nn.Linear(12, 8, bias=False)
with torch.no_grad():
    base.weight.copy_(oracle["boft.w_base"])
m = ButterflyOFTModule.make_module_from_state_dict(
    p, base, sd[f"{p}.oft_blocks"].float(), None, float(sd[f"{p}.alpha"]),
)
assert (m.boft_m, m.block_num, m.block_size) == (3, 4, 2), "boft: shape reconstruction drifted"
m.eval()
check("boft", F.linear(oracle["boft.x"].clone(), m.make_weight(scale=1).float()), oracle["boft.y"])

# ── DoRA (upstream lycoris DoRA == LoConModule(weight_decompose=True)) ────────
p = "lora_unet_dora_t"
sd = load_file("/tmp/mojo_dora.safetensors")
keycheck("dora", sd, [f"{p}.lora_down.weight", f"{p}.lora_up.weight",
                      f"{p}.dora_scale", f"{p}.alpha"])
base = torch.nn.Linear(12, 16, bias=False)
with torch.no_grad():
    base.weight.copy_(oracle["dora.w_base"])
m = LoConModule.make_module_from_state_dict(
    p, base,
    sd[f"{p}.lora_up.weight"].float(), sd[f"{p}.lora_down.weight"].float(),
    None, float(sd[f"{p}.alpha"]), sd[f"{p}.dora_scale"].float(),
)
assert m.wd, "dora: upstream did not enter weight-decompose mode from the Mojo file"
m.eval()
# FULL forward: DoRA replaces the effective weight (get_merged_weight).
check("dora", F.linear(oracle["dora.x"].clone(), m.get_merged_weight(multiplier=1)[0].float()),
      oracle["dora.y"])

if not ok:
    raise SystemExit("lycoris_family_load_check: FAIL (see lines above)")
print("lycoris_family_load_check ALL GATES PASS "
      "(upstream lycoris loads all 9 Mojo-saved files "
      "[locon/tucker/loha/oft/lokr x3/boft/dora], forward cos>=0.99999)")
