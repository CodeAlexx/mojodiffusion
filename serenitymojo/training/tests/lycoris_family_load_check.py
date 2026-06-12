# lycoris_family_load_check.py — ecosystem half of the T2.F family gate.
#
# Run AFTER /tmp/lycoris_family_parity (the Mojo gate), which writes
#   /tmp/mojo_locon.safetensors  /tmp/mojo_tucker.safetensors
#   /tmp/mojo_loha.safetensors   /tmp/mojo_oft.safetensors
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

if not ok:
    raise SystemExit("lycoris_family_load_check: FAIL (see lines above)")
print("lycoris_family_load_check ALL GATES PASS "
      "(upstream lycoris loads all 4 Mojo-saved files, forward cos>=0.99999)")
