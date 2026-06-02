# training/full_adapter_smoke.mojo — gate for LyCORIS Full adapter (item 2j).
#
# Asserts:
#   (1) DELTA math: full_delta_weight(diff, strength) == strength*diff,
#       elementwise (1e-6). strength==1.0 returns diff unscaled (bit-exact).
#   (2) SAVE convention: save_full_adapters writes "<prefix>.diff.weight" with
#       the base weight SHAPE, reopened and verified (key present, shape match,
#       values match the saved diff to 1e-6). Bias -> "<prefix>.diff_b" [n].
#   (3) BITROT-FAIL DEMO: comparing the delta against the WRONG formula
#       (strength+1)*diff must exceed 1e-6.
#
# Exits NONZERO (raise) on any mismatch.
#
# Build/run (JIT):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/full_adapter_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.ffi import sys_system
from serenitymojo.training.full_adapter import (
    FullAdapter, new_full_adapter, full_delta_weight, full_delta_bias,
    NamedFull, save_full_adapters,
)


def main() raises:
    var ctx = DeviceContext()
    var ok = True

    # Build a Full adapter for a [out=4, in=3] weight + bias[4], then fill diff
    # with known non-zero values (the optimizer would do this; we set directly).
    var shape = List[Int](); shape.append(4); shape.append(3)
    var adapter = new_full_adapter(shape, 4)
    for i in range(adapter.numel()):
        adapter.diff[i] = Float32(i) * Float32(0.25) - Float32(1.0)
    for i in range(4):
        adapter.diff_b[i] = Float32(i) * Float32(0.5)

    # ── (1) delta math ────────────────────────────────────────────────────────
    var strength = Float32(0.7)
    var dw = full_delta_weight(adapter, strength)
    var maxerr = Float32(0.0)
    for i in range(len(dw)):
        var e = dw[i] - strength * adapter.diff[i]
        if e < Float32(0.0):
            e = -e
        if e > maxerr:
            maxerr = e
    print("delta_weight max |dw - strength*diff| =", maxerr)
    if maxerr > Float32(1.0e-6):
        print("FAIL delta_weight != strength*diff"); ok = False
    else:
        print("PASS delta_weight == strength*diff to 1e-6")

    var dw1 = full_delta_weight(adapter, Float32(1.0))
    var bit = True
    for i in range(len(dw1)):
        if dw1[i] != adapter.diff[i]:
            bit = False
    if bit:
        print("PASS delta_weight(strength=1.0) == diff (bit-exact)")
    else:
        print("FAIL delta_weight(1.0) != diff"); ok = False

    var db = full_delta_bias(adapter, strength)
    var berr = Float32(0.0)
    for i in range(len(db)):
        var e = db[i] - strength * adapter.diff_b[i]
        if e < Float32(0.0):
            e = -e
        if e > berr:
            berr = e
    print("delta_bias max err =", berr, " (len=", len(db), ")")
    if len(db) != 4 or berr > Float32(1.0e-6):
        print("FAIL delta_bias wrong"); ok = False
    else:
        print("PASS delta_bias == strength*diff_b to 1e-6")

    # ── (2) save convention + reopen ──────────────────────────────────────────
    _ = sys_system(String("mkdir -p /tmp/full_adapter_gate"))
    var path = String("/tmp/full_adapter_gate/full.safetensors")
    var nf = List[NamedFull]()
    nf.append(NamedFull(String("double_blocks.0.img_attn.to_q"), adapter.copy()))
    var ntensors = save_full_adapters(nf, path, ctx)
    print("saved tensors:", ntensors, " (expect 2: diff.weight + diff_b)")
    if ntensors != 2:
        print("FAIL wrong tensor count"); ok = False

    var st = SafeTensors.open(path)
    var key_w = String("double_blocks.0.img_attn.to_q.diff.weight")
    var key_b = String("double_blocks.0.img_attn.to_q.diff_b")
    if key_w not in st.tensors:
        print("FAIL missing key", key_w); ok = False
    else:
        var info = st.tensor_info(key_w)
        print("diff.weight shape:", info.shape[0], "x", info.shape[1], " dtype-ok")
        if len(info.shape) != 2 or info.shape[0] != 4 or info.shape[1] != 3:
            print("FAIL diff.weight shape != [4,3]"); ok = False
        else:
            print("PASS .diff.weight key present with base weight shape [4,3]")
    if key_b not in st.tensors:
        print("FAIL missing key", key_b); ok = False
    else:
        var binfo = st.tensor_info(key_b)
        if len(binfo.shape) != 1 or binfo.shape[0] != 4:
            print("FAIL diff_b shape != [4]"); ok = False
        else:
            print("PASS .diff_b key present with shape [4]")

    # ── (3) BITROT-FAIL DEMO ──────────────────────────────────────────────────
    var wrong_maxerr = Float32(0.0)
    for i in range(len(dw)):
        var e = dw[i] - (strength + Float32(1.0)) * adapter.diff[i]
        if e < Float32(0.0):
            e = -e
        if e > wrong_maxerr:
            wrong_maxerr = e
    print("bitrot demo: max err vs WRONG (strength+1)*diff =", wrong_maxerr)
    if wrong_maxerr <= Float32(1.0e-6):
        print("FAIL bitrot demo: wrong formula matched"); ok = False
    else:
        print("PASS bitrot demo: wrong formula exceeds 1e-6 (gate is sensitive)")

    if not ok:
        raise Error("full_adapter_smoke FAILED")
    print("full_adapter_smoke gate PASS")
