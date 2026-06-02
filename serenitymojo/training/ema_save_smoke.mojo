# training/ema_save_smoke.mojo — gate for the EMA SHADOW CHECKPOINT (item 2i).
#
# Closes SKEPTIC FINDING 1: the EMA shadow was computed per-step but never saved
# (the checkpoint was byte-identical to EMA-off). save_klein_lora_ema now writes
# the shadow params to a sibling file. This gate proves the saved bytes come
# from the SHADOW lists, not the live set.
#
# Procedure:
#   - build a tiny KleinLoraSet (1 double + 1 single, D=8, rank=2);
#   - construct shadow A/B lists that DIFFER from live (live + 100.0);
#   - save_klein_lora(live) -> live.safetensors;
#   - save_klein_lora_ema(live, shadows) -> ema.safetensors;
#   - reopen both, read one adapter's lora_A.weight:
#       (1) ema A == shadow A  (1e-6)         -> shadow IS what got saved
#       (2) ema A != live A                    -> not silently saving live
#       (3) live file A == live set A (1e-6)   -> control
#   - BITROT DEMO: comparing ema A against live A must exceed 1e-6.
#
# Exits NONZERO (raise) on any mismatch.
#
# Build/run (JIT):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/ema_save_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.ffi import sys_system
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.lora_save import _read_f32
from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraSet, build_klein_lora_set, save_klein_lora, save_klein_lora_ema,
)


def _offset_group(src: List[Float32]) -> List[Float32]:
    """live + 100 so any collision with live is impossible."""
    var out = List[Float32]()
    for v in src:
        out.append(v + Float32(100.0))
    return out^


def main() raises:
    var ctx = DeviceContext()
    var ok = True

    var lora = build_klein_lora_set(1, 1, 8, 2, Float32(2.0))
    # shadow lists in the SAME flat order the trainer allocates them.
    var ema_dbl_a = List[List[Float32]]()
    var ema_dbl_b = List[List[Float32]]()
    for i in range(len(lora.dbl)):
        ema_dbl_a.append(_offset_group(lora.dbl[i].a))
        ema_dbl_b.append(_offset_group(lora.dbl[i].b))
    var ema_sgl_a = List[List[Float32]]()
    var ema_sgl_b = List[List[Float32]]()
    for i in range(len(lora.sgl)):
        ema_sgl_a.append(_offset_group(lora.sgl[i].a))
        ema_sgl_b.append(_offset_group(lora.sgl[i].b))

    _ = sys_system(String("mkdir -p /tmp/ema_save_gate"))
    var live_path = String("/tmp/ema_save_gate/alina_lora_step1.safetensors")
    var ema_path = String("/tmp/ema_save_gate/alina_lora_step1_ema.safetensors")
    var nlive = save_klein_lora(lora, live_path, ctx)
    var nema = save_klein_lora_ema(
        lora, ema_dbl_a, ema_dbl_b, ema_sgl_a, ema_sgl_b, ema_path, ctx
    )
    print("saved live pairs=", nlive, " ema pairs=", nema)
    if nlive != nema or nlive != 6:
        print("FAIL pair count mismatch (expect 6 each: 4 double-slot + 2 single-slot)"); ok = False

    # read one double-block adapter's A from both files
    var key_a = String("double_blocks.0.img_attn.qkv_proj.lora_A.weight")
    var st_live = SafeTensors.open(live_path)
    var st_ema = SafeTensors.open(ema_path)
    if key_a not in st_live.tensors:
        print("FAIL live file missing key", key_a); ok = False
    if key_a not in st_ema.tensors:
        print("FAIL ema file missing key", key_a); ok = False

    var live_a = _read_f32(st_live, key_a, ctx)
    var ema_a = _read_f32(st_ema, key_a, ctx)
    var live_set_a = lora.dbl[0].a.copy()   # slot 0 == img_attn.qkv_proj
    var shadow_a = ema_dbl_a[0].copy()

    # ── (3) control: live file == live set ────────────────────────────────────
    var ctrl_err = Float32(0.0)
    for i in range(len(live_a)):
        var e = live_a[i] - live_set_a[i]
        if e < Float32(0.0): e = -e
        if e > ctrl_err: ctrl_err = e
    print("control max |live_file - live_set| =", ctrl_err)
    if ctrl_err > Float32(1.0e-6):
        print("FAIL live file does not match live set"); ok = False
    else:
        print("PASS control: live file A == live set A")

    # ── (1) ema file == shadow ────────────────────────────────────────────────
    var shadow_err = Float32(0.0)
    for i in range(len(ema_a)):
        var e = ema_a[i] - shadow_a[i]
        if e < Float32(0.0): e = -e
        if e > shadow_err: shadow_err = e
    print("ema max |ema_file - shadow| =", shadow_err)
    if shadow_err > Float32(1.0e-6):
        print("FAIL ema file does NOT match the shadow params (shadow not saved!)"); ok = False
    else:
        print("PASS ema file A == shadow A (shadow IS what got written)")

    # ── (2) ema file != live (the FINDING-1 regression check) ─────────────────
    var diff = Float32(0.0)
    for i in range(len(ema_a)):
        var e = ema_a[i] - live_a[i]
        if e < Float32(0.0): e = -e
        if e > diff: diff = e
    print("ema vs live max |ema - live| =", diff)
    if diff <= Float32(1.0e-6):
        print("FAIL ema file is byte-identical to live (shadow silently discarded — FINDING 1)"); ok = False
    else:
        print("PASS ema file DIFFERS from live (shadow is not discarded)")

    # ── BITROT DEMO: claim ema==live (the wrong expectation) must fail ─────────
    var wrong_ok = diff <= Float32(1.0e-6)
    if wrong_ok:
        print("FAIL bitrot demo: ema matched live (gate insensitive)"); ok = False
    else:
        print("PASS bitrot demo: ema != live by", diff, "(gate is sensitive to the shadow write)")

    if not ok:
        raise Error("ema_save_smoke FAILED")
    print("ema_save_smoke gate PASS")
