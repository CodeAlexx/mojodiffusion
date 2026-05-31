# lora_save_smoke.mojo — round-trip gate for training/lora_save.mojo.
#
# Build two small LoraAdapters with KNOWN A/B host values, save them via
# save_lora_peft, read them back via load_lora_for_resume, and assert the A/B
# host lists are BYTE-EXACT (F32 round-trips with no truncation, the same
# property loop.mojo's master-checkpoint gate relies on — max_abs == 0).
#
# Run (after the compile lock frees):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/training/lora_save_smoke.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_save import (
    NamedLora,
    save_lora_peft,
    load_lora_for_resume,
)


comptime OUT = "/tmp/serenitymojo_lora_roundtrip.safetensors"


# Deterministic small adapter: A [rank,in] = ramp, B [out,rank] = ramp*0.5.
def _make_adapter(rank: Int, in_f: Int, out_f: Int, base: Float32) -> LoraAdapter:
    var a = List[Float32]()
    for i in range(rank * in_f):
        a.append(base + Float32(i) * 0.125)
    var b = List[Float32]()
    for i in range(out_f * rank):
        b.append(base * 0.5 - Float32(i) * 0.0625)
    var ma = List[Float32]()
    var va = List[Float32]()
    for _ in range(rank * in_f):
        ma.append(Float32(0.0))
        va.append(Float32(0.0))
    var mb = List[Float32]()
    var vb = List[Float32]()
    for _ in range(out_f * rank):
        mb.append(Float32(0.0))
        vb.append(Float32(0.0))
    return LoraAdapter(a^, b^, rank, in_f, out_f, Float32(1.0), ma^, va^, mb^, vb^)


def _assert_exact(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        raise Error(name + ": length mismatch " + String(len(a)) + " vs " + String(len(b)))
    var maxd = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        var ad = d if d >= 0.0 else -d
        if ad > maxd:
            maxd = ad
    if maxd != Float32(0.0):
        raise Error(name + ": NOT byte-exact, max_abs_diff=" + String(maxd))
    print("  ", name, " byte-exact (max_abs_diff=0, n=", len(a), ")")


def main() raises:
    var ctx = DeviceContext()
    print("=== lora_save round-trip smoke ===")

    # Two modules with distinct shapes (rank/in/out) and distinct values.
    var ad0 = _make_adapter(4, 8, 6, Float32(0.3))   # A[4,8], B[6,4]
    var ad1 = _make_adapter(2, 5, 7, Float32(-1.1))  # A[2,5], B[7,2]

    var pfx0 = String("double_blocks.0.img_attn.to_q")
    var pfx1 = String("single_blocks.3.linear1")

    # Keep host copies of the ORIGINALS for the post-load comparison.
    var a0_orig = ad0.a.copy()
    var b0_orig = ad0.b.copy()
    var a1_orig = ad1.a.copy()
    var b1_orig = ad1.b.copy()

    var advs = List[NamedLora]()
    advs.append(NamedLora(pfx0, ad0^))
    advs.append(NamedLora(pfx1, ad1^))

    var n = save_lora_peft(advs, String(OUT), ctx)
    print("  saved", n, "adapter(s) ->", OUT)

    var prefixes = List[String]()
    prefixes.append(pfx0)
    prefixes.append(pfx1)
    var loaded = load_lora_for_resume(prefixes, Float32(1.0), String(OUT), ctx)
    if len(loaded) != 2:
        raise Error("expected 2 reloaded adapters, got " + String(len(loaded)))

    # Module 0
    if loaded[0].adapter.rank != 4 or loaded[0].adapter.in_f != 8 or loaded[0].adapter.out_f != 6:
        raise Error("module 0 shape mismatch on reload")
    _assert_exact("module0.A", a0_orig, loaded[0].adapter.a)
    _assert_exact("module0.B", b0_orig, loaded[0].adapter.b)

    # Module 1
    if loaded[1].adapter.rank != 2 or loaded[1].adapter.in_f != 5 or loaded[1].adapter.out_f != 7:
        raise Error("module 1 shape mismatch on reload")
    _assert_exact("module1.A", a1_orig, loaded[1].adapter.a)
    _assert_exact("module1.B", b1_orig, loaded[1].adapter.b)

    print("lora_save round-trip smoke PASS (PEFT keys, F32 byte-exact)")
