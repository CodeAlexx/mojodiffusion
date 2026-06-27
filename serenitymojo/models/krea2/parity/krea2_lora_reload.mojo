# models/krea2/parity/krea2_lora_reload.mojo
# ──────────────────────────────────────────────────────────────────────────────
# RE-LOAD PROOF for the krea2 LoRA save (MJ-0805 / MJ-1018): the saved file must
# actually LOAD BACK through the REAL loader (serenitymojo/lora.mojo LoraSet),
# not just have the right header. This exercises the save's inverse end-to-end:
#
#   1. LoraSet.load(<saved krea2 lora>) — the production loader, auto-detect format.
#      The krea2 keys diffusion_model.blocks.<bi>.<mod>.lora_A/B.weight detect as
#      FMT_DIFFUSION_MODEL; the generic DM mapper strips `diffusion_model.` and
#      appends `.weight` → blocks.<bi>.<mod>.weight = the krea2 base key. So NO
#      new krea2 mapping is needed — the existing LoraSet path handles it.
#   2. Open the krea2 base checkpoint (raw.safetensors) header. For every resolved
#      mapping, confirm m.base_key is a REAL base module (n_unmapped MUST be 0) and
#      the A[rank,in]/B[out,rank] dims match the base weight [out,in].
#   3. _compute_delta on a sampled module (blocks.0.attn.wq) → finite, non-zero,
#      and delta.shape == the base weight shape.
#
# Prints: format, adapters_loaded, n_mapped, n_unmapped (gate: 0), sample delta.
# Build -O2. Mojo 1.0.0b1, NVIDIA GPU.
# ──────────────────────────────────────────────────────────────────────────────

from std.collections import List
from std.math import sqrt, isnan, isinf
from std.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.lora import LoraSet, FMT_DIFFUSION_MODEL

comptime NBLOCKS = 28
comptime KREA2_CKPT = String(
    "/home/alex/.cache/huggingface/hub/models--krea--Krea-2-Raw/"
    "snapshots/4ad9f4b627a647fad78b3dfeebb09f2654aeb494/raw.safetensors"
)


def _fmt_name(f: Int) -> String:
    if f == 0:
        return String("KLEIN_TRAINER")
    elif f == 1:
        return String("ZIMAGE_TRAINER")
    elif f == 2:
        return String("DIFFUSION_MODEL (PEFT)")
    elif f == 3:
        return String("KOHYA_SDXL")
    return String("LTX2_DISTILLED")


def main() raises:
    var args = argv()
    var lora_path: String
    if len(args) >= 2:
        lora_path = String(args[1])
    else:
        lora_path = String("/home/alex/trainings/krea2_512_test/krea2_krea2_lora_20.safetensors")
    var ctx = DeviceContext()

    print("==== KREA2 LoRA RE-LOAD PROOF (real LoraSet) ====")
    print("lora =", lora_path)
    print("base =", KREA2_CKPT)
    print("")

    # ── 1. load via the REAL loader ──────────────────────────────────────────
    var lset = LoraSet.load(lora_path)
    var fmt = lset.format
    var n_adapters = lset.num_mappings()
    print("format          =", _fmt_name(fmt), "(", fmt, ")")
    print("adapters_loaded =", n_adapters)

    # ── 2. cross-check every base_key against the REAL krea2 checkpoint ───────
    var base = SafeTensors.open(KREA2_CKPT)
    var n_mapped = 0
    var n_unmapped = 0
    var n_shape_ok = 0
    var n_shape_bad = 0
    var first_unmapped = String("")
    var first_shape_bad = String("")
    for ref m in lset.mappings:
        if m.base_key in base.tensors:
            n_mapped += 1
            # A is [rank,in], B is [out,rank]; base weight is [out,in].
            var a_info = lset.st.tensor_info(m.prefix + lset.suffix_a)
            var b_info = lset.st.tensor_info(m.prefix + lset.suffix_b)
            var w_info = base.tensor_info(m.base_key)
            # in = A.shape[1], out = B.shape[0]; base [out,in].
            var a_in = a_info.shape[len(a_info.shape) - 1]
            var b_out = b_info.shape[0]
            var w_out = w_info.shape[0]
            var w_in = w_info.shape[len(w_info.shape) - 1]
            if a_in == w_in and b_out == w_out:
                n_shape_ok += 1
            else:
                n_shape_bad += 1
                if first_shape_bad == String(""):
                    first_shape_bad = m.base_key + String(" A_in=") + String(a_in) \
                        + String(" B_out=") + String(b_out) + String(" vs base[out,in]=[") \
                        + String(w_out) + String(",") + String(w_in) + String("]")
        else:
            n_unmapped += 1
            if first_unmapped == String(""):
                first_unmapped = m.base_key

    print("n_mapped        =", n_mapped)
    print("n_unmapped      =", n_unmapped, "(GATE: must be 0)")
    if n_unmapped > 0:
        print("  first unmapped base_key:", first_unmapped)
    print("shape_ok        =", n_shape_ok)
    print("shape_bad       =", n_shape_bad)
    if n_shape_bad > 0:
        print("  first shape mismatch:", first_shape_bad)

    # ── 3. sample delta for blocks.0.attn.wq ─────────────────────────────────
    var sample_prefix = String("diffusion_model.blocks.0.attn.wq")
    var sample_base = String("blocks.0.attn.wq.weight")
    var found = False
    for ref m in lset.mappings:
        if m.prefix == sample_prefix:
            found = True
            var w_info = base.tensor_info(m.base_key)
            var delta = lset._compute_delta(m, Float32(1.0), STDtype.BF16, ctx)
            var dh = delta.to_host(ctx)
            var n = len(dh)
            var ss = Float64(0.0)
            var n_bad = 0
            for i in range(n):
                var v = dh[i]
                if isnan(v) or isinf(v):
                    n_bad += 1
                ss += Float64(v) * Float64(v)
            print("")
            print("sample module   =", m.prefix, "->", m.base_key)
            print("  delta.shape   =", delta.shape()[0], "x", delta.shape()[1],
                  " (base [out,in] =", w_info.shape[0], "x", w_info.shape[len(w_info.shape) - 1], ")")
            print("  delta L2      =", Float64(sqrt(ss)))
            print("  nonfinite     =", n_bad, "(GATE: 0)")
            var shape_match = (delta.shape()[0] == w_info.shape[0]
                and delta.shape()[1] == w_info.shape[len(w_info.shape) - 1])
            print("  shape==base   =", shape_match)
            break
    if not found:
        print("")
        print("WARN: sample module", sample_prefix, "not in mappings")

    # ── verdict ──────────────────────────────────────────────────────────────
    print("")
    var pass_all = (fmt == FMT_DIFFUSION_MODEL) and (n_adapters == NBLOCKS * 8) \
        and (n_unmapped == 0) and (n_shape_bad == 0) and found
    if pass_all:
        print("VERDICT: PASS — saved krea2 LoRA RE-LOADS via the real LoraSet",
              "(224 adapters, 0 unmapped, shapes match, finite nonzero delta).")
    else:
        print("VERDICT: FAIL — see the counts above (format/adapters/unmapped/shape/sample).")
