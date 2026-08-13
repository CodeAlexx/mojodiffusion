# serenitymojo/models/ltx2/parity/ltx2_av_ckpt_equiv_probe.mojo
#
# Does moving the AV TRAINING arm onto the fp8 checkpoint change the weights?
#
# The AV trainer today loads from `...-fp8-dequant-bf16.safetensors` via
# `LTX2AVBlockWeights.load` (explicit key list, BF16 straight off disk). The
# resident block store lives on `...-fp8.safetensors` and materialises via
# `LTX2BlockStream.load_block_bf16` -> `from_fp8_block` (F8_E4M3 dequanted to
# BF16 on GPU with the per-tensor weight_scale).
#
# If the dequant-bf16 file was produced by that same dequant, the two paths give
# BIT-IDENTICAL weights and the switch is numerically free. If not, every AV gate
# shifts. This probe answers that per key, rather than assuming either way.
#
# NOTE this is the DEQUANT path on both sides — NOT `load_block_fp8_resident` /
# `from_fp8_resident` (raw fp8 + linear_fp8), which `to_f32` refuses outright
# (ltx2_dit.mojo:1010-1017) and which the AV F32 training stack cannot use.
#
# Run: rm -f serenitymojo.mojopkg; pixi run mojo build -O2 -I . -Xlinker -lm \
#   -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_ckpt_equiv_probe.mojo \
#   -o /tmp/ltx2_av_ckpt_equiv && /tmp/ltx2_av_ckpt_equiv

from max.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.env import env_or, serenity_checkpoint
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2AVBlockWeights
from serenitymojo.offload.ltx2_block_stream import LTX2BlockStream

comptime CKPT_BF16_NAME = "ltx-2.3-22b-distilled-fp8-dequant-bf16.safetensors"
comptime CKPT_FP8_NAME = "ltx-2.3-22b-distilled-fp8.safetensors"


def _cos_and_maxabs(a: List[Float32], b: List[Float32]) raises -> Tuple[Float64, Float64, Int]:
    if len(a) != len(b):
        raise Error(String("length ") + String(len(a)) + " vs " + String(len(b)))
    var dot = 0.0; var na = 0.0; var nb = 0.0
    var mx = 0.0
    var ndiff = 0
    for i in range(len(a)):
        var x = Float64(a[i]); var y = Float64(b[i])
        dot += x * y; na += x * x; nb += y * y
        var d = x - y
        if d < 0.0: d = -d
        if d > mx: mx = d
        if a[i] != b[i]: ndiff += 1
    return (dot / (sqrt(na) * sqrt(nb) + 1e-30), mx, ndiff)


def _check_block(
    bi: Int, cfg: LTX2Config, ctx: DeviceContext,
    ckpt_bf16: String, ckpt_fp8: String,
) raises -> Int:
    print("")
    print("  ── block", bi, "──")
    var t0 = perf_counter_ns()
    var wa = LTX2AVBlockWeights.load(ckpt_bf16, bi, cfg, ctx)
    var t1 = perf_counter_ns()
    print("    A: LTX2AVBlockWeights.load(dequant-bf16 ckpt) =",
          Float64(t1 - t0) / 1.0e6, "ms")

    var stream = LTX2BlockStream.open(ckpt_fp8)
    var n_fp8 = stream.fp8_tensor_count(bi)
    var t2 = perf_counter_ns()
    var blk = stream.load_block_bf16(bi, ctx)
    var wb = LTX2AVBlockWeights.from_fp8_block(blk^, cfg, ctx)
    var t3 = perf_counter_ns()
    print("    B: stream.load_block_bf16(fp8 ckpt) -> from_fp8_block =",
          Float64(t3 - t2) / 1.0e6, "ms   (fp8 tensors in block:", n_fp8, ")")

    var n_keys = 0
    var n_exact = 0
    var n_shifted = 0
    var n_missing = 0
    var worst_cos = 1.0
    var worst_key = String("")
    var worst_max = 0.0
    for ref e in wa.name_to_idx.items():
        var key = e.key
        n_keys += 1
        if not wb._has(key):
            n_missing += 1
            print("    MISSING in B:", key)
            continue
        ref ta = wa._w(key)
        ref tb = wb._w(key)
        if ta.nbytes() != tb.nbytes():
            print("    SIZE MISMATCH", key, ta.nbytes(), "vs", tb.nbytes())
            n_shifted += 1
            continue
        var ha = ta.to_host(ctx)
        var hb = tb.to_host(ctx)
        var r = _cos_and_maxabs(ha, hb)
        if r[2] == 0:
            n_exact += 1
        else:
            n_shifted += 1
            if r[0] < worst_cos:
                worst_cos = r[0]
                worst_key = key
                worst_max = r[1]
    print("    keys=", n_keys, " BIT-IDENTICAL=", n_exact, " DIFFERING=", n_shifted,
          " MISSING=", n_missing)
    if n_shifted > 0:
        print("    worst key:", worst_key, " cos=", worst_cos, " max_abs=", worst_max)
    return n_shifted + n_missing


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var ckpt_bf16 = env_or(
        "LTX2_AV_CKPT_BF16", serenity_checkpoint(String(CKPT_BF16_NAME)),
    )
    var ckpt_fp8 = env_or(
        "LTX2_AV_CKPT_FP8", serenity_checkpoint(String(CKPT_FP8_NAME)),
    )
    print("=== AV checkpoint equivalence: dequant-bf16 file vs fp8 file (dequant path) ===")
    print("  A:", ckpt_bf16)
    print("  B:", ckpt_fp8)

    # block 0 = BF16 boundary block (no fp8 tensors); block 4 = inner fp8 block.
    var bad = 0
    bad += _check_block(0, cfg, ctx, ckpt_bf16, ckpt_fp8)
    bad += _check_block(4, cfg, ctx, ckpt_bf16, ckpt_fp8)

    print("")
    if bad == 0:
        print("VERDICT: the two checkpoints are BIT-IDENTICAL on the dequant path —")
        print("  switching the AV arm to the fp8 file moves NO weight bits.")
    else:
        print("VERDICT: the checkpoints DIFFER (", bad, "keys) — switching the AV arm")
        print("  to the fp8 file WILL shift every AV gate. Re-anchoring required.")
    print("ltx2_av_ckpt_equiv_probe DONE")
