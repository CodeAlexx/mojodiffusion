# Is Mojo's native Float32 -> BF16 cast BIASED relative to PyTorch's RNE?
#
# Motivation: gemma4_ltx_streamed's per-layer math gates at cos 0.99999 (both
# layer types, isolated against oracle-hook inputs) yet the 48-layer end-to-end
# lands at 0.9935 — ~10x more drift than linear accumulation of the per-layer
# error would explain. A SYSTEMATIC (signed) rounding bias compounds coherently
# across the 192 F32->BF16 stores a 48-layer pass performs (4 norms/layer),
# whereas a symmetric tie-break difference would not.
#
# This decides it on host arithmetic alone (no GPU, no checkpoint): compare
# `Float32.cast[bfloat16]()` against `ops.torch_bf16.torch_bf16_rne_value` over
# a spread of magnitudes, and report BOTH the disagreement rate and the MEAN
# SIGNED error (the bias). A near-zero mean signed error kills the hypothesis.

from serenitymojo.ops.torch_bf16 import torch_bf16_rne_value


def main() raises:
    print("== mojo native f32->bf16 cast vs torch RNE ==")

    # A deterministic spread across binades, including exact ties and values
    # just above/below the midpoint between adjacent BF16 values.
    var seed: Int = 12345
    var n = 0
    var n_diff = 0
    var sum_native: Float64 = 0.0
    var sum_rne: Float64 = 0.0
    var sum_signed_native: Float64 = 0.0  # (native - exact)
    var sum_signed_rne: Float64 = 0.0     # (rne    - exact)
    var max_gap: Float64 = 0.0

    for i in range(200000):
        # xorshift-ish LCG for a reproducible pseudo-random mantissa
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        var u = Float64(seed) / Float64(0x7FFFFFFF)  # [0,1)
        # spread over magnitudes typical of a residual stream: 1e-3 .. 3e2
        var mag_sel = i % 6
        var scale: Float64 = 1.0
        if mag_sel == 0:
            scale = 0.001
        elif mag_sel == 1:
            scale = 0.1
        elif mag_sel == 2:
            scale = 1.0
        elif mag_sel == 3:
            scale = 10.0
        elif mag_sel == 4:
            scale = 60.0
        else:
            scale = 300.0
        var exact = Float64((u * 2.0 - 1.0) * scale)
        if exact == 0.0:
            continue
        var v = Float32(exact)

        var native = Float64(v.cast[DType.bfloat16]().cast[DType.float32]())
        var rne = Float64(
            torch_bf16_rne_value(v).cast[DType.float32]()
        )

        n += 1
        sum_native += native
        sum_rne += rne
        sum_signed_native += native - Float64(v)
        sum_signed_rne += rne - Float64(v)
        if native != rne:
            n_diff += 1
            var g = native - rne
            if g < 0.0:
                g = -g
            if g > max_gap:
                max_gap = g

    print("samples              =", n)
    print("native != rne count  =", n_diff)
    print("disagreement rate    =", Float64(n_diff) / Float64(n))
    print("max |native - rne|   =", max_gap)
    print("mean signed err native =", sum_signed_native / Float64(n))
    print("mean signed err rne    =", sum_signed_rne / Float64(n))
    print("---")
    print(
        "VERDICT: a mean-signed-error near 0 for BOTH means the native cast is"
        " unbiased and rounding bias does NOT explain the e2e drift."
    )
