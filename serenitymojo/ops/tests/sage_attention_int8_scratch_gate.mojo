# serenitymojo/ops/tests/sage_attention_int8_scratch_gate.mojo
#
# Gate for the preallocated-scratch Sage entry point. The scratch path must be
# BIT-IDENTICAL to the per-call-allocation path (both run
# `_sage_enqueue_forward`), including when one scratch sized for a larger
# sequence is reused at smaller S — the product reuse pattern that fixes the
# per-call transient churn OOM (video-0177).

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.sage_attention_int8 import (
    SageInt8Scratch,
    sage_attention_int8_fwd_dynamic,
    sage_attention_int8_fwd_scratch,
)


comptime B = 1
comptime H = 56
comptime D = 128
# 2836 exercises the non-multiple-of-64 masked tail; 1024 exercises reuse of
# the same (larger) scratch at a smaller sequence.
comptime S_LARGE = 2836
comptime S_SMALL = 1024


def _check_case(
    label: String,
    s: Int,
    seed: Int,
    scratch: SageInt8Scratch,
    ctx: DeviceContext,
) raises:
    var shape: List[Int] = [B, s, H, D]
    var q = randn(shape.copy(), UInt64(seed), STDtype.BF16, ctx)
    var k = randn(shape.copy(), UInt64(seed + 1), STDtype.BF16, ctx)
    var v = randn(shape.copy(), UInt64(seed + 2), STDtype.BF16, ctx)
    var scale = Float32(1.0) / sqrt(Float32(D))

    var out_dynamic = sage_attention_int8_fwd_dynamic(q, k, v, scale, ctx)
    var host_dynamic = out_dynamic.to_host_bf16(ctx)
    var out_scratch = sage_attention_int8_fwd_scratch(
        q, k, v, scale, scratch, ctx
    )
    var host_scratch = out_scratch.to_host_bf16(ctx)

    if len(host_dynamic) != len(host_scratch):
        raise Error(label + ": output length mismatch")
    var mismatches = 0
    for i in range(len(host_dynamic)):
        if host_dynamic[i] != host_scratch[i]:
            mismatches += 1
    print(
        label, ": S=", s, " values=", len(host_dynamic),
        " bit mismatches=", mismatches,
    )
    if mismatches != 0:
        raise Error(label + ": scratch path is not bit-identical")


def main() raises:
    var ctx = DeviceContext()
    var scratch = SageInt8Scratch(S_LARGE, H, ctx)
    print(
        "sage scratch gate: preallocated",
        Float64(scratch.resident_bytes()) / (1024.0 * 1024.0),
        "MiB for max_s=", S_LARGE,
    )
    _check_case(String("full-capacity"), S_LARGE, 202, scratch, ctx)
    _check_case(String("reuse-smaller-s"), S_SMALL, 404, scratch, ctx)
    # Same S again through the same scratch: proves repeated reuse (the
    # 50-blocks-per-step product pattern) is stable call-over-call.
    _check_case(String("reuse-repeat"), S_SMALL, 606, scratch, ctx)
    # Capacity violations must fail loudly, never fall back.
    var overflow_shape: List[Int] = [B, S_LARGE + 64, H, D]
    var q = randn(overflow_shape.copy(), UInt64(808), STDtype.BF16, ctx)
    var k = randn(overflow_shape.copy(), UInt64(809), STDtype.BF16, ctx)
    var v = randn(overflow_shape.copy(), UInt64(810), STDtype.BF16, ctx)
    var rejected = False
    try:
        _ = sage_attention_int8_fwd_scratch(
            q, k, v, Float32(1.0) / sqrt(Float32(D)), scratch, ctx
        )
    except:
        rejected = True
    if not rejected:
        raise Error("oversized S was not rejected by the scratch path")
    print("oversized-S rejection: PASS")
    print("SAGE SCRATCH GATE: PASS")
