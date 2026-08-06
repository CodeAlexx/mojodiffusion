# MiniMax-H3 long-sequence BF16 projection parity.
#
# The production H3 block switches from the shared full-F32-accumulator
# `linear` to `minimax_h3_bf16_linear_chunked` above 16,384 sequence rows.
# This GPU gate straddles that boundary and compares the chunked result to the
# ordinary full-accumulator operator on identical BF16 inputs and weights.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.linear import linear
from serenitymojo.ops.random import randn
from serenitymojo.models.dit.minimax_h3_int8_linear import (
    minimax_h3_bf16_linear_chunked,
)


comptime M = 17001
comptime N = 512
comptime K = 256


def main() raises:
    var ctx = DeviceContext()
    var x = randn([M, K], 20260805, STDtype.BF16, ctx)
    var weight = randn([N, K], 20260806, STDtype.BF16, ctx)
    var reference = linear(x, weight, None, ctx).to_host(ctx)
    var chunked = minimax_h3_bf16_linear_chunked(x, weight, ctx)
    var result = ParityHarness(0.999999).compare(chunked, reference, ctx)
    print("MiniMax-H3 chunked BF16 linear parity:", result)
    if not result.passed:
        raise Error("MiniMax-H3 chunked BF16 linear parity failed")
