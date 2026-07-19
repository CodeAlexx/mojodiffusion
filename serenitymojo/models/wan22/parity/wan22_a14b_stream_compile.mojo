# Compile-only reachability gate for the Bernini-R A14B streamed expert adapter.

from std.collections import List
from std.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.wan22.wan22_a14b_streamed_dit import (
    Wan22A14BStreamedDiT,
)
from serenitymojo.tensor import Tensor


def main() raises:
    # Dynamic false arm: specializes and type-checks the complete CFG-paired
    # forward without requiring a real 14B cache for this compile gate.
    var args = argv()
    if len(args) == 999:
        var ctx = DeviceContext()
        var model = Wan22A14BStreamedDiT.open(String(args[998]), ctx)
        var latent_values = List[Float32]()
        for _ in range(16 * 1 * 2 * 2):
            latent_values.append(0.0)
        var text_values = List[Float32]()
        for _ in range(2 * 4096):
            text_values.append(0.0)
        var latent = Tensor.from_host(
            latent_values^, [16, 1, 2, 2], STDtype.BF16, ctx
        )
        var cond = Tensor.from_host(text_values.copy(), [2, 4096], STDtype.BF16, ctx)
        var uncond = Tensor.from_host(text_values^, [2, 4096], STDtype.BF16, ctx)
        _ = model.forward_cfg_pair[1, 1, 1, 1, 2, 2, 40, 128](
            latent, 900.0, cond, uncond, 2, 2, ctx
        )
    print("GATE PASS Wan A14B streamed adapter reachable")
