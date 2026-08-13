# Compile-only reachability gate for the SCAIL-2 streamed transformer adapter.
# The dynamic false arm specializes the exact 20/28-channel embedding path,
# 257-token image branch, SCAIL RoPE, one-block stream and video-only head.

from std.collections import List
from max.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.scail2.scail2_streamed_dit import Scail2StreamedDiT
from serenitymojo.tensor import Tensor


def _zeros(count: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(count):
        out.append(0.0)
    return out^


def main() raises:
    var args = argv()
    if len(args) == 999:
        var ctx = DeviceContext()
        var model = Scail2StreamedDiT.open(String(args[998]), ctx)
        var video = Tensor.from_host(
            _zeros(20 * 1 * 4 * 4), [20, 1, 4, 4], STDtype.BF16, ctx
        )
        var primary_ref = Tensor.from_host(
            _zeros(20 * 1 * 4 * 4), [20, 1, 4, 4], STDtype.BF16, ctx
        )
        var pose = Tensor.from_host(
            _zeros(20 * 1 * 2 * 2), [20, 1, 2, 2], STDtype.BF16, ctx
        )
        var main_masks = Tensor.from_host(
            _zeros(28 * 2 * 4 * 4), [28, 2, 4, 4], STDtype.BF16, ctx
        )
        var driving_masks = Tensor.from_host(
            _zeros(28 * 1 * 2 * 2), [28, 1, 2, 2], STDtype.BF16, ctx
        )
        var cond = Tensor.from_host(
            _zeros(2 * 4096), [2, 4096], STDtype.BF16, ctx
        )
        var uncond = Tensor.from_host(
            _zeros(2 * 4096), [2, 4096], STDtype.BF16, ctx
        )
        var clip = Tensor.from_host(
            _zeros(257 * 1280), [1, 257, 1280], STDtype.F16, ctx
        )
        # FT=1, patched GH=2/GW=2: ref+video=8, pose=1, total S=9.
        _ = model.forward_cfg_pair[1, 2, 2, 9, 512, 2, 257, 40, 128](
            video, primary_ref, pose, main_masks, driving_masks, 900.0,
            cond, uncond, clip, 512, 512, False, ctx,
        )
        var additional_ref = Tensor.from_host(
            _zeros(20 * 1 * 4 * 4), [20, 1, 4, 4], STDtype.BF16, ctx
        )
        var additional_mask = Tensor.from_host(
            _zeros(28 * 1 * 4 * 4), [28, 1, 4, 4], STDtype.BF16, ctx
        )
        # One additional ref contributes four more tokens: total S=13.
        _ = model.forward_cfg_pair_with_additional[
            1, 2, 2, 1, 13, 512, 2, 257, 40, 128,
        ](
            video, primary_ref, pose, main_masks, driving_masks,
            additional_ref, additional_mask, 900.0, cond, uncond, clip,
            512, 512, True, ctx,
        )
    print("GATE PASS SCAIL-2 streamed adapter reachable")
