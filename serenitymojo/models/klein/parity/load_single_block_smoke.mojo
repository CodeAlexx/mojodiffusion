# load_single_block_smoke.mojo — verify load_single_block_weights against REAL
# Klein-9B weights + finite single_block_forward at real dims (H=32, Dh=128,
# D=4096, F=12288). Mirrors load_double_block_smoke.mojo (the block math is
# already parity-proven in single_block_parity.mojo; this proves real weights
# load with compatible shapes and drive a finite forward).
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/models/klein/parity/load_single_block_smoke.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.single_block import (
    SingleModVecs, single_block_forward
)
from serenitymojo.models.klein.weights import load_single_block_weights


comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _all_finite(h: List[Float32]) -> Bool:
    for i in range(len(h)):
        var v = h[i]
        if v != v:
            return False
        var a = v if v >= 0.0 else -v
        if a > Float32(1.0e30):
            return False
    return True


def main() raises:
    var ctx = DeviceContext()
    comptime H = 32
    comptime Dh = 128
    comptime S = 6
    var D = 4096
    var F = 12288
    var eps = Float32(1.0e-6)

    print("=== load REAL Klein-9B single-block 0 + finite forward ===")
    var st = SafeTensors.open(KLEIN9B_PATH)
    var w = load_single_block_weights(st, 0, ctx)
    print("  w1 len =", len(w.w1), " w2 len =", len(w.w2),
          " q_norm len =", len(w.q_norm))
    # w1 = [3D+2F, D]; w2 = [D, D+F]
    if len(w.w1) != (3 * D + 2 * F) * D:
        raise Error("w1 unexpected size")
    if len(w.w2) != D * (D + F):
        raise Error("w2 unexpected size")

    var x = _fill(S * D, 100, 1.0)
    var mv = SingleModVecs(_fill(D, 1, 0.1), _fill(D, 2, 0.1), _fill(D, 3, 0.1))
    var cos_h = _fill(S * H * (Dh // 2), 500, 1.0)
    var sin_h = _fill(S * H * (Dh // 2), 600, 1.0)
    var cos = Tensor.from_host(cos_h, [S * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [S * H, Dh // 2], STDtype.F32, ctx)

    var fwd = single_block_forward[H, Dh, S](x, w, mv, cos, sin, D, F, eps, ctx)
    print("  out len =", len(fwd.out))
    if len(fwd.out) != S * D:
        raise Error("forward output shape wrong")
    if not _all_finite(fwd.out):
        raise Error("forward output not finite (NaN/inf) on real weights")
    print("PASS: real Klein-9B single-block-0 weights load + drive a FINITE forward")
