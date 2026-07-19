# load_double_block_smoke.mojo — G1 verification: load REAL Klein-9B double-block
# weights from safetensors and drive the verified double_block_forward at REAL
# dims (H=32, Dh=128, D=4096, F=12288). Proves the loader produces tensor shapes
# compatible with the block and a finite forward (the block math itself is already
# parity-proven vs torch in double_block_parity.mojo). Small N keeps it light.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/models/klein/parity/load_double_block_smoke.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.double_block import (
    ModVecs, double_block_forward
)
from serenitymojo.models.klein.weights import load_double_block_weights


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
        if v != v:            # NaN
            return False
        var a = v if v >= 0.0 else -v
        if a > Float32(1.0e30):   # inf / overflow
            return False
    return True


def _mod(d: Int, seed: UInt64) -> ModVecs:
    return ModVecs(
        _fill(d, seed + 1, 0.1), _fill(d, seed + 2, 0.1), _fill(d, seed + 3, 0.1),
        _fill(d, seed + 4, 0.1), _fill(d, seed + 5, 0.1), _fill(d, seed + 6, 0.1),
    )


def main() raises:
    var ctx = DeviceContext()
    comptime H = 32
    comptime Dh = 128
    comptime N_IMG = 4
    comptime N_TXT = 2
    comptime S = N_IMG + N_TXT
    var D = 4096
    var F = 12288
    var eps = Float32(1.0e-6)

    print("=== G1: load REAL Klein-9B double-block 0 + finite forward ===")
    print("  path:", KLEIN9B_PATH)
    var st = SafeTensors.open(KLEIN9B_PATH)
    var w = load_double_block_weights(st, 0, ctx)
    print("  loaded block 0:  img.wqkv len =", len(w.img.wqkv),
          " img.wd len =", len(w.img.wd), " txt.wqkv len =", len(w.txt.wqkv))
    # shape sanity: wqkv = [3*D, D]; wd = [D, F]
    if len(w.img.wqkv) != 3 * D * D:
        raise Error("img.wqkv unexpected size")
    if len(w.img.wd) != D * F:
        raise Error("img.wd unexpected size")

    var img = _fill(N_IMG * D, 100, 1.0)
    var txt = _fill(N_TXT * D, 200, 1.0)
    var img_mod = _mod(D, 300)
    var txt_mod = _mod(D, 400)
    var cos_h = _fill(S * H * (Dh // 2), 500, 1.0)
    var sin_h = _fill(S * H * (Dh // 2), 600, 1.0)
    var cos = Tensor.from_host(cos_h, [S * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [S * H, Dh // 2], STDtype.F32, ctx)

    var fwd = double_block_forward[H, Dh, N_IMG, N_TXT, S](
        img, txt, w, img_mod, txt_mod, cos, sin, D, F, eps, ctx
    )

    print("  img_out len =", len(fwd.img_out), " txt_out len =", len(fwd.txt_out))
    if len(fwd.img_out) != N_IMG * D or len(fwd.txt_out) != N_TXT * D:
        raise Error("forward output shape wrong")
    if not _all_finite(fwd.img_out) or not _all_finite(fwd.txt_out):
        raise Error("forward output not finite (NaN/inf) on real weights")
    print("PASS: real Klein-9B block-0 weights load + drive a FINITE forward at real dims")
