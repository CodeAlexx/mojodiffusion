# Z-Image L2P local decoder real-weight smoke.
#
# Runs checkpoint-backed MicroDiffusionModel math:
#   1. noisy_input NCHW -> enc1 conv3x3 + SiLU + maxpool2x2
#   2. cat([p4, feat_map]) -> bottleneck conv1x1 + SiLU
#   3. full tiny 32x32 encoder -> bottleneck -> decoder -> out_conv path
#
# This is intentionally a bounded gate. The native 1024 full decoder is too expensive
# for a smoke with the current naive NHWC conv backend.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.zimage_l2p_contract import (
    ZIMAGE_L2P_HIDDEN,
    ZIMAGE_L2P_LD_C4,
    zimage_l2p_default_checkpoint_path,
)
from serenitymojo.models.dit.zimage_l2p_local_decoder import ZImageL2PLocalDecoderGate
from serenitymojo.tensor import Tensor


comptime SMOKE_H = 32
comptime SMOKE_W = 32
comptime BOT_H = 4
comptime BOT_W = 4


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    sh.append(c)
    sh.append(d)
    return sh^


def _make_values(n: Int, scale: Float32) -> List[Float32]:
    var vals = List[Float32](capacity=n)
    for i in range(n):
        vals.append((Float32(i % 257) - 128.0) * scale)
    return vals^


def _require_shape4(
    label: String, got: List[Int], a: Int, b: Int, c: Int, d: Int
) raises:
    if len(got) != 4 or got[0] != a or got[1] != b or got[2] != c or got[3] != d:
        raise Error(String("Z-Image L2P shape mismatch for ") + label)


def _stats_nonzero(label: String, t: Tensor, ctx: DeviceContext) raises -> Float32:
    var h = t.to_host(ctx)
    if len(h) == 0:
        raise Error(String("Z-Image L2P empty tensor: ") + label)
    var max_abs = Float32(0.0)
    for i in range(len(h)):
        var v = h[i]
        if v < 0.0:
            v = -v
        if v > max_abs:
            max_abs = v
    if max_abs == 0.0:
        raise Error(String("Z-Image L2P all-zero tensor: ") + label)
    return max_abs


def main() raises:
    var ctx = DeviceContext()
    print("=== Z-Image L2P local-decoder real-weight smoke ===")
    var gate = ZImageL2PLocalDecoderGate.load_default(ctx)

    var noisy = Tensor.from_host(
        _make_values(1 * 3 * SMOKE_H * SMOKE_W, 0.01),
        _shape4(1, 3, SMOKE_H, SMOKE_W),
        STDtype.BF16,
        ctx,
    )
    var pooled = gate.enc1_pool[SMOKE_H, SMOKE_W](noisy, ctx)
    _require_shape4(String("enc1_pool"), pooled.shape(), 1, SMOKE_H // 2, SMOKE_W // 2, 64)
    var pooled_max = _stats_nonzero(String("enc1_pool"), pooled, ctx)

    var p4 = Tensor.from_host(
        _make_values(1 * BOT_H * BOT_W * ZIMAGE_L2P_LD_C4, 0.002),
        _shape4(1, BOT_H, BOT_W, ZIMAGE_L2P_LD_C4),
        STDtype.BF16,
        ctx,
    )
    var feat = Tensor.from_host(
        _make_values(1 * ZIMAGE_L2P_HIDDEN * BOT_H * BOT_W, 0.001),
        _shape4(1, ZIMAGE_L2P_HIDDEN, BOT_H, BOT_W),
        STDtype.BF16,
        ctx,
    )
    var bottleneck = gate.bottleneck[BOT_H, BOT_W](p4, feat, ctx)
    _require_shape4(
        String("bottleneck"),
        bottleneck.shape(),
        1,
        BOT_H,
        BOT_W,
        ZIMAGE_L2P_LD_C4,
    )
    var bottleneck_max = _stats_nonzero(String("bottleneck"), bottleneck, ctx)

    var feat_small = Tensor.from_host(
        _make_values(1 * ZIMAGE_L2P_HIDDEN * (SMOKE_H // 16) * (SMOKE_W // 16), 0.001),
        _shape4(1, ZIMAGE_L2P_HIDDEN, SMOKE_H // 16, SMOKE_W // 16),
        STDtype.BF16,
        ctx,
    )
    var full_out = gate.full_tiny_forward[SMOKE_H, SMOKE_W](noisy, feat_small, ctx)
    _require_shape4(
        String("full_tiny_forward"),
        full_out.shape(),
        1,
        3,
        SMOKE_H,
        SMOKE_W,
    )
    var full_max = _stats_nonzero(String("full_tiny_forward"), full_out, ctx)

    print("  checkpoint:", zimage_l2p_default_checkpoint_path())
    print("  enc1_pool shape/max_abs:", pooled.shape()[0], pooled.shape()[1], pooled.shape()[2], pooled.shape()[3], pooled_max)
    print("  bottleneck shape/max_abs:", bottleneck.shape()[0], bottleneck.shape()[1], bottleneck.shape()[2], bottleneck.shape()[3], bottleneck_max)
    print("  full_tiny output shape/max_abs:", full_out.shape()[0], full_out.shape()[1], full_out.shape()[2], full_out.shape()[3], full_max)
    print("Z-Image L2P local-decoder real-weight smoke PASS")
