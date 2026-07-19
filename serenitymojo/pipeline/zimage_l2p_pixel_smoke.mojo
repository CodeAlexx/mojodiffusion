# Z-Image L2P VAE-less pixel patch smoke.
#
# L2P operates directly in RGB pixel space with 16x16 patches, not through a
# VAE latent. This smoke exercises that model-specific GPU data path at the
# production 1024 profile: [1,3,1024,1024] <-> [1,4096,768].

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.zimage_l2p_contract import (
    ZIMAGE_L2P_DEFAULT_STEPS,
    ZIMAGE_L2P_HEIGHT,
    ZIMAGE_L2P_IMAGE_TOKENS,
    ZIMAGE_L2P_PATCH_SIZE,
    ZIMAGE_L2P_PATCH_VECTOR_DIM,
    ZIMAGE_L2P_PIXEL_CHANNELS,
    ZIMAGE_L2P_WIDTH,
    build_zimage_l2p_sigma_schedule,
    zimage_l2p_default_shift,
)
from serenitymojo.ops.layout import patchify, unpatchify
from serenitymojo.tensor import Tensor


def _make_pixels() -> List[Float32]:
    var n = ZIMAGE_L2P_PIXEL_CHANNELS * ZIMAGE_L2P_HEIGHT * ZIMAGE_L2P_WIDTH
    var vals = List[Float32](capacity=n)
    for i in range(n):
        vals.append(Float32(i % 251) / Float32(125.0) - Float32(1.0))
    return vals^


def _require_shape3(label: String, got: List[Int], a: Int, b: Int, c: Int) raises:
    if len(got) != 3 or got[0] != a or got[1] != b or got[2] != c:
        raise Error(String("shape mismatch for ") + label)


def _require_shape4(
    label: String, got: List[Int], a: Int, b: Int, c: Int, d: Int
) raises:
    if len(got) != 4 or got[0] != a or got[1] != b or got[2] != c or got[3] != d:
        raise Error(String("shape mismatch for ") + label)


def main() raises:
    var ctx = DeviceContext()
    print("=== Z-Image L2P pixel-space smoke ===")
    var shape = List[Int]()
    shape.append(1)
    shape.append(ZIMAGE_L2P_PIXEL_CHANNELS)
    shape.append(ZIMAGE_L2P_HEIGHT)
    shape.append(ZIMAGE_L2P_WIDTH)
    var x = Tensor.from_host(_make_pixels(), shape^, STDtype.BF16, ctx)
    var packed = patchify(x, ZIMAGE_L2P_PATCH_SIZE, ctx)
    var restored = unpatchify(
        packed,
        ZIMAGE_L2P_PIXEL_CHANNELS,
        ZIMAGE_L2P_HEIGHT,
        ZIMAGE_L2P_WIDTH,
        ZIMAGE_L2P_PATCH_SIZE,
        ctx,
    )

    _require_shape3(
        String("packed"),
        packed.shape(),
        1,
        ZIMAGE_L2P_IMAGE_TOKENS,
        ZIMAGE_L2P_PATCH_VECTOR_DIM,
    )
    _require_shape4(
        String("restored"),
        restored.shape(),
        1,
        ZIMAGE_L2P_PIXEL_CHANNELS,
        ZIMAGE_L2P_HEIGHT,
        ZIMAGE_L2P_WIDTH,
    )

    var before = x.to_host(ctx)
    var after = restored.to_host(ctx)
    var max_diff = Float32(0.0)
    for i in range(len(before)):
        var d = before[i] - after[i]
        if d < 0.0:
            d = -d
        if d > max_diff:
            max_diff = d
    if max_diff != 0.0:
        raise Error("Z-Image L2P patch roundtrip must be exact after BF16 storage")

    var sched = build_zimage_l2p_sigma_schedule(
        ZIMAGE_L2P_DEFAULT_STEPS, zimage_l2p_default_shift()
    )
    print(
        "  packed shape:",
        packed.shape()[0],
        packed.shape()[1],
        packed.shape()[2],
    )
    print("  roundtrip max_diff:", max_diff)
    print("  schedule first/mid/end:", sched[0], sched[15], sched[30])
    print("Z-Image L2P pixel-space smoke PASS")
