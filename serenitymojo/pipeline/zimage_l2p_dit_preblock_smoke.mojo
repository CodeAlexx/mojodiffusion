# Z-Image L2P DiT pre-block real-weight smoke.
#
# Gates the VAE-less L2P DiT path before transformer blocks:
# pixel patchify16 -> all_x_embedder, sigma -> timestep MLP, and
# cap_feats -> RMSNorm+Linear caption embedder.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.zimage_l2p_contract import (
    ZIMAGE_L2P_CAP_FEAT_DIM,
    ZIMAGE_L2P_HIDDEN,
    ZIMAGE_L2P_PATCH_VECTOR_DIM,
    validate_zimage_l2p_conditioning_header,
    zimage_l2p_default_checkpoint_path,
    zimage_l2p_default_conditioning_path,
)
from serenitymojo.models.dit.zimage_l2p_dit import (
    ZImageL2PDiTPreBlockGate,
    load_zimage_l2p_default_conditioning_bf16,
)
from serenitymojo.tensor import Tensor


comptime SMOKE_H = 32
comptime SMOKE_W = 32
comptime CAP = 32
comptime UNCOND_CAP = 8


def _shape2(a: Int, b: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    return sh^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    sh.append(c)
    return sh^


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
        vals.append((Float32(i % 251) - 125.0) * scale)
    return vals^


def _require_shape2(label: String, got: List[Int], a: Int, b: Int) raises:
    if len(got) != 2 or got[0] != a or got[1] != b:
        raise Error(String("Z-Image L2P shape mismatch for ") + label)


def _require_shape3(label: String, got: List[Int], a: Int, b: Int, c: Int) raises:
    if len(got) != 3 or got[0] != a or got[1] != b or got[2] != c:
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


def _require_bf16(label: String, t: Tensor) raises:
    if t.dtype() != STDtype.BF16:
        raise Error(String("Z-Image L2P expected BF16 tensor: ") + label)


def main() raises:
    var ctx = DeviceContext()
    print("=== Z-Image L2P DiT pre-block real-weight smoke ===")
    var gate = ZImageL2PDiTPreBlockGate.load_default(ctx)

    var pixels = Tensor.from_host(
        _make_values(1 * 3 * SMOKE_H * SMOKE_W, 0.01),
        _shape4(1, 3, SMOKE_H, SMOKE_W),
        STDtype.BF16,
        ctx,
    )
    var patches = gate.patchify16_pixel[SMOKE_H, SMOKE_W](pixels, ctx)
    _require_shape3(
        String("patches"),
        patches.shape(),
        1,
        (SMOKE_H // 16) * (SMOKE_W // 16),
        ZIMAGE_L2P_PATCH_VECTOR_DIM,
    )
    var pixel_emb = gate.pixel_embed[SMOKE_H, SMOKE_W](pixels, ctx)
    _require_shape3(
        String("pixel_embed"),
        pixel_emb.shape(),
        1,
        (SMOKE_H // 16) * (SMOKE_W // 16),
        ZIMAGE_L2P_HIDDEN,
    )

    var temb = gate.timestep_embed(1.0, ctx)
    _require_shape2(String("timestep_embed"), temb.shape(), 1, 256)

    var cap = Tensor.from_host(
        _make_values(1 * CAP * ZIMAGE_L2P_CAP_FEAT_DIM, 0.001),
        _shape3(1, CAP, ZIMAGE_L2P_CAP_FEAT_DIM),
        STDtype.BF16,
        ctx,
    )
    var cap_emb = gate.caption_embed[CAP](cap, ctx)
    _require_shape3(String("caption_embed"), cap_emb.shape(), 1, CAP, ZIMAGE_L2P_HIDDEN)

    var conditioning_path = zimage_l2p_default_conditioning_path()
    var cond_contract = validate_zimage_l2p_conditioning_header(
        conditioning_path, True
    )
    if cond_contract.cap_tokens != CAP or cond_contract.uncond_tokens != UNCOND_CAP:
        raise Error("Z-Image L2P default conditioning sidecar token shape changed")
    var real_cap = load_zimage_l2p_default_conditioning_bf16(
        String("cap_feats"), ctx
    )
    var real_uncond = load_zimage_l2p_default_conditioning_bf16(
        String("cap_feats_uncond"), ctx
    )
    _require_bf16(String("cap_feats"), real_cap)
    _require_bf16(String("cap_feats_uncond"), real_uncond)
    _require_shape3(
        String("cap_feats"),
        real_cap.shape(),
        1,
        CAP,
        ZIMAGE_L2P_CAP_FEAT_DIM,
    )
    _require_shape3(
        String("cap_feats_uncond"),
        real_uncond.shape(),
        1,
        UNCOND_CAP,
        ZIMAGE_L2P_CAP_FEAT_DIM,
    )
    var real_cap_emb = gate.caption_embed[CAP](real_cap, ctx)
    var real_uncond_emb = gate.caption_embed[UNCOND_CAP](real_uncond, ctx)
    _require_shape3(
        String("real_caption_embed"),
        real_cap_emb.shape(),
        1,
        CAP,
        ZIMAGE_L2P_HIDDEN,
    )
    _require_shape3(
        String("real_uncond_caption_embed"),
        real_uncond_emb.shape(),
        1,
        UNCOND_CAP,
        ZIMAGE_L2P_HIDDEN,
    )

    var patch_max = _stats_nonzero(String("patches"), patches, ctx)
    var pixel_max = _stats_nonzero(String("pixel_embed"), pixel_emb, ctx)
    var temb_max = _stats_nonzero(String("timestep_embed"), temb, ctx)
    var cap_max = _stats_nonzero(String("caption_embed"), cap_emb, ctx)
    var real_cap_max = _stats_nonzero(String("real_caption_embed"), real_cap_emb, ctx)
    var real_uncond_max = _stats_nonzero(
        String("real_uncond_caption_embed"), real_uncond_emb, ctx
    )

    print("  checkpoint:", zimage_l2p_default_checkpoint_path())
    print("  conditioning:", conditioning_path)
    print("  patches shape/max_abs:", patches.shape()[0], patches.shape()[1], patches.shape()[2], patch_max)
    print("  pixel_embed shape/max_abs:", pixel_emb.shape()[0], pixel_emb.shape()[1], pixel_emb.shape()[2], pixel_max)
    print("  timestep_embed shape/max_abs:", temb.shape()[0], temb.shape()[1], temb_max)
    print("  caption_embed shape/max_abs:", cap_emb.shape()[0], cap_emb.shape()[1], cap_emb.shape()[2], cap_max)
    print("  real cap_embed shape/max_abs:", real_cap_emb.shape()[0], real_cap_emb.shape()[1], real_cap_emb.shape()[2], real_cap_max)
    print("  real uncond_embed shape/max_abs:", real_uncond_emb.shape()[0], real_uncond_emb.shape()[1], real_uncond_emb.shape()[2], real_uncond_max)
    print("Z-Image L2P DiT pre-block real-weight smoke PASS")
