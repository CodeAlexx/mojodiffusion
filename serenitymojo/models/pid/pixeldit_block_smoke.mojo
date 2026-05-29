# models/pid/pixeldit_block_smoke.mojo — GPU smoke + parity for the PiD
# PixelDiT MMDiT joint-attention block (models/pid/pixeldit_block.mojo).
#
# Unit-gates one MMDiTBlockT2I.forward against a PyTorch reference (random
# seeded weights, tiny input) dumped via system python3 into
# parity/pixeldit_block_ref_data.mojo. F32 throughout.
#
# Gates (HARD RULE: real numeric on GPU, not compile-only):
#   1. x_out (image stream)  == MMDiTBlockT2I.forward(...)[0]   (cos>=0.999)
#   2. y_out (text stream)   == MMDiTBlockT2I.forward(...)[1]   (cos>=0.999)
#
# Run: pixi run mojo run -I . serenitymojo/models/pid/pixeldit_block_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.pid.pixeldit_block import (
    MMDiTBlockWeights, mmdit_block_forward,
)
from serenitymojo.models.pid.parity.pixeldit_block_ref_data import (
    HIDDEN, GROUPS, HEAD_DIM, B, NX, NY, ROPE_HALF, FF_HIDDEN,
    inp_x, inp_y, inp_c, rope_cos, rope_sin,
    adaln_img_w, adaln_img_b, adaln_txt_w, adaln_txt_b,
    norm_x1_w, norm_y1_w, norm_x2_w, norm_y2_w,
    qkv_x_w, qkv_y_w, q_norm_x_w, k_norm_x_w, q_norm_y_w, k_norm_y_w,
    proj_x_w, proj_x_b, proj_y_w, proj_y_b,
    mlp_x_w1, mlp_x_w3, mlp_x_w2, mlp_y_w1, mlp_y_w3, mlp_y_w2,
    out_x_ref, out_y_ref,
)


comptime F32 = STDtype.F32


def _report(label: String, r: ParityResult) -> Bool:
    var tag = "PASS" if r.passed else "FAIL"
    print(label, "  cos=", r.cos, "  max_abs=", r.max_abs, "  n=", r.n, "  [", tag, "]")
    return r.passed


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True
    var C = HIDDEN
    var H = GROUPS

    print("=== PiD PixelDiT MMDiTBlockT2I smoke — F32 vs PyTorch ref (seed 5151) ===")
    print("  C=", C, " groups=", H, " head_dim=", HEAD_DIM, " Nx=", NX, " Ny=", NY)

    # ── inputs ────────────────────────────────────────────────────────────────
    var x = Tensor.from_host(inp_x(), [B, NX, C], F32, ctx)
    var y = Tensor.from_host(inp_y(), [B, NY, C], F32, ctx)
    var c = Tensor.from_host(inp_c(), [B, 1, C], F32, ctx)
    var rcos = Tensor.from_host(rope_cos(), [NX, ROPE_HALF], F32, ctx)
    var rsin = Tensor.from_host(rope_sin(), [NX, ROPE_HALF], F32, ctx)

    # ── weights ────────────────────────────────────────────────────────────────
    var w = MMDiTBlockWeights(
        Tensor.from_host(adaln_img_w(), [6 * C, C], F32, ctx),
        Tensor.from_host(adaln_img_b(), [6 * C], F32, ctx),
        Tensor.from_host(adaln_txt_w(), [6 * C, C], F32, ctx),
        Tensor.from_host(adaln_txt_b(), [6 * C], F32, ctx),
        Tensor.from_host(norm_x1_w(), [C], F32, ctx),
        Tensor.from_host(norm_y1_w(), [C], F32, ctx),
        Tensor.from_host(norm_x2_w(), [C], F32, ctx),
        Tensor.from_host(norm_y2_w(), [C], F32, ctx),
        Tensor.from_host(qkv_x_w(), [3 * C, C], F32, ctx),
        Tensor.from_host(qkv_y_w(), [3 * C, C], F32, ctx),
        Tensor.from_host(q_norm_x_w(), [HEAD_DIM], F32, ctx),
        Tensor.from_host(k_norm_x_w(), [HEAD_DIM], F32, ctx),
        Tensor.from_host(q_norm_y_w(), [HEAD_DIM], F32, ctx),
        Tensor.from_host(k_norm_y_w(), [HEAD_DIM], F32, ctx),
        Tensor.from_host(proj_x_w(), [C, C], F32, ctx),
        Tensor.from_host(proj_x_b(), [C], F32, ctx),
        Tensor.from_host(proj_y_w(), [C, C], F32, ctx),
        Tensor.from_host(proj_y_b(), [C], F32, ctx),
        Tensor.from_host(mlp_x_w1(), [FF_HIDDEN, C], F32, ctx),
        Tensor.from_host(mlp_x_w3(), [FF_HIDDEN, C], F32, ctx),
        Tensor.from_host(mlp_x_w2(), [C, FF_HIDDEN], F32, ctx),
        Tensor.from_host(mlp_y_w1(), [FF_HIDDEN, C], F32, ctx),
        Tensor.from_host(mlp_y_w3(), [FF_HIDDEN, C], F32, ctx),
        Tensor.from_host(mlp_y_w2(), [C, FF_HIDDEN], F32, ctx),
    )

    # ── forward ────────────────────────────────────────────────────────────────
    var outs = mmdit_block_forward[GROUPS, HEAD_DIM, NX, NY](
        x, y, c, rcos, rsin, w, C, ctx
    )

    var r1 = h.compare(outs.x, out_x_ref(), ctx)
    all_pass = all_pass and _report("1 x_out (image stream)        ", r1)
    var r2 = h.compare(outs.y, out_y_ref(), ctx)
    all_pass = all_pass and _report("2 y_out (text stream)         ", r2)

    print("============================================================")
    if all_pass:
        print("ALL GATES PASS")
    else:
        print("SOME GATES FAILED")
        raise Error("pixeldit_block smoke: gate failure")
