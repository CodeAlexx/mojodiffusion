# models/pid/pit_block_smoke.mojo — GPU smoke + parity for pit_block.mojo.
#
# Unit-gates PiTBlock.forward against a PyTorch reference (random seeded weights,
# tiny input) dumped via system python3 into parity/pit_block_ref_data.mojo.
# F32 throughout. no-CP, mask=None path.
#
# Gate (HARD RULE: real numeric on GPU, not compile-only):
#   pit_block_forward(x, s_cond, weights) == PiTBlock.forward   (cos >= 0.999)
#
# Run: pixi run mojo run -I . serenitymojo/models/pid/pit_block_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.pid.pit_block import (
    PiTBlockWeights, pit_block_forward, gelu_erf,
)
from serenitymojo.models.pid.parity.pit_block_ref_data import (
    PIXEL_DIM, CONTEXT_DIM, ATTN_DIM, ATTN_HEADS, HEAD_DIM, PATCH_SIZE,
    P2, P2D, MLP_HIDDEN, IMG_H, IMG_W, HS, WS, L, B, BL, ROPE_REF,
    pit_x, pit_scond,
    pit_compress_w, pit_compress_b, pit_expand_w, pit_expand_b,
    pit_qkv_w, pit_proj_w, pit_proj_b, pit_qnorm_w, pit_knorm_w,
    pit_norm1_w, pit_norm2_w, pit_fc1_w, pit_fc1_b, pit_fc2_w, pit_fc2_b,
    pit_adaln_w, pit_adaln_b, pit_y_ref,
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

    print("=== PiTBlock smoke (pit_block.mojo) — F32 vs PyTorch ref (seed 4242) ===")

    # ── build weight tensors ─────────────────────────────────────────────────
    var compress_w = Tensor.from_host(pit_compress_w(), [ATTN_DIM, P2D], F32, ctx)
    var compress_b = Tensor.from_host(pit_compress_b(), [ATTN_DIM], F32, ctx)
    var expand_w = Tensor.from_host(pit_expand_w(), [P2D, ATTN_DIM], F32, ctx)
    var expand_b = Tensor.from_host(pit_expand_b(), [P2D], F32, ctx)
    var qkv_w = Tensor.from_host(pit_qkv_w(), [3 * ATTN_DIM, ATTN_DIM], F32, ctx)
    var proj_w = Tensor.from_host(pit_proj_w(), [ATTN_DIM, ATTN_DIM], F32, ctx)
    var proj_b = Tensor.from_host(pit_proj_b(), [ATTN_DIM], F32, ctx)
    var qnorm_w = Tensor.from_host(pit_qnorm_w(), [HEAD_DIM], F32, ctx)
    var knorm_w = Tensor.from_host(pit_knorm_w(), [HEAD_DIM], F32, ctx)
    var norm1_w = Tensor.from_host(pit_norm1_w(), [PIXEL_DIM], F32, ctx)
    var norm2_w = Tensor.from_host(pit_norm2_w(), [PIXEL_DIM], F32, ctx)
    var fc1_w = Tensor.from_host(pit_fc1_w(), [MLP_HIDDEN, PIXEL_DIM], F32, ctx)
    var fc1_b = Tensor.from_host(pit_fc1_b(), [MLP_HIDDEN], F32, ctx)
    var fc2_w = Tensor.from_host(pit_fc2_w(), [PIXEL_DIM, MLP_HIDDEN], F32, ctx)
    var fc2_b = Tensor.from_host(pit_fc2_b(), [PIXEL_DIM], F32, ctx)
    var adaln_w = Tensor.from_host(
        pit_adaln_w(), [6 * PIXEL_DIM * P2, CONTEXT_DIM], F32, ctx
    )
    var adaln_b = Tensor.from_host(pit_adaln_b(), [6 * PIXEL_DIM * P2], F32, ctx)

    var w = PiTBlockWeights(
        compress_w^, compress_b^, expand_w^, expand_b^,
        qkv_w^, proj_w^, proj_b^, qnorm_w^, knorm_w^,
        norm1_w^, norm2_w^, fc1_w^, fc1_b^, fc2_w^, fc2_b^,
        adaln_w^, adaln_b^,
    )

    # ── inputs ───────────────────────────────────────────────────────────────
    var x = Tensor.from_host(pit_x(), [BL, P2, PIXEL_DIM], F32, ctx)
    var s_cond = Tensor.from_host(pit_scond(), [BL, CONTEXT_DIM], F32, ctx)

    # ── forward ──────────────────────────────────────────────────────────────
    var y = pit_block_forward[B, L, ATTN_HEADS, HEAD_DIM](
        x, s_cond, w, PIXEL_DIM, CONTEXT_DIM, ATTN_DIM, P2,
        IMG_H, IMG_W, PATCH_SIZE, ROPE_REF, ctx,
    )

    # ── gate ─────────────────────────────────────────────────────────────────
    var r = h.compare(y, pit_y_ref(), ctx)
    all_pass = all_pass and _report("PiTBlock.forward            ", r)

    print("============================================================")
    if all_pass:
        print("ALL GATES PASS")
    else:
        print("SOME GATES FAILED")
        raise Error("pit_block smoke: gate failure")
