# models/dit/parity/krea2_txtfusion_parity_probe.mojo — PARITY GATE for krea2
# chunk 6 (TextFusionTransformer: 2 layerwise blocks over the 12-layer axis ->
# Linear(12->1) projector -> 2 refiner blocks over the Lt tokens) vs a real
# ai-toolkit torch oracle. TWO gated cases, both FAIL-LOUD (non-zero exit on
# cos < 0.999) so a stale oracle cannot false-green this.
#
#   CASE A (b==1 INFERENCE): keep is all-ones (no text padding at b==1 — see
#     gen_krea2_txtfusion.py). _mask(all-ones) is a uniform +1 bias => softmax-
#     invariant => no-op. Run the refiner with mask=None; compare vs out_nomask.
#   CASE B (PADDED, masked path): build a bf16 0/1 mask from keep_pad and run the
#     refiner through the masked sdpa path; compare vs out_masked. This exercises
#     the masked branch that chunk 7's pad-to-256 mask depends on.
#
# Oracle: krea2_txtfusion_oracle.safetensors (weights BF16, scales/keep/out F32):
#   context [1,24,12,2560], keep_pad [1,24] F32, projector_w [1,12],
#   out_nomask [1,24,2560], out_masked [1,24,2560], and per block (lw0/lw1/rf0/rf1):
#   prenorm/postnorm/qnorm/knorm (F32) + wq/wk/wv/gate/wo + mlp_gate/up/down (bf16).
#
# Run: cd /home/alex/mojodiffusion && \
#   pixi run mojo run -I . serenitymojo/models/dit/parity/krea2_txtfusion_parity_probe.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.dit.krea2_dit import (
    Krea2TextFusionWeights,
    build_krea2_text_mask,
    krea2_text_fusion,
)


comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_txtfusion_oracle.safetensors"


def _load_block(
    fx: ShardedSafeTensors, prefix: String, ctx: DeviceContext
) raises -> Krea2TextFusionWeights:
    """Load one TextFusionBlock's weights into a bundle. Scales F32; projections bf16."""
    return Krea2TextFusionWeights(
        ArcPointer(Tensor.from_view_as_f32(fx.tensor_view(prefix + "_prenorm"), ctx)),
        ArcPointer(Tensor.from_view_as_f32(fx.tensor_view(prefix + "_postnorm"), ctx)),
        ArcPointer(Tensor.from_view(fx.tensor_view(prefix + "_wq"), ctx)),
        ArcPointer(Tensor.from_view(fx.tensor_view(prefix + "_wk"), ctx)),
        ArcPointer(Tensor.from_view(fx.tensor_view(prefix + "_wv"), ctx)),
        ArcPointer(Tensor.from_view(fx.tensor_view(prefix + "_gate"), ctx)),
        ArcPointer(Tensor.from_view(fx.tensor_view(prefix + "_wo"), ctx)),
        ArcPointer(Tensor.from_view_as_f32(fx.tensor_view(prefix + "_qnorm"), ctx)),
        ArcPointer(Tensor.from_view_as_f32(fx.tensor_view(prefix + "_knorm"), ctx)),
        ArcPointer(Tensor.from_view(fx.tensor_view(prefix + "_mlp_gate"), ctx)),
        ArcPointer(Tensor.from_view(fx.tensor_view(prefix + "_mlp_up"), ctx)),
        ArcPointer(Tensor.from_view(fx.tensor_view(prefix + "_mlp_down"), ctx)),
    )


def _gate(name: String, res: ParityResult) raises:
    """FAIL-LOUD: print the result and raise (non-zero exit) on cos < 0.999."""
    print(name, "parity:", res)
    if res.cos < 0.999:
        raise Error(name + " FAILED: cos " + String(res.cos) + " < 0.999")


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(ORACLE)

    comptime LT = 24
    comptime NLAYERS = 12
    comptime HEADS = 20
    comptime HEADDIM = 128

    var context = Tensor.from_view(fx.tensor_view("context"), ctx)   # bf16 [1,24,12,2560]
    var projector_w = Tensor.from_view(fx.tensor_view("projector_w"), ctx)  # bf16 [1,12]

    var lw0 = _load_block(fx, "lw0", ctx)
    var lw1 = _load_block(fx, "lw1", ctx)
    var rf0 = _load_block(fx, "rf0", ctx)
    var rf1 = _load_block(fx, "rf1", ctx)

    # ── CASE A: b==1 inference (all-ones keep => no-op mask => refiner mask=None) ─
    var out_nomask_ref = Tensor.from_view_as_f32(fx.tensor_view("out_nomask"), ctx).to_host(ctx)
    var out_a = krea2_text_fusion[LT, NLAYERS, HEADS, HEADDIM](
        context, lw0, lw1, projector_w, rf0, rf1,
        Optional[Tensor](None), ctx,
    )
    _gate(
        "CASE-A refiner no-mask (b==1 inference)",
        ParityHarness(0.999).compare(out_a, out_nomask_ref, ctx),
    )

    # ── CASE B: padded masked path (chunk-7 mask coverage) ───────────────────
    # Build the bf16 0/1 additive mask from keep_pad; refiner runs the masked sdpa
    # path. q/k/v are bf16, so the mask must be bf16 (sdpa enforces dtype match;
    # 0.0/1.0 are bf16-exact, the math path adds it in F32).
    var keep_pad = Tensor.from_view_as_f32(fx.tensor_view("keep_pad"), ctx)  # [1,24]
    var keep_shape = List[Int]()
    keep_shape.append(LT)
    var keep_flat = reshape(keep_pad, keep_shape^, ctx)                      # [24]
    var mask_bf16 = build_krea2_text_mask(keep_flat, HEADS, LT, ctx, STDtype.BF16)  # [1,20,24,24] bf16
    var out_masked_ref = Tensor.from_view_as_f32(fx.tensor_view("out_masked"), ctx).to_host(ctx)
    var out_b = krea2_text_fusion[LT, NLAYERS, HEADS, HEADDIM](
        context, lw0, lw1, projector_w, rf0, rf1,
        Optional[Tensor](mask_bf16^), ctx,
    )
    _gate(
        "CASE-B refiner masked (padded)",
        ParityHarness(0.999).compare(out_b, out_masked_ref, ctx),
    )

    print("krea2_text_fusion: BOTH CASES PASS")
