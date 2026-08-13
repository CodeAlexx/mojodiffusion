# serenitymojo/models/klein/parity/klein_single_block_int8_parity.mojo
#
# PARITY GATE for the int8-W8A8 quantized-resident BASE path of the Klein
# SINGLE-STREAM DiT block (models/klein/single_block.mojo, slice 1).
#
# Builds ONE REAL-DIM Klein single block (H=32, Dh=128, D=4096, F=12288 — the
# Klein-9B config: num_heads/head_dim/inner_dim/mlp_hidden) with random
# non-degenerate inputs, then runs the SAME new forward twice:
#   * int8=None  → the bf16 base matmuls (unchanged `linear`) — the REFERENCE.
#   * int8=payload → the two frozen base matmuls run int8 W8A8 (int8_linear_fwd).
# LoRA is absent in both (base-only gate). We compare the two block outputs with
# ParityHarness. int8-vs-bf16 is the int8 W8A8 quant class; the tolerance MATCHES
# krea2's int8 gate (ops/tests/int8_linear_parity.mojo: cos >= 0.99). We ALSO
# report against the tighter cos >= 0.997 bar the task expects; the HUMAN gates
# on the printed cos + max_abs — this file only reports.
#
# Run (NO oracle needed — self-contained; ref is the bf16 base of the same block):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/klein/parity/klein_single_block_int8_parity.mojo

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt, log as flog, cos as fcos, pi
from std.memory import ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecs, SingleModVecsDevice,
    single_modvecs_to_device,
    SingleBlockLoraDevice,
    SingleBlockInt8, quantize_single_block_int8,
    single_block_int8_base_forward_device_resident,
)

comptime TArc = ArcPointer[Tensor]

# REAL Klein-9B single-block dims (configs/klein9b.json).
comptime H = 32
comptime Dh = 128
comptime D = H * Dh          # 4096  == inner_dim
comptime F = 12288           # mlp_hidden
comptime S = 128             # sequence length (modest, for gate speed)
comptime EPS = Float32(1e-06)


def _gaussian(n: Int, seed: Int, sd: Float32) -> List[Float32]:
    var out = List[Float32]()
    var st = UInt64(seed * 2654435761 + 12345)
    for _i in range(n):
        st = st * 6364136223846793005 + 1442695040888963407
        var u1 = (Float64(st >> 11) + 1.0) / Float64(1 << 53)
        st = st * 6364136223846793005 + 1442695040888963407
        var u2 = Float64(st >> 11) / Float64(1 << 53)
        var r = sqrt(-2.0 * flog(u1))
        out.append(Float32(r * fcos(2.0 * pi * u2)) * sd)
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("==== klein_single_block_int8_parity (int8 base vs bf16 base) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " F=", F, " S=", S)
    print("w1 [3D+2F, D] = [", 3 * D + 2 * F, ",", D, "]",
          "  w2 [D, D+F] = [", D, ",", D + F, "]")

    # ── non-degenerate random inputs ──
    var xh = _gaussian(S * D, 1, 1.0)
    var w1h = _gaussian((3 * D + 2 * F) * D, 2, 0.02)   # ~1/sqrt(D) init scale
    var w2h = _gaussian(D * (D + F), 3, 0.02)
    var qnh = _gaussian(Dh, 4, 1.0)
    var knh = _gaussian(Dh, 5, 1.0)
    var shifth = _gaussian(D, 6, 0.5)
    var scaleh = _gaussian(D, 7, 0.5)
    var gateh = _gaussian(D, 8, 0.5)
    var cosh = _gaussian(S * H * (Dh // 2), 9, 1.0)
    var sinh = _gaussian(S * H * (Dh // 2), 10, 1.0)

    # activations enter F32 (the Klein trainer regime: F32 activations, BF16
    # weights — klein_stack_lora.mojo:473-475).
    var x_arc = TArc(Tensor.from_host(xh.copy(), [S, D], STDtype.F32, ctx))

    # BASE weights BF16 (Klein transformer matrices are bf16 on disk).
    var w1_arc = TArc(Tensor.from_host(w1h.copy(), [3 * D + 2 * F, D], STDtype.BF16, ctx))
    var w2_arc = TArc(Tensor.from_host(w2h.copy(), [D, D + F], STDtype.BF16, ctx))
    var qn_arc = TArc(Tensor.from_host(qnh.copy(), [Dh], STDtype.F32, ctx))
    var kn_arc = TArc(Tensor.from_host(knh.copy(), [Dh], STDtype.F32, ctx))
    var w = SingleBlockWeights(w1_arc^, w2_arc^, qn_arc^, kn_arc^, D, F, ctx, True)

    var mv = single_modvecs_to_device(
        SingleModVecs(shifth.copy(), scaleh.copy(), gateh.copy()), D, ctx
    )
    var lora = SingleBlockLoraDevice(None, None)   # base-only gate (no LoRA)

    var cos_t = Tensor.from_host(cosh.copy(), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sinh.copy(), [S * H, Dh // 2], STDtype.F32, ctx)

    # int8 payload: quantize the two frozen bf16 base weights TENSORWISE, once.
    var payload = quantize_single_block_int8(w.w1[], w.w2[], ctx)

    # ── bf16 base forward (REFERENCE) ──
    var ref_fwd = single_block_int8_base_forward_device_resident[H, Dh, S](
        x_arc, w, mv, lora, Optional[SingleBlockInt8](None),
        cos_t, sin_t, D, F, EPS, ctx,
    )
    var ref_h = ref_fwd.out[].to_host(ctx)

    # ── int8 base forward ──
    var i8_fwd = single_block_int8_base_forward_device_resident[H, Dh, S](
        x_arc, w, mv, lora, Optional[SingleBlockInt8](payload^),
        cos_t, sin_t, D, F, EPS, ctx,
    )

    var harness = ParityHarness(0.997)
    var r = harness.compare(i8_fwd.out[], ref_h, ctx)

    print("")
    print("---- int8 base block output vs bf16 base block output ----")
    print("  cos =", r.cos, "  max_abs =", r.max_abs, "  n =", r.n)
    if r.cos >= 0.997:
        print("  RESULT: cos >= 0.997 (task bar) -> PASS")
    elif r.cos >= 0.99:
        print("  RESULT: 0.99 <= cos < 0.997 (krea2 int8 bar met, task bar missed)")
    else:
        print("  RESULT: cos < 0.99 -> FAIL")
    print("")
    print("NOTE: this file only REPORTS the number; the human gates.")
