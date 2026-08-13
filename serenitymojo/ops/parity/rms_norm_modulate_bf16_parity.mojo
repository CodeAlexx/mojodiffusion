# Bit-exact gate for the BF16 RMSNorm + modulation inference fusion.
#
# Run from the repository root:
#   pixi run mojo run -I . serenitymojo/ops/parity/rms_norm_modulate_bf16_parity.mojo

from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.norm import rms_norm, rms_norm_modulate_bf16
from serenitymojo.ops.random import randn


def _gate(name: String, rows: Int, d: Int, per_row: Bool) raises:
    var ctx = DeviceContext()
    var x = randn([rows, d], 2026081201, STDtype.BF16, ctx)
    var weight = randn([d], 2026081202, STDtype.BF16, ctx)
    var scale = (
        randn([rows, d], 2026081203, STDtype.BF16, ctx)
        if per_row
        else randn([d], 2026081203, STDtype.BF16, ctx)
    )
    var shift = (
        randn([rows, d], 2026081204, STDtype.BF16, ctx)
        if per_row
        else randn([d], 2026081204, STDtype.BF16, ctx)
    )
    var reference = modulate(
        rms_norm(x, weight, 1.0e-5, ctx), scale, shift, ctx
    ).to_host(ctx)
    var fused = rms_norm_modulate_bf16(
        x, weight, scale, shift, 1.0e-5, ctx
    ).to_host(ctx)
    var mismatches = 0
    var max_abs = Float32(0.0)
    for i in range(len(reference)):
        var delta = fused[i] - reference[i]
        var abs_delta = delta if delta >= 0.0 else -delta
        if abs_delta != 0.0:
            mismatches += 1
            if abs_delta > max_abs:
                max_abs = abs_delta
    print(name, " mismatches=", mismatches, " max_abs=", max_abs)
    if mismatches != 0:
        raise Error(name + " RMSNorm/modulate fusion is not bit-exact")


def main() raises:
    # H3's released width and per-token AdaLN addressing.
    _gate("H3 per-row [257,5376]", 257, 5376, True)
    # Shared-vector form exercises the ordinary modulate broadcasting contract.
    _gate("broadcast [513,1024]", 513, 1024, False)
    print("PASS: BF16 RMSNorm + modulation fusion is bit-exact")
