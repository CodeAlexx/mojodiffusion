# models/krea2/parity/krea2_flow_patchify_fused_smoke.mojo
# Verifies the fused train-latent kernel matches the old path:
#   randn(BF16) -> flow_match_noise_target -> krea2_patchify(x_t/target)
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/krea2/parity/krea2_flow_patchify_fused_smoke.mojo \
#     -o /tmp/krea2_flow_patchify_fused_smoke && \
#   /tmp/krea2_flow_patchify_fused_smoke

from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.random import randn
from serenitymojo.training.schedule import flow_match_noise_target
from serenitymojo.models.krea2.krea2_cache_reader import (
    krea2_patchify, krea2_flow_noise_patchify,
)


def _max_abs(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float32:
    if a.numel() != b.numel() or a.dtype() != b.dtype():
        raise Error("krea2_flow_patchify_fused_smoke: tensor mismatch")
    var ah = a.to_host(ctx)
    var bh = b.to_host(ctx)
    var m = Float32(0.0)
    for i in range(len(ah)):
        var d = ah[i] - bh[i]
        if d < Float32(0.0):
            d = -d
        if d > m:
            m = d
    return m


def main() raises:
    var ctx = DeviceContext()
    comptime LH = 64
    comptime LW = 64
    var seed_clean = UInt64(123)
    var seed_noise = UInt64(456)
    var sigma = Float32(0.37)

    var clean = randn([1, 16, LH, LW], seed_clean, STDtype.BF16, ctx)
    var noise = randn([1, 16, LH, LW], seed_noise, STDtype.BF16, ctx)
    var fm = flow_match_noise_target(clean, sigma, noise, ctx)
    var ref_img = krea2_patchify[LH, LW](fm.x_t, ctx)
    var ref_target = krea2_patchify[LH, LW](fm.target, ctx)

    var fused = krea2_flow_noise_patchify[LH, LW](clean, sigma, seed_noise, ctx)
    var img_diff = _max_abs(ref_img, fused.img[], ctx)
    var target_diff = _max_abs(ref_target, fused.target[], ctx)
    print("krea2 fused flow+patchify max_abs img=", img_diff, " target=", target_diff)
    if img_diff != Float32(0.0) or target_diff != Float32(0.0):
        raise Error("krea2 fused flow+patchify mismatch")
    print("ALL PASS — krea2_flow_patchify_fused_smoke")
