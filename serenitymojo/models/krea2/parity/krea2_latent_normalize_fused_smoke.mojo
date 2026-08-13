# models/krea2/parity/krea2_latent_normalize_fused_smoke.mojo
# Verifies krea2_prepare_cache's fused BF16 latent normalization matches the old
# BF16->F32 cast + sub + div + F32->BF16 path.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/krea2/parity/krea2_latent_normalize_fused_smoke.mojo \
#     -o /tmp/krea2_latent_normalize_fused_smoke && \
#   /tmp/krea2_latent_normalize_fused_smoke

from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import sub, div
from serenitymojo.models.vae.qwenimage_decoder import _vae_mean, _vae_std
from serenitymojo.models.krea2.krea2_prepare_cache import _normalize_latent_to_bf16


def _fill(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32(i % 37) * Float32(0.03125) - Float32(0.5))
    return out^


def _max_abs(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float32:
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
    comptime LH = 8
    comptime LW = 8
    var lat_f32 = Tensor.from_host(
        _fill(16 * LH * LW), [1, 16, LH, LW], STDtype.F32, ctx
    )
    var lat_bf16 = cast_tensor(lat_f32, STDtype.BF16, ctx)
    var mean = Tensor.from_host(_vae_mean(), [1, 16, 1, 1], STDtype.F32, ctx)
    var std = Tensor.from_host(_vae_std(), [1, 16, 1, 1], STDtype.F32, ctx)

    var old_lat_f32 = cast_tensor(lat_bf16, STDtype.F32, ctx)
    var centered = sub(old_lat_f32, mean, ctx)
    var old_clean_f32 = div(centered, std, ctx)
    var old_clean = cast_tensor(old_clean_f32, STDtype.BF16, ctx)
    var fused = _normalize_latent_to_bf16(lat_bf16, mean, std, ctx)

    var diff = _max_abs(old_clean, fused, ctx)
    print("krea2 fused latent normalize max_abs=", diff)
    if diff != Float32(0.0):
        raise Error("krea2 fused latent normalize mismatch")
    print("ALL PASS — krea2_latent_normalize_fused_smoke")
