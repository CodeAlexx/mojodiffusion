# wan22_cache_read_smoke.mojo — cheap standalone check of the Wan2.2 cache reader
# (no 28GB DiT checkpoint). Reuses _load_cache_sample / _count_cache_samples from
# the trainer so the SAME code path (SafeTensors mmap + patchify3d) is exercised.
# Prints each sample's latent std/mean (round-trip gate vs the torch builder).

from max.gpu.host import DeviceContext
from serenitymojo.training.train_wan22_real import (
    _load_cache_sample, _count_cache_samples,
)


def main() raises:
    var ctx = DeviceContext()
    var dir = String("/home/alex/.serenity/wan22_cache/40_woman")
    var n = _count_cache_samples(dir)
    print("cache samples:", n)
    for i in range(n):
        var std = Float32(0.0)
        var mean = Float32(0.0)
        var tl = 0
        var pair = _load_cache_sample(dir, i, ctx, std, mean, tl)
        print("sample", i, " std=", std, " mean=", mean, " txt_len=", tl,
              " len(x0)=", len(pair[0]), " len(text)=", len(pair[1]),
              " x0[0]=", pair[0][0])
