# krea2_omini_c5_regress_dump.mojo — CONDLEN=0 REGRESSION DUMPER (gate 4c).
#
# Dumps, from the UNCHANGED public krea2 cache-reader surface, the two artifacts
# the C5 reader edits could plausibly perturb for a NON-EDIT cache:
#   pos.bin      = KreaTrainCache.sample_padded[LH,LW,LTMAX](0).pos   [1,LFULL,3] F32
#   padmask.bin  = krea2_build_pad_mask(lt, LTMAX, IMGLEN)            [1,H,L,L] F32
#
# It is run TWICE — once at the pre-C5 tree, once after — and the two dumps are
# compared byte-for-byte (`cmp`). Nothing here is C5-specific, which is the point:
# it only calls API that existed before C5, so the same source compiles on both
# trees and any difference is a real regression.
#
# usage: krea2_omini_c5_regress_dump <cache.safetensors> <out_prefix>
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import argv
from max.gpu.host import DeviceContext

from serenitymojo.io.cap_cache import save_tensor_bin
from serenitymojo.models.krea2.krea2_cache_reader import (
    KreaTrainCache, krea2_build_pad_mask,
)

# The real non-edit cache on disk is the 1024px krea2 eri2 cache: clean.<i> is
# [1,16,128,128] and LT ~190, so LTMAX 256 is a valid (padding) bucket.
comptime LH = 128
comptime LW = 128
comptime LTMAX = 256
# Pad-mask shape is picked SMALL enough to allocate ([1,48,L,L] F32): the 512px
# training shape lt=282 / LTMAX=384 / IMGLEN=1024 -> L=1408 -> 380 MB.
comptime MASK_LT = 282
comptime MASK_LTMAX = 384
comptime MASK_IMGLEN = 1024


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error(
            "usage: krea2_omini_c5_regress_dump <cache.safetensors> <out_prefix>"
        )
    var cache_path = String(args[1])
    var prefix = String(args[2])
    var ctx = DeviceContext()

    var cache = KreaTrainCache.open(cache_path)
    print("[c5-regress] cache", cache_path, " samples=", cache.len())
    var s = cache.sample_padded[LH, LW, LTMAX](0, ctx)
    var psh = s.pos[].shape()
    print("[c5-regress] sample 0 LT=", s.text_len, " pos=[", psh[0], ",",
          psh[1], ",", psh[2], "]")
    save_tensor_bin(s.pos[], prefix + String("_pos.bin"), ctx)

    var mask = krea2_build_pad_mask(MASK_LT, MASK_LTMAX, MASK_IMGLEN, ctx)
    var msh = mask.shape()
    print("[c5-regress] padmask=[", msh[0], ",", msh[1], ",", msh[2], ",",
          msh[3], "]")
    save_tensor_bin(mask, prefix + String("_padmask.bin"), ctx)
    print("[c5-regress] WROTE", prefix + String("_pos.bin"), "and",
          prefix + String("_padmask.bin"))
