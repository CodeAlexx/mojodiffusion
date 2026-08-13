# krea2_fp8_cache_gate.mojo — BYTE-IDENTICAL round-trip gate for the fp8 sidecar.
#
# The cache's bar is bit-exactness: a store loaded from the sidecar must be
# BYTE-FOR-BYTE the store the quantizer produced (identical base bytes ⇒
# identical training math). This gate proves the whole write→read round trip:
#
#   1. quantize the 28 frozen krea2 blocks fresh (build_krea2_resident_fp8) —
#      the production quantizer, our oracle.
#   2. write the sidecar (save_krea2_fp8_cache) and validate its staleness meta
#      (krea2_fp8_cache_valid True for the real checkpoint; False for wrong
#      nblocks / a bogus checkpoint path).
#   3. for blocks 0 and 27, load every fp8-byte tensor + per-row scale tensor +
#      the 5 small tensors BACK from the sidecar and compare device bytes to the
#      fresh store — BYTE-IDENTICAL required (dtype + length + every byte).
#
# Only 2 blocks are byte-compared so peak VRAM stays ~13GB (fresh 12GB resident +
# a couple loaded blocks); the full 28-block load path is exercised by the warm
# training run. Usage:
#   krea2_fp8_cache_gate [<checkpoint.safetensors>] [<sidecar_path>]
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.sys import argv
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.ffi import sys_remove
from serenitymojo.models.krea2.krea2_stack import (
    Krea2ResidentFp8,
    build_krea2_resident_fp8,
)
from serenitymojo.models.krea2.krea2_fp8_cache import (
    krea2_fp8_cache_valid,
    save_krea2_fp8_cache,
    _load_raw_h2d,
    _block_prefix,
)

comptime NBLOCKS = 28   # Krea2Config single_mmdit_large_wide layer count
comptime _FP8_KEYS = 8

comptime _DEFAULT_CKPT = (
    "/home/alex/.cache/huggingface/hub/models--krea--Krea-2-Raw/"
    "snapshots/4ad9f4b627a647fad78b3dfeebb09f2654aeb494/raw.safetensors"
)


def _bytes_equal(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Bool:
    """D2H both device buffers and compare every byte (dtype + nbytes first)."""
    if a.dtype() != b.dtype():
        return False
    if a.nbytes() != b.nbytes():
        return False
    var n = a.nbytes()
    if n == 0:
        return False
    var ha = ctx.enqueue_create_host_buffer[DType.uint8](n)
    var hb = ctx.enqueue_create_host_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=ha, src_buf=a.buf)
    ctx.enqueue_copy(dst_buf=hb, src_buf=b.buf)
    ctx.synchronize()
    var pa = ha.unsafe_ptr()
    var pb = hb.unsafe_ptr()
    for i in range(n):
        if pa[i] != pb[i]:
            return False
    return True


def _compare_block(
    st: SafeTensors,
    fresh: Krea2ResidentFp8,
    bi: Int,
    ctx: DeviceContext,
) raises -> List[Int]:
    """Compare every sidecar-loaded tensor of block `bi` to the fresh store.
    Returns [checked, mismatches]."""
    var p = _block_prefix(bi)
    var checked = 0
    var bad = 0
    ref fb = fresh.blocks[bi]

    for ki in range(_FP8_KEYS):
        var lf = _load_raw_h2d(st, p + String("fp8.") + String(ki), ctx)
        checked += 1
        if not _bytes_equal(lf, fb.fp8[ki][], ctx):
            bad += 1
            print("  MISMATCH", p + String("fp8.") + String(ki))
        var ls = _load_raw_h2d(st, p + String("scale.") + String(ki), ctx)
        checked += 1
        if not _bytes_equal(ls, fb.scale[ki][], ctx):
            bad += 1
            print("  MISMATCH", p + String("scale.") + String(ki))

    # The 5 small resident tensors (Tensor is move-only → compare one at a time,
    # no List[Tensor]). Each: load-from-sidecar vs the fresh store's ArcPointer.
    checked += 1
    if not _bytes_equal(_load_raw_h2d(st, p + String("qnorm_scale"), ctx), fb.qnorm_scale[], ctx):
        bad += 1
        print("  MISMATCH", p + String("qnorm_scale"))
    checked += 1
    if not _bytes_equal(_load_raw_h2d(st, p + String("knorm_scale"), ctx), fb.knorm_scale[], ctx):
        bad += 1
        print("  MISMATCH", p + String("knorm_scale"))
    checked += 1
    if not _bytes_equal(_load_raw_h2d(st, p + String("prenorm_scale"), ctx), fb.prenorm_scale[], ctx):
        bad += 1
        print("  MISMATCH", p + String("prenorm_scale"))
    checked += 1
    if not _bytes_equal(_load_raw_h2d(st, p + String("postnorm_scale"), ctx), fb.postnorm_scale[], ctx):
        bad += 1
        print("  MISMATCH", p + String("postnorm_scale"))
    checked += 1
    if not _bytes_equal(_load_raw_h2d(st, p + String("mod_lin"), ctx), fb.mod_lin[], ctx):
        bad += 1
        print("  MISMATCH", p + String("mod_lin"))

    var out: List[Int] = [checked, bad]
    return out^


def main() raises:
    var args = argv()
    var checkpoint = String(_DEFAULT_CKPT)
    if len(args) >= 2:
        checkpoint = String(args[1])
    var cache_path = String("/tmp/krea2_fp8cache_gate.safetensors")
    if len(args) >= 3:
        cache_path = String(args[2])

    print("==== krea2 fp8 sidecar BYTE-IDENTICAL gate ====")
    print("checkpoint =", checkpoint)
    print("sidecar    =", cache_path)

    var ctx = DeviceContext()

    # 1) fresh quantize (the oracle).
    var st_ckpt = ShardedSafeTensors.open(checkpoint)
    print("quantizing", NBLOCKS, "blocks fresh (oracle) ...")
    var fresh = build_krea2_resident_fp8(st_ckpt, String(""), NBLOCKS, ctx)
    print("fresh quantize DONE")

    # 2) write sidecar + validate staleness meta.
    _ = sys_remove(cache_path)   # start clean
    save_krea2_fp8_cache(fresh, checkpoint, cache_path, NBLOCKS, ctx)
    print("sidecar written")

    var ok = krea2_fp8_cache_valid(cache_path, checkpoint, NBLOCKS)
    var neg_nblocks = krea2_fp8_cache_valid(cache_path, checkpoint, NBLOCKS + 1)
    var neg_ckpt = krea2_fp8_cache_valid(
        cache_path, checkpoint + String(".nope"), NBLOCKS
    )
    print("valid(correct)        =", ok, "(expect True)")
    print("valid(wrong nblocks)  =", neg_nblocks, "(expect False)")
    print("valid(bogus checkpoint)=", neg_ckpt, "(expect False)")

    # 3) byte-compare blocks 0 and 27 loaded-from-sidecar vs fresh.
    var st_cache = SafeTensors.open(cache_path)
    var total_checked = 0
    var total_bad = 0
    var probe = [0, NBLOCKS - 1]
    for pi in range(len(probe)):
        var bi = probe[pi]
        print("comparing block", bi, "...")
        var r = _compare_block(st_cache, fresh, bi, ctx)
        total_checked += r[0]
        total_bad += r[1]
        print("  block", bi, ":", r[0], "tensors checked,", r[1], "mismatches")

    var meta_ok = ok and (not neg_nblocks) and (not neg_ckpt)
    print("==== RESULT ====")
    print("tensors byte-compared:", total_checked, " mismatches:", total_bad)
    print("staleness meta gate:", "PASS" if meta_ok else "FAIL")
    if total_bad == 0 and meta_ok:
        print("GATE: PASS (byte-identical round trip + staleness guard)")
    else:
        print("GATE: FAIL")
        raise Error("krea2 fp8 sidecar gate FAILED")
