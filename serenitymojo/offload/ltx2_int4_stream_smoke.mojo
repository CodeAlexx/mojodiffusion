# offload/ltx2_int4_stream_smoke.mojo — GATE for the INT4 (SVDQuant class-A) DiT
# block streamer.
#
# Proves the dequant-first INT4 stream produces the EXACT dict contract the
# existing LTX-2 block forward consumes:
#   1) open the INT4 slab via LTX2Int4BlockStream,
#   2) load block 0 (reconstructs every quantized class-A linear to dense BF16),
#   3) print each reconstructed/passed-through weight's shape,
#   4) hand the block to LTX2BlockWeights.from_fp8_block — which raises if any
#      required canonical key is missing / mis-named — and assert it builds.
#
# A green run means: the slab keys/shapes are right, svdquant_reconstruct_weight
# accepts them, and the block-weights struct accepts the resulting dict. It does
# NOT run the forward (that is a later, heavier GPU gate).
#
# Build (serial, -O2; remove the stale pkg first):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/offload/ltx2_int4_stream_smoke.mojo -o /tmp/ltx2_int4_stream_smoke
#   /tmp/ltx2_int4_stream_smoke

from max.gpu.host import DeviceContext
from serenitymojo.offload.ltx2_int4_block_stream import LTX2Int4BlockStream
from serenitymojo.models.dit.ltx2_dit import LTX2Config, LTX2BlockWeights


comptime SLAB = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-svdint4-r32.safetensors"
comptime BLOCK_IDX = 0


def _shape_str(s: List[Int]) -> String:
    var out = String("[")
    for i in range(len(s)):
        if i > 0:
            out += ", "
        out += String(s[i])
    out += "]"
    return out^


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()

    print("=== LTX-2 INT4 (SVDQuant class-A) block-stream smoke ===")
    print("  slab :", SLAB)
    print("  block:", BLOCK_IDX)

    var stream = LTX2Int4BlockStream.open(String(SLAB))
    print("  block_count:", stream.block_count())

    print("  [load] LTX2Int4BlockStream.load_block_bf16 (int4 -> dense BF16)")
    var blk = stream.load_block_bf16(BLOCK_IDX, ctx)

    print("  reconstructed / passthrough tensors in block", BLOCK_IDX, ":", len(blk))
    var n_keys = 0
    for ref e in blk.items():
        print("    ", e.key, _shape_str(e.value[].shape()))
        n_keys += 1
    if n_keys == 0:
        raise Error("INT4 stream smoke: block 0 produced ZERO tensors")

    print("  [build] LTX2BlockWeights.from_fp8_block(block, LTX2Config.ltx2())")
    var w = LTX2BlockWeights.from_fp8_block(blk^, cfg, ctx)
    _ = w  # keep alive; the build succeeding IS the assertion

    print("LTX-2 INT4 block-stream smoke PASS "
          "(from_fp8_block accepted the reconstructed block)")
