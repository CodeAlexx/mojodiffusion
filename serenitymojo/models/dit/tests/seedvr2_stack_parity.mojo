# seedvr2_stack_parity.mojo — Full 32-block stack parity probe for SeedVR2-3B.
# Loads all 32 NaMMSRTransformerBlocks (blocks 0-9 SEPARATE .vid/.txt weights,
# blocks 10-31 SHARED .all weights) and runs dit_stack from the torch oracle's
# block-0 inputs, gating the final VID output vs block_31.out.
#
# The SAME emb + rope freqs feed every block (window structure is uniform for
# this input). BF16 compute accumulated over 32 layers lowers cos below the
# per-block 0.9999963 bar; PASS >= 0.99 (report the actual number).
#
#   weights: seedvr2_dit.safetensors (bf16)
#   oracle:  dit_oracle.safetensors (F32 refs -> bf16)
#     inputs  b0in_vid [256,2560]  b0in_txt [58,2560]  b0in_emb [1,15360]
#     freqs   b0rope_vid_freqs [256,126] (F32)  b0rope_txt_freqs [232,126] (F32)
#     target  block_31.out [256,2560]  (VID output after all 32 blocks)
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/dit/tests/seedvr2_stack_parity.mojo

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.seedvr2_dit import Block0Weights, load_block, dit_stack

comptime WEIGHTS = "/home/alex/models/seedvr2-3b/seedvr2_dit.safetensors"
comptime ORACLE = "/home/alex/models/seedvr2-3b/dit_oracle.safetensors"


def _cos_maxabs(a: List[Float32], b: List[Float32]) -> Tuple[Float32, Float32]:
    var n = len(a)
    if len(b) < n:
        n = len(b)
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    var maxd: Float32 = 0.0
    for i in range(n):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > maxd:
            maxd = d
    var denom = sqrt(na) * sqrt(nb)
    var cos: Float32 = 0.0
    if denom > 0.0:
        cos = Float32(dot / denom)
    return (cos, maxd)


def main() raises:
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))

    var orc = SafeTensors.open(ORACLE)
    var st = SafeTensors.open(WEIGHTS)

    # ── block-0 inputs (F32 oracle -> bf16) ────────────────────────────────────
    var vid = cast_tensor(load_bias(orc, "b0in_vid", ctx), STDtype.BF16, ctx)  # [256,2560]
    var txt = cast_tensor(load_bias(orc, "b0in_txt", ctx), STDtype.BF16, ctx)  # [58,2560]
    var emb = cast_tensor(load_bias(orc, "b0in_emb", ctx), STDtype.BF16, ctx)  # [1,15360]

    # ── rope angle tables (F32) — IDENTICAL for all 32 blocks (borrowed) ────────
    var vid_freqs = load_bias(orc, "b0rope_vid_freqs", ctx)   # F32 [256,126]
    var txt_freqs = load_bias(orc, "b0rope_txt_freqs", ctx)   # F32 [232,126]

    # ── load all 32 blocks (0-9 separate .vid/.txt, 10-31 shared .all) ──────────
    var blocks = List[ArcPointer[Block0Weights]]()
    for i in range(32):
        blocks.append(ArcPointer(load_block(st, i, ctx)))
        if i % 8 == 0 or i == 31:
            print("loaded block", i)
    print("all 32 blocks loaded")

    # ── run the full stack ──────────────────────────────────────────────────────
    var out = dit_stack(vid, txt, emb, vid_freqs, txt_freqs, blocks, ctx)
    var os = out.shape()
    print("final vid shape:", os[0], os[1])

    var out_h = out.to_host(ctx)
    var ref_h = load_bias(orc, "block_31.out", ctx).to_host(ctx)

    var r = _cos_maxabs(out_h, ref_h)
    print("stack vid cos=", r[0], " max_abs=", r[1])
    if r[0] >= 0.99:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
