# seedvr2_block_parity.mojo — Full-block parity probe for SeedVR2-3B
# NaMMSRTransformerBlock (block 0), composed from the 3 verified attention
# sub-pieces (attn_qkv_norm + apply_rope3d + window_attn_projout) plus the
# SwiGLU MLP and AdaSingle modulation.  Runs the whole dual-stream block from
# the torch oracle's block-0 inputs and gates the VID output vs block_00.out.
# BF16 compute, so the PASS bar is cos >= 0.999 (deep-block bf16 accumulation
# may sit slightly below the per-op bars).
#
#   weights: seedvr2_dit.safetensors (bf16), all keys under `blocks.0.`
#   oracle:  dit_oracle.safetensors (F32 refs -> bf16)
#     inputs  b0in_vid [256,2560]  b0in_txt [58,2560]  b0in_emb [1,15360]
#     freqs   b0rope_vid_freqs [256,126] (F32)  b0rope_txt_freqs [232,126] (F32)
#     target  block_00.out [256,2560]  (VID output of block 0)
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/dit/tests/seedvr2_block_parity.mojo

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.seedvr2_dit import load_block0, mmdit_block

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

    # ── rope angle tables (F32, consumed as-is) ────────────────────────────────
    var vid_freqs = load_bias(orc, "b0rope_vid_freqs", ctx)   # F32 [256,126]
    var txt_freqs = load_bias(orc, "b0rope_txt_freqs", ctx)   # F32 [232,126]

    # ── block-0 weights (bf16) ─────────────────────────────────────────────────
    var w = load_block0(st, ctx)

    var res = mmdit_block(vid, txt, emb, vid_freqs, txt_freqs, w, ctx)
    var vs = res[0].shape()
    var ts = res[1].shape()
    print("vid_out shape:", vs[0], vs[1], " txt_out shape:", ts[0], ts[1])

    var vid_h = res[0].to_host(ctx)
    var vid_ref = load_bias(orc, "block_00.out", ctx).to_host(ctx)

    var vres = _cos_maxabs(vid_h, vid_ref)
    print("block vid cos=", vres[0], " max_abs=", vres[1])
    if vres[0] >= 0.999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
