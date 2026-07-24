# seedvr2_vidin_embin_parity.mojo — Chunk-1 parity probe for SeedVR2-3B NaDiT
# input embeddings: vid_in (NaPatchIn) and emb_in (TimeEmbedding).
#
# Feeds the torch oracle's exact inputs through the pure-Mojo ops and compares
# each output against the torch bf16 reference (stored F32). BF16 compute, so
# PASS bar is cos >= 0.999 (not exactly 1.0).
#
#   weights: seedvr2_dit.safetensors (bf16)
#     vid_in.proj.weight [2560,132] / vid_in.proj.bias [2560]
#     emb_in.proj_in.weight [2560,256]/bias, proj_hid [2560,2560]/bias, proj_out [15360,2560]/bias
#   oracle:  dit_oracle.safetensors
#     vid_in_raw [1024,33] (F32 -> feed as bf16); vid_in.out [256,2560]
#     timestep [1] (=500.0);                      emb_in.out [1,15360]
#   vid_shape = (T=4, H=16, W=16) hardcoded.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/dit/tests/seedvr2_vidin_embin_parity.mojo

from std.gpu.host import DeviceContext
from math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.seedvr2_dit import vid_in, emb_in

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

    var st = SafeTensors.open(WEIGHTS)
    var orc = SafeTensors.open(ORACLE)

    # ── (A) vid_in ──────────────────────────────────────────────────────────
    var vid_raw_f32 = load_bias(orc, "vid_in_raw", ctx)          # [1024,33] F32
    var vid_raw = cast_tensor(vid_raw_f32, STDtype.BF16, ctx)    # -> bf16
    var vproj_w = load_bias(st, "vid_in.proj.weight", ctx)       # [2560,132] bf16
    var vproj_b = load_bias(st, "vid_in.proj.bias", ctx)         # [2560] bf16

    var vid_out = vid_in(vid_raw, vproj_w, vproj_b, 4, 16, 16, ctx)  # [256,2560]
    var vs = vid_out.shape()
    print("vid_in out shape:", vs[0], vs[1])

    var vid_h = vid_out.to_host(ctx)
    var vid_ref = load_bias(orc, "vid_in.out", ctx).to_host(ctx)
    var vres = _cos_maxabs(vid_h, vid_ref)

    # ── (B) emb_in ──────────────────────────────────────────────────────────
    var ts_h = load_bias(orc, "timestep", ctx).to_host(ctx)     # [1] F32
    var ts = ts_h[0]
    print("timestep:", ts)
    var pin_w = load_bias(st, "emb_in.proj_in.weight", ctx)
    var pin_b = load_bias(st, "emb_in.proj_in.bias", ctx)
    var phid_w = load_bias(st, "emb_in.proj_hid.weight", ctx)
    var phid_b = load_bias(st, "emb_in.proj_hid.bias", ctx)
    var pout_w = load_bias(st, "emb_in.proj_out.weight", ctx)
    var pout_b = load_bias(st, "emb_in.proj_out.bias", ctx)

    var emb_out = emb_in(ts, pin_w, pin_b, phid_w, phid_b, pout_w, pout_b, ctx)
    var es = emb_out.shape()
    print("emb_in out shape:", es[0], es[1])

    var emb_h = emb_out.to_host(ctx)
    var emb_ref = load_bias(orc, "emb_in.out", ctx).to_host(ctx)
    var eres = _cos_maxabs(emb_h, emb_ref)

    print(
        "vid_in cos=", vres[0], " max_abs=", vres[1],
        " emb_in cos=", eres[0], " max_abs=", eres[1],
    )
    if vres[0] >= 0.999 and eres[0] >= 0.999:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
