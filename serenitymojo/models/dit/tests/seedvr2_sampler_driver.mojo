# seedvr2_sampler_driver.mojo — SeedVR2-3B full sampler-loop parity probe.
#
# Runs the euler v-lerp sampler for 8 steps, each invoking the DiT twice (CFG
# pos txt_len=58 / neg txt_len=64) with PRELOADED weights (load_dit_weights once),
# and gates the accumulated latent x vs the torch full-pipeline `final_latent`.
#
#   weights: seedvr2_dit.safetensors (bf16, 6.8GB) -> load_dit_weights (once)
#   text:    seedvr2_text_emb.safetensors  pos_emb[58,5120] neg_emb[64,5120] (F32->bf16)
#   oracle:  pipeline_oracle.safetensors
#              cond_flat[1024,17]  x_T[1024,16]  tsched[8]  s_all[8]  final_latent[1024,16] (F32)
#
# Loop (per SeedVR common/diffusion euler v-lerp, CFG scale 7.5, T=1000):
#   x = x_T (F32)
#   for i in 0..7:
#     vid = concat(bf16(x), bf16(cond), dim=1)        # [1024,33]
#     pp  = full_dit_forward_pre(vid, pos_emb, 58, tsched[i], w)   # bf16 [1024,16]
#     pn  = full_dit_forward_pre(vid, neg_emb, 64, tsched[i], w)
#     pred = cfg(f32(pp), f32(pn), 7.5)               # F32
#     x   = euler_vlerp_step(pred, x, tsched[i], s_all[i], 1000)   # F32
#
# 16 bf16 DiT forwards + CFG-amplified (7.5) accumulation over 8 steps -> PASS if
# cos >= 0.98 vs final_latent.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . \
#        serenitymojo/models/dit/tests/seedvr2_sampler_driver.mojo

from max.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.real_weights import load_bias
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import concat
from serenitymojo.models.dit.seedvr2_dit import load_dit_weights, full_dit_forward_pre
from serenitymojo.models.dit.seedvr2_sampler import cfg, euler_vlerp_step

comptime WEIGHTS = "/home/alex/models/seedvr2-3b/seedvr2_dit.safetensors"
comptime TEXT = "/home/alex/models/seedvr2-3b/seedvr2_text_emb.safetensors"
comptime ORACLE = "/home/alex/models/seedvr2-3b/pipeline_oracle.safetensors"


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
    var txe = SafeTensors.open(TEXT)
    var orc = SafeTensors.open(ORACLE)

    # ── text embeddings (F32 -> bf16) ───────────────────────────────────────────
    var pos_emb = cast_tensor(load_bias(txe, "pos_emb", ctx), STDtype.BF16, ctx)  # [58,5120]
    var neg_emb = cast_tensor(load_bias(txe, "neg_emb", ctx), STDtype.BF16, ctx)  # [64,5120]
    print("pos_emb", pos_emb.shape()[0], pos_emb.shape()[1],
          " neg_emb", neg_emb.shape()[0], neg_emb.shape()[1])

    # ── sampler inputs ──────────────────────────────────────────────────────────
    var x = load_bias(orc, "x_T", ctx)                                # F32 [1024,16]
    var cond_bf = cast_tensor(load_bias(orc, "cond_flat", ctx), STDtype.BF16, ctx)  # [1024,17]
    var tsched = load_bias(orc, "tsched", ctx).to_host(ctx)           # [8]
    var s_all = load_bias(orc, "s_all", ctx).to_host(ctx)            # [8]
    print("x_T", x.shape()[0], x.shape()[1], " cond", cond_bf.shape()[0], cond_bf.shape()[1],
          " steps", len(tsched))

    # ── preload ALL DiT weights ONCE (heads + tail + 32 blocks) ─────────────────
    var w = load_dit_weights(st, ctx)
    print("DiT weights loaded (32 blocks + heads/tail)")

    # ── euler v-lerp sampler loop (8 steps x 2 CFG forwards) ────────────────────
    for i in range(8):
        var x_bf = cast_tensor(x, STDtype.BF16, ctx)                  # [1024,16] bf16
        var vid = concat(1, ctx, x_bf, cond_bf)                       # [1024,33] bf16
        var pp = full_dit_forward_pre(vid, pos_emb, 58, tsched[i], w, ctx)  # bf16 [1024,16]
        var pn = full_dit_forward_pre(vid, neg_emb, 64, tsched[i], w, ctx)
        var pred_bf = cfg(pp, pn, Float32(7.5), ctx)                  # bf16 CFG (torch autocast)
        var pred = cast_tensor(pred_bf, STDtype.F32, ctx)            # upcast for F32 step
        x = euler_vlerp_step(pred, x, tsched[i], s_all[i], Float32(1000.0), ctx)
        print("  step", i, "t=", tsched[i], "s=", s_all[i], "done")

    # ── gate x vs final_latent ──────────────────────────────────────────────────
    var out_h = x.to_host(ctx)
    var ref_h = load_bias(orc, "final_latent", ctx).to_host(ctx)
    var r = _cos_maxabs(out_h, ref_h)
    print("final_latent cos=", r[0], " max_abs=", r[1])
    if r[0] >= 0.98:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")
