# boogu_c5_probe.mojo — compile+run probe for Boogu-Image C5 output norm
# (LuminaLayerNormContinuous / norm_out) + the 2D unpatchify rearrange.
#
# (a) Loads norm_out from the REAL transformer dir, runs BooguNormOut.forward on a
#     synthetic hidden[1,272,3360] + temb[1,1024], prints the [1,272,64] output
#     shape + std.
# (b) Runs boogu_unpatchify on a synthetic [256,64] and prints the [16,32,32]
#     output shape.
# (c) Round-trip self-check: take a synthetic [16,32,32], patchify it the way C6
#     will (`c (h p1)(w p2)->(h w)(p1 p2 c)`) -> [256,64], then boogu_unpatchify
#     -> [16,32,32], and assert max-abs-diff == 0 (proves the layout is the exact
#     inverse of the C1/C6 patchify). Prints the round-trip max-abs.
#
# The orchestrator owns numeric parity vs the torch oracle (boogu_c5_oracle.py);
# this probe only proves the Mojo C5 code COMPILES and EXECUTES (exit 0) and that
# the unpatchify layout is the exact inverse of the patchify.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg \
#     && pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c5_probe.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.boogu_dit import BooguNormOut, boogu_unpatchify


comptime TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"
comptime SEQ = 272                       # joint seq (16 instruct + 256 img)
comptime HIDDEN = 3360
comptime TEMB_DIM = 1024
comptime OUT_DIM = 64                    # patch*patch*out_ch = 2*2*16
comptime IMG_LEN = 256                   # h_tok*w_tok
comptime H_TOK = 16
comptime W_TOK = 16
comptime OUT_CH = 16
comptime PATCH = 2
comptime H_OUT = 32                      # h_tok*patch
comptime W_OUT = 32                      # w_tok*patch


def _std(host: List[Float32]) -> Float32:
    var n = len(host)
    if n == 0:
        return Float32(0.0)
    var mean = Float32(0.0)
    for i in range(n):
        mean += host[i]
    mean /= Float32(n)
    var var_acc = Float32(0.0)
    for i in range(n):
        var d = host[i] - mean
        var_acc += d * d
    var_acc /= Float32(n)
    return sqrt(var_acc)


def main() raises:
    var ctx = DeviceContext()

    # ── (a) norm_out forward ─────────────────────────────────────────────────
    print("[c5-probe] loading norm_out from", TF_DIR)
    var st = ShardedSafeTensors.open(TF_DIR)
    var norm_out = BooguNormOut.load(st, "norm_out", ctx)
    print("[c5-probe] norm_out loaded")

    # synthetic hidden [1,272,3360] + temb [1,1024] (deterministic).
    var hidden_vals = List[Float32]()
    for i in range(SEQ * HIDDEN):
        var x = Float32((i * 2654435761) % 2000) / Float32(1000.0) - Float32(1.0)
        hidden_vals.append(x)
    var hidden = Tensor.from_host(hidden_vals, [1, SEQ, HIDDEN], STDtype.BF16, ctx)

    var temb_vals = List[Float32]()
    for i in range(TEMB_DIM):
        var x = Float32((i * 40503) % 2000) / Float32(1000.0) - Float32(1.0)
        temb_vals.append(x)
    var temb = Tensor.from_host(temb_vals, [1, TEMB_DIM], STDtype.BF16, ctx)

    print(
        "[c5-probe] hidden std:",
        _std(hidden.to_host(ctx)),
        " temb std:",
        _std(temb.to_host(ctx)),
    )

    var y = norm_out.forward(hidden, temb, ctx)        # [1,272,64]
    var ys = y.shape()
    print(
        "norm_out shape:", ys[0], ys[1], ys[2], "(expect 1", SEQ, OUT_DIM, ")"
    )
    print("norm_out std:", _std(y.to_host(ctx)))

    # ── (b) boogu_unpatchify on a synthetic [256,64] ─────────────────────────
    var tok_vals = List[Float32]()
    for i in range(IMG_LEN * OUT_DIM):
        var x = Float32((i * 2246822519) % 2000) / Float32(1000.0) - Float32(1.0)
        tok_vals.append(x)
    var tokens = Tensor.from_host(tok_vals, [IMG_LEN, OUT_DIM], STDtype.F32, ctx)
    var img = boogu_unpatchify(tokens, H_TOK, W_TOK, ctx)   # [16,32,32]
    var ims = img.shape()
    print(
        "unpatchify shape:", ims[0], ims[1], ims[2],
        "(expect", OUT_CH, H_OUT, W_OUT, ")",
    )

    # ── (c) round-trip self-check: image -> patchify (C6 layout) -> unpatchify.
    # Build a synthetic image [16,32,32] (row-major over [c, oh, ow]) and patchify
    # it host-side with the EXACT C6 layout `c (h p1)(w p2) -> (h w)(p1 p2 c)`:
    #   token = h*w_tok + w,  within = (p1*p + p2)*C + c
    #   patches[token, within] = image[c, h*p + p1, w*p + p2]
    # then boogu_unpatchify must recover the image bit-for-bit (max-abs == 0).
    var img_host = List[Float32]()
    for i in range(OUT_CH * H_OUT * W_OUT):
        var x = Float32((i * 2654435761) % 4093) / Float32(1000.0) - Float32(2.0)
        img_host.append(x)

    # host-side patchify into [256,64] following the C6 forward layout.
    var patch_host = List[Float32]()
    for _ in range(IMG_LEN * OUT_DIM):
        patch_host.append(Float32(0.0))
    for h in range(H_TOK):
        for w in range(W_TOK):
            var token = h * W_TOK + w
            for p1 in range(PATCH):
                for p2 in range(PATCH):
                    for c in range(OUT_CH):
                        var within = (p1 * PATCH + p2) * OUT_CH + c
                        var oh = h * PATCH + p1
                        var ow = w * PATCH + p2
                        var img_off = (c * H_OUT + oh) * W_OUT + ow
                        patch_host[token * OUT_DIM + within] = img_host[img_off]

    var patches = Tensor.from_host(
        patch_host, [IMG_LEN, OUT_DIM], STDtype.F32, ctx
    )
    var recovered = boogu_unpatchify(patches, H_TOK, W_TOK, ctx)   # [16,32,32]
    var rec_host = recovered.to_host(ctx)

    var max_abs = Float32(0.0)
    for i in range(OUT_CH * H_OUT * W_OUT):
        var d = rec_host[i] - img_host[i]
        if d < Float32(0.0):
            d = -d
        if d > max_abs:
            max_abs = d
    print("round-trip max-abs (MUST be 0):", max_abs)
    if max_abs != Float32(0.0):
        raise Error("boogu_unpatchify: NOT the exact inverse of C6 patchify")

    print("[c5-probe] OK")
