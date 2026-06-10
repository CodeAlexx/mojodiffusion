# pipeline/ltx2_decode_latent_gate.mojo — Mojo VAE decode vs ltx_core on the
# EXACT generated latent (quality-funnel step 1b).
#
# The ltx_core oracle decode of the Mojo MVP final latent produced coherent
# frames with a well-formed face (scripts/ltx2_decode_latent_oracle.py,
# output/ltx2_decode_oracle/). This gate decodes the SAME latent with the pure
# Mojo decoder and compares RGB-space output to the oracle:
#   - per-frame + global cos and max_abs on [0,1] RGB
#   - writes mojo frames next to the oracle's for eyeball diff
# If this gate fails while the oracle frames are clean, the Mojo decode path
# (decoder math, un-normalize stats, layout, or range conversion) is the
# face-distortion bug. If it passes, the bug is in the frame WRITING/mux only.
#
# Build/run (needs libcudnn on the loader path for conv3d_cudnn):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/pipeline/ltx2_decode_latent_gate.mojo -o /tmp/ltx2_decode_gate
#   LD_LIBRARY_PATH=/home/alex/libtorch-cu124/libtorch/lib /tmp/ltx2_decode_gate

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.vae.ltx2_vae_decoder import (
    LTX2VaeDecoderWeights, decode as decode_video,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import slice, reshape


comptime CKPT_BF16 = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
comptime LATENTS = "/home/alex/mojodiffusion/output/ltx2_mvp/final_latents.safetensors"
comptime ORACLE = "/home/alex/mojodiffusion/output/ltx2_decode_oracle/oracle_decode.safetensors"
comptime OUT_DIR = "/home/alex/mojodiffusion/output/ltx2_decode_oracle"
comptime NF = 2
comptime NH = 8
comptime NW = 8


def _sh4(a: Int, b: Int, c: Int, d: Int) raises -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d)
    return s^


def _pad2(n: Int) -> String:
    if n < 10:
        return String("0") + String(n)
    return String(n)


def main() raises:
    var ctx = DeviceContext()
    print("=== LTX2 decode-latent gate: Mojo VAE decode vs ltx_core on the exact latent ===")

    var st = ShardedSafeTensors.open(LATENTS)
    var video_x = Tensor.from_view(st.tensor_view("video_x"), ctx)
    var vsh = video_x.shape()
    print("latent video_x: [", vsh[0], ",", vsh[1], ",", vsh[2], ",", vsh[3], ",", vsh[4], "]")

    var ost = ShardedSafeTensors.open(ORACLE)
    var oracle = Tensor.from_view(ost.tensor_view("decoded_video"), ctx)
    var osh = oracle.shape()  # [F, H, W, C] F32 in [0,1]
    print("oracle decoded: [", osh[0], ",", osh[1], ",", osh[2], ",", osh[3], "]")
    var n_frames = osh[0]
    var hh = osh[1]
    var ww = osh[2]

    print("[load] Mojo LTX2 VAE decoder")
    var vae = LTX2VaeDecoderWeights.load(String(CKPT_BF16), ctx)
    var lat_bf16 = cast_tensor(video_x, STDtype.BF16, ctx)
    var frames = decode_video[1, 128, NF, NH, NW](vae, lat_bf16, ctx)  # [1,3,F,H,W] in [-1,1]
    var fsh = frames.shape()
    print("mojo decoded NCDHW: [", fsh[0], ",", fsh[1], ",", fsh[2], ",", fsh[3], ",", fsh[4], "]")
    if fsh[2] != n_frames or fsh[3] != hh or fsh[4] != ww:
        raise Error("decode shape mismatch vs oracle")

    # host copies, both as F32
    var mojo_h = cast_tensor(frames, STDtype.F32, ctx).to_host(ctx)   # NCDHW [-1,1]
    var oracle_h = oracle.to_host(ctx)                                # FHWC  [0,1]

    var worst_cos = Float64(2.0)
    var global_dot = Float64(0.0)
    var global_na = Float64(0.0)
    var global_nb = Float64(0.0)
    var global_max = Float64(0.0)
    for f in range(n_frames):
        var dot = Float64(0.0)
        var na = Float64(0.0)
        var nb = Float64(0.0)
        var mx = Float64(0.0)
        for y in range(hh):
            for x in range(ww):
                for c in range(3):
                    # mojo NCDHW [1,3,F,H,W] in [-1,1] -> [0,1]
                    var mi = ((c * n_frames + f) * hh + y) * ww + x
                    var mv = (Float64(mojo_h[mi]) + 1.0) * 0.5
                    if mv < 0.0:
                        mv = 0.0
                    if mv > 1.0:
                        mv = 1.0
                    # oracle FHWC
                    var oi = ((f * hh + y) * ww + x) * 3 + c
                    var ov = Float64(oracle_h[oi])
                    dot += mv * ov
                    na += mv * mv
                    nb += ov * ov
                    var d = mv - ov
                    if d < 0.0:
                        d = -d
                    if d > mx:
                        mx = d
        var cosv = dot / (sqrt(na) * sqrt(nb) + 1e-12)
        print("  frame", f, " cos=", cosv, " max_abs=", mx)
        if cosv < worst_cos:
            worst_cos = cosv
        global_dot += dot
        global_na += na
        global_nb += nb
        if mx > global_max:
            global_max = mx

    var gcos = global_dot / (sqrt(global_na) * sqrt(global_nb) + 1e-12)
    print("GLOBAL: cos=", gcos, " worst_frame_cos=", worst_cos, " max_abs=", global_max)

    # write mojo frames for eyeball diff next to the oracle's
    for fr in range(n_frames):
        var fslice = slice(frames, 2, fr, 1, ctx)
        var fs = fslice.shape()
        var chw = reshape(fslice, _sh4(fs[0], fs[1], fs[3], fs[4]), ctx)
        save_png(chw, String(OUT_DIR) + String("/mojo_frame") + _pad2(fr) + String(".png"), ctx, ValueRange.SIGNED)
    print("wrote mojo frames -> ", OUT_DIR, "/mojo_frame*.png")

    if gcos >= 0.999 and global_max <= 0.05:
        print("VERDICT: PASS — Mojo decode matches ltx_core on the real latent;")
        print("         face distortion must come from a different stage.")
    else:
        print("VERDICT: DIVERGES — Mojo decode path (math/stats/layout/range) is the suspect.")
