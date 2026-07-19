# MultiModalGuider.calculate unit gate.
#
# Oracle: /tmp/ltx2_guider_oracle.safetensors written by the python one-liner
# in the campaign log (serenityflow venv, torch F32):
#   pred = cond + (cfg-1)(cond-uncond) + (mod_scale-1)(cond-mod)
#   factor = rescale*(cond.std()/pred.std()) + (1-rescale); pred *= factor
# for the HQ video (cfg=3, rescale=0.45, mod=3) and audio (cfg=7, rescale=1,
# mod=3) param sets. Bar: cos >= 0.99999 for both.
#
# Run:
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/sampling/ltx2_multimodal_guider_smoke.mojo -o /tmp/guider_smoke
#   LD_LIBRARY_PATH=/home/alex/libtorch-cu124/libtorch/lib /tmp/guider_smoke

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.sampling.ltx2_multimodal_guider import guider_calculate

comptime ORACLE = "/tmp/ltx2_guider_oracle.safetensors"


def _cosine(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ha = a.to_host(ctx)
    var hb = b.to_host(ctx)
    if len(ha) != len(hb):
        raise Error("cosine: length mismatch")
    var dot = 0.0
    var na = 0.0
    var nb = 0.0
    for i in range(len(ha)):
        var x = Float64(ha[i])
        var y = Float64(hb[i])
        if x != x or y != y:
            raise Error("cosine: NaN")
        dot += x * y
        na += x * x
        nb += y * y
    return dot / (sqrt(na) * sqrt(nb) + 1e-30)


def _max_abs_rel(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Float64:
    var ha = a.to_host(ctx)
    var hb = b.to_host(ctx)
    var num = 0.0
    var den = 0.0
    for i in range(len(ha)):
        var d = Float64(ha[i]) - Float64(hb[i])
        num += d * d
        den += Float64(hb[i]) * Float64(hb[i])
    return sqrt(num) / (sqrt(den) + 1e-30)


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(ORACLE))
    var cond = Tensor.from_view_as_f32(st.tensor_view("cond"), ctx)
    var uncond = Tensor.from_view_as_f32(st.tensor_view("uncond"), ctx)
    var mod = Tensor.from_view_as_f32(st.tensor_view("mod"), ctx)
    var video_ref = Tensor.from_view_as_f32(st.tensor_view("video_expected"), ctx)
    var audio_ref = Tensor.from_view_as_f32(st.tensor_view("audio_expected"), ctx)

    print("=== LTX-2 MultiModalGuider unit gate ===")
    # HQ video params: cfg=3.0 stg=0 rescale=0.45 mod=3.0
    var video_pred = guider_calculate(
        cond, uncond, mod, 3.0, 0.0, 0.45, 3.0, ctx
    )
    # HQ audio params: cfg=7.0 stg=0 rescale=1.0 mod=3.0
    var audio_pred = guider_calculate(
        cond, uncond, mod, 7.0, 0.0, 1.0, 3.0, ctx
    )
    var vcos = _cosine(video_pred, video_ref, ctx)
    var acos = _cosine(audio_pred, audio_ref, ctx)
    var vrl2 = _max_abs_rel(video_pred, video_ref, ctx)
    var arl2 = _max_abs_rel(audio_pred, audio_ref, ctx)
    print("  VIDEO params  cos:", vcos, " relL2:", vrl2)
    print("  AUDIO params  cos:", acos, " relL2:", arl2)
    if vcos < 0.99999:
        raise Error(String("guider VIDEO FAIL: cos=") + String(vcos))
    if acos < 0.99999:
        raise Error(String("guider AUDIO FAIL: cos=") + String(acos))
    print("LTX-2 MultiModalGuider unit gate PASS")
