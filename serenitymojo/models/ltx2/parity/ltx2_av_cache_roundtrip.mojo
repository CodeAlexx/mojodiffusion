# serenitymojo/models/ltx2/parity/ltx2_av_cache_roundtrip.mojo
#
# LTX-2 P6.2 AV tri-pair cache-reader ROUND-TRIP gate. Reads the synthetic
# tri-pair written by scripts/ltx2_av_smoke_cache.py and asserts the reader
# (training/ltx2/av_cache.mojo) pairs + loads every stream by the musubi basename
# route: video latent + audio_latents [C,T,mel] (+ audio_lengths) + video/audio
# prompt embeds. Also checks the [C,T,mel] -> [1,S_A=T,patch_in=C*mel] reshape the
# audio patchify consumes.
#
# Prep: /home/alex/serenityflow-v2/.venv/bin/python scripts/ltx2_av_smoke_cache.py
# Run:  rm -f serenitymojo.mojopkg; pixi run mojo build -O2 -I . -Xlinker -lm \
#   -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_cache_roundtrip.mojo \
#   -o /tmp/ltx2_av_cache_roundtrip && /tmp/ltx2_av_cache_roundtrip

from std.gpu.host import DeviceContext
from std.collections import List

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.training.ltx2.av_cache import (
    audio_cache_path, te_cache_path, discover_audio_latents_key, discover_audio_lengths_key,
)

comptime ROOT = "/tmp/ltx2_av_smoke/cache"


def _expect(tag: String, cond: Bool, detail: String) -> Int:
    print("  ", "PASS" if cond else "FAIL", tag, detail)
    return 0 if cond else 1


def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^


def _shape_str(t: Tensor) -> String:
    var s = t.shape().copy()
    var out = String("[")
    for i in range(len(s)):
        out += String(s[i])
        if i + 1 < len(s):
            out += ","
    return out + "]"


def main() raises:
    var ctx = DeviceContext()
    print("=== LTX-2 P6.2 AV tri-pair cache round-trip gate ===")
    var video_path = String(ROOT) + "/sample0_ltx2.safetensors"
    var apath = audio_cache_path(video_path)
    var tpath = te_cache_path(video_path)
    print("  video:", video_path)
    print("  audio:", apath)
    print("  te   :", tpath)

    var fails = 0
    # basename route.
    fails += _expect(String("audio path = _ltx2_audio sibling"),
                     apath == String(ROOT) + "/sample0_ltx2_audio.safetensors", apath)
    fails += _expect(String("te path = _ltx2_te sibling"),
                     tpath == String(ROOT) + "/sample0_ltx2_te.safetensors", tpath)

    # video latent.
    var vst = ShardedSafeTensors.open(video_path)
    var vlat = Tensor.from_view_as_f32(vst.tensor_view("latents_4x9x16_bfloat16"), ctx)
    var vs = vlat.shape().copy()
    fails += _expect(String("video latent [128,4,9,16]"),
                     len(vs) == 4 and vs[0] == 128 and vs[1] == 4 and vs[2] == 9 and vs[3] == 16,
                     _shape_str(vlat))

    # audio latent [C,T,mel] via key discovery.
    var ast = ShardedSafeTensors.open(apath)
    var akey = discover_audio_latents_key(ast)
    fails += _expect(String("discover audio_latents key"),
                     akey == String("audio_latents_16x16x8_bfloat16"), akey)
    var alat = Tensor.from_view_as_f32(ast.tensor_view(akey), ctx)
    var ash = alat.shape().copy()
    var C = ash[0]; var T = ash[1]; var MEL = ash[2]
    fails += _expect(String("audio latent [C=8,T=16,mel=16]"),
                     len(ash) == 3 and C == 8 and T == 16 and MEL == 16, _shape_str(alat))
    var alen_key = discover_audio_lengths_key(ast)
    fails += _expect(String("discover audio_lengths key"),
                     alen_key == String("audio_lengths_int32"), alen_key)

    # [C,T,mel] -> [1, S_A=T, patch_in=C*mel] reshape (audio patchify input).
    var patch_in = C * MEL
    var host = alat.to_host(ctx)
    var out = List[Float32]()
    out.resize(T * patch_in, Float32(0.0))
    for c in range(C):
        for t in range(T):
            for m in range(MEL):
                out[t * patch_in + c * MEL + m] = host[(c * T + t) * MEL + m]
    var stream = Tensor.from_host(out, _sh3(1, T, patch_in), STDtype.F32, ctx)
    var ss = stream.shape().copy()
    fails += _expect(String("audio stream reshape [1,S_A=16,patch_in=128]"),
                     ss[0] == 1 and ss[1] == 16 and ss[2] == 128, _shape_str(stream))

    # tri-pair text: video + audio prompt embeds.
    var tst = ShardedSafeTensors.open(tpath)
    var vemb = Tensor.from_view_as_f32(tst.tensor_view("video_prompt_embeds_bfloat16"), ctx)
    var aemb = Tensor.from_view_as_f32(tst.tensor_view("audio_prompt_embeds_bfloat16"), ctx)
    var ve = vemb.shape().copy(); var ae = aemb.shape().copy()
    fails += _expect(String("video_prompt_embeds [1024,4096]"),
                     ve[0] == 1024 and ve[1] == 4096, _shape_str(vemb))
    fails += _expect(String("audio_prompt_embeds [1024,2048]"),
                     ae[0] == 1024 and ae[1] == 2048, _shape_str(aemb))

    print("")
    if fails == 0:
        print("LTX2 P6.2 AV tri-pair cache round-trip PASS: basename pairing + key discovery",
              "+ [C,T,mel]->[1,S_A,patch_in] reshape all match")
    else:
        raise Error(String("LTX2 P6.2 AV cache round-trip FAIL: ") + String(fails))
