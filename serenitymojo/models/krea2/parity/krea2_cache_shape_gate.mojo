# krea2_cache_shape_gate.mojo — CACHE-SHAPE GATE for the Krea-2 training data path.
#
# Proves the krea2 cache format (krea2_prepare_cache.mojo) + reader
# (krea2_cache_reader.mojo) produce per-sample inputs whose shapes EXACTLY match
# what the krea2 DiT forward (krea2_forward) and the stack LoRA forward consume —
# so Phase 4 (the trainer) can feed them with ZERO glue. This gate is encoder-FREE
# (no VAE / Qwen3-VL load): it writes a tiny SYNTHETIC cache (the exact tensor names
# + shapes the real prepare writes), reads it back through KreaTrainCache, and
# asserts every materialised shape against the forward's documented contract.
#
# WHAT IT ASSERTS (the krea2_forward signature, krea2_dit.mojo:1304-1317):
#   img     [1, imglen, 64]      imglen = (LH/2)*(LW/2)         (the `img` input)
#   context [1, LT, 12, 2560]                                   (the `context` input)
#   pos     [1, LT+imglen, 3]    == LFULL                       (the `pos` input)
#   clean   [1, 16, LH, LW]      ai-toolkit batch.latents (trainer noises this)
#   text_len == LT
# Plus the patchify/pos helpers reproduce the inference pipeline's _patchify/_build_pos.
#
# This is the GATE the lead re-runs. The REAL encode smoke (VAE + Qwen3-VL) is
# krea2_prepare_cache.mojo itself on real images — GPU-heavy, run by the orchestrator.
#
# Run (CPU/GPU-light, builds clean):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/krea2/parity/krea2_cache_shape_gate.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.krea2.krea2_cache_reader import (
    KreaTrainCache, krea2_patchify, krea2_build_pos,
)

comptime TArc = ArcPointer[Tensor]

# Small synthetic geometry: 64x64 latent (512px image bucket) keeps imglen=32*32=1024
# but tiny enough to write/read instantly. Use a SMALL latent here for speed: LH=LW=16
# -> imglen = 8*8 = 64. Two samples with DIFFERENT LT (prompt-dependent, as in real
# inference) to prove the reader handles per-sample LT.
comptime LH = 16
comptime LW = 16
comptime IMGLEN = (LH // 2) * (LW // 2)   # 64
comptime LT0 = 19                          # sample 0 caption length
comptime LT1 = 7                           # sample 1 (different LT)
comptime LTU = 5                           # uncond (empty caption) length
comptime CACHE = "/tmp/claude-1000/krea2_cache_shape_gate.safetensors"


def _fill(n: Int, c: Float32) -> List[Float32]:
    var o = List[Float32]()
    var x = Float64(0.0)
    for _ in range(n):
        x += 0.6180339887
        var frac = x - Float64(Int(x))
        o.append(Float32(frac * 2.0 - 1.0) * c)
    return o^


def _clean(ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(_fill(16 * LH * LW, 0.7), [1, 16, LH, LW], STDtype.F32, ctx)


def _context(lt: Int, ctx: DeviceContext) raises -> Tensor:
    var t = Tensor.from_host(_fill(lt * 12 * 2560, 0.3), [1, lt, 12, 2560], STDtype.F32, ctx)
    return cast_tensor(t, STDtype.BF16, ctx)


def _tl(lt: Int, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    h.append(Float32(lt))
    return Tensor.from_host(h^, [1], STDtype.F32, ctx)


def _expect(name: String, got: List[Int], want: List[Int]) raises:
    var ok = len(got) == len(want)
    if ok:
        for i in range(len(got)):
            if got[i] != want[i]:
                ok = False
    var gs = String("[")
    for i in range(len(got)):
        gs += String(got[i])
        if i + 1 < len(got):
            gs += ","
    gs += "]"
    var ws = String("[")
    for i in range(len(want)):
        ws += String(want[i])
        if i + 1 < len(want):
            ws += ","
    ws += "]"
    if ok:
        print("  OK  ", name, " = ", gs)
    else:
        print("  FAIL", name, " = ", gs, " expected ", ws)
        raise Error(String("shape mismatch: ") + name)


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_cache_shape_gate ====")
    print("LH=", LH, " LW=", LW, " IMGLEN=", IMGLEN, " (LT0=", LT0, " LT1=", LT1, ")")

    # ── 1) write a synthetic cache with the EXACT names the real prepare writes ──
    var names = List[String]()
    var tensors = List[TArc]()
    names.append(String("clean.0")); tensors.append(TArc(_clean(ctx)))
    names.append(String("context.0")); tensors.append(TArc(_context(LT0, ctx)))
    names.append(String("text_len.0")); tensors.append(TArc(_tl(LT0, ctx)))
    names.append(String("clean.1")); tensors.append(TArc(_clean(ctx)))
    names.append(String("context.1")); tensors.append(TArc(_context(LT1, ctx)))
    names.append(String("text_len.1")); tensors.append(TArc(_tl(LT1, ctx)))
    # uncond (caption dropout) context.
    names.append(String("context_uncond")); tensors.append(TArc(_context(LTU, ctx)))
    names.append(String("text_len_uncond")); tensors.append(TArc(_tl(LTU, ctx)))
    save_safetensors(names, tensors, String(CACHE), ctx)
    print("wrote synthetic cache ->", CACHE)

    # ── 2) read it back; assert the reader is well-formed ──
    var cache = KreaTrainCache.open(String(CACHE))
    if cache.len() != 2:
        raise Error(String("cache.len()=") + String(cache.len()) + " expected 2")
    print("cache.len() =", cache.len(), " OK")

    # ── 3) sample 0: assert every shape against the krea2_forward contract ──
    print("sample 0 (LT=", LT0, "):")
    var s0 = cache.sample[LH, LW](0, ctx)
    if s0.text_len != LT0:
        raise Error(String("s0.text_len=") + String(s0.text_len) + " expected " + String(LT0))
    print("  OK   text_len =", s0.text_len)
    _expect(String("clean  "), s0.clean[].shape(), [1, 16, LH, LW])
    _expect(String("img    "), s0.img[].shape(), [1, IMGLEN, 64])
    _expect(String("context"), s0.context[].shape(), [1, LT0, 12, 2560])
    _expect(String("pos    "), s0.pos[].shape(), [1, LT0 + IMGLEN, 3])

    # ── 4) sample 1: DIFFERENT LT -> pos/context track it ──
    print("sample 1 (LT=", LT1, "):")
    var s1 = cache.sample[LH, LW](1, ctx)
    _expect(String("context"), s1.context[].shape(), [1, LT1, 12, 2560])
    _expect(String("pos    "), s1.pos[].shape(), [1, LT1 + IMGLEN, 3])
    _expect(String("img    "), s1.img[].shape(), [1, IMGLEN, 64])

    # ── 5) uncond (caption dropout) conditioning ──
    print("uncond (LT=", LTU, "):")
    var su = cache.uncond[LH, LW](ctx)
    if su.text_len != LTU:
        raise Error(String("uncond text_len=") + String(su.text_len) + " expected " + String(LTU))
    _expect(String("context"), su.context[].shape(), [1, LTU, 12, 2560])
    _expect(String("pos    "), su.pos[].shape(), [1, LTU + IMGLEN, 3])

    # ── 6) the patchify / pos helpers reproduce the inference-pipeline order ──
    # pos: txt rows must be all-zero; first img row must be (0, 0, 0); the row for
    # img token (gh-1, gw-1) must be (0, gh-1, gw-1) in (gh,gw) row-major.
    comptime gh = LH // 2
    comptime gw = LW // 2
    var pos_h = s0.pos[].to_host(ctx)
    var ok_txt = True
    for i in range(LT0 * 3):
        if pos_h[i] != Float32(0.0):
            ok_txt = False
    var base_img = LT0 * 3
    var ok_first = (pos_h[base_img + 0] == 0.0 and pos_h[base_img + 1] == 0.0
                    and pos_h[base_img + 2] == 0.0)
    var last = base_img + (IMGLEN - 1) * 3
    var ok_last = (pos_h[last + 0] == 0.0
                   and pos_h[last + 1] == Float32(gh - 1)
                   and pos_h[last + 2] == Float32(gw - 1))
    if ok_txt and ok_first and ok_last:
        print("  OK   pos grid: txt-zeros + img (gh,gw) row-major")
    else:
        print("  FAIL pos grid: txt_zeros=", ok_txt, " first=", ok_first, " last=", ok_last)
        raise Error("pos grid order mismatch")

    # patchify a standalone latent (the trainer re-patchifies the NOISED latent this
    # way): a [1,16,LH,LW] -> [1,IMGLEN,64].
    var lat = _clean(ctx)
    var img2 = krea2_patchify[LH, LW](lat, ctx)
    _expect(String("patchify"), img2.shape(), [1, IMGLEN, 64])

    print("")
    print("VERDICT: PASS — cache format + reader deliver the exact krea2_forward inputs (img/context/pos/clean/text_len) with zero glue.")
