# serenitymojo/models/chroma/parity/chroma_block_device_parity.mojo
#
# HOST-vs-DEVICE parity gate for the device-resident Chroma (= Flux) DOUBLE +
# SINGLE LoRA block (models/chroma/chroma_block_device.mojo).
#
# The HOST block (models/chroma/chroma_block.mojo, = the torch-gated flux LoRA
# block) is the oracle. This gate generates deterministic pseudo-random inputs
# in-Mojo (weights, activations, mod vecs, LoRA A/B, cos/sin, upstream grads),
# runs BOTH the host block and the device block on BYTE-IDENTICAL inputs, and
# compares, per tensor:
#   forward img_out/txt_out (and single out)   : cos >= 0.999
#   backward d_x (d_img_x/d_txt_x, single d_x) : cos >= 0.999
#   every LoRA d_a/d_b slot                     : cos >= 0.999 AND rel-L2 <= 1e-3
# at REAL H=24, Dh=128, D=3072 with small N_IMG=8, N_TXT=6, S=14, Fmlp=64.
#
# Run (GPU; check `nvidia-smi` is idle first):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . -I /home/alex/MOJO-libs \
#     serenitymojo/models/chroma/parity/chroma_block_device_parity.mojo
# or build then run:
#   pixi run mojo build --optimization-level 2 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/chroma/parity/chroma_block_device_parity.mojo -o /tmp/chroma_dev_parity
#   /tmp/chroma_dev_parity

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.training.train_step import LoraAdapter

# HOST oracle (re-exported flux block)
from serenitymojo.models.chroma.chroma_block import (
    ChromaStreamWeights, ChromaDoubleBlockWeights, ChromaModVecs,
    ChromaSingleBlockWeights, ChromaSingleModVecs,
    ChromaStreamLora, ChromaDoubleBlockLora, ChromaSingleBlockLora,
    chroma_double_block_lora_forward, chroma_double_block_lora_backward,
    chroma_single_block_lora_forward, chroma_single_block_lora_backward,
    D_SQ, D_SK, D_SV, D_PROJ, D_MLP0, D_MLP2,
    S_SQ, S_SK, S_SV, S_PMLP, S_L2,
)

# DEVICE under test
from serenitymojo.models.chroma.chroma_block_device import (
    ModVecsDevice, SingleModVecsDevice,
    StreamLoraDevice, DoubleBlockLoraDevice, SingleBlockLoraDevice,
    modvecs_to_device, single_modvecs_to_device,
    double_block_lora_to_device, single_block_lora_to_device,
    chroma_double_block_lora_forward_device, chroma_double_block_lora_backward_device,
    chroma_single_block_lora_forward_device, chroma_single_block_lora_backward_device,
)


comptime H = 24
comptime Dh = 128
comptime D = H * Dh           # 3072
comptime N_IMG = 8
comptime N_TXT = 6
comptime S_D = N_IMG + N_TXT  # 14 (double joint seq)
comptime S_S = 14             # single seq
comptime FMLP = 64
comptime RANK = 4
comptime LSCALE = Float32(2.0)
comptime EPS = Float32(1e-06)


# ── deterministic pseudo-random (splitmix64) ─────────────────────────────────
def _next(mut s: UInt64) -> UInt64:
    s = s + 0x9E3779B97F4A7C15
    var z = s
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _randf(mut s: UInt64, r: Float32) -> Float32:
    var u = _next(s)
    var frac = Float32(u >> 40) / Float32(16777216.0)   # [0,1)
    return (frac * Float32(2.0) - Float32(1.0)) * r


def _rl(mut s: UInt64, n: Int, r: Float32) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(_randf(s, r))
    return o^


# q_norm/k_norm rms scales near 1.0
def _rl_near1(mut s: UInt64, n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(1.0) + _randf(s, Float32(0.1)))
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _adapter(mut s: UInt64, in_f: Int, out_f: Int) raises -> LoraAdapter:
    # NONZERO A and B so BOTH d_a and d_b are exercised (B=0 init would zero d_a).
    return LoraAdapter(
        _rl(s, RANK * in_f, Float32(0.05)), _rl(s, out_f * RANK, Float32(0.05)),
        RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _stream_weights(mut s: UInt64, ctx: DeviceContext) raises -> ChromaStreamWeights:
    return ChromaStreamWeights(
        _rl(s, 3 * D * D, Float32(0.04)), _rl(s, 3 * D, Float32(0.02)),
        _rl(s, D * D, Float32(0.04)), _rl(s, D, Float32(0.02)),
        _rl(s, FMLP * D, Float32(0.04)), _rl(s, FMLP, Float32(0.02)),
        _rl(s, D * FMLP, Float32(0.04)), _rl(s, D, Float32(0.02)),
        _rl_near1(s, Dh), _rl_near1(s, Dh),
        D, FMLP, Dh, ctx,
    )


def _mod(mut s: UInt64) raises -> ChromaModVecs:
    return ChromaModVecs(
        _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.5)),
        _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.5)),
    )


def _stream_lora(mut s: UInt64) raises -> ChromaStreamLora:
    return ChromaStreamLora(
        Optional[LoraAdapter](_adapter(s, D, D)),        # to_q
        Optional[LoraAdapter](_adapter(s, D, D)),        # to_k
        Optional[LoraAdapter](_adapter(s, D, D)),        # to_v
        Optional[LoraAdapter](_adapter(s, D, D)),        # proj
        Optional[LoraAdapter](_adapter(s, D, FMLP)),     # mlp0
        Optional[LoraAdapter](_adapter(s, FMLP, D)),     # mlp2
    )


# ── comparison helpers ───────────────────────────────────────────────────────
def _rel_l2(a: List[Float32], b: List[Float32]) -> Float64:
    var num: Float64 = 0.0
    var den: Float64 = 0.0
    for i in range(len(a)):
        var d = Float64(a[i]) - Float64(b[i])
        num += d * d
        den += Float64(b[i]) * Float64(b[i])
    if den == 0.0:
        return 0.0 if num == 0.0 else 1.0e30
    return sqrt(num / den)


def _ck(mut harness: ParityHarness, name: String, dev: Tensor, host: List[Float32],
        mut allok: Bool, ctx: DeviceContext) raises:
    var actual = dev.to_host(ctx)
    var r = harness.compare_host(actual, host)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs, "  n =", r.n,
          "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def _ck_lora(mut harness: ParityHarness, name: String, dev: Tensor, host: List[Float32],
             mut allok: Bool, ctx: DeviceContext) raises:
    var actual = dev.to_host(ctx)
    var r = harness.compare_host(actual, host)
    var rl2 = _rel_l2(actual, host)
    var ok = r.passed and (rl2 <= 1.0e-3)
    print("  cos(", name, ") =", r.cos, "  relL2 =", rl2, "  n =", r.n,
          "  ", "PASS" if ok else "FAIL")
    if not ok:
        allok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== chroma DEVICE-vs-HOST block parity (fwd+bwd+LoRA, bf16-native) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " S=", S_D, " FMLP=", FMLP, " RANK=", RANK)
    var harness = ParityHarness()
    var allok = True
    var s: UInt64 = 0x1234ABCD5678EF90

    # ════════════════════════ DOUBLE BLOCK ════════════════════════
    print("")
    print("################ DOUBLE BLOCK ################")
    var img = _rl(s, N_IMG * D, Float32(1.0))
    var txt = _rl(s, N_TXT * D, Float32(1.0))
    var iw = _stream_weights(s, ctx)
    var tw = _stream_weights(s, ctx)
    var w = ChromaDoubleBlockWeights(iw^, tw^)
    var im = _mod(s)
    var tm = _mod(s)
    var cos_h = _rl(s, S_D * H * (Dh // 2), Float32(1.0))
    var sin_h = _rl(s, S_D * H * (Dh // 2), Float32(1.0))
    var cos = Tensor.from_host(cos_h, [S_D * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [S_D * H, Dh // 2], STDtype.F32, ctx)
    var ilora = _stream_lora(s)
    var tlora = _stream_lora(s)
    var lora = ChromaDoubleBlockLora(ilora^, tlora^)
    var d_img = _rl(s, N_IMG * D, Float32(1.0))
    var d_txt = _rl(s, N_TXT * D, Float32(1.0))

    # device-side carriers (byte-identical to the host inputs)
    var img_x = ArcPointer[Tensor](Tensor.from_host(img.copy(), [N_IMG, D], STDtype.BF16, ctx))
    var txt_x = ArcPointer[Tensor](Tensor.from_host(txt.copy(), [N_TXT, D], STDtype.BF16, ctx))
    var imd = modvecs_to_device(im, D, ctx)
    var tmd = modvecs_to_device(tm, D, ctx)
    var lora_d = double_block_lora_to_device(lora, ctx)
    var d_io = ArcPointer[Tensor](Tensor.from_host(d_img.copy(), [N_IMG, D], STDtype.F32, ctx))
    var d_to = ArcPointer[Tensor](Tensor.from_host(d_txt.copy(), [N_TXT, D], STDtype.F32, ctx))

    # HOST forward + backward
    var hfwd = chroma_double_block_lora_forward[H, Dh, N_IMG, N_TXT, S_D](
        img.copy(), txt.copy(), w, im, tm, lora, cos, sin, D, FMLP, EPS, ctx)
    var hbwd = chroma_double_block_lora_backward[H, Dh, N_IMG, N_TXT, S_D](
        d_img.copy(), d_txt.copy(), w, im, tm, lora, hfwd.saved, cos, sin, D, FMLP, EPS, ctx)

    # DEVICE forward + backward — MATH arm (FLASH=False) = the bit-oracle
    var dfwd = chroma_double_block_lora_forward_device[H, Dh, N_IMG, N_TXT, S_D, False](
        img_x, txt_x, w, imd, tmd, lora_d, cos, sin, D, FMLP, EPS, ctx)
    var dbwd = chroma_double_block_lora_backward_device[H, Dh, N_IMG, N_TXT, S_D, False](
        d_io, d_to, w, imd, tmd, lora_d, dfwd.saved, cos, sin, D, FMLP, EPS, ctx)

    print("---- forward outputs (device vs host) ----")
    _ck(harness, "img_out", dfwd.img_out[], hfwd.img_out, allok, ctx)
    _ck(harness, "txt_out", dfwd.txt_out[], hfwd.txt_out, allok, ctx)

    print("---- input grads d_x (device vs host) ----")
    _ck(harness, "d_img_x", dbwd.img.d_x[], hbwd.base.img.d_x, allok, ctx)
    _ck(harness, "d_txt_x", dbwd.txt.d_x[], hbwd.base.txt.d_x, allok, ctx)

    print("---- IMG LoRA d_A/d_B (device vs host) ----")
    _ck_lora(harness, "img loA to_q", dbwd.img.d_a[D_SQ].value()[], hbwd.lora.img.d_a[D_SQ], allok, ctx)
    _ck_lora(harness, "img loB to_q", dbwd.img.d_b[D_SQ].value()[], hbwd.lora.img.d_b[D_SQ], allok, ctx)
    _ck_lora(harness, "img loA to_k", dbwd.img.d_a[D_SK].value()[], hbwd.lora.img.d_a[D_SK], allok, ctx)
    _ck_lora(harness, "img loB to_k", dbwd.img.d_b[D_SK].value()[], hbwd.lora.img.d_b[D_SK], allok, ctx)
    _ck_lora(harness, "img loA to_v", dbwd.img.d_a[D_SV].value()[], hbwd.lora.img.d_a[D_SV], allok, ctx)
    _ck_lora(harness, "img loB to_v", dbwd.img.d_b[D_SV].value()[], hbwd.lora.img.d_b[D_SV], allok, ctx)
    _ck_lora(harness, "img loA proj", dbwd.img.d_a[D_PROJ].value()[], hbwd.lora.img.d_a[D_PROJ], allok, ctx)
    _ck_lora(harness, "img loB proj", dbwd.img.d_b[D_PROJ].value()[], hbwd.lora.img.d_b[D_PROJ], allok, ctx)
    _ck_lora(harness, "img loA mlp0", dbwd.img.d_a[D_MLP0].value()[], hbwd.lora.img.d_a[D_MLP0], allok, ctx)
    _ck_lora(harness, "img loB mlp0", dbwd.img.d_b[D_MLP0].value()[], hbwd.lora.img.d_b[D_MLP0], allok, ctx)
    _ck_lora(harness, "img loA mlp2", dbwd.img.d_a[D_MLP2].value()[], hbwd.lora.img.d_a[D_MLP2], allok, ctx)
    _ck_lora(harness, "img loB mlp2", dbwd.img.d_b[D_MLP2].value()[], hbwd.lora.img.d_b[D_MLP2], allok, ctx)

    print("---- TXT LoRA d_A/d_B (device vs host) ----")
    _ck_lora(harness, "txt loA to_q", dbwd.txt.d_a[D_SQ].value()[], hbwd.lora.txt.d_a[D_SQ], allok, ctx)
    _ck_lora(harness, "txt loB to_q", dbwd.txt.d_b[D_SQ].value()[], hbwd.lora.txt.d_b[D_SQ], allok, ctx)
    _ck_lora(harness, "txt loA to_k", dbwd.txt.d_a[D_SK].value()[], hbwd.lora.txt.d_a[D_SK], allok, ctx)
    _ck_lora(harness, "txt loB to_k", dbwd.txt.d_b[D_SK].value()[], hbwd.lora.txt.d_b[D_SK], allok, ctx)
    _ck_lora(harness, "txt loA to_v", dbwd.txt.d_a[D_SV].value()[], hbwd.lora.txt.d_a[D_SV], allok, ctx)
    _ck_lora(harness, "txt loB to_v", dbwd.txt.d_b[D_SV].value()[], hbwd.lora.txt.d_b[D_SV], allok, ctx)
    _ck_lora(harness, "txt loA proj", dbwd.txt.d_a[D_PROJ].value()[], hbwd.lora.txt.d_a[D_PROJ], allok, ctx)
    _ck_lora(harness, "txt loB proj", dbwd.txt.d_b[D_PROJ].value()[], hbwd.lora.txt.d_b[D_PROJ], allok, ctx)
    _ck_lora(harness, "txt loA mlp0", dbwd.txt.d_a[D_MLP0].value()[], hbwd.lora.txt.d_a[D_MLP0], allok, ctx)
    _ck_lora(harness, "txt loB mlp0", dbwd.txt.d_b[D_MLP0].value()[], hbwd.lora.txt.d_b[D_MLP0], allok, ctx)
    _ck_lora(harness, "txt loA mlp2", dbwd.txt.d_a[D_MLP2].value()[], hbwd.lora.txt.d_a[D_MLP2], allok, ctx)
    _ck_lora(harness, "txt loB mlp2", dbwd.txt.d_b[D_MLP2].value()[], hbwd.lora.txt.d_b[D_MLP2], allok, ctx)

    # ════════════════════════ SINGLE BLOCK ════════════════════════
    print("")
    print("################ SINGLE BLOCK ################")
    var sx = _rl(s, S_S * D, Float32(1.0))
    var sw = ChromaSingleBlockWeights(
        _rl(s, (3 * D + FMLP) * D, Float32(0.04)), _rl(s, 3 * D + FMLP, Float32(0.02)),
        _rl(s, D * (D + FMLP), Float32(0.04)), _rl(s, D, Float32(0.02)),
        _rl_near1(s, Dh), _rl_near1(s, Dh),
        D, FMLP, Dh, ctx,
    )
    var smv = ChromaSingleModVecs(
        _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.5)),
    )
    var scos_h = _rl(s, S_S * H * (Dh // 2), Float32(1.0))
    var ssin_h = _rl(s, S_S * H * (Dh // 2), Float32(1.0))
    var scos = Tensor.from_host(scos_h, [S_S * H, Dh // 2], STDtype.F32, ctx)
    var ssin = Tensor.from_host(ssin_h, [S_S * H, Dh // 2], STDtype.F32, ctx)
    var slora = ChromaSingleBlockLora(
        Optional[LoraAdapter](_adapter(s, D, D)),          # to_q
        Optional[LoraAdapter](_adapter(s, D, D)),          # to_k
        Optional[LoraAdapter](_adapter(s, D, D)),          # to_v
        Optional[LoraAdapter](_adapter(s, D, FMLP)),       # proj_mlp
        Optional[LoraAdapter](_adapter(s, D + FMLP, D)),   # linear2
    )
    var s_d_out = _rl(s, S_S * D, Float32(1.0))

    var sx_d = ArcPointer[Tensor](Tensor.from_host(sx.copy(), [S_S, D], STDtype.BF16, ctx))
    var smvd = single_modvecs_to_device(smv, D, ctx)
    var slora_d = single_block_lora_to_device(slora, ctx)
    var s_d_out_d = ArcPointer[Tensor](Tensor.from_host(s_d_out.copy(), [S_S, D], STDtype.F32, ctx))

    var hsfwd = chroma_single_block_lora_forward[H, Dh, S_S](
        sx.copy(), sw, smv, slora, scos, ssin, D, FMLP, EPS, ctx)
    var hsbwd = chroma_single_block_lora_backward[H, Dh, S_S](
        s_d_out.copy(), sw, smv, slora, hsfwd.saved, scos, ssin, D, FMLP, EPS, ctx)

    var dsfwd = chroma_single_block_lora_forward_device[H, Dh, S_S, False](
        sx_d, sw, smvd, slora_d, scos, ssin, D, FMLP, EPS, ctx)
    var dsbwd = chroma_single_block_lora_backward_device[H, Dh, S_S, False](
        s_d_out_d, sw, smvd, slora_d, dsfwd.saved, scos, ssin, D, FMLP, EPS, ctx)

    print("---- forward output (device vs host) ----")
    _ck(harness, "s_out", dsfwd.out[], hsfwd.out, allok, ctx)
    print("---- input grad d_x (device vs host) ----")
    _ck(harness, "s_d_x", dsbwd.d_x[], hsbwd.base.d_x, allok, ctx)
    print("---- LoRA d_A/d_B (device vs host) ----")
    _ck_lora(harness, "s loA to_q", dsbwd.d_a[S_SQ].value()[], hsbwd.lora.d_a[S_SQ], allok, ctx)
    _ck_lora(harness, "s loB to_q", dsbwd.d_b[S_SQ].value()[], hsbwd.lora.d_b[S_SQ], allok, ctx)
    _ck_lora(harness, "s loA to_k", dsbwd.d_a[S_SK].value()[], hsbwd.lora.d_a[S_SK], allok, ctx)
    _ck_lora(harness, "s loB to_k", dsbwd.d_b[S_SK].value()[], hsbwd.lora.d_b[S_SK], allok, ctx)
    _ck_lora(harness, "s loA to_v", dsbwd.d_a[S_SV].value()[], hsbwd.lora.d_a[S_SV], allok, ctx)
    _ck_lora(harness, "s loB to_v", dsbwd.d_b[S_SV].value()[], hsbwd.lora.d_b[S_SV], allok, ctx)
    _ck_lora(harness, "s loA pmlp", dsbwd.d_a[S_PMLP].value()[], hsbwd.lora.d_a[S_PMLP], allok, ctx)
    _ck_lora(harness, "s loB pmlp", dsbwd.d_b[S_PMLP].value()[], hsbwd.lora.d_b[S_PMLP], allok, ctx)
    _ck_lora(harness, "s loA lin2", dsbwd.d_a[S_L2].value()[], hsbwd.lora.d_a[S_L2], allok, ctx)
    _ck_lora(harness, "s loB lin2", dsbwd.d_b[S_L2].value()[], hsbwd.lora.d_b[S_L2], allok, ctx)

    # ════════════════════ FLASH ARM (FLASH=True vs HOST oracle) ════════════════════
    # Same byte-identical inputs; cuDNN padmask flash fwd+bwd vs the HOST math
    # oracle. VALUE-CLASS (bf16 flash vs bf16 math): out/d_x cos>=0.999, one LoRA
    # dB cos>=0.995 (looser — the LoRA grad rides on the flash-perturbed d_att).
    print("")
    print("################ FLASH ARM (FLASH=True vs HOST) ################")
    var harness995 = ParityHarness(0.995)

    var dfwd_f = chroma_double_block_lora_forward_device[H, Dh, N_IMG, N_TXT, S_D, True](
        img_x, txt_x, w, imd, tmd, lora_d, cos, sin, D, FMLP, EPS, ctx)
    var dbwd_f = chroma_double_block_lora_backward_device[H, Dh, N_IMG, N_TXT, S_D, True](
        d_io, d_to, w, imd, tmd, lora_d, dfwd_f.saved, cos, sin, D, FMLP, EPS, ctx)
    print("---- DOUBLE flash out / d_x (vs host, cos>=0.999) ----")
    _ck(harness, "FLASH img_out", dfwd_f.img_out[], hfwd.img_out, allok, ctx)
    _ck(harness, "FLASH txt_out", dfwd_f.txt_out[], hfwd.txt_out, allok, ctx)
    _ck(harness, "FLASH d_img_x", dbwd_f.img.d_x[], hbwd.base.img.d_x, allok, ctx)
    _ck(harness, "FLASH d_txt_x", dbwd_f.txt.d_x[], hbwd.base.txt.d_x, allok, ctx)
    print("---- DOUBLE flash LoRA dB (vs host, cos>=0.995) ----")
    _ck(harness995, "FLASH img loB to_q", dbwd_f.img.d_b[D_SQ].value()[], hbwd.lora.img.d_b[D_SQ], allok, ctx)

    var dsfwd_f = chroma_single_block_lora_forward_device[H, Dh, S_S, True](
        sx_d, sw, smvd, slora_d, scos, ssin, D, FMLP, EPS, ctx)
    var dsbwd_f = chroma_single_block_lora_backward_device[H, Dh, S_S, True](
        s_d_out_d, sw, smvd, slora_d, dsfwd_f.saved, scos, ssin, D, FMLP, EPS, ctx)
    print("---- SINGLE flash out / d_x (vs host, cos>=0.999) ----")
    _ck(harness, "FLASH s_out", dsfwd_f.out[], hsfwd.out, allok, ctx)
    _ck(harness, "FLASH s_d_x", dsbwd_f.d_x[], hsbwd.base.d_x, allok, ctx)
    print("---- SINGLE flash LoRA dB (vs host, cos>=0.995) ----")
    _ck(harness995, "FLASH s loB to_q", dsbwd_f.d_b[S_SQ].value()[], hsbwd.lora.d_b[S_SQ], allok, ctx)

    print("")
    if allok:
        print("VERDICT: PASS — device block matches host (math bit bars + flash cos bars)")
    else:
        print("VERDICT: FAIL — at least one tensor diverged (see FAIL lines)")
