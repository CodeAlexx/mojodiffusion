# serenitymojo/models/flux/parity/flux_block_device_parity.mojo
#
# HOST-vs-DEVICE parity gate for the device-resident FLUX DOUBLE + SINGLE LoRA
# block backward that ALSO emits modulation-vector grads
# (models/flux/flux_block_device.mojo).
#
# The HOST flux block (models/flux/lora_block.mojo: double_block_lora_forward/
# backward, single_..) is the oracle. The DEVICE forward is chroma's
# (chroma_block_device — byte-for-byte the flux block, already gated); the DEVICE
# backward under test is flux's (flux_block_device) which additionally returns the
# per-block [6D]/[3D] modulation-vector grads. This gate proves, on byte-identical
# in-Mojo pseudo-random inputs at REAL H=24,Dh=128,D=3072 (small N/S):
#   forward img_out/txt_out/s_out : cos >= 0.999
#   backward d_x                  : cos >= 0.999
#   every LoRA d_a/d_b slot        : cos >= 0.999 AND rel-L2 <= 1e-3
#   MODULATION grad flats (NEW)    : cos >= 0.999 AND rel-L2 <= 1e-3
#     device d_modvec6/d_modvec3 vs host _modvec6/_single_modvec3
#
# Run (GPU; check `nvidia-smi` is idle first):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/flux/parity/flux_block_device_parity.mojo -o /tmp/flux_dev_parity
#   /tmp/flux_dev_parity

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.training.train_step import LoraAdapter

# HOST oracle (flux block) + flux weight/modvec/lora types
from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, ModVecs,
    SingleBlockWeights, SingleModVecs,
)
from serenitymojo.models.flux.lora_block import (
    StreamLora, DoubleBlockLora, SingleBlockLora,
    double_block_lora_forward, double_block_lora_backward,
    single_block_lora_forward, single_block_lora_backward,
    D_SQ, D_SK, D_SV, D_PROJ, D_MLP0, D_MLP2,
    S_SQ, S_SK, S_SV, S_PMLP, S_L2,
)
from serenitymojo.models.flux.flux_stack import _modvec6, _single_modvec3

# DEVICE forward (chroma, reused) + converters
from serenitymojo.models.chroma.chroma_block_device import (
    modvecs_to_device, single_modvecs_to_device,
    double_block_lora_to_device, single_block_lora_to_device,
    chroma_double_block_lora_forward_device, chroma_single_block_lora_forward_device,
)
# DEVICE backward under test (flux, emits mod grads)
from serenitymojo.models.flux.flux_block_device import (
    flux_double_block_lora_backward_device, flux_single_block_lora_backward_device,
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
    return LoraAdapter(
        _rl(s, RANK * in_f, Float32(0.05)), _rl(s, out_f * RANK, Float32(0.05)),
        RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _stream_weights(mut s: UInt64, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _rl(s, 3 * D * D, Float32(0.04)), _rl(s, 3 * D, Float32(0.02)),
        _rl(s, D * D, Float32(0.04)), _rl(s, D, Float32(0.02)),
        _rl(s, FMLP * D, Float32(0.04)), _rl(s, FMLP, Float32(0.02)),
        _rl(s, D * FMLP, Float32(0.04)), _rl(s, D, Float32(0.02)),
        _rl_near1(s, Dh), _rl_near1(s, Dh),
        D, FMLP, Dh, ctx,
    )


def _mod(mut s: UInt64) raises -> ModVecs:
    return ModVecs(
        _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.5)),
        _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.5)),
    )


def _stream_lora(mut s: UInt64) raises -> StreamLora:
    return StreamLora(
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


# host-list vs host-list (mod grads are already host [6D]/[3D] on both sides)
def _ck_lists(mut harness: ParityHarness, name: String, dev: List[Float32], host: List[Float32],
              mut allok: Bool) raises:
    var r = harness.compare_host(dev, host)
    var rl2 = _rel_l2(dev, host)
    var ok = r.passed and (rl2 <= 1.0e-3)
    print("  cos(", name, ") =", r.cos, "  relL2 =", rl2, "  n =", r.n,
          "  ", "PASS" if ok else "FAIL")
    if not ok:
        allok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== flux DEVICE-vs-HOST block parity (fwd+bwd+LoRA+MODGRADS) ====")
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
    var w = DoubleBlockWeights(iw^, tw^)
    var im = _mod(s)
    var tm = _mod(s)
    var cos_h = _rl(s, S_D * H * (Dh // 2), Float32(1.0))
    var sin_h = _rl(s, S_D * H * (Dh // 2), Float32(1.0))
    var cos = Tensor.from_host(cos_h, [S_D * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [S_D * H, Dh // 2], STDtype.F32, ctx)
    var ilora = _stream_lora(s)
    var tlora = _stream_lora(s)
    var lora = DoubleBlockLora(ilora^, tlora^)
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
    var hfwd = double_block_lora_forward[H, Dh, N_IMG, N_TXT, S_D](
        img.copy(), txt.copy(), w, im, tm, lora, cos, sin, D, FMLP, EPS, ctx)
    var hbwd = double_block_lora_backward[H, Dh, N_IMG, N_TXT, S_D](
        d_img.copy(), d_txt.copy(), w, im, tm, lora, hfwd.saved, cos, sin, D, FMLP, EPS, ctx)

    # DEVICE forward (chroma) + backward (flux, mod grads) — MATH arm (FLASH=False) = bit-oracle
    var dfwd = chroma_double_block_lora_forward_device[H, Dh, N_IMG, N_TXT, S_D, False](
        img_x, txt_x, w, imd, tmd, lora_d, cos, sin, D, FMLP, EPS, ctx)
    var dbwd = flux_double_block_lora_backward_device[H, Dh, N_IMG, N_TXT, S_D, False](
        d_io, d_to, w, imd, tmd, lora_d, dfwd.saved, cos, sin, D, FMLP, EPS, ctx)

    print("---- forward outputs (device vs host) ----")
    _ck(harness, "img_out", dfwd.img_out[], hfwd.img_out, allok, ctx)
    _ck(harness, "txt_out", dfwd.txt_out[], hfwd.txt_out, allok, ctx)

    print("---- input grads d_x (device vs host) ----")
    _ck(harness, "d_img_x", dbwd.img.d_x[], hbwd.base.img.d_x, allok, ctx)
    _ck(harness, "d_txt_x", dbwd.txt.d_x[], hbwd.base.txt.d_x, allok, ctx)

    print("---- MODULATION grads [6D] (device d_modvec6 vs host _modvec6) ----")
    _ck_lists(harness, "img d_modvec6", dbwd.img.d_modvec6, _modvec6(hbwd.base.img), allok)
    _ck_lists(harness, "txt d_modvec6", dbwd.txt.d_modvec6, _modvec6(hbwd.base.txt), allok)

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
    var sw = SingleBlockWeights(
        _rl(s, (3 * D + FMLP) * D, Float32(0.04)), _rl(s, 3 * D + FMLP, Float32(0.02)),
        _rl(s, D * (D + FMLP), Float32(0.04)), _rl(s, D, Float32(0.02)),
        _rl_near1(s, Dh), _rl_near1(s, Dh),
        D, FMLP, Dh, ctx,
    )
    var smv = SingleModVecs(
        _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.2)), _rl(s, D, Float32(0.5)),
    )
    var scos_h = _rl(s, S_S * H * (Dh // 2), Float32(1.0))
    var ssin_h = _rl(s, S_S * H * (Dh // 2), Float32(1.0))
    var scos = Tensor.from_host(scos_h, [S_S * H, Dh // 2], STDtype.F32, ctx)
    var ssin = Tensor.from_host(ssin_h, [S_S * H, Dh // 2], STDtype.F32, ctx)
    var slora = SingleBlockLora(
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

    var hsfwd = single_block_lora_forward[H, Dh, S_S](
        sx.copy(), sw, smv, slora, scos, ssin, D, FMLP, EPS, ctx)
    var hsbwd = single_block_lora_backward[H, Dh, S_S](
        s_d_out.copy(), sw, smv, slora, hsfwd.saved, scos, ssin, D, FMLP, EPS, ctx)

    var dsfwd = chroma_single_block_lora_forward_device[H, Dh, S_S, False](
        sx_d, sw, smvd, slora_d, scos, ssin, D, FMLP, EPS, ctx)
    var dsbwd = flux_single_block_lora_backward_device[H, Dh, S_S, False](
        s_d_out_d, sw, smvd, slora_d, dsfwd.saved, scos, ssin, D, FMLP, EPS, ctx)

    print("---- forward output (device vs host) ----")
    _ck(harness, "s_out", dsfwd.out[], hsfwd.out, allok, ctx)
    print("---- input grad d_x (device vs host) ----")
    _ck(harness, "s_d_x", dsbwd.d_x[], hsbwd.base.d_x, allok, ctx)
    print("---- MODULATION grads [3D] (device d_modvec3 vs host _single_modvec3) ----")
    _ck_lists(harness, "s d_modvec3", dsbwd.d_modvec3, _single_modvec3(hsbwd.base), allok)
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
    # Same byte-identical inputs; cuDNN padmask flash fwd(chroma)+bwd(flux) vs the
    # HOST math oracle. VALUE-CLASS (bf16 flash vs bf16 math): out/d_x cos>=0.999,
    # one LoRA dB cos>=0.995. (Mod grads unchanged from the math arm — not re-checked.)
    print("")
    print("################ FLASH ARM (FLASH=True vs HOST) ################")
    var harness995 = ParityHarness(0.995)

    var dfwd_f = chroma_double_block_lora_forward_device[H, Dh, N_IMG, N_TXT, S_D, True](
        img_x, txt_x, w, imd, tmd, lora_d, cos, sin, D, FMLP, EPS, ctx)
    var dbwd_f = flux_double_block_lora_backward_device[H, Dh, N_IMG, N_TXT, S_D, True](
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
    var dsbwd_f = flux_single_block_lora_backward_device[H, Dh, S_S, True](
        s_d_out_d, sw, smvd, slora_d, dsfwd_f.saved, scos, ssin, D, FMLP, EPS, ctx)
    print("---- SINGLE flash out / d_x (vs host, cos>=0.999) ----")
    _ck(harness, "FLASH s_out", dsfwd_f.out[], hsfwd.out, allok, ctx)
    _ck(harness, "FLASH s_d_x", dsbwd_f.d_x[], hsbwd.base.d_x, allok, ctx)
    print("---- SINGLE flash LoRA dB (vs host, cos>=0.995) ----")
    _ck(harness995, "FLASH s loB to_q", dsbwd_f.d_b[S_SQ].value()[], hsbwd.lora.d_b[S_SQ], allok, ctx)

    print("")
    if allok:
        print("VERDICT: PASS — flux device block matches host incl. mod grads (math bit bars + flash cos bars)")
    else:
        print("VERDICT: FAIL — at least one tensor diverged (see FAIL lines)")
