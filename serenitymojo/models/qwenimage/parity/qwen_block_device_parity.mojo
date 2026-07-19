# serenitymojo/models/qwenimage/parity/qwen_block_device_parity.mojo
#
# HOST-vs-DEVICE parity gate for the device-resident QwenImage DOUBLE LoRA
# block (models/qwenimage/qwenimage_block_device.mojo, MJ-1084 port).
#
# The HOST block (qwenimage_block.mojo double_block_lora_forward/backward — the
# torch-gated oracle) runs on BYTE-IDENTICAL inputs vs the device block.
# Per-tensor bars (chroma_block_device_parity precedent):
#   forward img_out/txt_out : cos >= 0.999
#   backward d_img_x/d_txt_x: cos >= 0.999
#   every LoRA d_a/d_b slot : cos >= 0.999 AND rel-L2 <= 1e-3
# at REAL H=24, Dh=128, D=3072 with small N_IMG=8, N_TXT=6, S=14, F=64.
#
# Run:
#   cd /home/alex/mojodiffusion
#   pixi run mojo build --optimization-level 2 -I . \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/qwenimage/parity/qwen_block_device_parity.mojo \
#     -o output/bin/qwen_block_device_parity && output/bin/qwen_block_device_parity

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.training.train_step import LoraAdapter

# HOST oracle
from serenitymojo.models.qwenimage.qwenimage_block import (
    ModVecs, StreamWeights, DoubleBlockWeights, StreamLora, DoubleBlockLora,
    double_block_lora_forward, double_block_lora_backward,
)

# DEVICE under test
from serenitymojo.models.qwenimage.qwenimage_block_device import (
    QwenModVecsDevice, qwen_modvecs_to_device,
    QwenDoubleBlockLoraDevice, qwen_double_block_lora_to_device,
    qwen_double_block_lora_forward_device, qwen_double_block_lora_backward_device,
    Q_SQ, Q_SK, Q_SV, Q_OUT, Q_UP, Q_DN,
)


comptime TArc = ArcPointer[Tensor]
comptime H = 24
comptime Dh = 128
comptime D = H * Dh           # 3072
comptime N_IMG = 8
comptime N_TXT = 6
comptime S_D = N_IMG + N_TXT  # 14
comptime FMLP = 64
comptime RANK = 4
comptime LSCALE = Float32(2.0)
comptime EPS = Float32(1e-06)


def _next(mut s: UInt64) -> UInt64:
    s = s + 0x9E3779B97F4A7C15
    var z = s
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _randf(mut s: UInt64, r: Float32) -> Float32:
    var u = _next(s)
    var frac = Float32(u >> 40) / Float32(16777216.0)
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
        _rl(s, D * D, Float32(0.04)), _rl(s, D * D, Float32(0.04)), _rl(s, D * D, Float32(0.04)),
        _rl(s, D, Float32(0.02)), _rl(s, D, Float32(0.02)), _rl(s, D, Float32(0.02)),
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
        Optional[LoraAdapter](_adapter(s, D, D)),        # q
        Optional[LoraAdapter](_adapter(s, D, D)),        # k
        Optional[LoraAdapter](_adapter(s, D, D)),        # v
        Optional[LoraAdapter](_adapter(s, D, D)),        # out
        Optional[LoraAdapter](_adapter(s, D, FMLP)),     # ff_up
        Optional[LoraAdapter](_adapter(s, FMLP, D)),     # ff_down
    )


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
    print("==== qwen DEVICE-vs-HOST double-block parity (fwd+bwd+LoRA) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " S=", S_D, " FMLP=", FMLP, " RANK=", RANK)
    var harness = ParityHarness()
    var allok = True
    var s: UInt64 = 0x51E57A7E2026_0706

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

    # device-side carriers — BYTE-IDENTICAL to host inputs. NOTE: the host
    # backward uploads d_out via `_ta` = BF16, so the device backward gets
    # BF16 d_out carriers too.
    var img_x = TArc(Tensor.from_host(img.copy(), [N_IMG, D], STDtype.BF16, ctx))
    var txt_x = TArc(Tensor.from_host(txt.copy(), [N_TXT, D], STDtype.BF16, ctx))
    var imd = qwen_modvecs_to_device(im, D, ctx)
    var tmd = qwen_modvecs_to_device(tm, D, ctx)
    var lora_d = qwen_double_block_lora_to_device(lora, ctx)
    var d_io = TArc(Tensor.from_host(d_img.copy(), [N_IMG, D], STDtype.BF16, ctx))
    var d_to = TArc(Tensor.from_host(d_txt.copy(), [N_TXT, D], STDtype.BF16, ctx))

    # HOST forward + backward (oracle)
    var hfwd = double_block_lora_forward[H, Dh, N_IMG, N_TXT, S_D](
        img.copy(), txt.copy(), w, im, tm, lora, cos, sin, D, FMLP, EPS, ctx)
    var hbwd = double_block_lora_backward[H, Dh, N_IMG, N_TXT, S_D](
        d_img.copy(), d_txt.copy(), w, im, tm, lora, hfwd.saved, cos, sin, D, FMLP, EPS, ctx)

    # DEVICE forward + backward
    var dfwd = qwen_double_block_lora_forward_device[H, Dh, N_IMG, N_TXT, S_D](
        img_x, txt_x, w, imd, tmd, lora_d, cos, sin, D, FMLP, EPS, ctx)
    var dbwd = qwen_double_block_lora_backward_device[H, Dh, N_IMG, N_TXT, S_D](
        d_io, d_to, w, imd, tmd, lora_d, dfwd.saved, cos, sin, D, FMLP, EPS, ctx)

    print("---- forward outputs (device vs host) ----")
    _ck(harness, "img_out", dfwd.img_out[], hfwd.img_out, allok, ctx)
    _ck(harness, "txt_out", dfwd.txt_out[], hfwd.txt_out, allok, ctx)

    print("---- input grads d_x (device vs host) ----")
    _ck(harness, "d_img_x", dbwd.img.d_x[], hbwd.base.img.d_x, allok, ctx)
    _ck(harness, "d_txt_x", dbwd.txt.d_x[], hbwd.base.txt.d_x, allok, ctx)

    print("---- IMG LoRA d_A/d_B (device vs host) ----")
    _ck_lora(harness, "img dA q", dbwd.img.d_a[Q_SQ].value()[], hbwd.img.q_d_a, allok, ctx)
    _ck_lora(harness, "img dB q", dbwd.img.d_b[Q_SQ].value()[], hbwd.img.q_d_b, allok, ctx)
    _ck_lora(harness, "img dA k", dbwd.img.d_a[Q_SK].value()[], hbwd.img.k_d_a, allok, ctx)
    _ck_lora(harness, "img dB k", dbwd.img.d_b[Q_SK].value()[], hbwd.img.k_d_b, allok, ctx)
    _ck_lora(harness, "img dA v", dbwd.img.d_a[Q_SV].value()[], hbwd.img.v_d_a, allok, ctx)
    _ck_lora(harness, "img dB v", dbwd.img.d_b[Q_SV].value()[], hbwd.img.v_d_b, allok, ctx)
    _ck_lora(harness, "img dA out", dbwd.img.d_a[Q_OUT].value()[], hbwd.img.out_d_a, allok, ctx)
    _ck_lora(harness, "img dB out", dbwd.img.d_b[Q_OUT].value()[], hbwd.img.out_d_b, allok, ctx)
    _ck_lora(harness, "img dA ffup", dbwd.img.d_a[Q_UP].value()[], hbwd.img.ff_up_d_a, allok, ctx)
    _ck_lora(harness, "img dB ffup", dbwd.img.d_b[Q_UP].value()[], hbwd.img.ff_up_d_b, allok, ctx)
    _ck_lora(harness, "img dA ffdn", dbwd.img.d_a[Q_DN].value()[], hbwd.img.ff_down_d_a, allok, ctx)
    _ck_lora(harness, "img dB ffdn", dbwd.img.d_b[Q_DN].value()[], hbwd.img.ff_down_d_b, allok, ctx)

    print("---- TXT LoRA d_A/d_B (device vs host) ----")
    _ck_lora(harness, "txt dA q", dbwd.txt.d_a[Q_SQ].value()[], hbwd.txt.q_d_a, allok, ctx)
    _ck_lora(harness, "txt dB q", dbwd.txt.d_b[Q_SQ].value()[], hbwd.txt.q_d_b, allok, ctx)
    _ck_lora(harness, "txt dA k", dbwd.txt.d_a[Q_SK].value()[], hbwd.txt.k_d_a, allok, ctx)
    _ck_lora(harness, "txt dB k", dbwd.txt.d_b[Q_SK].value()[], hbwd.txt.k_d_b, allok, ctx)
    _ck_lora(harness, "txt dA v", dbwd.txt.d_a[Q_SV].value()[], hbwd.txt.v_d_a, allok, ctx)
    _ck_lora(harness, "txt dB v", dbwd.txt.d_b[Q_SV].value()[], hbwd.txt.v_d_b, allok, ctx)
    _ck_lora(harness, "txt dA out", dbwd.txt.d_a[Q_OUT].value()[], hbwd.txt.out_d_a, allok, ctx)
    _ck_lora(harness, "txt dB out", dbwd.txt.d_b[Q_OUT].value()[], hbwd.txt.out_d_b, allok, ctx)
    _ck_lora(harness, "txt dA ffup", dbwd.txt.d_a[Q_UP].value()[], hbwd.txt.ff_up_d_a, allok, ctx)
    _ck_lora(harness, "txt dB ffup", dbwd.txt.d_b[Q_UP].value()[], hbwd.txt.ff_up_d_b, allok, ctx)
    _ck_lora(harness, "txt dA ffdn", dbwd.txt.d_a[Q_DN].value()[], hbwd.txt.ff_down_d_a, allok, ctx)
    _ck_lora(harness, "txt dB ffdn", dbwd.txt.d_b[Q_DN].value()[], hbwd.txt.ff_down_d_b, allok, ctx)

    print("")
    if allok:
        print("VERDICT: PASS - qwen device block matches host (cos>=0.999, LoRA relL2<=1e-3)")
    else:
        print("VERDICT: FAIL - at least one tensor diverged (see FAIL lines)")
