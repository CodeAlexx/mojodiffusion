# serenitymojo/models/qwenimage/parity/qwenimage_block_lora_parity.mojo
#
# PARITY GATE for the Qwen-Image double-stream block LoRA variant
# (models/qwenimage/qwenimage_block.mojo::double_block_lora_forward/backward).
# Loads inputs + torch-autograd reference d_A/d_B (every adapter) + input grads
# from qwenimage_block_lora_oracle.py, runs the LoRA fwd+bwd, compares at cos>=0.999.
#
# Run (oracle FIRST, SEPARATE command):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/qwenimage/parity/qwenimage_block_lora_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/qwenimage/parity/qwenimage_block_lora_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.lora_block import LoraAdapter
from serenitymojo.models.qwenimage.qwenimage_block import (
    StreamWeights, DoubleBlockWeights, ModVecs,
    StreamLora, DoubleBlockLora,
    double_block_lora_forward, double_block_lora_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/qwenimage/parity/"

comptime H = 24
comptime Dh = 16
comptime D = H * Dh        # 384
comptime N_IMG = 4
comptime N_TXT = 3
comptime F = 40
comptime RANK = 8
comptime ALPHA = Float32(8.0)
comptime SCALE_LORA = ALPHA / Float32(RANK)
comptime EPS = Float32(1e-06)


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the oracle first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _in(name: String) raises -> List[Float32]:
    return _read_bin_f32(REF_DIR + name + ".bin")


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _load_stream(prefix: String, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _in("lin_" + prefix + "_wq"), _in("lin_" + prefix + "_wk"), _in("lin_" + prefix + "_wv"),
        _in("lin_" + prefix + "_bq"), _in("lin_" + prefix + "_bk"), _in("lin_" + prefix + "_bv"),
        _in("lin_" + prefix + "_wout"), _in("lin_" + prefix + "_bout"),
        _in("lin_" + prefix + "_wup"), _in("lin_" + prefix + "_bup"),
        _in("lin_" + prefix + "_wdn"), _in("lin_" + prefix + "_bdn"),
        _in("lin_" + prefix + "_q_norm"), _in("lin_" + prefix + "_k_norm"),
        D, F, Dh, ctx,
    )


def _load_mod(prefix: String) raises -> ModVecs:
    return ModVecs(
        _in("lin_" + prefix + "_shift1"), _in("lin_" + prefix + "_scale1"),
        _in("lin_" + prefix + "_gate1"),
        _in("lin_" + prefix + "_shift2"), _in("lin_" + prefix + "_scale2"),
        _in("lin_" + prefix + "_gate2"),
    )


def _adapter(prefix: String, key: String, in_f: Int, out_f: Int) raises -> Optional[LoraAdapter]:
    var a = _in("lin_" + prefix + "_" + key + "_A")
    var b = _in("lin_" + prefix + "_" + key + "_B")
    return Optional[LoraAdapter](LoraAdapter(
        a^, b^, RANK, in_f, out_f, SCALE_LORA,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    ))


def _load_lora(prefix: String) raises -> StreamLora:
    return StreamLora(
        _adapter(prefix, "q", D, D), _adapter(prefix, "k", D, D),
        _adapter(prefix, "v", D, D), _adapter(prefix, "out", D, D),
        _adapter(prefix, "ff_up", D, F), _adapter(prefix, "ff_down", F, D),
    )


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== qwenimage_block_lora_parity (Qwen-Image LoRA fwd+bwd vs torch) ====")

    var img = _in("lin_img")
    var txt = _in("lin_txt")
    var iw = _load_stream("iw", ctx)
    var tw = _load_stream("tw", ctx)
    var im = _load_mod("im")
    var tm = _load_mod("tm")
    var ilo = _load_lora("ilo")
    var tlo = _load_lora("tlo")
    var cos = Tensor.from_host(_in("lin_cos"), [(N_IMG + N_TXT) * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("lin_sin"), [(N_IMG + N_TXT) * H, Dh // 2], STDtype.F32, ctx)
    var w = DoubleBlockWeights(iw^, tw^)
    var lora = DoubleBlockLora(ilo^, tlo^)

    var fwd = double_block_lora_forward[H, Dh, N_IMG, N_TXT, N_IMG + N_TXT](
        img.copy(), txt.copy(), w, im, tm, lora, cos, sin, D, F, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    var d_img = _in("lin_d_img")
    var d_txt = _in("lin_d_txt")
    var g = double_block_lora_backward[H, Dh, N_IMG, N_TXT, N_IMG + N_TXT](
        d_img, d_txt, w, im, tm, lora, fwd.saved, cos, sin, D, F, EPS, ctx,
    )

    print("")
    print("---- input grads vs torch ----")
    _check(harness, "d_img", g.base.img.d_x, _in("lref_d_img"), allok)
    _check(harness, "d_txt", g.base.txt.d_x, _in("lref_d_txt"), allok)

    print("")
    print("---- IMG LoRA d_A/d_B vs torch ----")
    _check(harness, "img q dA  ", g.img.q_d_a, _in("lref_img_q_dA"), allok)
    _check(harness, "img q dB  ", g.img.q_d_b, _in("lref_img_q_dB"), allok)
    _check(harness, "img k dA  ", g.img.k_d_a, _in("lref_img_k_dA"), allok)
    _check(harness, "img k dB  ", g.img.k_d_b, _in("lref_img_k_dB"), allok)
    _check(harness, "img v dA  ", g.img.v_d_a, _in("lref_img_v_dA"), allok)
    _check(harness, "img v dB  ", g.img.v_d_b, _in("lref_img_v_dB"), allok)
    _check(harness, "img out dA", g.img.out_d_a, _in("lref_img_out_dA"), allok)
    _check(harness, "img out dB", g.img.out_d_b, _in("lref_img_out_dB"), allok)
    _check(harness, "img ffu dA", g.img.ff_up_d_a, _in("lref_img_ff_up_dA"), allok)
    _check(harness, "img ffu dB", g.img.ff_up_d_b, _in("lref_img_ff_up_dB"), allok)
    _check(harness, "img ffd dA", g.img.ff_down_d_a, _in("lref_img_ff_down_dA"), allok)
    _check(harness, "img ffd dB", g.img.ff_down_d_b, _in("lref_img_ff_down_dB"), allok)

    print("")
    print("---- TXT LoRA d_A/d_B vs torch ----")
    _check(harness, "txt q dA  ", g.txt.q_d_a, _in("lref_txt_q_dA"), allok)
    _check(harness, "txt q dB  ", g.txt.q_d_b, _in("lref_txt_q_dB"), allok)
    _check(harness, "txt k dA  ", g.txt.k_d_a, _in("lref_txt_k_dA"), allok)
    _check(harness, "txt k dB  ", g.txt.k_d_b, _in("lref_txt_k_dB"), allok)
    _check(harness, "txt v dA  ", g.txt.v_d_a, _in("lref_txt_v_dA"), allok)
    _check(harness, "txt v dB  ", g.txt.v_d_b, _in("lref_txt_v_dB"), allok)
    _check(harness, "txt out dA", g.txt.out_d_a, _in("lref_txt_out_dA"), allok)
    _check(harness, "txt out dB", g.txt.out_d_b, _in("lref_txt_out_dB"), allok)
    _check(harness, "txt ffu dA", g.txt.ff_up_d_a, _in("lref_txt_ff_up_dA"), allok)
    _check(harness, "txt ffu dB", g.txt.ff_up_d_b, _in("lref_txt_ff_up_dB"), allok)
    _check(harness, "txt ffd dA", g.txt.ff_down_d_a, _in("lref_txt_ff_down_dA"), allok)
    _check(harness, "txt ffd dB", g.txt.ff_down_d_b, _in("lref_txt_ff_down_dB"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Qwen-Image LoRA fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one LoRA grad diverged")
