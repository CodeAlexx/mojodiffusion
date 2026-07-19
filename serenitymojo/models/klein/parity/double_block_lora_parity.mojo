# serenitymojo/models/klein/parity/double_block_lora_parity.mojo
#
# PARITY GATE for the Klein DOUBLE-STREAM DiT block LoRA training variant
# (models/klein/double_block.mojo `double_block_lora_forward/backward`). Loads
# the EXACT inputs + torch-autograd reference grads dumped by
# double_block_lora_oracle.py, runs the LoRA-aware forward+backward, and compares
# d_A AND d_B for every adapter (img/txt × qkv/proj) at cos >= 0.999, plus base
# input grads + a couple base weight grads (no-regression check).
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/double_block_lora_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       serenitymojo/models/klein/parity/double_block_lora_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.double_block import (
    StreamWeights, DoubleBlockWeights, ModVecs,
    StreamLora, DoubleBlockLora,
    double_block_lora_forward, double_block_lora_backward,
)
from serenitymojo.models.klein.lora_block import LoraAdapter


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/klein/parity/"

# dims MUST match double_block_lora_oracle.py
comptime H = 32
comptime Dh = 16
comptime D = H * Dh        # 512
comptime N_IMG = 4
comptime N_TXT = 2
comptime F = 24
comptime EPS = Float32(1e-06)
comptime RANK = 8
comptime LSCALE = Float32(16.0) / Float32(8.0)   # alpha/rank = 2.0


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
        o.append(Float32(0.0))
    return o^


def _load_stream(prefix: String, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _in("lin_" + prefix + "_wqkv"), _in("lin_" + prefix + "_wproj"),
        _in("lin_" + prefix + "_wgu"), _in("lin_" + prefix + "_wd"),
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


def _make_adapter(
    a: List[Float32], b: List[Float32], in_f: Int, out_f: Int
) -> LoraAdapter:
    return LoraAdapter(
        a.copy(), b.copy(), RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _load_stream_lora(prefix: String) raises -> StreamLora:
    var qkv = _make_adapter(
        _in("lin_" + prefix + "_qkv_A"), _in("lin_" + prefix + "_qkv_B"), D, 3 * D
    )
    var proj = _make_adapter(
        _in("lin_" + prefix + "_proj_A"), _in("lin_" + prefix + "_proj_B"), D, D
    )
    return StreamLora(Optional[LoraAdapter](qkv^), Optional[LoraAdapter](proj^))


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
    print("==== double_block_lora_parity (Klein double-stream + LoRA vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " F=", F, " RANK=", RANK)

    var img = _in("lin_img")
    var txt = _in("lin_txt")
    var iw = _load_stream("iw", ctx)
    var tw = _load_stream("tw", ctx)
    var im = _load_mod("im")
    var tm = _load_mod("tm")
    var cos_h = _in("lin_cos")
    var sin_h = _in("lin_sin")
    # Resident rope tables: upload ONCE, pass by borrow (matches the trainer).
    var cos = Tensor.from_host(cos_h, [(N_IMG + N_TXT) * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [(N_IMG + N_TXT) * H, Dh // 2], STDtype.F32, ctx)
    var w = DoubleBlockWeights(iw^, tw^)
    var ilo = _load_stream_lora("ilo")
    var tlo = _load_stream_lora("tlo")
    var lora = DoubleBlockLora(ilo^, tlo^)

    var fwd = double_block_lora_forward[H, Dh, N_IMG, N_TXT, N_IMG + N_TXT](
        img.copy(), txt.copy(), w, im, tm, lora, cos, sin,
        D, F, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward outputs vs torch ----")
    _check(harness, "img_out", fwd.img_out, _in("lref_img_out"), allok)
    _check(harness, "txt_out", fwd.txt_out, _in("lref_txt_out"), allok)

    var d_img = _in("lin_d_img")
    var d_txt = _in("lin_d_txt")
    var g = double_block_lora_backward[H, Dh, N_IMG, N_TXT, N_IMG + N_TXT](
        d_img, d_txt, w, im, tm, lora, fwd.saved, cos, sin,
        D, F, EPS, ctx,
    )

    print("")
    print("---- base input grads vs torch (no-regression) ----")
    _check(harness, "d_img", g.base.img.d_x, _in("lref_d_img"), allok)
    _check(harness, "d_txt", g.base.txt.d_x, _in("lref_d_txt"), allok)

    # NOTE: base NON-LoRA weight grads (d_wgu/d_wd/d_wqkv/d_wproj) are NO LONGER
    # checked here. The LoRA backward path now skips them (linear_backward_dx —
    # frozen base weights aren't trained, the grads were computed-then-discarded).
    # That exact d_w matmul math is still validated by the base gate
    # double_block_parity.mojo (all 8 d_w, img/txt × wqkv/wproj/wgu/wd). This gate
    # owns d_x + the LoRA d_A/d_B — exactly what the LoRA path must keep correct.

    print("")
    print("---- IMG LoRA grads d_A / d_B vs torch ----")
    _check(harness, "img qkv  d_A", g.img.qkv_d_a, _in("lref_img_qkv_dA"), allok)
    _check(harness, "img qkv  d_B", g.img.qkv_d_b, _in("lref_img_qkv_dB"), allok)
    _check(harness, "img proj d_A", g.img.proj_d_a, _in("lref_img_proj_dA"), allok)
    _check(harness, "img proj d_B", g.img.proj_d_b, _in("lref_img_proj_dB"), allok)

    print("")
    print("---- TXT LoRA grads d_A / d_B vs torch ----")
    _check(harness, "txt qkv  d_A", g.txt.qkv_d_a, _in("lref_txt_qkv_dA"), allok)
    _check(harness, "txt qkv  d_B", g.txt.qkv_d_b, _in("lref_txt_qkv_dB"), allok)
    _check(harness, "txt proj d_A", g.txt.proj_d_a, _in("lref_txt_proj_dA"), allok)
    _check(harness, "txt proj d_B", g.txt.proj_d_b, _in("lref_txt_proj_dB"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Klein double-stream LoRA fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
