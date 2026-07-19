# serenitymojo/models/klein/parity/double_block_ft_parity.mojo
#
# PARITY GATE for the Klein DOUBLE-STREAM block FULL-FINETUNE backward
# (double_block_ft_backward_dev): the 8 trainable matmul grads (per stream:
# d_wqkv/d_wproj/d_wgu/d_wd, F32 device) + the input grads d_img/d_txt vs
# torch autograd, cos >= 0.999. The forward is the DEVICE-RESIDENT base block
# with NO adapters (full-FT has no LoRA); the oracle is the EXISTING base-only
# double_block_oracle.py (its refs ref_{img,txt}_d_w* / ref_d_{img,txt} are
# torch W.grad / input.grad with no adapter in the graph — exactly the FT
# surface). v1 FROZEN params (q/k rms scales, mod vecs) are NOT compared —
# the FT backward intentionally skips them (see double_block.mojo FT header).
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/SerenityTrainer/venv/bin/python \
#       serenitymojo/models/klein/parity/double_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 \
#       -I . -I /home/alex/MOJO-libs -Xlinker -lm \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       -Xlinker -rpath -Xlinker $PWD/serenitymojo/ops/cshim/lib \
#       -Xlinker -rpath -Xlinker /home/alex/.serenity/cudnn/lib \
#       serenitymojo/models/klein/parity/double_block_ft_parity.mojo \
#       -o /tmp/double_block_ft_parity
#   /tmp/double_block_ft_parity

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.klein.double_block import (
    StreamWeights, DoubleBlockWeights, ModVecs, modvecs_to_device,
    StreamLoraDevice, DoubleBlockLoraDevice,
    double_block_lora_forward_device_resident,
    double_block_ft_backward_dev,
)

comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/klein/parity/"

# dims MUST match double_block_oracle.py
comptime H = 32
comptime Dh = 16
comptime D = H * Dh        # 512
comptime N_IMG = 4
comptime N_TXT = 2
comptime S = N_IMG + N_TXT
comptime F = 24
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


# chunk `idx` of width `d` out of a flat [k*d] host list.
def _chunk_host(src: List[Float32], idx: Int, d: Int) -> List[Float32]:
    var o = List[Float32]()
    var base = idx * d
    for i in range(d):
        o.append(src[base + i])
    return o^


def _load_stream(prefix: String, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _in("in_" + prefix + "_wqkv"), _in("in_" + prefix + "_wproj"),
        _in("in_" + prefix + "_wgu"), _in("in_" + prefix + "_wd"),
        _in("in_" + prefix + "_q_norm"), _in("in_" + prefix + "_k_norm"),
        D, F, Dh, ctx,
    )


def _load_mod(prefix: String) raises -> ModVecs:
    return ModVecs(
        _in("in_" + prefix + "_shift1"), _in("in_" + prefix + "_scale1"),
        _in("in_" + prefix + "_gate1"),
        _in("in_" + prefix + "_shift2"), _in("in_" + prefix + "_scale2"),
        _in("in_" + prefix + "_gate2"),
    )


def _none_stream_lora() -> StreamLoraDevice:
    return StreamLoraDevice(
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
        Optional[LoraAdapterDevice](None), Optional[LoraAdapterDevice](None),
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
    print("==== double_block_ft_parity (Klein double-block full-FT dW backward vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT, " F=", F)

    # ── load inputs (byte-identical to the oracle) ──
    var img = _in("in_img")
    var txt = _in("in_txt")
    var iw = _load_stream("iw", ctx)
    var tw = _load_stream("tw", ctx)
    var im = _load_mod("im")
    var tm = _load_mod("tm")
    var im_dev = modvecs_to_device(im, D, ctx)
    var tm_dev = modvecs_to_device(tm, D, ctx)
    var cos_h = _in("in_cos")
    var sin_h = _in("in_sin")
    var cos = Tensor.from_host(cos_h, [S * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [S * H, Dh // 2], STDtype.F32, ctx)
    var w = DoubleBlockWeights(iw^, tw^)

    # NO adapters: the full-FT forward is the base block.
    var lora = DoubleBlockLoraDevice(_none_stream_lora(), _none_stream_lora())

    # ── forward (device-resident base; saves the tape the FT backward consumes) ──
    var img_x = TArc(Tensor.from_host(img, [N_IMG, D], STDtype.F32, ctx))
    var txt_x = TArc(Tensor.from_host(txt, [N_TXT, D], STDtype.F32, ctx))
    var fwd = double_block_lora_forward_device_resident[H, Dh, N_IMG, N_TXT, S](
        img_x, txt_x, w, im_dev, tm_dev, lora, cos, sin, D, F, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward outputs vs torch ----")
    var img_out_h = fwd.img_out[].to_host(ctx)
    var txt_out_h = fwd.txt_out[].to_host(ctx)
    _check(harness, "img_out", img_out_h, _in("ref_img_out"), allok)
    _check(harness, "txt_out", txt_out_h, _in("ref_txt_out"), allok)

    # ── FT backward: d_img/d_txt + the 8 matmul dW ──
    var d_img_h = _in("in_d_img")
    var d_txt_h = _in("in_d_txt")
    var d_io = TArc(Tensor.from_host(d_img_h, [N_IMG, D], STDtype.F32, ctx))
    var d_to = TArc(Tensor.from_host(d_txt_h, [N_TXT, D], STDtype.F32, ctx))
    var g = double_block_ft_backward_dev[H, Dh, N_IMG, N_TXT, S, True](
        d_io, d_to, w, im_dev, tm_dev, fwd.saved, cos, sin, D, F, EPS, ctx,
    )

    print("")
    print("---- input grads vs torch ----")
    var d_img_x_h = g.d_img_x[].to_host(ctx)
    var d_txt_x_h = g.d_txt_x[].to_host(ctx)
    _check(harness, "d_img", d_img_x_h, _in("ref_d_img"), allok)
    _check(harness, "d_txt", d_txt_x_h, _in("ref_d_txt"), allok)

    print("")
    print("---- trainable weight grads d_W vs torch (8 matmuls, fixed dw order) ----")
    var names = List[String]()
    names.append(String("img_d_wqkv"))
    names.append(String("img_d_wproj"))
    names.append(String("img_d_wgu"))
    names.append(String("img_d_wd"))
    names.append(String("txt_d_wqkv"))
    names.append(String("txt_d_wproj"))
    names.append(String("txt_d_wgu"))
    names.append(String("txt_d_wd"))
    for i in range(8):
        var dw_h = g.dw[i][].to_host(ctx)
        _check(harness, names[i], dw_h, _in("ref_" + names[i]), allok)

    # ── SURF (Phase B): q/k rms-scale grads (chain-clean) ──
    print("")
    print("---- q/k rms-scale grads vs torch (Phase B) ----")
    var qk_names = List[String]()
    qk_names.append(String("img_d_qnorm"))
    qk_names.append(String("img_d_knorm"))
    qk_names.append(String("txt_d_qnorm"))
    qk_names.append(String("txt_d_knorm"))
    for i in range(4):
        var qk_h = g.d_qk[i][].to_host(ctx)
        _check(harness, qk_names[i], qk_h, _in("ref_" + qk_names[i]), allok)

    # ── SURF (Phase B): shared mod.lin FLAT grads, per-arm (unpack [1,6D]) ──
    print("")
    print("---- shared modulation-linear grads vs torch, per chunk (Phase B) ----")
    var chunk_names = List[String]()
    chunk_names.append(String("shift1"))
    chunk_names.append(String("scale1"))
    chunk_names.append(String("gate1"))
    chunk_names.append(String("shift2"))
    chunk_names.append(String("scale2"))
    chunk_names.append(String("gate2"))
    var im_flat = g.mod_flat[0][].to_host(ctx)   # [6D]
    var tm_flat = g.mod_flat[1][].to_host(ctx)
    for c in range(6):
        var im_c = _chunk_host(im_flat, c, D)
        _check(harness, "img_d_" + chunk_names[c], im_c, _in("ref_img_d_" + chunk_names[c]), allok)
    for c in range(6):
        var tm_c = _chunk_host(tm_flat, c, D)
        _check(harness, "txt_d_" + chunk_names[c], tm_c, _in("ref_txt_d_" + chunk_names[c]), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Klein double-block full-FT backward matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one grad diverged (see FAIL lines above)")
