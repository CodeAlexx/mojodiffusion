# serenitymojo/models/qwenimage/parity/qwen_block_direct_lycoris_identity_smoke.mojo
#
# Direct DoRA/OFT Qwen-Image double-block plumbing gate. At initialization,
# direct DoRA and OFT are identity substitutions for the frozen projection
# weights, so the direct block should match the base block without dense
# full-delta carriers.

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc

from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.flat_direct_lycoris_stack import (
    empty_flat_direct_dora_set, empty_flat_direct_oft_set,
)
from serenitymojo.models.qwenimage.qwenimage_direct_lycoris_stack import (
    build_qwen_direct_dora_set_from_weights, build_qwen_direct_oft_set,
    qwen_direct_dora_trainable_bytes, qwen_direct_oft_trainable_bytes,
)
from serenitymojo.models.qwenimage.qwenimage_block import (
    StreamWeights, DoubleBlockWeights, ModVecs,
    double_block_forward, double_block_backward,
    QwenBlockDirectLycoris, QWEN_DIRECT_ALGO_DORA, QWEN_DIRECT_ALGO_OFT,
    QWEN_DIRECT_TGT_ALL,
    double_block_direct_lycoris_forward,
    double_block_direct_lycoris_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/qwenimage/parity/"

comptime H = 24
comptime Dh = 16
comptime D = H * Dh
comptime N_IMG = 4
comptime N_TXT = 3
comptime S = N_IMG + N_TXT
comptime F = 40
comptime EPS = Float32(1e-06)
comptime RANK = 4
comptime ALPHA = Float32(4.0)
comptime OFT_BLOCK = 4


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref: ") + path)
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


def _load_weights(ctx: DeviceContext) raises -> DoubleBlockWeights:
    var iw = _load_stream("iw", ctx)
    var tw = _load_stream("tw", ctx)
    return DoubleBlockWeights(iw^, tw^)


def _load_mod(prefix: String) raises -> ModVecs:
    return ModVecs(
        _in("lin_" + prefix + "_shift1"), _in("lin_" + prefix + "_scale1"),
        _in("lin_" + prefix + "_gate1"),
        _in("lin_" + prefix + "_shift2"), _in("lin_" + prefix + "_scale2"),
        _in("lin_" + prefix + "_gate2"),
    )


def _block_weight_list() raises -> List[List[Float32]]:
    var out = List[List[Float32]]()
    out.append(_in("lin_iw_wq"))
    out.append(_in("lin_iw_wk"))
    out.append(_in("lin_iw_wv"))
    out.append(_in("lin_iw_wout"))
    out.append(_in("lin_iw_wup"))
    out.append(_in("lin_iw_wdn"))
    out.append(_in("lin_tw_wq"))
    out.append(_in("lin_tw_wk"))
    out.append(_in("lin_tw_wv"))
    out.append(_in("lin_tw_wout"))
    out.append(_in("lin_tw_wup"))
    out.append(_in("lin_tw_wdn"))
    return out^


def _l1(v: List[Float32]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(v)):
        var x = Float64(v[i])
        s += x if x >= 0.0 else -x
    return s


def _check(
    mut h: ParityHarness, name: String, got: List[Float32],
    expected: List[Float32], mut ok: Bool,
) raises:
    var r = h.compare_host(got, expected)
    print("  cos(", name, ") =", r.cos, " max_abs=", r.max_abs, " ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        ok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== qwen_block_direct_lycoris_identity_smoke ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT)

    var img = _in("lin_img")
    var txt = _in("lin_txt")
    var img_mod = _load_mod("im")
    var txt_mod = _load_mod("tm")
    var cos = Tensor.from_host(_in("lin_cos"), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("lin_sin"), [S * H, Dh // 2], STDtype.F32, ctx)
    var d_img = _in("lin_d_img")
    var d_txt = _in("lin_d_txt")

    var base_w = _load_weights(ctx)
    var base_fwd = double_block_forward[H, Dh, N_IMG, N_TXT, S](
        img.copy(), txt.copy(), base_w, img_mod, txt_mod, cos, sin, D, F, EPS, ctx,
    )
    var base_g = double_block_backward[H, Dh, N_IMG, N_TXT, S](
        d_img.copy(), d_txt.copy(), base_w, img_mod, txt_mod,
        base_fwd.saved, cos, sin, D, F, EPS, ctx,
    )

    var allok = True
    var h = ParityHarness(0.995)

    var dora_set = build_qwen_direct_dora_set_from_weights(
        _block_weight_list(), 1, D, F, RANK, ALPHA,
        QWEN_DIRECT_TGT_ALL, UInt64(9201), False,
    )
    var dora_direct = QwenBlockDirectLycoris(
        QWEN_DIRECT_ALGO_DORA, dora_set.copy(), empty_flat_direct_oft_set(),
        0, QWEN_DIRECT_TGT_ALL,
    )
    var dora_w = _load_weights(ctx)
    var dora_fwd = double_block_direct_lycoris_forward[H, Dh, N_IMG, N_TXT, S](
        img.copy(), txt.copy(), dora_w, img_mod, txt_mod, dora_direct,
        cos, sin, D, F, EPS, ctx,
    )
    var dora_g = double_block_direct_lycoris_backward[H, Dh, N_IMG, N_TXT, S](
        d_img.copy(), d_txt.copy(), dora_w, img_mod, txt_mod, dora_direct,
        dora_fwd.saved, cos, sin, D, F, EPS, ctx,
    )
    print("[direct-dora] trainable_bytes=", qwen_direct_dora_trainable_bytes(dora_set))
    _check(h, "dora img_out", dora_fwd.img_out, base_fwd.img_out, allok)
    _check(h, "dora txt_out", dora_fwd.txt_out, base_fwd.txt_out, allok)
    _check(h, "dora d_img", dora_g.img.d_x, base_g.img.d_x, allok)
    _check(h, "dora d_txt", dora_g.txt.d_x, base_g.txt.d_x, allok)
    var dora_grad_l1 = (
        _l1(dora_g.img.q.d_b) + _l1(dora_g.img.out_proj.d_b)
        + _l1(dora_g.img.ff_up.d_b) + _l1(dora_g.txt.q.d_b)
        + _l1(dora_g.txt.out_proj.d_b)
    )
    print("  dora selected_grad_l1=", dora_grad_l1)
    if dora_grad_l1 <= 0.0:
        allok = False

    var oft_set = build_qwen_direct_oft_set(1, D, F, OFT_BLOCK, QWEN_DIRECT_TGT_ALL)
    var oft_direct = QwenBlockDirectLycoris(
        QWEN_DIRECT_ALGO_OFT, empty_flat_direct_dora_set(), oft_set.copy(),
        0, QWEN_DIRECT_TGT_ALL,
    )
    var oft_w = _load_weights(ctx)
    var oft_fwd = double_block_direct_lycoris_forward[H, Dh, N_IMG, N_TXT, S](
        img.copy(), txt.copy(), oft_w, img_mod, txt_mod, oft_direct,
        cos, sin, D, F, EPS, ctx,
    )
    var oft_g = double_block_direct_lycoris_backward[H, Dh, N_IMG, N_TXT, S](
        d_img.copy(), d_txt.copy(), oft_w, img_mod, txt_mod, oft_direct,
        oft_fwd.saved, cos, sin, D, F, EPS, ctx,
    )
    print("[direct-oft] trainable_bytes=", qwen_direct_oft_trainable_bytes(oft_set))
    _check(h, "oft img_out", oft_fwd.img_out, base_fwd.img_out, allok)
    _check(h, "oft txt_out", oft_fwd.txt_out, base_fwd.txt_out, allok)
    _check(h, "oft d_img", oft_g.img.d_x, base_g.img.d_x, allok)
    _check(h, "oft d_txt", oft_g.txt.d_x, base_g.txt.d_x, allok)
    var oft_grad_l1 = (
        _l1(oft_g.img.q.d_vec) + _l1(oft_g.img.out_proj.d_vec)
        + _l1(oft_g.img.ff_up.d_vec) + _l1(oft_g.txt.q.d_vec)
        + _l1(oft_g.txt.out_proj.d_vec)
    )
    print("  oft selected_grad_l1=", oft_grad_l1)
    if oft_grad_l1 <= 0.0:
        allok = False

    if allok:
        print("ALL GATES PASS -- qwen_block_direct_lycoris_identity_smoke")
    else:
        raise Error("qwen_block_direct_lycoris_identity_smoke failed")
