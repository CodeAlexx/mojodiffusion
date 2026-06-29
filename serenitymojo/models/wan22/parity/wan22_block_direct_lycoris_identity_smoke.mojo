# serenitymojo/models/wan22/parity/wan22_block_direct_lycoris_identity_smoke.mojo
#
# Direct DoRA/OFT Wan2.2 block plumbing gate. At initialization, direct DoRA and
# OFT are identity substitutions for the frozen projection weights, so the block
# should match the base WanAttentionBlock without dense full-delta carriers.

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc

from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.flat_direct_lycoris_stack import (
    empty_flat_direct_dora_set, empty_flat_direct_oft_set,
)
from serenitymojo.models.wan22.wan22_direct_lycoris_stack import (
    build_wan22_direct_dora_set_from_weights, build_wan22_direct_oft_set,
    wan22_direct_dora_trainable_bytes, wan22_direct_oft_trainable_bytes,
)
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanModVecs, WanBlockDirectProjectionWeights,
    WanBlockDirectLycoris, WAN_DIRECT_ALGO_DORA, WAN_DIRECT_ALGO_OFT,
    WanBlockLora, wan22_block_lora_forward, wan22_block_lora_backward,
    wan22_block_direct_lycoris_forward, wan22_block_direct_lycoris_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/wan22/parity/"

comptime H = 24
comptime Dh = 8
comptime DIM = H * Dh
comptime S = 5
comptime TXT = 4
comptime FFN = 40
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


def _qk_norm(name: String) raises -> List[Float32]:
    var v = _in(name)
    if len(v) == DIM:
        return v^
    if len(v) != Dh:
        raise Error(String("bad q/k norm fixture length: ") + name)
    var out = List[Float32]()
    for _h in range(H):
        for i in range(Dh):
            out.append(v[i])
    return out^


def _load_weights(ctx: DeviceContext) raises -> WanBlockWeights:
    return WanBlockWeights(
        _in("lin_sa_wq"), _in("lin_sa_wk"), _in("lin_sa_wv"), _in("lin_sa_wo"),
        _in("lin_sa_bq"), _in("lin_sa_bk"), _in("lin_sa_bv"), _in("lin_sa_bo"),
        _qk_norm("lin_sa_qn"), _qk_norm("lin_sa_kn"),
        _in("lin_ca_wq"), _in("lin_ca_wk"), _in("lin_ca_wv"), _in("lin_ca_wo"),
        _in("lin_ca_bq"), _in("lin_ca_bk"), _in("lin_ca_bv"), _in("lin_ca_bo"),
        _qk_norm("lin_ca_qn"), _qk_norm("lin_ca_kn"),
        _in("lin_n3_w"), _in("lin_n3_b"),
        _in("lin_ffn0_w"), _in("lin_ffn0_b"), _in("lin_ffn2_w"), _in("lin_ffn2_b"),
        DIM, FFN, Dh, ctx,
    )


def _load_direct_w() raises -> WanBlockDirectProjectionWeights:
    return WanBlockDirectProjectionWeights(
        _in("lin_sa_wq"), _in("lin_sa_wk"), _in("lin_sa_wv"), _in("lin_sa_wo"),
        _in("lin_sa_bq"), _in("lin_sa_bk"), _in("lin_sa_bv"), _in("lin_sa_bo"),
        _in("lin_ca_wq"), _in("lin_ca_wk"), _in("lin_ca_wv"), _in("lin_ca_wo"),
        _in("lin_ca_bq"), _in("lin_ca_bk"), _in("lin_ca_bv"), _in("lin_ca_bo"),
    )


def _block_weight_list() raises -> List[List[Float32]]:
    var out = List[List[Float32]]()
    out.append(_in("lin_sa_wq"))
    out.append(_in("lin_sa_wk"))
    out.append(_in("lin_sa_wv"))
    out.append(_in("lin_sa_wo"))
    out.append(_in("lin_ca_wq"))
    out.append(_in("lin_ca_wk"))
    out.append(_in("lin_ca_wv"))
    out.append(_in("lin_ca_wo"))
    return out^


def _load_mod() raises -> WanModVecs:
    return WanModVecs(
        _in("lin_shift_sa"), _in("lin_scale_sa"), _in("lin_gate_sa"),
        _in("lin_shift_ffn"), _in("lin_scale_ffn"), _in("lin_gate_ffn"),
    )


def _empty_lora() -> WanBlockLora:
    return WanBlockLora(
        Optional[LoraAdapter](),
        Optional[LoraAdapter](),
        Optional[LoraAdapter](),
        Optional[LoraAdapter](),
        Optional[LoraAdapter](),
        Optional[LoraAdapter](),
        Optional[LoraAdapter](),
        Optional[LoraAdapter](),
    )


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
    print("==== wan22_block_direct_lycoris_identity_smoke ====")
    print("H=", H, " Dh=", Dh, " DIM=", DIM, " S=", S, " TXT=", TXT)

    var x = _in("lin_x")
    var context = _in("lin_context")
    var mv = _load_mod()
    var cos = Tensor.from_host(_in("lin_cos"), [S, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("lin_sin"), [S, Dh // 2], STDtype.F32, ctx)
    var d_out = _in("lin_d_out")

    var base_w = _load_weights(ctx)
    var ref_lora = _empty_lora()
    var base_fwd = wan22_block_lora_forward[H, Dh, S, TXT](
        x.copy(), context.copy(), mv, base_w, ref_lora, cos, sin, DIM, FFN, EPS, ctx,
    )
    var allok = True
    var h = ParityHarness(0.995)

    var dora_set = build_wan22_direct_dora_set_from_weights(
        _block_weight_list(), 1, DIM, RANK, ALPHA, UInt64(2201), False,
    )
    var dora_direct = WanBlockDirectLycoris(
        WAN_DIRECT_ALGO_DORA, dora_set.copy(), empty_flat_direct_oft_set(), 0,
    )
    var dora_w = _load_weights(ctx)
    var dora_fwd = wan22_block_direct_lycoris_forward[H, Dh, S, TXT](
        x.copy(), context.copy(), mv, dora_w, _load_direct_w(), dora_direct,
        cos, sin, DIM, FFN, EPS, ctx,
    )
    var dora_g = wan22_block_direct_lycoris_backward[H, Dh, S, TXT](
        d_out.copy(), mv, dora_w, _load_direct_w(), dora_direct,
        dora_fwd.saved, cos, sin, DIM, FFN, EPS, ctx,
    )
    print("[direct-dora] trainable_bytes=", wan22_direct_dora_trainable_bytes(dora_set))
    _check(h, "dora x_out", dora_fwd.x_out, base_fwd.x_out, allok)
    var dora_grad_l1 = _l1(dora_g.sa_q.d_b) + _l1(dora_g.sa_o.d_b) + _l1(dora_g.ca_o.d_b)
    print("  dora selected_grad_l1=", dora_grad_l1)
    if dora_grad_l1 <= 0.0:
        allok = False

    var oft_set = build_wan22_direct_oft_set(1, DIM, OFT_BLOCK)
    var oft_direct = WanBlockDirectLycoris(
        WAN_DIRECT_ALGO_OFT, empty_flat_direct_dora_set(), oft_set.copy(), 0,
    )
    var oft_w = _load_weights(ctx)
    var oft_fwd = wan22_block_direct_lycoris_forward[H, Dh, S, TXT](
        x.copy(), context.copy(), mv, oft_w, _load_direct_w(), oft_direct,
        cos, sin, DIM, FFN, EPS, ctx,
    )
    var oft_g = wan22_block_direct_lycoris_backward[H, Dh, S, TXT](
        d_out.copy(), mv, oft_w, _load_direct_w(), oft_direct,
        oft_fwd.saved, cos, sin, DIM, FFN, EPS, ctx,
    )
    print("[direct-oft] trainable_bytes=", wan22_direct_oft_trainable_bytes(oft_set))
    _check(h, "oft x_out", oft_fwd.x_out, base_fwd.x_out, allok)
    _check(h, "dora vs oft d_x", dora_g.d_x, oft_g.d_x, allok)
    _check(h, "dora vs oft d_context", dora_g.d_context, oft_g.d_context, allok)
    var oft_grad_l1 = _l1(oft_g.sa_q.d_vec) + _l1(oft_g.sa_o.d_vec) + _l1(oft_g.ca_o.d_vec)
    print("  oft selected_grad_l1=", oft_grad_l1)
    if oft_grad_l1 <= 0.0:
        allok = False

    if allok:
        print("ALL GATES PASS -- wan22_block_direct_lycoris_identity_smoke")
    else:
        raise Error("wan22_block_direct_lycoris_identity_smoke failed")
