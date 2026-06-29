# serenitymojo/models/flux/parity/flux_block_direct_lycoris_identity_smoke.mojo
#
# Direct DoRA/OFT Flux/Chroma shared block plumbing gate. At initialization,
# direct DoRA and OFT are identity substitutions for the frozen projection
# weights, so the direct double and single blocks should match the base block
# without dense full-delta carriers.

from std.gpu.host import DeviceContext
from std.collections import List

from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.flat_direct_lycoris_stack import (
    empty_flat_direct_dora_set, empty_flat_direct_oft_set,
)
from serenitymojo.models.flux.block import (
    StreamWeights, DoubleBlockWeights, ModVecs,
    SingleBlockWeights, SingleModVecs,
    double_block_forward, double_block_backward,
    single_block_forward, single_block_backward,
)
from serenitymojo.models.flux.flux_lycoris_stack import FLUX_LYCORIS_TGT_ALL
from serenitymojo.models.flux.flux_direct_lycoris_stack import (
    build_flux_direct_dora_set_from_weights, build_flux_direct_oft_set,
    flux_direct_active_slot_count,
    flux_direct_dora_trainable_bytes, flux_direct_oft_trainable_bytes,
)
from serenitymojo.models.flux.lora_block import (
    FluxDoubleBlockDirectLycoris, FluxSingleBlockDirectLycoris,
    FLUX_DIRECT_ALGO_DORA, FLUX_DIRECT_ALGO_OFT, FLUX_DIRECT_TGT_ALL,
    double_block_direct_lycoris_forward,
    double_block_direct_lycoris_backward,
    single_block_direct_lycoris_forward,
    single_block_direct_lycoris_backward,
)


comptime H = 2
comptime Dh = 8
comptime D = H * Dh
comptime N_IMG = 3
comptime N_TXT = 2
comptime S = N_IMG + N_TXT
comptime F = 32
comptime EPS = Float32(1e-06)
comptime RANK = 4
comptime ALPHA = Float32(4.0)
comptime OFT_BLOCK = 4


def _randn(n: Int, seed: UInt64, scale: Float32, bias: Float32 = Float32(0.0)) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale + bias)
    return out^


def _ones(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(1.0))
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _take_rows(src: List[Float32], start: Int, rows: Int, cols: Int) -> List[Float32]:
    var out = List[Float32]()
    for r in range(rows):
        var base = (start + r) * cols
        for c in range(cols):
            out.append(src[base + c])
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


def _stream_weights(prefix_seed: UInt64, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _randn(3 * D * D, prefix_seed + 1, Float32(0.12)),
        _randn(3 * D, prefix_seed + 2, Float32(0.02)),
        _randn(D * D, prefix_seed + 3, Float32(0.12)),
        _randn(D, prefix_seed + 4, Float32(0.02)),
        _randn(F * D, prefix_seed + 5, Float32(0.10)),
        _randn(F, prefix_seed + 6, Float32(0.02)),
        _randn(D * F, prefix_seed + 7, Float32(0.10)),
        _randn(D, prefix_seed + 8, Float32(0.02)),
        _ones(Dh), _ones(Dh),
        D, F, Dh, ctx,
    )


def _mod(seed: UInt64) -> ModVecs:
    return ModVecs(
        _randn(D, seed + 1, Float32(0.04)),
        _randn(D, seed + 2, Float32(0.04)),
        _randn(D, seed + 3, Float32(0.08)),
        _randn(D, seed + 4, Float32(0.04)),
        _randn(D, seed + 5, Float32(0.04)),
        _randn(D, seed + 6, Float32(0.08)),
    )


def _single_mod(seed: UInt64) -> SingleModVecs:
    return SingleModVecs(
        _randn(D, seed + 1, Float32(0.04)),
        _randn(D, seed + 2, Float32(0.04)),
        _randn(D, seed + 3, Float32(0.08)),
    )


def _append_stream_slots(mut out: List[List[Float32]], wqkv: List[Float32], wproj: List[Float32], wmlp0: List[Float32], wmlp2: List[Float32]):
    out.append(_take_rows(wqkv, 0, D, D))
    out.append(_take_rows(wqkv, D, D, D))
    out.append(_take_rows(wqkv, 2 * D, D, D))
    out.append(wproj.copy())
    out.append(wmlp0.copy())
    out.append(wmlp2.copy())


def main() raises:
    var ctx = DeviceContext()
    print("==== flux_block_direct_lycoris_identity_smoke ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT, " F=", F)

    var img = _randn(N_IMG * D, UInt64(10), Float32(0.20))
    var txt = _randn(N_TXT * D, UInt64(11), Float32(0.20))
    var sx = _randn(S * D, UInt64(12), Float32(0.20))
    var d_img = _randn(N_IMG * D, UInt64(13), Float32(0.10))
    var d_txt = _randn(N_TXT * D, UInt64(14), Float32(0.10))
    var d_s = _randn(S * D, UInt64(15), Float32(0.10))
    var cos = Tensor.from_host(_ones(S * H * (Dh // 2)), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_zeros(S * H * (Dh // 2)), [S * H, Dh // 2], STDtype.F32, ctx)
    var img_mod = _mod(UInt64(100))
    var txt_mod = _mod(UInt64(200))
    var smod = _single_mod(UInt64(300))

    var iw_wqkv = _randn(3 * D * D, UInt64(1001), Float32(0.12))
    var iw_bqkv = _randn(3 * D, UInt64(1002), Float32(0.02))
    var iw_wproj = _randn(D * D, UInt64(1003), Float32(0.12))
    var iw_bproj = _randn(D, UInt64(1004), Float32(0.02))
    var iw_wmlp0 = _randn(F * D, UInt64(1005), Float32(0.10))
    var iw_bmlp0 = _randn(F, UInt64(1006), Float32(0.02))
    var iw_wmlp2 = _randn(D * F, UInt64(1007), Float32(0.10))
    var iw_bmlp2 = _randn(D, UInt64(1008), Float32(0.02))

    var tw_wqkv = _randn(3 * D * D, UInt64(2001), Float32(0.12))
    var tw_bqkv = _randn(3 * D, UInt64(2002), Float32(0.02))
    var tw_wproj = _randn(D * D, UInt64(2003), Float32(0.12))
    var tw_bproj = _randn(D, UInt64(2004), Float32(0.02))
    var tw_wmlp0 = _randn(F * D, UInt64(2005), Float32(0.10))
    var tw_bmlp0 = _randn(F, UInt64(2006), Float32(0.02))
    var tw_wmlp2 = _randn(D * F, UInt64(2007), Float32(0.10))
    var tw_bmlp2 = _randn(D, UInt64(2008), Float32(0.02))

    var s_w1 = _randn((3 * D + F) * D, UInt64(3001), Float32(0.10))
    var s_b1 = _randn(3 * D + F, UInt64(3002), Float32(0.02))
    var s_w2 = _randn(D * (D + F), UInt64(3003), Float32(0.10))
    var s_b2 = _randn(D, UInt64(3004), Float32(0.02))

    var iw = StreamWeights(
        iw_wqkv.copy(), iw_bqkv.copy(), iw_wproj.copy(), iw_bproj.copy(),
        iw_wmlp0.copy(), iw_bmlp0.copy(), iw_wmlp2.copy(), iw_bmlp2.copy(),
        _ones(Dh), _ones(Dh), D, F, Dh, ctx,
    )
    var tw = StreamWeights(
        tw_wqkv.copy(), tw_bqkv.copy(), tw_wproj.copy(), tw_bproj.copy(),
        tw_wmlp0.copy(), tw_bmlp0.copy(), tw_wmlp2.copy(), tw_bmlp2.copy(),
        _ones(Dh), _ones(Dh), D, F, Dh, ctx,
    )
    var dw = DoubleBlockWeights(iw^, tw^)
    var sw = SingleBlockWeights(
        s_w1.copy(), s_b1.copy(), s_w2.copy(), s_b2.copy(),
        _ones(Dh), _ones(Dh), D, F, Dh, ctx,
    )

    var weights = List[List[Float32]]()
    _append_stream_slots(weights, iw_wqkv, iw_wproj, iw_wmlp0, iw_wmlp2)
    _append_stream_slots(weights, tw_wqkv, tw_wproj, tw_wmlp0, tw_wmlp2)
    weights.append(_take_rows(s_w1, 0, D, D))
    weights.append(_take_rows(s_w1, D, D, D))
    weights.append(_take_rows(s_w1, 2 * D, D, D))
    weights.append(_take_rows(s_w1, 3 * D, F, D))
    weights.append(s_w2.copy())

    var base_dbl = double_block_forward[H, Dh, N_IMG, N_TXT, S](
        img.copy(), txt.copy(), dw, img_mod, txt_mod, cos, sin, D, F, EPS, ctx,
    )
    var base_dbl_g = double_block_backward[H, Dh, N_IMG, N_TXT, S](
        d_img.copy(), d_txt.copy(), dw, img_mod, txt_mod,
        base_dbl.saved, cos, sin, D, F, EPS, ctx,
    )
    var base_sgl = single_block_forward[H, Dh, S](
        sx.copy(), sw, smod, cos, sin, D, F, EPS, ctx,
    )
    var base_sgl_g = single_block_backward[H, Dh, S](
        d_s.copy(), sw, smod, base_sgl.saved, cos, sin, D, F, EPS, ctx,
    )

    var h = ParityHarness(0.995)
    var allok = True
    var single_base_slot = flux_direct_active_slot_count(1, 0, FLUX_LYCORIS_TGT_ALL)

    var dora = build_flux_direct_dora_set_from_weights(
        weights, 1, 1, D, F, RANK, ALPHA, FLUX_LYCORIS_TGT_ALL, UInt64(9000), False,
    )
    var dora_dbl = FluxDoubleBlockDirectLycoris(
        FLUX_DIRECT_ALGO_DORA, dora.copy(), empty_flat_direct_oft_set(),
        0, FLUX_DIRECT_TGT_ALL,
    )
    var dora_sgl = FluxSingleBlockDirectLycoris(
        FLUX_DIRECT_ALGO_DORA, dora.copy(), empty_flat_direct_oft_set(),
        single_base_slot, FLUX_DIRECT_TGT_ALL,
    )
    var dora_dbl_f = double_block_direct_lycoris_forward[H, Dh, N_IMG, N_TXT, S](
        img.copy(), txt.copy(), dw, img_mod, txt_mod, dora_dbl,
        cos, sin, D, F, EPS, ctx,
    )
    var dora_dbl_g = double_block_direct_lycoris_backward[H, Dh, N_IMG, N_TXT, S](
        d_img.copy(), d_txt.copy(), dw, img_mod, txt_mod, dora_dbl,
        dora_dbl_f.saved, cos, sin, D, F, EPS, ctx,
    )
    var dora_sgl_f = single_block_direct_lycoris_forward[H, Dh, S](
        sx.copy(), sw, smod, dora_sgl, cos, sin, D, F, EPS, ctx,
    )
    var dora_sgl_g = single_block_direct_lycoris_backward[H, Dh, S](
        d_s.copy(), sw, smod, dora_sgl, dora_sgl_f.saved, cos, sin, D, F, EPS, ctx,
    )
    print("[direct-dora] trainable_bytes=", flux_direct_dora_trainable_bytes(dora))
    _check(h, "dora double img_out", dora_dbl_f.img_out, base_dbl.img_out, allok)
    _check(h, "dora double txt_out", dora_dbl_f.txt_out, base_dbl.txt_out, allok)
    _check(h, "dora double d_img", dora_dbl_g.img.d_x, base_dbl_g.img.d_x, allok)
    _check(h, "dora double d_txt", dora_dbl_g.txt.d_x, base_dbl_g.txt.d_x, allok)
    _check(h, "dora single out", dora_sgl_f.out, base_sgl.out, allok)
    _check(h, "dora single d_x", dora_sgl_g.d_x, base_sgl_g.d_x, allok)
    var dora_l1 = _l1(dora_dbl_g.img.q.d_b) + _l1(dora_dbl_g.txt.proj.d_b) + _l1(dora_sgl_g.linear2.d_b)
    print("  dora selected_grad_l1=", dora_l1)
    if dora_l1 <= 0.0:
        allok = False

    var oft = build_flux_direct_oft_set(1, 1, D, F, OFT_BLOCK, FLUX_LYCORIS_TGT_ALL)
    var oft_dbl = FluxDoubleBlockDirectLycoris(
        FLUX_DIRECT_ALGO_OFT, empty_flat_direct_dora_set(), oft.copy(),
        0, FLUX_DIRECT_TGT_ALL,
    )
    var oft_sgl = FluxSingleBlockDirectLycoris(
        FLUX_DIRECT_ALGO_OFT, empty_flat_direct_dora_set(), oft.copy(),
        single_base_slot, FLUX_DIRECT_TGT_ALL,
    )
    var oft_dbl_f = double_block_direct_lycoris_forward[H, Dh, N_IMG, N_TXT, S](
        img.copy(), txt.copy(), dw, img_mod, txt_mod, oft_dbl,
        cos, sin, D, F, EPS, ctx,
    )
    var oft_dbl_g = double_block_direct_lycoris_backward[H, Dh, N_IMG, N_TXT, S](
        d_img.copy(), d_txt.copy(), dw, img_mod, txt_mod, oft_dbl,
        oft_dbl_f.saved, cos, sin, D, F, EPS, ctx,
    )
    var oft_sgl_f = single_block_direct_lycoris_forward[H, Dh, S](
        sx.copy(), sw, smod, oft_sgl, cos, sin, D, F, EPS, ctx,
    )
    var oft_sgl_g = single_block_direct_lycoris_backward[H, Dh, S](
        d_s.copy(), sw, smod, oft_sgl, oft_sgl_f.saved, cos, sin, D, F, EPS, ctx,
    )
    print("[direct-oft] trainable_bytes=", flux_direct_oft_trainable_bytes(oft))
    _check(h, "oft double img_out", oft_dbl_f.img_out, base_dbl.img_out, allok)
    _check(h, "oft double txt_out", oft_dbl_f.txt_out, base_dbl.txt_out, allok)
    _check(h, "oft double d_img", oft_dbl_g.img.d_x, base_dbl_g.img.d_x, allok)
    _check(h, "oft double d_txt", oft_dbl_g.txt.d_x, base_dbl_g.txt.d_x, allok)
    _check(h, "oft single out", oft_sgl_f.out, base_sgl.out, allok)
    _check(h, "oft single d_x", oft_sgl_g.d_x, base_sgl_g.d_x, allok)
    var oft_l1 = _l1(oft_dbl_g.img.q.d_vec) + _l1(oft_dbl_g.txt.proj.d_vec) + _l1(oft_sgl_g.linear2.d_vec)
    print("  oft selected_grad_l1=", oft_l1)
    if oft_l1 <= 0.0:
        allok = False

    if allok:
        print("ALL GATES PASS -- flux_block_direct_lycoris_identity_smoke")
    else:
        raise Error("flux_block_direct_lycoris_identity_smoke failed")
