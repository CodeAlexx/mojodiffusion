# unet_stack_parity.mojo — GPU composition gate for the SDXL FULL conv-UNet stack
# fwd+bwd (models/sdxl/sdxl_unet_stack.mojo) vs torch autograd
# (unet_stack_oracle.py -> unet_stack_ref.txt).
#
# GATE: out + d_x + d_context + d_y + representative weight grads (conv_in,
# conv_out, a deep ResBlock conv1, a SpatialTransformer attn2 to_q, an output
# ResBlock conv2, the time/label embed linears) at cos >= 0.999.
#
# Inputs/weights reproduced on-device with the SAME deterministic fills as the
# oracle (only out + grads cross the boundary).
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sdxl/parity/unet_stack_oracle.py
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/models/sdxl/parity/unet_stack_parity.mojo -o /tmp/sdxl_stack_parity
#   /tmp/sdxl_stack_parity

from std.gpu.host import DeviceContext
from std.memory import ArcPointer, alloc
from std.collections import Optional
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY

from serenitymojo.models.sdxl.weights import ResBlockWeights
from serenitymojo.models.sdxl.embed import EmbWeights
from serenitymojo.models.sdxl.spatial_transformer import (
    SpatialTransformerWeights, BasicTransformerBlockWeights, AttnWeights,
)
from serenitymojo.models.sdxl.sdxl_unet_stack import (
    SdxlStackWeightsReduced, sdxl_unet_stack_forward_reduced,
    sdxl_unet_stack_backward_reduced,
)

comptime TArc = ArcPointer[Tensor]
comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/models/sdxl/parity/unet_stack_ref.txt"
)

# dims — MUST match unet_stack_oracle.py
comptime B = 1
comptime H0 = 8
comptime W0 = 8
comptime IN_CH = 4
comptime MC = 16
comptime OUT_CH = 4
comptime TEMB = 32
comptime SDIM = 16
comptime ADM = 24
comptime NKV = 7
comptime CCTX = 16


# ── deterministic fill (matches oracle fill(): (i*a % b - c) * scale) ─────────
def _fill(n: Int, a: Int, b: Int, c: Float32, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * a) % b) - c) * scale)
    return out^

def _sh1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^
def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^

def _t(n: Int, a: Int, b: Int, c: Float32, var sh: List[Int], scale: Float32,
       ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(_fill(n, a, b, c, scale), sh^, STDtype.F32, ctx)

def _ta(n: Int, a: Int, b: Int, c: Float32, var sh: List[Int], scale: Float32,
        ctx: DeviceContext) raises -> TArc:
    return TArc(_t(n, a, b, c, sh^, scale, ctx))


# ── ref reader (same as spatial_transformer_parity) ──────────────────────────
def _read_ref(tag: String) raises -> List[Float32]:
    var fd = sys_open(String(REF_PATH), O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ref: ") + String(REF_PATH))
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error("empty ref file")
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var prefix = tag + " "
    var pl = prefix.byte_length()
    var out = List[Float32]()
    var i = 0
    while i < done:
        var le = i
        while le < done and Int(buf[le]) != 0x0A:
            le += 1
        var is_match = (le - i) > pl
        if is_match:
            for j in range(pl):
                if Int(buf[i + j]) != ord(prefix[byte=j]):
                    is_match = False
                    break
        if is_match:
            var p = i + pl
            while p < le:
                var c = Int(buf[p])
                if c == 0x20:
                    p += 1
                    continue
                var ne = p
                while ne < le and Int(buf[ne]) != 0x20:
                    ne += 1
                var chars = List[UInt8]()
                for q in range(p, ne):
                    chars.append(buf[q])
                var s = String(from_utf8=chars)
                out.append(Float32(atof(s)))
                p = ne + 1
            buf.free()
            return out^
        i = le + 1
    buf.free()
    raise Error(String("ref tag not found: ") + tag)


# ── build a ResBlockWeights mirroring oracle alloc_resblock(pfx,cin,cout,ff) ──
def _make_rb(cin: Int, cout: Int, ff: Int, ctx: DeviceContext) raises -> ResBlockWeights:
    var s = ff
    var gn1_w = _t(cin, 5, 11, 5.0, _sh1(cin), 0.05, ctx)
    var gn1_b = _t(cin, 3, 9, 4.0, _sh1(cin), 0.05, ctx)
    var conv1_w = _t(3 * 3 * cin * cout, 5 + s, 11, 5.0, _sh4(3, 3, cin, cout), 0.02, ctx)
    var conv1_b = _t(cout, 4, 10, 5.0, _sh1(cout), 0.05, ctx)
    var emb_w = _t(cout * TEMB, 6 + s, 17, 8.0, _sh2(cout, TEMB), 0.02, ctx)
    var emb_b = _t(cout, 3, 9, 4.0, _sh1(cout), 0.05, ctx)
    var gn2_w = _t(cout, 5, 11, 5.0, _sh1(cout), 0.05, ctx)
    var gn2_b = _t(cout, 3, 9, 4.0, _sh1(cout), 0.05, ctx)
    var conv2_w = _t(3 * 3 * cout * cout, 6 + s, 11, 5.0, _sh4(3, 3, cout, cout), 0.02, ctx)
    var conv2_b = _t(cout, 4, 10, 5.0, _sh1(cout), 0.05, ctx)
    if cin != cout:
        var skip_w = _t(1 * 1 * cin * cout, 7 + s, 11, 5.0, _sh4(1, 1, cin, cout), 0.02, ctx)
        var skip_b = _t(cout, 4, 10, 5.0, _sh1(cout), 0.05, ctx)
        return ResBlockWeights(
            gn1_w^, gn1_b^, conv1_w^, conv1_b^, emb_w^, emb_b^,
            gn2_w^, gn2_b^, conv2_w^, conv2_b^, True, skip_w^, skip_b^)
    var ph1 = _t(1, 0, 1, 0.0, _sh1(1), 0.0, ctx)
    var ph2 = _t(1, 0, 1, 0.0, _sh1(1), 0.0, ctx)
    return ResBlockWeights(
        gn1_w^, gn1_b^, conv1_w^, conv1_b^, emb_w^, emb_b^,
        gn2_w^, gn2_b^, conv2_w^, conv2_b^, False, ph1^, ph2^)


# ── build one BasicTransformerBlockWeights mirroring oracle (s=j+1+ff) ───────
def _make_btb(C: Int, j: Int, ff: Int, ctx: DeviceContext) raises -> BasicTransformerBlockWeights:
    var s = j + 1 + ff
    var Cff = 2 * C
    var attn1 = AttnWeights(
        _ta(C * C, 5 + s, 13, 6.0, _sh2(C, C), 0.02, ctx),
        _ta(C * C, 6 + s, 13, 6.0, _sh2(C, C), 0.02, ctx),
        _ta(C * C, 7 + s, 13, 6.0, _sh2(C, C), 0.02, ctx),
        _ta(C * C, 8 + s, 13, 6.0, _sh2(C, C), 0.02, ctx),
        _ta(C, 4, 11, 5.0, _sh1(C), 0.05, ctx),
    )
    var attn2 = AttnWeights(
        _ta(C * C, 5 + s, 17, 8.0, _sh2(C, C), 0.02, ctx),
        _ta(C * CCTX, 6 + s, 17, 8.0, _sh2(C, CCTX), 0.02, ctx),
        _ta(C * CCTX, 7 + s, 17, 8.0, _sh2(C, CCTX), 0.02, ctx),
        _ta(C * C, 8 + s, 17, 8.0, _sh2(C, C), 0.02, ctx),
        _ta(C, 4, 11, 5.0, _sh1(C), 0.05, ctx),
    )
    return BasicTransformerBlockWeights(
        _ta(C, 3, 9, 4.0, _sh1(C), 0.05, ctx), _ta(C, 2, 7, 3.0, _sh1(C), 0.05, ctx),
        attn1^,
        _ta(C, 3, 9, 4.0, _sh1(C), 0.05, ctx), _ta(C, 2, 7, 3.0, _sh1(C), 0.05, ctx),
        attn2^,
        _ta(C, 3, 9, 4.0, _sh1(C), 0.05, ctx), _ta(C, 2, 7, 3.0, _sh1(C), 0.05, ctx),
        _ta(Cff * C, 5 + s, 13, 6.0, _sh2(Cff, C), 0.02, ctx),
        _ta(Cff, 4, 10, 5.0, _sh1(Cff), 0.05, ctx),
        _ta(C * (Cff // 2), 6 + s, 13, 6.0, _sh2(C, Cff // 2), 0.02, ctx),
        _ta(C, 3, 8, 3.0, _sh1(C), 0.05, ctx),
    )


# ── build a SpatialTransformerWeights (depth 1) mirroring oracle alloc_st ─────
def _make_st(C: Int, ff: Int, ctx: DeviceContext) raises -> SpatialTransformerWeights:
    var blocks = List[BasicTransformerBlockWeights]()
    blocks.append(_make_btb(C, 0, ff, ctx))
    return SpatialTransformerWeights(
        _ta(C, 3, 9, 4.0, _sh1(C), 0.05, ctx), _ta(C, 2, 7, 3.0, _sh1(C), 0.05, ctx),
        _ta(C * C, 5 + ff, 13, 6.0, _sh2(C, C), 0.02, ctx),
        _ta(C, 4, 11, 5.0, _sh1(C), 0.05, ctx),
        blocks^,
        _ta(C * C, 6 + ff, 17, 8.0, _sh2(C, C), 0.02, ctx),
        _ta(C, 3, 8, 3.0, _sh1(C), 0.05, ctx),
    )


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True

    # ── inputs ──
    var x = _t(B * H0 * W0 * IN_CH, 7, 13, 6.0, _sh4(B, H0, W0, IN_CH), 0.05, ctx)
    var context3 = _t(B * NKV * CCTX, 5, 11, 5.0, _sh3b(B, NKV, CCTX), 0.05, ctx)
    # timesteps t = fill(B,3,100,0,scale=1.0)
    var t = _t(B, 3, 100, 0.0, _sh1(B), 1.0, ctx)
    var y = _t(B * ADM, 4, 13, 6.0, _sh2(B, ADM), 0.05, ctx)

    # ── embed weights ──
    var emb_w = EmbWeights(
        _t(TEMB * SDIM, 5, 13, 6.0, _sh2(TEMB, SDIM), 0.02, ctx),
        _t(TEMB, 4, 11, 5.0, _sh1(TEMB), 0.05, ctx),
        _t(TEMB * TEMB, 6, 17, 8.0, _sh2(TEMB, TEMB), 0.02, ctx),
        _t(TEMB, 3, 9, 4.0, _sh1(TEMB), 0.05, ctx),
        _t(TEMB * ADM, 5, 13, 6.0, _sh2(TEMB, ADM), 0.02, ctx),
        _t(TEMB, 4, 11, 5.0, _sh1(TEMB), 0.05, ctx),
        _t(TEMB * TEMB, 6, 17, 8.0, _sh2(TEMB, TEMB), 0.02, ctx),
        _t(TEMB, 3, 9, 4.0, _sh1(TEMB), 0.05, ctx),
    )

    # ── conv_in / conv_out / final GN ──
    var conv_in_w = _t(3 * 3 * IN_CH * MC, 5, 11, 5.0, _sh4(3, 3, IN_CH, MC), 0.02, ctx)
    var conv_in_b = _t(MC, 4, 10, 5.0, _sh1(MC), 0.05, ctx)
    var out_gn_w = _t(MC, 5, 11, 5.0, _sh1(MC), 0.05, ctx)
    var out_gn_b = _t(MC, 3, 9, 4.0, _sh1(MC), 0.05, ctx)
    var conv_out_w = _t(3 * 3 * MC * OUT_CH, 6, 11, 5.0, _sh4(3, 3, MC, OUT_CH), 0.02, ctx)
    var conv_out_b = _t(OUT_CH, 4, 10, 5.0, _sh1(OUT_CH), 0.05, ctx)

    # ── per-block weights (ff seeds match oracle order) ──
    var in1 = _make_rb(16, 16, 1, ctx)
    var in3_st = _make_st(32, 1, ctx); var in3 = _make_rb(16, 32, 2, ctx)
    var in5_st = _make_st(64, 1, ctx); var in5 = _make_rb(32, 64, 3, ctx)
    var in2_op_w = _t(3 * 3 * 16 * 16, 5, 11, 5.0, _sh4(3, 3, 16, 16), 0.02, ctx)
    var in2_op_b = _t(16, 4, 10, 5.0, _sh1(16), 0.05, ctx)
    var in4_op_w = _t(3 * 3 * 32 * 32, 6, 11, 5.0, _sh4(3, 3, 32, 32), 0.02, ctx)
    var in4_op_b = _t(32, 4, 10, 5.0, _sh1(32), 0.05, ctx)
    var mid0 = _make_rb(64, 64, 4, ctx)
    var mid_st = _make_st(64, 5, ctx)
    var mid2 = _make_rb(64, 64, 6, ctx)
    var out0 = _make_rb(128, 64, 7, ctx); var out0_st = _make_st(64, 7, ctx)
    var out1 = _make_rb(96, 64, 8, ctx); var out1_st = _make_st(64, 8, ctx)
    var out1_up_w = _t(3 * 3 * 64 * 64, 5, 11, 5.0, _sh4(3, 3, 64, 64), 0.02, ctx)
    var out1_up_b = _t(64, 4, 10, 5.0, _sh1(64), 0.05, ctx)
    var out2 = _make_rb(96, 32, 9, ctx)
    var out3 = _make_rb(48, 32, 10, ctx)
    var out3_up_w = _t(3 * 3 * 32 * 32, 6, 11, 5.0, _sh4(3, 3, 32, 32), 0.02, ctx)
    var out3_up_b = _t(32, 4, 10, 5.0, _sh1(32), 0.05, ctx)
    var out4 = _make_rb(48, 16, 11, ctx)
    var out5 = _make_rb(32, 16, 12, ctx)

    var w = SdxlStackWeightsReduced(
        emb_w^, conv_in_w^, conv_in_b^, out_gn_w^, out_gn_b^, conv_out_w^, conv_out_b^,
        in1^, in3^, in3_st^, in5^, in5_st^,
        in2_op_w^, in2_op_b^, in4_op_w^, in4_op_b^,
        mid0^, mid_st^, mid2^,
        out0^, out0_st^, out1^, out1_st^, out1_up_w^, out1_up_b^,
        out2^, out3^, out3_up_w^, out3_up_b^, out4^, out5^,
    )

    # ── forward ──
    var fwd = sdxl_unet_stack_forward_reduced(x.clone(ctx), t.clone(ctx), y.clone(ctx),
                                              context3.clone(ctx), w, ctx)
    var r_out = h.compare_host(fwd.out.to_host(ctx), _read_ref(String("out")))
    print("UNet stack out        vs torch:", r_out)
    all_pass = all_pass and r_out.passed

    # ── backward ──
    var go = _t(B * H0 * W0 * OUT_CH, 2, 7, 3.0, _sh4(B, H0, W0, OUT_CH), 0.05, ctx)
    var g = sdxl_unet_stack_backward_reduced(
        go, x.clone(ctx), t.clone(ctx), y.clone(ctx), context3.clone(ctx), fwd.acts, w, ctx)

    var r_dx = h.compare_host(g.d_x.to_host(ctx), _read_ref(String("d_x")))
    var r_dctx = h.compare_host(g.d_context.to_host(ctx), _read_ref(String("d_context")))
    var r_dy = h.compare_host(g.d_y.to_host(ctx), _read_ref(String("d_y")))
    var r_cin = h.compare_host(g.d_conv_in_w.to_host(ctx), _read_ref(String("d_conv_in_w")))
    var r_cout = h.compare_host(g.d_conv_out_w.to_host(ctx), _read_ref(String("d_conv_out_w")))
    var r_mid = h.compare_host(g.d_mid0_conv1_w.to_host(ctx), _read_ref(String("d_mid0_conv1_w")))
    var r_q2 = h.compare_host(g.d_in5_st_q2.to_host(ctx), _read_ref(String("d_in5_st_b0_q2")))
    var r_o0c = h.compare_host(g.d_out0_conv2_w.to_host(ctx), _read_ref(String("d_out0_conv2_w")))
    var r_t0 = h.compare_host(g.d_t0_w.to_host(ctx), _read_ref(String("d_t0_w")))
    var r_l0 = h.compare_host(g.d_l0_w.to_host(ctx), _read_ref(String("d_l0_w")))

    print("UNet stack d_x        vs torch:", r_dx)
    print("UNet stack d_context  vs torch:", r_dctx)
    print("UNet stack d_y        vs torch:", r_dy)
    print("UNet stack d_conv_in_w   vs torch:", r_cin)
    print("UNet stack d_conv_out_w  vs torch:", r_cout)
    print("UNet stack d_mid0_conv1_w vs torch:", r_mid, "  (deep ResBlock)")
    print("UNet stack d_in5_st_q2   vs torch:", r_q2, "  (SpatialTransformer attn2 to_q)")
    print("UNet stack d_out0_conv2_w vs torch:", r_o0c)
    print("UNet stack d_t0_w        vs torch:", r_t0)
    print("UNet stack d_l0_w        vs torch:", r_l0)

    all_pass = (all_pass and r_dx.passed and r_dctx.passed and r_dy.passed
        and r_cin.passed and r_cout.passed and r_mid.passed and r_q2.passed
        and r_o0c.passed and r_t0.passed and r_l0.passed)

    print("")
    if all_pass:
        print("ALL SDXL UNET-STACK COMPOSITION FWD+BWD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("SDXL UNET-STACK COMPOSITION PARITY FAILURE")
        raise Error("unet_stack_parity gate failed")


def _sh3b(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^
