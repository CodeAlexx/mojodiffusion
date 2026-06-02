# spatial_transformer_parity.mojo — GPU gate for the SDXL SpatialTransformer
# cross-attn block fwd+bwd (models/sdxl/spatial_transformer.mojo) vs torch
# autograd (spatial_transformer_oracle.py). GATE: out + d_x + d_context + every
# weight grad at cos >= 0.999.
#
# The attn2 (cross-attn) Q/K/V/out grads + d_context are routed through the SHARED
# rectangular SDPA backward (sdpa_backward_rect, Sq=N=16 != Skv=77) — they are the
# key proof the shared primitive integrates correctly. attn1 (self) routes through
# the square sdpa_backward (Sq==Skv==16).
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sdxl/parity/spatial_transformer_oracle.py
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/models/sdxl/parity/spatial_transformer_parity.mojo -o /tmp/sdxl_st_parity
#   /tmp/sdxl_st_parity

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.sdxl.spatial_transformer import (
    spatial_transformer_forward, spatial_transformer_backward,
    SpatialTransformerWeights, BasicTransformerBlockWeights, AttnWeights,
)
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc

comptime TArc = ArcPointer[Tensor]
comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/models/sdxl/parity/spatial_transformer_ref_d2.txt"
)

# dims — MUST match spatial_transformer_oracle.py
comptime B = 1
comptime H = 4
comptime W = 4
comptime C = 128
comptime Dh = 64
comptime Hh = C // Dh    # 2
comptime N = H * W       # 16
comptime Nkv = 77
comptime Cctx = 16
comptime Cff = 32
comptime G = 32
comptime DEPTH = 2


# ── fills (identical to the oracle's fill()) ─────────────────────────────────
def _fill(n: Int, a: Int, b: Int, c: Float32, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * a) % b) - c) * scale)
    return out^

def _sh1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^
def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^
def _sh3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); return s^
def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^

def _t(n: Int, a: Int, b: Int, c: Float32, var sh: List[Int], scale: Float32,
       ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(_fill(n, a, b, c, scale), sh^, STDtype.F32, ctx)

def _ta(n: Int, a: Int, b: Int, c: Float32, var sh: List[Int], scale: Float32,
        ctx: DeviceContext) raises -> TArc:
    return TArc(_t(n, a, b, c, sh^, scale, ctx))


# ── ref reader (same as geglu_parity) ────────────────────────────────────────
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


# ── build one block's weights (mirror oracle bw dict, s = j+1) ───────────────
def _make_block(j: Int, ctx: DeviceContext) raises -> BasicTransformerBlockWeights:
    var s = j + 1
    var attn1 = AttnWeights(
        _ta(C * C, 5 + s, 13, 6.0, _sh2(C, C), 0.02, ctx),    # q1
        _ta(C * C, 6 + s, 13, 6.0, _sh2(C, C), 0.02, ctx),    # k1
        _ta(C * C, 7 + s, 13, 6.0, _sh2(C, C), 0.02, ctx),    # v1
        _ta(C * C, 8 + s, 13, 6.0, _sh2(C, C), 0.02, ctx),    # o1
        _ta(C, 4, 11, 5.0, _sh1(C), 0.05, ctx),               # o1b
    )
    var attn2 = AttnWeights(
        _ta(C * C, 5 + s, 17, 8.0, _sh2(C, C), 0.02, ctx),       # q2
        _ta(C * Cctx, 6 + s, 17, 8.0, _sh2(C, Cctx), 0.02, ctx), # k2 [C,Cctx]
        _ta(C * Cctx, 7 + s, 17, 8.0, _sh2(C, Cctx), 0.02, ctx), # v2 [C,Cctx]
        _ta(C * C, 8 + s, 17, 8.0, _sh2(C, C), 0.02, ctx),       # o2
        _ta(C, 4, 11, 5.0, _sh1(C), 0.05, ctx),                 # o2b
    )
    return BasicTransformerBlockWeights(
        _ta(C, 3, 9, 4.0, _sh1(C), 0.05, ctx),                  # n1w
        _ta(C, 2, 7, 3.0, _sh1(C), 0.05, ctx),                  # n1b
        attn1^,
        _ta(C, 3, 9, 4.0, _sh1(C), 0.05, ctx),                  # n2w
        _ta(C, 2, 7, 3.0, _sh1(C), 0.05, ctx),                  # n2b
        attn2^,
        _ta(C, 3, 9, 4.0, _sh1(C), 0.05, ctx),                  # n3w
        _ta(C, 2, 7, 3.0, _sh1(C), 0.05, ctx),                  # n3b
        _ta(2 * Cff * C, 5 + s, 13, 6.0, _sh2(2 * Cff, C), 0.02, ctx),  # fpw
        _ta(2 * Cff, 4, 10, 5.0, _sh1(2 * Cff), 0.05, ctx),     # fpb
        _ta(C * Cff, 6 + s, 13, 6.0, _sh2(C, Cff), 0.02, ctx),  # fow
        _ta(C, 3, 8, 3.0, _sh1(C), 0.05, ctx),                  # fob
    )


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True

    # ── inputs (NHWC x; context [B,Nkv,Cctx]) ──
    var x = _t(B * H * W * C, 7, 13, 6.0, _sh4(B, H, W, C), 0.05, ctx)
    var context = _t(B * Nkv * Cctx, 5, 11, 5.0, _sh3(B, Nkv, Cctx), 0.05, ctx)

    # ── ST-level weights ──
    var gn_w = _ta(C, 3, 9, 4.0, _sh1(C), 0.05, ctx)
    var gn_b = _ta(C, 2, 7, 3.0, _sh1(C), 0.05, ctx)
    var proj_in_w = _ta(C * C, 5, 13, 6.0, _sh2(C, C), 0.02, ctx)
    var proj_in_b = _ta(C, 4, 11, 5.0, _sh1(C), 0.05, ctx)
    var proj_out_w = _ta(C * C, 6, 17, 8.0, _sh2(C, C), 0.02, ctx)
    var proj_out_b = _ta(C, 3, 8, 3.0, _sh1(C), 0.05, ctx)

    var blocks = List[BasicTransformerBlockWeights]()
    for j in range(DEPTH):
        blocks.append(_make_block(j, ctx))

    var w = SpatialTransformerWeights(
        gn_w^, gn_b^, proj_in_w^, proj_in_b^, blocks^, proj_out_w^, proj_out_b^)

    # ── forward ──
    var fwd = spatial_transformer_forward[B, H, W, C, Nkv, Cctx, Hh, Dh, Cff, G, DEPTH](
        x.clone(ctx), context.clone(ctx), w, ctx)
    var r_out = h.compare_host(fwd.out.to_host(ctx), _read_ref(String("out")))
    print("ST out          vs torch:", r_out)
    all_pass = all_pass and r_out.passed

    # ── backward (go = NHWC seed) ──
    var go = _t(B * H * W * C, 2, 7, 3.0, _sh4(B, H, W, C), 0.05, ctx)
    var g = spatial_transformer_backward[B, H, W, C, Nkv, Cctx, Hh, Dh, Cff, G, DEPTH](
        go, fwd.acts, w, ctx)

    var r_dx = h.compare_host(g.d_x.to_host(ctx), _read_ref(String("d_x")))
    print("ST d_x          vs torch:", r_dx)
    all_pass = all_pass and r_dx.passed
    var r_dctx = h.compare_host(g.d_context.to_host(ctx), _read_ref(String("d_context")))
    print("ST d_context    vs torch:", r_dctx, "  (<- rect-SDPA bwd proof)")
    all_pass = all_pass and r_dctx.passed

    var r_gnw = h.compare_host(g.d_gn_w.to_host(ctx), _read_ref(String("d_gn_w")))
    var r_gnb = h.compare_host(g.d_gn_b.to_host(ctx), _read_ref(String("d_gn_b")))
    var r_piw = h.compare_host(g.d_proj_in_w.to_host(ctx), _read_ref(String("d_proj_in_w")))
    var r_pib = h.compare_host(g.d_proj_in_b.to_host(ctx), _read_ref(String("d_proj_in_b")))
    var r_pow = h.compare_host(g.d_proj_out_w.to_host(ctx), _read_ref(String("d_proj_out_w")))
    var r_pob = h.compare_host(g.d_proj_out_b.to_host(ctx), _read_ref(String("d_proj_out_b")))
    print("ST d_gn_w       vs torch:", r_gnw)
    print("ST d_gn_b       vs torch:", r_gnb)
    print("ST d_proj_in_w  vs torch:", r_piw)
    print("ST d_proj_in_b  vs torch:", r_pib)
    print("ST d_proj_out_w vs torch:", r_pow)
    print("ST d_proj_out_b vs torch:", r_pob)
    all_pass = (all_pass and r_gnw.passed and r_gnb.passed and r_piw.passed
                and r_pib.passed and r_pow.passed and r_pob.passed)

    # ── per-block weight grads ──
    for j in range(DEPTH):
        var p = String("b") + String(j) + String("_")
        ref bg = g.block_grads[j]
        var r_n1w = h.compare_host(bg.d_norm1_w[].to_host(ctx), _read_ref(p + String("d_n1w")))
        var r_n1b = h.compare_host(bg.d_norm1_b[].to_host(ctx), _read_ref(p + String("d_n1b")))
        var r_q1 = h.compare_host(bg.a1.d_to_q_w[].to_host(ctx), _read_ref(p + String("d_q1")))
        var r_k1 = h.compare_host(bg.a1.d_to_k_w[].to_host(ctx), _read_ref(p + String("d_k1")))
        var r_v1 = h.compare_host(bg.a1.d_to_v_w[].to_host(ctx), _read_ref(p + String("d_v1")))
        var r_o1 = h.compare_host(bg.a1.d_to_out_w[].to_host(ctx), _read_ref(p + String("d_o1")))
        var r_o1b = h.compare_host(bg.a1.d_to_out_b[].to_host(ctx), _read_ref(p + String("d_o1b")))
        var r_n2w = h.compare_host(bg.d_norm2_w[].to_host(ctx), _read_ref(p + String("d_n2w")))
        var r_n2b = h.compare_host(bg.d_norm2_b[].to_host(ctx), _read_ref(p + String("d_n2b")))
        var r_q2 = h.compare_host(bg.a2.d_to_q_w[].to_host(ctx), _read_ref(p + String("d_q2")))
        var r_k2 = h.compare_host(bg.a2.d_to_k_w[].to_host(ctx), _read_ref(p + String("d_k2")))
        var r_v2 = h.compare_host(bg.a2.d_to_v_w[].to_host(ctx), _read_ref(p + String("d_v2")))
        var r_o2 = h.compare_host(bg.a2.d_to_out_w[].to_host(ctx), _read_ref(p + String("d_o2")))
        var r_o2b = h.compare_host(bg.a2.d_to_out_b[].to_host(ctx), _read_ref(p + String("d_o2b")))
        var r_n3w = h.compare_host(bg.d_norm3_w[].to_host(ctx), _read_ref(p + String("d_n3w")))
        var r_n3b = h.compare_host(bg.d_norm3_b[].to_host(ctx), _read_ref(p + String("d_n3b")))
        var r_fpw = h.compare_host(bg.d_ff_proj_w[].to_host(ctx), _read_ref(p + String("d_fpw")))
        var r_fpb = h.compare_host(bg.d_ff_proj_b[].to_host(ctx), _read_ref(p + String("d_fpb")))
        var r_fow = h.compare_host(bg.d_ff_out_w[].to_host(ctx), _read_ref(p + String("d_fow")))
        var r_fob = h.compare_host(bg.d_ff_out_b[].to_host(ctx), _read_ref(p + String("d_fob")))
        print("  ", p, "d_n1w", r_n1w, " d_n1b", r_n1b)
        print("  ", p, "d_q1 ", r_q1, " d_k1 ", r_k1, " d_v1 ", r_v1, " d_o1 ", r_o1, " d_o1b", r_o1b)
        print("  ", p, "d_n2w", r_n2w, " d_n2b", r_n2b)
        print("  ", p, "d_q2 ", r_q2, " d_k2 ", r_k2, " d_v2 ", r_v2, " d_o2 ", r_o2, " d_o2b", r_o2b, "  (<- rect-SDPA bwd)")
        print("  ", p, "d_n3w", r_n3w, " d_n3b", r_n3b)
        print("  ", p, "d_fpw", r_fpw, " d_fpb", r_fpb, " d_fow", r_fow, " d_fob", r_fob)
        all_pass = (all_pass and r_n1w.passed and r_n1b.passed and r_q1.passed
            and r_k1.passed and r_v1.passed and r_o1.passed and r_o1b.passed
            and r_n2w.passed and r_n2b.passed and r_q2.passed and r_k2.passed
            and r_v2.passed and r_o2.passed and r_o2b.passed and r_n3w.passed
            and r_n3b.passed and r_fpw.passed and r_fpb.passed and r_fow.passed
            and r_fob.passed)

    print("")
    if all_pass:
        print("ALL SDXL SPATIAL-TRANSFORMER (DEPTH=2) FWD+BWD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("SDXL SPATIAL-TRANSFORMER PARITY FAILURE")
        raise Error("spatial_transformer_parity gate failed")
