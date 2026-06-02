# resblock_parity.mojo — GPU verification of the SDXL ResBlock fwd + bwd.
#
# GATE: forward `out` + backward `d_x` + every per-weight grad at cos >= 0.999
# vs PyTorch autograd (resblock_oracle.py -> resblock_ref.txt). Mirrors
# conv2d_bwd_parity / norm_bwd_parity: inputs are reproduced on-device with the
# SAME deterministic fills as the oracle; only the OUTPUT + GRADIENTS cross the
# boundary. Cin != Cout so the 1x1 skip conv path is exercised.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sdxl/parity/resblock_oracle.py
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/models/sdxl/parity/resblock_parity.mojo -o /tmp/sdxl_rb
#   /tmp/sdxl_rb

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.sdxl.weights import ResBlockWeights
from serenitymojo.models.sdxl.block import (
    resblock_forward, resblock_backward, ResBlockFwd, ResBlockGrads,
)
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/models/sdxl/parity/resblock_ref.txt"
)

# shape — MUST match resblock_oracle.py
comptime N = 2
comptime Hi = 8
comptime Wi = 8
comptime Cin = 64
comptime Cout = 128
comptime Eemb = 256
comptime G = 32


# ── deterministic fills (MUST match resblock_oracle.py) ──────────────────────
def _fill(n: Int, a: Int, b: Int, c: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * a) % b) - c) * 0.05)
    return out^

def _fill_x(n: Int) -> List[Float32]:    return _fill(n, 7, 13, 6.0)
def _fill_emb(n: Int) -> List[Float32]:  return _fill(n, 2, 7, 3.0)
def _fill_gnb(n: Int) -> List[Float32]:  return _fill(n, 3, 9, 4.0)
def _fill_conv(n: Int) -> List[Float32]: return _fill(n, 5, 11, 5.0)
def _fill_convb(n: Int) -> List[Float32]:return _fill(n, 4, 10, 5.0)
def _fill_embw(n: Int) -> List[Float32]: return _fill(n, 6, 17, 8.0)
def _fill_embb(n: Int) -> List[Float32]: return _fill(n, 3, 9, 4.0)
def _fill_go(n: Int) -> List[Float32]:   return _fill(n, 2, 7, 3.0)

def _fill_gnw(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 5) % 11) - 5.0) * 0.05 + 1.0)
    return out^


def _sh1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^

def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^

def _sh4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); s.append(c); s.append(d); return s^


# ── read one tagged float line from the ref (mirrors conv2d_bwd_parity) ──────
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


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True

    # ── build inputs + weights from the deterministic fills (F32, NHWC/RSCF) ──
    var x = Tensor.from_host(_fill_x(N * Hi * Wi * Cin), _sh4(N, Hi, Wi, Cin), STDtype.F32, ctx)
    var emb = Tensor.from_host(_fill_emb(N * Eemb), _sh2(N, Eemb), STDtype.F32, ctx)

    var gn1_w = Tensor.from_host(_fill_gnw(Cin), _sh1(Cin), STDtype.F32, ctx)
    var gn1_b = Tensor.from_host(_fill_gnb(Cin), _sh1(Cin), STDtype.F32, ctx)
    var conv1_w = Tensor.from_host(_fill_conv(3 * 3 * Cin * Cout), _sh4(3, 3, Cin, Cout), STDtype.F32, ctx)
    var conv1_b = Tensor.from_host(_fill_convb(Cout), _sh1(Cout), STDtype.F32, ctx)
    var emb_w = Tensor.from_host(_fill_embw(Cout * Eemb), _sh2(Cout, Eemb), STDtype.F32, ctx)
    var emb_b = Tensor.from_host(_fill_embb(Cout), _sh1(Cout), STDtype.F32, ctx)
    var gn2_w = Tensor.from_host(_fill_gnw(Cout), _sh1(Cout), STDtype.F32, ctx)
    var gn2_b = Tensor.from_host(_fill_gnb(Cout), _sh1(Cout), STDtype.F32, ctx)
    var conv2_w = Tensor.from_host(_fill_conv(3 * 3 * Cout * Cout), _sh4(3, 3, Cout, Cout), STDtype.F32, ctx)
    var conv2_b = Tensor.from_host(_fill_convb(Cout), _sh1(Cout), STDtype.F32, ctx)
    var skip_w = Tensor.from_host(_fill_conv(1 * 1 * Cin * Cout), _sh4(1, 1, Cin, Cout), STDtype.F32, ctx)
    var skip_b = Tensor.from_host(_fill_convb(Cout), _sh1(Cout), STDtype.F32, ctx)

    var w = ResBlockWeights(
        gn1_w^, gn1_b^, conv1_w^, conv1_b^, emb_w^, emb_b^,
        gn2_w^, gn2_b^, conv2_w^, conv2_b^, True, skip_w^, skip_b^,
    )

    # ── forward ──
    var fwd = resblock_forward[N, Hi, Wi, Cin, Cout, Eemb, G](x, emb, w, ctx)
    var out_host = fwd.out.to_host(ctx)
    var r_out = h.compare_host(out_host, _read_ref(String("out")))
    print("resblock out      vs torch:", r_out)
    all_pass = all_pass and r_out.passed

    # ── backward ──
    var go = Tensor.from_host(_fill_go(N * Hi * Wi * Cout), _sh4(N, Hi, Wi, Cout), STDtype.F32, ctx)
    var g = resblock_backward[N, Hi, Wi, Cin, Cout, Eemb, G](go, fwd.acts, w, ctx)

    var r_dx = h.compare_host(g.d_x.to_host(ctx), _read_ref(String("d_x")))
    var r_demb = h.compare_host(g.d_emb_in.to_host(ctx), _read_ref(String("d_emb_in")))
    var r_g1w = h.compare_host(g.d_gn1_w.to_host(ctx), _read_ref(String("d_gn1_w")))
    var r_g1b = h.compare_host(g.d_gn1_b.to_host(ctx), _read_ref(String("d_gn1_b")))
    var r_c1w = h.compare_host(g.d_conv1_w.to_host(ctx), _read_ref(String("d_conv1_w")))
    var r_c1b = h.compare_host(g.d_conv1_b.to_host(ctx), _read_ref(String("d_conv1_b")))
    var r_ew = h.compare_host(g.d_emb_w.to_host(ctx), _read_ref(String("d_emb_w")))
    var r_eb = h.compare_host(g.d_emb_b.to_host(ctx), _read_ref(String("d_emb_b")))
    var r_g2w = h.compare_host(g.d_gn2_w.to_host(ctx), _read_ref(String("d_gn2_w")))
    var r_g2b = h.compare_host(g.d_gn2_b.to_host(ctx), _read_ref(String("d_gn2_b")))
    var r_c2w = h.compare_host(g.d_conv2_w.to_host(ctx), _read_ref(String("d_conv2_w")))
    var r_c2b = h.compare_host(g.d_conv2_b.to_host(ctx), _read_ref(String("d_conv2_b")))
    var r_skw = h.compare_host(g.d_skip_w.to_host(ctx), _read_ref(String("d_skip_w")))
    var r_skb = h.compare_host(g.d_skip_b.to_host(ctx), _read_ref(String("d_skip_b")))

    print("resblock d_x      vs torch:", r_dx)
    print("resblock d_emb_in vs torch:", r_demb)
    print("resblock d_gn1_w  vs torch:", r_g1w)
    print("resblock d_gn1_b  vs torch:", r_g1b)
    print("resblock d_conv1_w vs torch:", r_c1w)
    print("resblock d_conv1_b vs torch:", r_c1b)
    print("resblock d_emb_w  vs torch:", r_ew)
    print("resblock d_emb_b  vs torch:", r_eb)
    print("resblock d_gn2_w  vs torch:", r_g2w)
    print("resblock d_gn2_b  vs torch:", r_g2b)
    print("resblock d_conv2_w vs torch:", r_c2w)
    print("resblock d_conv2_b vs torch:", r_c2b)
    print("resblock d_skip_w vs torch:", r_skw)
    print("resblock d_skip_b vs torch:", r_skb)

    all_pass = (all_pass and r_dx.passed and r_demb.passed
        and r_g1w.passed and r_g1b.passed and r_c1w.passed and r_c1b.passed
        and r_ew.passed and r_eb.passed and r_g2w.passed and r_g2b.passed
        and r_c2w.passed and r_c2b.passed and r_skw.passed and r_skb.passed)

    print("")
    if all_pass:
        print("ALL SDXL RESBLOCK FWD+BWD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("SDXL RESBLOCK PARITY FAILURE")
        raise Error("resblock_parity gate failed")
