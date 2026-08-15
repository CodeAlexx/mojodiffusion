# serenitymojo/models/wan22/parity/bernini_src_id_rope_parity.mojo
#
# PARITY GATE for the Bernini-R source-id RoPE
# (models/wan22/bernini_src_id_rope.mojo). Builds the STANDARD Wan 3D rope
# (wan22_build_rope) at REAL head_dim=128 on a small (ppf,pph,ppw) grid, applies
# the source-id rotation for source_id in {0, 2, 1.5}, and compares the rotated
# cos AND sin tables against the torch/diffusers reference dumped by
# bernini_src_id_rope_oracle.py at cos >= 0.999.
#
# Also asserts the IDENTITY property: source_id=0 rotation is BIT-identical to
# the stock wan rope table (max_abs == 0), the defining invariant.
#
# Run (oracle FIRST, SEPARATE command):
#   /home/alex/torchref-image/venv/bin/python \
#       serenitymojo/models/wan22/parity/bernini_src_id_rope_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       serenitymojo/models/wan22/parity/bernini_src_id_rope_parity.mojo

from max.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.wan22_dit import wan22_build_rope
from serenitymojo.models.wan22.bernini_src_id_rope import (
    apply_src_id_rope_rotation,
    build_bernini_src_id_rope,
    BerniniRopeSegment,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/wan22/parity/"

# MUST match bernini_src_id_rope_oracle.py
comptime HEAD_DIM = 128
comptime THETA = Float32(10000.0)
comptime PPF = 1
comptime PPH = 2
comptime PPW = 3
comptime S = PPF * PPH * PPW      # 6
comptime HALF = HEAD_DIM // 2     # 64


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


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


# Max |a - b| between two equal-length host lists (for the bit-identity assert).
def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("length mismatch in _max_abs_diff")
    var m = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def main() raises:
    var ctx = DeviceContext()
    print("==== bernini_src_id_rope_parity (Bernini-R source-id RoPE vs torch) ====")
    print("HEAD_DIM=", HEAD_DIM, " S=", S, " HALF=", HALF,
          " grid(f,h,w)=(", PPF, ",", PPH, ",", PPW, ")  theta=", THETA)

    # Standard Wan 3D rope (stock — the shared trainer builder).
    var stock = wan22_build_rope(PPF, PPH, PPW, HEAD_DIM, THETA, STDtype.F32, ctx)

    var harness = ParityHarness()
    var allok = True

    # ── Stock table vs torch (no source-id) ──
    print("")
    print("---- stock wan rope vs torch (sanity) ----")
    _check(harness, "stock_cos", stock[0].to_host(ctx), _in("ref_stock_cos"), allok)
    _check(harness, "stock_sin", stock[1].to_host(ctx), _in("ref_stock_sin"), allok)

    # ── source_id = 0 (identity) ──
    print("")
    print("---- source_id = 0 (identity) ----")
    var rot0 = apply_src_id_rope_rotation(stock[0], stock[1], 0.0, THETA, ctx)
    var rot0_cos = rot0[0].to_host(ctx)
    var rot0_sin = rot0[1].to_host(ctx)
    _check(harness, "cos sid=0 vs torch", rot0_cos, _in("ref_cos_0"), allok)
    _check(harness, "sin sid=0 vs torch", rot0_sin, _in("ref_sin_0"), allok)

    # Identity invariant: sid=0 rotation MUST be bit-identical to stock.
    var stock_cos_h = stock[0].to_host(ctx)
    var stock_sin_h = stock[1].to_host(ctx)
    var d_cos = _max_abs_diff(rot0_cos, stock_cos_h)
    var d_sin = _max_abs_diff(rot0_sin, stock_sin_h)
    print("  IDENTITY max|rot0_cos - stock_cos| =", d_cos,
          "  max|rot0_sin - stock_sin| =", d_sin,
          "  ", "PASS (bit-identical)" if (d_cos == 0.0 and d_sin == 0.0) else "FAIL")
    if not (d_cos == 0.0 and d_sin == 0.0):
        allok = False

    # ── source_id = 2 (integer) ──
    print("")
    print("---- source_id = 2 (integer) ----")
    var rot2 = apply_src_id_rope_rotation(stock[0], stock[1], 2.0, THETA, ctx)
    _check(harness, "cos sid=2 vs torch", rot2[0].to_host(ctx), _in("ref_cos_2"), allok)
    _check(harness, "sin sid=2 vs torch", rot2[1].to_host(ctx), _in("ref_sin_2"), allok)

    # ── source_id = 1.5 (FRACTIONAL — Bernini's key feature) ──
    print("")
    print("---- source_id = 1.5 (fractional) ----")
    var rotf = apply_src_id_rope_rotation(stock[0], stock[1], 1.5, THETA, ctx)
    _check(harness, "cos sid=1.5 vs torch", rotf[0].to_host(ctx), _in("ref_cos_1p5"), allok)
    _check(harness, "sin sid=1.5 vs torch", rotf[1].to_host(ctx), _in("ref_sin_1p5"), allok)

    # ── multi-segment concat helper (Tier 2b): [sid=0 | sid=1.5] ──
    # First segment == stock*identity, second == the 1.5-rotated table; the
    # concatenation must reproduce them stacked along the token axis.
    print("")
    print("---- multi-segment concat helper (Tier 2b seq-concat) ----")
    var segs = List[BerniniRopeSegment]()
    segs.append(BerniniRopeSegment(PPF, PPH, PPW, 0.0))
    segs.append(BerniniRopeSegment(PPF, PPH, PPW, 1.5))
    var seq = build_bernini_src_id_rope(segs, HEAD_DIM, THETA, STDtype.F32, ctx)
    var seq_cos = seq[0].to_host(ctx)
    var seq_sin = seq[1].to_host(ctx)
    # Expected: [ref_cos_0 (S rows) ; ref_cos_1p5 (S rows)].
    var exp_cos = _in("ref_cos_0")
    for v in _in("ref_cos_1p5"):
        exp_cos.append(v)
    var exp_sin = _in("ref_sin_0")
    for v in _in("ref_sin_1p5"):
        exp_sin.append(v)
    _check(harness, "concat cos [0|1.5]", seq_cos, exp_cos, allok)
    _check(harness, "concat sin [0|1.5]", seq_sin, exp_sin, allok)

    print("")
    if allok:
        print("VERDICT: PASS — Bernini-R source-id RoPE matches torch (cos>=0.999)"
              " and source_id=0 is bit-identical to stock wan rope")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
