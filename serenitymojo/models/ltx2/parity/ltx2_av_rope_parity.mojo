# serenitymojo/models/ltx2/parity/ltx2_av_rope_parity.mojo
#
# LTX-2 joint-AV RoPE builder parity gate (P6.2 runner prerequisite).
#
# Proves the importable Mojo rope builder (models/ltx2/ltx2_av_rope.mojo
# ltx2_av_build_rope) reproduces the oracle's 4 rope pairs (v/a/ca_v/ca_a),
# built at the MUSUBI-CORRECT causal_offset=1. The runner needs a proven rope;
# the stack gates only ever LOADED rope from the oracle dump (never built it),
# so this closes the last ungated AV numeric surface.
#
# Reference: the corrected offset=1 oracle dump
#   output/ltx2_av_stack/av_stack2_ref.safetensors (v_cos/v_sin/a_cos/a_sin/
#   ca_v_cos/ca_v_sin/ca_a_cos/ca_a_sin, stored [H,P,hrd]). The oracle rope is
#   F32; the Mojo builder emits F32; same causal_offset=1 both sides -> expect
#   near-bit (cos >= 0.999999, max_abs ~1e-7 class; worse = a finding).
#
# The oracle stores rope [H,P,hrd]; _load_rope transposes to [P*H,hrd] (the
# layout ltx2_av_build_rope emits and the block forward consumes).
#
# Run: rm -f serenitymojo.mojopkg; pixi run mojo build -O2 -I . -Xlinker -lm \
#   -Xlinker -lcuda serenitymojo/models/ltx2/parity/ltx2_av_rope_parity.mojo \
#   -o /tmp/ltx2_av_rope_parity && /tmp/ltx2_av_rope_parity

from max.gpu.host import DeviceContext
from std.collections import List
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.ltx2.ltx2_av_rope import ltx2_av_build_rope, LTX2AVRope

comptime REF = "/home/alex/mojodiffusion/output/ltx2_av_stack/av_stack2_ref.safetensors"
# geometry MUST match the oracle's factorization of S_V=128 (block oracle :77
# NF,NH,NW = 8,4,4 — NOT the MVP's 2,8,8; same S_V, different grid -> different
# video rope). Audio S_A=16.
comptime NF = 8
comptime NH = 4
comptime NW = 4
comptime S_A = 16
comptime COS_GATE = 0.999999


def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^


def _load_rope(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    # oracle rope stored [H, P, hrd] -> transpose to [P*H, hrd] (builder layout).
    var t = Tensor.from_view_as_f32(st.tensor_view(name), ctx)
    ref sh = t.shape()
    var H = sh[0]; var S = sh[1]; var hrd = sh[2]
    var host = t.to_host(ctx)
    var out = List[Float32]()
    out.resize(S * H * hrd, Float32(0.0))
    for h in range(H):
        for s in range(S):
            for j in range(hrd):
                out[(s * H + h) * hrd + j] = host[(h * S + s) * hrd + j]
    return Tensor.from_host(out, _sh2(S * H, hrd), STDtype.F32, ctx)


def _cos_maxabs(a: List[Float32], b: List[Float32]) raises -> Tuple[Float64, Float64]:
    if len(a) != len(b):
        raise Error(String("len ") + String(len(a)) + " vs " + String(len(b)))
    var dot = 0.0; var na = 0.0; var nb = 0.0; var mx = 0.0
    for i in range(len(a)):
        var x = Float64(a[i]); var y = Float64(b[i])
        if x != x or y != y:
            raise Error("NaN")
        dot += x * y; na += x * x; nb += y * y
        var d = x - y
        if d < 0.0: d = -d
        if d > mx: mx = d
    return (dot / (sqrt(na) * sqrt(nb) + 1e-30), mx)


def _check(tag: String, mojo: Tensor, oracle: Tensor, ctx: DeviceContext) raises -> Int:
    var m = mojo.to_host(ctx)
    var o = oracle.to_host(ctx)
    var cm = _cos_maxabs(m, o)
    var ok = cm[0] >= COS_GATE
    print("  ", "PASS" if ok else "FAIL", tag, "cos=", cm[0], "max_abs=", cm[1])
    return 0 if ok else 1


def main() raises:
    var ctx = DeviceContext()
    print("=== LTX-2 joint-AV RoPE builder parity gate (causal_offset=1) ===")
    print("  geometry NF/NH/NW/S_A:", NF, NH, NW, S_A, " S_V:", NF * NH * NW)
    print("  oracle:", REF)

    var dump = ShardedSafeTensors.open(String(REF))
    var rope = ltx2_av_build_rope[NF, NH, NW, S_A](ctx)

    var fails = 0
    fails += _check(String("v_cos"), rope.v_cos, _load_rope(dump, String("v_cos"), ctx), ctx)
    fails += _check(String("v_sin"), rope.v_sin, _load_rope(dump, String("v_sin"), ctx), ctx)
    fails += _check(String("a_cos"), rope.a_cos, _load_rope(dump, String("a_cos"), ctx), ctx)
    fails += _check(String("a_sin"), rope.a_sin, _load_rope(dump, String("a_sin"), ctx), ctx)
    fails += _check(String("ca_v_cos"), rope.ca_v_cos, _load_rope(dump, String("ca_v_cos"), ctx), ctx)
    fails += _check(String("ca_v_sin"), rope.ca_v_sin, _load_rope(dump, String("ca_v_sin"), ctx), ctx)
    fails += _check(String("ca_a_cos"), rope.ca_a_cos, _load_rope(dump, String("ca_a_cos"), ctx), ctx)
    fails += _check(String("ca_a_sin"), rope.ca_a_sin, _load_rope(dump, String("ca_a_sin"), ctx), ctx)

    print("")
    if fails == 0:
        print("LTX2 AV rope builder parity PASS: 8 rope tensors cos >=", COS_GATE, "(offset=1 both sides)")
    else:
        raise Error(String("LTX2 AV rope parity FAIL: ") + String(fails) + " tensors below gate")
