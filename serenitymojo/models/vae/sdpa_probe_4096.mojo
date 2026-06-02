# sdpa_probe_4096.mojo — SKEPTIC: does foundation flash_attention hold at the
# REAL mid-attn shape? B=1, S=4096 (64x64 latent), H=1 head, Dh=512.
# This is the size the 512^2 decode actually runs. A CPU single-head softmax
# reference is O(S^2 * Dh) = 4096*4096*512 ~ 8.6e9 mults — slow but bounded; we
# verify only a SUBSET of query rows (every 256th) against the GPU output to keep
# the CPU cost reasonable while still covering the full S range.
from std.math import sqrt, exp
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention import sdpa


def main() raises:
    var ctx = DeviceContext()
    comptime B = 1
    comptime S = 4096
    comptime Hd = 1
    comptime Dh = 512
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var qn = B * S * Hd * Dh
    var qv = List[Float32]()
    var kv = List[Float32]()
    var vv = List[Float32]()
    for i in range(qn):
        qv.append((Float32((i * 7) % 13) - 6.0) * 0.05)
        kv.append((Float32((i * 5) % 11) - 5.0) * 0.05)
        vv.append((Float32((i * 3) % 9) - 4.0) * 0.05)
    var qs = List[Int]()
    qs.append(B); qs.append(S); qs.append(Hd); qs.append(Dh)
    var ks = qs.copy()
    var vs = qs.copy()
    var q = Tensor.from_host(qv, qs^, STDtype.F32, ctx)
    var k = Tensor.from_host(kv, ks^, STDtype.F32, ctx)
    var v = Tensor.from_host(vv, vs^, STDtype.F32, ctx)
    var mn = B * Hd * S * S
    var mv = List[Float32]()
    for _ in range(mn):
        mv.append(0.0)
    var ms = List[Int]()
    ms.append(B); ms.append(Hd); ms.append(S); ms.append(S)
    var mask = Tensor.from_host(mv, ms^, STDtype.F32, ctx)

    print("running flash_attention at S=4096 Dh=512 ...")
    var out = sdpa[B, S, Hd, Dh](q, k, v, mask, scale, ctx)
    var ov = out.to_host(ctx)
    print("flash_attention done; checking subset of query rows on CPU ...")

    var maxd: Float32 = 0.0
    var checked = 0
    var qi = 0
    while qi < S:
        # scores for row qi over all keys
        var sc = List[Float32]()
        var mx = Float32(-3.0e38)
        for j in range(S):
            var dot: Float32 = 0.0
            for d in range(Dh):
                dot += qv[qi * Dh + d] * kv[j * Dh + d]
            dot *= scale
            sc.append(dot)
            if dot > mx:
                mx = dot
        var ssum: Float32 = 0.0
        for j in range(S):
            var e = exp(sc[j] - mx)
            sc[j] = e
            ssum += e
        # compare a few output dims for this row
        for d in range(Dh):
            var acc: Float32 = 0.0
            for j in range(S):
                acc += (sc[j] / ssum) * vv[j * Dh + d]
            var dd = ov[qi * Dh + d] - acc
            if dd < 0.0:
                dd = -dd
            if dd > maxd:
                maxd = dd
        checked += 1
        qi += 256
    print("checked", checked, "query rows; sdpa max_abs_diff vs CPU cref:", maxd)
    if maxd < 1e-3:
        print("SDPA-4096 PROBE PASS")
    else:
        print("SDPA-4096 PROBE FAIL")
