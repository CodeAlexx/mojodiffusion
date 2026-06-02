# sdpa_probe.mojo — does foundation sdpa handle the VAE mid-attn shape?
# B=1, S=64 (8x8 latent), H=1 head, Dh=512. scale=1/sqrt(512).
# Compare to a CPU single-head softmax-attention reference.
from std.math import sqrt, exp
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.attention import sdpa


def main() raises:
    var ctx = DeviceContext()
    comptime B = 1
    comptime S = 64
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
    # zero mask [B,H,S,S]
    var mn = B * Hd * S * S
    var mv = List[Float32]()
    for _ in range(mn):
        mv.append(0.0)
    var ms = List[Int]()
    ms.append(B); ms.append(Hd); ms.append(S); ms.append(S)
    var mask = Tensor.from_host(mv, ms^, STDtype.F32, ctx)

    var out = sdpa[B, S, Hd, Dh](q, k, v, mask, scale, ctx)
    var ov = out.to_host(ctx)

    # CPU cref (single head): for each query row i, scores[j]=scale*dot(q_i,k_j),
    # softmax over j, out_i = sum_j p_j * v_j.
    var cref = List[Float32]()
    for _ in range(qn):
        cref.append(0.0)
    for i in range(S):
        var sc = List[Float32]()
        var mx = Float32(-3.0e38)
        for j in range(S):
            var dot: Float32 = 0.0
            for d in range(Dh):
                dot += qv[i * Dh + d] * kv[j * Dh + d]
            dot *= scale
            sc.append(dot)
            if dot > mx:
                mx = dot
        var ssum: Float32 = 0.0
        for j in range(S):
            var e = exp(sc[j] - mx)
            sc[j] = e
            ssum += e
        for d in range(Dh):
            var acc: Float32 = 0.0
            for j in range(S):
                acc += (sc[j] / ssum) * vv[j * Dh + d]
            cref[i * Dh + d] = acc

    var maxd: Float32 = 0.0
    for i in range(len(ov)):
        var dd = ov[i] - cref[i]
        if dd < 0.0:
            dd = -dd
        if dd > maxd:
            maxd = dd
    print("sdpa max_abs_diff vs CPU cref:", maxd)
    if maxd < 1e-3:
        print("SDPA PROBE PASS")
    else:
        print("SDPA PROBE FAIL")
