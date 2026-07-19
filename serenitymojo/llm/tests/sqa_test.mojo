from std.gpu.host import DeviceContext
from std.math import exp, sqrt
from serenitymojo.llm.sqa import sqa_gpu

def _cpu_sqa(q: List[Float32], k: List[Float32], v: List[Float32],
             H: Int, H_kv: Int, L: Int, dh: Int) raises -> List[Float32]:
    var n_rep = H // H_kv
    var scale = Float32(1.0) / sqrt(Float32(dh))
    var o = List[Float32]()
    o.resize(H * dh, Float32(0.0))
    for hq in range(H):
        var kv = hq // n_rep
        var scores = List[Float32]()
        var m = Float32(-1.0e30)
        for l in range(L):
            var s = Float32(0.0)
            for d in range(dh):
                s += q[hq*dh+d] * k[(kv*L+l)*dh+d]
            s *= scale
            scores.append(s)
            if s > m: m = s
        var denom = Float32(0.0)
        for l in range(L): denom += exp(scores[l]-m)
        for l in range(L):
            var w = exp(scores[l]-m)/denom
            for d in range(dh):
                o[hq*dh+d] += w * v[(kv*L+l)*dh+d]
    return o^

def _rng(seed: Int, n: Int) -> List[Float32]:
    var s = UInt64(seed*2654435761 + 1)
    var out = List[Float32]()
    for _ in range(n):
        s = s ^ (s >> 12); s = s ^ (s << 25); s = s ^ (s >> 27)
        var u = Float32((s * 0x2545F4914F6CDD1D) >> 40) / Float32(16777216.0)
        out.append(u - 0.5)
    return out^

def check(mut p: Int, mut f: Int, cond: Bool, name: String):
    if cond: p += 1
    else:
        f += 1; print("  FAIL:", name)

def _maxdiff(a: List[Float32], b: List[Float32]) -> Float32:
    var m = Float32(0.0)
    for i in range(len(a)):
        var d = a[i]-b[i]
        if d < 0: d = -d
        if d > m: m = d
    return m

def main() raises:
    var ctx = DeviceContext()
    var p = 0
    var f = 0
    # case A: tiny
    var H=4; var Hkv=2; var L=8; var dh=16
    var q=_rng(1,H*dh); var k=_rng(2,Hkv*L*dh); var v=_rng(3,Hkv*L*dh)
    var g=sqa_gpu(ctx,q,k,v,H,Hkv,L,dh); var c=_cpu_sqa(q,k,v,H,Hkv,L,dh)
    var md=_maxdiff(g,c)
    print("case A (4/2/8/16) maxdiff:", md)
    check(p,f, md < 1e-3, "tiny GQA sqa matches CPU")
    # case B: Qwen3-0.6B shape (H=16,Hkv=8,dh=128), L=64
    var H2=16; var Hkv2=8; var L2=64; var dh2=128
    var q2=_rng(11,H2*dh2); var k2=_rng(12,Hkv2*L2*dh2); var v2=_rng(13,Hkv2*L2*dh2)
    var g2=sqa_gpu(ctx,q2,k2,v2,H2,Hkv2,L2,dh2); var c2=_cpu_sqa(q2,k2,v2,H2,Hkv2,L2,dh2)
    var md2=_maxdiff(g2,c2)
    print("case B (16/8/64/128) maxdiff:", md2)
    check(p,f, md2 < 1e-3, "qwen3-0.6B-shape GQA sqa matches CPU")
    print("passed:", p, " failed:", f)
    if f == 0: print("ALL SQA TESTS PASSED")
