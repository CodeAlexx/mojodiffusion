# sdpa_bwd_nondegen_parity.mojo — REGRESSION GATE: runs the REAL sdpa_backward
# with NON-DEGENERATE sinusoidal inputs at H=30 (Z-Image), H=6 (non-32-aligned),
# H=32 (32-aligned control), and S=384 (real Z-Image unified_len), gating
# d_q/d_k/d_v vs the torch oracle (sdpa_bwd_nondegen_oracle.py) at cos >= 0.999.
#
# This is the gate that PROVES the H=30 "failure" reported in
# BUG_sdpa_backward_H30_dq_dk_zero was a DEGENERATE-TEST-DATA artifact, not a
# kernel bug. The old realseq/toy oracles fill V via (i*3)%9; in BSHD the per-
# (head,dim) seq stride is H*Dh, and for H in {6,30} with Dh=128 that stride*3
# is ≡ 0 (mod 9) → V constant across seq → grad_attn rows constant → softmax-bwd
# grad_scores mathematically ZERO → d_q/d_k genuinely ~0 (torch: |d_q|≈2.5e-18).
# Cosine of two ~zero vectors is noise, which the old gate misread as FAIL.
#
# These sinusoidal fills never alias with H*Dh, so the gradient is genuinely
# nonzero and this is a real correctness test of the H=30 / H=6 head counts.
#
# Run the oracle first, then the gate:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/ops/parity/sdpa_bwd_nondegen_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/sdpa_bwd_nondegen_parity.mojo

from std.math import sqrt, sin, cos
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/ops/parity/"


# Non-degenerate sinusoidal fills (MUST match sdpa_bwd_nondegen_oracle.fills).
def _fq(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n): o.append(sin(0.07 * Float32(i) + 1.1) * 0.2)
    return o^
def _fk(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n): o.append(cos(0.05 * Float32(i) + 0.5) * 0.2)
    return o^
def _fv(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n): o.append(sin(0.10 * Float32(i) + 0.3) * 0.2)
    return o^
def _fdo(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n): o.append(cos(0.09 * Float32(i) + 0.2) * 0.2)
    return o^


def _bshd(B: Int, S: Int, H: Int, Dh: Int) -> List[Int]:
    var s = List[Int](); s.append(B); s.append(S); s.append(H); s.append(Dh); return s^


def _read_bin(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0: raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the oracle first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var g = sys_pread(fd, buf + done, n - done, done)
        if g <= 0: break
        done += g
    _ = sys_close(fd)
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(n // 4): out.append(fp[i])
    buf.free()
    return out^


def _run[B: Int, S: Int, H: Int, Dh: Int](ctx: DeviceContext, tag: String) raises -> Bool:
    var h = ParityHarness()
    var n = B * S * H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var sb = sdpa_backward[B, S, H, Dh](
        Tensor.from_host(_fq(n), _bshd(B,S,H,Dh), STDtype.F32, ctx),
        Tensor.from_host(_fk(n), _bshd(B,S,H,Dh), STDtype.F32, ctx),
        Tensor.from_host(_fv(n), _bshd(B,S,H,Dh), STDtype.F32, ctx),
        Tensor.from_host(_fdo(n), _bshd(B,S,H,Dh), STDtype.F32, ctx),
        scale, ctx,
    )
    var dq = sb.d_q.to_host(ctx)
    var dk = sb.d_k.to_host(ctx)
    var dv = sb.d_v.to_host(ctx)
    var rdq = _read_bin(REF_DIR + tag + "_dq.bin")
    var rdk = _read_bin(REF_DIR + tag + "_dk.bin")
    var rdv = _read_bin(REF_DIR + tag + "_dv.bin")
    var a = h.compare_host(dq, rdq)
    var b = h.compare_host(dk, rdk)
    var c = h.compare_host(dv, rdv)
    print("[", tag, "]")
    print("    d_q:", a)
    print("    d_k:", b)
    print("    d_v:", c)
    return a.passed and b.passed and c.passed


def main() raises:
    var ctx = DeviceContext()
    var ok = True
    print("=== NON-DEGENERATE sinusoidal SDPA BACKWARD parity (real Z-Image H=30) ===")
    ok = _run[1, 256, 30, 128](ctx, String("nd_S256_H30")) and ok   # real head count
    ok = _run[1, 256, 6, 128](ctx, String("nd_S256_H6")) and ok     # 2nd non-32-aligned
    ok = _run[1, 256, 32, 128](ctx, String("nd_S256_H32")) and ok   # 32-aligned control
    ok = _run[1, 384, 30, 128](ctx, String("nd_S384_H30")) and ok   # real unified_len
    print("")
    if ok:
        print("ALL NON-DEGENERATE GATES PASSED — sdpa_backward CORRECT at H=30/6/32 (cos >= 0.999)")
    else:
        print("NON-DEGENERATE SDPA BACKWARD PARITY FAILURE")
        raise Error("sdpa_bwd_nondegen_parity gate failed")
