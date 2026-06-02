# sdpa_rect_bwd_parity.mojo — PARITY GATE for the shared RECTANGULAR SDPA
# backward (sdpa_backward_rect, S_q != S_kv). Runs the REAL kernel with
# non-degenerate sinusoidal inputs at the two cross-attention classes that need
# it and gates d_q/d_k/d_v vs the torch oracle at cos >= 0.999:
#   Dh=64 : Sq=64, Skv=77  (SDXL cross-attn)
#   Dh=128: Sq=96, Skv=16  (Anima cross-attn)
#
# Run the oracle first, then the gate:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/ops/parity/sdpa_rect_bwd_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/sdpa_rect_bwd_parity.mojo

from std.math import sqrt, sin, cos
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.attention_backward import sdpa_backward_rect
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/ops/parity/"


# Non-degenerate sinusoidal fills (MUST match sdpa_rect_bwd_oracle.py).
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


def _run[B: Int, Sq: Int, Skv: Int, H: Int, Dh: Int](
    ctx: DeviceContext, tag: String
) raises -> Bool:
    var h = ParityHarness()
    var nq = B * Sq * H * Dh
    var nkv = B * Skv * H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var sb = sdpa_backward_rect[B, Sq, Skv, H, Dh](
        Tensor.from_host(_fq(nq), _bshd(B, Sq, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_fk(nkv), _bshd(B, Skv, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_fv(nkv), _bshd(B, Skv, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_fdo(nq), _bshd(B, Sq, H, Dh), STDtype.F32, ctx),
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
    print("=== RECTANGULAR SDPA BACKWARD parity (S_q != S_kv) ===")
    # SDXL cross-attn class (Dh=64).
    ok = _run[1, 64, 77, 5, 64](ctx, String("rect_Sq64_Skv77_H5_Dh64")) and ok
    # Anima cross-attn class (Dh=128).
    ok = _run[1, 96, 16, 4, 128](ctx, String("rect_Sq96_Skv16_H4_Dh128")) and ok
    print("")
    if ok:
        print("ALL RECTANGULAR GATES PASSED — sdpa_backward_rect CORRECT (cos >= 0.999)")
    else:
        print("RECTANGULAR SDPA BACKWARD PARITY FAILURE")
        raise Error("sdpa_rect_bwd_parity gate failed")
