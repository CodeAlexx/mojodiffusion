# pool_bwd_parity.mojo — GPU verification of the naive pool/upsample BACKWARD.
#
# Tier-5 gate: grad-parity cos >= 0.999 of maxpool d_x / upsample d_x vs a
# PyTorch reference (pool_bwd_oracle.py -> pool_bwd_ref.txt).
#
# Inputs use the SAME deterministic fills as pool_bwd_oracle.py, built in MOJO
# layout (NHWC). Only the reference GRADIENTS are read from the ref file (already
# permuted back to mojo NHWC by the oracle).
#
# Shapes:
#   MaxPool:  N=2, C=3, Hi=Wi=8, K=2, stride=2, pad=0  (Ho=Wo=4)
#   Upsample: N=2, C=3, in_h=in_w=5, scale=2           (Ho=Wo=10)
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/pool_bwd_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/pool_bwd_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.pool_backward import (
    maxpool2d_backward,
    upsample_nearest2d_backward,
)
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/pool_bwd_ref.txt"
)


# ── deterministic fills — MUST match pool_bwd_oracle.py ───────────────────────
def _fill_mp_x(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 37) % 257) - 128.0) * 0.01)
    return out^


def _fill_mp_gy(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 3) % 9) - 4.0) * 0.05)
    return out^


def _fill_us_gy(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 7) % 13) - 6.0) * 0.05)
    return out^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d)
    return s^


# ── read one tagged space-separated float line (mirrors conv2d_bwd_parity) ────
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

    # ── MaxPool2D backward ───────────────────────────────────────────────────
    comptime MP_N = 2
    comptime MP_Hi = 8
    comptime MP_Wi = 8
    comptime MP_C = 3
    comptime MP_K = 2
    comptime MP_S = 2
    comptime MP_Ho = (MP_Hi - MP_K) // MP_S + 1
    comptime MP_Wo = (MP_Wi - MP_K) // MP_S + 1
    comptime mp_nx = MP_N * MP_Hi * MP_Wi * MP_C
    comptime mp_ng = MP_N * MP_Ho * MP_Wo * MP_C

    var mp_x = Tensor.from_host(
        _fill_mp_x(mp_nx), _shape4(MP_N, MP_Hi, MP_Wi, MP_C), STDtype.F32, ctx
    )
    var mp_gy = Tensor.from_host(
        _fill_mp_gy(mp_ng), _shape4(MP_N, MP_Ho, MP_Wo, MP_C), STDtype.F32, ctx
    )
    var mp_dx = maxpool2d_backward[
        MP_N, MP_Hi, MP_Wi, MP_C, MP_K, MP_K, MP_S, MP_S
    ](mp_gy, mp_x, ctx)
    var mp_dx_host = mp_dx.to_host(ctx)
    var r_mp = h.compare_host(mp_dx_host, _read_ref(String("maxpool_dx")))
    print("maxpool2d d_x vs torch:", r_mp)
    all_pass = all_pass and r_mp.passed

    # ── UpsampleNearest2D backward ───────────────────────────────────────────
    comptime US_N = 2
    comptime US_h = 5
    comptime US_w = 5
    comptime US_C = 3
    comptime US_SCALE = 2
    comptime US_Ho = US_h * US_SCALE
    comptime US_Wo = US_w * US_SCALE
    comptime us_ng = US_N * US_Ho * US_Wo * US_C

    var us_gy = Tensor.from_host(
        _fill_us_gy(us_ng), _shape4(US_N, US_Ho, US_Wo, US_C), STDtype.F32, ctx
    )
    var us_dx = upsample_nearest2d_backward[
        US_N, US_h, US_w, US_C, US_SCALE
    ](us_gy, ctx)
    var us_dx_host = us_dx.to_host(ctx)
    var r_us = h.compare_host(us_dx_host, _read_ref(String("upsample_dx")))
    print("upsample_nearest2d d_x vs torch:", r_us)
    all_pass = all_pass and r_us.passed

    print("")
    if all_pass:
        print("ALL POOL BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("POOL BACKWARD PARITY FAILURE")
        raise Error("pool_bwd_parity gate failed")
