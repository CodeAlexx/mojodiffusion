# sdpa_bwd_parity.mojo — GPU verification of the decomposed SDPA BACKWARD.
#
# Phase T0 gate (FULL_PORT_TRAINING_PLAN §4): grad-parity cos >= 0.999 of
# d_q/d_k/d_v vs a PyTorch reference (sdpa_bwd_oracle.py → sdpa_bwd_ref.txt).
#
# Inputs use the SAME deterministic fills as sdpa_bwd_oracle.gen_qkv_dout (q/k/v
# match sdpa_math_oracle; d_out is the extra fill). Only the reference GRADIENTS
# are read from the ref file.
#
# Cases A (H=32,Dh=128) and B (H=8,Dh64) use the legacy modular fills + text ref.
# Cases C (H=30) and D (H=6) are the NON-32-ALIGNED head counts (Z-Image is H=30)
# added after BUG_sdpa_backward_H30_dq_dk_zero. They MUST use non-degenerate
# sinusoidal fills: the legacy V[i]=((i*3)%9-4)*0.05 fill is DEGENERATE at H=30
# and H=6 (in BSHD the per-(head,dim) seq stride H*Dh has stride*3 ≡ 0 (mod 9)
# for H∈{6,30},Dh=128 → V constant across seq → grad_scores mathematically zero
# → d_q/d_k genuinely ~0, and a cosine gate on ~zero vectors is noise). The
# sinusoidal fills never alias, so C/D are real H=30/H=6 correctness gates.
#
# Run BOTH oracles first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/sdpa_bwd_oracle.py
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/sdpa_bwd_nondegen_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/sdpa_bwd_parity.mojo

from std.math import sqrt, sin, cos
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/sdpa_bwd_ref.txt"
)


# Deterministic BSHD fills — MUST match sdpa_bwd_oracle.py gen_qkv_dout.
def _fill_q(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 7) % 13) - 6.0) * 0.05)
    return out^


def _fill_k(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 5) % 11) - 5.0) * 0.05)
    return out^


def _fill_v(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 3) % 9) - 4.0) * 0.05)
    return out^


def _fill_dout(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 2) % 7) - 3.0) * 0.05)
    return out^


def _bshd(B: Int, S: Int, H: Int, Dh: Int) -> List[Int]:
    var s = List[Int]()
    s.append(B); s.append(S); s.append(H); s.append(Dh)
    return s^


# ── NON-DEGENERATE sinusoidal fills (match sdpa_bwd_nondegen_oracle.fills) ────
comptime ND_REF_DIR = "/home/alex/mojodiffusion/serenitymojo/ops/parity/"
def _nd_q(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n): o.append(sin(0.07 * Float32(i) + 1.1) * 0.2)
    return o^
def _nd_k(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n): o.append(cos(0.05 * Float32(i) + 0.5) * 0.2)
    return o^
def _nd_v(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n): o.append(sin(0.10 * Float32(i) + 0.3) * 0.2)
    return o^
def _nd_do(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for i in range(n): o.append(cos(0.09 * Float32(i) + 0.2) * 0.2)
    return o^


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run sdpa_bwd_nondegen_oracle.py): ") + path)
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


# Non-degenerate H-arbitrary correctness case (used for H=30, H=6).
def _run_nd_case[
    B: Int, S: Int, H: Int, Dh: Int
](ctx: DeviceContext, h: ParityHarness, tag: String, label: String) raises -> Bool:
    var n = B * S * H * Dh
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var g = sdpa_backward[B, S, H, Dh](
        Tensor.from_host(_nd_q(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_nd_k(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_nd_v(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        Tensor.from_host(_nd_do(n), _bshd(B, S, H, Dh), STDtype.F32, ctx),
        scale, ctx,
    )
    var dq = g.d_q.to_host(ctx)
    var dk = g.d_k.to_host(ctx)
    var dv = g.d_v.to_host(ctx)
    var rdq = h.compare_host(dq, _read_bin_f32(ND_REF_DIR + tag + "_dq.bin"))
    var rdk = h.compare_host(dk, _read_bin_f32(ND_REF_DIR + tag + "_dk.bin"))
    var rdv = h.compare_host(dv, _read_bin_f32(ND_REF_DIR + tag + "_dv.bin"))
    print(label, "d_q vs torch:", rdq)
    print(label, "d_k vs torch:", rdk)
    print(label, "d_v vs torch:", rdv)
    return rdq.passed and rdk.passed and rdv.passed


# ── read one tagged space-separated float line (mirrors sdpa_math_parity) ─────
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

    # ── Case A: Dh=128 (B1,S8,H32,Dh128) ─────────────────────────────────────
    comptime BA = 1
    comptime SA = 8
    comptime HA = 32
    comptime DhA = 128
    var nA = BA * SA * HA * DhA
    var scaleA = Float32(1.0) / sqrt(Float32(DhA))
    var gradsA = sdpa_backward[BA, SA, HA, DhA](
        Tensor.from_host(_fill_q(nA), _bshd(BA, SA, HA, DhA), STDtype.F32, ctx),
        Tensor.from_host(_fill_k(nA), _bshd(BA, SA, HA, DhA), STDtype.F32, ctx),
        Tensor.from_host(_fill_v(nA), _bshd(BA, SA, HA, DhA), STDtype.F32, ctx),
        Tensor.from_host(_fill_dout(nA), _bshd(BA, SA, HA, DhA), STDtype.F32, ctx),
        scaleA, ctx,
    )
    var dqA = gradsA.d_q.to_host(ctx)
    var dkA = gradsA.d_k.to_host(ctx)
    var dvA = gradsA.d_v.to_host(ctx)
    var rA_dq = h.compare_host(dqA, _read_ref(String("dh128_dq")))
    var rA_dk = h.compare_host(dkA, _read_ref(String("dh128_dk")))
    var rA_dv = h.compare_host(dvA, _read_ref(String("dh128_dv")))
    print("A) Dh=128 d_q vs torch:", rA_dq)
    print("A) Dh=128 d_k vs torch:", rA_dk)
    print("A) Dh=128 d_v vs torch:", rA_dv)
    all_pass = all_pass and rA_dq.passed and rA_dk.passed and rA_dv.passed

    # ── Case B: Dh=64 (B1,S8,H8,Dh64) ────────────────────────────────────────
    comptime BB = 1
    comptime SB = 8
    comptime HB = 8
    comptime DhB = 64
    var nB = BB * SB * HB * DhB
    var scaleB = Float32(1.0) / sqrt(Float32(DhB))
    var gradsB = sdpa_backward[BB, SB, HB, DhB](
        Tensor.from_host(_fill_q(nB), _bshd(BB, SB, HB, DhB), STDtype.F32, ctx),
        Tensor.from_host(_fill_k(nB), _bshd(BB, SB, HB, DhB), STDtype.F32, ctx),
        Tensor.from_host(_fill_v(nB), _bshd(BB, SB, HB, DhB), STDtype.F32, ctx),
        Tensor.from_host(_fill_dout(nB), _bshd(BB, SB, HB, DhB), STDtype.F32, ctx),
        scaleB, ctx,
    )
    var dqB = gradsB.d_q.to_host(ctx)
    var dkB = gradsB.d_k.to_host(ctx)
    var dvB = gradsB.d_v.to_host(ctx)
    var rB_dq = h.compare_host(dqB, _read_ref(String("dh64_dq")))
    var rB_dk = h.compare_host(dkB, _read_ref(String("dh64_dk")))
    var rB_dv = h.compare_host(dvB, _read_ref(String("dh64_dv")))
    print("B) Dh=64  d_q vs torch:", rB_dq)
    print("B) Dh=64  d_k vs torch:", rB_dk)
    print("B) Dh=64  d_v vs torch:", rB_dv)
    all_pass = all_pass and rB_dq.passed and rB_dk.passed and rB_dv.passed

    # ── Case C: H=30 — Z-Image's REAL (non-32-aligned) head count ─────────────
    # Regression guard for BUG_sdpa_backward_H30_dq_dk_zero: non-degenerate data
    # so d_q/d_k are genuinely nonzero. Was the FALSE-GREEN gap in the old gate.
    var rC = _run_nd_case[1, 256, 30, 128](
        ctx, h, String("nd_S256_H30"), String("C) H=30 "))
    all_pass = all_pass and rC

    # ── Case D: H=6 — a second non-32-aligned head count ──────────────────────
    var rD = _run_nd_case[1, 256, 6, 128](
        ctx, h, String("nd_S256_H6"), String("D) H=6  "))
    all_pass = all_pass and rD

    print("")
    if all_pass:
        print("ALL SDPA BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("SDPA BACKWARD PARITY FAILURE")
        raise Error("sdpa_bwd_parity gate failed")
