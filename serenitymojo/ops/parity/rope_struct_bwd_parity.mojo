# rope_struct_bwd_parity.mojo — GPU verification of the BACKWARD of three DiT
# structural primitives (serenitymojo/ops/rope_struct_backward.mojo):
#   RoPePrecomputed  (rope_backward, interleaved)  -> rope_dx
#   QkvSplitPermute  (qkv_split_permute_backward)  -> qkv_dqkv
#   GateResidual     (gate_residual_backward)      -> gate_dx, gate_dg, gate_dy
#
# Gate: cos >= 0.999 vs a PyTorch reference (rope_struct_bwd_oracle.py ->
# rope_struct_bwd_ref.txt). Inputs use the SAME deterministic fills as the oracle.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/rope_struct_bwd_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/rope_struct_bwd_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.rope_struct_backward import (
    rope_backward,
    qkv_split_permute_backward,
    gate_residual_backward,
)
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/rope_struct_bwd_ref.txt"
)


# ── deterministic fills — MUST match rope_struct_bwd_oracle.fill ──────────────
def _fill(n: Int, mul: Int, sub: Float32, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * mul) % 13) - sub) * scale)
    return out^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b)
    return s^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d)
    return s^


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


# ── read one tagged space-separated float line (mirrors sdpa_bwd_parity) ──────
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

    # ── 1) RoPe interleaved backward: rows=16, D=64, half=32 ─────────────────
    comptime ROWS = 16
    comptime D = 64
    comptime HALF = D // 2
    var grad_out = Tensor.from_host(
        _fill(ROWS * D, 2, 6.0, 0.05), _shape2(ROWS, D), STDtype.F32, ctx)
    var cos = Tensor.from_host(
        _fill(ROWS * HALF, 5, 6.0, 0.10), _shape2(ROWS, HALF), STDtype.F32, ctx)
    var sin = Tensor.from_host(
        _fill(ROWS * HALF, 3, 6.0, 0.10), _shape2(ROWS, HALF), STDtype.F32, ctx)
    var dx = rope_backward(grad_out, cos, sin, True, ctx)  # interleaved
    var r_rope = h.compare_host(dx.to_host(ctx), _read_ref(String("rope_dx")))
    print("1) RoPe interleaved d_x vs torch:", r_rope)
    all_pass = all_pass and r_rope.passed

    # ── 1b) RoPe HALFSPLIT backward (Z-Image): same dims/fills, (i, i+half) ────
    # Reuses grad_out/cos/sin (identical fills); only the pairing convention
    # differs. interleaved=False dispatches _rope_bwd_halfsplit_kernel_f32.
    var grad_out_hs = Tensor.from_host(
        _fill(ROWS * D, 2, 6.0, 0.05), _shape2(ROWS, D), STDtype.F32, ctx)
    var cos_hs = Tensor.from_host(
        _fill(ROWS * HALF, 5, 6.0, 0.10), _shape2(ROWS, HALF), STDtype.F32, ctx)
    var sin_hs = Tensor.from_host(
        _fill(ROWS * HALF, 3, 6.0, 0.10), _shape2(ROWS, HALF), STDtype.F32, ctx)
    var dx_hs = rope_backward(grad_out_hs, cos_hs, sin_hs, False, ctx)  # halfsplit
    var r_rope_hs = h.compare_host(
        dx_hs.to_host(ctx), _read_ref(String("rope_halfsplit_dx")))
    print("1b) RoPe halfsplit d_x vs torch:", r_rope_hs)
    all_pass = all_pass and r_rope_hs.passed

    # ── 2) QkvSplitPermute backward: B1,N8,H4,Dh16 ───────────────────────────
    comptime B = 1
    comptime N = 8
    comptime H = 4
    comptime DH = 16
    comptime HD = H * DH
    var gq = Tensor.from_host(
        _fill(B * N * HD, 2, 6.0, 0.05), _shape4(B, N, H, DH), STDtype.F32, ctx)
    var gk = Tensor.from_host(
        _fill(B * N * HD, 3, 6.0, 0.05), _shape4(B, N, H, DH), STDtype.F32, ctx)
    var gv = Tensor.from_host(
        _fill(B * N * HD, 5, 6.0, 0.05), _shape4(B, N, H, DH), STDtype.F32, ctx)
    var dqkv = qkv_split_permute_backward(gq, gk, gv, ctx)
    var r_qkv = h.compare_host(dqkv.to_host(ctx), _read_ref(String("qkv_dqkv")))
    print("2) QkvSplitPermute d_qkv vs torch:", r_qkv)
    all_pass = all_pass and r_qkv.passed

    # ── 3) GateResidual backward: rows=12, C=32 ──────────────────────────────
    comptime GROWS = 12
    comptime C = 32
    var g_grad = Tensor.from_host(
        _fill(GROWS * C, 2, 6.0, 0.05), _shape2(GROWS, C), STDtype.F32, ctx)
    var g_x = Tensor.from_host(
        _fill(GROWS * C, 7, 6.0, 0.05), _shape2(GROWS, C), STDtype.F32, ctx)
    var g_gate = Tensor.from_host(
        _fill(C, 5, 6.0, 0.10), _shape1(C), STDtype.F32, ctx)
    var g_y = Tensor.from_host(
        _fill(GROWS * C, 3, 6.0, 0.05), _shape2(GROWS, C), STDtype.F32, ctx)
    var grads = gate_residual_backward(g_grad, g_x, g_gate, g_y, ctx)
    var r_gdx = h.compare_host(grads.d_x.to_host(ctx), _read_ref(String("gate_dx")))
    var r_gdg = h.compare_host(grads.d_g.to_host(ctx), _read_ref(String("gate_dg")))
    var r_gdy = h.compare_host(grads.d_y.to_host(ctx), _read_ref(String("gate_dy")))
    print("3) GateResidual d_x vs torch:", r_gdx)
    print("3) GateResidual d_g vs torch:", r_gdg)
    print("3) GateResidual d_y vs torch:", r_gdy)
    all_pass = all_pass and r_gdx.passed and r_gdg.passed and r_gdy.passed

    print("")
    if all_pass:
        print("ALL ROPE/QKV/GATE BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("ROPE_STRUCT BACKWARD PARITY FAILURE")
        raise Error("rope_struct_bwd_parity gate failed")
