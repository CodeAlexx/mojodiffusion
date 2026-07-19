# rope_halfsplit_full_parity.mojo — GPU gate for half-split RoPE FORWARD
# (rope_halfsplit_full) + BACKWARD (rope_halfsplit_full_backward) on a REAL
# interleaved-doubled angle table where cos[i] != cos[i+half].
#
# This is the gate the degenerate block_oracle never exercised. It FAILS on the
# old half-width single-angle backward (rope_backward(..., False) fed cos[:, :half])
# and PASSES on the corrected full-width backward.
#
# Run (oracle first, SEPARATE command after any mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/rope_halfsplit_full_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/rope_halfsplit_full_parity.mojo

from std.gpu.host import DeviceContext
from std.math import sin as msin, cos as mcos
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.rope import rope_halfsplit_full
from serenitymojo.ops.rope_struct_backward import (
    rope_backward, rope_halfsplit_full_backward,
)
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/rope_halfsplit_full_ref.txt"
)


def _fill(n: Int, mul: Int, sub: Float32, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * mul) % 13) - sub) * scale)
    return out^


# Interleaved-doubled angle table [a0,a0,a1,a1,...] -> cos/sin full-width [rows,D].
# Matches oracle build_interleaved_doubled_angles EXACTLY. want_cos picks which
# table to return (cos vs sin) so we avoid a tuple return.
def _build_table(rows: Int, D: Int, want_cos: Bool) -> List[Float32]:
    var half = D // 2
    var out = List[Float32]()
    for _ in range(rows * D):
        out.append(0.0)
    for r in range(rows):
        for j in range(half):
            var a = msin(0.11 * Float32(r + 1) + 0.07 * Float32(j)) * 1.3 + 0.2 * Float32(j)
            var v = mcos(a) if want_cos else msin(a)
            out[r * D + 2 * j] = v
            out[r * D + 2 * j + 1] = v
    return out^


def _half_slice(full: List[Float32], rows: Int, D: Int) -> List[Float32]:
    """First D/2 columns of each row of a [rows,D] table -> [rows,D/2]."""
    var half = D // 2
    var out = List[Float32]()
    for r in range(rows):
        for c in range(half):
            out.append(full[r * D + c])
    return out^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b)
    return s^


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

    comptime ROWS = 16
    comptime D = 64
    comptime HALF = D // 2

    var cos_full = _build_table(ROWS, D, True)
    var sin_full = _build_table(ROWS, D, False)
    var x_vals = _fill(ROWS * D, 7, 6.0, 0.05)
    var g_vals = _fill(ROWS * D, 2, 6.0, 0.05)

    var x = Tensor.from_host(x_vals.copy(), _shape2(ROWS, D), STDtype.F32, ctx)
    var cosF = Tensor.from_host(cos_full.copy(), _shape2(ROWS, D), STDtype.F32, ctx)
    var sinF = Tensor.from_host(sin_full.copy(), _shape2(ROWS, D), STDtype.F32, ctx)

    # ── FORWARD: rope_halfsplit_full on the real full-width table ────────────
    var fwd = rope_halfsplit_full(x, cosF, sinF, ctx)
    var r_fwd = h.compare_host(fwd.to_host(ctx), _read_ref(String("fwd_out")))
    print("FORWARD rope_halfsplit_full vs torch:", r_fwd)
    all_pass = all_pass and r_fwd.passed

    # ── OLD BACKWARD (buggy): half-width single-angle table ──────────────────
    # Feeds cos[:, :half] like ernie/block.mojo did. Expected to FAIL on real table.
    var cos_h = _half_slice(cos_full, ROWS, D)
    var sin_h = _half_slice(sin_full, ROWS, D)
    var g_old = Tensor.from_host(g_vals.copy(), _shape2(ROWS, D), STDtype.F32, ctx)
    var cosH = Tensor.from_host(cos_h.copy(), _shape2(ROWS, HALF), STDtype.F32, ctx)
    var sinH = Tensor.from_host(sin_h.copy(), _shape2(ROWS, HALF), STDtype.F32, ctx)
    var dx_old = rope_backward(g_old, cosH, sinH, False, ctx)
    var r_old = h.compare_host(dx_old.to_host(ctx), _read_ref(String("dx")))
    print("OLD halfsplit backward (half-width) vs torch:", r_old)

    # ── NEW BACKWARD (fixed): full-width table ───────────────────────────────
    var g_new = Tensor.from_host(g_vals.copy(), _shape2(ROWS, D), STDtype.F32, ctx)
    var cosF2 = Tensor.from_host(cos_full.copy(), _shape2(ROWS, D), STDtype.F32, ctx)
    var sinF2 = Tensor.from_host(sin_full.copy(), _shape2(ROWS, D), STDtype.F32, ctx)
    var dx_new = rope_halfsplit_full_backward(g_new, cosF2, sinF2, ctx)
    var r_new = h.compare_host(dx_new.to_host(ctx), _read_ref(String("dx")))
    print("NEW halfsplit_full backward (full-width) vs torch:", r_new)
    all_pass = all_pass and r_new.passed

    print("")
    if all_pass:
        print("ROPE HALFSPLIT_FULL FWD+BWD GATE PASSED (cos >= 0.999 vs torch)")
    else:
        print("ROPE HALFSPLIT_FULL PARITY FAILURE")
        raise Error("rope_halfsplit_full_parity gate failed")
