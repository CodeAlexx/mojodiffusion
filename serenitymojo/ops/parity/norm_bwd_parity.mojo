# norm_bwd_parity.mojo — GPU verification of the norm BACKWARD kernels.
#
# Gate: grad-parity cos >= 0.999 of d_x / d_g (/ d_b) vs a PyTorch reference
# (norm_bwd_oracle.py -> norm_bwd_ref.txt). Mirrors sdpa_bwd_parity.mojo:
# inputs are reproduced on-device via the SAME deterministic fills as the oracle
# (_fill_* below MUST match norm_bwd_oracle.py); only the reference GRADIENTS are
# read from the ref file.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/norm_bwd_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/norm_bwd_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.norm_backward import (
    rms_norm_backward,
    layer_norm_backward,
    group_norm_backward,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/norm_bwd_ref.txt"
)
comptime EPS = Float32(1e-5)


# ── deterministic fills — MUST match norm_bwd_oracle.py fill_* ───────────────
def _fill_x(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 7) % 13) - 6.0) * 0.05)
    return out^


def _fill_g(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 5) % 11) - 5.0) * 0.05 + 1.0)
    return out^


def _fill_b(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 3) % 9) - 4.0) * 0.05)
    return out^


def _fill_go(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 2) % 7) - 3.0) * 0.05)
    return out^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b)
    return s^


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d)
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
                var ch = Int(buf[p])
                if ch == 0x20:
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

    # ── RMSNorm: rows=8, D=64 ────────────────────────────────────────────────
    comptime ROWS = 8
    comptime D = 64
    var rn = ROWS * D
    var rms = rms_norm_backward(
        Tensor.from_host(_fill_go(rn), _shape2(ROWS, D), STDtype.F32, ctx),
        Tensor.from_host(_fill_x(rn), _shape2(ROWS, D), STDtype.F32, ctx),
        Tensor.from_host(_fill_g(D), _shape1(D), STDtype.F32, ctx),
        EPS, ctx,
    )
    var r_dx = h.compare(rms.d_x, _read_ref(String("rms_dx")), ctx)
    var r_dg = h.compare(rms.d_g, _read_ref(String("rms_dg")), ctx)
    print("RMSNorm d_x vs torch:", r_dx)
    print("RMSNorm d_g vs torch:", r_dg)
    all_pass = all_pass and r_dx.passed and r_dg.passed

    # ── RMSNorm BF16: same inputs downcast to BF16, SAME F32 torch ref. ────────
    # Relaxed gate cos >= 0.99 (BF16 ~3 decimal digits). The BF16 arm of
    # rms_norm_backward casts up to F32, runs the SAME reduction kernels, casts
    # grads back to BF16.
    var rms_bf = rms_norm_backward(
        cast_tensor(Tensor.from_host(_fill_go(rn), _shape2(ROWS, D), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(_fill_x(rn), _shape2(ROWS, D), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(_fill_g(D), _shape1(D), STDtype.F32, ctx), STDtype.BF16, ctx),
        EPS, ctx,
    )
    var r_dx_bf = h.compare(rms_bf.d_x, _read_ref(String("rms_dx")), ctx)
    var r_dg_bf = h.compare(rms_bf.d_g, _read_ref(String("rms_dg")), ctx)
    print("RMSNorm d_x _bf16 cos:", r_dx_bf.cos)
    print("RMSNorm d_g _bf16 cos:", r_dg_bf.cos)
    all_pass = all_pass and (r_dx_bf.cos >= 0.99) and (r_dg_bf.cos >= 0.99)

    # ── LayerNorm: rows=8, D=64 ──────────────────────────────────────────────
    var ln = layer_norm_backward(
        Tensor.from_host(_fill_go(rn), _shape2(ROWS, D), STDtype.F32, ctx),
        Tensor.from_host(_fill_x(rn), _shape2(ROWS, D), STDtype.F32, ctx),
        Tensor.from_host(_fill_g(D), _shape1(D), STDtype.F32, ctx),
        EPS, ctx,
    )
    var l_dx = h.compare(ln.d_x, _read_ref(String("ln_dx")), ctx)
    var l_dg = h.compare(ln.d_g, _read_ref(String("ln_dg")), ctx)
    var l_db = h.compare(ln.d_b, _read_ref(String("ln_db")), ctx)
    print("LayerNorm d_x vs torch:", l_dx)
    print("LayerNorm d_g vs torch:", l_dg)
    print("LayerNorm d_b vs torch:", l_db)
    all_pass = all_pass and l_dx.passed and l_dg.passed and l_db.passed

    # ── LayerNorm BF16: same inputs downcast to BF16, SAME F32 ref, thr 0.99. ──
    var ln_bf = layer_norm_backward(
        cast_tensor(Tensor.from_host(_fill_go(rn), _shape2(ROWS, D), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(_fill_x(rn), _shape2(ROWS, D), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(_fill_g(D), _shape1(D), STDtype.F32, ctx), STDtype.BF16, ctx),
        EPS, ctx,
    )
    var l_dx_bf = h.compare(ln_bf.d_x, _read_ref(String("ln_dx")), ctx)
    var l_dg_bf = h.compare(ln_bf.d_g, _read_ref(String("ln_dg")), ctx)
    var l_db_bf = h.compare(ln_bf.d_b, _read_ref(String("ln_db")), ctx)
    print("LayerNorm d_x _bf16 cos:", l_dx_bf.cos)
    print("LayerNorm d_g _bf16 cos:", l_dg_bf.cos)
    print("LayerNorm d_b _bf16 cos:", l_db_bf.cos)
    all_pass = all_pass and (l_dx_bf.cos >= 0.99) and (l_dg_bf.cos >= 0.99) and (l_db_bf.cos >= 0.99)

    # ── GroupNorm: N=2,H=4,W=4,C=8,G=4 (NHWC) ────────────────────────────────
    comptime N = 2
    comptime H = 4
    comptime W = 4
    comptime C = 8
    comptime G = 4
    var gn_n = N * H * W * C
    var gn = group_norm_backward(
        Tensor.from_host(_fill_go(gn_n), _shape4(N, H, W, C), STDtype.F32, ctx),
        Tensor.from_host(_fill_x(gn_n), _shape4(N, H, W, C), STDtype.F32, ctx),
        Tensor.from_host(_fill_g(C), _shape1(C), STDtype.F32, ctx),
        G, EPS, ctx,
    )
    var g_dx = h.compare(gn.d_x, _read_ref(String("gn_dx")), ctx)
    var g_dg = h.compare(gn.d_g, _read_ref(String("gn_dg")), ctx)
    var g_db = h.compare(gn.d_b, _read_ref(String("gn_db")), ctx)
    print("GroupNorm d_x vs torch:", g_dx)
    print("GroupNorm d_g vs torch:", g_dg)
    print("GroupNorm d_b vs torch:", g_db)
    all_pass = all_pass and g_dx.passed and g_dg.passed and g_db.passed

    # ── GroupNorm BF16: same inputs downcast to BF16, SAME F32 ref, thr 0.99. ──
    # The BF16 arm of group_norm_backward casts up to F32, runs the SAME NHWC
    # reduction kernels, casts grads back to BF16.
    var gn_bf = group_norm_backward(
        cast_tensor(Tensor.from_host(_fill_go(gn_n), _shape4(N, H, W, C), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(_fill_x(gn_n), _shape4(N, H, W, C), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(_fill_g(C), _shape1(C), STDtype.F32, ctx), STDtype.BF16, ctx),
        G, EPS, ctx,
    )
    var g_dx_bf = h.compare(gn_bf.d_x, _read_ref(String("gn_dx")), ctx)
    var g_dg_bf = h.compare(gn_bf.d_g, _read_ref(String("gn_dg")), ctx)
    var g_db_bf = h.compare(gn_bf.d_b, _read_ref(String("gn_db")), ctx)
    print("GroupNorm d_x _bf16 cos:", g_dx_bf.cos)
    print("GroupNorm d_g _bf16 cos:", g_dg_bf.cos)
    print("GroupNorm d_b _bf16 cos:", g_db_bf.cos)
    all_pass = all_pass and (g_dx_bf.cos >= 0.99) and (g_dg_bf.cos >= 0.99) and (g_db_bf.cos >= 0.99)

    print("")
    if all_pass:
        print("ALL NORM BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("NORM BACKWARD PARITY FAILURE")
