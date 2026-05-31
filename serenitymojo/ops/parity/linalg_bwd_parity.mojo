# linalg_bwd_parity.mojo — GPU verification of the GEMM-family backward kernels.
#
# Tier 2 gate (cos >= 0.999 vs PyTorch autograd): matmul, bmm, linear, addbias
# backward. Inputs are the SAME deterministic fills the oracle uses; only the
# reference grads are read from linalg_bwd_ref.txt.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/linalg_bwd_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/linalg_bwd_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.linalg_backward import (
    mm_backward,
    bmm_backward,
    linear_backward,
    addbias_backward,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/linalg_bwd_ref.txt"
)


# ── Deterministic fills — MUST match linalg_bwd_oracle.py fill() exactly. ─────
#   v[i] = (((i*a) % m) - sub) * 0.05
def _fill(n: Int, a: Int, m: Int, sub: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * a) % m) - sub) * 0.05)
    return out^


def _shape2(r: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(r)
    s.append(c)
    return s^


# ── read one tagged space-separated float line from the ref file ─────────────
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

    # ── matmul: C[M,N] = A[M,K] @ B[K,N] ─────────────────────────────────────
    comptime M = 6
    comptime N = 5
    comptime K = 7
    var a = Tensor.from_host(_fill(M * K, 7, 13, 6.0), _shape2(M, K), STDtype.F32, ctx)
    var b = Tensor.from_host(_fill(K * N, 5, 11, 5.0), _shape2(K, N), STDtype.F32, ctx)
    var gc = Tensor.from_host(_fill(M * N, 3, 9, 4.0), _shape2(M, N), STDtype.F32, ctx)
    var mm = mm_backward(gc, a, b, M, N, K, ctx)
    var r_mm_da = h.compare(mm.d_a, _read_ref(String("matmul_da")), ctx)
    var r_mm_db = h.compare(mm.d_b, _read_ref(String("matmul_db")), ctx)
    print("matmul_da vs torch:", r_mm_da)
    print("matmul_db vs torch:", r_mm_db)
    all_pass = all_pass and r_mm_da.passed and r_mm_db.passed

    # ── matmul BF16: same inputs downcast to BF16, SAME F32 torch ref. ─────────
    # Relaxed gate cos >= 0.99 (BF16 ~3 decimal digits; F32-exact 0.999 is
    # unreachable). mm_backward's BF16 arm casts up to F32, runs the SAME GEMMs,
    # casts grads back to BF16 — "runs & approximately right".
    var a_bf = cast_tensor(a, STDtype.BF16, ctx)
    var b_bf = cast_tensor(b, STDtype.BF16, ctx)
    var gc_bf = cast_tensor(gc, STDtype.BF16, ctx)
    var mm_bf = mm_backward(gc_bf, a_bf, b_bf, M, N, K, ctx)
    var r_mm_da_bf = h.compare(mm_bf.d_a, _read_ref(String("matmul_da")), ctx)
    var r_mm_db_bf = h.compare(mm_bf.d_b, _read_ref(String("matmul_db")), ctx)
    print("matmul_da _bf16 cos:", r_mm_da_bf.cos)
    print("matmul_db _bf16 cos:", r_mm_db_bf.cos)
    all_pass = all_pass and (r_mm_da_bf.cos >= 0.99) and (r_mm_db_bf.cos >= 0.99)

    # ── bmm: C[Bt,M,N] = A[Bt,M,K] @ B[Bt,K,N] ───────────────────────────────
    comptime Bt = 3
    var ba = Tensor.from_host(_fill(Bt * M * K, 7, 13, 6.0), _shape2(Bt * M, K), STDtype.F32, ctx)
    var bb = Tensor.from_host(_fill(Bt * K * N, 5, 11, 5.0), _shape2(Bt * K, N), STDtype.F32, ctx)
    var bgc = Tensor.from_host(_fill(Bt * M * N, 3, 9, 4.0), _shape2(Bt * M, N), STDtype.F32, ctx)
    var bmm = bmm_backward(bgc, ba, bb, Bt, M, N, K, ctx)
    var r_bmm_da = h.compare(bmm.d_a, _read_ref(String("bmm_da")), ctx)
    var r_bmm_db = h.compare(bmm.d_b, _read_ref(String("bmm_db")), ctx)
    print("bmm_da vs torch:", r_bmm_da)
    print("bmm_db vs torch:", r_bmm_db)
    all_pass = all_pass and r_bmm_da.passed and r_bmm_db.passed

    # ── bmm BF16: same inputs downcast to BF16, SAME F32 torch ref, thr 0.99. ──
    # bmm_backward's BF16 arm casts up to F32, runs the SAME per-batch GEMMs,
    # casts grads back to BF16.
    var ba_bf = cast_tensor(ba, STDtype.BF16, ctx)
    var bb_bf = cast_tensor(bb, STDtype.BF16, ctx)
    var bgc_bf = cast_tensor(bgc, STDtype.BF16, ctx)
    var bmm_bf = bmm_backward(bgc_bf, ba_bf, bb_bf, Bt, M, N, K, ctx)
    var r_bmm_da_bf = h.compare(bmm_bf.d_a, _read_ref(String("bmm_da")), ctx)
    var r_bmm_db_bf = h.compare(bmm_bf.d_b, _read_ref(String("bmm_db")), ctx)
    print("bmm_da _bf16 cos:", r_bmm_da_bf.cos)
    print("bmm_db _bf16 cos:", r_bmm_db_bf.cos)
    all_pass = all_pass and (r_bmm_da_bf.cos >= 0.99) and (r_bmm_db_bf.cos >= 0.99)

    # ── linear: y = x @ Wᵀ + b ───────────────────────────────────────────────
    comptime Mf = 4
    comptime inf = 7
    comptime outf = 5
    var x = Tensor.from_host(_fill(Mf * inf, 7, 13, 6.0), _shape2(Mf, inf), STDtype.F32, ctx)
    var w = Tensor.from_host(_fill(outf * inf, 5, 11, 5.0), _shape2(outf, inf), STDtype.F32, ctx)
    var gy = Tensor.from_host(_fill(Mf * outf, 2, 7, 3.0), _shape2(Mf, outf), STDtype.F32, ctx)
    var lin = linear_backward(gy, x, w, Mf, inf, outf, ctx)
    var r_lin_dx = h.compare(lin.d_x, _read_ref(String("linear_dx")), ctx)
    var r_lin_dw = h.compare(lin.d_w, _read_ref(String("linear_dw")), ctx)
    var r_lin_db = h.compare(lin.d_b, _read_ref(String("linear_db")), ctx)
    print("linear_dx vs torch:", r_lin_dx)
    print("linear_dw vs torch:", r_lin_dw)
    print("linear_db vs torch:", r_lin_db)
    all_pass = all_pass and r_lin_dx.passed and r_lin_dw.passed and r_lin_db.passed

    # ── linear BF16: same inputs downcast to BF16, SAME F32 torch ref, thr 0.99.
    var x_bf = cast_tensor(x, STDtype.BF16, ctx)
    var w_bf = cast_tensor(w, STDtype.BF16, ctx)
    var gy_bf = cast_tensor(gy, STDtype.BF16, ctx)
    var lin_bf = linear_backward(gy_bf, x_bf, w_bf, Mf, inf, outf, ctx)
    var r_lin_dx_bf = h.compare(lin_bf.d_x, _read_ref(String("linear_dx")), ctx)
    var r_lin_dw_bf = h.compare(lin_bf.d_w, _read_ref(String("linear_dw")), ctx)
    var r_lin_db_bf = h.compare(lin_bf.d_b, _read_ref(String("linear_db")), ctx)
    print("linear_dx _bf16 cos:", r_lin_dx_bf.cos)
    print("linear_dw _bf16 cos:", r_lin_dw_bf.cos)
    print("linear_db _bf16 cos:", r_lin_db_bf.cos)
    all_pass = all_pass and (r_lin_dx_bf.cos >= 0.99) and (r_lin_dw_bf.cos >= 0.99) and (r_lin_db_bf.cos >= 0.99)

    # ── addbias: y = x + b ───────────────────────────────────────────────────
    comptime Ma = 4
    comptime outa = 5
    var gya = Tensor.from_host(_fill(Ma * outa, 2, 7, 3.0), _shape2(Ma, outa), STDtype.F32, ctx)
    var ab = addbias_backward(gya, Ma, outa, ctx)
    var r_ab_db = h.compare(ab.d_b, _read_ref(String("addbias_db")), ctx)
    print("addbias_db vs torch:", r_ab_db)
    all_pass = all_pass and r_ab_db.passed
    # passthrough sanity: d_x must equal grad_y (gated against the same fill).
    var r_ab_dx = h.compare(ab.d_x, _fill(Ma * outa, 2, 7, 3.0), ctx)
    print("addbias_dx (passthrough) vs grad_y:", r_ab_dx)
    all_pass = all_pass and r_ab_dx.passed

    print("")
    if all_pass:
        print("ALL LINALG BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("LINALG BACKWARD PARITY FAILURE")
        raise Error("linalg_bwd_parity gate failed")
