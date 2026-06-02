# reduce_bwd_parity.mojo — GPU verification of the Tier-0/Tier-1 BACKWARD arms.
#
# Phase T1 gate (FULL_PORT_TRAINING_PLAN §5): grad-parity cos >= 0.999 of
# sqrt/square/log/softmax/logsoftmax/sum/mean backward vs a PyTorch reference
# (reduce_bwd_oracle.py -> reduce_bwd_ref.txt).
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/reduce_bwd_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/reduce_bwd_parity.mojo

from std.math import exp as _expf, log as _logf
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.reduce_backward import (
    sqrt_backward, square_backward, log_backward,
    softmax_backward, logsoftmax_backward, sum_backward, mean_backward,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/reduce_bwd_ref.txt"
)

comptime N_ELEM = 64
comptime ROWS = 8
comptime COLS = 16
comptime COLS_WIDE = 1024   # > _TPB=256; exercises the wide reduction path


# Deterministic fills — MUST match reduce_bwd_oracle.py.
def _fill_pos(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32(i % 13) + 1.0) * 0.25)
    return out^


def _fill_signed(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 7) % 13) - 6.0) * 0.1)
    return out^


def _fill_grad(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 2) % 7) - 3.0) * 0.05)
    return out^


def _shape1(n: Int) -> List[Int]:
    var s = List[Int]()
    s.append(n)
    return s^


def _shape2(r: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(r); s.append(c)
    return s^


# Compute softmax(x) over last dim on host (F64-style F32) to feed the bwd, which
# takes the softmax OUTPUT. Matches torch.softmax used by the oracle.
def _softmax_host(x: List[Float32], rows: Int, cols: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(rows * cols):
        out.append(Float32(0.0))
    for r in range(rows):
        var m = x[r * cols]
        for c in range(1, cols):
            if x[r * cols + c] > m:
                m = x[r * cols + c]
        var s = Float32(0.0)
        for c in range(cols):
            var e = _expf(x[r * cols + c] - m)
            out[r * cols + c] = e
            s += e
        for c in range(cols):
            out[r * cols + c] = out[r * cols + c] / s
    return out^


def _logsoftmax_host(x: List[Float32], rows: Int, cols: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(rows * cols):
        out.append(Float32(0.0))
    for r in range(rows):
        var m = x[r * cols]
        for c in range(1, cols):
            if x[r * cols + c] > m:
                m = x[r * cols + c]
        var s = Float32(0.0)
        for c in range(cols):
            s += _expf(x[r * cols + c] - m)
        var logsum = m + _logf(s)
        for c in range(cols):
            out[r * cols + c] = x[r * cols + c] - logsum
    return out^


# ── read one tagged space-separated float line (copied from sdpa_bwd_parity) ──
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

    var g_elem = _fill_grad(N_ELEM)
    var x_pos = _fill_pos(N_ELEM)

    # ── sqrt / square / log (positive x) ─────────────────────────────────────
    var d_sqrt = sqrt_backward(
        Tensor.from_host(g_elem, _shape1(N_ELEM), STDtype.F32, ctx),
        Tensor.from_host(x_pos, _shape1(N_ELEM), STDtype.F32, ctx), ctx)
    var r_sqrt = h.compare_host(d_sqrt.to_host(ctx), _read_ref(String("sqrt_dx")))
    print("sqrt_dx   vs torch:", r_sqrt)
    all_pass = all_pass and r_sqrt.passed

    # ── sqrt BF16: same inputs downcast to BF16, SAME F32 ref, thr 0.99. ───────
    # sqrt/square/log all route through _elementwise_bwd, whose BF16 arm casts up
    # to F32, runs the SAME kernel, casts grad back to BF16.
    var d_sqrt_bf = sqrt_backward(
        cast_tensor(Tensor.from_host(g_elem, _shape1(N_ELEM), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(x_pos, _shape1(N_ELEM), STDtype.F32, ctx), STDtype.BF16, ctx), ctx)
    var r_sqrt_bf = h.compare_host(d_sqrt_bf.to_host(ctx), _read_ref(String("sqrt_dx")))
    print("sqrt_dx   _bf16 cos:", r_sqrt_bf.cos)
    all_pass = all_pass and (r_sqrt_bf.cos >= 0.99)

    var d_sq = square_backward(
        Tensor.from_host(g_elem, _shape1(N_ELEM), STDtype.F32, ctx),
        Tensor.from_host(x_pos, _shape1(N_ELEM), STDtype.F32, ctx), ctx)
    var r_sq = h.compare_host(d_sq.to_host(ctx), _read_ref(String("square_dx")))
    print("square_dx vs torch:", r_sq)
    all_pass = all_pass and r_sq.passed

    # ── square BF16: SAME F32 ref, thr 0.99 (via _elementwise_bwd BF16 arm). ────
    var d_sq_bf = square_backward(
        cast_tensor(Tensor.from_host(g_elem, _shape1(N_ELEM), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(x_pos, _shape1(N_ELEM), STDtype.F32, ctx), STDtype.BF16, ctx), ctx)
    var r_sq_bf = h.compare_host(d_sq_bf.to_host(ctx), _read_ref(String("square_dx")))
    print("square_dx _bf16 cos:", r_sq_bf.cos)
    all_pass = all_pass and (r_sq_bf.cos >= 0.99)

    var d_log = log_backward(
        Tensor.from_host(g_elem, _shape1(N_ELEM), STDtype.F32, ctx),
        Tensor.from_host(x_pos, _shape1(N_ELEM), STDtype.F32, ctx), ctx)
    var r_log = h.compare_host(d_log.to_host(ctx), _read_ref(String("log_dx")))
    print("log_dx    vs torch:", r_log)
    all_pass = all_pass and r_log.passed

    # ── log BF16: SAME F32 ref, thr 0.99 (via _elementwise_bwd BF16 arm). ──────
    var d_log_bf = log_backward(
        cast_tensor(Tensor.from_host(g_elem, _shape1(N_ELEM), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(x_pos, _shape1(N_ELEM), STDtype.F32, ctx), STDtype.BF16, ctx), ctx)
    var r_log_bf = h.compare_host(d_log_bf.to_host(ctx), _read_ref(String("log_dx")))
    print("log_dx    _bf16 cos:", r_log_bf.cos)
    all_pass = all_pass and (r_log_bf.cos >= 0.99)

    # ── softmax / logsoftmax (2D [ROWS, COLS], reduce over COLS) ──────────────
    var x_sig = _fill_signed(ROWS * COLS)
    var g_sm = _fill_grad(ROWS * COLS)
    var sm_out = _softmax_host(x_sig, ROWS, COLS)
    var lsm_out = _logsoftmax_host(x_sig, ROWS, COLS)

    var d_sm = softmax_backward(
        Tensor.from_host(g_sm, _shape2(ROWS, COLS), STDtype.F32, ctx),
        Tensor.from_host(sm_out, _shape2(ROWS, COLS), STDtype.F32, ctx), ctx)
    var r_sm = h.compare_host(d_sm.to_host(ctx), _read_ref(String("softmax_dx")))
    print("softmax_dx vs torch:", r_sm)
    all_pass = all_pass and r_sm.passed

    # ── softmax BF16: same inputs downcast to BF16, SAME F32 ref, thr 0.99. ────
    # softmax_backward's BF16 arm casts up to F32, runs the SAME row-reduction
    # kernel, casts grad back to BF16. BF16 ~3 decimal digits -> gate 0.99.
    var d_sm_bf = softmax_backward(
        cast_tensor(Tensor.from_host(g_sm, _shape2(ROWS, COLS), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(sm_out, _shape2(ROWS, COLS), STDtype.F32, ctx), STDtype.BF16, ctx),
        ctx)
    var r_sm_bf = h.compare_host(d_sm_bf.to_host(ctx), _read_ref(String("softmax_dx")))
    print("softmax_dx _bf16 cos:", r_sm_bf.cos)
    all_pass = all_pass and (r_sm_bf.cos >= 0.99)

    var d_lsm = logsoftmax_backward(
        Tensor.from_host(g_sm, _shape2(ROWS, COLS), STDtype.F32, ctx),
        Tensor.from_host(lsm_out, _shape2(ROWS, COLS), STDtype.F32, ctx), ctx)
    var r_lsm = h.compare_host(
        d_lsm.to_host(ctx), _read_ref(String("logsoftmax_dx")))
    print("logsoftmax_dx vs torch:", r_lsm)
    all_pass = all_pass and r_lsm.passed

    # ── logsoftmax BF16: same inputs downcast to BF16, SAME F32 ref, thr 0.99. ──
    # logsoftmax_backward's BF16 arm casts up to F32, runs the SAME row-reduction
    # kernel, casts grad back to BF16.
    var d_lsm_bf = logsoftmax_backward(
        cast_tensor(Tensor.from_host(g_sm, _shape2(ROWS, COLS), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(lsm_out, _shape2(ROWS, COLS), STDtype.F32, ctx), STDtype.BF16, ctx), ctx)
    var r_lsm_bf = h.compare_host(d_lsm_bf.to_host(ctx), _read_ref(String("logsoftmax_dx")))
    print("logsoftmax_dx _bf16 cos:", r_lsm_bf.cos)
    all_pass = all_pass and (r_lsm_bf.cos >= 0.99)

    # ── softmax / logsoftmax WIDE (cols=1024 > _TPB): real attention width ────
    var x_sig_w = _fill_signed(ROWS * COLS_WIDE)
    var g_sm_w = _fill_grad(ROWS * COLS_WIDE)
    var sm_out_w = _softmax_host(x_sig_w, ROWS, COLS_WIDE)
    var lsm_out_w = _logsoftmax_host(x_sig_w, ROWS, COLS_WIDE)

    var d_sm_w = softmax_backward(
        Tensor.from_host(g_sm_w, _shape2(ROWS, COLS_WIDE), STDtype.F32, ctx),
        Tensor.from_host(sm_out_w, _shape2(ROWS, COLS_WIDE), STDtype.F32, ctx), ctx)
    var r_sm_w = h.compare_host(
        d_sm_w.to_host(ctx), _read_ref(String("softmax_wide_dx")))
    print("softmax_wide_dx (cols=1024) vs torch:", r_sm_w)
    all_pass = all_pass and r_sm_w.passed

    var d_lsm_w = logsoftmax_backward(
        Tensor.from_host(g_sm_w, _shape2(ROWS, COLS_WIDE), STDtype.F32, ctx),
        Tensor.from_host(lsm_out_w, _shape2(ROWS, COLS_WIDE), STDtype.F32, ctx), ctx)
    var r_lsm_w = h.compare_host(
        d_lsm_w.to_host(ctx), _read_ref(String("logsoftmax_wide_dx")))
    print("logsoftmax_wide_dx (cols=1024) vs torch:", r_lsm_w)
    all_pass = all_pass and r_lsm_w.passed

    # ── sum / mean (scalar grad 1.0 broadcast to [N_ELEM]) ───────────────────
    var d_sum = sum_backward(Float32(1.0), _shape1(N_ELEM), ctx)
    var r_sum = h.compare_host(d_sum.to_host(ctx), _read_ref(String("sum_dx")))
    print("sum_dx    vs torch:", r_sum)
    all_pass = all_pass and r_sum.passed

    var d_mean = mean_backward(Float32(1.0), _shape1(N_ELEM), ctx)
    var r_mean = h.compare_host(d_mean.to_host(ctx), _read_ref(String("mean_dx")))
    print("mean_dx   vs torch:", r_mean)
    all_pass = all_pass and r_mean.passed

    print("")
    if all_pass:
        print("ALL REDUCE BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("REDUCE BACKWARD PARITY FAILURE")
        raise Error("reduce_bwd_parity gate failed")
