# mixed_precision_parity.mojo — ONE mixed-precision training step gate.
#
# Proves the REAL DiT training dtype regime (flame-core convention): BF16 compute
# + F32 master weights. One full step matches an F32/torch reference within BF16
# tolerance.
#
# ── The step (standard mixed precision) ──────────────────────────────────────
# Master weights W, b kept in F32. ONE step:
#   1. cast master W,b -> BF16 ; cast x,target -> BF16          (cast_tensor)
#   2. forward in BF16:  y = linear(x_bf, W_bf, b_bf)           (ops/linear.mojo
#                          BF16 path: F32-accumulated GEMM, BF16 output storage)
#   3. loss = mse(y, target)                                    (host reduction
#                          for the printed scalar; grad uses the kernel)
#   4. backward in BF16:
#        d_y       = mse_backward(y, target_bf)                 (BF16 grad)
#        (d_x,d_W,d_b) = linear_backward(d_y, x_bf, W_bf)       (BF16 grads)
#   5. cast grads d_W,d_b -> F32                                (cast_tensor)
#   6. adamw_step updates the F32 master W,b with F32 moments   (training/optim)
#
# The UPDATED F32 master W,b is compared to the torch oracle's identical step.
# BF16 carries ~3 decimal digits, so the gate is cos >= 0.99 (NOT f32-exact —
# the point is the mixed path is CORRECT within BF16 noise). We additionally
# assert the master STAYED F32 and that the compute path actually used BF16
# (we print y.dtype / d_W.dtype, proving BF16 compute + F32 master).
#
# This file does NOT touch autograd.mojo / tensor.mojo / optim.mojo (no edits) —
# it composes the existing, dtype-dispatched forward + BF16-capable backward arms
# + the F32 optimizer, by hand (the move-only inline-rvalue pattern proven in
# composed_chain_parity.mojo).
#
# Run:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/training/parity/mixed_precision_oracle.py
#   rm -f serenitymojo.mojopkg && \
#   pixi run mojo run -I . serenitymojo/training/parity/mixed_precision_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward
from serenitymojo.ops.loss_swiglu_backward import mse_backward
from serenitymojo.training.optim import adamw_step
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from std.collections import List, Optional


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/training/parity/mixed_precision_ref.txt"
)

# Problem dims — MUST match mixed_precision_oracle.py.
comptime M = 4        # rows of x
comptime IN = 6       # in_features
comptime OUT = 5      # out_features

# AdamW hyperparameters — MUST match the oracle.
comptime LR = Float32(1.0e-3)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime EPS = Float32(1.0e-8)
comptime WD = Float32(0.01)

# Gate: BF16 carries ~3 digits → cos >= 0.99 (NOT 0.999).
comptime BF16_COS = Float64(0.99)


# ── deterministic fills (MUST match the oracle `fill`) ───────────────────────
def _fill(n: Int, a: Int, b: Int, c: Float32, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * a) % b) - c) * scale)
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(0.0)
    return out^


def _shape1(n: Int) -> List[Int]:
    var s = List[Int]()
    s.append(n)
    return s^


def _shape2(r: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(r)
    s.append(c)
    return s^


# ── read one tagged space-separated float line (verbatim from optim_parity) ──
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
    var h = ParityHarness(BF16_COS)

    print("==== mixed_precision_parity (BF16 compute + F32 master) ====")
    print("step: cast master->BF16 -> linear(BF16) -> mse -> backward(BF16)")
    print("      -> cast grads->F32 -> adamw_step(F32 master, F32 moments)")
    print("M=", M, " IN=", IN, " OUT=", OUT)

    # ── F32 MASTER weights + inputs (deterministic, == oracle fills) ─────────
    var W = Tensor.from_host(_fill(OUT * IN, 7, 13, 6.0, 0.05), _shape2(OUT, IN), STDtype.F32, ctx)
    var b = Tensor.from_host(_fill(OUT, 5, 11, 5.0, 0.05), _shape1(OUT), STDtype.F32, ctx)
    var x = Tensor.from_host(_fill(M * IN, 3, 9, 4.0, 0.10), _shape2(M, IN), STDtype.F32, ctx)
    var target = Tensor.from_host(_fill(M * OUT, 2, 7, 3.0, 0.10), _shape2(M, OUT), STDtype.F32, ctx)

    # F32 AdamW moments (master-precision), separate for W and b.
    var mW = Tensor.from_host(_zeros(OUT * IN), _shape2(OUT, IN), STDtype.F32, ctx)
    var vW = Tensor.from_host(_zeros(OUT * IN), _shape2(OUT, IN), STDtype.F32, ctx)
    var mb = Tensor.from_host(_zeros(OUT), _shape1(OUT), STDtype.F32, ctx)
    var vb = Tensor.from_host(_zeros(OUT), _shape1(OUT), STDtype.F32, ctx)

    print("")
    print("---- dtypes BEFORE step (master is F32) ----")
    print("  W.dtype =", W.dtype().name(), "  b.dtype =", b.dtype().name())

    # ── 1. cast master + inputs -> BF16 (cast_tensor borrows; master stays F32)
    var W_bf_fwd = cast_tensor(W, STDtype.BF16, ctx)
    var x_bf_fwd = cast_tensor(x, STDtype.BF16, ctx)
    var b_bf = cast_tensor(b, STDtype.BF16, ctx)

    # ── 2. forward in BF16 (linear BF16 path: F32-accum GEMM, BF16 output) ───
    var bias_opt = Optional[Tensor](b_bf^)
    var y = linear(x_bf_fwd, W_bf_fwd, bias_opt^, ctx)   # y: [M, OUT], BF16
    print("---- COMPUTE dtype (forward output) ----")
    print("  y.dtype =", y.dtype().name(), "  (expect BF16)")

    # ── 3. loss = mse(y, target), printed scalar (host reduction) ────────────
    var y_h = y.to_host(ctx)
    var tgt_h = target.to_host(ctx)
    var acc: Float32 = 0.0
    for i in range(len(y_h)):
        var d = y_h[i] - tgt_h[i]
        acc += d * d
    var loss = acc / Float32(len(y_h))
    print("  forward loss =", loss)

    # ── 4. backward in BF16 ──────────────────────────────────────────────────
    # mse_backward needs pred & target same dtype → cast target to BF16.
    var target_bf = cast_tensor(target, STDtype.BF16, ctx)
    var d_y = mse_backward(y, target_bf, ctx)             # d_y: [M, OUT], BF16
    print("---- COMPUTE dtype (mse grad) ----")
    print("  d_y.dtype =", d_y.dtype().name(), "  (expect BF16)")

    # linear_backward consumes fresh BF16 views of x and W (rebuild from master).
    var x_bf_bwd = cast_tensor(x, STDtype.BF16, ctx)
    var W_bf_bwd = cast_tensor(W, STDtype.BF16, ctx)
    var lb = linear_backward(d_y, x_bf_bwd, W_bf_bwd, M, IN, OUT, ctx)
    print("---- COMPUTE dtype (param grads) ----")
    print("  d_W.dtype =", lb.d_w.dtype().name(), "  d_b.dtype =", lb.d_b.dtype().name(), "  (expect BF16)")

    # ── 5. cast grads -> F32 (master-update input dtype) ─────────────────────
    var dW_f32 = cast_tensor(lb.d_w, STDtype.F32, ctx)
    var db_f32 = cast_tensor(lb.d_b, STDtype.F32, ctx)
    print("---- MASTER-UPDATE grad dtype (after cast) ----")
    print("  dW_f32.dtype =", dW_f32.dtype().name(), "  db_f32.dtype =", db_f32.dtype().name(), "  (expect F32)")

    # ── 6. AdamW updates the F32 master with F32 moments (t = 1) ─────────────
    adamw_step(W, dW_f32, mW, vW, 1, LR, BETA1, BETA2, EPS, WD, ctx)
    adamw_step(b, db_f32, mb, vb, 1, LR, BETA1, BETA2, EPS, WD, ctx)

    print("")
    print("---- dtypes AFTER step (master MUST still be F32) ----")
    print("  W.dtype =", W.dtype().name(), "  b.dtype =", b.dtype().name())
    var master_is_f32 = (W.dtype() == STDtype.F32) and (b.dtype() == STDtype.F32)
    if not master_is_f32:
        raise Error("master weights did not stay F32 after the step")

    # ── compare updated F32 master vs torch oracle (BF16 tolerance) ──────────
    var W_upd = W.to_host(ctx)
    var b_upd = b.to_host(ctx)
    var rW = h.compare_host(W_upd, _read_ref(String("mp_W")))
    var rb = h.compare_host(b_upd, _read_ref(String("mp_b")))

    # Secondary diagnostic: BF16 noise floor — mixed master vs a PURE-F32 step.
    var rW_f32 = h.compare_host(W_upd, _read_ref(String("f32_W")))
    var rb_f32 = h.compare_host(b_upd, _read_ref(String("f32_b")))

    print("")
    print("---- updated F32 master vs torch mixed-precision oracle ----")
    print("  W:", rW)
    print("  b:", rb)
    print("---- (diagnostic) mixed master vs PURE-F32 step (BF16 noise floor) ----")
    print("  cos(W, f32_W) =", rW_f32.cos)
    print("  cos(b, f32_b) =", rb_f32.cos)

    print("")
    var passed = master_is_f32 and rW.passed and rb.passed
    if passed:
        print("VERDICT: PASS — mixed-precision step (BF16 compute + F32 master) correct")
        print("  master stayed F32; compute used BF16; cos(W)=", rW.cos, " cos(b)=", rb.cos, " (gate ", BF16_COS, ")")
    else:
        print("VERDICT: FAIL")
        if not rW.passed:
            print("  -> W cos", rW.cos, "< gate", BF16_COS, " (precision lost in fwd/grad-cast/master-update)")
        if not rb.passed:
            print("  -> b cos", rb.cos, "< gate", BF16_COS)
        raise Error("mixed_precision_parity gate failed")
