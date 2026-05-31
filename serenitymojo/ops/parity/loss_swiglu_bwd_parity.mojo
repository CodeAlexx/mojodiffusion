# loss_swiglu_bwd_parity.mojo — GPU verification of MSE / Huber / SwiGLU BACKWARD.
#
# Phase T gate (FULL_PORT_TRAINING_PLAN): grad-parity cos >= 0.999 of
#   mse_backward.d_pred, huber_backward.d_pred, swiglu_backward.{d_gate,d_up}
# vs a PyTorch reference (loss_swiglu_bwd_oracle.py -> loss_swiglu_bwd_ref.txt).
#
# Inputs use the SAME deterministic fills as loss_swiglu_bwd_oracle.py; only the
# reference GRADIENTS are read from the ref file. _read_ref mirrors
# sdpa_bwd_parity.mojo verbatim.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/loss_swiglu_bwd_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/loss_swiglu_bwd_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.loss_swiglu_backward import (
    mse_backward,
    huber_backward,
    swiglu_backward,
)
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/loss_swiglu_bwd_ref.txt"
)

# MUST match loss_swiglu_bwd_oracle.py N and DELTA.
comptime N = 4096
comptime DELTA = Float32(1.0)


# Deterministic fills — MUST match loss_swiglu_bwd_oracle.py.
def _fill_pred(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 7) % 13) - 6.0) * 0.25)
    return out^


def _fill_target(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 5) % 11) - 5.0) * 0.25)
    return out^


def _fill_gate(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 3) % 9) - 4.0) * 0.20)
    return out^


def _fill_up(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 2) % 7) - 3.0) * 0.20)
    return out^


def _fill_grad_out(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 11) % 5) - 2.0) * 0.10)
    return out^


def _flat(n: Int) -> List[Int]:
    var s = List[Int]()
    s.append(n)
    return s^


# ── read one tagged space-separated float line (verbatim from sdpa_bwd_parity) ─
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

    # ── MSE backward ─────────────────────────────────────────────────────────
    var d_mse = mse_backward(
        Tensor.from_host(_fill_pred(N), _flat(N), STDtype.F32, ctx),
        Tensor.from_host(_fill_target(N), _flat(N), STDtype.F32, ctx),
        ctx,
    )
    var r_mse = h.compare_host(d_mse.to_host(ctx), _read_ref(String("mse_dpred")))
    print("MSE   d_pred vs torch:", r_mse)
    all_pass = all_pass and r_mse.passed

    # ── MSE BF16: same inputs downcast to BF16, SAME F32 ref, thr 0.99. ────────
    # mse_backward's BF16 arm casts up to F32, runs the SAME kernel, casts grad
    # back to BF16. BF16 ~3 decimal digits -> 0.999 unreachable, gate 0.99.
    var d_mse_bf = mse_backward(
        cast_tensor(Tensor.from_host(_fill_pred(N), _flat(N), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(_fill_target(N), _flat(N), STDtype.F32, ctx), STDtype.BF16, ctx),
        ctx,
    )
    var r_mse_bf = h.compare_host(d_mse_bf.to_host(ctx), _read_ref(String("mse_dpred")))
    print("MSE   d_pred _bf16 cos:", r_mse_bf.cos)
    all_pass = all_pass and (r_mse_bf.cos >= 0.99)

    # ── Huber backward ───────────────────────────────────────────────────────
    var d_hub = huber_backward(
        Tensor.from_host(_fill_pred(N), _flat(N), STDtype.F32, ctx),
        Tensor.from_host(_fill_target(N), _flat(N), STDtype.F32, ctx),
        DELTA,
        ctx,
    )
    var r_hub = h.compare_host(
        d_hub.to_host(ctx), _read_ref(String("huber_dpred"))
    )
    print("Huber d_pred vs torch:", r_hub)
    all_pass = all_pass and r_hub.passed

    # ── Huber BF16: same inputs downcast to BF16, SAME F32 ref, thr 0.99. ───────
    # huber_backward's BF16 arm casts up to F32, runs the SAME clamp kernel, casts
    # grad back to BF16. BF16 ~3 decimal digits -> 0.999 unreachable, gate 0.99.
    var d_hub_bf = huber_backward(
        cast_tensor(Tensor.from_host(_fill_pred(N), _flat(N), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(_fill_target(N), _flat(N), STDtype.F32, ctx), STDtype.BF16, ctx),
        DELTA,
        ctx,
    )
    var r_hub_bf = h.compare_host(d_hub_bf.to_host(ctx), _read_ref(String("huber_dpred")))
    print("Huber d_pred _bf16 cos:", r_hub_bf.cos)
    all_pass = all_pass and (r_hub_bf.cos >= 0.99)

    # ── SwiGLU backward ──────────────────────────────────────────────────────
    var sg = swiglu_backward(
        Tensor.from_host(_fill_grad_out(N), _flat(N), STDtype.F32, ctx),
        Tensor.from_host(_fill_gate(N), _flat(N), STDtype.F32, ctx),
        Tensor.from_host(_fill_up(N), _flat(N), STDtype.F32, ctx),
        ctx,
    )
    var r_dg = h.compare_host(
        sg.d_gate.to_host(ctx), _read_ref(String("swiglu_dgate"))
    )
    var r_du = h.compare_host(
        sg.d_up.to_host(ctx), _read_ref(String("swiglu_dup"))
    )
    print("SwiGLU d_gate vs torch:", r_dg)
    print("SwiGLU d_up   vs torch:", r_du)
    all_pass = all_pass and r_dg.passed and r_du.passed

    # ── SwiGLU BF16: same inputs downcast to BF16, SAME F32 ref, thr 0.99. ─────
    var sg_bf = swiglu_backward(
        cast_tensor(Tensor.from_host(_fill_grad_out(N), _flat(N), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(_fill_gate(N), _flat(N), STDtype.F32, ctx), STDtype.BF16, ctx),
        cast_tensor(Tensor.from_host(_fill_up(N), _flat(N), STDtype.F32, ctx), STDtype.BF16, ctx),
        ctx,
    )
    var r_dg_bf = h.compare_host(sg_bf.d_gate.to_host(ctx), _read_ref(String("swiglu_dgate")))
    var r_du_bf = h.compare_host(sg_bf.d_up.to_host(ctx), _read_ref(String("swiglu_dup")))
    print("SwiGLU d_gate _bf16 cos:", r_dg_bf.cos)
    print("SwiGLU d_up   _bf16 cos:", r_du_bf.cos)
    all_pass = all_pass and (r_dg_bf.cos >= 0.99) and (r_du_bf.cos >= 0.99)

    print("")
    if all_pass:
        print("ALL LOSS+SWIGLU BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("LOSS+SWIGLU BACKWARD PARITY FAILURE")
        raise Error("loss_swiglu_bwd_parity gate failed")
