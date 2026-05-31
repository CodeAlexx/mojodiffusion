# activation_bwd_parity.mojo — GPU verification of the Tier-1 activation
# BACKWARD kernels (serenitymojo/ops/activation_backward.mojo).
#
# Gate (FULL_PORT_TRAINING_PLAN §4): grad-parity cos >= 0.999 of d_x vs a
# PyTorch reference (activation_bwd_oracle.py → activation_bwd_ref.txt).
#
# Inputs use the SAME deterministic fills as activation_bwd_oracle.gen_inputs.
# Only the reference GRADIENTS are read from the ref file (tagged lines).
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/activation_bwd_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/activation_bwd_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.ops.activation_backward import (
    relu_backward,
    sigmoid_backward,
    tanh_backward,
    silu_backward,
    gelu_backward,
)
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime N = 4096
comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/ops/parity/activation_bwd_ref.txt"
)


# Deterministic fills — MUST match activation_bwd_oracle.gen_inputs.
def _fill_x(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 7) % 13) - 6.0) * 0.5)
    return out^


def _fill_go(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append((Float32((i * 5) % 11) - 5.0) * 0.3)
    return out^


def _shape1(n: Int) -> List[Int]:
    var s = List[Int]()
    s.append(n)
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

    var x = Tensor.from_host(_fill_x(N), _shape1(N), STDtype.F32, ctx)
    var go = Tensor.from_host(_fill_go(N), _shape1(N), STDtype.F32, ctx)

    var dx_relu = relu_backward(go, x, ctx)
    var r_relu = h.compare_host(dx_relu.to_host(ctx), _read_ref(String("relu_dx")))
    print("relu_dx    vs torch:", r_relu)
    all_pass = all_pass and r_relu.passed

    var dx_gelu = gelu_backward(go, x, ctx)
    var r_gelu = h.compare_host(dx_gelu.to_host(ctx), _read_ref(String("gelu_dx")))
    print("gelu_dx    vs torch:", r_gelu)
    all_pass = all_pass and r_gelu.passed

    var dx_silu = silu_backward(go, x, ctx)
    var r_silu = h.compare_host(dx_silu.to_host(ctx), _read_ref(String("silu_dx")))
    print("silu_dx    vs torch:", r_silu)
    all_pass = all_pass and r_silu.passed

    var dx_sig = sigmoid_backward(go, x, ctx)
    var r_sig = h.compare_host(dx_sig.to_host(ctx), _read_ref(String("sigmoid_dx")))
    print("sigmoid_dx vs torch:", r_sig)
    all_pass = all_pass and r_sig.passed

    var dx_tanh = tanh_backward(go, x, ctx)
    var r_tanh = h.compare_host(dx_tanh.to_host(ctx), _read_ref(String("tanh_dx")))
    print("tanh_dx    vs torch:", r_tanh)
    all_pass = all_pass and r_tanh.passed

    print("")
    if all_pass:
        print("ALL ACTIVATION BACKWARD GATES PASSED (cos >= 0.999 vs PyTorch)")
    else:
        print("ACTIVATION BACKWARD PARITY FAILURE")
        raise Error("activation_bwd_parity gate failed")
