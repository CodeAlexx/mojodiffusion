# checkpoint_parity.mojo -- Phase T0 checkpoint-offload backward parity gate.
#
# FULL_PORT_TRAINING_PLAN.md Phase T0 kill-gate: prove the checkpoint-recompute
# backward produces the SAME input gradient as the non-checkpointed backward,
# and that both match torch autograd of the same block.
#
# Block: y = silu(x @ W^T).  We compute dL/dx (and dL/dW) THREE ways:
#   A) save-all   : block_backward_saveall            (in-Mojo oracle)
#   B) checkpoint : offload x -> host, restore, recompute, backprop (deliverable)
#   C) torch      : autograd of the identical block   (cross-check oracle)
#
# GATES (real measurement only -- Tenet 4):
#   cos(B_dx, A_dx) >= 0.9999   self-consistency (checkpoint == save-all)
#   cos(B_dx, C_dx) >= 0.999    torch cross-check
#   cos(restore(offload(x)), x) >= 0.99999  AND  max_abs == 0  (byte fidelity)
#   (dW checked the same way.)
#
# Run (oracle first -- it writes checkpoint_ref.txt; deterministic, no RNG):
#   cd /home/alex/mojodiffusion
#   rm -f serenitymojo.mojopkg
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/training/parity/checkpoint_oracle.py
#   pixi run mojo run -I . serenitymojo/training/parity/checkpoint_parity.mojo

from collections import List
from math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.training.checkpoint import (
    HostOffload,
    offload_to_host,
    restore_to_device,
    block_forward,
    block_backward_saveall,
    checkpoint_recompute,
    BlockGrads,
)

# Fixed problem size (small, deterministic). y = silu(x @ W^T).
comptime M = 4   # rows of x
comptime K = 6   # cols of x / cols of W  (contraction = in_features)
comptime N = 5   # rows of W / cols of y  (out_features)

comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/training/parity/checkpoint_ref.txt"
)


# ---------------------------------------------------------------------------
# Deterministic host data (no RNG -- byte-reproducible across Mojo and torch).
# MUST match checkpoint_oracle.py exactly.
# ---------------------------------------------------------------------------
def _x_host() -> List[Float32]:
    var v = List[Float32]()
    for i in range(M * K):
        v.append(Float32(i) * 0.037 - 0.3)
    return v^


def _w_host() -> List[Float32]:
    var v = List[Float32]()
    for i in range(N * K):
        v.append(Float32(i) * 0.021 - 0.25)
    return v^


def _gout_host() -> List[Float32]:
    var v = List[Float32]()
    for i in range(M * N):
        v.append(Float32(i) * 0.05 - 0.15)
    return v^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _cos(a: List[Float32], b: List[Float32]) -> Float32:
    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    var n = len(a) if len(a) < len(b) else len(b)
    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def _max_abs(a: List[Float32], b: List[Float32]) -> Float32:
    var m: Float32 = 0.0
    var n = len(a) if len(a) < len(b) else len(b)
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


# ── read one tagged space-separated float line (ported from optim_parity) ─────
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
    print("=== checkpoint-offload backward parity (Phase T0) ===")
    print("block: y = silu(x @ W^T),  M,K,N =", M, K, N)
    # The torch oracle (checkpoint_oracle.py) must be run first; it writes
    # checkpoint_ref.txt (deterministic, no RNG). See header for the command.

    # ── build the block inputs on device (from_host(values, shape, dtype, ctx)) ─
    var x = Tensor.from_host(_x_host(), _shape2(M, K), STDtype.F32, ctx)
    var w = Tensor.from_host(_w_host(), _shape2(N, K), STDtype.F32, ctx)
    var gout = Tensor.from_host(_gout_host(), _shape2(M, N), STDtype.F32, ctx)

    # Sanity: the forward runs (and the recompute path reproduces it).
    var y = block_forward(x, w, ctx)
    _ = y^

    # ── Path A: save-all backward (in-Mojo oracle) ───────────────────────────
    var ga = block_backward_saveall(gout, x, w, M, K, N, ctx)
    var a_dx = ga.dx.to_host(ctx)
    var a_dW = ga.dW.to_host(ctx)

    # ── Path B: checkpoint-offload backward (the deliverable) ────────────────
    # offload x -> host (in a real loop the device x is then dropped to free
    # VRAM); checkpoint_recompute restores + recomputes + backprops using ONLY
    # the host copy + the weight.
    var off = offload_to_host(x, ctx)
    var gb = checkpoint_recompute(off, w, gout, M, K, N, ctx)
    var b_dx = gb.dx.to_host(ctx)
    var b_dW = gb.dW.to_host(ctx)

    # ── self-consistency gate: B == A ────────────────────────────────────────
    var cos_dx_self = _cos(b_dx, a_dx)
    var cos_dW_self = _cos(b_dW, a_dW)
    var max_dx_self = _max_abs(b_dx, a_dx)
    print("[self ] cos(dx_ckpt,dx_saveall)=", cos_dx_self, " max_abs=", max_dx_self)
    print("[self ] cos(dW_ckpt,dW_saveall)=", cos_dW_self)

    # ── torch cross-check: B vs torch ────────────────────────────────────────
    var t_dx = _read_ref(String("dx"))
    var t_dW = _read_ref(String("dw"))
    var cos_dx_torch = _cos(b_dx, t_dx)
    var cos_dW_torch = _cos(b_dW, t_dW)
    var max_dx_torch = _max_abs(b_dx, t_dx)
    print("[torch] cos(dx_ckpt,dx_torch)  =", cos_dx_torch, " max_abs=", max_dx_torch)
    print("[torch] cos(dW_ckpt,dW_torch)  =", cos_dW_torch)

    # ── offload round-trip byte-exactness check ──────────────────────────────
    var off2 = offload_to_host(x, ctx)
    var rx = restore_to_device(off2, ctx)
    var rx_host = rx.to_host(ctx)
    var x_host = x.to_host(ctx)
    var cos_rt = _cos(rx_host, x_host)
    var max_rt = _max_abs(rx_host, x_host)
    print("[rtrip] cos(restore(offload(x)),x)=", cos_rt, " max_abs=", max_rt)

    # ── verdict ──────────────────────────────────────────────────────────────
    var pass_self = cos_dx_self >= 0.9999 and cos_dW_self >= 0.9999
    var pass_torch = cos_dx_torch >= 0.999 and cos_dW_torch >= 0.999
    var pass_rt = cos_rt >= 0.99999 and max_rt == 0.0
    print("")
    if pass_self and pass_torch and pass_rt:
        print("PASS checkpoint-offload parity")
    else:
        print("FAIL checkpoint-offload parity",
              " self=", pass_self, " torch=", pass_torch, " rtrip=", pass_rt)
        raise Error("checkpoint parity gate failed")
