# optim_parity.mojo — GPU verification of the AdamW / SGD / grad-clip port.
#
# Phase T4 gate (FULL_PORT_TRAINING_PLAN §4/§5): one optimizer step on a fixed
# (param, grad, moment) tuple must match torch at cos >= 0.999; the clip
# total_norm scalar within rtol 1e-4.
#
# Reproduces the SAME deterministic param/grad fills as optim_oracle.py on the
# device, runs the REAL training/optim.mojo step(s) (in-place mut API), and
# compares the updated PARAMETER (and the clip total_norm) against the tags.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/training/parity/optim_oracle.py
#   pixi run mojo run -I . serenitymojo/training/parity/optim_parity.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.training.optim import adamw_step, sgd_step, clip_grad_global_norm
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/training/parity/optim_ref.txt"
)

# Hyperparameters — MUST match optim_oracle.py exactly.
comptime LR = Float32(1.0e-3)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime EPS = Float32(1.0e-8)
comptime WD = Float32(0.01)
comptime SGD_LR = Float32(0.1)
comptime SGD_MOMENTUM = Float32(0.9)
comptime SGD_WD = Float32(0.0)
comptime N = 64
comptime MAX_NORM = Float32(1.0)


# Deterministic fills — MUST match optim_oracle.py `fill`.
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


# ── run `steps` AdamW steps from the deterministic init, return updated param ─
def _run_adamw(steps: Int, ctx: DeviceContext) raises -> List[Float32]:
    var p = Tensor.from_host(_fill(N, 7, 13, 6.0, 0.05), _shape1(N), STDtype.F32, ctx)
    var m = Tensor.from_host(_zeros(N), _shape1(N), STDtype.F32, ctx)
    var v = Tensor.from_host(_zeros(N), _shape1(N), STDtype.F32, ctx)
    for t in range(1, steps + 1):
        # fresh grad each step (same deterministic fill, matches the oracle)
        var g = Tensor.from_host(_fill(N, 5, 11, 5.0, 0.05), _shape1(N), STDtype.F32, ctx)
        adamw_step(p, g, m, v, t, LR, BETA1, BETA2, EPS, WD, ctx)
    return p.to_host(ctx)


def _run_sgd(steps: Int, ctx: DeviceContext) raises -> List[Float32]:
    var p = Tensor.from_host(_fill(N, 7, 13, 6.0, 0.05), _shape1(N), STDtype.F32, ctx)
    var buf = Tensor.from_host(_zeros(N), _shape1(N), STDtype.F32, ctx)
    for _ in range(steps):
        var g = Tensor.from_host(_fill(N, 5, 11, 5.0, 0.05), _shape1(N), STDtype.F32, ctx)
        sgd_step(p, g, buf, SGD_LR, SGD_MOMENTUM, SGD_WD, ctx)
    return p.to_host(ctx)


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True

    # ── AdamW 1 step ─────────────────────────────────────────────────────────
    var a1 = _run_adamw(1, ctx)
    var rA1 = h.compare_host(a1, _read_ref(String("adamw_p1")))
    print("adamw_p1 vs torch:", rA1)
    all_pass = all_pass and rA1.passed

    # ── AdamW 3 steps (bias correction over t) ───────────────────────────────
    var a3 = _run_adamw(3, ctx)
    var rA3 = h.compare_host(a3, _read_ref(String("adamw_p3")))
    print("adamw_p3 vs torch:", rA3)
    all_pass = all_pass and rA3.passed

    # ── SGD 1 step ───────────────────────────────────────────────────────────
    var s1 = _run_sgd(1, ctx)
    var rS1 = h.compare_host(s1, _read_ref(String("sgd_p1")))
    print("sgd_p1 vs torch:", rS1)
    all_pass = all_pass and rS1.passed

    # ── SGD 3 steps (momentum accumulation) ──────────────────────────────────
    var s3 = _run_sgd(3, ctx)
    var rS3 = h.compare_host(s3, _read_ref(String("sgd_p3")))
    print("sgd_p3 vs torch:", rS3)
    all_pass = all_pass and rS3.passed

    # ── grad clip: two grads, global norm > max_norm ─────────────────────────
    var g1 = Tensor.from_host(_fill(N, 5, 11, 5.0, 0.5), _shape1(N), STDtype.F32, ctx)
    var g2 = Tensor.from_host(_fill(N, 3, 9, 4.0, 0.5), _shape1(N), STDtype.F32, ctx)
    var total_norm = clip_grad_global_norm(g1, g2, MAX_NORM, ctx)

    # concat scaled grads in the same order as the oracle (g1 then g2)
    var scaled = List[Float32]()
    var c1h = g1.to_host(ctx)
    for i in range(len(c1h)):
        scaled.append(c1h[i])
    var c2h = g2.to_host(ctx)
    for i in range(len(c2h)):
        scaled.append(c2h[i])
    var rClip = h.compare_host(scaled, _read_ref(String("clip_scaled")))
    print("clip_scaled vs torch:", rClip)
    all_pass = all_pass and rClip.passed

    # total_norm scalar within rtol 1e-4
    var ref_norm = _read_ref(String("clip_norm"))[0]
    var rel = (total_norm - ref_norm) / ref_norm
    if rel < 0.0:
        rel = -rel
    var norm_pass = rel <= 1.0e-4
    print(
        "clip_norm: mojo=", total_norm, " torch=", ref_norm,
        " rel=", rel, " passed=", norm_pass,
    )
    all_pass = all_pass and norm_pass

    print("")
    if all_pass:
        print("ALL OPTIMIZER GATES PASSED (cos >= 0.999, clip_norm rtol <= 1e-4)")
    else:
        print("OPTIMIZER PARITY FAILURE")
        raise Error("optim_parity gate failed")
