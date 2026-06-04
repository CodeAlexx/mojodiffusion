# fused_ln_parity.mojo — gate the two fused LN ops (serenitymojo/ops/fused_ln.mojo)
# AGAINST a PyTorch GPU-bf16 oracle (fused_ln_oracle.py -> fused_ln_ref.txt).
# Mirrors norm_bwd_parity.mojo: inputs reproduced deterministically here; only
# the reference OUTPUTS cross the boundary via the tagged ref file.
#
# Computes BOTH ops in bf16 on GPU and reports cos + magnitude ratio (||actual||
# / ||ref||) for each. Gate: cos>=0.999 for each op.
#
# Run (regenerate ref first):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/ops/parity/fused_ln_oracle.py
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#       pixi run mojo run -I . serenitymojo/ops/parity/fused_ln_parity.mojo

from std.math import sqrt
from std.memory import alloc
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.ops.fused_ln import layernorm_linear, residual_layernorm

comptime REF_PATH = "/home/alex/mojodiffusion/serenitymojo/ops/parity/fused_ln_ref.txt"

comptime ROWS = 64
comptime HIDDEN = 256
comptime OUT_FEAT = 320
comptime EPS = Float32(1e-5)


# ── deterministic fill — MUST match fused_ln_oracle.py _fill ──────────────────
def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


# ── read one tagged space-separated float line (mirrors norm_bwd_parity) ──────
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


def _mag_ratio(actual: List[Float32], reference: List[Float32]) -> Float64:
    var na: Float64 = 0.0
    var nr: Float64 = 0.0
    for i in range(len(actual)):
        na += Float64(actual[i]) * Float64(actual[i])
    for i in range(len(reference)):
        nr += Float64(reference[i]) * Float64(reference[i])
    if nr == 0.0:
        return 0.0
    return sqrt(na) / sqrt(nr)


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness(0.999)
    var all_pass = True

    # ── inputs (bf16 storage, F32 host fill) ──
    var x = Tensor.from_host(_fill(ROWS * HIDDEN, 11, 2.0), [ROWS, HIDDEN], STDtype.BF16, ctx)
    var gamma = Tensor.from_host(_fill(HIDDEN, 22, 1.0), [HIDDEN], STDtype.BF16, ctx)
    var beta = Tensor.from_host(_fill(HIDDEN, 33, 0.5), [HIDDEN], STDtype.BF16, ctx)
    var weight = Tensor.from_host(_fill(OUT_FEAT * HIDDEN, 44, 0.5), [OUT_FEAT, HIDDEN], STDtype.BF16, ctx)
    var bias = Tensor.from_host(_fill(OUT_FEAT, 55, 0.5), [OUT_FEAT], STDtype.BF16, ctx)
    var residual = Tensor.from_host(_fill(ROWS * HIDDEN, 66, 2.0), [ROWS, HIDDEN], STDtype.BF16, ctx)

    print("=== fused_ln parity vs torch GPU-bf16 (rows=", ROWS, " hidden=", HIDDEN, " out=", OUT_FEAT, ") ===")

    # ── layernorm_linear ──
    var y_ll = layernorm_linear(x, gamma, beta, weight, Optional[Tensor](bias^), EPS, ctx)
    var ref_ll = _read_ref(String("layernorm_linear"))
    var act_ll = y_ll.to_host(ctx)
    var r_ll = h.compare_host(act_ll, ref_ll)
    var mag_ll = _mag_ratio(act_ll, ref_ll)
    print("    layernorm_linear  :", r_ll, " magRatio=", mag_ll)
    if not r_ll.passed:
        all_pass = False

    # ── residual_layernorm ──
    var y_rl = residual_layernorm(x, residual, gamma, beta, EPS, ctx)
    var ref_rl = _read_ref(String("residual_layernorm"))
    var act_rl = y_rl.to_host(ctx)
    var r_rl = h.compare_host(act_rl, ref_rl)
    var mag_rl = _mag_ratio(act_rl, ref_rl)
    print("    residual_layernorm:", r_rl, " magRatio=", mag_rl)
    if not r_rl.passed:
        all_pass = False

    if all_pass:
        print("PASS: fused_ln both ops match torch GPU-bf16 cos>=0.999")
    else:
        raise Error("fused_ln_parity gate FAILED")
