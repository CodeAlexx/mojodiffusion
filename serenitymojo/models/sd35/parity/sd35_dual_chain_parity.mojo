# serenitymojo/models/sd35/parity/sd35_dual_chain_parity.mojo
#
# WIRING GATE: chain block A (DUAL) -> block B (STANDARD), proving the stack
# dispatch threads a dual block into a following standard block in forward AND
# backward (the exact composition sd35_stack_lora does for blocks 0-12 then 13+).
# Both block fwd/bwd are individually gate-verified; this checks the plumbing:
# A.(ctx_out,x_out) feed B; B's input grads become A's upstream grads.
#
# Run (oracle FIRST):
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sd35/parity/sd35_dual_chain_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/sd35/parity/sd35_dual_chain_parity.mojo -o /tmp/sd35_dual_chain
#   /tmp/sd35_dual_chain

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.sd35.sd35_block import (
    StreamWeights, Attn2Weights, ModVecs, JointBlockWeights,
    sd35_dual_joint_block_forward, sd35_dual_joint_block_backward,
    sd35_joint_block_forward, sd35_joint_block_backward,
)

comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/sd35/parity/"
comptime H = 24
comptime Dh = 8
comptime D = H * Dh
comptime N_CTX = 3
comptime N_IMG = 5
comptime S = N_CTX + N_IMG
comptime MLP = 32
comptime EPS = Float32(1e-6)
comptime QK_EPS = Float32(1e-6)
comptime SCALE = Float32(1.0) / Float32(2.828427)


def _read(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the oracle first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _in(name: String) raises -> List[Float32]:
    return _read(REF_DIR + name + ".bin")


def _stream(pfx: String) raises -> StreamWeights:
    return StreamWeights(
        _in(pfx + "_wqkv"), _in(pfx + "_bqkv"), _in(pfx + "_wproj"), _in(pfx + "_bproj"),
        _in(pfx + "_wfc1"), _in(pfx + "_bfc1"), _in(pfx + "_wfc2"), _in(pfx + "_bfc2"),
        _in(pfx + "_q_norm"), _in(pfx + "_k_norm"),
    )


def _attn2(pfx: String) raises -> Attn2Weights:
    return Attn2Weights(
        _in(pfx + "_wqkv"), _in(pfx + "_bqkv"), _in(pfx + "_wproj"), _in(pfx + "_bproj"),
        _in(pfx + "_q_norm"), _in(pfx + "_k_norm"),
    )


def _mod(pfx: String) raises -> ModVecs:
    return ModVecs(
        _in(pfx + "_shift_msa"), _in(pfx + "_scale_msa"), _in(pfx + "_gate_msa"),
        _in(pfx + "_shift_mlp"), _in(pfx + "_scale_mlp"), _in(pfx + "_gate_mlp"),
    )


def _check(mut h: ParityHarness, name: String, a: List[Float32], b: List[Float32], mut ok: Bool) raises:
    var r = h.compare_host(a, b)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        ok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== sd35_dual_chain_parity (dual block A -> standard block B) ====")

    var ctx0 = _in("chain_ctx0")
    var x0 = _in("chain_x0")
    # block A (dual)
    var Acw = _stream("chain_Acw")
    var Axw = _stream("chain_Axw")
    var Aa2 = _attn2("chain_Aa2")
    var Acm = _mod("chain_Acm")
    var Axm = _mod("chain_Axm")
    var A_shift2 = _in("chain_Axm2_shift_msa2")
    var A_scale2 = _in("chain_Axm2_scale_msa2")
    var A_gate2 = _in("chain_Axm2_gate_msa2")
    # block B (standard)
    var Bw = JointBlockWeights(_stream("chain_Bcw"), _stream("chain_Bxw"))
    var Bcm = _mod("chain_Bcm")
    var Bxm = _mod("chain_Bxm")

    var harness = ParityHarness()
    var ok = True

    # ── forward: A (dual) then B (standard) ──
    var fa = sd35_dual_joint_block_forward[1, S, N_IMG, H, Dh](
        ctx0.copy(), x0.copy(), Acw, Axw, Aa2, Acm, Axm,
        A_shift2.copy(), A_scale2.copy(), A_gate2.copy(),
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
    )
    var fb = sd35_joint_block_forward[1, S, H, Dh](
        fa.ctx_out.copy(), fa.x_out.copy(), Bw, Bcm, Bxm,
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
    )
    print("")
    print("---- chained forward outputs ----")
    _check(harness, "ctx_out", fb.ctx_out, _in("chain_ctx_out"), ok)
    _check(harness, "x_out  ", fb.x_out, _in("chain_x_out"), ok)

    # ── backward: B (standard) then A (dual) ──
    var gb = sd35_joint_block_backward[1, S, H, Dh](
        _in("chain_d_ctx_up"), _in("chain_d_x_up"), Bw, Bcm, Bxm, fb,
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
    )
    var ga = sd35_dual_joint_block_backward[1, S, N_IMG, H, Dh](
        gb.d_ctx, gb.d_x, Acw, Axw, Aa2, Acm, Axm,
        A_scale2.copy(), A_gate2.copy(), fa,
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
    )
    print("")
    print("---- input grads threaded back through BOTH blocks ----")
    _check(harness, "d_ctx0", ga.d_ctx, _in("chain_d_ctx0"), ok)
    _check(harness, "d_x0  ", ga.d_x, _in("chain_d_x0"), ok)
    print("")
    print("---- weight grads reach each block ----")
    _check(harness, "A.attn2 d_wqkv", ga.a2_g.d_wqkv, _in("chain_A_a2_d_wqkv"), ok)
    _check(harness, "B.x     d_wqkv", gb.x_g.d_wqkv, _in("chain_B_x_d_wqkv"), ok)

    print("")
    if ok:
        print("VERDICT: PASS — dual->standard chain threads fwd+bwd correctly (cos>=0.999)")
    else:
        print("VERDICT: FAIL — chain composition diverged (see FAIL lines)")
