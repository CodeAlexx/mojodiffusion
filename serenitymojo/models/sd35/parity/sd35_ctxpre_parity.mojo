# PARITY GATE for sd35_context_preonly_forward/backward (sd3.5 final block).
# Run: oracle first, then
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/models/sd35/parity/sd35_ctxpre_parity.mojo -o /tmp/sd35_ctxpre && /tmp/sd35_ctxpre
from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.sd35.sd35_block import (
    StreamWeights, ModVecs, sd35_context_preonly_forward, sd35_context_preonly_backward,
)

comptime REF = "/home/alex/mojodiffusion/serenitymojo/models/sd35/parity/"
comptime H = 24
comptime Dh = 8
comptime D = H * Dh
comptime N_CTX = 3
comptime N_IMG = 5
comptime S = N_CTX + N_IMG
comptime MLP = 32
comptime EPS = Float32(1e-6)
comptime SCALE = Float32(1.0) / Float32(2.828427)


def _r(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0: raise Error(String("open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd); raise Error(String("empty (run oracle): ") + path)
    var buf = alloc[UInt8](n); var done = 0
    while done < n:
        var g = sys_pread(fd, buf + done, n - done, done)
        if g <= 0: break
        done += g
    _ = sys_close(fd)
    var fp = buf.bitcast[Float32](); var o = List[Float32]()
    for i in range(n // 4): o.append(fp[i])
    buf.free(); return o^


def _in(name: String) raises -> List[Float32]:
    return _r(REF + name + ".bin")


def _xw() raises -> StreamWeights:
    return StreamWeights(
        _in("cp_xw_wqkv"), _in("cp_xw_bqkv"), _in("cp_xw_wproj"), _in("cp_xw_bproj"),
        _in("cp_xw_wfc1"), _in("cp_xw_bfc1"), _in("cp_xw_wfc2"), _in("cp_xw_bfc2"),
        _in("cp_xw_q_norm"), _in("cp_xw_k_norm"),
    )


def _xm() raises -> ModVecs:
    return ModVecs(
        _in("cp_xm_shift_msa"), _in("cp_xm_scale_msa"), _in("cp_xm_gate_msa"),
        _in("cp_xm_shift_mlp"), _in("cp_xm_scale_mlp"), _in("cp_xm_gate_mlp"),
    )


def _chk(mut h: ParityHarness, name: String, a: List[Float32], b: List[Float32], mut ok: Bool) raises:
    var r = h.compare_host(a, b)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed: ok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== sd35_ctxpre_parity (context_pre_only final block) ====")
    var context = _in("cp_context")
    var x = _in("cp_x")
    var cqkv_w = _in("cp_cqkv_w"); var cqkv_b = _in("cp_cqkv_b")
    var cqn = _in("cp_cqn"); var ckn = _in("cp_ckn")
    var cscale = _in("cp_ctx_scale"); var cshift = _in("cp_ctx_shift")
    var xw = _xw(); var xm = _xm()

    var h = ParityHarness(); var ok = True
    var fwd = sd35_context_preonly_forward[1, S, H, Dh](
        context.copy(), x.copy(), cqkv_w.copy(), cqkv_b.copy(), cqn.copy(), ckn.copy(),
        cscale.copy(), cshift.copy(), xw, xm,
        N_CTX, N_IMG, D, MLP, EPS, EPS, SCALE, ctx,
    )
    print("")
    print("---- forward ----")
    _chk(h, "x_out", fwd.x_out, _in("cp_x_out"), ok)

    var g = sd35_context_preonly_backward[1, S, H, Dh](
        _in("cp_d_x_up"), cqkv_w.copy(), cqn.copy(), ckn.copy(), cscale.copy(),
        xw, xm, fwd, N_CTX, N_IMG, D, MLP, EPS, EPS, SCALE, ctx,
    )
    print("")
    print("---- input grads ----")
    _chk(h, "d_x  ", g.d_x, _in("cp_d_x"), ok)
    _chk(h, "d_ctx", g.d_ctx, _in("cp_d_ctx"), ok)
    print("")
    print("---- x weight grads ----")
    _chk(h, "x d_wqkv", g.x_g.d_wqkv, _in("cp_x_d_wqkv"), ok)
    _chk(h, "x d_wfc1", g.x_g.d_wfc1, _in("cp_x_d_wfc1"), ok)

    print("")
    if ok:
        print("VERDICT: PASS — context_pre_only block fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — see FAIL lines")
