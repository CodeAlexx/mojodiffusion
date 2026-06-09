# serenitymojo/models/sd35/parity/sd35_dual_block_parity.mojo
#
# PARITY GATE for the SD3.5 DUAL-attention joint block (sd35_block.mojo
# sd35_dual_joint_block_forward/backward). Loads the torch refs dumped by
# sd35_dual_block_oracle.py (whose math is verified byte-faithful to the REAL
# diffusers JointTransformerBlock(use_dual_attention=True) in
# sd35_dual_block_vs_diffusers.py), runs the Mojo dual fwd+bwd, and compares the
# forward outputs, d_x/d_ctx, every weight grad (ctx + x + attn2), the modulation
# grads (incl the msa2 triple), and LoRA d_A/d_B on x-qkv AND attn2-qkv at cos>=0.999.
#
# Run (oracle FIRST):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sd35/parity/sd35_dual_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/sd35/parity/sd35_dual_block_parity.mojo -o /tmp/sd35_dual_block
#   /tmp/sd35_dual_block

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.sd35.sd35_block import (
    StreamWeights, Attn2Weights, ModVecs,
    sd35_dual_joint_block_forward, sd35_dual_joint_block_backward,
)

comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/sd35/parity/"
comptime H = 24
comptime Dh = 8
comptime D = H * Dh        # 192
comptime N_CTX = 3
comptime N_IMG = 5
comptime S = N_CTX + N_IMG # 8
comptime MLP = 32
comptime EPS = Float32(1e-6)
comptime QK_EPS = Float32(1e-6)
comptime SCALE = Float32(1.0) / Float32(2.828427)  # 1/sqrt(Dh=8)
comptime RANK = 4
comptime LSCALE = Float32(2.0)   # ALPHA(8)/RANK(4)


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


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _stream(pfx: String) raises -> StreamWeights:
    return StreamWeights(
        _in(pfx + "_wqkv"), _in(pfx + "_bqkv"), _in(pfx + "_wproj"), _in(pfx + "_bproj"),
        _in(pfx + "_wfc1"), _in(pfx + "_bfc1"), _in(pfx + "_wfc2"), _in(pfx + "_bfc2"),
        _in(pfx + "_q_norm"), _in(pfx + "_k_norm"),
    )


def _attn2() raises -> Attn2Weights:
    return Attn2Weights(
        _in("din_a2_wqkv"), _in("din_a2_bqkv"), _in("din_a2_wproj"), _in("din_a2_bproj"),
        _in("din_a2_q_norm"), _in("din_a2_k_norm"),
    )


def _mod(pfx: String) raises -> ModVecs:
    return ModVecs(
        _in(pfx + "_shift_msa"), _in(pfx + "_scale_msa"), _in(pfx + "_gate_msa"),
        _in(pfx + "_shift_mlp"), _in(pfx + "_scale_mlp"), _in(pfx + "_gate_mlp"),
    )


def _lora(a_name: String, b_name: String, in_f: Int, out_f: Int) raises -> LoraAdapter:
    return LoraAdapter(
        _in(a_name), _in(b_name), RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f), _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _check(mut h: ParityHarness, name: String, a: List[Float32], b: List[Float32], mut ok: Bool) raises:
    var r = h.compare_host(a, b)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        ok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== sd35_dual_block_parity (DUAL-attention joint block vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_CTX=", N_CTX, " N_IMG=", N_IMG, " MLP=", MLP)

    var context = _in("din_context")
    var x = _in("din_x")
    var cw = _stream("din_cw")
    var xw = _stream("din_xw")
    var a2 = _attn2()
    var cm = _mod("din_cm")
    var xm = _mod("din_xm")
    var shift_msa2 = _in("din_xm2_shift_msa2")
    var scale_msa2 = _in("din_xm2_scale_msa2")
    var gate_msa2 = _in("din_xm2_gate_msa2")

    var harness = ParityHarness()
    var ok = True

    # ── base (no LoRA) ──
    var fwd = sd35_dual_joint_block_forward[1, S, N_IMG, H, Dh](
        context.copy(), x.copy(), cw, xw, a2, cm, xm,
        shift_msa2.copy(), scale_msa2.copy(), gate_msa2.copy(),
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
    )
    print("")
    print("---- forward outputs ----")
    _check(harness, "ctx_out", fwd.ctx_out, _in("dref_ctx_out"), ok)
    _check(harness, "x_out  ", fwd.x_out, _in("dref_x_out"), ok)

    var g = sd35_dual_joint_block_backward[1, S, N_IMG, H, Dh](
        _in("din_d_ctx"), _in("din_d_x"), cw, xw, a2, cm, xm,
        scale_msa2.copy(), gate_msa2.copy(), fwd,
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
    )
    print("")
    print("---- input grads ----")
    _check(harness, "d_ctx", g.d_ctx, _in("dref_d_ctx"), ok)
    _check(harness, "d_x  ", g.d_x, _in("dref_d_x"), ok)

    print("")
    print("---- ctx-stream weight grads ----")
    _check(harness, "ctx d_wqkv ", g.ctx_g.d_wqkv, _in("dref_ctx_d_wqkv"), ok)
    _check(harness, "ctx d_wproj", g.ctx_g.d_wproj, _in("dref_ctx_d_wproj"), ok)
    _check(harness, "ctx d_wfc1 ", g.ctx_g.d_wfc1, _in("dref_ctx_d_wfc1"), ok)
    _check(harness, "ctx d_wfc2 ", g.ctx_g.d_wfc2, _in("dref_ctx_d_wfc2"), ok)
    _check(harness, "ctx d_qnorm", g.ctx_g.d_qnorm, _in("dref_ctx_d_q_norm"), ok)
    _check(harness, "ctx d_knorm", g.ctx_g.d_knorm, _in("dref_ctx_d_k_norm"), ok)

    print("")
    print("---- x-stream weight grads ----")
    _check(harness, "x d_wqkv ", g.x_g.d_wqkv, _in("dref_x_d_wqkv"), ok)
    _check(harness, "x d_wproj", g.x_g.d_wproj, _in("dref_x_d_wproj"), ok)
    _check(harness, "x d_wfc1 ", g.x_g.d_wfc1, _in("dref_x_d_wfc1"), ok)
    _check(harness, "x d_wfc2 ", g.x_g.d_wfc2, _in("dref_x_d_wfc2"), ok)
    _check(harness, "x d_qnorm", g.x_g.d_qnorm, _in("dref_x_d_q_norm"), ok)
    _check(harness, "x d_knorm", g.x_g.d_knorm, _in("dref_x_d_k_norm"), ok)

    print("")
    print("---- attn2 weight grads (the dual branch) ----")
    _check(harness, "a2 d_wqkv ", g.a2_g.d_wqkv, _in("dref_a2_d_wqkv"), ok)
    _check(harness, "a2 d_wproj", g.a2_g.d_wproj, _in("dref_a2_d_wproj"), ok)
    _check(harness, "a2 d_qnorm", g.a2_g.d_qnorm, _in("dref_a2_d_q_norm"), ok)
    _check(harness, "a2 d_knorm", g.a2_g.d_knorm, _in("dref_a2_d_k_norm"), ok)

    print("")
    print("---- modulation grads ----")
    _check(harness, "x d_gate_msa", g.x_g.d_gate_msa, _in("dref_x_d_gate_msa"), ok)
    _check(harness, "x d_scale_msa", g.x_g.d_scale_msa, _in("dref_x_d_scale_msa"), ok)
    _check(harness, "x d_gate_mlp", g.x_g.d_gate_mlp, _in("dref_x_d_gate_mlp"), ok)
    _check(harness, "x d_shift_msa2", g.d_shift_msa2, _in("dref_x_d_shift_msa2"), ok)
    _check(harness, "x d_scale_msa2", g.d_scale_msa2, _in("dref_x_d_scale_msa2"), ok)
    _check(harness, "x d_gate_msa2", g.d_gate_msa2, _in("dref_x_d_gate_msa2"), ok)

    # ── LoRA on x-qkv (joint) AND attn2-qkv (the new dual slot) ──
    var lx = _lora("din_lora_x_A", "din_lora_x_B", D, 3 * D)
    var la2 = _lora("din_lora_a2_A", "din_lora_a2_B", D, 3 * D)
    var lfwd = sd35_dual_joint_block_forward[1, S, N_IMG, H, Dh](
        context.copy(), x.copy(), cw, xw, a2, cm, xm,
        shift_msa2.copy(), scale_msa2.copy(), gate_msa2.copy(),
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
        Optional[LoraAdapter](lx.copy()), Optional[LoraAdapter](la2.copy()),
    )
    var lg = sd35_dual_joint_block_backward[1, S, N_IMG, H, Dh](
        _in("din_d_ctx"), _in("din_d_x"), cw, xw, a2, cm, xm,
        scale_msa2.copy(), gate_msa2.copy(), lfwd,
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
        Optional[LoraAdapter](lx.copy()), Optional[LoraAdapter](la2.copy()),
    )
    print("")
    print("---- LoRA d_A/d_B (x-qkv + attn2-qkv) ----")
    _check(harness, "lora x  d_A", lg.x_qkv_lora_d_a, _in("dref_lora_x_d_A"), ok)
    _check(harness, "lora x  d_B", lg.x_qkv_lora_d_b, _in("dref_lora_x_d_B"), ok)
    _check(harness, "lora a2 d_A", lg.a2_qkv_lora_d_a, _in("dref_lora_a2_d_A"), ok)
    _check(harness, "lora a2 d_B", lg.a2_qkv_lora_d_b, _in("dref_lora_a2_d_B"), ok)

    print("")
    if ok:
        print("VERDICT: PASS — SD3.5 dual-attention block fwd+bwd+LoRA matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines)")
