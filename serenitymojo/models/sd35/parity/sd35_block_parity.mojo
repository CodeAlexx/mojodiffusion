# serenitymojo/models/sd35/parity/sd35_block_parity.mojo
#
# PARITY GATE for the SD3.5 MMDiT JointTransformerBlock training unit
# (models/sd35/sd35_block.mojo). Loads the EXACT inputs + torch-autograd refs
# dumped by sd35_block_oracle.py, runs sd35_joint_block_forward +
# sd35_joint_block_backward, and compares ctx/x input grads, every trainable
# weight grad (qkv/proj/fc1/fc2 + biases, qk-norms), every modulation-vector grad,
# and LoRA d_A/d_B (x_block qkv adapter) at cos >= 0.999.
#
# REAL SD3.5 head count H = 24 (the dim that PASSES sdpa backward). Small N/Dh to
# keep the torch oracle fast. NON-DEGENERATE sinusoidal/randn inputs.
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/sd35/parity/sd35_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/sd35/parity/sd35_block_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.sd35.sd35_block import (
    StreamWeights, JointBlockWeights, ModVecs,
    sd35_joint_block_forward, sd35_joint_block_backward, sd35_stream_norm,
)
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.klein.lora_block import klein_lora_fwd, klein_lora_bwd


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/sd35/parity/"

# dims MUST match sd35_block_oracle.py
comptime H = 24
comptime Dh = 8
comptime D = H * Dh        # 192
comptime N_CTX = 3
comptime N_IMG = 5
comptime S = N_CTX + N_IMG
comptime MLP = 32
comptime EPS = Float32(1e-06)
comptime QK_EPS = Float32(1e-06)
comptime RANK = 4
comptime ALPHA = Float32(8.0)


def _read_bin_f32(path: String) raises -> List[Float32]:
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
    return _read_bin_f32(REF_DIR + name + ".bin")


def _load_stream(prefix: String) raises -> StreamWeights:
    return StreamWeights(
        _in("in_" + prefix + "_wqkv"), _in("in_" + prefix + "_bqkv"),
        _in("in_" + prefix + "_wproj"), _in("in_" + prefix + "_bproj"),
        _in("in_" + prefix + "_wfc1"), _in("in_" + prefix + "_bfc1"),
        _in("in_" + prefix + "_wfc2"), _in("in_" + prefix + "_bfc2"),
        _in("in_" + prefix + "_q_norm"), _in("in_" + prefix + "_k_norm"),
    )


def _load_mod(prefix: String) raises -> ModVecs:
    return ModVecs(
        _in("in_" + prefix + "_shift_msa"), _in("in_" + prefix + "_scale_msa"),
        _in("in_" + prefix + "_gate_msa"),
        _in("in_" + prefix + "_shift_mlp"), _in("in_" + prefix + "_scale_mlp"),
        _in("in_" + prefix + "_gate_mlp"),
    )


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def main() raises:
    var ctx = DeviceContext()
    print("==== sd35_block_parity (SD3.5 MMDiT joint block vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_CTX=", N_CTX, " N_IMG=", N_IMG, " MLP=", MLP)

    var context = _in("in_context")
    var x = _in("in_x")
    var cw = _load_stream("cw")
    var xw = _load_stream("xw")
    var cm = _load_mod("cm")
    var xm = _load_mod("xm")
    var w = JointBlockWeights(cw^, xw^)
    comptime SCALE = Float32(1.0) / Float32(2.8284271247461903)  # 1/sqrt(8)

    var harness = ParityHarness()
    var allok = True

    # ── BASE forward + backward (no LoRA) ──
    var no_lora = Optional[List[Float32]](None)
    var fwd = sd35_joint_block_forward[1, S, H, Dh](
        context.copy(), x.copy(), w, cm, xm,
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx, no_lora,
    )
    print("")
    print("---- forward outputs vs torch ----")
    _check(harness, "ctx_out", fwd.ctx_out, _in("ref_ctx_out"), allok)
    _check(harness, "x_out  ", fwd.x_out, _in("ref_x_out"), allok)

    var d_ctx_out = _in("in_d_ctx")
    var d_x_out = _in("in_d_x")
    var g = sd35_joint_block_backward[1, S, H, Dh](
        d_ctx_out, d_x_out, w, cm, xm, fwd,
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
    )

    print("")
    print("---- input grads vs torch ----")
    _check(harness, "d_ctx", g.d_ctx, _in("ref_d_ctx"), allok)
    _check(harness, "d_x  ", g.d_x, _in("ref_d_x"), allok)

    print("")
    print("---- CTX (context_block) weight grads vs torch ----")
    _check(harness, "ctx d_wqkv ", g.ctx_g.d_wqkv, _in("ref_ctx_d_wqkv"), allok)
    _check(harness, "ctx d_bqkv ", g.ctx_g.d_bqkv, _in("ref_ctx_d_bqkv"), allok)
    _check(harness, "ctx d_wproj", g.ctx_g.d_wproj, _in("ref_ctx_d_wproj"), allok)
    _check(harness, "ctx d_bproj", g.ctx_g.d_bproj, _in("ref_ctx_d_bproj"), allok)
    _check(harness, "ctx d_wfc1 ", g.ctx_g.d_wfc1, _in("ref_ctx_d_wfc1"), allok)
    _check(harness, "ctx d_bfc1 ", g.ctx_g.d_bfc1, _in("ref_ctx_d_bfc1"), allok)
    _check(harness, "ctx d_wfc2 ", g.ctx_g.d_wfc2, _in("ref_ctx_d_wfc2"), allok)
    _check(harness, "ctx d_bfc2 ", g.ctx_g.d_bfc2, _in("ref_ctx_d_bfc2"), allok)
    _check(harness, "ctx d_qnorm", g.ctx_g.d_qnorm, _in("ref_ctx_d_qnorm"), allok)
    _check(harness, "ctx d_knorm", g.ctx_g.d_knorm, _in("ref_ctx_d_knorm"), allok)

    print("")
    print("---- CTX modulation-vector grads vs torch ----")
    _check(harness, "ctx d_shift_msa", g.ctx_g.d_shift_msa, _in("ref_ctx_d_shift_msa"), allok)
    _check(harness, "ctx d_scale_msa", g.ctx_g.d_scale_msa, _in("ref_ctx_d_scale_msa"), allok)
    _check(harness, "ctx d_gate_msa ", g.ctx_g.d_gate_msa, _in("ref_ctx_d_gate_msa"), allok)
    _check(harness, "ctx d_shift_mlp", g.ctx_g.d_shift_mlp, _in("ref_ctx_d_shift_mlp"), allok)
    _check(harness, "ctx d_scale_mlp", g.ctx_g.d_scale_mlp, _in("ref_ctx_d_scale_mlp"), allok)
    _check(harness, "ctx d_gate_mlp ", g.ctx_g.d_gate_mlp, _in("ref_ctx_d_gate_mlp"), allok)

    print("")
    print("---- X (x_block) weight grads vs torch ----")
    _check(harness, "x d_wqkv ", g.x_g.d_wqkv, _in("ref_x_d_wqkv"), allok)
    _check(harness, "x d_bqkv ", g.x_g.d_bqkv, _in("ref_x_d_bqkv"), allok)
    _check(harness, "x d_wproj", g.x_g.d_wproj, _in("ref_x_d_wproj"), allok)
    _check(harness, "x d_bproj", g.x_g.d_bproj, _in("ref_x_d_bproj"), allok)
    _check(harness, "x d_wfc1 ", g.x_g.d_wfc1, _in("ref_x_d_wfc1"), allok)
    _check(harness, "x d_bfc1 ", g.x_g.d_bfc1, _in("ref_x_d_bfc1"), allok)
    _check(harness, "x d_wfc2 ", g.x_g.d_wfc2, _in("ref_x_d_wfc2"), allok)
    _check(harness, "x d_bfc2 ", g.x_g.d_bfc2, _in("ref_x_d_bfc2"), allok)
    _check(harness, "x d_qnorm", g.x_g.d_qnorm, _in("ref_x_d_qnorm"), allok)
    _check(harness, "x d_knorm", g.x_g.d_knorm, _in("ref_x_d_knorm"), allok)

    print("")
    print("---- X modulation-vector grads vs torch ----")
    _check(harness, "x d_shift_msa", g.x_g.d_shift_msa, _in("ref_x_d_shift_msa"), allok)
    _check(harness, "x d_scale_msa", g.x_g.d_scale_msa, _in("ref_x_d_scale_msa"), allok)
    _check(harness, "x d_gate_msa ", g.x_g.d_gate_msa, _in("ref_x_d_gate_msa"), allok)
    _check(harness, "x d_shift_mlp", g.x_g.d_shift_mlp, _in("ref_x_d_shift_mlp"), allok)
    _check(harness, "x d_scale_mlp", g.x_g.d_scale_mlp, _in("ref_x_d_scale_mlp"), allok)
    _check(harness, "x d_gate_mlp ", g.x_g.d_gate_mlp, _in("ref_x_d_gate_mlp"), allok)

    # ── LoRA path: x_block qkv adapter (representative LoRA target) ──
    print("")
    print("---- LoRA (x_block qkv) d_A / d_B vs torch ----")
    var lora_A = _in("in_lora_A")   # [RANK, D]
    var lora_B = _in("in_lora_B")   # [3D, RANK]
    var scale = ALPHA / Float32(RANK)
    var lo = LoraAdapter(
        lora_A.copy(), lora_B.copy(), RANK, D, 3 * D, scale,
        _zeros_list(RANK * D), _zeros_list(RANK * D),
        _zeros_list(3 * D * RANK), _zeros_list(3 * D * RANK),
    )
    # The LoRA input is the x-stream `norm` = modulate(layer_norm(x), ...).
    var x_norm = sd35_stream_norm(x.copy(), xm, N_IMG, D, EPS, ctx)
    var lora_delta = klein_lora_fwd(x_norm.copy(), lo, N_IMG, ctx)   # [N_IMG, 3D]
    # Re-run forward WITH the LoRA qkv delta, then backward, capture d_qkv@x.
    var fwd_l = sd35_joint_block_forward[1, S, H, Dh](
        context.copy(), x.copy(), w, cm, xm,
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
        Optional[List[Float32]](lora_delta^),
    )
    var g_l = sd35_joint_block_backward[1, S, H, Dh](
        d_ctx_out.copy(), d_x_out.copy(), w, cm, xm, fwd_l,
        N_CTX, N_IMG, D, MLP, EPS, QK_EPS, SCALE, ctx,
    )
    var lg = klein_lora_bwd(g_l.x_d_qkv.copy(), x_norm.copy(), lo, N_IMG, ctx)
    _check(harness, "lora d_A", lg.d_a, _in("ref_lora_d_A"), allok)
    _check(harness, "lora d_B", lg.d_b, _in("ref_lora_d_B"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — SD3.5 joint block fwd+bwd+LoRA matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")


def _zeros_list(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^
