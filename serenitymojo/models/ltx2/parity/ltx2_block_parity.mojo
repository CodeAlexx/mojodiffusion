# serenitymojo/models/ltx2/parity/ltx2_block_parity.mojo
#
# PARITY GATE for the LTX-2 core video transformer block training unit
# (models/ltx2/ltx2_block.mojo). Loads the EXACT inputs + torch-autograd
# reference grads dumped by ltx2_block_oracle.py, runs the packaged
# ltx2_block_forward + ltx2_block_backward (use_lora=True), and compares the
# input grad, EVERY base-weight grad, the modulation-vector grads, and every
# LoRA d_A/d_B at cos >= 0.999.
#
# REAL LTX-2 head count H = 32. Small Dh/S/FF for a fast oracle. NON-DEGENERATE
# sinusoidal/random inputs (no modular fills that alias and fake zero grads).
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/ltx2/parity/ltx2_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/ltx2/parity/ltx2_block_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.ltx2.ltx2_block import (
    LTX2BlockWeights, LTX2ModVecs, LTX2Lora,
    ltx2_block_forward, ltx2_block_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/ltx2/parity/"

# dims MUST match ltx2_block_oracle.py
comptime H = 32
comptime Dh = 16
comptime D = H * Dh        # 512
comptime S = 6
comptime FF = 32
comptime RANK = 4
comptime ALPHA = Float32(1.0)
comptime SCALE = ALPHA / Float32(RANK)
comptime EPS = Float32(1e-06)


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


def _load_weights(ctx: DeviceContext) raises -> LTX2BlockWeights:
    return LTX2BlockWeights(
        _in("in_wq"), _in("in_bq"), _in("in_wk"), _in("in_bk"),
        _in("in_wv"), _in("in_bv"), _in("in_wo"), _in("in_bo"),
        _in("in_qnorm"), _in("in_knorm"),
        _in("in_gate_w"), _in("in_gate_b"),
        _in("in_wff0"), _in("in_bff0"), _in("in_wff2"), _in("in_bff2"),
        D, H, FF, ctx,
    )


def _load_mod() raises -> LTX2ModVecs:
    return LTX2ModVecs(
        _in("in_shift_msa"), _in("in_scale_msa"), _in("in_gate_msa"),
        _in("in_shift_mlp"), _in("in_scale_mlp"), _in("in_gate_mlp"),
    )


def _lora(a_name: String, b_name: String) raises -> LTX2Lora:
    return LTX2Lora(_in(a_name), _in(b_name), RANK, D, D, SCALE)


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
    print("==== ltx2_block_parity (LTX-2 core video block vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " FF=", FF, " RANK=", RANK)

    var hidden = _in("in_hidden")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var lq = _lora("in_lq_a", "in_lq_b")
    var lk = _lora("in_lk_a", "in_lk_b")
    var lv = _lora("in_lv_a", "in_lv_b")
    var lo = _lora("in_lo_a", "in_lo_b")
    var cos = Tensor.from_host(_in("in_cos"), [S * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("in_sin"), [S * H, Dh // 2], STDtype.F32, ctx)

    # ── forward ──
    var fwd = ltx2_block_forward[H, Dh, S](
        hidden.copy(), w, mv, cos, sin, lq, lk, lv, lo, True,
        D, FF, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("ref_out"), allok)

    # ── backward ──
    var d_out = _in("in_d_out")
    var g = ltx2_block_backward[H, Dh, S](
        d_out, w, mv, fwd.saved, cos, sin, lq, lk, lv, lo, True,
        D, FF, EPS, ctx,
    )

    print("")
    print("---- input grad vs torch ----")
    _check(harness, "d_hidden", g.d_hidden, _in("ref_d_hidden"), allok)

    print("")
    print("---- base weight grads vs torch ----")
    _check(harness, "d_wq    ", g.d_wq, _in("ref_d_wq"), allok)
    _check(harness, "d_bq    ", g.d_bq, _in("ref_d_bq"), allok)
    _check(harness, "d_wk    ", g.d_wk, _in("ref_d_wk"), allok)
    _check(harness, "d_bk    ", g.d_bk, _in("ref_d_bk"), allok)
    _check(harness, "d_wv    ", g.d_wv, _in("ref_d_wv"), allok)
    _check(harness, "d_bv    ", g.d_bv, _in("ref_d_bv"), allok)
    _check(harness, "d_wo    ", g.d_wo, _in("ref_d_wo"), allok)
    _check(harness, "d_bo    ", g.d_bo, _in("ref_d_bo"), allok)
    _check(harness, "d_qnorm ", g.d_q_norm, _in("ref_d_qnorm"), allok)
    _check(harness, "d_knorm ", g.d_k_norm, _in("ref_d_knorm"), allok)
    _check(harness, "d_gate_w", g.d_gate_w, _in("ref_d_gate_w"), allok)
    _check(harness, "d_gate_b", g.d_gate_b, _in("ref_d_gate_b"), allok)
    _check(harness, "d_wff0  ", g.d_wff0, _in("ref_d_wff0"), allok)
    _check(harness, "d_bff0  ", g.d_bff0, _in("ref_d_bff0"), allok)
    _check(harness, "d_wff2  ", g.d_wff2, _in("ref_d_wff2"), allok)
    _check(harness, "d_bff2  ", g.d_bff2, _in("ref_d_bff2"), allok)

    print("")
    print("---- modulation-vector grads vs torch ----")
    _check(harness, "d_shift_msa", g.d_shift_msa, _in("ref_d_shift_msa"), allok)
    _check(harness, "d_scale_msa", g.d_scale_msa, _in("ref_d_scale_msa"), allok)
    _check(harness, "d_gate_msa ", g.d_gate_msa, _in("ref_d_gate_msa"), allok)
    _check(harness, "d_shift_mlp", g.d_shift_mlp, _in("ref_d_shift_mlp"), allok)
    _check(harness, "d_scale_mlp", g.d_scale_mlp, _in("ref_d_scale_mlp"), allok)
    _check(harness, "d_gate_mlp ", g.d_gate_mlp, _in("ref_d_gate_mlp"), allok)

    print("")
    print("---- LoRA d_A / d_B vs torch ----")
    _check(harness, "d_lq_a", g.d_lq_a, _in("ref_d_lq_a"), allok)
    _check(harness, "d_lq_b", g.d_lq_b, _in("ref_d_lq_b"), allok)
    _check(harness, "d_lk_a", g.d_lk_a, _in("ref_d_lk_a"), allok)
    _check(harness, "d_lk_b", g.d_lk_b, _in("ref_d_lk_b"), allok)
    _check(harness, "d_lv_a", g.d_lv_a, _in("ref_d_lv_a"), allok)
    _check(harness, "d_lv_b", g.d_lv_b, _in("ref_d_lv_b"), allok)
    _check(harness, "d_lo_a", g.d_lo_a, _in("ref_d_lo_a"), allok)
    _check(harness, "d_lo_b", g.d_lo_b, _in("ref_d_lo_b"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — LTX-2 core video block fwd+bwd+LoRA matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
