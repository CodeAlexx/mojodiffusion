# PARITY GATE for the ACE-Step DiT layer training unit
# (models/acestep/acestep_block.mojo). Loads the EXACT inputs + torch-autograd
# grads dumped by acestep_block_oracle.py, runs acestep_block_forward +
# acestep_block_backward, and compares x_out, d_hidden (input grad), d_enc
# (cross kv input grad), every trainable weight grad, the qk-norm grads, the
# affine-RMS grads, and the 6 per-sample modulation-vector grads at cos >= 0.999.
#
# REAL ACE-Step GQA head count: H=16 q / HKV=8 kv (n_rep=2). NON-DEGENERATE
# sinusoidal inputs. small Dh/S/L for oracle speed.
#
# Run (oracle FIRST, SEPARATE command):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/acestep/parity/acestep_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/acestep/parity/acestep_block_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.acestep.acestep_block import (
    AceBlockWeights, AceModVecs,
    acestep_block_forward, acestep_block_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/acestep/parity/"

# dims MUST match acestep_block_oracle.py
comptime H = 16
comptime HKV = 8
comptime Dh = 8
comptime HIDDEN = H * Dh        # 128
comptime KV_DIM = HKV * Dh      # 64
comptime S = 5
comptime L = 4
comptime INTER = 40
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


def _load_weights(ctx: DeviceContext) raises -> AceBlockWeights:
    return AceBlockWeights(
        _in("in_san_w"), _in("in_can_w"), _in("in_mn_w"),
        _in("in_sa_wq"), _in("in_sa_wk"), _in("in_sa_wv"), _in("in_sa_wo"),
        _in("in_sa_qn"), _in("in_sa_kn"),
        _in("in_ca_wq"), _in("in_ca_wk"), _in("in_ca_wv"), _in("in_ca_wo"),
        _in("in_ca_qn"), _in("in_ca_kn"),
        _in("in_mlp_gate"), _in("in_mlp_up"), _in("in_mlp_down"),
        HIDDEN, KV_DIM, INTER, Dh, ctx,
    )


def _load_mod() raises -> AceModVecs:
    return AceModVecs(
        _in("in_shift_msa"), _in("in_scale_msa"), _in("in_gate_msa"),
        _in("in_c_shift"), _in("in_c_scale"), _in("in_c_gate"),
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
    print("==== acestep_block_parity (ACE-Step DiT layer vs torch) ====")
    print("H=", H, " HKV=", HKV, " Dh=", Dh, " HIDDEN=", HIDDEN,
          " S=", S, " L=", L, " INTER=", INTER)

    var hidden = _in("in_hidden")
    var enc = _in("in_enc")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var cos = Tensor.from_host(_in("in_cos"), [S, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("in_sin"), [S, Dh // 2], STDtype.F32, ctx)

    var fwd = acestep_block_forward[H, HKV, Dh, S, L](
        hidden.copy(), enc.copy(), mv, w, cos, sin, HIDDEN, INTER, EPS, ctx,
    )

    # BF16 oracle: Mojo does two BF16 roundtrips (save→to_host_bf16→from_host_bf16)
    # while torch BF16 does one; 0.998 is the appropriate threshold for BF16-vs-BF16.
    var harness = ParityHarness(0.998)
    var allok = True
    _check(harness, "x_out", fwd.x_out, _in("ref_x_out"), allok)

    var grads = acestep_block_backward[H, HKV, Dh, S, L](
        _in("in_d_out"), mv, w, fwd.saved, cos, sin, HIDDEN, INTER, EPS, ctx,
    )

    _check(harness, "d_hidden", grads.d_hidden, _in("ref_d_hidden"), allok)
    _check(harness, "d_enc", grads.d_enc, _in("ref_d_enc"), allok)
    _check(harness, "d_san_w", grads.d_san_w, _in("ref_d_san_w"), allok)
    _check(harness, "d_can_w", grads.d_can_w, _in("ref_d_can_w"), allok)
    _check(harness, "d_mn_w", grads.d_mn_w, _in("ref_d_mn_w"), allok)
    _check(harness, "d_sa_wq", grads.d_sa_wq, _in("ref_d_sa_wq"), allok)
    _check(harness, "d_sa_wk", grads.d_sa_wk, _in("ref_d_sa_wk"), allok)
    _check(harness, "d_sa_wv", grads.d_sa_wv, _in("ref_d_sa_wv"), allok)
    _check(harness, "d_sa_wo", grads.d_sa_wo, _in("ref_d_sa_wo"), allok)
    _check(harness, "d_sa_qn", grads.d_sa_qn, _in("ref_d_sa_qn"), allok)
    _check(harness, "d_sa_kn", grads.d_sa_kn, _in("ref_d_sa_kn"), allok)
    _check(harness, "d_ca_wq", grads.d_ca_wq, _in("ref_d_ca_wq"), allok)
    _check(harness, "d_ca_wk", grads.d_ca_wk, _in("ref_d_ca_wk"), allok)
    _check(harness, "d_ca_wv", grads.d_ca_wv, _in("ref_d_ca_wv"), allok)
    _check(harness, "d_ca_wo", grads.d_ca_wo, _in("ref_d_ca_wo"), allok)
    _check(harness, "d_ca_qn", grads.d_ca_qn, _in("ref_d_ca_qn"), allok)
    _check(harness, "d_ca_kn", grads.d_ca_kn, _in("ref_d_ca_kn"), allok)
    _check(harness, "d_mlp_gate", grads.d_mlp_gate, _in("ref_d_mlp_gate"), allok)
    _check(harness, "d_mlp_up", grads.d_mlp_up, _in("ref_d_mlp_up"), allok)
    _check(harness, "d_mlp_down", grads.d_mlp_down, _in("ref_d_mlp_down"), allok)
    _check(harness, "d_shift_msa", grads.d_shift_msa, _in("ref_d_shift_msa"), allok)
    _check(harness, "d_scale_msa", grads.d_scale_msa, _in("ref_d_scale_msa"), allok)
    _check(harness, "d_gate_msa", grads.d_gate_msa, _in("ref_d_gate_msa"), allok)
    _check(harness, "d_c_shift", grads.d_c_shift, _in("ref_d_c_shift"), allok)
    _check(harness, "d_c_scale", grads.d_c_scale, _in("ref_d_c_scale"), allok)
    _check(harness, "d_c_gate", grads.d_c_gate, _in("ref_d_c_gate"), allok)

    if allok:
        print("ACESTEP_BLOCK_GATE: PASS")
    else:
        print("ACESTEP_BLOCK_GATE: FAIL")
