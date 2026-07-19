# serenitymojo/models/klein/parity/single_block_lora_parity.mojo
#
# PARITY GATE for the Klein SINGLE-STREAM DiT block LoRA training variant
# (models/klein/single_block.mojo `single_block_lora_forward/backward`). Loads
# the EXACT inputs + torch-autograd reference grads dumped by
# single_block_lora_oracle.py, runs the LoRA-aware forward+backward, and compares
# d_A AND d_B for both adapters (qkv-rows on w1, cols on w2) at cos >= 0.999,
# plus the base d_x (no-regression check).
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/single_block_lora_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       serenitymojo/models/klein/parity/single_block_lora_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecs,
    SingleBlockLora,
    single_block_lora_forward, single_block_lora_backward,
)
from serenitymojo.models.klein.lora_block import LoraAdapter


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/klein/parity/"

# dims MUST match single_block_lora_oracle.py
comptime H = 32
comptime Dh = 16
comptime D = H * Dh        # 512
comptime S = 6
comptime F = 24
comptime EPS = Float32(1e-06)
comptime RANK = 8
comptime LSCALE = Float32(16.0) / Float32(8.0)   # alpha/rank = 2.0


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


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _load_weights(ctx: DeviceContext) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _in("slin_w_w1"), _in("slin_w_w2"),
        _in("slin_w_q_norm"), _in("slin_w_k_norm"),
        D, F, Dh, ctx,
    )


def _load_mod() raises -> SingleModVecs:
    return SingleModVecs(
        _in("slin_m_shift"), _in("slin_m_scale"), _in("slin_m_gate"),
    )


def _make_adapter(
    a: List[Float32], b: List[Float32], in_f: Int, out_f: Int
) -> LoraAdapter:
    return LoraAdapter(
        a.copy(), b.copy(), RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


def _load_lora() raises -> SingleBlockLora:
    var qkv = _make_adapter(_in("slin_lo_qkv_A"), _in("slin_lo_qkv_B"), D, 3 * D)
    var out = _make_adapter(_in("slin_lo_out_A"), _in("slin_lo_out_B"), D, D)
    return SingleBlockLora(Optional[LoraAdapter](qkv^), Optional[LoraAdapter](out^))


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
    print("==== single_block_lora_parity (Klein single-stream + LoRA vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " F=", F, " RANK=", RANK)

    var x = _in("slin_x")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var lora = _load_lora()
    var cos_h = _in("slin_cos")
    var sin_h = _in("slin_sin")
    var cos = Tensor.from_host(cos_h, [S * H, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(sin_h, [S * H, Dh // 2], STDtype.F32, ctx)

    var fwd = single_block_lora_forward[H, Dh, S](
        x.copy(), w, mv, lora, cos, sin,
        D, F, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("slref_out"), allok)

    var d_out = _in("slin_d_out")
    var g = single_block_lora_backward[H, Dh, S](
        d_out, w, mv, lora, fwd.saved, cos, sin,
        D, F, EPS, ctx,
    )

    print("")
    print("---- base input grad vs torch (no-regression) ----")
    _check(harness, "d_x", g.base.d_x, _in("slref_d_x"), allok)

    print("")
    print("---- LoRA grads d_A / d_B vs torch ----")
    _check(harness, "qkv d_A", g.qkv_d_a, _in("slref_qkv_dA"), allok)
    _check(harness, "qkv d_B", g.qkv_d_b, _in("slref_qkv_dB"), allok)
    _check(harness, "out d_A", g.out_d_a, _in("slref_out_dA"), allok)
    _check(harness, "out d_B", g.out_d_b, _in("slref_out_dB"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Klein single-stream LoRA fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
