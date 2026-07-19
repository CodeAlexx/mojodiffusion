# PARITY GATE for the ACE-Step DiT layer LoRA training variant
# (models/acestep/acestep_block.mojo acestep_block_lora_forward/backward).
# Reloads the base in_* references (weights/mod/cos/sin/d_out — identical to the
# base oracle) + the lin_*_{A,B} LoRA adapters, runs the LoRA fwd+bwd, compares
# x_out, d_hidden, d_enc + the 16 LoRA d_A/d_B (self/cross {q,k,v,o}) vs torch.
#
# Run (BOTH oracles FIRST — base oracle dumps in_*, LoRA oracle dumps lin_/lref_):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/acestep/parity/acestep_block_oracle.py
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/acestep/parity/acestep_block_lora_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/acestep/parity/acestep_block_lora_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.acestep.acestep_block import (
    AceBlockWeights, AceModVecs, AceBlockLora,
    acestep_block_lora_forward, acestep_block_lora_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/acestep/parity/"

comptime H = 16
comptime HKV = 8
comptime Dh = 8
comptime HIDDEN = H * Dh
comptime KV_DIM = HKV * Dh
comptime S = 5
comptime L = 4
comptime INTER = 40
comptime RANK = 4
comptime SCALE = Float32(1.0)
comptime EPS = Float32(1e-06)


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run oracles first): ") + path)
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
        o.append(0.0)
    return o^


def _mk_lora(name: String, in_f: Int, out_f: Int) raises -> LoraAdapter:
    var a = _in(String("lin_") + name + "_A")    # [RANK,in_f]
    var b = _in(String("lin_") + name + "_B")    # [out_f,RANK]
    return LoraAdapter(
        a^, b^, RANK, in_f, out_f, SCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


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
    print("==== acestep_block_lora_parity (ACE-Step DiT LoRA vs torch) ====")
    print("H=", H, " HKV=", HKV, " Dh=", Dh, " HIDDEN=", HIDDEN,
          " S=", S, " L=", L, " RANK=", RANK)

    var hidden = _in("in_hidden")
    var enc = _in("in_enc")
    var w = _load_weights(ctx)
    var mv = _load_mod()
    var cos = Tensor.from_host(_in("in_cos"), [S, Dh // 2], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("in_sin"), [S, Dh // 2], STDtype.F32, ctx)

    var lora = AceBlockLora(
        Optional[LoraAdapter](_mk_lora("sa_q", HIDDEN, HIDDEN)),
        Optional[LoraAdapter](_mk_lora("sa_k", HIDDEN, KV_DIM)),
        Optional[LoraAdapter](_mk_lora("sa_v", HIDDEN, KV_DIM)),
        Optional[LoraAdapter](_mk_lora("sa_o", HIDDEN, HIDDEN)),
        Optional[LoraAdapter](_mk_lora("ca_q", HIDDEN, HIDDEN)),
        Optional[LoraAdapter](_mk_lora("ca_k", HIDDEN, KV_DIM)),
        Optional[LoraAdapter](_mk_lora("ca_v", HIDDEN, KV_DIM)),
        Optional[LoraAdapter](_mk_lora("ca_o", HIDDEN, HIDDEN)),
    )

    var fwd = acestep_block_lora_forward[H, HKV, Dh, S, L](
        hidden.copy(), enc.copy(), mv, w, lora, cos, sin, HIDDEN, INTER, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True
    _check(harness, "x_out", fwd.x_out, _in("lref_x_out"), allok)

    var g = acestep_block_lora_backward[H, HKV, Dh, S, L](
        _in("in_d_out"), mv, w, lora, fwd.saved, cos, sin, HIDDEN, INTER, EPS, ctx,
    )

    _check(harness, "d_hidden", g.base.d_hidden, _in("lref_d_hidden"), allok)
    _check(harness, "d_enc", g.base.d_enc, _in("lref_d_enc"), allok)
    _check(harness, "sa_q_dA", g.sa_q_da, _in("lref_sa_q_dA"), allok)
    _check(harness, "sa_q_dB", g.sa_q_db, _in("lref_sa_q_dB"), allok)
    _check(harness, "sa_k_dA", g.sa_k_da, _in("lref_sa_k_dA"), allok)
    _check(harness, "sa_k_dB", g.sa_k_db, _in("lref_sa_k_dB"), allok)
    _check(harness, "sa_v_dA", g.sa_v_da, _in("lref_sa_v_dA"), allok)
    _check(harness, "sa_v_dB", g.sa_v_db, _in("lref_sa_v_dB"), allok)
    _check(harness, "sa_o_dA", g.sa_o_da, _in("lref_sa_o_dA"), allok)
    _check(harness, "sa_o_dB", g.sa_o_db, _in("lref_sa_o_dB"), allok)
    _check(harness, "ca_q_dA", g.ca_q_da, _in("lref_ca_q_dA"), allok)
    _check(harness, "ca_q_dB", g.ca_q_db, _in("lref_ca_q_dB"), allok)
    _check(harness, "ca_k_dA", g.ca_k_da, _in("lref_ca_k_dA"), allok)
    _check(harness, "ca_k_dB", g.ca_k_db, _in("lref_ca_k_dB"), allok)
    _check(harness, "ca_v_dA", g.ca_v_da, _in("lref_ca_v_dA"), allok)
    _check(harness, "ca_v_dB", g.ca_v_db, _in("lref_ca_v_dB"), allok)
    _check(harness, "ca_o_dA", g.ca_o_da, _in("lref_ca_o_dA"), allok)
    _check(harness, "ca_o_dB", g.ca_o_db, _in("lref_ca_o_dB"), allok)

    if allok:
        print("ACESTEP_BLOCK_LORA_GATE: PASS")
    else:
        print("ACESTEP_BLOCK_LORA_GATE: FAIL")
