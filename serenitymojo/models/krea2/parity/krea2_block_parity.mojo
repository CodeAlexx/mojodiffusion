# serenitymojo/models/krea2/parity/krea2_block_parity.mojo
#
# PARITY GATE for the Krea-2-Raw SingleStreamBlock LoRA training unit
# (models/krea2/krea2_block.mojo). Loads the EXACT inputs + torch-autograd
# reference grads dumped by krea2_block_oracle.py, runs the LoRA-aware
# forward+backward, and compares the forward output, the base input grad d_x, and
# d_A AND d_B for ALL 8 block-Linear adapters at cos >= 0.999.
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/krea2/parity/krea2_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       serenitymojo/models/krea2/parity/krea2_block_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, Krea2LoraGrad,
    krea2_single_stream_block_lora,
    krea2_single_stream_block_lora_backward,
)

comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/"

# dims MUST match krea2_block_oracle.py
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM     # 6144
comptime HALF = HEADDIM // 2            # 64
comptime L = 8
comptime MLPDIM = 256
comptime EPS = Float32(1e-5)
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


def _t(name: String, var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(_in(name), shape^, STDtype.F32, ctx))


# Tile a per-token [L, HALF] cos/sin table to [L*heads, HALF] (row l*heads+h =
# table[l]) for BSHD rope_interleaved — mirrors krea2_dit._tile_rope_table.
def _tile(name: String, heads: Int, ctx: DeviceContext) raises -> Tensor:
    var tbl = _in(name)              # [L*HALF] row-major
    var out = List[Float32]()
    for l in range(L):
        for _h in range(heads):
            for c in range(HALF):
                out.append(tbl[l * HALF + c])
    var sh = List[Int]()
    sh.append(L * heads)
    sh.append(HALF)
    return Tensor.from_host(out^, sh^, STDtype.F32, ctx)


def _adapter(
    nm: String, in_f: Int, out_f: Int, ctx: DeviceContext
) raises -> Optional[LoraAdapterDevice]:
    var a = _t("kin_lo_" + nm + "_A", _shape2(RANK, in_f), ctx)
    var b = _t("kin_lo_" + nm + "_B", _shape2(out_f, RANK), ctx)
    return Optional[LoraAdapterDevice](
        LoraAdapterDevice(a^, b^, RANK, in_f, out_f, LSCALE)
    )


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def _check_lora(
    mut harness: ParityHarness, nm: String, g: Krea2LoraGrad, mut allok: Bool,
) raises:
    if not g.d_a or not g.d_b:
        print("  ", nm, " adapter MISSING grads — FAIL")
        allok = False
        return
    _check(harness, nm + " d_A", g.d_a.value(), _in("kref_" + nm + "_dA"), allok)
    _check(harness, nm + " d_B", g.d_b.value(), _in("kref_" + nm + "_dB"), allok)


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_block_parity (Krea-2 SingleStreamBlock + LoRA vs torch) ====")
    print("HEADS=", HEADS, " KVHEADS=", KVHEADS, " HEADDIM=", HEADDIM,
          " L=", L, " MLPDIM=", MLPDIM, " RANK=", RANK)

    # ── inputs ───────────────────────────────────────────────────────────────
    var x = _t("kin_x", _shape3(1, L, FEATURES), ctx)
    var vec_h = _in("kin_vec")
    var vec = Tensor.from_host(vec_h, _shape2(1, 6 * FEATURES), STDtype.F32, ctx)

    var w = Krea2BlockWeights(
        _t("kin_W_wq", _shape2(HEADS * HEADDIM, FEATURES), ctx),
        _t("kin_W_wk", _shape2(KVHEADS * HEADDIM, FEATURES), ctx),
        _t("kin_W_wv", _shape2(KVHEADS * HEADDIM, FEATURES), ctx),
        _t("kin_W_gate", _shape2(FEATURES, FEATURES), ctx),
        _t("kin_W_wo", _shape2(FEATURES, FEATURES), ctx),
        _t("kin_W_mlp_gate", _shape2(MLPDIM, FEATURES), ctx),
        _t("kin_W_mlp_up", _shape2(MLPDIM, FEATURES), ctx),
        _t("kin_W_mlp_down", _shape2(FEATURES, MLPDIM), ctx),
        _t("kin_qnorm", _shape1(HEADDIM), ctx),
        _t("kin_knorm", _shape1(HEADDIM), ctx),
        _t("kin_prenorm", _shape1(FEATURES), ctx),
        _t("kin_postnorm", _shape1(FEATURES), ctx),
        _t("kin_mod_lin", _shape1(6 * FEATURES), ctx),
    )

    var lora = Krea2BlockLora(
        _adapter("wq", FEATURES, HEADS * HEADDIM, ctx),
        _adapter("wk", FEATURES, KVHEADS * HEADDIM, ctx),
        _adapter("wv", FEATURES, KVHEADS * HEADDIM, ctx),
        _adapter("gate", FEATURES, FEATURES, ctx),
        _adapter("wo", FEATURES, FEATURES, ctx),
        _adapter("mlp_gate", FEATURES, MLPDIM, ctx),
        _adapter("mlp_up", FEATURES, MLPDIM, ctx),
        _adapter("mlp_down", MLPDIM, FEATURES, ctx),
    )

    # rope tables: per-token [L,HALF] tiled to q (HEADS) and k (KVHEADS).
    var cos_q = _tile("kin_cos", HEADS, ctx)
    var sin_q = _tile("kin_sin", HEADS, ctx)
    var cos_k = _tile("kin_cos", KVHEADS, ctx)
    var sin_k = _tile("kin_sin", KVHEADS, ctx)
    # cos/sin (untiled) passed for signature symmetry; not used by the block.
    var cos0 = _t("kin_cos", _shape2(L, HALF), ctx)
    var sin0 = _t("kin_sin", _shape2(L, HALF), ctx)

    # ── forward ──────────────────────────────────────────────────────────────
    var fwd = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
        x.copy(), vec, w, lora, cos0[], sin0[], cos_q, sin_q, cos_k, sin_k, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    var out_h = fwd.out[].to_host(ctx)
    _check(harness, "out", out_h, _in("kref_out"), allok)

    # ── backward ─────────────────────────────────────────────────────────────
    var d_out_h = _in("kin_d_out")
    var d_out = Tensor.from_host(d_out_h, _shape3(1, L, FEATURES), STDtype.F32, ctx)
    var grads = krea2_single_stream_block_lora_backward[L, HEADS, KVHEADS, HEADDIM](
        d_out, vec, w, lora, fwd.saved, cos_q, sin_q, cos_k, sin_k, EPS, ctx,
    )

    print("")
    print("---- base input grad vs torch (no-regression) ----")
    var d_x_h = grads.d_x[].to_host(ctx)
    _check(harness, "d_x", d_x_h, _in("kref_d_x"), allok)

    print("")
    print("---- LoRA grads d_A / d_B vs torch (all 8 block Linears) ----")
    _check_lora(harness, "wq", grads.wq, allok)
    _check_lora(harness, "wk", grads.wk, allok)
    _check_lora(harness, "wv", grads.wv, allok)
    _check_lora(harness, "gate", grads.gate_w, allok)
    _check_lora(harness, "wo", grads.wo, allok)
    _check_lora(harness, "mlp_gate", grads.mlp_gate_w, allok)
    _check_lora(harness, "mlp_up", grads.mlp_up_w, allok)
    _check_lora(harness, "mlp_down", grads.mlp_down_w, allok)

    print("")
    if allok:
        print("VERDICT: PASS — krea2 SingleStreamBlock LoRA fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^
