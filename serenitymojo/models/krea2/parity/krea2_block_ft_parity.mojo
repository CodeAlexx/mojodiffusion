# serenitymojo/models/krea2/parity/krea2_block_ft_parity.mojo
#
# PARITY GATE for the Krea-2 SingleStreamBlock FULL-FINETUNE backward
# (krea2_single_stream_block_ft_backward_dev): the 8 base weight grads d_W +
# the input grad d_x vs torch autograd, cos >= 0.999, PLUS the v2 FULL-SURFACE
# arms (FULL_SURFACE_PLAN_2026-07-08 Phase B row krea2) at cos >= 0.9999:
# qnorm/knorm/prenorm/postnorm rms-scale d_g + the mod.lin [6F] 1D-parameter
# grad (mmdit.py DoubleSharedModulation: out = vec + lin — NOT a Linear).
# The forward is the base block with NO adapters (full-FT has no LoRA); the
# oracle runs in FT mode (plain linear) and dumps kref_ft_* refs.
#
# Run (oracle FIRST, as a SEPARATE command):
#   cd /home/alex/mojodiffusion
#   KREA2_FT_ORACLE=1 /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/krea2/parity/krea2_block_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       serenitymojo/models/krea2/parity/krea2_block_ft_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora,
    krea2_single_stream_block_lora,
    krea2_single_stream_block_ft_backward_dev,
    KREA2_FT_SLOT_QNORM, KREA2_FT_SLOT_KNORM,
    KREA2_FT_SLOT_PRENORM, KREA2_FT_SLOT_POSTNORM,
    KREA2_FT_SLOT_MOD_LIN, KREA2_FT_SLOTS_V2,
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


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the FT oracle first): ") + path)
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


def _none_adapter() -> Optional[LoraAdapterDevice]:
    return Optional[LoraAdapterDevice](None)


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
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


def main() raises:
    var ctx = DeviceContext()
    print("==== krea2_block_ft_parity (full-FT dW backward vs torch) ====")
    print("HEADS=", HEADS, " KVHEADS=", KVHEADS, " HEADDIM=", HEADDIM,
          " L=", L, " MLPDIM=", MLPDIM)

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

    # NO adapters: the full-FT forward is the base block.
    var lora = Krea2BlockLora(
        _none_adapter(), _none_adapter(), _none_adapter(), _none_adapter(),
        _none_adapter(), _none_adapter(), _none_adapter(), _none_adapter(),
    )

    var cos_q = _tile("kin_cos", HEADS, ctx)
    var sin_q = _tile("kin_sin", HEADS, ctx)
    var cos_k = _tile("kin_cos", KVHEADS, ctx)
    var sin_k = _tile("kin_sin", KVHEADS, ctx)
    var cos0 = _t("kin_cos", _shape2(L, HALF), ctx)
    var sin0 = _t("kin_sin", _shape2(L, HALF), ctx)

    # ── forward (base-only; saves the tape the FT backward consumes) ────────
    var fwd = krea2_single_stream_block_lora[L, HEADS, KVHEADS, HEADDIM](
        x.copy(), vec, w, lora, cos0[], sin0[], cos_q, sin_q, cos_k, sin_k, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch (FT mode) ----")
    var out_h = fwd.out[].to_host(ctx)
    _check(harness, "out", out_h, _in("kref_ft_out"), allok)

    # ── FT backward: d_x + the 8 dW ─────────────────────────────────────────
    var d_out_h = _in("kin_d_out")
    var d_out = Tensor.from_host(d_out_h, _shape3(1, L, FEATURES), STDtype.F32, ctx)
    var grads = krea2_single_stream_block_ft_backward_dev[L, HEADS, KVHEADS, HEADDIM](
        d_out, vec, w, fwd.saved, cos_q, sin_q, cos_k, sin_k, EPS, ctx,
    )

    print("")
    print("---- input grad d_x vs torch ----")
    var d_x_h = grads.d_x[].to_host(ctx)
    _check(harness, "d_x", d_x_h, _in("kref_ft_d_x"), allok)

    print("")
    print("---- base weight grads d_W vs torch (all 8 block Linears) ----")
    var names = List[String]()
    names.append(String("wq")); names.append(String("wk"))
    names.append(String("wv")); names.append(String("gate"))
    names.append(String("wo")); names.append(String("mlp_gate"))
    names.append(String("mlp_up")); names.append(String("mlp_down"))
    for i in range(8):
        var dw_h = grads.dw[i][].to_host(ctx)
        _check(harness, names[i] + " d_W", dw_h, _in("kref_ft_" + names[i] + "_dW"), allok)

    # ── v2 FULL-SURFACE arms (Phase B row krea2): 4 rms scales + mod.lin,
    # bar 0.9999 (the campaign bar for the new 1D arms) ──────────────────────
    if len(grads.dw) != KREA2_FT_SLOTS_V2:
        raise Error(
            String("krea2 FT backward returned ") + String(len(grads.dw))
            + String(" grads, expected ") + String(KREA2_FT_SLOTS_V2)
        )
    print("")
    print("---- v2 full-surface grads vs torch (4 rms scales + mod.lin, bar 0.9999) ----")
    var h2 = ParityHarness(0.9999)
    var v2_names = List[String]()
    v2_names.append(String("qnorm")); v2_names.append(String("knorm"))
    v2_names.append(String("prenorm")); v2_names.append(String("postnorm"))
    v2_names.append(String("mod_lin"))
    var v2_idx = List[Int]()
    v2_idx.append(KREA2_FT_SLOT_QNORM); v2_idx.append(KREA2_FT_SLOT_KNORM)
    v2_idx.append(KREA2_FT_SLOT_PRENORM); v2_idx.append(KREA2_FT_SLOT_POSTNORM)
    v2_idx.append(KREA2_FT_SLOT_MOD_LIN)
    for i in range(len(v2_names)):
        var dg_h = grads.dw[v2_idx[i]][].to_host(ctx)
        _check(h2, v2_names[i] + " d_P", dg_h,
               _in("kref_ft_" + v2_names[i] + "_dW"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — krea2 full-FT block backward matches torch")
        print("  (9 v1 captures cos>=0.999 + 5 v2 full-surface arms cos>=0.9999)")
    else:
        print("VERDICT: FAIL — at least one grad diverged (see FAIL lines above)")
