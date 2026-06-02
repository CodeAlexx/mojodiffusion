# serenitymojo/models/klein/parity/klein_stack_lora_parity.mojo
#
# PARITY GATE for the Klein FULL DiT STACK *WITH LoRA* (small depth: 1 double +
# 1 single) — models/klein/klein_stack_lora.mojo. Loads the EXACT inputs + torch
# reference grads dumped by klein_stack_lora_oracle.py, runs
# klein_stack_lora_forward + klein_stack_lora_backward, and compares the output +
# the load-bearing input-token grads + the shared modvec grads + a sample of
# LoRA d_A/d_B from EACH block kind (every double adapter + both single adapters),
# all at cos >= 0.999.
#
# The per-block LoRA is already proven; this gate proves the STACK threads each
# block's adapters and COLLECTS each adapter's d_A/d_B into the flat KleinLoraGrads
# in the correct slot order.
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/klein/parity/klein_stack_lora_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/klein/parity/klein_stack_lora_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.models.klein.double_block import StreamWeights, DoubleBlockWeights, ModVecs
from serenitymojo.models.klein.single_block import SingleBlockWeights, SingleModVecs
from serenitymojo.models.klein.lora_block import LoraAdapter
from serenitymojo.models.klein.klein_stack import KleinStackBase
from serenitymojo.models.klein.klein_stack_lora import (
    KleinLoraSet, klein_stack_lora_forward, klein_stack_lora_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/klein/parity/"

# dims MUST match klein_stack_lora_oracle.py
comptime H = 4
comptime Dh = 8
comptime D = H * Dh            # 32
comptime N_IMG = 4
comptime N_TXT = 2
comptime S = N_TXT + N_IMG
comptime F = 24
comptime IN_CH = 10
comptime TXT_CH = 14
comptime OUT_CH = 6
comptime NUM_DOUBLE = 1
comptime NUM_SINGLE = 1
comptime EPS = Float32(1e-06)
comptime RANK = 4
comptime ALPHA = Float32(8.0)
comptime LSCALE = ALPHA / Float32(RANK)   # 2.0


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


def _load_stream(prefix: String, ctx: DeviceContext) raises -> StreamWeights:
    return StreamWeights(
        _in("klin_" + prefix + "_wqkv"), _in("klin_" + prefix + "_wproj"),
        _in("klin_" + prefix + "_wgu"), _in("klin_" + prefix + "_wd"),
        _in("klin_" + prefix + "_q_norm"), _in("klin_" + prefix + "_k_norm"),
        D, F, Dh, ctx,
    )


def _load_single(prefix: String, ctx: DeviceContext) raises -> SingleBlockWeights:
    return SingleBlockWeights(
        _in("klin_" + prefix + "_w1"), _in("klin_" + prefix + "_w2"),
        _in("klin_" + prefix + "_q_norm"), _in("klin_" + prefix + "_k_norm"),
        D, F, Dh, ctx,
    )


def _load_mod(prefix: String) raises -> ModVecs:
    return ModVecs(
        _in("klin_" + prefix + "_shift1"), _in("klin_" + prefix + "_scale1"),
        _in("klin_" + prefix + "_gate1"),
        _in("klin_" + prefix + "_shift2"), _in("klin_" + prefix + "_scale2"),
        _in("klin_" + prefix + "_gate2"),
    )


def _load_single_mod() raises -> SingleModVecs:
    return SingleModVecs(
        _in("klin_sm_shift"), _in("klin_sm_scale"), _in("klin_sm_gate"),
    )


# Build a LoraAdapter from loaded A/B with the canonical LSCALE + zeroed moments.
def _adapter(prefix: String, in_f: Int, out_f: Int) raises -> LoraAdapter:
    return LoraAdapter(
        _in("klin_" + prefix + "_A"), _in("klin_" + prefix + "_B"),
        RANK, in_f, out_f, LSCALE,
        _zeros(RANK * in_f), _zeros(RANK * in_f),
        _zeros(out_f * RANK), _zeros(out_f * RANK),
    )


# Build the flat KleinLoraSet from the loaded bins, in the SAME flat slot order
# klein_stack_lora.build_klein_lora_set uses (doubles first, slots
# img_qkv,img_proj,txt_qkv,txt_proj; singles slots qkv,out).
def _build_set() raises -> KleinLoraSet:
    var dbl = List[LoraAdapter]()
    for bi in range(NUM_DOUBLE):
        var p = String("d") + String(bi)
        dbl.append(_adapter(p + "_ilo_qkv", D, 3 * D))   # slot 0 img_qkv
        dbl.append(_adapter(p + "_ilo_proj", D, D))      # slot 1 img_proj
        dbl.append(_adapter(p + "_tlo_qkv", D, 3 * D))   # slot 2 txt_qkv
        dbl.append(_adapter(p + "_tlo_proj", D, D))      # slot 3 txt_proj
    var sgl = List[LoraAdapter]()
    for bi in range(NUM_SINGLE):
        var p = String("s") + String(bi)
        sgl.append(_adapter(p + "_qkv", D, 3 * D))       # slot 0 qkv
        sgl.append(_adapter(p + "_out", D, D))           # slot 1 out
    return KleinLoraSet(dbl^, sgl^, NUM_DOUBLE, NUM_SINGLE, RANK)


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
    print("==== klein_stack_lora_parity (Klein FULL stack + LoRA vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " N_IMG=", N_IMG, " N_TXT=", N_TXT,
          " F=", F, " num_double=", NUM_DOUBLE, " num_single=", NUM_SINGLE,
          " RANK=", RANK)

    var base = KleinStackBase(
        _in("klin_img_in"), _in("klin_txt_in"), _in("klin_final_lin"),
        _in("klin_final_shift"), _in("klin_final_scale"),
        D, IN_CH, TXT_CH, OUT_CH, ctx,
    )

    var dbw = List[DoubleBlockWeights]()
    for bi in range(NUM_DOUBLE):
        var p = String("d") + String(bi)
        dbw.append(DoubleBlockWeights(_load_stream(p + "_iw", ctx), _load_stream(p + "_tw", ctx)))
    var sbw = List[SingleBlockWeights]()
    for bi in range(NUM_SINGLE):
        sbw.append(_load_single(String("s") + String(bi), ctx))

    var im = _load_mod("im")
    var tm = _load_mod("tm")
    var sm = _load_single_mod()

    var lora = _build_set()

    var img_tokens = _in("klin_img_tokens")
    var txt_tokens = _in("klin_txt_tokens")
    var cos = _in("klin_cos")
    var sin = _in("klin_sin")

    var fwd = klein_stack_lora_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, lora, im, tm, sm, cos.copy(), sin.copy(),
        D, F, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch ----")
    _check(harness, "out", fwd.out, _in("klref_out"), allok)

    var d_out = _in("klin_d_out")
    var g = klein_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens.copy(), txt_tokens.copy(), base,
        dbw, sbw, lora, im, tm, sm, cos.copy(), sin.copy(), fwd,
        D, F, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    print("")
    print("---- load-bearing input-token grads vs torch ----")
    _check(harness, "d_img_tokens", g.d_img_tokens, _in("klref_d_img_tokens"), allok)
    _check(harness, "d_txt_tokens", g.d_txt_tokens, _in("klref_d_txt_tokens"), allok)

    print("")
    print("---- shared modulation-vector grads vs torch (summed across blocks) ----")
    _check(harness, "d_img_mod   ", g.d_img_mod, _in("klref_d_img_mod"), allok)
    _check(harness, "d_txt_mod   ", g.d_txt_mod, _in("klref_d_txt_mod"), allok)
    _check(harness, "d_single_mod", g.d_single_mod, _in("klref_d_single_mod"), allok)

    print("")
    print("---- DOUBLE block 0 LoRA grads d_A / d_B vs torch (flat slots 0..3) ----")
    # slot order: 0 img_qkv, 1 img_proj, 2 txt_qkv, 3 txt_proj.
    _check(harness, "d0 img qkv  d_A", g.dbl_d_a[0], _in("klref_d0_img_qkv_dA"), allok)
    _check(harness, "d0 img qkv  d_B", g.dbl_d_b[0], _in("klref_d0_img_qkv_dB"), allok)
    _check(harness, "d0 img proj d_A", g.dbl_d_a[1], _in("klref_d0_img_proj_dA"), allok)
    _check(harness, "d0 img proj d_B", g.dbl_d_b[1], _in("klref_d0_img_proj_dB"), allok)
    _check(harness, "d0 txt qkv  d_A", g.dbl_d_a[2], _in("klref_d0_txt_qkv_dA"), allok)
    _check(harness, "d0 txt qkv  d_B", g.dbl_d_b[2], _in("klref_d0_txt_qkv_dB"), allok)
    _check(harness, "d0 txt proj d_A", g.dbl_d_a[3], _in("klref_d0_txt_proj_dA"), allok)
    _check(harness, "d0 txt proj d_B", g.dbl_d_b[3], _in("klref_d0_txt_proj_dB"), allok)

    print("")
    print("---- SINGLE block 0 LoRA grads d_A / d_B vs torch (flat slots 0..1) ----")
    _check(harness, "s0 qkv d_A", g.sgl_d_a[0], _in("klref_s0_qkv_dA"), allok)
    _check(harness, "s0 qkv d_B", g.sgl_d_b[0], _in("klref_s0_qkv_dB"), allok)
    _check(harness, "s0 out d_A", g.sgl_d_a[1], _in("klref_s0_out_dA"), allok)
    _check(harness, "s0 out d_B", g.sgl_d_b[1], _in("klref_s0_out_dB"), allok)

    print("")
    if allok:
        print("VERDICT: PASS — Klein FULL stack + LoRA fwd+bwd composes & collects (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
