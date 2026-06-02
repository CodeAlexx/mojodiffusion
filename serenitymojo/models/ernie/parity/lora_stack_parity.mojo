# serenitymojo/models/ernie/parity/lora_stack_parity.mojo
#
# LoRA COMPOSITION PARITY GATE for the ERNIE-Image FULL STACK *WITH LoRA* on all
# 7 projections (models/ernie/ernie_stack_lora.mojo). Loads the EXACT base inputs
# (from stack_oracle.py) + LoRA A/B inits + torch-autograd LoRA grads (from
# lora_stack_oracle.py), builds an ErnieLoraSet from those SAME A/B, runs
# ernie_stack_lora_forward + ernie_stack_lora_backward at L=3 / S=8 / reduced F,
# and compares at cos >= 0.999:
#   * forward output (out) — LoRA-modified
#   * input-token grads (full-chain proof, threads through the summed LoRA d_x)
#   * final-layer modulation grads + summed shared-AdaLN mod [6D] (composition)
#   * ALL 7×L LoRA A/B grads (the deliverable: every adapter's d_A/d_B vs torch)
# Plus a base-no-regression check: with adapters absent the LoRA forward == base
# forward bit-for-bit (separate run via the base stack gate; here we additionally
# assert the LoRA-set out matches the LoRA oracle, and the d_A/d_B path is live).
#
# Run (oracles FIRST, SEPARATE commands — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/ernie/parity/stack_oracle.py
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/ernie/parity/lora_stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/ernie/parity/lora_stack_parity.mojo -o /tmp/ernie_lora_parity
#   /tmp/ernie_lora_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.ernie.weights import ErnieBlockWeights, ErnieStackBase
from serenitymojo.models.ernie.block import ErnieModVecs
from serenitymojo.models.ernie.lora_block import ERNIE_SLOTS
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.ernie.ernie_stack import ErnieStackForward
from serenitymojo.models.ernie.ernie_stack_lora import (
    ErnieLoraSet, ErnieLoraGrads,
    ernie_stack_lora_forward, ernie_stack_lora_backward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/ernie/parity/"

comptime H = 32
comptime Dh = 128
comptime D = H * Dh        # 4096
comptime N_IMG = 6
comptime N_TXT = 2
comptime S = N_IMG + N_TXT # 8
comptime F = 96
comptime IN_CH = 16
comptime TEXT_IN = 24
comptime OUT_CH = 16
comptime L = 3
comptime EPS = Float32(1e-06)
comptime RANK = 8
comptime ALPHA = Float32(16.0)

# slot order MUST match lora_block.mojo SLOT_* and the oracle SLOTS list.
def _slot_name(s: Int) -> String:
    if s == 0:
        return String("to_q")
    elif s == 1:
        return String("to_k")
    elif s == 2:
        return String("to_v")
    elif s == 3:
        return String("to_out")
    elif s == 4:
        return String("gate_proj")
    elif s == 5:
        return String("up_proj")
    return String("linear_fc2")


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the oracles first): ") + path)
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


def _t1(vals: List[Float32], n: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, [n], STDtype.F32, ctx))


def _t2(vals: List[Float32], a: Int, b: Int, ctx: DeviceContext) raises -> TArc:
    return TArc(Tensor.from_host(vals, [a, b], STDtype.F32, ctx))


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _load_block(l: Int, ctx: DeviceContext) raises -> ErnieBlockWeights:
    var pre = String("in_blk") + String(l) + String("_")
    return ErnieBlockWeights(
        _t1(_in(pre + String("sa_norm")), D, ctx),
        _t2(_in(pre + String("wq")), D, D, ctx),
        _t2(_in(pre + String("wk")), D, D, ctx),
        _t2(_in(pre + String("wv")), D, D, ctx),
        _t2(_in(pre + String("wo")), D, D, ctx),
        _t1(_in(pre + String("q_norm")), Dh, ctx),
        _t1(_in(pre + String("k_norm")), Dh, ctx),
        _t1(_in(pre + String("mlp_norm")), D, ctx),
        _t2(_in(pre + String("wgate")), F, D, ctx),
        _t2(_in(pre + String("wup")), F, D, ctx),
        _t2(_in(pre + String("wdown")), D, F, ctx),
    )


def _load_base(ctx: DeviceContext) raises -> ErnieStackBase:
    var dummy_d = _t1(_in("in_f_scale"), D, ctx)
    return ErnieStackBase(
        _t2(_in("in_patch_w"), D, IN_CH, ctx),
        _t1(_in("in_patch_b"), D, ctx),
        _t2(_in("in_text_proj"), D, TEXT_IN, ctx),
        dummy_d, dummy_d, dummy_d, dummy_d,
        dummy_d, dummy_d,
        dummy_d, dummy_d,
        _t2(_in("in_final_lin"), OUT_CH, D, ctx),
        _t1(_in("in_final_lin_b"), OUT_CH, ctx),
    )


def _load_mod() raises -> ErnieModVecs:
    return ErnieModVecs(
        _in("in_m_shift_msa"), _in("in_m_scale_msa"), _in("in_m_gate_msa"),
        _in("in_m_shift_mlp"), _in("in_m_scale_mlp"), _in("in_m_gate_mlp"),
    )


# slot in/out shapes (must match the oracle SLOT_SHAPE + build_ernie_lora_set)
def _slot_in(s: Int) -> Int:
    if s == 6:  # linear_fc2: in=F
        return F
    return D


def _slot_out(s: Int) -> Int:
    if s == 4 or s == 5:  # gate_proj/up_proj: out=F
        return F
    if s == 6:  # linear_fc2: out=D
        return D
    return D    # to_q/to_k/to_v/to_out: out=D


# Build an ErnieLoraSet from the oracle's lin_*.bin A/B (so inits are identical).
def _load_lora_set(ctx: DeviceContext) raises -> ErnieLoraSet:
    var scale = ALPHA / Float32(RANK)
    var ad = List[LoraAdapter]()
    for l in range(L):
        for s in range(ERNIE_SLOTS):
            var in_f = _slot_in(s)
            var out_f = _slot_out(s)
            var pre = String("lin_l") + String(l) + String("_") + _slot_name(s)
            var a = _in(pre + String("_A"))   # [rank, in]
            var b = _in(pre + String("_B"))   # [out, rank]
            ad.append(LoraAdapter(
                a^, b^, RANK, in_f, out_f, scale,
                _zeros(RANK * in_f), _zeros(RANK * in_f),
                _zeros(out_f * RANK), _zeros(out_f * RANK),
            ))
    return ErnieLoraSet(ad^, L, RANK)


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
    print("==== ernie LoRA stack_parity (ERNIE FULL STACK + 7-slot LoRA vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " F=", F, " L=", L,
          " RANK=", RANK, " ALPHA=", ALPHA)

    var img_tokens = _in("in_img_tokens")
    var txt_tokens = _in("in_txt_tokens")
    var base = _load_base(ctx)
    var mv = _load_mod()
    var f_scale = _in("in_f_scale")
    var f_shift = _in("in_f_shift")

    var blocks = List[ErnieBlockWeights]()
    for l in range(L):
        blocks.append(_load_block(l, ctx))

    var lora = _load_lora_set(ctx)

    var cos = Tensor.from_host(_in("in_cos"), [S * H, Dh], STDtype.F32, ctx)
    var sin = Tensor.from_host(_in("in_sin"), [S * H, Dh], STDtype.F32, ctx)

    var fwd = ernie_stack_lora_forward[H, Dh, N_IMG, N_TXT, S](
        img_tokens.copy(), txt_tokens.copy(), base, blocks, lora, mv,
        f_scale.copy(), f_shift.copy(), cos, sin,
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch (LoRA-modified) ----")
    _check(harness, "out", fwd.out, _in("lref_out"), allok)

    var d_out = _in("in_d_out")
    var g = ernie_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, img_tokens.copy(), txt_tokens.copy(), base, blocks, lora, mv,
        f_scale.copy(), f_shift.copy(), cos, sin, fwd,
        D, F, IN_CH, TEXT_IN, OUT_CH, EPS, ctx,
    )

    print("")
    print("---- input-token grads vs torch (full-chain proof through LoRA d_x) ----")
    _check(harness, "d_img_tokens", g.d_img_tokens, _in("lref_d_img_tokens"), allok)
    _check(harness, "d_txt_tokens", g.d_txt_tokens, _in("lref_d_txt_tokens"), allok)

    print("")
    print("---- final-layer + summed shared-AdaLN grads vs torch ----")
    _check(harness, "d_f_scale  ", g.d_f_scale, _in("lref_d_f_scale"), allok)
    _check(harness, "d_f_shift  ", g.d_f_shift, _in("lref_d_f_shift"), allok)
    _check(harness, "d_shared_mod", g.d_shared_mod, _in("lref_d_shared_mod"), allok)

    print("")
    print("---- ALL 7-slot LoRA A/B grads, every block vs torch (the deliverable) ----")
    for l in range(L):
        for s in range(ERNIE_SLOTS):
            var flat = l * ERNIE_SLOTS + s
            var nm = String("l") + String(l) + String("_") + _slot_name(s)
            var pre = String("lref_") + nm
            _check(harness, nm + String("_dA"), g.d_a[flat], _in(pre + String("_dA")), allok)
            _check(harness, nm + String("_dB"), g.d_b[flat], _in(pre + String("_dB")), allok)

    print("")
    print("nonfinite_lora_grads =", g.nonfinite_lora_grads)
    if allok:
        print("VERDICT: PASS — ERNIE LoRA composition fwd+bwd (all A/B grads) matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
