# serenitymojo/models/zimage/parity/lora_stack_parity.mojo
#
# LoRA COMPOSITION PARITY GATE for the Z-Image (NextDiT) FULL STACK *WITH LoRA* on
# all 7 projections per block (models/zimage/zimage_stack_lora.mojo). Loads the
# EXACT base inputs (from stack_oracle.py) + LoRA A/B inits + torch-autograd LoRA
# grads (from lora_stack_oracle.py), builds a ZImageLoraSet from those SAME A/B,
# runs zimage_stack_lora_forward + zimage_stack_lora_backward at NR=1/CR=1/MAIN=2,
# REAL H=30/Dh=128/D=3840, and compares at cos >= 0.999:
#   * forward output (out) — LoRA-modified
#   * input-token grads (full-chain proof through the summed LoRA d_x, both streams)
#   * final-layer scale grad + per-block RAW mod-vec grads (composition)
#   * ALL 7 × (NR+CR+MAIN) LoRA A/B grads (the deliverable: every adapter's d_A/d_B)
# Proves the COMPOSED LoRA backward = grad of the COMPOSED LoRA forward.
#
# Run (oracles FIRST, SEPARATE commands — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/zimage/parity/stack_oracle.py
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/zimage/parity/lora_stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/zimage/parity/lora_stack_parity.mojo -o /tmp/zimage_lora_parity
#   /tmp/zimage_lora_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import ZImageModVecs, ZImageBlockGrads
from serenitymojo.models.zimage.lora_block import (
    ZIMAGE_SLOTS, SLOT_W1, SLOT_W3, SLOT_W2,
)
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.zimage.zimage_stack import ZImageStackForward
from serenitymojo.models.zimage.zimage_stack_lora import (
    ZImageLoraSet, ZImageLoraGrads,
    zimage_stack_lora_forward, zimage_stack_lora_backward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/zimage/parity/"

# dims MUST match stack_oracle.py / lora_stack_oracle.py
comptime H = 30
comptime Dh = 128
comptime D = H * Dh        # 3840
comptime IMG_H = 2
comptime IMG_W = 3
comptime N_IMG = IMG_H * IMG_W   # 6
comptime N_TXT = 4
comptime S = N_IMG + N_TXT       # 10
comptime F = 96
comptime OUT_CH = 16
comptime HALF = Dh // 2          # 64
comptime EPS = Float32(1e-05)
comptime FINAL_EPS = Float32(1e-06)
comptime NUM_NR = 1
comptime NUM_CR = 1
comptime NUM_MAIN = 2
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
    elif s == SLOT_W1:
        return String("w1")
    elif s == SLOT_W3:
        return String("w3")
    return String("w2")


def _slot_in(s: Int) -> Int:
    if s == SLOT_W2:
        return F
    return D


def _slot_out(s: Int) -> Int:
    if s == SLOT_W1 or s == SLOT_W3:
        return F
    return D


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


def _load_block(prefix: String, ctx: DeviceContext) raises -> ZImageBlockWeights:
    return ZImageBlockWeights(
        _t1(_in(prefix + "_n1"), D, ctx),
        _t2(_in(prefix + "_wq"), D, D, ctx),
        _t2(_in(prefix + "_wk"), D, D, ctx),
        _t2(_in(prefix + "_wv"), D, D, ctx),
        _t2(_in(prefix + "_wo"), D, D, ctx),
        _t1(_in(prefix + "_q_norm"), Dh, ctx),
        _t1(_in(prefix + "_k_norm"), Dh, ctx),
        _t1(_in(prefix + "_n2"), D, ctx),
        _t1(_in(prefix + "_fn1"), D, ctx),
        _t2(_in(prefix + "_w1"), F, D, ctx),
        _t2(_in(prefix + "_w3"), F, D, ctx),
        _t2(_in(prefix + "_w2"), D, F, ctx),
        _t1(_in(prefix + "_fn2"), D, ctx),
    )


def _load_mod(prefix: String) raises -> ZImageModVecs:
    return ZImageModVecs(
        _in(prefix + "_scale_msa"), _in(prefix + "_gate_msa"),
        _in(prefix + "_scale_mlp"), _in(prefix + "_gate_mlp"),
    )


# append one segment-tagged block's 7 adapters from lin_*.bin into `ad`.
def _add_seg(mut ad: List[LoraAdapter], tag: String, ctx: DeviceContext) raises:
    var scale = ALPHA / Float32(RANK)
    for s in range(ZIMAGE_SLOTS):
        var in_f = _slot_in(s)
        var out_f = _slot_out(s)
        var pre = String("lin_") + tag + String("_") + _slot_name(s)
        var a = _in(pre + String("_A"))   # [rank, in]
        var b = _in(pre + String("_B"))   # [out, rank]
        ad.append(LoraAdapter(
            a^, b^, RANK, in_f, out_f, scale,
            _zeros(RANK * in_f), _zeros(RANK * in_f),
            _zeros(out_f * RANK), _zeros(out_f * RANK),
        ))


# Build a ZImageLoraSet from the oracle's lin_*.bin A/B (segment-tagged), so inits
# are identical on both sides. Flat order: nr | cr | main, each block 7 slots.
def _load_lora_set(ctx: DeviceContext) raises -> ZImageLoraSet:
    var ad = List[LoraAdapter]()
    for i in range(NUM_NR):
        _add_seg(ad, String("nr") + String(i), ctx)
    for i in range(NUM_CR):
        _add_seg(ad, String("cr") + String(i), ctx)
    for i in range(NUM_MAIN):
        _add_seg(ad, String("main") + String(i), ctx)
    return ZImageLoraSet(ad^, NUM_NR, NUM_CR, NUM_MAIN, RANK)


def _check(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def _pack4(g: ZImageBlockGrads) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(g.d_scale_msa)):
        o.append(g.d_scale_msa[i])
    for i in range(len(g.d_gate_msa)):
        o.append(g.d_gate_msa[i])
    for i in range(len(g.d_scale_mlp)):
        o.append(g.d_scale_mlp[i])
    for i in range(len(g.d_gate_mlp)):
        o.append(g.d_gate_mlp[i])
    return o^


def main() raises:
    var ctx = DeviceContext()
    print("==== zimage LoRA stack_parity (Z-Image FULL STACK + 7-slot LoRA vs torch) ====")
    print("H=", H, " Dh=", Dh, " D=", D, " S=", S, " F=", F,
          " NR=", NUM_NR, " CR=", NUM_CR, " MAIN=", NUM_MAIN,
          " RANK=", RANK, " ALPHA=", ALPHA)

    var x_seq = _in("sin_x_seq")
    var cap_seq = _in("sin_cap_seq")
    var f_scale = _in("sin_f_scale")
    var final_lin_w = Tensor.from_host(_in("sin_final_lin"), [OUT_CH, D], STDtype.F32, ctx)
    var final_lin_b = Tensor.from_host(_in("sin_final_lin_b"), [OUT_CH], STDtype.F32, ctx)

    var nr_blocks = List[ZImageBlockWeights]()
    var nr_mod = List[ZImageModVecs]()
    for i in range(NUM_NR):
        nr_blocks.append(_load_block(String("sin_nr") + String(i), ctx))
        nr_mod.append(_load_mod(String("sin_nr") + String(i)))
    var cr_blocks = List[ZImageBlockWeights]()
    for i in range(NUM_CR):
        cr_blocks.append(_load_block(String("sin_cr") + String(i), ctx))
    var main_blocks = List[ZImageBlockWeights]()
    var main_mod = List[ZImageModVecs]()
    for i in range(NUM_MAIN):
        main_blocks.append(_load_block(String("sin_main") + String(i), ctx))
        main_mod.append(_load_mod(String("sin_main") + String(i)))

    var lora = _load_lora_set(ctx)

    var x_cos = Tensor.from_host(_in("sin_x_cos"), [N_IMG * H, HALF], STDtype.F32, ctx)
    var x_sin = Tensor.from_host(_in("sin_x_sin"), [N_IMG * H, HALF], STDtype.F32, ctx)
    var cap_cos = Tensor.from_host(_in("sin_cap_cos"), [N_TXT * H, HALF], STDtype.F32, ctx)
    var cap_sin = Tensor.from_host(_in("sin_cap_sin"), [N_TXT * H, HALF], STDtype.F32, ctx)
    var uni_cos = Tensor.from_host(_in("sin_uni_cos"), [S * H, HALF], STDtype.F32, ctx)
    var uni_sin = Tensor.from_host(_in("sin_uni_sin"), [S * H, HALF], STDtype.F32, ctx)

    # ── forward ──
    var fwd = zimage_stack_lora_forward[H, Dh, N_IMG, N_TXT, S](
        x_seq.copy(), cap_seq.copy(),
        nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora,
        f_scale.copy(), final_lin_w, final_lin_b,
        x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch (LoRA-modified) ----")
    _check(harness, "out", fwd.out, _in("lref_out"), allok)

    # ── backward ──
    var d_out = _in("sin_d_out")
    var g = zimage_stack_lora_backward[H, Dh, N_IMG, N_TXT, S](
        d_out, nr_blocks, nr_mod, cr_blocks, main_blocks, main_mod, lora,
        f_scale.copy(), final_lin_w,
        x_cos, x_sin, cap_cos, cap_sin, uni_cos, uni_sin, fwd,
        D, F, OUT_CH, EPS, FINAL_EPS, ctx,
    )

    print("")
    print("---- input-token grads vs torch (full-chain proof through LoRA d_x) ----")
    _check(harness, "d_x_seq  ", g.d_x_seq, _in("lref_d_x_seq"), allok)
    _check(harness, "d_cap_seq", g.d_cap_seq, _in("lref_d_cap_seq"), allok)

    print("")
    print("---- final-layer + per-block RAW mod-vec grads vs torch ----")
    _check(harness, "d_f_scale", g.d_f_scale, _in("lref_d_f_scale"), allok)
    for i in range(NUM_NR):
        _check(harness, String("nr") + String(i) + String("_mod"),
               g.nr_mod[i], _in(String("lref_nr") + String(i) + String("_mod")), allok)
    for i in range(NUM_MAIN):
        _check(harness, String("main") + String(i) + String("_mod"),
               g.main_mod[i], _in(String("lref_main") + String(i) + String("_mod")), allok)

    print("")
    print("---- ALL 7-slot LoRA A/B grads, every block vs torch (the deliverable) ----")

    # build the segment tag for each flat block index (nr | cr | main).
    var tags = List[String]()
    for i in range(NUM_NR):
        tags.append(String("nr") + String(i))
    for i in range(NUM_CR):
        tags.append(String("cr") + String(i))
    for i in range(NUM_MAIN):
        tags.append(String("main") + String(i))

    for bidx in range(len(tags)):
        var base = bidx * ZIMAGE_SLOTS
        for s in range(ZIMAGE_SLOTS):
            var nm = tags[bidx] + String("_") + _slot_name(s)
            var pre = String("lref_") + nm
            _check(harness, nm + String("_dA"), g.d_a[base + s], _in(pre + String("_dA")), allok)
            _check(harness, nm + String("_dB"), g.d_b[base + s], _in(pre + String("_dB")), allok)

    print("")
    print("nonfinite_lora_grads =", g.nonfinite_lora_grads)
    if allok and g.nonfinite_lora_grads == 0:
        print("VERDICT: PASS — Z-Image LoRA composition fwd+bwd (all A/B grads) matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
