# serenitymojo/models/anima/parity/lora_stack_parity.mojo
#
# LoRA COMPOSITION PARITY GATE for the ANIMA FULL STACK *WITH LoRA* on all 10 target
# projections (models/anima/anima_stack_lora.mojo). Loads the EXACT base inputs (from
# stack_oracle.py) + LoRA A/B inits + torch-autograd LoRA grads (from
# lora_stack_oracle.py), builds an AnimaLoraSet from those SAME A/B, runs
# anima_stack_lora_forward + anima_stack_lora_backward at L=3, and compares at
# cos >= 0.999:
#   * forward output (out) — LoRA-modified
#   * input-patch + shared grads (full-chain proof: d_patches, d_t_silu)
#   * ALL 10×L LoRA A/B grads (the deliverable: every adapter's d_A/d_B vs torch)
#   * nonfinite count == 0
#
# This proves the COMPOSED LoRA backward = grad of the COMPOSED LoRA forward (the
# Klein composition-bug lesson: per-block-correct does NOT imply composition-correct).
# Base-no-regression (LoRA absent == base forward bit-for-bit) is guaranteed
# structurally (anima_lora_apply returns base_y unchanged when an adapter is absent)
# and proven by the separate base gate stack_parity.mojo.
#
# Run (oracles FIRST, SEPARATE commands — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/anima/parity/stack_oracle.py
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/anima/parity/lora_stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/anima/parity/lora_stack_parity.mojo -o /tmp/anima_lora_parity
#   /tmp/anima_lora_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc, ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.anima.weights import AnimaBlockWeights, AnimaStackBase
from serenitymojo.models.anima.lora_block import ANIMA_SLOTS
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.anima.anima_stack import AnimaStackForward
from serenitymojo.models.anima.anima_stack_lora import (
    AnimaLoraSet, AnimaLoraGrads,
    anima_stack_lora_forward, anima_stack_lora_backward,
)


comptime TArc = ArcPointer[Tensor]
comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/anima/parity/"

comptime B = 1
comptime H = 16
comptime Dh = 128
comptime D = H * Dh        # 2048
comptime S_IMG = 6
comptime S_TXT = 8
comptime JOINT = 1024
comptime F = 32
comptime IN_PATCH = 68
comptime OUT_PATCH = 64
comptime L = 3
comptime EPS = Float32(1e-06)
comptime RANK = 8
comptime ALPHA = Float32(16.0)


# slot order MUST match lora_block.mojo SLOT_* and the oracle SLOTS list.
def _slot_name(s: Int) -> String:
    if s == 0:
        return String("sa_q")
    elif s == 1:
        return String("sa_k")
    elif s == 2:
        return String("sa_v")
    elif s == 3:
        return String("sa_out")
    elif s == 4:
        return String("ca_q")
    elif s == 5:
        return String("ca_k")
    elif s == 6:
        return String("ca_v")
    elif s == 7:
        return String("ca_out")
    elif s == 8:
        return String("mlp1")
    return String("mlp2")


# slot in/out shapes (must match the oracle SLOT_SHAPE + build_anima_lora_set)
def _slot_in(s: Int) -> Int:
    if s == 5 or s == 6:   # ca_k / ca_v: in=JOINT
        return JOINT
    if s == 9:             # mlp2: in=F
        return F
    return D


def _slot_out(s: Int) -> Int:
    if s == 8:             # mlp1: out=F
        return F
    return D               # everything else out=D


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


def _t(name: String, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(_in(name), shape^, STDtype.F32, ctx)


def _sh(*dims: Int) -> List[Int]:
    var o = List[Int]()
    for d in dims:
        o.append(d)
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _load_block(l: Int, ctx: DeviceContext) raises -> AnimaBlockWeights:
    var p = String("in_blk") + String(l) + String("_")
    return AnimaBlockWeights(
        _t(p + "sa_mod1", _sh(256, D), ctx), _t(p + "sa_mod2", _sh(3 * D, 256), ctx),
        _t(p + "ca_mod1", _sh(256, D), ctx), _t(p + "ca_mod2", _sh(3 * D, 256), ctx),
        _t(p + "mlp_mod1", _sh(256, D), ctx), _t(p + "mlp_mod2", _sh(3 * D, 256), ctx),
        _t(p + "sa_q", _sh(D, D), ctx), _t(p + "sa_k", _sh(D, D), ctx),
        _t(p + "sa_v", _sh(D, D), ctx), _t(p + "sa_out", _sh(D, D), ctx),
        _t(p + "sa_qn", _sh(Dh), ctx), _t(p + "sa_kn", _sh(Dh), ctx),
        _t(p + "ca_q", _sh(D, D), ctx), _t(p + "ca_k", _sh(D, JOINT), ctx),
        _t(p + "ca_v", _sh(D, JOINT), ctx), _t(p + "ca_out", _sh(D, D), ctx),
        _t(p + "ca_qn", _sh(Dh), ctx), _t(p + "ca_kn", _sh(Dh), ctx),
        _t(p + "mlp1", _sh(F, D), ctx), _t(p + "mlp2", _sh(D, F), ctx),
    )


def _load_base(ctx: DeviceContext) raises -> AnimaStackBase:
    return AnimaStackBase(
        TArc(_t("in_x_embed", _sh(D, IN_PATCH), ctx)),
        TArc(_t("in_x_embed", _sh(D, IN_PATCH), ctx)),   # te_lin1 (unused by gate)
        TArc(_t("in_x_embed", _sh(D, IN_PATCH), ctx)),   # te_lin2 (unused by gate)
        TArc(_t("in_fl_lin", _sh(OUT_PATCH, D), ctx)),   # t_norm  (unused by gate)
        TArc(_t("in_fl_mod1", _sh(256, D), ctx)),
        TArc(_t("in_fl_mod2", _sh(2 * D, 256), ctx)),
        TArc(_t("in_fl_lin", _sh(OUT_PATCH, D), ctx)),
    )


def _expand_rope(name: String) raises -> List[Float32]:
    var half = Dh // 2
    var per_pos = _in(name)
    var out = List[Float32]()
    for _b in range(B):
        for s in range(S_IMG):
            for _h in range(H):
                for i in range(half):
                    out.append(per_pos[s * half + i])
    return out^


# Build an AnimaLoraSet from the oracle's lin_*.bin A/B (so inits are identical).
def _load_lora_set(ctx: DeviceContext) raises -> AnimaLoraSet:
    var scale = ALPHA / Float32(RANK)
    var ad = List[LoraAdapter]()
    for l in range(L):
        for s in range(ANIMA_SLOTS):
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
    return AnimaLoraSet(ad^, L, RANK)


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
    print("==== anima LoRA stack_parity (ANIMA FULL STACK + 10-slot LoRA vs torch) ====")
    print("B=", B, " H=", H, " Dh=", Dh, " D=", D, " S_img=", S_IMG,
          " S_txt=", S_TXT, " F=", F, " L=", L, " RANK=", RANK, " ALPHA=", ALPHA)

    var patches = _in("in_patches")
    var t_cond = _in("in_t_cond")
    var base_adaln = _in("in_base_adaln")
    var context = _in("in_context")
    var base = _load_base(ctx)

    var blocks = List[AnimaBlockWeights]()
    for l in range(L):
        blocks.append(_load_block(l, ctx))

    var lora = _load_lora_set(ctx)

    var half = Dh // 2
    var cos = Tensor.from_host(_expand_rope("in_cos"), [B * S_IMG * H, half], STDtype.F32, ctx)
    var sin = Tensor.from_host(_expand_rope("in_sin"), [B * S_IMG * H, half], STDtype.F32, ctx)

    # ── forward ──
    var fwd = anima_stack_lora_forward[H, Dh, S_IMG, S_TXT](
        patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, lora, cos, sin,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )

    var harness = ParityHarness()
    var allok = True

    print("")
    print("---- forward output vs torch (LoRA-modified) ----")
    _check(harness, "out", fwd.out, _in("lref_out"), allok)

    # ── backward ──
    var d_out = _in("in_d_out")
    var g = anima_stack_lora_backward[H, Dh, S_IMG, S_TXT](
        d_out, patches.copy(), t_cond.copy(), base_adaln.copy(), context.copy(),
        base, blocks, lora, cos, sin, fwd,
        B, D, JOINT, F, IN_PATCH, OUT_PATCH, EPS, ctx,
    )

    print("")
    print("---- input + shared grads vs torch (full-chain proof through LoRA d_x) ----")
    _check(harness, "d_patches", g.d_patches, _in("lref_d_patches"), allok)
    _check(harness, "d_t_silu ", g.d_t_silu, _in("lref_d_t_silu"), allok)

    print("")
    print("---- ALL 10-slot LoRA A/B grads, every block vs torch (the deliverable) ----")
    for l in range(L):
        for s in range(ANIMA_SLOTS):
            var flat = l * ANIMA_SLOTS + s
            var nm = String("l") + String(l) + String("_") + _slot_name(s)
            var pre = String("lref_") + nm
            _check(harness, nm + String("_dA"), g.d_a[flat], _in(pre + String("_dA")), allok)
            _check(harness, nm + String("_dB"), g.d_b[flat], _in(pre + String("_dB")), allok)

    print("")
    print("nonfinite_lora_grads =", g.nonfinite_lora_grads)
    if allok and g.nonfinite_lora_grads == 0:
        print("VERDICT: PASS — ANIMA LoRA composition fwd+bwd (all A/B grads) matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
