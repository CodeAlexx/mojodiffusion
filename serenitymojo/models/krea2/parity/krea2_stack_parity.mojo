# serenitymojo/models/krea2/parity/krea2_stack_parity.mojo
#
# PARITY GATE for the Krea-2-Raw SINGLE-STREAM STACK LoRA training composition
# (models/krea2/krea2_stack.mojo) at REDUCED depth (NBLOCKS=4). Loads the prepared
# single-stream inputs + per-block frozen weights + LoRA A/B + torch-autograd
# reference grads dumped by krea2_stack_oracle.py, runs:
#   (A) krea2_stack_lora_forward  → velocity (cos vs the reference velocity)
#   (B) krea2_stack_lora_backward → every per-block LoRA dA/dB (cos vs torch)
# at cos >= 0.999. Composition proof: re-confirms the reused block forward
# end-to-end (A) AND the new final-layer-bwd → single-stream-bwd×N chain (B).
#
# Sequence: TXTLEN=46, IMGLEN=210, L=256 (mult of 256 in the reference → NO pad →
# block SDPA == sdpa_nomask, the Phase-1 LoRA block's path). FAIL-LOUD: exit 1 on
# any cos < 0.999.
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/krea2/parity/krea2_stack_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       serenitymojo/models/krea2/parity/krea2_stack_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, Krea2LoraGrad,
)
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackWeights, Krea2StackLora, Krea2StackForward,
    Krea2StackLoraGrads, KREA2_SLOTS_PER_BLOCK,
    krea2_stack_lora_forward, krea2_stack_lora_backward,
)
from serenitymojo.models.dit.krea2_dit import build_krea2_rope

comptime TArc = ArcPointer[Tensor]
comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_stack_oracle.safetensors"

# dims MUST match krea2_stack_oracle.py
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM       # 6144
comptime HALF = HEADDIM // 2             # 64
comptime MLPDIM = 16384                  # SwiGLU hidden (mmdit.py:186-187, not features*mult)
comptime OUT_CH = 64                     # channels*patch^2
comptime NBLOCKS = 4
comptime L = 256
comptime TXTLEN = 46
comptime IMGLEN = 210
comptime RANK = 8
comptime EPS = Float32(1e-5)
comptime LSCALE = Float32(16.0) / Float32(8.0)   # alpha/rank = 2.0

# 8 LoRA slot names (order MUST match Krea2BlockLora fields / the backward scatter).
def _slot_name(s: Int) -> String:
    if s == 0:
        return String("wq")
    if s == 1:
        return String("wk")
    if s == 2:
        return String("wv")
    if s == 3:
        return String("gate")
    if s == 4:
        return String("wo")
    if s == 5:
        return String("mlp_gate")
    if s == 6:
        return String("mlp_up")
    return String("mlp_down")


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b)
    return s^


# load a dumped (f32) tensor by key into a device Tensor.
def _dev(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_f32(st.tensor_view(key), ctx)


def _arc(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> TArc:
    return TArc(_dev(st, key, ctx))


# read a dumped tensor as a host List[Float32] (reference grads).
def _host(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> List[Float32]:
    return _dev(st, key, ctx).to_host(ctx)


# LoRA in/out features per slot (mirrors the oracle SLOTS).
def _slot_io(slot: String) -> Tuple[Int, Int]:
    if slot == "wq":
        return (FEATURES, HEADS * HEADDIM)
    if slot == "wk":
        return (FEATURES, KVHEADS * HEADDIM)
    if slot == "wv":
        return (FEATURES, KVHEADS * HEADDIM)
    if slot == "gate":
        return (FEATURES, FEATURES)
    if slot == "wo":
        return (FEATURES, FEATURES)
    if slot == "mlp_gate":
        return (FEATURES, MLPDIM)
    if slot == "mlp_up":
        return (FEATURES, MLPDIM)
    return (MLPDIM, FEATURES)   # mlp_down


def _adapter(
    st: ShardedSafeTensors, bi: Int, slot: String, ctx: DeviceContext
) raises -> Optional[LoraAdapterDevice]:
    var io = _slot_io(slot)
    var in_f = io[0]
    var out_f = io[1]
    var a = _arc(st, "blk" + String(bi) + "." + slot + ".A", ctx)  # [rank, in]
    var b = _arc(st, "blk" + String(bi) + "." + slot + ".B", ctx)  # [out, rank]
    return Optional[LoraAdapterDevice](
        LoraAdapterDevice(a^, b^, RANK, in_f, out_f, LSCALE)
    )


def _block_weights(
    st: ShardedSafeTensors, bi: Int, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    var p = "blk" + String(bi) + "."
    return Krea2BlockWeights(
        _arc(st, p + "wq.W", ctx),
        _arc(st, p + "wk.W", ctx),
        _arc(st, p + "wv.W", ctx),
        _arc(st, p + "gate.W", ctx),
        _arc(st, p + "wo.W", ctx),
        _arc(st, p + "mlp_gate.W", ctx),
        _arc(st, p + "mlp_up.W", ctx),
        _arc(st, p + "mlp_down.W", ctx),
        _arc(st, p + "qnorm", ctx),
        _arc(st, p + "knorm", ctx),
        _arc(st, p + "prenorm", ctx),
        _arc(st, p + "postnorm", ctx),
        _arc(st, p + "mod_lin", ctx),
    )


def _block_lora(
    st: ShardedSafeTensors, bi: Int, ctx: DeviceContext
) raises -> Krea2BlockLora:
    return Krea2BlockLora(
        _adapter(st, bi, "wq", ctx),
        _adapter(st, bi, "wk", ctx),
        _adapter(st, bi, "wv", ctx),
        _adapter(st, bi, "gate", ctx),
        _adapter(st, bi, "wo", ctx),
        _adapter(st, bi, "mlp_gate", ctx),
        _adapter(st, bi, "mlp_up", ctx),
        _adapter(st, bi, "mlp_down", ctx),
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
    var st = ShardedSafeTensors.open(ORACLE)
    print("==== krea2_stack_parity (Krea-2 single-stream stack + LoRA vs torch) ====")
    print("NBLOCKS=", NBLOCKS, " L=", L, " TXTLEN=", TXTLEN, " IMGLEN=", IMGLEN,
          " FEATURES=", FEATURES, " MLPDIM=", MLPDIM, " RANK=", RANK)

    # ── stack weights + LoRA ─────────────────────────────────────────────────
    var blocks_w = List[Krea2BlockWeights]()
    var blocks_l = List[Krea2BlockLora]()
    for bi in range(NBLOCKS):
        blocks_w.append(_block_weights(st, bi, ctx))
        blocks_l.append(_block_lora(st, bi, ctx))
    var w = Krea2StackWeights(
        blocks_w^,
        _arc(st, "last.norm", ctx),
        _arc(st, "last.mod_lin", ctx),
        _arc(st, "last.lin_w", ctx),
        _arc(st, "last.lin_b", ctx),
    )
    var lora = Krea2StackLora(blocks_l^)

    # ── prepared single-stream inputs ────────────────────────────────────────
    var combined = _arc(st, "combined", ctx)               # [1,256,6144]
    var blk_vec = _dev(st, "tvec", ctx)                    # [1,1,6*6144] -> reshape [1,6F]
    var blk_vec2 = Tensor.from_host(
        blk_vec.to_host(ctx), _shape2(1, 6 * FEATURES), STDtype.F32, ctx,
    )
    var tmlp_out = _dev(st, "tmlp_out", ctx)               # [1,1,6144]

    # ── rope table from pos (build_krea2_rope; 3-axis [32,48,48], theta 1e3) ──
    var pos = _dev(st, "pos", ctx)                         # [1,256,3]
    var pos_flat = Tensor.from_host(pos.to_host(ctx), _shape1(L * 3), STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(32); axes.append(48); axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, Float32(1.0e3), ctx, STDtype.F32)
    # rope[0]=cos [256,64], rope[1]=sin [256,64]

    var harness = ParityHarness(0.999)
    var allok = True

    # ── (A) FORWARD → velocity ───────────────────────────────────────────────
    var fwd = krea2_stack_lora_forward[L, HEADS, KVHEADS, HEADDIM](
        combined, blk_vec2, tmlp_out, w, lora,
        rope[0], rope[1], EPS, TXTLEN, IMGLEN, ctx,
    )

    print("")
    print("---- (A) stack forward velocity vs torch ----")
    var vel_h = fwd.velocity[].to_host(ctx)
    _check(harness, "velocity", vel_h, _host(st, "velocity", ctx), allok)

    # ── (B) BACKWARD → every per-block LoRA dA/dB ────────────────────────────
    var d_velocity = _dev(st, "d_velocity", ctx)          # [1,IMGLEN,64]
    var grads = krea2_stack_lora_backward[L, HEADS, KVHEADS, HEADDIM](
        d_velocity, blk_vec2, tmlp_out, w, lora, fwd,
        rope[0], rope[1], EPS, ctx,
    )

    print("")
    print("---- (B) per-block LoRA dA/dB vs torch (all 8 slots x", NBLOCKS, "blocks) ----")
    for bi in range(NBLOCKS):
        var base = bi * KREA2_SLOTS_PER_BLOCK
        for s in range(KREA2_SLOTS_PER_BLOCK):
            var slot = _slot_name(s)
            var g = grads.grads[base + s].copy()
            if not g.d_a or not g.d_b:
                print("  blk", bi, slot, " adapter MISSING grads — FAIL")
                allok = False
                continue
            var pfx = "kref.blk" + String(bi) + "." + slot
            _check(harness, "blk" + String(bi) + "." + slot + " dA",
                   g.d_a.value(), _host(st, pfx + ".dA", ctx), allok)
            _check(harness, "blk" + String(bi) + "." + slot + " dB",
                   g.d_b.value(), _host(st, pfx + ".dB", ctx), allok)

    print("")
    if allok:
        print("VERDICT: PASS — krea2 single-stream stack LoRA fwd+bwd matches torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one output diverged (see FAIL lines above)")
        raise Error("krea2_stack_parity FAILED")
