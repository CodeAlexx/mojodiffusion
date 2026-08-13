# serenitymojo/models/krea2/parity/krea2_stack_ref_parity.mojo
#
# PARITY GATE for the KREA2 img-EDIT reference-conditioning projection (PORT of
# klein's img_in_ref) wired into the single-stream STACK LoRA composition
# (models/krea2/krea2_stack.mojo krea2_stack_lora_forward/backward), at REDUCED
# depth (NBLOCKS=4). Three checks:
#
#   (A) NON-ZERO img_in_ref + NON-DEGENERATE ref: assert d_combined (input-token
#       grads into the block-stack input) AND d_img_in_ref (the new trained param's
#       grad) vs torch.autograd (krea2_stack_ref_oracle.py) at cos >= 0.999.
#   (B) FLAGS-OFF (C13): ref=None must be BYTE-IDENTICAL to ref-on-with-ZERO-weight
#       (the zero-init contract: linear(ref,0)==0). Assert velocity + d_combined +
#       every per-block LoRA dA/dB are max_abs 0 between the two. Path A (ref=None)
#       IS the pre-change krea2_stack fwd+bwd (same function, defaulted args).
#   (C) PARAM: Krea2ImgInRefParam AdamW step + safetensors save/resume round-trip
#       BYTE-EXACT (weight bf16 + moments f32).
#
# The ref branch is a PURE INPUT LINEAR (dimension-independent) — reduced depth at
# krea2's real head config is a faithful oracle for the composition.
#
# Run (oracle FIRST, as a SEPARATE command — never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/SerenityTrainer/venv/bin/python \
#       serenitymojo/models/krea2/parity/krea2_stack_ref_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       serenitymojo/models/krea2/parity/krea2_stack_ref_parity.mojo

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, Krea2LoraGrad,
)
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackWeights, Krea2StackLora,
    Krea2StackLoraGrads, KREA2_SLOTS_PER_BLOCK,
    krea2_stack_lora_forward, krea2_stack_lora_backward,
)
from serenitymojo.models.krea2.krea2_img_in_ref_param import (
    Krea2ImgInRefParam, make_krea2_img_in_ref_param, krea2_img_in_ref_adamw_step,
    save_krea2_img_in_ref, load_krea2_img_in_ref_resume,
)
from serenitymojo.models.dit.krea2_dit import build_krea2_rope

comptime TArc = ArcPointer[Tensor]
comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_stack_ref_oracle.safetensors"

# dims MUST match krea2_stack_ref_oracle.py
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM       # 6144
comptime MLPDIM = 16384
comptime OUT_CH = 64
comptime IN_CH = 64                        # channels*patch^2 == first in_features
comptime NBLOCKS = 4
comptime L = 256
comptime TXTLEN = 46
comptime IMGLEN = 210
comptime RANK = 8
comptime EPS = Float32(1e-5)
comptime LSCALE = Float32(16.0) / Float32(8.0)   # alpha/rank = 2.0


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


def _dev(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_f32(st.tensor_view(key), ctx)


def _arc(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> TArc:
    return TArc(_dev(st, key, ctx))


def _host(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> List[Float32]:
    return _dev(st, key, ctx).to_host(ctx)


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
    var a = _arc(st, "blk" + String(bi) + "." + slot + ".A", ctx)
    var b = _arc(st, "blk" + String(bi) + "." + slot + ".B", ctx)
    return Optional[LoraAdapterDevice](
        LoraAdapterDevice(a^, b^, RANK, io[0], io[1], LSCALE)
    )


def _block_weights(
    st: ShardedSafeTensors, bi: Int, ctx: DeviceContext
) raises -> Krea2BlockWeights:
    var p = "blk" + String(bi) + "."
    return Krea2BlockWeights(
        _arc(st, p + "wq.W", ctx), _arc(st, p + "wk.W", ctx),
        _arc(st, p + "wv.W", ctx), _arc(st, p + "gate.W", ctx),
        _arc(st, p + "wo.W", ctx), _arc(st, p + "mlp_gate.W", ctx),
        _arc(st, p + "mlp_up.W", ctx), _arc(st, p + "mlp_down.W", ctx),
        _arc(st, p + "qnorm", ctx), _arc(st, p + "knorm", ctx),
        _arc(st, p + "prenorm", ctx), _arc(st, p + "postnorm", ctx),
        _arc(st, p + "mod_lin", ctx),
    )


def _block_lora(
    st: ShardedSafeTensors, bi: Int, ctx: DeviceContext
) raises -> Krea2BlockLora:
    return Krea2BlockLora(
        _adapter(st, bi, "wq", ctx), _adapter(st, bi, "wk", ctx),
        _adapter(st, bi, "wv", ctx), _adapter(st, bi, "gate", ctx),
        _adapter(st, bi, "wo", ctx), _adapter(st, bi, "mlp_gate", ctx),
        _adapter(st, bi, "mlp_up", ctx), _adapter(st, bi, "mlp_down", ctx),
    )


# cos check (>= 0.999) via the parity harness.
def _check_cos(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


# exact byte identity: max abs diff MUST be 0.
def _check_zero(
    name: String, a: List[Float32], b: List[Float32], mut allok: Bool
) raises:
    if len(a) != len(b):
        print("  ", name, " LEN MISMATCH", len(a), "vs", len(b), " FAIL")
        allok = False
        return
    var m = Float32(0.0)
    for i in range(len(a)):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > m:
            m = d
    print("  max_abs(", name, ") =", m, "  n =", len(a), "  ",
          "PASS" if m == Float32(0.0) else "FAIL")
    if m != Float32(0.0):
        allok = False


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(ORACLE)
    print("==== krea2_stack_ref_parity (img_in_ref projection port vs torch) ====")
    print("NBLOCKS=", NBLOCKS, " L=", L, " TXTLEN=", TXTLEN, " IMGLEN=", IMGLEN,
          " FEATURES=", FEATURES, " IN_CH=", IN_CH, " RANK=", RANK)

    # ── stack weights + LoRA ──
    var blocks_w = List[Krea2BlockWeights]()
    var blocks_l = List[Krea2BlockLora]()
    for bi in range(NBLOCKS):
        blocks_w.append(_block_weights(st, bi, ctx))
        blocks_l.append(_block_lora(st, bi, ctx))
    var w = Krea2StackWeights(
        blocks_w^, _arc(st, "last.norm", ctx), _arc(st, "last.mod_lin", ctx),
        _arc(st, "last.lin_w", ctx), _arc(st, "last.lin_b", ctx),
    )
    var lora = Krea2StackLora(blocks_l^)

    # ── prepared single-stream inputs (base combined = cat(ctx, first(img))) ──
    var combined = _arc(st, "combined", ctx)               # [1,256,6144]
    var blk_vec = _dev(st, "tvec", ctx)
    var blk_vec2 = Tensor.from_host(
        blk_vec.to_host(ctx), [1, 6 * FEATURES], STDtype.F32, ctx,
    )
    var tmlp_out = _dev(st, "tmlp_out", ctx)               # [1,1,6144]

    # ── rope table from pos ──
    var pos = _dev(st, "pos", ctx)                         # [1,256,3]
    var pos_flat = Tensor.from_host(pos.to_host(ctx), [L * 3], STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(32); axes.append(48); axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, Float32(1.0e3), ctx, STDtype.F32)

    # ── ref inputs ──
    var ref_tokens = _arc(st, "ref_tokens", ctx)           # [1,IMGLEN,64]
    var img_in_ref_nz = _arc(st, "img_in_ref", ctx)        # [6144,64] NONZERO param
    var img_in_ref_zero = TArc(zeros_device([FEATURES, IN_CH], STDtype.F32, ctx))

    var harness = ParityHarness(0.999)
    var allok = True

    var d_velocity = _dev(st, "d_velocity", ctx)           # [1,IMGLEN,64]

    # ══════════════════════════════════════════════════════════════════════════
    # (A) NON-ZERO img_in_ref + NON-DEGENERATE ref vs torch autograd
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("---- (A) nonzero-ref d_combined + d_img_in_ref vs torch ----")
    var fwd_nz = krea2_stack_lora_forward[L, HEADS, KVHEADS, HEADDIM](
        combined, blk_vec2, tmlp_out, w, lora,
        rope[0], rope[1], EPS, TXTLEN, IMGLEN, ctx,
        Optional[Int](None),
        Optional[TArc](ref_tokens.copy()), Optional[TArc](img_in_ref_nz.copy()),
    )
    # sanity: our forward velocity (WITH ref) should match the oracle's (WITH ref).
    _check_cos(harness, "velocity(with ref)", fwd_nz.velocity[].to_host(ctx),
               _host(st, "velocity", ctx), allok)
    var grads_nz = krea2_stack_lora_backward[L, HEADS, KVHEADS, HEADDIM](
        d_velocity, blk_vec2, tmlp_out, w, lora, fwd_nz,
        rope[0], rope[1], EPS, ctx,
        Optional[Int](None),
        Optional[TArc](ref_tokens.copy()), Optional[TArc](img_in_ref_nz.copy()),
    )
    _check_cos(harness, "d_combined", grads_nz.d_combined[].to_host(ctx),
               _host(st, "kref_ref.d_combined", ctx), allok)
    if not grads_nz.d_img_in_ref:
        print("  d_img_in_ref MISSING — FAIL")
        allok = False
    else:
        _check_cos(harness, "d_img_in_ref",
                   grads_nz.d_img_in_ref.value()[].to_host(ctx),
                   _host(st, "kref_ref.d_img_in_ref", ctx), allok)

    # ══════════════════════════════════════════════════════════════════════════
    # (B) FLAGS-OFF (C13): ref=None == ref-on-with-ZERO-weight, byte-identical
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("---- (B) flags-off (ref=None) == zero-init ref (max_abs 0) ----")
    var fwd_off = krea2_stack_lora_forward[L, HEADS, KVHEADS, HEADDIM](
        combined, blk_vec2, tmlp_out, w, lora,
        rope[0], rope[1], EPS, TXTLEN, IMGLEN, ctx,
    )   # NO ref args → the exact pre-change text-to-image path
    var grads_off = krea2_stack_lora_backward[L, HEADS, KVHEADS, HEADDIM](
        d_velocity, blk_vec2, tmlp_out, w, lora, fwd_off,
        rope[0], rope[1], EPS, ctx,
    )
    var fwd_z = krea2_stack_lora_forward[L, HEADS, KVHEADS, HEADDIM](
        combined, blk_vec2, tmlp_out, w, lora,
        rope[0], rope[1], EPS, TXTLEN, IMGLEN, ctx,
        Optional[Int](None),
        Optional[TArc](ref_tokens.copy()), Optional[TArc](img_in_ref_zero.copy()),
    )
    var grads_z = krea2_stack_lora_backward[L, HEADS, KVHEADS, HEADDIM](
        d_velocity, blk_vec2, tmlp_out, w, lora, fwd_z,
        rope[0], rope[1], EPS, ctx,
        Optional[Int](None),
        Optional[TArc](ref_tokens.copy()), Optional[TArc](img_in_ref_zero.copy()),
    )
    _check_zero("velocity off vs zero-init", fwd_off.velocity[].to_host(ctx),
                fwd_z.velocity[].to_host(ctx), allok)
    _check_zero("d_combined off vs zero-init", grads_off.d_combined[].to_host(ctx),
                grads_z.d_combined[].to_host(ctx), allok)
    for bi in range(NBLOCKS):
        var base = bi * KREA2_SLOTS_PER_BLOCK
        for s in range(KREA2_SLOTS_PER_BLOCK):
            var slot = _slot_name(s)
            var go = grads_off.grads[base + s].copy()
            var gz = grads_z.grads[base + s].copy()
            if not go.d_a or not gz.d_a or not go.d_b or not gz.d_b:
                print("  blk", bi, slot, " adapter MISSING — FAIL")
                allok = False
                continue
            _check_zero("blk" + String(bi) + "." + slot + " dA",
                        go.d_a.value(), gz.d_a.value(), allok)
            _check_zero("blk" + String(bi) + "." + slot + " dB",
                        go.d_b.value(), gz.d_b.value(), allok)

    # ══════════════════════════════════════════════════════════════════════════
    # (C) PARAM AdamW step + save/resume round-trip byte-exact
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("---- (C) Krea2ImgInRefParam AdamW + save/resume byte-exact ----")
    var param = make_krea2_img_in_ref_param(FEATURES, IN_CH)
    # non-degenerate grad (the emitted d_img_in_ref from path A).
    var dref = grads_nz.d_img_in_ref.value()[].to_host(ctx)
    krea2_img_in_ref_adamw_step(param, dref, 1, Float32(1.0e-3), ctx)
    var save_path = String("/tmp/krea2_img_in_ref_roundtrip.safetensors")
    var nsaved = save_krea2_img_in_ref(param, save_path, ctx)
    var reloaded = load_krea2_img_in_ref_resume(FEATURES, IN_CH, save_path, ctx)
    # byte-exact: weight (bf16), m, v (f32).
    var w_ok = True
    for i in range(len(param.w)):
        if param.w[i] != reloaded.w[i]:
            w_ok = False
            break
    var m_ok = True
    for i in range(len(param.m)):
        if param.m[i] != reloaded.m[i]:
            m_ok = False
            break
    var v_ok = True
    for i in range(len(param.v)):
        if param.v[i] != reloaded.v[i]:
            v_ok = False
            break
    print("  saved tensors =", nsaved, "  weight byte-exact =", w_ok,
          "  m byte-exact =", m_ok, "  v byte-exact =", v_ok)
    # AdamW must have MOVED the weight off zero (non-degenerate grad).
    var moved = False
    for i in range(len(param.w)):
        if param.w[i] != Float32(0.0).cast[DType.bfloat16]():
            moved = True
            break
    print("  AdamW moved weight off zero =", moved)
    if not (w_ok and m_ok and v_ok and moved):
        allok = False

    print("")
    if allok:
        print("VERDICT: PASS — krea2 img_in_ref projection matches torch (cos>=0.999),",
              "flags-off byte-identical, param round-trip byte-exact")
    else:
        print("VERDICT: FAIL — see FAIL lines above")
        raise Error("krea2_stack_ref_parity FAILED")
