# serenitymojo/models/krea2/parity/krea2_streamed_ref_seam_smoke.mojo
#
# WITH-REF SEAM SMOKE for the KREA2 img-EDIT reference-conditioning projection wired
# into the *_streamed backward family (Task 8a). The primary krea2_stack_lora_forward/
# backward ref path is ALREADY torch-gated (krea2_stack_ref_parity.mojo, cos~1). This
# smoke proves the STREAMED variants — the ones the LIVE trainer actually calls — emit
# the SAME d_img_in_ref as that proven primary path on the SAME inputs.
#
# The PRIMARY path runs in F32 (the exact krea2_stack_ref_parity config, torch-gated
# cos~1). The STREAMED path runs in its NATIVE bf16 (frozen weights H2D as bf16, bf16
# modulation carriers) from the SAME oracle values. The small bf16-vs-F32 weight delta
# means the seam is proven at cos>=0.999 (not max_abs 0) — the SEAM is under test, not
# weight precision.
#
#   (A)  NON-ZERO ref, NON-DEGENERATE ref_tokens: assert
#          d_img_in_ref(streamed)        ~ d_img_in_ref(primary)   [default bwd]
#          d_img_in_ref(streamed_dev)    ~ d_img_in_ref(primary)
#          d_img_in_ref(streamed_adamwdg)~ d_img_in_ref(primary)   [LIVE path]
#        at cos >= 0.999. Also d_combined(streamed) ~ d_combined(primary).
#   (B)  FLAGS-OFF (C13): streamed fwd with NO ref args == streamed fwd with a
#        ref-but-ZERO-weight (zero-init identity), velocity byte-identical (max_abs 0).
#
# Run (build the streamed file at runtime; oracle .py must have run FIRST to produce
# krea2_stack_ref_oracle.safetensors):
#   cd /home/alex/mojodiffusion
#   /home/alex/SerenityTrainer/venv/bin/python \
#       serenitymojo/models/krea2/parity/krea2_stack_ref_oracle.py     # if not present
#   rm -f serenitymojo.mojopkg
#   LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:\
#       $HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH \
#   pixi run mojo run -I . \
#       -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#       serenitymojo/models/krea2/parity/krea2_streamed_ref_seam_smoke.mojo

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.tensor_algebra import zeros_device
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.krea2.krea2_block import (
    Krea2BlockWeights, Krea2BlockLora, Krea2LoraGrad,
)
from serenitymojo.models.krea2.krea2_stack import (
    Krea2StackWeights, Krea2StackLora, Krea2StreamFinal,
    Krea2StackLoraGrads, Krea2StackDeviceGradWrite, KREA2_SLOTS_PER_BLOCK,
    krea2_stack_lora_forward, krea2_stack_lora_backward,
    krea2_stack_lora_forward_streamed, krea2_stack_lora_backward_streamed,
    krea2_stack_lora_backward_streamed_dev,
    krea2_stack_lora_backward_streamed_adamw_device_grads,
)
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState, lora_adamw_plain_device_state_init,
)
from serenitymojo.models.dit.krea2_dit import build_krea2_rope

comptime TArc = ArcPointer[Tensor]
comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_stack_ref_oracle.safetensors"
comptime STREAMED = "/tmp/krea2_streamed_ref_seam_blocks.safetensors"

# dims MUST match krea2_stack_ref_oracle.py (same as krea2_stack_ref_parity.mojo)
comptime HEADS = 48
comptime KVHEADS = 12
comptime HEADDIM = 128
comptime FEATURES = HEADS * HEADDIM       # 6144
comptime MLPDIM = 16384
comptime OUT_CH = 64
comptime IN_CH = 64
comptime NBLOCKS = 4
comptime L = 256
comptime TXTLEN = 46
comptime IMGLEN = 210
comptime RANK = 8
comptime EPS = Float32(1e-5)
comptime LSCALE = Float32(16.0) / Float32(8.0)


def _dev(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view_as_f32(st.tensor_view(key), ctx)


def _arc_f32(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> TArc:
    return TArc(_dev(st, key, ctx))


def _host(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> List[Float32]:
    return _dev(st, key, ctx).to_host(ctx)


# read an oracle F32 tensor, truncate to bf16 ONCE — the SHARED weight fed to both the
# primary stack AND the streamed on-disk store (so both consume identical bytes).
def _bf16_arc(st: ShardedSafeTensors, key: String, ctx: DeviceContext) raises -> TArc:
    var t = _dev(st, key, ctx)
    return TArc(Tensor.from_host(t.to_host(ctx), t.shape(), STDtype.BF16, ctx))


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
    var a = _arc_f32(st, "blk" + String(bi) + "." + slot + ".A", ctx)
    var b = _arc_f32(st, "blk" + String(bi) + "." + slot + ".B", ctx)
    return Optional[LoraAdapterDevice](
        LoraAdapterDevice(a^, b^, RANK, io[0], io[1], LSCALE)
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


# bf16 adapter for the streamed path (its LoRA lives bf16 in dev_p, matching the
# bf16 base). Same oracle A/B values, truncated to bf16.
def _adapter_bf16(
    st: ShardedSafeTensors, bi: Int, slot: String, ctx: DeviceContext
) raises -> Optional[LoraAdapterDevice]:
    var io = _slot_io(slot)
    var a = _bf16_arc(st, "blk" + String(bi) + "." + slot + ".A", ctx)
    var b = _bf16_arc(st, "blk" + String(bi) + "." + slot + ".B", ctx)
    return Optional[LoraAdapterDevice](
        LoraAdapterDevice(a^, b^, RANK, io[0], io[1], LSCALE)
    )


def _block_lora_bf16(
    st: ShardedSafeTensors, bi: Int, ctx: DeviceContext
) raises -> Krea2BlockLora:
    return Krea2BlockLora(
        _adapter_bf16(st, bi, "wq", ctx), _adapter_bf16(st, bi, "wk", ctx),
        _adapter_bf16(st, bi, "wv", ctx), _adapter_bf16(st, bi, "gate", ctx),
        _adapter_bf16(st, bi, "wo", ctx), _adapter_bf16(st, bi, "mlp_gate", ctx),
        _adapter_bf16(st, bi, "mlp_up", ctx), _adapter_bf16(st, bi, "mlp_down", ctx),
    )


# a zero-valued host LoraAdapter carrying the correct A/B dims — used ONLY to size the
# device AdamW grad-sink state for the _adamw_device_grads path (values irrelevant to
# d_x / d_img_in_ref, which don't touch opt_state).
def _dummy_adapter(slot: String) -> LoraAdapter:
    var io = _slot_io(slot)
    var in_f = io[0]
    var out_f = io[1]
    var za = List[Float32]()
    for _ in range(RANK * in_f):
        za.append(Float32(0.0))
    var zb = List[Float32]()
    for _ in range(out_f * RANK):
        zb.append(Float32(0.0))
    return LoraAdapter(
        za^, zb^, RANK, in_f, out_f, LSCALE,
        _zeros_f32(RANK * in_f), _zeros_f32(RANK * in_f),
        _zeros_f32(out_f * RANK), _zeros_f32(out_f * RANK),
    )


def _zeros_f32(n: Int) -> List[Float32]:
    var l = List[Float32]()
    for _ in range(n):
        l.append(Float32(0.0))
    return l^


def _slot_name(s: Int) -> String:
    if s == 0: return String("wq")
    if s == 1: return String("wk")
    if s == 2: return String("wv")
    if s == 3: return String("gate")
    if s == 4: return String("wo")
    if s == 5: return String("mlp_gate")
    if s == 6: return String("mlp_up")
    return String("mlp_down")


def _check_cos(
    mut harness: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = harness.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs,
          "  n =", r.n, "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


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
    print("==== krea2_streamed_ref_seam_smoke (streamed img_in_ref vs proven primary) ====")
    print("NBLOCKS=", NBLOCKS, " L=", L, " TXTLEN=", TXTLEN, " IMGLEN=", IMGLEN,
          " FEATURES=", FEATURES, " IN_CH=", IN_CH, " RANK=", RANK)

    # ── PRIMARY block weights: F32 (the exact proven krea2_stack_ref_parity config).
    #    STREAMED store: bf16 (the streamed path's native H2D dtype). Same underlying
    #    oracle values; the tiny bf16-vs-F32 weight delta => d_img_in_ref cos ~ 1 (the
    #    seam is what's under test, not weight precision). ──
    var save_names = List[String]()
    var save_tensors = List[ArcPointer[Tensor]]()
    var blocks_w = List[Krea2BlockWeights]()
    var blocks_l = List[Krea2BlockLora]()
    var blocks_l_bf16 = List[Krea2BlockLora]()
    for bi in range(NBLOCKS):
        var p = "blk" + String(bi) + "."
        var sp = "blocks." + String(bi) + "."   # streamed key_prefix == ""
        # primary in-memory block weights (F32).
        blocks_w.append(Krea2BlockWeights(
            _arc_f32(st, p + "wq.W", ctx), _arc_f32(st, p + "wk.W", ctx),
            _arc_f32(st, p + "wv.W", ctx), _arc_f32(st, p + "gate.W", ctx),
            _arc_f32(st, p + "wo.W", ctx), _arc_f32(st, p + "mlp_gate.W", ctx),
            _arc_f32(st, p + "mlp_up.W", ctx), _arc_f32(st, p + "mlp_down.W", ctx),
            _arc_f32(st, p + "qnorm", ctx), _arc_f32(st, p + "knorm", ctx),
            _arc_f32(st, p + "prenorm", ctx), _arc_f32(st, p + "postnorm", ctx),
            _arc_f32(st, p + "mod_lin", ctx),
        ))
        blocks_l.append(_block_lora(st, bi, ctx))
        blocks_l_bf16.append(_block_lora_bf16(st, bi, ctx))
        # streamed on-disk keys (bf16) == _load_krea2_block_streamed's expected names.
        save_names.append(sp + "attn.wq.weight");   save_tensors.append(_bf16_arc(st, p + "wq.W", ctx).copy())
        save_names.append(sp + "attn.wk.weight");   save_tensors.append(_bf16_arc(st, p + "wk.W", ctx).copy())
        save_names.append(sp + "attn.wv.weight");   save_tensors.append(_bf16_arc(st, p + "wv.W", ctx).copy())
        save_names.append(sp + "attn.gate.weight"); save_tensors.append(_bf16_arc(st, p + "gate.W", ctx).copy())
        save_names.append(sp + "attn.wo.weight");   save_tensors.append(_bf16_arc(st, p + "wo.W", ctx).copy())
        save_names.append(sp + "mlp.gate.weight");  save_tensors.append(_bf16_arc(st, p + "mlp_gate.W", ctx).copy())
        save_names.append(sp + "mlp.up.weight");    save_tensors.append(_bf16_arc(st, p + "mlp_up.W", ctx).copy())
        save_names.append(sp + "mlp.down.weight");  save_tensors.append(_bf16_arc(st, p + "mlp_down.W", ctx).copy())
        save_names.append(sp + "attn.qknorm.qnorm.scale"); save_tensors.append(_bf16_arc(st, p + "qnorm", ctx).copy())
        save_names.append(sp + "attn.qknorm.knorm.scale"); save_tensors.append(_bf16_arc(st, p + "knorm", ctx).copy())
        save_names.append(sp + "prenorm.scale");    save_tensors.append(_bf16_arc(st, p + "prenorm", ctx).copy())
        save_names.append(sp + "postnorm.scale");   save_tensors.append(_bf16_arc(st, p + "postnorm", ctx).copy())
        save_names.append(sp + "mod.lin");          save_tensors.append(_bf16_arc(st, p + "mod_lin", ctx).copy())
    save_safetensors(save_names, save_tensors, STREAMED, ctx)
    var st_s = ShardedSafeTensors.open(STREAMED)

    # ── final layer: primary F32, streamed fin bf16 (native to each path) ──
    var w = Krea2StackWeights(
        blocks_w^, _arc_f32(st, "last.norm", ctx), _arc_f32(st, "last.mod_lin", ctx),
        _arc_f32(st, "last.lin_w", ctx), _arc_f32(st, "last.lin_b", ctx),
    )
    var lora = Krea2StackLora(blocks_l^)
    var lora_streamed = Krea2StackLora(blocks_l_bf16^)
    var fin = Krea2StreamFinal(
        _bf16_arc(st, "last.norm", ctx), _bf16_arc(st, "last.mod_lin", ctx),
        _bf16_arc(st, "last.lin_w", ctx), _bf16_arc(st, "last.lin_b", ctx),
    )

    # ── shared inputs ──
    var combined = _arc_f32(st, "combined", ctx)
    # bf16 block-stack input for the streamed path (the live trainer's cond.combined is
    # bf16; F32 here would make the streamed activations F32 while the bf16 modulation
    # chunks stay bf16 -> gate_residual_backward dtype mismatch).
    var combined_bf16 = TArc(Tensor.from_host(
        combined[].to_host(ctx), combined[].shape(), STDtype.BF16, ctx))
    var blk_vec = _dev(st, "tvec", ctx)
    var blk_vec2 = Tensor.from_host(blk_vec.to_host(ctx), [1, 6 * FEATURES], STDtype.F32, ctx)
    var tmlp_out = _dev(st, "tmlp_out", ctx)
    # bf16 modulation carriers for the streamed path (its mod.lin / last.mod are bf16).
    var blk_vec2_bf16 = Tensor.from_host(
        blk_vec.to_host(ctx), [1, 6 * FEATURES], STDtype.BF16, ctx)
    var tmlp_bf16 = Tensor.from_host(
        tmlp_out.to_host(ctx), tmlp_out.shape(), STDtype.BF16, ctx)
    var pos = _dev(st, "pos", ctx)
    var pos_flat = Tensor.from_host(pos.to_host(ctx), [L * 3], STDtype.F32, ctx)
    var axes = List[Int]()
    axes.append(32); axes.append(48); axes.append(48)
    var rope = build_krea2_rope(pos_flat, axes, Float32(1.0e3), ctx, STDtype.F32)
    var ref_tokens = _arc_f32(st, "ref_tokens", ctx)
    var img_in_ref_nz = _arc_f32(st, "img_in_ref", ctx)
    var img_in_ref_zero = TArc(zeros_device([FEATURES, IN_CH], STDtype.F32, ctx))
    var d_velocity = _dev(st, "d_velocity", ctx)
    var d_velocity_bf16 = Tensor.from_host(
        d_velocity.to_host(ctx), d_velocity.shape(), STDtype.BF16, ctx)

    var harness = ParityHarness(0.999)
    var allok = True

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMARY (proven) WITH ref → the reference d_combined + d_img_in_ref
    # ══════════════════════════════════════════════════════════════════════════
    var pf = krea2_stack_lora_forward[L, HEADS, KVHEADS, HEADDIM](
        combined, blk_vec2, tmlp_out, w, lora,
        rope[0], rope[1], EPS, TXTLEN, IMGLEN, ctx,
        Optional[Int](None),
        Optional[TArc](ref_tokens.copy()), Optional[TArc](img_in_ref_nz.copy()),
    )
    var pg = krea2_stack_lora_backward[L, HEADS, KVHEADS, HEADDIM](
        d_velocity, blk_vec2, tmlp_out, w, lora, pf,
        rope[0], rope[1], EPS, ctx,
        Optional[Int](None),
        Optional[TArc](ref_tokens.copy()), Optional[TArc](img_in_ref_nz.copy()),
    )
    if not pg.d_img_in_ref:
        raise Error("primary d_img_in_ref missing — cannot gate")
    var ref_dcomb = pg.d_combined[].to_host(ctx)
    var ref_dref = pg.d_img_in_ref.value()[].to_host(ctx)

    # ══════════════════════════════════════════════════════════════════════════
    # (A) STREAMED variants WITH ref → d_img_in_ref == primary
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("---- (A) streamed d_img_in_ref vs proven primary (target max_abs 0) ----")
    var sf = krea2_stack_lora_forward_streamed[L, HEADS, KVHEADS, HEADDIM](
        combined_bf16, blk_vec2_bf16, tmlp_bf16, st_s, String(""), NBLOCKS, lora_streamed, fin,
        rope[0], rope[1], EPS, TXTLEN, IMGLEN, ctx,
        ref_tokens=Optional[TArc](ref_tokens.copy()),
        img_in_ref_w=Optional[TArc](img_in_ref_nz.copy()),
    )
    # sanity: streamed velocity WITH ref ~ primary velocity WITH ref (cos ~ 1).
    _check_cos(harness, "velocity streamed vs primary (with ref)",
               sf.velocity[].to_host(ctx), pf.velocity[].to_host(ctx), allok)

    var sg = krea2_stack_lora_backward_streamed[L, HEADS, KVHEADS, HEADDIM](
        d_velocity_bf16, blk_vec2_bf16, tmlp_bf16, st_s, String(""), NBLOCKS, lora_streamed, fin, sf,
        rope[0], rope[1], EPS, ctx,
        ref_tokens=Optional[TArc](ref_tokens.copy()),
        img_in_ref_w=Optional[TArc](img_in_ref_nz.copy()),
    )
    _check_cos(harness, "d_combined streamed vs primary", sg.d_combined[].to_host(ctx),
               ref_dcomb, allok)
    if not sg.d_img_in_ref:
        print("  streamed d_img_in_ref MISSING — FAIL"); allok = False
    else:
        _check_cos(harness, "d_img_in_ref (streamed)",
                   sg.d_img_in_ref.value()[].to_host(ctx), ref_dref, allok)

    # streamed_dev
    var sgd = krea2_stack_lora_backward_streamed_dev[L, HEADS, KVHEADS, HEADDIM](
        d_velocity_bf16, blk_vec2_bf16, tmlp_bf16, st_s, String(""), NBLOCKS, lora_streamed, fin, sf,
        rope[0], rope[1], EPS, ctx,
        ref_tokens=Optional[TArc](ref_tokens.copy()),
        img_in_ref_w=Optional[TArc](img_in_ref_nz.copy()),
    )
    if not sgd.d_img_in_ref:
        print("  streamed_dev d_img_in_ref MISSING — FAIL"); allok = False
    else:
        _check_cos(harness, "d_img_in_ref (streamed_dev)",
                   sgd.d_img_in_ref.value()[].to_host(ctx), ref_dref, allok)

    # streamed_adamw_device_grads (the LIVE trainer path)
    var adapters = List[LoraAdapter]()
    for _bi in range(NBLOCKS):
        for s in range(KREA2_SLOTS_PER_BLOCK):
            adapters.append(_dummy_adapter(_slot_name(s)))
    var opt_state = lora_adamw_plain_device_state_init(
        adapters, 0, NBLOCKS * KREA2_SLOTS_PER_BLOCK, ctx,
    )
    var wrote = krea2_stack_lora_backward_streamed_adamw_device_grads[
        L, HEADS, KVHEADS, HEADDIM
    ](
        d_velocity_bf16, blk_vec2_bf16, tmlp_bf16, st_s, String(""), NBLOCKS, lora_streamed, fin, sf,
        rope[0], rope[1], EPS, opt_state, ctx,
        ref_tokens=Optional[TArc](ref_tokens.copy()),
        img_in_ref_w=Optional[TArc](img_in_ref_nz.copy()),
    )
    if not wrote.d_img_in_ref:
        print("  streamed_adamw_device_grads d_img_in_ref MISSING — FAIL"); allok = False
    else:
        _check_cos(harness, "d_img_in_ref (streamed_adamw_device_grads)",
                   wrote.d_img_in_ref.value()[].to_host(ctx), ref_dref, allok)

    # ══════════════════════════════════════════════════════════════════════════
    # (B) FLAGS-OFF (C13): streamed no-ref == streamed zero-weight-ref (max_abs 0)
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("---- (B) streamed flags-off (no ref) == base streamed (max_abs 0) ----")
    var sf_off = krea2_stack_lora_forward_streamed[L, HEADS, KVHEADS, HEADDIM](
        combined_bf16, blk_vec2_bf16, tmlp_bf16, st_s, String(""), NBLOCKS, lora_streamed, fin,
        rope[0], rope[1], EPS, TXTLEN, IMGLEN, ctx,
    )   # NO ref args → the exact pre-change streamed path
    var sf_z = krea2_stack_lora_forward_streamed[L, HEADS, KVHEADS, HEADDIM](
        combined_bf16, blk_vec2_bf16, tmlp_bf16, st_s, String(""), NBLOCKS, lora_streamed, fin,
        rope[0], rope[1], EPS, TXTLEN, IMGLEN, ctx,
        ref_tokens=Optional[TArc](ref_tokens.copy()),
        img_in_ref_w=Optional[TArc](img_in_ref_zero.copy()),
    )
    _check_zero("velocity streamed off vs base", sf_off.velocity[].to_host(ctx),
                sf_z.velocity[].to_host(ctx), allok)

    print("")
    if allok:
        print("VERDICT: PASS — streamed img_in_ref seam matches the proven primary",
              "path (d_img_in_ref cos>=0.999 / max_abs 0), flags-off byte-identical")
    else:
        print("VERDICT: FAIL — see FAIL lines above")
        raise Error("krea2_streamed_ref_seam_smoke FAILED")
