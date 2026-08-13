# Replay the real Serenity Klein train-step dump through the Mojo forward path.
#
# This is the first real cached-data Klein train reference gate. It consumes
# parity/klein_train_ref_step000.safetensors from scripts/klein_dump_train_ref.py
# and compares the Mojo B=0 LoRA forward against Serenity's dumped
# trace.packed_predicted_flow and output.predicted.
#
# It is not the full train parity gate yet: loss/backward/AdamW adapter deltas are
# still compared by the next gate.

from std.collections import List, Optional
from max.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import (
    reshape as _reshape, reshape_owned as _reshape_owned, permute as _permute,
)
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.tensor import Tensor

from serenity_trainer.model.klein.double_block import DoubleBlockWeights
from serenity_trainer.model.klein.single_block import SingleBlockWeights
from serenity_trainer.model.klein.weights import (
    build_klein_vec_silu,
    build_klein_step_mods_device_cached,
    load_double_block_weights,
    load_klein_stack_base,
    load_klein_step_mod_weights,
    load_single_block_weights,
)
from serenity_trainer.model.KleinModel import (
    KDIM, KH, KDh, KIN_CH, KOUT_CH, KTXT_CH, KNUM_DOUBLE, KNUM_SINGLE,
    KTIMESTEP_DIM, build_klein9b_lora_set, build_klein_rope_tables_port,
    klein_inference_forward,
)
from serenity_trainer.model.KleinVAE import _unpatchify_packed
from serenity_trainer.model.klein.klein_stack_lora import klein_lora_set_to_device


comptime TArc = ArcPointer[Tensor]

comptime PARITY = "trainer/parity/klein_train_ref_step000.safetensors"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"

comptime HL = 32
comptime WL = 32
comptime N_IMG = HL * WL
comptime N_TXT = 512
comptime S = N_IMG + N_TXT
comptime TIMESTEP = Float32(545.0)
comptime LORA_RANK = 16
comptime LORA_ALPHA = Float32(16.0)

comptime SCRATCH_SLAB_BYTES = 256 * 1024 * 1024
comptime SCRATCH_NUM_SLABS = 4


def _sec(ns0: UInt, ns1: UInt) -> Float64:
    return Float64(ns1 - ns0) / Float64(1000000000.0)


def _compare_host(
    label: String, got: List[Float32], expected_tensor: Tensor,
    ctx: DeviceContext, min_cos: Float64,
) raises:
    var expected = expected_tensor.to_host(ctx)
    if len(got) != len(expected):
        raise Error(
            label + String(": len mismatch got ") + String(len(got))
            + String(" expected ") + String(len(expected))
        )

    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var max_abs = Float32(0.0)
    var nonfinite = 0
    for i in range(len(got)):
        var a = got[i]
        var b = expected[i]
        if a != a or b != b or (a - a) != Float32(0.0) or (b - b) != Float32(0.0):
            nonfinite += 1
            continue
        dot += Float64(a) * Float64(b)
        na += Float64(a) * Float64(a)
        nb += Float64(b) * Float64(b)
        var d = a - b
        var ad = d if d >= Float32(0.0) else -d
        if ad > max_abs:
            max_abs = ad

    var cos = dot / (sqrt(na) * sqrt(nb))
    print(label, "n =", len(got), "cos =", cos, "max_abs_diff =", max_abs, "nonfinite =", nonfinite)
    print(label, "got[0:3] =", got[0], got[1], got[2],
          "ref[0:3] =", expected[0], expected[1], expected[2])
    if cos < min_cos:
        raise Error(label + String(": cosine below gate"))


def _compare_tensor(
    label: String, got_tensor: Tensor, expected_tensor: Tensor,
    ctx: DeviceContext, min_cos: Float64,
) raises:
    var got = got_tensor.to_host(ctx)
    _compare_host(label, got^, expected_tensor, ctx, min_cos)


def main() raises:
    var ctx = DeviceContext()
    var all0 = perf_counter_ns()

    print("=== Klein train-ref forward replay ===")
    print("[parity]", PARITY)
    print("[ckpt]  ", CKPT)
    print("[shape] N_IMG =", N_IMG, "N_TXT =", N_TXT, "timestep =", TIMESTEP)

    var st = ShardedSafeTensors.open(String(PARITY))
    var img = cast_tensor(
        Tensor.from_view(st.tensor_view(String("trace.packed_latent_input")), ctx),
        STDtype.BF16,
        ctx,
    )
    var txt = cast_tensor(
        Tensor.from_view(st.tensor_view(String("trace.encoder_hidden_states")), ctx),
        STDtype.BF16,
        ctx,
    )
    var ref_packed = Tensor.from_view(st.tensor_view(String("trace.packed_predicted_flow")), ctx)
    var ref_predicted = Tensor.from_view(st.tensor_view(String("output.predicted")), ctx)

    var img_tok = _reshape_owned(img^, [N_IMG, KIN_CH])
    var txt_tok = _reshape_owned(txt^, [N_TXT, KTXT_CH])

    var load0 = perf_counter_ns()
    var ckpt = SafeTensors.open(String(CKPT))
    var ts_dev = Tensor.from_host([TIMESTEP], [1], STDtype.F32, ctx)
    var vec_silu = build_klein_vec_silu(ckpt, ts_dev, KTIMESTEP_DIM, KDIM, ctx)
    var base = load_klein_stack_base(ckpt, vec_silu, KDIM, ctx)
    var step_mod_w = load_klein_step_mod_weights(ckpt, KDIM, ctx)

    var dbw = List[DoubleBlockWeights]()
    for bi in range(KNUM_DOUBLE):
        dbw.append(load_double_block_weights(ckpt, bi, ctx))
    var sbw = List[SingleBlockWeights]()
    for bi in range(KNUM_SINGLE):
        sbw.append(load_single_block_weights(ckpt, bi, ctx))
    var load1 = perf_counter_ns()
    print("[load] base +", len(dbw), "double +", len(sbw), "single blocks")

    var lora_host = build_klein9b_lora_set(LORA_RANK, LORA_ALPHA)
    var lora = klein_lora_set_to_device(lora_host, ctx)

    var rope_tup = build_klein_rope_tables_port[N_IMG, N_TXT, KH, KDh](ctx, STDtype.BF16)
    ref cos_t = rope_tup[0]
    ref sin_t = rope_tup[1]

    var mods = build_klein_step_mods_device_cached(
        step_mod_w, TIMESTEP, Optional[Float32](None), KTIMESTEP_DIM, KDIM, ctx
    )
    var img_mod_dev = mods[0].copy()
    var txt_mod_dev = mods[1].copy()
    var single_mod_dev = mods[2].copy()

    print("[forward] replaying Serenity packed train inputs ...")
    var pred0 = perf_counter_ns()
    var scratch = ScratchRingAllocator(ctx, SCRATCH_SLAB_BYTES, SCRATCH_NUM_SLABS)
    var img_arc = TArc(img_tok^)
    var txt_arc = TArc(txt_tok^)
    var flow = klein_inference_forward[N_IMG, N_TXT, S](
        img_arc, txt_arc, base, dbw, sbw, lora,
        img_mod_dev, txt_mod_dev, single_mod_dev, cos_t, sin_t, ctx, scratch,
    )
    var pred1 = perf_counter_ns()

    var flow_tokens = Tensor.from_host(flow.copy(), [1, N_IMG, KOUT_CH], STDtype.BF16, ctx)
    var flow_b = _reshape(flow_tokens, [1, HL, WL, KOUT_CH], ctx)
    var flow_perm = _permute(flow_b, [0, 3, 1, 2], ctx)
    var predicted_flow_patch = _reshape_owned(flow_perm^, [1, KOUT_CH, HL, WL])
    var predicted = _unpatchify_packed(predicted_flow_patch, ctx)

    _compare_host("packed_flow", flow, ref_packed, ctx, Float64(0.999))
    _compare_tensor("output.predicted", predicted, ref_predicted, ctx, Float64(0.999))

    var all1 = perf_counter_ns()
    print("time_s: load =", _sec(load0, load1), "forward =", _sec(pred0, pred1),
          "total =", _sec(all0, all1))
    print("KLEIN TRAIN REF FORWARD REPLAY PASS")
