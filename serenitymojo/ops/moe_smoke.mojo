# ops/moe_smoke.mojo — MoE primitives GPU smoke + parity driver.
#
# Builds a DeviceContext, loads the SAME fixed inputs as the numpy oracle
# (serenitymojo/ops/parity/gen_moe_reference.py, materialised into the generated
# fixture moe_ref_data.mojo), runs the full MoE-FFN on the GPU:
#   top_k_router -> grouped_expert_ffn -> gated_scatter_add
# and compares each stage to the numpy reference via ParityHarness.
#
# Config: T=16 tokens, E=4 experts, top-2, hidden=32, ffn=64 (F32 storage so the
# main parity gate isolates op correctness from quantization). BF16/F16 storage
# smoke paths assert dtype preservation at MoE tensor boundaries. Gate:
# cos >= 0.999.
#
# Run: pixi run mojo run -I . serenitymojo/ops/moe_smoke.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.moe import (
    RouterPlan,
    top_k_router,
    grouped_expert_ffn,
    gated_scatter_add,
)
from serenitymojo.ops.parity.moe_ref_data import (
    MOE_T,
    MOE_E,
    MOE_K,
    MOE_H,
    MOE_F,
    tokens as ref_tokens,
    logits as ref_logits,
    gate_w as ref_gate_w,
    up_w as ref_up_w,
    down_w as ref_down_w,
    ref_expert_ids,
    ref_gating,
    ref_expert_out,
    ref_accum,
)


def _slot_indices() -> List[Int]:
    var indices = List[Int]()
    for t in range(MOE_T):
        for _ in range(MOE_K):
            indices.append(t)
    return indices^


def _zeros(n: Int) -> List[Float32]:
    var zeros = List[Float32]()
    for _ in range(n):
        zeros.append(Float32(0.0))
    return zeros^


def _run_storage_smoke(
    dtype: STDtype,
    name: String,
    plan: RouterPlan,
    indices: List[Int],
    ctx: DeviceContext,
    harness: ParityHarness,
) raises -> Bool:
    var tok = Tensor.from_host(ref_tokens(), [MOE_T, MOE_H], dtype, ctx)
    var gate_w = Tensor.from_host(
        ref_gate_w(), [MOE_E, MOE_F, MOE_H], dtype, ctx
    )
    var up_w = Tensor.from_host(
        ref_up_w(), [MOE_E, MOE_F, MOE_H], dtype, ctx
    )
    var down_w = Tensor.from_host(
        ref_down_w(), [MOE_E, MOE_H, MOE_F], dtype, ctx
    )

    var expert_out = grouped_expert_ffn(tok, gate_w, up_w, down_w, plan, ctx)
    if expert_out.dtype() != dtype:
        raise Error(
            name
            + String(" grouped_expert_ffn dtype changed to ")
            + expert_out.dtype().name()
        )
    var r_ffn = harness.compare(expert_out, ref_expert_out(), ctx)
    print("grouped_ffn/", name, " ", r_ffn, " dtype=", expert_out.dtype().name())

    var accum = Tensor.from_host(
        _zeros(MOE_T * MOE_H), [MOE_T, MOE_H], dtype, ctx
    )
    gated_scatter_add(expert_out, plan.gating.copy(), indices, accum, ctx)
    if accum.dtype() != dtype:
        raise Error(
            name
            + String(" gated_scatter_add accum dtype changed to ")
            + accum.dtype().name()
        )
    var r_final = harness.compare(accum, ref_accum(), ctx)
    print("scatter_add/", name, " ", r_final, " dtype=", accum.dtype().name())

    var accum_f32 = Tensor.from_host(
        _zeros(MOE_T * MOE_H), [MOE_T, MOE_H], STDtype.F32, ctx
    )
    gated_scatter_add(expert_out, plan.gating.copy(), indices, accum_f32, ctx)
    if accum_f32.dtype() != STDtype.F32:
        raise Error(name + String(" F32 accumulator dtype changed"))
    var r_final_f32 = harness.compare(accum_f32, ref_accum(), ctx)
    print(
        "scatter_add/", name, "->f32 ", r_final_f32,
        " dtype=", accum_f32.dtype().name(),
    )

    return r_ffn.passed and r_final.passed and r_final_f32.passed


def main() raises:
    var ctx = DeviceContext()
    var harness = ParityHarness()

    # ── upload inputs ─────────────────────────────────────────────────────────
    var tok = Tensor.from_host(ref_tokens(), [MOE_T, MOE_H], STDtype.F32, ctx)
    var logits = Tensor.from_host(ref_logits(), [MOE_T, MOE_E], STDtype.F32, ctx)
    var gate_w = Tensor.from_host(
        ref_gate_w(), [MOE_E, MOE_F, MOE_H], STDtype.F32, ctx
    )
    var up_w = Tensor.from_host(
        ref_up_w(), [MOE_E, MOE_F, MOE_H], STDtype.F32, ctx
    )
    var down_w = Tensor.from_host(
        ref_down_w(), [MOE_E, MOE_H, MOE_F], STDtype.F32, ctx
    )

    # ── stage 1: router ───────────────────────────────────────────────────────
    var plan = top_k_router(logits, MOE_K, ctx)

    # Expert-id parity (exact integer match) — cast both to F32 for the harness.
    var ids_f = List[Float32]()
    for i in range(len(plan.expert_ids)):
        ids_f.append(Float32(plan.expert_ids[i]))
    var ref_ids = ref_expert_ids()
    var ref_ids_f = List[Float32]()
    for i in range(len(ref_ids)):
        ref_ids_f.append(Float32(ref_ids[i]))
    var ids_exact = True
    if len(ids_f) != len(ref_ids_f):
        ids_exact = False
    else:
        for i in range(len(ids_f)):
            if ids_f[i] != ref_ids_f[i]:
                ids_exact = False
    var r_ids = harness.compare_host(ids_f, ref_ids_f)
    print("router/expert_ids ", r_ids, " exact=", ids_exact)

    var r_gate = harness.compare_host(plan.gating.copy(), ref_gating())
    print("router/gating     ", r_gate)

    # ── stage 2: grouped expert FFN ───────────────────────────────────────────
    var expert_out = grouped_expert_ffn(tok, gate_w, up_w, down_w, plan, ctx)
    var r_ffn = harness.compare(expert_out, ref_expert_out(), ctx)
    print("grouped_ffn       ", r_ffn)

    # ── stage 3: gated scatter-add ────────────────────────────────────────────
    # indices: token-major, slot s = t*K + j -> token t.
    var indices = _slot_indices()
    # accum starts at zero [T, H].
    var accum = Tensor.from_host(
        _zeros(MOE_T * MOE_H), [MOE_T, MOE_H], STDtype.F32, ctx
    )
    gated_scatter_add(expert_out, plan.gating.copy(), indices, accum, ctx)
    var r_final = harness.compare(accum, ref_accum(), ctx)
    print("scatter_add/final ", r_final)

    # ── BF16/F16 boundary contract smoke ─────────────────────────────────────
    var bf16_pass = _run_storage_smoke(
        STDtype.BF16, String("bf16"), plan, indices, ctx, harness
    )
    var f16_pass = _run_storage_smoke(
        STDtype.F16, String("f16"), plan, indices, ctx, harness
    )

    # ── overall gate ──────────────────────────────────────────────────────────
    var all_pass = (
        ids_exact
        and r_ids.passed
        and r_gate.passed
        and r_ffn.passed
        and r_final.passed
        and bf16_pass
        and f16_pass
    )
    print("")
    if all_pass:
        print("ALL MoE PARITY GATES PASSED (cos >= 0.999, expert_ids exact)")
    else:
        print("MoE PARITY FAILURE")
        raise Error("moe_smoke parity gate failed")
