# models/lingbotvideo/parity/a2_moe_probe.mojo — CHUNK A2 gate.
#
# Loads oracle_a2.safetensors (all tensors stored F32), runs the LingBotVideo
# grouped-sigmoid MoE (router + 128 SwiGLU experts top-8 + shared expert), and
# gates against the captured references.
#
# DTYPE CONTRACT (mirrors the oracle generator):
#   router GEMM: tokens.float() @ router_weightᵀ  -> F32   (weights F32, no bias)
#   experts + shared: bf16 tokens/weights, F32 combine
#
# TWO gate checks:
#   1. ROUTER: selected expert ids vs top_indices (position match, target 576/576)
#              + gating weights vs top_scores (cos >= 0.9999).
#   2. OUTPUT: full MoE out vs `out` (cos >= 0.999 + magnitude ratio).
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/a2_moe_probe.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.parity import ParityHarness
from serenitymojo.models.lingbotvideo.moe import (
    lingbot_grouped_sigmoid_router,
    lingbot_grouped_sigmoid_router_gpu,
    lingbot_video_moe,
    TOP_K,
)

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime S = 72
comptime H = 2048


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _load_bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(_load_f32(st, name, ctx), STDtype.BF16, ctx)


def _reshape_f32(t: Tensor, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(t.to_host(ctx), shape^, STDtype.F32, ctx)


def _reshape_bf16(t: Tensor, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(t.to_host(ctx), shape^, STDtype.BF16, ctx)


def main() raises:
    var ctx = DeviceContext()

    print("[A2] loading oracle_a2.safetensors")
    var st = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_a2.safetensors")

    # ffn_in [1,72,2048] stored f32 (of bf16 values) -> [72,2048] both dtypes.
    var ffn_raw = _load_f32(st, "ffn_in", ctx)
    var ffn_f32 = _reshape_f32(ffn_raw, [S, H], ctx)
    var ffn_bf16 = _reshape_bf16(ffn_raw, [S, H], ctx)

    # router: F32 weight + F32 bias.
    var router_weight = _load_f32(st, "router_weight", ctx)          # [128,2048]
    var bias = _load_f32(st, "e_score_correction_bias", ctx)         # [128]

    # experts + shared: bf16.
    var w1 = _load_bf16(st, "w1", ctx)   # [128,768,2048]
    var w3 = _load_bf16(st, "w3", ctx)   # [128,768,2048]
    var w2 = _load_bf16(st, "w2", ctx)   # [128,2048,768]
    var shared_gate = _load_bf16(st, "shared_gate", ctx)  # [768,2048]
    var shared_up = _load_bf16(st, "shared_up", ctx)      # [768,2048]
    var shared_down = _load_bf16(st, "shared_down", ctx)  # [2048,768]

    # ── GATE 1: ROUTER exactness ─────────────────────────────────────────────
    print("[A2] running router (GPU)")
    var logits = linear(ffn_f32, router_weight, None, ctx)  # [S,E] f32
    var plan = lingbot_grouped_sigmoid_router_gpu(logits, bias, ctx)

    # top_indices is stored I32 — read raw bytes (little-endian int32).
    var ti_bytes = st.tensor_bytes("top_indices")
    var ti_ref = List[Int]()
    for i in range(S * TOP_K):
        var b0 = Int(ti_bytes[i * 4])
        var b1 = Int(ti_bytes[i * 4 + 1])
        var b2 = Int(ti_bytes[i * 4 + 2])
        var b3 = Int(ti_bytes[i * 4 + 3])
        ti_ref.append(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
    var ts_ref = _load_f32(st, "top_scores", ctx).to_host(ctx)   # [S,8] f32

    var pos_match = 0
    for i in range(S * TOP_K):
        if ti_ref[i] == plan.expert_ids[i]:
            pos_match += 1

    # gating cos vs top_scores.
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(S * TOP_K):
        var a = Float64(plan.gating[i])
        var b = Float64(ts_ref[i])
        dot += a * b
        na += a * a
        nb += b * b
    var gate_cos = dot / (sqrt(na) * sqrt(nb))

    print("[A2] router position match:", pos_match, "/", S * TOP_K)
    print("[A2] gating cos vs top_scores:", gate_cos)
    var router_pass = (pos_match == S * TOP_K) and (gate_cos >= 0.9999)
    if router_pass:
        print("[A2] ROUTER GATE PASS (ids 576/576, gating cos >= 0.9999)")
    else:
        print("[A2] ROUTER GATE: check numbers above")

    # ── GATE 2: full MoE output ──────────────────────────────────────────────
    print("[A2] running full MoE forward")
    var mine = lingbot_video_moe(
        ffn_bf16, ffn_f32, router_weight, bias,
        w1, w3, w2, shared_gate, shared_up, shared_down, ctx,
    )

    var reference = _load_f32(st, "out", ctx)  # [1,72,2048]
    var ref_host = reference.to_host(ctx)
    var mine_host = mine.to_host(ctx)

    var harness = ParityHarness(0.999)
    var res = harness.compare_host(mine_host, ref_host)

    var nm: Float64 = 0.0
    var nr: Float64 = 0.0
    for i in range(len(mine_host)):
        nm += Float64(mine_host[i]) * Float64(mine_host[i])
        nr += Float64(ref_host[i]) * Float64(ref_host[i])
    var mag_ratio = sqrt(nm) / sqrt(nr)

    print("[A2] output cos =", res.cos, " max_abs_diff =", res.max_abs)
    print("[A2] |mine|/|ref| =", mag_ratio)
    if res.passed:
        print("[A2] OUTPUT GATE PASS (cos >= 0.999)")
    else:
        print("[A2] OUTPUT GATE FAIL (cos < 0.999)")

    if router_pass and res.passed:
        print("[A2] ===== A2 GATE PASS =====")
    else:
        print("[A2] ===== A2 GATE FAIL =====")
