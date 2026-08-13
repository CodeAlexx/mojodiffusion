# models/lingbotvideo/parity/a3_block_probe.mojo — CHUNK A3 gate.
#
# Loads oracle_a3.safetensors (all tensors stored F32), runs the full
# lingbot_video_block (AdaLN 6-way modulation + A1 attention + A2 MoE + the two
# gated residuals), and gates the block output vs the captured `out`.
#
# DTYPE CONTRACT (mirrors the oracle generator / the reference forward):
#   x + attn Linears (w_q/k/v/o, b_o) + experts (w1/w2/w3, shared_*) -> bf16.
#   temb6 + scale_shift_table + the 6 modulation tensors + the 4 block RMSNorm
#   gammas (norm1/norm2/norm_post_attn/norm_post_ffn) + norm_q/k + rope -> F32.
#   router weight + bias -> F32.
#
# Gate: full block output vs `out`, cos >= 0.999 + magnitude ratio |mine|/|ref|.
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/a3_block_probe.mojo

from std.math import sqrt
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.models.lingbotvideo.backbone import (
    LingBotAttnConfig,
    lingbot_video_block,
)

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime S = 72  # n_video(64) + text(8)
comptime H = 2048


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    # Oracle tensors are stored F32; from_view yields an F32 Tensor.
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _load_bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(_load_f32(st, name, ctx), STDtype.BF16, ctx)


def _reshape_f32(t: Tensor, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(t.to_host(ctx), shape^, STDtype.F32, ctx)


def main() raises:
    var ctx = DeviceContext()
    var cfg = LingBotAttnConfig.default()

    print("[A3] loading oracle_a3.safetensors")
    var st = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_a3.safetensors")

    # x: bf16 [1,72,2048] (stored f32 of bf16 values).
    var x = _load_bf16(st, "x", ctx)

    # AdaLN modulation source: F32.  temb6 [72,12288], scale_shift_table [1,12288].
    var temb6 = _load_f32(st, "temb6", ctx)
    var scale_shift_table = _load_f32(st, "scale_shift_table", ctx)

    # attention weights: bf16 Linears + bias, F32 norms + rope.
    var w_q = _load_bf16(st, "w_q", ctx)
    var w_k = _load_bf16(st, "w_k", ctx)
    var w_v = _load_bf16(st, "w_v", ctx)
    var w_o = _load_bf16(st, "w_o", ctx)
    var b_o = _load_bf16(st, "b_o", ctx)
    var norm_q_w = _load_f32(st, "norm_q_w", ctx)   # [128]
    var norm_k_w = _load_f32(st, "norm_k_w", ctx)   # [128]
    var cos = _load_f32(st, "freqs_cos", ctx)       # [72,64]
    var sin = _load_f32(st, "freqs_sin", ctx)       # [72,64]

    # block RMSNorm gammas: F32.
    var norm1_w = _load_f32(st, "norm1_w", ctx)
    var norm2_w = _load_f32(st, "norm2_w", ctx)
    var norm_post_attn_w = _load_f32(st, "norm_post_attn_w", ctx)
    var norm_post_ffn_w = _load_f32(st, "norm_post_ffn_w", ctx)

    # MoE: F32 router weight/bias, bf16 experts + shared.
    var router_weight = _load_f32(st, "router_weight", ctx)          # [128,2048]
    var e_score_correction_bias = _load_f32(st, "e_score_correction_bias", ctx)
    var w1 = _load_bf16(st, "w1", ctx)   # [128,768,2048]
    var w3 = _load_bf16(st, "w3", ctx)   # [128,768,2048]
    var w2 = _load_bf16(st, "w2", ctx)   # [128,2048,768]
    var shared_gate = _load_bf16(st, "shared_gate", ctx)  # [768,2048]
    var shared_up = _load_bf16(st, "shared_up", ctx)      # [768,2048]
    var shared_down = _load_bf16(st, "shared_down", ctx)  # [2048,768]

    print("[A3] running lingbot_video_block[S=", S, "]")
    var mine = lingbot_video_block[S](
        x, temb6, scale_shift_table,
        w_q, w_k, w_v, w_o, b_o, norm_q_w, norm_k_w, cos, sin,
        norm1_w, norm2_w, norm_post_attn_w, norm_post_ffn_w,
        router_weight, e_score_correction_bias,
        w1, w3, w2, shared_gate, shared_up, shared_down,
        cfg, ctx,
    )

    # ── parity vs `out` ──────────────────────────────────────────────────────
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

    print("[A3] output cos =", res.cos, " max_abs_diff =", res.max_abs)
    print("[A3] |mine|/|ref| =", mag_ratio)
    if res.passed:
        print("[A3] ===== A3 GATE PASS (cos >= 0.999) =====")
    else:
        print("[A3] ===== A3 GATE FAIL (cos < 0.999) =====")
