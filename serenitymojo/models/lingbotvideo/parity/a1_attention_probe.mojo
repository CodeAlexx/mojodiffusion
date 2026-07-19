# models/lingbotvideo/parity/a1_attention_probe.mojo — CHUNK A1 gate.
#
# Loads oracle_a1.safetensors (all tensors stored F32), runs lingbot_attention,
# and compares the result to the captured reference `out` with ParityHarness.
# DTYPE CONTRACT: x + to_q/k/v/out weights + b_o -> bf16; norm_q/k + freqs -> F32.
# Target: cos >= 0.999.
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/a1_attention_probe.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.parity import ParityHarness
from serenitymojo.models.lingbotvideo.backbone import (
    LingBotAttnConfig,
    lingbot_attention,
)

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime S = 72  # n_video(64) + text(8)


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    # Oracle tensors are stored F32; from_view yields an F32 Tensor.
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _load_bf16(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(_load_f32(st, name, ctx), STDtype.BF16, ctx)


def main() raises:
    var ctx = DeviceContext()
    var cfg = LingBotAttnConfig.default()

    print("[A1] loading oracle_a1.safetensors")
    var st = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_a1.safetensors")

    # bf16: x, q/k/v/o weights, output bias.
    var x = _load_bf16(st, "x", ctx)          # [1,72,2048]
    var w_q = _load_bf16(st, "w_q", ctx)      # [2048,2048]
    var w_k = _load_bf16(st, "w_k", ctx)
    var w_v = _load_bf16(st, "w_v", ctx)
    var w_o = _load_bf16(st, "w_o", ctx)
    var b_o = _load_bf16(st, "b_o", ctx)      # [2048]

    # f32: RMSNorm gamma + RoPE cos/sin tables.
    var norm_q_w = _load_f32(st, "norm_q_w", ctx)   # [128]
    var norm_k_w = _load_f32(st, "norm_k_w", ctx)   # [128]
    var cos = _load_f32(st, "freqs_cos", ctx)       # [72,64]
    var sin = _load_f32(st, "freqs_sin", ctx)       # [72,64]

    print("[A1] running lingbot_attention[S=", S, "]")
    var mine = lingbot_attention[S](
        x, w_q, w_k, w_v, w_o, b_o, norm_q_w, norm_k_w, cos, sin, cfg, ctx
    )

    # ── parity vs `out` ──────────────────────────────────────────────────────
    var reference = _load_f32(st, "out", ctx)
    var ref_host = reference.to_host(ctx)
    var mine_host = mine.to_host(ctx)

    var harness = ParityHarness(0.999)
    var res = harness.compare_host(mine_host, ref_host)

    # magnitude ratio |mine| / |ref| (L2).
    var nm: Float64 = 0.0
    var nr: Float64 = 0.0
    for i in range(len(mine_host)):
        nm += Float64(mine_host[i]) * Float64(mine_host[i])
        nr += Float64(ref_host[i]) * Float64(ref_host[i])
    var mag_ratio = sqrt(nm) / sqrt(nr)

    print("[A1] parity:", res)
    print("[A1] cos =", res.cos, " max_abs_diff =", res.max_abs)
    print("[A1] |mine|/|ref| =", mag_ratio)
    if res.passed:
        print("[A1] GATE PASS (cos >= 0.999)")
    else:
        print("[A1] GATE FAIL (cos < 0.999)")
