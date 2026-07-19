# models/dit/parity/krea2_attention_parity_probe.mojo — PARITY GATE for krea2
# chunk 3 (the Attention op: GQA + QKNorm + RoPE + sigmoid-gate) vs a real
# ai-toolkit torch oracle.
#
# Oracle: krea2_attention_oracle.safetensors (weights BF16, rest F32):
#   x [1,32,6144], wq/gate_w/wo [6144,6144], wk/wv [1536,6144],
#   qnorm_scale/knorm_scale [128] F32, pos [1,32,3] F32, cos/sin [32,64] F32,
#   y [1,32,6144] F32 (Attention output).
#
# Gate: cos >= 0.999 vs y (bf16 matmul + SDPA roundoff -> nonzero max_abs, fine).
#
# Run: cd /home/alex/mojodiffusion && \
#   pixi run mojo run -I . serenitymojo/models/dit/parity/krea2_attention_parity_probe.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.models.dit.krea2_dit import (
    Krea2Config,
    build_krea2_rope,
    krea2_attention,
)


comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_attention_oracle.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(ORACLE)

    comptime L = 32
    comptime HEADS = 48
    comptime KVHEADS = 12
    comptime HEADDIM = 128

    var cfg = Krea2Config.default()
    var axes = cfg.rope_axes()  # [32,48,48]

    # ── Load oracle tensors. Weights/x are bf16-stored (production dtype). ────
    var x = Tensor.from_view(fx.tensor_view("x"), ctx)             # bf16 [1,32,6144]
    var wq = Tensor.from_view(fx.tensor_view("wq"), ctx)           # bf16
    var wk = Tensor.from_view(fx.tensor_view("wk"), ctx)
    var wv = Tensor.from_view(fx.tensor_view("wv"), ctx)
    var gate_w = Tensor.from_view(fx.tensor_view("gate_w"), ctx)
    var wo = Tensor.from_view(fx.tensor_view("wo"), ctx)
    var qnorm_scale = Tensor.from_view_as_f32(fx.tensor_view("qnorm_scale"), ctx)  # F32 [128]
    var knorm_scale = Tensor.from_view_as_f32(fx.tensor_view("knorm_scale"), ctx)
    var y_ref = Tensor.from_view_as_f32(fx.tensor_view("y"), ctx).to_host(ctx)

    # ── Build the RoPE table from pos (full chunk-1 path) and sanity-check it ─
    var pos_raw = Tensor.from_view_as_f32(fx.tensor_view("pos"), ctx)  # [1,32,3]
    var pos_flat_shape = List[Int]()
    pos_flat_shape.append(L * 3)
    var pos = reshape(pos_raw, pos_flat_shape^, ctx)
    var tables = build_krea2_rope(pos, axes, cfg.theta, ctx, STDtype.F32)
    # sanity: built table matches the oracle cos/sin (closes the chunk-1 link).
    var cos_ref = Tensor.from_view_as_f32(fx.tensor_view("cos"), ctx).to_host(ctx)
    var sin_ref = Tensor.from_view_as_f32(fx.tensor_view("sin"), ctx).to_host(ctx)
    print("rope cos vs oracle:", ParityHarness(0.999).compare(tables[0], cos_ref, ctx))
    print("rope sin vs oracle:", ParityHarness(0.999).compare(tables[1], sin_ref, ctx))

    # ── Run the attention op with the EXACT weights ──────────────────────────
    var y_out = krea2_attention[L, HEADS, KVHEADS, HEADDIM](
        x, wq, wk, wv, gate_w, wo, qnorm_scale, knorm_scale,
        tables[0], tables[1], Optional[Tensor](None), Optional[Int](None), ctx,
    )
    var res = ParityHarness(0.999).compare(y_out, y_ref, ctx)
    print("krea2_attention parity:", res)
