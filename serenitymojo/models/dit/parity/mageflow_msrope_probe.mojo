# mageflow_msrope_probe.mojo — MageFlow msrope (multi-axis RoPE) parity gate.
#
# Proves which serenitymojo rope op(s) MageFlow's msrope needs. MageFlow's
# rope_type "msrope" == Qwen-Image QwenEmbedRope VERBATIM: axes_dim [16,56,56]
# (sum=128=head_dim), theta 10000, scale_rope=True (native-res centering),
# applied to IMAGE tokens only. The apply is
#   apply_rotary_emb_mageflow: view_as_complex(x.reshape(...,-1,2)) * freqs
# == adjacent-pair complex == INTERLEAVED (serenitymojo rope_interleaved).
# The cos/sin table (freqs.real/.imag, concat over axes) is exactly what
# build_multiaxis_rope_tables([16,56,56], theta=10000) produces given the
# scale_rope-centered per-axis positions (frame idx; h-,w- centered).
#
# Grid: frame=1, h=4, w=4 -> 16 img tokens, heads=24, head_dim=128.
# Rows = 16*24 = 384 (rope broadcast across heads, token-major then head).
# Positions come straight from the oracle (pos_rows) so the table gate also
# validates the native-res centering.
#
# Compile+run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo run -I . serenitymojo/models/dit/parity/mageflow_msrope_probe.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.rope_tables import build_multiaxis_rope_tables
from serenitymojo.ops.rope import rope_interleaved

comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/mageflow_fx_msrope.safetensors"
comptime FRAME = 1
comptime HEIGHT = 4
comptime WIDTH = 4
comptime HEADS = 24
comptime HEAD_DIM = 128
comptime S_IMG = FRAME * HEIGHT * WIDTH   # 16
comptime ROWS = S_IMG * HEADS             # 384
comptime HALF = 64


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(FX)

    # positions [ROWS, 3] F32 from the oracle -> flat [ROWS*3] token-major.
    var positions = Tensor.from_view(fx.tensor_view("pos_rows"), ctx)
    if positions.numel() != ROWS * 3:
        raise Error("pos_rows numel mismatch")

    var axes = [16, 56, 56]  # sum = 128 = head_dim
    var cs = build_multiaxis_rope_tables(
        positions, axes, Float32(10000.0), ctx, STDtype.F32
    )
    var sh = cs[0].shape()
    if len(sh) != 2 or sh[0] != ROWS or sh[1] != HALF:
        raise Error("msrope table shape mismatch")

    # ── table gate ───────────────────────────────────────────────────────
    var cos_ref = Tensor.from_view(fx.tensor_view("cos_rows"), ctx).to_host(ctx)
    var sin_ref = Tensor.from_view(fx.tensor_view("sin_rows"), ctx).to_host(ctx)
    var ht = ParityHarness(0.999)
    var cos_res = ht.compare(cs[0], cos_ref, ctx)
    var sin_res = ht.compare(cs[1], sin_ref, ctx)
    print("msrope cos table :", cos_res)
    print("msrope sin table :", sin_res)

    # ── apply gate (INTERLEAVED) ─────────────────────────────────────────
    var q = Tensor.from_view(fx.tensor_view("q"), ctx)   # [ROWS,128] F32
    var k = Tensor.from_view(fx.tensor_view("k"), ctx)
    var q_out = rope_interleaved(q, cs[0], cs[1], ctx)
    var k_out = rope_interleaved(k, cs[0], cs[1], ctx)
    var q_ref = Tensor.from_view(fx.tensor_view("q_out"), ctx).to_host(ctx)
    var k_ref = Tensor.from_view(fx.tensor_view("k_out"), ctx).to_host(ctx)
    var ha = ParityHarness(0.999)
    var q_res = ha.compare(q_out, q_ref, ctx)
    var k_res = ha.compare(k_out, k_ref, ctx)
    print("msrope applied-q :", q_res)
    print("msrope applied-k :", k_res)

    if not (cos_res.passed and sin_res.passed and q_res.passed and k_res.passed):
        raise Error("FAIL: MageFlow msrope parity below 0.999")
    print(
        "PASS: MageFlow msrope == build_multiaxis_rope_tables([16,56,56],",
        "theta=10000, scale_rope-centered positions) + rope_interleaved",
    )
