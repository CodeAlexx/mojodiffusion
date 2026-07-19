# models/dit/parity/krea2_block_parity_probe.mojo — PARITY GATE for krea2 chunk 4
# (the SingleStreamBlock: DoubleSharedModulation -> 2 AdaLN-Zero gated residual
# branches over chunk-3 Attention + chunk-2 SwiGLU) vs a real ai-toolkit torch
# oracle.
#
# Oracle: krea2_block_oracle.safetensors (weights BF16, scales/x-extras F32):
#   x [1,32,6144], vec [1,36864], mod_lin [36864] F32, prenorm/postnorm_scale [6144] F32,
#   wq/gate_w/wo [6144,6144], wk/wv [1536,6144], qnorm/knorm_scale [128] F32,
#   mlp_gate_w/up_w [16384,6144], mlp_down_w [6144,16384],
#   pos [1,32,3] F32, cos/sin [32,64] F32, y [1,32,6144] F32 (block output).
#
# Gate: cos >= 0.999 vs y (bf16 matmul + SDPA roundoff over the whole block).
#
# Run: cd /home/alex/mojodiffusion && \
#   pixi run mojo run -I . serenitymojo/models/dit/parity/krea2_block_parity_probe.mojo
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
    krea2_single_stream_block,
)


comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_block_oracle.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(ORACLE)

    comptime L = 32
    comptime HEADS = 48
    comptime KVHEADS = 12
    comptime HEADDIM = 128

    var cfg = Krea2Config.default()
    var axes = cfg.rope_axes()

    # ── Load oracle tensors. Weights/x/vec/mod_lin are bf16-stored (prod). ────
    var x = Tensor.from_view(fx.tensor_view("x"), ctx)               # bf16 [1,32,6144]
    var vec = Tensor.from_view(fx.tensor_view("vec"), ctx)           # bf16 [1,36864]
    var mod_lin = Tensor.from_view_as_bf16(fx.tensor_view("mod_lin"), ctx)  # bf16 [36864]
    var prenorm_scale = Tensor.from_view_as_f32(fx.tensor_view("prenorm_scale"), ctx)
    var postnorm_scale = Tensor.from_view_as_f32(fx.tensor_view("postnorm_scale"), ctx)
    var wq = Tensor.from_view(fx.tensor_view("wq"), ctx)
    var wk = Tensor.from_view(fx.tensor_view("wk"), ctx)
    var wv = Tensor.from_view(fx.tensor_view("wv"), ctx)
    var gate_w = Tensor.from_view(fx.tensor_view("gate_w"), ctx)
    var wo = Tensor.from_view(fx.tensor_view("wo"), ctx)
    var qnorm_scale = Tensor.from_view_as_f32(fx.tensor_view("qnorm_scale"), ctx)
    var knorm_scale = Tensor.from_view_as_f32(fx.tensor_view("knorm_scale"), ctx)
    var mlp_gate_w = Tensor.from_view(fx.tensor_view("mlp_gate_w"), ctx)
    var mlp_up_w = Tensor.from_view(fx.tensor_view("mlp_up_w"), ctx)
    var mlp_down_w = Tensor.from_view(fx.tensor_view("mlp_down_w"), ctx)
    var y_ref = Tensor.from_view_as_f32(fx.tensor_view("y"), ctx).to_host(ctx)

    # NOTE: the torch SingleStreamBlock keeps mod.lin in bf16 (the .to(DT) cast in
    # the gen). vec is bf16. So mod(vec) = vec + lin runs bf16 -> bf16 chunks,
    # which is what krea2_double_shared_modulation does (bf16 add). Loading mod_lin
    # as bf16 (not F32) matches the reference's dtype exactly.

    # ── Build the RoPE table from pos (chunk-1 path). ────────────────────────
    var pos_raw = Tensor.from_view_as_f32(fx.tensor_view("pos"), ctx)  # [1,32,3]
    var pos_flat_shape = List[Int]()
    pos_flat_shape.append(L * 3)
    var pos = reshape(pos_raw, pos_flat_shape^, ctx)
    var tables = build_krea2_rope(pos, axes, cfg.theta, ctx, STDtype.F32)

    # ── Run the block with the EXACT weights. ────────────────────────────────
    var y_out = krea2_single_stream_block[L, HEADS, KVHEADS, HEADDIM](
        x, vec, mod_lin, prenorm_scale, postnorm_scale,
        wq, wk, wv, gate_w, wo, qnorm_scale, knorm_scale,
        mlp_gate_w, mlp_up_w, mlp_down_w,
        tables[0], tables[1], Optional[Tensor](None), Optional[Int](None), ctx,
    )
    var res = ParityHarness(0.999).compare(y_out, y_ref, ctx)
    print("krea2_single_stream_block parity:", res)
