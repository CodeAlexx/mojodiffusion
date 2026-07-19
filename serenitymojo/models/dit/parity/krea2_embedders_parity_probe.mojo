# models/dit/parity/krea2_embedders_parity_probe.mojo — PARITY GATE for krea2
# chunk 5 (the embedders + input/output heads): temb, tmlp, tproj, txtmlp, first,
# LastLayer — each vs a real ai-toolkit torch oracle.
#
# Oracle: krea2_embedders_oracle.safetensors (weights BF16, scales/biases/te/t/
# vec/outputs F32). Per-piece tensors named with the obvious prefixes (see
# gen_krea2_embedders.py).
#
# Gate: cos >= 0.999 per piece (temb is exact-trig; the MLP heads carry bf16
# matmul + GELU roundoff -> nonzero max_abs, fine).
#
# Run: cd /home/alex/mojodiffusion && \
#   pixi run mojo run -I . serenitymojo/models/dit/parity/krea2_embedders_parity_probe.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.krea2_dit import (
    krea2_temb,
    krea2_tmlp,
    krea2_tproj,
    krea2_txtmlp,
    krea2_first,
    krea2_last_layer,
)


comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_embedders_oracle.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(ORACLE)

    comptime FEATURES = 6144
    comptime TDIM = 256

    # ── temb: t_in [1] F32 -> te [1,1,256] (cos-first, tfactor=1e3, period=1e4) ─
    var t_in = Tensor.from_view_as_f32(fx.tensor_view("t_in"), ctx)
    var te = krea2_temb(t_in, TDIM, ctx, STDtype.BF16)   # bf16, like the oracle dtype
    var te_ref = Tensor.from_view_as_f32(fx.tensor_view("te"), ctx).to_host(ctx)
    print("temb parity:", ParityHarness(0.999).compare(te, te_ref, ctx))

    # ── tmlp: te [1,1,256] bf16 -> t [1,1,6144]. Both Linears have bias. ──────
    var te_in = Tensor.from_view(fx.tensor_view("te"), ctx)   # bf16 [1,1,256]
    var tmlp_w1 = Tensor.from_view(fx.tensor_view("tmlp_w1"), ctx)
    var tmlp_b1 = Tensor.from_view_as_bf16(fx.tensor_view("tmlp_b1"), ctx)
    var tmlp_w2 = Tensor.from_view(fx.tensor_view("tmlp_w2"), ctx)
    var tmlp_b2 = Tensor.from_view_as_bf16(fx.tensor_view("tmlp_b2"), ctx)
    var t_out = krea2_tmlp(te_in, tmlp_w1, tmlp_b1, tmlp_w2, tmlp_b2, ctx)
    var t_ref = Tensor.from_view_as_f32(fx.tensor_view("t"), ctx).to_host(ctx)
    print("tmlp parity:", ParityHarness(0.999).compare(t_out, t_ref, ctx))

    # ── tproj: t [1,1,6144] bf16 -> vec [1,1,36864]. GELU then Linear(+bias). ──
    var t_in2 = Tensor.from_view(fx.tensor_view("t"), ctx)   # bf16 [1,1,6144]
    var tproj_w = Tensor.from_view(fx.tensor_view("tproj_w"), ctx)
    var tproj_b = Tensor.from_view_as_bf16(fx.tensor_view("tproj_b"), ctx)
    var vec_out = krea2_tproj(t_in2, tproj_w, tproj_b, ctx)
    var vec_ref = Tensor.from_view_as_f32(fx.tensor_view("vec"), ctx).to_host(ctx)
    print("tproj parity:", ParityHarness(0.999).compare(vec_out, vec_ref, ctx))

    # ── txtmlp: ctx_in [1,L,2560] bf16 -> [1,L,6144]. RMSNorm + 2 Linears. ────
    var ctx_in = Tensor.from_view(fx.tensor_view("ctx_in"), ctx)   # bf16
    var txt_rms = Tensor.from_view_as_f32(fx.tensor_view("txt_rms_scale"), ctx)
    var txt_w1 = Tensor.from_view(fx.tensor_view("txt_w1"), ctx)
    var txt_b1 = Tensor.from_view_as_bf16(fx.tensor_view("txt_b1"), ctx)
    var txt_w2 = Tensor.from_view(fx.tensor_view("txt_w2"), ctx)
    var txt_b2 = Tensor.from_view_as_bf16(fx.tensor_view("txt_b2"), ctx)
    var txt_out = krea2_txtmlp(ctx_in, txt_rms, txt_w1, txt_b1, txt_w2, txt_b2, ctx)
    var txt_ref = Tensor.from_view_as_f32(fx.tensor_view("txt"), ctx).to_host(ctx)
    print("txtmlp parity:", ParityHarness(0.999).compare(txt_out, txt_ref, ctx))

    # ── first: first_in [1,N,64] bf16 -> [1,N,6144]. Linear(+bias). ──────────
    var first_in = Tensor.from_view(fx.tensor_view("first_in"), ctx)
    var first_w = Tensor.from_view(fx.tensor_view("first_w"), ctx)
    var first_b = Tensor.from_view_as_bf16(fx.tensor_view("first_b"), ctx)
    var first_out = krea2_first(first_in, first_w, first_b, ctx)
    var first_ref = Tensor.from_view_as_f32(fx.tensor_view("first_out"), ctx).to_host(ctx)
    print("first parity:", ParityHarness(0.999).compare(first_out, first_ref, ctx))

    # ── LastLayer: last_x [1,L,6144] + last_tvec [1,1,6144] -> [1,L,64]. ──────
    var last_x = Tensor.from_view(fx.tensor_view("last_x"), ctx)
    # last_tvec was saved F32 (.float()) but it's the bf16 tmlp output; load bf16
    # so the SimpleModulation add runs bf16 (matches the reference's bf16 params).
    var last_tvec = Tensor.from_view_as_bf16(fx.tensor_view("last_tvec"), ctx)  # bf16 [1,1,6144]
    var last_norm = Tensor.from_view_as_f32(fx.tensor_view("last_norm_scale"), ctx)
    var last_mod = Tensor.from_view_as_bf16(fx.tensor_view("last_mod_lin"), ctx)  # [2,6144] bf16
    var last_lw = Tensor.from_view(fx.tensor_view("last_lin_w"), ctx)
    var last_lb = Tensor.from_view_as_bf16(fx.tensor_view("last_lin_b"), ctx)
    var last_out = krea2_last_layer(
        last_x, last_tvec, last_norm, last_mod, last_lw, last_lb, FEATURES, ctx
    )
    var last_ref = Tensor.from_view_as_f32(fx.tensor_view("last_out"), ctx).to_host(ctx)
    print("lastlayer parity:", ParityHarness(0.999).compare(last_out, last_ref, ctx))
