# models/dit/parity/krea2_blockops_parity_probe.mojo — PARITY GATE for krea2
# chunk 2 leaf ops (RMSNorm + SwiGLU) vs a real ai-toolkit torch oracle, plus
# compile-verified numeric self-checks for the two Modulations (no torch needed —
# they are vec + param then chunk).
#
# Oracle: krea2_blockops_oracle.safetensors (F32, generated bf16-on-cuda):
#   rms_x [17,6144], rms_scale [6144] (RAW scale), rms_y [17,6144]
#   swi_x [17,6144], swi_gate_w/up_w [16384,6144], swi_down_w [6144,16384], swi_y [17,6144]
#
# Gate: cos >= 0.999 for rmsnorm + swiglu (bf16 matmul/trig -> expect ~0.999+).
#
# Run: cd /home/alex/mojodiffusion && \
#   pixi run mojo run -I . serenitymojo/models/dit/parity/krea2_blockops_parity_probe.mojo
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.krea2_dit import (
    krea2_rmsnorm,
    krea2_swiglu,
    krea2_simple_modulation,
    krea2_double_shared_modulation,
)


comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/krea2_blockops_oracle.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var fx = ShardedSafeTensors.open(ORACLE)

    # ── RMSNorm: x bf16 (production), scale F32 (probe adds +1 internally) ────
    var rms_x = Tensor.from_view_as_bf16(fx.tensor_view("rms_x"), ctx)
    var rms_scale = Tensor.from_view_as_f32(fx.tensor_view("rms_scale"), ctx)
    var rms_y_ref = Tensor.from_view_as_f32(fx.tensor_view("rms_y"), ctx).to_host(ctx)
    var rms_out = krea2_rmsnorm(rms_x, rms_scale, Float32(1.0e-5), ctx)
    var rms_res = ParityHarness(0.999).compare(rms_out, rms_y_ref, ctx)
    print("rmsnorm parity:", rms_res)

    # ── SwiGLU: x + weights bf16 (production matmul) ─────────────────────────
    var swi_x = Tensor.from_view_as_bf16(fx.tensor_view("swi_x"), ctx)
    var swi_gate_w = Tensor.from_view_as_bf16(fx.tensor_view("swi_gate_w"), ctx)
    var swi_up_w = Tensor.from_view_as_bf16(fx.tensor_view("swi_up_w"), ctx)
    var swi_down_w = Tensor.from_view_as_bf16(fx.tensor_view("swi_down_w"), ctx)
    var swi_y_ref = Tensor.from_view_as_f32(fx.tensor_view("swi_y"), ctx).to_host(ctx)
    var swi_out = krea2_swiglu(swi_x, swi_gate_w, swi_up_w, swi_down_w, ctx)
    var swi_res = ParityHarness(0.999).compare(swi_out, swi_y_ref, ctx)
    print("swiglu parity:", swi_res)

    # ── SimpleModulation self-check (vec + lin[None], chunk(2,dim=1)) ─────────
    # vec [1, dim], lin [2, dim]. scale = vec + lin[0], shift = vec + lin[1].
    comptime DIM = 8
    var vec_host = List[Float32]()
    for i in range(DIM):
        vec_host.append(Float32(i) * Float32(0.5) - Float32(1.0))
    var lin_host = List[Float32]()
    for r in range(2):
        for i in range(DIM):
            lin_host.append(Float32(r) * Float32(10.0) + Float32(i) * Float32(0.1))
    var vec = Tensor.from_host(vec_host.copy(), [1, DIM], STDtype.F32, ctx)
    var lin = Tensor.from_host(lin_host.copy(), [2, DIM], STDtype.F32, ctx)
    var sm = krea2_simple_modulation(vec, lin, ctx)
    var scale_host = sm[0].to_host(ctx)   # [1,1,dim]
    var shift_host = sm[1].to_host(ctx)   # [1,1,dim]
    var sm_err: Float32 = 0.0
    for i in range(DIM):
        var ref_scale = vec_host[i] + lin_host[0 * DIM + i]
        var ref_shift = vec_host[i] + lin_host[1 * DIM + i]
        sm_err = max(sm_err, abs(scale_host[i] - ref_scale))
        sm_err = max(sm_err, abs(shift_host[i] - ref_shift))
    print("simple_modulation self-check max abs err =", sm_err,
          " scale[0..2]=", scale_host[0], scale_host[1], scale_host[2],
          " shift[0..2]=", shift_host[0], shift_host[1], shift_host[2])
    if sm_err > Float32(1.0e-5):
        raise Error("simple_modulation self-check FAILED")

    # ── DoubleSharedModulation self-check (vec + lin, chunk(6,dim=-1)) ────────
    # vec [1, 6*dim], lin [6*dim]. chunk_i = vec[i*dim:(i+1)*dim] + lin[...].
    var vec6_host = List[Float32]()
    for i in range(6 * DIM):
        vec6_host.append(Float32(i) * Float32(0.25) - Float32(2.0))
    var lin6_host = List[Float32]()
    for i in range(6 * DIM):
        lin6_host.append(Float32(i) * Float32(0.05) + Float32(0.3))
    var vec6 = Tensor.from_host(vec6_host.copy(), [1, 6 * DIM], STDtype.F32, ctx)
    var lin6 = Tensor.from_host(lin6_host.copy(), [6 * DIM], STDtype.F32, ctx)
    var dm = krea2_double_shared_modulation(vec6, lin6, ctx)
    var dm_err: Float32 = 0.0
    # 6 chunks: prescale, preshift, pregate, postscale, postshift, postgate.
    var c0 = dm[0].to_host(ctx)
    var c1 = dm[1].to_host(ctx)
    var c2 = dm[2].to_host(ctx)
    var c3 = dm[3].to_host(ctx)
    var c4 = dm[4].to_host(ctx)
    var c5 = dm[5].to_host(ctx)
    for i in range(DIM):
        dm_err = max(dm_err, abs(c0[i] - (vec6_host[0 * DIM + i] + lin6_host[0 * DIM + i])))
        dm_err = max(dm_err, abs(c1[i] - (vec6_host[1 * DIM + i] + lin6_host[1 * DIM + i])))
        dm_err = max(dm_err, abs(c2[i] - (vec6_host[2 * DIM + i] + lin6_host[2 * DIM + i])))
        dm_err = max(dm_err, abs(c3[i] - (vec6_host[3 * DIM + i] + lin6_host[3 * DIM + i])))
        dm_err = max(dm_err, abs(c4[i] - (vec6_host[4 * DIM + i] + lin6_host[4 * DIM + i])))
        dm_err = max(dm_err, abs(c5[i] - (vec6_host[5 * DIM + i] + lin6_host[5 * DIM + i])))
    print("double_shared_modulation self-check max abs err =", dm_err,
          " prescale[0]=", c0[0], " postgate[0]=", c5[0])
    if dm_err > Float32(1.0e-5):
        raise Error("double_shared_modulation self-check FAILED")

    print("CHUNK2 PROBE OK")
