# models/krea2/parity/krea2_fp8_resident_gate.mojo — fp8-resident round-trip cos
# on a REAL krea2 block weight (the T2.B "different-trajectory numerics" sanity).
#
# Loads block 0's 8 matmul weights from the REAL raw.safetensors, quantizes them
# ONCE through the resident builder path (fp8_e4m3_rowscale + fp8_e4m3_encode_perrow,
# the load-once step), dequants them back to bf16 (fp8_e4m3_dequant_perrow_to_bf16,
# the per-block step), and reports cos(deq, bf16-original) per weight. fp8 e4m3 is
# lossy (3 mantissa bits) → expect ~0.99+ (NOT bit-exact — this documents the new
# numerics class). This proves the resident store reconstructs real krea2 weights
# faithfully, on actual checkpoint data (the encode/decode KERNELS are separately
# gated bit-for-bit by ops/tests/fp8_quant_smoke.mojo on synthetic data).
#
# Build (ORCHESTRATOR runs the GPU smoke):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -L.pixi/envs/default/lib -Xlinker -lsqlite3 \
#     serenitymojo/models/krea2/parity/krea2_fp8_resident_gate.mojo \
#     -o /tmp/krea2_fp8_resident_gate && \
#   LD_LIBRARY_PATH=/home/alex/mojodiffusion/.pixi/envs/default/lib \
#     /tmp/krea2_fp8_resident_gate
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.fp8_quant import fp8_e4m3_rowscale, fp8_e4m3_encode_perrow
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16
from serenitymojo.models.krea2.config import krea2_raw


# cos(a, b) over two host F32 lists (the standard numeric-parity metric).
def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


# round-trip one real weight: bf16 view → rowscale+encode (load-once) → dequant
# (per-block) → cos vs the bf16 original.
def _roundtrip_cos(
    st: ShardedSafeTensors, key: String, ctx: DeviceContext
) raises -> Float64:
    var w_bf = Tensor.from_view_as_bf16(st.tensor_view(key), ctx)   # [out,in] BF16
    var ref_h = w_bf.to_host(ctx)
    var scale = fp8_e4m3_rowscale(w_bf, ctx)                        # F32 [out]
    var bytes = fp8_e4m3_encode_perrow(w_bf, scale, ctx)           # E4M3 [out,in]
    var deq = fp8_e4m3_dequant_perrow_to_bf16(bytes, scale, ctx)   # BF16 [out,in]
    var deq_h = deq.to_host(ctx)
    return _cos(ref_h, deq_h)


def main() raises:
    var ctx = DeviceContext()
    var cfg = krea2_raw()
    var st = ShardedSafeTensors.open(cfg.checkpoint)
    var p = String("blocks.0.")

    var keys = List[String]()
    keys.append(p + "attn.wq.weight")
    keys.append(p + "attn.wk.weight")
    keys.append(p + "attn.wv.weight")
    keys.append(p + "attn.gate.weight")
    keys.append(p + "attn.wo.weight")
    keys.append(p + "mlp.gate.weight")
    keys.append(p + "mlp.up.weight")
    keys.append(p + "mlp.down.weight")

    print("==== krea2 fp8-resident round-trip cos (REAL block 0 weights) ====")
    print("checkpoint=", cfg.checkpoint)
    var min_cos = Float64(2.0)
    for ki in range(len(keys)):
        var c = _roundtrip_cos(st, keys[ki], ctx)
        print("  ", keys[ki], "  cos(deq, bf16) =", c)
        if c < min_cos:
            min_cos = c

    print("min cos over the 8 matmul weights =", min_cos)
    # fp8 e4m3 is lossy; the resident base is faithful if cos >= 0.99 (documents
    # the different-trajectory numerics class — NOT a bit-exact claim).
    if min_cos >= 0.99:
        print("PASS: fp8-resident round-trip faithful (cos>=0.99, lossy as expected)")
    else:
        print("FAIL: fp8-resident round-trip cos", min_cos, "< 0.99")
        raise Error("krea2_fp8_resident_gate failed")
