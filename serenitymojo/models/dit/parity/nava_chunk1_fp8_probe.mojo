# NAVA chunk 1: Mojo fp8 per-row dequant of a NAVA Linear vs torch reference.
# Reuses the Ideogram fp8 path (ops/fp8) — confirms the same scheme on NAVA weights.
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_perrow_to_bf16

comptime NAVA = "/home/alex/.serenity/models/checkpoints/NAVA/NAVA_fp8.safetensors"
comptime FX = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/nava_fx_chunk1.safetensors"


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(NAVA)
    var key = String("backbone.double_blocks.0.cross_attn.k.weight")
    var wi = st.tensor_info(key)
    var w = Tensor.from_view_raw(from_parts(wi.dtype, wi.shape.copy(), st.tensor_bytes(key)), ctx)
    var si = st.tensor_info(key + "_scale")
    var scale = Tensor.from_view_as_f32(from_parts(si.dtype, si.shape.copy(), st.tensor_bytes(key + "_scale")), ctx)
    var deq = fp8_e4m3_dequant_perrow_to_bf16(w, scale, ctx)
    var fx = ShardedSafeTensors.open(FX)
    var ref_host = Tensor.from_view(fx.tensor_view("deq_bf16"), ctx).to_host(ctx)
    print("NAVA fp8 dequant vs torch:", ParityHarness(0.999).compare(deq, ref_host, ctx))
