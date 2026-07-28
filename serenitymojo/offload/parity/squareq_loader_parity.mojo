# offload/parity/squareq_loader_parity.mojo — chunk-3 gate: the
# TurboPlannedLoader squareq_w4-resident path is a faithful weight source.
#
# Flow (real Klein-4B checkpoint + real built slab):
#   1. open loader (klein 5 double + 20 single plan) on the BASE checkpoint
#   2. pin_residents_squareq(slab_dir, budget) — pins packed sidecar VERBATIM
#   3. await_block(0) (double 0) and await_block(5) (single 0): the squareq
#      fast path reconstructs BF16 per block
#   4. compare reconstructed weights vs the PYTHON-ORACLE W_hat from the SAME
#      slab bytes (loader_expected.safetensors):        cos >= 0.9999
#      and vs the streamed bf16 original (quant cos recorded, floor 0.99)
#   5. resident bytes reported vs squareq-plan.json bytes (human-checked)
#
# Build:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     -Xlinker -lcuda serenitymojo/offload/parity/squareq_loader_parity.mojo \
#     -o output/checks/squareq_loader_parity
# Run: LD_LIBRARY_PATH=.pixi/envs/default/lib output/checks/squareq_loader_parity

from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.offload.plan import build_klein_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

comptime CKPT = "/home/alex/mojodiffusion/models/klein4b/transformer.safetensors"
comptime SLAB = "/home/alex/mojodiffusion/models/klein4b/squareq_w4_r32"
comptime BUDGET = 6 * 1024 * 1024 * 1024
comptime N_DOUBLE = 5
comptime N_SINGLE = 20


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos: length mismatch " + String(len(a)) + " vs " + String(len(b)))
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("cos: zero norm")
    return dot / (sqrt(na) * sqrt(nb))


def _gate(name: String, cos: Float64, bar: Float64) raises:
    if cos >= bar:
        print("[sq-loader] PASS ", name, " cos=", cos, " (bar ", bar, ")")
    else:
        print("[sq-loader] FAIL ", name, " cos=", cos, " < ", bar)
        raise Error(String("squareq loader parity FAILED: ") + name)


def main() raises:
    var ctx = DeviceContext()
    var plan = build_klein_block_plan(N_DOUBLE, N_SINGLE)
    var loader = TurboPlannedLoader.open(
        String(CKPT), plan^, OffloadConfig.synchronous_single(), ctx,
        fill_block_store=False,
    )
    var pinned = loader.pin_residents_squareq(String(SLAB), BUDGET, ctx)
    if pinned != N_DOUBLE + N_SINGLE:
        raise Error(
            String("pinned ") + String(pinned) + " of "
            + String(N_DOUBLE + N_SINGLE) + " blocks"
        )
    ctx.synchronize()

    var expected = ShardedSafeTensors.open(
        String(SLAB) + String("/loader_expected.safetensors")
    )
    var base_st = ShardedSafeTensors.open(String(CKPT))

    # double block 0: img_attn.qkv
    var h0 = loader.await_block(0, ctx)
    var name0 = String("double_blocks.0.img_attn.qkv.weight")
    var w0 = h0.block[name0]
    var ref0 = Tensor.from_view(
        expected.tensor_view(String("double_blocks.0.img_attn.qkv.w_hat")), ctx
    )
    _gate(String("double0.qkv vs oracle W_hat"), _cos(w0[].to_host(ctx), ref0.to_host(ctx)), 0.9999)
    var orig0 = Tensor.from_view(base_st.tensor_view(name0), ctx)
    var qc0 = _cos(w0[].to_host(ctx), orig0.to_host(ctx))
    print("[sq-loader] double0.qkv vs bf16 original (quant cos) =", qc0)
    _gate(String("double0.qkv quant floor"), qc0, 0.99)

    # single block 0 (plan index N_DOUBLE): linear2
    var h5 = loader.await_block(N_DOUBLE, ctx)
    var name5 = String("single_blocks.0.linear2.weight")
    var w5 = h5.block[name5]
    var ref5 = Tensor.from_view(
        expected.tensor_view(String("single_blocks.0.linear2.w_hat")), ctx
    )
    _gate(String("single0.linear2 vs oracle W_hat"), _cos(w5[].to_host(ctx), ref5.to_host(ctx)), 0.9999)
    var orig5 = Tensor.from_view(base_st.tensor_view(name5), ctx)
    var qc5 = _cos(w5[].to_host(ctx), orig5.to_host(ctx))
    print("[sq-loader] single0.linear2 vs bf16 original (quant cos) =", qc5)
    _gate(String("single0.linear2 quant floor"), qc5, 0.99)

    # small-tensor passthrough must be exact
    var nname = String("double_blocks.0.img_attn.norm.query_norm.scale")
    var wn = h0.block[nname]
    var orign = Tensor.from_view(base_st.tensor_view(nname), ctx)
    _gate(String("double0.qnorm passthrough"), _cos(wn[].to_host(ctx), orign.to_host(ctx)), 0.999999)

    print("[sq-loader] ALL PASS")
