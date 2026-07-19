# models/nldiffusion/parity/image_head_probe.mojo — CHUNK C gate.
#
# Gates each of the 4 image-head pieces INDEPENDENTLY against oracle1.safetensors
# (all F32 in-file), loading bf16 weights from the 6-shard model. Reference bf16
# compute is mirrored (inputs cast to bf16). Gate cos >= 0.999.
#
#   gen_embedding : ids gen_embedding.in[1,4096] -> [1,4096,4096] vs .out
#   downsample_gen: downsample_gen.in[1,4096,4096] (64,64) -> [1,1024,4096] vs .out
#   upsample_gen  : upsample_gen.in[1,1024,4096] (32,32) -> [1,4096,4096] vs .out
#   gen_predictor : gen_predictor.in[4096,4096] first128 rows -> [128,131072] vs .out
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/nldiffusion/parity/image_head_probe.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import slice
from serenitymojo.parity import ParityHarness
from serenitymojo.models.nldiffusion.image_head import (
    nldiff_gen_embedding,
    nldiff_downsample_gen,
    nldiff_upsample_gen,
    nldiff_gen_predictor,
)

comptime MODEL_DIR = "/mnt/disk1/models/NL-Diffusion-Image"
comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"


def _load_oracle_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _l2(xs: List[Float32]) -> Float64:
    var acc: Float64 = 0.0
    for i in range(len(xs)):
        var v = Float64(xs[i])
        acc += v * v
    return sqrt(acc)


def _report(name: String, out_host: List[Float32], ref_host: List[Float32]) raises:
    var harness = ParityHarness(0.999)
    var res = harness.compare_host(out_host, ref_host)
    var ratio = _l2(out_host) / _l2(ref_host) if _l2(ref_host) != 0.0 else 0.0
    print(
        "[chunk C]",
        name,
        ": cos =",
        res.cos,
        " max_abs =",
        res.max_abs,
        " |mine|/|ref| =",
        ratio,
    )
    if res.passed:
        print("[chunk C]", name, "GATE PASS (cos >= 0.999)")
    else:
        print("[chunk C]", name, "GATE FAIL (cos < 0.999)")


def main() raises:
    var ctx = DeviceContext()
    var model = ShardedSafeTensors.open(MODEL_DIR)
    var oracle = ShardedSafeTensors.open(
        String(PARITY_DIR) + "/oracle1.safetensors"
    )

    # ── piece 1: gen_embedding ────────────────────────────────────────────────
    print("[chunk C] gen_embedding: loading ids (gen_embedding.in [1,4096])")
    var ids_f32 = _load_oracle_f32(oracle, "gen_embedding.in", ctx).to_host(ctx)
    var ids = List[Int]()
    for i in range(len(ids_f32)):
        ids.append(Int(ids_f32[i] + 0.5))  # exact int stored as f32
    var emb = nldiff_gen_embedding(ids, model, ctx)  # [1,4096,4096] bf16
    var emb_host = emb.to_host(ctx)
    var emb_ref = _load_oracle_f32(oracle, "gen_embedding.out", ctx).to_host(ctx)
    _report("gen_embedding", emb_host, emb_ref)
    _ = emb^

    # ── piece 2: downsample_gen ───────────────────────────────────────────────
    print("[chunk C] downsample_gen: loading downsample_gen.in [1,4096,4096]")
    var ds_in_f32 = _load_oracle_f32(oracle, "downsample_gen.in", ctx)
    var ds_in = cast_tensor(ds_in_f32, STDtype.BF16, ctx)
    var ds_cw = _load_oracle_f32(
        model, "encoder.downsample_gen.downsample.conv.weight", ctx
    )  # bf16 stored -> from_view keeps bf16
    var ds_nw = _load_oracle_f32(
        model, "encoder.downsample_gen.downsample.norm.weight", ctx
    )
    var ds_out = nldiff_downsample_gen[64, 64, 4096](ds_in, ds_cw, ds_nw, ctx)
    var ds_host = ds_out.to_host(ctx)
    var ds_ref = _load_oracle_f32(oracle, "downsample_gen.out", ctx).to_host(ctx)
    _report("downsample_gen", ds_host, ds_ref)
    _ = ds_out^

    # ── piece 3: upsample_gen ─────────────────────────────────────────────────
    print("[chunk C] upsample_gen: loading upsample_gen.in [1,1024,4096]")
    var us_in_f32 = _load_oracle_f32(oracle, "upsample_gen.in", ctx)
    var us_in = cast_tensor(us_in_f32, STDtype.BF16, ctx)
    var us_cw = _load_oracle_f32(
        model, "encoder.upsample_gen.upsample.conv.weight", ctx
    )
    var us_nw = _load_oracle_f32(
        model, "encoder.upsample_gen.upsample.norm.weight", ctx
    )
    var us_out = nldiff_upsample_gen[32, 32](us_in, us_cw, us_nw, ctx)
    var us_host = us_out.to_host(ctx)
    var us_ref = _load_oracle_f32(oracle, "upsample_gen.out", ctx).to_host(ctx)
    _report("upsample_gen", us_host, us_ref)
    _ = us_out^

    # ── piece 4: gen_predictor (first 128 rows) ───────────────────────────────
    print("[chunk C] gen_predictor: loading gen_predictor.in [4096,4096]")
    var gp_in_f32 = _load_oracle_f32(oracle, "gen_predictor.in", ctx)
    var gp_in = cast_tensor(gp_in_f32, STDtype.BF16, ctx)  # [4096,4096]
    var gp_in128 = slice(gp_in, 0, 0, 128, ctx)  # [128,4096]
    var gp_w = _load_oracle_f32(model, "encoder.gen_predictor.weight", ctx)
    var gp_out = nldiff_gen_predictor(gp_in128, gp_w, ctx)  # [128,131072]
    var gp_host = gp_out.to_host(ctx)
    var gp_ref = _load_oracle_f32(oracle, "gen_predictor.out", ctx).to_host(ctx)
    _report("gen_predictor", gp_host, gp_ref)
    _ = gp_out^
