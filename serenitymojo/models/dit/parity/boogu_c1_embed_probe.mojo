# boogu_c1_embed_probe.mojo — compile+run probe for Boogu-Image DiT embedders (C1).
#
# Builds SYNTHETIC weights + inputs of the correct shapes (the real 3-shard
# checkpoint may still be downloading — this probe does NOT depend on it) and
# runs BOTH embedder forwards (x_embed + time_caption_embed), printing output
# shapes and std. The orchestrator owns parity vs the torch oracle; this probe
# only proves the Mojo code COMPILES and EXECUTES (exit 0).
#
# Config: hidden_size=3360, x_embed_in=64, instruction_feat_dim=4096,
# frequency_embedding_size=256, time_embed_dim=1024.
#
# Run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg \
#     && pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c1_embed_probe.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.models.dit.boogu_dit import BooguEmbedders


def _std(host: List[Float32]) -> Float32:
    var n = len(host)
    if n == 0:
        return Float32(0.0)
    var mean = Float32(0.0)
    for i in range(n):
        mean += host[i]
    mean /= Float32(n)
    var var_acc = Float32(0.0)
    for i in range(n):
        var d = host[i] - mean
        var_acc += d * d
    var_acc /= Float32(n)
    return sqrt(var_acc)


def main() raises:
    var ctx = DeviceContext()

    # ── synthetic weights (BF16, correct shapes; small scale to keep std sane) ─
    # x_embedder: Linear(64 -> 3360, bias=True).
    var x_embedder_weight = randn([3360, 64], UInt64(1), STDtype.BF16, ctx)
    var x_embedder_bias = randn([3360], UInt64(2), STDtype.BF16, ctx)
    # timestep_embedder: TimestepEmbedding(256 -> 1024).
    var ts_linear_1_weight = randn([1024, 256], UInt64(3), STDtype.BF16, ctx)
    var ts_linear_1_bias = randn([1024], UInt64(4), STDtype.BF16, ctx)
    var ts_linear_2_weight = randn([1024, 1024], UInt64(5), STDtype.BF16, ctx)
    var ts_linear_2_bias = randn([1024], UInt64(6), STDtype.BF16, ctx)
    # caption_embedder: Sequential(RMSNorm(4096), Linear(4096 -> 3360, bias=True)).
    # RMSNorm gamma initialized to ones in the reference; synthetic ~1.0 here via
    # |randn|+1 would change std, but a plain randn gamma still exercises the path.
    var caption_norm_weight = randn([4096], UInt64(7), STDtype.BF16, ctx)
    var caption_linear_weight = randn([3360, 4096], UInt64(8), STDtype.BF16, ctx)
    var caption_linear_bias = randn([3360], UInt64(9), STDtype.BF16, ctx)

    var emb = BooguEmbedders(
        x_embedder_weight^,
        x_embedder_bias^,
        ts_linear_1_weight^,
        ts_linear_1_bias^,
        ts_linear_2_weight^,
        ts_linear_2_bias^,
        caption_norm_weight^,
        caption_linear_weight^,
        caption_linear_bias^,
    )

    # ── synthetic inputs ─────────────────────────────────────────────────────
    # tokens: already-patchified image tokens [1, 256, 64] BF16.
    var tokens = randn([1, 256, 64], UInt64(100), STDtype.BF16, ctx)
    # instruction_feats: [1, 16, 4096] BF16 (Qwen instruction features).
    var instruction_feats = randn([1, 16, 4096], UInt64(101), STDtype.BF16, ctx)
    # timestep: [1] F32 (a single fractional flow timestep in [0,1)).
    var timestep = Tensor.from_host([Float32(0.5)], [1], STDtype.F32, ctx)

    # ── x_embed forward ──────────────────────────────────────────────────────
    var x_out = emb.x_embed(tokens, ctx)
    var xs = x_out.shape()
    print("x_embed out shape:", xs[0], xs[1], xs[2])
    var x_host = x_out.to_host(ctx)
    print("x_embed std:", _std(x_host))

    # ── time_caption_embed forward ───────────────────────────────────────────
    var tc = emb.time_caption_embed(timestep, instruction_feats, ctx)
    var ts = tc[0].shape()
    print("time_embed shape:", ts[0], ts[1])
    var t_host = tc[0].to_host(ctx)
    print("time_embed std:", _std(t_host))
    var cs = tc[1].shape()
    print("caption_embed shape:", cs[0], cs[1], cs[2])
    var c_host = tc[1].to_host(ctx)
    print("caption_embed std:", _std(c_host))

    print("boogu_c1_embed_probe OK")
