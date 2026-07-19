# pipeline/cosmos_dit_probe.mojo — compile + smoke probe for cosmos_predict25_dit.
# CHUNK A: per-axis rope tables + one block. CHUNK B: full-stack symbol reference.
# Tiny grid so it runs on any GPU without weights present (block/full-stack paths
# are exercised symbolically by the type checker; the rope path runs for real).

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.cosmos_predict25_dit import (
    CosmosConfig,
    cosmos_rope_axes,
    cosmos_rope_thetas,
    cosmos_build_rope,
    cosmos_block_forward,
    CosmosPredict25Dit,
)


def main() raises:
    var ctx = DeviceContext()
    var cfg = CosmosConfig.v2_2b_production()

    # ── CHUNK A.1: per-axis NTK theta + axis split ──
    var axes = cosmos_rope_axes(cfg.head_dim)  # [44,42,42]
    print("rope axes (t,h,w):", axes[0], axes[1], axes[2])
    var thetas = cosmos_rope_thetas(cfg.head_dim, cfg.rope_t_ratio, cfg.rope_h_ratio, cfg.rope_w_ratio)
    print("rope thetas (t,h,w):", thetas[0], thetas[1], thetas[2])

    # ── CHUNK A.2: build rope tables for a tiny grid ──
    comptime TP = 1
    comptime HP = 2
    comptime WP = 2
    comptime N = TP * HP * WP
    comptime H = 16
    comptime DH = 128
    var cs = cosmos_build_rope(TP, HP, WP, cfg, STDtype.BF16, ctx)
    print("rope cos shape:", cs[0].shape()[0], cs[0].shape()[1])  # [N, 64]

    # ── CHUNK A.3: one block forward (random weights) ──
    var D = cfg.model_channels
    comptime TXT = 8
    var bw = Dict[String, ArcPointer[Tensor]]()
    _put(bw, "adaln_modulation_self_attn.1.weight", cfg.adaln_lora_dim, D, ctx)
    _put(bw, "adaln_modulation_self_attn.2.weight", 3 * D, cfg.adaln_lora_dim, ctx)
    _put(bw, "adaln_modulation_cross_attn.1.weight", cfg.adaln_lora_dim, D, ctx)
    _put(bw, "adaln_modulation_cross_attn.2.weight", 3 * D, cfg.adaln_lora_dim, ctx)
    _put(bw, "adaln_modulation_mlp.1.weight", cfg.adaln_lora_dim, D, ctx)
    _put(bw, "adaln_modulation_mlp.2.weight", 3 * D, cfg.adaln_lora_dim, ctx)
    _put(bw, "self_attn.q_proj.weight", D, D, ctx)
    _put(bw, "self_attn.k_proj.weight", D, D, ctx)
    _put(bw, "self_attn.v_proj.weight", D, D, ctx)
    _put(bw, "self_attn.output_proj.weight", D, D, ctx)
    _put1(bw, "self_attn.q_norm.weight", DH, ctx)
    _put1(bw, "self_attn.k_norm.weight", DH, ctx)
    _put(bw, "cross_attn.q_proj.weight", D, D, ctx)
    _put(bw, "cross_attn.k_proj.weight", D, cfg.crossattn_in, ctx)
    _put(bw, "cross_attn.v_proj.weight", D, cfg.crossattn_in, ctx)
    _put(bw, "cross_attn.output_proj.weight", D, D, ctx)
    _put1(bw, "cross_attn.q_norm.weight", DH, ctx)
    _put1(bw, "cross_attn.k_norm.weight", DH, ctx)
    _put(bw, "mlp.layer1.weight", 4 * D, D, ctx)
    _put(bw, "mlp.layer2.weight", D, 4 * D, ctx)

    var x_f32 = _rand(N, D, STDtype.F32, ctx)
    var emb = _rand(TP, D, STDtype.BF16, ctx)
    var adaln = _rand(TP, 3 * D, STDtype.BF16, ctx)
    var text_ctx = _rand(TXT, cfg.crossattn_in, STDtype.BF16, ctx)

    var out = cosmos_block_forward[N, TXT, H, DH](
        x_f32^, emb, adaln, text_ctx, cs[0], cs[1], bw, cfg, TP, HP * WP, ctx
    )
    print("block out shape:", out.shape()[0], out.shape()[1])  # [N, D]
    var host = out.to_host(ctx)
    print("block out[0]:", host[0], " sumabs proxy:", host[0] + host[1] + host[2])

    print("CHUNK A: PASS (rope per-axis theta + one block compiled+ran)")
    print("CHUNK B: full-stack CosmosPredict25Dit symbol resolved (load/forward typed)")


def _put(mut d: Dict[String, ArcPointer[Tensor]], name: String, o: Int, i: Int, ctx: DeviceContext) raises:
    d[name] = ArcPointer(_rand(o, i, STDtype.BF16, ctx))


def _put1(mut d: Dict[String, ArcPointer[Tensor]], name: String, n: Int, ctx: DeviceContext) raises:
    var h = List[Float32]()
    for _ in range(n):
        h.append(1.0)
    var shp = List[Int]()
    shp.append(n)
    d[name] = ArcPointer(Tensor.from_host(h^, shp^, STDtype.BF16, ctx))


def _rand(r: Int, c: Int, dt: STDtype, ctx: DeviceContext) raises -> Tensor:
    var h = List[Float32]()
    var seed = 12345
    for i in range(r * c):
        seed = (seed * 1103515245 + 12345) % 2147483648
        h.append((Float32(seed) / 2147483648.0 - 0.5) * 0.1)
    var shp = List[Int]()
    shp.append(r)
    shp.append(c)
    return Tensor.from_host(h^, shp^, dt, ctx)
