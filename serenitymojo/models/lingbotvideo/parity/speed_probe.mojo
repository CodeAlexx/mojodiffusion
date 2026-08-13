# models/lingbotvideo/parity/speed_probe.mojo — Phase-1 Gate-5 speed table.
#
# Times K full backbone forwards (each re-streams the whole transformer from the
# LINGBOT_CKPT dir) and reports s/forward. One denoise STEP = 2 forwards (CFG),
# so s/step = 2 * s/forward. Run once with the bf16 dir and once with the fp8 dir
# (transformer_fp8) for the before/after table:
#
#   cd /home/alex/mojodiffusion
#   pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/speed_probe.mojo   # bf16
#   LINGBOT_CKPT=/mnt/disk1/models/lingbot-video-moe/transformer_fp8 \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/speed_probe.mojo # fp8
#
# forward[0] is COLD (first disk read); forwards[1..] reflect warm-page-cache
# steady state — fp8 (30GB) fits the 60GB RAM cache, bf16 (57GB) largely does
# too but with more eviction pressure; both are reported.

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.env import env_or, env_int
from serenitymojo.models.lingbotvideo.backbone import (
    LingBotAttnConfig,
    lingbot_backbone,
    LingBotResidentStore,
)

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime MODEL_DIR_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer"
comptime GT = 1
comptime GH = 8
comptime GW = 8
comptime TEXT_LEN = 8
comptime N_VIDEO = GT * GH * GW
comptime S = N_VIDEO + TEXT_LEN   # 72
comptime DEPTH = 48
comptime OUT_CH = 16
comptime THETA = Float32(256.0)
comptime K = 4                    # forwards to time


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view(st.tensor_view(name), ctx)


def main() raises:
    var ctx = DeviceContext()
    var cfg = LingBotAttnConfig.default()

    var oracle = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_b2.safetensors")
    var latent = _load_f32(oracle, "latent", ctx)
    var timestep = _load_f32(oracle, "timestep", ctx)
    var text_embeds = _load_f32(oracle, "text_embeds", ctx)

    var model_dir = env_or(String("LINGBOT_CKPT"), String(MODEL_DIR_DEFAULT))
    var model = ShardedSafeTensors.open(model_dir)
    var n_res = env_int(String("LINGBOT_RESIDENT_BLOCKS"), 0)
    var store = LingBotResidentStore.load(model, n_res, ctx) if n_res > 0 else LingBotResidentStore.empty()
    print("[SPEED] ckpt =", model_dir, "  K =", K, " forwards  (S =", S,
          ")  resident_blocks =", n_res)

    var tap_indices = List[Int]()   # no taps — pure forward timing

    var warm_sum: Float64 = 0.0
    for k in range(K):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        var res = lingbot_backbone[S](
            latent, timestep, text_embeds, model,
            GT, GH, GW, TEXT_LEN, 1, 2, 2, OUT_CH, DEPTH, THETA,
            tap_indices, cfg, store, ctx,
        )
        # touch the output + fence so timing includes the whole forward.
        _ = res.velocity[].to_host(ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        var tag = "COLD" if k == 0 else "warm"
        print("[SPEED] forward", k, "(", tag, "):", secs, "s  -> step(x2):", secs * 2.0, "s")
        if k > 0:
            warm_sum += secs

    var warm_avg = warm_sum / Float64(K - 1)
    print("[SPEED] warm avg forward:", warm_avg, "s   =>  s/step (x2 CFG):", warm_avg * 2.0)
