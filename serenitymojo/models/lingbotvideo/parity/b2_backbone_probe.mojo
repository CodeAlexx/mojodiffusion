# models/lingbotvideo/parity/b2_backbone_probe.mojo — CHUNK B2 gate.
#
# The INTEGRATION gate: composes B1 (embeddings + rope + time) + the A3-gated
# `lingbot_video_block` × 48 (weights STREAMED one block ~1.2GB at a time from
# the 13-shard 60GB model) + the POST head, and gates INCREMENTALLY vs
# oracle_b2.safetensors to localize any divergence:
#   1. joint_preblock  (embeddings, REAL weights)  — expect ~exact.
#   2. block_0 / block_23 / block_47 per-block taps — accumulation lowers cos
#      (router near-tie flips over 48 layers is the known benign effect).
#   3. FINAL velocity cos >= 0.999 + |mine|/|ref| magnitude ratio + max_abs.
#
# Run (JIT; streams 60GB from NVMe — first run may take a minute or two):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/b2_backbone_probe.mojo

from std.math import sqrt
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.env import env_or, env_int
from serenitymojo.parity import ParityHarness
from serenitymojo.models.lingbotvideo.backbone import (
    LingBotAttnConfig,
    lingbot_backbone,
    LingBotResidentStore,
)

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime MODEL_DIR_DEFAULT = "/mnt/disk1/models/lingbot-video-moe/transformer"
# Override with LINGBOT_CKPT (e.g. .../transformer_fp8) for the fp8 dequant path.
comptime GT = 1
comptime GH = 8
comptime GW = 8
comptime TEXT_LEN = 8
comptime N_VIDEO = GT * GH * GW   # 64
comptime S = N_VIDEO + TEXT_LEN   # 72
comptime DEPTH = 48
comptime OUT_CH = 16
comptime THETA = Float32(256.0)


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _gate(
    tag: String, mine: List[Float32], reference: List[Float32], thr: Float64
) raises -> Bool:
    var harness = ParityHarness(thr)
    var res = harness.compare_host(mine, reference)
    var nm: Float64 = 0.0
    var nr: Float64 = 0.0
    for i in range(len(mine)):
        nm += Float64(mine[i]) * Float64(mine[i])
        nr += Float64(reference[i]) * Float64(reference[i])
    var mag = sqrt(nm) / sqrt(nr) if nr > 0.0 else Float64(0.0)
    var pass_tag = "PASS" if res.cos >= thr else "FAIL"
    print(
        "[B2]", tag, "cos =", res.cos, " max_abs =", res.max_abs,
        " |mine|/|ref| =", mag, "  ", pass_tag, "(thr", thr, ")",
    )
    return res.cos >= thr


def main() raises:
    var ctx = DeviceContext()
    var cfg = LingBotAttnConfig.default()

    print("[B2] loading oracle_b2.safetensors  S =", S, " depth =", DEPTH)
    var oracle = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_b2.safetensors")
    var latent = _load_f32(oracle, "latent", ctx)          # [1,16,1,16,16]
    var timestep = _load_f32(oracle, "timestep", ctx)      # [1]
    var text_embeds = _load_f32(oracle, "text_embeds", ctx)  # [1,8,2560]

    var model_dir = env_or(String("LINGBOT_CKPT"), String(MODEL_DIR_DEFAULT))
    print("[B2] opening transformer shards (streaming):", model_dir)
    var model = ShardedSafeTensors.open(model_dir)

    var tap_indices = List[Int]()
    for ti in [0, 1, 2, 4, 8, 12, 16, 20, 23, 28, 32, 36, 40, 44, 47]:
        tap_indices.append(ti)

    var n_res = env_int(String("LINGBOT_RESIDENT_BLOCKS"), 0)
    var store = LingBotResidentStore.load(model, n_res, ctx) if n_res > 0 else LingBotResidentStore.empty()
    print("[B2] running lingbot_backbone[S=", S, "] resident_blocks =", n_res)
    var res = lingbot_backbone[S](
        latent, timestep, text_embeds, model,
        GT, GH, GW, TEXT_LEN, 1, 2, 2, OUT_CH, DEPTH, THETA,
        tap_indices, cfg, store, ctx,
    )

    # ── 1. joint_preblock (embeddings, REAL weights) ──────────────────────────
    _ = _gate(
        "joint_preblock",
        res.joint_preblock[].to_host(ctx),
        _load_f32(oracle, "joint_preblock", ctx).to_host(ctx),
        0.999,
    )

    # ── 2. per-block taps (dense curve to localize divergence) ────────────────
    for k in range(len(tap_indices)):
        var idx = tap_indices[k]
        _ = _gate(
            String("block_") + String(idx),
            res.taps[k][].to_host(ctx),
            _load_f32(oracle, String("block_") + String(idx), ctx).to_host(ctx),
            0.999,
        )

    # ── 3. FINAL velocity gate (THE chunk gate) ───────────────────────────────
    var vel_mine = res.velocity[].to_host(ctx)
    var vel_ref = _load_f32(oracle, "velocity", ctx).to_host(ctx)
    var passed = _gate("velocity", vel_mine, vel_ref, 0.999)
    if passed:
        print("[B2] ===== B2 GATE PASS (velocity cos >= 0.999) =====")
    else:
        print("[B2] ===== B2 GATE FAIL (velocity cos < 0.999) =====")
