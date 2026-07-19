# models/lingbotvideo/parity/dense_backbone_probe.mojo — DENSE backbone gate.
#
# Composes the RESIDENT dense backbone (dense.mojo) — B1 embeddings + rope + time
# + lingbot_video_dense_block x 24 (weights loaded to VRAM ONCE) + POST head —
# and gates INCREMENTALLY vs the tap-forward captures in oracle_dense.safetensors
# (sf_* keys, S=72):
#   1. sf_joint_preblock  (embeddings, REAL weights)          — expect ~exact.
#   2. sf_block_0 .. sf_block_23 per-block taps               — dense = clean.
#   3. FINAL velocity cos >= 0.999 (dense has NO MoE router flips).
#
# Run (JIT):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/lingbotvideo/parity/dense_backbone_probe.mojo

from std.math import sqrt
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.lingbotvideo.backbone import LingBotAttnConfig
from serenitymojo.models.lingbotvideo.dense import (
    LingBotDenseModel,
    lingbot_dense_backbone,
)

comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/lingbotvideo/parity"
comptime MODEL_DIR = "/mnt/disk1/models/lingbot-video-dense/transformer"
comptime GT = 1
comptime GH = 8
comptime GW = 8
comptime TEXT_LEN = 8
comptime N_VIDEO = GT * GH * GW   # 64
comptime S = N_VIDEO + TEXT_LEN   # 72
comptime DEPTH = 24
comptime OUT_CH = 16
comptime THETA = Float32(256.0)


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_view(st.tensor_view(name), ctx)


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
        "[DENSE]", tag, "cos =", res.cos, " max_abs =", res.max_abs,
        " |mine|/|ref| =", mag, "  ", pass_tag, "(thr", thr, ")",
    )
    return res.cos >= thr


def main() raises:
    var ctx = DeviceContext()
    var cfg = LingBotAttnConfig.default()

    print("[DENSE] loading oracle_dense.safetensors  S =", S, " depth =", DEPTH)
    var oracle = ShardedSafeTensors.open(String(PARITY_DIR) + "/oracle_dense.safetensors")
    var latent = _load_f32(oracle, "sf_latent", ctx)          # [1,16,1,16,16]
    var timestep = _load_f32(oracle, "sf_timestep", ctx)      # [1]
    var text_embeds = _load_f32(oracle, "sf_text_embeds", ctx)  # [1,8,2560]

    print("[DENSE] opening dense transformer (single file):", MODEL_DIR)
    var model = ShardedSafeTensors.open(String(MODEL_DIR))

    var _t0 = perf_counter_ns()
    print("[DENSE] loading ALL", DEPTH, "blocks RESIDENT to VRAM (once)")
    var dm = LingBotDenseModel.load(model, DEPTH, ctx)
    var _t1 = perf_counter_ns()
    print("[DENSE] resident load done in", Float64(_t1 - _t0) / 1.0e9, "s")

    var tap_indices = List[Int]()
    for ti in range(DEPTH):
        tap_indices.append(ti)

    print("[DENSE] running lingbot_dense_backbone[S=", S, "]")
    var _f0 = perf_counter_ns()
    var res = lingbot_dense_backbone[S](
        latent, timestep, text_embeds, dm,
        GT, GH, GW, TEXT_LEN, 1, 2, 2, OUT_CH, THETA,
        tap_indices, cfg, ctx,
    )
    var _f1 = perf_counter_ns()
    print("[DENSE] forward wall =", Float64(_f1 - _f0) / 1.0e9, "s")

    # ── 1. joint_preblock ─────────────────────────────────────────────────────
    _ = _gate(
        "joint_preblock",
        res.joint_preblock[].to_host(ctx),
        _load_f32(oracle, "sf_joint_preblock", ctx).to_host(ctx),
        0.999,
    )

    # ── 2. per-block taps (should stay clean — dense, no router flips) ────────
    for k in range(len(tap_indices)):
        var idx = tap_indices[k]
        _ = _gate(
            String("block_") + String(idx),
            res.taps[k][].to_host(ctx),
            _load_f32(oracle, String("sf_block_") + String(idx), ctx).to_host(ctx),
            0.999,
        )

    # ── 3. FINAL velocity gate (THE chunk gate) ───────────────────────────────
    var vel_mine = res.velocity[].to_host(ctx)
    var vel_ref = _load_f32(oracle, "sf_velocity", ctx).to_host(ctx)
    var passed = _gate("velocity", vel_mine, vel_ref, 0.999)
    if passed:
        print("[DENSE] ===== DENSE BACKBONE GATE PASS (velocity cos >= 0.999) =====")
    else:
        print("[DENSE] ===== DENSE BACKBONE GATE FAIL (velocity cos < 0.999) =====")
