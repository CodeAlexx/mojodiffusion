# LTX-2 connector + projection parity smoke (P2.5).
#
# Runs the in-Mojo Embeddings1DConnector (video + audio) on the PRE-connector
# cached embeds and gates the post-connector contexts against the Python
# reference dumped by scripts/ltx2_dit_forward_parity_ref.py (which runs the
# connector LOCALLY on the same pre-connector embeds).
#
# HARD GATE: cosine similarity >= 0.999 for BOTH video_out and audio_out vs the
# Python reference outputs (output/ltx2_connector/connector_ref.safetensors).
#
# The connector REPLACES caption_projection in this checkpoint (no separate
# caption_projection keys exist); the connector IS the projection. Video output
# is the video context [1,1024,4096]; audio output is the audio context
# [1,1024,2048] (audio context is PROJECTED, not a sidecar).
#
# Inputs read from the ref file:
#   video_in   [1,1024,4096]  pre-connector video context (cached text_hidden)
#   audio_in   [1,1024,2048]  pre-connector audio context (deterministic)
#   video_out  [1,1024,4096]  Python connector output (GATE target)
#   audio_out  [1,1024,2048]  Python connector output (GATE target)
#
# GPU run required (HARD RULE: real numeric gate on GPU, not compile-only).

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ltx2_connector import (
    LTX2ConnectorConfig,
    LTX2ConnectorWeights,
    ltx2_connector_forward,
)


comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors"
comptime REF = "/home/alex/mojodiffusion/output/ltx2_connector/connector_ref.safetensors"
comptime N = 1024
comptime GATE = Float64(0.999)


def _cosine_sim(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cosine_sim: length mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        var av = Float64(a[i])
        var bv = Float64(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na < 1e-30 or nb < 1e-30:
        return Float64(1.0)
    return dot / (sqrt(na) * sqrt(nb))


def _max_abs_diff(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("max_abs_diff: length mismatch")
    var m = Float64(0.0)
    for i in range(len(a)):
        var d = Float64(a[i]) - Float64(b[i])
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def _stats(name: String, h: List[Float32]) raises:
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        if v != v:
            raise Error(String("NaN in ") + name)
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(len(h))
    var var_ = s2 / Float64(len(h)) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print("  ", name, "mean/std/absmax:", Float32(mean), Float32(sqrt(var_)), Float32(amax))


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("GATE FAIL: ") + msg)


def main() raises:
    var ctx = DeviceContext()
    print("=== LTX-2 connector parity smoke (P2.5) ===")

    # Load ref fixture (F32 on disk -> BF16 inputs to match the cached BF16
    # contract; targets read back as F32).
    print("  [load] ref fixture:", REF)
    var fix = ShardedSafeTensors.open(String(REF))
    var video_in = Tensor.from_view_as_bf16(fix.tensor_view("video_in"), ctx)
    var audio_in = Tensor.from_view_as_bf16(fix.tensor_view("audio_in"), ctx)
    var video_tgt = Tensor.from_view_as_bf16(fix.tensor_view("video_out"), ctx)
    var audio_tgt = Tensor.from_view_as_bf16(fix.tensor_view("audio_out"), ctx)

    var vi_sh = video_in.shape()
    var ai_sh = audio_in.shape()
    print("  video_in shape:", vi_sh[0], vi_sh[1], vi_sh[2])
    print("  audio_in shape:", ai_sh[0], ai_sh[1], ai_sh[2])
    if len(vi_sh) != 3 or vi_sh[0] != 1 or vi_sh[1] != N or vi_sh[2] != 4096:
        raise Error("video_in shape mismatch (expect [1,1024,4096])")
    if len(ai_sh) != 3 or ai_sh[0] != 1 or ai_sh[1] != N or ai_sh[2] != 2048:
        raise Error("audio_in shape mismatch (expect [1,1024,2048])")

    # ── VIDEO connector ──
    print("  [load] video connector weights")
    var v_cfg = LTX2ConnectorConfig.video()
    var v_w = LTX2ConnectorWeights.load(
        String(CKPT), String("video_embeddings_connector"), v_cfg, ctx
    )
    print("  [run] video connector forward")
    var video_out = ltx2_connector_forward[N, 32, 128](v_w, video_in, ctx)

    var v_host = video_out.to_host(ctx)
    var v_tgt_host = video_tgt.to_host(ctx)
    _stats(String("video_out (mojo)"), v_host)
    _stats(String("video_out (ref) "), v_tgt_host)
    var v_cos = _cosine_sim(v_host, v_tgt_host)
    var v_mad = _max_abs_diff(v_host, v_tgt_host)
    print("  [video] cosine:", Float32(v_cos), " max_abs_diff:", Float32(v_mad))

    # ── AUDIO connector ──
    print("  [load] audio connector weights")
    var a_cfg = LTX2ConnectorConfig.audio()
    var a_w = LTX2ConnectorWeights.load(
        String(CKPT), String("audio_embeddings_connector"), a_cfg, ctx
    )
    print("  [run] audio connector forward")
    var audio_out = ltx2_connector_forward[N, 32, 64](a_w, audio_in, ctx)

    var a_host = audio_out.to_host(ctx)
    var a_tgt_host = audio_tgt.to_host(ctx)
    _stats(String("audio_out (mojo)"), a_host)
    _stats(String("audio_out (ref) "), a_tgt_host)
    var a_cos = _cosine_sim(a_host, a_tgt_host)
    var a_mad = _max_abs_diff(a_host, a_tgt_host)
    print("  [audio] cosine:", Float32(a_cos), " max_abs_diff:", Float32(a_mad))

    print("=== HARD GATE: cos >= 0.999 (both contexts) ===")
    _check(v_cos >= GATE, "video cosine " + String(Float32(v_cos)) + " < 0.999")
    print("  [gate] video cos >= 0.999: PASS")
    _check(a_cos >= GATE, "audio cosine " + String(Float32(a_cos)) + " < 0.999")
    print("  [gate] audio cos >= 0.999: PASS")
    print("=== P2.5 connector parity: PASS ===")
