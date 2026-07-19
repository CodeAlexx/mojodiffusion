# LTX-2 connector REGISTER-REPLACEMENT parity probe.
#
# Gates ltx2_connector_forward(valid_len=...) — the reference semantics where
# pad rows of the right-padded context are replaced with the checkpoint's
# learnable registers before the blocks — against the ltx_core ground truth
# dumped by scripts/ltx2_connector_mask_oracle.py.
#
# HARD GATE: cos >= 0.999 for BOTH video and audio connector outputs.
#
# Build+run like the connector smoke (GPU; needs the cudnn sdpa shim links).

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.ltx2_connector import (
    LTX2ConnectorConfig,
    LTX2ConnectorWeights,
    ltx2_connector_forward,
)

comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled.safetensors"
comptime REF = "/home/alex/mojodiffusion/output/ltx2_connector/connector_mask_ref.safetensors"
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


def main() raises:
    var ctx = DeviceContext()
    print("=== LTX2 connector register-replacement parity probe ===")
    var st = ShardedSafeTensors.open(String(REF))
    var v_in = Tensor.from_view_as_bf16(st.tensor_view("video_in"), ctx)
    var a_in = Tensor.from_view_as_bf16(st.tensor_view("audio_in"), ctx)
    var lh = Tensor.from_view(st.tensor_view("valid_len"), ctx).to_host(ctx)
    var valid = Int(lh[0])
    print("  valid_len:", valid, "/", N)

    var v_w = LTX2ConnectorWeights.load(
        String(CKPT), String("video_embeddings_connector"),
        LTX2ConnectorConfig.video(), ctx,
    )
    var a_w = LTX2ConnectorWeights.load(
        String(CKPT), String("audio_embeddings_connector"),
        LTX2ConnectorConfig.audio(), ctx,
    )

    var v_out = ltx2_connector_forward[N, 32, 128](v_w, v_in, ctx, valid_len=valid)
    var a_out = ltx2_connector_forward[N, 32, 64](a_w, a_in, ctx, valid_len=valid)

    var v_ref = Tensor.from_view_as_bf16(st.tensor_view("video_out"), ctx)
    var a_ref = Tensor.from_view_as_bf16(st.tensor_view("audio_out"), ctx)

    var cv = _cosine_sim(v_out.to_host(ctx), v_ref.to_host(ctx))
    var ca = _cosine_sim(a_out.to_host(ctx), a_ref.to_host(ctx))
    print("  cos(video):", cv)
    print("  cos(audio):", ca)
    if cv < GATE or ca < GATE:
        raise Error("GATE FAIL: connector register-replacement cos < 0.999")
    print("GATE PASS: register-replacement connector matches ltx_core")
