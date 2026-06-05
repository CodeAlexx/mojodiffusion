# pipeline/ltx2_fp8_resident_smoke.mojo — lightweight gate for the LTX-2 FP8
# resident storage path.
#
# This does NOT run denoise. It preloads one inner FP8 block as raw GPU-resident
# bytes, materializes that block through `load_block_bf16`, and checks the
# existing AV-forward-required keys are present as BF16 tensors. This gates the
# speed-path plumbing without the cost of a full 22B generation.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.offload.ltx2_block_stream import LTX2BlockStream, drop_block


comptime CKPT = String(
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
)
comptime MIB = 1024.0 * 1024.0


def _require(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(msg)


def main() raises:
    var ctx = DeviceContext()
    var stream = LTX2BlockStream.open(CKPT)
    _require(stream.block_count() == 48, "expected 48 LTX2 blocks")

    print("=== LTX2 FP8 resident loader smoke ===")
    print("[resident] preload block 4 only")
    stream.enable_fp8_resident_range(4, 4, ctx)
    var resident_mib = Int(Float64(stream.resident_bytes()) / MIB)
    print("[resident] bytes:", stream.resident_bytes(), " (", resident_mib, " MiB )")
    _require(stream.resident_bytes() > 0, "resident byte count stayed zero")

    var n_fp8 = stream.fp8_tensor_count(4)
    print("[block4] fp8 tensor count:", n_fp8)
    _require(n_fp8 > 0, "block 4 did not expose FP8 tensors")

    var block = stream.load_block_bf16(4, ctx)
    _require("attn1.to_q.weight" in block, "resident block missing attn1.to_q.weight")
    _require("audio_attn1.to_q.weight" in block, "resident block missing audio_attn1.to_q.weight")
    _require("audio_to_video_attn.to_q.weight" in block, "resident block missing a2v q weight")
    _require("ff.net.0.proj.weight" in block, "resident block missing video FFN")
    _require("audio_ff.net.0.proj.weight" in block, "resident block missing audio FFN")

    var q_dtype = block["attn1.to_q.weight"][].dtype()
    var aq_dtype = block["audio_attn1.to_q.weight"][].dtype()
    print("[block4] attn1.to_q dtype:", q_dtype.name())
    print("[block4] audio_attn1.to_q dtype:", aq_dtype.name())
    _require(q_dtype == STDtype.BF16, "resident materialized video q is not BF16")
    _require(aq_dtype == STDtype.BF16, "resident materialized audio q is not BF16")

    drop_block(block^)
    ctx.synchronize()
    print("FP8 RESIDENT GATE PASS")
