# Parity: Ideogram-4 TEXT ENCODER (Qwen3-VL -> the 13-tap llm_features the DiT
# consumes) vs **ai-toolkit** oracle (NOT ideogram4-ref).
#
# Oracle = ideogram4_aitoolkit_encoder_oracle.py: loads the REAL bf16 Qwen3-VL text
# tower (via Qwen3VLForConditionalGeneration -- the public AutoModel path RANDOM-inits
# it in transformers 4.57.6) and runs ai-toolkit src/pipeline.py::get_qwen3_vl_features
# verbatim over FIXED token ids: taps QWEN3_VL_ACTIVATION_LAYERS =
# (0,3,6,9,12,15,18,21,24,27,30,33,35) = the post-decoder-layer hidden states (NO final
# model.norm), then stack->permute(1,2,3,0)->reshape => llm_features[..,f*13+t]=tap_t[..,f].
#
# This gate feeds the SAME dumped ids to the mojo encoder (load_ideogram_qwen3vl +
# encode_layer_states for per-tap, encode_ideogram_taps for the final concat) and
# compares a representative subset of taps (0/3/18/33/35) + the full llm_features at
# cos >= 0.999, FAIL-LOUD. Tokenization/caption-JSON-wrapping is a SEPARATE concern --
# this isolates the ENCODER MODEL given ids.
#
# Run (oracle FIRST):
#   /home/alex/serenityflow-v2/.venv/bin/python \
#     serenitymojo/models/text_encoder/parity/ideogram4_aitoolkit_encoder_oracle.py
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo run -I . \
#     serenitymojo/models/text_encoder/parity/ideogram4_aitoolkit_encoder_parity.mojo
from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.parity import ParityHarness
from serenitymojo.models.text_encoder.ideogram_qwen3vl import (
    load_ideogram_qwen3vl,
    encode_ideogram_taps,
)

comptime TE = "/home/alex/.serenity/models/ideogram-4-fp8/text_encoder/model.safetensors"
comptime DIR = "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/"
comptime FX = DIR + "ideogram4_aitoolkit_encoder.safetensors"
comptime IDS_BIN = DIR + "ideogram4_aitoolkit_encoder_ids.bin"
comptime COS_BAR = 0.999
comptime HIDDEN = 4096
comptime NTAPS = 13


def _read_ids_i32(path: String) raises -> List[Int]:
    """Read raw little-endian int32 ids dumped by the oracle."""
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open (run the oracle first): ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ids dump: ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var ni = n // 4
    var ip = buf.bitcast[Int32]()
    var out = List[Int]()
    for i in range(ni):
        out.append(Int(ip[i]))
    buf.free()
    return out^


def main() raises:
    var ctx = DeviceContext()
    print("=== Ideogram-4 TEXT ENCODER (Qwen3-VL 13-tap) parity vs ai-toolkit ===")

    var ids = _read_ids_i32(IDS_BIN)
    var L = len(ids)
    print("  loaded", L, "fixed token ids (first 5):", ids[0], ids[1], ids[2], ids[3], ids[4])

    var fx = ShardedSafeTensors.open(FX)

    print("  loading mojo Ideogram-4 Qwen3-VL encoder (fp8->bf16, ~9GB)...")
    var enc = load_ideogram_qwen3vl(TE, ctx)

    # ---- per-tap comparison (subset that the oracle dumped) ----
    # encode_layer_states[i] = output of layer i (post-layer, pre-final-norm) -- the
    # SAME convention as ai-toolkit's captured[layer_idx] (recorded AFTER decoder_layer).
    var states = enc.encode_layer_states(ids, ctx)  # 36 x [1,L,4096]
    if len(states) != 36:
        raise Error("expected 36 layer states, got " + String(len(states)))

    var want = [0, 3, 18, 33, 35]
    var all_pass = True
    for wi in range(len(want)):
        var li = want[wi]
        var tap_ref = Tensor.from_view(fx.tensor_view("tap_" + String(li)), ctx).to_host(ctx)
        if len(tap_ref) != L * HIDDEN:
            raise Error("tap_" + String(li) + " ref len wrong: " + String(len(tap_ref)))
        var r = ParityHarness(COS_BAR).compare(states[li][], tap_ref, ctx)
        print("  tap_" + String(li) + " (layer", li, "output):", r)
        if not r.passed:
            all_pass = False

    # ---- final concatenated llm_features [1, L, 4096*13=53248] ----
    var llm = encode_ideogram_taps(enc, ids, ctx)
    var ls = llm.shape()
    print("  llm_features shape:", ls[0], ls[1], ls[2], "(expect 1", L, HIDDEN * NTAPS, ")")
    if ls[0] != 1 or ls[1] != L or ls[2] != HIDDEN * NTAPS:
        raise Error("llm_features shape mismatch")
    var llm_ref = Tensor.from_view(fx.tensor_view("llm_features"), ctx).to_host(ctx)
    if len(llm_ref) != L * HIDDEN * NTAPS:
        raise Error("llm_features ref len wrong: " + String(len(llm_ref)))
    var r_llm = ParityHarness(COS_BAR).compare(llm, llm_ref, ctx)
    print("  llm_features (full 13-tap concat):", r_llm)
    if not r_llm.passed:
        all_pass = False

    # ---- FAIL LOUD ----
    if not all_pass:
        raise Error(
            "ideogram4 ai-toolkit TEXT ENCODER parity FAILED (cos < "
            + String(COS_BAR)
            + ") -- a wrong tap layer-set, RoPE/MRoPE phase, or fp8-vs-bf16 weight gap."
        )
    print("VERDICT: PASS -- mojo Ideogram-4 Qwen3-VL taps + llm_features match ai-toolkit (cos >= 0.999)")
