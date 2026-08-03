# serenitymojo/models/text_encoder/parity/minimax_h3_deepstack_gpu_gate.mojo
#
# Check E of the deepstack lane — the one minimax_h3_deepstack_gate.mojo
# (CPU-only by hard requirement) explicitly could NOT run: the COMPOSED
# 3-layer streamed decode on REAL FL2VA text_encoder weights, with the vision
# splice + deepstack taps injected, vs the torch oracle's own real-weight run.
#
# Oracle: scripts/minimax_h3_deepstack_oracle.py (read in full 2026-08-03):
#   - REAL layers 0/1/2 + embed_tokens from FL2VA text_encoder, F32, CPU.
#   - Position ids: model default = plain sequential (no real image grid, so
#     mrope degenerates to 1D rope — the oracle states this at its
#     forward_truncated() comment, and it is the same degenerate case
#     minimax_h3_encode_conditioning_streamed_depth's sequential
#     _build_rope_tables implements).
#   - out.hidden_00/01/02 are RAW pre-norm states after 1/2/3 layers (the
#     oracle swaps model.norm for Identity), i.e. exactly the
#     `hidden_states[num_layers]` convention streamed_depth returns.
#   - in.vision_embeds [10,5120] and in.deepstack [3,10,5120] are synthetic
#     fixed-seed (the tower is gated separately); the LAYER ARITHMETIC and the
#     injection composition are what this gate exercises.
#
# BAR: cos >= 0.999 per depth (the bar the CPU gate's own check-E text set).
# The oracle is all-F32; our streamed path holds hidden state in the
# checkpoint's native BF16 and splices/adds through F32 host round-trips, so
# bit-exactness is not expected — bf16 carry through 1-3 real layers is.
#
# Run (GPU):
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     -Xlinker -lcuda -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     serenitymojo/models/text_encoder/parity/minimax_h3_deepstack_gpu_gate.mojo \
#     -o <scratch>/dsgpu
#   <scratch>/dsgpu [ref.safetensors]

from std.sys import argv
from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_streamed import (
    H3_HIDDEN,
    minimax_h3_mm_token_type_ids,
    minimax_h3_visual_positions,
    minimax_h3_encode_conditioning_streamed_depth,
)
from serenitymojo.models.text_encoder.minimax_h3_qwen3vl_vision import (
    MiniMaxH3VisionOutput,
)

comptime DEFAULT_REF = "/tmp/claude-1000/-home-alex-mojodiffusion/7e1531cb-f7e2-44a5-9d63-8604853a656a/scratchpad/deepstack_ref.safetensors"
comptime TEXT_ENCODER_DIR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder"

# Same real special-token ids the oracle and the CPU gate hardcode
# (FL2VA text_encoder config.json image_token_id / video_token_id).
comptime IMAGE_TOKEN_ID = 151655
comptime VIDEO_TOKEN_ID = 151656

comptime COS_BAR = 0.999


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _load_i32(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.I32:
        raise Error(String("_load_i32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int32]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


struct _CosMax(Copyable, Movable):
    var cos: Float64
    var mx: Float64

    def __init__(out self, cos: Float64, mx: Float64):
        self.cos = cos
        self.mx = mx


def _cos_maxabs(a: List[Float32], b: List[Float32]) raises -> _CosMax:
    if len(a) != len(b):
        raise Error(
            "_cos_maxabs: length mismatch "
            + String(len(a))
            + " vs "
            + String(len(b))
        )
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var mx = Float64(0.0)
    for i in range(len(a)):
        var x = Float64(a[i])
        var y = Float64(b[i])
        dot += x * y
        na += x * x
        nb += y * y
        var d = x - y
        if d < 0:
            d = -d
        if d > mx:
            mx = d
    if na == 0.0 or nb == 0.0:
        raise Error("_cos_maxabs: zero-norm operand")
    return _CosMax(dot / (sqrt(na) * sqrt(nb)), mx)


def main() raises:
    var args = argv()
    var ref_path = String(DEFAULT_REF)
    if len(args) >= 2:
        ref_path = String(args[1])

    print("=== MiniMax-H3 deepstack COMPOSED GPU gate (check E) ===")
    print("  oracle:", ref_path)
    print("  weights:", String(TEXT_ENCODER_DIR), "(REAL, layers 0-2 streamed)")
    print("  bar: cos >=", COS_BAR, " per depth (F32 oracle vs bf16 stream)")
    print("")

    var st = SafeTensors.open(ref_path)
    var ids = _load_i32(st, "in.input_ids")
    var mm_ref = _load_i32(st, "in.mm_token_type_ids")
    var vision_embeds = _load_f32(st, "in.vision_embeds")
    var deepstack = _load_f32(st, "in.deepstack")

    var num_visual = len(vision_embeds) // H3_HIDDEN
    if len(deepstack) != 3 * num_visual * H3_HIDDEN:
        raise Error("gate: in.deepstack size does not match 3 x N x hidden")

    # Recompute the visual positions through OUR helpers (the CPU gate proved
    # them bit-exact; recomputing here keeps this gate self-contained).
    var mm = minimax_h3_mm_token_type_ids(ids, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID)
    if len(mm) != len(mm_ref):
        raise Error("gate: mm_token_type_ids length mismatch vs oracle")
    for i in range(len(mm)):
        if mm[i] != mm_ref[i]:
            raise Error("gate: mm_token_type_ids differs from oracle at " + String(i))
    var vpos = minimax_h3_visual_positions(mm)
    if len(vpos) != num_visual:
        raise Error("gate: visual position count != vision_embeds rows")
    print("  seq =", len(ids), " visual rows =", num_visual, " (helpers re-verified vs oracle)")

    var vision = MiniMaxH3VisionOutput(
        embeds=vision_embeds.copy(),
        deepstack=deepstack.copy(),
        num_tokens=num_visual,
    )

    # sdpa_dispatch enumerates power-of-2 seq cases — pad ids to the next
    # case with the tokenizer pad id and compare the real-row PREFIX only
    # (same pad-to-case law the product CLIs use; trailing pads under a
    # causal mask cannot influence rows 0..seq-1, and every visual position
    # is inside the real prefix).
    var real_seq = len(ids)
    var padded_seq = 1
    while padded_seq < real_seq:
        padded_seq *= 2
    var padded_ids = ids.copy()
    while len(padded_ids) < padded_seq:
        padded_ids.append(151643)
    if padded_seq != real_seq:
        print("  pad-to-case:", real_seq, "->", padded_seq, "(compare first", real_seq, "rows)")

    var ctx = DeviceContext()
    var failures = 0
    var names = List[String]()
    names.append(String("out.hidden_00"))
    names.append(String("out.hidden_01"))
    names.append(String("out.hidden_02"))

    for depth in range(1, 4):
        var want = _load_f32(st, names[depth - 1])
        var got_t = minimax_h3_encode_conditioning_streamed_depth(
            String(TEXT_ENCODER_DIR),
            padded_ids,
            depth,
            ctx,
            Optional[MiniMaxH3VisionOutput](vision.copy()),
            Optional[List[Int]](vpos.copy()),
        )
        var got_full = got_t.to_host(ctx)
        var got = List[Float32]()
        got.reserve(real_seq * H3_HIDDEN)
        for i in range(real_seq * H3_HIDDEN):
            got.append(got_full[i])
        var r = _cos_maxabs(got, want)
        var ok = r.cos >= Float64(COS_BAR)
        if not ok:
            failures += 1
        print(
            ("   PASS " if ok else "   FAIL "),
            " depth=",
            depth,
            " vs ",
            names[depth - 1],
            "  cos=",
            r.cos,
            "  max_abs=",
            r.mx,
        )

    print("")
    if failures > 0:
        raise Error(
            "minimax_h3_deepstack_gpu_gate: "
            + String(failures)
            + " depth(s) below cos bar "
            + String(COS_BAR)
        )
    print("PASS: composed deepstack GPU gate, 3 depths on real weights")
