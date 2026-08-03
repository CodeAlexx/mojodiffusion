# serenitymojo/models/minimax_h3/parity/minimax_h3_block_parity.mojo
#
# MiniMax-H3 transformer forward parity gate — the whole DiT math, on a tiny
# random-weight model, with no checkpoint in existence.
#
# Reference: diffusers PR huggingface/diffusers#14355 at head e1b518df, run by
# scripts/minimax_h3_block_oracle.py against the reference's OWN test fixture
# config. Both sides consume the same 39334 random parameters and the same
# inputs; the gate compares the two output heads.
#
# What passing this covers: RMSNorm, the wide QKV projections, per-head q/k
# norms, PARTIAL rotary (rotary_dim 12 < head_dim 16, so the pass-through path
# is exercised), full self-attention with softmax, the SwiGLU feed-forward with
# its value/gate order, three-modality AdaLN modulation addressed per row, the
# token refiner, the final timestep-indexed modulated norm, and both output
# heads with their row selection.
#
# BAR: this is a chain of matmuls and a softmax, both sides float32 but with
# different summation orders (torch's blocked GEMM vs a straight loop here), so
# bit-exactness is not the right bar and demanding it would be a test of BLAS.
# The bar is 2e-5 absolute on values of order 1, with the max deviation printed
# so drift is visible rather than merely under a threshold.
#
# Oracle: python3 scripts/minimax_h3_block_oracle.py
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_block_parity.mojo \
#     -o output/checks/minimax_h3_block_parity \
#   && output/checks/minimax_h3_block_parity

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.block_forward import (
    MiniMaxH3BlockConfig,
    MiniMaxH3Weights,
    minimax_h3_forward,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_block/block_ref.safetensors"
comptime TOL = Float32(2.0e-5)


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


def _load_f64_as_f64(ref st: SafeTensors, name: String) raises -> List[Float64]:
    """position_ids are dumped float32 by the fixture; the port's rope builder
    takes float64 as the packing modules emit, so widen on the way in."""
    var values = _load_f32(st, name)
    var out = List[Float64]()
    for i in range(len(values)):
        out.append(Float64(values[i]))
    return out^


def _load_i64(ref st: SafeTensors, name: String) raises -> List[Int]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.I64:
        raise Error(String("_load_i64: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Int64]()
    var out = List[Int]()
    for i in range(tv.numel()):
        out.append(Int(p[i]))
    return out^


def _weight_names() -> List[String]:
    """Every parameter the tiny model exposes, in the reference's own naming."""
    var names = List[String]()
    names.append(String("proj_in.weight"))
    names.append(String("proj_in.bias"))
    names.append(String("audio_proj_in.weight"))
    names.append(String("audio_proj_in.bias"))
    names.append(String("context_embedder.weight"))
    names.append(String("context_embedder.bias"))
    names.append(String("time_embedder.linear_1.weight"))
    names.append(String("time_embedder.linear_1.bias"))
    names.append(String("time_embedder.linear_2.weight"))
    names.append(String("time_embedder.linear_2.bias"))
    names.append(String("token_refiner.final_norm.weight"))
    names.append(String("norm_out.norm.weight"))
    names.append(String("norm_out.linear.weight"))
    names.append(String("norm_out.linear.bias"))
    names.append(String("proj_out.weight"))
    names.append(String("proj_out.bias"))
    names.append(String("audio_proj_out.weight"))
    names.append(String("audio_proj_out.bias"))
    for layer in range(2):
        var p = String("transformer_blocks.") + String(layer)
        names.append(p + ".norm1.weight")
        names.append(p + ".norm2.weight")
        names.append(p + ".attn.to_q.weight")
        names.append(p + ".attn.to_k.weight")
        names.append(p + ".attn.to_v.weight")
        names.append(p + ".attn.norm_q.weight")
        names.append(p + ".attn.norm_k.weight")
        names.append(p + ".attn.to_out.0.weight")
        names.append(p + ".ff.net.0.proj.weight")
        names.append(p + ".ff.net.2.weight")
        names.append(p + ".adaln_proj.linear.weight")
        names.append(p + ".adaln_proj.linear.bias")
    for layer in range(2):
        var p = String("token_refiner.refiner_blocks.") + String(layer)
        names.append(p + ".norm1.weight")
        names.append(p + ".norm2.weight")
        names.append(p + ".attn.to_q.weight")
        names.append(p + ".attn.to_k.weight")
        names.append(p + ".attn.to_v.weight")
        names.append(p + ".attn.norm_q.weight")
        names.append(p + ".attn.norm_k.weight")
        names.append(p + ".attn.to_out.0.weight")
        names.append(p + ".ff.net.0.proj.weight")
        names.append(p + ".ff.net.2.weight")
    return names^


def _compare(
    label: String, got: List[Float32], want: List[Float32], tol: Float32
) raises -> Bool:
    if len(got) != len(want):
        print("  FAIL", label, "length", len(got), "!=", len(want))
        return False
    var worst = Float32(0.0)
    var worst_index = 0
    for i in range(len(got)):
        var diff = got[i] - want[i]
        var mag = -diff if diff < 0.0 else diff
        if mag > worst:
            worst = mag
            worst_index = i
    if worst <= tol:
        print("  ok  ", label, len(got), "values, max_abs", worst)
        return True
    print(
        "  FAIL", label, "max_abs", worst, "at", worst_index,
        "got", got[worst_index], "want", want[worst_index],
    )
    return False


def main() raises:
    print("MiniMax-H3 transformer forward parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))

    # The reference's own fixture config.
    var config = MiniMaxH3BlockConfig(
        2,      # num_attention_heads
        16,     # attention_head_dim   -> inner 32, wider than hidden 24
        24,     # hidden_size
        2,      # num_layers
        2,      # num_refiner_layers
        32,     # ffn_dim
        4,      # in_channels          -> video patch dim 16
        6,      # audio_in_channels
        8,      # text_dim
        8,      # freq_dim
        16,     # time_embed_dim
        2,      # rope_freq_dim        -> rotary 12 < head_dim 16, PARTIAL rope
        Float32(1.0e-5),
        Float32(1.0e-5),
        Float32(1.0e-5),
    )

    var names = _weight_names()
    var values = List[List[Float32]]()
    for i in range(len(names)):
        values.append(_load_f32(st, String("w.") + names[i]))
    var weights = MiniMaxH3Weights(names^, values^)

    var out = minimax_h3_forward(
        weights,
        config,
        _load_f32(st, "in.hidden_states"),
        _load_f32(st, "in.audio_hidden_states"),
        _load_f32(st, "in.encoder_hidden_states"),
        _load_f32(st, "in.timestep"),
        _load_i64(st, "in.timestep_indices"),
        _load_i64(st, "in.token_tags"),
        _load_f64_as_f64(st, "in.position_ids"),
        _load_i64(st, "in.video_indices"),
        _load_i64(st, "in.audio_indices"),
        _load_i64(st, "in.text_indices"),
    )

    print("")
    print("[outputs]")
    var ok_video = _compare(
        String("sample (video velocity)"), out.sample, _load_f32(st, "out.sample"), TOL
    )
    var ok_audio = _compare(
        String("audio_sample (audio velocity)"),
        out.audio_sample,
        _load_f32(st, "out.audio_sample"),
        TOL,
    )

    print("")
    if ok_video and ok_audio:
        print("PASS: the full transformer forward matches the reference")
    else:
        raise Error("MiniMax-H3 transformer forward parity gate failed")
