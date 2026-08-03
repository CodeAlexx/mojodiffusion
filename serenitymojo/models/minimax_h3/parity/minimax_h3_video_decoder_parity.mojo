# serenitymojo/models/minimax_h3/parity/minimax_h3_video_decoder_parity.mojo
#
# MiniMax-H3 video ViT decoder parity gate: the latent-to-pixel path, on a tiny
# random-weight model, with no checkpoint in existence.
#
# Reference: diffusers PR huggingface/diffusers#14355 at head e1b518df, run by
# scripts/minimax_h3_video_decoder_oracle.py against the reference's OWN test
# fixture config, with every parameter re-randomized — `register_tokens` and
# both residual `scale` vectors are zero-initialized, and leaving them zero
# would make the entire transformer an identity and this gate vacuous.
#
# The fixture keeps `decoder_rope_dim_ratio` at its released 0.75, so the
# rotary width (6) is smaller than the head dim (8) and the PARTIAL rotary path
# is exercised rather than aliased away.
#
# `post_quant_conv` is included, so a pass covers the whole latent-to-pixel
# path rather than the transformer alone.
#
# BAR: 2e-5 absolute, as for the other forward gates — different summation
# orders on both sides make bit-exactness a test of BLAS.
#
# Oracle: python3 scripts/minimax_h3_video_decoder_oracle.py
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_video_decoder_parity.mojo \
#     -o output/checks/minimax_h3_video_decoder_parity \
#   && output/checks/minimax_h3_video_decoder_parity

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.video_decoder import (
    MiniMaxH3VideoDecoderConfig,
    MiniMaxH3VideoWeights,
    minimax_h3_video_decode,
    video_position_grid,
    video_rope_inv_freq,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_video/video_decoder_ref.safetensors"
comptime TOL = Float32(2.0e-5)

comptime LATENT_FRAMES = 2
comptime LATENT_HEIGHT = 3
comptime LATENT_WIDTH = 3


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


def _max_abs(got: List[Float32], want: List[Float32]) raises -> Float32:
    if len(got) != len(want):
        raise Error("length mismatch")
    var worst = Float32(0.0)
    for i in range(len(got)):
        var diff = got[i] - want[i]
        var mag = -diff if diff < 0.0 else diff
        if mag > worst:
            worst = mag
    return worst


def _tensor_names() -> List[String]:
    var names = List[String]()
    names.append(String("post_quant_conv.weight"))
    names.append(String("post_quant_conv.bias"))
    names.append(String("decoder.proj_in.weight"))
    names.append(String("decoder.proj_in.bias"))
    names.append(String("decoder.register_tokens"))
    names.append(String("decoder.norm_out.weight"))
    names.append(String("decoder.norm_out.bias"))
    names.append(String("decoder.proj_out.weight"))
    names.append(String("decoder.proj_out.bias"))
    for layer in range(2):
        var p = String("decoder.transformer_blocks.") + String(layer)
        names.append(p + ".norm1.weight")
        names.append(p + ".norm2.weight")
        names.append(p + ".scale1")
        names.append(p + ".scale2")
        names.append(p + ".attn.to_q.weight")
        names.append(p + ".attn.to_q.bias")
        names.append(p + ".attn.to_k.weight")
        names.append(p + ".attn.to_k.bias")
        names.append(p + ".attn.to_v.weight")
        names.append(p + ".attn.to_v.bias")
        names.append(p + ".attn.to_out.0.weight")
        names.append(p + ".attn.to_out.0.bias")
        names.append(p + ".ff.net.0.proj.weight")
        names.append(p + ".ff.net.0.proj.bias")
        names.append(p + ".ff.net.2.weight")
        names.append(p + ".ff.net.2.bias")
    return names^


def main() raises:
    print("MiniMax-H3 video ViT decoder parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))
    var checks = 0
    var failures = 0

    print("")
    print("[1] rotary construction")
    # head_dim 8, ratio 0.75 -> dim 6, step 6/6 = 1 -> exactly ONE frequency,
    # so the rotary width is 2*3*1 = 6 against a head dim of 8.
    var inv_freq = video_rope_inv_freq(6, Float64(100.0))
    checks += 1
    if len(inv_freq) == 1 and inv_freq[0] == Float32(1.0):
        print("  ok   inv_freq has 1 entry = 1.0 (rotary width 6 < head_dim 8)")
    else:
        failures += 1
        print("  FAIL inv_freq", len(inv_freq), "entries")

    var grid = video_position_grid(LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, 3)
    checks += 1
    # First voxel: (2*0.5/2 - 1, 2*0.5/3 - 1, 2*0.5/3 - 1) = (-0.5, -2/3, -2/3);
    # the three suffix rows are zero.
    var first_ok = grid[0] == -0.5
    var suffix_ok = grid[len(grid) - 1] == 0.0 and grid[len(grid) - 9] == 0.0
    if first_ok and suffix_ok:
        print("  ok   position grid normalized to [-1, 1), suffix at zero")
    else:
        failures += 1
        print("  FAIL position grid: first", grid[0], "last", grid[len(grid) - 1])

    print("")
    print("[2] decoded pixels")
    var names = _tensor_names()
    var values = List[List[Float32]]()
    for i in range(len(names)):
        values.append(_load_f32(st, String("w.") + names[i]))
    var weights = MiniMaxH3VideoWeights(names^, values^)

    var config = MiniMaxH3VideoDecoderConfig(
        4,      # latent_channels
        3,      # out_channels
        2,      # num_layers
        2,      # num_attention_heads
        8,      # attention_head_dim   -> dim 16
        2,      # num_register_tokens  -> 3 suffix rows with the zero token
        2,      # ffn_mult
        4,      # patch_size           (spatial compression ratio)
        4,      # patch_size_t         (temporal compression ratio)
        Float64(100.0),
        Float64(0.75),
        Float32(1.0e-5),
    )

    var pixels = minimax_h3_video_decode(
        weights, config, _load_f32(st, "in.latents"),
        LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH,
    )
    var want = _load_f32(st, "out.pixels")
    checks += 1
    if len(pixels) != len(want):
        failures += 1
        print("  FAIL pixel count", len(pixels), "!=", len(want))
    else:
        var worst = _max_abs(pixels, want)
        if worst <= TOL:
            print("  ok   pixels", len(pixels), "values, max_abs", worst)
        else:
            failures += 1
            print("  FAIL pixels max_abs", worst)

    print("")
    if failures == 0:
        print("PASS:", checks, "checks")
    else:
        print("FAIL:", failures, "of", checks, "checks")
        raise Error("MiniMax-H3 video decoder parity gate failed")
