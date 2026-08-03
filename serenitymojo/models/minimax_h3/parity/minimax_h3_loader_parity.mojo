# serenitymojo/models/minimax_h3/parity/minimax_h3_loader_parity.mojo
#
# MiniMax-H3 loader parity gate: ORIGINAL checkpoint bytes in, matching
# velocities out — with no checkpoint in existence.
#
# The oracle synthesizes an original-layout checkpoint by running the diffusers
# converter's transforms BACKWARDS on the tiny fixture model: fusing q/k/v back
# into per-head interleave, swapping `fc1` to `[gate; value]`, and using the
# original key spelling. This gate then loads THAT, applies
# `serenitymojo/models/minimax_h3/loader.mojo`, feeds the result to the unit-9
# forward, and compares against the same reference velocities unit 9 matched.
#
# So a pass means the whole chain holds: checkpoint spelling -> tensor
# placement -> arithmetic -> output. The two halves are gated separately as
# well, because "the output is right" is a weak signal when one error can mask
# another:
#   [1] the dtype rule (which 12 keys are float32)
#   [2] key renaming
#   [3] the QKV de-interleave and split, against torch's own to_q/to_k/to_v
#   [4] the fc1 half-swap, against torch's own ff.net.0.proj
#   [5] end to end
#
# Oracle: python3 scripts/minimax_h3_block_oracle.py
# Run:
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_loader_parity.mojo \
#     -o output/checks/minimax_h3_loader_parity \
#   && output/checks/minimax_h3_loader_parity

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.block_forward import (
    MiniMaxH3BlockConfig,
    MiniMaxH3Weights,
    minimax_h3_forward,
)
from serenitymojo.models.minimax_h3.loader import (
    minimax_h3_deinterleave_qkv,
    minimax_h3_is_dropped_key,
    minimax_h3_is_fp32_key,
    minimax_h3_rename_key,
    minimax_h3_split_qkv,
    minimax_h3_swap_fc1,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_block/block_ref.safetensors"
comptime TOL = Float32(2.0e-5)

comptime HEADS = 2
comptime HEAD_DIM = 16
comptime HIDDEN = 24
comptime FFN = 32


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


def _load_f64_widened(ref st: SafeTensors, name: String) raises -> List[Float64]:
    var values = _load_f32(st, name)
    var out = List[Float64]()
    for i in range(len(values)):
        out.append(Float64(values[i]))
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


struct Report(Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def truth(mut self, label: String, condition: Bool):
        self.checks += 1
        if condition:
            print("  ok  ", label)
        else:
            self.failures += 1
            print("  FAIL", label)

    def close(mut self, label: String, got: List[Float32], want: List[Float32]) raises:
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var worst = _max_abs(got, want)
        if worst <= TOL:
            print("  ok  ", label, len(got), "values, max_abs", worst)
        else:
            self.failures += 1
            print("  FAIL", label, "max_abs", worst)


def _add(
    mut names: List[String],
    mut values: List[List[Float32]],
    ref st: SafeTensors,
    original: String,
) raises:
    """Rename an ORIGINAL key and load its tensor under the new name."""
    names.append(minimax_h3_rename_key(original))
    values.append(_load_f32(st, String("orig.") + original))


def main() raises:
    print("MiniMax-H3 loader parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))
    var report = Report()

    print("")
    print("[1] the mixed-precision rule: which original keys are float32")
    report.truth("video_patch_proj.weight is fp32", minimax_h3_is_fp32_key(String("video_patch_proj.weight")))
    report.truth("audio_patch_proj.bias is fp32", minimax_h3_is_fp32_key(String("audio_patch_proj.bias")))
    report.truth("time_embedder.proj_in.weight is fp32", minimax_h3_is_fp32_key(String("time_embedder.proj_in.weight")))
    report.truth("final_layer.video_out.weight is fp32", minimax_h3_is_fp32_key(String("final_layer.video_out.weight")))
    report.truth("final_layer.audio_out.bias is fp32", minimax_h3_is_fp32_key(String("final_layer.audio_out.bias")))
    # The trap: adaLN sits next to the fp32 heads but is bfloat16 like the stack.
    report.truth(
        "final_layer.adaln_proj is NOT fp32",
        not minimax_h3_is_fp32_key(String("final_layer.adaln_proj.linear.weight")),
    )
    report.truth(
        "blocks.0.adaln_proj is NOT fp32",
        not minimax_h3_is_fp32_key(String("blocks.0.adaln_proj.linear.weight")),
    )
    report.truth("condition_proj is NOT fp32", not minimax_h3_is_fp32_key(String("condition_proj.weight")))
    report.truth("rope.inv_freq is dropped", minimax_h3_is_dropped_key(String("rope.inv_freq")))

    print("")
    print("[2] key renaming")
    report.truth(
        "blocks.0.norm1.weight",
        minimax_h3_rename_key(String("blocks.0.norm1.weight")) == String("transformer_blocks.0.norm1.weight"),
    )
    report.truth(
        "token_refiner.blocks.1.mlp.fc2.weight",
        minimax_h3_rename_key(String("token_refiner.blocks.1.mlp.fc2.weight"))
        == String("token_refiner.refiner_blocks.1.ff.net.2.weight"),
    )
    report.truth(
        "blocks.0.attn.out_proj.weight",
        minimax_h3_rename_key(String("blocks.0.attn.out_proj.weight"))
        == String("transformer_blocks.0.attn.to_out.0.weight"),
    )
    report.truth(
        "final_layer.adaln_proj.linear.bias",
        minimax_h3_rename_key(String("final_layer.adaln_proj.linear.bias")) == String("norm_out.linear.bias"),
    )
    report.truth(
        "video_patch_proj.weight",
        minimax_h3_rename_key(String("video_patch_proj.weight")) == String("proj_in.weight"),
    )
    report.truth(
        "time_embedder.proj_out.bias",
        minimax_h3_rename_key(String("time_embedder.proj_out.bias")) == String("time_embedder.linear_2.bias"),
    )

    print("")
    print("[3] QKV de-interleave + split, against torch's own to_q/to_k/to_v")
    for layer in range(2):
        var fused = _load_f32(
            st, String("orig.blocks.") + String(layer) + ".attn.qkv_proj.weight"
        )
        var reordered = minimax_h3_deinterleave_qkv(fused, HEADS, HEAD_DIM, HIDDEN)
        var names = [String("to_q"), String("to_k"), String("to_v")]
        for part in range(3):
            report.close(
                String("blocks.") + String(layer) + "." + names[part],
                minimax_h3_split_qkv(reordered, HEADS, HEAD_DIM, HIDDEN, part),
                _load_f32(
                    st,
                    String("w.transformer_blocks.") + String(layer) + ".attn."
                    + names[part] + ".weight",
                ),
            )

    print("")
    print("[4] fc1 half-swap, against torch's own ff.net.0.proj")
    for layer in range(2):
        report.close(
            String("blocks.") + String(layer) + ".fc1 -> ff.net.0.proj",
            minimax_h3_swap_fc1(
                _load_f32(st, String("orig.blocks.") + String(layer) + ".mlp.fc1.weight"),
                FFN, HIDDEN,
            ),
            _load_f32(
                st, String("w.transformer_blocks.") + String(layer) + ".ff.net.0.proj.weight"
            ),
        )

    print("")
    print("[5] end to end: original bytes -> loader -> forward -> velocities")
    var names = List[String]()
    var values = List[List[Float32]]()

    _add(names, values, st, String("video_patch_proj.weight"))
    _add(names, values, st, String("video_patch_proj.bias"))
    _add(names, values, st, String("audio_patch_proj.weight"))
    _add(names, values, st, String("audio_patch_proj.bias"))
    _add(names, values, st, String("condition_proj.weight"))
    _add(names, values, st, String("condition_proj.bias"))
    _add(names, values, st, String("time_embedder.proj_in.weight"))
    _add(names, values, st, String("time_embedder.proj_in.bias"))
    _add(names, values, st, String("time_embedder.proj_out.weight"))
    _add(names, values, st, String("time_embedder.proj_out.bias"))
    _add(names, values, st, String("token_refiner.final_norm.weight"))
    _add(names, values, st, String("final_layer.norm.weight"))
    _add(names, values, st, String("final_layer.adaln_proj.linear.weight"))
    _add(names, values, st, String("final_layer.adaln_proj.linear.bias"))
    _add(names, values, st, String("final_layer.video_out.weight"))
    _add(names, values, st, String("final_layer.video_out.bias"))
    _add(names, values, st, String("final_layer.audio_out.weight"))
    _add(names, values, st, String("final_layer.audio_out.bias"))

    for group in range(2):
        var prefix = String("blocks.") if group == 0 else String("token_refiner.blocks.")
        for layer in range(2):
            var base = prefix + String(layer)
            _add(names, values, st, base + ".norm1.weight")
            _add(names, values, st, base + ".norm2.weight")
            _add(names, values, st, base + ".attn.q_norm.weight")
            _add(names, values, st, base + ".attn.k_norm.weight")
            _add(names, values, st, base + ".attn.out_proj.weight")
            _add(names, values, st, base + ".mlp.fc2.weight")
            if group == 0:
                _add(names, values, st, base + ".adaln_proj.linear.weight")
                _add(names, values, st, base + ".adaln_proj.linear.bias")

            # The two transformed tensors.
            var fused = _load_f32(st, String("orig.") + base + ".attn.qkv_proj.weight")
            var reordered = minimax_h3_deinterleave_qkv(fused, HEADS, HEAD_DIM, HIDDEN)
            var stem = (
                String("transformer_blocks.") + String(layer)
                if group == 0
                else String("token_refiner.refiner_blocks.") + String(layer)
            )
            names.append(stem + ".attn.to_q.weight")
            values.append(minimax_h3_split_qkv(reordered, HEADS, HEAD_DIM, HIDDEN, 0))
            names.append(stem + ".attn.to_k.weight")
            values.append(minimax_h3_split_qkv(reordered, HEADS, HEAD_DIM, HIDDEN, 1))
            names.append(stem + ".attn.to_v.weight")
            values.append(minimax_h3_split_qkv(reordered, HEADS, HEAD_DIM, HIDDEN, 2))
            names.append(stem + ".ff.net.0.proj.weight")
            values.append(
                minimax_h3_swap_fc1(
                    _load_f32(st, String("orig.") + base + ".mlp.fc1.weight"), FFN, HIDDEN
                )
            )

    var weights = MiniMaxH3Weights(names^, values^)
    var config = MiniMaxH3BlockConfig(
        HEADS, HEAD_DIM, HIDDEN, 2, 2, FFN, 4, 6, 8, 8, 16, 2,
        Float32(1.0e-5), Float32(1.0e-5), Float32(1.0e-5),
    )

    var out = minimax_h3_forward(
        weights, config,
        _load_f32(st, "in.hidden_states"),
        _load_f32(st, "in.audio_hidden_states"),
        _load_f32(st, "in.encoder_hidden_states"),
        _load_f32(st, "in.timestep"),
        _load_i64(st, "in.timestep_indices"),
        _load_i64(st, "in.token_tags"),
        _load_f64_widened(st, "in.position_ids"),
        _load_i64(st, "in.video_indices"),
        _load_i64(st, "in.audio_indices"),
        _load_i64(st, "in.text_indices"),
    )
    report.close(String("sample from ORIGINAL layout"), out.sample, _load_f32(st, "out.sample"))
    report.close(
        String("audio_sample from ORIGINAL layout"),
        out.audio_sample,
        _load_f32(st, "out.audio_sample"),
    )

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("MiniMax-H3 loader parity gate failed")
