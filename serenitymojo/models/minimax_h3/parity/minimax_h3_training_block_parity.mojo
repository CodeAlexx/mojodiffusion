# MiniMax-H3 reduced F32 LoRA block forward/backward parity gate.
#
# Oracle fixture:
#   uv run --with torch --with safetensors --with numpy \
#     scripts/minimax_h3_training_block_oracle.py
#
# Geometry keeps the released H3 head count H=56, but deliberately reduces
# S=3, Dh=4, D=8, F=12, rank=2. This gates topology and every LoRA gradient;
# it is NOT released-geometry, BF16, device, stack, optimizer or trainer proof.
#
# Build/run (lead owns serial Mojo compilation):
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs -Xlinker -lm \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_training_block_parity.mojo \
#     -o output/checks/minimax_h3_training_block_parity
#   output/checks/minimax_h3_training_block_parity

from std.collections import List
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.parity.training_block_reference import (
    MiniMaxH3BlockModulation,
    MiniMaxH3LoraAdapter,
    MiniMaxH3TrainingBlockConfig,
    MiniMaxH3TrainingBlockLora,
    MiniMaxH3TrainingBlockWeights,
    minimax_h3_training_block_reference_backward,
    minimax_h3_training_block_reference_forward,
)


comptime REF = "serenitymojo/models/minimax_h3/parity/fixtures/minimax_h3_training_block.safetensors"
comptime S = 3
comptime H = 56
comptime DH = 4
comptime D = 8
comptime FF = 12
comptime ROT = 2
comptime RANK = 2
comptime MAX_ABS_TOL = Float32(4.0e-4)
comptime COS_TOL = Float32(0.9999)


def _load(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("expected F32 fixture tensor: ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _check(label: String, got: List[Float32], want: List[Float32]) raises -> Bool:
    if len(got) != len(want):
        print("  FAIL", label, "length", len(got), "!=", len(want))
        return False
    var dot = Float64(0.0)
    var ng = Float64(0.0)
    var nw = Float64(0.0)
    var worst = Float32(0.0)
    var worst_i = 0
    for i in range(len(got)):
        var g = Float64(got[i])
        var w = Float64(want[i])
        dot += g * w
        ng += g * g
        nw += w * w
        var d = got[i] - want[i]
        var ad = -d if d < 0.0 else d
        if ad > worst:
            worst = ad
            worst_i = i
    var cos = Float32(dot / sqrt(ng * nw))
    var ok = cos >= COS_TOL and worst <= MAX_ABS_TOL
    if ok:
        print("  ok  ", label, "cos", cos, "max_abs", worst)
    else:
        print("  FAIL", label, "cos", cos, "max_abs", worst, "at", worst_i)
    return ok


def _adapter(ref st: SafeTensors, name: String, inf: Int, outf: Int) raises -> MiniMaxH3LoraAdapter:
    var multiplier = _load(st, "meta.lora_multiplier")[0]
    var alpha = _load(st, "meta.lora_alpha")[0]
    var rank = _load(st, "meta.lora_rank")[0]
    var scale = multiplier * alpha / rank
    var expected = _load(st, "meta.lora_scale")[0]
    if abs(scale - expected) > 1.0e-7:
        raise Error("MiniMax-H3 LoRA multiplier*alpha/rank mapping mismatch")
    return MiniMaxH3LoraAdapter(
        _load(st, String("lora.") + name + ".a"),
        _load(st, String("lora.") + name + ".b"),
        RANK, inf, outf, scale,
    )


def _check_lora_scale_mapping() raises:
    # Musubi default alpha=rank and multiplier=1.0 => 1.0.
    var default_scale = Float32(1.0) * Float32(RANK) / Float32(RANK)
    if abs(default_scale - 1.0) > 1.0e-7:
        raise Error("MiniMax-H3 default LoRA scale mapping failed")
    # Non-default fixture case: multiplier=1.4, alpha=1, rank=2 => 0.7.
    var configured_scale = Float32(1.4) * Float32(1.0) / Float32(2.0)
    if abs(configured_scale - 0.7) > 1.0e-7:
        raise Error("MiniMax-H3 configured LoRA scale mapping failed")


def main() raises:
    print("MiniMax-H3 reduced F32 training-block parity")
    print("  H=56 released head count; S/Dh/D/F/rank are reduced")
    print("  fused qkv=[all-q;all-k;all-v], raw fc1=[gate;value]")
    print("  oracle musubi-tuner@b8717864713c9e4e7ef3d56eba1fc695a9b626a5")
    _check_lora_scale_mapping()
    var st = SafeTensors.open(String(REF))
    var cfg = MiniMaxH3TrainingBlockConfig(H, DH, D, FF, ROT, 1.0e-5, 1.0e-5)
    var weights = MiniMaxH3TrainingBlockWeights(
        _load(st, "w.norm1"), _load(st, "w.qkv"),
        _load(st, "w.q_norm"), _load(st, "w.k_norm"),
        _load(st, "w.out_proj"), _load(st, "w.norm2"),
        _load(st, "w.fc1"), _load(st, "w.fc2"),
    )
    var mod = MiniMaxH3BlockModulation(
        _load(st, "mod.shift_msa"), _load(st, "mod.scale_msa"),
        _load(st, "mod.gate_msa"), _load(st, "mod.shift_mlp"),
        _load(st, "mod.scale_mlp"), _load(st, "mod.gate_mlp"),
    )
    var lora = MiniMaxH3TrainingBlockLora(
        _adapter(st, "qkv", D, 3 * H * DH),
        _adapter(st, "out_proj", H * DH, D),
        _adapter(st, "fc1", D, 2 * FF),
        _adapter(st, "fc2", FF, D),
    )
    var x = _load(st, "in.x")
    var cos = _load(st, "in.cos")
    var sin = _load(st, "in.sin")

    var y = minimax_h3_training_block_reference_forward(
        x, weights, mod, lora, cos, sin, S, cfg
    )
    # Structs are copyable because this bounded host gate recomputes the block;
    # the production stack will stream/recompute device tensors instead.
    var b = minimax_h3_training_block_reference_backward(
        _load(st, "in.dy"), x, weights, mod, lora, cos, sin, S, cfg
    )

    var ok = True
    ok = _check("forward.y", y, _load(st, "out.y")) and ok
    ok = _check("grad.x", b.d_x, _load(st, "grad.x")) and ok
    ok = _check("grad.qkv.A", b.qkv.d_a, _load(st, "grad.qkv.a")) and ok
    ok = _check("grad.qkv.B", b.qkv.d_b, _load(st, "grad.qkv.b")) and ok
    ok = _check("grad.out_proj.A", b.out_proj.d_a, _load(st, "grad.out_proj.a")) and ok
    ok = _check("grad.out_proj.B", b.out_proj.d_b, _load(st, "grad.out_proj.b")) and ok
    ok = _check("grad.fc1.A", b.fc1.d_a, _load(st, "grad.fc1.a")) and ok
    ok = _check("grad.fc1.B", b.fc1.d_b, _load(st, "grad.fc1.b")) and ok
    ok = _check("grad.fc2.A", b.fc2.d_a, _load(st, "grad.fc2.a")) and ok
    ok = _check("grad.fc2.B", b.fc2.d_b, _load(st, "grad.fc2.b")) and ok
    if ok:
        print("PASS: MiniMax-H3 block forward, d_x and all LoRA A/B grads match")
    else:
        raise Error("MiniMax-H3 training-block parity failed")
