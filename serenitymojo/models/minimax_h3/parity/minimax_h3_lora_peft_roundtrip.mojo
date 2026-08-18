# MiniMax-H3 external LoRA contract gate.
#
# Proves the exact trainer save path's host-F32 -> BF16 PEFT writer, canonical
# H3 key mapping, H3 overlay reload values/scale, and legacy Musubi/Kohya
# down/up/alpha compatibility on small deterministic adapters. It also gates
# the shared training-store QKV deinterleave, proves canonical QKV B remains
# unchanged, proves FC1 B swaps [gate;value] -> [value;gate], and checks the
# activation-delta math at released output widths.
#
# Compile in the rootless, hard-capped transient build service:
#   MEM_MAX=24G SWAP_MAX=2G scripts/mem_safe_runtime.sh \
#     pixi run mojo build --optimization-level 2 -j 1 -I . \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_lora_peft_roundtrip.mojo \
#     -o /tmp/minimax_h3_lora_peft_roundtrip

from std.collections import List
from std.sys import argv
from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.lora_save import (
    F32NamedLora,
    NamedLora,
    save_lora_peft_host_f32,
    save_lora_serenity_trainer,
)
from serenitymojo.models.minimax_h3.h3_lora_format import (
    h3_lora_peft_prefix,
    h3_lora_legacy_prefix,
)
from serenitymojo.models.minimax_h3.h3_lora_overlay import H3LoraOverlay
from serenitymojo.models.minimax_h3.h3_qkv_layout import (
    h3_qkv_deinterleave_rows,
)
from serenitymojo.models.minimax_h3.h3_train_fence_policy import (
    h3_fp8_forward_should_fence,
    h3_fp8_backward_should_fence,
)

comptime PEFT_OUT = "/tmp/minimax_h3_lora_peft_roundtrip.safetensors"
comptime LEGACY_OUT = "/tmp/minimax_h3_lora_legacy_roundtrip.safetensors"
comptime LAYOUT_OUT = "/tmp/minimax_h3_lora_layout_roundtrip.safetensors"
comptime H3_INNER = 56 * 128
comptime H3_FFN = 14336


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _values(base: Float32) -> Tuple[List[Float32], List[Float32]]:
    var a: List[Float32] = [
        base + 0.0, base + 0.125, base + 0.25,
        base + 0.375, base + 0.5, base + 0.625,
    ]  # [rank=2, in=3]
    var b: List[Float32] = [
        base - 0.0, base - 0.0625,
        base - 0.125, base - 0.1875,
        base - 0.25, base - 0.3125,
        base - 0.375, base - 0.4375,
    ]  # [out=4, rank=2]
    return (a^, b^)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _adapter(base: Float32, scale: Float32) -> LoraAdapter:
    var values = _values(base)
    return LoraAdapter(
        values[0].copy(), values[1].copy(), 2, 3, 4, scale,
        _zeros(6), _zeros(6), _zeros(8), _zeros(8),
    )


def _assert_bf16_roundtrip(
    label: String, got: List[Float32], expected: List[Float32],
) raises:
    _require(len(got) == len(expected), label + ": length mismatch")
    var max_abs = Float32(0.0)
    for i in range(len(got)):
        var want = expected[i].cast[DType.bfloat16]().cast[DType.float32]()
        var diff = got[i] - want
        var ad = diff if diff >= 0.0 else -diff
        if ad > max_abs:
            max_abs = ad
    _require(max_abs == Float32(0.0), label + ": not BF16-exact")
    print("  ", label, " BF16-exact, n=", len(got))


def _layout_a() -> List[Float32]:
    # x=[2,0,0], A below -> hidden=[2,0]. With multiplier 0.5 and the
    # default alpha/rank=1, activation_delta must equal B's first column.
    return [
        Float32(1.0), Float32(0.0), Float32(0.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
    ]


def _layout_b(rows: Int, offset: Int) -> List[Float32]:
    var out = List[Float32](capacity=rows * 2)
    for row in range(rows):
        # Small integers are exactly representable in BF16.
        out.append(Float32(offset + row % 16))
        out.append(Float32(offset + 32 + row % 16))
    return out^


def _assert_selected_rows(
    label: String, got: List[Float32], expected_b: List[Float32],
    rows: List[Int],
) raises:
    for row in rows:
        var want = expected_b[row * 2].cast[DType.bfloat16]().cast[DType.float32]()
        _require(got[row] == want, label + ": activation row mismatch at " + String(row))
    print("  ", label, " selected activation rows exact")


def _gate_training_qkv_layout(ctx: DeviceContext) raises:
    # Tiny [heads=2, head_dim=2] row labels. Raw order is
    # h0(q0,q1,k0,k1,v0,v1), h1(...); expected is q_all,k_all,v_all.
    var raw_values = List[Float32]()
    for row in range(12):
        raw_values.append(Float32(row))
    var raw_shape: List[Int] = [12, 1]
    var raw = Tensor.from_host(raw_values, raw_shape^, STDtype.BF16, ctx)
    var transformed = h3_qkv_deinterleave_rows(raw, 2, 2, ctx).to_host(ctx)
    var expected: List[Float32] = [
        0.0, 1.0, 6.0, 7.0,
        2.0, 3.0, 8.0, 9.0,
        4.0, 5.0, 10.0, 11.0,
    ]
    _require(len(transformed) == len(expected), String("QKV layout length mismatch"))
    for i in range(len(expected)):
        _require(transformed[i] == expected[i], String("QKV layout row mismatch"))
    print("PASS: H3 training QKV rows deinterleave to [q_all;k_all;v_all]")


def _gate_fence_policy() raises:
    comptime blocks = 50
    comptime resident = 42
    var forward_fences = 0
    var backward_fences = 0
    for block in range(blocks):
        var ff = h3_fp8_forward_should_fence(block, resident)
        var bf = h3_fp8_backward_should_fence(block, resident)
        if ff:
            forward_fences += 1
        if bf:
            backward_fences += 1
        if block >= resident:
            _require(ff and bf, String("streamed tail block missing required fence"))
        elif block % 8 != 0:
            _require(not bf, String("resident backward unexpectedly serialized"))
    _require(forward_fences == 14, String("unexpected forward fence count"))
    _require(backward_fences == 14, String("unexpected backward fence count"))
    print("PASS: H3 FP8 stack fences every tail block and pipelines residents (14/50)")


def _gate_runtime_layout_and_activation(ctx: DeviceContext) raises:
    var a = _layout_a()
    var qkv_b = _layout_b(3 * H3_INNER, 1)
    var fc1_b = _layout_b(2 * H3_FFN, 65)
    var adapters = List[F32NamedLora]()
    adapters.append(F32NamedLora(
        h3_lora_peft_prefix(1, 0), a.copy(), qkv_b.copy(),
        2, 3, 3 * H3_INNER,
    ))
    adapters.append(F32NamedLora(
        h3_lora_peft_prefix(1, 2), a.copy(), fc1_b.copy(),
        2, 3, 2 * H3_FFN,
    ))
    _require(
        save_lora_peft_host_f32(adapters, String(LAYOUT_OUT)) == 2,
        String("expected two layout PEFT pairs"),
    )
    var overlay = H3LoraOverlay.load(String(LAYOUT_OUT), Float32(0.5), ctx)
    var qidx = 1 * 4 + 0
    var fidx = 1 * 4 + 2
    _require(overlay.has(1, 0) and overlay.has(1, 2), String("layout overlay reload failed"))
    _assert_bf16_roundtrip(
        "QKV B canonical/runtime unchanged",
        overlay.runtime_up[qidx].value()[].to_host(ctx), qkv_b,
    )

    var fc1_expected = List[Float32](capacity=len(fc1_b))
    for row in range(H3_FFN, 2 * H3_FFN):
        fc1_expected.append(fc1_b[row * 2])
        fc1_expected.append(fc1_b[row * 2 + 1])
    for row in range(H3_FFN):
        fc1_expected.append(fc1_b[row * 2])
        fc1_expected.append(fc1_b[row * 2 + 1])
    _assert_bf16_roundtrip(
        "FC1 B runtime half-swap",
        overlay.runtime_up[fidx].value()[].to_host(ctx), fc1_expected,
    )

    var x_values: List[Float32] = [2.0, 0.0, 0.0]
    var x_shape: List[Int] = [1, 3]
    var x = Tensor.from_host(x_values, x_shape^, STDtype.BF16, ctx)
    var q_delta = overlay.activation_delta(1, 0, x, ctx).to_host(ctx)
    var f_delta = overlay.activation_delta(1, 2, x, ctx).to_host(ctx)
    var q_rows: List[Int] = [0, 1, 255, 256, 3 * H3_INNER - 1]
    var f_rows: List[Int] = [0, 1, H3_FFN - 1, H3_FFN, 2 * H3_FFN - 1]
    _assert_selected_rows("QKV", q_delta, qkv_b, q_rows)
    _assert_selected_rows("FC1", f_delta, fc1_expected, f_rows)
    print("PASS: H3 canonical QKV/FC1 B layout + activation LoRA math")


def main() raises:
    _require(
        h3_lora_peft_prefix(0, 0)
        == String("diffusion_model.blocks.0.attn.qkv_proj"),
        String("qkv PEFT prefix mismatch"),
    )
    _require(
        h3_lora_peft_prefix(7, 1)
        == String("diffusion_model.blocks.7.attn.out_proj"),
        String("out PEFT prefix mismatch"),
    )
    _require(
        h3_lora_peft_prefix(12, 2)
        == String("diffusion_model.blocks.12.mlp.fc1"),
        String("fc1 PEFT prefix mismatch"),
    )
    _require(
        h3_lora_peft_prefix(49, 3)
        == String("diffusion_model.blocks.49.mlp.fc2"),
        String("fc2 PEFT prefix mismatch"),
    )

    var source = _values(Float32(0.3))
    var host_adapters = List[F32NamedLora]()
    host_adapters.append(F32NamedLora(
        h3_lora_peft_prefix(0, 1),
        source[0].copy(), source[1].copy(), 2, 3, 4,
    ))
    var saved = save_lora_peft_host_f32(host_adapters, String(PEFT_OUT))
    _require(saved == 1, String("expected one saved PEFT pair"))

    var st = SafeTensors.open(String(PEFT_OUT))
    var pfx = h3_lora_peft_prefix(0, 1)
    var akey = pfx + ".lora_A.weight"
    var bkey = pfx + ".lora_B.weight"
    _require(len(st.tensors) == 2, String("PEFT file must contain exactly A/B"))
    _require(st.has_tensor(akey) and st.has_tensor(bkey), String("canonical PEFT keys missing"))
    _require(not st.has_tensor(h3_lora_legacy_prefix(0, 1) + ".lora_down.weight"), String("legacy key leaked into new save"))
    var ai = st.tensor_info(akey)
    var bi = st.tensor_info(bkey)
    _require(ai.dtype == STDtype.BF16 and bi.dtype == STDtype.BF16, String("PEFT A/B must be BF16"))
    _require(len(ai.shape) == 2 and ai.shape[0] == 2 and ai.shape[1] == 3, String("PEFT A shape mismatch"))
    _require(len(bi.shape) == 2 and bi.shape[0] == 4 and bi.shape[1] == 2, String("PEFT B shape mismatch"))

    var args = argv()
    if len(args) > 1 and String(args[1]) == String("--host-only"):
        print("PASS: H3 host-direct BF16 PEFT save + canonical key inventory")
        return

    var ctx = DeviceContext()
    _gate_fence_policy()
    _gate_training_qkv_layout(ctx)
    var overlay = H3LoraOverlay.load(String(PEFT_OUT), Float32(0.5), ctx)
    _require(overlay.adapters == 1 and overlay.has(0, 1), String("PEFT overlay reload failed"))
    _require(overlay.scale[1] == Float32(0.5), String("PEFT default alpha/rank scale mismatch"))
    _assert_bf16_roundtrip("PEFT A", overlay.down[1].value()[].to_host(ctx), source[0])
    _assert_bf16_roundtrip("PEFT B", overlay.up[1].value()[].to_host(ctx), source[1])

    # Existing H3 checkpoints remain valid. The legacy writer stores
    # alpha=scale*rank, so scale 0.75 and multiplier 0.5 must reload as 0.375.
    var legacy = List[NamedLora]()
    legacy.append(NamedLora(
        h3_lora_legacy_prefix(0, 1), _adapter(Float32(-0.2), Float32(0.75)),
    ))
    _ = save_lora_serenity_trainer(legacy, String(LEGACY_OUT), ctx)
    var old_overlay = H3LoraOverlay.load(String(LEGACY_OUT), Float32(0.5), ctx)
    _require(old_overlay.adapters == 1 and old_overlay.has(0, 1), String("legacy overlay reload failed"))
    _require(old_overlay.scale[1] == Float32(0.375), String("legacy alpha/rank scale mismatch"))

    _gate_runtime_layout_and_activation(ctx)
    print("PASS: H3 canonical PEFT save/reload + legacy load compatibility")
