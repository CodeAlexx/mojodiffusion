# ernie_lora_ot_save_key_smoke.mojo -- ERNIE OneTrainer save-key contract.
#
# Tiny structural gate: build one synthetic ERNIE LoRA layer, save through
# save_ernie_lora, then inspect the safetensors header and raw OT alpha/down/up
# keys. No model checkpoint, transformer math, or sampler run.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.ernie.ernie_stack_lora import (
    build_ernie_lora_set,
    ernie_lora_prefixes,
    load_ernie_lora_resume,
    save_ernie_lora,
)
from serenitymojo.training.lora_save import _read_f32


comptime OUT = "/tmp/ernie_lora_ot_save_key_smoke.safetensors"
comptime SMOKE_NUM_LAYERS = 1
comptime SMOKE_HIDDEN = 8
comptime SMOKE_FFN = 16
comptime SMOKE_RANK = 2
comptime SMOKE_ADAPTERS = 7
comptime SMOKE_TENSORS = SMOKE_ADAPTERS * 3


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _require_rank2(st: SafeTensors, key: String, dim0: Int, dim1: Int) raises:
    _require(key in st.tensors, String("missing key ") + key)
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.BF16, String("expected BF16 for ") + key)
    _require(len(info.shape) == 2, String("expected rank-2 tensor for ") + key)
    _require(
        info.shape[0] == dim0 and info.shape[1] == dim1,
        String("shape mismatch for ") + key,
    )


def _require_alpha(st: SafeTensors, key: String, expected: Float32, ctx: DeviceContext) raises:
    _require(key in st.tensors, String("missing key ") + key)
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.BF16, String("expected BF16 alpha for ") + key)
    _require(len(info.shape) == 0, String("expected scalar alpha for ") + key)
    var alpha = _read_f32(st, key, ctx)
    _require(len(alpha) == 1, String("expected one alpha value for ") + key)
    var err = alpha[0] - expected
    if err < Float32(0.0):
        err = -err
    _require(err == Float32(0.0), String("alpha value mismatch for ") + key)


def _contains(values: List[String], expected: String) -> Bool:
    for i in range(len(values)):
        if values[i] == expected:
            return True
    return False


def _require_prefix(values: List[String], idx: Int, expected: String) raises:
    _require(idx < len(values), String("missing prefix index ") + String(idx))
    _require(
        values[idx] == expected,
        String("prefix[") + String(idx) + String("] mismatch: ")
        + values[idx] + String(" != ") + expected,
    )


def _append_expected(mut names: List[String], prefix: String):
    names.append(prefix + String(".alpha"))
    names.append(prefix + String(".lora_down.weight"))
    names.append(prefix + String(".lora_up.weight"))


def _expected_names() -> List[String]:
    var names = List[String]()
    _append_expected(names, String("transformer.layers.0.self_attention.to_q"))
    _append_expected(names, String("transformer.layers.0.self_attention.to_k"))
    _append_expected(names, String("transformer.layers.0.self_attention.to_v"))
    _append_expected(names, String("transformer.layers.0.self_attention.to_out.0"))
    _append_expected(names, String("transformer.layers.0.mlp.gate_proj"))
    _append_expected(names, String("transformer.layers.0.mlp.up_proj"))
    _append_expected(names, String("transformer.layers.0.mlp.linear_fc2"))
    return names^


def _require_exact_prefixes(values: List[String]) raises:
    _require(len(values) == SMOKE_ADAPTERS, String("expected 7 Ernie prefixes"))
    _require_prefix(values, 0, String("transformer.layers.0.self_attention.to_q"))
    _require_prefix(values, 1, String("transformer.layers.0.self_attention.to_k"))
    _require_prefix(values, 2, String("transformer.layers.0.self_attention.to_v"))
    _require_prefix(values, 3, String("transformer.layers.0.self_attention.to_out.0"))
    _require_prefix(values, 4, String("transformer.layers.0.mlp.gate_proj"))
    _require_prefix(values, 5, String("transformer.layers.0.mlp.up_proj"))
    _require_prefix(values, 6, String("transformer.layers.0.mlp.linear_fc2"))


def _require_exact_inventory(st: SafeTensors) raises:
    var actual = st.names()
    var expected = _expected_names()
    _require(
        len(actual) == SMOKE_TENSORS,
        String("expected 21 tensors: 7 modules x alpha/down/up"),
    )
    _require(len(expected) == SMOKE_TENSORS, String("expected-name builder mismatch"))
    for i in range(len(expected)):
        _require(_contains(actual, expected[i]), String("missing saved key ") + expected[i])
    for i in range(len(actual)):
        _require(
            _contains(expected, actual[i]),
            String("unexpected saved key ") + actual[i],
        )


def _require_adapter(
    st: SafeTensors, prefix: String, in_f: Int, out_f: Int, ctx: DeviceContext
) raises:
    _require_alpha(st, prefix + String(".alpha"), Float32(4.0), ctx)
    _require_rank2(st, prefix + String(".lora_down.weight"), SMOKE_RANK, in_f)
    _require_rank2(st, prefix + String(".lora_up.weight"), out_f, SMOKE_RANK)


def main() raises:
    var ctx = DeviceContext()
    var prefixes = ernie_lora_prefixes(SMOKE_NUM_LAYERS)
    _require_exact_prefixes(prefixes)

    var lora = build_ernie_lora_set(
        SMOKE_NUM_LAYERS, SMOKE_HIDDEN, SMOKE_FFN, SMOKE_RANK, Float32(4.0)
    )
    _require(len(lora.ad) == SMOKE_ADAPTERS, String("expected 7 built ERNIE adapters"))
    _require(lora.num_layers == SMOKE_NUM_LAYERS, String("smoke layer count mismatch"))
    _require(lora.rank == SMOKE_RANK, String("smoke rank mismatch"))

    var saved = save_ernie_lora(lora, String(OUT), ctx)
    _require(saved == 7, String("expected 7 saved ERNIE adapters"))

    var st = SafeTensors.open(String(OUT))
    _require_exact_inventory(st)

    var q = String("transformer.layers.0.self_attention.to_q")
    _require_adapter(st, q, SMOKE_HIDDEN, SMOKE_HIDDEN, ctx)

    var k = String("transformer.layers.0.self_attention.to_k")
    _require_adapter(st, k, SMOKE_HIDDEN, SMOKE_HIDDEN, ctx)

    var v = String("transformer.layers.0.self_attention.to_v")
    _require_adapter(st, v, SMOKE_HIDDEN, SMOKE_HIDDEN, ctx)

    var o = String("transformer.layers.0.self_attention.to_out.0")
    _require_adapter(st, o, SMOKE_HIDDEN, SMOKE_HIDDEN, ctx)

    var gate = String("transformer.layers.0.mlp.gate_proj")
    _require_adapter(st, gate, SMOKE_HIDDEN, SMOKE_FFN, ctx)

    var up = String("transformer.layers.0.mlp.up_proj")
    _require_adapter(st, up, SMOKE_HIDDEN, SMOKE_FFN, ctx)

    var down = String("transformer.layers.0.mlp.linear_fc2")
    _require_adapter(st, down, SMOKE_FFN, SMOKE_HIDDEN, ctx)

    var resumed = load_ernie_lora_resume(
        SMOKE_NUM_LAYERS, SMOKE_RANK, Float32(4.0), String(OUT), ctx
    )
    _require(len(resumed.ad) == 7, String("expected 7 resumed adapters"))
    _require(resumed.ad[0].rank == SMOKE_RANK, String("resumed rank mismatch"))
    _require(resumed.ad[0].in_f == SMOKE_HIDDEN, String("resumed q input dim mismatch"))
    _require(resumed.ad[0].out_f == SMOKE_HIDDEN, String("resumed q output dim mismatch"))
    _require(resumed.ad[4].in_f == SMOKE_HIDDEN, String("resumed gate input dim mismatch"))
    _require(resumed.ad[4].out_f == SMOKE_FFN, String("resumed gate output dim mismatch"))
    _require(resumed.ad[6].in_f == SMOKE_FFN, String("resumed fc2 input dim mismatch"))
    _require(resumed.ad[6].out_f == SMOKE_HIDDEN, String("resumed fc2 output dim mismatch"))

    print("[ernie-lora-ot-save-key] PASS:", OUT)
