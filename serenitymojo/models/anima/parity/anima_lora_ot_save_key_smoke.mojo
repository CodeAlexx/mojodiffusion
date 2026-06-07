# anima_lora_ot_save_key_smoke.mojo -- Anima OneTrainer save-key contract.
#
# Tiny structural gate: build one synthetic Anima LoRA block, save through
# save_anima_lora, inspect the raw OT alpha/down/up safetensors header, then
# load it back through the Anima resume loader. No model checkpoint or denoise.

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.anima.lora_block import (
    ANIMA_SLOTS,
    SLOT_SA_Q,
    SLOT_SA_K,
    SLOT_SA_V,
    SLOT_SA_O,
    SLOT_CA_Q,
    SLOT_CA_K,
    SLOT_CA_V,
    SLOT_CA_O,
    SLOT_MLP1,
    SLOT_MLP2,
)
from serenitymojo.models.anima.anima_stack_lora import (
    build_anima_lora_set,
    load_anima_lora_resume,
    save_anima_lora,
)
from serenitymojo.training.lora_save import _read_f32


comptime OUT = "/tmp/anima_lora_ot_save_key_smoke.safetensors"
comptime SMOKE_D = 8
comptime SMOKE_JOINT = 4
comptime SMOKE_F = 16
comptime SMOKE_RANK = 2
comptime SMOKE_ALPHA = 4.0


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


def _module_for_slot(slot: Int) raises -> String:
    if slot == SLOT_SA_Q:
        return String("attn1.to_q")
    elif slot == SLOT_SA_K:
        return String("attn1.to_k")
    elif slot == SLOT_SA_V:
        return String("attn1.to_v")
    elif slot == SLOT_SA_O:
        return String("attn1.to_out.0")
    elif slot == SLOT_CA_Q:
        return String("attn2.to_q")
    elif slot == SLOT_CA_K:
        return String("attn2.to_k")
    elif slot == SLOT_CA_V:
        return String("attn2.to_v")
    elif slot == SLOT_CA_O:
        return String("attn2.to_out.0")
    elif slot == SLOT_MLP1:
        return String("ff.net.0.proj")
    elif slot == SLOT_MLP2:
        return String("ff.net.2")
    raise Error(String("unsupported Anima LoRA slot ") + String(slot))


def _prefix_for_slot(slot: Int) raises -> String:
    return String("transformer.transformer_blocks.0.") + _module_for_slot(slot)


def _slot_in(slot: Int) raises -> Int:
    if slot == SLOT_CA_K or slot == SLOT_CA_V:
        return SMOKE_JOINT
    elif slot == SLOT_MLP2:
        return SMOKE_F
    elif (
        slot == SLOT_SA_Q or slot == SLOT_SA_K or slot == SLOT_SA_V
        or slot == SLOT_SA_O or slot == SLOT_CA_Q or slot == SLOT_CA_O
        or slot == SLOT_MLP1
    ):
        return SMOKE_D
    raise Error(String("unsupported Anima LoRA slot input dim ") + String(slot))


def _slot_out(slot: Int) raises -> Int:
    if slot == SLOT_MLP1:
        return SMOKE_F
    elif (
        slot == SLOT_SA_Q or slot == SLOT_SA_K or slot == SLOT_SA_V
        or slot == SLOT_SA_O or slot == SLOT_CA_Q or slot == SLOT_CA_K
        or slot == SLOT_CA_V or slot == SLOT_CA_O or slot == SLOT_MLP2
    ):
        return SMOKE_D
    raise Error(String("unsupported Anima LoRA slot output dim ") + String(slot))


def _expect_key(mut expected: Dict[String, Bool], key: String):
    expected[key] = True


def _require_exact_inventory(st: SafeTensors, ctx: DeviceContext) raises:
    var expected = Dict[String, Bool]()
    for s in range(ANIMA_SLOTS):
        var p = _prefix_for_slot(s)
        _expect_key(expected, p + String(".alpha"))
        _expect_key(expected, p + String(".lora_down.weight"))
        _expect_key(expected, p + String(".lora_up.weight"))

    _require(
        len(st.tensors) == len(expected),
        String("expected exact Anima smoke inventory tensor count"),
    )

    var names = st.names()
    for i in range(len(names)):
        _require(
            names[i] in expected,
            String("unsupported Anima LoRA saved key ") + names[i],
        )

    for s in range(ANIMA_SLOTS):
        var p = _prefix_for_slot(s)
        var in_f = _slot_in(s)
        var out_f = _slot_out(s)
        _require_alpha(st, p + String(".alpha"), Float32(SMOKE_ALPHA), ctx)
        _require_rank2(st, p + String(".lora_down.weight"), SMOKE_RANK, in_f)
        _require_rank2(st, p + String(".lora_up.weight"), out_f, SMOKE_RANK)


def main() raises:
    var ctx = DeviceContext()
    var lora = build_anima_lora_set(
        1, SMOKE_D, SMOKE_JOINT, SMOKE_F, SMOKE_RANK, Float32(SMOKE_ALPHA)
    )
    var saved = save_anima_lora(lora, String(OUT), ctx)
    _require(saved == ANIMA_SLOTS, String("expected 10 saved Anima adapters"))

    var st = SafeTensors.open(String(OUT))
    _require_exact_inventory(st, ctx)

    var resumed = load_anima_lora_resume(
        1, SMOKE_RANK, Float32(SMOKE_ALPHA), String(OUT), ctx
    )
    _require(len(resumed.ad) == ANIMA_SLOTS, String("expected 10 resumed adapters"))
    for s in range(ANIMA_SLOTS):
        _require(resumed.ad[s].rank == SMOKE_RANK, String("resumed rank mismatch"))
        _require(
            resumed.ad[s].in_f == _slot_in(s),
            String("resumed input dim mismatch for slot ") + String(s),
        )
        _require(
            resumed.ad[s].out_f == _slot_out(s),
            String("resumed output dim mismatch for slot ") + String(s),
        )
        _require(
            resumed.ad[s].scale == Float32(SMOKE_ALPHA) / Float32(SMOKE_RANK),
            String("resumed scale mismatch for slot ") + String(s),
        )

    print("[anima-lora-ot-save-key] PASS:", OUT)
