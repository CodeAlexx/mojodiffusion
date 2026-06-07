# Klein full-finetune checkpoint collector smoke.
#
# Synthetic, bounded gate only: creates tiny BF16 live runtime structs, saves the
# 201-key Klein full-finetune payload, validates the name manifest, and loads the
# flat payload back. This is not a full-finetune training or resume-rebind gate.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.klein.double_block import (
    DoubleBlockWeights,
    StreamWeights,
)
from serenitymojo.models.klein.full_finetune_checkpoint import (
    collect_klein_full_finetune_tensors,
    load_klein_full_finetune_payload_only,
    save_klein_full_finetune_checkpoint,
)
from serenitymojo.models.klein.full_finetune_inventory import (
    klein_full_finetune_checkpoint_key_manifest,
    klein_full_finetune_inventory_expected_count,
)
from serenitymojo.models.klein.klein_stack import KleinStackBase
from serenitymojo.models.klein.single_block import SingleBlockWeights
from serenitymojo.models.klein.weights import KleinStepModWeights
from serenitymojo.tensor import Tensor
from serenitymojo.training.full_finetune_save import (
    assert_full_finetune_name_manifest_matches,
)


comptime TArc = ArcPointer[Tensor]
comptime OUT = "/tmp/klein_full_finetune_checkpoint_smoke.safetensors"
comptime MANIFEST = "/tmp/klein_full_finetune_checkpoint_smoke.names.safetensors"


def _shape1(a: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    return s^


def _shape2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _bf16_values(n: Int, start: Float32) -> List[BFloat16]:
    var out = List[BFloat16]()
    for i in range(n):
        out.append((start + Float32(i) * Float32(0.125)).cast[DType.bfloat16]())
    return out^


def _bf16_tensor(
    n: Int, start: Float32, var shape: List[Int], ctx: DeviceContext
) raises -> Tensor:
    return Tensor.from_host_bf16(_bf16_values(n, start), shape^, ctx)


def _bf16_arc(
    n: Int, start: Float32, var shape: List[Int], ctx: DeviceContext
) raises -> TArc:
    return TArc(_bf16_tensor(n, start, shape^, ctx))


def _stream(seed: Int, ctx: DeviceContext) raises -> StreamWeights:
    var s = Float32(seed)
    return StreamWeights(
        _bf16_arc(48, s + Float32(0.0), _shape2(12, 4), ctx),
        _bf16_arc(16, s + Float32(1.0), _shape2(4, 4), ctx),
        _bf16_arc(64, s + Float32(2.0), _shape2(16, 4), ctx),
        _bf16_arc(32, s + Float32(3.0), _shape2(4, 8), ctx),
        _bf16_arc(2, s + Float32(4.0), _shape1(2), ctx),
        _bf16_arc(2, s + Float32(5.0), _shape1(2), ctx),
    )


def _double_block(idx: Int, ctx: DeviceContext) raises -> DoubleBlockWeights:
    return DoubleBlockWeights(
        _stream(1000 + idx * 10, ctx),
        _stream(2000 + idx * 10, ctx),
    )


def _single_block(idx: Int, ctx: DeviceContext) raises -> SingleBlockWeights:
    var s = Float32(3000 + idx * 10)
    return SingleBlockWeights(
        _bf16_arc(112, s + Float32(0.0), _shape2(28, 4), ctx),
        _bf16_arc(48, s + Float32(1.0), _shape2(4, 12), ctx),
        _bf16_arc(2, s + Float32(2.0), _shape1(2), ctx),
        _bf16_arc(2, s + Float32(3.0), _shape1(2), ctx),
        4,
        8,
        ctx,
    )


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def main() raises:
    var ctx = DeviceContext()

    var base = KleinStackBase(
        _bf16_arc(12, Float32(1.0), _shape2(4, 3), ctx),
        _bf16_arc(20, Float32(2.0), _shape2(4, 5), ctx),
        _bf16_arc(8, Float32(3.0), _shape2(2, 4), ctx),
        _bf16_arc(4, Float32(4.0), _shape1(4), ctx),
        _bf16_arc(4, Float32(5.0), _shape1(4), ctx),
    )
    var step_mod = KleinStepModWeights(
        _bf16_tensor(16, Float32(10.0), _shape2(4, 4), ctx),
        _bf16_tensor(16, Float32(11.0), _shape2(4, 4), ctx),
        _bf16_tensor(96, Float32(12.0), _shape2(24, 4), ctx),
        _bf16_tensor(96, Float32(13.0), _shape2(24, 4), ctx),
        _bf16_tensor(48, Float32(14.0), _shape2(12, 4), ctx),
        _bf16_tensor(32, Float32(15.0), _shape2(8, 4), ctx),
    )

    var doubles = List[DoubleBlockWeights]()
    for i in range(8):
        doubles.append(_double_block(i, ctx))
    var singles = List[SingleBlockWeights]()
    for i in range(24):
        singles.append(_single_block(i, ctx))

    var collected = collect_klein_full_finetune_tensors(
        base, step_mod, doubles, singles, ctx
    )
    var expected = klein_full_finetune_inventory_expected_count()
    _require(len(collected) == expected, String("collector count mismatch"))
    _require(collected[0].name == String("img_in.weight"), String("first key mismatch"))
    _require(
        collected[len(collected) - 1].name
        == String("single_blocks.23.norm.key_norm.scale"),
        String("last key mismatch"),
    )
    _require(
        collected[0].tensor[].dtype() == STDtype.BF16,
        String("first dtype mismatch"),
    )
    _require(
        collected[2].tensor[].dtype() == STDtype.BF16,
        String("step-mod dtype mismatch"),
    )

    var saved = save_klein_full_finetune_checkpoint(
        base, step_mod, doubles, singles, String(OUT), String(MANIFEST), ctx
    )
    _require(saved == expected, String("saved tensor count mismatch"))

    var st = SafeTensors.open(String(OUT))
    _require(st.count() == expected, String("safetensors count mismatch"))
    var first = st.tensor_info(String("img_in.weight"))
    _require(first.dtype == STDtype.BF16, String("saved img_in dtype mismatch"))
    var mod = st.tensor_info(String("time_in.in_layer.weight"))
    _require(mod.dtype == STDtype.BF16, String("saved step-mod dtype mismatch"))

    var names = klein_full_finetune_checkpoint_key_manifest()
    assert_full_finetune_name_manifest_matches(names, String(MANIFEST))

    var payload = load_klein_full_finetune_payload_only(
        String(OUT), String(MANIFEST), ctx
    )
    _require(len(payload) == expected, String("payload-only load count mismatch"))
    _require(payload[0].name == names[0], String("payload first key mismatch"))
    _require(payload[0].tensor[].dtype() == STDtype.BF16, String("payload dtype mismatch"))

    print("Klein full-finetune checkpoint smoke PASS count=", expected)
