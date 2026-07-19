# training/full_finetune_save_smoke.mojo -- full-finetune save scaffold smoke.
#
# Builds one BF16 model tensor and one F32 tensor, saves them, reopens with
# SafeTensors, loads them back through load_full_finetune_model_tensors, and
# asserts key, dtype, shape, and BF16 raw-value preservation. The BF16 tensor is
# created through Tensor.from_host_bf16(List[BFloat16], ...), not through a host
# Float32 tensor boundary.
#
# Run:
#   timeout 180 prlimit --as=16000000000 pixi run mojo run \
#     --target-accelerator sm_86 -I . \
#     serenitymojo/training/full_finetune_save_smoke.mojo

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.training.full_finetune_save import (
    FullFinetuneTensor,
    assert_full_finetune_name_manifest_matches,
    load_full_finetune_name_manifest,
    load_full_finetune_model_tensors,
    read_full_finetune_tensor_header,
    save_full_finetune_name_manifest,
    save_full_finetune_model_tensors,
)


comptime OUT = "/tmp/full_finetune_save_smoke.safetensors"
comptime MANIFEST = "/tmp/full_finetune_save_smoke.names.safetensors"


def _shape2(a: Int, b: Int) -> List[Int]:
    var sh = List[Int]()
    sh.append(a)
    sh.append(b)
    return sh^


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _require_shape(got: List[Int], expected: List[Int], label: String) raises:
    _require(len(got) == len(expected), label + ": rank mismatch")
    for i in range(len(expected)):
        _require(
            got[i] == expected[i],
            label + ": dim " + String(i) + " mismatch",
        )


def _assert_bf16_bytes(
    st: SafeTensors, key: String, expected: List[BFloat16]
) raises:
    var bytes = st.tensor_bytes(key)
    _require(
        len(bytes) == len(expected) * STDtype.BF16.byte_size(),
        key + ": BF16 byte length mismatch",
    )
    var bp = bytes.unsafe_ptr().bitcast[BFloat16]()
    for i in range(len(expected)):
        _require(bp[i] == expected[i], key + ": BF16 raw value mismatch")


def _assert_tensor_bf16_bytes(
    tensor: Tensor, expected: List[BFloat16], ctx: DeviceContext
) raises:
    _require(tensor.dtype() == STDtype.BF16, "loaded tensor is not BF16")
    _require(
        tensor.nbytes() == len(expected) * STDtype.BF16.byte_size(),
        "loaded BF16 tensor byte length mismatch",
    )
    var host = ctx.enqueue_create_host_buffer[DType.uint8](tensor.nbytes())
    ctx.enqueue_copy(dst_buf=host, src_buf=tensor.buf)
    ctx.synchronize()
    var bp = host.unsafe_ptr().bitcast[BFloat16]()
    for i in range(len(expected)):
        _require(bp[i] == expected[i], "loaded BF16 tensor raw value mismatch")


def main() raises:
    print("=== full_finetune_save scaffold smoke ===")
    var ctx = DeviceContext()

    var bf16_values = List[BFloat16]()
    bf16_values.append(Float32(1.0).cast[DType.bfloat16]())
    bf16_values.append(Float32(-2.0).cast[DType.bfloat16]())
    bf16_values.append(Float32(0.5).cast[DType.bfloat16]())
    bf16_values.append(Float32(3.0).cast[DType.bfloat16]())
    var bf16_shape = _shape2(2, 2)
    var bf16_tensor = Tensor.from_host_bf16(bf16_values.copy(), bf16_shape.copy(), ctx)

    var f32_values = List[Float32]()
    f32_values.append(Float32(0.25))
    f32_values.append(Float32(-0.75))
    f32_values.append(Float32(2.5))
    var f32_shape = List[Int]()
    f32_shape.append(3)
    var f32_tensor = Tensor.from_host(f32_values^, f32_shape.copy(), STDtype.F32, ctx)

    var key_bf16 = String("model.block0.linear.weight")
    var key_f32 = String("model.norm.weight")

    var named = List[FullFinetuneTensor]()
    named.append(FullFinetuneTensor(key_bf16, ArcPointer[Tensor](bf16_tensor^)))
    named.append(FullFinetuneTensor(key_f32, ArcPointer[Tensor](f32_tensor^)))

    var saved = save_full_finetune_model_tensors(named, String(OUT), ctx)
    _require(saved == 2, "expected two saved tensors")
    print("saved", saved, "tensor(s) ->", OUT)

    var st = SafeTensors.open(String(OUT))
    _require(st.count() == 2, "expected two tensors after reopen")
    _require(key_bf16 in st.tensors, "missing BF16 key after reopen")
    _require(key_f32 in st.tensors, "missing F32 key after reopen")

    var h_bf16 = read_full_finetune_tensor_header(st, key_bf16)
    _require(h_bf16.dtype == STDtype.BF16, "BF16 tensor dtype widened or changed")
    _require_shape(h_bf16.shape, bf16_shape, "BF16 tensor shape")
    _require(
        h_bf16.nbytes == 4 * STDtype.BF16.byte_size(),
        "BF16 tensor byte size mismatch",
    )
    _assert_bf16_bytes(st, key_bf16, bf16_values)
    print("BF16 key/dtype/shape/raw-byte readback PASS")

    var h_f32 = read_full_finetune_tensor_header(st, key_f32)
    _require(h_f32.dtype == STDtype.F32, "F32 tensor dtype changed")
    _require_shape(h_f32.shape, f32_shape, "F32 tensor shape")
    _require(
        h_f32.nbytes == 3 * STDtype.F32.byte_size(),
        "F32 tensor byte size mismatch",
    )
    print("F32 key/dtype/shape readback PASS")

    var requested = List[String]()
    requested.append(key_bf16)
    requested.append(key_f32)

    var manifest_saved = save_full_finetune_name_manifest(
        requested, String(MANIFEST), ctx
    )
    _require(manifest_saved == 2, "expected two manifest names")
    var manifest_names = load_full_finetune_name_manifest(String(MANIFEST))
    _require(len(manifest_names) == 2, "expected two loaded manifest names")
    _require(manifest_names[0] == key_bf16, "manifest BF16 name/order mismatch")
    _require(manifest_names[1] == key_f32, "manifest F32 name/order mismatch")
    assert_full_finetune_name_manifest_matches(requested, String(MANIFEST))
    print("full-finetune tensor-name manifest PASS")

    var loaded = load_full_finetune_model_tensors(requested, String(OUT), ctx)
    _require(len(loaded) == 2, "expected two loaded tensors")
    _require(loaded[0].name == key_bf16, "loaded BF16 key order mismatch")
    _require(loaded[1].name == key_f32, "loaded F32 key order mismatch")

    _require(loaded[0].tensor[].dtype() == STDtype.BF16, "loaded BF16 dtype mismatch")
    _require_shape(loaded[0].tensor[].shape(), bf16_shape, "loaded BF16 tensor shape")
    _assert_tensor_bf16_bytes(loaded[0].tensor[], bf16_values, ctx)
    print("BF16 load dtype/shape/raw-value preservation PASS")

    _require(loaded[1].tensor[].dtype() == STDtype.F32, "loaded F32 dtype mismatch")
    _require_shape(loaded[1].tensor[].shape(), f32_shape, "loaded F32 tensor shape")
    _require(
        loaded[1].tensor[].nbytes() == 3 * STDtype.F32.byte_size(),
        "loaded F32 tensor byte size mismatch",
    )
    print("F32 load dtype/shape preservation PASS")

    print("full_finetune_save scaffold smoke PASS")
