# training/full_finetune_save.mojo -- shared full-finetune model tensor save scaffold.
#
# This module is deliberately small. It writes already-named model tensors to a
# safetensors file via io/safetensors_writer.save_safetensors, preserving each
# Tensor's storage dtype at the file boundary. BF16/F16/FP8 tensors are copied as
# raw bytes by the writer; this wrapper does not call Tensor.to_host() or route
# model storage through host Float32.
#
# Scope boundary:
#   * This is model-tensor save scaffolding only.
#   * Optimizer/master state is a separate checkpoint artifact.
#   * This is not a substitute for model-specific full-finetune parity: each
#     model still has to supply the exact key inventory and reload mapping.

from std.collections import List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.tensor import Tensor


comptime FULL_FINETUNE_NAME_MANIFEST_KEY = "__full_finetune_tensor_names_utf8__"


@fieldwise_init
struct FullFinetuneTensor(Copyable, Movable):
    """A named model tensor for full-finetune checkpoint save scaffolding.

    `name` must be the already-final safetensors key. This module intentionally
    does not translate model fields into keys or map checkpoint keys back to
    model structs.
    """

    var name: String
    var tensor: ArcPointer[Tensor]


@fieldwise_init
struct FullFinetuneTensorHeader(Copyable, Movable):
    """Header-only readback metadata for smoke/contract validation."""

    var name: String
    var dtype: STDtype
    var shape: List[Int]
    var nbytes: Int


def _has_name(names: List[String], name: String) -> Bool:
    for i in range(len(names)):
        if names[i] == name:
            return True
    return False


def _validate_name_list(names: List[String], caller: String) raises:
    if len(names) == 0:
        raise Error(caller + ": refusing empty tensor names")
    var seen = List[String]()
    for i in range(len(names)):
        if names[i].byte_length() == 0:
            raise Error(caller + ": empty tensor name")
        var src = names[i].as_bytes()
        for j in range(len(src)):
            if src[j] == UInt8(10) or src[j] == UInt8(13):
                raise Error(caller + ": tensor names may not contain line breaks")
        if _has_name(seen, names[i]):
            raise Error(caller + ": duplicate tensor name: " + names[i])
        seen.append(names[i])


def _u8_1d(values: List[UInt8], ctx: DeviceContext) raises -> Tensor:
    var shape = List[Int]()
    shape.append(len(values))
    var host = ctx.enqueue_create_host_buffer[DType.uint8](len(values))
    var hp = host.unsafe_ptr()
    for i in range(len(values)):
        hp[i] = values[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](len(values))
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, shape^, STDtype.U8)


def save_full_finetune_model_tensors(
    tensors: List[FullFinetuneTensor], path: String, ctx: DeviceContext
) raises -> Int:
    """Save named full-finetune model tensors as one safetensors file.

    Returns the number of tensors written. The caller owns model-specific key
    naming, completeness, trainable/frozen filtering, and reload mapping. This
    function only validates the flat named-tensor list and delegates to the
    dtype-preserving raw-byte safetensors writer.
    """

    var names = List[String]()
    var arcs = List[ArcPointer[Tensor]]()

    for ref entry in tensors:
        names.append(entry.name)
        arcs.append(entry.tensor.copy())

    _validate_name_list(names, String("save_full_finetune_model_tensors"))
    save_safetensors(names, arcs, path, ctx)
    return len(tensors)


def save_full_finetune_name_manifest(
    names: List[String], path: String, ctx: DeviceContext
) raises -> Int:
    """Save the deterministic full-finetune tensor order used by TrainState.

    `training/loop.mojo` stores optimizer/master sidecars as opaque
    `param.N`/`adam_m.N`/`adam_v.N` tensors. This manifest binds those indices
    back to OneTrainer/model tensor names by saving newline-separated UTF-8
    names as one U8 safetensors tensor. Model loops must use the same ordered
    name list for full-weight save, TrainState construction, and resume.
    """

    _validate_name_list(names, String("save_full_finetune_name_manifest"))

    var bytes = List[UInt8]()
    for i in range(len(names)):
        var src = names[i].as_bytes()
        for j in range(len(src)):
            bytes.append(src[j])
        bytes.append(UInt8(10))

    var manifest = _u8_1d(bytes, ctx)
    var out_names = List[String]()
    out_names.append(String(FULL_FINETUNE_NAME_MANIFEST_KEY))
    var tensors = List[ArcPointer[Tensor]]()
    tensors.append(ArcPointer[Tensor](manifest^))
    save_safetensors(out_names, tensors, path, ctx)
    return len(names)


def load_full_finetune_name_manifest(path: String) raises -> List[String]:
    """Load the tensor-order manifest written by save_full_finetune_name_manifest."""

    var st = SafeTensors.open(path)
    var key = String(FULL_FINETUNE_NAME_MANIFEST_KEY)
    if key not in st.tensors:
        raise Error("load_full_finetune_name_manifest: missing manifest tensor")
    var info = st.tensor_info(key)
    if info.dtype != STDtype.U8 or len(info.shape) != 1:
        raise Error("load_full_finetune_name_manifest: manifest must be U8 [N]")

    var bytes = st.tensor_bytes(key)
    var out = List[String]()
    var cur = List[UInt8]()
    for i in range(len(bytes)):
        if bytes[i] == UInt8(10):
            if len(cur) == 0:
                raise Error("load_full_finetune_name_manifest: empty manifest name")
            out.append(String(unsafe_from_utf8=cur))
            cur = List[UInt8]()
        else:
            cur.append(bytes[i])
    if len(cur) != 0:
        out.append(String(unsafe_from_utf8=cur))

    _validate_name_list(out, String("load_full_finetune_name_manifest"))
    return out^


def assert_full_finetune_name_manifest_matches(
    expected: List[String], path: String
) raises:
    """Fail if a TrainState sidecar order would not match the model tensors."""

    _validate_name_list(expected, String("assert_full_finetune_name_manifest_matches"))
    var actual = load_full_finetune_name_manifest(path)
    if len(actual) != len(expected):
        raise Error("full-finetune manifest length mismatch")
    for i in range(len(expected)):
        if actual[i] != expected[i]:
            raise Error(
                String("full-finetune manifest mismatch at index ")
                + String(i)
                + String(": expected ")
                + expected[i]
                + String(" got ")
                + actual[i]
            )


def load_full_finetune_model_tensors(
    names: List[String], path: String, ctx: DeviceContext
) raises -> List[FullFinetuneTensor]:
    """Load exactly the requested model tensors from a safetensors file.

    The returned list preserves request order and carries each tensor as a
    `FullFinetuneTensor(name, ArcPointer[Tensor])`. Loading uses
    `Tensor.from_view_raw`, so the on-disk dtype label and bytes are preserved
    for BF16/F16/F32 and non-compute storage such as FP8.

    This is only flat model-tensor resume scaffolding. Full training resume also
    requires a separate optimizer/master-state sidecar and model-specific
    key-to-struct mapping, both intentionally out of scope here.
    """

    _validate_name_list(names, String("load_full_finetune_model_tensors"))

    var st = SafeTensors.open(path)
    var out = List[FullFinetuneTensor]()
    for i in range(len(names)):
        var name = names[i]
        if name not in st.tensors:
            raise Error(
                String("load_full_finetune_model_tensors: missing tensor: ") + name
            )
        var info = st.tensor_info(name)
        var bytes = st.tensor_bytes(name)
        var view = from_parts(info.dtype, info.shape.copy(), bytes)
        var tensor = Tensor.from_view_raw(view, ctx)
        out.append(FullFinetuneTensor(name, ArcPointer[Tensor](tensor^)))

    return out^


def read_full_finetune_tensor_header(
    st: SafeTensors, name: String
) raises -> FullFinetuneTensorHeader:
    """Return header metadata for a saved tensor.

    This is intentionally header/readback-only. It does not load tensor data into
    a model and does not implement any model-specific key mapping.
    """

    var info = st.tensor_info(name)
    return FullFinetuneTensorHeader(name, info.dtype, info.shape.copy(), info.size)
