# 1:1 port of Serenity modules/util/enum/DataType.py
# Source of truth: /home/alex/Serenity/modules/util/enum/DataType.py
#
# comptime-int constants matching the Python DataType members exactly
# (names + order). String value == member name.

from serenitymojo.io.dtype import STDtype

comptime DATA_TYPE_NONE = 0           # DataType.NONE
comptime DATA_TYPE_FLOAT_8 = 1        # DataType.FLOAT_8
comptime DATA_TYPE_FLOAT_16 = 2       # DataType.FLOAT_16
comptime DATA_TYPE_FLOAT_32 = 3       # DataType.FLOAT_32
comptime DATA_TYPE_BFLOAT_16 = 4      # DataType.BFLOAT_16
comptime DATA_TYPE_TFLOAT_32 = 5      # DataType.TFLOAT_32
comptime DATA_TYPE_INT_8 = 6          # DataType.INT_8
comptime DATA_TYPE_NFLOAT_4 = 7       # DataType.NFLOAT_4
comptime DATA_TYPE_FLOAT_W8A8 = 8     # DataType.FLOAT_W8A8
comptime DATA_TYPE_INT_W8A8 = 9       # DataType.INT_W8A8
comptime DATA_TYPE_GGUF = 10          # DataType.GGUF
comptime DATA_TYPE_GGUF_A8_FLOAT = 11 # DataType.GGUF_A8_FLOAT
comptime DATA_TYPE_GGUF_A8_INT = 12   # DataType.GGUF_A8_INT


# is_quantized  (DataType.py:46-51)
def data_type_is_quantized(kind: Int) -> Bool:
    return (
        kind == DATA_TYPE_FLOAT_8
        or kind == DATA_TYPE_INT_8
        or kind == DATA_TYPE_FLOAT_W8A8
        or kind == DATA_TYPE_INT_W8A8
        or kind == DATA_TYPE_NFLOAT_4
    )


# is_gguf  (DataType.py:53-56)
def data_type_is_gguf(kind: Int) -> Bool:
    return (
        kind == DATA_TYPE_GGUF
        or kind == DATA_TYPE_GGUF_A8_FLOAT
        or kind == DATA_TYPE_GGUF_A8_INT
    )


# enable_tf  (DataType.py:43-44)
def data_type_enable_tf(kind: Int) -> Bool:
    return kind == DATA_TYPE_TFLOAT_32


# quantize_fp8  (DataType.py:58-59)
def data_type_quantize_fp8(kind: Int) -> Bool:
    return kind == DATA_TYPE_FLOAT_8


# quantize_int8  (DataType.py:61-62)
def data_type_quantize_int8(kind: Int) -> Bool:
    return kind == DATA_TYPE_INT_8


# quantize_fpW8A8  (DataType.py:64-65)
def data_type_quantize_fpW8A8(kind: Int) -> Bool:
    return kind == DATA_TYPE_FLOAT_W8A8


# quantize_intW8A8  (DataType.py:67-68)
def data_type_quantize_intW8A8(kind: Int) -> Bool:
    return kind == DATA_TYPE_INT_W8A8


# quantize_nf4  (DataType.py:70-71)
def data_type_quantize_nf4(kind: Int) -> Bool:
    return kind == DATA_TYPE_NFLOAT_4


# torch_dtype  (DataType.py:24-41) mapped to STDtype where representable.
# Returns (has_value, dtype): NONE/INT_8/quantized-unmatched cases map to no
# value (Python returns None). FLOAT_16->F16, FLOAT_32/TFLOAT_32->F32,
# BFLOAT_16->BF16. When is_quantized and not supports_quantization -> float16.
def data_type_torch_dtype(
    kind: Int, supports_quantization: Bool
) -> (Bool, STDtype):
    if data_type_is_quantized(kind) and not supports_quantization:
        return (True, STDtype.F16)  # torch.float16

    if kind == DATA_TYPE_FLOAT_16:
        return (True, STDtype.F16)
    elif kind == DATA_TYPE_FLOAT_32:
        return (True, STDtype.F32)
    elif kind == DATA_TYPE_BFLOAT_16:
        return (True, STDtype.BF16)
    elif kind == DATA_TYPE_TFLOAT_32:
        return (True, STDtype.F32)
    else:
        return (False, STDtype.F32)  # Python None


def data_type_str(kind: Int) -> String:
    if kind == DATA_TYPE_NONE:
        return "NONE"
    elif kind == DATA_TYPE_FLOAT_8:
        return "FLOAT_8"
    elif kind == DATA_TYPE_FLOAT_16:
        return "FLOAT_16"
    elif kind == DATA_TYPE_FLOAT_32:
        return "FLOAT_32"
    elif kind == DATA_TYPE_BFLOAT_16:
        return "BFLOAT_16"
    elif kind == DATA_TYPE_TFLOAT_32:
        return "TFLOAT_32"
    elif kind == DATA_TYPE_INT_8:
        return "INT_8"
    elif kind == DATA_TYPE_NFLOAT_4:
        return "NFLOAT_4"
    elif kind == DATA_TYPE_FLOAT_W8A8:
        return "FLOAT_W8A8"
    elif kind == DATA_TYPE_INT_W8A8:
        return "INT_W8A8"
    elif kind == DATA_TYPE_GGUF:
        return "GGUF"
    elif kind == DATA_TYPE_GGUF_A8_FLOAT:
        return "GGUF_A8_FLOAT"
    else:
        return "GGUF_A8_INT"
