# STDtype — safetensors dtype enum mirroring serenity-safetensors src/lib.rs.
#
# Reference (read line-by-line, do NOT infer):
#   /home/alex/serenity-safetensors/src/lib.rs lines 70-99 (byte-size groups +
#   the exact dtype name strings).
#
# byte_size() groups (lib.rs:72-75):
#   F64 | I64 | U64                                  => 8
#   F32 | I32 | U32                                  => 4
#   F16 | BF16 | I16 | U16                           => 2
#   F8_E4M3 | F8_E5M2 | I8 | U8 | BOOL               => 1
#
# name strings (lib.rs:82-96): BOOL,U8,I8,F8_E5M2,F8_E4M3,I16,U16,F16,BF16,
#   I32,U32,F32,F64,I64,U64.

from std.builtin.dtype import DType


# Integer tags for the enum-like struct. Kept as module-level comptime ints so
# both the struct constants and external code can pattern-match cheaply.
comptime _BOOL = 0
comptime _U8 = 1
comptime _I8 = 2
comptime _F8_E5M2 = 3
comptime _F8_E4M3 = 4
comptime _I16 = 5
comptime _U16 = 6
comptime _F16 = 7
comptime _BF16 = 8
comptime _I32 = 9
comptime _U32 = 10
comptime _F32 = 11
comptime _F64 = 12
comptime _I64 = 13
comptime _U64 = 14


@fieldwise_init
struct STDtype(Copyable, Movable, ImplicitlyCopyable, Equatable):
    """A safetensors dtype. Enum-like: a single integer tag."""

    var tag: Int

    # Associated constants — one per safetensors dtype.
    comptime BOOL = Self(_BOOL)
    comptime U8 = Self(_U8)
    comptime I8 = Self(_I8)
    comptime F8_E5M2 = Self(_F8_E5M2)
    comptime F8_E4M3 = Self(_F8_E4M3)
    comptime I16 = Self(_I16)
    comptime U16 = Self(_U16)
    comptime F16 = Self(_F16)
    comptime BF16 = Self(_BF16)
    comptime I32 = Self(_I32)
    comptime U32 = Self(_U32)
    comptime F32 = Self(_F32)
    comptime F64 = Self(_F64)
    comptime I64 = Self(_I64)
    comptime U64 = Self(_U64)

    def __eq__(self, other: Self) -> Bool:
        return self.tag == other.tag

    def __ne__(self, other: Self) -> Bool:
        return self.tag != other.tag

    def byte_size(self) -> Int:
        """Element size in bytes. Mirrors lib.rs:70-78 dtype_element_size."""
        var t = self.tag
        # 8-byte group: F64 | I64 | U64
        if t == _F64 or t == _I64 or t == _U64:
            return 8
        # 4-byte group: F32 | I32 | U32
        if t == _F32 or t == _I32 or t == _U32:
            return 4
        # 2-byte group: F16 | BF16 | I16 | U16
        if t == _F16 or t == _BF16 or t == _I16 or t == _U16:
            return 2
        # 1-byte group: F8_E4M3 | F8_E5M2 | I8 | U8 | BOOL
        # (everything remaining)
        return 1

    def name(self) -> String:
        """Safetensors dtype string. Mirrors lib.rs:80-99
        dtype_to_safetensors_str."""
        var t = self.tag
        if t == _BOOL:
            return String("BOOL")
        if t == _U8:
            return String("U8")
        if t == _I8:
            return String("I8")
        if t == _F8_E5M2:
            return String("F8_E5M2")
        if t == _F8_E4M3:
            return String("F8_E4M3")
        if t == _I16:
            return String("I16")
        if t == _U16:
            return String("U16")
        if t == _F16:
            return String("F16")
        if t == _BF16:
            return String("BF16")
        if t == _I32:
            return String("I32")
        if t == _U32:
            return String("U32")
        if t == _F32:
            return String("F32")
        if t == _F64:
            return String("F64")
        if t == _I64:
            return String("I64")
        # _U64
        return String("U64")

    @staticmethod
    def from_name(s: String) raises -> STDtype:
        """Parse a safetensors dtype string. Accepts the canonical uppercase
        names emitted by safetensors (lib.rs:82-96). Raises on unknown."""
        if s == "BOOL":
            return Self.BOOL
        if s == "U8":
            return Self.U8
        if s == "I8":
            return Self.I8
        if s == "F8_E5M2":
            return Self.F8_E5M2
        if s == "F8_E4M3":
            return Self.F8_E4M3
        if s == "I16":
            return Self.I16
        if s == "U16":
            return Self.U16
        if s == "F16":
            return Self.F16
        if s == "BF16":
            return Self.BF16
        if s == "I32":
            return Self.I32
        if s == "U32":
            return Self.U32
        if s == "F32":
            return Self.F32
        if s == "F64":
            return Self.F64
        if s == "I64":
            return Self.I64
        if s == "U64":
            return Self.U64
        raise Error(String("Unknown dtype: ") + s)

    def to_mojo_dtype(self) raises -> DType:
        """Map to a Mojo DType for compute. Only BF16/F16/F32 are supported
        compute dtypes (PLAN.md: BF16-first). Others raise for now."""
        var t = self.tag
        if t == _BF16:
            return DType.bfloat16
        if t == _F16:
            return DType.float16
        if t == _F32:
            return DType.float32
        raise Error(
            String("unsupported compute dtype: ") + self.name()
        )
