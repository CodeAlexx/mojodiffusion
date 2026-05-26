# mojo_dump.mojo — parity dumper. Opens a safetensors file via the pure-Mojo
# reader and prints, for EVERY tensor: name|dtype|shape|offset|size|fnv1a64
# where fnv1a64 is a 64-bit FNV-1a hash over the mmap'd tensor bytes (proves the
# returned pointer addresses the exact bytes the oracle hashes).
#
# Run: pixi run mojo run -I . serenitymojo/io/parity/mojo_dump.mojo <path>

from std.sys import argv
from serenitymojo.io.safetensors import SafeTensors, TensorRef


comptime FNV_OFFSET: UInt64 = 0xCBF29CE484222325
comptime FNV_PRIME: UInt64 = 0x100000001B3
comptime WINDOW = 65536  # match oracle.py: first WINDOW + last WINDOW bytes


def _fnv1a(read st: SafeTensors, name: String, size: Int) raises -> UInt64:
    # Origin-bound view: the Span ties the mmap'd bytes to `st`'s lifetime.
    var p = st.tensor_bytes(name)
    var h: UInt64 = FNV_OFFSET
    if size <= 2 * WINDOW:
        for i in range(size):
            h = (h ^ UInt64(Int(p[i]))) * FNV_PRIME
        return h
    # first WINDOW bytes
    for i in range(WINDOW):
        h = (h ^ UInt64(Int(p[i]))) * FNV_PRIME
    # last WINDOW bytes
    for i in range(size - WINDOW, size):
        h = (h ^ UInt64(Int(p[i]))) * FNV_PRIME
    return h


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: mojo_dump <path.safetensors>")
        return
    var path = String(args[1])
    var st = SafeTensors.open(path)
    print("#TENSORS", st.count())
    print("#DATASEG", st.data_size())
    var names = st.names()
    for ref nm in names:
        var info = st.tensor_info(nm)
        var shape_str = String("[")
        for i in range(len(info.shape)):
            if i > 0:
                shape_str += ","
            shape_str += String(info.shape[i])
        shape_str += "]"
        var h = _fnv1a(st, nm, info.size)
        print(
            nm,
            "|",
            info.dtype.name(),
            "|",
            shape_str,
            "|",
            info.offset,
            "|",
            info.size,
            "|",
            h,
        )
