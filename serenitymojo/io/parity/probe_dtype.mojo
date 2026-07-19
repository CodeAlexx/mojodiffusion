from serenitymojo.io.dtype import STDtype

def check(d: STDtype) raises:
    print(d.name(), "size=", d.byte_size(), "roundtrip=", STDtype.from_name(d.name()).name())

def main() raises:
    var all = [
        STDtype.BOOL, STDtype.U8, STDtype.I8, STDtype.F8_E5M2, STDtype.F8_E4M3,
        STDtype.I16, STDtype.U16, STDtype.F16, STDtype.BF16,
        STDtype.I32, STDtype.U32, STDtype.F32, STDtype.F64, STDtype.I64, STDtype.U64,
    ]
    for ref d in all:
        check(d)
