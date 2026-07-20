# One-shot/resumable SCAIL-2 14B persistent E4M3 block-cache preparation.
# Usage: scail2_prepare_fp8_cache <canonical transformer dir> <cache dir>

from std.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.models.scail2.scail2_fp8_stream import (
    prepare_scail2_14b_fp8_cache,
)


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error(
            "usage: scail2_prepare_fp8_cache <canonical transformer dir> "
            "<cache dir>"
        )
    var source = String(args[1])
    var cache = String(args[2])
    print("=== SCAIL-2 14B persistent FP8 cache ===")
    print("  source:", source)
    print("  cache:", cache)
    var ctx = DeviceContext()
    prepare_scail2_14b_fp8_cache(source, cache, ctx)
