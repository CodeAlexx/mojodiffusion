# One-shot Mojo cache preparation for one Bernini-R Wan A14B expert.
# Usage: bernini_prepare_fp8_cache <source transformer dir> <cache dir>

from max.gpu.host import DeviceContext
from std.sys import argv

from serenitymojo.models.wan22.wan22_fp8_stream import (
    prepare_wan22_a14b_fp8_cache,
)


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error(
            "usage: bernini_prepare_fp8_cache <source transformer dir> <cache dir>"
        )
    var source = String(args[1])
    var cache = String(args[2])
    print("=== Bernini-R Wan A14B persistent FP8 cache ===")
    print("  source:", source)
    print("  cache:", cache)
    var ctx = DeviceContext()
    prepare_wan22_a14b_fp8_cache(source, cache, ctx)
