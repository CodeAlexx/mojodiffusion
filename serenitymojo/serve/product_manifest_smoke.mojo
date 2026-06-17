from std.testing import assert_equal

from serenitymojo.serve.product_manifest import json_bool, json_escape, peak_vram_mib


def main() raises:
    assert_equal(json_escape(String("plain")), String("plain"))
    assert_equal(json_escape(String("quote\"slash\\tab\t")), String("quote\\\"slash\\\\tab\\t"))
    assert_equal(json_escape(String("café")), String("café"))
    assert_equal(json_bool(True), String("true"))
    assert_equal(json_bool(False), String("false"))
    assert_equal(peak_vram_mib(1048576 * 12, 1048576 * 2), 10.0)
