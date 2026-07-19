# snapshot tests — pure-function units.
#   pixi run --manifest-path /home/alex/rill/pixi.toml mojo run -I . \
#       snapshot/tests/snapshot_test.mojo

from std.testing import assert_equal, assert_true, TestSuite

from snapshot.snapshot import (
    fnv1a_hex,
    to_hex16,
    count_lines,
    ext_of,
    is_text_ext,
    is_excluded_dir,
    human_size,
    symbols_for,
)


def test_hex16() raises:
    assert_equal(to_hex16(UInt64(255)), "00000000000000ff")
    assert_equal(to_hex16(UInt64(0)), "0000000000000000")


def test_fnv1a_deterministic() raises:
    var a = fnv1a_hex(String("hello world"))
    var b = fnv1a_hex(String("hello world"))
    var c = fnv1a_hex(String("hello worle"))
    assert_equal(a, b)  # same input -> same hash
    assert_true(a != c)  # one byte change -> different hash
    assert_equal(a.byte_length(), 16)


def test_count_lines() raises:
    assert_equal(count_lines(String("a\nb\nc\n")), 3)
    assert_equal(count_lines(String("no newline")), 0)
    assert_equal(count_lines(String("")), 0)


def test_ext_of() raises:
    assert_equal(ext_of(String("foo.MOJO")), "mojo")  # lowercased
    assert_equal(ext_of(String("a.b.py")), "py")  # final segment
    assert_equal(ext_of(String("README")), "")  # no dot
    assert_equal(ext_of(String(".gitignore")), "gitignore")


def test_classify() raises:
    assert_true(is_text_ext(String("mojo")))
    assert_true(is_text_ext(String("rs")))
    assert_true(not is_text_ext(String("png")))
    assert_true(is_excluded_dir(String(".git")))
    assert_true(is_excluded_dir(String("node_modules")))
    assert_true(not is_excluded_dir(String("src")))


def test_human_size() raises:
    assert_equal(human_size(512), "512B")
    assert_equal(human_size(1024), "1.0K")
    assert_equal(human_size(1024 * 1024), "1.0M")
    assert_equal(human_size(1536), "1.5K")


def test_symbols_mojo() raises:
    var src = String(
        "struct Foo(Copyable):\n"
        + "    var x: Int\n"
        + "    def method(self):\n"  # indented -> skipped (top-level only)
        + "        pass\n"
        + "def top_level() -> Int:\n"
        + "    return 1\n"
        + "trait Bar:\n"
        + "    pass\n"
    )
    var syms = symbols_for(src, String("mojo"))
    assert_equal(len(syms), 3)
    assert_equal(syms[0], "struct Foo")
    assert_equal(syms[1], "def top_level")
    assert_equal(syms[2], "trait Bar")


def test_symbols_python() raises:
    var src = String(
        "class C:\n    def m(self):\n        pass\nasync def fetch():\n    pass\n"
    )
    var syms = symbols_for(src, String("py"))
    assert_equal(len(syms), 2)
    assert_equal(syms[0], "class C")
    assert_equal(syms[1], "async def fetch")


def test_symbols_none() raises:
    # non-source extension -> no symbols
    var syms = symbols_for(String("# title\nbody\n"), String("md"))
    assert_equal(len(syms), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
