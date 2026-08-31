# Pure-Mojo H3 bucket geometry parity against pinned Musubi evidence.

from json.parser import loads
from std.pathlib import Path

from serenitymojo.training.minimax_h3.bucket_geometry import (
    MINIMAX_H3_BUCKET_STEP,
    minimax_h3_select_bucket,
)


comptime FIXTURE = "serenitymojo/training/parity/fixtures/minimax_h3_bucket_geometry_v1.json"
comptime SCHEMA = "serenity.minimax_h3.bucket_geometry_oracle.v1"
comptime ORACLE_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
comptime SOURCE_FILE = "src/musubi_tuner/dataset/bucket.py"
comptime SOURCE_SHA256 = "2cb4d4c1c74f3becb00070aac84597529cec1313865c9042a7149c2ae41cc1ed"


def _check(condition: Bool, message: String) raises:
    if not condition:
        raise Error(String("MiniMax H3 bucket parity failed: ") + message)


def main() raises:
    var doc = loads(Path(String(FIXTURE)).read_text())
    _check(doc[String("schema")].as_string() == String(SCHEMA), "schema")
    _check(
        doc[String("oracle_commit")].as_string() == String(ORACLE_COMMIT),
        "oracle commit",
    )
    _check(doc[String("source_file")].as_string() == String(SOURCE_FILE), "source file")
    _check(
        doc[String("source_sha256")].as_string() == String(SOURCE_SHA256),
        "source SHA-256",
    )
    _check(doc[String("step")].as_int() == MINIMAX_H3_BUCKET_STEP, "step")
    var resolution = doc[String("resolution")]
    _check(
        resolution.length() == 2
        and resolution[0].as_int() == 1024
        and resolution[1].as_int() == 1024,
        "resolution",
    )
    var cases = doc[String("cases")]
    _check(cases.length() == 36, "case count")
    var bucket_count = 0
    var no_upscale_count = 0
    var fixed_count = 0
    for index in range(cases.length()):
        var item = cases[index]
        var source = item[String("source")]
        var target = item[String("target")]
        var mode = item[String("mode")].as_string()
        if mode == String("bucket"):
            bucket_count += 1
        elif mode == String("no_upscale"):
            no_upscale_count += 1
        elif mode == String("fixed"):
            fixed_count += 1
        else:
            raise Error(String("MiniMax H3 bucket parity unknown mode: ") + mode)
        var got = minimax_h3_select_bucket(
            source[0].as_int(),
            source[1].as_int(),
            resolution[0].as_int(),
            resolution[1].as_int(),
            mode != String("fixed"),
            mode == String("no_upscale"),
        )
        _check(
            got.width == target[0].as_int() and got.height == target[1].as_int(),
            String("case ") + String(index),
        )
    _check(
        bucket_count == 12 and no_upscale_count == 12 and fixed_count == 12,
        "mode inventory",
    )
    print("PASS MiniMax H3 bucket geometry Musubi parity; cases:", cases.length())
