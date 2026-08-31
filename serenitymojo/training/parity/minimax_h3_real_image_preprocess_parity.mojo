# Real eri_with_trigger decode + bucket resize/crop parity gate.
# No model, DeviceContext, VAE, cache write, or trainer is entered here.

from json.parser import loads
from json.value import JSONValue
from std.pathlib import Path
from std.sys import argv

from serenitymojo.training.minimax_h3.bucket_geometry import minimax_h3_select_bucket
from serenitymojo.training.minimax_h3.image_preprocess import minimax_h3_resize_image_to_bucket
from serenitymojo.training.minimax_h3.sha256 import minimax_h3_sha256_bytes
from serenitymojo.training.minimax_h3.source_image import minimax_h3_decode_source_rgb8


comptime SCHEMA = "serenity.minimax_h3.real_image_preprocess.v1"
comptime MUSUBI_COMMIT = "b8717864713c9e4e7ef3d56eba1fc695a9b626a5"
comptime MEDIA_UTILS_SHA256 = "b3f99b70183eab2fac857eb96a74f01b6c00902abeacb6d6aa581c7d8977ceec"


def _require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def _read_json(path: String) raises -> JSONValue:
    return loads(Path(path).read_text())


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error("usage: real_image_preprocess_parity <dataset> <fixture.json>")
    var dataset = args[1]
    var fixture = _read_json(args[2])
    _require(fixture["schema"].as_string() == String(SCHEMA), "fixture schema mismatch")
    _require(fixture["musubi_commit"].as_string() == String(MUSUBI_COMMIT), "oracle commit mismatch")
    _require(fixture["media_utils_sha256"].as_string() == String(MEDIA_UTILS_SHA256), "media oracle SHA mismatch")
    _require(fixture["dataset_identity"].as_string() == String("eri_with_trigger"), "dataset identity mismatch")
    var samples = fixture["samples"]
    _require(samples.length() == 3, "real image fixture count mismatch")
    for index in range(samples.length()):
        var row = samples[index]
        var relative = row["relative_path"].as_string()
        var decoded = minimax_h3_decode_source_rgb8(dataset + String("/") + relative)
        _require(decoded.width == Int(row["source_width"].as_int()), "decoded width mismatch: " + relative)
        _require(decoded.height == Int(row["source_height"].as_int()), "decoded height mismatch: " + relative)
        _require(minimax_h3_sha256_bytes(decoded.pixels) == row["decoded_rgb_sha256"].as_string(), "decoded RGB bytes mismatch: " + relative)
        var bucket = minimax_h3_select_bucket(decoded.width, decoded.height)
        _require(bucket.width == Int(row["bucket_width"].as_int()), "bucket width mismatch: " + relative)
        _require(bucket.height == Int(row["bucket_height"].as_int()), "bucket height mismatch: " + relative)
        var prepared = minimax_h3_resize_image_to_bucket(decoded, bucket.width, bucket.height)
        _require(prepared.width == bucket.width and prepared.height == bucket.height, "prepared geometry mismatch: " + relative)
        var prepared_sha = minimax_h3_sha256_bytes(prepared.pixels)
        if prepared_sha != row["prepared_rgb_sha256"].as_string():
            print("prepared RGB got:", prepared_sha)
            print("prepared RGB expected:", row["prepared_rgb_sha256"].as_string())
        _require(prepared_sha == row["prepared_rgb_sha256"].as_string(), "prepared RGB bytes mismatch: " + relative)
        print("PASS real H3 source image:", relative, String(decoded.width) + String("x") + String(decoded.height), String("->"), String(bucket.width) + String("x") + String(bucket.height))
    print("PASS native Mojo decode plus exact Musubi bucket preprocessing for three real eri_with_trigger images")
    print("NOT TESTED: released VAE moments, tiled encode, posterior RNG, cache artifact, or trainer")
