# Byte-corruption gate for the production SCAIL-2 cache checksum validator.
# Usage: scail2_cache_checksum_gate <artifact-path> <expected-valid:0|1>

from std.sys import argv

from serenitymojo.models.scail2.scail2_fp8_stream import (
    validate_scail2_artifact_checksum,
)


def main() raises:
    var args = argv()
    if len(args) != 3:
        raise Error(
            "usage: scail2_cache_checksum_gate "
            "<artifact-path> <expected-valid:0|1>"
        )
    var expected = String(args[2]) == String("1")
    var actual = validate_scail2_artifact_checksum(String(args[1]))
    if actual != expected:
        raise Error("SCAIL-2 artifact checksum gate result mismatch")
    print("GATE PASS SCAIL-2 artifact checksum valid=", actual)
