# static_entrypoints_smoke.mojo - compile/run gate for static entry contracts.

from serenitymojo.runtime.model_manifest import ModelFamily
from serenitymojo.runtime.request import GenerationRequest
from serenitymojo.runtime.static_entrypoints import (
    StaticEntrypointContract,
    find_static_entrypoint,
    static_entrypoint_at,
    static_entrypoint_count,
    validate_request_for_static_entrypoint,
    validate_static_entrypoint,
)


def _check_int(name: String, got: Int, expected: Int) raises:
    print("[static-entrypoint]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("static entrypoint int mismatch: ") + name)


def _check_bool(name: String, got: Bool, expected: Bool) raises:
    print("[static-entrypoint]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("static entrypoint bool mismatch: ") + name)


def _check_string(name: String, got: String, expected: String) raises:
    print("[static-entrypoint]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("static entrypoint string mismatch: ") + name)


def _print_contract(contract: StaticEntrypointContract):
    print(
        "[static-entrypoint]",
        contract.model_id,
        contract.profile_name,
        contract.static_type_name,
        "kind=",
        contract.entry_kind,
        "shape=",
        contract.width,
        "x",
        contract.height,
        "seq=",
        contract.total_sequence,
        "path=",
        contract.declared_entry_path,
    )


def _request_for(contract: StaticEntrypointContract) -> GenerationRequest:
    return GenerationRequest(
        contract.model_id,
        ModelFamily.text_to_image(),
        String("static entrypoint smoke"),
        String(""),
        contract.width,
        contract.height,
        contract.frames,
        1,
        UInt64(42),
        Float32(1.0),
        String("/tmp/static_entrypoint.png"),
    )


def main() raises:
    _check_int(String("count"), static_entrypoint_count(), 4)

    var sense_smoke = find_static_entrypoint(
        String("sensenova_u1"), String("sensenova_u1_64_text18")
    )
    var sense_native = find_static_entrypoint(
        String("sensenova_u1"), String("sensenova_u1_2048_text512")
    )
    var hidream_smoke = find_static_entrypoint(
        String("hidream_o1"), String("hidream_o1_64_s20")
    )
    var hidream_native = find_static_entrypoint(
        String("hidream_o1"), String("hidream_o1_2048_s4608")
    )

    validate_static_entrypoint(sense_smoke)
    validate_static_entrypoint(sense_native)
    validate_static_entrypoint(hidream_smoke)
    validate_static_entrypoint(hidream_native)

    _check_string(
        String("sense wrapper"),
        sense_smoke.wrapper_name,
        String("sensenova_u1_static_entry"),
    )
    _check_string(
        String("hidream wrapper"),
        hidream_smoke.wrapper_name,
        String("hidream_o1_static_entry"),
    )
    _check_bool(String("sense smoke runnable"), sense_smoke.runnable_smoke, True)
    _check_bool(String("sense native runnable"), sense_native.runnable_smoke, False)
    _check_bool(String("sense text bucket"), sense_smoke.text_bucket_matches(18), True)
    _check_bool(
        String("hidream common cfg S"),
        hidream_native.requires_common_cfg_sequence,
        True,
    )
    _check_int(String("sense native tokens"), sense_native.image_tokens, 4096)
    _check_int(String("hidream native seq"), hidream_native.total_sequence, 4608)

    var req = _request_for(sense_smoke)
    validate_request_for_static_entrypoint(req, sense_smoke)

    for i in range(static_entrypoint_count()):
        var contract = static_entrypoint_at(i)
        validate_static_entrypoint(contract)
        _print_contract(contract)
