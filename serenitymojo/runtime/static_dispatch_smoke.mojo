# static_dispatch_smoke.mojo - compile/run gate for static specialization table.

from serenitymojo.runtime.static_dispatch import (
    find_static_specialization,
    static_specialization_at,
    static_specialization_count,
)


def _check(name: String, got: Int, expected: Int) raises:
    print("[static-dispatch]", name, "got=", got, "expected=", expected)
    if got != expected:
        raise Error(String("static dispatch mismatch: ") + name)


def main() raises:
    _check(String("count"), static_specialization_count(), 4)
    var sense_smoke = find_static_specialization(
        String("sensenova_u1"), String("sensenova_u1_64_text18")
    )
    var hidream_native = find_static_specialization(
        String("hidream_o1"), String("hidream_o1_2048_s4608")
    )
    _check(String("sense smoke tokens"), sense_smoke.image_tokens, 4)
    _check(String("sense smoke text"), sense_smoke.text_tokens, 18)
    _check(String("hidream native seq"), hidream_native.total_sequence, 4608)
    for i in range(static_specialization_count()):
        var spec = static_specialization_at(i)
        print(
            "[static-dispatch]",
            spec.model_id,
            spec.profile_name,
            spec.entry_name,
            "seq=",
            spec.total_sequence,
        )
