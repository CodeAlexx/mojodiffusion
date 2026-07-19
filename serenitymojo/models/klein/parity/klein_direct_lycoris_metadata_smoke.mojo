# models/klein/parity/klein_direct_lycoris_metadata_smoke.mojo
#
# Proves Klein direct DoRA/OFT slot metadata and byte preflight. This is not the
# live GPU block lowering gate.
#
# Build/run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/klein/parity/klein_direct_lycoris_metadata_smoke.mojo \
#     -o /tmp/klein_direct_lycoris_metadata_smoke && \
#   /tmp/klein_direct_lycoris_metadata_smoke

from serenitymojo.models.klein.klein_direct_lycoris_stack import (
    KLEIN_DIRECT_24_GIB,
    klein_direct_active_slot_count,
    klein_direct_dense_carrier_bytes,
    klein_direct_dora_trainable_bytes_estimate,
    klein_direct_oft_trainable_bytes_estimate,
    klein_direct_dora_preflight,
    klein_direct_oft_preflight,
    build_klein_direct_oft_set,
    klein_direct_oft_trainable_bytes,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(msg)


def main() raises:
    print("=== klein direct LyCORIS metadata smoke ===")

    var D = 32
    var F = 48
    _check(klein_direct_active_slot_count(2, 1, 1) == 16, "target 1 count mismatch")
    _check(klein_direct_active_slot_count(2, 1, 2) == 24, "target 2 count mismatch")
    _check(klein_direct_active_slot_count(2, 1, 3) == 26, "target 3 count mismatch")

    var dense_t1 = klein_direct_dense_carrier_bytes(2, 1, D, F, 1)
    var dense_t2 = klein_direct_dense_carrier_bytes(2, 1, D, F, 2)
    var dense_t3 = klein_direct_dense_carrier_bytes(2, 1, D, F, 3)
    print("  small dense carrier bytes: t1=", dense_t1, " t2=", dense_t2, " t3=", dense_t3)
    _check(dense_t1 < dense_t2 and dense_t2 < dense_t3, "carrier bytes should grow by target set")

    var dora_t3 = klein_direct_dora_trainable_bytes_estimate(2, 1, D, F, 4, 3)
    var oft_t3 = klein_direct_oft_trainable_bytes_estimate(2, 1, D, F, 4, 3)
    print("  small direct bytes: dora_t3=", dora_t3, " oft_t3=", oft_t3)
    _check(dora_t3 < dense_t3, "DoRA direct state should be smaller than dense carrier")
    _check(oft_t3 < dense_t3, "OFT direct state should be smaller than dense carrier")

    var oft = build_klein_direct_oft_set(2, 1, D, F, 4, 3)
    _check(klein_direct_oft_trainable_bytes(oft) == oft_t3, "OFT live bytes mismatch")

    var RD = 4096
    var RF = 12288
    var rdense1 = klein_direct_dense_carrier_bytes(8, 24, RD, RF, 1)
    var rdense2 = klein_direct_dense_carrier_bytes(8, 24, RD, RF, 2)
    var rdense3 = klein_direct_dense_carrier_bytes(8, 24, RD, RF, 3)
    var rdora2 = klein_direct_dora_preflight(8, 24, RD, RF, 16, 2, KLEIN_DIRECT_24_GIB)
    var roft2 = klein_direct_oft_preflight(8, 24, RD, RF, 4, 2, KLEIN_DIRECT_24_GIB)
    var rdora3 = klein_direct_dora_preflight(8, 24, RD, RF, 16, 3, KLEIN_DIRECT_24_GIB)
    var roft3 = klein_direct_oft_preflight(8, 24, RD, RF, 4, 3, KLEIN_DIRECT_24_GIB)
    print("  real dense carrier bytes: t1=", rdense1, " t2=", rdense2, " t3=", rdense3)
    print("  real direct bytes: dora_t2=", rdora2, " oft_t2=", roft2, " dora_t3=", rdora3, " oft_t3=", roft3)
    _check(rdora2 < rdense2 and roft2 < rdense2, "target 2 direct state should beat carrier")
    _check(rdora3 < rdense3 and roft3 < rdense3, "target 3 direct state should beat carrier")

    print("PASS -- klein direct LyCORIS metadata/preflight smoke")
