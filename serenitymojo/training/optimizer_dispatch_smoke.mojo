# optimizer_dispatch_smoke.mojo — fixed checks for OneTrainer optimizer dispatch.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/optimizer_dispatch_smoke.mojo

from serenitymojo.training.optimizer_dispatch import (
    OPT_BACKEND_ADAFACTOR_HOST,
    OPT_BACKEND_FUSED_ADAMW,
    OPT_BACKEND_SGD_MOMENTUM,
    OPT_BACKEND_UNSUPPORTED,
    OPT_STATE_ADAFACTOR_FACTORED_HOST_F32,
    OPT_STATE_ADAM_M_V_F32,
    OPT_STATUS_ALIASED,
    OPT_STATUS_SUPPORTED,
    OPT_STATUS_UNSUPPORTED,
    optimizer_dispatch_for_identifier,
)


def _assert_dispatch(
    identifier: String,
    canonical: String,
    backend: Int,
    state_kind: Int,
    status: Int,
) raises:
    var d = optimizer_dispatch_for_identifier(identifier)
    if d.canonical != canonical:
        print("dispatch canonical mismatch:", d)
        raise Error("optimizer dispatch canonical mismatch")
    if d.backend != backend:
        print("dispatch backend mismatch:", d)
        raise Error("optimizer dispatch backend mismatch")
    if d.state_kind != state_kind:
        print("dispatch state mismatch:", d)
        raise Error("optimizer dispatch state mismatch")
    if d.status != status:
        print("dispatch status mismatch:", d)
        raise Error("optimizer dispatch status mismatch")


def main() raises:
    _assert_dispatch(
        "<missing>",
        "ADAMW",
        OPT_BACKEND_FUSED_ADAMW,
        OPT_STATE_ADAM_M_V_F32,
        OPT_STATUS_SUPPORTED,
    )
    _assert_dispatch(
        "None",
        "ADAMW",
        OPT_BACKEND_FUSED_ADAMW,
        OPT_STATE_ADAM_M_V_F32,
        OPT_STATUS_SUPPORTED,
    )
    _assert_dispatch(
        "ADAMW",
        "ADAMW",
        OPT_BACKEND_FUSED_ADAMW,
        OPT_STATE_ADAM_M_V_F32,
        OPT_STATUS_SUPPORTED,
    )
    _assert_dispatch(
        "ADAFACTOR",
        "ADAFACTOR",
        OPT_BACKEND_ADAFACTOR_HOST,
        OPT_STATE_ADAFACTOR_FACTORED_HOST_F32,
        OPT_STATUS_SUPPORTED,
    )

    # Explicitly document non-target names that must not silently fall through.
    _assert_dispatch(
        "SGD",
        "SGD",
        OPT_BACKEND_SGD_MOMENTUM,
        2,
        OPT_STATUS_ALIASED,
    )
    _assert_dispatch(
        "ADAMW_8BIT",
        "ADAMW_8BIT",
        OPT_BACKEND_UNSUPPORTED,
        0,
        OPT_STATUS_UNSUPPORTED,
    )
    _assert_dispatch(
        "SCHEDULE_FREE_ADAMW",
        "SCHEDULE_FREE_ADAMW",
        OPT_BACKEND_UNSUPPORTED,
        0,
        OPT_STATUS_UNSUPPORTED,
    )

    print("PASS: optimizer dispatch covers target OneTrainer identifiers")
