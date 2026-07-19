# training/parity/ltx2_masked_av_loss_parity.mojo
#
# Musubi LTX2 masked video/audio loss parity gate.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/parity/ltx2_masked_av_loss_parity.mojo

from std.collections import List
from std.math import abs

from serenitymojo.training.ltx2.masked_loss import (
    LTX2_LOSS_HUBER,
    LTX2_LOSS_MAE,
    LTX2_LOSS_MSE,
    ltx2_loss_kind_from_string,
    masked_loss_audio_bt_mask,
    masked_loss_audio_4d_mask,
    masked_loss_unmasked,
    masked_loss_video_bf_mask,
    masked_loss_video_5d_mask,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("LTX2 masked AV loss parity failed: ") + msg)


def _close(a: Float32, b: Float32, tol: Float32 = Float32(1.0e-5)) -> Bool:
    return abs(a - b) <= tol


def _video_pred() -> List[Float32]:
    var out = List[Float32]()
    for i in range(24):
        out.append((Float32(i % 7) - Float32(3.0)) * Float32(0.25))
    return out^


def _video_target() -> List[Float32]:
    var out = List[Float32]()
    for i in range(24):
        out.append((Float32((i * 2 + 1) % 9) - Float32(4.0)) * Float32(0.2))
    return out^


def _video_bf_mask() -> List[Bool]:
    var out = List[Bool]()
    out.append(True)
    out.append(False)
    out.append(True)
    out.append(False)
    out.append(True)
    out.append(True)
    return out^


def _video_5d_b1f11_mask() -> List[Bool]:
    var out = List[Bool]()
    out.append(False)
    out.append(True)
    out.append(True)
    out.append(True)
    out.append(True)
    out.append(False)
    return out^


def _all_false(n: Int) -> List[Bool]:
    var out = List[Bool]()
    for _ in range(n):
        out.append(False)
    return out^


def _audio_pred() -> List[Float32]:
    var out = List[Float32]()
    for i in range(20):
        out.append((Float32(i % 11) - Float32(5.0)) * Float32(0.125))
    return out^


def _audio_target() -> List[Float32]:
    var out = List[Float32]()
    for i in range(20):
        out.append((Float32((i * 3 + 2) % 13) - Float32(6.0)) * Float32(0.1))
    return out^


def _audio_bt_mask() -> List[Bool]:
    var out = List[Bool]()
    out.append(True)
    out.append(True)
    out.append(False)
    out.append(True)
    out.append(False)
    return out^


def run_ltx2_masked_av_loss_parity(print_details: Bool = True) raises:
    if print_details:
        print("--- Musubi masked AV loss parity ---")

    _check(ltx2_loss_kind_from_string("mse") == LTX2_LOSS_MSE, "mse parser")
    _check(ltx2_loss_kind_from_string("l1") == LTX2_LOSS_MAE, "l1 parser")
    _check(ltx2_loss_kind_from_string("smooth_l1") == LTX2_LOSS_HUBER, "smooth_l1 parser")

    var vp = _video_pred()
    var vt = _video_target()
    var vmask = _video_bf_mask()
    var vmask5 = _video_5d_b1f11_mask()
    var vfalse = _all_false(6)

    _check(
        _close(masked_loss_video_bf_mask(vp, vt, vmask, 2, 2, 3, 2, 1, LTX2_LOSS_MSE), 0.4765625),
        "video [B,F] MSE",
    )
    _check(
        _close(masked_loss_video_bf_mask(vp, vt, vmask, 2, 2, 3, 2, 1, LTX2_LOSS_MAE), 0.574999988),
        "video [B,F] MAE",
    )
    _check(
        _close(
            masked_loss_video_bf_mask(vp, vt, vmask, 2, 2, 3, 2, 1, LTX2_LOSS_HUBER, Float32(0.5)),
            0.365000010,
        ),
        "video [B,F] Huber",
    )
    _check(
        _close(masked_loss_video_5d_mask(vp, vt, vmask5, 2, 2, 3, 2, 1, LTX2_LOSS_MSE), 0.653281212),
        "video [B,1,F,1,1] mask MSE",
    )
    _check(
        _close(masked_loss_unmasked(vp, vt, LTX2_LOSS_MSE), 0.567708313),
        "video unmasked MSE",
    )
    _check(
        _close(masked_loss_video_bf_mask(vp, vt, vfalse, 2, 2, 3, 2, 1, LTX2_LOSS_MSE), 0.567708313),
        "video all-false fallback",
    )

    var ap = _audio_pred()
    var at = _audio_target()
    var amask = _audio_bt_mask()
    var afalse = _all_false(5)

    _check(
        _close(masked_loss_audio_bt_mask(ap, at, amask, 1, 2, 5, 2, LTX2_LOSS_MSE), 0.352031261),
        "audio [B,T] MSE",
    )
    _check(
        _close(masked_loss_audio_bt_mask(ap, at, amask, 1, 2, 5, 2, LTX2_LOSS_MAE), 0.493750006),
        "audio [B,T] MAE",
    )
    _check(
        _close(
            masked_loss_audio_4d_mask(ap, at, amask, 1, 2, 5, 2, LTX2_LOSS_HUBER, Float32(0.25)),
            0.387395799,
        ),
        "audio [B,T] Huber via 4D broadcast",
    )
    _check(
        _close(masked_loss_unmasked(ap, at, LTX2_LOSS_MSE), 0.320093811),
        "audio unmasked MSE",
    )
    _check(
        _close(masked_loss_audio_bt_mask(ap, at, afalse, 1, 2, 5, 2, LTX2_LOSS_MSE), 0.320093811),
        "audio all-false fallback",
    )

    if print_details:
        print("  PASS masked video/audio loss broadcast and fallback parity")


def main() raises:
    run_ltx2_masked_av_loss_parity(True)
