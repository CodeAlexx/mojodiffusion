# training/parity/ltx2_audio_ref_ic_contract.mojo
#
# Musubi LTX2 audio_ref_only_ic conditioning contract.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/parity/ltx2_audio_ref_ic_contract.mojo

from std.collections import List
from std.math import abs

from serenitymojo.training.ltx2.audio_ref_ic import (
    audio_ref_a2v_reference_mask,
    audio_ref_position_times,
    audio_ref_text_reference_mask,
    audio_ref_total_time,
    prepend_audio_ref_latents,
    prepend_ref_false_loss_mask,
    prepend_zero_ref_target,
    prepend_zero_ref_timesteps,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("LTX2 audio-ref IC contract failed: ") + msg)


def _close(a: Float32, b: Float32, tol: Float32 = Float32(1.0e-6)) -> Bool:
    return abs(a - b) <= tol


def _check_float_list(got: List[Float32], expected: List[Float32], msg: String) raises:
    _check(len(got) == len(expected), msg + String(" length"))
    for i in range(len(got)):
        _check(_close(got[i], expected[i]), msg + String(" at ") + String(i))


def _check_bool_list(got: List[Bool], expected: List[Bool], msg: String) raises:
    _check(len(got) == len(expected), msg + String(" length"))
    for i in range(len(got)):
        _check(got[i] == expected[i], msg + String(" at ") + String(i))


def _seq(start: Float32, n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(start + Float32(i))
    return out^


def run_ltx2_audio_ref_ic_contract(print_details: Bool = True) raises:
    if print_details:
        print("--- Musubi audio_ref_only_ic contract ---")

    comptime B = 1
    comptime C = 1
    comptime REF_T = 3
    comptime TARGET_T = 5
    comptime MEL = 2

    _check(audio_ref_total_time(REF_T, TARGET_T) == 8, "total time")

    var ref_latents = _seq(Float32(100.0), B * C * REF_T * MEL)
    var noisy_target = _seq(Float32(1.0), B * C * TARGET_T * MEL)
    var combined = prepend_audio_ref_latents(ref_latents, noisy_target, B, C, REF_T, TARGET_T, MEL)
    var expected_combined = List[Float32]()
    for i in range(6):
        expected_combined.append(Float32(100 + i))
    for i in range(10):
        expected_combined.append(Float32(1 + i))
    _check_float_list(combined, expected_combined, "ref + noisy target concat")

    var target = _seq(Float32(10.0), B * C * TARGET_T * MEL)
    var with_zero_ref_target = prepend_zero_ref_target(target, B, C, REF_T, TARGET_T, MEL)
    var expected_target = List[Float32]()
    for _ in range(6):
        expected_target.append(Float32(0.0))
    for i in range(10):
        expected_target.append(Float32(10 + i))
    _check_float_list(with_zero_ref_target, expected_target, "zero-ref target concat")

    var scalar_ts = List[Float32]()
    scalar_ts.append(Float32(0.7))
    var timesteps = prepend_zero_ref_timesteps(scalar_ts, B, REF_T, TARGET_T)
    var expected_ts = List[Float32]()
    expected_ts.append(0.0)
    expected_ts.append(0.0)
    expected_ts.append(0.0)
    for _ in range(TARGET_T):
        expected_ts.append(0.7)
    _check_float_list(timesteps, expected_ts, "zero-ref timestep scalar expansion")

    var per_token_ts = List[Float32]()
    for i in range(TARGET_T):
        per_token_ts.append(Float32(i + 1) * Float32(0.1))
    var timesteps_token = prepend_zero_ref_timesteps(per_token_ts, B, REF_T, TARGET_T)
    var expected_ts_token = List[Float32]()
    expected_ts_token.append(0.0)
    expected_ts_token.append(0.0)
    expected_ts_token.append(0.0)
    expected_ts_token.append(0.1)
    expected_ts_token.append(0.2)
    expected_ts_token.append(0.3)
    expected_ts_token.append(0.4)
    expected_ts_token.append(0.5)
    _check_float_list(timesteps_token, expected_ts_token, "zero-ref timestep token sequence")

    var target_mask = List[Bool]()
    target_mask.append(True)
    target_mask.append(True)
    target_mask.append(False)
    target_mask.append(True)
    target_mask.append(False)
    var loss_mask = prepend_ref_false_loss_mask(target_mask, B, REF_T, TARGET_T)
    var expected_loss_mask = List[Bool]()
    expected_loss_mask.append(False)
    expected_loss_mask.append(False)
    expected_loss_mask.append(False)
    expected_loss_mask.append(True)
    expected_loss_mask.append(True)
    expected_loss_mask.append(False)
    expected_loss_mask.append(True)
    expected_loss_mask.append(False)
    _check_bool_list(loss_mask, expected_loss_mask, "ref-false loss mask")

    var positions = audio_ref_position_times(REF_T, TARGET_T, False, Float32(0.5))
    var expected_positions = List[Float32]()
    expected_positions.append(0.0)
    expected_positions.append(0.5)
    expected_positions.append(1.0)
    expected_positions.append(0.0)
    expected_positions.append(0.5)
    expected_positions.append(1.0)
    expected_positions.append(1.5)
    expected_positions.append(2.0)
    _check_float_list(positions, expected_positions, "default separate positions")

    var neg_positions = audio_ref_position_times(REF_T, TARGET_T, True, Float32(0.5))
    var expected_neg = List[Float32]()
    expected_neg.append(-1.5)
    expected_neg.append(-1.0)
    expected_neg.append(-0.5)
    expected_neg.append(0.0)
    expected_neg.append(0.5)
    expected_neg.append(1.0)
    expected_neg.append(1.5)
    expected_neg.append(2.0)
    _check_float_list(neg_positions, expected_neg, "negative ref positions")

    var a2v = audio_ref_a2v_reference_mask(B, 2, REF_T, TARGET_T)
    var expected_a2v = List[Bool]()
    for _ in range(2):
        expected_a2v.append(True)
        expected_a2v.append(True)
        expected_a2v.append(True)
        expected_a2v.append(False)
        expected_a2v.append(False)
        expected_a2v.append(False)
        expected_a2v.append(False)
        expected_a2v.append(False)
    _check_bool_list(a2v, expected_a2v, "A2V mask reference columns")

    var text_valid = List[Bool]()
    text_valid.append(True)
    text_valid.append(False)
    text_valid.append(True)
    text_valid.append(True)
    var text_mask = audio_ref_text_reference_mask(text_valid, B, REF_T, TARGET_T, 4)
    _check(len(text_mask) == 32, "audio text mask shape")
    for a in range(8):
        for txt in range(4):
            var got = text_mask[a * 4 + txt]
            var expected = a < REF_T or txt == 1
            _check(got == expected, "audio text mask content")

    if print_details:
        print("  PASS audio-ref concat/timestep/target/loss-mask/position/mask contract")


def main() raises:
    run_ltx2_audio_ref_ic_contract(True)
