# Deterministic MiniMax-H3 FC1 LoRA checkpoint/runtime layout round-trip smoke.

from std.collections import List

from serenitymojo.training.minimax_h3.lora_layout import (
    minimax_h3_fc1_lora_up_musubi_to_runtime,
    minimax_h3_fc1_lora_up_runtime_to_musubi,
)


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def main() raises:
    comptime F = 3
    comptime R = 2
    # Musubi [gate rows; value rows], each row has rank columns.
    var raw = List[Float32]()
    for i in range(2 * F * R):
        raw.append(Float32(i + 1))

    var runtime = minimax_h3_fc1_lora_up_musubi_to_runtime(raw, F, R)
    for row in range(F):
        for col in range(R):
            _require(
                runtime[row * R + col] == raw[(F + row) * R + col],
                "MiniMax-H3 FC1 LoRA import did not place value rows first",
            )
            _require(
                runtime[(F + row) * R + col] == raw[row * R + col],
                "MiniMax-H3 FC1 LoRA import did not place gate rows second",
            )

    var roundtrip = minimax_h3_fc1_lora_up_runtime_to_musubi(runtime, F, R)
    _require(len(roundtrip) == len(raw), "MiniMax-H3 FC1 LoRA roundtrip length")
    for i in range(len(raw)):
        _require(roundtrip[i] == raw[i], "MiniMax-H3 FC1 LoRA roundtrip mismatch")

    var bad_rank_rejected = False
    try:
        _ = minimax_h3_fc1_lora_up_musubi_to_runtime(raw, F, 0)
    except:
        bad_rank_rejected = True
    _require(bad_rank_rejected, "MiniMax-H3 FC1 LoRA zero rank must fail")

    print("MiniMax-H3 FC1 LoRA layout roundtrip SMOKE PASS")
