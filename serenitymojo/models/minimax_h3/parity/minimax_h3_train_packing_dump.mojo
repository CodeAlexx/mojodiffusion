# minimax_h3_train_packing_dump — twin of scripts/minimax_h3_train_packing_oracle.py:
# emits the SAME bitcast-integer text format from the Mojo builder so a plain
# line diff gates the TRAINING packed layout against the pinned Musubi fork.
# CPU-only (no DeviceContext) — safe to run while a training job owns the GPU.
#
# Compare with: python3 scripts/minimax_h3_train_packing_compare.py
from std.memory import bitcast
from serenitymojo.models.minimax_h3.packing import (
    minimax_h3_build_packed_sequence, minimax_h3_build_row_timesteps,
    MINIMAX_H3_ANCHOR_FIRST, MINIMAX_H3_ANCHOR_LAST,
)
from serenitymojo.models.minimax_h3.h3_train_sigma import h3_shift_sigma

comptime U = Float32(0.4375)
comptime TEXT = 87
comptime F = 37
comptime LH = 16
comptime LW = 28
comptime AT = 207


def _emit_case(name: String, anchors: List[Int]) raises:
    var tags = List[Int]()
    for _ in range(TEXT):
        tags.append(1)
    var lay = minimax_h3_build_packed_sequence(
        tags, F, LH, LW, AT, 2, 2, anchors, Float64(1.0)
    )
    var t_v = Float32(1.0) - h3_shift_sigma(U, Float32(12.0))
    var t_a = Float32(1.0) - h3_shift_sigma(U, Float32(3.0))
    var ts = minimax_h3_build_row_timesteps(
        lay, t_v, t_a, Float32(0.999), Float32(1.0)
    )
    print("case", name)
    print("S", lay.sequence_length)
    # segment table in packed order: text | visual conditions | audio | video
    var n_cond = lay.num_condition_video_rows
    var rows_pf = (LH // 2) * (LW // 2)
    print("seg text 0", TEXT)
    var cursor = TEXT
    for i in range(len(anchors)):
        print(
            "seg visual_condition_" + String(i), cursor, cursor + rows_pf
        )
        cursor += rows_pf
    if n_cond != len(anchors) * rows_pf:
        raise Error("condition row count mismatch")
    print("seg target_audio", cursor, cursor + 2 * AT)
    cursor += 2 * AT
    print("seg target_video", cursor, cursor + F * rows_pf)
    print("pos")
    for i in range(len(lay.position_ids)):
        print(bitcast[DType.uint64](lay.position_ids[i]))
    print("rowts")
    for i in range(len(ts.indices)):
        print(bitcast[DType.uint32](ts.values[ts.indices[i]]))
    print("tags")
    for i in range(len(lay.token_tags)):
        print(lay.token_tags[i])
    print("adaln")
    for i in range(len(ts.indices)):
        var tag = lay.token_tags[i]
        if tag < 0:
            tag = 0
        print(ts.indices[i] * 3 + tag)
    print("endcase")


def main() raises:
    var t_v = Float32(1.0) - h3_shift_sigma(U, Float32(12.0))
    var t_a = Float32(1.0) - h3_shift_sigma(U, Float32(3.0))
    print("tv", bitcast[DType.uint32](t_v))
    print("ta", bitcast[DType.uint32](t_a))
    var none = List[Int]()
    _emit_case(String("av_t2va"), none)
    var fl = List[Int]()
    fl.append(MINIMAX_H3_ANCHOR_FIRST)
    fl.append(MINIMAX_H3_ANCHOR_LAST)
    _emit_case(String("av_fl2va"), fl)
