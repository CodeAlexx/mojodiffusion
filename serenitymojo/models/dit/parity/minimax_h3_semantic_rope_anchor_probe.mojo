# serenitymojo/models/dit/parity/minimax_h3_semantic_rope_anchor_probe.mojo
#
# Exact host gate for opt-in mk2va semantic-grid time localization.

from serenitymojo.models.dit.minimax_h3_sampling import (
    MINIMAX_H3_ANCHOR_FRACTION_BASE,
    MINIMAX_H3_ANCHOR_FRACTION_SCALE,
    minimax_h3_build_sampling_geometry,
    minimax_h3_localize_semantic_rope_anchors,
)


def main() raises:
    var text_tokens = 24
    var keyframes = 6
    var latent_height = 8
    var latent_width = 8
    var patch_h = 2
    var patch_w = 2
    var rows_per_frame = 16
    var tags = List[Int]()
    for _ in range(text_tokens):
        tags.append(1)
    var anchors: List[Int] = [
        0,
        MINIMAX_H3_ANCHOR_FRACTION_BASE + 200000,
        MINIMAX_H3_ANCHOR_FRACTION_BASE + 400000,
        MINIMAX_H3_ANCHOR_FRACTION_BASE + 600000,
        MINIMAX_H3_ANCHOR_FRACTION_BASE + 800000,
        1,
    ]
    var geometry = minimax_h3_build_sampling_geometry(
        tags, 22, latent_height, latent_width, 40,
        patch_h, patch_w, anchors,
    )
    var before = geometry.position_ids.copy()
    # Two semantic rows per picture, deliberately interleaved with ordinary
    # prose rows. Group order is the Qwen presentation's picture order.
    var pads: List[Int] = [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22]
    var rows = minimax_h3_localize_semantic_rope_anchors(
        geometry, pads, text_tokens, keyframes, rows_per_frame,
    )
    if rows != 2:
        raise Error("semantic RoPE probe: rows-per-picture mismatch")

    var changed = 0
    for text_row in range(text_tokens):
        var is_pad = False
        var pad_slot = -1
        for i in range(len(pads)):
            if pads[i] == text_row:
                is_pad = True
                pad_slot = i
                break
        if is_pad:
            var keyframe = pad_slot // rows
            var condition_row = text_tokens + keyframe * rows_per_frame
            var expected = geometry.position_ids[3 * condition_row]
            if geometry.position_ids[3 * text_row] != expected:
                raise Error("semantic RoPE probe: wrong anchor time")
            if geometry.position_ids[3 * text_row] != before[3 * text_row]:
                changed += 1
        elif geometry.position_ids[3 * text_row] != before[3 * text_row]:
            raise Error("semantic RoPE probe: changed a prose token")
        if geometry.position_ids[3 * text_row + 1] != before[3 * text_row + 1]:
            raise Error("semantic RoPE probe: changed text height coordinate")
        if geometry.position_ids[3 * text_row + 2] != before[3 * text_row + 2]:
            raise Error("semantic RoPE probe: changed text width coordinate")

    if changed != len(pads):
        raise Error("semantic RoPE probe: not every image-pad row changed")
    print(
        "semantic anchors keyframes=", keyframes,
        " rows_per_picture=", rows,
        " changed_image_pad_rows=", changed,
        " prose_rows_unchanged=", text_tokens - changed,
    )
    print("PASS: semantic-grid temporal RoPE localization")
