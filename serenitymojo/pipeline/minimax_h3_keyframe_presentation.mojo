# serenitymojo/pipeline/minimax_h3_keyframe_presentation.mojo
#
# The I2VA / FL2VA / L2VA conditioner presentation: prompt + keyframes -> token
# ids, per-token modality tags, and the `<|image_pad|>` positions the vision
# tower's embeds are spliced into. Host only: no Tensor, no DeviceContext, no
# GPU, no weights.
#
# ── WHY THIS FILE EXISTS WHEN BOTH HALVES ARE ALREADY GATED ────────────────
# `models/minimax_h3/image_grid.mojo` is gated (source size -> conditioner grid
# and token count) and `models/minimax_h3/presentation.mojo` is gated (prompt +
# token COUNTS -> ids and tags). Neither gate JOINS them, and the join is where
# a keyframe request's three load-bearing numbers come from:
#
#   1. TEXT_TOKENS. Comptime in `minimax_h3_i2va.mojo`. The keyframe's vision
#      block is part of the text run, so the budget is NOT "the prompt's
#      length" — it is the prompt plus a `"<Picture i>: "` label plus
#      `2 + vision_tokens` per keyframe. At the 1184x768 canvas that is 888
#      extra rows for ONE keyframe. Getting it wrong is a loud rebuild; getting
#      it silently wrong shifts every media row's rotary coordinate, because
#      text length is the media clock's origin (packing.py:410-413).
#   2. THE TAGS. A keyframe's vision block is tagged VIDEO (0) inside an
#      otherwise TEXT (1) run — encoders.py:166. That is the ONE way a keyframe
#      request's text region differs from t2va's, and the transformer's AdaLN
#      keys off it, so a mistagged row is modulated with the wrong parameters.
#   3. THE PAD POSITIONS. Where the tower's `[num_tokens, 5120]` embeds are
#      substituted, and where the three deepstack features are added at LANGUAGE
#      decoder layers 0/1/2 (`_deepstack_process`, modeling_qwen3_vl.py:876-884
#      — "at visual token positions only"). A position list of the right LENGTH
#      but the wrong PLACEMENT yields a perfectly-shaped, completely wrong
#      conditioning, which is exactly the failure no shape check catches.
#
# ── THE GRID IS THE CANVAS'S, NOT THE SOURCE PHOTO'S ───────────────────────
# The vendor prepares the keyframes onto the target canvas FIRST
# (before_encoder.py:190-193) and only then hands them to the image processor
# (encoders.py:154). So the conditioner grid is resolved from the CANVAS size.
# Passing the original photo's dimensions here would produce a different token
# count, a different TEXT_TOKENS, and a different rotary origin — while looking
# entirely reasonable.
#
# Gate: pipeline/parity/minimax_h3_keyframe_presentation_probe.mojo, against the
# REAL H3 tokenizer and the REAL image processor. Host, no GPU.

from std.collections import List

from serenitymojo.models.minimax_h3.image_grid import (
    MiniMaxH3ImageGrid,
    minimax_h3_image_grid,
)
from serenitymojo.models.minimax_h3.presentation import (
    MINIMAX_H3_IMAGE_PAD,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    minimax_h3_special_id,
    minimax_h3_t2va_presentation,
)
from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


@fieldwise_init
struct MiniMaxH3KeyframePresentation(Copyable, Movable):
    """What the conditioner runs on for a keyframe request.

    `pad_positions` indexes `token_ids`, in ascending order, and its length is
    `num_keyframes * vision_tokens_each` by construction — the probe checks
    both, because a correct count with a wrong placement is the silent failure
    this struct exists to prevent."""

    var token_ids: List[Int]
    var token_tags: List[Int]
    var pad_positions: List[Int]
    var vision_tokens_each: Int
    var num_keyframes: Int
    # The conditioner PATCH grid the canvas resolved to. Carried so a consumer
    # can build the tower's `MiniMaxH3VisionGrid(1, grid_h, grid_w)` from the
    # SAME resolution this presentation counted tokens from — re-resolving it at
    # the call site is how the pad count and the tower's token count drift apart.
    var grid_h: Int
    var grid_w: Int

    def num_text_tokens(self) -> Int:
        """The `TEXT_TOKENS` budget a binary must be compiled for."""
        return len(self.token_ids)

    def validate(self) raises:
        if len(self.token_ids) != len(self.token_tags):
            raise Error(
                "minimax_h3_keyframe_presentation: ids and tags differ in length"
            )
        var expected_pads = self.num_keyframes * self.vision_tokens_each
        if len(self.pad_positions) != expected_pads:
            raise Error(
                String("minimax_h3_keyframe_presentation: ")
                + String(len(self.pad_positions)) + " image_pad rows but "
                + String(self.num_keyframes) + " keyframe(s) x "
                + String(self.vision_tokens_each) + " implies "
                + String(expected_pads)
            )
        # Every pad row must be VIDEO-tagged. A pad that is TEXT-tagged would be
        # modulated as text and would also not be replaced by a vision embed.
        for i in range(len(self.pad_positions)):
            if self.token_tags[self.pad_positions[i]] != MINIMAX_H3_VIDEO_TAG:
                raise Error(
                    String("minimax_h3_keyframe_presentation: pad row ")
                    + String(self.pad_positions[i]) + " is not tagged VIDEO"
                )


def minimax_h3_keyframe_presentation(
    ref tokenizer: Qwen3Tokenizer,
    prompt: String,
    canvas_height: Int,
    canvas_width: Int,
    num_keyframes: Int,
) raises -> MiniMaxH3KeyframePresentation:
    """Compose the gated grid and the gated presentation for a keyframe request.

    `canvas_height`/`canvas_width` are the TARGET CANVAS — see this file's
    header. Every keyframe of a request is on that same canvas, so they all
    contribute the same token count; the vendor still processes them as a batch
    and numbers the labels 1..N in packed order, which
    `minimax_h3_t2va_presentation` already does."""
    if num_keyframes < 1:
        raise Error(
            "minimax_h3_keyframe_presentation: a keyframe request needs at least"
            " one keyframe — use the t2va path for none"
        )
    var grid = minimax_h3_image_grid(canvas_height, canvas_width)
    var counts = List[Int]()
    for _ in range(num_keyframes):
        counts.append(grid.num_vision_tokens)

    var presentation = minimax_h3_t2va_presentation(tokenizer, prompt, counts)

    var pad_id = minimax_h3_special_id(tokenizer, String(MINIMAX_H3_IMAGE_PAD))
    var pads = List[Int]()
    for i in range(len(presentation.ids)):
        if presentation.ids[i] == pad_id:
            pads.append(i)

    var out = MiniMaxH3KeyframePresentation(
        presentation.ids.copy(), presentation.tags.copy(), pads^,
        grid.num_vision_tokens, num_keyframes, grid.grid_h, grid.grid_w,
    )
    out.validate()
    return out^


def minimax_h3_keyframe_vision_grid(
    canvas_height: Int, canvas_width: Int
) raises -> MiniMaxH3ImageGrid:
    """The conditioner grid a canvas resolves to — exposed so a pipeline can
    report the token count before building the presentation, and so the probe
    can check the grid independently of the tokenization around it."""
    return minimax_h3_image_grid(canvas_height, canvas_width)


def minimax_h3_keyframe_text_token_budget(
    ref tokenizer: Qwen3Tokenizer,
    prompt: String,
    canvas_height: Int,
    canvas_width: Int,
    num_keyframes: Int,
) raises -> Int:
    """The `-D H3_TEXT_TOKENS=` a keyframe request needs.

    Separate entry point because a caller usually wants this ALONE, before
    building anything — the pipeline's geometry is comptime, so the budget has
    to be known before the binary that will use it is compiled."""
    var p = minimax_h3_keyframe_presentation(
        tokenizer, prompt, canvas_height, canvas_width, num_keyframes
    )
    return p.num_text_tokens()


def minimax_h3_keyframe_tag_summary(
    presentation: MiniMaxH3KeyframePresentation
) raises -> List[Int]:
    """`[num_text_tagged, num_video_tagged]`.

    The layout builder consumes the tags wholesale; this is for reporting and
    for the probe's cross-check that the two tag classes partition the run."""
    var text = 0
    var video = 0
    for i in range(len(presentation.token_tags)):
        var t = presentation.token_tags[i]
        if t == MINIMAX_H3_TEXT_TAG:
            text += 1
        elif t == MINIMAX_H3_VIDEO_TAG:
            video += 1
        else:
            raise Error(
                String("minimax_h3_keyframe_presentation: unexpected tag ")
                + String(t)
            )
    return [text, video]
