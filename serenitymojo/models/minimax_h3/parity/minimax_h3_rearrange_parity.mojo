# serenitymojo/models/minimax_h3/parity/minimax_h3_rearrange_parity.mojo
#
# MiniMax-H3 rearrangement parity gate: patchify / unpatchify / audio unpack /
# rotate-half rope.
#
# Reference: diffusers PR huggingface/diffusers#14355 at head e1b518df, run by
# scripts/minimax_h3_rearrange_oracle.py.
#
# The tensors are `arange`-filled, so every element carries its own source index
# and a misplacement shows up as a wrong INTEGER rather than a small numeric
# difference. That is the point: these functions have no arithmetic to check,
# only index order, and a transposed axis pair yields correctly shaped, silently
# scrambled output.
#
# The rope case is compared bit-exactly: multiply and add on values the oracle
# supplies, no transcendentals evaluated on our side (cos/sin come from the
# dump), so there is nothing here that two math libraries could legitimately
# disagree about.
#
# Oracle: python3 scripts/minimax_h3_rearrange_oracle.py
# Run:
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/models/minimax_h3/parity/minimax_h3_rearrange_parity.mojo \
#     -o output/checks/minimax_h3_rearrange_parity \
#   && output/checks/minimax_h3_rearrange_parity

from std.collections import List

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.models.minimax_h3.rearrange import (
    minimax_h3_apply_rotary,
    minimax_h3_patchify_video,
    minimax_h3_unpack_audio,
    minimax_h3_unpatchify_video,
)

comptime REF = "/home/alex/mojodiffusion/output/minimax_h3_rearrange/rearrange_ref.safetensors"


def _load_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    if tv.dtype != STDtype.F32:
        raise Error(String("_load_f32: unexpected dtype for ") + name)
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


struct Report(Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def exact(mut self, label: String, got: List[Float32], want: List[Float32]):
        self.checks += 1
        if len(got) != len(want):
            self.failures += 1
            print("  FAIL", label, "length", len(got), "!=", len(want))
            return
        var bad = 0
        var first = -1
        for i in range(len(got)):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            print("  ok  ", label, "bit-exact over", len(got), "values")
        else:
            self.failures += 1
            print(
                "  FAIL", label, bad, "of", len(got), "differ; first at", first,
                "got", got[first], "want", want[first],
            )


def _arange(count: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(count):
        out.append(Float32(i))
    return out^


def _check_video(
    mut report: Report,
    ref st: SafeTensors,
    name: String,
    channels: Int,
    frames: Int,
    height: Int,
    width: Int,
) raises:
    var latents = _arange(channels * frames * height * width)
    var rows = minimax_h3_patchify_video(latents, channels, frames, height, width, 1, 2, 2)
    report.exact(
        String("video.") + name + ".rows", rows, _load_f32(st, String("video.") + name + ".rows")
    )
    var back = minimax_h3_unpatchify_video(rows, channels, frames, height, width, 1, 2, 2)
    report.exact(
        String("video.") + name + ".roundtrip",
        back,
        _load_f32(st, String("video.") + name + ".roundtrip"),
    )
    # The reference's own round trip is exact, so ours must return the input.
    report.exact(String("video.") + name + ".roundtrip_is_input", back, latents)


def _check_rope(
    mut report: Report,
    ref st: SafeTensors,
    name: String,
    sequence_length: Int,
    heads: Int,
    head_dim: Int,
    rotary_dim: Int,
) raises:
    var key = String("rope.") + name
    var out = minimax_h3_apply_rotary(
        _load_f32(st, key + ".hidden"),
        _load_f32(st, key + ".cos"),
        _load_f32(st, key + ".sin"),
        sequence_length,
        heads,
        head_dim,
        rotary_dim,
    )
    report.exact(key + ".out", out, _load_f32(st, key + ".out"))


def main() raises:
    print("MiniMax-H3 rearrangement parity gate")
    print("  reference:", REF)
    var st = SafeTensors.open(String(REF))
    var report = Report()

    print("[1] video patchify / unpatchify")
    _check_video(report, st, String("small"), 4, 2, 4, 6)
    _check_video(report, st, String("channels24"), 24, 2, 4, 4)
    _check_video(report, st, String("tall"), 3, 3, 8, 4)
    _check_video(report, st, String("single_frame"), 5, 1, 6, 6)

    print("[2] audio unpack (channel-major stereo)")
    report.exact(
        "audio.small.unpacked",
        minimax_h3_unpack_audio(_load_f32(st, "audio.small.rows"), 5, 4),
        _load_f32(st, "audio.small.unpacked"),
    )
    report.exact(
        "audio.channels32.unpacked",
        minimax_h3_unpack_audio(_load_f32(st, "audio.channels32.rows"), 3, 32),
        _load_f32(st, "audio.channels32.unpacked"),
    )

    print("[3] rotate-half rotary application")
    _check_rope(report, st, String("tiny"), 4, 2, 8, 6)
    _check_rope(report, st, String("wide"), 3, 3, 12, 12)
    _check_rope(report, st, String("h3like"), 2, 2, 16, 12)

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks, all bit-exact")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks differ")
        raise Error("MiniMax-H3 rearrangement parity gate failed")
