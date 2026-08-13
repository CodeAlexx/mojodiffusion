# Exact mapping gate for the inference-only MiniMax-H3 stream packer.

from std.collections import List
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.minimax_h3_frontend import (
    minimax_h3_scatter_streams,
)
from serenitymojo.tensor import Tensor


def _rows(start: Int, rows: Int, width: Int) -> List[Float32]:
    var values = List[Float32](capacity=rows * width)
    for row in range(rows):
        for column in range(width):
            values.append(Float32(start + row * width + column))
    return values^


def main() raises:
    var ctx = DeviceContext()
    var width = 16  # one aligned u64x4 chunk per BF16 row
    # Keep every integer below 128 so the FP32 fixture is exactly representable
    # after the BF16 upload boundary.
    var video_host = _rows(0, 2, width)
    var audio_host = _rows(32, 1, width)
    var text_host = _rows(48, 3, width)
    var video = Tensor.from_host(video_host, [2, width], STDtype.BF16, ctx)
    var audio = Tensor.from_host(audio_host, [1, width], STDtype.BF16, ctx)
    var text = Tensor.from_host(text_host, [3, width], STDtype.BF16, ctx)

    # Deliberately interleave every modality. Destination rows should be:
    # video[1], text[0], audio[0], video[0], text[2], text[1].
    var video_indices: List[Int] = [3, 0]
    var audio_indices: List[Int] = [2]
    var text_indices: List[Int] = [1, 5, 4]
    var packed = minimax_h3_scatter_streams(
        video, audio, text, video_indices, audio_indices, text_indices,
        6, width, ctx,
    )
    var got = packed.to_host(ctx)
    var expected = List[Float32](capacity=6 * width)
    for column in range(width):
        expected.append(video_host[width + column])
    for column in range(width):
        expected.append(text_host[column])
    for column in range(width):
        expected.append(audio_host[column])
    for column in range(width):
        expected.append(video_host[column])
    for column in range(width):
        expected.append(text_host[2 * width + column])
    for column in range(width):
        expected.append(text_host[width + column])

    if len(got) != len(expected):
        raise Error("MiniMax-H3 forward scatter gate: output length mismatch")
    for index in range(len(expected)):
        if got[index] != expected[index]:
            raise Error(
                "MiniMax-H3 forward scatter gate: mapping mismatch at "
                + String(index)
            )
    print("PASS: MiniMax-H3 disjoint forward scatter is bit-exact")
