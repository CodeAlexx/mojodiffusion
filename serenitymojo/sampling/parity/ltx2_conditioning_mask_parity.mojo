"""Executable LTX2 I2V/V2V mask and timestep parity gate.

The independent scalar oracle is the upstream contract:
  denoise_mask = 1 - source_strength
  token_timestep = sigma * denoise_mask
The Mojo model receives timesteps multiplied by 1000 before AdaLN.
"""

from std.collections import List
from max.gpu.host import DeviceContext

from serenitymojo.pipeline.ltx2_t2v_av_hq import (
    _request_hq_i2v_mask,
    _request_hq_v2v_mask,
    _request_hq_v2v_spatial_mask,
    _request_hq_video_timestep_values,
)


def _abs(value: Float32) -> Float32:
    return value if value >= Float32(0.0) else -value


def _check(
    label: String,
    actual: List[Float32],
    expected: List[Float32],
    tolerance: Float32 = Float32(1.0e-6),
) raises:
    if len(actual) != len(expected):
        raise Error(
            label + String(": length mismatch ")
            + String(len(actual)) + String(" != ") + String(len(expected))
        )
    for i in range(len(actual)):
        if _abs(actual[i] - expected[i]) > tolerance:
            raise Error(
                label + String(": value mismatch at ") + String(i)
                + String(": ") + String(actual[i])
                + String(" != ") + String(expected[i])
            )
    print("[PASS]", label, "rows=", len(actual))


def _repeat(count: Int, value: Float32) -> List[Float32]:
    var values = List[Float32]()
    for _ in range(count):
        values.append(value)
    return values^


def main() raises:
    var ctx = DeviceContext()
    var sequence_tokens = 8
    var frame_zero_tokens = 2
    var source_strength = Float32(0.75)
    var denoise = Float32(1.0) - source_strength
    var scale = Float32(0.8)
    var sigma = Float32(0.6)

    var i2v_expected = _repeat(sequence_tokens, scale)
    for i in range(frame_zero_tokens):
        i2v_expected[i] = denoise * scale
    _check(
        String("I2V noiser mask"),
        _request_hq_i2v_mask(
            sequence_tokens, frame_zero_tokens, denoise, scale, ctx
        ).to_host(ctx),
        i2v_expected,
    )

    _check(
        String("V2V noiser mask"),
        _request_hq_v2v_mask(
            sequence_tokens, denoise, scale, ctx
        ).to_host(ctx),
        _repeat(sequence_tokens, denoise * scale),
    )

    var i2v_ts_expected = _repeat(
        sequence_tokens, sigma * Float32(1000.0)
    )
    for i in range(frame_zero_tokens):
        i2v_ts_expected[i] = sigma * denoise * Float32(1000.0)
    _check(
        String("I2V model timesteps"),
        _request_hq_video_timestep_values(
            sigma, sequence_tokens, frame_zero_tokens, denoise
        ),
        i2v_ts_expected,
    )

    _check(
        String("V2V model timesteps"),
        _request_hq_video_timestep_values(
            sigma, sequence_tokens, sequence_tokens, denoise
        ),
        _repeat(sequence_tokens, sigma * denoise * Float32(1000.0)),
    )

    var spatial_values: List[Float32] = [
        0.0, denoise, 0.0, denoise,
        0.0, denoise, 0.0, denoise,
    ]
    var spatial_scaled = List[Float32]()
    var spatial_timesteps = List[Float32]()
    for value in spatial_values:
        spatial_scaled.append(value * scale)
        spatial_timesteps.append(value * sigma * Float32(1000.0))
    _check(
        String("V2V spatial noiser mask"),
        _request_hq_v2v_spatial_mask(
            spatial_values, sequence_tokens, scale, ctx
        ).to_host(ctx),
        spatial_scaled,
    )
    _check(
        String("V2V spatial model timesteps"),
        _request_hq_video_timestep_values(
            sigma, sequence_tokens, sequence_tokens, denoise, False,
            Optional[List[Float32]](spatial_values.copy()),
        ),
        spatial_timesteps,
    )

    _check(
        String("T2V uniform broadcast timestep"),
        _request_hq_video_timestep_values(
            sigma, sequence_tokens, 0, Float32(1.0), True
        ),
        _repeat(1, sigma * Float32(1000.0)),
    )
    print("LTX2 conditioning mask/timestep parity PASS")
