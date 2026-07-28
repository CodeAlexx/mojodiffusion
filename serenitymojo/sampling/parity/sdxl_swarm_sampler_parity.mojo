# Scalar and tensor parity gate for the SDXL SwarmUI sampler slice.
#
# Oracle: the bundled SwarmUI ComfyUI checkout:
#   comfy/model_sampling.py::ModelSamplingDiscrete
#   comfy/samplers.py::{normal,simple,ddim}_scheduler
#   comfy/k_diffusion/sampling.py::{get_sigmas_karras,
#                                   get_sigmas_exponential,
#                                   sample_dpmpp_2m}

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.sampling.sdxl_euler import (
    build_sdxl_swarm_sigmas,
    build_sdxl_swarm_timesteps,
    sdxl_dpmpp_2m_step,
)


def _abs(value: Float32) -> Float32:
    return value if value >= 0.0 else -value


def _check(actual: Float32, expected: Float32, tolerance: Float32, label: String) raises:
    if _abs(actual - expected) > tolerance:
        raise Error(
            label + String(": got ") + String(actual)
            + String(" expected ") + String(expected)
        )


def _check_schedule(name: String, expected: List[Float32], expected_t: List[Int]) raises:
    var sigmas = build_sdxl_swarm_sigmas(5, name)
    var timesteps = build_sdxl_swarm_timesteps(sigmas)
    if len(sigmas) != len(expected) or len(timesteps) != len(expected_t):
        raise Error(name + String(": schedule length mismatch"))
    for i in range(len(expected)):
        _check(sigmas[i], expected[i], 2.0e-5, name + String(" sigma ") + String(i))
    for i in range(len(expected_t)):
        if Int(timesteps[i]) != expected_t[i]:
            raise Error(
                name + String(": timestep mismatch at ") + String(i)
                + String(" got ") + String(timesteps[i])
                + String(" expected ") + String(expected_t[i])
            )


def _list6(
    a: Float32, b: Float32, c: Float32, d: Float32, e: Float32, f: Float32
) -> List[Float32]:
    var out = List[Float32]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    out.append(e)
    out.append(f)
    return out^


def _t5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    out.append(e)
    return out^


def main() raises:
    _check_schedule(
        String("normal"),
        _list6(14.614640, 4.086081, 1.615580, 0.695150, 0.029167, 0.0),
        _t5(999, 749, 500, 250, 0),
    )
    _check_schedule(
        String("simple"),
        _list6(14.614641, 5.087763, 2.276463, 1.160578, 0.569285, 0.0),
        _t5(999, 799, 599, 399, 199),
    )
    _check_schedule(
        String("ddim_uniform"),
        _list6(5.134431, 2.292854, 1.168216, 0.574050, 0.041314, 0.0),
        _t5(801, 601, 401, 201, 1),
    )
    _check_schedule(
        String("karras"),
        _list6(14.614643, 4.796535, 1.274101, 0.247950, 0.029167, 0.0),
        _t5(999, 786, 427, 59, 0),
    )
    _check_schedule(
        String("exponential"),
        _list6(14.614640, 3.088976, 0.652892, 0.137996, 0.029167, 0.0),
        _t5(999, 681, 233, 20, 0),
    )

    var ctx = DeviceContext()
    var x = Tensor.from_host(_list6(2.0, 2.0, 2.0, 2.0, 2.0, 2.0), [6], STDtype.F32, ctx)
    var d0 = Tensor.from_host(_list6(0.5, 0.5, 0.5, 0.5, 0.5, 0.5), [6], STDtype.F32, ctx)
    var step0 = sdxl_dpmpp_2m_step(
        x, d0, d0, False, 10.0, 3.0, 0.0, ctx
    )
    _check(step0.to_host(ctx)[0], 0.95000005, 2.0e-6, String("DPM++ 2M step 0"))

    var d1 = Tensor.from_host(_list6(0.25, 0.25, 0.25, 0.25, 0.25, 0.25), [6], STDtype.F32, ctx)
    var step1 = sdxl_dpmpp_2m_step(
        step0, d1, d0, True, 3.0, 1.0, 10.0, ctx
    )
    _check(step1.to_host(ctx)[0], 0.40729260, 2.0e-6, String("DPM++ 2M step 1"))

    var d2 = Tensor.from_host(_list6(0.1, 0.1, 0.1, 0.1, 0.1, 0.1), [6], STDtype.F32, ctx)
    var terminal = sdxl_dpmpp_2m_step(
        step1, d2, d1, True, 1.0, 0.0, 3.0, ctx
    )
    _check(terminal.to_host(ctx)[0], 0.1, 1.0e-7, String("DPM++ 2M terminal"))
    print("SDXL SwarmUI sampler parity: PASS")
