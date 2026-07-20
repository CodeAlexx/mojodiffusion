# SCAIL-2's pinned FlowUniPC schedule contract.
#
# The creator constructs the training sigma table in torch.float32, takes its
# 0.999... endpoint, uses NumPy float64 linspace for inference, applies the
# runtime shift, then stores step sigmas as float32 and timesteps as truncated
# int64.  This differs from the Bernini schedule and must stay model-specific.

from std.collections import List


def build_scail2_unipc_sigma_schedule(
    num_inference_steps: Int,
    shift: Float64,
    num_train_timesteps: Int = 1000,
) raises -> List[Float64]:
    if num_inference_steps <= 0:
        raise Error("SCAIL-2 UniPC: num_inference_steps must be > 0")
    if num_train_timesteps <= 0:
        raise Error("SCAIL-2 UniPC: num_train_timesteps must be > 0")
    var ntrain = Float64(num_train_timesteps)
    # Match the creator's initial torch.float32 sigma table endpoint.
    var sigma_max = Float64(Float32((ntrain - 1.0) / ntrain))
    var out = List[Float64]()
    for i in range(num_inference_steps):
        var raw = sigma_max * (
            1.0 - Float64(i) / Float64(num_inference_steps)
        )
        var shifted = shift * raw / (1.0 + (shift - 1.0) * raw)
        out.append(Float64(Float32(shifted)))
    out.append(0.0)
    return out^


def build_scail2_unipc_timesteps(
    num_inference_steps: Int,
    shift: Float64,
    num_train_timesteps: Int = 1000,
) raises -> List[Float32]:
    if num_inference_steps <= 0:
        raise Error("SCAIL-2 UniPC: num_inference_steps must be > 0")
    if num_train_timesteps <= 0:
        raise Error("SCAIL-2 UniPC: num_train_timesteps must be > 0")
    var ntrain = Float64(num_train_timesteps)
    var sigma_max = Float64(Float32((ntrain - 1.0) / ntrain))
    var out = List[Float32]()
    for i in range(num_inference_steps):
        var raw = sigma_max * (
            1.0 - Float64(i) / Float64(num_inference_steps)
        )
        var shifted = shift * raw / (1.0 + (shift - 1.0) * raw)
        out.append(Float32(Int(shifted * ntrain)))
    return out^
