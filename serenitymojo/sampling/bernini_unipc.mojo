# Bernini-R's exact Diffusers UniPC flow schedule.
#
# The shared Wan/Cosmos helper starts from (N_train-1)/N_train and ends at
# zero before appending the terminal zero.  Bernini's pinned Diffusers
# scheduler instead uses:
#   linspace(1, 1/N_train, steps+1)[:-1]
#   shift*s / (1 + (shift-1)*s)
#   first_sigma -= 1e-6
# and stores the resulting sigma table as F32.  Keep this model-specific
# schedule separate so the already-gated Wan/Cosmos contract is unchanged.

from std.collections import List


def build_bernini_unipc_sigma_schedule(
    num_inference_steps: Int, shift: Float64, num_train_timesteps: Int,
) raises -> List[Float64]:
    if num_inference_steps <= 0:
        raise Error("Bernini UniPC: num_inference_steps must be > 0")
    if num_train_timesteps <= 0:
        raise Error("Bernini UniPC: num_train_timesteps must be > 0")
    var out = List[Float64]()
    var sigma_min = 1.0 / Float64(num_train_timesteps)
    for i in range(num_inference_steps):
        var raw = 1.0 + (sigma_min - 1.0) * (
            Float64(i) / Float64(num_inference_steps)
        )
        var shifted = shift * raw / (1.0 + (shift - 1.0) * raw)
        if i == 0:
            # diffusers UniPC avoids log(alpha=0) on its first update.
            shifted -= 1.0e-6
        # The creator scheduler materializes `sigmas` as np.float32 before
        # stepping.  Round here too; coefficients then intentionally run F64
        # over those creator-owned F32 values.
        out.append(Float64(Float32(shifted)))
    out.append(0.0)
    return out^


def build_bernini_unipc_timesteps(
    num_inference_steps: Int, shift: Float64, num_train_timesteps: Int,
) raises -> List[Float32]:
    """Creator model timesteps (shifted sigma*1000, truncated to int64)."""
    if num_inference_steps <= 0:
        raise Error("Bernini UniPC: num_inference_steps must be > 0")
    var out = List[Float32]()
    var sigma_min = 1.0 / Float64(num_train_timesteps)
    for i in range(num_inference_steps):
        var raw = 1.0 + (sigma_min - 1.0) * (
            Float64(i) / Float64(num_inference_steps)
        )
        var shifted = shift * raw / (1.0 + (shift - 1.0) * raw)
        if i == 0:
            shifted -= 1.0e-6
        out.append(Float32(Int(shifted * Float64(num_train_timesteps))))
    return out^
