# timestep_inverted_parabola_smoke.mojo — parity probe for the SerenityTrainer
# INVERTED_PARABOLA discrete timestep draw (schedule.mojo).
#
# Prints the N-bucket weight table (compared elementwise against SerenityTrainer's
# torch-computed weights by scripts/check_inverted_parabola_dist.py) and a
# 200k-draw histogram (compared against the exact normalized PMF).
#
#   pixi run mojo run -I . serenitymojo/training/timestep_inverted_parabola_smoke.mojo
#
# Chroma 24GB-LoRA preset operating point: weight=7.7, bias=0.0, shift=1.0,
# N=1000.

from serenitymojo.training.schedule import (
    inverted_parabola_weight_table,
    sample_timestep_idx_from_weight_table,
)

comptime N = 1000
comptime WEIGHT = Float32(7.7)
comptime BIAS = Float32(0.0)
comptime SHIFT = Float32(1.0)
comptime DRAWS = 2000000


def main() raises:
    var table = inverted_parabola_weight_table(WEIGHT, BIAS, SHIFT, N)
    print("WEIGHTS", len(table))
    for i in range(len(table)):
        print(table[i])

    var hist = List[Int]()
    for _ in range(N):
        hist.append(0)
    for k in range(DRAWS):
        var idx = sample_timestep_idx_from_weight_table(UInt64(k), table)
        hist[idx] += 1
    print("HIST", DRAWS)
    for i in range(N):
        print(hist[i])
