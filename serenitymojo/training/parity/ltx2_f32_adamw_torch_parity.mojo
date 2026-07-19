# ltx2_f32_adamw_torch_parity.mojo — digit-level gate for _adamw_host_list_f32
# (the F32-master AdamW the ltx2 driver runs per step, landed 2026-07-16 after
# the bf16-master write-back was convicted — 30-57% update absorption).
#
# Emits p/m/v after K steps on deterministic integer-hash inputs; the
# companion scripts/check_f32_adamw_vs_torch.py replays the SAME inputs
# through torch.optim.AdamW (single-tensor, f32) and compares. Bar: max
# |Δ| ≤ 4 ulp of the value (the ideogram4 fused-AdamW gate bar).
#
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/training/parity/ltx2_f32_adamw_torch_parity.mojo \
#     -o /tmp/ltx2_f32_adamw_gate
#   /tmp/ltx2_f32_adamw_gate > /tmp/f32_adamw_mojo.txt
#   python scripts/check_f32_adamw_vs_torch.py /tmp/f32_adamw_mojo.txt

from serenitymojo.training.train_step import _adamw_host_list_f32

comptime N = 4096
comptime K = 10
comptime LR = Float32(1.0e-4)
comptime WD = Float32(0.01)


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _u01(seed: UInt64) -> Float32:
    # top 24 bits -> [0,1); EXACTLY mirrored in the python checker.
    return Float32(Float64(_splitmix64(seed) >> 40) * (1.0 / 16777216.0))


def main() raises:
    # Two magnitude regimes: A-class (kaiming rms ~9e-3) and B-class (~1e-3),
    # grads at the measured per-element scale ~1.2e-5.
    for regime in range(2):
        var pscale = Float32(0.02) if regime == 0 else Float32(0.002)
        var p = List[Float32]()
        var m = List[Float32]()
        var v = List[Float32]()
        for i in range(N):
            p.append((_u01(UInt64(regime) * 7919 + UInt64(i)) - Float32(0.5)) * pscale)
            m.append(Float32(0.0))
            v.append(Float32(0.0))
        for t in range(1, K + 1):
            var g = List[Float32]()
            for i in range(N):
                g.append(
                    (_u01(UInt64(regime) * 104729 + UInt64(t) * 1299709 + UInt64(i))
                     - Float32(0.5)) * Float32(2.4e-5)
                )
            _adamw_host_list_f32(p, g, m, v, t, LR, Float32(0.9), Float32(0.999),
                                 Float32(1.0e-8), WD)
        # dump as raw f32 bit patterns (hex) for exact comparison
        for i in range(N):
            var pb = p[i]
            var mb = m[i]
            var vb = v[i]
            print("R", regime, i,
                  Int(pb.to_bits[DType.uint32]()),
                  Int(mb.to_bits[DType.uint32]()),
                  Int(vb.to_bits[DType.uint32]()))
    print("MOJO-DONE N=", N, " K=", K)
