# training/oft_stack.mojo — OneTrainer-OFT via the shared (a,b) carrier (FULL
# materialized delta), reusing the proven stack path (no klein-stack surgery).
# OFT W_eff[o,base+k] = Σ_c R_g[k,c]·W[o,base+c] (input-block rotation folded into
# the weight). The carrier is the full delta:
#       a = I_in [in,in]   b = ΔW_oft = W_eff - W [out,in]   r_eff = in
# y = x@Wᵀ + x@bᵀ = x@W_effᵀ (== oft_ot_forward). VRAM-bound (r_eff=in) → preflight
# fails loud at large in. Re-materialized each step.
#
# Backward: the stack's d_b == ∂L/∂ΔW_oft == d_W_eff. Chain per block:
#   d_R_g[k,c] = Σ_o d_W_eff[o,base+k]·W[o,base+c]  →  d_Q_g (Neumann bwd)  →  d_vec_g.
# Gated vs oft_ot_backward in tests/oft_carrier_parity.mojo.

from std.collections import List
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.training.oft_onetrainer import (
    oft_ot_skew, oft_ot_neumann_r, _neumann_backward,
)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def oft_ot_carrier_r_eff(IN: Int) -> Int:
    return IN


# Materialize the FULL-delta carrier (a=I_in, b=W_eff-W). Carrier scale == 1.
def oft_ot_carrier_adapter(
    vec: List[Float32], w: List[Float32], IN: Int, OUT: Int, b: Int, r: Int
) raises -> LoraAdapter:
    var ne = b * (b - 1) // 2
    var a = _zeros(IN * IN)                              # I_in
    for i in range(IN):
        a[i * IN + i] = Float32(1.0)
    var bb = _zeros(OUT * IN)                            # ΔW_oft = W_eff - W
    for g in range(r):
        var vblk = List[Float32]()
        for t in range(ne):
            vblk.append(vec[g * ne + t])
        var q = oft_ot_skew(vblk, b)
        var rg = oft_ot_neumann_r(q, b)                 # [b,b]
        var base = g * b
        for o in range(OUT):
            for k in range(b):
                var weff = Float32(0.0)
                for c in range(b):
                    weff += rg[k * b + c] * w[o * IN + base + c]
                bb[o * IN + base + k] = weff - w[o * IN + base + k]
    return LoraAdapter(
        a^, bb^, IN, IN, OUT, Float32(1.0),
        List[Float32](), List[Float32](), List[Float32](), List[Float32](),
    )


# Chain carrier grads → d_vec. d_b_carrier == d_W_eff; d_a_carrier discarded.
def oft_ot_chain_carrier_grads(
    vec: List[Float32], w: List[Float32],
    d_a_carrier: List[Float32], d_b_carrier: List[Float32],
    IN: Int, OUT: Int, b: Int, r: Int,
) raises -> List[Float32]:
    var ne = b * (b - 1) // 2
    if len(d_b_carrier) != OUT * IN:
        raise Error("oft_ot_chain_carrier_grads: d_b numel mismatch")
    var d_vec = _zeros(r * ne)
    for g in range(r):
        var base = g * b
        # d_R_g[k,c] = Σ_o d_W_eff[o,base+k]·W[o,base+c]
        var d_rg = _zeros(b * b)
        for k in range(b):
            for c in range(b):
                var acc = Float32(0.0)
                for o in range(OUT):
                    acc += d_b_carrier[o * IN + base + k] * w[o * IN + base + c]
                d_rg[k * b + c] = acc
        var vblk = List[Float32]()
        for t in range(ne):
            vblk.append(vec[g * ne + t])
        var q = oft_ot_skew(vblk, b)
        var dq = _neumann_backward(q, d_rg, b)
        var t = 0
        for i in range(b):
            for j in range(i + 1, b):
                d_vec[g * ne + t] = dq[i * b + j] - dq[j * b + i]
                t += 1
    return d_vec^
