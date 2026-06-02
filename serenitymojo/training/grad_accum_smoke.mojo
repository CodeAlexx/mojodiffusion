# training/grad_accum_smoke.mojo — gate for gradient accumulation (item 2h).
#
# The trainer wires accumulation as: SUM the four LoRA grad groups across N
# micro-steps, then MEAN (÷N) before clip+AdamW. The decisive property the gate
# checks (which makes the optimizer step identical for identical micro-samples):
#
#   accum_steps=2 on TWO IDENTICAL micro-grads g:
#       acc = 0 ; acc += g ; acc += g ; acc *= 1/2  ==  g   (bit-exact F32)
#   => one AdamW step on the meaned grad == one AdamW step on a single g.
#
#   accum_steps=1:  acc = 0 ; acc += g ; acc *= 1/1  ==  g   (bit-exact, the
#   current per-step path is byte-unchanged).
#
# Asserts:
#   (1) N=2 mean over identical grads == g (bit-exact, F32 equality).
#   (2) N=1 == g (bit-exact).
#   (3) SUM correctness: acc += g1 ; acc += g2 == g1+g2 elementwise (1e-6).
#   (4) BITROT-FAIL DEMO: WRONG scale (÷1 instead of ÷2 for N=2) must differ.
#
# Exits NONZERO (raise) on any mismatch.
#
# Build/run (JIT):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/training/grad_accum_smoke.mojo

from serenitymojo.training.grad_accum import (
    accumulate_grad_group, scale_grad_group, zeros_like_group,
)


def _group(seed: Int, n_adapters: Int, numel: Int) -> List[List[Float32]]:
    var out = List[List[Float32]]()
    for i in range(n_adapters):
        var inner = List[Float32]()
        for j in range(numel):
            inner.append(Float32((i * 13 + j * 7 + seed) % 11) * Float32(0.137) - Float32(0.5))
        out.append(inner^)
    return out^


def _bit_equal(a: List[List[Float32]], b: List[List[Float32]]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]):
            return False
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]:
                return False
    return True


def main() raises:
    var ok = True
    var n_adapters = 6
    var numel = 20

    var g = _group(1, n_adapters, numel)

    # ── (1) N=2 mean over identical grads == g (bit-exact) ────────────────────
    var acc2 = zeros_like_group(g)
    accumulate_grad_group(acc2, g)
    accumulate_grad_group(acc2, g)        # second identical micro-grad
    scale_grad_group(acc2, Float32(1.0) / Float32(2.0))
    if _bit_equal(acc2, g):
        print("PASS N=2 mean over identical micro-grads == g (bit-exact)")
    else:
        print("FAIL N=2 mean != g"); ok = False

    # ── (2) N=1 == g (bit-exact: current per-step path unchanged) ─────────────
    var acc1 = zeros_like_group(g)
    accumulate_grad_group(acc1, g)
    scale_grad_group(acc1, Float32(1.0) / Float32(1.0))
    if _bit_equal(acc1, g):
        print("PASS N=1 == g (default-off byte-unchanged)")
    else:
        print("FAIL N=1 != g"); ok = False

    # ── (3) SUM correctness over two DIFFERENT grads ──────────────────────────
    var g1 = _group(2, n_adapters, numel)
    var g2 = _group(5, n_adapters, numel)
    var accs = zeros_like_group(g1)
    accumulate_grad_group(accs, g1)
    accumulate_grad_group(accs, g2)
    var maxerr = Float32(0.0)
    for i in range(n_adapters):
        for j in range(numel):
            var e = accs[i][j] - (g1[i][j] + g2[i][j])
            if e < Float32(0.0):
                e = -e
            if e > maxerr:
                maxerr = e
    print("sum max |acc - (g1+g2)| =", maxerr)
    if maxerr > Float32(1.0e-6):
        print("FAIL sum accumulation incorrect"); ok = False
    else:
        print("PASS acc += g1; acc += g2 == g1+g2 to 1e-6")

    # ── (4) BITROT-FAIL DEMO: wrong scale (÷1 for N=2) must differ from g ──────
    var accw = zeros_like_group(g)
    accumulate_grad_group(accw, g)
    accumulate_grad_group(accw, g)
    scale_grad_group(accw, Float32(1.0))   # WRONG: should be ÷2
    if _bit_equal(accw, g):
        print("FAIL bitrot demo: wrong scale still matched g"); ok = False
    else:
        print("PASS bitrot demo: wrong scale (÷1 for N=2) differs from g (gate is sensitive)")

    if not ok:
        raise Error("grad_accum_smoke FAILED")
    print("grad_accum_smoke gate PASS")
