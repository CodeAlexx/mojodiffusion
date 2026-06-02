# training/grad_accum.mojo — micro-batch gradient accumulation wiring (item 2h).
#
# schedule.mojo:456 `grad_accumulate` (device: acc += new_grad) is the per-tensor
# SUM primitive. This module wires it for the Klein LoRA grad set (host
# List[List[Float32]], the four AdamW-fed grad groups) and supplies the MEAN
# rescale that the accumulation policy requires before the optimizer step.
#
# ── Convention (AGENT-DEFAULT for review) ─────────────────────────────────────
# SUM across N micro-steps, then divide by N (MEAN) before clip+AdamW. The
# schedule.mojo `grad_accumulate` header explicitly states "divide by the
# accumulation count ... before the optimizer step", so MEAN is the documented
# policy. MEAN (not raw SUM) makes the AdamW step invariant to grad_accum_steps
# for identical micro-samples, which is the behavior the gate checks:
#   accum_steps=2 on two identical micro-grads g -> mean = (g+g)/2 = g
#   -> one AdamW step on g, identical to accum_steps=1 on g.
# accum_steps=1 is byte-identical to the current per-step path (sum of one, /1).
#
# Pure host F32 (the Klein LoRA grads already live as host lists). The device
# `grad_accumulate` is the same `acc += g` math for any trainer whose grads are
# device tensors; this host version keeps the Klein path allocation-light.
#
# Mojo 1.0.0b1.

from std.collections import List


# acc[i][j] += add[i][j]  (host SUM accumulation, mirrors schedule.mojo grad_accumulate).
def accumulate_grad_group(mut acc: List[List[Float32]], add: List[List[Float32]]) raises:
    if len(acc) != len(add):
        raise Error("accumulate_grad_group: group count mismatch")
    for i in range(len(acc)):
        if len(acc[i]) != len(add[i]):
            raise Error("accumulate_grad_group: tensor numel mismatch")
        for j in range(len(acc[i])):
            acc[i][j] = acc[i][j] + add[i][j]


# acc[i][j] *= s  (MEAN rescale: pass s = 1/N before the optimizer step).
def scale_grad_group(mut acc: List[List[Float32]], s: Float32):
    for i in range(len(acc)):
        for j in range(len(acc[i])):
            acc[i][j] = acc[i][j] * s


# Zero-cloned accumulator matching the shape of a grad group (one List per
# adapter, sized to that adapter's grad numel).
def zeros_like_group(src: List[List[Float32]]) -> List[List[Float32]]:
    var out = List[List[Float32]]()
    for i in range(len(src)):
        var inner = List[Float32]()
        for _ in range(len(src[i])):
            inner.append(Float32(0.0))
        out.append(inner^)
    return out^
