# v2v_loss.mojo -- LTX-2 IC-LoRA / V2V target-only masked-MSE loss + the
# ref-sliced block-output cotangent (P5 unit 2). HOST-side, mirroring the
# existing fwd->bwd host loss seam (the levers-loss seam).
#
# torchref contract (ltx2_train_network.py:3452-3459 + ltx2_train.py:178-225):
#   * SLICE the reference off before the loss:  target_pred = pred[:, ref_seq_len:]
#   * target_velocity = patchify(noise - latents)  (target tokens only)
#   * target_loss_mask = ~target_conditioning_mask
#   * reduction = _masked_mse == sum(per_elem*mask)/sum(mask)  (target-token norm)
#
# The block backward takes a cotangent at the COMBINED block output; the ref
# slice IS zeros on the ref rows of that cotangent, and the target rows carry the
# masked-MSE gradient  d = 2*(pred-tgt)*mask / sum(mask_broadcast).  This mirrors
# the oracle's d_block_out = [zeros_ref ; d(masked_loss)/d(target_pred)] exactly.

from std.math import abs as _abs


# loss = sum((pred-tgt)^2 * mask) / (n_true * dim)  == torchref _masked_mse (mse),
# target-token normalized. mask is per TARGET TOKEN, broadcast over dim.
# d_block_out [s_comb*dim] row-major: ref rows (0..s_ref-1) = 0; target row t,
# channel d = 2*(pred-tgt)*mask[t] / (n_true*dim).
def v2v_target_cotangent(
    block_out: List[Float32],          # [s_comb*dim], ref rows first
    target_velocity: List[Float32],    # [s_tgt*dim]
    target_loss_mask: List[Bool],      # [s_tgt]  (True = in loss)
    s_ref: Int, s_tgt: Int, dim: Int,
) raises -> Tuple[Float32, List[Float32]]:
    var s_comb = s_ref + s_tgt
    if len(block_out) != s_comb * dim:
        raise Error(String("v2v_target_cotangent: block_out len ")
            + String(len(block_out)) + " != s_comb*dim " + String(s_comb * dim))
    if len(target_velocity) != s_tgt * dim:
        raise Error(String("v2v_target_cotangent: target_velocity len ")
            + String(len(target_velocity)) + " != s_tgt*dim " + String(s_tgt * dim))
    if len(target_loss_mask) != s_tgt:
        raise Error(String("v2v_target_cotangent: mask len ")
            + String(len(target_loss_mask)) + " != s_tgt " + String(s_tgt))

    var n_true = 0
    for t in range(s_tgt):
        if target_loss_mask[t]:
            n_true += 1

    var d_block = List[Float32]()
    d_block.resize(s_comb * dim, Float32(0.0))     # ref rows stay 0 (the slice)

    # denominator = number of masked ELEMENTS = n_true * dim (sum over broadcast)
    if n_true == 0:
        # torchref fallback: denom==0 -> per_elem.mean(); no target in loss -> the
        # loss is the plain mean and the gradient normalizes by all target elems.
        var msum0 = Float32(0.0)
        var denom0 = Float32(s_tgt * dim)
        for t in range(s_tgt):
            var base_o = (s_ref + t) * dim
            var base_v = t * dim
            for d in range(dim):
                var diff = block_out[base_o + d] - target_velocity[base_v + d]
                msum0 += diff * diff
                d_block[base_o + d] = Float32(2.0) * diff / denom0
        return (msum0 / denom0, d_block^)

    var denom = Float32(n_true * dim)
    var masked_sum = Float32(0.0)
    for t in range(s_tgt):
        var in_loss = target_loss_mask[t]
        var base_o = (s_ref + t) * dim
        var base_v = t * dim
        for d in range(dim):
            var diff = block_out[base_o + d] - target_velocity[base_v + d]
            if in_loss:
                masked_sum += diff * diff
                d_block[base_o + d] = Float32(2.0) * diff / denom
            # masked-out target rows: cotangent stays 0
    return (masked_sum / denom, d_block^)
