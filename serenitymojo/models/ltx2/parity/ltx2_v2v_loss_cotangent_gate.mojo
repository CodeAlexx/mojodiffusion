# serenitymojo/models/ltx2/parity/ltx2_v2v_loss_cotangent_gate.mojo
#
# LTX-2 IC-LoRA / V2V loss+cotangent self-check gate (P5 unit 2). HOST-ONLY --
# no DeviceContext, no GPU. Validates v2v_target_cotangent:
#   * loss == sum((pred-tgt)^2 * mask)/(n_true*dim)  (torchref _masked_mse, mse),
#   * ref rows of d_block_out are ZERO (the ref slice),
#   * masked-out target rows are ZERO,
#   * target-in-loss rows: d_block == d(loss)/d(block_out) by FINITE DIFFERENCE.
#
# Run (no GPU):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . \
#       serenitymojo/models/ltx2/parity/ltx2_v2v_loss_cotangent_gate.mojo \
#       -o /tmp/ltx2_v2v_loss_cotangent_gate && /tmp/ltx2_v2v_loss_cotangent_gate

from std.math import abs as _abs
from serenitymojo.training.ltx2.v2v_loss import v2v_target_cotangent

comptime S_REF = 2
comptime S_TGT = 3
comptime DIM = 2
comptime S_COMB = S_REF + S_TGT


def _mk_block() -> List[Float32]:
    var b = List[Float32]()
    for i in range(S_COMB * DIM):
        b.append(Float32(0.1) * Float32(i + 1) - Float32(0.3))   # spread of signs
    return b^


def _mk_tv() -> List[Float32]:
    var v = List[Float32]()
    for i in range(S_TGT * DIM):
        v.append(Float32(0.05) * Float32(i) - Float32(0.2))
    return v^


def _mk_mask() -> List[Bool]:
    # token 1 masked-out (e.g. first-frame conditioning); 0 and 2 in loss
    var m = List[Bool]()
    m.append(True)
    m.append(False)
    m.append(True)
    return m^


def _loss_only(block: List[Float32]) raises -> Float32:
    var r = v2v_target_cotangent(block, _mk_tv(), _mk_mask(), S_REF, S_TGT, DIM)
    return r[0]


def main() raises:
    print("=== LTX-2 v2v loss+cotangent self-check (host-only) ===")
    var block = _mk_block()
    var tv = _mk_tv()
    var mask = _mk_mask()
    var res = v2v_target_cotangent(block, tv, mask, S_REF, S_TGT, DIM)
    var loss = res[0]
    var dblk = res[1].copy()

    var fails = 0

    # hand loss: sum over in-loss target rows of (pred-tgt)^2 / (n_true*dim)
    var n_true = 2
    var hand = Float32(0.0)
    for t in range(S_TGT):
        if mask[t]:
            for d in range(DIM):
                var diff = block[(S_REF + t) * DIM + d] - tv[t * DIM + d]
                hand += diff * diff
    hand = hand / Float32(n_true * DIM)
    if _abs(loss - hand) < Float32(1e-6):
        print("  [PASS] loss == masked_sum/(n_true*dim):", loss)
    else:
        fails += 1
        print("  [FAIL] loss", loss, "!= hand", hand)

    # ref rows zero
    var ref_nonzero = 0
    for i in range(S_REF * DIM):
        if dblk[i] != Float32(0.0):
            ref_nonzero += 1
    if ref_nonzero == 0:
        print("  [PASS] ref rows of cotangent are zero (the ref slice)")
    else:
        fails += 1
        print("  [FAIL] ref rows nonzero:", ref_nonzero)

    # masked-out target row (token 1) zero
    var masked_row_nonzero = 0
    for d in range(DIM):
        if dblk[(S_REF + 1) * DIM + d] != Float32(0.0):
            masked_row_nonzero += 1
    if masked_row_nonzero == 0:
        print("  [PASS] masked-out target row cotangent is zero")
    else:
        fails += 1
        print("  [FAIL] masked-out target row nonzero:", masked_row_nonzero)

    # finite-difference the in-loss target elements
    var eps = Float32(1e-3)
    var max_fd_err = Float32(0.0)
    for t in range(S_TGT):
        if mask[t]:
            for d in range(DIM):
                var idx = (S_REF + t) * DIM + d
                var bp = block.copy()
                bp[idx] = bp[idx] + eps
                var bm = block.copy()
                bm[idx] = bm[idx] - eps
                var fd = (_loss_only(bp) - _loss_only(bm)) / (Float32(2.0) * eps)
                var err = _abs(fd - dblk[idx])
                if err > max_fd_err:
                    max_fd_err = err
    if max_fd_err < Float32(1e-4):
        print("  [PASS] cotangent == d(loss)/d(block_out) by FD, max err", max_fd_err)
    else:
        fails += 1
        print("  [FAIL] FD gradient mismatch, max err", max_fd_err)

    if fails > 0:
        raise Error(String("LTX-2 V2V LOSS/COTANGENT GATE FAIL: ") + String(fails) + " check(s)")
    print("LTX-2 V2V LOSS/COTANGENT SELF-CHECK PASS (loss + ref-slice + mask + FD grad)")
