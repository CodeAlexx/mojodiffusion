# klein_single_block_smoke.mojo — compile + run smoke for ONE Klein (FLUX.2)
# single-stream DiT block: forward (saves activations) + hand-chained backward
# returning the per-adapter LoRA d_A/d_B. This is the unit the 24-deep Klein
# single stack (model/KleinModel.mojo / model/klein/klein_stack_lora.mojo) chains.
#
# Tiny comptime dims (NOT the 9B dims) so the smoke is fast: H=2, Dh=128 (Klein
# head dim is fixed 128 for the rope axis math), D=H*Dh=256, F=512, S=8.
# Builds deterministic weights + ONE qkv LoRA adapter (slot0) + ONE out adapter
# (slot1), runs single_block_lora_forward then single_block_lora_backward, and
# checks the returned d_A/d_B are the right lengths and finite.
#
# BORROWED block math: serenity_trainer/model/klein/single_block.mojo (copied from
# serenitymojo/models/klein/single_block.mojo). Foundation ops imported unchanged.

from std.collections import List, Optional
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

from serenity_trainer.model.klein.single_block import (
    SingleBlockWeights, SingleModVecs, SingleBlockLora,
    single_block_lora_forward, single_block_lora_backward,
)
from serenity_trainer.model.klein.lora_adapter import LoraAdapter
from serenity_trainer.model.klein.klein_stack_lora import make_lora_adapter
from serenity_trainer.model.KleinModel import build_klein_rope_tables_port


comptime SH = 2          # heads (smoke)
comptime SDh = 128       # head dim (Klein fixed)
comptime SD = SH * SDh   # model dim = 256
comptime SF = 512        # mlp hidden (smoke)
comptime SS = 8          # seq len (N_TXT=4 + N_IMG=4, N_IMG square)
comptime S_NTXT = 4
comptime S_NIMG = 4
comptime SEPS = Float32(1e-6)


def _randn(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _ones(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(1.0))
    return out^


def _all_finite(v: List[Float32]) -> Bool:
    for i in range(len(v)):
        var x = v[i]
        if x != x:
            return False
        if x > Float32(1e30) or x < Float32(-1e30):
            return False
    return True


def main() raises:
    var ctx = DeviceContext()

    # rope tables for the joint sequence: [SS*SH, SDh/2]
    var ct: Tensor
    var st: Tensor
    (ct, st) = build_klein_rope_tables_port[S_NIMG, S_NTXT, SH, SDh](ctx, STDtype.F32)

    # weights: w1 [3D+2F, D], w2 [D, D+F], q_norm/k_norm [Dh]=ones.
    var w = SingleBlockWeights(
        _randn((3 * SD + 2 * SF) * SD, 11, 0.02),   # w1
        _randn(SD * (SD + SF), 22, 0.02),           # w2
        _ones(SDh),                                  # q_norm
        _ones(SDh),                                  # k_norm
        SD, SF, SDh, ctx, True,
    )

    # single modulation vectors (shift/scale/gate), each [D].
    var mv = SingleModVecs(
        _randn(SD, 31, 0.05), _randn(SD, 32, 0.05), _randn(SD, 33, 0.05),
    )

    # ONE qkv adapter over the FULL to_qkv_mlp_proj Linear (slot0, in=D,
    # out=3D+2F) + ONE out adapter over the FULL to_out Linear (slot1, in=D+F,
    # out=D). This matches Serenity's per-Linear LoRAModuleWrapper
    # (Flux2LoRASetup.py:57) wrapping the fused parallel-block projections
    # (transformer_flux2.py:753-755 to_qkv_mlp_proj, :763 to_out).
    var rank = 4
    var alpha = Float32(4.0)
    var lora = SingleBlockLora(
        Optional[LoraAdapter](make_lora_adapter(rank, alpha, SD, 3 * SD + 2 * SF, 100)),
        Optional[LoraAdapter](make_lora_adapter(rank, alpha, SD + SF, SD, 101)),
    )

    var x = _randn(SS * SD, 7, 0.1)

    # FORWARD (saves activations for backward).
    var fwd = single_block_lora_forward[SH, SDh, SS](
        x.copy(), w, mv, lora, ct, st, SD, SF, SEPS, ctx,
    )
    if len(fwd.out) != SS * SD:
        raise Error("klein single smoke: forward out wrong length")
    if not _all_finite(fwd.out):
        raise Error("klein single smoke: forward out not finite")

    # BACKWARD (hand-chained) -> per-adapter d_A/d_B.
    var d_out = _randn(SS * SD, 9, 0.1)
    var grads = single_block_lora_backward[SH, SDh, SS](
        d_out^, w, mv, lora, fwd.saved.copy(), ct, st, SD, SF, SEPS, ctx,
    )

    # qkv adapter: A [rank, D], B [3D+2F, rank]; out adapter: A [rank, D+F],
    # B [D, rank] (FULL fused-projection widths, incl. mlp columns).
    if len(grads.qkv_d_a) != rank * SD:
        raise Error("klein single smoke: qkv d_A wrong length")
    if len(grads.qkv_d_b) != (3 * SD + 2 * SF) * rank:
        raise Error("klein single smoke: qkv d_B wrong length")
    if len(grads.out_d_a) != rank * (SD + SF):
        raise Error("klein single smoke: out d_A wrong length")
    if len(grads.out_d_b) != SD * rank:
        raise Error("klein single smoke: out d_B wrong length")
    if not _all_finite(grads.qkv_d_a) or not _all_finite(grads.qkv_d_b):
        raise Error("klein single smoke: qkv grads not finite")
    if not _all_finite(grads.out_d_a) or not _all_finite(grads.out_d_b):
        raise Error("klein single smoke: out grads not finite")

    print("klein_single_block_smoke OK: fwd+bwd, d_A/d_B shapes verified")
