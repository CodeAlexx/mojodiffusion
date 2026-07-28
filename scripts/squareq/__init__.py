"""SquareQ — grouped-Hadamard + low-rank-corrected 4-bit quantization.

Public Python reference for the serenitymojo `squareq_w4` format: the offline
builder, pack/unpack codecs, and a PyTorch reference runtime that consumes the
exact slab the Mojo trainer/inference stack consumes. Third parties can verify
every SquareQ result with this package alone (no Mojo required).

Format v1 (`squareq_w4_v1`), per quantized linear `W [out, in]`:
    W = lora_up @ lora_down^T + Rrot @ H_bd
        qweight   u8   [out, in/2]   int4 (twos-complement, low nibble = even k)
                                     of Rrot = (W - lora_up@lora_down^T) @ H_bd
        wscales   bf16 [in/64, out]  group-64-along-in x per-out scale of Rrot
        lora_down bf16 [in, r]
        lora_up   bf16 [out, r]
    H_bd = block-diagonal normalized Hadamard-256 (symmetric, self-inverse),
    so reconstruction is W_hat = dequant4(qweight, wscales) @ H_bd
                                 + lora_up @ lora_down^T.
Conventions are pinned to serenitymojo/ops/svdquant.mojo (SIGN_MODE=twos,
NIBBLE_ORDER=lo_even, Class-A wscales layout).
"""

from .core import (  # noqa: F401
    FORMAT_TAG,
    GROUP,
    HBLOCK,
    dequant_int4_g64,
    hadamard_matrix,
    nvfp4_decode,
    nvfp4_encode,
    pack_int4,
    quant_int4_g64,
    quantize_layer,
    reconstruct_weight,
    rht_grouped,
    svd_lowrank_init,
    unpack_int4,
)
