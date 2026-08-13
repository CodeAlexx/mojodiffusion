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

Format `convrot_w8_v1`, per quantized linear `W [out, in]` (no low-rank branch):
    W = Rrot @ H_bd
        qweight   i8   [out, in]      round(Rrot / wscale), clamped [-127, 127]
        wscale    f32  [out]          per-output-channel absmax(Rrot) / 127
    with Rrot = W @ H_bd and the SAME H_bd, so W_hat = dequant8 @ H_bd.
"""

from .core import (  # noqa: F401
    FORMAT_TAG,
    FORMAT_TAG_CONVROT,
    GROUP,
    HBLOCK,
    dequant_int4_g64,
    dequant_int8_perchannel,
    hadamard_matrix,
    nvfp4_decode,
    nvfp4_encode,
    pack_int4,
    quant_int4_g64,
    quant_int8_perchannel,
    quantize_layer,
    quantize_layer_convrot,
    reconstruct_weight,
    reconstruct_weight_convrot,
    rht_grouped,
    svd_lowrank_init,
    unpack_int4,
)
