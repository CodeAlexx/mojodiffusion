# Flux2 runtime-loader helper import/key contract.
#
# This is a non-GPU compile/run guard for the runtime helper split from
# modelLoader/Flux2ModelLoader.mojo. It does not instantiate DeviceContext or
# load safetensors; the full key/load roundtrip remains
# smoke/klein_lora_key_parity.mojo.

from serenity_trainer.modelLoader.Flux2RuntimeLoader import (
    flux2_double_prefix,
    flux2_double_weight_key,
    flux2_single_prefix,
    flux2_single_weight_key,
)


def _expect_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(name + String(": got ") + got + String(", expected ") + expected)


def main() raises:
    _expect_string("double prefix", flux2_double_prefix(3), String("transformer_blocks.3"))
    _expect_string("single prefix", flux2_single_prefix(4), String("single_transformer_blocks.4"))
    _expect_string(
        "double weight key",
        flux2_double_weight_key(2, String("attn.to_q")),
        String("transformer_blocks.2.attn.to_q.weight"),
    )
    _expect_string(
        "single weight key",
        flux2_single_weight_key(5, String("attn.to_qkv_mlp_proj")),
        String("single_transformer_blocks.5.attn.to_qkv_mlp_proj.weight"),
    )

    print("FLUX2 RUNTIME LOADER KEY CONTRACT OK")
