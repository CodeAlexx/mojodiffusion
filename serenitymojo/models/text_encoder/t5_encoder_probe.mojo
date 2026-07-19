# t5_encoder_probe.mojo — COMPILE-ONLY probe for T5Encoder.
# Imports + constructs the config and references the encoder type. Does NOT run
# (GPU wedged). Verification = clean compile (EXIT=0).

from serenitymojo.models.text_encoder.t5_encoder import (
    T5Config,
    T5Encoder,
    _t5_relative_position_bucket,
)


def main() raises:
    var cfg = T5Config.t5_xxl()
    # Sanity: head*kv == d_model.
    if cfg.num_heads * cfg.d_kv != cfg.d_model:
        raise Error("T5 config: num_heads * d_kv != d_model")
    if cfg.num_layers != 24:
        raise Error("T5 config: expected 24 layers")
    # Reference the bucket helper (host scalar) and the comptime-param type so
    # both monomorphize at S=512.
    var b = _t5_relative_position_bucket(1, True, 32, 128)
    comptime EncT = T5Encoder[512]
    # Force load/encode bodies to monomorphize (bound static/instance methods).
    comptime LoadFn = T5Encoder[512].load
    comptime EncFn = T5Encoder[512].encode
    print("t5_encoder probe constructed; bucket(1)=", b)
