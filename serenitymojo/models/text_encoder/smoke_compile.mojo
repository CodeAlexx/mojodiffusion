# smoke_compile.mojo — minimal compile/typecheck driver for qwen3_encoder.
# Does NOT touch the GPU heavy path beyond a config print; the real parity run
# lives in parity/qwen3_parity.mojo.

from serenitymojo.models.text_encoder.qwen3_encoder import Qwen3Config


def main() raises:
    var cfg = Qwen3Config.zimage()
    print("Qwen3Config.zimage:")
    print("  hidden_size =", cfg.hidden_size)
    print("  num_layers  =", cfg.num_layers)
    print("  num_heads   =", cfg.num_heads)
    print("  num_kv_heads=", cfg.num_kv_heads)
    print("  head_dim    =", cfg.head_dim)
    print("  rope_theta  =", cfg.rope_theta)
    print("compile OK")
