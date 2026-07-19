# sensenova_u1_probe.mojo — COMPILE-ONLY probe for the SenseNova-U1 DiT.
# Imports + names the public types and the config so the compiler type-checks
# the whole module. Does NOT run (GPU wedged). Build:
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/models/dit/sensenova_u1_probe.mojo -o /tmp/snprobe

from serenitymojo.models.dit.sensenova_u1 import (
    SenseNovaU1, SenseNovaU1Config, KvCache,
)


def main() raises:
    var cfg = SenseNovaU1Config.sensenova_u1_8b()
    # Touch the config fields so they are type-checked.
    var n = cfg.num_layers + cfg.hidden_size + cfg.num_heads
    n += cfg.rope_dim_t() + cfg.rope_dim_h() + cfg.rope_dim_w()
    print("sensenova_u1 config layers/hidden/heads:", cfg.num_layers, cfg.hidden_size, cfg.num_heads)
    print("rope dims t/h/w:", cfg.rope_dim_t(), cfg.rope_dim_h(), cfg.rope_dim_w())
    print("fm_head_out_dim:", cfg.fm_head_out_dim, " merge:", cfg.merge_size)
    print("noise scale sum probe:", n)
    # Name the comptime-parameterized model type (no construction — no GPU).
    comptime ModelTy = SenseNovaU1[1024, 256]
    print("SenseNovaU1[L=1024,T=256] type named ok")
