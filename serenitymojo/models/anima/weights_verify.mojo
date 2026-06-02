# models/anima/weights_verify.mojo — scaffold smoke: load Anima block-0 real
# weights and assert the 20 tensor shapes against the verified checkpoint header.
# This is the Phase-1 weights-load verification (Tenet 4: measured, not asserted).
#
# Run: cd /home/alex/mojodiffusion && \
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/models/anima/weights_verify.mojo -o /tmp/anima_weights_verify && \
#   /tmp/anima_weights_verify

from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.anima.config import anima
from serenitymojo.models.anima.weights import load_anima_block_weights


def _check(name: String, got: List[Int], e0: Int, e1: Int) raises:
    var ok = len(got) == 2 and got[0] == e0 and got[1] == e1
    if len(got) == 1:
        ok = e1 == -1 and got[0] == e0
    var got_str = String("[")
    for i in range(len(got)):
        if i > 0:
            got_str += String(", ")
        got_str += String(got[i])
    got_str += String("]")
    print("  ", name, "shape", got_str, "expected [", e0, ",", e1, "] ->",
          "OK" if ok else "MISMATCH")
    if not ok:
        raise Error(String("anima block-0 shape mismatch: ") + name)


def main() raises:
    var ctx = DeviceContext()
    print("############################################################")
    print("# Anima scaffold: config + block-0 weights load verification")
    print("############################################################")

    var cfg = anima()
    print("config: d_model", cfg.d_model, "num_single", cfg.num_single,
          "n_heads", cfg.n_heads, "head_dim", cfg.head_dim,
          "mlp_hidden", cfg.mlp_hidden, "in_ch", cfg.in_channels,
          "out_ch", cfg.out_channels, "joint_dim", cfg.joint_attention_dim,
          "lr", cfg.lr, "rank", cfg.lora_rank, "alpha", cfg.lora_alpha,
          "shift", cfg.timestep_shift)

    print("opening checkpoint:", cfg.checkpoint)
    var st = SafeTensors.open(cfg.checkpoint)

    var w = load_anima_block_weights(st, 0, ctx)
    print("loaded block-0 weights. verifying shapes:")
    _check(String("sa_mod1"), w.sa_mod1[].shape(), 256, 2048)
    _check(String("sa_mod2"), w.sa_mod2[].shape(), 6144, 256)
    _check(String("ca_mod1"), w.ca_mod1[].shape(), 256, 2048)
    _check(String("ca_mod2"), w.ca_mod2[].shape(), 6144, 256)
    _check(String("mlp_mod1"), w.mlp_mod1[].shape(), 256, 2048)
    _check(String("mlp_mod2"), w.mlp_mod2[].shape(), 6144, 256)
    _check(String("sa_q"), w.sa_q[].shape(), 2048, 2048)
    _check(String("sa_k"), w.sa_k[].shape(), 2048, 2048)
    _check(String("sa_v"), w.sa_v[].shape(), 2048, 2048)
    _check(String("sa_out"), w.sa_out[].shape(), 2048, 2048)
    _check(String("sa_qn"), w.sa_qn[].shape(), 128, -1)
    _check(String("sa_kn"), w.sa_kn[].shape(), 128, -1)
    _check(String("ca_q"), w.ca_q[].shape(), 2048, 2048)
    _check(String("ca_k"), w.ca_k[].shape(), 2048, 1024)
    _check(String("ca_v"), w.ca_v[].shape(), 2048, 1024)
    _check(String("ca_out"), w.ca_out[].shape(), 2048, 2048)
    _check(String("ca_qn"), w.ca_qn[].shape(), 128, -1)
    _check(String("ca_kn"), w.ca_kn[].shape(), 128, -1)
    _check(String("mlp1"), w.mlp1[].shape(), 8192, 2048)
    _check(String("mlp2"), w.mlp2[].shape(), 2048, 8192)

    print("ALL 20 block-0 shapes verified.")
