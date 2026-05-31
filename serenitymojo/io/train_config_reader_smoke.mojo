# train_config_reader_smoke.mojo — read the real OneTrainer Klein9B config and
# print the parsed TrainConfig + OptimExtra. Also asserts the JSON-derived
# recipe scalars match the known values in the file.
#
# Run (after the compile lock frees):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/io/train_config_reader_smoke.mojo
#
# Known values in /home/alex/OneTrainer/configs/klein9b_loss_compare.json
# (verified 2026-05-30):
#   learning_rate 0.0004 | lora_rank 16 | lora_alpha 16 | model_type "FLUX_2"
#   optimizer.eps 1e-08 | optimizer.weight_decay 0.01 | beta1 0.9 | beta2 0.999

from serenitymojo.io.train_config_reader import read_train_config


comptime CONFIG = "/home/alex/OneTrainer/configs/klein9b_loss_compare.json"

# Klein9B model dims (NOT in the OneTrainer JSON — caller-supplied, mirroring the
# per-model config constructor the train_config.mojo header points to). Klein9B:
# inner_dim 4096 = 32 heads * 128 head_dim; 8 double + 24 single = 32 blocks.
comptime D_MODEL = 4096
comptime N_HEADS = 32
comptime HEAD_DIM = 128
comptime MLP_HIDDEN = 16384
comptime N_LAYERS = 32


def _close(name: String, a: Float32, b: Float32) raises:
    var d = a - b
    var ad = d if d >= 0.0 else -d
    if ad > Float32(1e-6):
        raise Error(name + " mismatch: got " + String(a) + " expected " + String(b))


def main() raises:
    print("=== train_config_reader smoke: ", CONFIG, " ===")
    var r = read_train_config(
        String(CONFIG), D_MODEL, N_HEADS, HEAD_DIM, MLP_HIDDEN, N_LAYERS
    )
    var c = r.cfg.copy()

    print("  name           =", c.name)
    print("  lr             =", c.lr)
    print("  lora_rank      =", c.lora_rank)
    print("  lora_alpha     =", c.lora_alpha)
    print("  eps            =", c.eps)
    print("  timestep_shift =", c.timestep_shift, "(caller default 1.8)")
    print("  d_model        =", c.d_model)
    print("  n_heads        =", c.n_heads)
    print("  head_dim       =", c.head_dim)
    print("  mlp_hidden     =", c.mlp_hidden)
    print("  n_layers       =", c.n_layers)
    print("  optim.weight_decay =", r.optim.weight_decay)
    print("  optim.beta1        =", r.optim.beta1)
    print("  optim.beta2        =", r.optim.beta2)

    # Assertions against the known file values.
    _close("lr", c.lr, Float32(0.0004))
    if c.lora_rank != 16:
        raise Error("lora_rank expected 16, got " + String(c.lora_rank))
    _close("lora_alpha", c.lora_alpha, Float32(16.0))
    _close("eps", c.eps, Float32(1e-8))
    _close("weight_decay", r.optim.weight_decay, Float32(0.01))
    _close("beta1", r.optim.beta1, Float32(0.9))
    _close("beta2", r.optim.beta2, Float32(0.999))
    if c.name != String("FLUX_2"):
        raise Error("model_type expected FLUX_2, got " + c.name)

    print("train_config_reader smoke PASS (recipe scalars match the file)")
