# train_config_reader_smoke.mojo — read serenitymojo/configs/klein4b.json and
# assert the parsed TrainConfig matches the checkpoint-verified values.
#
# Run (after the compile lock frees):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/io/train_config_reader_smoke.mojo
#
# Values verified against flux-2-klein-base-4b.safetensors header (2026-05-31).

from serenitymojo.io.train_config_reader import read_model_config


comptime CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/klein4b.json"


def _close(name: String, a: Float32, b: Float32) raises:
    var d = a - b
    var ad = d if d >= 0.0 else -d
    if ad > Float32(1e-6):
        raise Error(name + " mismatch: got " + String(a) + " expected " + String(b))


def _eq(name: String, a: Int, b: Int) raises:
    if a != b:
        raise Error(name + " expected " + String(b) + ", got " + String(a))


def main() raises:
    print("=== train_config_reader smoke: ", CONFIG, " ===")
    var c = read_model_config(String(CONFIG))

    print("  name          =", c.name)
    print("  checkpoint     =", c.checkpoint)
    print("  vae            =", c.vae)
    print("  d_model        =", c.d_model)
    print("  in_channels    =", c.in_channels)
    print("  joint_attn_dim =", c.joint_attention_dim)
    print("  out_channels   =", c.out_channels)
    print("  num_double     =", c.num_double)
    print("  num_single     =", c.num_single)
    print("  n_heads        =", c.n_heads)
    print("  head_dim       =", c.head_dim)
    print("  mlp_hidden     =", c.mlp_hidden)
    print("  timestep_dim   =", c.timestep_dim)
    print("  rope_theta     =", c.rope_theta)
    print("  lr             =", c.lr)
    print("  lora_rank      =", c.lora_rank)
    print("  lora_alpha     =", c.lora_alpha)
    print("  timestep_shift =", c.timestep_shift)
    print("  eps            =", c.eps)
    print("  weight_decay   =", c.weight_decay)
    print("  beta1/beta2    =", c.beta1, c.beta2)
    print("  max_grad_norm  =", c.max_grad_norm)
    print("  max_steps      =", c.max_steps)
    print("  save/sample    =", c.save_every, c.sample_every)
    print("  n_layers()     =", c.n_layers())

    # Arch — verified from the 4B checkpoint header.
    _eq("d_model", c.d_model, 3072)
    _eq("in_channels", c.in_channels, 128)
    _eq("joint_attention_dim", c.joint_attention_dim, 7680)
    _eq("out_channels", c.out_channels, 128)
    _eq("num_double", c.num_double, 5)
    _eq("num_single", c.num_single, 20)
    _eq("n_heads", c.n_heads, 24)
    _eq("head_dim", c.head_dim, 128)
    _eq("mlp_hidden", c.mlp_hidden, 9216)
    _eq("timestep_dim", c.timestep_dim, 256)
    _eq("n_layers", c.n_layers(), 25)
    # Consistency: heads * head_dim == inner_dim.
    _eq("n_heads*head_dim==d_model", c.n_heads * c.head_dim, c.d_model)

    # Recipe.
    _close("lr", c.lr, Float32(1e-4))
    _eq("lora_rank", c.lora_rank, 16)
    _close("lora_alpha", c.lora_alpha, Float32(16.0))
    _close("timestep_shift", c.timestep_shift, Float32(1.8))
    _close("eps", c.eps, Float32(1e-6))
    _close("weight_decay", c.weight_decay, Float32(0.0))
    _close("beta1", c.beta1, Float32(0.9))
    _close("beta2", c.beta2, Float32(0.999))
    if c.name != String("klein"):
        raise Error("model_type expected klein, got " + c.name)

    print("train_config_reader smoke PASS (arch + recipe match the file)")
