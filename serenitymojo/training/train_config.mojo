# pipeline/train_config.mojo — the ONE training config descriptor.
#
# Unifies the former per-file KleinTrainConfig / ZImageTrainConfig (which were
# identical modulo Z-Image's n_layers field). Carries the RUNTIME recipe scalars
# (lr, shift, rank, alpha, eps) + the model dims. Per-model constructors live in
# serenitymojo/models/<model>/config.mojo and return this type.
#
# NOTE (Mojo constraint): the attention SHAPE (B,S,H,Dh) is a COMPTIME param of
# the train step, NOT carried here — only the recipe + nominal dims are runtime.
# See RECOMMENDED_TRAINER_STRUCTURE.md "forced trade-off".

from std.collections import List


struct TrainConfig(Copyable, Movable):
    var name: String
    var d_model: Int         # inner_dim (== n_heads * head_dim)
    var n_heads: Int
    var head_dim: Int
    var mlp_hidden: Int
    var n_layers: Int        # total blocks (Klein: 8 double + 24 single = 32; Z-Image: 30)
    var lr: Float32          # AdamW lr (OneTrainer config)
    var timestep_shift: Float32
    var lora_rank: Int
    var lora_alpha: Float32
    var eps: Float32

    def __init__(
        out self, var name: String, d_model: Int, n_heads: Int, head_dim: Int,
        mlp_hidden: Int, n_layers: Int, lr: Float32, timestep_shift: Float32,
        lora_rank: Int, lora_alpha: Float32, eps: Float32,
    ):
        self.name = name^
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.mlp_hidden = mlp_hidden
        self.n_layers = n_layers
        self.lr = lr
        self.timestep_shift = timestep_shift
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.eps = eps
