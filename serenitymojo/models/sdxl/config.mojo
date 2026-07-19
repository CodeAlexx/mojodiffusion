# models/sdxl/config.mojo — SDXL UNet config (conv-UNet, NOT a DiT).
#
# Binding user rule (config-driven dims): arch + recipe come from a JSON config
# file (serenitymojo/configs/sdxl.json), NOT hardcoded into model files. This
# mirrors models/klein/config.mojo's thin-reader discipline.
#
# SDXL differs from Klein/Z-Image: it is a convolutional UNet, so its config is
# NOT the DiT-shaped TrainConfig (inner_dim/joint_attention_dim/num_double/rope).
# It carries conv-UNet arch (model_channels, channel_mult, num_res_blocks,
# transformer_depths, adm_in_channels) verified line-by-line against the Rust
# oracle inference-flame/src/models/sdxl_unet.rs (SDXLConfig::default, lines
# 185-202) and the skeptic-verified Mojo forward (models/dit/sdxl_unet.mojo).
#
# The SCALAR arch fields (channels, head_dim, num_groups, eps) + recipe
# (lr/rank/alpha/shift/paths) are READ from the JSON via the shared cursor/parse
# machinery in io.train_config_reader. The structural depth LISTS
# (channel_mult, transformer_depth_*) are SDXL topology invariants — they define
# the model identity, not a tunable, and are documented + cross-checked against
# the oracle's build_block_descriptors.

from serenitymojo.io.train_config_reader import _read_file_bytes, _read_scalar
from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value


comptime SDXL_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/sdxl.json"

# GroupNorm eps split — verified vs Rust oracle (sdxl_unet.rs):
#   ResBlock GroupNorm:           eps = 1e-5  (lines 597, 615, 987)
#   SpatialTransformer GroupNorm: eps = 1e-6  (line 775)
comptime GN_EPS_RES: Float32 = 1e-5
comptime GN_EPS_ST: Float32 = 1e-6


struct SDXLConfig(Copyable, Movable):
    """SDXL UNet architecture + recipe. Scalars read from JSON; the structural
    depth topology (channel_mult / transformer_depth_*) is SDXL-invariant and
    exposed via accessor methods documented against the Rust oracle."""

    # ── paths ────────────────────────────────────────────────────────────────
    var checkpoint: String
    var vae: String

    # ── conv-UNet arch (scalars from JSON) ─────────────────────────────────────
    var in_channels: Int        # latent channels (4)
    var out_channels: Int       # eps-prediction channels (4)
    var model_channels: Int     # base width (320)
    var num_res_blocks: Int     # res blocks per level (2)
    var context_dim: Int        # cross-attn text dim (2048 = CLIP-L 768 + CLIP-G 1280)
    var head_dim: Int           # attention head dim (64)
    var adm_in_channels: Int    # pooled+time_ids vector cond dim (2816)
    var num_groups: Int         # GroupNorm groups (32)
    var time_embed_dim: Int     # time/label embedding dim (1280)
    var transformer_depth_middle: Int  # middle block ST depth (10)

    # ── recipe (from JSON) ─────────────────────────────────────────────────────
    var lr: Float32
    var lora_rank: Int
    var lora_alpha: Float32
    var timestep_shift: Float32
    var max_grad_norm: Float32
    var max_steps: Int
    var save_every: Int
    var sample_every: Int

    # ── diffusion schedule (from JSON) ─────────────────────────────────────────
    var beta_start: Float32
    var beta_end: Float32
    var num_train_timesteps: Int

    def __init__(out self):
        # defaults mirror the Rust SDXLConfig::default; overwritten by JSON parse.
        self.checkpoint = String("")
        self.vae = String("")
        self.in_channels = 4
        self.out_channels = 4
        self.model_channels = 320
        self.num_res_blocks = 2
        self.context_dim = 2048
        self.head_dim = 64
        self.adm_in_channels = 2816
        self.num_groups = 32
        self.time_embed_dim = 1280
        self.transformer_depth_middle = 10
        self.lr = 1e-4
        self.lora_rank = 16
        self.lora_alpha = 16.0
        self.timestep_shift = 1.0
        self.max_grad_norm = 1.0
        self.max_steps = 3000
        self.save_every = 500
        self.sample_every = 250
        self.beta_start = 0.00085
        self.beta_end = 0.012
        self.num_train_timesteps = 1000

    # ── structural topology (SDXL invariants — verified vs the Rust oracle) ────
    # channel_mult = (1, 2, 4)  →  per-level out channels = model_channels * mult
    def channel_mult(self) -> List[Int]:
        var m = List[Int]()
        m.append(1); m.append(2); m.append(4)
        return m^

    # transformer_depth_input = [0, 0, 2, 2, 10, 10]  (one per input res-block)
    def transformer_depth_input(self) -> List[Int]:
        var d = List[Int]()
        d.append(0); d.append(0); d.append(2); d.append(2); d.append(10); d.append(10)
        return d^

    # transformer_depth_output = [10, 10, 10, 2, 2, 2, 0, 0, 0] (one per output res-block)
    def transformer_depth_output(self) -> List[Int]:
        var d = List[Int]()
        d.append(10); d.append(10); d.append(10)
        d.append(2); d.append(2); d.append(2)
        d.append(0); d.append(0); d.append(0)
        return d^

    # num attention heads at a given channel width = channels / head_dim.
    def num_heads(self, channels: Int) -> Int:
        return channels // self.head_dim


def _eq(a: String, b: StringLiteral) -> Bool:
    return a == String(b)


def read_sdxl_config(json_path: String) raises -> SDXLConfig:
    """Parse the SDXL JSON config → SDXLConfig. Reads scalar arch + recipe
    fields; structural depth lists are SDXL invariants (see struct doc).
    Missing keys keep the SDXLConfig() defaults."""
    var bytes = _read_file_bytes(json_path)
    var cur = _Cursor(bytes^)
    var cfg = SDXLConfig()

    cur.expect(0x7B)  # '{'
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance()
        return cfg^

    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)  # ':'

        if _eq(key, "checkpoint"):
            var sc = _read_scalar(cur)
            cfg.checkpoint = sc.s
        elif _eq(key, "vae"):
            var sc = _read_scalar(cur)
            cfg.vae = sc.s
        elif _eq(key, "in_channels"):
            var sc = _read_scalar(cur)
            cfg.in_channels = Int(sc.num)
        elif _eq(key, "out_channels"):
            var sc = _read_scalar(cur)
            cfg.out_channels = Int(sc.num)
        elif _eq(key, "model_channels"):
            var sc = _read_scalar(cur)
            cfg.model_channels = Int(sc.num)
        elif _eq(key, "num_res_blocks"):
            var sc = _read_scalar(cur)
            cfg.num_res_blocks = Int(sc.num)
        elif _eq(key, "context_dim"):
            var sc = _read_scalar(cur)
            cfg.context_dim = Int(sc.num)
        elif _eq(key, "head_dim"):
            var sc = _read_scalar(cur)
            cfg.head_dim = Int(sc.num)
        elif _eq(key, "adm_in_channels"):
            var sc = _read_scalar(cur)
            cfg.adm_in_channels = Int(sc.num)
        elif _eq(key, "num_groups"):
            var sc = _read_scalar(cur)
            cfg.num_groups = Int(sc.num)
        elif _eq(key, "time_embed_dim"):
            var sc = _read_scalar(cur)
            cfg.time_embed_dim = Int(sc.num)
        elif _eq(key, "transformer_depth_middle"):
            var sc = _read_scalar(cur)
            cfg.transformer_depth_middle = Int(sc.num)
        elif _eq(key, "learning_rate"):
            var sc = _read_scalar(cur)
            cfg.lr = Float32(sc.num)
        elif _eq(key, "lora_rank"):
            var sc = _read_scalar(cur)
            cfg.lora_rank = Int(sc.num)
        elif _eq(key, "lora_alpha"):
            var sc = _read_scalar(cur)
            cfg.lora_alpha = Float32(sc.num)
        elif _eq(key, "timestep_shift"):
            var sc = _read_scalar(cur)
            cfg.timestep_shift = Float32(sc.num)
        elif _eq(key, "max_grad_norm"):
            var sc = _read_scalar(cur)
            cfg.max_grad_norm = Float32(sc.num)
        elif _eq(key, "max_steps"):
            var sc = _read_scalar(cur)
            cfg.max_steps = Int(sc.num)
        elif _eq(key, "save_every"):
            var sc = _read_scalar(cur)
            cfg.save_every = Int(sc.num)
        elif _eq(key, "sample_every"):
            var sc = _read_scalar(cur)
            cfg.sample_every = Int(sc.num)
        elif _eq(key, "beta_start"):
            var sc = _read_scalar(cur)
            cfg.beta_start = Float32(sc.num)
        elif _eq(key, "beta_end"):
            var sc = _read_scalar(cur)
            cfg.beta_end = Float32(sc.num)
        elif _eq(key, "num_train_timesteps"):
            var sc = _read_scalar(cur)
            cfg.num_train_timesteps = Int(sc.num)
        else:
            # Unknown key (e.g. model_type, channel_mult list, transformer_depth
            # lists, prediction_type, nested optimizer object): skip its value.
            _skip_value(cur)

        cur.skip_ws()
        var nb = cur.peek()
        if nb == 0x2C:  # ','
            cur.advance()
            cur.skip_ws()
            continue
        elif nb == 0x7D:  # '}'
            cur.advance()
            break
        else:
            raise Error("read_sdxl_config: expected ',' or '}'")

    return cfg^


def sdxl() raises -> SDXLConfig:
    return read_sdxl_config(String(SDXL_CONFIG))
