# models/krea2/config.mojo — Krea-2-Raw (krea2) per-variant training config.
#
# Same pattern as models/klein/config.mojo: NO hardcoded arch/recipe — the
# params live in serenitymojo/configs/krea2.json (single source of truth, the
# values below are verified against the REAL checkpoint header, see the receipts
# block). `krea2_raw()` reads that JSON via read_model_config.
#
# ── DIMS (confirmed by a python struct-read of the raw.safetensors header) ─────
#   blocks.0.attn.wq.weight        BF16 [6144, 6144]   (HEADS·HEADDIM = 48·128)
#   blocks.0.attn.wk.weight        BF16 [1536, 6144]   (KVHEADS·HEADDIM = 12·128)
#   blocks.0.attn.wv.weight        BF16 [1536, 6144]
#   blocks.0.attn.gate.weight      BF16 [6144, 6144]   (the sigmoid-gate proj)
#   blocks.0.attn.wo.weight        BF16 [6144, 6144]
#   blocks.0.attn.qknorm.qnorm.scale  F32 [128]        (= HEADDIM)
#   blocks.0.attn.qknorm.knorm.scale  F32 [128]
#   blocks.0.mlp.gate.weight       BF16 [16384, 6144]  (mlpdim = 16384)
#   blocks.0.mlp.up.weight         BF16 [16384, 6144]
#   blocks.0.mlp.down.weight       BF16 [6144, 16384]
#   blocks.0.mod.lin               F32  [36864]        (= 6·6144, a bare Parameter)
#   blocks.0.prenorm.scale         F32  [6144]
#   blocks.0.postnorm.scale        F32  [6144]
#   28 blocks (num_single). features=6144, heads=48, kvheads=12, headdim=128,
#   theta=1e3 (krea2.py:55-68 KREA2_MMDIT_CONFIG + SingleMMDiTConfig defaults).
#   in/out channels for the `first`/`last` Linears = channels·patch² = 16·4 = 64.
#
# ── LoRA RECIPE (ai-toolkit krea2.py + lora_special.py + config_modules.py) ────
#   krea2.py:148  self.target_lora_modules = ["SingleStreamDiT"]  → the LoRA
#   network (LoRASpecialNetwork, lora_special.py:338-405) wraps every nn.Linear
#   (LINEAR_MODULES, lora_special.py:29) found UNDER a SingleStreamDiT-class
#   module whose dotted name contains "blocks" (transformer_only filter,
#   get_transformer_block_names()=["blocks"], krea2.py:492-493). That is the
#   8 per-block Linears:
#       attn.wq  attn.wk  attn.wv  attn.gate  attn.wo
#       mlp.gate  mlp.up  mlp.down
#   NOTE: `mod.lin` is a torch.nn.Parameter (DoubleSharedModulation.lin,
#   mmdit.py:125), NOT an nn.Linear, so ai-toolkit does NOT LoRA-wrap it. The
#   prenorm/postnorm/qknorm scales are also non-Linear → frozen. So per block:
#   8 LoRA adapters (NOT 9 — the brief's "mod_lin" target is not what the oracle
#   produces; flagged in the handoff).
#
#   rank/alpha: NetworkConfig defaults rank=4, alpha=1.0 (config_modules.py:180-
#   183); scale = alpha/rank (lora_special.py:116). The shipped krea2.json pins
#   rank=16/alpha=16 (scale 1.0) — a typical SDXL/flux-style LoRA preset; both
#   are recipe choices, override in the JSON. lora_down (A=[rank,in]) kaiming-
#   uniform a=sqrt(5); lora_up (B=[out,rank]) ZERO-init (lora_special.py:120-122).
#   lr: TrainConfig.lr default 1e-6 (config_modules.py:359); krea2.json pins 1e-4.
#   scheduler: flow-match — CustomFlowMatchEulerDiscreteScheduler, exponential
#   dynamic time-shift between (256-res → base_shift 0.5) and (1280-res → max_shift
#   1.15), num_train_timesteps=1000 (krea2.py:76-85). Velocity target = noise -
#   clean, no time flip (krea2.py:399-403, get_noise_prediction divides t by 1000).

from serenitymojo.training.train_config import TrainConfig
from serenitymojo.io.train_config_reader import read_model_config


comptime KREA2_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/krea2.json"


def krea2_raw() raises -> TrainConfig:
    """Krea-2-Raw (oss_raw) single_mmdit_large_wide LoRA training config."""
    return read_model_config(String(KREA2_CONFIG))
