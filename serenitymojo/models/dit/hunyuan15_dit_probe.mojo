# hunyuan15_dit_probe.mojo — COMPILE-ONLY probe for Hunyuan15Dit.
# Imports + references the types and monomorphizes the comptime helpers (rope +
# one double-stream block + patch embed via forward). Does NOT run a numeric gate
# (HunyuanVideo-1.5 weights are NOT on disk — numeric gate is BLOCKED-no-weights).
# Verification = clean compile (EXIT=0) + reference fidelity vs hunyuan15_dit.rs.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.models.dit.hunyuan15_dit import (
    Hunyuan15Config,
    Hunyuan15Dit,
    hunyuan15_double_block,
    hunyuan15_build_rope,
    hunyuan15_rope_positions,
)


def main() raises:
    var cfg = Hunyuan15Config.default()

    # Config sanity (matches hunyuan15_dit.rs:68-85).
    if cfg.num_double_blocks != 54:
        raise Error("Hunyuan15 config: expected 54 double-stream blocks")
    if cfg.num_heads * cfg.head_dim != cfg.hidden_size:
        raise Error("Hunyuan15 config: num_heads*head_dim != hidden_size")
    var axes = cfg.rope_axes()
    var axsum = axes[0] + axes[1] + axes[2]
    if axsum != cfg.head_dim:
        raise Error("Hunyuan15 config: rope_dim_list must sum to head_dim")
    if cfg.patch_f != 1 or cfg.patch_h != 1 or cfg.patch_w != 1:
        raise Error("Hunyuan15 config: expected patch_size (1,1,1)")

    # Monomorphize the comptime types + helpers so their BODIES compile.
    comptime DitT = Hunyuan15Dit
    # CHUNK A: rope tables + ONE double-stream block at a tiny token grid.
    comptime RopePosFn = hunyuan15_rope_positions
    comptime RopeFn = hunyuan15_build_rope
    # One double-stream block: S_IMG=4, S_TXT=2, H=16, DH=128.
    comptime BlockFn = hunyuan15_double_block[4, 2, 16, 128]
    # CHUNK B: the full forward (patch embed + 54 blocks + final + unpatchify),
    # monomorphized at a tiny grid TT=1,TH=2,TW=2 -> S_IMG=4, S_TXT=2.
    comptime FwdFn = Hunyuan15Dit.forward[1, 2, 2, 4, 2, 16, 128]

    print("hunyuan15_dit probe constructed; hidden_size=", cfg.hidden_size,
          " blocks=", cfg.num_double_blocks,
          " rope_axes=", axes[0], axes[1], axes[2],
          " theta=", cfg.rope_theta)
    print("hunyuan15_dit probe compiled")
