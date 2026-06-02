# Compile probe for nucleus_dit.mojo — imports + constructs the type + config.
# COMPILE ONLY (GPU wedged). EXIT=0 == pass.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.models.dit.nucleus_dit import (
    NucleusConfig,
    NucleusDiT,
    NucleusRope,
    build_nucleus_3d_rope,
)


def main() raises:
    var cfg = NucleusConfig.nucleus_image()
    print("layers:", cfg.num_layers, "experts:", cfg.num_experts)
    print("dense_inner_dim:", cfg.dense_inner_dim())
    print("cap@3:", cfg.capacity_factor_for(3), "cap@5:", cfg.capacity_factor_for(5))

    # Construct an empty DiT to force struct instantiation at S_IMG=1024, S_TXT=512.
    var weights = List[ArcPointer[Tensor]]()
    var n2i = Dict[String, Int]()
    var dit = NucleusDiT[1024, 512](weights^, n2i^, cfg)
    print("NucleusDiT[1024,512] constructed; heads:", dit.config.num_heads)
    print("nucleus_dit probe compiled")
