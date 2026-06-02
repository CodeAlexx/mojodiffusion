# Compile-only probe for the HiDream-O1 DiT. Constructs the config + DiT type
# and references the public forward signature. DO NOT RUN (GPU wedged).
# Build: pixi run mojo build -I . -Xlinker -lm \
#   serenitymojo/models/dit/hidream_o1_probe.mojo -o /tmp/hdprobe

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.hidream_o1 import (
    HiDreamO1Config,
    HiDreamO1DiT,
    build_mrope_positions,
)


def main() raises:
    var cfg = HiDreamO1Config.dev_8b()
    # Sanity: section sums to head_dim/2.
    if cfg.mrope_t + cfg.mrope_h + cfg.mrope_w != cfg.head_dim // 2:
        raise Error("mrope_section must sum to head_dim/2")

    # Build a tiny position-id stream (compile-time exercise of the builder).
    var full_ids = List[Int]()
    full_ids.append(1)  # text
    full_ids.append(cfg.tms_token_id)
    full_ids.append(cfg.vision_start_token_id)
    full_ids.append(cfg.image_token_id)
    var pos = build_mrope_positions(
        full_ids, cfg.image_token_id, cfg.vision_start_token_id, 1, 1, cfg.fix_point
    )
    _ = pos

    # Construct an empty DiT (S=4) to force monomorphization of the comptime
    # type and its methods. We reference forward() behind a False guard so the
    # compiler typechecks the full parametric body (mrope/mask/layer/sdpa) WITHOUT
    # executing any GPU op (build does not run).
    comptime S = 4
    var weights = List[ArcPointer[Tensor]]()
    var name_to_idx = Dict[String, Int]()
    var dit = HiDreamO1DiT[S](weights^, name_to_idx^, cfg)
    _ = dit._has(String("nonexistent"))

    if False:
        var ctx = DeviceContext()
        var ids = List[Int]()
        ids.append(1)
        ids.append(cfg.tms_token_id)
        var t = pos[0].copy()
        var hh = pos[1].copy()
        var ww = pos[2].copy()
        var dummy_vals = List[Float32]()
        for _ in range(2 * 3072):
            dummy_vals.append(Float32(0.0))
        var dsh = List[Int]()
        dsh.append(1); dsh.append(2); dsh.append(3072)
        var patches = Tensor.from_host(dummy_vals, dsh^, STDtype.BF16, ctx)
        var out = dit.forward(ids, patches, t^, hh^, ww^, 2, Float32(0.5), ctx)
        _ = out

    print("hidream_o1 DiT probe compiled")
