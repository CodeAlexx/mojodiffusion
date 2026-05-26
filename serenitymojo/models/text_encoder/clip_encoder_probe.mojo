# clip_encoder_probe.mojo — compile/typecheck driver for clip_encoder.
# Constructs both configs + a DeviceContext, exercises the type. Does NOT load
# weights or run the GPU forward (GPU wedged; compile-only verification).

from std.gpu.host import DeviceContext
from serenitymojo.models.text_encoder.clip_encoder import ClipConfig, ClipEncoder


# Force monomorphization of load()+encode_sdxl[77]() for both configs without
# executing GPU work (`if False:` guard; GPU wedged, compile-only).
def _instantiate(ctx: DeviceContext) raises:
    if False:
        var clip_l = ClipEncoder.load(String("/nonexistent"), ClipConfig.clip_l(), ctx)
        var ids = List[Int]()
        ids.append(49406)
        ids.append(49407)
        var pair_l = clip_l.encode_sdxl(ids.copy(), ctx)
        print(pair_l[0].shape()[2], pair_l[1].shape()[1])
        var clip_g = ClipEncoder.load(String("/nonexistent"), ClipConfig.clip_g(), ctx)
        var pair_g = clip_g.encode_sdxl(ids^, ctx)
        print(pair_g[0].shape()[2], pair_g[1].shape()[1])


def main() raises:
    var l = ClipConfig.clip_l()
    var g = ClipConfig.clip_g()
    print("CLIP-L: hidden", l.hidden_size, "layers", l.num_layers,
          "heads", l.num_heads, "quick_gelu", l.use_quick_gelu)
    print("CLIP-G: hidden", g.hidden_size, "layers", g.num_layers,
          "heads", g.num_heads, "quick_gelu", g.use_quick_gelu)
    var ctx = DeviceContext()
    print("ctx ok", Int(ctx.id()))
    _instantiate(ctx)
    print("clip_encoder compile OK")
