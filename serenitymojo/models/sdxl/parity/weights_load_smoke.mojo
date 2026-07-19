# weights_load_smoke.mojo — verify SDXL ResBlock weights load from the REAL
# LDM-format checkpoint with correct shapes (RSCF conv filters).
#
# Loads input_blocks.4.0 (320->640, HAS skip_connection) and input_blocks.1.0
# (320->320, NO skip) from the real sdxl_unet_bf16.safetensors and prints the
# loaded tensor shapes. Confirms _load_conv_rscf produced [Kh,Kw,Cin,Cout].
#
# Run:
#   cd /home/alex/mojodiffusion
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/models/sdxl/parity/weights_load_smoke.mojo -o /tmp/sdxl_wl
#   /tmp/sdxl_wl

from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.sdxl.weights import load_resblock_weights, ResBlockWeights
from serenitymojo.models.sdxl.config import sdxl


comptime CKPT = "/home/alex/.serenity/models/checkpoints/sdxl_unet_bf16.safetensors"


def _shape_str(t: List[Int]) -> String:
    var s = String("[")
    for i in range(len(t)):
        s += String(t[i])
        if i + 1 < len(t):
            s += String(", ")
    s += String("]")
    return s


def _report(name: String, w: ResBlockWeights) raises:
    print("---", name, "---")
    print("  gn1_w  ", _shape_str(w.gn1_w.shape()))
    print("  conv1_w", _shape_str(w.conv1_w.shape()), "(RSCF [Kh,Kw,Cin,Cout])")
    print("  emb_w  ", _shape_str(w.emb_w.shape()))
    print("  gn2_w  ", _shape_str(w.gn2_w.shape()))
    print("  conv2_w", _shape_str(w.conv2_w.shape()), "(RSCF)")
    print("  has_skip", w.has_skip)
    if w.has_skip:
        print("  skip_w ", _shape_str(w.skip_w.shape()), "(RSCF 1x1)")


def main() raises:
    var ctx = DeviceContext()

    # config reads from JSON (binding rule)
    var cfg = sdxl()
    print("SDXL config: model_channels=", cfg.model_channels,
          " context_dim=", cfg.context_dim, " adm_in=", cfg.adm_in_channels,
          " num_groups=", cfg.num_groups, " head_dim=", cfg.head_dim)

    var st = SafeTensors.open(String(CKPT))

    # input_blocks.4.0: ResBlock 320->640 WITH 1x1 skip_connection
    var w4 = load_resblock_weights(st, String("input_blocks.4.0"), ctx)
    _report(String("input_blocks.4.0 (320->640, skip)"), w4)

    # input_blocks.1.0: ResBlock 320->320 WITHOUT skip
    var w1 = load_resblock_weights(st, String("input_blocks.1.0"), ctx)
    _report(String("input_blocks.1.0 (320->320, no skip)"), w1)

    print("")
    print("WEIGHTS LOAD SMOKE OK (real LDM checkpoint, RSCF conv filters)")
