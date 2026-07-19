# load_flux_weights_smoke.mojo — Blocker B GATE: load REAL flux1-dev block weights
# (double 0, last double, last single) + assert embedder shapes match config.
#
# Proves the Blocker-B loader (models/flux/weights.mojo) produces tensor-struct
# shapes compatible with models/flux/block.mojo at the real flux1-dev config
# (D=3072 inner, 24 heads, Dh=128, Fmlp=12288). Shape-derivation in the loader is
# checkpoint-driven; this gate re-asserts every derived dim against the JSON config
# and against the raw safetensors header for the embedders.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/models/flux/parity/load_flux_weights_smoke.mojo

from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.models.flux.config import flux_dev
from serenitymojo.models.flux.weights import (
    load_double_block_weights, load_single_block_weights, _dim0, _dim1,
)


comptime FLUX_PATH = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"


def _expect(name: String, got: Int, want: Int) raises:
    if got != want:
        raise Error(name + " = " + String(got) + " expected " + String(want))
    print("  OK ", name, "=", got)


def main() raises:
    var ctx = DeviceContext()
    print("=== Blocker B GATE: load REAL flux1-dev block weights + shape check ===")
    print("  path:", FLUX_PATH)

    # Config from JSON (source of truth for the asserted dims).
    var cfg = flux_dev()
    var D = cfg.d_model            # 3072 (JSON inner_dim -> d_model)
    var Dh = cfg.head_dim          # 128
    var H = cfg.n_heads            # 24  (JSON num_heads -> n_heads)
    var Fmlp = cfg.mlp_hidden      # 12288
    var num_double = cfg.num_double  # 19 -> last = 18
    var num_single = cfg.num_single  # 38 -> last = 37
    print("  config: D =", D, " Dh =", Dh, " heads =", H, " Fmlp =", Fmlp,
          " #double =", num_double, " #single =", num_single)

    # head_dim * num_heads must equal inner_dim (3072 == 128 * 24).
    _expect(String("Dh * H"), Dh * H, D)

    var st = SafeTensors.open(FLUX_PATH)

    # ---- DOUBLE block 0 ----
    var d0 = load_double_block_weights(st, 0, ctx)
    print("--- double block 0 ---")
    _expect(String("d0.img.wqkv"), d0.img.wqkv[].numel(), 3 * D * D)
    _expect(String("d0.img.bqkv"), d0.img.bqkv[].numel(), 3 * D)
    _expect(String("d0.img.wproj"), d0.img.wproj[].numel(), D * D)
    _expect(String("d0.img.bproj"), d0.img.bproj[].numel(), D)
    _expect(String("d0.img.wmlp0"), d0.img.wmlp0[].numel(), Fmlp * D)
    _expect(String("d0.img.bmlp0"), d0.img.bmlp0[].numel(), Fmlp)
    _expect(String("d0.img.wmlp2"), d0.img.wmlp2[].numel(), D * Fmlp)
    _expect(String("d0.img.bmlp2"), d0.img.bmlp2[].numel(), D)
    _expect(String("d0.img.q_norm"), d0.img.q_norm[].numel(), Dh)
    _expect(String("d0.img.k_norm"), d0.img.k_norm[].numel(), Dh)
    _expect(String("d0.txt.wqkv"), d0.txt.wqkv[].numel(), 3 * D * D)
    _expect(String("d0.txt.q_norm"), d0.txt.q_norm[].numel(), Dh)

    # ---- LAST DOUBLE block (num_double - 1) ----
    var dL = load_double_block_weights(st, num_double - 1, ctx)
    print("--- last double block", num_double - 1, "---")
    _expect(String("dL.img.wqkv"), dL.img.wqkv[].numel(), 3 * D * D)
    _expect(String("dL.img.wmlp2"), dL.img.wmlp2[].numel(), D * Fmlp)
    _expect(String("dL.txt.wproj"), dL.txt.wproj[].numel(), D * D)

    # ---- LAST SINGLE block (num_single - 1) ----
    var sL = load_single_block_weights(st, num_single - 1, ctx)
    print("--- last single block", num_single - 1, "---")
    _expect(String("sL.w1"), sL.w1[].numel(), (3 * D + Fmlp) * D)
    _expect(String("sL.b1"), sL.b1[].numel(), 3 * D + Fmlp)
    _expect(String("sL.w2"), sL.w2[].numel(), D * (D + Fmlp))
    _expect(String("sL.b2"), sL.b2[].numel(), D)
    _expect(String("sL.q_norm"), sL.q_norm[].numel(), Dh)
    _expect(String("sL.k_norm"), sL.k_norm[].numel(), Dh)

    # ---- EMBEDDERS: assert raw header shapes (stack-phase weights, shape-only) ----
    print("--- embedders (header shape check) ---")
    _expect(String("img_in [D,in_ch] dim0"), _dim0(st, String("img_in.weight")), D)
    _expect(String("img_in in_ch (=in_channels)"), _dim1(st, String("img_in.weight")), cfg.in_channels)
    _expect(String("txt_in [D,joint] dim0"), _dim0(st, String("txt_in.weight")), D)
    _expect(String("txt_in joint dim1"), _dim1(st, String("txt_in.weight")), cfg.joint_attention_dim)
    _expect(String("time_in.in_layer dim0"), _dim0(st, String("time_in.in_layer.weight")), D)
    _expect(String("time_in.in_layer dim1 (=timestep_dim)"), _dim1(st, String("time_in.in_layer.weight")), cfg.timestep_dim)
    _expect(String("vector_in.in_layer dim0"), _dim0(st, String("vector_in.in_layer.weight")), D)
    _expect(String("guidance_in.in_layer dim0"), _dim0(st, String("guidance_in.in_layer.weight")), D)
    _expect(String("final_layer.linear dim0 (=out_channels)"), _dim0(st, String("final_layer.linear.weight")), cfg.out_channels)
    _expect(String("final_layer.linear dim1"), _dim1(st, String("final_layer.linear.weight")), D)
    _expect(String("final_layer.adaLN dim0 (=2D)"), _dim0(st, String("final_layer.adaLN_modulation.1.weight")), 2 * D)

    print("VERDICT: PASS — flux1-dev double(0,", num_double - 1,
          ") + single(", num_single - 1,
          ") + embedders load with shapes matching config (D=3072, 24 heads, Dh=128)")
