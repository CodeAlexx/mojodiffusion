# Proves gemma4_ltx_streamed loads BOTH shipping key layouts.
#
#   google/gemma-4-12B-it          -> model.language_model.*
#   LTX-2.5 gemma4-12b-with-proj-* -> model.*
#
# Fixtures come from parity/gemma4_prefix_fixture.py (run it first). Each holds
# the exact key set `_load_layer` requests — sliding layer 0 WITH v_proj, global
# layer 5 WITHOUT it — plus, in the LTX fixture, the extra
# `text_embedding_projection.*` / `vision_model.*` tensors the real file carries
# and the text path must ignore.
#
# `layer_scalar` is stamped 0.375 + layer_idx in the fixture, so printing it
# proves the loader resolved the RIGHT tensor rather than merely finding a name.
#
# Run (no shim needed — no attention on this path):
#   pixi run mojo run -I . \
#     serenitymojo/models/text_encoder/parity/gemma4_prefix_probe.mojo

from std.gpu.host import DeviceContext

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.text_encoder.gemma4_ltx_streamed import (
    _load_layer,
    detect_gemma4_prefix,
)


comptime FIX = String(
    "/home/alex/mojodiffusion/serenitymojo/models/text_encoder/parity/gemma4_prefix_fixtures"
)


def _check(label: String, dir: String, want_prefix: String, ctx: DeviceContext) raises:
    var st = ShardedSafeTensors.open(dir)
    var prefix = detect_gemma4_prefix(st)
    var ok = prefix == want_prefix
    print(label, " detected='", prefix, "' expected='", want_prefix, "'",
          " PREFIX_OK" if ok else " PREFIX_WRONG")
    if not ok:
        raise Error("prefix mismatch for " + label)

    var l0 = _load_layer(st, 0, False, prefix, ctx)
    print(
        "   layer0 sliding: q=", l0.q_proj.numel(), " v=", l0.v_proj.numel(),
        " layer_scalar=", l0.layer_scalar, " (want 0.375)",
    )
    var l5 = _load_layer(st, 5, True, prefix, ctx)
    print(
        "   layer5 global : q=", l5.q_proj.numel(),
        " layer_scalar=", l5.layer_scalar, " (want 5.375)",
    )
    # v_proj on a global layer is the 1-element placeholder, never the file.
    if l5.v_proj.numel() != 1:
        raise Error("global layer v_proj placeholder was not used")


def main() raises:
    var ctx = DeviceContext()
    print("== gemma4 checkpoint-prefix resolution ==")
    _check(
        String("google_style"), FIX + "/google_style",
        String("model.language_model."), ctx,
    )
    _check(
        String("ltx25_style "), FIX + "/ltx25_style",
        String("model."), ctx,
    )
    print("BOTH LAYOUTS LOAD")
