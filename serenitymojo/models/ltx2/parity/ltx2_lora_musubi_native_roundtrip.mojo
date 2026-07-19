# ltx2_lora_musubi_native_roundtrip.mojo — round-trip musubi's NATIVE
# 1152-key LTX-2 LoRA artifact (lora_unet_* flat underscored keys, per-module
# {alpha, lora_down.weight, lora_up.weight}, all BF16) through the Mojo
# resume/save surface. Companion to ltx2_lora_musubi_real_artifact_roundtrip
# (which covers the 768-key comfy conversion); payload byte comparison is done
# by scripts/check_ltx2_lora_keys.py.
#
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/models/ltx2/parity/ltx2_lora_musubi_native_roundtrip.mojo \
#     -o /tmp/ltx2_lora_native_roundtrip
#   /tmp/ltx2_lora_native_roundtrip <in.safetensors> <out.safetensors>
#
# Oracle for the 2026-07-16 gate:
#   output/ltx2_musubi_ref/ltx2_musubi_ref.safetensors (1152 keys, rank 32)

from std.sys import argv
from std.gpu.host import DeviceContext
from std.collections import List

from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.training.lora_save import (
    NamedLora, load_lora_for_resume, save_lora_serenity_trainer,
)

comptime NUM_LAYERS = 48
comptime SCALE_FALLBACK = Float32(1.0)  # .alpha present -> loader reconstructs
# t2v = 384 modules (8/block) -> 1152 keys; v2v = 480 (10/block) -> 1440 keys.


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _native_prefixes(v2v: Bool) raises -> List[String]:
    # musubi native: lora_unet_model_transformer_blocks_{bi}_{fam}_{proj}
    # with dots flattened to underscores (to_out.0 -> to_out_0). v2v appends the
    # two video FFN modules per block (ff.net.0.proj -> ff_net_0_proj, ff.net.2 ->
    # ff_net_2) AFTER the 8 attention modules, matching the driver slot order.
    var fams = List[String]()
    fams.append("attn1")
    fams.append("attn2")
    var projs = List[String]()
    projs.append("to_q")
    projs.append("to_k")
    projs.append("to_v")
    projs.append("to_out_0")
    var out = List[String]()
    for bi in range(NUM_LAYERS):
        var stem = String("lora_unet_model_transformer_blocks_") + String(bi) + "_"
        for ref f in fams:
            for ref p in projs:
                out.append(stem + f + "_" + p)
        if v2v:
            out.append(stem + "ff_net_0_proj")
            out.append(stem + "ff_net_2")
    return out^


def main() raises:
    var args = argv()
    _require(len(args) >= 3, String("usage: <in.safetensors> <out.safetensors> [--v2v]"))
    var in_path = String(args[1])
    var out_path = String(args[2])
    var v2v = False
    for i in range(len(args)):
        if String(args[i]) == "--v2v":
            v2v = True
    var n_mod = 480 if v2v else 384
    var n_keys = n_mod * 3

    var ctx = DeviceContext()

    var src = SafeTensors.open(in_path)
    _require(
        len(src.tensors) == n_keys,
        String("expected ") + String(n_keys) + " source tensors, got "
        + String(len(src.tensors)),
    )

    var prefixes = _native_prefixes(v2v)
    _require(len(prefixes) == n_mod,
             String("expected ") + String(n_mod) + " native prefixes")

    var resumed = load_lora_for_resume(prefixes, SCALE_FALLBACK, in_path, ctx)
    _require(len(resumed) == n_mod,
             String("expected ") + String(n_mod) + " resumed adapters")
    # .alpha present -> scale must be reconstructed alpha/rank = 32/32 = 1.0
    for ref nl in resumed:
        _require(
            nl.adapter.scale == Float32(1.0),
            String("alpha-derived scale != 1.0 for ") + nl.prefix,
        )

    var saved = save_lora_serenity_trainer(resumed, out_path, ctx)
    _require(saved == n_mod, String("expected ") + String(n_mod) + " saved modules")

    var dst = SafeTensors.open(out_path)
    _require(
        len(dst.tensors) == n_keys,
        String("expected ") + String(n_keys) + " re-saved tensors, got "
        + String(len(dst.tensors)),
    )
    for key in src.tensors:
        _require(key in dst.tensors, String("re-saved artifact missing key ") + key)
    print("[roundtrip-native] PASS:", n_keys, "keys present after Mojo resume->save (v2v=", v2v, ")")
