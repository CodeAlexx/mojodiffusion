# LensLoRASaver.mojo — 1:1 port of Serenity
#   modules/modelSaver/lens/LensLoRASaver.py  (pr-1510)
# composed with the save mechanics of
#   modules/modelSaver/mixin/LoRASaverMixin.py  (_get_state_dict → __save_*)
# and the dispatch wrapper
#   modules/modelSaver/LensLoRAModelSaver.py     (make_lora_model_saver(...)).
# structurally mirrored on modelSaver/zImage/ZImageLoRASaver.mojo.
#
# Serenity SOURCE (LensLoRASaver.py:_get_state_dict):
#   state_dict = {}
#   if model.transformer_lora is not None:
#       state_dict |= model.transformer_lora.state_dict()   # the trained adapters
#   if model.lora_state_dict is not None:
#       state_dict |= model.lora_state_dict                 # any preloaded extras
#   return state_dict
# _get_convert_key_sets returns None (LensLoRASaver.py) → NO OMI/legacy key remap;
# the diffusers/PEFT keys are written verbatim. LoRASaverMixin._save with
# enable_omi_format=False routes ModelFormat.SAFETENSORS → __save_legacy_safetensors,
# which (since key_sets is None) is byte-identical to __save_safetensors: it just
# save_file()s the raw state dict.
#
# SAVED KEY NAMING (LoRAModule's nn.Linear submodules are named lora_down/lora_up;
# the state_dict carries those names verbatim, with the "transformer" wrapper
# prefix from LoRAModuleWrapper(model.transformer, "transformer", ...)):
#   transformer.<module>.lora_down.weight = a [rank, in]
#   transformer.<module>.lora_up.weight   = b [out, rank]
#   transformer.<module>.alpha            = scalar (= scale*rank = alpha)
# The Lens LoRA prefixes (the SHIPPED "attn-mlp" preset, 480 adapters) are
# enumerated by modelSetup/lensLoraTargets.lens_lora_target_prefixes: 48 blocks × 10
# per-block attn/mlp Linears (block-major, slot-minor) — NO mod, NO top-level (the
# layer_filter "attn,mlp" wraps only attn.*/img_mlp.*/txt_mlp.* Linears). The
# prefixes already carry the "transformer." host prefix. set.block MUST be in this
# SAME flat order.
#
# REAL MODEL TYPES (wired here, NOT the stale ZImage copy):
#   LensLoraSet  (model/lens/lens_backward.mojo) — fields: block: List[LoraAdapter]
#                (480, block-major/slot-minor), rank: Int.  (No .ad / .n_blocks.)
#   LoraAdapter  (module/LensLoRAModule.mojo)    — a: List[BFloat16] [rank,in],
#                b: List[BFloat16] [out,rank], in_f, out_f, rank, scale: Float32
#                (= alpha/rank).  There is NO `alpha` field → alpha = scale*rank.
# The host BF16 A/B lists are uploaded to device BF16 tensors with
# Tensor.from_host_bf16 (the same re-upload the LoRA forward/backward use).
#
# Reuses ONLY serenitymojo {tensor, io, ops}. No serenitymojo model code imported.
# Dtype: BF16 storage written verbatim (D2H raw byte copy); a `dtype` override is
# supported by casting each tensor before write (LoRASaverMixin._convert_state_dict_dtype).

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import zeros_device, add_scalar

from serenity_trainer.model.lens.lens_backward import LensLoraSet
from serenity_trainer.modelSetup import lensLoraTargets as LT


comptime TArc = ArcPointer[Tensor]

# Whether to also emit a "<prefix>.alpha" scalar per adapter (PEFT/ai-toolkit do).
comptime EMIT_ALPHA = True


struct LensLoraStateDict(Movable):
    var names: List[String]
    var tensors: List[TArc]

    def __init__(out self, var names: List[String], var tensors: List[TArc]):
        self.names = names^
        self.tensors = tensors^


# Build the LoRA state dict (names + tensors) from the trained adapter set.
# Mirrors LensLoRASaver._get_state_dict: iterate the transformer_lora adapters in
# the SAME order the setup created them and emit the saved key pair (+ alpha) for
# each. The authoritative order is lens_lora_target_prefixes() — 480 per-block
# (block-major, slot-minor), attn-mlp preset, NO top-level — and set.block is
# parallel to it (idx i ↔ prefixes[i]). The prefixes already carry the
# "transformer." host prefix.
def build_lens_lora_state_dict(
    set: LensLoraSet,
    ctx: DeviceContext,
    dtype: STDtype = STDtype.BF16,
) raises -> LensLoraStateDict:
    var names = List[String]()
    var tensors = List[TArc]()
    var prefixes = LT.lens_lora_target_prefixes()
    if len(prefixes) != len(set.block):
        raise Error(
            String("build_lens_lora_state_dict: adapter/prefix count mismatch: ")
            + String(len(set.block)) + String(" adapters vs ")
            + String(len(prefixes)) + String(" prefixes")
        )
    for i in range(len(prefixes)):
        var prefix = prefixes[i]
        ref ad = set.block[i]

        # A = lora_down.weight  [rank, in]  (host BF16 → device BF16, no F32 detour)
        var a_sh = List[Int](); a_sh.append(ad.rank); a_sh.append(ad.in_f)
        var a_t = Tensor.from_host_bf16(ad.a.copy(), a_sh^, ctx)
        names.append(prefix + String(".lora_down.weight"))
        tensors.append(TArc(_maybe_cast(a_t, dtype, ctx)))

        # B = lora_up.weight  [out, rank]
        var b_sh = List[Int](); b_sh.append(ad.out_f); b_sh.append(ad.rank)
        var b_t = Tensor.from_host_bf16(ad.b.copy(), b_sh^, ctx)
        names.append(prefix + String(".lora_up.weight"))
        tensors.append(TArc(_maybe_cast(b_t, dtype, ctx)))

        if EMIT_ALPHA:
            # LoraAdapter stores scale = alpha/rank → alpha = scale*rank.
            var alpha = ad.scale * Float32(ad.rank)
            names.append(prefix + String(".alpha"))
            tensors.append(TArc(_scalar(alpha, dtype, ctx)))
    return LensLoraStateDict(names^, tensors^)


# LoRASaverMixin.save → _save → __save_legacy_safetensors (key_sets None ⇒ raw).
# Writes the single-file safetensors at `destination`.
def save_lens_lora(
    set: LensLoraSet,
    destination: String,
    ctx: DeviceContext,
    dtype: STDtype = STDtype.BF16,
) raises:
    var sd = build_lens_lora_state_dict(set, ctx, dtype)
    save_safetensors(sd.names, sd.tensors, destination, ctx)


# ── helpers ───────────────────────────────────────────────────────────────────
def _maybe_cast(t: Tensor, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    # _convert_state_dict_dtype: cast each tensor to the requested save dtype; a
    # dtype no-op is still a device copy so caller-owned adapters are never aliased.
    return cast_tensor(t, dtype, ctx)


def _scalar(val: Float32, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    # Serenity stores alpha as a 0-dim scalar tensor (shape () ), not [1].
    var sh = List[Int]()          # empty shape → 0-dim
    var z = zeros_device(sh^, dtype, ctx)
    return add_scalar(z, val, ctx)
