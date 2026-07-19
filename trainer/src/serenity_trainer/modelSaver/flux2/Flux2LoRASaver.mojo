# Flux2LoRASaver.mojo — 1:1 surface port of Serenity
#   modules/modelSaver/flux2/Flux2LoRASaver.py  (Flux2LoRASaver)
# composed with the save mechanics of
#   modules/modelSaver/mixin/LoRASaverMixin.py   (_get_state_dict → __save_*)
# and the dispatch wrapper
#   modules/modelSaver/Flux2LoRAModelSaver.py     (make_lora_model_saver(...)).
#
# Serenity's Flux2LoRASaver._get_state_dict (Flux2LoRASaver.py:20-30):
#   state_dict = {}
#   state_dict |= model.transformer_lora.state_dict()   # the trained adapters
#   state_dict |= model.lora_state_dict                 # any preloaded extras
#   return state_dict
# _get_convert_key_sets returns None (Flux2LoRASaver.py:17-18) → no OMI/legacy key
# remap; LoRASaverMixin saves the raw Serenity LoRAModuleWrapper keys. The
# contract records that route behavior only; no numeric parity claim is made.
#
# ── SAVE-KEY LAYOUT (the Serenity-faithful diffusers names) ─────────────────
# The wrapped module is named "transformer" (Flux2LoRASetup.py:57); each wrapped
# diffusers Linear <name> writes the LoRAModule (LoRAModule.py:287-329) state dict:
#   transformer.<diffusers_name>.lora_down.weight  = lora_down.weight  [rank, in]
#   transformer.<diffusers_name>.lora_up.weight    = lora_up.weight    [out, rank]
#   transformer.<diffusers_name>.alpha             = alpha scalar (LoRAModule.py:303)
# diffusers_name enumerated by modelSetup/flux2LoraTargets (block-major,
# suffix-minor): "transformer_blocks.<i>.attn.to_q", … , "single_transformer_blocks.<i>.attn.to_out".
#
# ── ADAPTER SLOT ↔ Serenity KEY RECONCILIATION (load-bearing) ───────────────
# The current KleinLoraSet is already 1:1 with Serenity's LoRAModuleWrapper over
# Flux2LoRASetup.py:57-58: DBL_SLOTS=12 per double block and SGL_SLOTS=2 per single
# block. Therefore the saver must emit each adapter directly, with no fused qkv
# splitting:
#   double slots 0..11:
#     attn.to_q, attn.to_k, attn.to_v, attn.to_out.0,
#     ff.linear_in, ff.linear_out,
#     attn.add_q_proj, attn.add_k_proj, attn.add_v_proj, attn.to_add_out,
#     ff_context.linear_in, ff_context.linear_out
#   single slots 0..1:
#     attn.to_qkv_mlp_proj, attn.to_out
#
# Reuses ONLY serenitymojo {tensor, io, ops}. No serenitymojo model code imported.
# Dtype: BF16 storage written verbatim; a `dtype` override casts before write.

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import zeros_device, add_scalar

from serenity_trainer.model.klein.klein_stack_lora import (
    KleinLoraSet, DBL_SLOTS, SGL_SLOTS,
)
from serenity_trainer.modelSetup.flux2LoraTargets import (
    double_block_diffusers_suffixes,
    flux2_double_module, flux2_single_module, flux2_lora_save_prefix,
)


comptime TArc = ArcPointer[Tensor]


comptime FLUX2_FMT_DIFFUSERS = 0
comptime FLUX2_FMT_CKPT = 1
comptime FLUX2_FMT_SAFETENSORS = 2
comptime FLUX2_FMT_LEGACY_SAFETENSORS = 3
comptime FLUX2_FMT_COMFY_LORA = 4
comptime FLUX2_FMT_INTERNAL = 5


struct Flux2LoraStateDictContract(Movable):
    var includes_transformer_lora: Bool
    var includes_preloaded_lora_state_dict: Bool
    var includes_text_encoder_lora: Bool
    var includes_embedding_saver_state: Bool
    var can_bundle_additional_embeddings: Bool
    var has_convert_key_sets: Bool
    var convert_key_sets_source: String
    var wrapper_prefix: String
    var lora_down_suffix: String
    var lora_up_suffix: String
    var alpha_suffix: String
    var alpha_buffer_included: Bool
    var key_namespace: String
    var source_count: Int
    var preserves_tensor_storage_dtype: Bool

    def __init__(out self):
        self.includes_transformer_lora = True
        self.includes_preloaded_lora_state_dict = True
        self.includes_text_encoder_lora = False
        self.includes_embedding_saver_state = False
        self.can_bundle_additional_embeddings = False
        self.has_convert_key_sets = False
        self.convert_key_sets_source = String("None")
        self.wrapper_prefix = String("transformer")
        self.lora_down_suffix = String(".lora_down.weight")
        self.lora_up_suffix = String(".lora_up.weight")
        self.alpha_suffix = String(".alpha")
        self.alpha_buffer_included = True
        self.key_namespace = String("transformer.<diffusers module>.{lora_down.weight,lora_up.weight,alpha}")
        self.source_count = 2
        self.preserves_tensor_storage_dtype = True


struct Flux2LoraSavePlan(Movable):
    var output_model_format: Int
    var output_model_destination: String
    var route_name: String
    var target_key_namespace: String
    var writes_safetensors: Bool
    var internal_destination: String
    var saver_mixin_name: String
    var save_delegates_to_mixin: Bool
    var safetensors_uses_legacy_route_without_omi: Bool
    var legacy_safetensors_uses_legacy_route: Bool
    var internal_uses_safetensors_route: Bool
    var internal_passes_dtype_none: Bool
    var wrapper_saves_internal_data_after_leaf_save: Bool
    var dtype_override_applies_before_key_route: Bool
    var has_convert_key_sets: Bool
    var convert_key_sets_source: String
    var creates_safetensors_header: Bool
    var creates_parent_directory: Bool
    var state_dict_contract: Flux2LoraStateDictContract
    var preserves_storage_dtype_without_override: Bool
    var runtime_write_implemented_for_contract: Bool

    def __init__(
        out self,
        output_model_format: Int,
        var output_model_destination: String,
        var route_name: String,
        var target_key_namespace: String,
        var internal_destination: String,
        writes_safetensors: Bool,
    ):
        self.output_model_format = output_model_format
        self.output_model_destination = output_model_destination^
        self.route_name = route_name^
        self.target_key_namespace = target_key_namespace^
        self.writes_safetensors = writes_safetensors
        self.internal_destination = internal_destination^
        self.saver_mixin_name = String("LoRASaverMixin")
        self.save_delegates_to_mixin = True
        self.safetensors_uses_legacy_route_without_omi = output_model_format == FLUX2_FMT_SAFETENSORS
        self.legacy_safetensors_uses_legacy_route = output_model_format == FLUX2_FMT_LEGACY_SAFETENSORS
        self.internal_uses_safetensors_route = output_model_format == FLUX2_FMT_INTERNAL
        self.internal_passes_dtype_none = output_model_format == FLUX2_FMT_INTERNAL
        self.wrapper_saves_internal_data_after_leaf_save = output_model_format == FLUX2_FMT_INTERNAL
        self.dtype_override_applies_before_key_route = (
            output_model_format == FLUX2_FMT_SAFETENSORS
            or output_model_format == FLUX2_FMT_LEGACY_SAFETENSORS
        )
        self.has_convert_key_sets = False
        self.convert_key_sets_source = String("None")
        self.creates_safetensors_header = writes_safetensors
        self.creates_parent_directory = writes_safetensors
        self.state_dict_contract = Flux2LoraStateDictContract()
        self.preserves_storage_dtype_without_override = True
        self.runtime_write_implemented_for_contract = False


def flux2_lora_saver_has_convert_key_sets() -> Bool:
    return False


def flux2_lora_state_dict_contract() -> Flux2LoraStateDictContract:
    return Flux2LoraStateDictContract()


def flux2_lora_state_dict_source_names() -> List[String]:
    var names = List[String]()
    names.append(String("model.transformer_lora.state_dict() when model.transformer_lora is not None"))
    names.append(String("model.lora_state_dict when model.lora_state_dict is not None"))
    return names^


def flux2_lora_save_plan(
    output_model_format: Int,
    output_model_destination: String,
) raises -> Flux2LoraSavePlan:
    if output_model_format == FLUX2_FMT_SAFETENSORS:
        return Flux2LoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("legacy_safetensors"),
            String("raw_diffusers_loramodule_keys"),
            String(),
            True,
        )
    if output_model_format == FLUX2_FMT_LEGACY_SAFETENSORS:
        return Flux2LoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("legacy_safetensors"),
            String("raw_diffusers_loramodule_keys"),
            String(),
            True,
        )
    if output_model_format == FLUX2_FMT_INTERNAL:
        return Flux2LoraSavePlan(
            output_model_format,
            output_model_destination.copy(),
            String("internal_lora"),
            String("raw_diffusers_loramodule_keys"),
            output_model_destination + String("/lora/lora.safetensors"),
            True,
        )
    if output_model_format == FLUX2_FMT_DIFFUSERS:
        raise Error("Flux2LoRASaver: DIFFUSERS LoRA output is not implemented in Serenity")
    raise Error("Flux2LoRASaver: unsupported ModelFormat")


struct Flux2LoraStateDict(Movable):
    var names: List[String]
    var tensors: List[TArc]

    def __init__(out self, var names: List[String], var tensors: List[TArc]):
        self.names = names^
        self.tensors = tensors^


# Build A device tensor from a KleinLoraSet adapter's host BF16 list (shape [rank,in]).
def _adapter_a_tensor(set: KleinLoraSet, dbl: Bool, flat: Int, ctx: DeviceContext) raises -> Tensor:
    if dbl:
        ref ad = set.dbl[flat]
        var sh = List[Int](); sh.append(ad.rank); sh.append(ad.in_f)
        return Tensor.from_host_bf16(ad.a.copy(), sh^, ctx)
    ref ad = set.sgl[flat]
    var sh = List[Int](); sh.append(ad.rank); sh.append(ad.in_f)
    return Tensor.from_host_bf16(ad.a.copy(), sh^, ctx)


# Build B device tensor (shape [out,rank]).
def _adapter_b_tensor(set: KleinLoraSet, dbl: Bool, flat: Int, ctx: DeviceContext) raises -> Tensor:
    if dbl:
        ref ad = set.dbl[flat]
        var sh = List[Int](); sh.append(ad.out_f); sh.append(ad.rank)
        return Tensor.from_host_bf16(ad.b.copy(), sh^, ctx)
    ref ad = set.sgl[flat]
    var sh = List[Int](); sh.append(ad.out_f); sh.append(ad.rank)
    return Tensor.from_host_bf16(ad.b.copy(), sh^, ctx)


def _alpha_of(set: KleinLoraSet, dbl: Bool, flat: Int) -> Float32:
    if dbl:
        ref ad = set.dbl[flat]
        return ad.scale * Float32(ad.rank)
    ref ad = set.sgl[flat]
    return ad.scale * Float32(ad.rank)   # alpha = (alpha/rank) * rank


def _double_slot_for_suffix_index(suffix_index: Int) -> Int:
    # double_block_diffusers_suffixes() order is the Serenity save-key order.
    # KleinLoraSet stores slots stream-major: img q/k/v/out/ff_in/ff_out, then
    # txt q/k/v/out/ff_in/ff_out.
    if suffix_index < 4:
        return suffix_index
    if suffix_index < 8:
        return suffix_index + 2
    if suffix_index < 10:
        return suffix_index - 4
    return suffix_index


def _emit(
    mut names: List[String], mut tensors: List[TArc],
    diffusers_module: String, var a: Tensor, var b: Tensor, alpha: Float32,
    dtype: STDtype, ctx: DeviceContext,
) raises:
    var prefix = flux2_lora_save_prefix(diffusers_module)
    names.append(prefix + String(".lora_down.weight"))
    tensors.append(TArc(_maybe_cast(a^, dtype, ctx)))
    names.append(prefix + String(".lora_up.weight"))
    tensors.append(TArc(_maybe_cast(b^, dtype, ctx)))
    # Serenity's LoRAModule keeps alpha in the module state_dict (registered
    # buffer, LoRAModule.py:303), so the saved file always carries it.
    names.append(prefix + String(".alpha"))
    tensors.append(TArc(_scalar(alpha, dtype, ctx)))


# Build the Serenity-faithful state dict from a KleinLoraSet. `D` is retained
# for the existing saver API; adapter shapes already carry the in/out dimensions.
def build_flux2_lora_state_dict(
    set: KleinLoraSet, D: Int, ctx: DeviceContext, dtype: STDtype = STDtype.BF16,
) raises -> Flux2LoraStateDict:
    if D <= 0:
        raise Error("build_flux2_lora_state_dict: D must be positive")
    var names = List[String]()
    var tensors = List[TArc]()

    for bi in range(set.num_double):
        var base = bi * DBL_SLOTS
        var dsuf = double_block_diffusers_suffixes()
        for suffix_index in range(len(dsuf)):
            var slot = _double_slot_for_suffix_index(suffix_index)
            _emit(names, tensors, flux2_double_module(bi, dsuf[suffix_index]),
                  _adapter_a_tensor(set, True, base + slot, ctx),
                  _adapter_b_tensor(set, True, base + slot, ctx),
                  _alpha_of(set, True, base + slot), dtype, ctx)

    for bi in range(set.num_single):
        var base = bi * SGL_SLOTS
        _emit(names, tensors, flux2_single_module(bi, String("attn.to_qkv_mlp_proj")),
              _adapter_a_tensor(set, False, base + 0, ctx),
              _adapter_b_tensor(set, False, base + 0, ctx),
              _alpha_of(set, False, base + 0), dtype, ctx)
        _emit(names, tensors, flux2_single_module(bi, String("attn.to_out")),
              _adapter_a_tensor(set, False, base + 1, ctx),
              _adapter_b_tensor(set, False, base + 1, ctx),
              _alpha_of(set, False, base + 1), dtype, ctx)

    return Flux2LoraStateDict(names^, tensors^)


# LoRASaverMixin.save → _save → __save_legacy_safetensors (key_sets None ⇒ raw).
def save_flux2_lora(
    set: KleinLoraSet, D: Int, destination: String, ctx: DeviceContext,
    dtype: STDtype = STDtype.BF16,
) raises:
    var sd = build_flux2_lora_state_dict(set, D, ctx, dtype)
    save_safetensors(sd.names, sd.tensors, destination, ctx)


# ── helpers ───────────────────────────────────────────────────────────────────
def _maybe_cast(var t: Tensor, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    return cast_tensor(t, dtype, ctx)


def _scalar(val: Float32, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int](); sh.append(1)
    var z = zeros_device(sh^, dtype, ctx)
    return add_scalar(z, val, ctx)


struct Flux2LoRASaver(Movable):
    def __init__(out self):
        pass

    def _get_convert_key_sets(self) -> Bool:
        return flux2_lora_saver_has_convert_key_sets()

    def state_dict_contract(self) -> Flux2LoraStateDictContract:
        return flux2_lora_state_dict_contract()

    def state_dict_source_names(self) -> List[String]:
        return flux2_lora_state_dict_source_names()

    def save_plan(
        self,
        output_model_format: Int,
        output_model_destination: String,
    ) raises -> Flux2LoraSavePlan:
        return flux2_lora_save_plan(output_model_format, output_model_destination)
