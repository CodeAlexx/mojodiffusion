# Flux2 runtime tensor/safetensors loader helpers.
#
# These helpers are split from modelLoader/Flux2ModelLoader.mojo so the
# Serenity-named loader surface can build as a lightweight source-contract
# gate. The runtime path still mirrors Flux2ModelLoader.py + Flux2LoRASetup.py:
# BF16 tensors are loaded from safetensors views without persistent F32 storage.

from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors

from serenity_trainer.model.klein.lora_adapter import LoraAdapter
from serenity_trainer.model.klein.klein_stack_lora import (
    DBL_SLOTS,
    KleinLoraSet,
    SGL_SLOTS,
)
from serenity_trainer.modelSetup.flux2LoraTargets import (
    flux2_double_module,
    flux2_lora_save_prefix,
    flux2_single_module,
)


comptime TArc = ArcPointer[Tensor]


# Frozen transformer weight store (name -> BF16 device Tensor).
struct Flux2Weights(Movable):
    var weights: List[TArc]
    var name_to_idx: Dict[String, Int]

    def __init__(out self, var weights: List[TArc], var name_to_idx: Dict[String, Int]):
        self.weights = weights^
        self.name_to_idx = name_to_idx^

    @staticmethod
    def load(dir: String, ctx: DeviceContext) raises -> Flux2Weights:
        var sharded = ShardedSafeTensors.open(dir)
        var weights = List[TArc]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return Flux2Weights(weights^, name_to_idx^)

    def get(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("Flux2Weights: missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def has(self, name: String) -> Bool:
        return name in self.name_to_idx

    def count(self) -> Int:
        return len(self.weights)


def flux2_double_prefix(block_idx: Int) -> String:
    return String("transformer_blocks.") + String(block_idx)


def flux2_single_prefix(block_idx: Int) -> String:
    return String("single_transformer_blocks.") + String(block_idx)


def flux2_double_weight_key(block_idx: Int, suffix: String) -> String:
    return flux2_double_prefix(block_idx) + String(".") + suffix + String(".weight")


def flux2_single_weight_key(block_idx: Int, suffix: String) -> String:
    return flux2_single_prefix(block_idx) + String(".") + suffix + String(".weight")


struct Flux2LoraReload(Movable):
    var a: List[TArc]
    var b: List[TArc]
    var alpha: List[Float32]
    var rank: Int

    def __init__(out self, var a: List[TArc], var b: List[TArc], var alpha: List[Float32], rank: Int):
        self.a = a^
        self.b = b^
        self.alpha = alpha^
        self.rank = rank


def load_flux2_lora(
    path: String,
    prefixes: List[String],
    ctx: DeviceContext,
) raises -> Flux2LoraReload:
    var sharded = ShardedSafeTensors.open(path)
    var have = Dict[String, Int]()
    for ref nm in sharded.names():
        have[nm] = 1
    var a = List[TArc]()
    var b = List[TArc]()
    var alpha = List[Float32]()
    var rank = -1
    for i in range(len(prefixes)):
        var pre = prefixes[i]
        var ak = pre + String(".lora_down.weight")
        var bk = pre + String(".lora_up.weight")
        var at = Tensor.from_view(sharded.tensor_view(ak), ctx)
        var bt = Tensor.from_view(sharded.tensor_view(bk), ctx)
        if rank < 0:
            rank = at.shape()[0]
        var ah = pre + String(".alpha")
        var al = Float32(rank)
        if ah in have:
            var alt = Tensor.from_view(sharded.tensor_view(ah), ctx)
            al = alt.to_host(ctx)[0]
        a.append(TArc(at^))
        b.append(TArc(bt^))
        alpha.append(al)
    return Flux2LoraReload(a^, b^, alpha^, rank)


struct _AbHost(Movable):
    var a: List[Float32]
    var b: List[Float32]
    var rank: Int
    var in_f: Int

    def __init__(out self, var a: List[Float32], var b: List[Float32], rank: Int, in_f: Int):
        self.a = a^
        self.b = b^
        self.rank = rank
        self.in_f = in_f


def _read_ab_host(
    sharded: ShardedSafeTensors,
    prefix: String,
    ctx: DeviceContext,
) raises -> _AbHost:
    var at = Tensor.from_view(sharded.tensor_view(prefix + String(".lora_down.weight")), ctx)
    var bt = Tensor.from_view(sharded.tensor_view(prefix + String(".lora_up.weight")), ctx)
    var ash = at.shape()
    var rank = ash[0]
    var in_f = ash[1]
    var ah = at.to_host(ctx)
    var bh = bt.to_host(ctx)
    return _AbHost(ah^, bh^, rank, in_f)


def _alpha_host(
    sharded: ShardedSafeTensors,
    have: Dict[String, Int],
    prefix: String,
    rank: Int,
    ctx: DeviceContext,
) raises -> Float32:
    var ah = prefix + String(".alpha")
    if ah in have:
        var alt = Tensor.from_view(sharded.tensor_view(ah), ctx)
        return alt.to_host(ctx)[0]
    return Float32(rank)


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _simple_adapter(
    a: List[Float32],
    b: List[Float32],
    rank: Int,
    in_f: Int,
    alpha: Float32,
) raises -> LoraAdapter:
    var out_f = len(b) // rank
    var scale = alpha / Float32(rank)
    return LoraAdapter(
        a.copy(),
        b.copy(),
        rank,
        in_f,
        out_f,
        scale,
        _zeros(rank * in_f),
        _zeros(rank * in_f),
        _zeros(out_f * rank),
        _zeros(out_f * rank),
    )


def _double_suffixes_carrier_order() -> List[String]:
    var o = List[String]()
    o.append(String("attn.to_q"))
    o.append(String("attn.to_k"))
    o.append(String("attn.to_v"))
    o.append(String("attn.to_out.0"))
    o.append(String("ff.linear_in"))
    o.append(String("ff.linear_out"))
    o.append(String("attn.add_q_proj"))
    o.append(String("attn.add_k_proj"))
    o.append(String("attn.add_v_proj"))
    o.append(String("attn.to_add_out"))
    o.append(String("ff_context.linear_in"))
    o.append(String("ff_context.linear_out"))
    return o^


def _single_suffixes_carrier_order() -> List[String]:
    var o = List[String]()
    o.append(String("attn.to_qkv_mlp_proj"))
    o.append(String("attn.to_out"))
    return o^


def _phase_prefix(phase: String, key_prefix: String) -> String:
    if phase.byte_length() == 0:
        return key_prefix
    return phase + String(".") + key_prefix


def _lora_key_prefix(phase: String, module_name: String) -> String:
    if phase.byte_length() == 0:
        return flux2_lora_save_prefix(module_name)
    # Serenity's in-step adapter dumps are keyed by trainable parameter name
    # without the LoRAModuleWrapper "transformer." save prefix.
    return _phase_prefix(phase, module_name)


def load_flux2_lora_fused_phase(
    path: String,
    phase: String,
    num_double: Int,
    num_single: Int,
    ctx: DeviceContext,
) raises -> KleinLoraSet:
    var sharded = ShardedSafeTensors.open(path)
    var have = Dict[String, Int]()
    for ref nm in sharded.names():
        have[nm] = 1

    var dbl = List[LoraAdapter]()
    var rank = -1
    var dsuf = _double_suffixes_carrier_order()
    if len(dsuf) != DBL_SLOTS:
        raise Error("load_flux2_lora_fused: double suffix table does not match DBL_SLOTS")
    for bi in range(num_double):
        for si in range(len(dsuf)):
            var p = _lora_key_prefix(phase, flux2_double_module(bi, dsuf[si]))
            var r = _read_ab_host(sharded, p, ctx)
            if rank < 0:
                rank = r.rank
            dbl.append(_simple_adapter(r.a, r.b, r.rank, r.in_f, _alpha_host(sharded, have, p, r.rank, ctx)))

    var sgl = List[LoraAdapter]()
    var ssuf = _single_suffixes_carrier_order()
    if len(ssuf) != SGL_SLOTS:
        raise Error("load_flux2_lora_fused: single suffix table does not match SGL_SLOTS")
    for bi in range(num_single):
        for si in range(len(ssuf)):
            var p = _lora_key_prefix(phase, flux2_single_module(bi, ssuf[si]))
            var r = _read_ab_host(sharded, p, ctx)
            if rank < 0:
                rank = r.rank
            sgl.append(_simple_adapter(r.a, r.b, r.rank, r.in_f, _alpha_host(sharded, have, p, r.rank, ctx)))

    return KleinLoraSet(dbl^, sgl^, num_double, num_single, rank)


def load_flux2_lora_fused(
    path: String,
    num_double: Int,
    num_single: Int,
    ctx: DeviceContext,
) raises -> KleinLoraSet:
    return load_flux2_lora_fused_phase(path, String(), num_double, num_single, ctx)
