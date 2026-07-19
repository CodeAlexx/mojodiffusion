# weights.mojo — Z-Image transformer safetensors weight store.
#
# Loads the Z-Image NextDiT transformer weights (FROZEN base, BF16 storage) into
# a name→Tensor map, mirroring the reference loader in
#   mojodiffusion/serenitymojo/models/dit/zimage_dit.mojo::NextDiT.load (:200-213)
# but as a standalone reusable store the Serenity block fwd/bwd consume. We do
# NOT import serenitymojo.models — only the io loaders + Tensor.from_view (which
# preserves the on-disk dtype; Z-Image ships BF16).
#
# Key naming (diffusers ZImageTransformer2DModel state dict; the SAME keys the
# reference _block reads). For one MAIN block `layers.<i>`:
#   layers.<i>.adaLN_modulation.0.{weight,bias}      AdaLN: Linear -> 4*dim
#   layers.<i>.attention_norm1.weight                RMSNorm  (pre-attn)
#   layers.<i>.attention_norm2.weight                RMSNorm  (post-attn)
#   layers.<i>.ffn_norm1.weight                      RMSNorm  (pre-mlp)
#   layers.<i>.ffn_norm2.weight                      RMSNorm  (post-mlp)
#   layers.<i>.attention.to_q.weight   [dim,dim]     (no bias)
#   layers.<i>.attention.to_k.weight   [dim,dim]
#   layers.<i>.attention.to_v.weight   [dim,dim]
#   layers.<i>.attention.norm_q.weight [Dh]          per-head RMSNorm
#   layers.<i>.attention.norm_k.weight [Dh]
#   layers.<i>.attention.to_out.0.weight [dim,dim]
#   layers.<i>.feed_forward.w1.weight  [ff,dim]
#   layers.<i>.feed_forward.w3.weight  [ff,dim]
#   layers.<i>.feed_forward.w2.weight  [dim,ff]
#
# All base weights are FROZEN (LoRA training): they are loaded but never tracked
# on a tape and never receive gradients (only their LoRA d_A/d_B do).

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors


comptime TArc = ArcPointer[Tensor]


# A frozen weight store: name → device Tensor (BF16). Tensor is move-only, so we
# box each in ArcPointer and key by name through an Int index (List/Dict can't
# hold a bare Tensor).
struct ZImageWeights(Movable):
    var weights: List[TArc]
    var name_to_idx: Dict[String, Int]

    def __init__(out self, var weights: List[TArc], var name_to_idx: Dict[String, Int]):
        self.weights = weights^
        self.name_to_idx = name_to_idx^

    # Load every transformer tensor from a sharded safetensors dir (H2D copy,
    # dtype preserved). Mirrors NextDiT.load.
    @staticmethod
    def load(dir: String, ctx: DeviceContext) raises -> ZImageWeights:
        var sharded = ShardedSafeTensors.open(dir)
        var weights = List[TArc]()
        var name_to_idx = Dict[String, Int]()
        for ref nm in sharded.names():
            var tv = sharded.tensor_view(nm)
            var t = Tensor.from_view(tv, ctx)
            var idx = len(weights)
            weights.append(ArcPointer(t^))
            name_to_idx[nm] = idx
        return ZImageWeights(weights^, name_to_idx^)

    # Reference to a weight by full name (raises if missing).
    def get(self, name: String) raises -> ref [self.weights] Tensor:
        if name not in self.name_to_idx:
            raise Error(String("ZImageWeights: missing weight: ") + name)
        var idx = self.name_to_idx[name]
        return self.weights[idx][]

    def has(self, name: String) -> Bool:
        return name in self.name_to_idx

    def count(self) -> Int:
        return len(self.weights)


# ── per-block key helpers (block fwd/bwd use these to fetch frozen weights) ────
def block_prefix(block_idx: Int) -> String:
    return String("layers.") + String(block_idx)


# Bundle of the frozen base-weight names a single MAIN block reads. The fwd/bwd
# fetch each via ZImageWeights.get(prefix + suffix).
@fieldwise_init
struct ZImageBlockKeys(Copyable, Movable):
    var prefix: String

    @staticmethod
    def for_block(block_idx: Int) -> ZImageBlockKeys:
        return ZImageBlockKeys(block_prefix(block_idx))

    def adaln_w(self) -> String:    return self.prefix + String(".adaLN_modulation.0.weight")
    def adaln_b(self) -> String:    return self.prefix + String(".adaLN_modulation.0.bias")
    def attn_norm1(self) -> String: return self.prefix + String(".attention_norm1.weight")
    def attn_norm2(self) -> String: return self.prefix + String(".attention_norm2.weight")
    def ffn_norm1(self) -> String:  return self.prefix + String(".ffn_norm1.weight")
    def ffn_norm2(self) -> String:  return self.prefix + String(".ffn_norm2.weight")
    def to_q(self) -> String:       return self.prefix + String(".attention.to_q.weight")
    def to_k(self) -> String:       return self.prefix + String(".attention.to_k.weight")
    def to_v(self) -> String:       return self.prefix + String(".attention.to_v.weight")
    def norm_q(self) -> String:     return self.prefix + String(".attention.norm_q.weight")
    def norm_k(self) -> String:     return self.prefix + String(".attention.norm_k.weight")
    def to_out(self) -> String:     return self.prefix + String(".attention.to_out.0.weight")
    def ff_w1(self) -> String:      return self.prefix + String(".feed_forward.w1.weight")
    def ff_w3(self) -> String:      return self.prefix + String(".feed_forward.w3.weight")
    def ff_w2(self) -> String:      return self.prefix + String(".feed_forward.w2.weight")


# ══════════════════════════════════════════════════════════════════════════════
# LoRA reload — the inverse of modelSaver/zImage/ZImageLoRASaver.save_zimage_lora.
#
# Mirrors Serenity ZImageLoRASetup.setup_model (ZImageLoRASetup.py:61-63):
#   if model.lora_state_dict:
#       model.transformer_lora.load_state_dict(model.lora_state_dict)
# i.e. a previously-saved adapter checkpoint is read back so training can RESUME
# from it (vs the default A~randn / B=0 cold start). We read the PEFT key pair
#   "<prefix>.lora_A.weight" → A [rank, in]
#   "<prefix>.lora_B.weight" → B [out, rank]
# for each (block, slot) prefix (and the optional "<prefix>.alpha" scalar) from a
# single-file or sharded safetensors and return a flat (A,B,alpha) bundle the
# block fwd/bwd consume.  Dtype preserved (BF16) via Tensor.from_view.
#
# This is a DATA reload; the setup wires it into a ZImageLoraSet (the model unit's
# build path takes A/B tensors directly — see cross-slice notes). It does NOT
# import the model struct to avoid an import cycle (loader ← model ← loader).
struct ZImageLoraReload(Movable):
    var a: List[TArc]        # [n_layers*7] each [rank, in]
    var b: List[TArc]        # [n_layers*7] each [out, rank]
    var alpha: List[Float32] # [n_layers*7] per-adapter alpha (rank if absent)
    var rank: Int

    def __init__(out self, var a: List[TArc], var b: List[TArc], var alpha: List[Float32], rank: Int):
        self.a = a^
        self.b = b^
        self.alpha = alpha^
        self.rank = rank


# Read the saved LoRA adapters from `path` (file or dir). `prefixes` is the
# block-major/slot-minor module list from ZImageLoRASetup.zimage_lora_target_prefixes.
def load_zimage_lora(
    path: String,
    prefixes: List[String],
    ctx: DeviceContext,
) raises -> ZImageLoraReload:
    var sharded = ShardedSafeTensors.open(path)
    # Build a name set once (ShardedSafeTensors has no membership predicate; we
    # gate the optional alpha read against this set).
    var have = Dict[String, Int]()
    for ref nm in sharded.names():
        have[nm] = 1
    var a = List[TArc]()
    var b = List[TArc]()
    var alpha = List[Float32]()
    var rank = -1
    for i in range(len(prefixes)):
        var pre = prefixes[i]
        var ak = pre + String(".lora_A.weight")
        var bk = pre + String(".lora_B.weight")
        var at = Tensor.from_view(sharded.tensor_view(ak), ctx)
        var bt = Tensor.from_view(sharded.tensor_view(bk), ctx)
        if rank < 0:
            rank = at.shape()[0]            # A is [rank, in]
        # alpha: read "<prefix>.alpha" (1-elem) if present; else default to rank
        # (PEFT/diffusers identity-scale convention when alpha is omitted).
        var ah = pre + String(".alpha")
        var al = Float32(rank)
        if ah in have:
            var alt = Tensor.from_view(sharded.tensor_view(ah), ctx)
            al = alt.to_host(ctx)[0]
        a.append(TArc(at^))
        b.append(TArc(bt^))
        alpha.append(al)
    return ZImageLoraReload(a^, b^, alpha^, rank)
