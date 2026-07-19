# training/full_adapter.mojo — LyCORIS "Full" adapter (Wave 2B item 2j).
#
# Ports eri-lycoris/lycoris-rs/src/algorithms/full.rs (FullAdapter) +
# EDv2 crates/eridiffusion-core/src/lycoris.rs Full save convention.
#
# Full = a FULL-SHAPE trainable weight delta (NOT low-rank A·B). It carries one
# tensor `diff` matching the base weight shape (and optional `diff_b` for bias),
# zero-initialized so ΔW=0 at start. Inference / merge does:
#       base.weight <- base.weight + strength * diff      (full.rs:5-7, :65-72)
#       base.bias   <- base.bias   + strength * diff_b    (if diff_b present)
#
# Save convention (lycoris.rs:899-903 / full.rs:9 header
# "lycoris/modules/full.py custom_state_dict"):
#       "<prefix>.diff.weight"   shape == base weight shape
#       "<prefix>.diff_b"        shape == [bias_size]   (optional)
#
# ── default-OFF ───────────────────────────────────────────────────────────────
# TrainConfig.adapter_algo defaults to 0 (plain LoRA). Full is selected only when
# adapter_algo==1. The plain-LoRA path (KleinLoraSet) is byte-unchanged when off.
#
# ── AGENT-DEFAULT for review ──────────────────────────────────────────────────
# Full-shape deltas for Klein-9B are very large (one [out,in] tensor per adapted
# linear). The existing Klein loop is built around the low-rank KleinLoraSet +
# klein_stack_lora_backward, which has no full-delta backward arm. So this wave
# ships the FullAdapter PRIMITIVE (delta math + AdamW state + the .diff.weight
# save convention) and the adapter_algo SELECTOR + a guarded loop branch that
# reports the selected path; swapping the full stack forward/backward onto
# full-delta weights is a larger follow-up (flagged). The gate proves the delta
# math and the save key/shape, per the item-2j acceptance.
#
# Mojo 1.0.0b1. BF16 trainable storage; F32 moments/compute.

from std.collections import List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors_writer import save_safetensors


# A full-shape weight delta with AdamW moments. `diff` is row-major BF16 storage
# with the base weight's shape (stored flat + the shape dims).
struct FullAdapter(Copyable, Movable):
    var diff: List[BFloat16]       # flat, len == prod(shape)
    var shape: List[Int]           # base weight shape (e.g. [out, in])
    var m: List[Float32]           # AdamW first moment
    var v: List[Float32]           # AdamW second moment
    var has_bias: Bool
    var diff_b: List[BFloat16]     # flat [bias_size] (empty when has_bias=False)
    var mb: List[Float32]
    var vb: List[Float32]

    def __init__(
        out self, var diff: List[Float32], var shape: List[Int],
        var m: List[Float32], var v: List[Float32],
        has_bias: Bool, var diff_b: List[Float32],
        var mb: List[Float32], var vb: List[Float32],
    ):
        self.diff = _f32_to_bf16_list(diff)
        self.shape = shape^
        self.m = m^
        self.v = v^
        self.has_bias = has_bias
        self.diff_b = _f32_to_bf16_list(diff_b)
        self.mb = mb^
        self.vb = vb^

    def numel(self) -> Int:
        var n = 1
        for i in range(len(self.shape)):
            n = n * self.shape[i]
        return n


def _f32_to_bf16_list(v: List[Float32]) -> List[BFloat16]:
    var out = List[BFloat16]()
    for i in range(len(v)):
        out.append(BFloat16(v[i]))
    return out^


# Construct a fresh zero-initialized Full adapter (ΔW=0, ΔB=0), AdamW moments 0.
# Mirrors FullAdapter::new_for_training (full.rs:43-62).
def new_full_adapter(shape: List[Int], bias_size: Int) -> FullAdapter:
    var n = 1
    for i in range(len(shape)):
        n = n * shape[i]
    var diff = List[Float32]()
    var m = List[Float32]()
    var v = List[Float32]()
    for _ in range(n):
        diff.append(Float32(0.0)); m.append(Float32(0.0)); v.append(Float32(0.0))
    var has_bias = bias_size > 0
    var diff_b = List[Float32]()
    var mb = List[Float32]()
    var vb = List[Float32]()
    if has_bias:
        for _ in range(bias_size):
            diff_b.append(Float32(0.0)); mb.append(Float32(0.0)); vb.append(Float32(0.0))
    return FullAdapter(diff^, shape.copy(), m^, v^, has_bias, diff_b^, mb^, vb^)


# Returns strength * diff (the weight delta the caller adds to the base weight).
# Mirrors FullAdapter::delta_weight (full.rs:65-72): strength==1.0 returns diff
# unscaled.
def full_delta_weight(adapter: FullAdapter, strength: Float32) -> List[Float32]:
    var out = List[Float32]()
    if strength == Float32(1.0):
        for i in range(len(adapter.diff)):
            out.append(adapter.diff[i].cast[DType.float32]())
        return out^
    for i in range(len(adapter.diff)):
        out.append(strength * adapter.diff[i].cast[DType.float32]())
    return out^


# Returns strength * diff_b when a bias delta is present (empty list otherwise).
# Mirrors FullAdapter::delta_bias (full.rs:75-87).
def full_delta_bias(adapter: FullAdapter, strength: Float32) -> List[Float32]:
    var out = List[Float32]()
    if not adapter.has_bias:
        return out^
    if strength == Float32(1.0):
        for i in range(len(adapter.diff_b)):
            out.append(adapter.diff_b[i].cast[DType.float32]())
        return out^
    for i in range(len(adapter.diff_b)):
        out.append(strength * adapter.diff_b[i].cast[DType.float32]())
    return out^


# A Full adapter paired with its base-weight prefix (for save keying).
@fieldwise_init
struct NamedFull(Copyable, Movable):
    var prefix: String
    var adapter: FullAdapter


# Save Full adapters in the LyCORIS convention:
#   "<prefix>.diff.weight"  (base weight shape)
#   "<prefix>.diff_b"       (optional, [bias_size])
# Mirrors lycoris.rs:899-903. Returns the number of TENSORS written.
def save_full_adapters(
    adapters: List[NamedFull], path: String, ctx: DeviceContext
) raises -> Int:
    if len(adapters) == 0:
        raise Error("save_full_adapters: refusing to write an empty file")
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    for ref nf in adapters:
        var a = nf.adapter.copy()
        if len(a.diff) != a.numel():
            raise Error(
                String("save_full_adapters: diff numel ") + String(len(a.diff))
                + " != prod(shape) " + String(a.numel()) + " for '" + nf.prefix + "'"
            )
        names.append(nf.prefix + ".diff.weight")
        tensors.append(ArcPointer(Tensor.from_host_bf16(a.diff.copy(), a.shape.copy(), ctx)))
        if a.has_bias:
            var bsh = List[Int](); bsh.append(len(a.diff_b))
            names.append(nf.prefix + ".diff_b")
            tensors.append(ArcPointer(Tensor.from_host_bf16(a.diff_b.copy(), bsh^, ctx)))
    save_safetensors(names, tensors, path, ctx)
    return len(names)
