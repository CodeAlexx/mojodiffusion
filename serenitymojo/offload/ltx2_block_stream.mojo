# offload/ltx2_block_stream.mojo — FP8 DiT block streamer for LTX-2.3 22B.
#
# P2 (LTX2_PORT_PLAN_2026-05-28 §P2): prove the existing per-block streaming
# discipline (BlockLoader: load → use → drop) works for the FP8 distilled
# checkpoint, where blocks 4-46 store the attn/FFN weight matrices as
# float8_e4m3fn (1 byte/element) with a per-tensor F32 `weight_scale`.
#
# WHY a new module (not turbo_planned_loader): the Klein turbo loader does a
# RAW byte H2D copy (on-disk dtype == in-model dtype, both BF16). The FP8
# checkpoint stores weights in F8_E4M3 — a raw copy would land FP8 bytes on the
# GPU; the block forward needs BF16. This loader DEQUANTS FP8 → BF16 on-use
# (ops/fp8.mojo), keeping BF16/F32 tensors as a plain H2D copy. It mirrors the
# Rust production path fp8_resident.rs (RawWeight::FP8 → dequant_fp8_to_bf16
# with the per-tensor weight_scale) and ltx2_model.rs:3303-3364 (scale_map
# built from the 0-D `*.weight_scale` scalars; `input_scale` is NOT used for
# the weight dequant).
#
# Streaming model (matches BlockLoader semantics): each block is a value the
# caller owns; dropping it frees its VRAM. A small resident window is achieved
# by loading block i, running its forward, then letting it drop before loading
# block i+1 (single-resident window). The streamer instruments peak VRAM via
# DeviceContext so the memory ceiling can be gated.
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.memory import ArcPointer, bitcast
from std.gpu.host import DeviceContext
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.fp8 import fp8_e4m3_dequant_to_bf16


# A streamed block: tensor-name (prefix-stripped) -> BF16 device Tensor.
# Same Arc-wrapped Dict discipline as offload/block_loader.Block.
comptime FP8Block = Dict[String, ArcPointer[Tensor]]


def _substr(s: String, start: Int, end: Int) -> String:
    """Codepoint-indexed substring [start, end). Mojo-beta-safe (string slicing
    via `[a:b]` is unavailable in the current dialect)."""
    var out = String("")
    var i = 0
    for ch in s.codepoint_slices():
        if i >= start and i < end:
            out += String(ch)
        i += 1
    return out^


def _first_dot(s: String) -> Int:
    """Index of the first '.' in `s` (codepoint index), or -1 if absent."""
    var i = 0
    for ch in s.codepoint_slices():
        if String(ch) == ".":
            return i
        i += 1
    return -1


@always_inline
def _read_f32_scalar[
    mut: Bool, //, origin: Origin[mut=mut]
](data: Span[UInt8, origin]) -> Float32:
    """Read a little-endian F32 from the first 4 bytes of a 0-D scalar view."""
    var b0 = UInt32(Int(data[0]))
    var b1 = UInt32(Int(data[1]))
    var b2 = UInt32(Int(data[2]))
    var b3 = UInt32(Int(data[3]))
    var bits = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](bits))


struct LTX2BlockStream(Movable):
    """Streams LTX-2.3 22B FP8 DiT transformer blocks one at a time.

    Holds the mmap-backed ShardedSafeTensors and the ComfyUI key prefix
    (`model.diffusion_model.transformer_blocks.`). `load_block_bf16(i, ctx)`
    returns one block as BF16 device Tensors — FP8 weights dequantized on-use
    with their per-tensor `weight_scale`, BF16/F32 copied verbatim/upcast.

    The returned FP8Block owns its VRAM; drop it (or let it fall out of scope)
    to evict before loading the next block — this is the single-resident window
    that keeps peak VRAM bounded well under 24 GB."""

    var sharded: ShardedSafeTensors
    var prefix: String
    var n_blocks: Int

    @staticmethod
    def open(checkpoint_path: String) raises -> LTX2BlockStream:
        var st = ShardedSafeTensors.open(checkpoint_path)
        # Detect prefix + block count from the names.
        var pfx_comfy = String("model.diffusion_model.transformer_blocks.")
        var pfx_diff = String("transformer_blocks.")
        var prefix = pfx_comfy
        var found = False
        for ref nm in st.names():
            if nm.startswith(pfx_comfy):
                prefix = pfx_comfy
                found = True
                break
            if nm.startswith(pfx_diff):
                prefix = pfx_diff
                found = True
                break
        if not found:
            raise Error("LTX2BlockStream: no transformer_blocks.* keys found")
        # Count blocks: max index + 1 over keys matching prefix.{N}.
        var max_idx = -1
        for ref nm in st.names():
            if nm.startswith(prefix):
                var rest = _substr(nm, len(prefix), len(nm))
                # rest = "{idx}.{...}"
                var dot = _first_dot(rest)
                if dot > 0:
                    try:
                        var idx = Int(_substr(rest, 0, dot))
                        if idx > max_idx:
                            max_idx = idx
                    except:
                        pass
        var n = max_idx + 1
        return LTX2BlockStream(st^, prefix, n)

    def __init__(
        out self, var sharded: ShardedSafeTensors, prefix: String, n_blocks: Int
    ):
        self.sharded = sharded^
        self.prefix = prefix
        self.n_blocks = n_blocks

    def block_count(self) -> Int:
        return self.n_blocks

    def _scale_for(self, weight_key: String) raises -> Float32:
        """Return the per-tensor weight_scale for an FP8 weight, or 1.0 if the
        `{weight_key}_scale` scalar is absent. Mirrors ltx2_model.rs scale_map
        (key = full name with `_scale` suffix stripped)."""
        var scale_key = weight_key + "_scale"
        for ref nm in self.sharded.names():
            if nm == scale_key:
                var tv = self.sharded.tensor_view(scale_key)
                return _read_f32_scalar(tv.data)
        return 1.0

    def load_block_bf16(
        self, block_idx: Int, ctx: DeviceContext
    ) raises -> FP8Block:
        """Load every tensor under `prefix{block_idx}.` to the GPU as BF16.

        - F8_E4M3 weights → from_view_raw (verbatim FP8 bytes) → GPU dequant to
          BF16 with the per-tensor weight_scale (ops/fp8.mojo).
        - BF16/F32 tensors → from_view_as_bf16 (upcast F32→BF16 host-side).
        - `*_scale` / `*input_scale` scalars are NOT placed in the block (they
          are consumed only as the dequant multiplier), matching the Rust loader
          which skips `_scale`/`input_scale` keys when building block weights.

        Keys are stripped of the block prefix so the block dict is keyed by the
        canonical sub-name (e.g. `attn1.to_q.weight`)."""
        var block = FP8Block()
        var bp = self.prefix + String(block_idx) + "."
        for ref nm in self.sharded.names():
            if not nm.startswith(bp):
                continue
            # Skip scale scalars (consumed as dequant multipliers, not weights).
            if nm.endswith("_scale") or nm.endswith("input_scale"):
                continue
            var canon = _substr(nm, len(bp), len(nm))
            var tv = self.sharded.tensor_view(nm)
            if tv.dtype == STDtype.F8_E4M3:
                var scale = self._scale_for(nm)
                var raw = Tensor.from_view_raw(tv, ctx)
                var deq = fp8_e4m3_dequant_to_bf16(raw, scale, ctx)
                block[canon] = ArcPointer(deq^)
            else:
                # BF16 / F32 → BF16 device tensor.
                var t = Tensor.from_view_as_bf16(tv, ctx)
                block[canon] = ArcPointer(t^)
        return block^

    def fp8_tensor_count(self, block_idx: Int) raises -> Int:
        """How many F8_E4M3 weight tensors block `block_idx` has (diagnostic)."""
        var bp = self.prefix + String(block_idx) + "."
        var n = 0
        for ref nm in self.sharded.names():
            if nm.startswith(bp) and not nm.endswith("_scale") and not nm.endswith(
                "input_scale"
            ):
                var tv = self.sharded.tensor_view(nm)
                if tv.dtype == STDtype.F8_E4M3:
                    n += 1
        return n


def drop_block(var block: FP8Block):
    """Explicitly evict a streamed block, freeing its VRAM. Equivalent to
    letting `block` fall out of scope. Call as `drop_block(block^)`."""
    _ = block^
