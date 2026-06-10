# offload/ltx2_block_stream.mojo — FP8 DiT block streamer for LTX-2.3 22B.
#
# P2 (LTX2_PORT_PLAN_2026-05-28 P2): prove the existing per-block streaming
# discipline (BlockLoader: load → use → drop) works for the FP8 distilled
# checkpoint, where blocks 4-46 store the attn/FFN weight matrices as
# float8_e4m3fn (1 byte/element) with a per-tensor F32 scale.
#
# WHY a new module (not turbo_planned_loader): the Klein turbo loader does a
# RAW byte H2D copy (on-disk dtype == in-model dtype, both BF16). The FP8
# checkpoint stores weights in F8_E4M3 — a raw copy would land FP8 bytes on the
# GPU; the block forward needs BF16. This loader DEQUANTS FP8 → BF16 on-use
# (ops/fp8.mojo), keeping BF16/F32 tensors as a plain H2D copy. It mirrors the
# Rust production path fp8_resident.rs (RawWeight::FP8 → dequant_fp8_to_bf16
# with the per-tensor scale) and Flame Core's two scale naming conventions:
# `weight_key + "_scale"` and `weight_key.strip_suffix(".weight") + ".scale_weight"`.
# `input_scale` / `scale_input` are NOT used for weight dequant.
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
    var resident_enabled: Bool
    var resident_loaded: List[Bool]
    var resident_blocks: List[FP8Block]
    var resident_scales: List[Dict[String, Float32]]
    var resident_bytes_: Int
    var scale_cache: Dict[String, Float32]

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
        var scales = Dict[String, Float32]()
        for ref nm in st.names():
            if nm.endswith("_scale"):
                var weight_key = _substr(nm, 0, len(nm) - len(String("_scale")))
                var tv = st.tensor_view(nm)
                scales[weight_key] = _read_f32_scalar(tv.data)
            elif nm.endswith(".scale_weight"):
                var base_key = _substr(
                    nm, 0, len(nm) - len(String(".scale_weight"))
                )
                var tv = st.tensor_view(nm)
                scales[base_key + ".weight"] = _read_f32_scalar(tv.data)
        return LTX2BlockStream(st^, prefix, n, scales^)

    def __init__(
        out self,
        var sharded: ShardedSafeTensors,
        prefix: String,
        n_blocks: Int,
        var scale_cache: Dict[String, Float32],
    ):
        self.sharded = sharded^
        self.prefix = prefix
        self.n_blocks = n_blocks
        self.resident_enabled = False
        self.resident_loaded = List[Bool]()
        self.resident_blocks = List[FP8Block]()
        self.resident_scales = List[Dict[String, Float32]]()
        self.resident_bytes_ = 0
        self.scale_cache = scale_cache^

    def block_count(self) -> Int:
        return self.n_blocks

    def resident_bytes(self) -> Int:
        return self.resident_bytes_

    def _scale_for(self, weight_key: String) raises -> Float32:
        """Return the per-tensor scale for an FP8 weight, or 1.0 if absent.

        Mirrors Flame Core's safetensors/offload loaders:
        - LTX-style sidecar: `{weight_key}_scale`
        - Comfy-scaled sidecar: `{base}.scale_weight` for `{base}.weight`
        """
        if weight_key in self.scale_cache:
            return self.scale_cache[weight_key]
        return 1.0

    def load_block_bf16(
        self, block_idx: Int, ctx: DeviceContext
    ) raises -> FP8Block:
        """Load every tensor under `prefix{block_idx}.` to the GPU as BF16.

        - F8_E4M3 weights → from_view_raw (verbatim FP8 bytes) → GPU dequant to
          BF16 with the per-tensor scale (ops/fp8.mojo).
        - BF16/F32 tensors → from_view_as_bf16 (upcast F32→BF16 host-side).
        - `*_scale` / `*input_scale` scalars are NOT placed in the block (they
          are consumed only as the dequant multiplier), matching the Rust loader
          which skips `_scale`/`input_scale` keys when building block weights.

        Keys are stripped of the block prefix so the block dict is keyed by the
        canonical sub-name (e.g. `attn1.to_q.weight`)."""
        if (
            self.resident_enabled
            and block_idx >= 0
            and block_idx < len(self.resident_loaded)
            and self.resident_loaded[block_idx]
        ):
            return self._load_resident_block_bf16(block_idx, ctx)

        var block = FP8Block()
        var bp = self.prefix + String(block_idx) + "."
        for ref nm in self.sharded.names():
            if not nm.startswith(bp):
                continue
            # Skip scale scalars (consumed as dequant multipliers, not weights).
            if (
                nm.endswith("_scale")
                or nm.endswith("input_scale")
                or nm.endswith(".scale_weight")
                or nm.endswith(".scale_input")
            ):
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

    def _load_resident_block_bf16(
        self, block_idx: Int, ctx: DeviceContext
    ) raises -> FP8Block:
        """Materialize one BF16 block from GPU-resident raw storage.

        FP8 tensors are already on the GPU as raw bytes, so this avoids the
        repeated host→device copy in the streamed path. Non-FP8 tensors are
        stored resident as BF16 and reused by Arc; the caller materializes a
        mutable BF16 block and kernels do any F32 accumulation internally."""
        var block = FP8Block()
        ref raw_block = self.resident_blocks[block_idx]
        ref scale_block = self.resident_scales[block_idx]
        for ref e in raw_block.items():
            if e.value[].dtype() == STDtype.F8_E4M3:
                var scale = Float32(1.0)
                if e.key in scale_block:
                    scale = scale_block[e.key]
                var deq = fp8_e4m3_dequant_to_bf16(e.value[], scale, ctx)
                block[e.key] = ArcPointer(deq^)
            else:
                block[e.key] = e.value
        return block^

    def enable_fp8_resident_range(
        mut self, first_block: Int, last_block: Int, ctx: DeviceContext
    ) raises:
        """Preload raw FP8/storage tensors for `first_block..last_block`.

        This is the production-speed path for repeated denoise evals: FP8 bytes
        stay in VRAM, while each eval still materializes only the current block
        as BF16/F32 for the existing forward. The range form lets callers keep
        BF16 boundary blocks out of the resident set on 24 GB cards."""
        if first_block < 0 or last_block >= self.n_blocks or first_block > last_block:
            raise Error("enable_fp8_resident_range: invalid block range")
        if len(self.resident_blocks) != 0:
            raise Error("enable_fp8_resident_range: resident store already initialized")

        self.resident_enabled = True
        self.resident_bytes_ = 0
        for block_idx in range(self.n_blocks):
            var block = FP8Block()
            var scales = Dict[String, Float32]()
            var loaded = block_idx >= first_block and block_idx <= last_block
            if loaded:
                var bp = self.prefix + String(block_idx) + "."
                for ref nm in self.sharded.names():
                    if not nm.startswith(bp):
                        continue
                    if (
                        nm.endswith("_scale")
                        or nm.endswith("input_scale")
                        or nm.endswith(".scale_weight")
                        or nm.endswith(".scale_input")
                    ):
                        continue
                    var canon = _substr(nm, len(bp), len(nm))
                    var tv = self.sharded.tensor_view(nm)
                    if tv.dtype == STDtype.F8_E4M3:
                        var raw = Tensor.from_view_raw(tv, ctx)
                        self.resident_bytes_ += raw.nbytes()
                        scales[canon] = self._scale_for(nm)
                        block[canon] = ArcPointer(raw^)
                    else:
                        var t = Tensor.from_view_as_bf16(tv, ctx)
                        self.resident_bytes_ += t.nbytes()
                        block[canon] = ArcPointer(t^)
            self.resident_blocks.append(block^)
            self.resident_scales.append(scales^)
            self.resident_loaded.append(loaded)

    def load_block_fp8_resident(
        self, block_idx: Int, mut scales_out: FP8Block, ctx: DeviceContext
    ) raises -> FP8Block:
        """RESIDENT-FP8 load: one H2D per tensor, ONCE — the caller keeps the
        returned block alive for every subsequent eval (SPEED_CONTRACT clause 5:
        no re-upload churn; clause 1: zero per-step host stalls in the loop).

        - 2-D F8_E4M3 matmul weights upload as RAW fp8 bytes (1 B/param, NO
          dequant). For each, a per-output-row F32 [N] scale tensor is built
          once from the per-tensor `weight_scale` (_scale_for) and stored in
          `scales_out` under the SAME prefix-stripped key — the exact layout
          `ops/fp8_gemm.linear_fp8(x, w_fp8, scale_f32_per_row, ...)` consumes.
        - Any other tensor (norms/biases/scale_shift tables, and a non-2D fp8
          tensor should one ever appear) loads BF16 exactly as load_block_bf16.

        Mirrors the proven Ideogram4 resident-fp8 pattern
        (models/dit/ideogram4_resident.mojo): raw fp8 + scale alongside,
        dtype-dispatched at the linear call site."""
        var block = FP8Block()
        var bp = self.prefix + String(block_idx) + "."
        for ref nm in self.sharded.names():
            if not nm.startswith(bp):
                continue
            # Skip scale scalars (consumed as the per-row scale fill value).
            if (
                nm.endswith("_scale")
                or nm.endswith("input_scale")
                or nm.endswith(".scale_weight")
                or nm.endswith(".scale_input")
            ):
                continue
            var canon = _substr(nm, len(bp), len(nm))
            var tv = self.sharded.tensor_view(nm)
            if tv.dtype == STDtype.F8_E4M3 and len(tv.shape) == 2:
                var raw = Tensor.from_view_raw(tv, ctx)
                var n_rows = raw.shape()[0]
                var s = self._scale_for(nm)
                var srow = List[Float32]()
                for _ in range(n_rows):
                    srow.append(s)
                var srow_sh = List[Int]()
                srow_sh.append(n_rows)
                var st = Tensor.from_host(srow, srow_sh^, STDtype.F32, ctx)
                scales_out[canon] = ArcPointer(st^)
                block[canon] = ArcPointer(raw^)
            elif tv.dtype == STDtype.F8_E4M3:
                # Non-2D fp8 (not present in the LTX-2.3 checkpoint): no GEMM
                # consumes it, so dequant to BF16 like the streamed path.
                var scale = self._scale_for(nm)
                var raw = Tensor.from_view_raw(tv, ctx)
                var deq = fp8_e4m3_dequant_to_bf16(raw, scale, ctx)
                block[canon] = ArcPointer(deq^)
            else:
                block[canon] = ArcPointer(Tensor.from_view_as_bf16(tv, ctx))
        return block^

    def fp8_tensor_count(self, block_idx: Int) raises -> Int:
        """How many F8_E4M3 weight tensors block `block_idx` has (diagnostic)."""
        var bp = self.prefix + String(block_idx) + "."
        var n = 0
        for ref nm in self.sharded.names():
            if (
                nm.startswith(bp)
                and not nm.endswith("_scale")
                and not nm.endswith("input_scale")
                and not nm.endswith(".scale_weight")
                and not nm.endswith(".scale_input")
            ):
                var tv = self.sharded.tensor_view(nm)
                if tv.dtype == STDtype.F8_E4M3:
                    n += 1
        return n


def drop_block(var block: FP8Block):
    """Explicitly evict a streamed block, freeing its VRAM. Equivalent to
    letting `block` fall out of scope. Call as `drop_block(block^)`."""
    _ = block^
