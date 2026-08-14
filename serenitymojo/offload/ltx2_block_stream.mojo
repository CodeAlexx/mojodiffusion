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
from std.collections import Optional
from max.gpu.host import DeviceContext, DeviceBuffer
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.fp8 import (
    fp8_e4m3_dequant_to_bf16,
    fp8_e4m3_dequant_to_bf16_no_sync,
)
# INT4 backing (MJ-1096): when a slab is attached, load_block_bf16 reconstructs
# each block bf16 from the svdint4 slab instead of dequanting fp8 — same block
# dict, so from_fp8_block + the whole gen forward are unchanged.
from serenitymojo.ops.svdquant import (
    SvdquantLinearA, svdquant_reconstruct_weight, svdquant_reconstruct_weight_raw,
)
from serenitymojo.offload.plan import BlockKind, BlockPlan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader


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


def _align_up(n: Int, alignment: Int) -> Int:
    """Round `n` up to a multiple of `alignment` (StepSlab's 256-byte idiom —
    cuBLAS kernel selection is pointer-alignment sensitive, so each staged
    tensor starts at a 256-aligned offset)."""
    if alignment <= 1:
        return n
    var rem = n % alignment
    if rem == 0:
        return n
    return n + (alignment - rem)


@fieldwise_init
struct LTX2FixedEntry(Copyable, Movable):
    """One tensor's fixed placement inside an `LTX2FixedStage` buffer: the
    prefix-stripped canonical name, its 256-aligned byte offset, its BF16 byte
    length, and its output shape. Computed ONCE from the reference block and
    reused for every block (all 48 video blocks are shape-identical)."""
    var name: String
    var offset: Int
    var nbytes: Int
    var shape: List[Int]


struct LTX2FixedStage(Movable):
    """ONE fixed device buffer that stages a single LTX-2.3 block's BF16 weights
    at STABLE device addresses so a captured CUDA graph can be replayed against
    them (contract C8, AUTOGRAD_V2_MOJO_DESIGN.md: replay re-runs recorded
    kernels on recorded pointers, so block weights must land at the same address
    every visit).

    The per-tensor `layout` is computed once from a reference block by
    `LTX2BlockStream.build_fixed_stage` and reused for every block.
    `load_block_bf16_fixed` dequants/upcasts each tensor into a temp EXACTLY as
    the streamed `load_block_bf16` path does, `enqueue_copy`s it into `buf` at
    the layout offset, and returns a block whose Tensors are `create_sub_buffer`
    VIEWS into `buf` — the scratch_ring idiom (the stage owns the memory; views
    never outlive it).

    C13 (gate-don't-delete): opt-in only. Nothing routes here unless a caller
    builds a stage and calls `load_block_bf16_fixed`; the default streaming path
    (`load_block_bf16`, fresh per-tensor allocs) is untouched and stays the
    default."""

    var buf: DeviceBuffer[DType.uint8]
    var layout: List[LTX2FixedEntry]
    var total_bytes: Int
    var alignment: Int
    var addrs: List[Int]        # device address of each entry's sub-view (recorded on 1st visit)
    var addr_recorded: Bool

    def __init__(
        out self,
        var buf: DeviceBuffer[DType.uint8],
        var layout: List[LTX2FixedEntry],
        total_bytes: Int,
        alignment: Int,
        var addrs: List[Int],
        addr_recorded: Bool,
    ):
        self.buf = buf^
        self.layout = layout^
        self.total_bytes = total_bytes
        self.alignment = alignment
        self.addrs = addrs^
        self.addr_recorded = addr_recorded

    def n_entries(self) -> Int:
        return len(self.layout)

    def slot_for(self, name: String) raises -> Int:
        """Index of `name` in the layout, or FAIL-LOUD if this block has a
        tensor the reference block did not (blocks not shape-identical)."""
        for i in range(len(self.layout)):
            if self.layout[i].name == name:
                return i
        raise Error(
            String("LTX2FixedStage: tensor '") + name
            + "' not in the reference-block layout (blocks not shape-identical?)"
        )


struct LTX2StandaloneStage(Movable):
    """CAPTURE-MODE weight stage: ONE STANDALONE device buffer PER tensor (each
    holds a single weight at offset 0), NOT a packed buffer with sub-buffer views.

    WHY (MEASURED 2026-07-17, ltx2_capture_external_refill_constraint): memory
    REFILLED FROM OUTSIDE a captured graph between replays must be a STANDALONE
    buffer (or an offset-0 view) to be re-read by replay. A non-zero-offset
    sub-buffer view into a shared buffer is NOT re-read — the packed
    `LTX2FixedStage` refill propagates to the memory (to_host confirms) but the
    replayed kernels still read the OLD block's weights. Intra-graph slab
    intermediates are unaffected (they are written by the graph each replay —
    proven by the 834cb0e capture smoke over the slab path). So the per-block
    weights the captured graph reads must live in their own buffers, refilled in
    place per block.

    Sibling of `LTX2FixedStage` (C13: the packed stage + its byte gate stay —
    they are correct for the non-capture streamed-loader test). Same layout, same
    dequant/copy; only the buffer topology differs (N standalone vs 1 packed).
    Total VRAM is identical (~737 MiB); 86 allocations, once, at init."""

    var bufs: List[DeviceBuffer[DType.uint8]]   # one standalone buffer per tensor
    var layout: List[LTX2FixedEntry]            # name/nbytes/shape (offset field unused, == 0)
    var addrs: List[Int]                        # device address per buffer (recorded 1st visit)
    var addr_recorded: Bool

    def __init__(
        out self,
        var bufs: List[DeviceBuffer[DType.uint8]],
        var layout: List[LTX2FixedEntry],
        var addrs: List[Int],
        addr_recorded: Bool,
    ):
        self.bufs = bufs^
        self.layout = layout^
        self.addrs = addrs^
        self.addr_recorded = addr_recorded

    def n_entries(self) -> Int:
        return len(self.layout)

    def total_bytes(self) -> Int:
        var t = 0
        for i in range(len(self.layout)):
            t += self.layout[i].nbytes
        return t

    def slot_for(self, name: String) raises -> Int:
        for i in range(len(self.layout)):
            if self.layout[i].name == name:
                return i
        raise Error(
            String("LTX2StandaloneStage: tensor '") + name
            + "' not in the reference-block layout (blocks not shape-identical?)"
        )


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
    var int4_slab: Optional[ShardedSafeTensors]   # attached svdint4 slab (else fp8)
    var checkpoint_path: String
    # Product sampling uses the shared complete host-resident loader. Keeping it
    # behind the existing LTX2 block-source surface avoids a second denoiser
    # orchestration path while making mmap fallback impossible after step 0.
    var product_host_loaders: List[ArcPointer[TurboPlannedLoader]]
    var product_host_active: Bool
    # Explicit low-host-RAM product mode. The default remains the complete
    # pinned host store; callers must opt in before denoise. In checkpoint
    # streaming mode each synchronized block upload is followed by
    # MADV_DONTNEED/POSIX_FADV_DONTNEED so clean 22B checkpoint pages do not
    # accumulate in the product cgroup and trigger systemd-oomd.
    var product_checkpoint_streaming: Bool
    var product_host_bytes_: Int

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
                var rest = _substr(nm, prefix.byte_length(), nm.byte_length())
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
                var weight_key = _substr(nm, 0, nm.byte_length() - String("_scale").byte_length())
                var tv = st.tensor_view(nm)
                scales[weight_key] = _read_f32_scalar(tv.data)
            elif nm.endswith(".scale_weight"):
                var base_key = _substr(
                    nm, 0, nm.byte_length() - String(".scale_weight").byte_length()
                )
                var tv = st.tensor_view(nm)
                scales[base_key + ".weight"] = _read_f32_scalar(tv.data)
        return LTX2BlockStream(st^, prefix, n, scales^, checkpoint_path)

    def __init__(
        out self,
        var sharded: ShardedSafeTensors,
        prefix: String,
        n_blocks: Int,
        var scale_cache: Dict[String, Float32],
        checkpoint_path: String,
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
        self.int4_slab = None
        self.checkpoint_path = checkpoint_path
        self.product_host_loaders = List[ArcPointer[TurboPlannedLoader]]()
        self.product_host_active = False
        self.product_checkpoint_streaming = False
        self.product_host_bytes_ = 0

    def block_count(self) -> Int:
        return self.n_blocks

    def attach_int4(mut self, slab_path: String) raises:
        """Back this stream with an svdint4 slab. load_block_bf16 then
        reconstructs each block bf16 from the slab (dequant4 + low-rank) instead
        of the fp8 path. The slab uses the same block key prefix."""
        self.int4_slab = Optional[ShardedSafeTensors](ShardedSafeTensors.open(slab_path))

    def has_int4(self) -> Bool:
        return Bool(self.int4_slab)

    def resident_bytes(self) -> Int:
        return self.resident_bytes_

    def product_host_bytes(self) -> Int:
        return self.product_host_bytes_

    def resident_storage_bytes_for_block(self, block_idx: Int) raises -> Int:
        """Return the device bytes used by resident storage for one block.

        FP8 tensors remain raw FP8 in the resident store. Every other storage
        dtype is materialized as BF16, matching `enable_fp8_resident_range`.
        This metadata-only query lets the product pipeline size a partial GPU
        cache from the free VRAM measured after its exact-shape warmup.
        """
        if block_idx < 0 or block_idx >= self.n_blocks:
            raise Error("resident_storage_bytes_for_block: invalid block index")
        var total = 0
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
            var tv = self.sharded.tensor_view(nm)
            if tv.dtype == STDtype.F8_E4M3:
                total += tv.nbytes()
            else:
                var numel = 1
                for dim_idx in range(len(tv.shape)):
                    numel *= tv.shape[dim_idx]
                total += numel * STDtype.BF16.byte_size()
        return total

    def release_checkpoint_pages(mut self):
        """Drop clean checkpoint pages after an explicit preload boundary."""
        self.sharded.release_to_os()

    def enable_product_checkpoint_streaming(mut self) raises:
        """Use the existing byte-identical per-block checkpoint loader.

        This is an explicit product choice for hosts that cannot pin the full
        BF16 DiT in RAM. It changes only the block source: weights still land
        as BF16 GPU tensors and the same LTX2 block forward executes.
        """
        if self.int4_slab:
            raise Error(
                "LTX2 checkpoint streaming is not valid for the int4 resident slab"
            )
        if len(self.product_host_loaders) != 0:
            raise Error(
                "LTX2 checkpoint streaming must be selected before resident preload"
            )
        self.product_checkpoint_streaming = True

    def enable_product_memory_resident(mut self, ctx: DeviceContext) raises -> Int:
        """Copy every FP8/BF16 block byte into pinned RAM before denoise.

        The shared TurboPlannedLoader supplies double-buffered copy-stream H2D
        and FP8 dequant. Sampling is admitted only after all 48 plan entries are
        present in RAM (or, for int4, already present in the complete GPU store).
        No product call can silently fall back to the checkpoint mapping.
        """
        if self.int4_slab:
            if len(self.resident_loaded) != self.n_blocks:
                raise Error(
                    "LTX2 product refuses int4 sampling without a complete "
                    "device-resident block store"
                )
            for i in range(self.n_blocks):
                if not self.resident_loaded[i]:
                    raise Error(
                        String("LTX2 product int4 block is not resident: ")
                        + String(i)
                    )
            return self.n_blocks
        if len(self.product_host_loaders) != 0:
            self.product_host_loaders[0][].require_all_blocks_memory_resident()
            return self.n_blocks

        var plan = BlockPlan(String("ltx2_product"))
        self.product_host_bytes_ = 0
        for block_idx in range(self.n_blocks):
            var bp = self.prefix + String(block_idx) + "."
            plan.append(bp, BlockKind.transformer())
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
                self.product_host_bytes_ += self.sharded.tensor_view(nm).nbytes()

        var loader = TurboPlannedLoader.open(
            self.checkpoint_path,
            plan^,
            OffloadConfig.synchronous_cfg_paired(),
            ctx,
            fill_block_store=False,
        )
        var pinned = loader.pin_residents_fp8_host_raw(1 << 60, ctx)
        loader.require_all_blocks_memory_resident()
        loader.release_checkpoint_pages()
        loader.discard_unused_raw_streaming_slots(ctx)
        loader.set_fp8h_overlap(True)
        loader.prefetch_with_ctx(0, ctx)
        self.sharded.release_to_os()
        self.product_host_loaders.append(ArcPointer(loader^))
        return pinned

    def load_block_bf16_product(
        mut self, block_idx: Int, ctx: DeviceContext
    ) raises -> FP8Block:
        """Load one product block from the explicitly selected source."""
        if self.int4_slab:
            return self.load_block_bf16(block_idx, ctx)
        if self.product_checkpoint_streaming:
            var block = self.load_block_bf16(block_idx, ctx)
            # load_block_bf16 synchronizes the H2D/dequant work before return,
            # so the source pages are no longer needed by the active GPU work.
            # Drop both mapping residency and clean page-cache charge now rather
            # than letting 37+ GiB accumulate under the desktop user slice.
            self.sharded.release_to_os()
            return block^
        if len(self.product_host_loaders) != 1:
            raise Error(
                "LTX2 product refuses sampling: complete host-resident block "
                "store was not prepared and checkpoint streaming is disabled"
            )
        if self.product_host_active:
            self.product_host_loaders[0][].mark_active_block_done(ctx)
        var handle = self.product_host_loaders[0][].await_block(block_idx, ctx)
        self.product_host_loaders[0][].prefetch_next_with_ctx(block_idx, ctx)

        var block = FP8Block()
        var bp = self.prefix + String(block_idx) + "."
        for ref e in handle.block.items():
            if not e.key.startswith(bp):
                raise Error(
                    String("LTX2 product loader returned key outside block prefix: ")
                    + e.key
                )
            var canon = _substr(e.key, bp.byte_length(), e.key.byte_length())
            block[canon] = e.value
        self.product_host_active = True
        return block^

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
        if self.int4_slab:
            if (
                self.resident_enabled
                and block_idx >= 0
                and block_idx < len(self.resident_loaded)
                and self.resident_loaded[block_idx]
            ):
                return self._load_block_int4_resident(block_idx, ctx)
            return self._load_block_int4(block_idx, ctx)
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
            var canon = _substr(nm, bp.byte_length(), nm.byte_length())
            var tv = self.sharded.tensor_view(nm)
            if tv.dtype == STDtype.F8_E4M3:
                var scale = self._scale_for(nm)
                var raw = Tensor.from_view_raw(tv, ctx)
                # stream-dequant-contract: one-fence-after-complete-block
                # NO-SYNC dequant per tensor + ONE fence at block end (below).
                # The per-tensor SYNCED variant cost ~0.61s per block visit —
                # ~86 fences/block dominated the whole 58s training step
                # (probe ltx2_stream_cost_probe.mojo 2026-07-09). Numerics
                # identical: same kernel, same scale; the block is complete
                # before return either way.
                var deq = fp8_e4m3_dequant_to_bf16_no_sync(raw, scale, ctx)
                block[canon] = ArcPointer(deq^)
            else:
                # BF16 / F32 → BF16 device tensor.
                var t = Tensor.from_view_as_bf16(tv, ctx)
                block[canon] = ArcPointer(t^)
        ctx.synchronize()
        return block^

    def build_fixed_stage(
        self, ctx: DeviceContext, layout_block: Int = 0, alignment: Int = 256
    ) raises -> LTX2FixedStage:
        """Compute the fixed per-tensor layout from ONE reference block
        (`layout_block`, default 0 — all 48 video blocks are shape-identical) and
        allocate ONE device buffer sized to hold a block's BF16 weights at
        256-aligned offsets. NO GPU load happens here: only tensor-view shapes
        are read. Each layout entry is the streamed path's OUTPUT tensor (always
        BF16, shape == on-disk shape, nbytes == numel*2) placed at a cursor-bumped
        aligned offset — the same set of tensors `load_block_bf16` places (scale
        scalars skipped). This buffer is the stable staging region CUDA-graph
        replay reads from (contract C8)."""
        if self.int4_slab:
            raise Error(
                "build_fixed_stage: int4 slab path not supported (fp8 streamed only)"
            )
        var layout = List[LTX2FixedEntry]()
        var bp = self.prefix + String(layout_block) + "."
        var cursor = 0
        for ref nm in self.sharded.names():
            if not nm.startswith(bp):
                continue
            # Skip scale scalars (consumed as dequant multipliers, not weights) —
            # EXACTLY the streamed path's skip set.
            if (
                nm.endswith("_scale")
                or nm.endswith("input_scale")
                or nm.endswith(".scale_weight")
                or nm.endswith(".scale_input")
            ):
                continue
            var canon = _substr(nm, len(bp), len(nm))
            var tv = self.sharded.tensor_view(nm)
            var numel = 1
            for k in range(len(tv.shape)):
                numel *= tv.shape[k]
            var nbytes = numel * STDtype.BF16.byte_size()
            var off = cursor
            layout.append(LTX2FixedEntry(canon, off, nbytes, tv.shape.copy()))
            cursor = _align_up(off + nbytes, alignment)
        if len(layout) == 0:
            raise Error(
                String("build_fixed_stage: no tensors under prefix for block ")
                + String(layout_block)
            )
        var total = cursor
        var dev = ctx.enqueue_create_buffer[DType.uint8](total)
        var addrs = List[Int]()
        for _ in range(len(layout)):
            addrs.append(0)
        return LTX2FixedStage(dev^, layout^, total, alignment, addrs^, False)

    def load_block_bf16_fixed(
        self, block_idx: Int, mut stage: LTX2FixedStage, ctx: DeviceContext
    ) raises -> FP8Block:
        """FIXED-BUFFER twin of `load_block_bf16` (contract C8: CUDA-graph replay
        reads STABLE addresses). Dequant/upcast each tensor into a TEMP EXACTLY as
        the streamed path (from_view_raw→fp8_e4m3_dequant_to_bf16_no_sync for FP8,
        from_view_as_bf16 otherwise), then `enqueue_copy` the temp into
        `stage.buf` at the layout offset and return a block whose Tensors are
        `create_sub_buffer` views into `stage.buf`. The refill kernels are
        byte-identical to `load_block_bf16` — only the copy DESTINATION differs.

        FAIL-LOUD: a tensor absent from the reference-block layout, a byte-length
        that differs from its layout entry, or a sub-view address that drifts
        across visits all raise (never trust allocator determinism)."""
        if self.int4_slab:
            raise Error(
                "load_block_bf16_fixed: int4 slab path not supported (fp8 streamed only)"
            )
        var block = FP8Block()
        var bp = self.prefix + String(block_idx) + "."
        var first_visit = not stage.addr_recorded
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
            var slot = stage.slot_for(canon)
            # Copy the layout entry out (no live borrow of stage while we later
            # mutate stage.addrs).
            var e_off = stage.layout[slot].offset
            var e_nbytes = stage.layout[slot].nbytes
            var e_shape = stage.layout[slot].shape.copy()

            var tv = self.sharded.tensor_view(nm)
            # Materialize the BF16/dequant TEMP exactly as the streamed path.
            var tmp: Tensor
            if tv.dtype == STDtype.F8_E4M3:
                var scale = self._scale_for(nm)
                var raw = Tensor.from_view_raw(tv, ctx)
                tmp = fp8_e4m3_dequant_to_bf16_no_sync(raw, scale, ctx)
            else:
                tmp = Tensor.from_view_as_bf16(tv, ctx)
            if tmp.nbytes() != e_nbytes:
                raise Error(
                    String("load_block_bf16_fixed: tensor '") + canon
                    + "' bytes=" + String(tmp.nbytes())
                    + " != layout nbytes=" + String(e_nbytes)
                    + " (blocks not shape-identical?)"
                )
            # Refill the fixed slot (D2D), then build the returned VIEW from the
            # SAME sub-buffer. NO new dequant kernel — same kernels, new dest.
            var view = stage.buf.create_sub_buffer[DType.uint8](e_off, e_nbytes)
            ctx.enqueue_copy(dst_buf=view, src_buf=tmp.buf)
            var addr = Int(view.unsafe_ptr())
            if first_visit:
                stage.addrs[slot] = addr
            else:
                if addr != stage.addrs[slot]:
                    raise Error(
                        String("load_block_bf16_fixed: address drift for '") + canon
                        + "' (recorded " + String(stage.addrs[slot])
                        + " != " + String(addr) + ") — fixed buffer not stable"
                    )
            block[canon] = ArcPointer(Tensor(view^, e_shape^, STDtype.BF16))
        # Key-set equality with the reference block, BOTH directions: slot_for
        # (above) fail-louds on a key the reference layout lacked; this catches a
        # block MISSING a reference key. Together they assert block N's key set ==
        # block 0's (blocks not shape-identical => fail loud).
        if len(block) != stage.n_entries():
            raise Error(
                String("load_block_bf16_fixed: block ") + String(block_idx)
                + " produced " + String(len(block)) + " tensors != reference layout "
                + String(stage.n_entries()) + " (blocks not shape-identical?)"
            )
        if first_visit:
            stage.addr_recorded = True
        ctx.synchronize()
        return block^

    def build_standalone_stage(
        self, ctx: DeviceContext, layout_block: Int = 0
    ) raises -> LTX2StandaloneStage:
        """Capture-mode twin of `build_fixed_stage`: same per-tensor layout from a
        reference block, but allocate ONE STANDALONE device buffer PER tensor
        (offset 0) instead of one packed buffer. This is the topology CUDA-graph
        replay re-reads under an EXTERNAL per-block refill (see LTX2StandaloneStage
        WHY). NO GPU load here — only tensor-view shapes are read."""
        if self.int4_slab:
            raise Error(
                "build_standalone_stage: int4 slab path not supported (fp8 streamed only)"
            )
        var layout = List[LTX2FixedEntry]()
        var bufs = List[DeviceBuffer[DType.uint8]]()
        var bp = self.prefix + String(layout_block) + "."
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
            var canon = _substr(nm, bp.byte_length(), nm.byte_length())
            var tv = self.sharded.tensor_view(nm)
            var numel = 1
            for k in range(len(tv.shape)):
                numel *= tv.shape[k]
            var nbytes = numel * STDtype.BF16.byte_size()
            layout.append(LTX2FixedEntry(canon, 0, nbytes, tv.shape.copy()))
            bufs.append(ctx.enqueue_create_buffer[DType.uint8](nbytes))
        if len(layout) == 0:
            raise Error(
                String("build_standalone_stage: no tensors under prefix for block ")
                + String(layout_block)
            )
        var addrs = List[Int]()
        for _ in range(len(layout)):
            addrs.append(0)
        return LTX2StandaloneStage(bufs^, layout^, addrs^, False)

    def load_block_bf16_standalone(
        self, block_idx: Int, mut stage: LTX2StandaloneStage, ctx: DeviceContext,
        sync: Bool = True,
    ) raises -> FP8Block:
        """STANDALONE-BUFFER twin of `load_block_bf16_fixed` — identical dequant/
        upcast + byte-length/address/key-set fail-louds, but each tensor is
        `enqueue_copy`d into its OWN standalone buffer (offset 0) and the returned
        Tensor wraps that buffer directly (no sub-buffer view). This is the
        capture-safe external-refill topology; the refilled weights ARE re-read by
        replay (LTX2StandaloneStage WHY).

        sync=True (default, streamed use) fences at the end. The CAPTURE REPLAY
        loop passes sync=False: the H2D+dequant+D2D refill runs on the SAME stream
        as the following cuGraphLaunch, so stream ordering guarantees the refill
        completes before the launch reads the buffers — a per-block ctx.synchronize
        here would be 48 host stalls/step (the sync the rung-2 slab path removed)."""
        if self.int4_slab:
            raise Error(
                "load_block_bf16_standalone: int4 slab path not supported (fp8 streamed only)"
            )
        # SPEED (2026-07-17): if this block is in the resident fp8 pool, refill the
        # stage FROM VRAM (device-only dequant, zero H2D-from-mmap, zero sync) rather
        # than the streamed loop below (which pays an H2D + internal sync PER TENSOR).
        # Mirrors load_block_bf16's residency dispatch; the mmap loop below is the
        # legitimate path for the non-resident boundary blocks. Byte-identical.
        var use_resident = (
            self.resident_enabled
            and block_idx >= 0
            and block_idx < len(self.resident_loaded)
            and self.resident_loaded[block_idx]
        )
        if use_resident:
            return self._load_block_bf16_standalone_resident(block_idx, stage, ctx, sync)
        var block = FP8Block()
        var bp = self.prefix + String(block_idx) + "."
        var first_visit = not stage.addr_recorded
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
            var canon = _substr(nm, bp.byte_length(), nm.byte_length())
            var slot = stage.slot_for(canon)
            var e_nbytes = stage.layout[slot].nbytes
            var e_shape = stage.layout[slot].shape.copy()

            var tv = self.sharded.tensor_view(nm)
            var tmp: Tensor
            if tv.dtype == STDtype.F8_E4M3:
                var scale = self._scale_for(nm)
                var raw = Tensor.from_view_raw(tv, ctx)
                tmp = fp8_e4m3_dequant_to_bf16_no_sync(raw, scale, ctx)
            else:
                tmp = Tensor.from_view_as_bf16(tv, ctx)
            if tmp.nbytes() != e_nbytes:
                raise Error(
                    String("load_block_bf16_standalone: tensor '") + canon
                    + "' bytes=" + String(tmp.nbytes())
                    + " != layout nbytes=" + String(e_nbytes)
                    + " (blocks not shape-identical?)"
                )
            # Owned handle to the standalone buffer (Arc bump — same memory), so we
            # never hold a `stage` borrow while mutating stage.addrs below.
            var bh = stage.bufs[slot].copy()
            ctx.enqueue_copy(dst_buf=bh, src_buf=tmp.buf)
            var addr = Int(bh.unsafe_ptr())
            if first_visit:
                stage.addrs[slot] = addr
            else:
                if addr != stage.addrs[slot]:
                    raise Error(
                        String("load_block_bf16_standalone: address drift for '") + canon
                        + "' (recorded " + String(stage.addrs[slot])
                        + " != " + String(addr) + ") — standalone buffer not stable"
                    )
            block[canon] = ArcPointer(Tensor(bh^, e_shape^, STDtype.BF16))
        if len(block) != stage.n_entries():
            raise Error(
                String("load_block_bf16_standalone: block ") + String(block_idx)
                + " produced " + String(len(block)) + " tensors != reference layout "
                + String(stage.n_entries()) + " (blocks not shape-identical?)"
            )
        if first_visit:
            stage.addr_recorded = True
        if sync:
            ctx.synchronize()
        return block^

    def _load_block_bf16_standalone_resident(
        self, block_idx: Int, mut stage: LTX2StandaloneStage, ctx: DeviceContext,
        sync: Bool,
    ) raises -> FP8Block:
        """RESIDENT-sourced refill of the standalone stage (SPEED — kills the
        per-block H2D-from-mmap the streamed twin pays). For a block in the resident
        fp8 pool: each F8_E4M3 tensor is dequanted -> bf16 ON DEVICE
        (fp8_e4m3_dequant_to_bf16_no_sync — zero H2D, zero sync), each already-bf16
        resident tensor is used directly; the result is enqueue_copy'd IN PLACE into
        the SAME stage.bufs[slot] the captured graph baked (addresses UNCHANGED —
        same asserts). Byte-identical to the streamed twin: same raw fp8 bytes, same
        _scale_for scale (stored in resident_scales at enable time), same dequant
        kernel. Same byte-length / address-drift / key-set fail-louds."""
        var block = FP8Block()
        var first_visit = not stage.addr_recorded
        ref rb = self.resident_blocks[block_idx]
        ref sc = self.resident_scales[block_idx]
        for ref e in rb.items():
            var canon = e.key
            var slot = stage.slot_for(canon)
            var e_nbytes = stage.layout[slot].nbytes
            var e_shape = stage.layout[slot].shape.copy()
            # In-place discipline (loader-authority shape, verbatim):
            #   fp8     -> tmp = dequant(resident, scale)   [transient device-only]
            #   non-fp8 -> tmp = resident bf16 (Arc-shared) [already resident]
            #   D2D       enqueue_copy(dst=stage.bufs[slot], src=tmp.buf)  [existing buffer]
            #   return    Tensor(stage.bufs[slot].copy(), ...)            [SAME buffer]
            var tmp: Tensor
            if e.value[].dtype() == STDtype.F8_E4M3:
                var scale = Float32(1.0)
                if canon in sc:
                    scale = sc[canon]
                tmp = fp8_e4m3_dequant_to_bf16_no_sync(e.value[], scale, ctx)
            else:
                tmp = Tensor(e.value[].buf.copy(), e.value[].shape(), e.value[].dtype())
            if tmp.nbytes() != e_nbytes:
                raise Error(
                    String("_load_block_bf16_standalone_resident: tensor '") + canon
                    + "' bytes=" + String(tmp.nbytes()) + " != layout nbytes="
                    + String(e_nbytes) + " (blocks not shape-identical?)")
            # Owned handle to the standalone buffer (Arc bump — SAME memory = the baked
            # address), so we never hold a `stage` borrow while mutating stage.addrs.
            var bh = stage.bufs[slot].copy()
            ctx.enqueue_copy(dst_buf=bh, src_buf=tmp.buf)   # D2D into the EXISTING buffer
            var addr = Int(bh.unsafe_ptr())
            if first_visit:
                stage.addrs[slot] = addr
            else:
                if addr != stage.addrs[slot]:
                    raise Error(
                        String("_load_block_bf16_standalone_resident: address drift for '")
                        + canon + "' (recorded " + String(stage.addrs[slot]) + " != "
                        + String(addr) + ") — standalone buffer not stable")
            block[canon] = ArcPointer(Tensor(bh^, e_shape^, STDtype.BF16))
        if len(block) != stage.n_entries():
            raise Error(
                String("_load_block_bf16_standalone_resident: block ") + String(block_idx)
                + " produced " + String(len(block)) + " tensors != reference layout "
                + String(stage.n_entries()) + " (blocks not shape-identical?)")
        if first_visit:
            stage.addr_recorded = True
        if sync:
            ctx.synchronize()
        return block^

    def _load_block_int4(
        self, block_idx: Int, ctx: DeviceContext
    ) raises -> FP8Block:
        """Reconstruct block `block_idx` bf16 from the attached svdint4 slab.
        A class-A linear is detected by `<base>.qweight`; its 6 slab tensors are
        bundled into SvdquantLinearA and reconstructed to a dense BF16 W
        (dequant4 + low-rank), placed under the prefix-stripped `<base>.weight`
        with its `<base>.bias`. int4 companions are consumed; everything else
        (norms, scale_shift_table, gate_logits, non-quantized weights/biases)
        passes through verbatim BF16. Same dict contract as the fp8 path."""
        ref st = self.int4_slab.value()
        var block = FP8Block()
        var bp = self.prefix + String(block_idx) + "."
        for ref nm in st.names():
            if not nm.startswith(bp):
                continue
            var canon = _substr(nm, bp.byte_length(), nm.byte_length())
            if canon.endswith(".qweight"):
                var qsuf = String(".qweight").byte_length()
                var base = _substr(canon, 0, canon.byte_length() - qsuf)
                var full = _substr(nm, 0, nm.byte_length() - qsuf)
                var qweight = Tensor.from_view_raw(st.tensor_view(full + ".qweight"), ctx)
                var wscales = Tensor.from_view(st.tensor_view(full + ".wscales"), ctx)
                var lora_down = Tensor.from_view(st.tensor_view(full + ".lora_down"), ctx)
                var lora_up = Tensor.from_view(st.tensor_view(full + ".lora_up"), ctx)
                var smooth = Tensor.from_view(st.tensor_view(full + ".smooth"), ctx)
                var bias = Tensor.from_view(st.tensor_view(full + ".bias"), ctx)
                var in_f = qweight.shape()[1] * 2
                var out_f = qweight.shape()[0]
                var rank = lora_down.shape()[1]
                block[base + ".bias"] = ArcPointer(bias.clone(ctx))
                var w = SvdquantLinearA(
                    qweight^, wscales^, lora_down^, lora_up^, smooth^, bias^,
                    in_f, out_f, rank,
                )
                block[base + ".weight"] = ArcPointer(svdquant_reconstruct_weight(w, ctx))
            elif (
                canon.endswith(".wscales") or canon.endswith(".lora_down")
                or canon.endswith(".lora_up") or canon.endswith(".smooth")
            ):
                continue
            elif canon.endswith(".bias"):
                var bsuf = String(".bias").byte_length()
                var bbase = _substr(nm, 0, nm.byte_length() - bsuf)
                if (bbase + ".qweight") in st.name_to_shard:
                    continue
                block[canon] = ArcPointer(Tensor.from_view_as_bf16(st.tensor_view(nm), ctx))
            else:
                block[canon] = ArcPointer(Tensor.from_view_as_bf16(st.tensor_view(nm), ctx))
        ctx.synchronize()
        return block^

    def enable_int4_resident_range(
        mut self, first_block: Int, last_block: Int, ctx: DeviceContext
    ) raises:
        """Preload the svdint4 slab's per-block tensors onto the GPU for blocks
        [first,last]. Then load_block_bf16 reconstructs each block bf16 ON-USE
        from the resident GPU tensors (dequant4 + low-rank) with NO disk/host
        re-read — the int4 analog of enable_fp8_resident_range. The whole 22B
        int4 slab (~15.4GB) fits a 24GB card. Tensors are stored keyed by the
        prefix-stripped canonical name (attn1.to_q.qweight, ...)."""
        if not self.int4_slab:
            raise Error("enable_int4_resident_range: no int4 slab attached")
        if len(self.resident_blocks) != 0:
            raise Error("enable_int4_resident_range: resident store already initialized")
        ref st = self.int4_slab.value()
        self.resident_enabled = True
        self.resident_bytes_ = 0
        for block_idx in range(self.n_blocks):
            var rb = FP8Block()
            var loaded = block_idx >= first_block and block_idx <= last_block
            if loaded:
                var bp = self.prefix + String(block_idx) + "."
                for ref nm in st.names():
                    if not nm.startswith(bp):
                        continue
                    var canon = _substr(nm, bp.byte_length(), nm.byte_length())
                    var t: Tensor
                    if canon.endswith(".qweight"):
                        t = Tensor.from_view_raw(st.tensor_view(nm), ctx)   # I8 resident
                    else:
                        t = Tensor.from_view_as_bf16(st.tensor_view(nm), ctx)
                    self.resident_bytes_ += t.nbytes()
                    rb[canon] = ArcPointer(t^)
            self.resident_blocks.append(rb^)
            self.resident_scales.append(Dict[String, Float32]())
            self.resident_loaded.append(loaded)
        ctx.synchronize()

    def _load_block_int4_resident(
        self, block_idx: Int, ctx: DeviceContext
    ) raises -> FP8Block:
        """Reconstruct block `block_idx` bf16 from the GPU-RESIDENT int4 tensors
        (no disk/host re-read). For each resident `<base>.qweight`, reconstruct
        the dense W via svdquant_reconstruct_weight_raw (borrows the resident
        tensors); passthrough tensors are shared by Arc."""
        ref rb = self.resident_blocks[block_idx]
        var block = FP8Block()
        for ref e in rb.items():
            var canon = e.key
            if canon.endswith(".qweight"):
                var qsuf = String(".qweight").byte_length()
                var base = _substr(canon, 0, canon.byte_length() - qsuf)
                ref qweight = rb[base + ".qweight"][]
                ref wscales = rb[base + ".wscales"][]
                ref lora_down = rb[base + ".lora_down"][]
                ref lora_up = rb[base + ".lora_up"][]
                var in_f = qweight.shape()[1] * 2
                var out_f = qweight.shape()[0]
                var W = svdquant_reconstruct_weight_raw(
                    qweight, wscales, lora_down, lora_up, in_f, out_f, ctx)
                block[base + ".weight"] = ArcPointer(W^)
                block[base + ".bias"] = rb[base + ".bias"]   # share Arc
            elif (
                canon.endswith(".wscales") or canon.endswith(".lora_down")
                or canon.endswith(".lora_up") or canon.endswith(".smooth")
            ):
                continue
            elif canon.endswith(".bias"):
                var bsuf = String(".bias").byte_length()
                var bbase = _substr(canon, 0, canon.byte_length() - bsuf)
                if (bbase + ".qweight") in rb:
                    continue
                block[canon] = e.value       # share Arc
            else:
                block[canon] = e.value       # share Arc (norms, scale_shift, gate)
        return block^

    def _load_resident_block_bf16(
        self, block_idx: Int, ctx: DeviceContext
    ) raises -> FP8Block:
        """Materialize one BF16 block from GPU-resident raw storage.

        FP8 tensors are already on the GPU as raw bytes, so this avoids the
        repeated host→device copy in the streamed path. The resident-only
        dequant call also avoids a per-tensor host/device fence; later kernels
        consume the BF16 tensors on the same DeviceContext stream. Non-FP8
        tensors are stored resident as BF16 and reused by Arc."""
        var block = FP8Block()
        ref raw_block = self.resident_blocks[block_idx]
        ref scale_block = self.resident_scales[block_idx]
        for ref e in raw_block.items():
            if e.value[].dtype() == STDtype.F8_E4M3:
                var scale = Float32(1.0)
                if e.key in scale_block:
                    scale = scale_block[e.key]
                var deq = fp8_e4m3_dequant_to_bf16_no_sync(e.value[], scale, ctx)
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
                    var canon = _substr(nm, bp.byte_length(), nm.byte_length())
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
