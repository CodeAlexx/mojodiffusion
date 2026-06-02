# block_loader.mojo — BlockLoader: on-demand transformer-block weight streaming.
#
# Mirrors the Rust `BlockLoader` (inference-flame/src/offload.rs) and the
# `OffloaderApi`/`OffloaderBlock` surface (inference-flame/src/offload_api.rs).
# Pure Mojo, GPU-resident blocks. Mojo 1.0.0b1, Linux x86-64, NVIDIA.
#
# What it does (mirror of the Rust semantics):
#   * Holds an mmap-backed `ShardedSafeTensors` (the byte-parity-verified loader,
#     io/sharded.mojo). No safetensors re-implementation here.
#   * `load_block(prefix, ctx)`: load EVERY tensor whose name starts with
#     `prefix` (prefix = literal string match, exactly Rust `key.starts_with`)
#     to a fresh device `Tensor` via `Tensor.from_view` (H2D copy), returning the
#     block as a Dict. Only that block's weights are resident on GPU at a time.
#   * `unload_block`: dropping the returned Dict frees the device buffers (see
#     "Ownership" below). Rust's `unload_block` is `self.cache.clear()`; our
#     equivalent is "let the returned Dict go out of scope (or call drop)".
#   * `prefetch_block(prefix)`: MADV_WILLNEED each tensor in the block (warm the
#     page cache before H2D), via the owning shard's `prefetch_tensor`.
#
# Block in-memory representation (matches Rust `OffloaderBlock(Arc<HashMap<...>>)`):
#   `Dict[String, ArcPointer[Tensor]]`, keyed by the FULL tensor name (we do NOT
#   strip the prefix — the Rust default `BlockLoader::new` has `key_prefix=""`,
#   so its cache keys are the full names too; the prefix-stripping `new_with_prefix`
#   variant is a separate Rust constructor we don't need for Z-Image).
#
#   Why `ArcPointer[Tensor]` and not a bare `Tensor`: `Tensor` is
#   Movable-but-NOT-Copyable (it uniquely owns a `DeviceBuffer`), and Mojo
#   1.0.0b1's `Dict[K, V]` REQUIRES `V: Copyable & ImplicitlyDestructible`
#   (verified: a bare `Dict[String, Tensor]` fails to compile). `ArcPointer` is
#   Copyable (copy == refcount bump), so `Dict[String, ArcPointer[Tensor]]`
#   compiles AND the device buffer is freed exactly when the last Arc to it
#   drops. This is the SAME discipline io/sharded.mojo uses for
#   `List[ArcPointer[SafeTensors]]`, and it mirrors the Rust block's `Arc<...>`
#   wrapping one-for-one.
#
# Ownership / lifetime (origin-safe):
#   `Tensor.from_view` COPIES the mmap bytes into a fresh device `DeviceBuffer`
#   (tensor.mojo from_view docstring: "The bytes are COPIED into a fresh device
#   buffer, so the resulting Tensor does not alias (and does not keep alive) the
#   source mmap region."). So a returned block Tensor does NOT borrow the
#   loader's `ShardedSafeTensors` / mmap — no origin entanglement, no dangling.
#   The block can outlive nothing it depends on; it owns its own VRAM. Dropping
#   the Dict (Arc refcount -> 0) runs `Tensor.__del__`, which frees the
#   `DeviceBuffer`. There is no explicit-free API on `Tensor`; drop IS the free.

from std.memory import ArcPointer
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor
from std.gpu.host import DeviceContext


# A loaded block: tensor-name -> device Tensor (Arc-wrapped). Mirrors Rust
# `OffloaderBlock(Arc<HashMap<String, Tensor>>)`.
comptime Block = Dict[String, ArcPointer[Tensor]]


struct BlockLoader(Movable):
    """Streams transformer-block weights from a sharded safetensors directory,
    one block at a time, so large models fit in 24 GB. Mirrors the Rust
    `BlockLoader` (inference-flame/src/offload.rs): prefix-keyed load, drop to
    unload.

    Holds a `ShardedSafeTensors` (mmap-backed). `ShardedSafeTensors` is
    Movable-not-Copyable, so `BlockLoader` is Movable-not-Copyable too — it
    uniquely owns the open shards / mmaps, freed on `__del__`."""

    var sharded: ShardedSafeTensors

    @staticmethod
    def open(dir: String) raises -> BlockLoader:
        """Open the model directory (index-detected shards or single-file
        fallback) via the verified `ShardedSafeTensors.open`."""
        var st = ShardedSafeTensors.open(dir)
        return BlockLoader(st^)

    def __init__(out self, var sharded: ShardedSafeTensors):
        self.sharded = sharded^

    def _block_prefix(self, prefix: String) -> String:
        # Normalize to a dot-delimited block boundary: append '.' unless present.
        # Closes the footgun where "layers.1" would match "layers.10."; matches the
        # effective behavior of Rust load_block (which appends '.'). F1 fix 2026-05-25.
        return prefix if prefix.endswith(".") else prefix + "."

    def block_count_for(self, prefix: String) -> Int:
        """Number of tensors in the block at `prefix` (cheap; no H2D)."""
        var p = self._block_prefix(prefix)
        var n = 0
        for ref nm in self.sharded.names():
            if nm.startswith(p):
                n += 1
        return n

    def prefetch_block(self, prefix: String) raises:
        """MADV_WILLNEED every tensor in the block to warm the page cache before
        H2D. Mirrors `OffloaderApi::prefetch_block` (the warm-ahead step). Reaches
        each tensor's OWNING shard (via `shard_index`) and calls that shard's
        `prefetch_tensor` — `ShardedSafeTensors` doesn't expose prefetch directly,
        so we go through its public `shards` list."""
        var p = self._block_prefix(prefix)
        for ref nm in self.sharded.names():
            if nm.startswith(p):
                var idx = self.sharded.shard_index(nm)
                self.sharded.shards[idx][].prefetch_tensor(nm)

    def load_block(self, prefix: String, ctx: DeviceContext) raises -> Block:
        """Load every tensor whose name starts with `prefix` onto the GPU as a
        device `Tensor` (H2D via `Tensor.from_view`), returning the block Dict.

        The prefix is normalized to a dot boundary (`_block_prefix`): a trailing
        '.' is appended unless present, so both `"layers.1"` and `"layers.1."` load
        ONLY layer 1 (never `"layers.10."`). Matches Rust `load_block`'s internal
        dot-append. NOTE: dtype is preserved on load (no BF16 coercion, unlike Rust);
        inert for all-BF16 models (e.g. Z-Image). The returned block owns its VRAM;
        drop it to unload."""
        var block = Block()
        var p = self._block_prefix(prefix)
        for ref nm in self.sharded.names():
            if nm.startswith(p):
                var tv = self.sharded.tensor_view(nm)
                var t = Tensor.from_view(tv, ctx)
                block[nm] = ArcPointer(t^)
        return block^

    def load_block_as_bf16(self, prefix: String, ctx: DeviceContext) raises -> Block:
        """Load every tensor in `prefix` as BF16 device storage, converting
        F32/F16 safetensors on the host before H2D. Use this for large F32
        checkpoints where a raw `from_view` would put F32 layer weights on the
        GPU and blow the memory budget."""
        var block = Block()
        var p = self._block_prefix(prefix)
        for ref nm in self.sharded.names():
            if nm.startswith(p):
                var tv = self.sharded.tensor_view(nm)
                var t = Tensor.from_view_as_bf16(tv, ctx)
                block[nm] = ArcPointer(t^)
        return block^


# ── unload note ───────────────────────────────────────────────────────────────
# There is no `unload_block(self, ...)` method: the block is a value the caller
# owns, and Mojo frees it deterministically when it goes out of scope or is
# explicitly transferred away. To unload mid-scope, rebind the variable
# (`block = Block()`) or transfer it into a sink. Each pattern drops the
# ArcPointers; the last Arc to each `Tensor` runs `Tensor.__del__` -> frees the
# `DeviceBuffer` (VRAM). This is the Mojo-idiomatic equivalent of Rust's
# `self.cache.clear()`. A free helper is provided for an explicit, readable
# unload at call sites:
def unload_block(var block: Block):
    """Explicitly drop a loaded block, freeing its device buffers. Equivalent to
    letting `block` fall out of scope; provided for an explicit, readable unload.
    Takes the block by `var` (owned), so the Dict + its ArcPointers are destroyed
    when this function returns, dropping the last reference to each `Tensor` and
    freeing its VRAM. Call as `unload_block(block^)`."""
    _ = block^
