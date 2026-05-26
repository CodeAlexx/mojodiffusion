# serenitymojo.offload — on-demand transformer-block weight streaming.
#
# Mirrors the Rust BlockLoader (inference-flame/src/offload.rs) + OffloaderApi
# (inference-flame/src/offload_api.rs). Reuses the verified io/ read path
# (ShardedSafeTensors) and tensor.mojo (Tensor.from_view, H2D). No safetensors
# or H2D re-implementation. Linux x86-64, NVIDIA, GPU-resident blocks.
#
# Modules:
#   block_loader  — BlockLoader: open(dir), load_block(prefix, ctx) -> Block
#                   (Dict[String, ArcPointer[Tensor]]), prefetch_block(prefix),
#                   unload_block(block). Block drop frees the device buffers.
