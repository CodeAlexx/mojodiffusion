# serenitymojo.io — pure-Mojo safetensors read path.
#
# Ported from the Rust crate serenity-safetensors (read path only), mirroring
# src/mmap.rs logic exactly. Linux x86-64. No Python in the runtime path; the
# JSON header is hand-rolled (json_header.mojo). See io/BUILD_PLAN.md.
#
# Modules:
#   dtype         — STDtype enum + byte_size/from_name/name/to_mojo_dtype
#   ffi           — libc externs (mmap/munmap/madvise/sysconf/open/close/pread)
#   mmap          — MmapRegion (uncommitted MAP_NORESERVE mmap of data segment)
#   json_header   — minimal flat-schema safetensors header parser
#   safetensors   — SafeTensors single-file reader (header + mmap'd data)
#   tensor_view   — TensorView: typed metadata + ORIGIN-BOUND byte span (chunk 2)
#   sharded       — ShardedSafeTensors: index/weight_map multi-shard loader,
#                   origin-bound views into shards owned inside the struct
#                   (chunk 2)
