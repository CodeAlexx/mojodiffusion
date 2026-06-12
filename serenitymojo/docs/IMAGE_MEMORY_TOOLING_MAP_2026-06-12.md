# Image/Memory Tooling Map - 2026-06-12

Scope: `/home/alex/mojodiffusion` and `/home/alex/MOJO-libs`, with Qwen and
video excluded for the next implementation slice. Python is allowed here only as
audit/tooling; runtime product paths remain Mojo.

## Search Result

- No repo-local `*.map` files were found under the two requested roots.
- Repo-local map/wayfinding files that do exist:
  - `serenitymojo/MAP.md`
  - `T5_ZIMAGE_TRAINING_MAP.md`
  - `FULL_PORT_ROADMAP.md`
  - `HANDOFF_2026-05-30_PERF_ROADMAP.md`
- A broad `/home/alex` search found unrelated JavaScript source maps inside
  Python/Gradio/Jupyter environments; those are not model/tooling maps and were
  excluded.

## Available Tooling

### Pure Mojo Image I/O

- `/home/alex/MOJO-libs/image/`
  - PNG/JPEG/WebP decode.
  - PNG/JPEG/WebP encode.
  - resize, rotate, crop, color, filters, EXIF, ICC, CMYK, depth/studio ops.
  - `image/gpu.mojo` has GPU color/filter ops, but codecs and resize remain CPU.
- Product adoption in `mojodiffusion`:
  - `serenitymojo/serve/image_io.mojo` decodes init images through MOJO-libs
    PNG/JPEG/WebP.
  - `serenitymojo/serve/zimage_backend.mojo` writes PNGs with
    `serenity.genparams.v1` and uses `resize_bilinear`.
  - `serenitymojo/serve/qwenimage_backend.mojo` also writes metadata PNGs, but
    Qwen is excluded for this slice.
  - `serenitymojo/serve/stub_backend.mojo` is a metadata/PNG proof path only.

### CPU Memory Allocators

- `/home/alex/MOJO-libs/mem/`
  - arena, growable arena, pool, slab, byte ring, aligned buffer, stats.
  - These are CPU/raw-pointer allocators; useful for daemon request parsing,
    gallery indexing, thumbnails, and host-side churn.
- Product adoption:
  - No meaningful model denoise path adoption found yet.
  - Do not assume these fix GPU generation speed; they are host-side tools unless
    explicitly wired into a host hot path.

### GPU Scratch/Activation Memory

- `serenitymojo/scratch_ring.mojo`
  - `ScratchRingAllocator`: GPU slab-backed temporary tensor allocator with
    forward and reverse allocation, mark/rewind, and peak byte tracking.
- Product/training adoption:
  - Used heavily by Klein blocks, Klein stack, autograd v2, linear backward,
    attention backward, and tensor algebra scratch variants.
  - Not the current Z-Image product inference carrier. Z-Image has some device
    no-saved paths, but the broader ring/scratch strategy is not yet a product
    sampler-wide allocation policy there.

### Offload And Residency

- `serenitymojo/offload/turbo_planned_loader.mojo`
  - `TurboPlannedLoader`: async double-buffer block loader with
    `prefetch_with_ctx`, `prefetch_next_with_ctx`, and
    `mark_active_block_done`.
  - This is the existing route for overlapping H2D weight transfer with block
    compute.
- `serenitymojo/offload/vmm_cuda.mojo`
  - CUDA driver/VMM FFI, `cu_mem_get_info`, and `cu_mempool_trim_current`.
- `serenitymojo/offload/vmm_slab.mojo`
  - `VmmSlabAllocator`: reserved virtual address slab with map/unmap,
    refcount, and eviction primitive.
  - Current status: parked. It is implemented but not broadly wired into product
    model paths.
- Product/training adoption:
  - Klein sampler uses `TurboPlannedLoader` and `ScratchRingAllocator`.
  - Flux/Chroma/SD35/Wan22/LTX2 training stacks use `TurboPlannedLoader`.
  - Z-Image product generation does not currently use Turbo/VMM weight
    streaming; it loads the Z-Image pieces directly.
  - `cu_mem_get_info` and `cu_mempool_trim_current` are used by Z-Image and Qwen
    serve backends and by dispatch backend memory trimming.

### Attention / Denoise Kernels

- `serenitymojo/ops/attention.mojo`
  - `sdpa` legacy masked/math route.
  - `sdpa_tiled` and `sdpa_nomask_tiled`: online-softmax large-S paths that do
    not materialize `[S,S]` score tensors. Dh <= 128.
- `serenitymojo/ops/attention_flash.mojo`
  - cuDNN v9 BF16 flash SDPA forward/backward through
    `serenitymojo/ops/cshim/lib/libserenity_cudnn_sdpa.so`.
  - This is the current in-tree fast attention primitive.
- Search result:
  - No `cutlass` or `cudlass` implementation was found in the audited roots.
  - The actual fast external shim found here is cuDNN flash SDPA; GEMM paths use
    the existing linear/vendor matmul stack.
- Product/training adoption:
  - Z-Image no-saved product forwards now dispatch through `sdpa_flash_train_fwd`
    on the path edited earlier in this session.
  - Klein block paths use flash F32 wrappers and scratch carriers in several
    training/sampler paths.
  - Many large/video DiTs use `sdpa_nomask_tiled`, but video is excluded because
    current video models do not work.

## Non-Qwen, Non-Video Candidate Paths

### Z-Image

Why it matters:
- The SwarmUI audit names it as the first resident daemon/model target.
- It already has a product backend, metadata PNG output, resize/image tooling,
  result manifests, and the new flash SDPA product forward route.

What is still weak:
- It is not yet accepted for speed parity.
- It needs a real product sample artifact with prompt, dimensions, timings, and
  positive peak VRAM.
- It does not yet use the broader scratch/Turbo/VMM memory tooling as a sampler
  allocation/offload policy.

### Klein

Why it matters:
- It is the clearest path already using the memory/offload tools:
  `ScratchRingAllocator` plus `TurboPlannedLoader`.
- The existing turbo/VMM audit says the problem is adoption/measurement of the
  overlap contract, not missing primitives.

What is still weak:
- Heavy model compared with Z-Image.
- Existing docs warn some inference loops historically used legacy prefetch
  order and needed measurement before claiming overlap.

### SDXL / SD3 / Flux / Chroma / ERNIE

Why they matter:
- Smaller or more contained image smoke/product paths exist in places.
- They reuse the shared VAE/sampler/image output stack.

What is still weak:
- Some are cached-input or staged smoke paths rather than full raw prompt to
  production image paths.
- They are not the first SwarmUI resident target in the current audit.

## Next Implementation Rule

Do not choose a model by name alone.

For the next non-Qwen image slice, pick one of:

1. Z-Image if the goal is SwarmUI product parity first:
   resident image generation, metadata/gallery, real output timing, VRAM, and
   denoise fast-path evidence.
2. Klein if the goal is specifically memory/offload mechanics:
   verify or fix Turbo overlap, scratch allocation, and peak VRAM measurement.

Do not use video in this slice. Do not use Qwen in this slice.

## Commands Used

```bash
rg --files /home/alex/mojodiffusion /home/alex/MOJO-libs | rg '(\\.map$|mem|memory|alloc|allocator|slab|scratch|offload|cache|cudnn|cutlass|cuda|image|png|jpeg|jpg|latent|vae|denoise|video|mp4|gallery)'
find /home/alex/mojodiffusion /home/alex/MOJO-libs -type f \( -name '*.map' -o -iname '*map*' \) -print
rg -n -i '(cutlass|cudlass|cublas|cudnn|flash|wmma|mma|tensor core|matmul|gemm|sdpa|scratch|slab|ring|vmm|turbo|offload|mempool|cuMem|cuda dma|DMA)' /home/alex/mojodiffusion/serenitymojo /home/alex/MOJO-libs --glob '*.mojo' --glob '*.md' --glob '*.cpp' --glob '*.h'
rg -n 'from image\.|from mem\.|from http\.|from json\.|from sqlite\.|MOJO-libs|ScratchRingAllocator|TurboPlannedLoader|VmmSlabAllocator|cu_mem_get_info|cu_mempool_trim_current|sdpa_flash_train_fwd|sdpa_nomask_tiled|sdpa_tiled|encode_png_with_text|decode_png_bytes|read_png_text' serenitymojo scripts --glob '*.mojo' --glob '*.py' --glob '*.md'
```
