# Serenitymojo — pure-Mojo, inference-only, GPU-only tensor/kernel library.
# Standalone (no MAX graph/engine dependency); leans on the Mojo SDK
# (linalg, nn, layout, gpu) for the heavy kernels and hand-writes only the
# diffusion-specific fused elementwise/reshape ops. See ../PLAN.md.

comptime VERSION = "0.0.1"
