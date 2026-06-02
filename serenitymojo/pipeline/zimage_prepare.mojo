# pipeline/zimage_prepare.mojo — Z-Image cache reader / inspector.
#
# TRANSLATION NOTE (2026-06-01): prepare_zimage.rs encodes images through the
# Z-Image (Qwen-Image-family) VAE + Qwen3 text encoder and writes per-sample
# {latent, text_embedding, text_mask} safetensors. For the FIRST real run the
# cache ALREADY EXISTS at
#   /home/alex/EriDiffusion/EriDiffusion-v2/cache/alina_zimage_512
# (51 files; latent F32 [1,16,64,64], text_embedding F32 [1,512,2560],
# text_mask F32 [1,512]) — so the VAE-encoder port is SKIPPED (FAST PATH per the
# task). This module is the cache READER: it reuses the model-agnostic
# KleinCache (same latent/text_embedding/text_mask schema) and exposes an
# inspector main() that prints the keys/shapes so the trainer's assumptions are
# verifiable. When the VAE-encoder port lands, the encode side is added here
# mirroring prepare_zimage.rs.
#
# Run (inspect the cache):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/pipeline/zimage_prepare.mojo

from std.gpu.host import DeviceContext
from serenitymojo.training.klein_dataset import KleinCache


comptime ZIMAGE_CACHE_DIR = "/home/alex/EriDiffusion/EriDiffusion-v2/cache/alina_zimage_512"


def main() raises:
    var ctx = DeviceContext()
    print("=== Z-Image cache inspector:", ZIMAGE_CACHE_DIR, "===")
    var cache = KleinCache(String(ZIMAGE_CACHE_DIR))
    print("samples:", cache.count())
    var key = cache.peek_key(0, ctx)
    print("sample[0] latent C/H/W =", key.c, key.h, key.w, " text_seq =", key.seq)
    var s = cache.load(0, ctx)
    print("latent shape:", s.latent.shape()[0], s.latent.shape()[1], s.latent.shape()[2], s.latent.shape()[3])
    print("text_embedding shape:", s.text_embedding.shape()[0], s.text_embedding.shape()[1], s.text_embedding.shape()[2])
    print("OK: cache readable; trainer can consume {latent, text_embedding, text_mask}")
