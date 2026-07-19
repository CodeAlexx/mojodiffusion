#!/usr/bin/env python3
"""Static contract for the Flux worker's executed VAE strategy manifest field."""

from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1] / "serenitymojo/serve/flux_backend.mojo"


def main() -> None:
    source = BACKEND.read_text(encoding="utf-8")
    required = (
        "var vae_decode_grid: String",
        'self.vae_decode_grid = String("whole_image")',
        'self.vae_decode_grid = String("3x3_tiled_fallback")',
        '"vae_decode_tile_grid":"\') + json_escape(self.vae_decode_grid)',
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise SystemExit(f"FAIL Flux VAE manifest contract missing: {missing}")
    stale = '\"vae_decode_tile_grid\":\"5x5_lowmem\"'
    if stale in source:
        raise SystemExit("FAIL Flux manifest still hard-codes the retired 5x5 path")
    whole = source.index('self.vae_decode_grid = String("whole_image")')
    tiled = source.index('self.vae_decode_grid = String("3x3_tiled_fallback")')
    whole_branch = source.index("if mem.free_bytes > WHOLE_DECODE_MIN_FREE_BYTES:")
    tiled_call = source.index("var img = flux_tiled_decode[LATENT_H_, LATENT_W_]")
    if not whole_branch < whole < tiled_call < tiled:
        raise SystemExit("FAIL Flux decode-strategy labels are not attached to their executed branches")
    print("PASS Flux result manifest reports executed whole_image vs 3x3_tiled_fallback")


if __name__ == "__main__":
    main()
