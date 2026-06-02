# flux1_dit_probe.mojo — COMPILE-ONLY probe for Flux1DiT / Flux1Offloaded.
# Imports + references the types and monomorphizes the comptime helpers. Does NOT
# run (GPU wedged). Verification = clean compile (EXIT=0).

from serenitymojo.models.dit.flux1_dit import (
    Flux1Config,
    Flux1DiT,
    Flux1Offloaded,
    build_flux1_rope_tables,
)


def main() raises:
    var cfg = Flux1Config.dev()
    if cfg.num_double != 19 or cfg.num_single != 38:
        raise Error("FLUX.1 config: expected 19 double + 38 single blocks")
    if cfg.num_heads * cfg.head_dim != cfg.inner_dim:
        raise Error("FLUX.1 config: num_heads*head_dim != inner_dim")
    if not cfg.has_guidance:
        raise Error("FLUX.1 dev config: expected has_guidance")
    var sc = Flux1Config.schnell()
    if sc.has_guidance:
        raise Error("FLUX.1 schnell config: expected no guidance")

    # Reference the types + the comptime rope-table builder so they monomorphize.
    comptime DitT = Flux1DiT
    comptime OffT = Flux1Offloaded
    comptime RopeFn = build_flux1_rope_tables[256, 16, 24, 128]
    # Force the generic forward methods to monomorphize at a tiny token grid so
    # their BODIES compile (alias of the bound generic method; not executed).
    comptime FwdFn = Flux1Offloaded.forward[4, 2, 6]
    comptime DblFn = Flux1DiT._double_block[4, 2, 6]
    comptime SglFn = Flux1DiT._single_block[6]
    comptime AttnFn = Flux1DiT._attn_rope_only[6]
    print("flux1_dit probe constructed; inner_dim=", cfg.inner_dim)
