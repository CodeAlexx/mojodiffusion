# pipeline/ltx2_stream_ceiling_smoke.mojo — P2 FP8 streaming memory-ceiling spike.
#
# LTX2_PORT_PLAN_2026-05-28 §P2 (de-risk #2). Proves that the 48 DiT transformer
# blocks of ltx-2.3-22b-distilled-fp8.safetensors (29 GB on disk, blocks 4-46
# in float8_e4m3fn) can be STREAMED one at a time — load → dequant FP8→BF16
# on-use → run a representative block forward → DROP (evict) — staying well
# under the 24 GB ceiling, with peak VRAM bounded (weights evicted, NOT
# accumulating to 29 GB).
#
# This is a MEMORY gate, not a parity gate. The "forward" is a faithful proxy:
# for each block we run the attn1 QKV projections, the attn1 output projection,
# and the FFN (net.0.proj + net.2) on an MVP-shape token sequence using the
# block's REAL (dequantized) FP8 weights. This exercises dequant-on-use and the
# largest weight matrices (the [16384,4096] / [4096,16384] FFN), which dominate
# both compute and the transient BF16 footprint.
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/ltx2_stream_ceiling_smoke.mojo -o /tmp/ltx2_stream_ceiling_smoke
#
# GATE (P2): 48-block forward completes, exit 0, all outputs FINITE, peak
# resident < 22000 MiB, and peak does NOT grow toward 29 GB (eviction works).
#
# HEAVY GPU: requires >= 22 GB free. Run only after Chroma finishes.

from std.gpu.host import DeviceContext
from std.math import isfinite
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.offload.ltx2_block_stream import (
    LTX2BlockStream,
    FP8Block,
    drop_block,
)


comptime CKPT = String(
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
)
comptime MIB = 1024.0 * 1024.0


def _has(block: FP8Block, key: String) -> Bool:
    return key in block


def _w(block: FP8Block, key: String) raises -> ref [block] Tensor:
    return block[key][]


def _finite_count(t: Tensor, ctx: DeviceContext) raises -> Tuple[Int, Int]:
    """Return (n_finite, n_total) for a tensor (host readback, diagnostic only)."""
    var h = t.to_host(ctx)
    var nf = 0
    for i in range(len(h)):
        if isfinite(h[i]):
            nf += 1
    return (nf, len(h))


def _make_input(
    n_tokens: Int, dim: Int, ctx: DeviceContext
) raises -> Tensor:
    """A small deterministic BF16 activation [1, n_tokens, dim]."""
    var vals = List[Float32]()
    var total = n_tokens * dim
    for i in range(total):
        # bounded, deterministic, nonzero
        var v = Float32((i % 17) - 8) * 0.01
        vals.append(v)
    return Tensor.from_host(vals^, [1, n_tokens, dim], STDtype.BF16, ctx)


def _clone_t(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape(), x.dtype())


def _lin(
    block: FP8Block, x: Tensor, w_key: String, b_key: String, ctx: DeviceContext
) raises -> Tensor:
    ref w = _w(block, w_key)
    ref b = _w(block, b_key)
    return linear(x, w, Optional[Tensor](_clone_t(b, ctx)), ctx)


def _block_forward_proxy(
    block: FP8Block, x: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """Representative forward over one block's dequantized weights.

    Runs attn1 QKV + out projections and the FFN. Uses the largest weight
    matrices so the transient BF16 footprint and dequant cost are realistic.
    Returns the FFN output (used only for a finiteness check)."""
    # attn1 QKV (each [4096,4096], bias [4096]).
    var q = _lin(block, x, "attn1.to_q.weight", "attn1.to_q.bias", ctx)
    var k = _lin(block, x, "attn1.to_k.weight", "attn1.to_k.bias", ctx)
    var v = _lin(block, x, "attn1.to_v.weight", "attn1.to_v.bias", ctx)
    # attn1 out projection on q (proxy for the attention-weighted value).
    var o = _lin(block, q, "attn1.to_out.0.weight", "attn1.to_out.0.bias", ctx)
    # FFN: net.0.proj [16384,4096] then net.2 [4096,16384] — the largest mats.
    var h1 = _lin(block, o, "ff.net.0.proj.weight", "ff.net.0.proj.bias", ctx)
    var h2 = _lin(block, h1, "ff.net.2.weight", "ff.net.2.bias", ctx)
    # keep k/v alive through the proxy so their VRAM is part of the peak
    _ = k.numel()
    _ = v.numel()
    return h2^


def main() raises:
    var ctx = DeviceContext()

    var stream = LTX2BlockStream.open(CKPT)
    var n_blocks = stream.block_count()
    print("[open] checkpoint blocks:", n_blocks)

    var mi0 = ctx.get_memory_info()
    var total_vram = Int(mi0[1])
    var free_before = Int(mi0[0])
    print(
        "[vram] total:", Int(Float64(total_vram) / MIB), "MiB  free at start:",
        Int(Float64(free_before) / MIB), "MiB",
    )

    # MVP video-token shape: 256x256 / 16f → ~128 video tokens, dim 4096.
    var n_tokens = 128
    var dim = 4096
    var x = _make_input(n_tokens, dim, ctx)

    var min_free = free_before     # track minimum free seen → peak usage
    var all_finite = True
    var fp8_blocks_seen = 0
    var max_block_peak_mib: Float64 = 0.0
    # SKEPTIC eviction proof: capture the fwd-peak of the FIRST and LAST FP8
    # block. If eviction is broken, the 43 FP8 blocks accumulate (each FFN
    # [16384,4096]≈256 MB dequant) and last >> first, climbing toward 29 GB.
    # A genuine single-resident window keeps last ≈ first. This replaces the
    # old tautological `gate_evict = (peak < 22000)` duplicate.
    var first_fp8_peak_mib: Float64 = -1.0
    var last_fp8_peak_mib: Float64 = -1.0

    for bi in range(n_blocks):
        # Load + dequant one block.
        var block = stream.load_block_bf16(bi, ctx)
        var n_fp8 = stream.fp8_tensor_count(bi)
        if n_fp8 > 0:
            fp8_blocks_seen += 1

        # Free VRAM right after load+dequant (block + transients resident).
        var mi_loaded = Int(ctx.get_memory_info()[0])
        if mi_loaded < min_free:
            min_free = mi_loaded
        var block_peak_mib = Float64(total_vram - mi_loaded) / MIB
        if block_peak_mib > max_block_peak_mib:
            max_block_peak_mib = block_peak_mib

        # Run the representative forward (only if this block has attn1 — all do).
        var ok_fwd = _has(block, "attn1.to_q.weight") and _has(
            block, "ff.net.0.proj.weight"
        )
        if ok_fwd:
            var out = _block_forward_proxy(block, x, ctx)
            # Peak during forward (transients live).
            var mi_fwd = Int(ctx.get_memory_info()[0])
            if mi_fwd < min_free:
                min_free = mi_fwd
            var fwd_peak_mib = Float64(total_vram - mi_fwd) / MIB
            if fwd_peak_mib > max_block_peak_mib:
                max_block_peak_mib = fwd_peak_mib
            # Eviction proof: record first/last FP8-block fwd peak.
            if n_fp8 > 0:
                if first_fp8_peak_mib < 0.0:
                    first_fp8_peak_mib = fwd_peak_mib
                last_fp8_peak_mib = fwd_peak_mib
            # SKEPTIC: shape assertion on EVERY block (must be [1,128,4096]).
            var osh = out.shape()
            var shape_ok = (
                len(osh) == 3 and osh[0] == 1 and osh[1] == n_tokens
                and osh[2] == dim
            )
            if not shape_ok:
                all_finite = False
                print("  [block", bi, "] BAD SHAPE: len", len(osh))
            # SKEPTIC: finiteness on EVERY block, not just 0/mid/last.
            var fc = _finite_count(out, ctx)
            if fc[0] != fc[1]:
                all_finite = False
                print("  [block", bi, "] NON-FINITE:", fc[1] - fc[0], "/", fc[1])
            # Per-block VRAM print to prove flatness across ALL 48 blocks.
            print(
                "  [block", bi, "] finite", fc[0] == fc[1], "shape_ok", shape_ok,
                "fp8", n_fp8, "fwd_peak", Int(fwd_peak_mib), "MiB",
            )
            _ = out^
        else:
            print("  [block", bi, "] no attn1 (skipped forward)  fp8:", n_fp8)

        # EVICT: drop the block before loading the next (single-resident window).
        drop_block(block^)
        ctx.synchronize()

    # Final reclaim check.
    _ = x^
    ctx.synchronize()
    var mi_end = Int(ctx.get_memory_info()[0])
    var free_end = Int(Float64(mi_end) / MIB)
    var peak_mib = Int(Float64(total_vram - min_free) / MIB)

    # Eviction-growth across the 43 FP8 blocks (first → last FP8 block peak).
    var evict_growth_mib = last_fp8_peak_mib - first_fp8_peak_mib

    print("──────────────────────────────")
    print("[result] blocks streamed:", n_blocks, " (with FP8 weights:", fp8_blocks_seen, ")")
    print("[result] peak resident:", peak_mib, "MiB  (max per-block peak:", Int(max_block_peak_mib), "MiB )")
    print("[result] free VRAM at end:", free_end, "MiB")
    print(
        "[result] FP8 fwd-peak first:", Int(first_fp8_peak_mib),
        "MiB  last:", Int(last_fp8_peak_mib),
        "MiB  growth:", Int(evict_growth_mib), "MiB",
    )
    print("[result] all sampled forwards finite:", all_finite)

    # ── GATE ──────────────────────────────────────────────────────────────────
    var gate_ceiling = peak_mib < 22000
    # GENUINE eviction proof (not a duplicate of the ceiling): across all 43
    # FP8 blocks the per-block fwd peak must NOT climb. If eviction leaked, each
    # block adds ≥256 MB (FFN dequant) and growth would be multi-GB by block 46.
    # Allow a small slack for pool fragmentation / measurement jitter.
    var gate_evict = (evict_growth_mib < 512.0) and (first_fp8_peak_mib > 0.0)
    var gate = gate_ceiling and gate_evict and all_finite and (
        fp8_blocks_seen > 0
    )
    if gate:
        print("P2 GATE PASS: 48 blocks streamed, peak", peak_mib,
              "MiB < 22000, finite, FP8 dequant-on-use, eviction bounded",
              "(FP8 peak growth", Int(evict_growth_mib), "MiB < 512).")
    else:
        print("P2 GATE FAIL: ceiling", gate_ceiling, " evict_bounded", gate_evict,
              " finite", all_finite, " fp8_seen", fp8_blocks_seen,
              " evict_growth", Int(evict_growth_mib), "MiB")
        raise Error("P2 streaming ceiling gate FAILED")
