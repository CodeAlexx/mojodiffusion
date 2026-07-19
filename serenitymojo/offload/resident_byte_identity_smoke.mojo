# offload/resident_byte_identity_smoke.mojo — [GPU] byte-identity gate for
# TurboPlannedLoader.pin_residents (Phase-4 residency, 2026-06-11).
#
# For block 0 (double) and block 8 (first single) of Klein-9B:
#   1. await_block STREAMED (before any pinning) -> to_host every tensor
#   2. pin_residents(enough for these blocks) -> await_block RESIDENT -> to_host
#   3. compare element-exact. Any mismatch = the residency path corrupts bytes.
# PASS = "ALL TENSORS IDENTICAL". This decides whether the measured 8e-5
# step-1 loss shift (scoreboard log 2026-06-11) is wrong-bytes (bug) or
# pointer-alignment GEMM algo selection (acceptable-class, like the optimizer
# m/v ties).
#
# Build + run:
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/offload/resident_byte_identity_smoke.mojo -o /tmp/resid_id \
#     && /tmp/resid_id

from std.gpu.host import DeviceContext

from serenitymojo.offload.plan import build_klein_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

comptime CKPT = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"


def _check_block(
    idx: Int, mut loader: TurboPlannedLoader, ctx: DeviceContext,
    streamed_names: List[String], streamed_vals: List[List[Float32]],
) raises -> Int:
    var h = loader.await_block(idx, ctx)
    var bad = 0
    for i in range(len(streamed_names)):
        var nm = streamed_names[i].copy()
        var t = h.block[nm][].to_host(ctx)
        if len(t) != len(streamed_vals[i]):
            print("  LEN MISMATCH", nm, len(t), "vs", len(streamed_vals[i]))
            bad += 1
            continue
        var d = 0
        for j in range(len(t)):
            if t[j] != streamed_vals[i][j]:
                d += 1
        if d != 0:
            print("  BYTE MISMATCH block", idx, nm, " elems=", d)
            bad += d
    return bad


def _snapshot(
    idx: Int, mut loader: TurboPlannedLoader, ctx: DeviceContext,
    mut names: List[String], mut vals: List[List[Float32]],
) raises:
    var h = loader.await_block(idx, ctx)
    for entry in h.block.items():
        names.append(entry.key.copy())
        vals.append(entry.value[].to_host(ctx))
    loader.mark_active_block_done(ctx)


def main() raises:
    print("=== resident byte-identity gate (Klein-9B blocks 0 and 8) ===")
    var ctx = DeviceContext()
    var plan = build_klein_block_plan(8, 24)
    var loader = TurboPlannedLoader.open(
        CKPT, plan^, OffloadConfig.synchronous_single(), ctx
    )
    # 1) streamed snapshots (no residents yet)
    var n0 = List[String]()
    var v0 = List[List[Float32]]()
    _snapshot(0, loader, ctx, n0, v0)
    var n8 = List[String]()
    var v8 = List[List[Float32]]()
    _snapshot(8, loader, ctx, n8, v8)
    print("streamed snapshots:", len(n0), "+", len(n8), "tensors")
    # 2) pin enough blocks to cover 0..8 (~6 GiB) then compare resident reads
    var pinned = loader.pin_residents(7 * 1024 * 1024 * 1024, ctx)
    print("pinned:", pinned, "blocks")
    if pinned < 9:
        raise Error("gate needs blocks 0..8 resident; pinned=" + String(pinned))
    var bad = _check_block(0, loader, ctx, n0, v0)
    bad += _check_block(8, loader, ctx, n8, v8)
    if bad != 0:
        raise Error("resident bytes differ: " + String(bad))
    print("=== ALL TENSORS IDENTICAL — resident bytes == streamed bytes ===")
