# serenitymojo/models/text_encoder/minimax_h3_repeat_kv_probe.mojo
#
# Verifies qwen3_encoder.mojo's `_repeat_kv` at MiniMax-H3's GQA ratio
# n_rep = 64/8 = 8 — every existing production preset only ever exercises
# n_rep in {4 (Z-Image/Klein, 32/8 or 4096-hidden Klein-9B), 7 (Qwen-Image,
# qwen25vl_encoder.mojo)}. n_rep=8 has never been run in this repo. This is a
# RUN, not an inference from reading the kernel: builds real per-(seq,kv_head)
# distinguishable KV data, calls `_repeat_kv`, and checks EVERY output
# element against its source kv head (kvh = head // n_rep, matching
# `_repeat_kv_kernel_f32`'s own index math, qwen3_encoder.mojo:229) — not
# just the output shape.
#
# Run: pixi run mojo run -I . serenitymojo/models/text_encoder/minimax_h3_repeat_kv_probe.mojo

from max.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.text_encoder.qwen3_encoder import _repeat_kv


def main() raises:
    var ctx = DeviceContext()
    comptime SEQ = 3
    comptime H = 64
    comptime H_KV = 8
    comptime DH = 128
    comptime N_REP = H // H_KV  # 8 — the value under test

    if N_REP != 8:
        raise Error("minimax_h3_repeat_kv_probe: N_REP arithmetic is wrong")

    # [1, SEQ, H_KV, DH] with a value UNIQUE per (seq, kv_head) so every
    # repeated output head is checkable against its specific source, not
    # merely shape-checked.
    var data = List[Float32]()
    for s in range(SEQ):
        for hk in range(H_KV):
            var v = Float32(s * 100 + hk)
            for _d in range(DH):
                data.append(v)
    var shape = List[Int]()
    shape.append(1)
    shape.append(SEQ)
    shape.append(H_KV)
    shape.append(DH)
    var kv = Tensor.from_host(data, shape^, STDtype.F32, ctx)

    var out = _repeat_kv(kv^, H, H_KV, ctx)
    var os = out.shape()
    if len(os) != 4 or os[0] != 1 or os[1] != SEQ or os[2] != H or os[3] != DH:
        raise Error("minimax_h3_repeat_kv_probe: wrong output shape")
    if out.dtype() != STDtype.F32:
        raise Error("minimax_h3_repeat_kv_probe: wrong output dtype")

    var host = out.to_host(ctx)
    var mismatches = 0
    for s in range(SEQ):
        for h in range(H):
            var kvh = h // N_REP  # matches _repeat_kv_kernel_f32:229 exactly
            var expected = Float32(s * 100 + kvh)
            var base = ((s * H) + h) * DH
            for d in range(DH):
                if host[base + d] != expected:
                    mismatches += 1
    if mismatches != 0:
        raise Error(
            String("minimax_h3_repeat_kv_probe: ")
            + String(mismatches)
            + " mismatched elements out of "
            + String(SEQ * H * DH)
        )

    print(
        "minimax_h3_repeat_kv_probe: out shape", os[0], "x", os[1], "x", os[2], "x", os[3],
        " n_rep=", N_REP, " 0/", SEQ * H * DH, "mismatches",
    )
    print("PASS — _repeat_kv verified at n_rep=8 (RUN, not inferred)")
