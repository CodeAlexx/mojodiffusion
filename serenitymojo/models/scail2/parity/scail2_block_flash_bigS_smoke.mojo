# serenitymojo/models/scail2/parity/scail2_block_flash_bigS_smoke.mojo
#
# G2 LARGE-S OOM-RELIEF SMOKE (CHUNK 2c) for the SCAIL-2 i2v FLASH training block.
#
# Runs ONE scail2 block fwd+bwd with FLASH=True at the REAL transformer dims
# (H=40, DH=128, DIM=5120, FFN=13824, TXT=512, IMG=257) and a LARGE sequence S
# that the DENSE sdpa_nomask path provably CANNOT hold: the dense self-attn
# materializes an [1,H,S,S] score tensor = H*S*S*4 bytes (F32 interior). At the
# real video S=39,872 that is ~254 GB → guaranteed OOM on the 16 GB 5080. The
# FLASH path never materializes S×S; its self-attn memory is O(S).
#
# Random/sinusoidal inputs + random LoRA (A!=0, B!=0). The per-token AdaLN
# modulation is broadcast to the full [S,dim] exactly as the real streamed stack
# does (scail2_stack_train_full.scail2_modvecs_from_stream) so the block's true
# linear-in-S memory profile is honest (6 × [S,dim] F32 ≈ 4.9 GB at real S).
#
# ASSERTS: forward x_out finite, d_x finite, every LoRA grad finite and some
# nonzero, no OOM. Prints S, dense-path S² memory (for contrast), and fwd+bwd
# wall-time. Peak VRAM is captured EXTERNALLY via an nvidia-smi poll.
#
# Build (BINARY — flash needs cuDNN-shim + CUDA linkage; JIT can't cuInit):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build --optimization-level 2 -I . \
#     -Xlinker -lm -Xlinker -ldl \
#     -Xlinker -L"$CONDA_PREFIX/targets/x86_64-linux/lib/stubs" -Xlinker -lcuda \
#     -Xlinker -L"$CONDA_PREFIX/lib" -Xlinker -rpath-link -Xlinker "$CONDA_PREFIX/lib" \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     serenitymojo/models/scail2/parity/scail2_block_flash_bigS_smoke.mojo \
#     -o output/bin/scail2_block_flash_bigS_smoke
#
# Run with a VRAM poller:
#   ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; \
#       sleep 0.2; done > /tmp/vram.log & POLL=$!; \
#     LD_LIBRARY_PATH=serenitymojo/ops/cshim/lib:.pixi/envs/default/lib \
#       output/bin/scail2_block_flash_bigS_smoke; \
#     kill $POLL; echo "peak VRAM MiB:"; sort -n /tmp/vram.log | tail -1 )

from std.collections import List, Optional
from max.gpu.host import DeviceContext
from std.math import sin, cos, sqrt
from std.time import perf_counter

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.wan22.wan22_block import (
    WanBlockWeights, WanI2VBlockWeights, WanModVecs,
    WanBlockLora, WanI2VBlockLora,
)
from serenitymojo.models.scail2.scail2_block_train import (
    scail2_block_lora_forward, scail2_block_lora_backward,
)


# ── REAL SCAIL-2 i2v transformer dims (scail2_config.mojo) ──
comptime H = 40
comptime DH = 128
comptime DIM = H * DH          # 5120
comptime FFN = 13824
comptime TXT = 512
comptime IMG = 257
comptime EPS = Float32(1.0e-6)
comptime RANK = 8
comptime LSCALE = Float32(1.0)  # alpha/rank = 8/8

# LARGE sequence. Real video target (65f / 896×512) is S≈39,872. Flash KILLS the
# S² self-attn OOM (254 GB → gone), but the CHUNK-1 block's O(S) F32 terms — the
# 6×[S,dim] AdaLN mod vectors (~4.9 GB at real S) + the F32 residual/grad streams
# + the [S,FFN] hidden — still cap a SINGLE block on 16 GB. Measured on the 5080
# (2026-07-23): fwd+bwd FITS at S=16384 (peak ~14.2 GB); backward OOMs at 18432
# and 20480 (fwd alone fits to ~20480). So 16384 is the verified single-block
# max-fit; the real S=39,872 reaches it only via the 40-block STREAMING stack
# (one block resident) + a lower-memory mod-vector/residual representation.
comptime S = 16384


# ── cheap host RNG (LCG) → uniform in [-scale, scale] ──
def _rand_list(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var o = List[Float32]()
    o.reserve(n)
    var st = seed
    for _ in range(n):
        st = st * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(st >> 40)) * Float32(1.0 / 16777216.0)  # [0,1)
        o.append((u - 0.5) * 2.0 * scale)
    return o^


def _ones_list(n: Int) -> List[Float32]:
    var o = List[Float32]()
    o.reserve(n)
    for _ in range(n):
        o.append(1.0)
    return o^


def _zeros_list(n: Int) -> List[Float32]:
    var o = List[Float32]()
    o.reserve(n)
    for _ in range(n):
        o.append(0.0)
    return o^


def _make_adapter(rank: Int, in_f: Int, out_f: Int, seed: UInt64) -> LoraAdapter:
    var a = _rand_list(rank * in_f, seed, 0.04)
    var b = _rand_list(out_f * rank, seed + 12345, 0.04)
    return LoraAdapter(
        a^, b^, rank, in_f, out_f, LSCALE,
        _zeros_list(rank * in_f), _zeros_list(rank * in_f),
        _zeros_list(out_f * rank), _zeros_list(out_f * rank),
    )


def _adp(rank: Int, in_f: Int, out_f: Int, seed: UInt64) -> Optional[LoraAdapter]:
    return Optional[LoraAdapter](_make_adapter(rank, in_f, out_f, seed))


def _build_weights(ctx: DeviceContext) raises -> WanI2VBlockWeights:
    var base = WanBlockWeights(
        _rand_list(DIM * DIM, 101, 0.02), _rand_list(DIM * DIM, 102, 0.02),
        _rand_list(DIM * DIM, 103, 0.02), _rand_list(DIM * DIM, 104, 0.02),
        _zeros_list(DIM), _zeros_list(DIM), _zeros_list(DIM), _zeros_list(DIM),
        _ones_list(DIM), _ones_list(DIM),
        _rand_list(DIM * DIM, 105, 0.02), _rand_list(DIM * DIM, 106, 0.02),
        _rand_list(DIM * DIM, 107, 0.02), _rand_list(DIM * DIM, 108, 0.02),
        _zeros_list(DIM), _zeros_list(DIM), _zeros_list(DIM), _zeros_list(DIM),
        _ones_list(DIM), _ones_list(DIM),
        _ones_list(DIM), _zeros_list(DIM),
        _rand_list(FFN * DIM, 109, 0.02), _zeros_list(FFN),
        _rand_list(DIM * FFN, 110, 0.02), _zeros_list(DIM),
        DIM, FFN, DH, ctx,
    )
    return WanI2VBlockWeights(
        base^,
        _rand_list(DIM * DIM, 111, 0.02), _zeros_list(DIM),
        _rand_list(DIM * DIM, 112, 0.02), _zeros_list(DIM),
        _ones_list(DIM),
        DIM, ctx,
    )


# Broadcast e0[6*dim] + block_mod[6*dim] → 6 × [S,dim] (the real stack layout).
def _build_mod() raises -> WanModVecs:
    var e0 = _rand_list(6 * DIM, 201, 0.02)
    var bm = _rand_list(6 * DIM, 202, 0.02)
    var shift_sa = List[Float32](); shift_sa.reserve(S * DIM)
    var scale_sa = List[Float32](); scale_sa.reserve(S * DIM)
    var gate_sa = List[Float32](); gate_sa.reserve(S * DIM)
    var shift_ffn = List[Float32](); shift_ffn.reserve(S * DIM)
    var scale_ffn = List[Float32](); scale_ffn.reserve(S * DIM)
    var gate_ffn = List[Float32](); gate_ffn.reserve(S * DIM)
    for _ in range(S):
        for d in range(DIM):
            shift_sa.append(e0[0 * DIM + d] + bm[0 * DIM + d])
            scale_sa.append(e0[1 * DIM + d] + bm[1 * DIM + d])
            gate_sa.append(e0[2 * DIM + d] + bm[2 * DIM + d])
            shift_ffn.append(e0[3 * DIM + d] + bm[3 * DIM + d])
            scale_ffn.append(e0[4 * DIM + d] + bm[4 * DIM + d])
            gate_ffn.append(e0[5 * DIM + d] + bm[5 * DIM + d])
    return WanModVecs(
        shift_sa^, scale_sa^, gate_sa^, shift_ffn^, scale_ffn^, gate_ffn^
    )


def _build_lora() raises -> WanI2VBlockLora:
    var base = WanBlockLora(
        _adp(RANK, DIM, DIM, 301), _adp(RANK, DIM, DIM, 302),
        _adp(RANK, DIM, DIM, 303), _adp(RANK, DIM, DIM, 304),
        _adp(RANK, DIM, DIM, 305), _adp(RANK, DIM, DIM, 306),
        _adp(RANK, DIM, DIM, 307), _adp(RANK, DIM, DIM, 308),
        _adp(RANK, DIM, FFN, 309), _adp(RANK, FFN, DIM, 310),
    )
    return WanI2VBlockLora(base^, _adp(RANK, DIM, DIM, 311), _adp(RANK, DIM, DIM, 312))


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


def _sin_list(n: Int, phase: Float32) -> List[Float32]:
    var o = List[Float32]()
    o.reserve(n)
    for i in range(n):
        o.append(0.02 * sin(Float32(i) * phase))
    return o^


def main() raises:
    var ctx = DeviceContext()
    print("=== SCAIL-2 i2v FLASH block LARGE-S OOM-relief smoke (CHUNK 2c / G2) ===")
    print("H=", H, " DH=", DH, " DIM=", DIM, " FFN=", FFN, " TXT=", TXT,
          " IMG=", IMG, " S=", S)

    # dense-path contrast: [1,H,S,S] F32 scores that sdpa_nomask would materialize
    var dense_bytes = Float64(H) * Float64(S) * Float64(S) * 4.0
    print("  DENSE self-attn [1,H,S,S] F32 score tensor would be",
          dense_bytes / (1024.0 * 1024.0 * 1024.0), "GiB  ->  OOM on 16 GB")

    # inputs
    var x = _sin_list(S * DIM, 0.0007)
    var context_txt = _sin_list(TXT * DIM, 0.0013)
    var context_img = _sin_list(IMG * DIM, 0.0019)
    var cos_t = Tensor.from_host(_sin_list(S * (DH // 2), 0.011), [S, DH // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(_sin_list(S * (DH // 2), 0.013), [S, DH // 2], STDtype.F32, ctx)

    var w = _build_weights(ctx)
    var mv = _build_mod()
    var lora = _build_lora()
    print("  built weights + [S,dim] modulation + 12 LoRA adapters")

    var t0 = perf_counter()
    var fwd = scail2_block_lora_forward[S, TXT, IMG, H, DH, True](
        x.copy(), context_txt.copy(), context_img.copy(), mv, w, lora,
        cos_t, sin_t, DIM, FFN, EPS, ctx,
    )
    ctx.synchronize()
    var t_fwd = perf_counter() - t0
    var nf_fwd = _nonfinite(fwd.x_out)
    print("  FLASH forward done: x_out nonfinite =", nf_fwd, "  (", t_fwd, "s )")

    # upstream grad = sinusoidal
    var d_out = _sin_list(S * DIM, 0.0005)
    var t1 = perf_counter()
    var g = scail2_block_lora_backward[S, TXT, IMG, H, DH, True](
        d_out, mv, w, lora, fwd.saved, cos_t, sin_t, DIM, FFN, EPS, ctx,
    )
    ctx.synchronize()
    var t_bwd = perf_counter() - t1

    var nf_dx = _nonfinite(g.base.base.d_x)
    var nf_dctx = _nonfinite(g.base.base.d_context) + _nonfinite(g.d_context_img)

    # LoRA grad finiteness + nonzero (borrow each grad in place; no copies)
    var nf_lora = 0
    var nonzero = 0

    def _scan(v: List[Float32], mut nf: Int, mut nz: Int):
        nf += _nonfinite(v)
        for j in range(len(v)):
            if v[j] != Float32(0.0):
                nz += 1

    _scan(g.base.sa_q_da, nf_lora, nonzero); _scan(g.base.sa_q_db, nf_lora, nonzero)
    _scan(g.base.sa_k_da, nf_lora, nonzero); _scan(g.base.sa_k_db, nf_lora, nonzero)
    _scan(g.base.sa_v_da, nf_lora, nonzero); _scan(g.base.sa_v_db, nf_lora, nonzero)
    _scan(g.base.sa_o_da, nf_lora, nonzero); _scan(g.base.sa_o_db, nf_lora, nonzero)
    _scan(g.base.ca_q_da, nf_lora, nonzero); _scan(g.base.ca_q_db, nf_lora, nonzero)
    _scan(g.base.ca_k_da, nf_lora, nonzero); _scan(g.base.ca_k_db, nf_lora, nonzero)
    _scan(g.base.ca_v_da, nf_lora, nonzero); _scan(g.base.ca_v_db, nf_lora, nonzero)
    _scan(g.base.ca_o_da, nf_lora, nonzero); _scan(g.base.ca_o_db, nf_lora, nonzero)
    _scan(g.base.ffn0_da, nf_lora, nonzero); _scan(g.base.ffn0_db, nf_lora, nonzero)
    _scan(g.base.ffn2_da, nf_lora, nonzero); _scan(g.base.ffn2_db, nf_lora, nonzero)
    _scan(g.img_k_da, nf_lora, nonzero); _scan(g.img_k_db, nf_lora, nonzero)
    _scan(g.img_v_da, nf_lora, nonzero); _scan(g.img_v_db, nf_lora, nonzero)

    print("  FLASH backward done: d_x nonfinite =", nf_dx,
          "  d_context nonfinite =", nf_dctx,
          "  LoRA-grad nonfinite =", nf_lora,
          "  nonzero =", nonzero, "  (", t_bwd, "s )")
    print("  wall: fwd =", t_fwd, "s   bwd =", t_bwd, "s   fwd+bwd =",
          t_fwd + t_bwd, "s")

    var ok = (nf_fwd == 0) and (nf_dx == 0) and (nf_dctx == 0) \
        and (nf_lora == 0) and (nonzero > 0)
    print("")
    if ok:
        print("VERDICT: PASS -- FLASH block fwd+bwd at S=", S,
              "/ H=40 / DIM=5120 is finite with NO OOM (dense would need",
              dense_bytes / (1024.0 * 1024.0 * 1024.0), "GiB for scores alone)")
    else:
        print("VERDICT: FAIL -- see the metrics above")
