# serenitymojo/pipeline/parity/minimax_h3_torch_cpu_rng_probe.mojo
#
# Gates `pipeline/minimax_h3_torch_cpu_rng.mojo` against REAL PyTorch. Host
# only — no DeviceContext, no GPU, no model weights. The oracle runs torch on
# the CPU with `CUDA_VISIBLE_DEVICES=""`, so this whole gate is runnable while
# the GPU is held by another job.
#
# Oracle: scripts/minimax_h3_torch_cpu_rng_oracle.py, which dumps
# `torch.rand`/`torch.randn` under `torch.Generator().manual_seed(...)` — the
# exact call MiniMax-H3's keyframe encode makes at seed 42 (encoders.py:297).
#
# THE TWO BARS ARE DIFFERENT ON PURPOSE:
#   uniforms  BIT-EXACT. Pure integer MT19937 plus a 24-bit mask and a power-of-
#             two scale; there is no rounding freedom, so anything but equality
#             is a bug.
#   normals   MEASURED, not asserted at a guessed bar. torch's float32 normal
#             fill compiles to an AVX2 variant using polynomial log/sincos
#             (DistributionTemplates.h:88-105), so exact libm cannot match it
#             bit for bit. What this probe proves is (a) the STREAM is right —
#             the same uniforms in the same order, which the bit-exact uniform
#             check already establishes — and (b) the residual is far below the
#             float16 rounding that immediately follows in the real chain.
#
# Run:
#   CUDA_VISIBLE_DEVICES="" /home/alex/torchref/venv/bin/python \
#     scripts/minimax_h3_torch_cpu_rng_oracle.py
#   pixi run mojo build -O0 -j 1 -I . -I vendor/mojo-libs \
#     serenitymojo/pipeline/parity/minimax_h3_torch_cpu_rng_probe.mojo \
#     -o /tmp/h3_torch_cpu_rng_probe -Xlinker -lm \
#   && /tmp/h3_torch_cpu_rng_probe output/minimax_h3_keyframe/torch_cpu_rng_ref.safetensors

from std.sys import argv
from std.collections import List
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.pipeline.minimax_h3_torch_cpu_rng import (
    MiniMaxH3TorchCpuGenerator,
    minimax_h3_torch_cpu_randn,
    minimax_h3_torch_cpu_randn_from,
    minimax_h3_torch_cpu_uniform,
)


struct Report(Copyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def ok(mut self, label: String, detail: String):
        self.checks += 1
        print("  ok  ", label, "—", detail)

    def fail(mut self, label: String, detail: String):
        self.checks += 1
        self.failures += 1
        print("  FAIL", label, "—", detail)


def _read_f32(ref st: SafeTensors, name: String) raises -> List[Float32]:
    if not st.has_tensor(name):
        raise Error(String("probe: reference file has no tensor ") + name)
    var info = st.tensor_info(name)
    var tv = from_parts(info.dtype, info.shape.copy(), st.tensor_bytes(name))
    if tv.dtype != STDtype.F32:
        raise Error(String("probe: ") + name + " is not F32")
    var p = tv.data.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32](capacity=tv.numel())
    for i in range(tv.numel()):
        out.append(p[i])
    return out^


def _fp16_round(x: Float32) -> Float32:
    return Float32(Float16(x))


def main() raises:
    var args = argv()
    if len(args) < 2:
        print("usage: minimax_h3_torch_cpu_rng_probe <torch_cpu_rng_ref.safetensors>")
        return
    var ref_path = String(args[1])
    print("MiniMax-H3 keyframe unit — torch CPU RNG probe")
    print("  reference:", ref_path)
    print("")
    var report = Report()
    var st = SafeTensors.open(ref_path)

    # ── [1] uniforms: BIT-EXACT ──────────────────────────────────────────────
    print("[1] torch.rand(n, generator=Generator().manual_seed(42)) — bit-exact")
    var sizes = [16, 17, 31, 1024, 37440]
    for si in range(len(sizes)):
        var n = sizes[si]
        var want = _read_f32(st, String("uniform_") + String(n))
        var got = minimax_h3_torch_cpu_uniform(n, UInt64(42))
        var bad = 0
        var first = -1
        for i in range(n):
            if got[i] != want[i]:
                bad += 1
                if first < 0:
                    first = i
        if bad == 0:
            report.ok(
                String("uniform n=") + String(n),
                String("bit-exact over ") + String(n) + " values",
            )
        else:
            report.fail(
                String("uniform n=") + String(n),
                String(bad) + " differ, first at " + String(first) + ": got "
                + String(got[first]) + ", want " + String(want[first]),
            )

    # ── [2] normals: stream identity + measured residual ─────────────────────
    print("")
    print("[2] torch.randn(n, generator=Generator().manual_seed(42)) — measured")
    for si in range(len(sizes)):
        var n = sizes[si]
        var want = _read_f32(st, String("normal_") + String(n))
        var got = minimax_h3_torch_cpu_randn(n, UInt64(42))
        var max_abs = Float32(0.0)
        var max_rel = Float32(0.0)
        var at = -1
        var dot = Float64(0.0)
        var na = Float64(0.0)
        var nb = Float64(0.0)
        var sign_flips = 0
        for i in range(n):
            var d = got[i] - want[i]
            if d < 0:
                d = -d
            if d > max_abs:
                max_abs = d
                at = i
            var mag = want[i] if want[i] >= 0 else -want[i]
            if mag > Float32(1.0e-3):
                var r = d / mag
                if r > max_rel:
                    max_rel = r
            if (got[i] < 0) != (want[i] < 0):
                sign_flips += 1
            dot += Float64(got[i]) * Float64(want[i])
            na += Float64(got[i]) * Float64(got[i])
            nb += Float64(want[i]) * Float64(want[i])
        var cosine = dot / (sqrt(na) * sqrt(nb))
        # A wrong STREAM shows up here, not in the ulp counts: an unrelated
        # normal sequence of the same length lands at cosine ~0.
        if cosine > 0.9999999 and sign_flips == 0:
            report.ok(
                String("normal n=") + String(n),
                String("cos=") + String(cosine) + " max_abs=" + String(max_abs)
                + " max_rel=" + String(max_rel) + " sign_flips=0",
            )
        else:
            report.fail(
                String("normal n=") + String(n),
                String("cos=") + String(cosine) + " sign_flips="
                + String(sign_flips) + " max_abs=" + String(max_abs)
                + " (first at " + String(at) + ")",
            )

    # ── [3] the seed is actually honoured ────────────────────────────────────
    print("")
    print("[3] other seeds — the stream tracks manual_seed, not a constant")
    var seeds = [UInt64(0), UInt64(1), UInt64(12345)]
    for si in range(len(seeds)):
        var s = seeds[si]
        var want = _read_f32(st, String("normal_s") + String(s) + "_1024")
        var got = minimax_h3_torch_cpu_randn(1024, s)
        var dot = Float64(0.0)
        var na = Float64(0.0)
        var nb = Float64(0.0)
        for i in range(1024):
            dot += Float64(got[i]) * Float64(want[i])
            na += Float64(got[i]) * Float64(got[i])
            nb += Float64(want[i]) * Float64(want[i])
        var cosine = dot / (sqrt(na) * sqrt(nb))
        if cosine > 0.9999999:
            report.ok(String("seed ") + String(s), String("cos=") + String(cosine))
        else:
            report.fail(String("seed ") + String(s), String("cos=") + String(cosine))

    # ── [3b] SEQUENTIAL draws off one generator ──────────────────────────────
    print("")
    print("[3b] a request's own draw order: conditions, then video, then audio")
    var seq_names = [String("seq_a"), String("seq_b"), String("seq_c"), String("seq_d")]
    var seq_sizes = [24 * 1 * 30 * 52, 24 * 1 * 30 * 52, 24 * 4 * 30 * 52, 74 * 32]
    var gen = MiniMaxH3TorchCpuGenerator(UInt64(7))
    for si in range(len(seq_names)):
        var want = _read_f32(st, seq_names[si])
        var got = minimax_h3_torch_cpu_randn_from(gen, seq_sizes[si])
        var dot = Float64(0.0)
        var na = Float64(0.0)
        var nb = Float64(0.0)
        var flips = 0
        for i in range(seq_sizes[si]):
            if (got[i] < 0) != (want[i] < 0):
                flips += 1
            dot += Float64(got[i]) * Float64(want[i])
            na += Float64(got[i]) * Float64(got[i])
            nb += Float64(want[i]) * Float64(want[i])
        var cosine = dot / (sqrt(na) * sqrt(nb))
        if cosine > 0.9999999 and flips == 0:
            report.ok(
                String("draw ") + String(si) + " (" + seq_names[si] + ")",
                String("cos=") + String(cosine) + " n=" + String(seq_sizes[si]),
            )
        else:
            report.fail(
                String("draw ") + String(si) + " (" + seq_names[si] + ")",
                String("cos=") + String(cosine) + " sign_flips=" + String(flips)
                + " — the generator did not advance the way torch's does",
            )

    # A draw whose size is not a multiple of 16 burns 16 EXTRA words on the tail
    # recompute; the next draw lands wrong if that is not reproduced.
    var gen_odd = MiniMaxH3TorchCpuGenerator(UInt64(7))
    var odd_a = minimax_h3_torch_cpu_randn_from(gen_odd, 31)
    var odd_b = minimax_h3_torch_cpu_randn_from(gen_odd, 1024)
    var want_odd = _read_f32(st, String("seqodd_b"))
    _ = odd_a
    var odd_flips = 0
    var odd_dot = Float64(0.0)
    var odd_na = Float64(0.0)
    var odd_nb = Float64(0.0)
    for i in range(1024):
        if (odd_b[i] < 0) != (want_odd[i] < 0):
            odd_flips += 1
        odd_dot += Float64(odd_b[i]) * Float64(want_odd[i])
        odd_na += Float64(odd_b[i]) * Float64(odd_b[i])
        odd_nb += Float64(want_odd[i]) * Float64(want_odd[i])
    var odd_cos = odd_dot / (sqrt(odd_na) * sqrt(odd_nb))
    if odd_cos > 0.9999999 and odd_flips == 0:
        report.ok(
            "draw after a non-multiple-of-16 draw",
            String("cos=") + String(odd_cos) + " (the 16-word tail recompute is accounted for)",
        )
    else:
        report.fail(
            "draw after a non-multiple-of-16 draw",
            String("cos=") + String(odd_cos) + " sign_flips=" + String(odd_flips),
        )

    # ── [4] what the residual costs AFTER the fp16 round the chain applies ───
    print("")
    print("[4] survival across the float16 round the keyframe encode applies next")
    var n4 = 37440
    var want4 = _read_f32(st, String("normal_") + String(n4))
    var got4 = minimax_h3_torch_cpu_randn(n4, UInt64(42))
    var changed = 0
    var max_fp16_gap = Float32(0.0)
    for i in range(n4):
        var a = _fp16_round(got4[i])
        var b = _fp16_round(want4[i])
        if a != b:
            changed += 1
            var d = a - b
            if d < 0:
                d = -d
            if d > max_fp16_gap:
                max_fp16_gap = d
    var pct = Float64(changed) * 100.0 / Float64(n4)
    report.ok(
        "fp16-rounded disagreement",
        String(changed) + " of " + String(n4) + " values (" + String(pct)
        + "%), max gap " + String(max_fp16_gap),
    )

    print("")
    if report.failures == 0:
        print("PASS:", report.checks, "checks")
    else:
        print("FAIL:", report.failures, "of", report.checks, "checks")
        raise Error("minimax_h3_torch_cpu_rng probe FAILED")
