# serenitymojo/models/sdxl/parity/conv_lora_parity.mojo
#
# CONV-LoRA (LyCORIS LoCon) PARITY GATE vs torch.autograd. Loads the x / down / up
# / go tensors + torch delta-forward + torch conv-autograd grads written by
# conv_lora_oracle.py, builds the LoConConvAdapter from the SAME down/up, runs
# locon_forward + locon_backward, and compares at cos >= 0.999:
#   * forward Δy            (delta conv output)
#   * d_down  (spatial down-conv kernel grad)
#   * d_up    (1x1 up-conv kernel grad)
#   * d_x     (delta-path input grad — the term threaded into the frozen base
#             conv's d_x at the resblock integration layer)
# Three configs cover the SDXL conv-LoRA family kernel/stride variants:
#   k3s1p1 (conv1/conv2/conv_in/conv_out/upsampler), k3s2p1 (downsampler),
#   k1s1p0 (conv_shortcut). All share ONE LoCon code path.
#
# Run (oracle FIRST, SEPARATE command):
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sdxl/parity/conv_lora_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/sdxl/parity/conv_lora_parity.mojo -o /tmp/conv_lora_parity
#   /tmp/conv_lora_parity

from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.training.locon_conv_adapter import (
    LoConConvAdapter, locon_out_h, locon_out_w, locon_forward, locon_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/sdxl/parity/"
comptime ALPHA = Float32(4.0)


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref (run the oracle first): ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nf = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(nf):
        out.append(fp[i])
    buf.free()
    return out^


def _in(name: String) raises -> List[Float32]:
    return _read_bin_f32(REF_DIR + name + ".bin")


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _check(
    mut h: ParityHarness, name: String,
    actual: List[Float32], expected: List[Float32], mut allok: Bool,
) raises:
    var r = h.compare_host(actual, expected)
    print("  cos(", name, ") =", r.cos, "  max_abs =", r.max_abs, "  n =", r.n,
          "  ", "PASS" if r.passed else "FAIL")
    if not r.passed:
        allok = False


def _run_config(
    mut h: ParityHarness, tag: String,
    N: Int, Hi: Int, Wi: Int, Cin: Int, Cout: Int, Kh: Int, Kw: Int, rank: Int,
    sh: Int, sw: Int, ph: Int, pw: Int, mut allok: Bool, ctx: DeviceContext,
) raises:
    print("---- conv-LoRA", tag, " (N=", N, " HxW=", Hi, "x", Wi, " Cin=", Cin,
          " Cout=", Cout, " K=", Kh, "x", Kw, " rank=", rank, " s=", sh, " p=", ph, ") ----")
    var x = _in(String("conv_") + tag + String("_x"))
    var down = _in(String("conv_") + tag + String("_down"))
    var up = _in(String("conv_") + tag + String("_up"))
    var go = _in(String("conv_") + tag + String("_go"))

    var lo = LoConConvAdapter(
        down^, up^, Cin, Cout, Kh, Kw, rank, sh, sw, ph, pw, ALPHA,
        _zeros(Kh * Kw * Cin * rank), _zeros(Kh * Kw * Cin * rank),
        _zeros(rank * Cout), _zeros(rank * Cout),
    )

    var y = locon_forward(x, lo, N, Hi, Wi)
    _check(h, tag + String("_ydelta"), y, _in(String("conv_") + tag + String("_ydelta")), allok)

    var g = locon_backward(go, x, lo, N, Hi, Wi)
    _check(h, tag + String("_d_down "), g.d_down, _in(String("conv_") + tag + String("_d_down")), allok)
    _check(h, tag + String("_d_up   "), g.d_up, _in(String("conv_") + tag + String("_d_up")), allok)
    _check(h, tag + String("_d_x    "), g.d_x, _in(String("conv_") + tag + String("_d_x")), allok)
    print("")


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var allok = True
    print("==== SDXL conv-LoRA (LoCon) parity vs torch.autograd ====")
    print("")
    # MUST match conv_lora_oracle.py dims.
    _run_config(h, String("k3s1p1"), 2, 5, 4, 3, 4, 3, 3, 2, 1, 1, 1, 1, allok, ctx)
    _run_config(h, String("k3s2p1"), 1, 6, 6, 2, 3, 3, 3, 2, 2, 2, 1, 1, allok, ctx)
    _run_config(h, String("k1s1p0"), 1, 4, 4, 3, 5, 1, 1, 2, 1, 1, 0, 0, allok, ctx)
    if allok:
        print("VERDICT: PASS — SDXL conv-LoRA (LoCon) fwd + d_down/d_up/d_x match torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one conv-LoRA output diverged (see FAIL lines)")
