# serenitymojo/models/sdxl/parity/embed_lora_parity.mojo
#
# EMBED/PROJ LINEAR-LoRA PARITY GATE vs torch.autograd. Loads x/A/B/Wbase/go +
# torch forward (base+lora) + torch LoRA-branch grads written by
# embed_lora_oracle.py, builds a LoraAdapter from the SAME A/B, runs the EXACT
# wiring primitive (sdxl_lora_apply forward + sdxl_lora_bwd backward) the SDXL
# embed/proj LoRA would use, and compares at cos>=0.999:
#   * forward y_full (base + LoRA)
#   * d_A, d_B (the trained grads), d_x (LoRA-branch input grad)
# Covers time_embedding.linear_1/2, add_embedding.linear_1/2, resnet
# time_emb_proj — all one linear-LoRA primitive at rectangular embed dims.
#
# Run (oracle FIRST): /home/alex/serenityflow-v2/.venv/bin/python \
#       serenitymojo/models/sdxl/parity/embed_lora_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -I . -Xlinker -lm \
#       serenitymojo/models/sdxl/parity/embed_lora_parity.mojo -o /tmp/embed_lora_parity
#   /tmp/embed_lora_parity

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import alloc
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.sdxl.lora_block import sdxl_lora_apply, sdxl_lora_bwd


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/sdxl/parity/"
comptime RANK = 4
comptime ALPHA = Float32(8.0)


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


def _run(
    mut h: ParityHarness, tag: String, M: Int, in_f: Int, out_f: Int,
    mut allok: Bool, ctx: DeviceContext,
) raises:
    print("---- embed-LoRA", tag, " (M=", M, " in=", in_f, " out=", out_f, " rank=", RANK, ") ----")
    var scale = ALPHA / Float32(RANK)
    var x_h = _in(String("emb_") + tag + String("_x"))
    var a_h = _in(String("emb_") + tag + String("_A"))
    var b_h = _in(String("emb_") + tag + String("_B"))
    var wb_h = _in(String("emb_") + tag + String("_Wbase"))
    var go_h = _in(String("emb_") + tag + String("_go"))

    var adapter = LoraAdapter(
        a_h^, b_h^, RANK, in_f, out_f, scale,
        _zeros(RANK * in_f), _zeros(RANK * in_f), _zeros(out_f * RANK), _zeros(out_f * RANK),
    )

    var xs = List[Int](); xs.append(M); xs.append(in_f)
    var x = Tensor.from_host(x_h.copy(), xs^, STDtype.F32, ctx)
    var ws = List[Int](); ws.append(out_f); ws.append(in_f)
    var wbase = Tensor.from_host(wb_h^, ws^, STDtype.F32, ctx)

    var nb = Optional[Tensor](None)
    var base_y = linear(x.clone(ctx), wbase, nb^, ctx)             # [M,out]
    var y = sdxl_lora_apply(base_y, x.clone(ctx), Optional[LoraAdapter](adapter.copy()), M, out_f, ctx)
    _check(h, tag + String("_yfull"), y.to_host(ctx), _in(String("emb_") + tag + String("_yfull")), allok)

    var g = sdxl_lora_bwd(go_h.copy(), x_h.copy(), adapter, M, ctx)
    _check(h, tag + String("_dA   "), g.d_a, _in(String("emb_") + tag + String("_dA")), allok)
    _check(h, tag + String("_dB   "), g.d_b, _in(String("emb_") + tag + String("_dB")), allok)
    _check(h, tag + String("_dx   "), g.d_x, _in(String("emb_") + tag + String("_dx")), allok)
    print("")


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var allok = True
    print("==== SDXL embed/proj LINEAR-LoRA parity vs torch.autograd ====")
    print("")
    _run(h, String("te1"), 4, 12, 20, allok, ctx)
    _run(h, String("te2"), 3, 20, 16, allok, ctx)
    _run(h, String("add1"), 2, 28, 20, allok, ctx)
    if allok:
        print("VERDICT: PASS — SDXL embed/proj linear-LoRA fwd + d_A/d_B/d_x match torch (cos>=0.999)")
    else:
        print("VERDICT: FAIL — at least one embed-LoRA output diverged (see FAIL lines)")
