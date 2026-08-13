# serenitymojo/models/scail2/parity/scail2_head_parity.mojo
#
# G1 PARITY GATE for the SCAIL-2 video HEAD forward + its NEW backward
#   models/scail2/scail2_stack_train_full.mojo::scail2_head_video_{forward,backward}
# vs the torch-autograd oracle scail2_head_oracle.py (float32, reduced dim, AR=0).
#
# Gates (cos>=0.999): head forward output AND d_out-into-video-region grad. Head
# weights are FROZEN, so only the input-grad path is checked.
#
# Run (oracle FIRST, SEPARATE command):
#   /home/alex/ai-toolkit/venv/bin/python \
#       serenitymojo/models/scail2/parity/scail2_head_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/scail2/parity/scail2_head_parity.mojo

from max.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.scail2.scail2_stack_train_full import (
    scail2_head_video_forward, scail2_head_video_backward,
)


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/scail2/parity/"

comptime DIM = 512
comptime OUT_DIM = 16
comptime FT = 1
comptime GH = 2
comptime GW = 2
comptime AR = 0
comptime VIDEO_OFFSET = (AR + 1) * GH * GW     # 4
comptime VIDEO_ROWS = FT * GH * GW             # 4
comptime S = VIDEO_OFFSET + VIDEO_ROWS         # 8
comptime PD = OUT_DIM * 4                       # 64
comptime EPS = Float32(1e-6)


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open (run the oracle first): ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing ref: ") + path)
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


def main() raises:
    var ctx = DeviceContext()
    print("==== scail2_head_parity (SCAIL-2 video head fwd + NEW backward) ====")
    print("DIM=", DIM, " OUT_DIM=", OUT_DIM, " FT=", FT, " GH=", GH, " GW=", GW,
          " VIDEO_ROWS=", VIDEO_ROWS, " S=", S, " AR=", AR)

    var img_h = _in("s2h_img")             # [S,DIM]
    var img_seq = Tensor.from_host(img_h.copy(), [1, S, DIM], STDtype.F32, ctx)
    var e_head = Tensor.from_host(_in("s2h_e_head"), [1, 1, DIM], STDtype.F32, ctx)
    var head_mod = Tensor.from_host(_in("s2h_head_mod"), [1, 2, DIM], STDtype.F32, ctx)
    var head_w = Tensor.from_host(_in("s2h_head_w"), [PD, DIM], STDtype.F32, ctx)
    var head_b = Tensor.from_host(_in("s2h_head_b"), [PD], STDtype.F32, ctx)
    var d_out = Tensor.from_host(_in("s2h_d_out"), [OUT_DIM, FT, GH * 2, GW * 2], STDtype.F32, ctx)

    var harness = ParityHarness()          # cos>=0.999
    var allok = True

    # ── forward ──
    var out = scail2_head_video_forward[FT, GH, GW, AR, S](
        img_seq, e_head, head_mod, head_w, head_b, OUT_DIM, DIM, EPS, ctx,
    )
    var out_h = out.to_host(ctx)
    var rf = harness.compare_host(out_h, _in("s2h_ref_out"))
    print("  cos(head forward out) =", rf.cos, "  max_abs =", rf.max_abs,
          "  ", "PASS" if rf.passed else "FAIL")
    if not rf.passed:
        allok = False

    # ── backward (video-region input grad) ──
    var video_region = List[Float32]()
    for r in range(VIDEO_ROWS):
        var srcb = (VIDEO_OFFSET + r) * DIM
        for d in range(DIM):
            video_region.append(img_h[srcb + d])
    var d_seq = scail2_head_video_backward[FT, GH, GW, AR, S](
        d_out, video_region, e_head, head_mod, head_w, OUT_DIM, DIM, EPS, ctx,
    )
    var d_video = List[Float32]()
    for r in range(VIDEO_ROWS):
        var srcb = (VIDEO_OFFSET + r) * DIM
        for d in range(DIM):
            d_video.append(d_seq[srcb + d])
    var rb = harness.compare_host(d_video, _in("s2h_ref_d_video"))
    print("  cos(d_out into video region) =", rb.cos, "  max_abs =", rb.max_abs,
          "  ", "PASS" if rb.passed else "FAIL")
    if not rb.passed:
        allok = False

    # non-video rows must be exactly zero (scatter isolates the video region).
    var nonzero_outside = 0
    for r in range(S):
        if r >= VIDEO_OFFSET and r < VIDEO_OFFSET + VIDEO_ROWS:
            continue
        for d in range(DIM):
            if d_seq[r * DIM + d] != Float32(0.0):
                nonzero_outside += 1
    print("  non-video rows nonzero count =", nonzero_outside, " (must be 0)")
    if nonzero_outside != 0:
        allok = False

    print("")
    if allok:
        print("VERDICT: PASS -- SCAIL-2 head fwd + NEW backward match torch autograd")
    else:
        print("VERDICT: FAIL -- see FAIL lines above")
