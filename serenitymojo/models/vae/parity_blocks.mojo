# parity_blocks.mojo — PER-BLOCK parity vs the diffusers oracle.
#
# Validates each kit block in ISOLATION: feed a block its oracle INPUT and
# compare its output to the oracle OUTPUT, so errors are not masked or
# accumulated across the stack. Stage dumps (NCHW) come from parity/gen_oracle.py
#   conv_in    : input z_rescaled  -> output conv_in       (tests conv_in)
#   mid_block  : input conv_in     -> output mid_block      (res0+attn+res1)
#   up_block_0 : input mid_block   -> output up_block_0     (3 resnets + upsample)
#   up_block_1 : input up_block_0  -> output up_block_1
#   up_block_2 : input up_block_1  -> output up_block_2     (shortcut + upsample)
#   up_block_3 : input up_block_2  -> output up_block_3     (shortcut, no upsamp)
#   head       : input up_block_3  -> output final          (gn+silu+conv_out)
#
# Each block runs in NHWC internally; we permute oracle NCHW -> NHWC at entry and
# the block output NHWC -> NCHW for the compare.
# Run: pixi run mojo run -I . serenitymojo/models/vae/parity_blocks.mojo

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.conv import conv2d
from serenitymojo.ops.norm import group_norm
from serenitymojo.ops.activations import silu
from serenitymojo.models.vae.decoder2d import (
    ResnetBlock,
    AttnBlock,
    Upsample,
    nchw_to_nhwc,
    nhwc_to_nchw,
    _load_weight,
    _load_conv_weight_rscf,
    GN_GROUPS,
    GN_EPS,
)
from serenitymojo.models.vae.vae_ops import clone


comptime VAE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/vae"
)
comptime PD = "/home/alex/mojodiffusion/serenitymojo/models/vae/parity"
comptime LH = 8
comptime LW = 8


def _read_f32_bin(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var n = file_size(fd)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var count = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(count):
        out.append(fp[i])
    buf.free()
    return out^


def _load_nchw_bf16(
    stem: String, n: Int, c: Int, h: Int, w: Int, ctx: DeviceContext
) raises -> Tensor:
    """Load an oracle NCHW dump as a BF16 GPU Tensor [n,c,h,w]."""
    var v = _read_f32_bin(String(PD) + "/" + stem + ".bin")
    var sh = List[Int]()
    sh.append(n)
    sh.append(c)
    sh.append(h)
    sh.append(w)
    return Tensor.from_host(v, sh^, STDtype.BF16, ctx)


def _check(
    name: String, out_nhwc: Tensor, ref_stem: String, ctx: DeviceContext
) raises:
    """Compare an NHWC block output to an NCHW oracle dump."""
    var out_nchw = nhwc_to_nchw(out_nhwc, ctx)
    var refv = _read_f32_bin(String(PD) + "/" + ref_stem + ".bin")
    var harness = ParityHarness(0.99)
    var res = harness.compare(out_nchw, refv, ctx)
    print("[block]", name, ":", res)


def main() raises:
    var ctx = DeviceContext()
    var st = ShardedSafeTensors.open(String(VAE_DIR))
    var p = String("decoder")

    # ── conv_in: z_rescaled [1,16,8,8] -> [1,512,8,8] ──
    var z_in = _load_nchw_bf16("z_rescaled", 1, 16, LH, LW, ctx)
    var z_nhwc = nchw_to_nhwc(z_in, ctx)
    var ciw = _load_conv_weight_rscf(st, p + ".conv_in.weight", ctx)
    var cib = _load_weight(st, p + ".conv_in.bias", ctx)
    var ci_out = conv2d[1, LH, LW, 16, 3, 3, 512, 1, 1, 1, 1](
        z_nhwc, ciw, Optional[Tensor](cib^), ctx
    )
    _check("conv_in   ", ci_out, "conv_in", ctx)

    # ── mid_block: conv_in [1,512,8,8] -> [1,512,8,8] ──
    var mid_in = _load_nchw_bf16("conv_in", 1, 512, LH, LW, ctx)
    var mid_nhwc = nchw_to_nhwc(mid_in, ctx)
    var mid_r0 = ResnetBlock[1, LH, LW, 512, 512].load(
        st, p + ".mid_block.resnets.0", ctx
    )
    var mid_attn = AttnBlock[1, LH, LW, 512].load(
        st, p + ".mid_block.attentions.0", ctx
    )
    var mid_r1 = ResnetBlock[1, LH, LW, 512, 512].load(
        st, p + ".mid_block.resnets.1", ctx
    )
    var m = mid_r0.forward(mid_nhwc, ctx)
    m = mid_attn.forward(m, ctx)
    m = mid_r1.forward(m, ctx)
    _check("mid_block ", m, "mid_block", ctx)

    # ── up_block_0: mid_block [1,512,8,8] -> [1,512,16,16] ──
    var u0_in = _load_nchw_bf16("mid_block", 1, 512, LH, LW, ctx)
    var u0_nhwc = nchw_to_nhwc(u0_in, ctx)
    var u0r0 = ResnetBlock[1, LH, LW, 512, 512].load(st, p + ".up_blocks.0.resnets.0", ctx)
    var u0r1 = ResnetBlock[1, LH, LW, 512, 512].load(st, p + ".up_blocks.0.resnets.1", ctx)
    var u0r2 = ResnetBlock[1, LH, LW, 512, 512].load(st, p + ".up_blocks.0.resnets.2", ctx)
    var u0up = Upsample[1, LH, LW, 512].load(st, p + ".up_blocks.0.upsamplers.0", ctx)
    var h0 = u0r0.forward(u0_nhwc, ctx)
    h0 = u0r1.forward(h0, ctx)
    h0 = u0r2.forward(h0, ctx)
    h0 = u0up.forward(h0, ctx)
    _check("up_block_0", h0, "up_block_0", ctx)

    # ── up_block_1: up_block_0 [1,512,16,16] -> [1,512,32,32] ──
    var u1_in = _load_nchw_bf16("up_block_0", 1, 512, 2 * LH, 2 * LW, ctx)
    var u1_nhwc = nchw_to_nhwc(u1_in, ctx)
    var u1r0 = ResnetBlock[1, 2 * LH, 2 * LW, 512, 512].load(st, p + ".up_blocks.1.resnets.0", ctx)
    var u1r1 = ResnetBlock[1, 2 * LH, 2 * LW, 512, 512].load(st, p + ".up_blocks.1.resnets.1", ctx)
    var u1r2 = ResnetBlock[1, 2 * LH, 2 * LW, 512, 512].load(st, p + ".up_blocks.1.resnets.2", ctx)
    var u1up = Upsample[1, 2 * LH, 2 * LW, 512].load(st, p + ".up_blocks.1.upsamplers.0", ctx)
    var h1 = u1r0.forward(u1_nhwc, ctx)
    h1 = u1r1.forward(h1, ctx)
    h1 = u1r2.forward(h1, ctx)
    h1 = u1up.forward(h1, ctx)
    _check("up_block_1", h1, "up_block_1", ctx)

    # ── up_block_2: up_block_1 [1,512,32,32] -> [1,256,64,64] (shortcut+upsample) ──
    var u2_in = _load_nchw_bf16("up_block_1", 1, 512, 4 * LH, 4 * LW, ctx)
    var u2_nhwc = nchw_to_nhwc(u2_in, ctx)
    var u2r0 = ResnetBlock[1, 4 * LH, 4 * LW, 512, 256].load(st, p + ".up_blocks.2.resnets.0", ctx)
    var u2r1 = ResnetBlock[1, 4 * LH, 4 * LW, 256, 256].load(st, p + ".up_blocks.2.resnets.1", ctx)
    var u2r2 = ResnetBlock[1, 4 * LH, 4 * LW, 256, 256].load(st, p + ".up_blocks.2.resnets.2", ctx)
    var u2up = Upsample[1, 4 * LH, 4 * LW, 256].load(st, p + ".up_blocks.2.upsamplers.0", ctx)
    var h2 = u2r0.forward(u2_nhwc, ctx)
    h2 = u2r1.forward(h2, ctx)
    h2 = u2r2.forward(h2, ctx)
    h2 = u2up.forward(h2, ctx)
    _check("up_block_2", h2, "up_block_2", ctx)

    # ── up_block_3: up_block_2 [1,256,64,64] -> [1,128,64,64] (shortcut, no upsample) ──
    var u3_in = _load_nchw_bf16("up_block_2", 1, 256, 8 * LH, 8 * LW, ctx)
    var u3_nhwc = nchw_to_nhwc(u3_in, ctx)
    var u3r0 = ResnetBlock[1, 8 * LH, 8 * LW, 256, 128].load(st, p + ".up_blocks.3.resnets.0", ctx)
    var u3r1 = ResnetBlock[1, 8 * LH, 8 * LW, 128, 128].load(st, p + ".up_blocks.3.resnets.1", ctx)
    var u3r2 = ResnetBlock[1, 8 * LH, 8 * LW, 128, 128].load(st, p + ".up_blocks.3.resnets.2", ctx)
    var h3 = u3r0.forward(u3_nhwc, ctx)
    h3 = u3r1.forward(h3, ctx)
    h3 = u3r2.forward(h3, ctx)
    _check("up_block_3", h3, "up_block_3", ctx)

    # ── head: up_block_3 [1,128,64,64] -> final [1,3,64,64] ──
    var hd_in = _load_nchw_bf16("up_block_3", 1, 128, 8 * LH, 8 * LW, ctx)
    var hd_nhwc = nchw_to_nhwc(hd_in, ctx)
    var now = _load_weight(st, p + ".conv_norm_out.weight", ctx)
    var nob = _load_weight(st, p + ".conv_norm_out.bias", ctx)
    var gn = group_norm(hd_nhwc, now, nob, GN_GROUPS, GN_EPS, ctx)
    gn = silu(gn, ctx)
    var cow = _load_conv_weight_rscf(st, p + ".conv_out.weight", ctx)
    var cob = _load_weight(st, p + ".conv_out.bias", ctx)
    var fin = conv2d[1, 8 * LH, 8 * LW, 128, 3, 3, 3, 1, 1, 1, 1](
        gn, cow, Optional[Tensor](cob^), ctx
    )
    _check("head      ", fin, "final", ctx)
