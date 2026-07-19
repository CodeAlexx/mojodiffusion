# wan22_block_parity.mojo — GPU bf16 parity for ONE Wan2.2 transformer block.
#
# Drives serenitymojo/models/dit/wan22_dit.wan22_block_forward (block 0) against
# the canonical WanModel oracle (wan22_gen_oracle.py, hooked on blocks[0]). Both
# sides are fed byte-identical f32 inputs (block0_in, block0_e0, block0_context)
# and the SAME RoPE grid; the Mojo path runs in bf16 on GPU. Gate: cos >= 0.999.
#
# Run the oracle first, then the probe:
#   cd /home/alex/mojodiffusion
#   /home/alex/SimpleTuner/.venv/bin/python serenitymojo/models/dit/parity/wan22_gen_oracle.py
#   pixi run mojo run -I . serenitymojo/models/dit/parity/wan22_block_parity.mojo

from std.gpu.host import DeviceContext
from std.memory import alloc, ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.wan22_dit import (
    Wan22Config, wan22_build_rope, wan22_block_forward,
)


comptime DIR = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-bf16"

# Grid from wan22_grid.txt: F=1,H=4,W=4 -> seq_len 16; text_len 512; dim 3072.
comptime S = 16
comptime TXT = 512
comptime NH = 24
comptime HD = 128
comptime DIM = 3072
comptime F_G = 1
comptime H_G = 4
comptime W_G = 4


def _read_f32_bin(name: String) raises -> List[Float32]:
    var path = String(DIR) + name
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path + " (run the oracle first)")
    var nbytes = file_size(fd)
    if nbytes <= 0:
        _ = sys_close(fd)
        raise Error(String("empty bin: ") + path)
    var buf = alloc[UInt8](nbytes)
    var done = 0
    while done < nbytes:
        var got = sys_pread(fd, buf + done, nbytes - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var nfloats = done // 4
    var out = List[Float32]()
    var fptr = buf.bitcast[Float32]()
    for i in range(nfloats):
        out.append(fptr[i])
    buf.free()
    return out^


# Load a block-0 weight by full checkpoint name into a bf16 device Tensor.
def _load_w(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def main() raises:
    var ctx = DeviceContext()
    print("=== Wan2.2 block-0 parity (S=", S, " TXT=", TXT, ", bf16 GPU) ===")

    var cfg = Wan22Config.ti2v_5b()

    # ── Load oracle inputs (f32 bytes) -> device bf16 ──
    var xin_h = _read_f32_bin("wan22_block0_in.bin")        # [1,S,DIM]
    var e0_h = _read_f32_bin("wan22_block0_e0.bin")         # [1,S,6,DIM]
    var ctx_h = _read_f32_bin("wan22_block0_context.bin")   # [1,TXT,DIM]
    var ref_h = _read_f32_bin("wan22_block0_out.bin")       # [1,S,DIM]

    if len(xin_h) != S * DIM:
        raise Error("block0_in size mismatch")
    if len(e0_h) != S * 6 * DIM:
        raise Error("block0_e0 size mismatch")
    if len(ctx_h) != TXT * DIM:
        raise Error("block0_context size mismatch")
    if len(ref_h) != S * DIM:
        raise Error("block0_out size mismatch")

    var x_f32 = Tensor.from_host(xin_h.copy(), [1, S, DIM], STDtype.F32, ctx)
    var x_bf16 = cast_tensor(x_f32, STDtype.BF16, ctx)

    # e0 stays F32 (the modulation math is F32 in the oracle).
    var e0 = Tensor.from_host(e0_h.copy(), [1, S, 6, DIM], STDtype.F32, ctx)

    var ctx_f32 = Tensor.from_host(ctx_h.copy(), [1, TXT, DIM], STDtype.F32, ctx)
    var ctx_bf16 = cast_tensor(ctx_f32, STDtype.BF16, ctx)

    # ── RoPE tables (bf16, interleaved) for the (F,H,W) grid ──
    var cs = wan22_build_rope(F_G, H_G, W_G, HD, cfg.rope_theta, STDtype.BF16, ctx)

    # ── Load block-0 weights (keys stripped of "blocks.0." prefix) ──
    var st = ShardedSafeTensors.open(CKPT)
    var w = Dict[String, ArcPointer[Tensor]]()
    var keys = [
        "modulation",
        "self_attn.q.weight", "self_attn.q.bias",
        "self_attn.k.weight", "self_attn.k.bias",
        "self_attn.v.weight", "self_attn.v.bias",
        "self_attn.o.weight", "self_attn.o.bias",
        "self_attn.norm_q.weight", "self_attn.norm_k.weight",
        "cross_attn.q.weight", "cross_attn.q.bias",
        "cross_attn.k.weight", "cross_attn.k.bias",
        "cross_attn.v.weight", "cross_attn.v.bias",
        "cross_attn.o.weight", "cross_attn.o.bias",
        "cross_attn.norm_q.weight", "cross_attn.norm_k.weight",
        "norm3.weight", "norm3.bias",
        "ffn.0.weight", "ffn.0.bias",
        "ffn.2.weight", "ffn.2.bias",
    ]
    for kk in keys:
        var key = String(kk)
        var full = String("blocks.0.") + key
        w[key] = ArcPointer(_load_w(st, full, ctx))

    # ── Forward ──
    var out_bf16 = wan22_block_forward[S, TXT, NH, HD](
        x_bf16, e0, ctx_bf16, cs[0], cs[1], w, cfg, ctx
    )
    var out_f32 = cast_tensor(out_bf16, STDtype.F32, ctx)

    var harness = ParityHarness(0.999)
    var r = harness.compare(out_f32, ref_h, ctx)
    print("    wan22 block0 (bf16):", r)
    if r.passed:
        print("GATE PASS blockGateCos=", r.cos)
    else:
        print("GATE FAIL blockGateCos=", r.cos)
