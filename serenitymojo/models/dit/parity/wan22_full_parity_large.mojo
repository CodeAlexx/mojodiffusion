# wan22_full_parity_large.mojo — GPU bf16 parity for the FULL Wan2.2 DiT forward
# at a LARGE video grid (S=4096) that exercises the tiled (online-softmax) SDPA.
#
# Grid: latent F=4,H=64,W=64 -> patch (1,2,2) -> grid (4,32,32) -> S=4096.
# At head_dim=128 the math-mode SDPA would materialize [1,24,4096,4096] f32
# scores (~1.6 GB/block) and OOM; wan22_block_forward dispatches S>512 to
# sdpa_nomask_tiled. Deep 30-block chain -> gate cos >= 0.99.
#
# Run the large oracle first, then this probe:
#   cd /home/alex/mojodiffusion
#   /home/alex/SimpleTuner/.venv/bin/python serenitymojo/models/dit/parity/wan22_gen_oracle_large.py
#   pixi run mojo run -I . serenitymojo/models/dit/parity/wan22_full_parity_large.mojo

from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.wan22_dit import Wan22Config, Wan22DiT


comptime DIR = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/"
comptime CKPT = "/home/alex/.serenity/models/checkpoints/Wan2.2-TI2V-5B-bf16"

# Large grid: latent F=4,H=64,W=64 -> patch grid (4,32,32) -> S=4096.
comptime IN_DIM = 48
comptime F_LAT = 4
comptime H_LAT = 64
comptime W_LAT = 64
comptime FG = 4
comptime HG = 32
comptime WG = 32
comptime S = 4096
comptime TXT = 512
comptime CTXL = 12
comptime TEXT_DIM = 4096
comptime NH = 24
comptime HD = 128
comptime TIMESTEP = Float32(500.0)


def _read_f32_bin(name: String) raises -> List[Float32]:
    var path = String(DIR) + name
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path + " (run the large oracle first)")
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


def main() raises:
    var ctx = DeviceContext()
    print("=== Wan2.2 FULL forward parity LARGE (S=", S, ", 30 blocks, tiled SDPA, bf16 GPU) ===")

    var cfg = Wan22Config.ti2v_5b()

    var x_h = _read_f32_bin("wan22_full_x_in_large.bin")          # [IN_DIM,F,H,W]
    var ctx_h = _read_f32_bin("wan22_full_context_raw_large.bin") # [CTXL,TEXT_DIM]
    var ref_h = _read_f32_bin("wan22_full_out_large.bin")         # [IN_DIM,F,H,W]

    if len(x_h) != IN_DIM * F_LAT * H_LAT * W_LAT:
        raise Error("full_x_in_large size mismatch")
    if len(ctx_h) != CTXL * TEXT_DIM:
        raise Error("full_context_raw_large size mismatch")
    if len(ref_h) != IN_DIM * F_LAT * H_LAT * W_LAT:
        raise Error("full_out_large size mismatch")

    var x_f32 = Tensor.from_host(x_h.copy(), [IN_DIM, F_LAT, H_LAT, W_LAT], STDtype.F32, ctx)
    var x_bf = cast_tensor(x_f32, STDtype.BF16, ctx)
    var ctx_f32 = Tensor.from_host(ctx_h.copy(), [CTXL, TEXT_DIM], STDtype.F32, ctx)
    var ctx_bf = cast_tensor(ctx_f32, STDtype.BF16, ctx)

    print("    loading Wan2.2-TI2V-5B weights (bf16, resident)...")
    var model = Wan22DiT.load(CKPT, cfg, ctx)
    print("    weights loaded; running 30-block forward at S=", S, " (tiled SDPA)...")

    var out_bf = model.forward[FG, HG, WG, S, TXT, CTXL, NH, HD](
        x_bf, TIMESTEP, ctx_bf, ctx
    )
    var out_f32 = cast_tensor(out_bf, STDtype.F32, ctx)

    var harness = ParityHarness(0.99)
    var r = harness.compare(out_f32, ref_h, ctx)
    print("    wan22 LARGE full forward (bf16, tiled):", r)
    if r.passed:
        print("GATE PASS largeFullForwardCos=", r.cos)
    else:
        print("GATE FAIL largeFullForwardCos=", r.cos)
