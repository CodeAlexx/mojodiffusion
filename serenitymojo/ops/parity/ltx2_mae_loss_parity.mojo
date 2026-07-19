# ltx2_mae_loss_parity.mojo — torch L1 (MAE) parity for ops/loss_fns.mojo
# mae_loss_grad (added for the ltx2 loss lever loss_type mae|l1).
#
# Reads the byte-identical inputs the oracle dumped, runs mae_loss_grad, and
# compares the loss scalar + the full d_pred vector to torch's l1_loss value +
# autograd. loss = mean(|d|), grad = sign(d)/N — pure host F32 (grad is
# rational so bit-exact; the loss reduction is F64-accumulated here vs torch F32,
# a ~1e-6 class). Bar: abs <= 1e-5.
#
# Run the oracle first, then the gate:
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/ops/parity/ltx2_mae_loss_oracle.py
#   pixi run mojo run -I . serenitymojo/ops/parity/ltx2_mae_loss_parity.mojo

from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.ops.loss_fns import mae_loss_grad


comptime DIR = "/home/alex/mojodiffusion/serenitymojo/ops/parity/"
comptime N = 512
comptime BAR = Float32(1.0e-5)


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


def main() raises:
    var pred = _read_f32_bin("ltx2_mae_pred.bin")
    var target = _read_f32_bin("ltx2_mae_target.bin")
    var ref_loss = _read_f32_bin("ltx2_mae_loss.bin")
    var ref_grad = _read_f32_bin("ltx2_mae_grad.bin")
    if len(pred) != N or len(target) != N or len(ref_grad) != N or len(ref_loss) != 1:
        raise Error("ltx2_mae_loss_parity: input/ref size mismatch with oracle")

    var lg = mae_loss_grad(pred, target)

    var loss_err = abs(lg.loss - ref_loss[0])
    if loss_err > BAR:
        print("  FAIL loss  mojo=", lg.loss, " torch=", ref_loss[0], " abs=", loss_err)
        raise Error("[ltx2-mae-parity] FAIL: loss abs > 1e-5")

    if len(lg.d_pred) != N:
        raise Error("[ltx2-mae-parity] d_pred length mismatch")
    var worst_grad = Float32(0.0)
    for i in range(N):
        var e = abs(lg.d_pred[i] - ref_grad[i])
        if e > worst_grad:
            worst_grad = e
        if e > BAR:
            print("  FAIL grad[", i, "] mojo=", lg.d_pred[i], " torch=", ref_grad[i], " abs=", e)
            raise Error("[ltx2-mae-parity] FAIL: grad abs > 1e-5")

    print("[ltx2-mae-parity] N=", N, " loss mojo=", lg.loss, " torch=", ref_loss[0],
          " abs=", loss_err)
    print("[ltx2-mae-parity] grad worst abs=", worst_grad, " (bar 1e-5)")
    print("ALL PASS")
