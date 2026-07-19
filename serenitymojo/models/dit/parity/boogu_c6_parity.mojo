# boogu_c6_parity.mojo — C6 (full DiT forward) INTEGRATION parity gate vs torch.
#
# Loads the full BooguDiT (40 blocks + 6 refiners + embedders + norm_out), feeds
# the synthetic latent/timestep/instruction dumped by boogu_c6_oracle.py, runs the
# full forward, compares the velocity [1,16,32,32] vs torch (cos >= 0.999 + magnitude).
# This is the wiring test: a wrong patchify/rope-segment/fusion/extract tanks cos.
#
# Run (oracle FIRST):
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/dit/parity/boogu_c6_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c6_parity.mojo
from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.boogu_dit import BooguDiT

comptime DUMP = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_dumps/"
comptime TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open (run boogu_c6_oracle.py first): ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing oracle dump: ") + path)
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


def main() raises:
    var ctx = DeviceContext()
    print("=== C6 (Boogu full DiT forward) INTEGRATION parity vs torch ===")

    var lat_h = _read_bin_f32(DUMP + "c6_in_latent.bin")
    var ts_h = _read_bin_f32(DUMP + "c6_in_timestep.bin")
    var ins_h = _read_bin_f32(DUMP + "c6_in_instr.bin")
    var ref_vel = _read_bin_f32(DUMP + "c6_out_velocity.bin")
    if len(lat_h) != 16 * 32 * 32 or len(ts_h) != 1 or len(ins_h) != 16 * 4096:
        raise Error("input bin sizes wrong")

    var latent = Tensor.from_host(lat_h, [1, 16, 32, 32], STDtype.BF16, ctx)
    var timestep = Tensor.from_host(ts_h, [1], STDtype.F32, ctx)
    var instr = Tensor.from_host(ins_h, [1, 16, 4096], STDtype.BF16, ctx)

    print("  loading full BooguDiT (40 blocks + refiners + embedders + norm_out)…")
    var dit = BooguDiT.load(String(TF_DIR), ctx)
    var vel = dit.forward[16, 16, 16](latent, timestep, instr, ctx)  # [1,16,32,32]

    var sh = vel.shape()
    if len(sh) != 4 or sh[1] != 16 or sh[2] != 32 or sh[3] != 32:
        raise Error("velocity shape wrong")

    var h = ParityHarness()
    var r = h.compare(vel, ref_vel, ctx)
    print("  velocity", r)
    if not r.passed:
        raise Error("C6 full-DiT integration parity FAIL (cos < 0.999)")
    print("VERDICT: C6 PASS — full DiT forward matches torch (cos >= 0.999)")
