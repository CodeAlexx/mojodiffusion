# boogu_c5_parity.mojo — C5 (norm_out) REAL-weight parity gate vs torch.
#
# Loads norm_out via BooguNormOut, feeds the synthetic hidden+temb dumped by
# boogu_c5_oracle.py, runs forward, compares [1,272,64] vs torch (cos >= 0.999).
# (The unpatchify is already self-verified bit-exact by the C5 probe round-trip.)
#
# Run (oracle FIRST):
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/dit/parity/boogu_c5_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c5_parity.mojo
from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.boogu_dit import BooguNormOut

comptime DUMP = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_dumps/"
comptime TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"
comptime SEQ = 272
comptime HIDDEN = 3360
comptime OUTDIM = 64


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open (run boogu_c5_oracle.py first): ") + path)
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
    print("=== C5 (Boogu norm_out) REAL-weight parity vs torch ===")

    var hid_h = _read_bin_f32(DUMP + "c5_in_hidden.bin")
    var tmb_h = _read_bin_f32(DUMP + "c5_in_temb.bin")
    var ref_out = _read_bin_f32(DUMP + "c5_out.bin")
    if len(hid_h) != SEQ * HIDDEN or len(tmb_h) != 1024 or len(ref_out) != SEQ * OUTDIM:
        raise Error("bin sizes wrong")
    var hidden = Tensor.from_host(hid_h, [1, SEQ, HIDDEN], STDtype.BF16, ctx)
    var temb = Tensor.from_host(tmb_h, [1, 1024], STDtype.BF16, ctx)

    var st = ShardedSafeTensors.open(String(TF_DIR))
    var norm_out = BooguNormOut.load(st, String("norm_out"), ctx)
    var y = norm_out.forward(hidden, temb, ctx)   # [1,272,64]

    var h = ParityHarness()
    var r = h.compare(y, ref_out, ctx)
    print("  norm_out", r)
    if not r.passed:
        raise Error("C5 norm_out parity FAIL (cos < 0.999)")
    print("VERDICT: C5 PASS — norm_out matches torch (cos >= 0.999)")
