# boogu_c3_parity.mojo — C3 (single-stream block) REAL-weight parity gate vs torch.
#
# Loads single_stream_layers.0 via BooguBlock, builds rope (build_boogu_rope_tables
# 16/16/16 -> seq 272), feeds the synthetic hidden+temb dumped by boogu_c3_oracle.py,
# runs forward[272], compares to the torch block output (cos >= 0.999 + magnitude).
#
# Run (oracle FIRST):
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/dit/parity/boogu_c3_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c3_parity.mojo
from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.boogu_dit import BooguBlock, build_boogu_rope_tables

comptime DUMP = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_dumps/"
comptime TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"
comptime SEQ = 272
comptime HIDDEN = 3360


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open (run boogu_c3_oracle.py first): ") + path)
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
    print("=== C3 (Boogu single-stream block) REAL-weight parity vs torch ===")

    var hid_h = _read_bin_f32(DUMP + "c3_in_hidden.bin")
    var tmb_h = _read_bin_f32(DUMP + "c3_in_temb.bin")
    var ref_out = _read_bin_f32(DUMP + "c3_out.bin")
    if len(hid_h) != SEQ * HIDDEN or len(tmb_h) != 1024 or len(ref_out) != SEQ * HIDDEN:
        raise Error("bin sizes wrong")
    var hidden = Tensor.from_host(hid_h, [1, SEQ, HIDDEN], STDtype.BF16, ctx)
    var temb = Tensor.from_host(tmb_h, [1, 1024], STDtype.BF16, ctx)

    var st = ShardedSafeTensors.open(String(TF_DIR))
    var block = BooguBlock.load(st, String("single_stream_layers.0"), True, ctx)
    var tables = build_boogu_rope_tables(16, 16, 16, ctx)   # (cos[272,60], sin[272,60])

    var y = block.forward[SEQ](hidden, temb, tables[0], tables[1], ctx)  # [1,272,3360]

    var h = ParityHarness()
    var r = h.compare(y, ref_out, ctx)
    print("  block out", r)
    if not r.passed:
        raise Error("C3 block parity FAIL (cos < 0.999)")
    print("VERDICT: C3 PASS — single-stream block matches torch (cos >= 0.999)")
