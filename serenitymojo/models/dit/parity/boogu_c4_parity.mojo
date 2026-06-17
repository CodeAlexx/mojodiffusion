# boogu_c4_parity.mojo — C4 (double-stream block) REAL-weight parity gate vs torch.
#
# Loads double_stream_layers.0 via BooguDoubleStreamBlock, feeds the synthetic
# img/instruct/temb dumped by boogu_c4_oracle.py, runs forward[16,256], compares
# both outputs (img_out, instruct_out) to torch (cos >= 0.999 + magnitude).
#
# Run (oracle FIRST):
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/dit/parity/boogu_c4_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c4_parity.mojo
from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.boogu_dit import BooguDoubleStreamBlock

comptime DUMP = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_dumps/"
comptime TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"
comptime LINS = 16
comptime LIMG = 256
comptime HIDDEN = 3360


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open (run boogu_c4_oracle.py first): ") + path)
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
    print("=== C4 (Boogu double-stream block) REAL-weight parity vs torch ===")

    var img_h = _read_bin_f32(DUMP + "c4_in_img.bin")
    var ins_h = _read_bin_f32(DUMP + "c4_in_instruct.bin")
    var tmb_h = _read_bin_f32(DUMP + "c4_in_temb.bin")
    var ref_img = _read_bin_f32(DUMP + "c4_out_img.bin")
    var ref_ins = _read_bin_f32(DUMP + "c4_out_instruct.bin")
    if len(img_h) != LIMG * HIDDEN or len(ins_h) != LINS * HIDDEN or len(tmb_h) != 1024:
        raise Error("input bin sizes wrong")

    var img = Tensor.from_host(img_h, [1, LIMG, HIDDEN], STDtype.BF16, ctx)
    var instruct = Tensor.from_host(ins_h, [1, LINS, HIDDEN], STDtype.BF16, ctx)
    var temb = Tensor.from_host(tmb_h, [1, 1024], STDtype.BF16, ctx)

    var st = ShardedSafeTensors.open(String(TF_DIR))
    var block = BooguDoubleStreamBlock.load(st, String("double_stream_layers.0"), ctx)
    var outs = block.forward[LINS, LIMG](img, instruct, temb, 16, 16, ctx)  # (img_out, instruct_out)

    var h = ParityHarness()
    var r_img = h.compare(outs[0], ref_img, ctx)
    var r_ins = h.compare(outs[1], ref_ins, ctx)
    print("  img_out     ", r_img)
    print("  instruct_out", r_ins)
    if not (r_img.passed and r_ins.passed):
        raise Error("C4 double-stream block parity FAIL (cos < 0.999)")
    print("VERDICT: C4 PASS — double-stream block matches torch (cos >= 0.999)")
