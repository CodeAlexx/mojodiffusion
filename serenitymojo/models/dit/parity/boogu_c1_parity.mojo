# boogu_c1_parity.mojo — C1 (embedders) REAL-weight parity gate vs torch oracle.
#
# Reads the deterministic inputs + reference outputs dumped by boogu_c1_oracle.py
# (raw little-endian F32 .bin), loads the REAL Boogu transformer embedder weights
# via BooguEmbedders.load, runs x_embed + time_caption_embed, and compares to the
# torch references via ParityHarness (cos >= 0.999).
#
# Run (oracle FIRST, separate command, never chained after a mojo build):
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/dit/parity/boogu_c1_oracle.py
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . serenitymojo/models/dit/parity/boogu_c1_parity.mojo
from std.gpu.host import DeviceContext
from std.collections import List
from std.memory import alloc
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.models.dit.boogu_dit import BooguEmbedders

comptime DUMP = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity/boogu_dumps/"
comptime TF_DIR = "/home/alex/Boogu-Image/models/Boogu-Image-0.1-Base/transformer"


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open (run boogu_c1_oracle.py first): ") + path)
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
    print("=== C1 (Boogu embedders) REAL-weight parity vs torch ===")

    # Inputs (dumped F32; instr/tokens were bf16 in the model -> upload as BF16).
    var ts_h = _read_bin_f32(DUMP + "c1_in_timestep.bin")
    var instr_h = _read_bin_f32(DUMP + "c1_in_instr.bin")
    var tok_h = _read_bin_f32(DUMP + "c1_in_tokens.bin")
    if len(ts_h) != 1 or len(instr_h) != 16 * 4096 or len(tok_h) != 256 * 64:
        raise Error("input bin sizes wrong")
    var timestep = Tensor.from_host(ts_h, [1], STDtype.F32, ctx)
    var instr = Tensor.from_host(instr_h, [1, 16, 4096], STDtype.BF16, ctx)
    var tokens = Tensor.from_host(tok_h, [1, 256, 64], STDtype.BF16, ctx)

    # Real weights.
    var emb = BooguEmbedders.load(String(TF_DIR), ctx)
    var xembed = emb.x_embed(tokens, ctx)                       # [1,256,3360]
    var tc = emb.time_caption_embed(timestep, instr, ctx)       # (temb[1,1024], caption[1,16,3360])

    # References.
    var ref_temb = _read_bin_f32(DUMP + "c1_out_temb.bin")
    var ref_cap = _read_bin_f32(DUMP + "c1_out_caption.bin")
    var ref_xemb = _read_bin_f32(DUMP + "c1_out_xembed.bin")

    var h = ParityHarness()
    var r_temb = h.compare(tc[0], ref_temb, ctx)
    var r_cap = h.compare(tc[1], ref_cap, ctx)
    var r_xemb = h.compare(xembed, ref_xemb, ctx)
    print("  temb    ", r_temb)
    print("  caption ", r_cap)
    print("  xembed  ", r_xemb)

    if not (r_temb.passed and r_cap.passed and r_xemb.passed):
        raise Error("C1 parity FAIL (cos < 0.999 on one or more embedder outputs)")
    print("VERDICT: C1 PASS — embedders match torch (cos >= 0.999 on temb/caption/xembed)")
