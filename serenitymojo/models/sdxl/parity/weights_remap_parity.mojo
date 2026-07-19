# weights_remap_parity.mojo — VALUE-level gate on the OIHW->RSCF conv-weight
# remap, on the REAL SDXL checkpoint. Closes the [LOW] skeptic gap: the existing
# smoke checks shapes only; this checks the actual element VALUES of the remapped
# RSCF buffer against a numpy-permuted reference (weights_remap_oracle.py).
#
# Loads input_blocks.4.0.in_layers.2 (3x3) + .skip_connection (1x1) via the SAME
# load_resblock_weights path the trainer uses, takes the RSCF conv1_w / skip_w
# tensors to host, and compares element-for-element against the numpy ref. A
# value SCRAMBLE (wrong axis order) would tank cos here even though shapes match.
#
# Run the oracle first, then:
#   cd /home/alex/mojodiffusion
#   /home/alex/serenityflow-v2/.venv/bin/python serenitymojo/models/sdxl/parity/weights_remap_oracle.py
#   pixi run mojo build -I . -Xlinker -lm serenitymojo/models/sdxl/parity/weights_remap_parity.mojo -o /tmp/sdxl_wr
#   /tmp/sdxl_wr

from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.models.sdxl.weights import load_resblock_weights, ResBlockWeights
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc


comptime CKPT = "/home/alex/.serenity/models/checkpoints/sdxl_unet_bf16.safetensors"
comptime REF_PATH = (
    "/home/alex/mojodiffusion/serenitymojo/models/sdxl/parity/weights_remap_ref.txt"
)


def _read_ref(tag: String) raises -> List[Float32]:
    var fd = sys_open(String(REF_PATH), O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ref: ") + String(REF_PATH))
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error("empty ref file")
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)

    var prefix = tag + " "
    var pl = prefix.byte_length()
    var out = List[Float32]()
    var i = 0
    while i < done:
        var le = i
        while le < done and Int(buf[le]) != 0x0A:
            le += 1
        var is_match = (le - i) > pl
        if is_match:
            for j in range(pl):
                if Int(buf[i + j]) != ord(prefix[byte=j]):
                    is_match = False
                    break
        if is_match:
            var p = i + pl
            while p < le:
                var c = Int(buf[p])
                if c == 0x20:
                    p += 1
                    continue
                var ne = p
                while ne < le and Int(buf[ne]) != 0x20:
                    ne += 1
                var chars = List[UInt8]()
                for q in range(p, ne):
                    chars.append(buf[q])
                var s = String(from_utf8=chars)
                out.append(Float32(atof(s)))
                p = ne + 1
            buf.free()
            return out^
        i = le + 1
    buf.free()
    raise Error(String("ref tag not found: ") + tag)


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True

    var st = SafeTensors.open(String(CKPT))
    var w4 = load_resblock_weights(st, String("input_blocks.4.0"), ctx)

    # conv1_w RSCF [3,3,320,640] — value compare vs numpy permute(2,3,1,0)
    var conv1_host = w4.conv1_w.to_host(ctx)
    var r_conv1 = h.compare_host(conv1_host, _read_ref(String("conv1")))
    print("remap conv1_w (3x3 OIHW->RSCF) vs numpy:", r_conv1)
    all_pass = all_pass and r_conv1.passed

    # skip_w RSCF [1,1,320,640]
    var skip_host = w4.skip_w.to_host(ctx)
    var r_skip = h.compare_host(skip_host, _read_ref(String("skip")))
    print("remap skip_w  (1x1 OIHW->RSCF) vs numpy:", r_skip)
    all_pass = all_pass and r_skip.passed

    print("")
    if all_pass:
        print("WEIGHTS REMAP VALUE GATE PASSED (OIHW->RSCF preserves values, cos >= 0.999)")
    else:
        print("WEIGHTS REMAP VALUE GATE FAILURE")
        raise Error("weights_remap_parity gate failed")
