# parity_blkdbg.mojo — block-by-block DiT parity at the REAL divergence point
# (lat_step_27, cond, t=0.82353). Compares each named intermediate of my
# NextDiT against the GPU-bf16 diffusers oracle (blkdbg/<name>.bin) via
# ParityHarness cos. Walks stages in forward order to localize where cos drops
# below ~0.9998 (the bad op) vs gradual bf16 accumulation.
#
# Run: cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#        serenitymojo/pipeline/parity/parity_blkdbg.mojo
from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.models.dit.zimage_dit import NextDiT
from serenitymojo.parity import ParityHarness

comptime TRANSFORMER = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
)
comptime PD = "/home/alex/mojodiffusion/serenitymojo/pipeline/parity"
comptime BD = "/home/alex/mojodiffusion/serenitymojo/pipeline/parity/blkdbg"
comptime HL = 32
comptime WL = 32
comptime CAPLEN = 173
comptime HIDDEN = 2560
comptime T0 = Float32(0.82353)


def _read(path: String) raises -> List[Float32]:
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
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(n // 4):
        out.append(fp[i])
    buf.free()
    return out^


# truncate a host ref to the first `n` elements (for cap_after_embedder pad).
def _head(v: List[Float32], n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(v[i])
    return out^


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness(0.9998)

    var x = Tensor.from_host(_read(String(PD) + "/lat_step_27.bin"), [1, 16, HL, WL], STDtype.BF16, ctx)
    var cap = Tensor.from_host(_read(String(PD) + "/cond.bin"), [CAPLEN, HIDDEN], STDtype.BF16, ctx)
    var dit = NextDiT[HL, WL, CAPLEN].load(TRANSFORMER, ctx)

    print("=== Z-Image DiT block-by-block parity (lat_step_27, t=", T0, ") ===")
    print("    threshold cos>=0.9998 ; ref=GPU bf16 diffusers oracle (blkdbg/)")
    print("")

    # ── stage walk (forward order) ──
    # (stage code, oracle filename, head_truncate or -1)
    # t_emb
    var s0 = dit.debug_stage(x, T0, cap, 0, ctx)
    print("  [0]  t_emb                      ", h.compare(s0, _read(String(BD) + "/t_emb.bin"), ctx))

    var s1 = dit.debug_stage(x, T0, cap, 1, ctx)  # [img_tokens,dim]=[256,3840]
    print("  [1]  x_after_embedder           ", h.compare(s1, _read(String(BD) + "/x_after_embedder.bin"), ctx))

    var s17 = dit.debug_stage(x, T0, cap, 17, ctx)  # [1,256,3840]
    print("  [17] x_after_prepare            ", h.compare(s17, _read(String(BD) + "/x_after_prepare.bin"), ctx))

    # cap_after_embedder: Mojo [173,3840], oracle [192,3840] -> head 173*3840
    var s2 = dit.debug_stage(x, T0, cap, 2, ctx)
    var cae = _read(String(BD) + "/cap_after_embedder.bin")
    print("  [2]  cap_after_embedder (h173)  ", h.compare(s2, _head(cae, CAPLEN * 3840), ctx))

    # noise_refiner.0 sub-steps
    var s14 = dit.debug_stage(x, T0, cap, 14, ctx)
    print("  [14] nr0_norm1                  ", h.compare(s14, _read(String(BD) + "/nr0_norm1.bin"), ctx))
    var s15 = dit.debug_stage(x, T0, cap, 15, ctx)
    print("  [15] nr0_norm1_scaled           ", h.compare(s15, _read(String(BD) + "/nr0_norm1_scaled.bin"), ctx))
    var s16 = dit.debug_stage(x, T0, cap, 16, ctx)
    print("  [16] nr0_attn_out               ", h.compare(s16, _read(String(BD) + "/nr0_attn_out.bin"), ctx))

    var s11 = dit.debug_stage(x, T0, cap, 11, ctx)
    print("  [11] x_after_noise_refiner_0    ", h.compare(s11, _read(String(BD) + "/x_after_noise_refiner_0.bin"), ctx))
    var s3 = dit.debug_stage(x, T0, cap, 3, ctx)
    print("  [3]  x_after_noise_refiner_1    ", h.compare(s3, _read(String(BD) + "/x_after_noise_refiner_1.bin"), ctx))

    var s4 = dit.debug_stage(x, T0, cap, 4, ctx)
    print("  [4]  cap_after_context_refiner_1", h.compare(s4, _read(String(BD) + "/cap_after_context_refiner_1.bin"), ctx))

    var s5 = dit.debug_stage(x, T0, cap, 5, ctx)
    print("  [5]  unified_initial            ", h.compare(s5, _read(String(BD) + "/unified_initial.bin"), ctx))

    # main layers 0..29
    for li in range(30):
        var sl = dit.debug_main_layer(x, T0, cap, li, ctx)
        print("  [L", li, "] unified_after_layer       ",
              h.compare(sl, _read(String(BD) + "/unified_after_layer_" + String(li) + ".bin"), ctx))

    var s8 = dit.debug_stage(x, T0, cap, 8, ctx)
    print("  [8]  after_final_layer          ", h.compare(s8, _read(String(BD) + "/after_final_layer.bin"), ctx))

    var s9 = dit.debug_stage(x, T0, cap, 9, ctx)
    print("  [9]  out (velocity)             ", h.compare(s9, _read(String(BD) + "/out.bin"), ctx))
