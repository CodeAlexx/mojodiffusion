# parity_forward.mojo — per-stage + final parity for the Z-Image NextDiT forward
# vs the diffusers oracle (parity/gen_oracle.py).
#
# Run: pixi run mojo run -I . serenitymojo/models/dit/parity_forward.mojo
#
# Oracle dumps live in parity/*.bin (+ .shape). DEV-ONLY: Python never runs here.
# Compile-time HL/WL/CAPLEN must match the oracle invocation (default 8 8 32).

from std.gpu.host import DeviceContext
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.models.dit.zimage_dit import NextDiT


comptime XFMR_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/transformer"
)
comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/dit/parity"


def _read_f32_bin(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var n = file_size(fd)
    if n <= 0 or n % 4 != 0:
        _ = sys_close(fd)
        raise Error(String("bad bin size for ") + path)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    var count = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(count):
        out.append(fp[i])
    buf.free()
    return out^


def main() raises:
    var ctx = DeviceContext()
    comptime HL = 8
    comptime WL = 8
    comptime CAPLEN = 32

    print("[parity] loading NextDiT from", XFMR_DIR)
    var model = NextDiT[HL, WL, CAPLEN].load(String(XFMR_DIR), ctx)
    print("[parity] loaded")

    # Inputs (match oracle): image latent [1,16,HL,WL], t=0.7, cap [CAPLEN,2560].
    var img_v = _read_f32_bin(String(PARITY_DIR) + "/in_img.bin")  # (16,1,HL,WL)
    var x_sh = List[Int]()
    x_sh.append(1)
    x_sh.append(16)
    x_sh.append(HL)
    x_sh.append(WL)
    var x = Tensor.from_host(img_v, x_sh^, STDtype.BF16, ctx)

    var cap_v = _read_f32_bin(String(PARITY_DIR) + "/in_cap.bin")  # (CAPLEN,2560)
    var c_sh = List[Int]()
    c_sh.append(CAPLEN)
    c_sh.append(2560)
    var cap = Tensor.from_host(cap_v, c_sh^, STDtype.BF16, ctx)

    var t = Float32(0.7)
    var harness = ParityHarness(0.99)

    # ── per-stage parity ──
    var ref_t = _read_f32_bin(String(PARITY_DIR) + "/t_emb.bin")
    var s0 = model.debug_stage(x, t, cap, 0, ctx)
    print("[stage 0] t_emb:", harness.compare(s0, ref_t, ctx))

    var ref_cap = _read_f32_bin(String(PARITY_DIR) + "/cap_after_embedder.bin")
    var s2 = model.debug_stage(x, t, cap, 2, ctx)
    print("[stage 2] cap_after_embedder:", harness.compare(s2, ref_cap, ctx))

    # localize noise_refiner.0 modulation chunks
    var ref_sm = _read_f32_bin(String(PARITY_DIR) + "/nr0_scale_msa.bin")
    print("[nr0] scale_msa:", harness.compare(model.debug_nr0_mod(t, 0, ctx), ref_sm, ctx))
    var ref_gm = _read_f32_bin(String(PARITY_DIR) + "/nr0_gate_msa.bin")
    print("[nr0] gate_msa:", harness.compare(model.debug_nr0_mod(t, 1, ctx), ref_gm, ctx))
    var ref_smlp = _read_f32_bin(String(PARITY_DIR) + "/nr0_scale_mlp.bin")
    print("[nr0] scale_mlp:", harness.compare(model.debug_nr0_mod(t, 2, ctx), ref_smlp, ctx))
    var ref_gmlp = _read_f32_bin(String(PARITY_DIR) + "/nr0_gate_mlp.bin")
    print("[nr0] gate_mlp:", harness.compare(model.debug_nr0_mod(t, 3, ctx), ref_gmlp, ctx))

    var ref_xnr0 = _read_f32_bin(String(PARITY_DIR) + "/x_after_noise_refiner_0.bin")
    print("[stage 11] x_after_noise_refiner_0:", harness.compare(model.debug_stage(x, t, cap, 11, ctx), ref_xnr0, ctx))

    var ref_ropecos = _read_f32_bin(String(PARITY_DIR) + "/unified_rope_cos_h0.bin")
    print("[stage 13] uni RoPE cos (head 0):", harness.compare(model.debug_stage(x, t, cap, 13, ctx), ref_ropecos, ctx))

    var ref_xprep = _read_f32_bin(String(PARITY_DIR) + "/x_after_prepare.bin")
    print("[stage 17] x_seq (pre-noise-refiner):", harness.compare(model.debug_stage(x, t, cap, 17, ctx), ref_xprep, ctx))

    var ref_n1 = _read_f32_bin(String(PARITY_DIR) + "/nr0_norm1.bin")
    print("[stage 14] nr0 norm1:", harness.compare(model.debug_stage(x, t, cap, 14, ctx), ref_n1, ctx))
    var ref_n1s = _read_f32_bin(String(PARITY_DIR) + "/nr0_norm1_scaled.bin")
    print("[stage 15] nr0 norm1_scaled:", harness.compare(model.debug_stage(x, t, cap, 15, ctx), ref_n1s, ctx))
    var ref_ao = _read_f32_bin(String(PARITY_DIR) + "/nr0_attn_out.bin")
    print("[stage 16] nr0 attn_out:", harness.compare(model.debug_stage(x, t, cap, 16, ctx), ref_ao, ctx))

    var ref_xnr0real = _read_f32_bin(String(PARITY_DIR) + "/x_after_noise_refiner_0_real.bin")
    print("[stage 12] x_after_noise_refiner_0 REAL tokens:", harness.compare(model.debug_stage(x, t, cap, 12, ctx), ref_xnr0real, ctx))

    var ref_xnr = _read_f32_bin(String(PARITY_DIR) + "/x_after_noise_refiner_1.bin")
    var s3 = model.debug_stage(x, t, cap, 3, ctx)
    print("[stage 3] x_after_noise_refiner_1:", harness.compare(s3, ref_xnr, ctx))

    var ref_ccr = _read_f32_bin(String(PARITY_DIR) + "/cap_after_context_refiner_1.bin")
    var s4 = model.debug_stage(x, t, cap, 4, ctx)
    print("[stage 4] cap_after_context_refiner_1:", harness.compare(s4, ref_ccr, ctx))

    var ref_uni = _read_f32_bin(String(PARITY_DIR) + "/unified_initial.bin")
    var s5 = model.debug_stage(x, t, cap, 5, ctx)
    print("[stage 5] unified_initial:", harness.compare(s5, ref_uni, ctx))

    var ref_l0 = _read_f32_bin(String(PARITY_DIR) + "/unified_after_layer_0.bin")
    var s6 = model.debug_stage(x, t, cap, 6, ctx)
    print("[stage 6] unified_after_layer_0:", harness.compare(s6, ref_l0, ctx))

    var ref_main = _read_f32_bin(String(PARITY_DIR) + "/unified_after_main.bin")
    var s7 = model.debug_stage(x, t, cap, 7, ctx)
    print("[stage 7] unified_after_main:", harness.compare(s7, ref_main, ctx))

    var ref_fin = _read_f32_bin(String(PARITY_DIR) + "/after_final_layer.bin")
    var s8 = model.debug_stage(x, t, cap, 8, ctx)
    print("[stage 8] after_final_layer:", harness.compare(s8, ref_fin, ctx))

    # ── final forward ──
    var ref_out = _read_f32_bin(String(PARITY_DIR) + "/out.bin")
    var out = model.forward(x, t, cap, ctx)
    var osh = out.shape()
    print("[forward] out shape:", osh[0], osh[1], osh[2], osh[3])
    print("[forward] FINAL:", harness.compare(out, ref_out, ctx))
