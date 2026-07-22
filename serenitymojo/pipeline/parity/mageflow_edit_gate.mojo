# mageflow_edit_gate.mojo — E2E parity gate for the MageFlow *edit* turbo pipeline.
#
# Loads the oracle dumps (mageflow_edit_oracle.py) for a FIXED (image, instruction,
# seed, steps, resolution): the exact processor pixel_values / input_ids / grid, the
# target-res ref pixel tensor, the VAE ref latent, the target init noise, the DiT
# context, the final pre-VAE target latent, and the decoded edited RGB. Then runs
# the REAL Mojo edit stages on the SAME inputs:
#   (1) Mojo encode_mageflow_edit(pixel_values, ids, grid)  → txt [1,92,2560]
#         vs oracle editcond_txt  (INFORMATIONAL: fused image-token rows are bf16
#         ill-conditioned — same property the editcond probe documented)
#   (2) Mojo mageflow_encode(ref_pixel)                      → ref tokens [1,256,128]
#         vs oracle ref_latent  (GATE cos>=0.999 — the VAE-encode-in-DiT-path)
#   (3) Mojo 4-step edit denoise from the ORACLE target noise + Mojo ref tokens,
#       concat [target,ref] each step, frame=2 msrope, target-only step
#         (a) fed the MOJO txt   → e2e final target latent  (PRIMARY gate)
#         (b) fed the ORACLE txt → isolated denoise latent  (LOCALIZATION)
#         both vs oracle final_latent (cos)
#   (4) Mojo MageVAE decode the e2e target → compare RGB + write the Mojo PNG
#
# Gate: final-latent cos >= 0.999 goal. The decoded image side-by-side is the
# ultimate proof (report the number + the visual verdict vs oracle_edit.png).
#
# Run (oracle FIRST):
#   cd /home/alex/mojodiffusion
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VF_HF_ATTN_IMPL=eager \
#     PYTHONPATH=/home/alex/Mage /home/alex/OneTrainer/venv/bin/python \
#     serenitymojo/pipeline/parity/mageflow_edit_oracle.py
#   rm -f serenitymojo.mojopkg
#   LD_LIBRARY_PATH="serenitymojo/ops/cshim/lib:<cudnn wheel lib>" \
#   pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -L/usr/lib/x86_64-linux-gnu -Xlinker -lcuda \
#     serenitymojo/pipeline/parity/mageflow_edit_gate.mojo

from std.gpu.host import DeviceContext
from std.memory import alloc

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.parity import ParityHarness
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.ops.torch_bf16 import torch_f32_to_bf16_rne
from serenitymojo.pipeline.mageflow_edit_pipeline import (
    mageflow_edit_encode_text,
    mageflow_edit_vae_encode,
    mageflow_edit_denoise,
    MFE_TE_DIR,
    MFE_TRANSFORMER,
    MFE_VAE,
)
from serenitymojo.pipeline.mageflow_pipeline import mageflow_decode_latent

comptime DUMP = "serenitymojo/pipeline/parity/mageflow_edit_dumps/"

# comptimes from the oracle SET GATE COMPTIME line:
comptime N_TXT = 92
comptime S_VIS = 288       # vision patch seq (grid 1x24x12)
comptime GRID_H = 24
comptime GRID_W = 12
comptime Lt = 256          # target tokens (SH*SH, 256²)
comptime Lr = 256          # ref tokens
comptime N_IMG = 512       # Lt + Lr
comptime S = 604           # N_IMG + N_TXT
comptime SH = 16
comptime IH = 256
comptime IW = 256
comptime MOJO_PNG = "/home/alex/mojodiffusion/output/mageflow/edit_gate_dog_256.png"


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open (run mageflow_edit_oracle.py first): ") + path)
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
    var cnt = done // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(cnt):
        out.append(fp[i])
    buf.free()
    return out^


def main() raises:
    var ctx = DeviceContext()
    var harness = ParityHarness(0.999)
    print("=== MageFlow EDIT E2E parity gate (dog+snowy-forest, 256², turbo 4-step) ===")

    # ── inputs from the oracle ──
    var ids_f = _read_bin_f32(DUMP + "input_ids.bin")
    var ids = List[Int]()
    for i in range(len(ids_f)):
        ids.append(Int(ids_f[i] + 0.5))
    if len(ids) - 64 != N_TXT:
        raise Error(String("gate: N_TXT mismatch, ids L=") + String(len(ids)))
    print("[gate] edit ids L =", len(ids), " keep(N_TXT) =", len(ids) - 64)

    var pv_h = _read_bin_f32(DUMP + "pixel_values.bin")   # [288,1536]
    var pixel_values = Tensor.from_host(pv_h, [S_VIS, 1536], STDtype.F32, ctx)

    # ── (1) Mojo edit conditioning (VL encoder freed on return) ──
    print("[gate] STAGE1 encode_mageflow_edit ...")
    var txt = mageflow_edit_encode_text[S_VIS](
        String(MFE_TE_DIR), ids, 1, GRID_H, GRID_W, pixel_values, ctx
    )
    var ts = txt.shape()
    print("[gate]   txt =", ts[0], "x", ts[1], "x", ts[2])
    var ref_txt = _read_bin_f32(DUMP + "editcond_txt.bin")
    var r_txt = harness.compare(txt, ref_txt, ctx)
    print("[gate] EDITCOND TXT (informational, bf16 ill-cond) ", r_txt)

    # ── (2) Mojo VAE-encode the ref (GATE) ──
    print("[gate] STAGE2a mageflow_encode ref ...")
    var refpx_h = _read_bin_f32(DUMP + "ref_pixel.bin")   # [3,256,256]
    var img_nchw = Tensor.from_host(refpx_h, [1, 3, IH, IW], STDtype.F32, ctx)
    var ref_tokens = mageflow_edit_vae_encode[IH, IW](String(MFE_VAE), img_nchw, ctx)
    var ref_lat_ref = _read_bin_f32(DUMP + "ref_latent.bin")   # [1,256,128]
    var r_ref = harness.compare(ref_tokens, ref_lat_ref, ctx)
    print("[gate] REF LATENT (VAE-encode-in-DiT-path) ", r_ref)

    # ── (3a) e2e denoise: Mojo txt + Mojo ref tokens + oracle target noise ──
    print("[gate] STAGE2b edit denoise (e2e: Mojo txt) ...")
    var noise_h = _read_bin_f32(DUMP + "init_noise.bin")   # [1,256,128]
    var tgt_noise = Tensor.from_host(noise_h, [1, Lt, 128], STDtype.F32, ctx)
    var latent_e2e = mageflow_edit_denoise[Lt, Lr, N_TXT, N_IMG, S, SH, SH](
        String(MFE_TRANSFORMER), txt, tgt_noise, ref_tokens, 4, ctx
    )
    var ref_final = _read_bin_f32(DUMP + "final_latent.bin")
    var r_e2e = harness.compare(latent_e2e, ref_final, ctx)
    print("[gate] FINAL LATENT (e2e, Mojo txt) ", r_e2e)

    # ── (3b) isolated denoise: ORACLE txt (localize encode vs denoise) ──
    print("[gate] STAGE2b edit denoise (isolated: oracle txt) ...")
    var otxt_h = _read_bin_f32(DUMP + "editcond_txt.bin")  # [1,92,2560]
    var otxt_f32 = Tensor.from_host(otxt_h, [1, N_TXT, 2560], STDtype.F32, ctx)
    var otxt = torch_f32_to_bf16_rne(otxt_f32, ctx)  # DiT context must be BF16
    var tgt_noise2 = Tensor.from_host(noise_h, [1, Lt, 128], STDtype.F32, ctx)
    var latent_iso = mageflow_edit_denoise[Lt, Lr, N_TXT, N_IMG, S, SH, SH](
        String(MFE_TRANSFORMER), otxt, tgt_noise2, ref_tokens, 4, ctx
    )
    var r_iso = harness.compare(latent_iso, ref_final, ctx)
    print("[gate] FINAL LATENT (isolated, oracle txt) ", r_iso)

    # ── (4) decode the e2e target → compare RGB + write PNG ──
    print("[gate] STAGE3 MageVAE decode (e2e latent) ...")
    var rgb = mageflow_decode_latent[Lt, SH, SH](String(MFE_VAE), latent_e2e, ctx)
    var ref_rgb = _read_bin_f32(DUMP + "rgb.bin")
    var r_rgb = harness.compare(rgb, ref_rgb, ctx)
    print("[gate] DECODED RGB (e2e) ", r_rgb)
    save_png(rgb, MOJO_PNG, ctx, ValueRange.SIGNED)
    print("[gate] mojo edit saved:", MOJO_PNG)
    print("[gate] oracle edit:", DUMP + "oracle_edit.png", " source:", DUMP + "input_source.png")

    print("")
    print("[gate] SUMMARY:")
    print("       ref-latent  cos =", r_ref.cos, " (VAE encode gate, goal >=0.999)")
    print("       editcond    cos =", r_txt.cos, " (informational)")
    print("       final(iso)  cos =", r_iso.cos, " (denoise math, oracle txt)")
    print("       final(e2e)  cos =", r_e2e.cos, " (full Mojo path)")
    print("       decoded rgb cos =", r_rgb.cos)
    if r_e2e.passed:
        print("[gate] E2E LATENT GATE PASS (cos >= 0.999)")
    else:
        print("[gate] e2e cos < 0.999 — check iso vs e2e to localize; the decoded")
        print("       image side-by-side is the verdict (accept ~0.997 if image matches).")
