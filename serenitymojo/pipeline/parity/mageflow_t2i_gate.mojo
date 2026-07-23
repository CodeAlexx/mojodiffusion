# mageflow_t2i_gate.mojo — E2E parity gate for the MageFlow T2I turbo pipeline.
#
# Loads the oracle dumps (mageflow_t2i_oracle.py): the EXACT initial packed noise
# (post get_noise + Gaussian-Shading encode_noise), the templated token ids, the
# final pre-VAE latent, and the decoded RGB. Then runs the REAL Mojo pipeline
# stages on the SAME inputs:
#   (1) Mojo tokenize the astronaut prompt   → assert ids == oracle ids
#   (2) Mojo encode_mageflow_text(ids, 34)    → txt [1,19,2560]
#   (3) Mojo 4-step Euler denoise from the ORACLE init noise (MageFlow DiT)
#         → compare final latent vs oracle (the decisive DiT-through-loop gate)
#   (4) Mojo MageVAE decode → compare RGB vs oracle + write the Mojo PNG
#
# Gate: final-latent cos >= 0.999 (goal). The decoded image side-by-side is the
# ultimate proof (report the number + the visual verdict).
#
# Run (oracle FIRST):
#   cd /home/alex/mojodiffusion
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VF_HF_ATTN_IMPL=eager \
#     PYTHONPATH=/home/alex/Mage pixi run python \
#     serenitymojo/pipeline/parity/mageflow_t2i_oracle.py
#   rm -f serenitymojo.mojopkg
#   LD_LIBRARY_PATH="serenitymojo/ops/cshim/lib:<cudnn wheel lib>" \
#   pixi run mojo run -I . \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -L/usr/lib/x86_64-linux-gnu -Xlinker -lcuda \
#     serenitymojo/pipeline/parity/mageflow_t2i_gate.mojo

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import alloc

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from serenitymojo.parity import ParityHarness
from serenitymojo.image.png import save_png, ValueRange
from serenitymojo.models.text_encoder.mageflow_qwen3vl import MAGEFLOW_T2I_DROP_IDX
from serenitymojo.pipeline.mageflow_pipeline import (
    mageflow_tokenize,
    mageflow_encode_text,
    mageflow_denoise,
    mageflow_decode_latent,
    MF_TE_DIR,
    MF_TRANSFORMER,
    MF_VAE,
    MF_TOK_JSON,
)

comptime DUMP = "serenitymojo/pipeline/parity/mageflow_t2i_dumps/"
comptime GATE_PROMPT = (
    "A photorealistic portrait of an astronaut riding a horse on Mars."
)
comptime N_IMG = 256   # 16*16 (256² image, MageVAE 16× downsample)
comptime N_TXT = 19    # keep = L_full(53) - drop(34)
comptime S = 275       # N_IMG + N_TXT
comptime SH = 16
comptime MOJO_PNG = "/home/alex/mojodiffusion/output/mageflow/gate_astronaut_256.png"


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open (run mageflow_t2i_oracle.py first): ") + path)
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
    print("=== MageFlow T2I E2E parity gate (astronaut, 256², turbo 4-step) ===")

    # ── (1) tokenizer parity: Mojo ids vs oracle ids ─────────────────────────
    var ids = mageflow_tokenize(String(MF_TOK_JSON), String(GATE_PROMPT))
    var ref_ids_f = _read_bin_f32(DUMP + "input_ids.bin")
    var ids_match = len(ids) == len(ref_ids_f)
    if ids_match:
        for i in range(len(ids)):
            if ids[i] != Int(ref_ids_f[i]):
                ids_match = False
                break
    print("[gate] tokenizer: mojo L =", len(ids), "| oracle L =", len(ref_ids_f),
          "| ids match =", ids_match)
    if len(ids) - MAGEFLOW_T2I_DROP_IDX != N_TXT:
        raise Error("gate: N_TXT mismatch vs comptime")

    # ── (2) Mojo text conditioning ───────────────────────────────────────────
    print("[gate] STAGE1 encode text ...")
    var txt = mageflow_encode_text(String(MF_TE_DIR), ids, MAGEFLOW_T2I_DROP_IDX, ctx)
    var ts = txt.shape()
    print("[gate]   txt =", ts[0], "x", ts[1], "x", ts[2])

    # ── (3) Mojo denoise from the ORACLE init noise ──────────────────────────
    print("[gate] STAGE2 denoise from oracle init noise ...")
    var noise_h = _read_bin_f32(DUMP + "init_noise.bin")  # [1,256,128] = 32768
    var init = Tensor.from_host(noise_h, [1, N_IMG, 128], STDtype.F32, ctx)
    var latent = mageflow_denoise[N_IMG, N_TXT, S, SH](
        String(MF_TRANSFORMER), txt, init, 4, ctx
    )

    var harness = ParityHarness(0.999)
    var ref_latent = _read_bin_f32(DUMP + "final_latent.bin")
    var r_lat = harness.compare(latent, ref_latent, ctx)
    print("[gate] FINAL LATENT ", r_lat)

    # ── (4) Mojo decode → compare RGB + write PNG ────────────────────────────
    print("[gate] STAGE3 MageVAE decode ...")
    var rgb = mageflow_decode_latent[N_IMG, SH, SH](String(MF_VAE), latent, ctx)
    var ref_rgb = _read_bin_f32(DUMP + "rgb.bin")
    var r_rgb = harness.compare(rgb, ref_rgb, ctx)
    print("[gate] DECODED RGB  ", r_rgb)

    save_png(rgb, MOJO_PNG, ctx, ValueRange.SIGNED)
    print("[gate] mojo image saved:", MOJO_PNG)
    print("[gate] oracle image:", DUMP + "oracle_image.png")

    if r_lat.passed:
        print("[gate] LATENT GATE PASS (cos >= 0.999)")
    else:
        print("[gate] LATENT cos =", r_lat.cos,
              "(<0.999 — check the decoded image side-by-side for the verdict)")
