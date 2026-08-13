# models/nldiffusion/parity/vq_decode_probe.mojo — CHUNK D gate.
#
# Gates the Emu3p5 VQ-VAE DECODE path (image token ids -> pixels) against
# vq_oracle.safetensors (all F32 in-file; the Emu3 VQ-VAE is fp32). Weights load
# F32 from the standalone emu3_vqvae/model.safetensors (single file, NOT sharded).
# Gate cos >= 0.999.
#
#   get_codebook_entry : ids vqdec.ids[1,256] -> [1,256,256] vs vqdec.codebook_entry
#   reshape            : -> z NCHW [1,256,16,16]           vs vqdec.z
#   decode             : post_quant_conv + Decoder -> [1,3,256,256] vs vqdec.decoded_raw
#                        also clamp(-1,1)                  vs vqdec.decoded_clamped
#
# Run (JIT; RAM-heavy, no parallel compiles):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/nldiffusion/parity/vq_decode_probe.mojo

from max.gpu.host import DeviceContext

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.nldiffusion.emu3_vq_decoder import (
    Emu3VQDecoder,
    emu3_get_codebook_entry,
    emu3_codebook_to_z,
)

comptime VQ_FILE = "/mnt/disk1/models/NL-Diffusion-Image/emu3_vqvae/model.safetensors"
comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"
comptime LH = 16
comptime LW = 16


def _load_oracle_f32(
    st: ShardedSafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _report(name: String, out_host: List[Float32], ref_host: List[Float32]) raises:
    var harness = ParityHarness(0.999)
    var res = harness.compare_host(out_host, ref_host)
    print(
        "[chunk D]", name, ": cos =", res.cos, " max_abs =", res.max_abs,
        " n =", len(out_host),
    )
    if res.passed:
        print("[chunk D]", name, "GATE PASS (cos >= 0.999)")
    else:
        print("[chunk D]", name, "GATE FAIL (cos < 0.999)")


def main() raises:
    var ctx = DeviceContext()
    var oracle = ShardedSafeTensors.open(
        String(PARITY_DIR) + "/vq_oracle.safetensors"
    )

    # ── input ids (stored F32, exact ints) ────────────────────────────────────
    print("[chunk D] loading vqdec.ids [1,256]")
    var ids_f32 = _load_oracle_f32(oracle, "vqdec.ids", ctx).to_host(ctx)
    var ids = List[Int]()
    for i in range(len(ids_f32)):
        ids.append(Int(ids_f32[i] + 0.5))

    # ── get_codebook_entry -> [1,256,256] ─────────────────────────────────────
    var cb = emu3_get_codebook_entry(ids, VQ_FILE, ctx)
    var cb_host = cb.to_host(ctx)
    var cb_ref = _load_oracle_f32(oracle, "vqdec.codebook_entry", ctx).to_host(ctx)
    _report("get_codebook_entry", cb_host, cb_ref)

    # ── reshape b (h w) d -> b d h w -> z NCHW [1,256,16,16] ───────────────────
    var z = emu3_codebook_to_z[LH, LW](cb, ctx)
    var z_host = z.to_host(ctx)
    var z_ref = _load_oracle_f32(oracle, "vqdec.z", ctx).to_host(ctx)
    _report("reshape_z", z_host, z_ref)

    # ── full decode (post_quant_conv + Decoder) -> [1,3,256,256] ───────────────
    print("[chunk D] loading Emu3VQDecoder weights (F32, ~1.8GB single file)")
    var dec = Emu3VQDecoder[LH, LW](VQ_FILE, ctx)
    var img = dec.decode(z, ctx)
    var img_host = img.to_host(ctx)
    var img_ref = _load_oracle_f32(oracle, "vqdec.decoded_raw", ctx).to_host(ctx)
    _report("decoded_raw", img_host, img_ref)

    # ── clamp(-1,1) vs decoded_clamped (host clamp) ────────────────────────────
    var clamped = List[Float32]()
    for i in range(len(img_host)):
        var v = img_host[i]
        if v < -1.0:
            v = -1.0
        elif v > 1.0:
            v = 1.0
        clamped.append(v)
    var clamped_ref = _load_oracle_f32(oracle, "vqdec.decoded_clamped", ctx).to_host(ctx)
    _report("decoded_clamped", clamped, clamped_ref)
