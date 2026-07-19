# models/nldiffusion/parity/tiled_decode_probe.mojo — tiled 1024² decode gate.
#
# Gates emu3_tiled_decode (3x3 overlap + feathered blend) against the FULL
# (non-tiled) Python decode reference vq_full_decode_ref.safetensors key `pixels`
# [1,3,1024,1024] (raw float, pre-clamp) — produced by scratchpad/vq_full_ref.py
# from the same mojo_tokens.safetensors[x0_img] token grid.
#
# The whole point: this MUST NOT OOM on the 16 GB 5080 (the single-shot 1024²
# decode does). Feathered blend makes the tiled result near-identical to the
# full decode; gate cos >= 0.999.
#
# Run (JIT; RAM-heavy, no parallel compiles):
#   cd /home/alex/mojodiffusion && \
#     pixi run mojo run -I . serenitymojo/models/nldiffusion/parity/tiled_decode_probe.mojo

from std.gpu.host import DeviceContext
from std.memory import ArcPointer, alloc, UnsafePointer
from std.builtin.type_aliases import MutExternalOrigin
from std.ffi import external_call

from serenitymojo.tensor import Tensor
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.parity import ParityHarness
from serenitymojo.models.nldiffusion.emu3_vq_decoder import (
    Emu3VQDecoder,
    emu3_get_codebook_entry,
    emu3_codebook_to_z,
)
from serenitymojo.models.nldiffusion.emu3_tiled_decode import (
    emu3_decode_to_up1res,
    emu3_tiled_tail_3x3,
)

comptime VQ_FILE = "/mnt/disk1/models/NL-Diffusion-Image/emu3_vqvae/model.safetensors"
comptime PARITY_DIR = "/home/alex/mojodiffusion/serenitymojo/models/nldiffusion/parity"
comptime LH = 64      # full latent grid (64x64 -> 1024²)
comptime LW = 64
comptime TILE = 32    # LATENT/2: 3x3 512² tail tiles; only up.0+head GroupNorm is tiled


def _load_f32(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx).to_host(ctx)


comptime _CStr = UnsafePointer[UInt8, MutExternalOrigin]


def _cstr(s: String) -> _CStr:
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    return _CStr(unsafe_from_address=Int(buf))


def _setenv(name: String, val: String):
    # Enable the runtime's SYNCHRONOUS device allocator (direct cuMemAlloc that
    # frees back to the driver each op) instead of the default async pool that
    # grows to the cumulative transient peak and OOMs. Must run BEFORE the first
    # DeviceContext(). Env-var equivalent: MODULAR_DEVICE_CONTEXT_SYNC_MODE=true.
    _ = external_call["setenv", Int32](_cstr(name), _cstr(val), Int32(1))


def main() raises:
    _setenv(String("MODULAR_DEVICE_CONTEXT_SYNC_MODE"), String("true"))
    var ctx = DeviceContext()

    # ── load the Mojo token grid x0_img [1,4096] ──────────────────────────────
    print("[tiled] loading mojo_tokens.safetensors x0_img [1,4096]")
    var toks = ShardedSafeTensors.open(String(PARITY_DIR) + "/mojo_tokens.safetensors")
    var ids_f32 = _load_f32(toks, "x0_img", ctx)
    var ids = List[Int]()
    for i in range(len(ids_f32)):
        ids.append(Int(ids_f32[i] + 0.5))

    # ── codebook gather -> z NCHW [1,256,64,64] ───────────────────────────────
    var cb = emu3_get_codebook_entry(ids, VQ_FILE, ctx)   # [1,4096,256]
    var z = emu3_codebook_to_z[LH, LW](cb, ctx)           # [1,256,64,64]
    _ = cb^

    # ── deep-split tiled decode -> image NCHW [1,3,1024,1024] (must NOT OOM) ────
    # PHASE A: the full-shaped decoder runs GLOBALLY through up.1's resnets (all
    # attention + 15/21 GroupNorms over the full grid, max res 512² → fits), then
    # is FREED so only dec_tile is resident for the 1024²-producing tail.
    print("[tiled] PHASE A: full-shaped decoder global pass -> up.1 resnets (512² feature)")
    var dec_full = Emu3VQDecoder[LH, LW](VQ_FILE, ctx)
    var feat = emu3_decode_to_up1res[LH, LW](dec_full, z, ctx)  # [1,512,512,256] NHWC
    ctx.synchronize()
    _ = z^
    _ = dec_full^                                               # free 1.8GB before tail
    ctx.synchronize()
    # PHASE B: tile the 1024²-producing tail (up.1.upsample + up.0 + head); only
    # up.0's 5 GN blocks + head are tiled → tiny residual GroupNorm drift.
    print("[tiled] PHASE B: tile-shaped decoder, 3x3 512² tail tiles (up1_up+up0+head)")
    var dec_tile = Emu3VQDecoder[TILE, TILE](VQ_FILE, ctx)
    var img = emu3_tiled_tail_3x3[LH, LW, TILE, TILE](feat, dec_tile, ctx)
    _ = feat^
    var img_host = img.to_host(ctx)

    # ── compare vs full (non-tiled) reference ─────────────────────────────────
    var ref_st = ShardedSafeTensors.open(
        String(PARITY_DIR) + "/vq_full_decode_ref.safetensors"
    )
    var ref_host = _load_f32(ref_st, "pixels", ctx)
    var harness = ParityHarness(0.999)
    var res = harness.compare_host(img_host, ref_host)
    print("[tiled] tiled-vs-full : cos =", res.cos, " max_abs =", res.max_abs,
          " n =", len(img_host))
    if res.passed:
        print("[tiled] GATE PASS (cos >= 0.999) — 1024² produced with NO OOM")
    else:
        print("[tiled] GATE FAIL (cos < 0.999)")

    # ── save the Mojo tiled pixels for external PNG render ─────────────────────
    var names = List[String]()
    names.append(String("pixels"))
    var tens = List[ArcPointer[Tensor]]()
    tens.append(ArcPointer(img^))
    save_safetensors(names, tens, String(PARITY_DIR) + "/mojo_tiled_pixels.safetensors", ctx)
    print("[tiled] SAVED tiled pixels -> parity/mojo_tiled_pixels.safetensors [1,3,1024,1024]")
