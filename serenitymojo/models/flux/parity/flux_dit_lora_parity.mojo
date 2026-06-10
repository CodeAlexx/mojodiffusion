# flux_dit_lora_parity.mojo — GATE B-LoRA: FLUX DiT forward WITH a Kohya-BFL LoRA
# overlay, parity vs the BFL torch oracle (flux_dit_lora_oracle.py). Confirms the
# runtime additive overlay (flux_lora_overlay.mojo) is numerically CORRECT (right
# transpose + scale), not merely non-trivial.
#
# Same Gate B inputs / tiny 4x4 grid; the only difference vs flux_dit_parity is
# Flux1Offloaded.load_with_lora (multiplier 1.0). Metric: cosine vs the LoRA'd
# oracle pred. Bar: cos >= 0.99 (bf16-compute floor over 57 blocks, as Gate B).
#
# Run (oracles FIRST):
#   python3 serenitymojo/models/flux/parity/flux_dit_oracle.py 4 4 16   # inputs
#   python3 serenitymojo/models/flux/parity/flux_dit_lora_oracle.py     # LoRA pred
#   rm -f serenitymojo.mojopkg
#   pixi run mojo run -I . -Xlinker -lcuda serenitymojo/models/flux/parity/flux_dit_lora_parity.mojo

from std.math import sqrt
from std.collections import List
from std.gpu.host import DeviceContext
from serenitymojo.io.ffi import sys_open, sys_close, sys_pread, file_size, O_RDONLY
from std.memory import alloc
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.models.dit.flux1_dit import Flux1Config, Flux1Offloaded, build_flux1_rope_tables


comptime REF_DIR = "/home/alex/mojodiffusion/serenitymojo/models/flux/parity/"
comptime DIT_PATH = "/home/alex/.serenity/models/checkpoints/flux1-dev.safetensors"
comptime LORA_PATH = "/home/alex/.serenity/models/loras/Fluxass (1).safetensors"
comptime H2 = 4
comptime W2 = 4
comptime N_IMG = H2 * W2
comptime N_TXT = 16
comptime S = N_IMG + N_TXT


def _read_bin_f32(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open: ") + path)
    var n = file_size(fd)
    if n <= 0:
        _ = sys_close(fd)
        raise Error(String("empty/missing oracle (run flux_dit_lora_oracle.py first): ") + path)
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


def _abs(v: Float32) -> Float32:
    return v if v >= 0.0 else -v


def main() raises:
    var ctx = DeviceContext()
    print("=== GATE B-LoRA: FLUX DiT forward + Kohya LoRA overlay parity vs BFL ===")

    var img_h = _read_bin_f32(REF_DIR + "flux_dit_img.bin")
    var txt_h = _read_bin_f32(REF_DIR + "flux_dit_txt.bin")
    var vec_h = _read_bin_f32(REF_DIR + "flux_dit_vec.bin")
    var tg = _read_bin_f32(REF_DIR + "flux_dit_tg.bin")
    if len(img_h) != N_IMG * 64 or len(txt_h) != N_TXT * 4096 or len(vec_h) != 768 or len(tg) != 2:
        raise Error("input bin sizes wrong (run flux_dit_oracle.py 4 4 16 first)")

    var img = cast_tensor(Tensor.from_host(img_h, [1, N_IMG, 64], STDtype.F32, ctx), STDtype.BF16, ctx)
    var txt = cast_tensor(Tensor.from_host(txt_h, [1, N_TXT, 4096], STDtype.F32, ctx), STDtype.BF16, ctx)
    var vector = cast_tensor(Tensor.from_host(vec_h, [1, 768], STDtype.F32, ctx), STDtype.BF16, ctx)

    var tvals = List[Float32]()
    tvals.append(tg[0] * 1000.0)
    var t_vec = Tensor.from_host(tvals, [1], STDtype.F32, ctx)
    var gvals = List[Float32]()
    gvals.append(tg[1] * 1000.0)
    var g_vec = Tensor.from_host(gvals, [1], STDtype.F32, ctx)

    var rope = build_flux1_rope_tables[N_IMG, N_TXT, 24, 128](H2, W2, ctx, STDtype.BF16)

    print("  loading FLUX.1-dev DiT (offloaded) + LoRA overlay")
    var model = Flux1Offloaded.load_with_lora(
        DIT_PATH, Flux1Config.dev(), String(LORA_PATH), Float32(1.0), ctx
    )
    var pred = cast_tensor(
        model.forward[N_IMG, N_TXT, S](
            img, txt, t_vec, Optional[Tensor](g_vec^), vector, rope[0], rope[1], ctx,
        ),
        STDtype.F32, ctx,
    )
    var ph = pred.to_host(ctx)
    for i in range(len(ph)):
        if not (ph[i] == ph[i]) or _abs(ph[i]) > 1.0e30:
            raise Error("pred non-finite at " + String(i))

    var oracle = _read_bin_f32(REF_DIR + "flux_dit_lora_pred.bin")
    if len(oracle) != len(ph):
        raise Error("oracle/mojo size mismatch " + String(len(oracle)) + " vs " + String(len(ph)))

    var dot: Float32 = 0.0
    var na: Float32 = 0.0
    var nb: Float32 = 0.0
    var sad: Float32 = 0.0
    for i in range(len(oracle)):
        dot += ph[i] * oracle[i]
        na += ph[i] * ph[i]
        nb += oracle[i] * oracle[i]
        sad += _abs(ph[i] - oracle[i])
    var cos = dot / (sqrt(na) * sqrt(nb)) if (na > 0.0 and nb > 0.0) else Float32(1.0)
    print("  cos vs BFL+LoRA oracle =", cos)
    print("  mean-abs-diff          =", sad / Float32(len(oracle)))
    if cos < 0.99:
        raise Error("FLUX DiT LoRA-overlay parity FAIL: cos " + String(cos) + " < 0.99")
    print("VERDICT: PASS — FLUX DiT + Kohya LoRA overlay correct, cos vs BFL+LoRA =",
          cos, "(>= 0.99)")
