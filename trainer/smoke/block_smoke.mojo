# block_smoke.mojo — COMPILE smoke for the Z-Image block unit. Exercises the
# public surface of block.mojo / sigma_map.mojo / lora_targets.mojo / weights.mojo
# so the orchestrator's integrated build type-checks the whole unit. Build-only
# (GPU-free build per unit instructions). Do NOT run.
#
# Uses a SMALL comptime S so the smoke compiles fast; the block ops are
# comptime-shaped on [S, H=30, Dh=128, dim=3840] (real Z-Image main-block dims).

from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.tensor_algebra import zeros_device

from serenity_trainer.module.LoRAModule import make_lora_adapter
from serenity_trainer.modelLoader.ZImageModelLoader import (
    ZImageWeights, ZImageBlockKeys, block_prefix,
)
from serenity_trainer.model.ZImageModel import (
    ZH, ZDh, ZDIM, LArc,
    zimage_block_forward, zimage_block_backward,
    ZImageBlockActs, ZImageBlockOut, ZImageBlockBwd, ZImageBlockLoraGrads,
)
from serenity_trainer.modelSetup import ZImageLoRASetup as LT
from serenity_trainer.modelSetup.BaseZImageSetup import (
    sigma_from_timestep, model_t_from_timestep, timestep_from_sigma,
    snr_from_sigma, ZIMAGE_NUM_TRAIN_TIMESTEPS,
)


comptime TArc = ArcPointer[Tensor]
comptime FF_HIDDEN = 2560  # arbitrary smoke ff width (real = ~10240)


def _w2(a: Int, b: Int) -> List[Int]:
    var s = List[Int](); s.append(a); s.append(b); return s^

def _w1(a: Int) -> List[Int]:
    var s = List[Int](); s.append(a); return s^


# Build a synthetic frozen weight store for ONE main block (block 0) with all
# keys the block forward/backward reads (zeros — smoke is build-only).
def _build_synth_weights(ctx: DeviceContext) raises -> ZImageWeights:
    var weights = List[TArc]()
    var name_to_idx = Dict[String, Int]()
    var names = List[String]()
    var shapes = List[List[Int]]()

    var keys = ZImageBlockKeys.for_block(0)
    # adaLN: Linear adaln_embed(256) -> 4*dim ; weight [4*dim, 256], bias [4*dim]
    names.append(keys.adaln_w());    shapes.append(_w2(4 * ZDIM, 256))
    names.append(keys.adaln_b());    shapes.append(_w1(4 * ZDIM))
    # RMSNorm weights [dim]
    names.append(keys.attn_norm1()); shapes.append(_w1(ZDIM))
    names.append(keys.attn_norm2()); shapes.append(_w1(ZDIM))
    names.append(keys.ffn_norm1());  shapes.append(_w1(ZDIM))
    names.append(keys.ffn_norm2());  shapes.append(_w1(ZDIM))
    # per-head RMSNorm [Dh]
    names.append(keys.norm_q());     shapes.append(_w1(ZDh))
    names.append(keys.norm_k());     shapes.append(_w1(ZDh))
    # attention projections [dim,dim]
    names.append(keys.to_q());       shapes.append(_w2(ZDIM, ZDIM))
    names.append(keys.to_k());       shapes.append(_w2(ZDIM, ZDIM))
    names.append(keys.to_v());       shapes.append(_w2(ZDIM, ZDIM))
    names.append(keys.to_out());     shapes.append(_w2(ZDIM, ZDIM))
    # feed forward: w1/w3 [ff,dim], w2 [dim,ff]
    names.append(keys.ff_w1());      shapes.append(_w2(FF_HIDDEN, ZDIM))
    names.append(keys.ff_w3());      shapes.append(_w2(FF_HIDDEN, ZDIM))
    names.append(keys.ff_w2());      shapes.append(_w2(ZDIM, FF_HIDDEN))

    for i in range(len(names)):
        var t = zeros_device(shapes[i].copy(), STDtype.BF16, ctx)
        weights.append(ArcPointer(t^))
        name_to_idx[names[i]] = i

    return ZImageWeights(weights^, name_to_idx^)


# Build the 7 per-block LoRA adapters in slot order (boxed, move-only).
def _build_loras(rank: Int, alpha: Float32, ctx: DeviceContext) raises -> List[LArc]:
    var loras = List[LArc]()
    var seed = UInt64(1)
    # to_q,to_k,to_v,to_out: in=dim,out=dim
    loras.append(LArc(make_lora_adapter(ZDIM, ZDIM, rank, alpha, seed + 0, ctx)))
    loras.append(LArc(make_lora_adapter(ZDIM, ZDIM, rank, alpha, seed + 1, ctx)))
    loras.append(LArc(make_lora_adapter(ZDIM, ZDIM, rank, alpha, seed + 2, ctx)))
    loras.append(LArc(make_lora_adapter(ZDIM, ZDIM, rank, alpha, seed + 3, ctx)))
    # ff_w1, ff_w3: in=dim,out=ff
    loras.append(LArc(make_lora_adapter(ZDIM, FF_HIDDEN, rank, alpha, seed + 4, ctx)))
    loras.append(LArc(make_lora_adapter(ZDIM, FF_HIDDEN, rank, alpha, seed + 5, ctx)))
    # ff_w2: in=ff,out=dim
    loras.append(LArc(make_lora_adapter(FF_HIDDEN, ZDIM, rank, alpha, seed + 6, ctx)))
    return loras^


def main() raises:
    var ctx = DeviceContext()

    # ── sigma_map surface ─────────────────────────────────────────────────────
    var t = 250
    var sigma = sigma_from_timestep(t)
    var t_model = model_t_from_timestep(t)
    var t_back = timestep_from_sigma(sigma)
    var snr = snr_from_sigma(sigma)
    _ = ZIMAGE_NUM_TRAIN_TIMESTEPS
    _ = t_model; _ = t_back; _ = snr

    # ── lora_targets surface ──────────────────────────────────────────────────
    var prefixes = LT.zimage_lora_target_prefixes()
    _ = LT.zimage_lora_count()
    _ = prefixes
    _ = LT.lora_module_prefix(0, LT.LORA_TO_Q)

    # ── weights + block fwd/bwd ───────────────────────────────────────────────
    comptime S = 64
    var w = _build_synth_weights(ctx)
    _ = w.count()
    var keys = ZImageBlockKeys.for_block(0)
    var loras = _build_loras(8, Float32(16.0), ctx)

    # block input x [S,dim], adaln [1,256], rope cos/sin [S*H, Dh/2]
    var x = zeros_device(_w2(S, ZDIM), STDtype.BF16, ctx)
    var adaln = zeros_device(_w2(1, 256), STDtype.BF16, ctx)
    var cos = zeros_device(_w2(S * ZH, ZDh // 2), STDtype.BF16, ctx)
    var sin = zeros_device(_w2(S * ZH, ZDh // 2), STDtype.BF16, ctx)

    var fwd = zimage_block_forward[S](x, adaln, cos, sin, w, keys, loras, ctx)
    _ = fwd.out.shape()

    # backward: seed d_out = ones-ish (use zeros for build smoke)
    var d_out = zeros_device(_w2(S, ZDIM), STDtype.BF16, ctx)
    var bwd = zimage_block_backward[S](d_out, fwd.acts^, cos, sin, w, keys, loras, ctx)
    _ = bwd.d_x.shape()
    _ = len(bwd.lora_grads.d_a)
    _ = len(bwd.lora_grads.d_b)

    print("zimage block smoke OK")
