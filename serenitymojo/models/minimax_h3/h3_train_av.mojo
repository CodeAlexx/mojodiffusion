# serenitymojo/models/minimax_h3/h3_train_av.mojo
#
# AV-training tensor plumbing shared by the trainer and its step gate:
# audio latent <-> packed-row transforms and the audio loss-mask tensor.
#
# Row contract (packing.py:315, inverse gated in rearrange.mojo
# minimax_h3_unpack_audio): audio rows are CHANNEL-MAJOR — the block for
# stereo channel 0 comes first, then channel 1; row (s, t) carries the
# 32-dim latent column at [s, :, t]. So [2, 32, T] -> [2T, 32] is
# permute(0, 2, 1) then a flatten of the leading two dims — and back.
from max.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra import permute, reshape
from serenitymojo.ops.cast import cast_tensor


def h3_audio_latents_to_rows(
    audio: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """[2, C, T] -> [2T, C] channel-major rows (dtype preserved)."""
    var sh = audio.shape()
    if len(sh) != 3 or sh[0] != 2:
        raise Error("h3_audio_latents_to_rows: expected [2, C, T]")
    var c = sh[1]
    var t = sh[2]
    var perm: List[Int] = [0, 2, 1]
    var p = permute(audio, perm, ctx)          # [2, T, C]
    var rows_sh: List[Int] = [2 * t, c]
    return reshape(p, rows_sh^, ctx)


def h3_audio_rows_to_latents(
    rows: Tensor, num_audio_latents: Int, ctx: DeviceContext
) raises -> Tensor:
    """[2T, C] channel-major rows -> [2, C, T] (dtype preserved)."""
    var sh = rows.shape()
    if len(sh) != 2 or sh[0] != 2 * num_audio_latents:
        raise Error("h3_audio_rows_to_latents: expected [2T, C]")
    var c = sh[1]
    var mid_sh: List[Int] = [2, num_audio_latents, c]
    var m = reshape(rows, mid_sh^, ctx)        # [2, T, C]
    var perm: List[Int] = [0, 2, 1]
    return permute(m, perm, ctx)               # [2, C, T]


def h3_audio_mask_tensor(
    mask: List[Bool], channels: Int, ctx: DeviceContext
) raises -> Tensor:
    """[T] bool -> full-shape [2, C, T] bf16 0/1 tensor (h3_loss_grad's mask
    multiply is exact on 0/1)."""
    var t = len(mask)
    var vals = List[Float32](capacity=2 * channels * t)
    for _ in range(2 * channels):
        for i in range(t):
            vals.append(Float32(1.0) if mask[i] else Float32(0.0))
    var sh: List[Int] = [2, channels, t]
    var f = Tensor.from_host(vals, sh^, STDtype.F32, ctx)
    return cast_tensor(f, STDtype.BF16, ctx)
