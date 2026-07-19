# Ideogram4CacheReader.mojo — streaming cache reader for full Ideogram4 LoRA
# training.
#
# This reader deliberately materialises ONE sample at a time. It never stores a
# list of sample tensors, because Ideogram4 latents + Qwen features get large and
# the full train step already owns the activation memory.
#
# Accepted safetensors layouts:
#   indexed:
#     clean.<i>       [1,128,GH,GW] F32/BF16
#     llm.<i>         [1,NT,53248] BF16/F32
#     optional noise.<i>, noisy.<i>, t_flow.<i>
#   single:
#     clean, llm, optional noise/noisy/t_flow
#   parity fixture:
#     clean_latent, llm_features, optional noise/noisy
#
# If noise/noisy are absent, the reader creates deterministic device noise for
# the current step and computes noisy = (1-t)*clean + t*noise.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random import randn

from serenity_trainer.model.Ideogram4Predict import ideogram4_add_noise
from serenity_trainer.modelSampler.Ideogram4Sampler import (
    IDEOGRAM4_PACKED_CHANNELS,
    IDEOGRAM4_TEXT_FEATURE_DIM,
)


comptime TArc = ArcPointer[Tensor]


struct Ideogram4TrainSample(Copyable, Movable):
    var clean: TArc
    var noise: TArc
    var noisy: TArc
    var llm_features: TArc
    var t_flow: Float32
    # natural (pre-pad) caption token count; defaults to NT (all real) when the
    # cache predates the text_len.<i> scalar. Passed to ideogram4_build_packed_inputs
    # so the DiT indicator is 0 at text-pad positions (ai-toolkit pipeline.py:249).
    var text_len: Int
    var index: Int

    def __init__(
        out self,
        var clean: TArc,
        var noise: TArc,
        var noisy: TArc,
        var llm_features: TArc,
        t_flow: Float32,
        text_len: Int,
        index: Int,
    ):
        self.clean = clean^
        self.noise = noise^
        self.noisy = noisy^
        self.llm_features = llm_features^
        self.t_flow = t_flow
        self.text_len = text_len
        self.index = index


struct Ideogram4TrainCache(Movable):
    var src: ShardedSafeTensors
    var clean_keys: List[String]
    var llm_keys: List[String]
    var noise_keys: List[String]
    var noisy_keys: List[String]
    var t_flow_keys: List[String]
    # natural (pre-pad) caption token length per sample (text_len.<i> scalar,
    # written by ideogram4_prepare_cache). Empty when the cache predates the fix;
    # callers then default text_len to NT (all-real-text behavior).
    var text_len_keys: List[String]
    # T1.D caption dropout: cached empty-caption features llm_uncond
    # [1,NT,53248] ("" = absent; written by ideogram4_prepare_cache when the
    # stage dir has uncond.txt from `ideogram4_stage_images.py --uncond`).
    var llm_uncond_key: String

    def __init__(
        out self,
        var src: ShardedSafeTensors,
        var clean_keys: List[String],
        var llm_keys: List[String],
        var noise_keys: List[String],
        var noisy_keys: List[String],
        var t_flow_keys: List[String],
        var text_len_keys: List[String],
        var llm_uncond_key: String,
    ):
        self.src = src^
        self.clean_keys = clean_keys^
        self.llm_keys = llm_keys^
        self.noise_keys = noise_keys^
        self.noisy_keys = noisy_keys^
        self.t_flow_keys = t_flow_keys^
        self.text_len_keys = text_len_keys^
        self.llm_uncond_key = llm_uncond_key^

    @staticmethod
    def open(path: String) raises -> Ideogram4TrainCache:
        var src = ShardedSafeTensors.open(path)
        var clean = List[String]()
        var llm = List[String]()
        var noise = List[String]()
        var noisy = List[String]()
        var tflow = List[String]()
        var tlen = List[String]()
        _discover_ideogram4_cache(src, clean, llm, noise, noisy, tflow, tlen)
        if len(clean) == 0:
            raise Error(
                String("Ideogram4TrainCache: no samples in ") + path
                + String(" — expected clean.<i>/llm.<i>, clean/llm, or ")
                + String("clean_latent/llm_features")
            )
        if len(clean) != len(llm):
            raise Error("Ideogram4TrainCache: clean/llm key count mismatch")
        if len(noise) != 0 and len(noise) != len(clean):
            raise Error("Ideogram4TrainCache: partial noise keys are not supported")
        if len(noisy) != 0 and len(noisy) != len(clean):
            raise Error("Ideogram4TrainCache: partial noisy keys are not supported")
        if len(tflow) != 0 and len(tflow) != len(clean):
            raise Error("Ideogram4TrainCache: partial t_flow keys are not supported")
        if len(tlen) != 0 and len(tlen) != len(clean):
            raise Error("Ideogram4TrainCache: partial text_len keys are not supported")
        var uncond_key = String("")
        if String("llm_uncond") in src.name_to_shard:
            uncond_key = String("llm_uncond")
        return Ideogram4TrainCache(
            src^, clean^, llm^, noise^, noisy^, tflow^, tlen^, uncond_key^
        )

    def len(self) -> Int:
        return len(self.clean_keys)

    def uncond[NT: Int](self, ctx: DeviceContext) raises -> Tensor:
        """T1.D caption dropout: the cached empty-caption (uncond) features
        [1,NT,53248] BF16. Fail-loud when the cache predates the --uncond
        stager (default-off trainers never call this)."""
        if self.llm_uncond_key.byte_length() == 0:
            raise Error(
                "Ideogram4TrainCache: caption_dropout enabled but cache has no"
                " llm_uncond (re-run stager --uncond + prepare)"
            )
        var llm = cast_tensor(
            Tensor.from_view(self.src.tensor_view(self.llm_uncond_key), ctx),
            STDtype.BF16,
            ctx,
        )
        _validate_llm_shape[NT](llm)
        return llm^

    def uncond_text_len[NT: Int](self, ctx: DeviceContext) raises -> Int:
        """Natural (pre-pad) token length of the uncond render; NT (all-real)
        when the cache has no text_len_uncond scalar (predates the fix)."""
        if String("text_len_uncond") not in self.src.name_to_shard:
            return NT
        var tl = Tensor.from_view(
            self.src.tensor_view(String("text_len_uncond")), ctx
        )
        var tlh = tl.to_host(ctx)
        if len(tlh) == 0:
            return NT
        return Int(tlh[0])

    def sample[NT: Int, GH: Int, GW: Int](
        self,
        index: Int,
        default_t_flow: Float32,
        noise_seed: UInt64,
        ctx: DeviceContext,
    ) raises -> Ideogram4TrainSample:
        if index < 0 or index >= self.len():
            raise Error(
                String("Ideogram4TrainCache.sample: index ") + String(index)
                + String(" out of range [0,") + String(self.len()) + String(")")
            )

        var clean = cast_tensor(
            Tensor.from_view(self.src.tensor_view(self.clean_keys[index]), ctx),
            STDtype.F32,
            ctx,
        )
        _validate_clean_shape[GH, GW](clean)

        var llm = cast_tensor(
            Tensor.from_view(self.src.tensor_view(self.llm_keys[index]), ctx),
            STDtype.BF16,
            ctx,
        )
        _validate_llm_shape[NT](llm)

        var flow = default_t_flow
        if len(self.t_flow_keys) == self.len():
            var tf = Tensor.from_view(self.src.tensor_view(self.t_flow_keys[index]), ctx)
            var tfh = tf.to_host(ctx)
            if len(tfh) > 0:
                flow = tfh[0]

        # natural caption length: read text_len.<i> if present, else NT (all-real).
        var text_len = NT
        if len(self.text_len_keys) == self.len():
            var tl = Tensor.from_view(self.src.tensor_view(self.text_len_keys[index]), ctx)
            var tlh = tl.to_host(ctx)
            if len(tlh) > 0:
                text_len = Int(tlh[0])

        var noise = _load_or_make_noise[GH, GW](
            self, index, clean, noise_seed, ctx
        )
        var noisy = _load_or_make_noisy[GH, GW](
            self, index, clean, noise, flow, ctx
        )

        return Ideogram4TrainSample(
            TArc(clean^),
            TArc(noise^),
            TArc(noisy^),
            TArc(llm^),
            flow,
            text_len,
            index,
        )


def _load_or_make_noise[GH: Int, GW: Int](
    cache: Ideogram4TrainCache,
    index: Int,
    clean: Tensor,
    noise_seed: UInt64,
    ctx: DeviceContext,
) raises -> Tensor:
    if len(cache.noise_keys) == cache.len():
        var noise = cast_tensor(
            Tensor.from_view(cache.src.tensor_view(cache.noise_keys[index]), ctx),
            STDtype.F32,
            ctx,
        )
        _validate_clean_shape[GH, GW](noise)
        return noise^
    return randn(clean.shape().copy(), noise_seed, STDtype.F32, ctx)


def _load_or_make_noisy[GH: Int, GW: Int](
    cache: Ideogram4TrainCache,
    index: Int,
    clean: Tensor,
    noise: Tensor,
    t_flow: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    if len(cache.noisy_keys) == cache.len():
        var noisy = cast_tensor(
            Tensor.from_view(cache.src.tensor_view(cache.noisy_keys[index]), ctx),
            STDtype.F32,
            ctx,
        )
        _validate_clean_shape[GH, GW](noisy)
        return noisy^
    return ideogram4_add_noise[GH, GW](clean, noise, t_flow, ctx)


def _validate_clean_shape[GH: Int, GW: Int](x: Tensor) raises:
    var sh = x.shape()
    if (
        len(sh) != 4
        or sh[0] != 1
        or sh[1] != IDEOGRAM4_PACKED_CHANNELS
        or sh[2] != GH
        or sh[3] != GW
    ):
        raise Error(
            String("Ideogram4TrainCache: latent shape mismatch, expected [1,")
            + String(IDEOGRAM4_PACKED_CHANNELS) + String(",")
            + String(GH) + String(",") + String(GW) + String("]")
        )


def _validate_llm_shape[NT: Int](x: Tensor) raises:
    var sh = x.shape()
    if (
        len(sh) != 3
        or sh[0] != 1
        or sh[1] != NT
        or sh[2] != IDEOGRAM4_TEXT_FEATURE_DIM
    ):
        raise Error(
            String("Ideogram4TrainCache: llm shape mismatch, expected [1,")
            + String(NT) + String(",") + String(IDEOGRAM4_TEXT_FEATURE_DIM)
            + String("]")
        )


def _discover_ideogram4_cache(
    src: ShardedSafeTensors,
    mut clean: List[String],
    mut llm: List[String],
    mut noise: List[String],
    mut noisy: List[String],
    mut tflow: List[String],
    mut tlen: List[String],
) raises:
    # Indexed cache: clean.<i>/llm.<i>, optional noise/noisy/t_flow/text_len.
    var i = 0
    while True:
        var ckey = String("clean.") + String(i)
        var lkey = String("llm.") + String(i)
        if ckey in src.name_to_shard and lkey in src.name_to_shard:
            clean.append(ckey)
            llm.append(lkey)
            var nkey = String("noise.") + String(i)
            var xkey = String("noisy.") + String(i)
            var tkey = String("t_flow.") + String(i)
            var tlkey = String("text_len.") + String(i)
            if nkey in src.name_to_shard:
                noise.append(nkey)
            if xkey in src.name_to_shard:
                noisy.append(xkey)
            if tkey in src.name_to_shard:
                tflow.append(tkey)
            if tlkey in src.name_to_shard:
                tlen.append(tlkey)
            i += 1
        else:
            break
    if len(clean) > 0:
        return

    # Single-sample cache.
    if String("clean") in src.name_to_shard and String("llm") in src.name_to_shard:
        clean.append(String("clean"))
        llm.append(String("llm"))
        if String("noise") in src.name_to_shard:
            noise.append(String("noise"))
        if String("noisy") in src.name_to_shard:
            noisy.append(String("noisy"))
        if String("t_flow") in src.name_to_shard:
            tflow.append(String("t_flow"))
        if String("text_len") in src.name_to_shard:
            tlen.append(String("text_len"))
        return

    # Existing real Giger predict fixture.
    if (
        String("clean_latent") in src.name_to_shard
        and String("llm_features") in src.name_to_shard
    ):
        clean.append(String("clean_latent"))
        llm.append(String("llm_features"))
        if String("noise") in src.name_to_shard:
            noise.append(String("noise"))
        if String("noisy") in src.name_to_shard:
            noisy.append(String("noisy"))
