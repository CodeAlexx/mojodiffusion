# models/dit/sd15_contract.mojo - SD 1.5 metadata/header contract.
#
# Header-only gate for the local diffusers SD 1.5 snapshot. It validates CLIP-L,
# UNet, VAE, scheduler, tokenizer paths, and representative tensor metadata
# without DeviceContext setup, tensor allocations, H2D loads, denoise, or decode.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.runtime.model_manifest import ModelFamily, ModelManifest


comptime SD15_ROOT = (
    "/home/alex/.cache/huggingface/hub/"
    "models--stable-diffusion-v1-5--stable-diffusion-v1-5/"
    "snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)
comptime SD15_UNET_FILE = SD15_ROOT + "/unet/diffusion_pytorch_model.safetensors"
comptime SD15_TEXT_ENCODER_FILE = SD15_ROOT + "/text_encoder/model.safetensors"
comptime SD15_VAE_FILE = SD15_ROOT + "/vae/diffusion_pytorch_model.safetensors"
comptime SD15_TOKENIZER_VOCAB = SD15_ROOT + "/tokenizer/vocab.json"

comptime SD15_DEFAULT_WIDTH = 512
comptime SD15_DEFAULT_HEIGHT = 512
comptime SD15_DEFAULT_FRAMES = 1
comptime SD15_DEFAULT_STEPS = 30
comptime SD15_DEFAULT_CFG_X10 = 75
comptime SD15_TRAIN_STEPS = 1000
comptime SD15_LATENT_DOWNSAMPLE = 8
comptime SD15_LATENT_CHANNELS = 4
comptime SD15_LATENT_H = SD15_DEFAULT_HEIGHT // SD15_LATENT_DOWNSAMPLE
comptime SD15_LATENT_W = SD15_DEFAULT_WIDTH // SD15_LATENT_DOWNSAMPLE
comptime SD15_IMAGE_TOKENS = SD15_LATENT_H * SD15_LATENT_W
comptime SD15_TEXT_TOKENS = 77
comptime SD15_TOTAL_SEQUENCE = SD15_IMAGE_TOKENS + SD15_TEXT_TOKENS
comptime SD15_PATCH_SIZE = 1

comptime SD15_UNET_TENSORS = 686
comptime SD15_TEXT_ENCODER_TENSORS = 197
comptime SD15_VAE_TENSORS = 248

comptime SD15_UNET_MODEL_CHANNELS = 320
comptime SD15_UNET_CONTEXT_DIM = 768
comptime SD15_UNET_NUM_HEADS = 8
comptime SD15_CLIP_VOCAB = 49408
comptime SD15_CLIP_HIDDEN = 768
comptime SD15_CLIP_LAYERS = 12
comptime SD15_CLIP_HEADS = 12
comptime SD15_CLIP_INTERMEDIATE = 3072
comptime SD15_VAE_SCALING_X100000 = 18215


def sd15_default_cfg_scale() -> Float32:
    return Float32(SD15_DEFAULT_CFG_X10) / 10.0


def sd15_vae_scaling_factor() -> Float32:
    return Float32(SD15_VAE_SCALING_X100000) / 100000.0


def sd15_latent_spatial_dim(image_dim: Int) raises -> Int:
    if image_dim <= 0:
        raise Error("sd15_latent_spatial_dim: image_dim must be > 0")
    if image_dim % SD15_LATENT_DOWNSAMPLE != 0:
        raise Error("SD1.5 image dimension must divide by latent_downsample=8")
    return image_dim // SD15_LATENT_DOWNSAMPLE


@fieldwise_init
struct SD15TokenPlan(Copyable, Movable):
    var width: Int
    var height: Int
    var frames: Int
    var latent_channels: Int
    var latent_h: Int
    var latent_w: Int
    var image_tokens: Int
    var text_tokens: Int
    var total_sequence: Int
    var patch_size: Int
    var latent_elements: Int

    def validate_512_contract(self) raises:
        if self.width != SD15_DEFAULT_WIDTH or self.height != SD15_DEFAULT_HEIGHT:
            raise Error("SD1.5 contract currently targets 512x512")
        if self.frames != SD15_DEFAULT_FRAMES:
            raise Error("SD1.5 contract is image-only")
        if self.latent_channels != SD15_LATENT_CHANNELS:
            raise Error("SD1.5 latent channel mismatch")
        if self.latent_h != SD15_LATENT_H or self.latent_w != SD15_LATENT_W:
            raise Error("SD1.5 latent grid must be 64x64")
        if self.image_tokens != SD15_IMAGE_TOKENS:
            raise Error("SD1.5 image token count must be 4096")
        if self.text_tokens != SD15_TEXT_TOKENS:
            raise Error("SD1.5 CLIP token count must be 77")
        if self.total_sequence != SD15_TOTAL_SEQUENCE:
            raise Error("SD1.5 total sequence mismatch")


def build_sd15_token_plan(width: Int, height: Int, frames: Int) raises -> SD15TokenPlan:
    if frames != 1:
        raise Error("build_sd15_token_plan: SD1.5 is image-only")
    var lh = sd15_latent_spatial_dim(height)
    var lw = sd15_latent_spatial_dim(width)
    return SD15TokenPlan(
        width,
        height,
        frames,
        SD15_LATENT_CHANNELS,
        lh,
        lw,
        lh * lw,
        SD15_TEXT_TOKENS,
        lh * lw + SD15_TEXT_TOKENS,
        SD15_PATCH_SIZE,
        SD15_LATENT_CHANNELS * lh * lw,
    )


def _shape1(a: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    return out^


def _shape2(a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    return out^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    return out^


def _shape_string(shape: List[Int]) -> String:
    var s = String("[")
    for i in range(len(shape)):
        if i > 0:
            s += String(", ")
        s += String(shape[i])
    s += String("]")
    return s^


def _numel(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    return n


def _require_path(label: String, path: String) raises:
    if not path_exists(path):
        raise Error(label + String(" missing: ") + path)


def _check_int(label: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            String("SD1.5 contract int mismatch: ")
            + label
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def _check_close(name: String, got: Float32, expected: Float32, tol: Float32) raises:
    var diff = got - expected
    if diff < 0.0:
        diff = -diff
    if diff > tol:
        raise Error(
            String("SD1.5 contract float mismatch: ")
            + name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def _check_tensor(
    ref st: ShardedSafeTensors,
    name: String,
    dtype: STDtype,
    expected_shape: List[Int],
) raises:
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(
            String("SD1.5 tensor dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    if len(info.shape) != len(expected_shape):
        raise Error(
            String("SD1.5 tensor rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=")
            + _shape_string(expected_shape)
        )
    for i in range(len(expected_shape)):
        if info.shape[i] != expected_shape[i]:
            raise Error(
                String("SD1.5 tensor shape mismatch for ")
                + name
                + String(": actual=")
                + _shape_string(info.shape)
                + String(" expected=")
                + _shape_string(expected_shape)
            )
    var expected_nbytes = _numel(expected_shape) * info.dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("SD1.5 tensor byte-size mismatch for ")
            + name
            + String(": actual=")
            + String(info.size)
            + String(" expected=")
            + String(expected_nbytes)
        )


def validate_sd15_manifest_contract(manifest: ModelManifest) raises:
    if manifest.model_id != "sd15":
        raise Error(String("SD1.5 contract got manifest: ") + manifest.model_id)
    if manifest.family != ModelFamily.text_to_image():
        raise Error("SD1.5 manifest family mismatch")
    if manifest.variant != "stable-diffusion-v1-5":
        raise Error(String("SD1.5 manifest variant mismatch: ") + manifest.variant)
    if manifest.default_width != SD15_DEFAULT_WIDTH:
        raise Error("SD1.5 manifest width mismatch")
    if manifest.default_height != SD15_DEFAULT_HEIGHT:
        raise Error("SD1.5 manifest height mismatch")
    if manifest.latent_channels != SD15_LATENT_CHANNELS:
        raise Error("SD1.5 manifest latent channel mismatch")
    if manifest.latent_downsample_s != SD15_LATENT_DOWNSAMPLE:
        raise Error("SD1.5 manifest downsample mismatch")
    if manifest.image_tokens != SD15_IMAGE_TOKENS:
        raise Error("SD1.5 manifest image tokens mismatch")
    if manifest.text_tokens != SD15_TEXT_TOKENS:
        raise Error("SD1.5 manifest text tokens mismatch")
    if manifest.total_sequence != SD15_TOTAL_SEQUENCE:
        raise Error("SD1.5 manifest sequence mismatch")
    if not manifest.uses_vae():
        raise Error("SD1.5 manifest must keep the VAE path")


def validate_sd15_local_paths() raises -> Int:
    _require_path(String("SD1.5 root"), String(SD15_ROOT))
    _require_path(String("SD1.5 model_index"), String(SD15_ROOT + "/model_index.json"))
    _require_path(String("SD1.5 tokenizer vocab"), String(SD15_TOKENIZER_VOCAB))
    _require_path(String("SD1.5 tokenizer merges"), String(SD15_ROOT + "/tokenizer/merges.txt"))
    _require_path(
        String("SD1.5 tokenizer config"),
        String(SD15_ROOT + "/tokenizer/tokenizer_config.json"),
    )
    _require_path(String("SD1.5 scheduler config"), String(SD15_ROOT + "/scheduler/scheduler_config.json"))
    _require_path(String("SD1.5 text config"), String(SD15_ROOT + "/text_encoder/config.json"))
    _require_path(String("SD1.5 text encoder"), String(SD15_TEXT_ENCODER_FILE))
    _require_path(String("SD1.5 UNet config"), String(SD15_ROOT + "/unet/config.json"))
    _require_path(String("SD1.5 UNet"), String(SD15_UNET_FILE))
    _require_path(String("SD1.5 VAE config"), String(SD15_ROOT + "/vae/config.json"))
    _require_path(String("SD1.5 VAE"), String(SD15_VAE_FILE))
    return 12


def validate_sd15_static_contract() raises -> SD15TokenPlan:
    var plan = build_sd15_token_plan(
        SD15_DEFAULT_WIDTH, SD15_DEFAULT_HEIGHT, SD15_DEFAULT_FRAMES
    )
    plan.validate_512_contract()
    _check_int(String("clip_head_dim"), SD15_CLIP_HIDDEN, SD15_CLIP_HEADS * 64)
    _check_int(String("unet_num_heads"), SD15_UNET_NUM_HEADS, 8)
    _check_close(String("cfg_scale"), sd15_default_cfg_scale(), 7.5, 0.000001)
    _check_close(String("vae_scale"), sd15_vae_scaling_factor(), 0.18215, 0.000001)
    return plan^


def validate_sd15_unet_header() raises:
    var st = ShardedSafeTensors.open(String(SD15_UNET_FILE))
    _check_int(String("unet_tensors"), st.num_tensors(), SD15_UNET_TENSORS)
    _check_tensor(
        st,
        String("time_embedding.linear_1.weight"),
        STDtype.F32,
        _shape2(1280, SD15_UNET_MODEL_CHANNELS),
    )
    _check_tensor(
        st,
        String("conv_in.weight"),
        STDtype.F32,
        _shape4(SD15_UNET_MODEL_CHANNELS, SD15_LATENT_CHANNELS, 3, 3),
    )
    _check_tensor(
        st,
        String("down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight"),
        STDtype.F32,
        _shape2(SD15_UNET_MODEL_CHANNELS, SD15_UNET_CONTEXT_DIM),
    )
    _check_tensor(
        st,
        String("mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight"),
        STDtype.F32,
        _shape2(1280, SD15_UNET_CONTEXT_DIM),
    )
    _check_tensor(
        st,
        String("up_blocks.3.resnets.2.conv2.weight"),
        STDtype.F32,
        _shape4(SD15_UNET_MODEL_CHANNELS, SD15_UNET_MODEL_CHANNELS, 3, 3),
    )
    _check_tensor(st, String("conv_norm_out.weight"), STDtype.F32, _shape1(SD15_UNET_MODEL_CHANNELS))
    _check_tensor(
        st,
        String("conv_out.weight"),
        STDtype.F32,
        _shape4(SD15_LATENT_CHANNELS, SD15_UNET_MODEL_CHANNELS, 3, 3),
    )


def validate_sd15_text_encoder_header() raises:
    var st = ShardedSafeTensors.open(String(SD15_TEXT_ENCODER_FILE))
    _check_int(
        String("text_encoder_tensors"), st.num_tensors(), SD15_TEXT_ENCODER_TENSORS
    )
    _check_tensor(
        st,
        String("text_model.embeddings.token_embedding.weight"),
        STDtype.F32,
        _shape2(SD15_CLIP_VOCAB, SD15_CLIP_HIDDEN),
    )
    _check_tensor(
        st,
        String("text_model.embeddings.position_embedding.weight"),
        STDtype.F32,
        _shape2(SD15_TEXT_TOKENS, SD15_CLIP_HIDDEN),
    )
    _check_tensor(
        st,
        String("text_model.encoder.layers.0.self_attn.q_proj.weight"),
        STDtype.F32,
        _shape2(SD15_CLIP_HIDDEN, SD15_CLIP_HIDDEN),
    )
    _check_tensor(
        st,
        String("text_model.encoder.layers.11.mlp.fc2.weight"),
        STDtype.F32,
        _shape2(SD15_CLIP_HIDDEN, SD15_CLIP_INTERMEDIATE),
    )
    _check_tensor(
        st,
        String("text_model.final_layer_norm.weight"),
        STDtype.F32,
        _shape1(SD15_CLIP_HIDDEN),
    )


def validate_sd15_vae_header() raises:
    var st = ShardedSafeTensors.open(String(SD15_VAE_FILE))
    _check_int(String("vae_tensors"), st.num_tensors(), SD15_VAE_TENSORS)
    _check_tensor(
        st,
        String("post_quant_conv.weight"),
        STDtype.F32,
        _shape4(SD15_LATENT_CHANNELS, SD15_LATENT_CHANNELS, 1, 1),
    )
    _check_tensor(
        st,
        String("decoder.conv_in.weight"),
        STDtype.F32,
        _shape4(512, SD15_LATENT_CHANNELS, 3, 3),
    )
    _check_tensor(
        st,
        String("decoder.mid_block.attentions.0.query.weight"),
        STDtype.F32,
        _shape2(512, 512),
    )
    _check_tensor(st, String("decoder.conv_norm_out.weight"), STDtype.F32, _shape1(128))
    _check_tensor(
        st,
        String("decoder.conv_out.weight"),
        STDtype.F32,
        _shape4(3, 128, 3, 3),
    )


def validate_sd15_metadata_contract(manifest: ModelManifest) raises -> SD15TokenPlan:
    validate_sd15_manifest_contract(manifest)
    var plan = validate_sd15_static_contract()
    validate_sd15_unet_header()
    validate_sd15_text_encoder_header()
    validate_sd15_vae_header()
    return plan^
