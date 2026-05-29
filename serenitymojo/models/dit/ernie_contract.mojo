# models/dit/ernie_contract.mojo - ERNIE-Image metadata/header contract.
#
# Header-only gate for Baidu ERNIE-Image 8B. It validates the local Mistral3B
# text encoder, ERNIE DiT, Klein VAE dependency, and FlowMatch Euler schedule
# without DeviceContext setup, H2D weight loads, denoise math, or VAE decode.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.runtime.model_manifest import ModelFamily, ModelManifest


comptime ERNIE_ROOT = "/home/alex/models/ERNIE-Image"
comptime ERNIE_TRANSFORMER_DIR = ERNIE_ROOT + "/transformer"
comptime ERNIE_TRANSFORMER_SHARD_0 = (
    ERNIE_TRANSFORMER_DIR + "/diffusion_pytorch_model-00001-of-00002.safetensors"
)
comptime ERNIE_TRANSFORMER_SHARD_1 = (
    ERNIE_TRANSFORMER_DIR + "/diffusion_pytorch_model-00002-of-00002.safetensors"
)
comptime ERNIE_TEXT_ENCODER_FILE = ERNIE_ROOT + "/text_encoder/model.safetensors"
comptime ERNIE_TOKENIZER_FILE = ERNIE_ROOT + "/tokenizer/tokenizer.json"
comptime ERNIE_VAE_FILE = ERNIE_ROOT + "/vae/diffusion_pytorch_model.safetensors"

comptime ERNIE_DEFAULT_WIDTH = 1024
comptime ERNIE_DEFAULT_HEIGHT = 1024
comptime ERNIE_DEFAULT_FRAMES = 1
comptime ERNIE_DEFAULT_STEPS = 50
comptime ERNIE_DEFAULT_CFG_X10 = 40
comptime ERNIE_SCHEDULER_SHIFT_X10 = 30
comptime ERNIE_SCHEDULER_TRAIN_STEPS = 1000
comptime ERNIE_LATENT_DOWNSAMPLE = 16
comptime ERNIE_LATENT_CHANNELS = 128
comptime ERNIE_LATENT_H = ERNIE_DEFAULT_HEIGHT // ERNIE_LATENT_DOWNSAMPLE
comptime ERNIE_LATENT_W = ERNIE_DEFAULT_WIDTH // ERNIE_LATENT_DOWNSAMPLE
comptime ERNIE_PATCH_SIZE = 1
comptime ERNIE_IMAGE_TOKENS = ERNIE_LATENT_H * ERNIE_LATENT_W
comptime ERNIE_TEXT_MAX_TOKENS = 256
comptime ERNIE_TOTAL_SEQUENCE = ERNIE_IMAGE_TOKENS + ERNIE_TEXT_MAX_TOKENS

comptime ERNIE_DIT_HIDDEN = 4096
comptime ERNIE_DIT_HEADS = 32
comptime ERNIE_DIT_HEAD_DIM = 128
comptime ERNIE_DIT_LAYERS = 36
comptime ERNIE_DIT_FFN_HIDDEN = 12288
comptime ERNIE_DIT_TEXT_IN_DIM = 3072
comptime ERNIE_DIT_ROPE_THETA = 256
comptime ERNIE_DIT_ROPE_AXIS_0 = 32
comptime ERNIE_DIT_ROPE_AXIS_1 = 48
comptime ERNIE_DIT_ROPE_AXIS_2 = 48
comptime ERNIE_DIT_ADALN_DIM = 6 * ERNIE_DIT_HIDDEN
comptime ERNIE_DIT_FINAL_ADALN_DIM = 2 * ERNIE_DIT_HIDDEN

comptime ERNIE_MISTRAL_VOCAB = 131072
comptime ERNIE_MISTRAL_HIDDEN = 3072
comptime ERNIE_MISTRAL_LAYERS = 26
comptime ERNIE_MISTRAL_EXTRACT_LAYER = 24
comptime ERNIE_MISTRAL_HEADS = 32
comptime ERNIE_MISTRAL_KV_HEADS = 8
comptime ERNIE_MISTRAL_HEAD_DIM = 128
comptime ERNIE_MISTRAL_Q_DIM = ERNIE_MISTRAL_HEADS * ERNIE_MISTRAL_HEAD_DIM
comptime ERNIE_MISTRAL_KV_DIM = ERNIE_MISTRAL_KV_HEADS * ERNIE_MISTRAL_HEAD_DIM
comptime ERNIE_MISTRAL_INTERMEDIATE = 9216
comptime ERNIE_MISTRAL_MAX_SEQ_LEN = 512
comptime ERNIE_MISTRAL_YARN_FACTOR = 16

comptime ERNIE_TRANSFORMER_TENSORS = 409
comptime ERNIE_TEXT_ENCODER_TENSORS = 458
comptime ERNIE_VAE_TENSORS = 251
comptime ERNIE_KLEIN_VAE_LATENT_CHANNELS = 32


def ernie_default_cfg_scale() -> Float32:
    return Float32(ERNIE_DEFAULT_CFG_X10) / 10.0


def ernie_default_shift() -> Float32:
    return Float32(ERNIE_SCHEDULER_SHIFT_X10) / 10.0


def ernie_sigma(index: Int, num_steps: Int, shift: Float32) raises -> Float32:
    if num_steps <= 0:
        raise Error("ernie_sigma: num_steps must be > 0")
    if index < 0 or index > num_steps:
        raise Error("ernie_sigma: index out of range")
    if index == 0:
        return 1.0
    if index == num_steps:
        return 0.0
    var t = 1.0 - Float32(index) / Float32(num_steps)
    return shift * t / (1.0 + (shift - 1.0) * t)


def ernie_euler_delta(index: Int, num_steps: Int, shift: Float32) raises -> Float32:
    if index < 0 or index >= num_steps:
        raise Error("ernie_euler_delta: index out of range")
    return ernie_sigma(index + 1, num_steps, shift) - ernie_sigma(
        index, num_steps, shift
    )


def ernie_model_timestep(sigma: Float32) -> Float32:
    return sigma * Float32(ERNIE_SCHEDULER_TRAIN_STEPS)


def ernie_latent_spatial_dim(image_dim: Int) raises -> Int:
    if image_dim <= 0:
        raise Error("ernie_latent_spatial_dim: image_dim must be > 0")
    if image_dim % ERNIE_LATENT_DOWNSAMPLE != 0:
        raise Error("ERNIE image dimension must divide by latent_downsample=16")
    return image_dim // ERNIE_LATENT_DOWNSAMPLE


@fieldwise_init
struct ErnieTokenPlan(Copyable, Movable):
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

    def validate_1024_contract(self) raises:
        if self.width != ERNIE_DEFAULT_WIDTH or self.height != ERNIE_DEFAULT_HEIGHT:
            raise Error("ERNIE contract currently targets 1024x1024")
        if self.frames != ERNIE_DEFAULT_FRAMES:
            raise Error("ERNIE contract is image-only")
        if self.latent_channels != ERNIE_LATENT_CHANNELS:
            raise Error("ERNIE latent channel mismatch")
        if self.latent_h != ERNIE_LATENT_H or self.latent_w != ERNIE_LATENT_W:
            raise Error("ERNIE latent grid must be 64x64")
        if self.image_tokens != ERNIE_IMAGE_TOKENS:
            raise Error("ERNIE image token count must be 4096")
        if self.text_tokens != ERNIE_TEXT_MAX_TOKENS:
            raise Error("ERNIE max conditioning tokens must be 256")
        if self.total_sequence != ERNIE_TOTAL_SEQUENCE:
            raise Error("ERNIE total sequence must be 4352")
        if self.patch_size != ERNIE_PATCH_SIZE:
            raise Error("ERNIE patch size must be 1")


def build_ernie_token_plan(
    width: Int, height: Int, frames: Int, text_tokens: Int
) raises -> ErnieTokenPlan:
    if frames != 1:
        raise Error("build_ernie_token_plan: ERNIE is image-only")
    if text_tokens <= 0:
        raise Error("build_ernie_token_plan: text_tokens must be > 0")
    var lh = ernie_latent_spatial_dim(height)
    var lw = ernie_latent_spatial_dim(width)
    return ErnieTokenPlan(
        width,
        height,
        frames,
        ERNIE_LATENT_CHANNELS,
        lh,
        lw,
        lh * lw,
        text_tokens,
        lh * lw + text_tokens,
        ERNIE_PATCH_SIZE,
        ERNIE_LATENT_CHANNELS * lh * lw,
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
            label
            + String(" mismatch: actual=")
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
            String("ERNIE float mismatch: ")
            + name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def _check_shape(
    ref st: ShardedSafeTensors, name: String, expected_shape: List[Int]
) raises:
    var info = st.tensor_info(name)
    if len(info.shape) != len(expected_shape):
        raise Error(
            String("ERNIE tensor rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=")
            + _shape_string(expected_shape)
        )
    for i in range(len(expected_shape)):
        if info.shape[i] != expected_shape[i]:
            raise Error(
                String("ERNIE tensor shape mismatch for ")
                + name
                + String(": actual=")
                + _shape_string(info.shape)
                + String(" expected=")
                + _shape_string(expected_shape)
            )
    var expected_nbytes = _numel(expected_shape) * info.dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("ERNIE tensor byte-size mismatch for ")
            + name
            + String(": actual=")
            + String(info.size)
            + String(" expected=")
            + String(expected_nbytes)
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
            String("ERNIE tensor dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    _check_shape(st, name, expected_shape)


def validate_ernie_manifest_contract(manifest: ModelManifest) raises:
    if manifest.model_id != "ernie_image":
        raise Error(String("ERNIE contract got manifest: ") + manifest.model_id)
    if manifest.family != ModelFamily.text_to_image():
        raise Error("ERNIE manifest family mismatch")
    if manifest.variant != "ernie-image-8b":
        raise Error(String("ERNIE manifest variant mismatch: ") + manifest.variant)
    if manifest.profile_name != "ernie_image_1024":
        raise Error(String("ERNIE manifest profile mismatch: ") + manifest.profile_name)
    if manifest.default_width != ERNIE_DEFAULT_WIDTH:
        raise Error("ERNIE manifest width mismatch")
    if manifest.default_height != ERNIE_DEFAULT_HEIGHT:
        raise Error("ERNIE manifest height mismatch")
    if manifest.default_frames != ERNIE_DEFAULT_FRAMES:
        raise Error("ERNIE manifest frames mismatch")
    if manifest.latent_channels != ERNIE_LATENT_CHANNELS:
        raise Error("ERNIE manifest latent channels mismatch")
    if manifest.latent_downsample_s != ERNIE_LATENT_DOWNSAMPLE:
        raise Error("ERNIE manifest spatial downsample mismatch")
    if manifest.image_tokens != ERNIE_IMAGE_TOKENS:
        raise Error("ERNIE manifest image tokens mismatch")
    if manifest.text_tokens != ERNIE_TEXT_MAX_TOKENS:
        raise Error("ERNIE manifest text tokens mismatch")
    if manifest.total_sequence != ERNIE_TOTAL_SEQUENCE:
        raise Error("ERNIE manifest sequence mismatch")
    if manifest.patch_size != ERNIE_PATCH_SIZE:
        raise Error("ERNIE manifest patch size mismatch")
    if not manifest.uses_vae():
        raise Error("ERNIE manifest must keep the Klein VAE path")
    if manifest.production_entry != "serenitymojo/pipeline/ernie_contract_smoke.mojo":
        raise Error("ERNIE manifest production entry mismatch")


def validate_ernie_local_paths() raises -> Int:
    _require_path(String("ERNIE root"), String(ERNIE_ROOT))
    _require_path(String("ERNIE model_index"), String(ERNIE_ROOT + "/model_index.json"))
    _require_path(
        String("ERNIE transformer config"), String(ERNIE_TRANSFORMER_DIR + "/config.json")
    )
    _require_path(
        String("ERNIE transformer index"),
        String(ERNIE_TRANSFORMER_DIR + "/diffusion_pytorch_model.safetensors.index.json"),
    )
    _require_path(String("ERNIE transformer shard 0"), String(ERNIE_TRANSFORMER_SHARD_0))
    _require_path(String("ERNIE transformer shard 1"), String(ERNIE_TRANSFORMER_SHARD_1))
    _require_path(
        String("ERNIE text encoder config"),
        String(ERNIE_ROOT + "/text_encoder/config.json"),
    )
    _require_path(String("ERNIE text encoder"), String(ERNIE_TEXT_ENCODER_FILE))
    _require_path(String("ERNIE tokenizer"), String(ERNIE_TOKENIZER_FILE))
    _require_path(
        String("ERNIE tokenizer config"),
        String(ERNIE_ROOT + "/tokenizer/tokenizer_config.json"),
    )
    _require_path(
        String("ERNIE scheduler"), String(ERNIE_ROOT + "/scheduler/scheduler_config.json")
    )
    _require_path(String("ERNIE VAE config"), String(ERNIE_ROOT + "/vae/config.json"))
    _require_path(String("ERNIE VAE weights"), String(ERNIE_VAE_FILE))
    return 13


def validate_ernie_static_contract() raises -> ErnieTokenPlan:
    var plan = build_ernie_token_plan(
        ERNIE_DEFAULT_WIDTH,
        ERNIE_DEFAULT_HEIGHT,
        ERNIE_DEFAULT_FRAMES,
        ERNIE_TEXT_MAX_TOKENS,
    )
    plan.validate_1024_contract()
    _check_int(String("hidden"), ERNIE_DIT_HIDDEN, ERNIE_DIT_HEADS * ERNIE_DIT_HEAD_DIM)
    _check_int(String("depth"), ERNIE_DIT_LAYERS, 36)
    _check_int(String("mistral_layers"), ERNIE_MISTRAL_LAYERS, 26)
    _check_int(String("mistral_extract_layer"), ERNIE_MISTRAL_EXTRACT_LAYER, 24)
    _check_int(
        String("mistral_q_dim"),
        ERNIE_MISTRAL_Q_DIM,
        ERNIE_MISTRAL_HEADS * ERNIE_MISTRAL_HEAD_DIM,
    )
    _check_int(
        String("mistral_kv_dim"),
        ERNIE_MISTRAL_KV_DIM,
        ERNIE_MISTRAL_KV_HEADS * ERNIE_MISTRAL_HEAD_DIM,
    )
    _check_close(
        String("cfg_scale"), ernie_default_cfg_scale(), 4.0, 0.000001
    )
    _check_close(
        String("sigma[0]"),
        ernie_sigma(0, ERNIE_DEFAULT_STEPS, ernie_default_shift()),
        1.0,
        0.000001,
    )
    _check_close(
        String("sigma[mid]"),
        ernie_sigma(ERNIE_DEFAULT_STEPS // 2, ERNIE_DEFAULT_STEPS, ernie_default_shift()),
        0.75,
        0.000001,
    )
    _check_close(
        String("sigma[end]"),
        ernie_sigma(ERNIE_DEFAULT_STEPS, ERNIE_DEFAULT_STEPS, ernie_default_shift()),
        0.0,
        0.000001,
    )
    if ernie_euler_delta(0, ERNIE_DEFAULT_STEPS, ernie_default_shift()) >= 0.0:
        raise Error("ERNIE Euler delta should be sigma_next - sigma")
    _check_close(String("timestep_sigma1"), ernie_model_timestep(1.0), 1000.0, 0.000001)
    _check_close(String("timestep_sigma0"), ernie_model_timestep(0.0), 0.0, 0.000001)
    return plan^


def validate_ernie_transformer_header() raises:
    var st = ShardedSafeTensors.open(String(ERNIE_TRANSFORMER_DIR))
    _check_int(String("transformer_tensors"), st.num_tensors(), ERNIE_TRANSFORMER_TENSORS)
    _check_tensor(
        st,
        String("x_embedder.proj.weight"),
        STDtype.BF16,
        _shape4(ERNIE_DIT_HIDDEN, ERNIE_LATENT_CHANNELS, 1, 1),
    )
    _check_tensor(st, String("x_embedder.proj.bias"), STDtype.BF16, _shape1(ERNIE_DIT_HIDDEN))
    _check_tensor(
        st,
        String("text_proj.weight"),
        STDtype.BF16,
        _shape2(ERNIE_DIT_HIDDEN, ERNIE_DIT_TEXT_IN_DIM),
    )
    _check_tensor(
        st,
        String("time_embedding.linear_1.weight"),
        STDtype.BF16,
        _shape2(ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("time_embedding.linear_2.weight"),
        STDtype.BF16,
        _shape2(ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("adaLN_modulation.1.weight"),
        STDtype.BF16,
        _shape2(ERNIE_DIT_ADALN_DIM, ERNIE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("layers.0.adaLN_sa_ln.weight"),
        STDtype.BF16,
        _shape1(ERNIE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("layers.0.self_attention.to_q.weight"),
        STDtype.BF16,
        _shape2(ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("layers.0.self_attention.to_out.0.weight"),
        STDtype.BF16,
        _shape2(ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("layers.0.self_attention.norm_q.weight"),
        STDtype.BF16,
        _shape1(ERNIE_DIT_HEAD_DIM),
    )
    _check_tensor(
        st,
        String("layers.0.mlp.gate_proj.weight"),
        STDtype.BF16,
        _shape2(ERNIE_DIT_FFN_HIDDEN, ERNIE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("layers.0.mlp.linear_fc2.weight"),
        STDtype.BF16,
        _shape2(ERNIE_DIT_HIDDEN, ERNIE_DIT_FFN_HIDDEN),
    )
    _check_tensor(
        st,
        String("layers.35.self_attention.to_v.weight"),
        STDtype.BF16,
        _shape2(ERNIE_DIT_HIDDEN, ERNIE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("layers.35.mlp.linear_fc2.weight"),
        STDtype.BF16,
        _shape2(ERNIE_DIT_HIDDEN, ERNIE_DIT_FFN_HIDDEN),
    )
    _check_tensor(
        st,
        String("final_norm.linear.weight"),
        STDtype.BF16,
        _shape2(ERNIE_DIT_FINAL_ADALN_DIM, ERNIE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("final_linear.weight"),
        STDtype.BF16,
        _shape2(ERNIE_LATENT_CHANNELS, ERNIE_DIT_HIDDEN),
    )


def validate_ernie_text_encoder_header() raises:
    var st = ShardedSafeTensors.open(String(ERNIE_TEXT_ENCODER_FILE))
    _check_int(
        String("text_encoder_tensors"), st.num_tensors(), ERNIE_TEXT_ENCODER_TENSORS
    )
    _check_tensor(
        st,
        String("language_model.model.embed_tokens.weight"),
        STDtype.BF16,
        _shape2(ERNIE_MISTRAL_VOCAB, ERNIE_MISTRAL_HIDDEN),
    )
    _check_tensor(
        st,
        String("language_model.model.layers.0.self_attn.q_proj.weight"),
        STDtype.BF16,
        _shape2(ERNIE_MISTRAL_Q_DIM, ERNIE_MISTRAL_HIDDEN),
    )
    _check_tensor(
        st,
        String("language_model.model.layers.0.self_attn.k_proj.weight"),
        STDtype.BF16,
        _shape2(ERNIE_MISTRAL_KV_DIM, ERNIE_MISTRAL_HIDDEN),
    )
    _check_tensor(
        st,
        String("language_model.model.layers.0.self_attn.o_proj.weight"),
        STDtype.BF16,
        _shape2(ERNIE_MISTRAL_HIDDEN, ERNIE_MISTRAL_Q_DIM),
    )
    _check_tensor(
        st,
        String("language_model.model.layers.24.mlp.down_proj.weight"),
        STDtype.BF16,
        _shape2(ERNIE_MISTRAL_HIDDEN, ERNIE_MISTRAL_INTERMEDIATE),
    )
    _check_tensor(
        st,
        String("language_model.model.norm.weight"),
        STDtype.BF16,
        _shape1(ERNIE_MISTRAL_HIDDEN),
    )


def validate_ernie_vae_header() raises:
    var st = ShardedSafeTensors.open(String(ERNIE_VAE_FILE))
    _check_int(String("vae_tensors"), st.num_tensors(), ERNIE_VAE_TENSORS)
    _check_tensor(
        st,
        String("post_quant_conv.weight"),
        STDtype.BF16,
        _shape4(
            ERNIE_KLEIN_VAE_LATENT_CHANNELS,
            ERNIE_KLEIN_VAE_LATENT_CHANNELS,
            1,
            1,
        ),
    )
    _check_tensor(
        st,
        String("decoder.conv_in.weight"),
        STDtype.BF16,
        _shape4(512, ERNIE_KLEIN_VAE_LATENT_CHANNELS, 3, 3),
    )
    _check_tensor(
        st,
        String("decoder.up_blocks.3.resnets.2.conv2.weight"),
        STDtype.BF16,
        _shape4(128, 128, 3, 3),
    )
    _check_tensor(
        st,
        String("decoder.conv_out.weight"),
        STDtype.BF16,
        _shape4(3, 128, 3, 3),
    )


def validate_ernie_metadata_contract(manifest: ModelManifest) raises -> ErnieTokenPlan:
    validate_ernie_manifest_contract(manifest)
    _ = validate_ernie_local_paths()
    var plan = validate_ernie_static_contract()
    validate_ernie_transformer_header()
    validate_ernie_text_encoder_header()
    validate_ernie_vae_header()
    return plan^
