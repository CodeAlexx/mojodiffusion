# models/dit/chroma_contract.mojo - Chroma metadata/header contract.
#
# Header-only gate for lodestones Chroma1-HD. It validates local checkpoint
# visibility, static image/latent/T5 geometry, and representative safetensors
# tensor metadata without DeviceContext setup, tensor allocations, H2D loads, or
# denoise/VAE math.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import path_exists


comptime CHROMA_SINGLE_DIT_CHECKPOINT = "/home/alex/.serenity/models/checkpoints/chroma1_hd_bf16.safetensors"
comptime CHROMA_SNAPSHOT_ROOT = "/home/alex/.cache/huggingface/hub/models--lodestones--Chroma1-HD/snapshots/0e0c60ece1e82b17cb7f77342d765ba5024c40c0"
comptime CHROMA_TRANSFORMER_DIR = CHROMA_SNAPSHOT_ROOT + "/transformer"
comptime CHROMA_TRANSFORMER_INDEX = (
    CHROMA_TRANSFORMER_DIR + "/diffusion_pytorch_model.safetensors.index.json"
)
comptime CHROMA_TRANSFORMER_SHARD_0 = (
    CHROMA_TRANSFORMER_DIR + "/diffusion_pytorch_model-00001-of-00002.safetensors"
)
comptime CHROMA_TRANSFORMER_SHARD_1 = (
    CHROMA_TRANSFORMER_DIR + "/diffusion_pytorch_model-00002-of-00002.safetensors"
)
comptime CHROMA_TRANSFORMER_CONFIG = CHROMA_TRANSFORMER_DIR + "/config.json"
comptime CHROMA_TEXT_ENCODER_DIR = CHROMA_SNAPSHOT_ROOT + "/text_encoder"
comptime CHROMA_TEXT_ENCODER_INDEX = (
    CHROMA_TEXT_ENCODER_DIR + "/model.safetensors.index.json"
)
comptime CHROMA_TEXT_ENCODER_SHARD_0 = (
    CHROMA_TEXT_ENCODER_DIR + "/model-00001-of-00002.safetensors"
)
comptime CHROMA_TEXT_ENCODER_SHARD_1 = (
    CHROMA_TEXT_ENCODER_DIR + "/model-00002-of-00002.safetensors"
)
comptime CHROMA_TEXT_ENCODER_CONFIG = CHROMA_TEXT_ENCODER_DIR + "/config.json"
comptime CHROMA_TOKENIZER_DIR = CHROMA_SNAPSHOT_ROOT + "/tokenizer"
comptime CHROMA_TOKENIZER_MODEL = CHROMA_TOKENIZER_DIR + "/spiece.model"
comptime CHROMA_TOKENIZER_CONFIG = CHROMA_TOKENIZER_DIR + "/tokenizer_config.json"
comptime CHROMA_SCHEDULER_CONFIG = CHROMA_SNAPSHOT_ROOT + "/scheduler/scheduler_config.json"
comptime CHROMA_VAE_FILE = (
    CHROMA_SNAPSHOT_ROOT + "/vae/diffusion_pytorch_model.safetensors"
)
comptime CHROMA_VAE_CONFIG = CHROMA_SNAPSHOT_ROOT + "/vae/config.json"

comptime CHROMA_DEFAULT_WIDTH = 1024
comptime CHROMA_DEFAULT_HEIGHT = 1024
comptime CHROMA_DEFAULT_FRAMES = 1
comptime CHROMA_DEFAULT_STEPS = 40
comptime CHROMA_DEFAULT_CFG_X10 = 40
comptime CHROMA_SCHEDULE_SHIFT_X100 = 115
comptime CHROMA_LATENT_DOWNSAMPLE = 8
comptime CHROMA_LATENT_CHANNELS = 16
comptime CHROMA_LATENT_H = CHROMA_DEFAULT_HEIGHT // CHROMA_LATENT_DOWNSAMPLE
comptime CHROMA_LATENT_W = CHROMA_DEFAULT_WIDTH // CHROMA_LATENT_DOWNSAMPLE
comptime CHROMA_PACK_PATCH = 2
comptime CHROMA_PATCH_GRID_H = CHROMA_LATENT_H // CHROMA_PACK_PATCH
comptime CHROMA_PATCH_GRID_W = CHROMA_LATENT_W // CHROMA_PACK_PATCH
comptime CHROMA_IMAGE_TOKENS = CHROMA_PATCH_GRID_H * CHROMA_PATCH_GRID_W
comptime CHROMA_PATCH_VECTOR_DIM = (
    CHROMA_LATENT_CHANNELS * CHROMA_PACK_PATCH * CHROMA_PACK_PATCH
)
comptime CHROMA_T5_SEQ_LEN = 512
comptime CHROMA_T5_HIDDEN = 4096
comptime CHROMA_T5_VOCAB = 32128
comptime CHROMA_T5_LAYERS = 24
comptime CHROMA_T5_HEADS = 64
comptime CHROMA_T5_HEAD_DIM = 64
comptime CHROMA_T5_FFN_HIDDEN = 10240
comptime CHROMA_TOTAL_SEQUENCE = CHROMA_IMAGE_TOKENS + CHROMA_T5_SEQ_LEN

comptime CHROMA_DIT_HIDDEN = 3072
comptime CHROMA_DIT_DOUBLE_BLOCKS = 19
comptime CHROMA_DIT_SINGLE_BLOCKS = 38
comptime CHROMA_DIT_TOTAL_BLOCKS = (
    CHROMA_DIT_DOUBLE_BLOCKS + CHROMA_DIT_SINGLE_BLOCKS
)
comptime CHROMA_DIT_HEADS = 24
comptime CHROMA_DIT_HEAD_DIM = 128
comptime CHROMA_DIT_MLP_HIDDEN = 12288
comptime CHROMA_DIT_CONTEXT_DIM = CHROMA_T5_HIDDEN
comptime CHROMA_DIT_ROPE_AXIS_0 = 16
comptime CHROMA_DIT_ROPE_AXIS_1 = 56
comptime CHROMA_DIT_ROPE_AXIS_2 = 56
comptime CHROMA_DIT_APPROX_IN = 64
comptime CHROMA_DIT_APPROX_HIDDEN = 5120
comptime CHROMA_DIT_APPROX_LAYERS = 5
comptime CHROMA_DIT_MOD_INDEX = (
    3 * CHROMA_DIT_SINGLE_BLOCKS + 2 * 6 * CHROMA_DIT_DOUBLE_BLOCKS + 2
)
comptime CHROMA_DIT_TENSORS = 1023
comptime CHROMA_TEXT_ENCODER_TENSORS = 219
comptime CHROMA_VAE_TENSORS = 244
comptime CHROMA_VAE_SCALE_X10000 = 3611
comptime CHROMA_VAE_SHIFT_X10000 = 1159


def chroma_default_checkpoint_path() -> String:
    return String(CHROMA_SINGLE_DIT_CHECKPOINT)


def chroma_transformer_dir() -> String:
    return String(CHROMA_TRANSFORMER_DIR)


def chroma_text_encoder_dir() -> String:
    return String(CHROMA_TEXT_ENCODER_DIR)


def chroma_vae_path() -> String:
    return String(CHROMA_VAE_FILE)


def chroma_default_cfg_scale() -> Float32:
    return Float32(CHROMA_DEFAULT_CFG_X10) / 10.0


def chroma_schedule_shift() -> Float32:
    return Float32(CHROMA_SCHEDULE_SHIFT_X100) / 100.0


def chroma_vae_scale() -> Float32:
    return Float32(CHROMA_VAE_SCALE_X10000) / 10000.0


def chroma_vae_shift() -> Float32:
    return Float32(CHROMA_VAE_SHIFT_X10000) / 10000.0


def chroma_shifted_sigma(index: Int, num_steps: Int, shift: Float32) raises -> Float32:
    if num_steps <= 0:
        raise Error("chroma_shifted_sigma: num_steps must be > 0")
    if index < 0 or index > num_steps:
        raise Error("chroma_shifted_sigma: index out of range")
    if index == num_steps:
        return 0.0
    var t = 1.0 - Float32(index) / Float32(num_steps)
    return shift * t / (1.0 + (shift - 1.0) * t)


def chroma_schedule_delta(index: Int, num_steps: Int, shift: Float32) raises -> Float32:
    if index < 0 or index >= num_steps:
        raise Error("chroma_schedule_delta: index out of range")
    var sigma = chroma_shifted_sigma(index, num_steps, shift)
    var sigma_next = chroma_shifted_sigma(index + 1, num_steps, shift)
    return sigma_next - sigma


def chroma_latent_spatial_dim(image_dim: Int) raises -> Int:
    if image_dim <= 0:
        raise Error("chroma_latent_spatial_dim: image_dim must be > 0")
    if image_dim % CHROMA_LATENT_DOWNSAMPLE != 0:
        raise Error("Chroma image dimension must divide by latent downsample=8")
    return image_dim // CHROMA_LATENT_DOWNSAMPLE


def chroma_patch_grid_dim(latent_dim: Int) raises -> Int:
    if latent_dim <= 0:
        raise Error("chroma_patch_grid_dim: latent_dim must be > 0")
    if latent_dim % CHROMA_PACK_PATCH != 0:
        raise Error("Chroma latent dimension must divide by pack patch=2")
    return latent_dim // CHROMA_PACK_PATCH


@fieldwise_init
struct ChromaTokenPlan(Copyable, Movable):
    var width: Int
    var height: Int
    var frames: Int
    var text_tokens: Int
    var latent_channels: Int
    var latent_h: Int
    var latent_w: Int
    var patch_size: Int
    var patch_grid_h: Int
    var patch_grid_w: Int
    var image_tokens: Int
    var patch_vector_dim: Int
    var latent_elements: Int
    var total_sequence: Int

    def validate_1024_contract(self) raises:
        if self.width != CHROMA_DEFAULT_WIDTH or self.height != CHROMA_DEFAULT_HEIGHT:
            raise Error("Chroma contract currently targets 1024x1024")
        if self.frames != CHROMA_DEFAULT_FRAMES:
            raise Error("Chroma contract is image-only")
        if self.text_tokens != CHROMA_T5_SEQ_LEN:
            raise Error("Chroma T5 token count must be 512")
        if self.latent_channels != CHROMA_LATENT_CHANNELS:
            raise Error("Chroma latent channel mismatch")
        if self.latent_h != CHROMA_LATENT_H or self.latent_w != CHROMA_LATENT_W:
            raise Error("Chroma latent grid must be 128x128")
        if (
            self.patch_grid_h != CHROMA_PATCH_GRID_H
            or self.patch_grid_w != CHROMA_PATCH_GRID_W
        ):
            raise Error("Chroma packed token grid must be 64x64")
        if self.image_tokens != CHROMA_IMAGE_TOKENS:
            raise Error("Chroma image token count must be 4096")
        if self.patch_vector_dim != CHROMA_PATCH_VECTOR_DIM:
            raise Error("Chroma packed token channel dim must be 64")
        if self.total_sequence != CHROMA_TOTAL_SEQUENCE:
            raise Error("Chroma image+text sequence mismatch")


def build_chroma_token_plan(
    width: Int, height: Int, frames: Int, text_tokens: Int
) raises -> ChromaTokenPlan:
    if frames != 1:
        raise Error("build_chroma_token_plan: Chroma is image-only")
    if text_tokens <= 0:
        raise Error("build_chroma_token_plan: text_tokens must be > 0")
    var lh = chroma_latent_spatial_dim(height)
    var lw = chroma_latent_spatial_dim(width)
    var gh = chroma_patch_grid_dim(lh)
    var gw = chroma_patch_grid_dim(lw)
    return ChromaTokenPlan(
        width,
        height,
        frames,
        text_tokens,
        CHROMA_LATENT_CHANNELS,
        lh,
        lw,
        CHROMA_PACK_PATCH,
        gh,
        gw,
        gh * gw,
        CHROMA_PATCH_VECTOR_DIM,
        CHROMA_LATENT_CHANNELS * lh * lw,
        gh * gw + text_tokens,
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
            String("Chroma contract int mismatch: ")
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
            String("Chroma contract float mismatch: ")
            + name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def _check_shape_and_bytes(
    name: String, dtype: STDtype, shape: List[Int], size: Int, expected_shape: List[Int]
) raises:
    if len(shape) != len(expected_shape):
        raise Error(
            String("Chroma tensor rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(shape)
            + String(" expected=")
            + _shape_string(expected_shape)
        )
    for i in range(len(expected_shape)):
        if shape[i] != expected_shape[i]:
            raise Error(
                String("Chroma tensor shape mismatch for ")
                + name
                + String(": actual=")
                + _shape_string(shape)
                + String(" expected=")
                + _shape_string(expected_shape)
            )
    var expected_nbytes = _numel(expected_shape) * dtype.byte_size()
    if size != expected_nbytes:
        raise Error(
            String("Chroma tensor byte-size mismatch for ")
            + name
            + String(": actual=")
            + String(size)
            + String(" expected=")
            + String(expected_nbytes)
        )


def _check_tensor(
    ref st: SafeTensors, name: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(
            String("Chroma tensor dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    _check_shape_and_bytes(name, info.dtype, info.shape, info.size, expected_shape)


def _check_sharded_tensor(
    ref st: ShardedSafeTensors, name: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(
            String("Chroma sharded tensor dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    _check_shape_and_bytes(name, info.dtype, info.shape, info.size, expected_shape)


def validate_chroma_local_paths() raises -> Int:
    _require_path(String("Chroma single DiT"), String(CHROMA_SINGLE_DIT_CHECKPOINT))
    _require_path(String("Chroma transformer config"), String(CHROMA_TRANSFORMER_CONFIG))
    _require_path(String("Chroma transformer index"), String(CHROMA_TRANSFORMER_INDEX))
    _require_path(String("Chroma transformer shard 0"), String(CHROMA_TRANSFORMER_SHARD_0))
    _require_path(String("Chroma transformer shard 1"), String(CHROMA_TRANSFORMER_SHARD_1))
    _require_path(String("Chroma text config"), String(CHROMA_TEXT_ENCODER_CONFIG))
    _require_path(String("Chroma text index"), String(CHROMA_TEXT_ENCODER_INDEX))
    _require_path(String("Chroma text shard 0"), String(CHROMA_TEXT_ENCODER_SHARD_0))
    _require_path(String("Chroma text shard 1"), String(CHROMA_TEXT_ENCODER_SHARD_1))
    _require_path(String("Chroma tokenizer model"), String(CHROMA_TOKENIZER_MODEL))
    _require_path(String("Chroma tokenizer config"), String(CHROMA_TOKENIZER_CONFIG))
    _require_path(String("Chroma scheduler config"), String(CHROMA_SCHEDULER_CONFIG))
    _require_path(String("Chroma VAE config"), String(CHROMA_VAE_CONFIG))
    _require_path(String("Chroma VAE weights"), String(CHROMA_VAE_FILE))
    return 14


def validate_chroma_static_contract() raises -> ChromaTokenPlan:
    var plan = build_chroma_token_plan(
        CHROMA_DEFAULT_WIDTH,
        CHROMA_DEFAULT_HEIGHT,
        CHROMA_DEFAULT_FRAMES,
        CHROMA_T5_SEQ_LEN,
    )
    plan.validate_1024_contract()
    _check_int(
        String("hidden=heads*head_dim"),
        CHROMA_DIT_HIDDEN,
        CHROMA_DIT_HEADS * CHROMA_DIT_HEAD_DIM,
    )
    _check_int(
        String("rope_axes=head_dim"),
        CHROMA_DIT_ROPE_AXIS_0 + CHROMA_DIT_ROPE_AXIS_1 + CHROMA_DIT_ROPE_AXIS_2,
        CHROMA_DIT_HEAD_DIM,
    )
    _check_int(
        String("mod_index"),
        CHROMA_DIT_MOD_INDEX,
        344,
    )
    _check_int(
        String("patch_vector_dim"),
        CHROMA_PATCH_VECTOR_DIM,
        CHROMA_DIT_APPROX_IN,
    )
    _check_close(
        String("sigma[0]"),
        chroma_shifted_sigma(0, CHROMA_DEFAULT_STEPS, chroma_schedule_shift()),
        1.0,
        0.000001,
    )
    _check_close(
        String("sigma[end]"),
        chroma_shifted_sigma(
            CHROMA_DEFAULT_STEPS, CHROMA_DEFAULT_STEPS, chroma_schedule_shift()
        ),
        0.0,
        0.000001,
    )
    return plan^


def validate_chroma_dit_header(checkpoint_path: String) raises:
    if not path_exists(checkpoint_path):
        raise Error(String("Chroma DiT checkpoint missing: ") + checkpoint_path)
    var st = SafeTensors.open(checkpoint_path)
    _check_int(String("single DiT tensor count"), st.count(), CHROMA_DIT_TENSORS)
    _check_chroma_dit_tensors(st)


def validate_chroma_transformer_sharded_header(transformer_dir: String) raises:
    var st = ShardedSafeTensors.open(transformer_dir)
    _check_int(
        String("sharded transformer tensor count"),
        st.num_tensors(),
        CHROMA_DIT_TENSORS,
    )
    _check_chroma_sharded_dit_tensors(st)


def _check_chroma_dit_tensors(ref st: SafeTensors) raises:
    _check_tensor(
        st,
        String("x_embedder.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_HIDDEN, CHROMA_PATCH_VECTOR_DIM),
    )
    _check_tensor(
        st,
        String("context_embedder.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_HIDDEN, CHROMA_DIT_CONTEXT_DIM),
    )
    _check_tensor(
        st,
        String("distilled_guidance_layer.in_proj.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_APPROX_HIDDEN, CHROMA_DIT_APPROX_IN),
    )
    _check_tensor(
        st,
        String("distilled_guidance_layer.layers.0.linear_1.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_APPROX_HIDDEN, CHROMA_DIT_APPROX_HIDDEN),
    )
    _check_tensor(
        st,
        String("distilled_guidance_layer.norms.4.weight"),
        STDtype.BF16,
        _shape1(CHROMA_DIT_APPROX_HIDDEN),
    )
    _check_tensor(
        st,
        String("distilled_guidance_layer.out_proj.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_HIDDEN, CHROMA_DIT_APPROX_HIDDEN),
    )
    _check_tensor(
        st,
        String("transformer_blocks.0.attn.to_q.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_HIDDEN, CHROMA_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("transformer_blocks.0.attn.add_q_proj.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_HIDDEN, CHROMA_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("transformer_blocks.0.attn.norm_q.weight"),
        STDtype.BF16,
        _shape1(CHROMA_DIT_HEAD_DIM),
    )
    _check_tensor(
        st,
        String("transformer_blocks.")
        + String(CHROMA_DIT_DOUBLE_BLOCKS - 1)
        + String(".ff.net.0.proj.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_MLP_HIDDEN, CHROMA_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("transformer_blocks.")
        + String(CHROMA_DIT_DOUBLE_BLOCKS - 1)
        + String(".ff_context.net.2.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_HIDDEN, CHROMA_DIT_MLP_HIDDEN),
    )
    _check_tensor(
        st,
        String("single_transformer_blocks.0.attn.to_q.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_HIDDEN, CHROMA_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("single_transformer_blocks.0.proj_mlp.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_MLP_HIDDEN, CHROMA_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("single_transformer_blocks.")
        + String(CHROMA_DIT_SINGLE_BLOCKS - 1)
        + String(".proj_out.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_HIDDEN, CHROMA_DIT_HIDDEN + CHROMA_DIT_MLP_HIDDEN),
    )
    _check_tensor(
        st,
        String("proj_out.weight"),
        STDtype.BF16,
        _shape2(CHROMA_PATCH_VECTOR_DIM, CHROMA_DIT_HIDDEN),
    )


def _check_chroma_sharded_dit_tensors(ref st: ShardedSafeTensors) raises:
    _check_sharded_tensor(
        st,
        String("x_embedder.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_HIDDEN, CHROMA_PATCH_VECTOR_DIM),
    )
    _check_sharded_tensor(
        st,
        String("transformer_blocks.0.attn.to_q.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_HIDDEN, CHROMA_DIT_HIDDEN),
    )
    _check_sharded_tensor(
        st,
        String("transformer_blocks.")
        + String(CHROMA_DIT_DOUBLE_BLOCKS - 1)
        + String(".attn.to_out.0.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_HIDDEN, CHROMA_DIT_HIDDEN),
    )
    _check_sharded_tensor(
        st,
        String("single_transformer_blocks.")
        + String(CHROMA_DIT_SINGLE_BLOCKS - 1)
        + String(".proj_out.weight"),
        STDtype.BF16,
        _shape2(CHROMA_DIT_HIDDEN, CHROMA_DIT_HIDDEN + CHROMA_DIT_MLP_HIDDEN),
    )
    _check_sharded_tensor(
        st,
        String("proj_out.weight"),
        STDtype.BF16,
        _shape2(CHROMA_PATCH_VECTOR_DIM, CHROMA_DIT_HIDDEN),
    )


def validate_chroma_text_encoder_header(text_encoder_dir: String) raises:
    var st = ShardedSafeTensors.open(text_encoder_dir)
    _check_int(
        String("T5 text encoder tensor count"),
        st.num_tensors(),
        CHROMA_TEXT_ENCODER_TENSORS,
    )
    _check_sharded_tensor(
        st,
        String("shared.weight"),
        STDtype.BF16,
        _shape2(CHROMA_T5_VOCAB, CHROMA_T5_HIDDEN),
    )
    _check_sharded_tensor(
        st,
        String("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
        STDtype.BF16,
        _shape2(32, CHROMA_T5_HEADS),
    )
    _check_sharded_tensor(
        st,
        String("encoder.block.0.layer.0.SelfAttention.q.weight"),
        STDtype.BF16,
        _shape2(CHROMA_T5_HIDDEN, CHROMA_T5_HIDDEN),
    )
    _check_sharded_tensor(
        st,
        String("encoder.block.0.layer.1.DenseReluDense.wi_0.weight"),
        STDtype.BF16,
        _shape2(CHROMA_T5_FFN_HIDDEN, CHROMA_T5_HIDDEN),
    )
    _check_sharded_tensor(
        st,
        String("encoder.block.23.layer.1.DenseReluDense.wo.weight"),
        STDtype.BF16,
        _shape2(CHROMA_T5_HIDDEN, CHROMA_T5_FFN_HIDDEN),
    )
    _check_sharded_tensor(
        st,
        String("encoder.final_layer_norm.weight"),
        STDtype.BF16,
        _shape1(CHROMA_T5_HIDDEN),
    )


def validate_chroma_vae_header(vae_path: String) raises:
    if not path_exists(vae_path):
        raise Error(String("Chroma VAE checkpoint missing: ") + vae_path)
    var st = SafeTensors.open(vae_path)
    _check_int(String("VAE tensor count"), st.count(), CHROMA_VAE_TENSORS)
    _check_tensor(
        st,
        String("decoder.conv_in.weight"),
        STDtype.BF16,
        _shape4(512, CHROMA_LATENT_CHANNELS, 3, 3),
    )
    _check_tensor(
        st,
        String("decoder.mid_block.attentions.0.to_q.weight"),
        STDtype.BF16,
        _shape2(512, 512),
    )
    _check_tensor(
        st,
        String("decoder.up_blocks.0.resnets.0.conv1.weight"),
        STDtype.BF16,
        _shape4(512, 512, 3, 3),
    )
    _check_tensor(
        st,
        String("decoder.conv_out.weight"),
        STDtype.BF16,
        _shape4(3, 128, 3, 3),
    )
    _check_tensor(
        st,
        String("encoder.conv_in.weight"),
        STDtype.BF16,
        _shape4(128, 3, 3, 3),
    )


def validate_chroma_default_checkpoint_contract() raises -> ChromaTokenPlan:
    var plan = validate_chroma_static_contract()
    _ = validate_chroma_local_paths()
    validate_chroma_dit_header(chroma_default_checkpoint_path())
    validate_chroma_transformer_sharded_header(chroma_transformer_dir())
    validate_chroma_text_encoder_header(chroma_text_encoder_dir())
    validate_chroma_vae_header(chroma_vae_path())
    return plan^
