# models/dit/sd3_contract.mojo - SD3.5 manifest/header/schedule contracts.
#
# Metadata-only helpers for the SD3.5 Large/Medium 1024 lanes. This file intentionally
# has no DeviceContext, MMDiT, VAE, or text-encoder runtime dependency.

from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import validate_manifest_paths
from serenitymojo.runtime.model_manifest import ModelFamily, ModelManifest


comptime SD3_LARGE_WIDTH = 1024
comptime SD3_LARGE_HEIGHT = 1024
comptime SD3_LARGE_LATENT_DOWNSAMPLE = 8
comptime SD3_LARGE_LATENT_H = SD3_LARGE_HEIGHT // SD3_LARGE_LATENT_DOWNSAMPLE
comptime SD3_LARGE_LATENT_W = SD3_LARGE_WIDTH // SD3_LARGE_LATENT_DOWNSAMPLE
comptime SD3_LARGE_LATENT_CHANNELS = 16
comptime SD3_LARGE_PATCH_SIZE = 2
comptime SD3_LARGE_PATCH_GRID_H = SD3_LARGE_LATENT_H // SD3_LARGE_PATCH_SIZE
comptime SD3_LARGE_PATCH_GRID_W = SD3_LARGE_LATENT_W // SD3_LARGE_PATCH_SIZE
comptime SD3_LARGE_IMAGE_TOKENS = (
    SD3_LARGE_PATCH_GRID_H * SD3_LARGE_PATCH_GRID_W
)
comptime SD3_LARGE_PATCH_VECTOR_DIM = (
    SD3_LARGE_LATENT_CHANNELS * SD3_LARGE_PATCH_SIZE * SD3_LARGE_PATCH_SIZE
)
comptime SD3_LARGE_LATENT_ELEMENTS = (
    SD3_LARGE_LATENT_CHANNELS * SD3_LARGE_LATENT_H * SD3_LARGE_LATENT_W
)
comptime SD3_LARGE_CLIP_SEQ_LEN = 77
comptime SD3_LARGE_T5_SEQ_LEN = 256
comptime SD3_LARGE_TEXT_TOKENS = (
    SD3_LARGE_CLIP_SEQ_LEN + SD3_LARGE_CLIP_SEQ_LEN + SD3_LARGE_T5_SEQ_LEN
)
comptime SD3_LARGE_TOTAL_SEQUENCE = (
    SD3_LARGE_IMAGE_TOKENS + SD3_LARGE_TEXT_TOKENS
)
comptime SD3_LARGE_CONTEXT_DIM = 4096
comptime SD3_LARGE_POOLED_DIM = 2048
comptime SD3_LARGE_HIDDEN = 2432
comptime SD3_LARGE_DEPTH = 38
comptime SD3_LARGE_NUM_HEADS = 38
comptime SD3_LARGE_HEAD_DIM = 64
comptime SD3_LARGE_TIMESTEP_DIM = 256
comptime SD3_LARGE_POS_EMBED_TOKENS = 36864
comptime SD3_LARGE_POS_EMBED_GRID = 192
comptime SD3_LARGE_NUM_STEPS = 28
comptime SD3_LARGE_CHECKED_PATHS = 10
comptime SD3_LARGE_CHECKPOINT_TENSORS = 1167

comptime SD3_MEDIUM_WIDTH = 1024
comptime SD3_MEDIUM_HEIGHT = 1024
comptime SD3_MEDIUM_LATENT_DOWNSAMPLE = 8
comptime SD3_MEDIUM_LATENT_H = SD3_MEDIUM_HEIGHT // SD3_MEDIUM_LATENT_DOWNSAMPLE
comptime SD3_MEDIUM_LATENT_W = SD3_MEDIUM_WIDTH // SD3_MEDIUM_LATENT_DOWNSAMPLE
comptime SD3_MEDIUM_LATENT_CHANNELS = 16
comptime SD3_MEDIUM_PATCH_SIZE = 2
comptime SD3_MEDIUM_PATCH_GRID_H = SD3_MEDIUM_LATENT_H // SD3_MEDIUM_PATCH_SIZE
comptime SD3_MEDIUM_PATCH_GRID_W = SD3_MEDIUM_LATENT_W // SD3_MEDIUM_PATCH_SIZE
comptime SD3_MEDIUM_IMAGE_TOKENS = (
    SD3_MEDIUM_PATCH_GRID_H * SD3_MEDIUM_PATCH_GRID_W
)
comptime SD3_MEDIUM_PATCH_VECTOR_DIM = (
    SD3_MEDIUM_LATENT_CHANNELS * SD3_MEDIUM_PATCH_SIZE * SD3_MEDIUM_PATCH_SIZE
)
comptime SD3_MEDIUM_LATENT_ELEMENTS = (
    SD3_MEDIUM_LATENT_CHANNELS * SD3_MEDIUM_LATENT_H * SD3_MEDIUM_LATENT_W
)
comptime SD3_MEDIUM_CLIP_SEQ_LEN = 77
comptime SD3_MEDIUM_T5_SEQ_LEN = 256
comptime SD3_MEDIUM_TEXT_TOKENS = (
    SD3_MEDIUM_CLIP_SEQ_LEN + SD3_MEDIUM_CLIP_SEQ_LEN + SD3_MEDIUM_T5_SEQ_LEN
)
comptime SD3_MEDIUM_TOTAL_SEQUENCE = (
    SD3_MEDIUM_IMAGE_TOKENS + SD3_MEDIUM_TEXT_TOKENS
)
comptime SD3_MEDIUM_CONTEXT_DIM = 4096
comptime SD3_MEDIUM_POOLED_DIM = 2048
comptime SD3_MEDIUM_HIDDEN = 1536
comptime SD3_MEDIUM_DEPTH = 24
comptime SD3_MEDIUM_NUM_HEADS = 24
comptime SD3_MEDIUM_HEAD_DIM = 64
comptime SD3_MEDIUM_TIMESTEP_DIM = 256
comptime SD3_MEDIUM_POS_EMBED_TOKENS = 147456
comptime SD3_MEDIUM_POS_EMBED_GRID = 384
comptime SD3_MEDIUM_DUAL_ATTENTION_BLOCKS = 13
comptime SD3_MEDIUM_NUM_STEPS = 28
comptime SD3_MEDIUM_CHECKED_PATHS = 10
comptime SD3_MEDIUM_CHECKPOINT_TENSORS = 909


def sd3_large_cfg_scale() -> Float32:
    return 4.5


def sd3_large_schedule_shift() -> Float32:
    return 3.0


def sd3_medium_cfg_scale() -> Float32:
    return 4.5


def sd3_medium_schedule_shift() -> Float32:
    return 3.0


def sd3_large_vae_scale() -> Float32:
    return 1.5305


def sd3_large_vae_shift() -> Float32:
    return 0.0609


def sd3_medium_vae_scale() -> Float32:
    return 1.5305


def sd3_medium_vae_shift() -> Float32:
    return 0.0609


def sd3_large_model_timestep(sigma: Float32) -> Float32:
    return sigma * 1000.0


def sd3_medium_model_timestep(sigma: Float32) -> Float32:
    return sigma * 1000.0


def sd3_shifted_sigma(index: Int, num_steps: Int, shift: Float32) raises -> Float32:
    if num_steps <= 0:
        raise Error("sd3_shifted_sigma: num_steps must be > 0")
    if index < 0 or index > num_steps:
        raise Error("sd3_shifted_sigma: index out of range")
    var t = 1.0 - Float32(index) / Float32(num_steps)
    return shift * t / (1.0 + (shift - 1.0) * t)


def sd3_schedule_delta(index: Int, num_steps: Int, shift: Float32) raises -> Float32:
    if index < 0 or index >= num_steps:
        raise Error("sd3_schedule_delta: index out of range")
    var sigma = sd3_shifted_sigma(index, num_steps, shift)
    var sigma_next = sd3_shifted_sigma(index + 1, num_steps, shift)
    return sigma_next - sigma


def build_sd3_shifted_schedule(num_steps: Int, shift: Float32) raises -> List[Float32]:
    if num_steps <= 0:
        raise Error("build_sd3_shifted_schedule: num_steps must be > 0")
    var result = List[Float32]()
    for i in range(num_steps + 1):
        result.append(sd3_shifted_sigma(i, num_steps, shift))
    return result^


def sd3_large_latent_spatial_dim(image_dim: Int) raises -> Int:
    if image_dim <= 0:
        raise Error("sd3_large_latent_spatial_dim: image_dim must be > 0")
    if image_dim % SD3_LARGE_LATENT_DOWNSAMPLE != 0:
        raise Error("sd3_large_latent_spatial_dim: image_dim must divide by 8")
    return image_dim // SD3_LARGE_LATENT_DOWNSAMPLE


def sd3_large_patch_grid_dim(latent_dim: Int) raises -> Int:
    if latent_dim <= 0:
        raise Error("sd3_large_patch_grid_dim: latent_dim must be > 0")
    if latent_dim % SD3_LARGE_PATCH_SIZE != 0:
        raise Error("sd3_large_patch_grid_dim: latent_dim must divide by patch size")
    return latent_dim // SD3_LARGE_PATCH_SIZE


def sd3_medium_latent_spatial_dim(image_dim: Int) raises -> Int:
    if image_dim <= 0:
        raise Error("sd3_medium_latent_spatial_dim: image_dim must be > 0")
    if image_dim % SD3_MEDIUM_LATENT_DOWNSAMPLE != 0:
        raise Error("sd3_medium_latent_spatial_dim: image_dim must divide by 8")
    return image_dim // SD3_MEDIUM_LATENT_DOWNSAMPLE


def sd3_medium_patch_grid_dim(latent_dim: Int) raises -> Int:
    if latent_dim <= 0:
        raise Error("sd3_medium_patch_grid_dim: latent_dim must be > 0")
    if latent_dim % SD3_MEDIUM_PATCH_SIZE != 0:
        raise Error("sd3_medium_patch_grid_dim: latent_dim must divide by patch size")
    return latent_dim // SD3_MEDIUM_PATCH_SIZE


struct SD3LargeTokenPlan(Copyable, Movable):
    var width: Int
    var height: Int
    var text_tokens: Int
    var latent_channels: Int
    var patch_size: Int
    var latent_h: Int
    var latent_w: Int
    var patch_grid_h: Int
    var patch_grid_w: Int
    var image_tokens: Int
    var patch_vector_dim: Int
    var latent_elements: Int
    var total_sequence: Int

    def __init__(out self, width: Int, height: Int, text_tokens: Int) raises:
        if text_tokens <= 0:
            raise Error("SD3LargeTokenPlan: text_tokens must be > 0")
        self.width = width
        self.height = height
        self.text_tokens = text_tokens
        self.latent_channels = SD3_LARGE_LATENT_CHANNELS
        self.patch_size = SD3_LARGE_PATCH_SIZE
        self.latent_h = sd3_large_latent_spatial_dim(height)
        self.latent_w = sd3_large_latent_spatial_dim(width)
        self.patch_grid_h = sd3_large_patch_grid_dim(self.latent_h)
        self.patch_grid_w = sd3_large_patch_grid_dim(self.latent_w)
        self.image_tokens = self.patch_grid_h * self.patch_grid_w
        self.patch_vector_dim = self.latent_channels * self.patch_size * self.patch_size
        self.latent_elements = self.latent_channels * self.latent_h * self.latent_w
        self.total_sequence = self.image_tokens + self.text_tokens

    def validate_large_1024_contract(self) raises:
        if self.width != SD3_LARGE_WIDTH or self.height != SD3_LARGE_HEIGHT:
            raise Error("SD3.5 Large contract currently targets 1024x1024")
        if (
            self.latent_h != SD3_LARGE_LATENT_H
            or self.latent_w != SD3_LARGE_LATENT_W
        ):
            raise Error("SD3.5 Large latent grid must be 128x128")
        if (
            self.patch_grid_h != SD3_LARGE_PATCH_GRID_H
            or self.patch_grid_w != SD3_LARGE_PATCH_GRID_W
        ):
            raise Error("SD3.5 Large patch grid must be 64x64")
        if self.image_tokens != SD3_LARGE_IMAGE_TOKENS:
            raise Error("SD3.5 Large image token count must be 4096")
        if self.patch_vector_dim != SD3_LARGE_PATCH_VECTOR_DIM:
            raise Error("SD3.5 Large patch vector dim must be 64")
        if self.latent_elements != SD3_LARGE_LATENT_ELEMENTS:
            raise Error("SD3.5 Large latent element count mismatch")
        if self.total_sequence != SD3_LARGE_TOTAL_SEQUENCE:
            raise Error("SD3.5 Large total sequence mismatch")


def build_sd3_large_token_plan(
    width: Int, height: Int, text_tokens: Int
) raises -> SD3LargeTokenPlan:
    return SD3LargeTokenPlan(width, height, text_tokens)


struct SD3MediumTokenPlan(Copyable, Movable):
    var width: Int
    var height: Int
    var text_tokens: Int
    var latent_channels: Int
    var patch_size: Int
    var latent_h: Int
    var latent_w: Int
    var patch_grid_h: Int
    var patch_grid_w: Int
    var image_tokens: Int
    var patch_vector_dim: Int
    var latent_elements: Int
    var total_sequence: Int

    def __init__(out self, width: Int, height: Int, text_tokens: Int) raises:
        if text_tokens <= 0:
            raise Error("SD3MediumTokenPlan: text_tokens must be > 0")
        self.width = width
        self.height = height
        self.text_tokens = text_tokens
        self.latent_channels = SD3_MEDIUM_LATENT_CHANNELS
        self.patch_size = SD3_MEDIUM_PATCH_SIZE
        self.latent_h = sd3_medium_latent_spatial_dim(height)
        self.latent_w = sd3_medium_latent_spatial_dim(width)
        self.patch_grid_h = sd3_medium_patch_grid_dim(self.latent_h)
        self.patch_grid_w = sd3_medium_patch_grid_dim(self.latent_w)
        self.image_tokens = self.patch_grid_h * self.patch_grid_w
        self.patch_vector_dim = self.latent_channels * self.patch_size * self.patch_size
        self.latent_elements = self.latent_channels * self.latent_h * self.latent_w
        self.total_sequence = self.image_tokens + self.text_tokens

    def validate_medium_1024_contract(self) raises:
        if self.width != SD3_MEDIUM_WIDTH or self.height != SD3_MEDIUM_HEIGHT:
            raise Error("SD3.5 Medium contract currently targets 1024x1024")
        if (
            self.latent_h != SD3_MEDIUM_LATENT_H
            or self.latent_w != SD3_MEDIUM_LATENT_W
        ):
            raise Error("SD3.5 Medium latent grid must be 128x128")
        if (
            self.patch_grid_h != SD3_MEDIUM_PATCH_GRID_H
            or self.patch_grid_w != SD3_MEDIUM_PATCH_GRID_W
        ):
            raise Error("SD3.5 Medium patch grid must be 64x64")
        if self.image_tokens != SD3_MEDIUM_IMAGE_TOKENS:
            raise Error("SD3.5 Medium image token count must be 4096")
        if self.patch_vector_dim != SD3_MEDIUM_PATCH_VECTOR_DIM:
            raise Error("SD3.5 Medium patch vector dim must be 64")
        if self.latent_elements != SD3_MEDIUM_LATENT_ELEMENTS:
            raise Error("SD3.5 Medium latent element count mismatch")
        if self.total_sequence != SD3_MEDIUM_TOTAL_SEQUENCE:
            raise Error("SD3.5 Medium total sequence mismatch")


def build_sd3_medium_token_plan(
    width: Int, height: Int, text_tokens: Int
) raises -> SD3MediumTokenPlan:
    return SD3MediumTokenPlan(width, height, text_tokens)


def _shape1(a: Int) -> List[Int]:
    var result = List[Int]()
    result.append(a)
    return result^


def _shape2(a: Int, b: Int) -> List[Int]:
    var result = List[Int]()
    result.append(a)
    result.append(b)
    return result^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var result = List[Int]()
    result.append(a)
    result.append(b)
    result.append(c)
    return result^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var result = List[Int]()
    result.append(a)
    result.append(b)
    result.append(c)
    result.append(d)
    return result^


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


def _check_bool(name: String, got: Bool, expected: Bool) raises:
    if got != expected:
        raise Error(String("SD3 contract bool mismatch: ") + name)


def _check_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            String("SD3 contract int mismatch: ")
            + name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def _check_string(name: String, got: String, expected: String) raises:
    if got != expected:
        raise Error(
            String("SD3 contract string mismatch: ")
            + name
            + String(" got=")
            + got
            + String(" expected=")
            + expected
        )


def _check_tensor_shape(
    ref st: ShardedSafeTensors, name: String, expected_shape: List[Int]
) raises:
    var info = st.tensor_info(name)
    if len(info.shape) != len(expected_shape):
        raise Error(
            String("SD3 tensor rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=")
            + _shape_string(expected_shape)
        )
    for i in range(len(expected_shape)):
        if info.shape[i] != expected_shape[i]:
            raise Error(
                String("SD3 tensor shape mismatch for ")
                + name
                + String(": actual=")
                + _shape_string(info.shape)
                + String(" expected=")
                + _shape_string(expected_shape)
            )
    var expected_nbytes = _numel(expected_shape) * info.dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("SD3 tensor byte-size mismatch for ")
            + name
            + String(": actual=")
            + String(info.size)
            + String(" expected=")
            + String(expected_nbytes)
        )


def validate_sd3_large_manifest_contract(manifest: ModelManifest) raises:
    _check_string(String("model_id"), manifest.model_id, String("sd3_5_large"))
    _check_bool(
        String("family"),
        manifest.family == ModelFamily.text_to_image(),
        True,
    )
    _check_string(
        String("variant"),
        manifest.variant,
        String("stable-diffusion-v3.5-large"),
    )
    _check_string(
        String("profile"),
        manifest.profile_name,
        String("sd3_5_large_1024"),
    )
    _check_string(
        String("entry"),
        manifest.production_entry,
        String("serenitymojo/pipeline/sd3_pipeline_contract_smoke.mojo"),
    )
    _check_int(String("width"), manifest.default_width, SD3_LARGE_WIDTH)
    _check_int(String("height"), manifest.default_height, SD3_LARGE_HEIGHT)
    _check_int(String("frames"), manifest.default_frames, 1)
    _check_int(
        String("latent_channels"),
        manifest.latent_channels,
        SD3_LARGE_LATENT_CHANNELS,
    )
    _check_int(
        String("latent_downsample_s"),
        manifest.latent_downsample_s,
        SD3_LARGE_LATENT_DOWNSAMPLE,
    )
    _check_int(String("latent_height"), manifest.latent_height(), SD3_LARGE_LATENT_H)
    _check_int(String("latent_width"), manifest.latent_width(), SD3_LARGE_LATENT_W)
    _check_int(String("image_tokens"), manifest.image_tokens, SD3_LARGE_IMAGE_TOKENS)
    _check_int(String("text_tokens"), manifest.text_tokens, SD3_LARGE_TEXT_TOKENS)
    _check_int(
        String("total_sequence"),
        manifest.total_sequence,
        SD3_LARGE_TOTAL_SEQUENCE,
    )
    _check_int(String("patch_size"), manifest.patch_size, SD3_LARGE_PATCH_SIZE)
    _check_bool(String("uses_vae"), manifest.uses_vae(), True)
    _check_bool(
        String("embedded_vae_path"),
        manifest.vae_path == manifest.denoiser_path,
        True,
    )
    var plan = build_sd3_large_token_plan(
        manifest.default_width, manifest.default_height, manifest.text_tokens
    )
    plan.validate_large_1024_contract()
    _check_int(
        String("manifest patch_grid_h"), plan.patch_grid_h, SD3_LARGE_PATCH_GRID_H
    )
    _check_int(
        String("manifest patch_grid_w"), plan.patch_grid_w, SD3_LARGE_PATCH_GRID_W
    )
    _check_int(
        String("manifest patch_vector_dim"),
        plan.patch_vector_dim,
        SD3_LARGE_PATCH_VECTOR_DIM,
    )
    _check_int(
        String("manifest latent_elements"),
        plan.latent_elements,
        SD3_LARGE_LATENT_ELEMENTS,
    )
    _check_int(
        String("pos_embed square grid"),
        SD3_LARGE_POS_EMBED_GRID * SD3_LARGE_POS_EMBED_GRID,
        SD3_LARGE_POS_EMBED_TOKENS,
    )
    if (
        plan.patch_grid_h > SD3_LARGE_POS_EMBED_GRID
        or plan.patch_grid_w > SD3_LARGE_POS_EMBED_GRID
    ):
        raise Error("SD3.5 Large patch grid exceeds learned pos_embed grid")

    var status = validate_manifest_paths(manifest)
    _check_int(String("manifest path checks"), status.checked, SD3_LARGE_CHECKED_PATHS)
    _check_int(String("manifest missing paths"), status.missing, 0)


def validate_sd3_medium_manifest_contract(manifest: ModelManifest) raises:
    _check_string(String("model_id"), manifest.model_id, String("sd3_5_medium"))
    _check_bool(
        String("family"),
        manifest.family == ModelFamily.text_to_image(),
        True,
    )
    _check_string(
        String("variant"),
        manifest.variant,
        String("stable-diffusion-v3.5-medium"),
    )
    _check_string(
        String("profile"),
        manifest.profile_name,
        String("sd3_5_medium_1024"),
    )
    _check_string(
        String("entry"),
        manifest.production_entry,
        String("serenitymojo/pipeline/sd3_medium_pipeline_contract_smoke.mojo"),
    )
    _check_int(String("width"), manifest.default_width, SD3_MEDIUM_WIDTH)
    _check_int(String("height"), manifest.default_height, SD3_MEDIUM_HEIGHT)
    _check_int(String("frames"), manifest.default_frames, 1)
    _check_int(
        String("latent_channels"),
        manifest.latent_channels,
        SD3_MEDIUM_LATENT_CHANNELS,
    )
    _check_int(
        String("latent_downsample_s"),
        manifest.latent_downsample_s,
        SD3_MEDIUM_LATENT_DOWNSAMPLE,
    )
    _check_int(String("latent_height"), manifest.latent_height(), SD3_MEDIUM_LATENT_H)
    _check_int(String("latent_width"), manifest.latent_width(), SD3_MEDIUM_LATENT_W)
    _check_int(String("image_tokens"), manifest.image_tokens, SD3_MEDIUM_IMAGE_TOKENS)
    _check_int(String("text_tokens"), manifest.text_tokens, SD3_MEDIUM_TEXT_TOKENS)
    _check_int(
        String("total_sequence"),
        manifest.total_sequence,
        SD3_MEDIUM_TOTAL_SEQUENCE,
    )
    _check_int(String("patch_size"), manifest.patch_size, SD3_MEDIUM_PATCH_SIZE)
    _check_bool(String("uses_vae"), manifest.uses_vae(), True)
    _check_bool(
        String("embedded_vae_path"),
        manifest.vae_path == manifest.denoiser_path,
        True,
    )
    var plan = build_sd3_medium_token_plan(
        manifest.default_width, manifest.default_height, manifest.text_tokens
    )
    plan.validate_medium_1024_contract()
    _check_int(
        String("manifest patch_grid_h"), plan.patch_grid_h, SD3_MEDIUM_PATCH_GRID_H
    )
    _check_int(
        String("manifest patch_grid_w"), plan.patch_grid_w, SD3_MEDIUM_PATCH_GRID_W
    )
    _check_int(
        String("manifest patch_vector_dim"),
        plan.patch_vector_dim,
        SD3_MEDIUM_PATCH_VECTOR_DIM,
    )
    _check_int(
        String("manifest latent_elements"),
        plan.latent_elements,
        SD3_MEDIUM_LATENT_ELEMENTS,
    )
    _check_int(
        String("pos_embed square grid"),
        SD3_MEDIUM_POS_EMBED_GRID * SD3_MEDIUM_POS_EMBED_GRID,
        SD3_MEDIUM_POS_EMBED_TOKENS,
    )
    if (
        plan.patch_grid_h > SD3_MEDIUM_POS_EMBED_GRID
        or plan.patch_grid_w > SD3_MEDIUM_POS_EMBED_GRID
    ):
        raise Error("SD3.5 Medium patch grid exceeds learned pos_embed grid")

    var status = validate_manifest_paths(manifest)
    _check_int(String("manifest path checks"), status.checked, SD3_MEDIUM_CHECKED_PATHS)
    _check_int(String("manifest missing paths"), status.missing, 0)


def validate_sd3_large_checkpoint_header(manifest: ModelManifest) raises:
    var st = ShardedSafeTensors.open(manifest.denoiser_path)
    _check_int(String("tensor_count"), st.num_tensors(), SD3_LARGE_CHECKPOINT_TENSORS)
    _check_tensor_shape(
        st,
        String("model.diffusion_model.pos_embed"),
        _shape3(1, SD3_LARGE_POS_EMBED_TOKENS, SD3_LARGE_HIDDEN),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.x_embedder.proj.weight"),
        _shape4(
            SD3_LARGE_HIDDEN,
            SD3_LARGE_LATENT_CHANNELS,
            SD3_LARGE_PATCH_SIZE,
            SD3_LARGE_PATCH_SIZE,
        ),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.final_layer.linear.weight"),
        _shape2(
            SD3_LARGE_LATENT_CHANNELS
            * SD3_LARGE_PATCH_SIZE
            * SD3_LARGE_PATCH_SIZE,
            SD3_LARGE_HIDDEN,
        ),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.context_embedder.weight"),
        _shape2(SD3_LARGE_HIDDEN, SD3_LARGE_CONTEXT_DIM),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.y_embedder.mlp.0.weight"),
        _shape2(SD3_LARGE_HIDDEN, SD3_LARGE_POOLED_DIM),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.t_embedder.mlp.0.weight"),
        _shape2(SD3_LARGE_HIDDEN, SD3_LARGE_TIMESTEP_DIM),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.joint_blocks.0.x_block.adaLN_modulation.1.weight"),
        _shape2(6 * SD3_LARGE_HIDDEN, SD3_LARGE_HIDDEN),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.joint_blocks.")
        + String(SD3_LARGE_DEPTH - 1)
        + String(".context_block.adaLN_modulation.1.weight"),
        _shape2(2 * SD3_LARGE_HIDDEN, SD3_LARGE_HIDDEN),
    )
    _check_tensor_shape(
        st,
        String("first_stage_model.decoder.conv_in.weight"),
        _shape4(512, SD3_LARGE_LATENT_CHANNELS, 3, 3),
    )


def validate_sd3_medium_checkpoint_header(manifest: ModelManifest) raises:
    var st = ShardedSafeTensors.open(manifest.denoiser_path)
    _check_int(String("tensor_count"), st.num_tensors(), SD3_MEDIUM_CHECKPOINT_TENSORS)
    _check_tensor_shape(
        st,
        String("model.diffusion_model.pos_embed"),
        _shape3(1, SD3_MEDIUM_POS_EMBED_TOKENS, SD3_MEDIUM_HIDDEN),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.x_embedder.proj.weight"),
        _shape4(
            SD3_MEDIUM_HIDDEN,
            SD3_MEDIUM_LATENT_CHANNELS,
            SD3_MEDIUM_PATCH_SIZE,
            SD3_MEDIUM_PATCH_SIZE,
        ),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.final_layer.linear.weight"),
        _shape2(
            SD3_MEDIUM_LATENT_CHANNELS
            * SD3_MEDIUM_PATCH_SIZE
            * SD3_MEDIUM_PATCH_SIZE,
            SD3_MEDIUM_HIDDEN,
        ),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.context_embedder.weight"),
        _shape2(SD3_MEDIUM_HIDDEN, SD3_MEDIUM_CONTEXT_DIM),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.y_embedder.mlp.0.weight"),
        _shape2(SD3_MEDIUM_HIDDEN, SD3_MEDIUM_POOLED_DIM),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.t_embedder.mlp.0.weight"),
        _shape2(SD3_MEDIUM_HIDDEN, SD3_MEDIUM_TIMESTEP_DIM),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.joint_blocks.0.x_block.adaLN_modulation.1.weight"),
        _shape2(9 * SD3_MEDIUM_HIDDEN, SD3_MEDIUM_HIDDEN),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.joint_blocks.")
        + String(SD3_MEDIUM_DUAL_ATTENTION_BLOCKS - 1)
        + String(".x_block.attn2.qkv.weight"),
        _shape2(3 * SD3_MEDIUM_HIDDEN, SD3_MEDIUM_HIDDEN),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.joint_blocks.")
        + String(SD3_MEDIUM_DUAL_ATTENTION_BLOCKS - 1)
        + String(".x_block.attn2.ln_q.weight"),
        _shape1(SD3_MEDIUM_HEAD_DIM),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.joint_blocks.")
        + String(SD3_MEDIUM_DUAL_ATTENTION_BLOCKS)
        + String(".x_block.adaLN_modulation.1.weight"),
        _shape2(6 * SD3_MEDIUM_HIDDEN, SD3_MEDIUM_HIDDEN),
    )
    _check_tensor_shape(
        st,
        String("model.diffusion_model.joint_blocks.")
        + String(SD3_MEDIUM_DEPTH - 1)
        + String(".context_block.adaLN_modulation.1.weight"),
        _shape2(2 * SD3_MEDIUM_HIDDEN, SD3_MEDIUM_HIDDEN),
    )
    _check_tensor_shape(
        st,
        String("first_stage_model.decoder.conv_in.weight"),
        _shape4(512, SD3_MEDIUM_LATENT_CHANNELS, 3, 3),
    )


def validate_sd3_large_pipeline_contract(manifest: ModelManifest) raises:
    validate_sd3_large_manifest_contract(manifest)
    validate_sd3_large_checkpoint_header(manifest)


def validate_sd3_medium_pipeline_contract(manifest: ModelManifest) raises:
    validate_sd3_medium_manifest_contract(manifest)
    validate_sd3_medium_checkpoint_header(manifest)
