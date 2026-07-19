# models/dit/qwenimage_edit_contract.mojo - Qwen-Image-Edit metadata contract.
#
# Header-only gate for Qwen-Image-Edit-2511. It validates the local edit
# snapshot, 5-shard 60-block DiT, Qwen2.5-VL text encoder, Qwen image VAE, and
# edit-specific target+reference token geometry without DeviceContext setup,
# H2D weight loads, denoise math, or VAE decode.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.runtime.model_manifest import ModelFamily, ModelManifest
from serenitymojo.sampling.flow_match import build_qwen_sigma_schedule, qwen_mu


comptime QWENIMAGE_EDIT_ROOT = (
    "/home/alex/.cache/huggingface/hub/"
    "models--Qwen--Qwen-Image-Edit-2511/"
    "snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
)
comptime QWENIMAGE_EDIT_TRANSFORMER_DIR = QWENIMAGE_EDIT_ROOT + "/transformer"
comptime QWENIMAGE_EDIT_TEXT_ENCODER_DIR = QWENIMAGE_EDIT_ROOT + "/text_encoder"
comptime QWENIMAGE_EDIT_TOKENIZER_FILE = QWENIMAGE_EDIT_ROOT + "/processor/tokenizer.json"
comptime QWENIMAGE_EDIT_VAE_FILE = QWENIMAGE_EDIT_ROOT + "/vae/diffusion_pytorch_model.safetensors"

comptime QWENIMAGE_EDIT_DEFAULT_WIDTH = 1024
comptime QWENIMAGE_EDIT_DEFAULT_HEIGHT = 1024
comptime QWENIMAGE_EDIT_DEFAULT_FRAMES = 1
comptime QWENIMAGE_EDIT_DEFAULT_STEPS = 50
comptime QWENIMAGE_EDIT_DEFAULT_CFG_X10 = 40
comptime QWENIMAGE_EDIT_LATENT_DOWNSAMPLE = 8
comptime QWENIMAGE_EDIT_LATENT_CHANNELS = 16
comptime QWENIMAGE_EDIT_LATENT_H = (
    QWENIMAGE_EDIT_DEFAULT_HEIGHT // QWENIMAGE_EDIT_LATENT_DOWNSAMPLE
)
comptime QWENIMAGE_EDIT_LATENT_W = (
    QWENIMAGE_EDIT_DEFAULT_WIDTH // QWENIMAGE_EDIT_LATENT_DOWNSAMPLE
)
comptime QWENIMAGE_EDIT_PATCH_SIZE = 2
comptime QWENIMAGE_EDIT_PATCH_GRID_H = QWENIMAGE_EDIT_LATENT_H // QWENIMAGE_EDIT_PATCH_SIZE
comptime QWENIMAGE_EDIT_PATCH_GRID_W = QWENIMAGE_EDIT_LATENT_W // QWENIMAGE_EDIT_PATCH_SIZE
comptime QWENIMAGE_EDIT_TARGET_TOKENS = (
    QWENIMAGE_EDIT_PATCH_GRID_H * QWENIMAGE_EDIT_PATCH_GRID_W
)
comptime QWENIMAGE_EDIT_REFERENCE_TOKENS = QWENIMAGE_EDIT_TARGET_TOKENS
comptime QWENIMAGE_EDIT_IMAGE_TOKENS = (
    QWENIMAGE_EDIT_TARGET_TOKENS + QWENIMAGE_EDIT_REFERENCE_TOKENS
)
comptime QWENIMAGE_EDIT_TEXT_MAX_TOKENS = 1024
comptime QWENIMAGE_EDIT_TOTAL_SEQUENCE = (
    QWENIMAGE_EDIT_IMAGE_TOKENS + QWENIMAGE_EDIT_TEXT_MAX_TOKENS
)
comptime QWENIMAGE_EDIT_PACKED_CHANNELS = (
    QWENIMAGE_EDIT_LATENT_CHANNELS
    * QWENIMAGE_EDIT_PATCH_SIZE
    * QWENIMAGE_EDIT_PATCH_SIZE
)
comptime QWENIMAGE_EDIT_DROP_IDX = 34
comptime QWENIMAGE_EDIT_PAD_ID = 151643
comptime QWENIMAGE_EDIT_ZERO_COND_T = True

comptime QWENIMAGE_EDIT_DIT_HIDDEN = 3072
comptime QWENIMAGE_EDIT_DIT_LAYERS = 60
comptime QWENIMAGE_EDIT_DIT_HEADS = 24
comptime QWENIMAGE_EDIT_DIT_HEAD_DIM = 128
comptime QWENIMAGE_EDIT_DIT_IN_CHANNELS = 64
comptime QWENIMAGE_EDIT_DIT_JOINT_ATTENTION_DIM = 3584
comptime QWENIMAGE_EDIT_DIT_MLP_HIDDEN = 12288
comptime QWENIMAGE_EDIT_DIT_TIMESTEP_DIM = 256
comptime QWENIMAGE_EDIT_DIT_FINAL_ADALN_DIM = 2 * QWENIMAGE_EDIT_DIT_HIDDEN
comptime QWENIMAGE_EDIT_TEXT_VOCAB = 152064
comptime QWENIMAGE_EDIT_TEXT_HIDDEN = 3584
comptime QWENIMAGE_EDIT_TEXT_LAYERS = 28
comptime QWENIMAGE_EDIT_TEXT_HEADS = 28
comptime QWENIMAGE_EDIT_TEXT_KV_HEADS = 4
comptime QWENIMAGE_EDIT_TEXT_HEAD_DIM = 128
comptime QWENIMAGE_EDIT_TEXT_Q_DIM = QWENIMAGE_EDIT_TEXT_HEADS * QWENIMAGE_EDIT_TEXT_HEAD_DIM
comptime QWENIMAGE_EDIT_TEXT_KV_DIM = (
    QWENIMAGE_EDIT_TEXT_KV_HEADS * QWENIMAGE_EDIT_TEXT_HEAD_DIM
)
comptime QWENIMAGE_EDIT_TEXT_INTERMEDIATE = 18944

comptime QWENIMAGE_EDIT_TRANSFORMER_TENSORS = 1933
comptime QWENIMAGE_EDIT_TEXT_ENCODER_TENSORS = 729
comptime QWENIMAGE_EDIT_VAE_TENSORS = 194


def qwenimage_edit_default_cfg_scale() -> Float32:
    return Float32(QWENIMAGE_EDIT_DEFAULT_CFG_X10) / 10.0


def qwenimage_edit_latent_spatial_dim(image_dim: Int) raises -> Int:
    if image_dim <= 0:
        raise Error("qwenimage_edit_latent_spatial_dim: image_dim must be > 0")
    if image_dim % QWENIMAGE_EDIT_LATENT_DOWNSAMPLE != 0:
        raise Error("Qwen-Image-Edit dimension must divide by latent_downsample=8")
    return image_dim // QWENIMAGE_EDIT_LATENT_DOWNSAMPLE


def qwenimage_edit_patch_grid_dim(latent_dim: Int) raises -> Int:
    if latent_dim <= 0:
        raise Error("qwenimage_edit_patch_grid_dim: latent_dim must be > 0")
    if latent_dim % QWENIMAGE_EDIT_PATCH_SIZE != 0:
        raise Error("Qwen-Image-Edit latent dimension must divide by patch_size=2")
    return latent_dim // QWENIMAGE_EDIT_PATCH_SIZE


@fieldwise_init
struct QwenImageEditTokenPlan(Copyable, Movable):
    var width: Int
    var height: Int
    var frames: Int
    var latent_channels: Int
    var latent_h: Int
    var latent_w: Int
    var patch_grid_h: Int
    var patch_grid_w: Int
    var target_tokens: Int
    var reference_tokens: Int
    var image_tokens: Int
    var text_tokens: Int
    var total_sequence: Int
    var packed_channels: Int

    def validate_1024_contract(self) raises:
        if self.width != QWENIMAGE_EDIT_DEFAULT_WIDTH or self.height != QWENIMAGE_EDIT_DEFAULT_HEIGHT:
            raise Error("Qwen-Image-Edit contract currently targets 1024x1024")
        if self.frames != QWENIMAGE_EDIT_DEFAULT_FRAMES:
            raise Error("Qwen-Image-Edit contract is image-only")
        if self.latent_channels != QWENIMAGE_EDIT_LATENT_CHANNELS:
            raise Error("Qwen-Image-Edit latent channel mismatch")
        if self.latent_h != QWENIMAGE_EDIT_LATENT_H or self.latent_w != QWENIMAGE_EDIT_LATENT_W:
            raise Error("Qwen-Image-Edit latent grid must be 128x128")
        if self.target_tokens != QWENIMAGE_EDIT_TARGET_TOKENS:
            raise Error("Qwen-Image-Edit target token mismatch")
        if self.reference_tokens != QWENIMAGE_EDIT_REFERENCE_TOKENS:
            raise Error("Qwen-Image-Edit reference token mismatch")
        if self.image_tokens != QWENIMAGE_EDIT_IMAGE_TOKENS:
            raise Error("Qwen-Image-Edit total image token mismatch")
        if self.total_sequence != QWENIMAGE_EDIT_TOTAL_SEQUENCE:
            raise Error("Qwen-Image-Edit total sequence mismatch")


def build_qwenimage_edit_token_plan(
    width: Int, height: Int, frames: Int, text_tokens: Int
) raises -> QwenImageEditTokenPlan:
    if frames != 1:
        raise Error("build_qwenimage_edit_token_plan: edit path expects image frames=1")
    var lh = qwenimage_edit_latent_spatial_dim(height)
    var lw = qwenimage_edit_latent_spatial_dim(width)
    var gh = qwenimage_edit_patch_grid_dim(lh)
    var gw = qwenimage_edit_patch_grid_dim(lw)
    var target = gh * gw
    return QwenImageEditTokenPlan(
        width,
        height,
        frames,
        QWENIMAGE_EDIT_LATENT_CHANNELS,
        lh,
        lw,
        gh,
        gw,
        target,
        target,
        target * 2,
        text_tokens,
        target * 2 + text_tokens,
        QWENIMAGE_EDIT_PACKED_CHANNELS,
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


def _shape5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    out.append(e)
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
            String("Qwen-Image-Edit float mismatch: ")
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
            String("Qwen-Image-Edit tensor dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    if len(info.shape) != len(expected_shape):
        raise Error(
            String("Qwen-Image-Edit tensor rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=")
            + _shape_string(expected_shape)
        )
    for i in range(len(expected_shape)):
        if info.shape[i] != expected_shape[i]:
            raise Error(
                String("Qwen-Image-Edit tensor shape mismatch for ")
                + name
                + String(": actual=")
                + _shape_string(info.shape)
                + String(" expected=")
                + _shape_string(expected_shape)
            )
    var expected_nbytes = _numel(expected_shape) * info.dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error("Qwen-Image-Edit tensor byte-size mismatch for " + name)


def validate_qwenimage_edit_manifest_contract(manifest: ModelManifest) raises:
    if manifest.model_id != "qwen_image_edit":
        raise Error(String("Qwen-Image-Edit contract got manifest: ") + manifest.model_id)
    if manifest.family != ModelFamily.image_to_image():
        raise Error("Qwen-Image-Edit manifest family mismatch")
    if manifest.variant != "qwen-image-edit-2511":
        raise Error("Qwen-Image-Edit manifest variant mismatch")
    if manifest.image_tokens != QWENIMAGE_EDIT_IMAGE_TOKENS:
        raise Error("Qwen-Image-Edit manifest image tokens mismatch")
    if manifest.text_tokens != QWENIMAGE_EDIT_TEXT_MAX_TOKENS:
        raise Error("Qwen-Image-Edit manifest text tokens mismatch")
    if manifest.total_sequence != QWENIMAGE_EDIT_TOTAL_SEQUENCE:
        raise Error("Qwen-Image-Edit manifest sequence mismatch")
    if manifest.patch_size != QWENIMAGE_EDIT_PATCH_SIZE:
        raise Error("Qwen-Image-Edit manifest patch size mismatch")


def validate_qwenimage_edit_local_paths() raises -> Int:
    _require_path(String("Qwen-Image-Edit root"), String(QWENIMAGE_EDIT_ROOT))
    _require_path(
        String("Qwen-Image-Edit model_index"),
        String(QWENIMAGE_EDIT_ROOT + "/model_index.json"),
    )
    _require_path(
        String("Qwen-Image-Edit transformer config"),
        String(QWENIMAGE_EDIT_TRANSFORMER_DIR + "/config.json"),
    )
    _require_path(
        String("Qwen-Image-Edit transformer index"),
        String(QWENIMAGE_EDIT_TRANSFORMER_DIR + "/diffusion_pytorch_model.safetensors.index.json"),
    )
    for shard in range(1, 6):
        _require_path(
            String("Qwen-Image-Edit transformer shard ") + String(shard),
            String(
                QWENIMAGE_EDIT_TRANSFORMER_DIR
                + "/diffusion_pytorch_model-0000"
                + String(shard)
                + "-of-00005.safetensors"
            ),
        )
    _require_path(
        String("Qwen-Image-Edit text config"),
        String(QWENIMAGE_EDIT_TEXT_ENCODER_DIR + "/config.json"),
    )
    _require_path(
        String("Qwen-Image-Edit text index"),
        String(QWENIMAGE_EDIT_TEXT_ENCODER_DIR + "/model.safetensors.index.json"),
    )
    for shard in range(1, 5):
        _require_path(
            String("Qwen-Image-Edit text shard ") + String(shard),
            String(
                QWENIMAGE_EDIT_TEXT_ENCODER_DIR
                + "/model-0000"
                + String(shard)
                + "-of-00004.safetensors"
            ),
        )
    _require_path(String("Qwen-Image-Edit tokenizer"), String(QWENIMAGE_EDIT_TOKENIZER_FILE))
    _require_path(
        String("Qwen-Image-Edit processor template"),
        String(QWENIMAGE_EDIT_ROOT + "/processor/chat_template.jinja"),
    )
    _require_path(
        String("Qwen-Image-Edit scheduler"),
        String(QWENIMAGE_EDIT_ROOT + "/scheduler/scheduler_config.json"),
    )
    _require_path(String("Qwen-Image-Edit VAE config"), String(QWENIMAGE_EDIT_ROOT + "/vae/config.json"))
    _require_path(String("Qwen-Image-Edit VAE weights"), String(QWENIMAGE_EDIT_VAE_FILE))
    return 20


def validate_qwenimage_edit_static_contract() raises -> QwenImageEditTokenPlan:
    var plan = build_qwenimage_edit_token_plan(
        QWENIMAGE_EDIT_DEFAULT_WIDTH,
        QWENIMAGE_EDIT_DEFAULT_HEIGHT,
        QWENIMAGE_EDIT_DEFAULT_FRAMES,
        QWENIMAGE_EDIT_TEXT_MAX_TOKENS,
    )
    plan.validate_1024_contract()
    _check_int(
        String("hidden"), QWENIMAGE_EDIT_DIT_HIDDEN,
        QWENIMAGE_EDIT_DIT_HEADS * QWENIMAGE_EDIT_DIT_HEAD_DIM,
    )
    _check_int(String("depth"), QWENIMAGE_EDIT_DIT_LAYERS, 60)
    _check_close(
        String("mu_4096"),
        qwen_mu(Float32(QWENIMAGE_EDIT_TARGET_TOKENS)),
        0.6935484,
        0.00001,
    )
    var sigmas = build_qwen_sigma_schedule(
        QWENIMAGE_EDIT_DEFAULT_STEPS, Float32(QWENIMAGE_EDIT_TARGET_TOKENS)
    )
    _check_int(String("sigma_count"), len(sigmas), QWENIMAGE_EDIT_DEFAULT_STEPS + 1)
    _check_close(String("sigma0"), sigmas[0], 1.0, 0.000001)
    _check_close(
        String("sigma_preterminal"),
        sigmas[QWENIMAGE_EDIT_DEFAULT_STEPS - 1],
        0.02,
        0.000001,
    )
    _check_close(String("sigma_end"), sigmas[QWENIMAGE_EDIT_DEFAULT_STEPS], 0.0, 0.000001)
    return plan^


def validate_qwenimage_edit_transformer_header() raises:
    var st = ShardedSafeTensors.open(String(QWENIMAGE_EDIT_TRANSFORMER_DIR))
    _check_int(
        String("transformer_tensors"),
        st.num_tensors(),
        QWENIMAGE_EDIT_TRANSFORMER_TENSORS,
    )
    _check_tensor(
        st,
        String("img_in.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_EDIT_DIT_HIDDEN, QWENIMAGE_EDIT_DIT_IN_CHANNELS),
    )
    _check_tensor(
        st,
        String("txt_norm.weight"),
        STDtype.BF16,
        _shape1(QWENIMAGE_EDIT_DIT_JOINT_ATTENTION_DIM),
    )
    _check_tensor(
        st,
        String("txt_in.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_EDIT_DIT_HIDDEN, QWENIMAGE_EDIT_DIT_JOINT_ATTENTION_DIM),
    )
    _check_tensor(
        st,
        String("time_text_embed.timestep_embedder.linear_1.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_EDIT_DIT_HIDDEN, QWENIMAGE_EDIT_DIT_TIMESTEP_DIM),
    )
    _check_tensor(
        st,
        String("transformer_blocks.0.attn.to_q.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_EDIT_DIT_HIDDEN, QWENIMAGE_EDIT_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("transformer_blocks.59.attn.to_v.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_EDIT_DIT_HIDDEN, QWENIMAGE_EDIT_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("norm_out.linear.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_EDIT_DIT_FINAL_ADALN_DIM, QWENIMAGE_EDIT_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("proj_out.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_EDIT_PACKED_CHANNELS, QWENIMAGE_EDIT_DIT_HIDDEN),
    )


def validate_qwenimage_edit_text_encoder_header() raises:
    var st = ShardedSafeTensors.open(String(QWENIMAGE_EDIT_TEXT_ENCODER_DIR))
    _check_int(
        String("text_encoder_tensors"),
        st.num_tensors(),
        QWENIMAGE_EDIT_TEXT_ENCODER_TENSORS,
    )
    _check_tensor(
        st,
        String("model.embed_tokens.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_EDIT_TEXT_VOCAB, QWENIMAGE_EDIT_TEXT_HIDDEN),
    )
    _check_tensor(
        st,
        String("model.layers.0.self_attn.q_proj.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_EDIT_TEXT_Q_DIM, QWENIMAGE_EDIT_TEXT_HIDDEN),
    )
    _check_tensor(
        st,
        String("model.layers.0.self_attn.k_proj.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_EDIT_TEXT_KV_DIM, QWENIMAGE_EDIT_TEXT_HIDDEN),
    )
    _check_tensor(
        st,
        String("model.layers.27.mlp.down_proj.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_EDIT_TEXT_HIDDEN, QWENIMAGE_EDIT_TEXT_INTERMEDIATE),
    )
    _check_tensor(
        st,
        String("model.norm.weight"),
        STDtype.BF16,
        _shape1(QWENIMAGE_EDIT_TEXT_HIDDEN),
    )


def validate_qwenimage_edit_vae_header() raises:
    var st = ShardedSafeTensors.open(String(QWENIMAGE_EDIT_VAE_FILE))
    _check_int(String("vae_tensors"), st.num_tensors(), QWENIMAGE_EDIT_VAE_TENSORS)
    _check_tensor(
        st,
        String("post_quant_conv.weight"),
        STDtype.BF16,
        _shape5(16, 16, 1, 1, 1),
    )
    _check_tensor(
        st,
        String("decoder.conv_in.weight"),
        STDtype.BF16,
        _shape5(384, 16, 3, 3, 3),
    )
    _check_tensor(
        st,
        String("decoder.conv_out.weight"),
        STDtype.BF16,
        _shape5(3, 96, 3, 3, 3),
    )


def validate_qwenimage_edit_metadata_contract(
    manifest: ModelManifest,
) raises -> QwenImageEditTokenPlan:
    validate_qwenimage_edit_manifest_contract(manifest)
    var plan = validate_qwenimage_edit_static_contract()
    validate_qwenimage_edit_transformer_header()
    validate_qwenimage_edit_text_encoder_header()
    validate_qwenimage_edit_vae_header()
    return plan^
