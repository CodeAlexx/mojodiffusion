# models/dit/anima_contract.mojo - Anima metadata/header contract.
#
# Metadata-only gate for the Anima 2B image path. It validates the local
# safetensors headers and static shape facts from inference-flame without
# DeviceContext, H2D weight loads, denoise math, or VAE execution.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import path_exists


comptime ANIMA_ROOT = "/home/alex/.serenity/models/anima"
comptime ANIMA_DIT_PATH = (
    "/home/alex/.serenity/models/anima/split_files/diffusion_models/"
    "anima-base-v1.0.safetensors"
)
comptime ANIMA_QWEN3_PATH = (
    "/home/alex/.serenity/models/anima/split_files/text_encoders/"
    "qwen_3_06b_base.safetensors"
)
comptime ANIMA_VAE_PATH = (
    "/home/alex/.serenity/models/anima/split_files/vae/"
    "qwen_image_vae.safetensors"
)
comptime ANIMA_DEFAULT_CONDITIONING_PATH = (
    "/home/alex/EriDiffusion/inference-flame/output/anima_embeddings.safetensors"
)
comptime ANIMA_DEFAULT_RUST_LATENT_PATH = (
    "/home/alex/EriDiffusion/inference-flame/output/anima_rust_latent.safetensors"
)

comptime ANIMA_DEFAULT_WIDTH = 1024
comptime ANIMA_DEFAULT_HEIGHT = 1024
comptime ANIMA_DEFAULT_FRAMES = 1
comptime ANIMA_NUM_STEPS = 30
comptime ANIMA_CFG_SCALE_X10 = 45
comptime ANIMA_LATENT_DOWNSAMPLE = 8
comptime ANIMA_LATENT_CHANNELS = 16
comptime ANIMA_LATENT_T = 1
comptime ANIMA_LATENT_H = ANIMA_DEFAULT_HEIGHT // ANIMA_LATENT_DOWNSAMPLE
comptime ANIMA_LATENT_W = ANIMA_DEFAULT_WIDTH // ANIMA_LATENT_DOWNSAMPLE
comptime ANIMA_PATCH_SIZE = 2
comptime ANIMA_PATCH_GRID_H = ANIMA_LATENT_H // ANIMA_PATCH_SIZE
comptime ANIMA_PATCH_GRID_W = ANIMA_LATENT_W // ANIMA_PATCH_SIZE
comptime ANIMA_IMAGE_TOKENS = (
    ANIMA_LATENT_T * ANIMA_PATCH_GRID_H * ANIMA_PATCH_GRID_W
)
comptime ANIMA_PATCH_IN_DIM = (
    (ANIMA_LATENT_CHANNELS + 1) * ANIMA_PATCH_SIZE * ANIMA_PATCH_SIZE
)
comptime ANIMA_PATCH_OUT_DIM = (
    ANIMA_LATENT_CHANNELS * ANIMA_PATCH_SIZE * ANIMA_PATCH_SIZE
)
comptime ANIMA_HIDDEN = 2048
comptime ANIMA_DEPTH = 28
comptime ANIMA_NUM_HEADS = 16
comptime ANIMA_HEAD_DIM = 128
comptime ANIMA_MLP_HIDDEN = 8192
comptime ANIMA_ADALN_LORA_DIM = 256
comptime ANIMA_ADALN_DIM = 3 * ANIMA_HIDDEN
comptime ANIMA_ADAPTER_DIM = 1024
comptime ANIMA_ADAPTER_HEADS = 16
comptime ANIMA_ADAPTER_HEAD_DIM = 64
comptime ANIMA_ADAPTER_BLOCKS = 6
comptime ANIMA_ADAPTER_MLP_HIDDEN = 4096
comptime ANIMA_ADAPTER_VOCAB = 32128
comptime ANIMA_MAX_SEQ_LEN = 256
comptime ANIMA_QWEN3_VOCAB = 151936
comptime ANIMA_QWEN3_HIDDEN = 1024
comptime ANIMA_QWEN3_LAYERS = 28
comptime ANIMA_QWEN3_PAD_ID = 151643
comptime ANIMA_T5_PAD_ID = 0

comptime ANIMA_DIT_TENSOR_COUNT = 685
comptime ANIMA_QWEN3_TENSOR_COUNT = 310
comptime ANIMA_VAE_TENSOR_COUNT = 194


def anima_cfg_scale() -> Float32:
    return Float32(ANIMA_CFG_SCALE_X10) / 10.0


def anima_default_conditioning_path() -> String:
    return String(ANIMA_DEFAULT_CONDITIONING_PATH)


def anima_default_rust_latent_path() -> String:
    return String(ANIMA_DEFAULT_RUST_LATENT_PATH)


def anima_sigma(index: Int, num_steps: Int) raises -> Float32:
    if num_steps <= 0:
        raise Error("anima_sigma: num_steps must be > 0")
    if index < 0 or index > num_steps:
        raise Error("anima_sigma: index out of range")
    return 1.0 - Float32(index) / Float32(num_steps)


def anima_euler_delta(index: Int, num_steps: Int) raises -> Float32:
    if index < 0 or index >= num_steps:
        raise Error("anima_euler_delta: index out of range")
    return anima_sigma(index + 1, num_steps) - anima_sigma(index, num_steps)


def anima_latent_spatial_dim(image_dim: Int) raises -> Int:
    if image_dim <= 0:
        raise Error("anima_latent_spatial_dim: image_dim must be > 0")
    if image_dim % ANIMA_LATENT_DOWNSAMPLE != 0:
        raise Error("anima_latent_spatial_dim: image_dim must divide by 8")
    return image_dim // ANIMA_LATENT_DOWNSAMPLE


def anima_patch_grid_dim(latent_dim: Int) raises -> Int:
    if latent_dim <= 0:
        raise Error("anima_patch_grid_dim: latent_dim must be > 0")
    if latent_dim % ANIMA_PATCH_SIZE != 0:
        raise Error("anima_patch_grid_dim: latent_dim must divide by patch size")
    return latent_dim // ANIMA_PATCH_SIZE


struct AnimaTokenPlan(Copyable, Movable):
    var width: Int
    var height: Int
    var frames: Int
    var text_tokens: Int
    var latent_channels: Int
    var latent_t: Int
    var latent_h: Int
    var latent_w: Int
    var patch_grid_h: Int
    var patch_grid_w: Int
    var image_tokens: Int
    var patch_in_dim: Int
    var patch_out_dim: Int
    var latent_elements: Int

    def __init__(out self, width: Int, height: Int, frames: Int, text_tokens: Int) raises:
        if frames != 1:
            raise Error("AnimaTokenPlan: image-only contract expects one frame")
        if text_tokens <= 0:
            raise Error("AnimaTokenPlan: text_tokens must be > 0")
        self.width = width
        self.height = height
        self.frames = frames
        self.text_tokens = text_tokens
        self.latent_channels = ANIMA_LATENT_CHANNELS
        self.latent_t = 1
        self.latent_h = anima_latent_spatial_dim(height)
        self.latent_w = anima_latent_spatial_dim(width)
        self.patch_grid_h = anima_patch_grid_dim(self.latent_h)
        self.patch_grid_w = anima_patch_grid_dim(self.latent_w)
        self.image_tokens = self.latent_t * self.patch_grid_h * self.patch_grid_w
        self.patch_in_dim = (
            (self.latent_channels + 1) * ANIMA_PATCH_SIZE * ANIMA_PATCH_SIZE
        )
        self.patch_out_dim = self.latent_channels * ANIMA_PATCH_SIZE * ANIMA_PATCH_SIZE
        self.latent_elements = (
            self.latent_t * self.latent_h * self.latent_w * self.latent_channels
        )

    def validate_1024_contract(self) raises:
        if self.width != ANIMA_DEFAULT_WIDTH or self.height != ANIMA_DEFAULT_HEIGHT:
            raise Error("Anima contract currently targets 1024x1024")
        if self.frames != ANIMA_DEFAULT_FRAMES:
            raise Error("Anima contract is image-only")
        if self.latent_h != ANIMA_LATENT_H or self.latent_w != ANIMA_LATENT_W:
            raise Error("Anima latent grid must be 128x128")
        if (
            self.patch_grid_h != ANIMA_PATCH_GRID_H
            or self.patch_grid_w != ANIMA_PATCH_GRID_W
        ):
            raise Error("Anima patch grid must be 64x64")
        if self.image_tokens != ANIMA_IMAGE_TOKENS:
            raise Error("Anima image token count must be 4096")
        if self.patch_in_dim != ANIMA_PATCH_IN_DIM:
            raise Error("Anima patch input dim must be 68")
        if self.patch_out_dim != ANIMA_PATCH_OUT_DIM:
            raise Error("Anima patch output dim must be 64")


def build_anima_token_plan(
    width: Int, height: Int, frames: Int, text_tokens: Int
) raises -> AnimaTokenPlan:
    return AnimaTokenPlan(width, height, frames, text_tokens)


struct AnimaConditioningContract(Copyable, Movable):
    var batch: Int
    var text_tokens: Int
    var hidden: Int
    var has_uncond: Bool

    def __init__(
        out self, batch: Int, text_tokens: Int, hidden: Int, has_uncond: Bool
    ):
        self.batch = batch
        self.text_tokens = text_tokens
        self.hidden = hidden
        self.has_uncond = has_uncond

    def validate(self) raises:
        _check_int(String("conditioning_batch"), self.batch, 1)
        _check_int(String("conditioning_text_tokens"), self.text_tokens, ANIMA_MAX_SEQ_LEN)
        _check_int(String("conditioning_hidden"), self.hidden, ANIMA_ADAPTER_DIM)
        if not self.has_uncond:
            raise Error("Anima conditioning missing context_uncond")


struct AnimaLatentOracleContract(Copyable, Movable):
    var batch: Int
    var channels: Int
    var frames: Int
    var latent_h: Int
    var latent_w: Int

    def __init__(
        out self, batch: Int, channels: Int, frames: Int, latent_h: Int, latent_w: Int
    ):
        self.batch = batch
        self.channels = channels
        self.frames = frames
        self.latent_h = latent_h
        self.latent_w = latent_w

    def validate(self) raises:
        _check_int(String("latent_batch"), self.batch, 1)
        _check_int(String("latent_channels"), self.channels, ANIMA_LATENT_CHANNELS)
        _check_int(String("latent_frames"), self.frames, ANIMA_LATENT_T)
        _check_int(String("latent_h"), self.latent_h, ANIMA_LATENT_H)
        _check_int(String("latent_w"), self.latent_w, ANIMA_LATENT_W)


def _shape1(a: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    return out^


def _shape2(a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
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
            String("Anima contract int mismatch: ")
            + label
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
            String("Anima tensor dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    if len(info.shape) != len(expected_shape):
        raise Error(
            String("Anima tensor rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=")
            + _shape_string(expected_shape)
        )
    for i in range(len(expected_shape)):
        if info.shape[i] != expected_shape[i]:
            raise Error(
                String("Anima tensor shape mismatch for ")
                + name
                + String(": actual=")
                + _shape_string(info.shape)
                + String(" expected=")
                + _shape_string(expected_shape)
            )
    var expected_nbytes = _numel(expected_shape) * dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("Anima tensor byte-size mismatch for ")
            + name
            + String(": actual=")
            + String(info.size)
            + String(" expected=")
            + String(expected_nbytes)
        )


def _has_tensor(ref st: SafeTensors, name: String) -> Bool:
    var names = st.names()
    for i in range(len(names)):
        if names[i] == name:
            return True
    return False


def _check_conditioning_tensor(
    ref st: SafeTensors, name: String
) raises -> AnimaConditioningContract:
    var info = st.tensor_info(name)
    if info.dtype != STDtype.BF16 and info.dtype != STDtype.F32:
        raise Error(
            String("Anima conditioning dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=BF16 or F32")
        )
    if len(info.shape) != 3:
        raise Error(
            String("Anima conditioning rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=[1, 256, 1024]")
        )
    if info.shape[0] != 1:
        raise Error(
            String("Anima conditioning batch mismatch for ")
            + name
            + String(": actual=")
            + String(info.shape[0])
            + String(" expected=1")
        )
    if info.shape[1] != ANIMA_MAX_SEQ_LEN:
        raise Error(
            String("Anima conditioning token mismatch for ")
            + name
            + String(": actual=")
            + String(info.shape[1])
            + String(" expected=")
            + String(ANIMA_MAX_SEQ_LEN)
        )
    if info.shape[2] != ANIMA_ADAPTER_DIM:
        raise Error(
            String("Anima conditioning hidden mismatch for ")
            + name
            + String(": actual=")
            + String(info.shape[2])
            + String(" expected=")
            + String(ANIMA_ADAPTER_DIM)
        )
    var expected_nbytes = _numel(info.shape) * info.dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("Anima conditioning byte-size mismatch for ")
            + name
            + String(": actual=")
            + String(info.size)
            + String(" expected=")
            + String(expected_nbytes)
        )
    return AnimaConditioningContract(
        info.shape[0], info.shape[1], info.shape[2], False
    )


def validate_anima_local_paths() raises -> Int:
    _require_path(String("Anima root"), String(ANIMA_ROOT))
    _require_path(String("Anima DiT"), String(ANIMA_DIT_PATH))
    _require_path(String("Anima Qwen3 encoder"), String(ANIMA_QWEN3_PATH))
    _require_path(String("Anima Qwen-Image VAE"), String(ANIMA_VAE_PATH))
    return 4


def validate_anima_static_contract() raises -> AnimaTokenPlan:
    var plan = build_anima_token_plan(
        ANIMA_DEFAULT_WIDTH,
        ANIMA_DEFAULT_HEIGHT,
        ANIMA_DEFAULT_FRAMES,
        ANIMA_MAX_SEQ_LEN,
    )
    plan.validate_1024_contract()
    _check_int(String("hidden"), ANIMA_HIDDEN, 2048)
    _check_int(String("depth"), ANIMA_DEPTH, 28)
    _check_int(String("num_heads"), ANIMA_NUM_HEADS, 16)
    _check_int(String("head_dim"), ANIMA_HEAD_DIM, 128)
    _check_int(String("mlp_hidden"), ANIMA_MLP_HIDDEN, 8192)
    _check_int(String("adapter_dim"), ANIMA_ADAPTER_DIM, 1024)
    _check_int(String("adapter_blocks"), ANIMA_ADAPTER_BLOCKS, 6)
    _check_int(String("adapter_vocab"), ANIMA_ADAPTER_VOCAB, 32128)
    _check_int(String("qwen3_vocab"), ANIMA_QWEN3_VOCAB, 151936)
    _check_int(String("qwen3_layers"), ANIMA_QWEN3_LAYERS, 28)
    if anima_cfg_scale() != 4.5:
        raise Error("Anima CFG scale must default to 4.5")
    if anima_sigma(0, ANIMA_NUM_STEPS) != 1.0:
        raise Error("Anima sigma schedule must start at 1.0")
    if anima_sigma(ANIMA_NUM_STEPS, ANIMA_NUM_STEPS) != 0.0:
        raise Error("Anima sigma schedule must end at 0.0")
    if anima_euler_delta(0, ANIMA_NUM_STEPS) >= 0.0:
        raise Error("Anima Euler delta should be sigma_next - sigma")
    return plan^


def validate_anima_dit_header() raises:
    var dit = ShardedSafeTensors.open(String(ANIMA_DIT_PATH))
    _check_int(String("DiT tensor count"), dit.num_tensors(), ANIMA_DIT_TENSOR_COUNT)
    _check_tensor(
        dit,
        String("net.x_embedder.proj.1.weight"),
        STDtype.BF16,
        _shape2(ANIMA_HIDDEN, ANIMA_PATCH_IN_DIM),
    )
    _check_tensor(
        dit,
        String("net.t_embedder.1.linear_1.weight"),
        STDtype.BF16,
        _shape2(ANIMA_HIDDEN, ANIMA_HIDDEN),
    )
    _check_tensor(
        dit,
        String("net.t_embedder.1.linear_2.weight"),
        STDtype.BF16,
        _shape2(ANIMA_ADALN_DIM, ANIMA_HIDDEN),
    )
    _check_tensor(
        dit,
        String("net.t_embedding_norm.weight"),
        STDtype.BF16,
        _shape1(ANIMA_HIDDEN),
    )
    _check_tensor(
        dit,
        String("net.final_layer.adaln_modulation.1.weight"),
        STDtype.BF16,
        _shape2(ANIMA_ADALN_LORA_DIM, ANIMA_HIDDEN),
    )
    _check_tensor(
        dit,
        String("net.final_layer.adaln_modulation.2.weight"),
        STDtype.BF16,
        _shape2(2 * ANIMA_HIDDEN, ANIMA_ADALN_LORA_DIM),
    )
    _check_tensor(
        dit,
        String("net.final_layer.linear.weight"),
        STDtype.BF16,
        _shape2(ANIMA_PATCH_OUT_DIM, ANIMA_HIDDEN),
    )
    _check_tensor(
        dit,
        String("net.llm_adapter.embed.weight"),
        STDtype.BF16,
        _shape2(ANIMA_ADAPTER_VOCAB, ANIMA_ADAPTER_DIM),
    )
    _check_tensor(
        dit,
        String("net.llm_adapter.blocks.0.self_attn.q_proj.weight"),
        STDtype.BF16,
        _shape2(ANIMA_ADAPTER_DIM, ANIMA_ADAPTER_DIM),
    )
    _check_tensor(
        dit,
        String("net.llm_adapter.blocks.0.cross_attn.k_proj.weight"),
        STDtype.BF16,
        _shape2(ANIMA_ADAPTER_DIM, ANIMA_QWEN3_HIDDEN),
    )
    _check_tensor(
        dit,
        String("net.llm_adapter.blocks.5.mlp.2.weight"),
        STDtype.BF16,
        _shape2(ANIMA_ADAPTER_DIM, ANIMA_ADAPTER_MLP_HIDDEN),
    )
    _check_tensor(
        dit,
        String("net.blocks.0.self_attn.q_proj.weight"),
        STDtype.BF16,
        _shape2(ANIMA_HIDDEN, ANIMA_HIDDEN),
    )
    _check_tensor(
        dit,
        String("net.blocks.0.cross_attn.k_proj.weight"),
        STDtype.BF16,
        _shape2(ANIMA_HIDDEN, ANIMA_ADAPTER_DIM),
    )
    _check_tensor(
        dit,
        String("net.blocks.0.mlp.layer1.weight"),
        STDtype.BF16,
        _shape2(ANIMA_MLP_HIDDEN, ANIMA_HIDDEN),
    )
    _check_tensor(
        dit,
        String("net.blocks.27.mlp.layer2.weight"),
        STDtype.BF16,
        _shape2(ANIMA_HIDDEN, ANIMA_MLP_HIDDEN),
    )


def validate_anima_qwen3_header() raises:
    var qwen3 = ShardedSafeTensors.open(String(ANIMA_QWEN3_PATH))
    _check_int(
        String("Qwen3 tensor count"), qwen3.num_tensors(), ANIMA_QWEN3_TENSOR_COUNT
    )
    _check_tensor(
        qwen3,
        String("model.embed_tokens.weight"),
        STDtype.BF16,
        _shape2(ANIMA_QWEN3_VOCAB, ANIMA_QWEN3_HIDDEN),
    )
    _check_tensor(
        qwen3,
        String("model.layers.0.self_attn.q_proj.weight"),
        STDtype.BF16,
        _shape2(2 * ANIMA_QWEN3_HIDDEN, ANIMA_QWEN3_HIDDEN),
    )
    _check_tensor(
        qwen3,
        String("model.layers.27.mlp.down_proj.weight"),
        STDtype.BF16,
        _shape2(ANIMA_QWEN3_HIDDEN, 3072),
    )
    _check_tensor(
        qwen3,
        String("model.norm.weight"),
        STDtype.BF16,
        _shape1(ANIMA_QWEN3_HIDDEN),
    )


def validate_anima_vae_header() raises:
    var vae = ShardedSafeTensors.open(String(ANIMA_VAE_PATH))
    _check_int(String("VAE tensor count"), vae.num_tensors(), ANIMA_VAE_TENSOR_COUNT)
    _check_tensor(
        vae,
        String("encoder.conv1.weight"),
        STDtype.BF16,
        _shape5(96, 3, 3, 3, 3),
    )
    _check_tensor(
        vae,
        String("decoder.conv1.weight"),
        STDtype.BF16,
        _shape5(384, ANIMA_LATENT_CHANNELS, 3, 3, 3),
    )
    _check_tensor(
        vae,
        String("decoder.head.2.weight"),
        STDtype.BF16,
        _shape5(3, 96, 3, 3, 3),
    )


def validate_anima_metadata_contract() raises -> AnimaTokenPlan:
    _ = validate_anima_local_paths()
    var plan = validate_anima_static_contract()
    validate_anima_dit_header()
    validate_anima_qwen3_header()
    validate_anima_vae_header()
    return plan^


def validate_anima_conditioning_header(
    embeddings_path: String
) raises -> AnimaConditioningContract:
    if not path_exists(embeddings_path):
        raise Error(String("Anima conditioning missing: ") + embeddings_path)

    var st = SafeTensors.open(embeddings_path)
    if st.count() != 2:
        raise Error(
            String("Anima conditioning tensor count mismatch: actual=")
            + String(st.count())
            + String(" expected=2")
        )
    var names = st.names()
    for i in range(len(names)):
        if names[i] != String("context_cond") and names[i] != String("context_uncond"):
            raise Error(
                String("Anima conditioning unexpected tensor: ")
                + names[i]
                + String(" expected only context_cond/context_uncond")
            )

    var cond = _check_conditioning_tensor(st, String("context_cond"))
    if not _has_tensor(st, String("context_uncond")):
        raise Error("Anima conditioning missing context_uncond")
    var uncond = _check_conditioning_tensor(st, String("context_uncond"))
    if (
        uncond.batch != cond.batch
        or uncond.text_tokens != cond.text_tokens
        or uncond.hidden != cond.hidden
    ):
        raise Error("Anima conditioning shape mismatch between cond/uncond")

    var result = AnimaConditioningContract(
        cond.batch, cond.text_tokens, cond.hidden, True
    )
    result.validate()
    return result^


def validate_anima_rust_latent_header(
    latent_path: String
) raises -> AnimaLatentOracleContract:
    if not path_exists(latent_path):
        raise Error(String("Anima Rust latent missing: ") + latent_path)

    var st = SafeTensors.open(latent_path)
    if st.count() != 1:
        raise Error(
            String("Anima Rust latent tensor count mismatch: actual=")
            + String(st.count())
            + String(" expected=1")
        )
    var names = st.names()
    if len(names) != 1 or names[0] != String("latent"):
        raise Error("Anima Rust latent expected only tensor 'latent'")

    var info = st.tensor_info(String("latent"))
    if info.dtype != STDtype.F32:
        raise Error(
            String("Anima Rust latent dtype mismatch: actual=")
            + info.dtype.name()
            + String(" expected=F32")
        )
    if len(info.shape) != 5:
        raise Error(
            String("Anima Rust latent rank mismatch: actual=")
            + _shape_string(info.shape)
            + String(" expected=[1, 16, 1, 128, 128]")
        )
    var result = AnimaLatentOracleContract(
        info.shape[0], info.shape[1], info.shape[2], info.shape[3], info.shape[4]
    )
    result.validate()
    var expected_nbytes = _numel(info.shape) * info.dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("Anima Rust latent byte-size mismatch: actual=")
            + String(info.size)
            + String(" expected=")
            + String(expected_nbytes)
        )
    return result^
