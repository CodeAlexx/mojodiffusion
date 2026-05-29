# models/dit/qwenimage_contract.mojo - Qwen-Image metadata/header contract.
#
# Header-only gate for Qwen-Image-2512. It validates the local diffusers
# snapshot, Qwen2.5-VL text encoder, 60-block Qwen MMDiT, Qwen image VAE, and
# dynamic FlowMatch schedule facts without DeviceContext setup, H2D weight
# loads, denoise math, or VAE decode.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import path_exists
from serenitymojo.runtime.model_manifest import ModelFamily, ModelManifest
from serenitymojo.sampling.flow_match import build_qwen_sigma_schedule, qwen_mu


comptime QWENIMAGE_ROOT = "/home/alex/.serenity/models/checkpoints/qwen-image-2512"
comptime QWENIMAGE_TRANSFORMER_DIR = QWENIMAGE_ROOT + "/transformer"
comptime QWENIMAGE_TEXT_ENCODER_DIR = QWENIMAGE_ROOT + "/text_encoder"
comptime QWENIMAGE_TOKENIZER_FILE = QWENIMAGE_ROOT + "/tokenizer/tokenizer.json"
comptime QWENIMAGE_VAE_FILE = QWENIMAGE_ROOT + "/vae/diffusion_pytorch_model.safetensors"

comptime QWENIMAGE_DEFAULT_WIDTH = 1024
comptime QWENIMAGE_DEFAULT_HEIGHT = 1024
comptime QWENIMAGE_DEFAULT_FRAMES = 1
comptime QWENIMAGE_DEFAULT_STEPS = 50
comptime QWENIMAGE_DEFAULT_CFG_X10 = 40
comptime QWENIMAGE_SCHEDULER_TRAIN_STEPS = 1000
comptime QWENIMAGE_SCHEDULER_BASE_SHIFT_X10 = 5
comptime QWENIMAGE_SCHEDULER_MAX_SHIFT_X10 = 9
comptime QWENIMAGE_SCHEDULER_SHIFT_TERMINAL_X100 = 2
comptime QWENIMAGE_LATENT_DOWNSAMPLE = 8
comptime QWENIMAGE_LATENT_CHANNELS = 16
comptime QWENIMAGE_LATENT_H = QWENIMAGE_DEFAULT_HEIGHT // QWENIMAGE_LATENT_DOWNSAMPLE
comptime QWENIMAGE_LATENT_W = QWENIMAGE_DEFAULT_WIDTH // QWENIMAGE_LATENT_DOWNSAMPLE
comptime QWENIMAGE_PATCH_SIZE = 2
comptime QWENIMAGE_PATCH_GRID_H = QWENIMAGE_LATENT_H // QWENIMAGE_PATCH_SIZE
comptime QWENIMAGE_PATCH_GRID_W = QWENIMAGE_LATENT_W // QWENIMAGE_PATCH_SIZE
comptime QWENIMAGE_IMAGE_TOKENS = QWENIMAGE_PATCH_GRID_H * QWENIMAGE_PATCH_GRID_W
comptime QWENIMAGE_TEXT_MAX_TOKENS = 1024
comptime QWENIMAGE_DROP_IDX = 34
comptime QWENIMAGE_PAD_ID = 151643
comptime QWENIMAGE_TOTAL_SEQUENCE = QWENIMAGE_IMAGE_TOKENS + QWENIMAGE_TEXT_MAX_TOKENS
comptime QWENIMAGE_PACKED_CHANNELS = (
    QWENIMAGE_LATENT_CHANNELS * QWENIMAGE_PATCH_SIZE * QWENIMAGE_PATCH_SIZE
)

comptime QWENIMAGE_DIT_HIDDEN = 3072
comptime QWENIMAGE_DIT_LAYERS = 60
comptime QWENIMAGE_DIT_HEADS = 24
comptime QWENIMAGE_DIT_HEAD_DIM = 128
comptime QWENIMAGE_DIT_IN_CHANNELS = 64
comptime QWENIMAGE_DIT_OUT_CHANNELS = 16
comptime QWENIMAGE_DIT_JOINT_ATTENTION_DIM = 3584
comptime QWENIMAGE_DIT_MLP_HIDDEN = 12288
comptime QWENIMAGE_DIT_TIMESTEP_DIM = 256
comptime QWENIMAGE_DIT_ROPE_AXIS_0 = 16
comptime QWENIMAGE_DIT_ROPE_AXIS_1 = 56
comptime QWENIMAGE_DIT_ROPE_AXIS_2 = 56
comptime QWENIMAGE_DIT_ROPE_THETA = 10000
comptime QWENIMAGE_DIT_ADALN_DIM = 6 * QWENIMAGE_DIT_HIDDEN
comptime QWENIMAGE_DIT_FINAL_ADALN_DIM = 2 * QWENIMAGE_DIT_HIDDEN

comptime QWENIMAGE_TEXT_VOCAB = 152064
comptime QWENIMAGE_TEXT_HIDDEN = 3584
comptime QWENIMAGE_TEXT_LAYERS = 28
comptime QWENIMAGE_TEXT_HEADS = 28
comptime QWENIMAGE_TEXT_KV_HEADS = 4
comptime QWENIMAGE_TEXT_HEAD_DIM = 128
comptime QWENIMAGE_TEXT_Q_DIM = QWENIMAGE_TEXT_HEADS * QWENIMAGE_TEXT_HEAD_DIM
comptime QWENIMAGE_TEXT_KV_DIM = QWENIMAGE_TEXT_KV_HEADS * QWENIMAGE_TEXT_HEAD_DIM
comptime QWENIMAGE_TEXT_INTERMEDIATE = 18944
comptime QWENIMAGE_TEXT_ROPE_THETA = 1000000

comptime QWENIMAGE_TRANSFORMER_TENSORS = 1933
comptime QWENIMAGE_TEXT_ENCODER_TENSORS = 729
comptime QWENIMAGE_VAE_TENSORS = 194


def qwenimage_default_cfg_scale() -> Float32:
    return Float32(QWENIMAGE_DEFAULT_CFG_X10) / 10.0


def qwenimage_model_timestep(sigma: Float32) -> Float32:
    return sigma * Float32(QWENIMAGE_SCHEDULER_TRAIN_STEPS)


def qwenimage_latent_spatial_dim(image_dim: Int) raises -> Int:
    if image_dim <= 0:
        raise Error("qwenimage_latent_spatial_dim: image_dim must be > 0")
    if image_dim % QWENIMAGE_LATENT_DOWNSAMPLE != 0:
        raise Error("Qwen-Image image dimension must divide by latent_downsample=8")
    return image_dim // QWENIMAGE_LATENT_DOWNSAMPLE


def qwenimage_patch_grid_dim(latent_dim: Int) raises -> Int:
    if latent_dim <= 0:
        raise Error("qwenimage_patch_grid_dim: latent_dim must be > 0")
    if latent_dim % QWENIMAGE_PATCH_SIZE != 0:
        raise Error("Qwen-Image latent dimension must divide by patch_size=2")
    return latent_dim // QWENIMAGE_PATCH_SIZE


@fieldwise_init
struct QwenImageTokenPlan(Copyable, Movable):
    var width: Int
    var height: Int
    var frames: Int
    var latent_channels: Int
    var latent_h: Int
    var latent_w: Int
    var patch_size: Int
    var patch_grid_h: Int
    var patch_grid_w: Int
    var image_tokens: Int
    var text_tokens: Int
    var total_sequence: Int
    var packed_channels: Int
    var latent_elements: Int

    def validate_1024_contract(self) raises:
        if self.width != QWENIMAGE_DEFAULT_WIDTH or self.height != QWENIMAGE_DEFAULT_HEIGHT:
            raise Error("Qwen-Image contract currently targets 1024x1024")
        if self.frames != QWENIMAGE_DEFAULT_FRAMES:
            raise Error("Qwen-Image contract is image-only")
        if self.latent_channels != QWENIMAGE_LATENT_CHANNELS:
            raise Error("Qwen-Image latent channel mismatch")
        if self.latent_h != QWENIMAGE_LATENT_H or self.latent_w != QWENIMAGE_LATENT_W:
            raise Error("Qwen-Image latent grid must be 128x128")
        if (
            self.patch_grid_h != QWENIMAGE_PATCH_GRID_H
            or self.patch_grid_w != QWENIMAGE_PATCH_GRID_W
        ):
            raise Error("Qwen-Image patch grid must be 64x64")
        if self.image_tokens != QWENIMAGE_IMAGE_TOKENS:
            raise Error("Qwen-Image image token count must be 4096")
        if self.text_tokens != QWENIMAGE_TEXT_MAX_TOKENS:
            raise Error("Qwen-Image max conditioning tokens must be 1024")
        if self.total_sequence != QWENIMAGE_TOTAL_SEQUENCE:
            raise Error("Qwen-Image total sequence must be 5120")
        if self.packed_channels != QWENIMAGE_PACKED_CHANNELS:
            raise Error("Qwen-Image packed token channels must be 64")


def build_qwenimage_token_plan(
    width: Int, height: Int, frames: Int, text_tokens: Int
) raises -> QwenImageTokenPlan:
    if frames != 1:
        raise Error("build_qwenimage_token_plan: Qwen-Image is image-only")
    if text_tokens <= 0:
        raise Error("build_qwenimage_token_plan: text_tokens must be > 0")
    var lh = qwenimage_latent_spatial_dim(height)
    var lw = qwenimage_latent_spatial_dim(width)
    var gh = qwenimage_patch_grid_dim(lh)
    var gw = qwenimage_patch_grid_dim(lw)
    return QwenImageTokenPlan(
        width,
        height,
        frames,
        QWENIMAGE_LATENT_CHANNELS,
        lh,
        lw,
        QWENIMAGE_PATCH_SIZE,
        gh,
        gw,
        gh * gw,
        text_tokens,
        gh * gw + text_tokens,
        QWENIMAGE_PACKED_CHANNELS,
        QWENIMAGE_LATENT_CHANNELS * lh * lw,
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


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
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
            String("Qwen-Image float mismatch: ")
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
            String("Qwen-Image tensor rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=")
            + _shape_string(expected_shape)
        )
    for i in range(len(expected_shape)):
        if info.shape[i] != expected_shape[i]:
            raise Error(
                String("Qwen-Image tensor shape mismatch for ")
                + name
                + String(": actual=")
                + _shape_string(info.shape)
                + String(" expected=")
                + _shape_string(expected_shape)
            )
    var expected_nbytes = _numel(expected_shape) * info.dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("Qwen-Image tensor byte-size mismatch for ")
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
            String("Qwen-Image tensor dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    _check_shape(st, name, expected_shape)


def validate_qwenimage_manifest_contract(manifest: ModelManifest) raises:
    if manifest.model_id != "qwen_image":
        raise Error(String("Qwen-Image contract got manifest: ") + manifest.model_id)
    if manifest.family != ModelFamily.text_to_image():
        raise Error("Qwen-Image manifest family mismatch")
    if manifest.variant != "qwen-image-2512":
        raise Error(String("Qwen-Image manifest variant mismatch: ") + manifest.variant)
    if manifest.profile_name != "qwen_image_1024":
        raise Error(String("Qwen-Image manifest profile mismatch: ") + manifest.profile_name)
    if manifest.default_width != QWENIMAGE_DEFAULT_WIDTH:
        raise Error("Qwen-Image manifest width mismatch")
    if manifest.default_height != QWENIMAGE_DEFAULT_HEIGHT:
        raise Error("Qwen-Image manifest height mismatch")
    if manifest.default_frames != QWENIMAGE_DEFAULT_FRAMES:
        raise Error("Qwen-Image manifest frames mismatch")
    if manifest.latent_channels != QWENIMAGE_LATENT_CHANNELS:
        raise Error("Qwen-Image manifest latent channels mismatch")
    if manifest.latent_downsample_s != QWENIMAGE_LATENT_DOWNSAMPLE:
        raise Error("Qwen-Image manifest spatial downsample mismatch")
    if manifest.image_tokens != QWENIMAGE_IMAGE_TOKENS:
        raise Error("Qwen-Image manifest image tokens mismatch")
    if manifest.text_tokens != QWENIMAGE_TEXT_MAX_TOKENS:
        raise Error("Qwen-Image manifest text tokens mismatch")
    if manifest.total_sequence != QWENIMAGE_TOTAL_SEQUENCE:
        raise Error("Qwen-Image manifest sequence mismatch")
    if manifest.patch_size != QWENIMAGE_PATCH_SIZE:
        raise Error("Qwen-Image manifest patch size mismatch")
    if not manifest.uses_vae():
        raise Error("Qwen-Image manifest must keep the VAE path")
    if manifest.production_entry != "serenitymojo/pipeline/qwenimage_contract_smoke.mojo":
        raise Error("Qwen-Image manifest production entry mismatch")


def validate_qwenimage_local_paths() raises -> Int:
    _require_path(String("Qwen-Image root"), String(QWENIMAGE_ROOT))
    _require_path(String("Qwen-Image model_index"), String(QWENIMAGE_ROOT + "/model_index.json"))
    _require_path(
        String("Qwen-Image transformer config"),
        String(QWENIMAGE_TRANSFORMER_DIR + "/config.json"),
    )
    _require_path(
        String("Qwen-Image transformer index"),
        String(QWENIMAGE_TRANSFORMER_DIR + "/diffusion_pytorch_model.safetensors.index.json"),
    )
    for shard in range(1, 10):
        var suffix = String(shard)
        if shard < 10:
            suffix = String("0") + suffix
        _require_path(
            String("Qwen-Image transformer shard ") + String(shard),
            String(
                QWENIMAGE_TRANSFORMER_DIR
                + "/diffusion_pytorch_model-000"
                + suffix
                + "-of-00009.safetensors"
            ),
        )
    _require_path(
        String("Qwen-Image text encoder config"),
        String(QWENIMAGE_TEXT_ENCODER_DIR + "/config.json"),
    )
    _require_path(
        String("Qwen-Image text encoder index"),
        String(QWENIMAGE_TEXT_ENCODER_DIR + "/model.safetensors.index.json"),
    )
    for shard in range(1, 5):
        var suffix = String(shard)
        if shard < 10:
            suffix = String("0") + suffix
        _require_path(
            String("Qwen-Image text encoder shard ") + String(shard),
            String(
                QWENIMAGE_TEXT_ENCODER_DIR
                + "/model-000"
                + suffix
                + "-of-00004.safetensors"
            ),
        )
    _require_path(String("Qwen-Image tokenizer"), String(QWENIMAGE_TOKENIZER_FILE))
    _require_path(
        String("Qwen-Image tokenizer config"),
        String(QWENIMAGE_ROOT + "/tokenizer/tokenizer_config.json"),
    )
    _require_path(
        String("Qwen-Image chat template"),
        String(QWENIMAGE_ROOT + "/tokenizer/chat_template.jinja"),
    )
    _require_path(
        String("Qwen-Image scheduler"), String(QWENIMAGE_ROOT + "/scheduler/scheduler_config.json")
    )
    _require_path(String("Qwen-Image VAE config"), String(QWENIMAGE_ROOT + "/vae/config.json"))
    _require_path(String("Qwen-Image VAE weights"), String(QWENIMAGE_VAE_FILE))
    return 25


def validate_qwenimage_static_contract() raises -> QwenImageTokenPlan:
    var plan = build_qwenimage_token_plan(
        QWENIMAGE_DEFAULT_WIDTH,
        QWENIMAGE_DEFAULT_HEIGHT,
        QWENIMAGE_DEFAULT_FRAMES,
        QWENIMAGE_TEXT_MAX_TOKENS,
    )
    plan.validate_1024_contract()
    _check_int(
        String("hidden"),
        QWENIMAGE_DIT_HIDDEN,
        QWENIMAGE_DIT_HEADS * QWENIMAGE_DIT_HEAD_DIM,
    )
    _check_int(String("depth"), QWENIMAGE_DIT_LAYERS, 60)
    _check_int(String("packed_channels"), QWENIMAGE_PACKED_CHANNELS, 64)
    _check_int(String("text_layers"), QWENIMAGE_TEXT_LAYERS, 28)
    _check_int(
        String("text_q_dim"),
        QWENIMAGE_TEXT_Q_DIM,
        QWENIMAGE_TEXT_HEADS * QWENIMAGE_TEXT_HEAD_DIM,
    )
    _check_int(
        String("text_kv_dim"),
        QWENIMAGE_TEXT_KV_DIM,
        QWENIMAGE_TEXT_KV_HEADS * QWENIMAGE_TEXT_HEAD_DIM,
    )
    _check_close(String("cfg_scale"), qwenimage_default_cfg_scale(), 4.0, 0.000001)
    _check_close(String("mu_4096"), qwen_mu(Float32(QWENIMAGE_IMAGE_TOKENS)), 0.6935484, 0.00001)
    var sigmas = build_qwen_sigma_schedule(
        QWENIMAGE_DEFAULT_STEPS, Float32(QWENIMAGE_IMAGE_TOKENS)
    )
    _check_int(String("sigma_count"), len(sigmas), QWENIMAGE_DEFAULT_STEPS + 1)
    _check_close(String("sigma[0]"), sigmas[0], 1.0, 0.000001)
    _check_close(
        String("sigma[last_pre_terminal]"),
        sigmas[QWENIMAGE_DEFAULT_STEPS - 1],
        0.02,
        0.000001,
    )
    _check_close(String("sigma[end]"), sigmas[QWENIMAGE_DEFAULT_STEPS], 0.0, 0.000001)
    if sigmas[1] >= sigmas[0]:
        raise Error("Qwen-Image sigma schedule should descend")
    if sigmas[1] - sigmas[0] >= 0.0:
        raise Error("Qwen-Image Euler delta should be sigma_next - sigma")
    _check_close(String("timestep_sigma1"), qwenimage_model_timestep(1.0), 1000.0, 0.000001)
    _check_close(String("timestep_sigma0"), qwenimage_model_timestep(0.0), 0.0, 0.000001)
    return plan^


def validate_qwenimage_transformer_header() raises:
    var st = ShardedSafeTensors.open(String(QWENIMAGE_TRANSFORMER_DIR))
    _check_int(
        String("transformer_tensors"), st.num_tensors(), QWENIMAGE_TRANSFORMER_TENSORS
    )
    _check_tensor(
        st,
        String("img_in.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_DIT_HIDDEN, QWENIMAGE_DIT_IN_CHANNELS),
    )
    _check_tensor(st, String("img_in.bias"), STDtype.BF16, _shape1(QWENIMAGE_DIT_HIDDEN))
    _check_tensor(
        st,
        String("txt_norm.weight"),
        STDtype.BF16,
        _shape1(QWENIMAGE_DIT_JOINT_ATTENTION_DIM),
    )
    _check_tensor(
        st,
        String("txt_in.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_DIT_HIDDEN, QWENIMAGE_DIT_JOINT_ATTENTION_DIM),
    )
    _check_tensor(
        st,
        String("time_text_embed.timestep_embedder.linear_1.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_DIT_HIDDEN, QWENIMAGE_DIT_TIMESTEP_DIM),
    )
    _check_tensor(
        st,
        String("time_text_embed.timestep_embedder.linear_2.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_DIT_HIDDEN, QWENIMAGE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("transformer_blocks.0.attn.to_q.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_DIT_HIDDEN, QWENIMAGE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("transformer_blocks.0.attn.to_q.bias"),
        STDtype.BF16,
        _shape1(QWENIMAGE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("transformer_blocks.0.attn.norm_q.weight"),
        STDtype.BF16,
        _shape1(QWENIMAGE_DIT_HEAD_DIM),
    )
    _check_tensor(
        st,
        String("transformer_blocks.0.attn.add_q_proj.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_DIT_HIDDEN, QWENIMAGE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("transformer_blocks.0.attn.to_add_out.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_DIT_HIDDEN, QWENIMAGE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("transformer_blocks.0.img_mod.1.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_DIT_ADALN_DIM, QWENIMAGE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("transformer_blocks.0.img_mlp.net.0.proj.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_DIT_MLP_HIDDEN, QWENIMAGE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("transformer_blocks.59.attn.to_v.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_DIT_HIDDEN, QWENIMAGE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("norm_out.linear.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_DIT_FINAL_ADALN_DIM, QWENIMAGE_DIT_HIDDEN),
    )
    _check_tensor(
        st,
        String("proj_out.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_PACKED_CHANNELS, QWENIMAGE_DIT_HIDDEN),
    )


def validate_qwenimage_text_encoder_header() raises:
    var st = ShardedSafeTensors.open(String(QWENIMAGE_TEXT_ENCODER_DIR))
    _check_int(
        String("text_encoder_tensors"), st.num_tensors(), QWENIMAGE_TEXT_ENCODER_TENSORS
    )
    _check_tensor(
        st,
        String("model.embed_tokens.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_TEXT_VOCAB, QWENIMAGE_TEXT_HIDDEN),
    )
    _check_tensor(
        st,
        String("model.layers.0.self_attn.q_proj.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_TEXT_Q_DIM, QWENIMAGE_TEXT_HIDDEN),
    )
    _check_tensor(
        st,
        String("model.layers.0.self_attn.q_proj.bias"),
        STDtype.BF16,
        _shape1(QWENIMAGE_TEXT_Q_DIM),
    )
    _check_tensor(
        st,
        String("model.layers.0.self_attn.k_proj.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_TEXT_KV_DIM, QWENIMAGE_TEXT_HIDDEN),
    )
    _check_tensor(
        st,
        String("model.layers.0.self_attn.k_proj.bias"),
        STDtype.BF16,
        _shape1(QWENIMAGE_TEXT_KV_DIM),
    )
    _check_tensor(
        st,
        String("model.layers.0.mlp.gate_proj.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_TEXT_INTERMEDIATE, QWENIMAGE_TEXT_HIDDEN),
    )
    _check_tensor(
        st,
        String("model.layers.27.mlp.down_proj.weight"),
        STDtype.BF16,
        _shape2(QWENIMAGE_TEXT_HIDDEN, QWENIMAGE_TEXT_INTERMEDIATE),
    )
    _check_tensor(
        st,
        String("model.norm.weight"),
        STDtype.BF16,
        _shape1(QWENIMAGE_TEXT_HIDDEN),
    )


def validate_qwenimage_vae_header() raises:
    var st = ShardedSafeTensors.open(String(QWENIMAGE_VAE_FILE))
    _check_int(String("vae_tensors"), st.num_tensors(), QWENIMAGE_VAE_TENSORS)
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
        String("decoder.mid_block.attentions.0.to_qkv.weight"),
        STDtype.BF16,
        _shape4(1152, 384, 1, 1),
    )
    _check_tensor(
        st,
        String("decoder.norm_out.gamma"),
        STDtype.BF16,
        _shape4(96, 1, 1, 1),
    )
    _check_tensor(
        st,
        String("decoder.conv_out.weight"),
        STDtype.BF16,
        _shape5(3, 96, 3, 3, 3),
    )


def validate_qwenimage_metadata_contract(manifest: ModelManifest) raises -> QwenImageTokenPlan:
    validate_qwenimage_manifest_contract(manifest)
    var plan = validate_qwenimage_static_contract()
    validate_qwenimage_transformer_header()
    validate_qwenimage_text_encoder_header()
    validate_qwenimage_vae_header()
    return plan^
