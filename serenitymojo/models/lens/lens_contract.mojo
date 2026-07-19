# models/lens/lens_contract.mojo - Microsoft Lens metadata/header contract.
#
# Metadata-only gate for the local microsoft/Lens sidecar. This intentionally
# has no DeviceContext, no MXFP4 dequant, and no Lens DiT/GPT-OSS/VAE math.

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.registry.checkpoints import path_exists


comptime LENS_ROOT = "/home/alex/.serenity/models/microsoft_lens"
comptime LENS_TRANSFORMER_DIR = LENS_ROOT + "/transformer"
comptime LENS_TEXT_ENCODER_DIR = LENS_ROOT + "/text_encoder"
comptime LENS_TOKENIZER_DIR = LENS_ROOT + "/tokenizer"
comptime LENS_VAE_FILE = LENS_ROOT + "/vae/diffusion_pytorch_model.safetensors"
comptime LENS_CAPTURE_1024_METADATA = (
    "/home/alex/EriDiffusion/inference-flame/lens/parity/captures/"
    "capture_metadata.json"
)
comptime LENS_CAPTURE_512_METADATA = (
    "/home/alex/EriDiffusion/inference-flame/lens/parity/captures_512/"
    "capture_metadata.json"
)
comptime LENS_TEXT_SMOKE_METADATA = (
    "/home/alex/EriDiffusion/inference-flame/lens/parity/captures_text_smoke/"
    "metadata.json"
)
comptime LENS_TEXT_SMOKE_DIR = (
    "/home/alex/EriDiffusion/inference-flame/lens/parity/captures_text_smoke"
)
comptime LENS_TEXT_SMOKE_INPUT_IDS = LENS_TEXT_SMOKE_DIR + "/input_ids.safetensors"
comptime LENS_TEXT_SMOKE_ATTENTION_MASK = (
    LENS_TEXT_SMOKE_DIR + "/attention_mask.safetensors"
)
comptime LENS_TEXT_SMOKE_HIDDEN_05 = LENS_TEXT_SMOKE_DIR + "/hidden_layer_05.safetensors"
comptime LENS_TEXT_SMOKE_HIDDEN_11 = LENS_TEXT_SMOKE_DIR + "/hidden_layer_11.safetensors"
comptime LENS_TEXT_SMOKE_HIDDEN_17 = LENS_TEXT_SMOKE_DIR + "/hidden_layer_17.safetensors"
comptime LENS_TEXT_SMOKE_HIDDEN_23 = LENS_TEXT_SMOKE_DIR + "/hidden_layer_23.safetensors"

comptime LENS_WIDTH = 1024
comptime LENS_HEIGHT = 1024
comptime LENS_LATENT_DOWNSAMPLE = 16
comptime LENS_LATENT_H = LENS_HEIGHT // LENS_LATENT_DOWNSAMPLE
comptime LENS_LATENT_W = LENS_WIDTH // LENS_LATENT_DOWNSAMPLE
comptime LENS_IMAGE_TOKENS = LENS_LATENT_H * LENS_LATENT_W
comptime LENS_PATCH_SIZE = 2
comptime LENS_OUT_CHANNELS = 32
comptime LENS_PATCH_VECTOR_DIM = (
    LENS_OUT_CHANNELS * LENS_PATCH_SIZE * LENS_PATCH_SIZE
)
comptime LENS_TRANSFORMER_TENSORS = 1264
comptime LENS_TEXT_ENCODER_TENSORS = 459
comptime LENS_VAE_TENSORS = 251
comptime LENS_DIT_LAYERS = 48
comptime LENS_DIT_HEADS = 24
comptime LENS_DIT_HEAD_DIM = 64
comptime LENS_DIT_INNER_DIM = 1536
comptime LENS_DIT_MLP_HIDDEN = 4096
comptime LENS_GPT_OSS_HIDDEN = 2880
comptime LENS_GPT_OSS_LAYERS = 24
comptime LENS_GPT_OSS_Q_HEADS = 64
comptime LENS_GPT_OSS_KV_HEADS = 8
comptime LENS_GPT_OSS_HEAD_DIM = 64
comptime LENS_GPT_OSS_EXPERTS = 32
comptime LENS_GPT_OSS_EXPERTS_PER_TOKEN = 4
comptime LENS_GPT_OSS_INTERMEDIATE = 2880
comptime LENS_GPT_OSS_VOCAB = 201088
comptime LENS_GPT_OSS_SLIDING_WINDOW = 128
comptime LENS_GPT_OSS_ROPE_THETA = 150000
comptime LENS_GPT_OSS_YARN_FACTOR = 32
comptime LENS_TEXT_SMOKE_SEQ_LEN = 64
comptime LENS_TEXT_SMOKE_LAYER_05 = 5
comptime LENS_TEXT_SMOKE_LAYER_11 = 11
comptime LENS_TEXT_SMOKE_LAYER_17 = 17
comptime LENS_TEXT_SMOKE_LAYER_23 = 23
comptime LENS_MAX_TEXT_LEN = 512
comptime LENS_TXT_OFFSET = 97
comptime LENS_POST_OFFSET_TEXT_TOKENS = LENS_MAX_TEXT_LEN - LENS_TXT_OFFSET
comptime LENS_ZERO_FEATURE_TEXT_TOKENS = 256
comptime LENS_NUM_STEPS = 20
comptime LENS_CFG_SCALE = 5.0
comptime LENS_SCHEDULER_TRAIN_STEPS = 1000
comptime LENS_SCHEDULER_SHIFT_X10 = 30
comptime LENS_SCHEDULER_BASE_IMAGE_SEQ_LEN = 256
comptime LENS_SCHEDULER_MAX_IMAGE_SEQ_LEN = 4096
comptime LENS_SCHEDULER_BASE_SHIFT_X10 = 5
comptime LENS_SCHEDULER_MAX_SHIFT_X100 = 115
comptime LENS_SCHEDULER_DYNAMIC_SHIFTING = 1
comptime LENS_SCHEDULER_EXPONENTIAL_SHIFT = 1
comptime LENS_SCHEDULER_KARRAS_SIGMAS = 0
comptime LENS_SCHEDULER_EXPONENTIAL_SIGMAS = 0
comptime LENS_SCHEDULER_BETA_SIGMAS = 0


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


def _check_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            String("Lens contract int mismatch: ")
            + name
            + String(" got=")
            + String(got)
            + String(" expected=")
            + String(expected)
        )


def _require_path(label: String, path: String) raises:
    if not path_exists(path):
        raise Error(label + String(" missing: ") + path)


def _check_tensor_shape(
    ref st: ShardedSafeTensors, name: String, expected_shape: List[Int]
) raises:
    var info = st.tensor_info(name)
    if len(info.shape) != len(expected_shape):
        raise Error(
            String("Lens tensor rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=")
            + _shape_string(expected_shape)
        )
    for i in range(len(expected_shape)):
        if info.shape[i] != expected_shape[i]:
            raise Error(
                String("Lens tensor shape mismatch for ")
                + name
                + String(": actual=")
                + _shape_string(info.shape)
                + String(" expected=")
                + _shape_string(expected_shape)
            )
    var expected_nbytes = _numel(expected_shape) * info.dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("Lens tensor byte-size mismatch for ")
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
            String("Lens tensor dtype mismatch for ")
            + name
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    _check_tensor_shape(st, name, expected_shape)


def _check_safe_tensor_shape(
    ref st: SafeTensors, name: String, expected_shape: List[Int]
) raises:
    var info = st.tensor_info(name)
    if len(info.shape) != len(expected_shape):
        raise Error(
            String("Lens smoke tensor rank mismatch for ")
            + name
            + String(": actual=")
            + _shape_string(info.shape)
            + String(" expected=")
            + _shape_string(expected_shape)
        )
    for i in range(len(expected_shape)):
        if info.shape[i] != expected_shape[i]:
            raise Error(
                String("Lens smoke tensor shape mismatch for ")
                + name
                + String(": actual=")
                + _shape_string(info.shape)
                + String(" expected=")
                + _shape_string(expected_shape)
            )
    var expected_nbytes = _numel(expected_shape) * info.dtype.byte_size()
    if info.size != expected_nbytes:
        raise Error(
            String("Lens smoke tensor byte-size mismatch for ")
            + name
            + String(": actual=")
            + String(info.size)
            + String(" expected=")
            + String(expected_nbytes)
        )


def _check_single_safe_tensor(path: String, dtype: STDtype, expected_shape: List[Int]) raises:
    var st = SafeTensors.open(path)
    if st.count() != 1:
        raise Error(
            String("Lens smoke tensor count mismatch for ")
            + path
            + String(": actual=")
            + String(st.count())
            + String(" expected=1")
        )
    var info = st.tensor_info(String("tensor"))
    if info.dtype != dtype:
        raise Error(
            String("Lens smoke tensor dtype mismatch for ")
            + path
            + String(": actual=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    _check_safe_tensor_shape(st, String("tensor"), expected_shape)


struct LensTokenPlan(Copyable, Movable):
    var width: Int
    var height: Int
    var text_tokens: Int
    var latent_h: Int
    var latent_w: Int
    var image_tokens: Int
    var patch_vector_dim: Int
    var total_sequence: Int

    def __init__(out self, width: Int, height: Int, text_tokens: Int) raises:
        if text_tokens < 0:
            raise Error("LensTokenPlan: text_tokens must be >= 0")
        if width <= 0 or height <= 0:
            raise Error("LensTokenPlan: image dimensions must be > 0")
        if width % LENS_LATENT_DOWNSAMPLE != 0:
            raise Error("LensTokenPlan: width must divide by 16")
        if height % LENS_LATENT_DOWNSAMPLE != 0:
            raise Error("LensTokenPlan: height must divide by 16")
        self.width = width
        self.height = height
        self.text_tokens = text_tokens
        self.latent_h = height // LENS_LATENT_DOWNSAMPLE
        self.latent_w = width // LENS_LATENT_DOWNSAMPLE
        self.image_tokens = self.latent_h * self.latent_w
        self.patch_vector_dim = LENS_PATCH_VECTOR_DIM
        self.total_sequence = self.image_tokens + self.text_tokens

    def validate_1024_contract(self) raises:
        _check_int(String("width"), self.width, LENS_WIDTH)
        _check_int(String("height"), self.height, LENS_HEIGHT)
        _check_int(String("latent_h"), self.latent_h, LENS_LATENT_H)
        _check_int(String("latent_w"), self.latent_w, LENS_LATENT_W)
        _check_int(String("image_tokens"), self.image_tokens, LENS_IMAGE_TOKENS)
        _check_int(
            String("patch_vector_dim"),
            self.patch_vector_dim,
            LENS_PATCH_VECTOR_DIM,
        )


def build_lens_token_plan(width: Int, height: Int, text_tokens: Int) raises -> LensTokenPlan:
    return LensTokenPlan(width, height, text_tokens)


def validate_lens_static_contract() raises:
    _check_int(
        String("dit_inner"),
        LENS_DIT_HEADS * LENS_DIT_HEAD_DIM,
        LENS_DIT_INNER_DIM,
    )
    _check_int(String("patch_vector_dim"), LENS_PATCH_VECTOR_DIM, 128)
    _check_int(
        String("gpt_oss_attention_width"),
        LENS_GPT_OSS_Q_HEADS * LENS_GPT_OSS_HEAD_DIM,
        4096,
    )
    _check_int(
        String("gpt_oss_gqa_ratio"),
        LENS_GPT_OSS_Q_HEADS // LENS_GPT_OSS_KV_HEADS,
        8,
    )
    var real_plan = build_lens_token_plan(
        LENS_WIDTH, LENS_HEIGHT, LENS_POST_OFFSET_TEXT_TOKENS
    )
    real_plan.validate_1024_contract()
    _check_int(String("post_offset_text_tokens"), real_plan.text_tokens, 415)
    _check_int(String("real_total_sequence"), real_plan.total_sequence, 4511)

    var zero_feature_plan = build_lens_token_plan(
        LENS_WIDTH, LENS_HEIGHT, LENS_ZERO_FEATURE_TEXT_TOKENS
    )
    zero_feature_plan.validate_1024_contract()
    _check_int(String("zero_feature_text_tokens"), zero_feature_plan.text_tokens, 256)
    _check_int(String("zero_feature_total_sequence"), zero_feature_plan.total_sequence, 4352)
    _check_int(String("scheduler_train_steps"), LENS_SCHEDULER_TRAIN_STEPS, 1000)
    _check_int(String("scheduler_shift_x10"), LENS_SCHEDULER_SHIFT_X10, 30)
    _check_int(
        String("scheduler_base_image_seq_len"), LENS_SCHEDULER_BASE_IMAGE_SEQ_LEN, 256
    )
    _check_int(
        String("scheduler_max_image_seq_len"), LENS_SCHEDULER_MAX_IMAGE_SEQ_LEN, 4096
    )
    _check_int(String("scheduler_base_shift_x10"), LENS_SCHEDULER_BASE_SHIFT_X10, 5)
    _check_int(String("scheduler_max_shift_x100"), LENS_SCHEDULER_MAX_SHIFT_X100, 115)
    _check_int(String("scheduler_dynamic_shifting"), LENS_SCHEDULER_DYNAMIC_SHIFTING, 1)
    _check_int(String("scheduler_exponential_shift"), LENS_SCHEDULER_EXPONENTIAL_SHIFT, 1)
    _check_int(String("scheduler_karras_sigmas"), LENS_SCHEDULER_KARRAS_SIGMAS, 0)
    _check_int(
        String("scheduler_exponential_sigmas"), LENS_SCHEDULER_EXPONENTIAL_SIGMAS, 0
    )
    _check_int(String("scheduler_beta_sigmas"), LENS_SCHEDULER_BETA_SIGMAS, 0)


def validate_lens_local_paths() raises -> Int:
    var checked = 0
    _require_path(String("Lens root"), String(LENS_ROOT))
    checked += 1
    _require_path(String("Lens model_index"), String(LENS_ROOT + "/model_index.json"))
    checked += 1
    _require_path(String("Lens transformer config"), String(LENS_TRANSFORMER_DIR + "/config.json"))
    checked += 1
    _require_path(
        String("Lens transformer index"),
        String(LENS_TRANSFORMER_DIR + "/diffusion_pytorch_model.safetensors.index.json"),
    )
    checked += 1
    _require_path(String("Lens text encoder config"), String(LENS_TEXT_ENCODER_DIR + "/config.json"))
    checked += 1
    _require_path(
        String("Lens text encoder index"),
        String(LENS_TEXT_ENCODER_DIR + "/model.safetensors.index.json"),
    )
    checked += 1
    _require_path(String("Lens tokenizer"), String(LENS_TOKENIZER_DIR + "/tokenizer.json"))
    checked += 1
    _require_path(String("Lens chat template"), String(LENS_TOKENIZER_DIR + "/chat_template.jinja"))
    checked += 1
    _require_path(String("Lens scheduler"), String(LENS_ROOT + "/scheduler/scheduler_config.json"))
    checked += 1
    _require_path(String("Lens VAE config"), String(LENS_ROOT + "/vae/config.json"))
    checked += 1
    _require_path(String("Lens VAE weights"), String(LENS_VAE_FILE))
    checked += 1
    _require_path(String("Lens 1024 capture metadata"), String(LENS_CAPTURE_1024_METADATA))
    checked += 1
    _require_path(String("Lens 512 capture metadata"), String(LENS_CAPTURE_512_METADATA))
    checked += 1
    _require_path(String("Lens text smoke metadata"), String(LENS_TEXT_SMOKE_METADATA))
    checked += 1
    _require_path(String("Lens text smoke input_ids"), String(LENS_TEXT_SMOKE_INPUT_IDS))
    checked += 1
    _require_path(
        String("Lens text smoke attention_mask"), String(LENS_TEXT_SMOKE_ATTENTION_MASK)
    )
    checked += 1
    _require_path(String("Lens text smoke hidden layer 05"), String(LENS_TEXT_SMOKE_HIDDEN_05))
    checked += 1
    _require_path(String("Lens text smoke hidden layer 11"), String(LENS_TEXT_SMOKE_HIDDEN_11))
    checked += 1
    _require_path(String("Lens text smoke hidden layer 17"), String(LENS_TEXT_SMOKE_HIDDEN_17))
    checked += 1
    _require_path(String("Lens text smoke hidden layer 23"), String(LENS_TEXT_SMOKE_HIDDEN_23))
    checked += 1
    return checked


def validate_lens_transformer_header() raises:
    var st = ShardedSafeTensors.open(String(LENS_TRANSFORMER_DIR))
    _check_int(String("transformer_shards"), st.num_shards(), 2)
    _check_int(String("transformer_tensors"), st.num_tensors(), LENS_TRANSFORMER_TENSORS)
    _check_tensor(st, String("img_in.weight"), STDtype.F32, _shape2(1536, 128))
    _check_tensor(st, String("txt_in.weight"), STDtype.F32, _shape2(1536, 11520))
    _check_tensor(
        st,
        String("time_text_embed.timestep_embedder.linear_1.weight"),
        STDtype.F32,
        _shape2(1536, 256),
    )
    _check_tensor(
        st,
        String("transformer_blocks.0.attn.img_qkv.weight"),
        STDtype.F32,
        _shape2(4608, 1536),
    )
    _check_tensor(
        st,
        String("transformer_blocks.47.img_mlp.w2.weight"),
        STDtype.F32,
        _shape2(1536, LENS_DIT_MLP_HIDDEN),
    )
    _check_tensor(
        st,
        String("norm_out.linear.weight"),
        STDtype.F32,
        _shape2(2 * LENS_DIT_INNER_DIM, LENS_DIT_INNER_DIM),
    )
    _check_tensor(
        st,
        String("proj_out.weight"),
        STDtype.F32,
        _shape2(LENS_PATCH_VECTOR_DIM, LENS_DIT_INNER_DIM),
    )


def validate_lens_text_encoder_header() raises:
    var st = ShardedSafeTensors.open(String(LENS_TEXT_ENCODER_DIR))
    _check_int(String("text_encoder_shards"), st.num_shards(), 3)
    _check_int(String("text_encoder_tensors"), st.num_tensors(), LENS_TEXT_ENCODER_TENSORS)
    _check_tensor(
        st,
        String("model.embed_tokens.weight"),
        STDtype.BF16,
        _shape2(LENS_GPT_OSS_VOCAB, LENS_GPT_OSS_HIDDEN),
    )
    _check_tensor(
        st,
        String("model.layers.0.self_attn.q_proj.weight"),
        STDtype.BF16,
        _shape2(LENS_GPT_OSS_Q_HEADS * LENS_GPT_OSS_HEAD_DIM, LENS_GPT_OSS_HIDDEN),
    )
    _check_tensor(
        st,
        String("model.layers.0.self_attn.k_proj.weight"),
        STDtype.BF16,
        _shape2(LENS_GPT_OSS_KV_HEADS * LENS_GPT_OSS_HEAD_DIM, LENS_GPT_OSS_HIDDEN),
    )
    _check_tensor(
        st,
        String("model.layers.0.self_attn.sinks"),
        STDtype.BF16,
        _shape1(LENS_GPT_OSS_Q_HEADS),
    )
    _check_tensor(
        st,
        String("model.layers.0.mlp.router.weight"),
        STDtype.BF16,
        _shape2(LENS_GPT_OSS_EXPERTS, LENS_GPT_OSS_HIDDEN),
    )
    _check_tensor(
        st,
        String("model.layers.0.mlp.experts.gate_up_proj_bias"),
        STDtype.F32,
        _shape2(LENS_GPT_OSS_EXPERTS, 2 * LENS_GPT_OSS_INTERMEDIATE),
    )
    _check_tensor(
        st,
        String("model.layers.0.mlp.experts.gate_up_proj_blocks"),
        STDtype.U8,
        _shape4(LENS_GPT_OSS_EXPERTS, 2 * LENS_GPT_OSS_INTERMEDIATE, 90, 16),
    )
    _check_tensor(
        st,
        String("model.layers.0.mlp.experts.gate_up_proj_scales"),
        STDtype.U8,
        _shape3(LENS_GPT_OSS_EXPERTS, 2 * LENS_GPT_OSS_INTERMEDIATE, 90),
    )
    _check_tensor(
        st,
        String("model.layers.0.mlp.experts.down_proj_blocks"),
        STDtype.U8,
        _shape4(LENS_GPT_OSS_EXPERTS, LENS_GPT_OSS_HIDDEN, 90, 16),
    )
    _check_tensor(
        st,
        String("model.layers.23.mlp.experts.down_proj_bias"),
        STDtype.F32,
        _shape2(LENS_GPT_OSS_EXPERTS, LENS_GPT_OSS_HIDDEN),
    )


def validate_lens_vae_header() raises:
    var st = ShardedSafeTensors.open(String(LENS_VAE_FILE))
    _check_int(String("vae_shards"), st.num_shards(), 1)
    _check_int(String("vae_tensors"), st.num_tensors(), LENS_VAE_TENSORS)
    _check_tensor(
        st,
        String("post_quant_conv.weight"),
        STDtype.F32,
        _shape4(32, 32, 1, 1),
    )
    _check_tensor(
        st,
        String("decoder.conv_in.weight"),
        STDtype.F32,
        _shape4(512, 32, 3, 3),
    )
    _check_tensor(
        st,
        String("decoder.up_blocks.0.resnets.0.conv1.weight"),
        STDtype.F32,
        _shape4(512, 512, 3, 3),
    )
    _check_tensor(st, String("bn.running_mean"), STDtype.F32, _shape1(128))
    _check_tensor(st, String("bn.running_var"), STDtype.F32, _shape1(128))


def validate_lens_text_smoke_header() raises:
    _check_int(String("text_smoke_seq_len"), LENS_TEXT_SMOKE_SEQ_LEN, 64)
    _check_int(String("text_smoke_layer_05"), LENS_TEXT_SMOKE_LAYER_05, 5)
    _check_int(String("text_smoke_layer_11"), LENS_TEXT_SMOKE_LAYER_11, 11)
    _check_int(String("text_smoke_layer_17"), LENS_TEXT_SMOKE_LAYER_17, 17)
    _check_int(String("text_smoke_layer_23"), LENS_TEXT_SMOKE_LAYER_23, 23)
    _check_single_safe_tensor(
        String(LENS_TEXT_SMOKE_INPUT_IDS),
        STDtype.F32,
        _shape2(1, LENS_TEXT_SMOKE_SEQ_LEN),
    )
    _check_single_safe_tensor(
        String(LENS_TEXT_SMOKE_ATTENTION_MASK),
        STDtype.F32,
        _shape2(1, LENS_TEXT_SMOKE_SEQ_LEN),
    )
    _check_single_safe_tensor(
        String(LENS_TEXT_SMOKE_HIDDEN_05),
        STDtype.BF16,
        _shape3(1, LENS_TEXT_SMOKE_SEQ_LEN, LENS_GPT_OSS_HIDDEN),
    )
    _check_single_safe_tensor(
        String(LENS_TEXT_SMOKE_HIDDEN_11),
        STDtype.BF16,
        _shape3(1, LENS_TEXT_SMOKE_SEQ_LEN, LENS_GPT_OSS_HIDDEN),
    )
    _check_single_safe_tensor(
        String(LENS_TEXT_SMOKE_HIDDEN_17),
        STDtype.BF16,
        _shape3(1, LENS_TEXT_SMOKE_SEQ_LEN, LENS_GPT_OSS_HIDDEN),
    )
    _check_single_safe_tensor(
        String(LENS_TEXT_SMOKE_HIDDEN_23),
        STDtype.BF16,
        _shape3(1, LENS_TEXT_SMOKE_SEQ_LEN, LENS_GPT_OSS_HIDDEN),
    )


def validate_lens_sidecar_contract() raises -> Int:
    validate_lens_static_contract()
    var checked = validate_lens_local_paths()
    validate_lens_transformer_header()
    validate_lens_text_encoder_header()
    validate_lens_vae_header()
    validate_lens_text_smoke_header()
    return checked
