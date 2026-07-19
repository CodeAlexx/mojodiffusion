# AIToolkitIdeogram4Config.mojo - native bridge for ai-toolkit Ideogram4 recipes.
#
# Deliberately no yaml dependency and no Python. This is a small scanner for the
# ai-toolkit config shape used by /home/alex/ai-toolkit/config/*ideogram4*.yaml.

from serenity_trainer.ui.TrainerConfigModel import (
    TrainerUIConfig,
    TrainerUIConcept,
    TrainerUISample,
)
from serenity_trainer.util.config.TrainConfig import (
    TrainConfig,
    TSDIST_UNIFORM,
)


comptime IDEOGRAM4_LOCAL_ROOT = "/home/alex/.serenity/models/ideogram-4-fp8"
comptime IDEOGRAM4_HF_REPO = "ideogram-ai/ideogram-4-fp8"


def _read_text_file(path: String) raises -> String:
    var f = open(path, "r")
    var text = f.read()
    f.close()
    return text^


def _ascii_working_copy(text: String) -> String:
    var out = String("")
    for ch in text.codepoint_slices():
        var s = String(ch)
        if s.byte_length() == 1:
            out += s
        else:
            out += String(" ")
    return out^


def _find_token(text: String, token: String, start: Int = 0) -> Int:
    var text_len = text.byte_length()
    var token_len = token.byte_length()
    if token_len <= 0 or text_len < token_len:
        return -1
    var i = start
    if i < 0:
        i = 0
    var last = text_len - token_len
    while i <= last:
        if String(text[byte=i:i + token_len]) == token:
            return i
        i = i + 1
    return -1


def _line_end(text: String, start: Int) -> Int:
    var i = start
    var n = text.byte_length()
    while i < n:
        if String(text[byte=i]) == "\n":
            return i
        i = i + 1
    return n


def _is_space(ch: String) -> Bool:
    return ch == " " or ch == "\t" or ch == "\n" or ch == "\r"


def _trim(text: String) -> String:
    var start = 0
    var end = text.byte_length()
    while start < end and _is_space(String(text[byte=start])):
        start = start + 1
    while end > start and _is_space(String(text[byte=end - 1])):
        end = end - 1
    return String(text[byte=start:end])


def _strip_comment(line: String) -> String:
    var hash_pos = _find_token(line, String("#"))
    if hash_pos < 0:
        return line.copy()
    return String(line[byte=0:hash_pos])


def _strip_quotes(text: String) -> String:
    var s = _trim(text)
    if s.byte_length() >= 2:
        var first = String(s[byte=0])
        var last = String(s[byte=s.byte_length() - 1])
        if (first == "\"" and last == "\"") or (first == "'" and last == "'"):
            return String(s[byte=1:s.byte_length() - 1])
    return s^


def _starts_with(text: String, prefix: String) -> Bool:
    if text.byte_length() < prefix.byte_length():
        return False
    return String(text[byte=0:prefix.byte_length()]) == prefix


def _is_digit(ch: String) -> Bool:
    return (
        ch == "0"
        or ch == "1"
        or ch == "2"
        or ch == "3"
        or ch == "4"
        or ch == "5"
        or ch == "6"
        or ch == "7"
        or ch == "8"
        or ch == "9"
    )


def _value_after_key_from(text: String, key: String, start: Int = 0) -> String:
    var token = key + String(":")
    var pos = _find_token(text, token, start)
    if pos < 0:
        return String("")
    var value_start = pos + token.byte_length()
    var value_end = _line_end(text, value_start)
    return _strip_quotes(_trim(_strip_comment(String(text[byte=value_start:value_end]))))


def _section_start(text: String, section: String) -> Int:
    return _find_token(text, section + String(":"))


def _parse_bool_or_default(text: String, default_value: Bool) -> Bool:
    var v = _trim(text)
    if v == "true" or v == "True" or v == "yes":
        return True
    if v == "false" or v == "False" or v == "no":
        return False
    return default_value


def _parse_int_or_default(text: String, default_value: Int) raises -> Int:
    var v = _trim(text)
    if v.byte_length() == 0:
        return default_value
    return Int(atol(v))


def _parse_float_or_default(text: String, default_value: Float32) raises -> Float32:
    var v = _trim(text)
    if v.byte_length() == 0:
        return default_value
    return Float32(atof(v))


def _max_int_in_text(text: String, default_value: Int) raises -> Int:
    var best = default_value
    var i = 0
    var n = text.byte_length()
    while i < n:
        var ch = String(text[byte=i])
        if _is_digit(ch):
            var start = i
            while i < n and _is_digit(String(text[byte=i])):
                i = i + 1
            var val = Int(atol(String(text[byte=start:i])))
            if val > best:
                best = val
        else:
            i = i + 1
    return best


def _read_prompt_list(text: String, sample_start: Int) -> List[String]:
    var prompts = List[String]()
    var pos = _find_token(text, String("prompts:"), sample_start)
    if pos < 0:
        return prompts^
    var line = _line_end(text, pos) + 1
    var n = text.byte_length()
    while line < n:
        var end = _line_end(text, line)
        var raw = String(text[byte=line:end])
        var trimmed = _trim(raw)
        if _starts_with(trimmed, String("neg:")) or _starts_with(trimmed, String("seed:")):
            break
        if _starts_with(trimmed, String("- ")):
            var value = _strip_quotes(_trim(String(trimmed[byte=2:])))
            prompts.append(value^)
        line = end + 1
    return prompts^


struct AIToolkitIdeogram4Config(Movable):
    var name: String
    var training_folder: String
    var device: String
    var trigger_word: String
    var lora_rank: Int
    var lora_alpha: Float32
    var save_dtype: String
    var save_every: Int
    var max_step_saves_to_keep: Int
    var dataset_folder_path: String
    var caption_ext: String
    var caption_dropout_rate: Float32
    var shuffle_tokens: Bool
    var cache_latents_to_disk: Bool
    var resolution_max: Int
    var batch_size: Int
    var cache_text_embeddings: Bool
    var steps: Int
    var gradient_accumulation: Int
    var train_unet: Bool
    var train_text_encoder: Bool
    var gradient_checkpointing: Bool
    var noise_scheduler: String
    var optimizer: String
    var lr: Float32
    var train_dtype: String
    var timestep_type: String
    var model_name_or_path: String
    var model_arch: String
    var quantize: Bool
    var quantize_te: Bool
    var qtype_te: String
    var low_vram: Bool
    var sample_sampler: String
    var sample_every: Int
    var sample_steps: Int
    var guidance_scale: Float32
    var width: Int
    var height: Int
    var sample_negative_prompt: String
    var seed: Int
    var walk_seed: Bool
    var sample_prompts: List[String]

    def __init__(out self):
        self.name = String("gigerver3_ideogram4_lora_v1")
        self.training_folder = String("output")
        self.device = String("cuda:0")
        self.trigger_word = String("gigerver3")
        self.lora_rank = 16
        self.lora_alpha = 16.0
        self.save_dtype = String("float16")
        self.save_every = 250
        self.max_step_saves_to_keep = 4
        self.dataset_folder_path = String("/home/alex/1/datasets/gigerver3_json")
        self.caption_ext = String("json")
        self.caption_dropout_rate = 0.05
        self.shuffle_tokens = False
        self.cache_latents_to_disk = True
        self.resolution_max = 1024
        self.batch_size = 1
        self.cache_text_embeddings = True
        self.steps = 2000
        self.gradient_accumulation = 1
        self.train_unet = True
        self.train_text_encoder = False
        self.gradient_checkpointing = True
        self.noise_scheduler = String("flowmatch")
        self.optimizer = String("adamw8bit")
        self.lr = 0.0001
        self.train_dtype = String("bf16")
        self.timestep_type = String("linear")
        self.model_name_or_path = String(IDEOGRAM4_HF_REPO)
        self.model_arch = String("ideogram4")
        self.quantize = True
        self.quantize_te = True
        self.qtype_te = String("qfloat8")
        self.low_vram = True
        self.sample_sampler = String("flowmatch")
        self.sample_every = 500
        self.sample_steps = 20
        self.guidance_scale = 7.0
        self.width = 1024
        self.height = 1024
        self.sample_negative_prompt = String("")
        self.seed = 42
        self.walk_seed = True
        self.sample_prompts = List[String]()


def read_ai_toolkit_ideogram4_config(path: String) raises -> AIToolkitIdeogram4Config:
    var raw_text = _read_text_file(path)
    var text = _ascii_working_copy(raw_text)
    var cfg = AIToolkitIdeogram4Config()

    var config_start = _section_start(text, String("config"))
    var network_start = _section_start(text, String("network"))
    var save_start = _section_start(text, String("save"))
    var dataset_start = _section_start(text, String("datasets"))
    var train_start = _section_start(text, String("train"))
    var model_start = _section_start(text, String("model"))
    var sample_start = _section_start(text, String("sample"))

    var v = _value_after_key_from(text, String("name"), config_start)
    if v.byte_length() > 0:
        cfg.name = v^
    v = _value_after_key_from(text, String("training_folder"), config_start)
    if v.byte_length() > 0:
        cfg.training_folder = v^
    v = _value_after_key_from(text, String("device"), config_start)
    if v.byte_length() > 0:
        cfg.device = v^
    v = _value_after_key_from(text, String("trigger_word"), config_start)
    if v.byte_length() > 0:
        cfg.trigger_word = v^

    cfg.lora_rank = _parse_int_or_default(_value_after_key_from(text, String("linear"), network_start), cfg.lora_rank)
    cfg.lora_alpha = _parse_float_or_default(_value_after_key_from(text, String("linear_alpha"), network_start), cfg.lora_alpha)

    v = _value_after_key_from(text, String("dtype"), save_start)
    if v.byte_length() > 0:
        cfg.save_dtype = v^
    cfg.save_every = _parse_int_or_default(_value_after_key_from(text, String("save_every"), save_start), cfg.save_every)
    cfg.max_step_saves_to_keep = _parse_int_or_default(_value_after_key_from(text, String("max_step_saves_to_keep"), save_start), cfg.max_step_saves_to_keep)

    v = _value_after_key_from(text, String("folder_path"), dataset_start)
    if v.byte_length() > 0:
        cfg.dataset_folder_path = v^
    v = _value_after_key_from(text, String("caption_ext"), dataset_start)
    if v.byte_length() > 0:
        cfg.caption_ext = v^
    cfg.caption_dropout_rate = _parse_float_or_default(_value_after_key_from(text, String("caption_dropout_rate"), dataset_start), cfg.caption_dropout_rate)
    cfg.shuffle_tokens = _parse_bool_or_default(_value_after_key_from(text, String("shuffle_tokens"), dataset_start), cfg.shuffle_tokens)
    cfg.cache_latents_to_disk = _parse_bool_or_default(_value_after_key_from(text, String("cache_latents_to_disk"), dataset_start), cfg.cache_latents_to_disk)
    cfg.resolution_max = _max_int_in_text(_value_after_key_from(text, String("resolution"), dataset_start), cfg.resolution_max)

    cfg.batch_size = _parse_int_or_default(_value_after_key_from(text, String("batch_size"), train_start), cfg.batch_size)
    cfg.cache_text_embeddings = _parse_bool_or_default(_value_after_key_from(text, String("cache_text_embeddings"), train_start), cfg.cache_text_embeddings)
    cfg.steps = _parse_int_or_default(_value_after_key_from(text, String("steps"), train_start), cfg.steps)
    cfg.gradient_accumulation = _parse_int_or_default(_value_after_key_from(text, String("gradient_accumulation"), train_start), cfg.gradient_accumulation)
    cfg.train_unet = _parse_bool_or_default(_value_after_key_from(text, String("train_unet"), train_start), cfg.train_unet)
    cfg.train_text_encoder = _parse_bool_or_default(_value_after_key_from(text, String("train_text_encoder"), train_start), cfg.train_text_encoder)
    cfg.gradient_checkpointing = _parse_bool_or_default(_value_after_key_from(text, String("gradient_checkpointing"), train_start), cfg.gradient_checkpointing)
    v = _value_after_key_from(text, String("noise_scheduler"), train_start)
    if v.byte_length() > 0:
        cfg.noise_scheduler = v^
    v = _value_after_key_from(text, String("optimizer"), train_start)
    if v.byte_length() > 0:
        cfg.optimizer = v^
    cfg.lr = _parse_float_or_default(_value_after_key_from(text, String("lr"), train_start), cfg.lr)
    v = _value_after_key_from(text, String("dtype"), train_start)
    if v.byte_length() > 0:
        cfg.train_dtype = v^
    v = _value_after_key_from(text, String("timestep_type"), train_start)
    if v.byte_length() > 0:
        cfg.timestep_type = v^

    v = _value_after_key_from(text, String("name_or_path"), model_start)
    if v.byte_length() > 0:
        cfg.model_name_or_path = v^
    v = _value_after_key_from(text, String("arch"), model_start)
    if v.byte_length() > 0:
        cfg.model_arch = v^
    cfg.quantize = _parse_bool_or_default(_value_after_key_from(text, String("quantize"), model_start), cfg.quantize)
    cfg.quantize_te = _parse_bool_or_default(_value_after_key_from(text, String("quantize_te"), model_start), cfg.quantize_te)
    v = _value_after_key_from(text, String("qtype_te"), model_start)
    if v.byte_length() > 0:
        cfg.qtype_te = v^
    cfg.low_vram = _parse_bool_or_default(_value_after_key_from(text, String("low_vram"), model_start), cfg.low_vram)

    v = _value_after_key_from(text, String("sampler"), sample_start)
    if v.byte_length() > 0:
        cfg.sample_sampler = v^
    cfg.sample_every = _parse_int_or_default(_value_after_key_from(text, String("sample_every"), sample_start), cfg.sample_every)
    cfg.sample_steps = _parse_int_or_default(_value_after_key_from(text, String("sample_steps"), sample_start), cfg.sample_steps)
    cfg.guidance_scale = _parse_float_or_default(_value_after_key_from(text, String("guidance_scale"), sample_start), cfg.guidance_scale)
    cfg.width = _parse_int_or_default(_value_after_key_from(text, String("width"), sample_start), cfg.width)
    cfg.height = _parse_int_or_default(_value_after_key_from(text, String("height"), sample_start), cfg.height)
    v = _value_after_key_from(text, String("neg"), sample_start)
    if v.byte_length() > 0:
        cfg.sample_negative_prompt = v^
    cfg.seed = _parse_int_or_default(_value_after_key_from(text, String("seed"), sample_start), cfg.seed)
    cfg.walk_seed = _parse_bool_or_default(_value_after_key_from(text, String("walk_seed"), sample_start), cfg.walk_seed)
    cfg.sample_prompts = _read_prompt_list(text, sample_start)

    return cfg^


def ai_toolkit_ideogram4_resolved_model_root(cfg: AIToolkitIdeogram4Config) -> String:
    if cfg.model_name_or_path == String(IDEOGRAM4_HF_REPO):
        return String(IDEOGRAM4_LOCAL_ROOT)
    return cfg.model_name_or_path.copy()


def ai_toolkit_ideogram4_to_train_config(src: AIToolkitIdeogram4Config) -> TrainConfig:
    var cfg = TrainConfig.adamw_lora_defaults()
    cfg.learning_rate = src.lr
    cfg.epochs = 1
    cfg.batch_size = src.batch_size
    cfg.gradient_accumulation_steps = src.gradient_accumulation
    cfg.lora_rank = src.lora_rank
    cfg.lora_alpha = src.lora_alpha
    cfg.seed = UInt32(src.seed)
    cfg.timestep_distribution = TSDIST_UNIFORM
    cfg.guidance_scale = src.guidance_scale
    return cfg^


def ai_toolkit_ideogram4_to_trainer_ui_config(src: AIToolkitIdeogram4Config) -> TrainerUIConfig:
    var cfg = TrainerUIConfig()
    cfg.backend_target = String("ideogram4")
    cfg.run_name = src.name.copy()
    cfg.base_model_name = ai_toolkit_ideogram4_resolved_model_root(src)
    cfg.model_arch = src.model_arch.copy()
    cfg.model_quantize = src.quantize
    cfg.model_quantize_text_encoder = src.quantize_te
    cfg.model_low_vram = src.low_vram
    cfg.model_qtype_text_encoder = src.qtype_te.copy()
    cfg.output_model_destination = String("/home/alex/trainings/") + src.name.copy() + String("/output")
    if src.save_dtype == "float16":
        cfg.output_dtype = String("FLOAT_16")
    elif src.save_dtype == "bf16":
        cfg.output_dtype = String("BFLOAT_16")
    else:
        cfg.output_dtype = src.save_dtype.copy()
    cfg.workspace_dir = String("/home/alex/trainings/") + src.name.copy()
    cfg.cache_dir = cfg.workspace_dir.copy() + String("/cache")
    cfg.dataset_path = src.dataset_folder_path.copy()
    cfg.sample_output_dir = cfg.workspace_dir.copy() + String("/samples")
    cfg.concepts = List[TrainerUIConcept]()
    cfg.concepts.append(TrainerUIConcept(src.trigger_word.copy(), src.dataset_folder_path.copy(), src.trigger_word.copy(), 0, 1, String("STANDARD"), True))
    cfg.latent_caching = src.cache_latents_to_disk
    cfg.cache_text_embeddings = src.cache_text_embeddings
    cfg.caption_extension = src.caption_ext.copy()
    cfg.caption_dropout = src.caption_dropout_rate
    cfg.resolution = String(src.resolution_max)
    cfg.learning_rate = src.lr
    cfg.transformer_learning_rate = src.lr
    cfg.epochs = 1.0
    cfg.max_train_steps = Float32(src.steps)
    cfg.batch_size = Float32(src.batch_size)
    cfg.gradient_accumulation_steps = Float32(src.gradient_accumulation)
    cfg.train_transformer = src.train_unet
    cfg.train_text_encoder = src.train_text_encoder
    cfg.gradient_checkpointing = src.gradient_checkpointing
    cfg.train_dtype = String("BFLOAT_16")
    if src.train_dtype == "float16":
        cfg.train_dtype = String("FLOAT_16")
    cfg.fallback_train_dtype = cfg.train_dtype.copy()
    cfg.optimizer_index = 0
    cfg.noise_scheduler = src.noise_scheduler.copy()
    cfg.timestep_type = src.timestep_type.copy()
    cfg.timestep_distribution = String("UNIFORM")
    cfg.lora_model_name = src.name.copy()
    cfg.lora_rank = Float32(src.lora_rank)
    cfg.lora_alpha = src.lora_alpha
    cfg.samples = List[TrainerUISample]()
    if len(src.sample_prompts) > 0:
        for i in range(len(src.sample_prompts)):
            cfg.samples.append(TrainerUISample(src.sample_prompts[i].copy(), src.sample_negative_prompt.copy(), Int32(src.seed + i)))
    else:
        cfg.samples.append(TrainerUISample(src.trigger_word.copy(), src.sample_negative_prompt.copy(), Int32(src.seed)))
    cfg.sample_after = Float32(src.sample_every)
    cfg.sample_steps = Float32(src.sample_steps)
    cfg.sample_cfg = src.guidance_scale
    cfg.sample_sampler = String("Ideogram4 FlowMatch")
    cfg.sampler_preset = String("V4_DEFAULT_20")
    cfg.save_every = Float32(src.save_every)
    cfg.save_max_keep = Float32(src.max_step_saves_to_keep)
    cfg.save_filename_prefix = src.name.copy()
    return cfg^
