# reader_levers_reachability_smoke.mojo — prove EVERY Wave 2 (2A+2B) config key
# is REACHABLE through train_config_reader and maps to the correct TrainConfig
# field (incl. enum string -> comptime-Int).
#
# Run (after the compile lock frees):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo run -I . serenitymojo/io/reader_levers_reachability_smoke.mojo
#
# Three sub-gates (TENET 4 + parity-bitrot guard):
#   (a) ALL-SET   — a JSON with every new key at a NON-default value; assert each
#                   TrainConfig field reflects it (esp. "cosine"->2, "later"->1,
#                   "uniform"->0, "full"->1).
#   (b) NONE-SET  — a JSON setting NONE of the new keys; assert every new field ==
#                   TrainConfig.default() (absent-key default preservation /
#                   baseline byte-invariance / DEFAULT-OFF rule).
#   (c) BOGUS     — a JSON with lr_scheduler="bogus"; assert the reader RAISES
#                   (fail-loud, not silent-default).
# Plus a deliberate-wrong assertion demo proving the gate EXITS NONZERO on bad
# values (parity-bitrot guard).
#
# Mojo 1.0.0b1; pure-Mojo file I/O (no Python).

from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.io.ffi import (
    sys_open, sys_close, sys_pwrite, BytePtr,
    O_WRONLY, O_CREAT, O_TRUNC,
)
from serenitymojo.training.train_config import (
    TrainConfig,
    TRAIN_MODALITY_VIDEO, TRAIN_MODALITY_AV,
    LORA_TARGET_LEGACY_VIDEO_ATTN1, LORA_TARGET_LTX2_V2V,
    TRAIN_ADAPTER_ALGO_FULL, TRAIN_ADAPTER_ALGO_LOCON,
    TRAIN_ADAPTER_ALGO_LOHA, TRAIN_ADAPTER_ALGO_LOKR,
)
from std.memory import alloc


# ── write a String to `path` via raw syscalls (O_CREAT|O_WRONLY|O_TRUNC) ─────
def _write_file(path: String, content: String) raises:
    var fd = sys_open(path, O_CREAT | O_WRONLY | O_TRUNC, Int32(0o644))
    if fd < 0:
        raise Error(String("reader smoke: cannot create ") + path)
    var n = content.byte_length()
    var buf = alloc[UInt8](n if n > 0 else 1)
    var src = content.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    var wrote = sys_pwrite(fd, BytePtr(unsafe_from_address=Int(buf)), n, 0)
    buf.free()
    _ = sys_close(fd)
    if wrote != n:
        raise Error(String("reader smoke: short write to ") + path)


def _close(name: String, a: Float32, b: Float32) raises:
    var d = a - b
    var ad = d if d >= 0.0 else -d
    if ad > Float32(1e-6):
        raise Error(name + " mismatch: got " + String(a) + " expected " + String(b))


def _eq(name: String, a: Int, b: Int) raises:
    if a != b:
        raise Error(name + " expected " + String(b) + ", got " + String(a))


def _eqb(name: String, a: Bool, b: Bool) raises:
    if a != b:
        raise Error(name + " expected " + String(b) + ", got " + String(a))


def _eqs(name: String, a: String, b: String) raises:
    if a != b:
        raise Error(name + " expected " + b + ", got " + a)


# A minimal-but-valid arch stub shared by both JSONs so the parser walks a
# realistic top-level object (arch keys exercise the pre-existing branches).
comptime _ARCH = String(
    '"model_type":"klein","inner_dim":64,"in_channels":4,'
    + '"joint_attention_dim":64,"out_channels":4,"num_double":1,'
    + '"num_single":1,"num_heads":2,"head_dim":32,"mlp_hidden":128,'
    + '"timestep_dim":256,"learning_rate":1e-4,"lora_rank":16,"lora_alpha":16.0'
)


# ─────────────────────────────────────────────────────────────────────────────
def _gate_all_set() raises:
    print("--- gate (a): ALL Wave 2 keys set to non-default ---")
    var path = String("/tmp/reader_levers_all.json")
    var js = String("{") + _ARCH + ","
    # Wave 2A
    js += '"lr_scheduler":"cosine",'
    js += '"lr_warmup_steps":100,'
    js += '"lr_min_factor":0.1,'
    js += '"lr_cycles":3.0,'
    js += '"min_snr_gamma":5.0,'
    js += '"debiased":true,'
    js += '"loss_mse_strength":0.5,'
    js += '"loss_mae_strength":0.25,'
    js += '"loss_huber_strength":0.75,'
    js += '"timestep_bias_strategy":"later",'
    js += '"timestep_bias_multiplier":0.6,'
    js += '"timestep_bias_range_min":0.2,'
    js += '"timestep_bias_range_max":0.8,'
    js += '"timestep_distribution":"uniform",'
    js += '"timestep_noising_weight":1.8,'
    js += '"timestep_noising_bias":0.3,'
    # Wave 2B
    js += '"caption_dropout_prob":0.1,'
    js += '"offset_noise_weight":0.05,'
    js += '"offset_noise_prob":0.5,'
    js += '"input_perturbation":0.1,'
    js += '"multires_iterations":4,'
    js += '"multires_discount":0.6,'
    js += '"grad_accum_steps":2,'
    js += '"ema_enabled":true,'
    js += '"ema_inv_gamma":2.0,'
    js += '"ema_power":0.75,'
    js += '"ema_update_after_step":50,'
    js += '"ema_min_decay":0.1,'
    js += '"ema_max_decay":0.999,'
    # Cached-input / AV trainer contract.
    js += '"train_modality":"av",'
    js += '"lora_target_preset":"v2v",'
    js += '"dataset_cache_dir":"/tmp/ltx2_av_cache",'
    js += '"require_cached_video_latents":true,'
    js += '"require_cached_text_embeddings":true,'
    js += '"require_cached_audio_latents":true,'
    js += '"hot_loop_device_only":true,'
    js += '"video_loss_weight":1.0,'
    js += '"audio_loss_weight":0.25,'
    js += '"algo":"full"'
    js += "}"
    _write_file(path, js)

    var c = read_model_config(path)

    # Wave 2A — lr scheduler (string -> int)
    _eq("lr_scheduler(cosine->LR_COSINE)", c.lr_scheduler, 2)
    _eq("lr_warmup_steps", c.lr_warmup_steps, 100)
    _close("lr_min_factor", c.lr_min_factor, Float32(0.1))
    _close("lr_cycles", c.lr_cycles, Float32(3.0))
    # Wave 2A — loss weighting
    _close("min_snr_gamma", c.min_snr_gamma, Float32(5.0))
    _eqb("debiased", c.debiased, True)
    _close("loss_mse_strength", c.loss_mse_strength, Float32(0.5))
    _close("loss_mae_strength", c.loss_mae_strength, Float32(0.25))
    _close("loss_huber_strength", c.loss_huber_strength, Float32(0.75))
    # Wave 2A — timestep bias (string -> int)
    _eq("timestep_bias_strategy(later->TSB_LATER)", c.timestep_bias_strategy, 1)
    _close("timestep_bias_multiplier", c.timestep_bias_multiplier, Float32(0.6))
    _close("timestep_bias_range_min", c.timestep_bias_range_min, Float32(0.2))
    _close("timestep_bias_range_max", c.timestep_bias_range_max, Float32(0.8))
    # Wave 2A — timestep distribution (string -> int)
    _eq("timestep_distribution(uniform->TSD_UNIFORM)", c.timestep_distribution, 0)
    _close("timestep_noising_weight", c.timestep_noising_weight, Float32(1.8))
    _close("timestep_noising_bias", c.timestep_noising_bias, Float32(0.3))
    # Wave 2B — caption dropout
    _close("caption_dropout_prob", c.caption_dropout_prob, Float32(0.1))
    # Wave 2B — noise modifiers
    _close("offset_noise_weight", c.offset_noise_weight, Float32(0.05))
    _close("offset_noise_prob", c.offset_noise_prob, Float32(0.5))
    _close("input_perturbation", c.input_perturbation, Float32(0.1))
    _eq("multires_iterations", c.multires_iterations, 4)
    _close("multires_discount", c.multires_discount, Float32(0.6))
    # Wave 2B — grad accumulation
    _eq("grad_accum_steps", c.grad_accum_steps, 2)
    # Wave 2B — EMA
    _eqb("ema_enabled", c.ema_enabled, True)
    _close("ema_inv_gamma", c.ema_inv_gamma, Float32(2.0))
    _close("ema_power", c.ema_power, Float32(0.75))
    _eq("ema_update_after_step", c.ema_update_after_step, 50)
    _close("ema_min_decay", c.ema_min_decay, Float32(0.1))
    _close("ema_max_decay", c.ema_max_decay, Float32(0.999))
    # Cached-input / AV trainer contract.
    _eq("train_modality(av->1)", c.train_modality, TRAIN_MODALITY_AV)
    _eq("lora_target_preset(v2v->2)", c.lora_target_preset, LORA_TARGET_LTX2_V2V)
    _eqs("dataset_cache_dir", c.dataset_cache_dir, String("/tmp/ltx2_av_cache"))
    _eqb("require_cached_video_latents", c.require_cached_video_latents, True)
    _eqb("require_cached_text_embeddings", c.require_cached_text_embeddings, True)
    _eqb("require_cached_audio_latents", c.require_cached_audio_latents, True)
    _eqb("hot_loop_device_only", c.hot_loop_device_only, True)
    _close("video_loss_weight", c.video_loss_weight, Float32(1.0))
    _close("audio_loss_weight", c.audio_loss_weight, Float32(0.25))
    # Wave 2B — adapter algo (string -> int)
    _eq("adapter_algo(full->1)", c.adapter_algo, TRAIN_ADAPTER_ALGO_FULL)
    print("  gate (a) PASS — Wave 2 keys and AV cache contract keys reachable + mapped correctly")


# ─────────────────────────────────────────────────────────────────────────────
def _gate_none_set() raises:
    print("--- gate (b): NO Wave 2 keys set -> every field == default() ---")
    var path = String("/tmp/reader_levers_none.json")
    var js = String("{") + _ARCH + "}"  # arch only, no Wave 2 keys
    _write_file(path, js)

    var c = read_model_config(path)
    var d = TrainConfig.default()

    # Every Wave 2 field must equal the default() value (DEFAULT-OFF invariance).
    _eq("lr_scheduler", c.lr_scheduler, d.lr_scheduler)
    _eq("lr_warmup_steps", c.lr_warmup_steps, d.lr_warmup_steps)
    _close("lr_min_factor", c.lr_min_factor, d.lr_min_factor)
    _close("lr_cycles", c.lr_cycles, d.lr_cycles)
    _close("min_snr_gamma", c.min_snr_gamma, d.min_snr_gamma)
    _eqb("debiased", c.debiased, d.debiased)
    _close("loss_mse_strength", c.loss_mse_strength, d.loss_mse_strength)
    _close("loss_mae_strength", c.loss_mae_strength, d.loss_mae_strength)
    _close("loss_huber_strength", c.loss_huber_strength, d.loss_huber_strength)
    _eq("timestep_bias_strategy", c.timestep_bias_strategy, d.timestep_bias_strategy)
    _close("timestep_bias_multiplier", c.timestep_bias_multiplier, d.timestep_bias_multiplier)
    _close("timestep_bias_range_min", c.timestep_bias_range_min, d.timestep_bias_range_min)
    _close("timestep_bias_range_max", c.timestep_bias_range_max, d.timestep_bias_range_max)
    _eq("timestep_distribution", c.timestep_distribution, d.timestep_distribution)
    _close("timestep_noising_weight", c.timestep_noising_weight, d.timestep_noising_weight)
    _close("timestep_noising_bias", c.timestep_noising_bias, d.timestep_noising_bias)
    _close("caption_dropout_prob", c.caption_dropout_prob, d.caption_dropout_prob)
    _close("offset_noise_weight", c.offset_noise_weight, d.offset_noise_weight)
    _close("offset_noise_prob", c.offset_noise_prob, d.offset_noise_prob)
    _close("input_perturbation", c.input_perturbation, d.input_perturbation)
    _eq("multires_iterations", c.multires_iterations, d.multires_iterations)
    _close("multires_discount", c.multires_discount, d.multires_discount)
    _eq("grad_accum_steps", c.grad_accum_steps, d.grad_accum_steps)
    _eqb("ema_enabled", c.ema_enabled, d.ema_enabled)
    _close("ema_inv_gamma", c.ema_inv_gamma, d.ema_inv_gamma)
    _close("ema_power", c.ema_power, d.ema_power)
    _eq("ema_update_after_step", c.ema_update_after_step, d.ema_update_after_step)
    _close("ema_min_decay", c.ema_min_decay, d.ema_min_decay)
    _close("ema_max_decay", c.ema_max_decay, d.ema_max_decay)
    _eq("train_modality", c.train_modality, TRAIN_MODALITY_VIDEO)
    _eq("lora_target_preset", c.lora_target_preset, LORA_TARGET_LEGACY_VIDEO_ATTN1)
    _eqs("dataset_cache_dir", c.dataset_cache_dir, d.dataset_cache_dir)
    _eqb("require_cached_video_latents", c.require_cached_video_latents, d.require_cached_video_latents)
    _eqb("require_cached_text_embeddings", c.require_cached_text_embeddings, d.require_cached_text_embeddings)
    _eqb("require_cached_audio_latents", c.require_cached_audio_latents, d.require_cached_audio_latents)
    _eqb("hot_loop_device_only", c.hot_loop_device_only, d.hot_loop_device_only)
    _close("video_loss_weight", c.video_loss_weight, d.video_loss_weight)
    _close("audio_loss_weight", c.audio_loss_weight, d.audio_loss_weight)
    _eq("adapter_algo", c.adapter_algo, d.adapter_algo)
    print("  gate (b) PASS — absent keys preserve default() (baseline byte-unchanged)")


# ─────────────────────────────────────────────────────────────────────────────
def _gate_bogus_enum() raises:
    print("--- gate (c): bogus enum string must FAIL LOUD ---")
    var path = String("/tmp/reader_levers_bogus.json")
    var js = String("{") + _ARCH + ',"lr_scheduler":"bogus"}'
    _write_file(path, js)

    var raised = False
    try:
        var c = read_model_config(path)
        _ = c.lr_scheduler  # keep `c` live; should be unreachable
    except e:
        raised = True
        print("  reader raised as expected:", String(e))
    if not raised:
        raise Error("gate (c) FAIL — bogus lr_scheduler did NOT raise (silent default!)")
    print("  gate (c) PASS — unknown enum string raises (fail-loud)")


def _gate_adapter_aliases() raises:
    print("--- gate (d): network_algorithm aliases map to adapter ids ---")
    var locon_path = String("/tmp/reader_adapter_locon.json")
    _write_file(locon_path, String("{") + _ARCH + ',"network_algorithm":"locon"}')
    var locon = read_model_config(locon_path)
    _eq("network_algorithm(locon->7)", locon.adapter_algo, TRAIN_ADAPTER_ALGO_LOCON)

    var loha_path = String("/tmp/reader_adapter_loha.json")
    _write_file(loha_path, String("{") + _ARCH + ',"adapter_algo":"LOHA"}')
    var loha = read_model_config(loha_path)
    _eq("adapter_algo(LOHA->2)", loha.adapter_algo, TRAIN_ADAPTER_ALGO_LOHA)

    var lokr_path = String("/tmp/reader_adapter_lokr.json")
    _write_file(lokr_path, String("{") + _ARCH + ',"network_algorithm":"lokr"}')
    var lokr = read_model_config(lokr_path)
    _eq("network_algorithm(lokr->4)", lokr.adapter_algo, TRAIN_ADAPTER_ALGO_LOKR)
    print("  gate (d) PASS — lora/locon/loha/lokr aliases are reachable")


# ─────────────────────────────────────────────────────────────────────────────
# Parity-bitrot demo: a deliberately-wrong expected value MUST make the gate
# raise (-> nonzero exit). Toggle SHOW_BITROT_DEMO to True to witness it fail.
comptime SHOW_BITROT_DEMO = False


def _bitrot_demo() raises:
    print("--- bitrot guard demo: deliberate-wrong assertion exits nonzero ---")
    var path = String("/tmp/reader_levers_all.json")  # written by gate (a)
    var c = read_model_config(path)
    # WRONG on purpose: "cosine" maps to 2, we assert 99 -> must raise.
    _eq("DELIBERATE-WRONG lr_scheduler", c.lr_scheduler, 99)


def main() raises:
    print("=== reader Wave 2 levers reachability smoke ===")
    _gate_all_set()
    _gate_none_set()
    _gate_bogus_enum()
    _gate_adapter_aliases()

    comptime if SHOW_BITROT_DEMO:
        _bitrot_demo()  # raises -> process exits nonzero (bitrot guard proof)

    print("reader_levers_reachability_smoke PASS (a+b+c+d green)")
