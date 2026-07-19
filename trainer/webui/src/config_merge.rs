// config_merge.rs — the SINGLE source of truth for how the web supervisor turns
// a preset (base template + recipe + UI overrides) into the runner config JSON,
// PLUS a schema validator that mirrors the trainer's own config reader
// (mojodiffusion serenitymojo/io/train_config_reader.mojo) so a UI-emitted value
// the reader would REJECT is caught here — offline, without a GPU or a mojo build.
//
// Shared between the server (src/main.rs `mod config_merge;`) and the offline
// smoke (`src/bin/config_smoke.rs` includes this same file via #[path]). Keeping
// ONE merge fn is the whole point: the smoke must merge EXACTLY as the server
// does, or it validates a config the server never builds.
//
// GAP (documented, wave 2): validate_config_enums replicates the reader's
// fail-loud ENUM string sets only (the "#2/#3 class": quantized_resident, ema,
// dtypes, optimizer, training_method, adapter_algo, lr_scheduler, time units,
// timestep_* …). It does NOT run the real Mojo parser, nor the reader's three
// cross-field validators (validate_training_method_config /
// validate_offload_checkpoint_config / validate_serenity_trainer_policy_config). Those
// need a mojo build; the enum layer is the cheap proxy the task asks for.

use serde_json::{json, Value};
use std::path::Path;

/// Build the runner config exactly like the server's `launch` handler:
///   base template (or {}) ← preset recipe ← UI overrides, then workspace_dir /
///   save_filename_prefix, then the sampling-strip for the backends whose inline
///   sampler is not wired. Returns (config, human notes). Err = base template
///   named but missing/unreadable (the server maps this to a 422).
pub fn build_merged_config(
    repo_root: &str,
    base_config: &str,
    recipe: &Value,
    backend: &str,
    run_name: &str,
    overrides: &Value,
) -> Result<(Value, Vec<String>), String> {
    let mut cfg: Value = if base_config.is_empty() {
        json!({}) // e.g. ideogram4: all-argv contract, no runner config template
    } else {
        let base_path = Path::new(repo_root).join(base_config);
        match std::fs::read_to_string(&base_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
        {
            Some(v) => v,
            None => {
                return Err(format!(
                    "base config missing/unreadable: {} — run this model once via CLI or add the template",
                    base_path.display()
                ))
            }
        }
    };
    let workspace = Path::new(repo_root)
        .join("output")
        .join(run_name)
        .to_string_lossy()
        .into_owned();
    if let Value::Object(m) = &mut cfg {
        if let Value::Object(r) = recipe {
            for (k, v) in r {
                m.insert(k.clone(), v.clone());
            }
        }
        if let Value::Object(o) = overrides {
            for (k, v) in o {
                m.insert(k.clone(), v.clone());
            }
        }
        m.insert("workspace_dir".into(), json!(workspace));
        m.insert("save_filename_prefix".into(), json!(run_name));
    }
    // inline sampling during training is not wired for these backends — their
    // trainers FAIL LOUD on a sample cadence / prompt file. Strip the triggers
    // (identical to launch()).
    let mut notes: Vec<String> = vec![];
    if matches!(backend, "sd35" | "hidream" | "ideogram4") {
        if let Value::Object(m) = &mut cfg {
            if m.get("sample_every").and_then(|v| v.as_u64()).unwrap_or(0) > 0 {
                m.insert("sample_every".into(), json!(0));
                notes.push(format!(
                    "sampling disabled: inline sampler not wired for backend {backend} — sample_every forced to 0"
                ));
            }
            for k in ["validation_prompts_file", "sample_definition_file_name"] {
                if m.remove(k).is_some() {
                    notes.push(format!(
                        "removed {k}: inline sampling not supported for backend {backend}"
                    ));
                }
            }
        }
    }
    Ok((cfg, notes))
}

// ── enum schema mirrored from train_config_reader.mojo ───────────────────────
// Each accepted set is transcribed from the reader's fail-loud mapper of the
// same name. Line references are to serenitymojo/io/train_config_reader.mojo.

// _dtype_int (l.194)
const DTYPES: &[&str] = &[
    "NONE", "FLOAT_8", "FLOAT_16", "FLOAT_32", "BFLOAT_16", "TFLOAT_32", "INT_8",
    "NFLOAT_4", "FLOAT_W8A8", "INT_W8A8", "GGUF", "GGUF_A8_FLOAT", "GGUF_A8_INT",
];
// _optimizer_int (l.224) — the SUPPORTED tags only (ADAM/CAME/LION/… fail loud)
const OPTIMIZERS: &[&str] = &[
    "ADAMW", "ADAMW_ADV", "ADAMW_8BIT", "ADAFACTOR", "SCHEDULE_FREE_ADAMW",
    "AUTOMAGIC3", "AUTOMAGIC_3", "AUTOMAGIC-3",
];
// _time_unit_int (l.269)
const TIME_UNITS: &[&str] = &[
    "EPOCH", "epoch", "STEP", "step", "SECOND", "second", "MINUTE", "minute",
    "HOUR", "hour", "NEVER", "never", "ALWAYS", "always",
];
// _ema_mode_int (l.287)
const EMA_MODES: &[&str] = &["OFF", "off", "GPU", "gpu", "CPU", "cpu", "EMA", "ema"];
// quantized_resident (l.832) — "" is accepted (default/unset)
const QUANT_RESIDENT: &[&str] =
    &["", "OFF", "fp8_e4m3", "fp8_e4m3_host", "streamed_base_opt_in"];
// _training_method_int (l.626)
const TRAINING_METHODS: &[&str] = &[
    "LORA", "lora", "LoRA", "FINE_TUNE", "fine_tune", "FineTune", "finetune",
    "FINETUNE", "full", "FULL", "Full",
];
// _lr_scheduler_int (l.524)
const LR_SCHEDULERS: &[&str] = &[
    "constant", "CONSTANT", "linear", "LINEAR", "cosine", "COSINE",
    "cosine_with_restarts", "COSINE_WITH_RESTARTS", "polynomial", "POLYNOMIAL",
    "rex", "REX",
];
// _loss_fn_int (l.545)
const LOSS_FNS: &[&str] = &["mse", "MSE", "huber", "HUBER", "smooth_l1", "SMOOTH_L1"];
// _timestep_bias_int (l.560)
const TS_BIAS: &[&str] = &["none", "later", "earlier", "range"];
// _timestep_distribution_int (l.576)
const TS_DIST: &[&str] = &[
    "uniform", "UNIFORM", "sigmoid", "SIGMOID", "logit_normal", "LOGIT_NORMAL",
    "HEAVY_TAIL", "heavy_tail", "COS_MAP", "cos_map", "INVERTED_PARABOLA",
    "inverted_parabola",
];
// _adapter_algo_int (l.598) — boft is explicitly REJECTED
const ADAPTER_ALGOS: &[&str] = &[
    "lora", "LORA", "locon", "LOCON", "lycoris", "LYCORIS", "full", "FULL",
    "loha", "LOHA", "dora", "DORA", "lokr", "LOKR", "oft", "OFT",
];
// _gradient_checkpointing_int (l.647) — string form
const GRAD_CKPT: &[&str] = &["OFF", "off", "ON", "on", "CPU_OFFLOADED", "cpu_offloaded"];
// _train_modality_int (l.670) string form (int 0/1/2 also accepted)
const TRAIN_MODALITY: &[&str] = &["video", "v", "av", "audio_video", "va", "audio", "a"];
// _lora_target_preset_int (l.700) string form (int 0..5 also accepted)
const LORA_TARGET_PRESET: &[&str] =
    &["legacy_video_attn1", "legacy", "t2v", "v2v", "audio", "audio_ref_only_ic", "full"];

fn dtype_keys() -> &'static [&'static str] {
    &[
        "train_dtype", "fallback_train_dtype", "weight_dtype", "output_dtype",
        "lora_weight_dtype", "embedding_weight_dtype", "unet_weight_dtype",
        "prior_weight_dtype", "transformer_weight_dtype",
        "text_encoder_weight_dtype", "text_encoder_2_weight_dtype",
        "text_encoder_3_weight_dtype", "text_encoder_4_weight_dtype",
        "vae_weight_dtype",
    ]
}

fn check_str_enum(cfg: &Value, key: &str, accepted: &[&str], out: &mut Vec<String>) {
    if let Some(v) = cfg.get(key) {
        if let Some(s) = v.as_str() {
            if !accepted.contains(&s) {
                out.push(format!(
                    "{key} = {:?} rejected by train_config_reader (accepted: {})",
                    s,
                    accepted.join("|")
                ));
            }
        } else {
            out.push(format!("{key} must be a string (reader reads it as a string enum)"));
        }
    }
}

/// Validate a merged config's enum-valued keys against the trainer's config
/// reader. Returns one message PER violation; empty = the reader would accept
/// every enum in this config. This is the offline proxy for the "#2/#3 class":
/// the UI emitting a value the reader fail-louds on.
pub fn validate_config_enums(cfg: &Value) -> Vec<String> {
    let mut out = vec![];

    check_str_enum(cfg, "quantized_resident", QUANT_RESIDENT, &mut out);
    check_str_enum(cfg, "ema", EMA_MODES, &mut out);
    for k in dtype_keys() {
        check_str_enum(cfg, k, DTYPES, &mut out);
    }
    for k in ["training_method", "train_method", "method"] {
        check_str_enum(cfg, k, TRAINING_METHODS, &mut out);
    }
    for k in ["network_algorithm", "algo", "adapter_algo"] {
        check_str_enum(cfg, k, ADAPTER_ALGOS, &mut out);
    }
    for k in ["lr_scheduler", "learning_rate_scheduler"] {
        check_str_enum(cfg, k, LR_SCHEDULERS, &mut out);
    }
    check_str_enum(cfg, "loss_fn", LOSS_FNS, &mut out);
    check_str_enum(cfg, "timestep_bias_strategy", TS_BIAS, &mut out);
    check_str_enum(cfg, "timestep_distribution", TS_DIST, &mut out);
    for k in [
        "stop_training_after_unit", "save_every_unit", "save_after_unit",
        "sample_after_unit", "validate_after_unit", "backup_after_unit",
    ] {
        check_str_enum(cfg, k, TIME_UNITS, &mut out);
    }

    // gradient_checkpointing: string (enum) OR bool/number (both accepted).
    if let Some(v) = cfg.get("gradient_checkpointing") {
        if let Some(s) = v.as_str() {
            if !GRAD_CKPT.contains(&s) {
                out.push(format!(
                    "gradient_checkpointing = {:?} rejected (accepted: {} | bool | number)",
                    s,
                    GRAD_CKPT.join("|")
                ));
            }
        }
    }

    // train_modality / lora_target_preset: string enum OR bounded int.
    for k in ["train_modality", "ltx2_mode", "modality"] {
        if let Some(v) = cfg.get(k) {
            if let Some(s) = v.as_str() {
                if !TRAIN_MODALITY.contains(&s) {
                    out.push(format!(
                        "{k} = {:?} rejected (accepted: {} | int 0..2)",
                        s,
                        TRAIN_MODALITY.join("|")
                    ));
                }
            } else if let Some(n) = v.as_i64() {
                if !(0..=2).contains(&n) {
                    out.push(format!("{k} = {n} rejected (int must be 0=video,1=av,2=audio)"));
                }
            }
        }
    }
    for k in ["lora_target_preset", "ltx2_lora_target_preset"] {
        if let Some(v) = cfg.get(k) {
            if let Some(s) = v.as_str() {
                if !LORA_TARGET_PRESET.contains(&s) {
                    out.push(format!(
                        "{k} = {:?} rejected (accepted: {} | int 0..5)",
                        s,
                        LORA_TARGET_PRESET.join("|")
                    ));
                }
            } else if let Some(n) = v.as_i64() {
                if !(0..=5).contains(&n) {
                    out.push(format!("{k} = {n} rejected (int must be 0..5)"));
                }
            }
        }
    }

    // lokr_targets: string attn|attn+ff|all OR int 1..3
    if let Some(v) = cfg.get("lokr_targets") {
        if let Some(s) = v.as_str() {
            if !matches!(s, "attn" | "attn+ff" | "all") {
                out.push(format!("lokr_targets = {s:?} rejected (accepted: attn|attn+ff|all | int 1..3)"));
            }
        } else if let Some(n) = v.as_i64() {
            if !(1..=3).contains(&n) {
                out.push(format!("lokr_targets = {n} rejected (int must be 1..3)"));
            }
        }
    }

    // numeric floors the reader fail-louds on
    if let Some(n) = cfg.get("controlnet_layers").and_then(|v| v.as_i64()) {
        if n < 0 {
            out.push(format!("controlnet_layers = {n} rejected (must be >= 0)"));
        }
    }
    if let Some(n) = cfg.get("init_lokr_norm").and_then(|v| v.as_f64()) {
        if n < 0.0 {
            out.push(format!("init_lokr_norm = {n} rejected (must be >= 0)"));
        }
    }

    // nested optimizer object: the reader reads optimizer.optimizer as the tag,
    // and structurally REQUIRES an object (a bare string fails cur.expect('{')).
    if let Some(v) = cfg.get("optimizer") {
        if let Some(o) = v.as_object() {
            if let Some(tag) = o.get("optimizer").and_then(|t| t.as_str()) {
                if !OPTIMIZERS.contains(&tag) {
                    out.push(format!(
                        "optimizer.optimizer = {:?} rejected (accepted: {})",
                        tag,
                        OPTIMIZERS.join("|")
                    ));
                }
            }
        } else {
            out.push("optimizer must be a JSON object (reader parses a nested object)".into());
        }
    }

    out
}
