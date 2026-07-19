// config_smoke — offline launch-config round-trip validator (webui audit wave 2,
// cross-cutting item). For EVERY preset in presets.json it builds the merged
// runner config with the SERVER'S OWN merge fn (config_merge::build_merged_config,
// shared verbatim via #[path]) and runs the trainer-reader enum schema over it
// (config_merge::validate_config_enums, transcribed from
// mojodiffusion serenitymojo/io/train_config_reader.mojo). No GPU, no mojo build,
// no running server.
//
// The bar (task): the "#2/#3 class" — the UI emits a value the trainer's config
// reader REJECTS — must be CAUGHT here. Proven three ways below:
//   1. every preset's merged config is clean (a regression would flag it),
//   2. an end-to-end injection: a bad UI override through the REAL merge is caught,
//   3. unit negative/positive tables pin the validator itself.
//
// Run:  cd webui && cargo run --release --bin config_smoke
// Exit: 0 = all clean + all self-tests pass; 1 = a real preset violation OR the
//       validator failed to catch an injected bad value (the class escaped).
//
// GAP: enum layer only — the reader's three cross-field validators
// (validate_training_method_config / _offload_checkpoint_config /
// _serenity_trainer_policy_config) and the actual byte-parse are NOT replicated here
// (they need a mojo build). Documented in config_merge.rs.

#[path = "../config_merge.rs"]
mod config_merge;

use serde_json::{json, Value};

fn repo_root() -> String {
    // CARGO_MANIFEST_DIR = <repo>/webui ; base_config paths are relative to <repo>
    let manifest = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest)
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .expect("trainer repository root")
        .to_string_lossy()
        .into_owned()
}

fn main() {
    let root = repo_root();
    let presets_path = format!("{root}/webui/presets.json");
    let raw = std::fs::read_to_string(&presets_path)
        .unwrap_or_else(|e| panic!("read {presets_path}: {e}"));
    let doc: Value = serde_json::from_str(&raw).expect("presets.json parse");
    let presets = doc["presets"].as_array().expect("presets array");

    let mut failed = false;
    println!("== config_smoke: {} presets from {presets_path} ==\n", presets.len());

    // ── 1) every preset merges clean through the server's merge fn ────────────
    for p in presets {
        let id = p["id"].as_str().unwrap_or("?");
        let backend = p["backend"].as_str().unwrap_or("");
        let wired = p["wired"].as_bool().unwrap_or(false);
        let base_config = p["base_config"].as_str().unwrap_or("");
        let run_name = p["run_name"].as_str().unwrap_or(id);
        let recipe = p.get("recipe").cloned().unwrap_or_else(|| json!({}));
        let overrides = json!({});

        if !wired {
            println!("  [skip] {id:<10} wired=false (never launches a config)");
            continue;
        }

        // Merge exactly as the server does. If the base template file is absent
        // (a not-yet-built target/*.json), fall back to an empty base so the
        // recipe/override layer — which carries quantized_resident etc. — is
        // still validated. The server itself 422s on a missing base, so this is
        // strictly ADDITIONAL coverage, noted below.
        let (cfg, note) = match config_merge::build_merged_config(
            &root, base_config, &recipe, backend, run_name, &overrides,
        ) {
            Ok((c, _notes)) => (c, String::new()),
            Err(_) => {
                let (c, _) = config_merge::build_merged_config(
                    &root, "", &recipe, backend, run_name, &overrides,
                )
                .expect("empty-base merge cannot fail");
                (c, format!(" (base template absent: {base_config} — recipe-only)"))
            }
        };

        let violations = config_merge::validate_config_enums(&cfg);
        if violations.is_empty() {
            println!("  [ok]   {id:<10} backend={backend:<9} clean{note}");
        } else {
            failed = true;
            println!("  [FAIL] {id:<10} backend={backend:<9}{note}");
            for v in &violations {
                println!("           ✗ {v}");
            }
        }
    }

    // ── 2) end-to-end injection: a bad UI override through the REAL merge ──────
    // Simulate the #2/#3 class directly: the form sends quantized_resident="fp8"
    // (a plausible typo of fp8_e4m3). It must survive the merge and be REJECTED.
    println!("\n-- injection (UI override through build_merged_config) --");
    {
        let krea2 = presets.iter().find(|p| p["id"] == json!("krea2")).expect("krea2 preset");
        let recipe = krea2["recipe"].clone();
        let bad = json!({"quantized_resident": "fp8", "ema": "YES"});
        let (cfg, _) = config_merge::build_merged_config(
            &root, "", &recipe, "krea2", "smoke", &bad,
        )
        .expect("merge");
        let v = config_merge::validate_config_enums(&cfg);
        let caught_quant = v.iter().any(|s| s.contains("quantized_resident"));
        let caught_ema = v.iter().any(|s| s.contains("ema"));
        if caught_quant && caught_ema {
            println!("  [ok]   injected quantized_resident=fp8 + ema=YES were CAUGHT ({} violations)", v.len());
        } else {
            failed = true;
            println!("  [FAIL] injected bad UI overrides ESCAPED the validator: {v:?}");
        }
    }

    // ── 3) unit tables: pin the validator (negatives must flag, positives must not)
    println!("\n-- validator unit table --");
    let negatives: &[(&str, Value)] = &[
        ("quantized_resident=fp8", json!({"quantized_resident": "fp8"})),
        ("ema=YES", json!({"ema": "YES"})),
        ("optimizer=LION", json!({"optimizer": {"optimizer": "LION"}})),
        ("adapter_algo=boft", json!({"adapter_algo": "boft"})),
        ("train_dtype=FP8", json!({"train_dtype": "FP8"})),
        ("training_method=peft", json!({"training_method": "peft"})),
        ("lr_scheduler=warmup", json!({"lr_scheduler": "warmup"})),
        ("loss_fn=l2", json!({"loss_fn": "l2"})),
        ("timestep_distribution=normal", json!({"timestep_distribution": "normal"})),
        ("stop_training_after_unit=steps", json!({"stop_training_after_unit": "steps"})),
        ("lora_target_preset=9", json!({"lora_target_preset": 9})),
        ("controlnet_layers=-1", json!({"controlnet_layers": -1})),
    ];
    let positives: &[(&str, Value)] = &[
        ("quantized_resident=fp8_e4m3", json!({"quantized_resident": "fp8_e4m3"})),
        ("quantized_resident=OFF", json!({"quantized_resident": "OFF"})),
        ("ema=EMA", json!({"ema": "EMA"})),
        ("optimizer=AUTOMAGIC3", json!({"optimizer": {"optimizer": "AUTOMAGIC3"}})),
        ("adapter_algo=lokr", json!({"adapter_algo": "lokr"})),
        ("train_dtype=BFLOAT_16", json!({"train_dtype": "BFLOAT_16"})),
        ("timestep_distribution=logit_normal", json!({"timestep_distribution": "logit_normal"})),
        ("train_modality=1(int)", json!({"train_modality": 1})),
        ("empty", json!({})),
    ];
    for (name, cfg) in negatives {
        let v = config_merge::validate_config_enums(cfg);
        if v.is_empty() {
            failed = true;
            println!("  [FAIL] negative {name} was NOT caught");
        } else {
            println!("  [ok]   negative {name} -> {}", v[0]);
        }
    }
    for (name, cfg) in positives {
        let v = config_merge::validate_config_enums(cfg);
        if v.is_empty() {
            println!("  [ok]   positive {name} accepted");
        } else {
            failed = true;
            println!("  [FAIL] positive {name} wrongly flagged: {v:?}");
        }
    }

    println!();
    if failed {
        eprintln!("config_smoke: FAIL — see ✗ lines above");
        std::process::exit(1);
    }
    println!("config_smoke: PASS — every preset merges to a reader-accepted config; validator table green");
}
