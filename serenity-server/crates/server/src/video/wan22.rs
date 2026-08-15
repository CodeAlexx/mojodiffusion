//! Wan 2.2 video admission, request validation, and Mojo orchestration.

use super::*;

/// Wan2.2 two-process video: encode UMT5 conditions, then run the bounded T2V
/// pipeline through the repository-built Mojo executables.

pub(super) const WAN22_ENCODE: &str = "output/bin/wan22_encode_prompt";
pub(super) const WAN22_T2V: &str = "output/bin/wan22_t2v_1280x704";
pub(super) const WAN22_T2V_PORTRAIT: &str = "output/bin/wan22_t2v_704x1280";
pub(super) const WAN22_I2V_LANDSCAPE: &str = "output/bin/wan22_t2v_1248x704";
pub(super) const WAN22_I2V_PORTRAIT: &str = "output/bin/wan22_t2v_704x1248";
pub(super) const WAN22_FIRST_FRAME_LANDSCAPE: &str = "output/bin/wan22_encode_first_frame_1280x704";
pub(super) const WAN22_FIRST_FRAME_PORTRAIT: &str = "output/bin/wan22_encode_first_frame_704x1280";
pub(super) const WAN22_FIRST_FRAME_I2V_LANDSCAPE: &str =
    "output/bin/wan22_encode_first_frame_1248x704";
pub(super) const WAN22_FIRST_FRAME_I2V_PORTRAIT: &str =
    "output/bin/wan22_encode_first_frame_704x1248";
pub(super) const WAN22_A14B_LORA_T2V: &str = "output/bin/wan22_a14b_lora_t2v";
pub(super) const WAN22_A14B_HIGH: &str = "checkpoints/wan2.2_t2v_a14b_fp8_e4m3/high";
pub(super) const WAN22_A14B_LOW: &str = "checkpoints/wan2.2_t2v_a14b_fp8_e4m3/low";
pub(super) const WAN22_A14B_VAE: &str = "lingbot-video-moe/vae/diffusion_pytorch_model.safetensors";
pub(super) const WAN22_MODEL_ROOT: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo";
pub(super) const WAN22_ARTIFACT_MANIFEST: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/serenity_wan22_manifest.json";
pub(super) const WAN22_TRANSFORMER_SHARD_1: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/diffusion_pytorch_model-00001-of-00003.safetensors";
pub(super) const WAN22_TRANSFORMER_SHARD_2: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/diffusion_pytorch_model-00002-of-00003.safetensors";
pub(super) const WAN22_TRANSFORMER_SHARD_3: &str =
    "checkpoints/Wan2.2-TI2V-5B-Mojo/diffusion_pytorch_model-00003-of-00003.safetensors";
pub(super) const WAN22_UMT5_FILE: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo/umt5/model.safetensors";
pub(super) const WAN22_TOKENIZER: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo/tokenizer.json";
pub(super) const WAN22_SPIECE: &str = "checkpoints/Wan2.2-TI2V-5B-Mojo/spiece.model";
pub(super) const WAN22_VAE: &str = "vaes/wan2.2_vae.safetensors";
pub(super) const WAN22_PRODUCT_GATE: &str = "output/checks/wan22_product_gate.json";
pub(super) const WAN22_HF_REVISION: &str = "installed-official-native";
pub(super) const WAN22_CREATOR_REVISION: &str = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc";
pub(super) const WAN22_TRANSFORMER_INDEX_SHA256: &str =
    "cd769dd8bddb0825ffb3516a39d64fc2ac3a5946fb93337f8594af926d6a0f56";
pub(super) const WAN22_LOCAL_TRANSFORMER_INDEX_SHA256: &str =
    "ff3fe4b6936ac924f881863bcaeda0e5e1e54c8b7e2202b2990aba8fcf18ce47";
pub(super) const WAN22_TRANSFORMER_SHARD_SHA256: [&str; 3] = [
    "07cddfa20368c5e0884ee6660ed82b29d7ac97a9207b31fb630e4557c5308eb7",
    "38b79f68c95618f5341d4deae5ab364f9c74f10e8e903326499d0cb95353f1ff",
    "8d76abc71dee3e61a59ccc3a2e40889bb52ec9697acebfa7110de73f2a510452",
];
pub(super) const WAN22_VAE_SHA256: &str =
    "e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156";
pub(super) const WAN22_RUNNER_SOURCE_BUNDLE_SHA256: &str =
    "ea317b6ae0914c4828d85489c1e5a2d0952d2ca3880a122e56287771f65d24fe";
pub(super) const WAN22_DEFAULT_NEGATIVE: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";
pub(super) const WAN22_CUDA_CACHE: &str = "/dev/shm/serenity-wan22-cuda-cache";

pub(super) const WAN22_FRAMES: i64 = 121;
pub(super) const WAN22_WIDTH: i64 = 1280;
pub(super) const WAN22_HEIGHT: i64 = 704;
pub(super) const WAN22_PORTRAIT_WIDTH: i64 = 704;
pub(super) const WAN22_PORTRAIT_HEIGHT: i64 = 1280;
pub(super) const WAN22_I2V_LANDSCAPE_WIDTH: i64 = 1248;
pub(super) const WAN22_I2V_LANDSCAPE_HEIGHT: i64 = 704;
pub(super) const WAN22_I2V_PORTRAIT_WIDTH: i64 = 704;
pub(super) const WAN22_I2V_PORTRAIT_HEIGHT: i64 = 1248;
pub(super) const WAN22_FPS: i64 = 24;
pub(super) const WAN22_DEFAULT_STEPS: i64 = 50;
pub(super) const WAN22_I2V_STEPS: i64 = 50;
pub(super) const WAN22_DEFAULT_GUIDANCE: f64 = 5.0;
pub(super) const WAN22_A14B_FRAMES: i64 = 81;
pub(super) const WAN22_A14B_WIDTH: i64 = 832;
pub(super) const WAN22_A14B_HEIGHT: i64 = 480;
pub(super) const WAN22_A14B_FPS: i64 = 16;
pub(super) const WAN22_A14B_STEPS: i64 = 40;
pub(super) const WAN22_A14B_GUIDANCE: f64 = 3.0;

pub(super) fn wan22_missing() -> Vec<String> {
    let mut m = Vec::new();
    if !bin_x(WAN22_ENCODE) {
        m.push(WAN22_ENCODE.to_string());
    }
    if !bin_x(WAN22_T2V) {
        m.push(WAN22_T2V.to_string());
    }
    if !bin_x(WAN22_T2V_PORTRAIT) {
        m.push(WAN22_T2V_PORTRAIT.to_string());
    }
    if !bin_x(WAN22_I2V_LANDSCAPE) {
        m.push(WAN22_I2V_LANDSCAPE.to_string());
    }
    if !bin_x(WAN22_I2V_PORTRAIT) {
        m.push(WAN22_I2V_PORTRAIT.to_string());
    }
    for binary in [
        WAN22_FIRST_FRAME_LANDSCAPE,
        WAN22_FIRST_FRAME_PORTRAIT,
        WAN22_FIRST_FRAME_I2V_LANDSCAPE,
        WAN22_FIRST_FRAME_I2V_PORTRAIT,
    ] {
        if !bin_x(binary) {
            m.push(binary.to_string());
        }
    }
    for path in [
        WAN22_ARTIFACT_MANIFEST,
        WAN22_TRANSFORMER_SHARD_1,
        WAN22_TRANSFORMER_SHARD_2,
        WAN22_TRANSFORMER_SHARD_3,
        WAN22_UMT5_FILE,
        WAN22_TOKENIZER,
        WAN22_SPIECE,
        WAN22_VAE,
    ] {
        let resolved = model_path(path);
        if !nonempty_file(&resolved) {
            m.push(resolved.to_string_lossy().into_owned());
        }
    }
    m
}

pub(super) fn wan22_a14b_cache_complete(path: &std::path::Path) -> bool {
    nonempty_file(&path.join("shared.safetensors"))
        && (0..40).all(|index| nonempty_file(&path.join(format!("block_{index:02}.safetensors"))))
}

pub(super) fn wan22_a14b_missing() -> Vec<String> {
    let mut missing = Vec::new();
    for binary in [WAN22_ENCODE, WAN22_A14B_LORA_T2V] {
        if !bin_x(binary) {
            missing.push(binary.to_string());
        }
    }
    for cache in [WAN22_A14B_HIGH, WAN22_A14B_LOW] {
        let resolved = model_path(cache);
        if !wan22_a14b_cache_complete(&resolved) {
            missing.push(resolved.to_string_lossy().into_owned());
        }
    }
    let vae = model_path(WAN22_A14B_VAE);
    if !nonempty_file(&vae) {
        missing.push(vae.to_string_lossy().into_owned());
    }
    missing
}

pub(super) fn wan22_a14b_lora(body: &Value) -> Result<(std::path::PathBuf, f64, String), String> {
    let rows = body
        .get("lora")
        .or_else(|| body.get("loras"))
        .and_then(Value::as_array)
        .ok_or_else(|| "Wan2.2 A14B preview requires exactly one authored LoRA".to_string())?;
    if rows.len() != 1 {
        return Err("Wan2.2 A14B preview requires exactly one authored LoRA".to_string());
    }
    let row = rows[0]
        .as_object()
        .ok_or_else(|| "Wan2.2 A14B lora[0] must be an object".to_string())?;
    let name = row.get("name").and_then(Value::as_str).unwrap_or("").trim();
    if name.is_empty() {
        return Err("Wan2.2 A14B lora[0].name is required".to_string());
    }
    let weight = row
        .get("weight")
        .or_else(|| row.get("strength"))
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    if !weight.is_finite() || !(-10.0..=10.0).contains(&weight) {
        return Err("Wan2.2 A14B lora[0] weight must be finite in [-10, 10]".to_string());
    }
    let Some((path, arch)) = crate::models::lora_path_and_arch(name) else {
        return Err(format!(
            "Wan2.2 A14B LoRA not found in the model registry: {name}"
        ));
    };
    if !path.is_file() {
        return Err(format!(
            "Wan2.2 A14B LoRA path is missing: {}",
            path.display()
        ));
    }
    if arch != "wan2.2" {
        return Err(format!(
            "Wan2.2 A14B LoRA '{name}' targets '{arch}', not wan2.2"
        ));
    }
    Ok((path, weight, name.to_string()))
}

pub(super) fn wan22_ti2v5b_lora_header(doc: &Value) -> Result<usize, String> {
    let Some(tensors) = doc.as_object() else {
        return Err("invalid safetensors header".to_string());
    };
    let mut pairs = 0;
    for (key, entry) in tensors {
        if key == "__metadata__" || !key.ends_with(".lora_A.weight") {
            continue;
        }
        if !key.starts_with("diffusion_model.blocks.") {
            return Err(format!(
                "unsupported Wan LoRA key '{key}'; TI2V-5B admits torchref/DiffusionModel block adapters"
            ));
        }
        let prefix = key.trim_end_matches(".lora_A.weight");
        let b_key = format!("{prefix}.lora_B.weight");
        let Some(b_entry) = tensors.get(&b_key) else {
            return Err(format!("Wan LoRA is missing pair tensor '{b_key}'"));
        };
        let a_shape = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("Wan LoRA tensor '{key}' has no shape"))?;
        let b_shape = b_entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("Wan LoRA tensor '{b_key}' has no shape"))?;
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(format!("Wan TI2V-5B LoRA tensor '{key}' must be rank-2"));
        }
        let rank = a_shape[0]
            .as_u64()
            .ok_or_else(|| format!("Wan LoRA tensor '{key}' has invalid rank"))?;
        let input = a_shape[1]
            .as_u64()
            .ok_or_else(|| format!("Wan LoRA tensor '{key}' has invalid input size"))?;
        let output = b_shape[0]
            .as_u64()
            .ok_or_else(|| format!("Wan LoRA tensor '{b_key}' has invalid output size"))?;
        let b_rank = b_shape[1]
            .as_u64()
            .ok_or_else(|| format!("Wan LoRA tensor '{b_key}' has invalid rank"))?;
        if rank == 0 || b_rank != rank {
            return Err(format!("Wan LoRA A/B rank mismatch for '{prefix}'"));
        }
        let module = prefix
            .split_once(".blocks.")
            .and_then(|(_, tail)| tail.split_once('.').map(|(_, module)| module))
            .unwrap_or("");
        let expected = match module {
            "self_attn.q" | "self_attn.k" | "self_attn.v" | "self_attn.o" | "cross_attn.q"
            | "cross_attn.k" | "cross_attn.v" | "cross_attn.o" => (3_072, 3_072),
            "ffn.0" => (14_336, 3_072),
            "ffn.2" => (3_072, 14_336),
            _ => {
                return Err(format!(
                    "Wan TI2V-5B LoRA module '{module}' is not an admitted block linear"
                ));
            }
        };
        if (output, input) != expected {
            return Err(format!(
                "Wan LoRA module '{module}' has [{output},{input}], expected TI2V-5B [{},{}]; this is probably a 14B adapter",
                expected.0, expected.1
            ));
        }
        for (tensor_key, tensor_entry) in [(key.as_str(), entry), (b_key.as_str(), b_entry)] {
            let dtype = tensor_entry
                .get("dtype")
                .and_then(Value::as_str)
                .unwrap_or("");
            if !matches!(dtype, "BF16" | "F16" | "F32") {
                return Err(format!(
                    "Wan LoRA tensor '{tensor_key}' uses unsupported dtype '{dtype}'"
                ));
            }
        }
        pairs += 1;
    }
    if pairs == 0 {
        return Err(
            "no torchref/DiffusionModel Wan TI2V-5B LoRA A/B pairs were found".to_string(),
        );
    }
    Ok(pairs)
}

pub(super) fn wan22_ti2v5b_lora(
    body: &Value,
) -> Result<Option<(std::path::PathBuf, f64, String, usize)>, String> {
    let Some(rows) = body
        .get("lora")
        .or_else(|| body.get("loras"))
        .and_then(Value::as_array)
    else {
        return Ok(None);
    };
    if rows.is_empty() {
        return Ok(None);
    }
    if rows.len() != 1 {
        return Err("Wan2.2-TI2V-5B currently accepts one resident LoRA per render".to_string());
    }
    let row = rows[0]
        .as_object()
        .ok_or_else(|| "Wan2.2-TI2V-5B lora[0] must be an object".to_string())?;
    let name = row.get("name").and_then(Value::as_str).unwrap_or("").trim();
    if name.is_empty() {
        return Err("Wan2.2-TI2V-5B lora[0].name is required".to_string());
    }
    let weight = row
        .get("weight")
        .or_else(|| row.get("strength"))
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    if !weight.is_finite() || !(-10.0..=10.0).contains(&weight) {
        return Err("Wan2.2-TI2V-5B lora[0] weight must be finite in [-10, 10]".to_string());
    }
    if crate::models::lora_usage(name) != "overlay" {
        return Err(format!(
            "Wan2.2-TI2V-5B LoRA '{name}' is a feature adapter, not a model overlay"
        ));
    }
    let Some((path, arch)) = crate::models::lora_path_and_arch(name) else {
        return Err(format!(
            "Wan2.2-TI2V-5B LoRA not found in the model registry: {name}"
        ));
    };
    if arch != "wan2.2" {
        return Err(format!(
            "Wan2.2-TI2V-5B LoRA '{name}' targets '{arch}', not wan2.2"
        ));
    }
    let header = safetensors_header(&path).ok_or_else(|| {
        format!(
            "cannot read Wan LoRA safetensors header: {}",
            path.display()
        )
    })?;
    let pairs = wan22_ti2v5b_lora_header(&header)?;
    Ok(Some((path, weight, name.to_string(), pairs)))
}

/// Read acceptance only from the machine-local evidence gate. The report is
/// regenerated by scripts/check_wan22_product_gate.py after verifying the
/// pinned native BF16 shards/VAE, runtime parity, representative frame
/// bytes, muxed T2V/I2V artifacts, visual inspection, wall time, and peak VRAM.
pub(super) fn wan22_product_gate_passed() -> bool {
    let Ok(bytes) = std::fs::read(repo_path(WAN22_PRODUCT_GATE)) else {
        return false;
    };
    let Ok(doc) = serde_json::from_slice::<Value>(&bytes) else {
        return false;
    };
    doc.get("schema").and_then(Value::as_str) == Some("serenity.wan22.product_gate.v3")
        && doc.get("passed").and_then(Value::as_bool) == Some(true)
        && doc.pointer("/pins/hf_revision").and_then(Value::as_str) == Some(WAN22_HF_REVISION)
        && doc
            .pointer("/pins/creator_revision")
            .and_then(Value::as_str)
            == Some(WAN22_CREATOR_REVISION)
        && doc
            .pointer("/pins/source_transformer_index_sha256")
            .and_then(Value::as_str)
            == Some(WAN22_TRANSFORMER_INDEX_SHA256)
        && doc
            .pointer("/pins/local_transformer_index_sha256")
            .and_then(Value::as_str)
            == Some(WAN22_LOCAL_TRANSFORMER_INDEX_SHA256)
        && (0..WAN22_TRANSFORMER_SHARD_SHA256.len()).all(|index| {
            doc.pointer(&format!("/pins/bf16_transformer_shard_sha256/{index}"))
                .and_then(Value::as_str)
                == Some(WAN22_TRANSFORMER_SHARD_SHA256[index])
        })
        && doc.pointer("/pins/bf16_vae_sha256").and_then(Value::as_str) == Some(WAN22_VAE_SHA256)
        && doc
            .pointer("/pins/runner_source_bundle_sha256")
            .and_then(Value::as_str)
            == Some(WAN22_RUNNER_SOURCE_BUNDLE_SHA256)
        && doc.pointer("/profile/width").and_then(Value::as_i64) == Some(WAN22_WIDTH)
        && doc.pointer("/profile/height").and_then(Value::as_i64) == Some(WAN22_HEIGHT)
        && doc.pointer("/profile/frames").and_then(Value::as_i64) == Some(WAN22_FRAMES)
        && doc.pointer("/profile/steps").and_then(Value::as_i64) == Some(WAN22_DEFAULT_STEPS)
        && doc.pointer("/profile/guidance").and_then(Value::as_f64) == Some(WAN22_DEFAULT_GUIDANCE)
        && doc.pointer("/profile/shift").and_then(Value::as_f64) == Some(5.0)
        && doc.pointer("/profile/quant").and_then(Value::as_str) == Some("bf16")
        && doc.pointer("/i2v_profile/steps").and_then(Value::as_i64) == Some(WAN22_I2V_STEPS)
        && doc.pointer("/i2v_profile/shift").and_then(Value::as_f64) == Some(5.0)
        && doc.pointer("/i2v_profile/quant").and_then(Value::as_str) == Some("bf16")
        && doc.pointer("/i2v_profile/width").and_then(Value::as_i64)
            == Some(WAN22_I2V_PORTRAIT_WIDTH)
        && doc.pointer("/i2v_profile/height").and_then(Value::as_i64)
            == Some(WAN22_I2V_PORTRAIT_HEIGHT)
        && doc
            .pointer("/checks/i2v_first_frame_identity")
            .and_then(Value::as_bool)
            == Some(true)
        && doc
            .pointer("/checks/vae_encoder_mojo_parity")
            .and_then(Value::as_bool)
            == Some(true)
        && doc
            .pointer("/checks/creator_prompt_extension")
            .and_then(Value::as_bool)
            == Some(true)
        && doc
            .pointer("/checks/transformer_bf16_stream_parity")
            .and_then(Value::as_bool)
            == Some(true)
        && doc
            .pointer("/performance/requires_isolated_gpu_worker")
            .and_then(Value::as_bool)
            == Some(true)
}

pub(super) fn normalized_wan22_camera_motion_request(body: &Value) -> Result<Value, String> {
    let mut normalized = body.clone();
    let motion = body
        .get("camera_motion")
        .and_then(Value::as_str)
        .unwrap_or("none")
        .trim();
    let suffix = match motion {
        "none" => "",
        "static" => ", static camera, locked off shot, no camera movement",
        "focus_shift" => ", focus shift, rack focus, changing focal point",
        "dolly_in" => ", dolly in, camera pushing forward, smooth forward movement",
        "dolly_out" => ", dolly out, camera pulling back, smooth backward movement",
        "dolly_left" => ", dolly left, camera tracking left, lateral movement",
        "dolly_right" => ", dolly right, camera tracking right, lateral movement",
        "jib_up" => ", jib up, camera rising up, upward crane movement",
        "jib_down" => ", jib down, camera lowering down, downward crane movement",
        other => return Err(format!("unsupported Wan camera_motion '{other}'")),
    };
    let object = normalized
        .as_object_mut()
        .ok_or_else(|| "Wan request must be an object".to_string())?;
    object.insert("camera_motion".to_string(), json!(motion));
    if !suffix.is_empty()
        && object
            .get("creator_camera_motion_applied")
            .and_then(Value::as_bool)
            != Some(true)
    {
        let prompt = object
            .get("prompt")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        object.insert("creator_prompt".to_string(), json!(prompt));
        object.insert("prompt".to_string(), json!(format!("{prompt}{suffix}")));
        object.insert("creator_camera_motion_suffix".to_string(), json!(suffix));
        object.insert("creator_camera_motion_applied".to_string(), json!(true));
    }
    Ok(normalized)
}

pub(super) fn validate_wan22_a14b_request(body: &Value) -> Result<(), String> {
    if body
        .get("prompt")
        .and_then(Value::as_str)
        .is_none_or(|value| value.trim().is_empty())
    {
        return Err("Wan2.2 A14B preview requires a non-empty prompt".to_string());
    }
    for (key, required) in [
        ("width", WAN22_A14B_WIDTH),
        ("height", WAN22_A14B_HEIGHT),
        ("frames", WAN22_A14B_FRAMES),
        ("steps", WAN22_A14B_STEPS),
        ("fps", WAN22_A14B_FPS),
    ] {
        let actual = body.get(key).and_then(Value::as_i64).unwrap_or(required);
        if actual != required {
            return Err(format!(
                "Wan2.2 A14B preview requires {key}={required}; requested {actual}"
            ));
        }
    }
    let guidance = body
        .get("guidance")
        .and_then(Value::as_f64)
        .unwrap_or(WAN22_A14B_GUIDANCE);
    if !guidance.is_finite() || (guidance - WAN22_A14B_GUIDANCE).abs() > f64::EPSILON {
        return Err(format!(
            "Wan2.2 A14B preview requires CFG {WAN22_A14B_GUIDANCE}"
        ));
    }
    let _ = ltx2_seed(body)?;
    let _ = wan22_a14b_lora(body)?;
    Ok(())
}

pub(super) fn wan22_command(bin_abs: &std::path::Path) -> std::process::Command {
    let root = repo_root();
    let pixi_lib = root.join(".pixi/envs/default/lib");
    let mut preload = [
        "libcudnn_graph.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_engines_tensor_ir.so.9",
        "libcudnn_heuristic.so.9",
    ]
    .into_iter()
    .map(|name| pixi_lib.join(name))
    .collect::<Vec<_>>();
    preload.extend([
        std::path::PathBuf::from("/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1"),
        std::path::PathBuf::from("/usr/lib/x86_64-linux-gnu/libnvidia-nvvm70.so.4"),
    ]);
    if let Ok(entries) = std::fs::read_dir("/usr/lib/x86_64-linux-gnu") {
        if let Some(gpucomp) = entries.flatten().map(|entry| entry.path()).find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("libnvidia-gpucomp.so."))
        }) {
            preload.push(gpucomp);
        }
    }
    if let Some(existing) = std::env::var_os("LD_PRELOAD") {
        preload.extend(std::env::split_paths(&existing));
    }
    let ram_cache_ready = std::fs::create_dir_all(WAN22_CUDA_CACHE).is_ok();
    let mut command = std::process::Command::new(bin_abs);
    command
        .current_dir(root)
        .env("LD_LIBRARY_PATH", mojo_ld_path())
        .env(
            "LD_PRELOAD",
            std::env::join_paths(preload).unwrap_or_default(),
        )
        .env("CUDA_CACHE_PATH", WAN22_CUDA_CACHE);
    if !ram_cache_ready {
        command.env("CUDA_CACHE_DISABLE", "1");
    }
    command
}

/// First `*.mp4` produced under `dir` (newest by mtime). "" if none — the runner
/// owns the output name, so we glob rather than hardcode it.
pub(super) fn find_mp4(dir: &std::path::Path) -> String {
    let mut best: Option<(std::time::SystemTime, String)> = None;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for ent in rd.flatten() {
            let p = ent.path();
            let is_mp4 = p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("mp4"))
                .unwrap_or(false);
            if !is_mp4 {
                continue;
            }
            let mtime = ent
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            let path = p.to_string_lossy().into_owned();
            if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
                best = Some((mtime, path));
            }
        }
    }
    best.map(|(_, p)| p).unwrap_or_default()
}

/// Count `frame_*.png` written by wan22_t2v under `dir`.
pub(super) fn count_wan22_frames(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .map(|n| n.starts_with("frame_") && n.ends_with(".png"))
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}

/// wan22_t2v writes `frame_%d.png` frames + prints a manual ffmpeg mux command
/// (it does NOT produce an mp4 itself). Do that mux here → `<dir>/wan22_t2v.mp4`.
/// Returns the mp4 path on success. Mirrors the runner's exact ffmpeg args (24fps,
/// libx264, yuv420p, +faststart).
pub(super) fn mux_wan22_frames(dir: &std::path::Path, fps: i64) -> Result<String, String> {
    let pattern = dir.join("frame_%d.png");
    let mp4 = dir.join("wan22_t2v.mp4");
    let out = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            &fps.to_string(),
            "-start_number",
            "0",
            "-i",
            &pattern.to_string_lossy(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            &mp4.to_string_lossy(),
        ])
        .output()
        .map_err(|e| format!("ffmpeg spawn: {e}"))?;
    if out.status.success() && mp4.exists() {
        Ok(mp4.to_string_lossy().into_owned())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// Creator `best_output_size` for TI2V-5B I2V. The CLI's 1280x704/704x1280
/// values are maximum-area buckets; the actual output follows the source
/// aspect ratio on a 32-pixel grid. Keep this byte-for-byte equivalent to
/// `wan/utils/utils.py::best_output_size` instead of stretching every source
/// into the two T2V shapes.
pub(super) fn wan22_creator_i2v_size(source_width: u32, source_height: u32) -> (i64, i64) {
    const ALIGN: f64 = 32.0;
    const EXPECTED_AREA: f64 = (WAN22_WIDTH * WAN22_HEIGHT) as f64;
    let ratio = source_width as f64 / source_height as f64;
    let output_width = (EXPECTED_AREA * ratio).sqrt();
    let output_height = EXPECTED_AREA / output_width;

    let width_first = (output_width / ALIGN).floor() * ALIGN;
    let height_from_width = (EXPECTED_AREA / width_first / ALIGN).floor() * ALIGN;
    let ratio_width_first = width_first / height_from_width;

    let height_first = (output_height / ALIGN).floor() * ALIGN;
    let width_from_height = (EXPECTED_AREA / height_first / ALIGN).floor() * ALIGN;
    let ratio_height_first = width_from_height / height_first;

    let distortion_width = (ratio / ratio_width_first).max(ratio_width_first / ratio);
    let distortion_height = (ratio / ratio_height_first).max(ratio_height_first / ratio);
    if distortion_width < distortion_height {
        (width_first as i64, height_from_width as i64)
    } else {
        (width_from_height as i64, height_first as i64)
    }
}

/// Wan2.2 T2V arm — two-process orchestration (encode umt5 conds → t2v → mp4).
/// Synchronous, multi-minute blocking render (matches the LTX2 arm's convention). All
/// validation happens BEFORE the binary-presence check + spawn, so a bad request
/// never launches the GPU.
pub(super) fn post_video_wan22(st: &AppState, b: &Value) -> Response {
    let s = |k: &str, d: &str| b.get(k).and_then(|v| v.as_str()).unwrap_or(d).to_string();
    let prompt = s("prompt", "").trim().to_string();
    if prompt.is_empty() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "wan22: 'prompt' is required",
        );
    }
    let requested_negative = s("negative_prompt", "").trim().to_string();
    let neg_prompt = if requested_negative.is_empty() {
        WAN22_DEFAULT_NEGATIVE.to_string()
    } else {
        requested_negative
    };
    let image_path = s("image_path", "").trim().to_string();
    let is_i2v = !image_path.is_empty();
    if is_i2v && !std::path::Path::new(&image_path).is_file() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("Wan TI2V first-frame image not found: {image_path}"),
        );
    }
    if b.get("last_image_path")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
    {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Wan2.2-TI2V-5B officially accepts one input image; last-frame conditioning requires a different FLF2V model and is not faked",
        );
    }
    for unsupported in [
        "video_path",
        "vace_path",
        "control_video_path",
        "motion_track_path",
    ] {
        if b.get(unsupported)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty())
        {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "Wan2.2-TI2V-5B does not support '{unsupported}' with the installed weights; VACE/control/motion models are not installed"
                ),
            );
        }
    }
    let lora = match wan22_ti2v5b_lora(b) {
        Ok(value) => value,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
    };
    // Both official BF16 storage and the row-scaled E4M3 cache are explicit
    // runtime choices. Never silently alias a BF16 UI selection to FP8.
    let quant = s("quant", "bf16");
    if quant != "fp8" && quant != "bf16" {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("wan22_t2v quant '{quant}' is unsupported; choose bf16 or fp8"),
        );
    }
    let frames = b
        .get("frames")
        .and_then(|v| v.as_i64())
        .unwrap_or(WAN22_FRAMES);
    if !(1..=121).contains(&frames) {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "'frames' out of range [1..121]",
        );
    }
    if frames != WAN22_FRAMES {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "wan22_t2v is comptime-compiled for {WAN22_FRAMES} frames ({WAN22_WIDTH}x{WAN22_HEIGHT}); requested {frames} requires a rebuild"
            ),
        );
    }
    let width = b
        .get("width")
        .and_then(|v| v.as_i64())
        .unwrap_or(WAN22_WIDTH);
    let height = b
        .get("height")
        .and_then(|v| v.as_i64())
        .unwrap_or(WAN22_HEIGHT);
    let fps = b.get("fps").and_then(|v| v.as_i64()).unwrap_or(24);
    let t2v_landscape = width == WAN22_WIDTH && height == WAN22_HEIGHT;
    let t2v_portrait = width == WAN22_PORTRAIT_WIDTH && height == WAN22_PORTRAIT_HEIGHT;
    let i2v_landscape = width == WAN22_I2V_LANDSCAPE_WIDTH && height == WAN22_I2V_LANDSCAPE_HEIGHT;
    let i2v_portrait = width == WAN22_I2V_PORTRAIT_WIDTH && height == WAN22_I2V_PORTRAIT_HEIGHT;
    if fps != WAN22_FPS {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("Wan2.2 creator profile requires {WAN22_FPS} fps; requested {fps}"),
        );
    }
    if is_i2v {
        let (source_width, source_height) =
            match image::image_dimensions(std::path::Path::new(&image_path)) {
                Ok(dimensions) => dimensions,
                Err(error) => {
                    return err_detail(
                        StatusCode::UNPROCESSABLE_ENTITY,
                        &format!(
                            "cannot inspect Wan I2V source dimensions for creator sizing: {error}"
                        ),
                    )
                }
            };
        let creator_size = wan22_creator_i2v_size(source_width, source_height);
        if (width, height) != creator_size {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "Wan2.2 creator I2V sizing for source {source_width}x{source_height} is {}x{}, not {width}x{height}",
                    creator_size.0, creator_size.1
                ),
            );
        }
        if !(t2v_landscape || t2v_portrait || i2v_landscape || i2v_portrait) {
            return err_detail(
                StatusCode::UNPROCESSABLE_ENTITY,
                &format!(
                    "Wan2.2 creator-derived I2V profile {width}x{height} is not precompiled on this installation"
                ),
            );
        }
    } else if !(t2v_landscape || t2v_portrait) {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "Wan2.2 T2V native profiles are {WAN22_WIDTH}x{WAN22_HEIGHT} and {WAN22_PORTRAIT_WIDTH}x{WAN22_PORTRAIT_HEIGHT}; requested {width}x{height}"
            ),
        );
    }
    let steps = b
        .get("steps")
        .and_then(|v| v.as_i64())
        .unwrap_or(if is_i2v {
            WAN22_I2V_STEPS
        } else {
            WAN22_DEFAULT_STEPS
        });
    let required_steps = if is_i2v {
        WAN22_I2V_STEPS
    } else {
        WAN22_DEFAULT_STEPS
    };
    if steps != required_steps {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "Wan2.2 {} creator profile requires exactly {required_steps} steps",
                if is_i2v { "I2V" } else { "T2V" }
            ),
        );
    }
    let seed = match ltx2_seed(b) {
        Ok(seed) => seed,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, error),
    };
    let guidance = b
        .get("guidance")
        .and_then(|v| v.as_f64())
        .unwrap_or(WAN22_DEFAULT_GUIDANCE);
    if !guidance.is_finite() || (guidance - WAN22_DEFAULT_GUIDANCE).abs() > f64::EPSILON {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!("wan22_t2v high-quality profile requires CFG {WAN22_DEFAULT_GUIDANCE}"),
        );
    }

    // Binary presence gate (cwd-relative, like the LTX2 runner). 422 naming absent.
    let missing = wan22_missing();
    if !missing.is_empty() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            &format!(
                "wan22 runtime prerequisites missing: {}",
                missing.join(", ")
            ),
        );
    }
    if !wan22_product_gate_passed() {
        return err_detail(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Wan2.2 machine-local high-quality product gate is not current",
        );
    }

    let n = st
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
    let video_id = format!("video-{n:04}");
    let out_dir = st.out_dir.join(&video_id);
    let _ = std::fs::create_dir_all(&out_dir);
    let conds = out_dir.join("wan22_conds.safetensors");
    let conds_s = conds.to_string_lossy().into_owned();
    let out_dir_s = out_dir.to_string_lossy().into_owned();
    let enc_log = out_dir.join("wan22_encode.log");
    let first_frame_log = out_dir.join("wan22_first_frame_encode.log");
    let t2v_log = out_dir.join("wan22_t2v.log");

    let abs_encode = repo_path(WAN22_ENCODE);
    let runner = match (width, height) {
        (WAN22_WIDTH, WAN22_HEIGHT) => WAN22_T2V,
        (WAN22_PORTRAIT_WIDTH, WAN22_PORTRAIT_HEIGHT) => WAN22_T2V_PORTRAIT,
        (WAN22_I2V_LANDSCAPE_WIDTH, WAN22_I2V_LANDSCAPE_HEIGHT) => WAN22_I2V_LANDSCAPE,
        (WAN22_I2V_PORTRAIT_WIDTH, WAN22_I2V_PORTRAIT_HEIGHT) => WAN22_I2V_PORTRAIT,
        _ => unreachable!("Wan profile validated before runner selection"),
    };
    let abs_t2v = repo_path(runner);
    let abs_first_frame = if is_i2v {
        let binary = match (width, height) {
            (WAN22_WIDTH, WAN22_HEIGHT) => WAN22_FIRST_FRAME_LANDSCAPE,
            (WAN22_PORTRAIT_WIDTH, WAN22_PORTRAIT_HEIGHT) => WAN22_FIRST_FRAME_PORTRAIT,
            (WAN22_I2V_LANDSCAPE_WIDTH, WAN22_I2V_LANDSCAPE_HEIGHT) => {
                WAN22_FIRST_FRAME_I2V_LANDSCAPE
            }
            (WAN22_I2V_PORTRAIT_WIDTH, WAN22_I2V_PORTRAIT_HEIGHT) => WAN22_FIRST_FRAME_I2V_PORTRAIT,
            _ => unreachable!("Wan I2V profile validated before encoder selection"),
        };
        Some(repo_path(binary))
    } else {
        None
    };
    let lora_path = lora
        .as_ref()
        .map(|(path, _, _, _)| path.to_string_lossy().into_owned())
        .unwrap_or_else(|| "-".to_string());
    let lora_weight = lora
        .as_ref()
        .map(|(_, weight, _, _)| *weight)
        .unwrap_or(1.0);

    // ── Step A: encode umt5 conds (prompt strings passed directly as argv) ──
    let ta = std::time::Instant::now();
    let mut encode = wan22_command(&abs_encode);
    encode.args([&prompt, &neg_prompt, &conds_s]);
    let enc = run_logged_with_gpu_peak(&mut encode, &enc_log);
    let enc_secs = ta.elapsed().as_secs_f64();
    let (enc_rc, encode_peak_vram_mib) = match enc {
        Ok(measured) => measured,
        Err(e) => {
            let _ = std::fs::write(&enc_log, format!("spawn failed: {e}"));
            (-1, None)
        }
    };
    if enc_rc != 0 {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1", "video_id": video_id, "model": "wan22",
                "state": "failed", "failed_step": "encode", "encode_exit_code": enc_rc,
                "encode_log": enc_log.to_string_lossy(), "out_dir": out_dir_s,
                "encode_seconds": enc_secs, "encode_peak_vram_mib": encode_peak_vram_mib,
                "error": "wan22_encode_prompt failed; inspect encode_log",
            }),
        );
    }

    // ── Step B: process-isolated creator first-frame VAE encode. ───────────
    // A same-process VAE->DiT handoff leaves allocator residue/fragmentation
    // and OOMs the exact BF16 stream on a 24 GB card. Persist only the small
    // first latent, then let process exit reclaim every VAE allocation.
    let first_frame_cache = out_dir.join("wan22_first_frame.safetensors");
    let first_frame_cache_s = first_frame_cache.to_string_lossy().into_owned();
    let (first_frame_arg, first_frame_rc, first_frame_secs, first_frame_peak_vram_mib) =
        if let Some(first_frame_binary) = abs_first_frame.as_ref() {
            let started = std::time::Instant::now();
            let mut encode_first = wan22_command(first_frame_binary);
            encode_first.args([&image_path, &first_frame_cache_s]);
            let measured = run_logged_with_gpu_peak(&mut encode_first, &first_frame_log);
            let seconds = started.elapsed().as_secs_f64();
            let (code, peak) = match measured {
                Ok(result) => result,
                Err(error) => {
                    let _ = std::fs::write(&first_frame_log, format!("spawn failed: {error}"));
                    (-1, None)
                }
            };
            (first_frame_cache_s.clone(), code, seconds, peak)
        } else {
            (String::new(), 0, 0.0, None)
        };
    if first_frame_rc != 0 {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1", "video_id": video_id, "model": "wan22",
                "state": "failed", "failed_step": "first_frame_encode",
                "encode_exit_code": enc_rc, "first_frame_exit_code": first_frame_rc,
                "encode_log": enc_log.to_string_lossy(),
                "first_frame_log": first_frame_log.to_string_lossy(),
                "out_dir": out_dir_s, "conds": conds_s,
                "encode_seconds": enc_secs,
                "first_frame_seconds": first_frame_secs,
                "encode_peak_vram_mib": encode_peak_vram_mib,
                "first_frame_peak_vram_mib": first_frame_peak_vram_mib,
                "error": "wan22_encode_first_frame failed; inspect first_frame_log",
            }),
        );
    }

    // ── Step C: creator-native 121-frame render in the selected precision ──
    let tb = std::time::Instant::now();
    let mut t2v = wan22_command(&abs_t2v);
    t2v.arg(&conds_s)
        .arg(&out_dir_s)
        .arg(frames.to_string())
        .arg(steps.to_string())
        .arg(seed.to_string())
        .arg(format!("{guidance}"))
        .arg("1")
        .arg(&first_frame_arg)
        .arg(&lora_path)
        .arg(format!("{lora_weight}"))
        .arg(&quant);
    let t2v = run_logged_with_gpu_peak(&mut t2v, &t2v_log);
    let t2v_secs = tb.elapsed().as_secs_f64();
    let (t2v_rc, t2v_peak_vram_mib) = match t2v {
        Ok(measured) => measured,
        Err(e) => {
            let _ = std::fs::write(&t2v_log, format!("spawn failed: {e}"));
            (-1, None)
        }
    };
    let total_wall = enc_secs + first_frame_secs + t2v_secs;
    if t2v_rc != 0 {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1", "video_id": video_id, "model": "wan22",
                "state": "failed", "failed_step": "t2v",
                "encode_exit_code": enc_rc, "first_frame_exit_code": first_frame_rc,
                "t2v_exit_code": t2v_rc,
                "encode_log": enc_log.to_string_lossy(),
                "first_frame_log": first_frame_log.to_string_lossy(),
                "t2v_log": t2v_log.to_string_lossy(),
                "out_dir": out_dir_s, "conds": conds_s,
                "encode_seconds": enc_secs, "first_frame_seconds": first_frame_secs,
                "t2v_seconds": t2v_secs, "total_wall_seconds": total_wall,
                "encode_peak_vram_mib": encode_peak_vram_mib,
                "first_frame_peak_vram_mib": first_frame_peak_vram_mib,
                "t2v_peak_vram_mib": t2v_peak_vram_mib,
                "error": "wan22_t2v failed; inspect t2v_log",
            }),
        );
    }

    // ── Step D: mux the frame_*.png the runner wrote into an mp4 (24fps). ──
    let frames_written = count_wan22_frames(&out_dir);
    let (mp4, mux) = if frames_written > 0 {
        match mux_wan22_frames(&out_dir, 24) {
            Ok(p) => (p, "muxed".to_string()),
            Err(e) => (String::new(), format!("mux_failed: {e}")),
        }
    } else {
        // fallback in case a runner ever writes an mp4 directly
        (find_mp4(&out_dir), "no_frames_written".to_string())
    };
    let mux_ok = mux == "muxed";
    let probe = if mux_ok {
        probe_video_path(&mp4).ok()
    } else {
        None
    };
    let artifact_ok = probe.as_ref().is_some_and(|value| {
        probe_matches_video_profile(value, width, height, WAN22_FRAMES, WAN22_FPS, false)
    }) && frames_written == WAN22_FRAMES as usize;
    let parity_ok = wan22_product_gate_passed();
    json_resp(
        if artifact_ok {
            StatusCode::OK
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        },
        &json!({
            "schema": "serenity.video_result.v1", "video_id": video_id, "model": "wan22",
            "backend": BACKEND_NAME, "control_plane": "serenity-server",
            "resident": if quant == "bf16" { "bf16_native_shards_pinned_host" } else { "fp8_e4m3_cached" },
            "mode": if is_i2v { "i2v_first_frame" } else { "t2v" },
            "readiness_label": if parity_ok { "quality_profile_ready" } else { "product_gate_required" },
            "accepted_video_artifact": artifact_ok, "accepted_video_parity": parity_ok,
            "target_width": width, "target_height": height, "frames": frames,
            "frames_written": frames_written, "mux": mux, "fps": WAN22_FPS,
            "steps": steps, "seed": seed, "guidance": guidance, "quant": quant,
            "flow_shift": 5.0,
            "image_path": if is_i2v { image_path.as_str() } else { "" },
            "camera_motion": b.get("camera_motion").and_then(Value::as_str).unwrap_or("none"),
            "creator_prompt": b.get("creator_prompt").and_then(Value::as_str).unwrap_or(&prompt),
            "lora": lora.as_ref().map(|(path, weight, name, pairs)| json!({
                "name": name,
                "weight": weight,
                "path": path,
                "matched_modules": pairs,
                "merge": if quant == "bf16" {
                    "exact_bf16_pinned_host_additive"
                } else {
                    "resident_fp8_requantized_once"
                },
            })),
            "negative_prompt_source": if b.get("negative_prompt").and_then(Value::as_str).is_some_and(|value| !value.trim().is_empty()) { "request" } else { "creator_default" },
            "encode_exit_code": enc_rc, "first_frame_exit_code": first_frame_rc,
            "t2v_exit_code": t2v_rc,
            "out_dir": out_dir_s, "conds": conds_s, "mp4": mp4,
            "first_frame_cache": if is_i2v { first_frame_cache_s.as_str() } else { "" },
            "mp4_url": if artifact_ok { format!("/out/{video_id}/wan22_t2v.mp4") } else { String::new() },
            "probe": probe,
            "encode_log": enc_log.to_string_lossy(),
            "first_frame_log": first_frame_log.to_string_lossy(),
            "t2v_log": t2v_log.to_string_lossy(),
            "encode_seconds": enc_secs, "first_frame_seconds": first_frame_secs,
            "t2v_seconds": t2v_secs, "total_wall_seconds": total_wall,
            "encode_peak_vram_mib": encode_peak_vram_mib,
            "first_frame_peak_vram_mib": first_frame_peak_vram_mib,
            "t2v_peak_vram_mib": t2v_peak_vram_mib,
            "note": if is_i2v {
                format!("Wan2.2-TI2V-5B creator-native I2V profile: process-isolated cover-resize/center-crop source VAE encode, clean frame-0 replacement before and after each step, per-token zero timestep for conditioned frame patches, {quant} DiT, Flow-UniPC 50-step shift-5 sampling, and 24 fps MP4 mux.")
            } else {
                format!("Wan2.2-TI2V-5B creator T2V profile: official UMT5 conditioning and default negative prompt, {quant} DiT, Flow-UniPC 50-step shift-5 sampling, tiled VAE decode, and 24 fps MP4 mux.")
            },
        }),
    )
}

/// Bounded Wan2.2 T2V-A14B LoRA preview. This is intentionally separate from
/// the accepted TI2V-5B profile: it runs the image-trained A14B adapter on the
/// matching dual-expert base and returns a short MP4 for checkpoint evaluation.
pub(super) fn post_video_wan22_a14b(st: &AppState, b: &Value) -> Response {
    if let Err(error) = validate_wan22_a14b_request(b) {
        return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error);
    }
    let prompt = b["prompt"].as_str().unwrap_or("").trim().to_string();
    let requested_negative = b
        .get("negative_prompt")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    let negative = if requested_negative.is_empty() {
        WAN22_DEFAULT_NEGATIVE.to_string()
    } else {
        requested_negative
    };
    let seed = match ltx2_seed(b) {
        Ok(seed) => seed,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, error),
    };
    let (lora_path, lora_weight, lora_name) = match wan22_a14b_lora(b) {
        Ok(value) => value,
        Err(error) => return err_detail(StatusCode::UNPROCESSABLE_ENTITY, &error),
    };

    let n = st
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        + 1;
    let video_id = format!("video-{n:04}");
    let out_dir = st.out_dir.join(&video_id);
    let _ = std::fs::create_dir_all(&out_dir);
    let conds = out_dir.join("wan22_a14b_conds.safetensors");
    let encode_log = out_dir.join("wan22_a14b_encode.log");
    let render_log = out_dir.join("wan22_a14b_t2v.log");
    let conds_s = conds.to_string_lossy().into_owned();
    let out_dir_s = out_dir.to_string_lossy().into_owned();

    let encode_started = std::time::Instant::now();
    let mut encode = wan22_command(&repo_path(WAN22_ENCODE));
    encode.args([&prompt, &negative, &conds_s]);
    let encode_result = run_logged_with_gpu_peak(&mut encode, &encode_log);
    let encode_seconds = encode_started.elapsed().as_secs_f64();
    let (encode_exit_code, encode_peak_vram_mib) = match encode_result {
        Ok(measured) => measured,
        Err(error) => {
            let _ = std::fs::write(&encode_log, format!("spawn failed: {error}"));
            (-1, None)
        }
    };
    if encode_exit_code != 0 {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1",
                "video_id": video_id,
                "model": "wan22_a14b",
                "state": "failed",
                "failed_step": "encode",
                "encode_exit_code": encode_exit_code,
                "encode_log": encode_log.to_string_lossy(),
                "out_dir": out_dir_s,
                "error": "wan22_encode_prompt failed; inspect encode_log",
            }),
        );
    }

    let render_started = std::time::Instant::now();
    let mut render = wan22_command(&repo_path(WAN22_A14B_LORA_T2V));
    render
        .arg(&conds_s)
        .arg(model_path(WAN22_A14B_HIGH))
        .arg(model_path(WAN22_A14B_LOW))
        .arg(&out_dir_s)
        .arg(&lora_path)
        .arg(format!("{lora_weight}"))
        .arg(WAN22_A14B_STEPS.to_string())
        .arg(seed.to_string())
        .arg("1")
        .arg(format!("{WAN22_A14B_GUIDANCE}"))
        .arg("12.0")
        .arg("0")
        .arg("4.0");
    let render_result = run_logged_with_gpu_peak(&mut render, &render_log);
    let render_seconds = render_started.elapsed().as_secs_f64();
    let (render_exit_code, render_peak_vram_mib) = match render_result {
        Ok(measured) => measured,
        Err(error) => {
            let _ = std::fs::write(&render_log, format!("spawn failed: {error}"));
            (-1, None)
        }
    };
    if render_exit_code != 0 {
        return json_resp(
            StatusCode::INTERNAL_SERVER_ERROR,
            &json!({
                "schema": "serenity.video_result.v1",
                "video_id": video_id,
                "model": "wan22_a14b",
                "state": "failed",
                "failed_step": "t2v",
                "render_exit_code": render_exit_code,
                "render_log": render_log.to_string_lossy(),
                "out_dir": out_dir_s,
                "error": "wan22_a14b_lora_t2v failed; inspect render_log",
            }),
        );
    }

    let mp4 = out_dir.join("wan22_a14b_lora_t2v.mp4");
    let mp4_s = mp4.to_string_lossy().into_owned();
    let probe = probe_video_path(&mp4_s).ok();
    let artifact_ok = probe.as_ref().is_some_and(|value| {
        probe_matches_video_profile(
            value,
            WAN22_A14B_WIDTH,
            WAN22_A14B_HEIGHT,
            WAN22_A14B_FRAMES,
            WAN22_A14B_FPS,
            false,
        )
    });
    json_resp(
        if artifact_ok {
            StatusCode::OK
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        },
        &json!({
            "schema": "serenity.video_result.v1",
            "video_id": video_id,
            "model": "wan22_a14b",
            "backend": BACKEND_NAME,
            "control_plane": "serenity-server",
            "state": if artifact_ok { "complete" } else { "failed" },
            "readiness_label": "experimental_lora_preview",
            "accepted_video_artifact": artifact_ok,
            "accepted_video_parity": false,
            "width": WAN22_A14B_WIDTH,
            "height": WAN22_A14B_HEIGHT,
            "frames": WAN22_A14B_FRAMES,
            "fps": WAN22_A14B_FPS,
            "steps": WAN22_A14B_STEPS,
            "seed": seed,
            "guidance": WAN22_A14B_GUIDANCE,
            "shift": 12.0,
            "expert": "dual_expert_t2v_boundary_0.875",
            "lora": [{"name": lora_name, "weight": lora_weight, "path": lora_path}],
            "out_dir": out_dir_s,
            "conds": conds_s,
            "mp4": mp4_s,
            "mp4_url": if artifact_ok {
                format!("/out/{video_id}/wan22_a14b_lora_t2v.mp4")
            } else {
                String::new()
            },
            "probe": probe,
            "encode_log": encode_log.to_string_lossy(),
            "render_log": render_log.to_string_lossy(),
            "encode_seconds": encode_seconds,
            "render_seconds": render_seconds,
            "total_wall_seconds": encode_seconds + render_seconds,
            "encode_peak_vram_mib": encode_peak_vram_mib,
            "render_peak_vram_mib": render_peak_vram_mib,
            "note": "Bounded T2V-A14B LoRA checkpoint preview; not the accepted TI2V-5B product profile.",
        }),
    )
}
