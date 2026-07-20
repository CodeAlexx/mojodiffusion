//! ComfyUI-API compatibility adapters (Tier A of the Konva-canvas pivot).
//!
//! The Konva web canvas (serenityflow-v2's `canvas/`) speaks the **ComfyUI API**:
//! `POST /prompt`, a single global `GET /ws`, `GET /view`, `GET /object_info`, plus a
//! handful of read-only endpoints. serenity-server's engine speaks `/v1/*`. These thin
//! adapters map the ComfyUI names onto the existing engine + graph-lowering path so the
//! canvas can drive a real generation end-to-end without any client changes.
//!
//! Boundaries:
//!   - `/prompt` reuses [`crate::enqueue_generate`] — the SAME lower→validate→enqueue
//!     core as `/v1/generate` — so both surface identical structured errors.
//!   - `/ws` subscribes to the global [`crate::ComfyBus`] (fed by every `JobChannel`
//!     via `attach_global`) and translates each `WorkerEvent` into ComfyUI messages.
//!   - `/object_info` is the canvas's MODEL-LIST source: `CheckpointLoaderSimple.
//!     ckpt_name` / `UNETLoader.unet_name` combos are filled from the disk scan.
//!   - `/view` serves a produced PNG from `out_dir` (the `executed` message points here).

use std::collections::HashMap;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::broadcast;

use serenity_wire::WorkerEvent;

use crate::{enqueue_generate, AppState, DriverCtl};

// ── POST /prompt ────────────────────────────────────────────────────────────────

/// ComfyUI queue endpoint. Body: `{"prompt": <graph>, "client_id": <id>}` where
/// `<graph>` is the API-prompt form `{"1": {"class_type", "inputs"}, ...}` the
/// canvas's WorkflowBuilder emits. We rewrap the graph under `workflow` so the shared
/// enqueue core runs the comfy-api-prompt lowering, and return `{"prompt_id": ...}`.
pub async fn post_prompt(State(st): State<AppState>, Json(body): Json<Value>) -> Response {
    let graph = body.get("prompt").cloned().unwrap_or(Value::Null);
    if !graph.is_object() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "type": "invalid_prompt",
                    "message": "request is missing a 'prompt' graph object",
                },
                "node_errors": {},
            })),
        )
            .into_response();
    }
    // Rewrap: the shared core lowers only when a `workflow` key is present.
    let req_value = json!({ "workflow": graph });
    match enqueue_generate(&st, req_value) {
        Ok(job_id) => (
            StatusCode::OK,
            Json(json!({
                "prompt_id": job_id,
                "number": 0,
                "node_errors": {},
            })),
        )
            .into_response(),
        // The engine already built a structured error Response (with status). The
        // canvas only checks `resp.ok` and shows the text, so surface it verbatim.
        Err(resp) => resp,
    }
}

// ── GET /ws — single global ComfyUI event socket ────────────────────────────────

/// The canvas opens ONE `/ws?clientId=` socket at page load (before any job) and
/// expects ComfyUI-format messages for every job keyed by `prompt_id`. We subscribe
/// to the global bus and translate each `WorkerEvent`.
pub async fn get_ws(State(st): State<AppState>, ws: WebSocketUpgrade) -> Response {
    let rx = st.comfy_ws.subscribe();
    ws.on_upgrade(move |socket| run_comfy_ws(socket, rx))
}

async fn run_comfy_ws(mut socket: WebSocket, mut rx: broadcast::Receiver<(String, WorkerEvent)>) {
    // Initial ComfyUI status frame (idle queue). Harmless if the canvas ignores it.
    let hello = json!({
        "type": "status",
        "data": { "status": { "exec_info": { "queue_remaining": 0 } } }
    })
    .to_string();
    if socket.send(Message::Text(hello)).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            // Drain inbound frames (close / stray text). Ping/Pong is handled below
            // the axum layer; we just need to notice a client-initiated close.
            inbound = socket.recv() => {
                match inbound {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => {}
                    Some(Err(_)) => break,
                }
            }
            ev = rx.recv() => {
                match ev {
                    Ok((prompt_id, event)) => {
                        for msg in comfy_messages_for(&prompt_id, &event) {
                            if socket.send(Message::Text(msg)).await.is_err() {
                                return;
                            }
                        }
                    }
                    // A slow socket that lagged past the bus cap: keep going from the
                    // current position (a dropped progress frame is cosmetic).
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    // All senders dropped — server shutting down.
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }
    let _ = socket.close().await;
}

/// Translate one `WorkerEvent` for `prompt_id` into the ComfyUI WS message(s) the
/// canvas consumes. The canvas listens for: `execution_start`, `progress`,
/// `executed` (reads `data.output.images[0].{filename,subfolder,type}`),
/// `execution_success`, and `execution_error` (`data.exception_message`).
fn comfy_messages_for(prompt_id: &str, ev: &WorkerEvent) -> Vec<String> {
    match ev {
        WorkerEvent::Ready => vec![],
        WorkerEvent::Progress { step, total, .. } => {
            let mut out = Vec::new();
            // Announce start on the first step so the canvas resets/enables its
            // progress UI even if it connected after the job was queued.
            if *step <= 1 {
                out.push(
                    json!({ "type": "execution_start", "data": { "prompt_id": prompt_id } })
                        .to_string(),
                );
            }
            out.push(
                json!({
                    "type": "progress",
                    "data": { "value": step, "max": total, "prompt_id": prompt_id }
                })
                .to_string(),
            );
            out
        }
        WorkerEvent::Done { output_path } => {
            let filename = std::path::Path::new(output_path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            vec![
                json!({
                    "type": "executed",
                    "data": {
                        "node": "save",
                        "display_node": "save",
                        "prompt_id": prompt_id,
                        "output": {
                            "images": [ { "filename": filename, "subfolder": "", "type": "output" } ]
                        }
                    }
                })
                .to_string(),
                json!({ "type": "execution_success", "data": { "prompt_id": prompt_id } })
                    .to_string(),
            ]
        }
        WorkerEvent::Failed { error } => vec![json!({
            "type": "execution_error",
            "data": {
                "prompt_id": prompt_id,
                "exception_message": error,
                "node_id": "",
                "node_type": "",
                "executed": []
            }
        })
        .to_string()],
        WorkerEvent::Cancelled => vec![json!({
            "type": "execution_interrupted",
            "data": { "prompt_id": prompt_id }
        })
        .to_string()],
    }
}

// ── GET /view — serve a produced image ──────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ViewQuery {
    filename: String,
    #[serde(default)]
    subfolder: String,
    #[serde(default, rename = "type")]
    #[allow(dead_code)]
    ftype: String,
}

/// `GET /view?filename=&subfolder=&type=` — serve `<out_dir>/<subfolder>/<filename>`.
/// Only `output` images live in `out_dir`; `type` is accepted but not used to pick a
/// different root (uploads are also under `out_dir/uploads`, reachable via subfolder).
pub async fn get_view(State(st): State<AppState>, Query(q): Query<ViewQuery>) -> Response {
    if q.filename.contains("..") || q.subfolder.contains("..") {
        return (StatusCode::BAD_REQUEST, "bad path").into_response();
    }
    let mut p = st.out_dir.clone();
    for seg in q.subfolder.split('/').filter(|s| !s.is_empty()) {
        p.push(seg);
    }
    for seg in q.filename.split('/').filter(|s| !s.is_empty()) {
        p.push(seg);
    }
    match std::fs::read(&p) {
        Ok(bytes) => (
            [
                (header::CONTENT_TYPE, view_content_type(&p)),
                (header::CACHE_CONTROL, "no-store, must-revalidate"),
            ],
            bytes,
        )
            .into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

fn view_content_type(p: &std::path::Path) -> &'static str {
    match p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "mp4" => "video/mp4",
        _ => "application/octet-stream",
    }
}

// ── GET /object_info — node schema map (drives the model dropdown) ───────────────

/// A single-combo required input in ComfyUI `/object_info` shape: `[[options], meta]`.
/// The canvas reads `input.required.<name>[0]` and expects an array of option strings.
fn combo(options: &[String]) -> Value {
    json!([options, {}])
}

/// `GET /object_info` — emit just enough of the ComfyUI node schema for the canvas.
/// CRITICAL: the model dropdown is built by merging `CheckpointLoaderSimple.ckpt_name`
/// and `UNETLoader.unet_name`, so every disk-scanned model is listed under BOTH.
pub async fn get_object_info() -> Response {
    let models = crate::models::checkpoint_names();
    let loras = crate::models::lora_names();
    // VAE options: the canvas's zimage/flux builders hardcode `ae.safetensors`; expose
    // a small curated set so the advanced VAE picker is non-empty.
    let vaes: Vec<String> = vec![
        "ae.safetensors".into(),
        "flux2-vae.safetensors".into(),
        "sdxl_vae.safetensors".into(),
        "wan2.2_vae.safetensors".into(),
    ];
    let clips: Vec<String> = vec![
        "qwen_3_4b.safetensors".into(),
        "clip_l.safetensors".into(),
        "clip_g.safetensors".into(),
        "t5xxl_fp16.safetensors".into(),
    ];
    let dtypes: Vec<String> = vec!["default".into(), "fp8_e4m3fn".into(), "fp8_e5m2".into()];

    // Node catalog with FULL linked-input + widget declarations. The canvas
    // builds node SOCKETS from these; a node type with empty/missing inputs
    // renders socketless, silently drops every connection on template load,
    // and the round-trip serialization then 501s at queue time ("missing clip
    // input") — the gate-observed 2026-07-16 failure. Covers every class_type
    // used by the 40 shipped workflow templates.
    fn lk(t: &str) -> serde_json::Value {
        json!([t])
    }
    let int = |d: i64, mn: i64, mx: i64| json!(["INT", {"default": d, "min": mn, "max": mx}]);
    let fl = |d: f64, mn: f64, mx: f64| json!(["FLOAT", {"default": d, "min": mn, "max": mx, "step": 0.01}]);
    let str_w = || json!(["STRING", {"multiline": false}]);
    let text_w = || json!(["STRING", {"multiline": true}]);
    let samplers = combo(&[
        "euler".into(),
        "euler_ancestral".into(),
        "dpmpp_2m".into(),
        "uni_pc".into(),
        "uni_pc_bh2".into(),
        "flowmatch_euler".into(),
        "res_multistep".into(),
    ]);
    let scheds = combo(&[
        "simple".into(),
        "normal".into(),
        "sgm_uniform".into(),
        "flux2".into(),
        "karras".into(),
        "beta".into(),
    ]);

    let doc = json!({
        "CheckpointLoaderSimple": { "input": { "required": { "ckpt_name": combo(&models) } },
            "output": ["MODEL","CLIP","VAE"], "output_name": ["MODEL","CLIP","VAE"], "name": "CheckpointLoaderSimple", "category": "loaders" },
        "UNETLoader": { "input": { "required": { "unet_name": combo(&models), "weight_dtype": combo(&dtypes) } },
            "output": ["MODEL"], "output_name": ["MODEL"], "name": "UNETLoader", "category": "loaders" },
        "LTXVLoader": { "input": { "required": {
                "checkpoint_path": combo(&models),
                "gemma_path": str_w(),
                "dtype": combo(&dtypes),
                "quantization": combo(&["auto".into(),"fp8".into(),"int4".into()]),
                "backend": combo(&["mojo".into()])
            } },
            "output": ["MODEL","VAE"], "output_name": ["MODEL","VAE"], "name": "LTXVLoader", "category": "loaders" },
        "VAELoader": { "input": { "required": { "vae_name": combo(&vaes) } },
            "output": ["VAE"], "output_name": ["VAE"], "name": "VAELoader", "category": "loaders" },
        "CLIPLoader": { "input": { "required": { "clip_name": combo(&clips), "type": combo(&["zimage".into(),"flux".into(),"sd3".into(),"stable_diffusion".into(),"klein".into(),"krea2".into(),"ideogram4".into(),"ltxv".into(),"wan".into()]), "device": combo(&["default".into(),"cpu".into()]) } },
            "output": ["CLIP"], "output_name": ["CLIP"], "name": "CLIPLoader", "category": "loaders" },
        "DualCLIPLoader": { "input": { "required": { "clip_name1": combo(&clips), "clip_name2": combo(&clips), "type": combo(&["flux".into(),"sdxl".into(),"sd3".into()]) } },
            "output": ["CLIP"], "output_name": ["CLIP"], "name": "DualCLIPLoader", "category": "loaders" },
        "TripleCLIPLoader": { "input": { "required": { "clip_name1": combo(&clips), "clip_name2": combo(&clips), "clip_name3": combo(&clips) } },
            "output": ["CLIP"], "output_name": ["CLIP"], "name": "TripleCLIPLoader", "category": "loaders" },
        "CLIPVisionLoader": { "input": { "required": { "clip_name": combo(&clips) } },
            "output": ["CLIP_VISION"], "output_name": ["CLIP_VISION"], "name": "CLIPVisionLoader", "category": "loaders" },
        "LoraLoader": { "input": { "required": { "model": lk("MODEL"), "clip": lk("CLIP"), "lora_name": combo(&loras), "strength_model": fl(1.0,-20.0,20.0), "strength_clip": fl(1.0,-20.0,20.0) } },
            "output": ["MODEL","CLIP"], "output_name": ["MODEL","CLIP"], "name": "LoraLoader", "category": "loaders" },
        "LoraLoaderModelOnly": { "input": { "required": { "model": lk("MODEL"), "lora_name": combo(&loras), "strength_model": fl(1.0,-20.0,20.0) } },
            "output": ["MODEL"], "output_name": ["MODEL"], "name": "LoraLoaderModelOnly", "category": "loaders" },
        "CLIPTextEncode": { "input": { "required": { "clip": lk("CLIP"), "text": text_w() } },
            "output": ["CONDITIONING"], "output_name": ["CONDITIONING"], "name": "CLIPTextEncode", "category": "conditioning" },
        "CLIPTextEncodeFlux": { "input": { "required": { "clip": lk("CLIP"), "clip_l": text_w(), "t5xxl": text_w(), "guidance": fl(3.5,0.0,100.0) } },
            "output": ["CONDITIONING"], "output_name": ["CONDITIONING"], "name": "CLIPTextEncodeFlux", "category": "conditioning" },
        "TextEncodeQwenImageEditPlus": { "input": { "required": { "clip": lk("CLIP"), "prompt": text_w() }, "optional": { "vae": lk("VAE"), "image1": lk("IMAGE"), "image2": lk("IMAGE"), "image3": lk("IMAGE") } },
            "output": ["CONDITIONING"], "output_name": ["CONDITIONING"], "name": "TextEncodeQwenImageEditPlus", "category": "conditioning" },
        "CLIPVisionEncode": { "input": { "required": { "clip_vision": lk("CLIP_VISION"), "image": lk("IMAGE"), "crop": combo(&["center".into(),"none".into()]) } },
            "output": ["CLIP_VISION_OUTPUT"], "output_name": ["CLIP_VISION_OUTPUT"], "name": "CLIPVisionEncode", "category": "conditioning" },
        "ConditioningZeroOut": { "input": { "required": { "conditioning": lk("CONDITIONING") } },
            "output": ["CONDITIONING"], "output_name": ["CONDITIONING"], "name": "ConditioningZeroOut", "category": "conditioning" },
        "FluxGuidance": { "input": { "required": { "conditioning": lk("CONDITIONING"), "guidance": fl(3.5,0.0,100.0) } },
            "output": ["CONDITIONING"], "output_name": ["CONDITIONING"], "name": "FluxGuidance", "category": "conditioning" },
        "ReferenceLatent": { "input": { "required": { "conditioning": lk("CONDITIONING"), "latent": lk("LATENT") } },
            "output": ["CONDITIONING"], "output_name": ["CONDITIONING"], "name": "ReferenceLatent", "category": "conditioning" },
        "InpaintModelConditioning": { "input": { "required": { "positive": lk("CONDITIONING"), "negative": lk("CONDITIONING"), "vae": lk("VAE"), "pixels": lk("IMAGE"), "mask": lk("MASK"), "noise_mask": json!(["BOOLEAN", {"default": true}]) } },
            "output": ["CONDITIONING","CONDITIONING","LATENT"], "output_name": ["positive","negative","latent"], "name": "InpaintModelConditioning", "category": "conditioning" },
        "EmptyLatentImage": { "input": { "required": { "width": int(1024,64,8192), "height": int(1024,64,8192), "batch_size": int(1,1,64) } },
            "output": ["LATENT"], "output_name": ["LATENT"], "name": "EmptyLatentImage", "category": "latent" },
        "EmptySD3LatentImage": { "input": { "required": { "width": int(1024,64,8192), "height": int(1024,64,8192), "batch_size": int(1,1,64) } },
            "output": ["LATENT"], "output_name": ["LATENT"], "name": "EmptySD3LatentImage", "category": "latent" },
        "EmptyFlux2LatentImage": { "input": { "required": { "width": int(1024,64,8192), "height": int(1024,64,8192), "batch_size": int(1,1,64) } },
            "output": ["LATENT"], "output_name": ["LATENT"], "name": "EmptyFlux2LatentImage", "category": "latent" },
        "EmptyHunyuanLatentVideo": { "input": { "required": { "width": int(832,64,4096), "height": int(480,64,4096), "length": int(121,1,1024), "batch_size": int(1,1,16) } },
            "output": ["LATENT"], "output_name": ["LATENT"], "name": "EmptyHunyuanLatentVideo", "category": "latent" },
        "VAEEncode": { "input": { "required": { "pixels": lk("IMAGE"), "vae": lk("VAE") } },
            "output": ["LATENT"], "output_name": ["LATENT"], "name": "VAEEncode", "category": "latent" },
        "VAEDecode": { "input": { "required": { "samples": lk("LATENT"), "vae": lk("VAE") } },
            "output": ["IMAGE"], "output_name": ["IMAGE"], "name": "VAEDecode", "category": "latent" },
        "LoadImage": { "input": { "required": { "image": str_w() } },
            "output": ["IMAGE","MASK"], "output_name": ["IMAGE","MASK"], "name": "LoadImage", "category": "image" },
        "LoadAudio": { "input": { "required": { "audio": str_w() } },
            "output": ["AUDIO"], "output_name": ["AUDIO"], "name": "LoadAudio", "category": "audio" },
        "ImageScaleToTotalPixels": { "input": { "required": { "image": lk("IMAGE"), "upscale_method": combo(&["lanczos".into(),"bilinear".into(),"nearest-exact".into()]), "megapixels": fl(1.0,0.01,64.0) } },
            "output": ["IMAGE"], "output_name": ["IMAGE"], "name": "ImageScaleToTotalPixels", "category": "image" },
        "ModelSamplingAuraFlow": { "input": { "required": { "model": lk("MODEL"), "shift": fl(3.0,0.0,100.0) } },
            "output": ["MODEL"], "output_name": ["MODEL"], "name": "ModelSamplingAuraFlow", "category": "advanced" },
        "ModelSamplingSD3": { "input": { "required": { "model": lk("MODEL"), "shift": fl(3.0,0.0,100.0) } },
            "output": ["MODEL"], "output_name": ["MODEL"], "name": "ModelSamplingSD3", "category": "advanced" },
        "DifferentialDiffusion": { "input": { "required": { "model": lk("MODEL") } },
            "output": ["MODEL"], "output_name": ["MODEL"], "name": "DifferentialDiffusion", "category": "advanced" },
        "KSampler": { "input": { "required": { "model": lk("MODEL"), "positive": lk("CONDITIONING"), "negative": lk("CONDITIONING"), "latent_image": lk("LATENT"), "seed": int(0,0,281474976710655), "steps": int(20,1,4096), "cfg": fl(4.5,0.0,100.0), "sampler_name": samplers, "scheduler": scheds, "denoise": fl(1.0,0.0,1.0) } },
            "output": ["LATENT"], "output_name": ["LATENT"], "name": "KSampler", "category": "sampling" },
        "KSamplerSelect": { "input": { "required": { "sampler_name": samplers } },
            "output": ["SAMPLER"], "output_name": ["SAMPLER"], "name": "KSamplerSelect", "category": "sampling" },
        "RandomNoise": { "input": { "required": { "noise_seed": int(0,0,281474976710655) } },
            "output": ["NOISE"], "output_name": ["NOISE"], "name": "RandomNoise", "category": "sampling" },
        "CFGGuider": { "input": { "required": { "model": lk("MODEL"), "positive": lk("CONDITIONING"), "negative": lk("CONDITIONING"), "cfg": fl(4.5,0.0,100.0) } },
            "output": ["GUIDER"], "output_name": ["GUIDER"], "name": "CFGGuider", "category": "sampling" },
        "Flux2Scheduler": { "input": { "required": { "model": lk("MODEL"), "steps": int(35,1,4096), "width": int(1024,64,8192), "height": int(1024,64,8192) } },
            "output": ["SIGMAS"], "output_name": ["SIGMAS"], "name": "Flux2Scheduler", "category": "sampling" },
        "SamplerCustomAdvanced": { "input": { "required": { "noise": lk("NOISE"), "guider": lk("GUIDER"), "sampler": lk("SAMPLER"), "sigmas": lk("SIGMAS"), "latent_image": lk("LATENT") } },
            "output": ["LATENT","LATENT"], "output_name": ["output","denoised_output"], "name": "SamplerCustomAdvanced", "category": "sampling" },
        "LTXVSampler": { "input": { "required": {
                "ltxv_model": lk("MODEL"),
                "prompt": text_w(),
                "negative_prompt": text_w(),
                "width": int(960,960,960),
                "height": int(544,544,544),
                "num_frames": int(241,241,241),
                "steps": int(15,15,15),
                "cfg": fl(3.0,3.0,3.0),
                "seed": int(0,0,281474976710655),
                "frame_rate": int(24,24,24),
                "stg_scale": fl(1.0,1.0,1.0),
                "mode": combo(&["distilled".into(),"dev".into()])
            }, "optional": {
                "guide_image": lk("IMAGE"),
                "guide_strength": fl(1.0,0.0,1.0),
                "guide_frame_idx": int(0,0,240)
            } },
            "output": ["LATENT","VIDEO","AUDIO"], "output_name": ["LATENT","VIDEO","AUDIO"], "name": "LTXVSampler", "category": "sampling" },
        "SCAIL2Animation": { "input": { "required": {
                "model": str_w(),
                "prompt": text_w(),
                "negative_prompt": text_w(),
                "mode": combo(&["animation".into(),"replacement".into()]),
                "reference_image": str_w(),
                "reference_mask": str_w(),
                "driving_video": str_w(),
                "driving_mask_video": str_w(),
                "width": int(896,896,896),
                "height": int(512,512,512),
                "frames": int(65,65,65),
                "fps": int(16,16,16),
                "steps": int(40,40,40),
                "cfg": fl(5.0,5.0,5.0),
                "seed": int(0,0,281474976710655),
                "quant": combo(&["fp8".into()])
            }, "optional": {
                "additional_reference_images": str_w(),
                "additional_reference_masks": str_w()
            } },
            "output": ["VIDEO"], "output_name": ["VIDEO"], "name": "SCAIL2Animation", "category": "video" },
        "WanImageToVideo": { "input": { "required": { "positive": lk("CONDITIONING"), "negative": lk("CONDITIONING"), "vae": lk("VAE"), "width": int(832,64,4096), "height": int(480,64,4096), "length": int(81,1,1024), "batch_size": int(1,1,16) }, "optional": { "clip_vision_output": lk("CLIP_VISION_OUTPUT"), "start_image": lk("IMAGE") } },
            "output": ["CONDITIONING","CONDITIONING","LATENT"], "output_name": ["positive","negative","latent"], "name": "WanImageToVideo", "category": "video" },
        "EmptyLatentVideo": { "input": { "required": {
                "width": int(832,832,832),
                "height": int(480,480,480),
                "length": int(121,121,121),
                "batch_size": int(1,1,1)
            } },
            "output": ["LATENT"], "output_name": ["LATENT"], "name": "EmptyLatentVideo", "category": "video" },
        "SaveImage": { "input": { "required": { "images": lk("IMAGE"), "filename_prefix": str_w() } },
            "output": [], "output_name": [], "name": "SaveImage", "category": "image" },
        "SaveVideo": { "input": { "required": { "video": lk("VIDEO"), "filename_prefix": str_w() } },
            "output": [], "output_name": [], "name": "SaveVideo", "category": "video" },
        "SaveAnimatedWEBP": { "input": { "required": { "images": lk("IMAGE"), "filename_prefix": str_w(), "fps": int(24,24,24) } },
            "output": [], "output_name": [], "name": "SaveAnimatedWEBP", "category": "video" },
        "SaveAudioOpus": { "input": { "required": { "audio": lk("AUDIO"), "filename_prefix": str_w() } },
            "output": [], "output_name": [], "name": "SaveAudioOpus", "category": "audio" },
    });
    (
        [(header::CONTENT_TYPE, "application/json")],
        serde_json::to_string(&doc).unwrap_or_else(|_| "{}".into()),
    )
        .into_response()
}

/// `GET /object_info/{node}` — some clients probe one node. Return the whole map;
/// the canvas only ever fetches the full `/object_info`, so this is a convenience.
pub async fn get_object_info_node(Path(_node): Path<String>) -> Response {
    get_object_info().await
}

// ── read-only stubs the canvas polls on load (all defensive client-side) ─────────

/// `GET /system_stats` — minimal ComfyUI system report so the canvas footer/badges
/// render. VRAM totals are cosmetic here.
pub async fn get_system_stats() -> Response {
    Json(json!({
        "system": {
            "os": "posix",
            "comfyui_version": "serenity-server-0.1",
            "python_version": "n/a",
            "embedded_python": false,
        },
        "devices": [{
            "name": "cuda:0",
            "type": "cuda",
            "index": 0,
            "vram_total": 25_757_220_864_u64,
            "vram_free": 0,
            "torch_vram_total": 0,
            "torch_vram_free": 0,
        }],
    }))
    .into_response()
}

/// `GET /queue` — running/pending. The canvas tracks its own pending set locally
/// (QueueTab.registerPending), so empty arrays are correct and safe.
pub async fn get_queue() -> Response {
    Json(json!({ "queue_running": [], "queue_pending": [] })).into_response()
}

/// `POST /queue` — ComfyUI uses this to clear/delete queued items. No-op accept.
pub async fn post_queue() -> Response {
    (StatusCode::OK, Json(json!({}))).into_response()
}

/// `GET /history` — completed-prompt map. The canvas reads results over the WS
/// `executed` message, not history, so an empty map is sufficient for Tier A.
pub async fn get_history() -> Response {
    Json(json!({})).into_response()
}

/// `GET /history/{prompt_id}` — single entry (empty map).
pub async fn get_history_one(Path(_id): Path<String>) -> Response {
    Json(json!({})).into_response()
}

/// `POST /interrupt` — cancel whatever job is currently in flight (single-GPU).
pub async fn post_interrupt(State(st): State<AppState>) -> Response {
    let cur = st.in_flight.lock().ok().and_then(|g| g.clone());
    if let Some(id) = cur {
        let _ = st.ctl.send(DriverCtl::Cancel(id));
    }
    (StatusCode::OK, Json(json!({}))).into_response()
}

/// `GET /models` — delegate to the existing `/v1/models` disk-scan browser doc.
pub async fn get_models_comfy(q: Query<HashMap<String, String>>) -> Response {
    crate::models::get_models(q).await
}

/// `GET /models/loras` — bare array of LoRA name strings (the canvas maps each to an
/// `<option>` value).
pub async fn get_loras_comfy() -> Response {
    Json(crate::models::lora_names()).into_response()
}

/// `GET /embeddings` — bare array of textual-inversion names. None wired yet.
pub async fn get_embeddings() -> Response {
    Json(json!([])).into_response()
}
