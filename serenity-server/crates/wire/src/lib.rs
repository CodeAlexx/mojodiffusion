//! serenity-wire — the FROZEN IPC contract between the Rust control plane and the
//! unchanged Mojo inference workers.
//!
//! This crate is the single source of truth for the wire bytes. It is a 1:1
//! reproduction of `serenitymojo/serve/ipc_codec.mojo` (encode_start / decode_start /
//! encode_ev) and `backend.mojo` (JobParams defaults). DO NOT add/rename/reorder
//! fields without changing the Mojo side in lockstep — the worker's `decode_start`
//! reads these keys by name, and `serenity.genparams.v1` parity depends on it.
//!
//! Messages (one JSON object per line, '\n' framed):
//!   parent->child : {"cmd":"start", <all JobParams fields>}  |  {"cmd":"cancel"}
//!   child->parent : {"ev":"ready"}
//!                   {"ev":"progress","step":N,"total":M,"phase":"...","preview":"..."}
//!                   {"ev":"done","output_path":"..."}
//!                   {"ev":"failed","error":"..."}
//!                   {"ev":"cancelled"}

use serde::{Deserialize, Serialize};

/// One LoRA overlay (ipc_codec encodes `{"name":..,"weight":..}` in the `lora` array).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoraSpec {
    pub name: String,
    pub weight: f64,
}

/// Everything a backend needs to run one generation job. Field NAMES, JSON TYPES,
/// and DEFAULTS mirror `backend.mojo::JobParams.__init__` and the keys emitted by
/// `ipc_codec.mojo::encode_start`. Declaration order matches encode_start (cosmetic
/// — the worker reads by key — but kept for faithful bytes).
///
/// Integer fields are `i64` (encoded `from_int`), real fields `f64` (`from_float`),
/// text `String` (`from_string`), flags `bool` (`from_bool`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct JobParams {
    pub job_id: String,
    pub model: String,
    pub prompt: String,
    pub negative: String,
    pub width: i64,
    pub height: i64,
    pub steps: i64,
    pub seed: i64,
    pub cfg: f64,
    pub cfg_override: f64,
    pub cfg_override_start_percent: f64,
    pub cfg_override_end_percent: f64,
    pub sampler: String,
    pub scheduler: String,
    pub sigma_shift: f64,
    pub variation_seed: i64,
    pub variation_strength: f64,
    pub images: i64,
    pub image_index: i64,
    pub image_count: i64,
    pub workflow_save_prefix: String,
    pub init_image: String,
    pub mask_image: String,
    pub lanpaint_mask_channel: String,
    pub inpaint_conditioning_image: String,
    pub inpaint_conditioning_mask: String,
    pub inpaint_conditioning_noise_mask: bool,
    pub qwen_edit_conditioning_image: String,
    pub sample_caps_pos: String,
    pub sample_caps_neg: String,
    pub conditioning_mask_image: String,
    pub conditioning_mask_channel: String,
    pub conditioning_mask_strength: f64,
    pub conditioning_mask_set_area_to_bounds: bool,
    pub outpaint_left: i64,
    pub outpaint_top: i64,
    pub outpaint_right: i64,
    pub outpaint_bottom: i64,
    pub outpaint_feathering: i64,
    pub threshold_mask_value: f64,
    pub threshold_mask_operator: String,
    pub lanpaint_mask_blend_overlap: i64,
    pub lanpaint_num_steps: i64,
    pub lanpaint_lambda: f64,
    pub lanpaint_step_size: f64,
    pub lanpaint_beta: f64,
    pub lanpaint_friction: f64,
    pub lanpaint_prompt_mode: String,
    pub lanpaint_inpainting_mode: String,
    pub lanpaint_add_noise: String,
    pub lanpaint_noise_seed: i64,
    pub lanpaint_start_at_step: i64,
    pub lanpaint_end_at_step: i64,
    pub lanpaint_return_with_leftover_noise: String,
    pub lanpaint_early_stop: i64,
    pub lanpaint_inner_threshold: f64,
    pub lanpaint_inner_patience: i64,
    pub reference_image: String,
    pub reference_latent_method: String,
    pub reference_latent_count: i64,
    pub creativity: f64,
    // ── advanced-sampling knobs (UI _section_advanced). Plumbed end-to-end so
    //    the worker can HONOR what it supports and warn-loud on what it can't —
    //    NEVER silently dropped. Defaults = "unset" sentinels (clip_skip 0 / the
    //    sigma/eta -1.0 / restart_sampling false / vae ""). ──
    pub clip_skip: i64,
    pub eta: f64,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub restart_sampling: bool,
    pub vae: String,
    pub out_dir: String,
    pub params_json: String,
    /// Serialized under the key `lora` (matches encode_start), not `loras`.
    #[serde(rename = "lora")]
    pub loras: Vec<LoraSpec>,
}

impl Default for JobParams {
    /// Mirrors backend.mojo::JobParams.__init__ defaults exactly.
    fn default() -> Self {
        JobParams {
            job_id: String::new(),
            model: String::new(),
            prompt: String::new(),
            negative: String::new(),
            width: 512,
            height: 512,
            steps: 20,
            seed: 0,
            cfg: 4.5,
            cfg_override: -1.0,
            cfg_override_start_percent: 0.0,
            cfg_override_end_percent: 1.0,
            sampler: String::new(),
            scheduler: String::new(),
            sigma_shift: 3.0,
            variation_seed: 0,
            variation_strength: 0.0,
            images: 1,
            image_index: 0,
            image_count: 1,
            workflow_save_prefix: String::new(),
            init_image: String::new(),
            mask_image: String::new(),
            lanpaint_mask_channel: String::new(),
            inpaint_conditioning_image: String::new(),
            inpaint_conditioning_mask: String::new(),
            inpaint_conditioning_noise_mask: false,
            qwen_edit_conditioning_image: String::new(),
            sample_caps_pos: String::new(),
            sample_caps_neg: String::new(),
            conditioning_mask_image: String::new(),
            conditioning_mask_channel: String::new(),
            conditioning_mask_strength: -1.0,
            conditioning_mask_set_area_to_bounds: false,
            outpaint_left: -1,
            outpaint_top: -1,
            outpaint_right: -1,
            outpaint_bottom: -1,
            outpaint_feathering: -1,
            threshold_mask_value: -1.0,
            threshold_mask_operator: String::new(),
            lanpaint_mask_blend_overlap: -1,
            lanpaint_num_steps: -1,
            lanpaint_lambda: -1.0,
            lanpaint_step_size: -1.0,
            lanpaint_beta: -1.0,
            lanpaint_friction: -1.0,
            lanpaint_prompt_mode: String::new(),
            lanpaint_inpainting_mode: String::new(),
            lanpaint_add_noise: String::new(),
            lanpaint_noise_seed: -1,
            lanpaint_start_at_step: -1,
            lanpaint_end_at_step: -1,
            lanpaint_return_with_leftover_noise: String::new(),
            lanpaint_early_stop: -1,
            lanpaint_inner_threshold: -1.0,
            lanpaint_inner_patience: -1,
            reference_image: String::new(),
            reference_latent_method: String::new(),
            reference_latent_count: 0,
            creativity: 0.5,
            // advanced-sampling "unset" sentinels (mirror backend.mojo).
            clip_skip: 0,
            eta: -1.0,
            sigma_min: -1.0,
            sigma_max: -1.0,
            restart_sampling: false,
            vae: String::new(),
            out_dir: String::new(),
            params_json: String::new(),
            loras: Vec::new(),
        }
    }
}

impl JobParams {
    /// Serialize to a `{"cmd":"start", ...}` line WITHOUT the trailing '\n'
    /// (the IPC layer appends the newline frame). Inverse of the worker's
    /// `decode_start`.
    pub fn to_start_line(&self) -> String {
        #[derive(Serialize)]
        struct StartCmd<'a> {
            cmd: &'static str,
            #[serde(flatten)]
            params: &'a JobParams,
        }
        serde_json::to_string(&StartCmd {
            cmd: "start",
            params: self,
        })
        .expect("JobParams always serializes")
    }
}

/// The cancel command line (matches `encode_cancel`). No trailing '\n'.
pub const CANCEL_LINE: &str = r#"{"cmd":"cancel"}"#;

/// child->parent events (inverse of `encode_ev` / `decode_ev`). Tagged by `ev`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "ev", rename_all = "lowercase")]
pub enum WorkerEvent {
    /// Emitted once on worker startup, before any job.
    Ready,
    Progress {
        step: i64,
        total: i64,
        #[serde(default)]
        phase: String,
        #[serde(default)]
        preview: String,
    },
    Done {
        output_path: String,
    },
    Failed {
        error: String,
    },
    Cancelled,
}

impl WorkerEvent {
    /// Parse one received line (no trailing '\n').
    pub fn parse(line: &str) -> serde_json::Result<WorkerEvent> {
        serde_json::from_str(line)
    }
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            WorkerEvent::Done { .. } | WorkerEvent::Failed { .. } | WorkerEvent::Cancelled
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_line_has_cmd_and_core_fields() {
        let mut p = JobParams::default();
        p.job_id = "j1".into();
        p.model = "stub".into();
        p.prompt = "hello".into();
        p.out_dir = "/tmp".into();
        let line = p.to_start_line();
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["cmd"], "start");
        assert_eq!(v["job_id"], "j1");
        assert_eq!(v["model"], "stub");
        assert_eq!(v["width"], 512); // default carried
        assert_eq!(v["cfg"], 4.5);
        assert!(v.get("lora").unwrap().is_array());
        // no trailing newline in the body
        assert!(!line.ends_with('\n'));
    }

    #[test]
    fn parse_events() {
        assert_eq!(
            WorkerEvent::parse(r#"{"ev":"ready"}"#).unwrap(),
            WorkerEvent::Ready
        );
        assert_eq!(
            WorkerEvent::parse(r#"{"ev":"done","output_path":"/tmp/j1.png"}"#).unwrap(),
            WorkerEvent::Done {
                output_path: "/tmp/j1.png".into()
            }
        );
        let p =
            WorkerEvent::parse(r#"{"ev":"progress","step":3,"total":20,"phase":"","preview":""}"#)
                .unwrap();
        assert!(matches!(
            p,
            WorkerEvent::Progress {
                step: 3,
                total: 20,
                ..
            }
        ));
        assert!(WorkerEvent::parse(r#"{"ev":"failed","error":"boom"}"#)
            .unwrap()
            .is_terminal());
    }

    #[test]
    fn cancel_line_matches_mojo() {
        assert_eq!(CANCEL_LINE, r#"{"cmd":"cancel"}"#);
    }

    #[test]
    fn advanced_sampling_knobs_default_and_serialize() {
        let p = JobParams::default();
        // "unset" sentinels.
        assert_eq!(p.clip_skip, 0);
        assert_eq!(p.eta, -1.0);
        assert_eq!(p.sigma_min, -1.0);
        assert_eq!(p.sigma_max, -1.0);
        assert!(!p.restart_sampling);
        assert_eq!(p.vae, "");
        // they ride the start line by name (lockstep with ipc_codec.mojo).
        let mut q = JobParams::default();
        q.clip_skip = 2;
        q.eta = 0.3;
        q.sigma_min = 0.03;
        q.sigma_max = 14.6;
        q.restart_sampling = true;
        q.vae = "sdxl_vae.safetensors".into();
        let v: serde_json::Value = serde_json::from_str(&q.to_start_line()).unwrap();
        assert_eq!(v["clip_skip"], 2);
        assert_eq!(v["eta"], 0.3);
        assert_eq!(v["sigma_min"], 0.03);
        assert_eq!(v["sigma_max"], 14.6);
        assert_eq!(v["restart_sampling"], true);
        assert_eq!(v["vae"], "sdxl_vae.safetensors");
    }
}
