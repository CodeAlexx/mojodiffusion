use serde_json::{json, Map, Value};

#[derive(Debug, Copy, Clone)]
struct BlockProfileSpec {
    profile: &'static str,
    family: &'static str,
    source: &'static str,
    block_count: Option<i64>,
    block_kinds: &'static [(&'static str, i64)],
    tensor_count_hint: Option<i64>,
    byte_count_hint_per_block: Option<i64>,
    byte_count_hint_total: Option<i64>,
    storage_dtype: &'static str,
    offload_policy: &'static str,
    vmm_handle_available: bool,
    turbo_hot_path: bool,
}

impl BlockProfileSpec {
    fn to_json(self) -> Value {
        let mut kinds = Map::new();
        for (kind, count) in self.block_kinds {
            kinds.insert((*kind).to_string(), json!(count));
        }

        let mut out = json!({
            "profile": self.profile,
            "family": self.family,
            "source": self.source,
            "block_count": self.block_count,
            "block_kinds": kinds,
            "storage_dtype": self.storage_dtype,
            "offload_policy": self.offload_policy,
            "vmm_handle_available": self.vmm_handle_available,
            "turbo_hot_path": self.turbo_hot_path,
            "control_plane_owner": "rust",
            "runtime_owner": "mojo",
            "memory_block_policy_owner": "rust_preflight_plus_mojo_runtime",
            "vmm_manager": if self.vmm_handle_available {
                "serenitymojo/offload/vmm_manager.mojo"
            } else {
                ""
            },
            "runtime_dependency_on_external_repos": false,
        });

        if let Some(obj) = out.as_object_mut() {
            if let Some(v) = self.tensor_count_hint {
                obj.insert("tensor_count_hint".to_string(), json!(v));
            }
            if let Some(v) = self.byte_count_hint_per_block {
                obj.insert("byte_count_hint_per_block".to_string(), json!(v));
            }
            if let Some(v) = self.byte_count_hint_total {
                obj.insert("byte_count_hint_total".to_string(), json!(v));
            }
        }
        out
    }
}

const QWEN_IMAGE: BlockProfileSpec = BlockProfileSpec {
    profile: "qwen_image_transformer",
    family: "qwenimage",
    source: "serenitymojo/offload/plan.mojo::build_qwenimage_block_plan",
    block_count: Some(60),
    block_kinds: &[("double_stream", 60)],
    tensor_count_hint: Some(1920),
    byte_count_hint_per_block: Some(679_662_592),
    byte_count_hint_total: Some(40_779_755_520),
    storage_dtype: "BF16",
    offload_policy: "planned_loader",
    vmm_handle_available: true,
    turbo_hot_path: false,
};

const KLEIN9B_FLUX2: BlockProfileSpec = BlockProfileSpec {
    profile: "klein9b_flux2_dit",
    family: "flux2",
    source: "serenitymojo/offload/plan.mojo::build_klein9b_block_plan",
    block_count: Some(32),
    block_kinds: &[("double_stream", 8), ("single_stream", 24)],
    tensor_count_hint: None,
    byte_count_hint_per_block: None,
    byte_count_hint_total: None,
    storage_dtype: "BF16",
    offload_policy: "planned_loader_turbo_candidate",
    vmm_handle_available: true,
    turbo_hot_path: false,
};

const FLUX1_DEV: BlockProfileSpec = BlockProfileSpec {
    profile: "flux1_dev_dit",
    family: "flux",
    source: "serenitymojo/offload/plan.mojo::build_flux1_dev_block_plan",
    block_count: Some(57),
    block_kinds: &[("double_stream", 19), ("single_stream", 38)],
    tensor_count_hint: None,
    byte_count_hint_per_block: None,
    byte_count_hint_total: None,
    storage_dtype: "BF16",
    offload_policy: "planned_loader",
    vmm_handle_available: true,
    turbo_hot_path: false,
};

const SD35_LARGE: BlockProfileSpec = BlockProfileSpec {
    profile: "sd35_large_mmdit",
    family: "sd3",
    source: "serenitymojo/offload/plan.mojo::build_sd35_large_block_plan",
    block_count: Some(38),
    block_kinds: &[("joint_double_stream", 38)],
    tensor_count_hint: None,
    byte_count_hint_per_block: None,
    byte_count_hint_total: None,
    storage_dtype: "BF16",
    offload_policy: "planned_loader",
    vmm_handle_available: true,
    turbo_hot_path: false,
};

const ZIMAGE_NEXTDIT: BlockProfileSpec = BlockProfileSpec {
    profile: "zimage_nextdit",
    family: "zimage",
    source: "serenitymojo/models/dit/zimage_dit.mojo::NextDiTConfig.zimage",
    block_count: Some(34),
    block_kinds: &[
        ("noise_refiner", 2),
        ("context_refiner", 2),
        ("main_layers", 30),
    ],
    tensor_count_hint: None,
    byte_count_hint_per_block: None,
    byte_count_hint_total: None,
    storage_dtype: "BF16",
    offload_policy: "resident_worker",
    vmm_handle_available: false,
    turbo_hot_path: false,
};

const IDEOGRAM4_FP8: BlockProfileSpec = BlockProfileSpec {
    profile: "ideogram4_fp8_resident",
    family: "ideogram4",
    source: "serenitymojo/models/dit/ideogram4_dit.mojo::ideogram4_forward",
    block_count: Some(34),
    block_kinds: &[("transformer_layers", 34)],
    tensor_count_hint: None,
    byte_count_hint_per_block: None,
    byte_count_hint_total: None,
    storage_dtype: "FP8_E4M3 weights + BF16 activations",
    offload_policy: "resident_fp8_worker",
    vmm_handle_available: false,
    turbo_hot_path: false,
};

const SDXL_UNET: BlockProfileSpec = BlockProfileSpec {
    profile: "sdxl_unet",
    family: "sdxl",
    source: "serenitymojo/models/dit/sdxl_unet.mojo::SDXLUNet",
    block_count: Some(21),
    block_kinds: &[
        ("input_blocks", 9),
        ("middle_blocks", 3),
        ("output_blocks", 9),
    ],
    tensor_count_hint: None,
    byte_count_hint_per_block: None,
    byte_count_hint_total: None,
    storage_dtype: "BF16",
    offload_policy: "resident_worker",
    vmm_handle_available: false,
    turbo_hot_path: false,
};

const ANIMA_ADAPTER: BlockProfileSpec = BlockProfileSpec {
    profile: "anima_adapter",
    family: "anima",
    source: "serenitymojo/models/dit/anima_contract.mojo",
    block_count: Some(34),
    block_kinds: &[("adapter_blocks", 6), ("qwen3_layers", 28)],
    tensor_count_hint: None,
    byte_count_hint_per_block: None,
    byte_count_hint_total: None,
    storage_dtype: "BF16",
    offload_policy: "resident_worker",
    vmm_handle_available: false,
    turbo_hot_path: false,
};

const HIDREAM_O1: BlockProfileSpec = BlockProfileSpec {
    profile: "hidream_o1",
    family: "hidream",
    source: "serenitymojo/offload/plan.mojo::build_hidream_o1_block_plan",
    block_count: Some(36),
    block_kinds: &[("transformer", 36)],
    tensor_count_hint: None,
    byte_count_hint_per_block: None,
    byte_count_hint_total: None,
    storage_dtype: "BF16",
    offload_policy: "planned_loader",
    vmm_handle_available: true,
    turbo_hot_path: false,
};

const SENSENOVA_U1: BlockProfileSpec = BlockProfileSpec {
    profile: "sensenova_u1",
    family: "sensenova",
    source: "serenitymojo/offload/plan.mojo::build_sensenova_u1_block_plan",
    block_count: Some(42),
    block_kinds: &[("transformer", 42)],
    tensor_count_hint: None,
    byte_count_hint_per_block: None,
    byte_count_hint_total: None,
    storage_dtype: "BF16",
    offload_policy: "planned_loader",
    vmm_handle_available: true,
    turbo_hot_path: false,
};

const LANCE_T2V: BlockProfileSpec = BlockProfileSpec {
    profile: "lance_t2v",
    family: "lance",
    source: "serenitymojo/offload/plan.mojo::build_lance_t2v_block_plan",
    block_count: Some(36),
    block_kinds: &[("transformer", 36)],
    tensor_count_hint: None,
    byte_count_hint_per_block: None,
    byte_count_hint_total: None,
    storage_dtype: "BF16",
    offload_policy: "planned_loader",
    vmm_handle_available: true,
    turbo_hot_path: false,
};

fn unknown_profile() -> Value {
    BlockProfileSpec {
        profile: "unknown",
        family: "unknown",
        source: "none",
        block_count: None,
        block_kinds: &[],
        tensor_count_hint: None,
        byte_count_hint_per_block: None,
        byte_count_hint_total: None,
        storage_dtype: "unknown",
        offload_policy: "unknown",
        vmm_handle_available: false,
        turbo_hot_path: false,
    }
    .to_json()
}

pub(crate) fn local_block_profile(model: &str) -> Value {
    let m = model.trim().to_ascii_lowercase();
    if m.contains("qwen") {
        QWEN_IMAGE.to_json()
    } else if m.contains("klein")
        || m.contains("flux2")
        || m.contains("flux-2")
        || m.contains("flux_2")
    {
        KLEIN9B_FLUX2.to_json()
    } else if m.contains("flux") {
        FLUX1_DEV.to_json()
    } else if m.contains("sd3") || m.contains("sd35") || m.contains("sd3.5") {
        SD35_LARGE.to_json()
    } else if m.contains("zimage") || m.contains("z-image") || m.contains("z_image") {
        ZIMAGE_NEXTDIT.to_json()
    } else if m.contains("ideogram") {
        IDEOGRAM4_FP8.to_json()
    } else if m.contains("sdxl") || m.contains("stable-diffusion-xl") || m.contains("animagine") {
        SDXL_UNET.to_json()
    } else if m.contains("anima") {
        ANIMA_ADAPTER.to_json()
    } else if m.contains("hidream") || m.contains("hi-dream") || m.contains("hi_dream") {
        HIDREAM_O1.to_json()
    } else if m.contains("sensenova") || m.contains("sense_nova") || m.contains("sense-nova") {
        SENSENOVA_U1.to_json()
    } else if m.contains("lance") {
        LANCE_T2V.to_json()
    } else {
        unknown_profile()
    }
}

#[cfg(test)]
mod tests {
    use super::local_block_profile;

    #[test]
    fn qwen_profile_is_rust_owned_and_keeps_budget_hints() {
        let profile = local_block_profile("qwen-image");
        assert_eq!(profile["family"], "qwenimage");
        assert_eq!(profile["block_count"], 60);
        assert_eq!(profile["tensor_count_hint"], 1920);
        assert_eq!(profile["byte_count_hint_total"], 40_779_755_520i64);
        assert_eq!(profile["control_plane_owner"], "rust");
        assert_eq!(profile["runtime_owner"], "mojo");
        assert_eq!(profile["runtime_dependency_on_external_repos"], false);
    }

    #[test]
    fn unknown_profile_is_fail_closed() {
        let profile = local_block_profile("not-a-model");
        assert_eq!(profile["profile"], "unknown");
        assert_eq!(profile["block_count"], serde_json::Value::Null);
        assert_eq!(profile["vmm_handle_available"], false);
    }
}
